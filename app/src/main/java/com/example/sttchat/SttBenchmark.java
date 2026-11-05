package com.example.sttchat;

import android.content.Context;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.os.Build;
import android.os.Debug;
import android.os.SystemClock;
import android.util.Log;
import com.k2fsa.sherpa.onnx.OnlineRecognizer;
import com.k2fsa.sherpa.onnx.OnlineStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

/**
 * Runs WAV files through the app's STT recognizer on the phone and writes the results as TSV.
 *
 * <p>The PC tool {@code docs/stt-model-replacement/tools/device_benchmark.py} pushes the WAV files
 * to {@code <external files>/benchmark/input/<input>/}, starts the activity with the extras below,
 * waits for {@code <run>.done} and pulls the results from {@code <external files>/benchmark/output/}.
 *
 * <p>Each file is decoded in chunks of the microphone buffer size, like {@code recordLoop()}, but
 * without endpoint resets: the whole file gives one text. The first file is decoded once before
 * timing starts, because the first decode after loading is slower.
 */
final class SttBenchmark {
  interface Listener {
    void onProgress(String message);

    void onFinished(String message);
  }

  /** Optional step after STT, for example an LLM correction. Runs on the benchmark thread. */
  interface TextProcessor {
    String process(String text) throws Exception;
  }

  static final String EXTRA_INPUT = "benchmark_input";
  static final String EXTRA_RUN = "benchmark_run";
  static final String EXTRA_REPEAT = "benchmark_repeat";

  private static final String TAG = "SttBenchmark";
  private static final int SAMPLE_RATE = 16000;
  private static final float TAIL_PADDING_SECONDS = 0.5f;
  private static final Pattern SAFE_NAME = Pattern.compile("[A-Za-z0-9_-][A-Za-z0-9._-]{0,63}");

  private final Context context;
  private final String inputName;
  private final String runName;
  private final int repeat;
  private final TextProcessor processor;
  private final Listener listener;

  SttBenchmark(
      Context context,
      String inputName,
      String runName,
      int repeat,
      TextProcessor processor,
      Listener listener) {
    this.context = context.getApplicationContext();
    this.inputName = inputName;
    this.runName = runName;
    this.repeat = repeat;
    this.processor = processor;
    this.listener = listener;
  }

  /** Accepts only simple file names, so an intent extra cannot point outside the folders. */
  static boolean isValidName(String name) {
    return name != null && SAFE_NAME.matcher(name).matches();
  }

  /** Runs the benchmark on the calling thread. Never throws. */
  void run() {
    File base = new File(context.getExternalFilesDir(null), "benchmark");
    File outputDirectory = new File(base, "output");
    File doneFile = new File(outputDirectory, runName + ".done");
    try {
      if (!outputDirectory.isDirectory() && !outputDirectory.mkdirs()) {
        throw new IOException("Cannot create " + outputDirectory);
      }
      if (doneFile.exists() && !doneFile.delete()) {
        throw new IOException("Cannot delete " + doneFile);
      }
      String summary = execute(new File(base, "input/" + inputName), outputDirectory);
      writeText(doneFile, "ok\n");
      Log.i(TAG, "Benchmark finished: " + summary);
      listener.onFinished(summary);
    } catch (Exception | OutOfMemoryError error) {
      Log.e(TAG, "Benchmark failed", error);
      String message = "Benchmark failed: " + error;
      try {
        writeText(doneFile, "error: " + clean(String.valueOf(error)) + "\n");
      } catch (IOException ignored) {
        // The PC tool reports a timeout instead.
      }
      listener.onFinished(message);
    }
  }

  private String execute(File inputDirectory, File outputDirectory) throws IOException {
    File[] wavFiles =
        inputDirectory.listFiles((dir, name) -> name.toLowerCase(Locale.ROOT).endsWith(".wav"));
    if (wavFiles == null || wavFiles.length == 0) {
      throw new IOException("No WAV files in " + inputDirectory);
    }
    Arrays.sort(wavFiles);

    int minBufferBytes =
        AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
    int chunkSamples = minBufferBytes > 0 ? minBufferBytes / Short.BYTES : SAMPLE_RATE / 10;

    long pssBeforeLoadKb = totalPssKb();
    listener.onProgress("Loading model…");
    long loadStart = SystemClock.elapsedRealtime();
    OnlineRecognizer recognizer = OfflineSpeechRecognizer.createRecognizer(context);
    long loadMs = SystemClock.elapsedRealtime() - loadStart;
    long pssAfterLoadKb = totalPssKb();
    long pssPeakKb = pssAfterLoadKb;

    List<String[]> sttRows = new ArrayList<>();
    List<String[]> processedRows = new ArrayList<>();
    double totalAudioSeconds = 0;
    double totalDecodeMs = 0;
    try {
      listener.onProgress("Warm-up…");
      recognize(recognizer, readWav(wavFiles[0]), chunkSamples);

      for (int index = 0; index < wavFiles.length; index++) {
        File wavFile = wavFiles[index];
        listener.onProgress(
            "File " + (index + 1) + " / " + wavFiles.length + ": " + wavFile.getName());
        float[] samples = readWav(wavFile);
        double audioSeconds = samples.length / (double) SAMPLE_RATE;

        double[] decodeMs = new double[repeat];
        Result result = null;
        for (int run = 0; run < repeat; run++) {
          long start = SystemClock.elapsedRealtimeNanos();
          result = recognize(recognizer, samples, chunkSamples);
          decodeMs[run] = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0;
        }
        double medianMs = median(decodeMs);
        pssPeakKb = Math.max(pssPeakKb, totalPssKb());
        totalAudioSeconds += audioSeconds;
        totalDecodeMs += medianMs;
        sttRows.add(
            new String[] {
              wavFile.getName(),
              clean(result.text),
              format("%.2f", audioSeconds),
              format("%.0f", medianMs),
              format("%.3f", medianMs / 1000 / audioSeconds),
              result.firstEndpointSeconds < 0 ? "" : format("%.2f", result.firstEndpointSeconds)
            });

        if (processor != null) {
          processedRows.add(process(wavFile.getName(), result.text));
        }
      }
    } finally {
      recognizer.release();
    }

    writeTsv(
        new File(outputDirectory, runName + ".tsv"),
        new String[] {"file", "hypothesis", "audio_s", "decode_ms", "rtf", "first_endpoint_s"},
        sttRows);
    if (processor != null) {
      writeTsv(
          new File(outputDirectory, runName + "-processed.tsv"),
          new String[] {"file", "stt", "hypothesis", "process_ms", "status"},
          processedRows);
    }

    List<String[]> summary = new ArrayList<>();
    summary.add(new String[] {"run", runName});
    summary.add(new String[] {"input", inputName});
    summary.add(new String[] {"files", String.valueOf(wavFiles.length)});
    summary.add(new String[] {"repeat", String.valueOf(repeat)});
    summary.add(new String[] {"chunk_samples", String.valueOf(chunkSamples)});
    summary.add(new String[] {"device", Build.MANUFACTURER + " " + Build.MODEL});
    summary.add(
        new String[] {"android", Build.VERSION.RELEASE + " (API " + Build.VERSION.SDK_INT + ")"});
    summary.add(new String[] {"abi", Build.SUPPORTED_ABIS[0]});
    summary.add(new String[] {"model_load_ms", String.valueOf(loadMs)});
    summary.add(new String[] {"total_audio_s", format("%.2f", totalAudioSeconds)});
    summary.add(new String[] {"total_decode_ms", format("%.0f", totalDecodeMs)});
    summary.add(new String[] {"rtf", format("%.3f", totalDecodeMs / 1000 / totalAudioSeconds)});
    summary.add(new String[] {"pss_before_load_mb", format("%.1f", pssBeforeLoadKb / 1024.0)});
    summary.add(new String[] {"pss_after_load_mb", format("%.1f", pssAfterLoadKb / 1024.0)});
    summary.add(new String[] {"pss_peak_mb", format("%.1f", pssPeakKb / 1024.0)});
    writeTsv(
        new File(outputDirectory, runName + ".summary.tsv"),
        new String[] {"key", "value"},
        summary);

    return String.format(
        Locale.ROOT,
        "%d files, RTF %.3f, load %d ms, peak PSS %.0f MB",
        wavFiles.length,
        totalDecodeMs / 1000 / totalAudioSeconds,
        loadMs,
        pssPeakKb / 1024.0);
  }

  private String[] process(String fileName, String sttText) {
    if (sttText.isEmpty()) {
      return new String[] {fileName, "", "", "0", "skipped: empty STT text"};
    }
    String processed = sttText;
    String status = "ok";
    long start = SystemClock.elapsedRealtime();
    try {
      processed = processor.process(sttText);
    } catch (Exception error) {
      status = "error: " + error.getMessage();
    }
    long processMs = SystemClock.elapsedRealtime() - start;
    return new String[] {
      fileName, clean(sttText), clean(processed), String.valueOf(processMs), clean(status)
    };
  }

  private static final class Result {
    final String text;
    final double firstEndpointSeconds;

    Result(String text, double firstEndpointSeconds) {
      this.text = text;
      this.firstEndpointSeconds = firstEndpointSeconds;
    }
  }

  private static Result recognize(OnlineRecognizer recognizer, float[] samples, int chunkSamples) {
    OnlineStream stream = recognizer.createStream("");
    try {
      double firstEndpointSeconds = -1;
      for (int offset = 0; offset < samples.length; offset += chunkSamples) {
        int end = Math.min(samples.length, offset + chunkSamples);
        stream.acceptWaveform(Arrays.copyOfRange(samples, offset, end), SAMPLE_RATE);
        while (recognizer.isReady(stream)) {
          recognizer.decode(stream);
        }
        // The app stops at its first endpoint. Record where that would happen.
        if (firstEndpointSeconds < 0 && recognizer.isEndpoint(stream)) {
          firstEndpointSeconds = end / (double) SAMPLE_RATE;
        }
      }
      stream.acceptWaveform(new float[(int) (TAIL_PADDING_SECONDS * SAMPLE_RATE)], SAMPLE_RATE);
      stream.inputFinished();
      while (recognizer.isReady(stream)) {
        recognizer.decode(stream);
      }
      return new Result(recognizer.getResult(stream).getText().trim(), firstEndpointSeconds);
    } finally {
      stream.release();
    }
  }

  /** Reads a 16 kHz, mono, 16-bit PCM WAV file. Parses the chunks, because headers vary. */
  static float[] readWav(File file) throws IOException {
    byte[] data = Files.readAllBytes(file.toPath());
    ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    if (data.length < 12 || !"RIFF".equals(chunkId(data, 0)) || !"WAVE".equals(chunkId(data, 8))) {
      throw new IOException(file.getName() + ": not a WAV file");
    }
    boolean formatChecked = false;
    int position = 12;
    while (position + 8 <= data.length) {
      String id = chunkId(data, position);
      int size = buffer.getInt(position + 4);
      int body = position + 8;
      if (size < 0 || size > data.length - body) {
        size = data.length - body; // Some writers leave a wrong size in the last chunk.
      }
      if ("fmt ".equals(id)) {
        int format = buffer.getShort(body) & 0xffff;
        int channels = buffer.getShort(body + 2) & 0xffff;
        int sampleRate = buffer.getInt(body + 4);
        int bits = buffer.getShort(body + 14) & 0xffff;
        if (format != 1 || channels != 1 || sampleRate != SAMPLE_RATE || bits != 16) {
          throw new IOException(file.getName() + ": expected 16 kHz, mono, 16-bit PCM");
        }
        formatChecked = true;
      } else if ("data".equals(id)) {
        if (!formatChecked) {
          throw new IOException(file.getName() + ": data chunk before fmt chunk");
        }
        float[] samples = new float[size / Short.BYTES];
        for (int index = 0; index < samples.length; index++) {
          // Same scaling as OfflineSpeechRecognizer.recordLoop().
          samples[index] = buffer.getShort(body + index * Short.BYTES) / (float) Short.MAX_VALUE;
        }
        return samples;
      }
      position = body + size + (size & 1);
    }
    throw new IOException(file.getName() + ": no data chunk");
  }

  private static String chunkId(byte[] data, int offset) {
    return new String(data, offset, 4, StandardCharsets.US_ASCII);
  }

  private static long totalPssKb() {
    Debug.MemoryInfo info = new Debug.MemoryInfo();
    Debug.getMemoryInfo(info);
    return info.getTotalPss();
  }

  private static double median(double[] values) {
    double[] sorted = values.clone();
    Arrays.sort(sorted);
    int middle = sorted.length / 2;
    return sorted.length % 2 == 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  private static String clean(String text) {
    return text == null ? "" : text.replaceAll("[\\t\\r\\n]+", " ").trim();
  }

  private static String format(String pattern, double value) {
    return String.format(Locale.ROOT, pattern, value);
  }

  private static void writeTsv(File file, String[] header, List<String[]> rows)
      throws IOException {
    StringBuilder text = new StringBuilder(String.join("\t", header)).append('\n');
    for (String[] row : rows) {
      text.append(String.join("\t", row)).append('\n');
    }
    writeText(file, text.toString());
  }

  private static void writeText(File file, String text) throws IOException {
    try (Writer writer =
        new OutputStreamWriter(new FileOutputStream(file), StandardCharsets.UTF_8)) {
      writer.write(text);
    }
  }
}
