package com.example.sttchat;

import android.content.Context;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;
import com.k2fsa.sherpa.onnx.OnlineRecognizer;
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig;
import com.k2fsa.sherpa.onnx.OnlineStream;
import java.util.Arrays;

/** Offline Sherpa ONNX speech recognizer. */
final class OfflineSpeechRecognizer {
  interface Listener {
    void onModelReady();

    void onReady();

    void onPartialText(String text);

    void onFinalText(String text);

    void onError(String message);
  }

  private static final String TAG = "OfflineSpeechRecognizer";
  private static final int SAMPLE_RATE = 16000;
  private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
  private static final int ENCODING = AudioFormat.ENCODING_PCM_16BIT;
  private static final float RULE1_MIN_TRAILING_SILENCE_SECONDS = 0.8f;
  private static final float RULE2_MIN_TRAILING_SILENCE_SECONDS = 1.2f;
  private static final float RULE3_MIN_UTTERANCE_SECONDS = 20.0f;

  private final Context context;
  private final Listener listener;
  private final Object audioLock = new Object();

  private OnlineRecognizer recognizer;
  private OnlineStream stream;
  private AudioRecord audioRecord;
  private Thread recordingThread;
  private volatile boolean listening;
  private volatile boolean submitWhenStopped;

  OfflineSpeechRecognizer(Context context, Listener listener) {
    this.context = context.getApplicationContext();
    this.listener = listener;
  }

  void prepare() {
    if (recognizer != null) {
      return;
    }

    try {
      recognizer = createRecognizer(context);
      stream = recognizer.createStream("");
      listener.onModelReady();
    } catch (Throwable error) {
      Log.e(TAG, "Could not load offline STT model", error);
      recognizer = null;
      stream = null;
      listener.onError("Offline speech model could not start");
    }
  }

  /**
   * Creates the STT recognizer. The microphone mode and {@link SttBenchmark} both use this method,
   * so a model change here applies to both.
   */
  static OnlineRecognizer createRecognizer(Context context) {
    OnlineRecognizerConfig config = new OnlineRecognizerConfig();
    config.getFeatConfig().setSampleRate(SAMPLE_RATE);
    config.getFeatConfig().setFeatureDim(80);
    config
        .getEndpointConfig()
        .getRule1()
        .setMinTrailingSilence(RULE1_MIN_TRAILING_SILENCE_SECONDS);
    config
        .getEndpointConfig()
        .getRule2()
        .setMinTrailingSilence(RULE2_MIN_TRAILING_SILENCE_SECONDS);
    config.getEndpointConfig().getRule3().setMinUtteranceLength(RULE3_MIN_UTTERANCE_SECONDS);

    String modelDirectory = "models/stt/";
    config.getModelConfig().getTransducer().setEncoder(modelDirectory + "encoder.int8.onnx");
    config.getModelConfig().getTransducer().setDecoder(modelDirectory + "decoder.int8.onnx");
    config.getModelConfig().getTransducer().setJoiner(modelDirectory + "joiner.int8.onnx");
    config.getModelConfig().setTokens(modelDirectory + "tokens.txt");
    config.getModelConfig().setModelingUnit("bpe");
    config.getModelConfig().setBpeVocab(modelDirectory + "bpe.vocab");
    config.setHotwordsFile(modelDirectory + "hotwords.txt");
    config.setHotwordsScore(1.5f);
    config.getModelConfig().setNumThreads(4);
    config.setDecodingMethod("modified_beam_search");
    config.setMaxActivePaths(4);

    return new OnlineRecognizer(context.getAssets(), config);
  }

  void start() {
    if (listening) {
      return;
    }
    prepare();
    if (recognizer == null || stream == null) {
      return;
    }

    int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, ENCODING);
    if (bufferSize <= 0) {
      listener.onError("16 kHz mono microphone capture is not supported");
      return;
    }

    try {
      recognizer.reset(stream);
      AudioRecord recorder =
          new AudioRecord(
              MediaRecorder.AudioSource.VOICE_RECOGNITION,
              SAMPLE_RATE,
              CHANNEL_CONFIG,
              ENCODING,
              bufferSize);
      if (recorder.getState() != AudioRecord.STATE_INITIALIZED) {
        recorder.release();
        listener.onError("Microphone could not start");
        return;
      }

      synchronized (audioLock) {
        audioRecord = recorder;
      }
      submitWhenStopped = false;
      listening = true;
      recorder.startRecording();
      recordingThread = new Thread(() -> recordLoop(recorder), "offline-stt");
      recordingThread.start();
    } catch (SecurityException error) {
      listener.onError("Microphone permission is required");
    }
  }

  boolean isListening() {
    return listening;
  }

  void stopAndSubmit() {
    submitWhenStopped = true;
    stop();
  }

  void stop() {
    listening = false;
    releaseAudioRecord(null);
  }

  void destroy() {
    stop();
  }

  private void recordLoop(AudioRecord recorder) {
    int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, ENCODING);
    short[] pcm = new short[bufferSize / Short.BYTES];
    float[] samples = null;
    boolean readySent = false;

    try {
      while (listening) {
        int read = recorder.read(pcm, 0, pcm.length);
        if (read < 0) {
          if (listening) {
            listener.onError("Microphone stopped delivering audio");
          }
          break;
        }
        if (read == 0) {
          continue;
        }

        samples = new float[read];
        for (int index = 0; index < read; index++) {
          samples[index] = pcm[index] / (float) Short.MAX_VALUE;
        }
        stream.acceptWaveform(samples, SAMPLE_RATE);
        if (!readySent) {
          readySent = true;
          listener.onReady();
        }
        while (recognizer.isReady(stream)) {
          recognizer.decode(stream);
        }

        String text = recognizer.getResult(stream).getText();
        if (!text.isEmpty()) {
          if (recognizer.isEndpoint(stream)) {
            listener.onFinalText(text);
            recognizer.reset(stream);
            listening = false;
          } else {
            listener.onPartialText(text);
          }
        }
        Arrays.fill(samples, 0f);
        samples = null;
      }
    } catch (RuntimeException error) {
      if (listening) {
        Log.e(TAG, "Speech recognition failed", error);
        listener.onError("Speech recognition failed");
      }
    } finally {
      Arrays.fill(pcm, (short) 0);
      if (samples != null) {
        Arrays.fill(samples, 0f);
      }
      releaseAudioRecord(recorder);
      if (submitWhenStopped) {
        submitWhenStopped = false;
        submitPendingText();
      }
    }
  }

  private void submitPendingText() {
    try {
      stream.inputFinished();
      while (recognizer.isReady(stream)) {
        recognizer.decode(stream);
      }
      String text = recognizer.getResult(stream).getText();
      recognizer.reset(stream);
      if (!text.isEmpty()) {
        listener.onFinalText(text);
      }
    } catch (RuntimeException error) {
      Log.e(TAG, "Could not finish transcript", error);
    }
  }

  private void releaseAudioRecord(AudioRecord ownedRecorder) {
    synchronized (audioLock) {
      if (ownedRecorder != null && audioRecord != ownedRecorder) {
        return;
      }
      if (audioRecord == null) {
        return;
      }
      try {
        audioRecord.stop();
      } catch (IllegalStateException ignored) {
      }
      audioRecord.release();
      audioRecord = null;
    }
  }
}
