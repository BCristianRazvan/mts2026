package com.example.sttchat;

import android.Manifest;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.graphics.Rect;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Displays offline speech recognition as a simple chat transcript. */
public class SttChatActivity extends AppCompatActivity {
  private static final int RECORD_AUDIO_REQUEST_CODE = 1;

  private EditText transcript;
  private TextView status;
  private Button microphoneButton;
  private final ExecutorService speechExecutor = Executors.newSingleThreadExecutor();
  private String finalizedText = "";
  private OfflineSpeechRecognizer recognizer;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_stt_chat);

    transcript = findViewById(R.id.transcript);
    status = findViewById(R.id.status);
    microphoneButton = findViewById(R.id.microphone_button);
    recognizer = new OfflineSpeechRecognizer(this, new SpeechListener());
    microphoneButton.setOnClickListener(view -> toggleListening());

    // savedInstanceState != null: the activity was recreated (for example after a rotation).
    // Do not start the same benchmark again.
    if (savedInstanceState != null || !startBenchmarkIfRequested()) {
      speechExecutor.execute(recognizer::prepare);
    }
  }

  /**
   * Starts the on-device benchmark when the activity is launched with the benchmark extras, for
   * example by {@code tools/device_benchmark.py}. Only debuggable builds accept the request.
   */
  private boolean startBenchmarkIfRequested() {
    String input = getIntent().getStringExtra(SttBenchmark.EXTRA_INPUT);
    if (input == null) {
      return false;
    }
    String run = getIntent().getStringExtra(SttBenchmark.EXTRA_RUN);
    if (run == null) {
      run = input;
    }
    int repeat = Math.max(1, Math.min(10, getIntent().getIntExtra(SttBenchmark.EXTRA_REPEAT, 3)));
    boolean debuggable = (getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) != 0;
    if (!debuggable || !SttBenchmark.isValidName(input) || !SttBenchmark.isValidName(run)) {
      // Show the reason in the transcript and start in normal microphone mode.
      finalizedText += getString(R.string.benchmark_rejected) + "\n";
      transcript.setText(finalizedText);
      return false;
    }

    // A screen that turns off slows down the CPU and changes the timing.
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    status.setText(getString(R.string.benchmark_running, run));
    SttBenchmark benchmark =
        new SttBenchmark(this, input, run, repeat, null, new BenchmarkListener());
    speechExecutor.execute(benchmark::run);
    return true;
  }

  @Override
  protected void onDestroy() {
    speechExecutor.execute(recognizer::destroy);
    speechExecutor.shutdown();
    super.onDestroy();
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == RECORD_AUDIO_REQUEST_CODE
        && grantResults.length > 0
        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
      startListening();
    } else if (requestCode == RECORD_AUDIO_REQUEST_CODE) {
      showError("Microphone permission is required");
    }
  }

  private void toggleListening() {
    if (recognizer.isListening()) {
      microphoneButton.setEnabled(false);
      status.setText(R.string.finishing_transcript);
      speechExecutor.execute(recognizer::stopAndSubmit);
      return;
    }

    if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
        != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(
          this, new String[] {Manifest.permission.RECORD_AUDIO}, RECORD_AUDIO_REQUEST_CODE);
      return;
    }
    startListening();
  }

  private void startListening() {
    microphoneButton.setEnabled(false);
    status.setText(R.string.starting_microphone);
    speechExecutor.execute(recognizer::start);
  }

  /** Hides the keyboard when the user touches anywhere outside the focused text box. */
  @Override
  public boolean dispatchTouchEvent(MotionEvent event) {
    if (event.getAction() == MotionEvent.ACTION_DOWN) {
      View focused = getCurrentFocus();
      if (focused instanceof EditText) {
        Rect bounds = new Rect();
        focused.getGlobalVisibleRect(bounds);
        if (!bounds.contains((int) event.getRawX(), (int) event.getRawY())) {
          focused.clearFocus();
          InputMethodManager keyboard = getSystemService(InputMethodManager.class);
          if (keyboard != null) {
            keyboard.hideSoftInputFromWindow(focused.getWindowToken(), 0);
          }
        }
      }
    }
    return super.dispatchTouchEvent(event);
  }

  private void showTranscript(String partialText) {
    String displayedText = finalizedText + partialText;
    transcript.setText(displayedText);
    transcript.setSelection(displayedText.length());
  }

  private void completeTranscript(String finalText) {
    finalizedText += finalText.trim() + "\n";
    transcript.setText(finalizedText);
    transcript.setSelection(finalizedText.length());
    status.setText(R.string.tap_to_listen);
    microphoneButton.setText(R.string.start_listening);
    microphoneButton.setEnabled(true);
  }

  private void showError(String message) {
    status.setText(message);
    microphoneButton.setText(R.string.start_listening);
    microphoneButton.setEnabled(false);
  }

  private final class SpeechListener implements OfflineSpeechRecognizer.Listener {
    @Override
    public void onModelReady() {
      runOnUiThread(
          () -> {
            status.setText(R.string.tap_to_listen);
            microphoneButton.setEnabled(true);
          });
    }

    @Override
    public void onReady() {
      runOnUiThread(
          () -> {
            status.setText(R.string.listening);
            microphoneButton.setText(R.string.stop_and_add);
            microphoneButton.setEnabled(true);
          });
    }

    @Override
    public void onPartialText(String text) {
      runOnUiThread(() -> showTranscript(text));
    }

    @Override
    public void onFinalText(String text) {
      runOnUiThread(() -> completeTranscript(text));
    }

    @Override
    public void onError(String message) {
      runOnUiThread(() -> showError(message));
    }
  }

  private final class BenchmarkListener implements SttBenchmark.Listener {
    @Override
    public void onProgress(String message) {
      runOnUiThread(
          () -> {
            if (!isDestroyed()) {
              status.setText(message);
            }
          });
    }

    @Override
    public void onFinished(String message) {
      runOnUiThread(
          () -> {
            if (isDestroyed() || speechExecutor.isShutdown()) {
              return;
            }
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            // Keep the result visible, then switch to normal microphone mode.
            finalizedText += message + "\n";
            transcript.setText(finalizedText);
            status.setText(R.string.loading_model);
            speechExecutor.execute(recognizer::prepare);
          });
    }
  }
}
