package com.example.sttchat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
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

    speechExecutor.execute(recognizer::prepare);
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
}
