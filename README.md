# Offline STT Chat

An Android app that turns microphone input into a simple chat-style transcript. It runs speech recognition on-device using the bundled Sherpa-ONNX transducer model, so transcription does not require a network connection.

The app requests microphone permission when listening starts, displays partial recognition results, and adds finalized speech to the transcript.

## Workshop guides

The speech-to-text improvement workshop (two student groups: a stronger offline model, and a cloud LLM after local STT) is described in [docs/stt-model-replacement/README.md](docs/stt-model-replacement/README.md).

## Prerequisites

Before building the application, install:

- **Oracle JDK 21**
- **Android Studio Quail 4**
- **Android SDK API 35**
- **Android SDK Platform-Tools**
- An Android device or emulator for installation and testing

---

## 1. Install Oracle JDK 21

Download and install **Oracle JDK 21 for Windows (x64)** using the official Oracle installer:

[Download Oracle JDK 21 for Windows x64](https://download.oracle.com/java/21/latest/jdk-21_windows-x64_bin.exe)

Run the installer and follow the installation wizard.

The default installation directory is typically similar to:

```text
C:\Program Files\Java\jdk-21
```

The exact directory may vary depending on the installer configuration.

After installation, open a **new PowerShell window** and verify Java:

```powershell
java -version
```

The output should indicate Java 21.

---

## 2. Configure `JAVA_HOME`

`JAVA_HOME` must point to the **JDK installation directory**, not to the `bin` directory.

For the default Oracle JDK 21 installation:

```text
C:\Program Files\Java\jdk-21
```

### Configure using Windows Settings

1. Open the **Start** menu.
2. Search for **Environment Variables**.
3. Select **Edit the system environment variables**.
4. In the **System Properties** window, click **Environment Variables...**.
5. Under **User variables**, click **New...**.
6. Configure:

```text
Variable name:
JAVA_HOME

Variable value:
C:\Program Files\Java\jdk-21
```

7. Click **OK**.
8. Under **User variables**, select `Path` and click **Edit**.
9. Add:

```text
%JAVA_HOME%\bin
```

10. Click **OK** to close all environment-variable dialogs.

Close any existing PowerShell or Command Prompt windows and open a **new PowerShell window** so the updated environment variables are loaded.

### Verify `JAVA_HOME`

Run:

```powershell
$env:JAVA_HOME
```

Expected:

```text
C:\Program Files\Java\jdk-21
```

Verify the Java executable:

```powershell
where.exe java
```

The result should contain:

```text
C:\Program Files\Java\jdk-21\bin\java.exe
```

Finally:

```powershell
java -version
```

Make sure Java 21 is reported.

---

## 3. Install Android Studio

Download and install **Android Studio Quail 4 for Windows** using the following installer:

[Download Android Studio Quail 4 for Windows](https://edgedl.me.gvt1.com/android/studio/install/2026.1.4.7/android-studio-quail4-windows.exe)

Run the installer and follow the installation wizard.

During installation, keep the recommended components selected, including:

- Android Studio
- Android SDK
- Android SDK Platform
- Android SDK Platform-Tools
- Android Emulator

After Android Studio starts:

1. Open **Tools → SDK Manager**.
2. Make sure **Android SDK Platform 35** is installed.
3. Make sure **Android SDK Platform-Tools** is installed.
4. If you intend to use an emulator, open **Tools → Device Manager** and create an Android Virtual Device.

The default Android SDK location on Windows is usually:

```text
C:\Users\<your-user>\AppData\Local\Android\Sdk
```

---

## 4. Configure the Android SDK

If Gradle cannot find the Android SDK, create a `local.properties` file in the project root.

Example:

```properties
sdk.dir=C\:\\Users\\your-user\\AppData\\Local\\Android\\Sdk
```

Replace `your-user` with your Windows username.

For example:

```properties
sdk.dir=C\:\\Users\\cbalasoiu\\AppData\\Local\\Android\\Sdk
```

> `local.properties` is normally a machine-specific file and should not be committed to the repository.

---

## 5. Verify the Complete Environment

Open a **new PowerShell window** and run:

```powershell
java -version
$env:JAVA_HOME
adb.exe version
```

You should have:

- Oracle JDK 21 installed
- `JAVA_HOME` pointing to the Oracle JDK 21 directory
- Android Studio installed
- Android SDK API 35 installed
- Android SDK Platform-Tools available

---

## 6. Build the Application

Run the commands below from the project root in PowerShell.

Build the debug APK:

```powershell
.\gradlew :app:assembleDebug
```

The debug APK is generated at:

```text
app\build\outputs\apk\debug\app-debug.apk
```

---

## 7. Connect an Android Device or Emulator

Connect an Android device using USB, or start an Android emulator.

For a physical device:

1. Enable **Developer Options**.
2. Enable **USB debugging**.
3. Connect the device to the computer.
4. Accept the USB debugging authorization prompt on the device.

Check the connected devices:

```powershell
adb.exe devices
```

Example:

```text
List of devices attached
XXXXXXXXXXXX    device
```

The device should appear with the status:

```text
device
```

If it appears as:

```text
unauthorized
```

unlock the phone and accept the USB debugging authorization prompt.

---

## 8. Install the Debug APK

After successfully building the application, install it with:

```powershell
adb.exe install -r .\app\build\outputs\apk\debug\app-debug.apk
```

The `-r` option reinstalls the application while attempting to preserve its existing application data.

---

## 9. Inspect Application Logs

Start Android logcat:

```powershell
adb.exe logcat
```

Stop `adb logcat` with:

```text
Ctrl+C
```

---

## 10. Complete Build and Installation Flow

Once the environment is configured, the normal development workflow is:

```powershell
.\gradlew :app:assembleDebug

adb.exe devices

adb.exe install -r .\app\build\outputs\apk\debug\app-debug.apk

adb.exe logcat
```

The debug APK is located at:

```text
app\build\outputs\apk\debug\app-debug.apk
```

## Requirements Summary

| Component | Requirement |
|---|---|
| Operating System | Windows |
| Java | Oracle JDK 21 |
| `JAVA_HOME` | `C:\Program Files\Java\jdk-21` |
| IDE | Android Studio Quail 4 |
| Android SDK | API 35 |
| Platform Tools | `adb` |
| Device | Android device or emulator |
| Speech Recognition | Sherpa-ONNX |
| Network | Not required for speech recognition |