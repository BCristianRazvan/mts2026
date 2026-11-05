# Step 1 of 2: generates the clean synthetic speech for the test set with the
# Windows built-in speech synthesizer (offline). Output: 16 kHz, mono, 16-bit PCM.
#
# Text comes from transcripts.tsv, voice and speaking rate from conditions.tsv.
# The clean files go to testset\clean\. Step 2 (add_noise.py) adds car noise
# and writes the final test files to testset\.
#
# Usage (PowerShell, from the repository root):
#   powershell -ExecutionPolicy Bypass -File docs\stt-model-replacement\testset\generate_testset.ps1
#   uv run --no-project --python 3.12 --with numpy python docs\stt-model-replacement\testset\add_noise.py

Add-Type -AssemblyName System.Speech

$testsetDir = $PSScriptRoot
$cleanDir = Join-Path $testsetDir 'clean'
New-Item -ItemType Directory -Force -Path $cleanDir | Out-Null

$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)

$conditions = @{}
Import-Csv -Path (Join-Path $testsetDir 'conditions.tsv') -Delimiter "`t" | ForEach-Object {
    $conditions[$_.file] = $_
}

$synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer

Get-Content -Path (Join-Path $testsetDir 'transcripts.tsv') -Encoding UTF8 | ForEach-Object {
    if ($_.Trim() -eq '') { return }
    $name, $text = $_ -split "`t", 2
    $condition = $conditions[$name]
    if ($null -eq $condition) { throw "$name is missing in conditions.tsv" }

    # Rate: -10 (slow) to 10 (fast). 0 is the normal speed.
    $synthesizer.SelectVoice($condition.voice)
    $synthesizer.Rate = [int]$condition.rate
    $synthesizer.SetOutputToWaveFile((Join-Path $cleanDir $name), $format)
    $synthesizer.Speak($text)
    $synthesizer.SetOutputToNull()
    Write-Output "$name  voice=$($condition.voice)  rate=$($condition.rate)"
}

$synthesizer.Dispose()
