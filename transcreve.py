# Codigo de transcricao

import assemblyai as aai

# Set your API key in the script if not using environment variables
# aai.settings.api_key = "YOUR_API_KEY"

transcriber = aai.Transcriber()

# Provide a local file path or a publicly accessible URL
audio_file = "somente_audio.mp3" # Replace with your audio file URL or path

print(f"Starting transcription for: {audio_file}")
# Uses universal-3-pro for en, es, de, fr, it, pt. Else uses universal-2 for support across all other languages
config = aai.TranscriptionConfig(speech_models=["universal-3-pro", "universal-2"], language_detection=True)

transcript = aai.Transcriber(config=config).transcribe(audio_file)

if transcript.status == "error":
  raise RuntimeError(f"Transcription failed: {transcript.error}")


print('--------------------------------------------')
print(transcript.text)
print('--------------------------------------------')

