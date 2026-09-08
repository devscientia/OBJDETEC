
import speech_recognition as sr

# Initialize the recognizer
recognizer = sr.Recognizer()

# Load your offline audio file (must be .wav format)
with sr.AudioFile("somente_audio.wav") as source:
    print("Loading audio...")
    audio_data = recognizer.record(source)

try:
    # Perform offline speech recognition using PocketSphinx
    print("Transcribing offline with PocketSphinx...")
    text = recognizer.recognize_sphinx(audio_data)
    print(f"Result: {text}")
except sr.UnknownValueError:
    print("PocketSphinx could not understand the audio.")
except sr.RequestError as e:
    print(f"Sphinx error; {e}")