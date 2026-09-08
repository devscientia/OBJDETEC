##########################################################
# Teste de transcricao de audio usando a biblioteca SpeechRecognition
#
#
############################################################33

import speech_recognition as sr
from pydub import AudioSegment


#file_audio = sr.AudioFile("/workspaces/OBJDETEC/atend_teste_01.wav")
# carregando o canal de audio em mp3
src=(r"/workspaces/OBJDETEC/atende03.mp3")


# converter de mp3 para wav
sound = AudioSegment.from_mp3(src)
sound.export("atende03.wav", format="wav")
file_audio = sr.AudioFile("atende03.wav")


# Usar o audio em WAV para a lib de transcricao 
r = sr.Recognizer()
with file_audio as source:
   audio_text = r.record(source)
   text = r.recognize_google(audio_text,language='pt-BR')


print('-----------------------------------------------------------------')
print('::Confira aqui o texto extraido do audio : ', text)
print('-----------------------------------------------------------------')
