
###############################################333
# Diarization with pyannote.audio

# Etapas
# Install ffmpeg (aplicativo executal que deve ser instalado no SO)
# pip install ffmpeg-python (lib para chamar o ff mpeg)
# pip install torch (lib para processamento de dados, necessário para o pyannote)
# pip install torch torchaudio
#----------------------------------------------------------
# Instalando o pyannote.audio e dependências
# pip install pyannote.audio
# pip install torch torchaudio
# pip install pyannote.pipeline
# pip install pyannote.metrics
# pip install pyannote.database
# pip install pyannote.core

from pyannote.audio import Pipeline
import torchaudio

from dotenv import load_dotenv
import os

# Load the .env file and set the variables
load_dotenv()


# Importa e coloca dentro da sessao
HF_TOKEN = os.getenv("HF_TOKEN")  
    

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN # Changed from use_auth_token
)


# Run the pipeline on an audio file
diarization = pipeline("atend_teste_01.wav")


for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"[{turn.start:.1f}s - {turn.end:.1f}s] {speaker}")



