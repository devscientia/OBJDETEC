
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
# You must generate a token from hf.co/settings/tokens and accept user conditions for pyannote/speaker-diarization-community-1
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", use_auth_token="YOUR_HUGGINGFACE_ACCESS_TOKEN")
# Run the pipeline on an audio file
diarization = pipeline("audio.wav")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"[{turn.start:.1f}s - {turn.end:.1f}s] {speaker}")



