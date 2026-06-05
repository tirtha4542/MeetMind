import os
from faster_whisper import WhisperModel

whisper_model = os.getenv("WHISPER_MODEL", "small")
_model = None
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "saaras:v2.5")

def load_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {whisper_model}...")
        _model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
        print("Model loaded successfully.")
    return _model

def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
    model = load_model()
    task = "translate" if translate else "transcribe"
    segments, info = model.transcribe(chunk_path, task=task)
    return " ".join(segment.text for segment in segments)

def transcribe_all(chunks: list, translate: bool = False) -> str:
    full_transcript = ""
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}: {chunk}...")
        transcript = transcribe_chunk(chunk, translate)
        full_transcript += transcript + " "
    return full_transcript.strip()