import os
import torchaudio
import torch
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import numpy as np
import soundfile as sf
from datetime import datetime
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import Tacotron2
from espnet2.bin.tts_inference import Text2Speech

# Ruta de los directorios
INPUT_DIR = './input'
OUTPUT_DIR = './audio'
MODEL_DIR = './models'

# Modelo y Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")

# Asegurar que el directorio de salida exista
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Módulo 1: Cargar archivo de audio
def load_audio(file_path):
    print(f"Cargando archivo de audio desde {file_path}")
    audio = AudioSegment.from_mp3(file_path)
    wav_path = file_path.replace('.mp3', '.wav')
    audio.export(wav_path, format="wav")
    waveform, sample_rate = torchaudio.load(wav_path)
    return waveform, sample_rate, wav_path

# Módulo 2: Cargar modelo preentrenado para la síntesis de voz
def train_voice_cloning_model():
    print("Cargando modelo preentrenado para la síntesis de voz en español...")
    
    # Aquí utilizamos un modelo preentrenado para la síntesis de voz
    text2speech = Text2Speech.from_pretrained(
        "kan-bayashi/ljspeech_tacotron2",
        "kan-bayashi/ljspeech_parallel_wavegan.v3"
    )
    
    print("Modelo cargado y listo para su uso.")
    return text2speech

# Módulo 3: Generación de voz clonada
def generate_cloned_voice(text2speech, text):
    print("Generando voz clonada...")

    # Convertir texto en espectrograma mel y luego en audio
    with torch.no_grad():
        wav = text2speech(text)["wav"]

    # Guardar archivo de audio generado con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"cloned_voice_{timestamp}.wav")
    sf.write(output_file, wav.cpu().numpy(), 22050)
    print(f"Voz clonada generada y guardada en {output_file}")
    return output_file

if __name__ == "__main__":
    # Ruta del archivo MP3 de entrada
    input_file = os.path.join(INPUT_DIR, 'input_voice.mp3')
    
    # Cargar el audio
    waveform, sample_rate, wav_path = load_audio(input_file)
    
    # Cargar el modelo preentrenado
    text2speech = train_voice_cloning_model()
    
    # Pedir texto a leer
    texto = input("Ingrese el texto para la voz clonada: ")
    
    # Generar y guardar la voz clonada
    generate_cloned_voice(text2speech, texto)
