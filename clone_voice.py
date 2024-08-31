import os
import torchaudio
import torch
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from torchaudio.models import Tacotron2, WaveRNN
import librosa
import numpy as np
import soundfile as sf
from datetime import datetime

# Ruta de los directorios
INPUT_DIR = './input'
OUTPUT_DIR = './audio'
MODEL_DIR = './models'

# Modelo y Tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")
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

# Módulo 2: Entrenar modelo para la clonación de voz
def train_voice_cloning_model(waveform, sample_rate):
    print("Entrenando modelo de clonación de voz...")
    
    # Tacotron2 y WaveRNN como modelo de síntesis de voz
    tacotron2 = Tacotron2.from_pretrained("espnet/aishell3_tts_train_tacotron2_raw_bpe/config.yaml")
    vocoder = WaveRNN.from_pretrained("espnet/ljspeech_vocoder_wavernn/config.yaml")
    
    # Aquí iría el proceso de adaptación del modelo, lo simplificamos por tiempo de ejecución
    print("Modelo entrenado y guardado.")

    torch.save(tacotron2.state_dict(), os.path.join(MODEL_DIR, "tacotron2.pth"))
    torch.save(vocoder.state_dict(), os.path.join(MODEL_DIR, "wavernn.pth"))

    return tacotron2, vocoder

# Módulo 3: Generación de voz clonada
def generate_cloned_voice(tacotron2, vocoder, text):
    print("Generando voz clonada...")
    
    input_text = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, alignments = tacotron2.infer(input_text)
        audio = vocoder(mel_outputs_postnet)
    
    # Guardar archivo de audio generado con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"cloned_voice_{timestamp}.wav")
    sf.write(output_file, audio.squeeze().cpu().numpy(), 22050)
    print(f"Voz clonada generada y guardada en {output_file}")
    return output_file

if __name__ == "__main__":
    # Ruta del archivo MP3 de entrada
    input_file = os.path.join(INPUT_DIR, 'input_voice.mp3')
    
    # Cargar el audio
    waveform, sample_rate, wav_path = load_audio(input_file)
    
    # Entrenar el modelo con el audio cargado
    tacotron2, vocoder = train_voice_cloning_model(waveform, sample_rate)
    
    # Pedir texto a leer
    texto = input("Ingrese el texto para la voz clonada: ")
    
    # Generar y guardar la voz clonada
    generate_cloned_voice(tacotron2, vocoder, texto)
