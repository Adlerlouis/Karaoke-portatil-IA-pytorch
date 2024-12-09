#!/usr/bin/env python3
import numpy as np
np.float = float  # Solución temporal para compatibilidad
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import mir_eval
from pathlib import Path
import os
from scipy.ndimage import zoom
import soundfile as sf
# ==== CONFIGURACIONES ====
MODEL_PATH = "unet_model_separacion_voz_instrumentos.h5"  # Ruta al modelo entrenado
TEST_AUDIO_PATH = "/home/adler/PYTHON3INTELIGENCEARTIFICAL/samples/04-Cancion3.mp3"  # Ruta al archivo de audio mixto
OUTPUT_DIR = Path("/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/separados/")  # Directorio para guardar audios separados
os.makedirs(OUTPUT_DIR, exist_ok=True)
TARGET_LENGTH = 654

# Funciones auxiliares
def pad_or_trim_spectrogram(spectrogram, target_length=654):
    if spectrogram.shape[1] > target_length:
        spectrogram = spectrogram[:, :target_length]
    elif spectrogram.shape[1] < target_length:
        padding = target_length - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant', constant_values=0)
    return spectrogram

def resize_mask(mask, target_length):
    factor = target_length / mask.shape[1]
    return zoom(mask, (1, factor), order=1)

def apply_mask_and_reconstruct(spectrogram, mask, sr):
    spectrogram = pad_or_trim_spectrogram(spectrogram, target_length=mask.shape[1])
    masked_spectrogram = spectrogram * mask
    audio = librosa.griffinlim(masked_spectrogram)
    return audio

def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    spectrogram = np.abs(librosa.stft(y, n_fft=2048))
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    log_spectrogram = pad_or_trim_spectrogram(log_spectrogram, TARGET_LENGTH)
    return log_spectrogram, spectrogram, sr

# Cargar modelo
model = load_model(MODEL_PATH)
print("Modelo cargado.")

# Procesar audio
test_spectrogram_db, test_spectrogram, sr = audio_to_spectrogram(TEST_AUDIO_PATH)
test_spectrogram_db = np.expand_dims(test_spectrogram_db, axis=(0, -1))

# Predicciones
prediction = model.predict(test_spectrogram_db)
mask_voice = prediction[0, :, :, 0]
mask_instruments = 1 - mask_voice

# Ajustar máscaras
mask_voice_resized = resize_mask(mask_voice, test_spectrogram.shape[1])
mask_instruments_resized = resize_mask(mask_instruments, test_spectrogram.shape[1])

# Reconstruir audio
voice_audio = apply_mask_and_reconstruct(test_spectrogram, mask_voice_resized, sr)
instrument_audio = apply_mask_and_reconstruct(test_spectrogram, mask_instruments_resized, sr)
sf.write(os.path.join(OUTPUT_DIR, "voz_separada.mp3"), voice_audio, sr)
sf.write(os.path.join(OUTPUT_DIR, "instrumentos_separados.mp3"), instrument_audio, sr)
print("Archivos separados guardados.")

