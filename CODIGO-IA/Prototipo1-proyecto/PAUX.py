#!/usr/bin/env python3
import os
from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Ruta donde se encuentran los archivos .txt
TEXT_INPUT_PATH = Path("/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/txt_spect")

# Tamaño objetivo para el espectrograma
TARGET_LENGTH = 861  # Esto debe coincidir con la cantidad de frames que necesitas
TARGET_FREQUENCY_BINS = 1025  # Número de bins de frecuencia

def pad_or_trim_spectrogram(spectrogram, target_length=TARGET_LENGTH):
    """
    Recorta o rellena un espectrograma para que tenga la longitud deseada en el eje temporal.
    """
    # Recortar si el espectrograma es más largo
    if spectrogram.shape[1] > target_length:
        spectrogram = spectrogram[:, :target_length]
    # Rellenar con ceros si el espectrograma es más corto
    elif spectrogram.shape[1] < target_length:
        padding = target_length - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant', constant_values=0)
    
    return spectrogram

def load_spectrogram_from_txt(freq_file, time_file, spec_file):
    """
    Carga los datos de frecuencia, tiempo y espectrograma desde archivos .txt.
    Solo valida las dimensiones sin hacer cambios, pero recorta o rellena los espectrogramas.
    """
    # Cargar los archivos de texto
    freqs = np.loadtxt(freq_file)
    times = np.loadtxt(time_file)
    spectrogram_db = np.loadtxt(spec_file)

    # Recortar o rellenar el espectrograma para que tenga la longitud correcta
    spectrogram_db = pad_or_trim_spectrogram(spectrogram_db, target_length=TARGET_LENGTH)

    # Verificar las dimensiones de los datos
    print(f"Shape of freqs: {freqs.shape}")
    print(f"Shape of times: {times.shape}")
    print(f"Shape of spectrogram_db: {spectrogram_db.shape}")

    # Validar que las dimensiones sean correctas
    if spectrogram_db.shape[1] != TARGET_LENGTH:
        print(f"Warning: Expected {TARGET_LENGTH} time bins, but got {spectrogram_db.shape[1]}")
    if freqs.shape[0] != TARGET_FREQUENCY_BINS:
        print(f"Warning: Expected {TARGET_FREQUENCY_BINS} frequency bins, but got {freqs.shape[0]}")
    if times.shape[0] != TARGET_LENGTH:
        print(f"Warning: Expected {TARGET_LENGTH} time bins, but got {times.shape[0]}")

    return freqs, times, spectrogram_db

def load_all_spectrograms_from_txt(txt_folder):
    spectrograms = []
    freqs_list = []
    times_list = []
    
    for txt_file in txt_folder.glob("*.txt"):
        if "frecuencias" in txt_file.name:
            # Buscar los archivos correspondientes para cada muestra de audio
            base_name = txt_file.stem.split("_frecuencias")[0]
            freq_file = txt_folder / f"{base_name}_frecuencias.txt"
            time_file = txt_folder / f"{base_name}_tiempos.txt"
            spec_file = txt_folder / f"{base_name}_espectrograma_db.txt"
            
            # Asegurar que los archivos existen
            if freq_file.exists() and time_file.exists() and spec_file.exists():
                freqs, times, spectrogram = load_spectrogram_from_txt(freq_file, time_file, spec_file)
                spectrograms.append(spectrogram)
                freqs_list.append(freqs)
                times_list.append(times)
            else:
                print(f"Archivos faltantes para: {base_name}")
    
    # No intentamos convertir a numpy si no son consistentes
    return freqs_list, times_list, spectrograms

# Cargar y preprocesar todos los espectrogramas
freqs_list, times_list, spectrograms = load_all_spectrograms_from_txt(TEXT_INPUT_PATH)

# Verificar las formas de los espectrogramas cargados
print(f"Total de archivos cargados: {len(freqs_list)}")
print(f"Shape of freqs_list: {len(freqs_list)}")
print(f"Shape of times_list: {len(times_list)}")
print(f"Shape of spectrograms: {len(spectrograms)}")

# Mostrar un resumen de las dimensiones
if len(spectrograms) > 0:
    print(f"Primer espectrograma shape: {spectrograms[0].shape}")
    print(f"Primer freqs shape: {freqs_list[0].shape}")
    print(f"Primer times shape: {times_list[0].shape}")

