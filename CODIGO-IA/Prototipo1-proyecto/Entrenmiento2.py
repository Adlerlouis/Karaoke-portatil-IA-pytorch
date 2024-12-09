#!/usr/bin/env python3
import pandas as pd
import os
from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model, Input

# Ruta donde se encuentran los archivos .txt
TEXT_INPUT_PATH = Path("/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/txt_spect")

# Tamaño objetivo para el espectrograma
TARGET_LENGTH = 861  # Esto debe coincidir con la cantidad de frames que necesitas
TARGET_FREQUENCY_BINS = 1025  # Número de bins de frecuencia

# ==== FUNCIONES DE PROCESAMIENTO ====
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
    """
    freqs = np.loadtxt(freq_file)
    times = np.loadtxt(time_file)
    spectrogram_db = np.loadtxt(spec_file)

    # Recortar o rellenar el espectrograma
    spectrogram_db = pad_or_trim_spectrogram(spectrogram_db, target_length=TARGET_LENGTH)

    # Validaciones
    if spectrogram_db.shape[1] != TARGET_LENGTH:
        print(f"Warning: Expected {TARGET_LENGTH} time bins, but got {spectrogram_db.shape[1]}")
    if freqs.shape[0] != TARGET_FREQUENCY_BINS:
        print(f"Warning: Expected {TARGET_FREQUENCY_BINS} frequency bins, but got {freqs.shape[0]}")
    if times.shape[0] != TARGET_LENGTH:
        print(f"Warning: Expected {TARGET_LENGTH} time bins, but got {times.shape[0]}")

    return freqs, times, spectrogram_db

def load_all_spectrograms_from_txt(txt_folder):
    spectrograms = []
    for txt_file in txt_folder.glob("*.txt"):
        if "frecuencias" in txt_file.name:
            base_name = txt_file.stem.split("_frecuencias")[0]
            freq_file = txt_folder / f"{base_name}_frecuencias.txt"
            time_file = txt_folder / f"{base_name}_tiempos.txt"
            spec_file = txt_folder / f"{base_name}_espectrograma_db.txt"
            
            if freq_file.exists() and time_file.exists() and spec_file.exists():
                _, _, spectrogram = load_spectrogram_from_txt(freq_file, time_file, spec_file)
                spectrograms.append(spectrogram)
            else:
                print(f"Archivos faltantes para: {base_name}")
    
    return spectrograms

def prepare_spectrograms_for_unet(spectrograms):
    """
    Ajusta la forma de los espectrogramas para que sean compatibles con la U-Net.
    """
    spectrograms = np.array(spectrograms)
    spectrograms = np.expand_dims(spectrograms, axis=-1)  # Añadir canal
    return spectrograms

# ==== MODELO U-NET ====
def encoder_block(input_tensor, filters):
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, skip_tensor, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    x = layers.Concatenate()([x, skip_tensor])
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    return x

def build_unet(input_shape):
    inputs = Input(input_shape)
    
    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    
    # Bottleneck
    b = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    
    # Decoder
    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    
    # Output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d4)
    
    return Model(inputs, outputs)
#caracteristicas del modelo estadisiticamente hablando 


# ==== PROCESAMIENTO Y ENTRENAMIENTO ====
spectrograms = load_all_spectrograms_from_txt(TEXT_INPUT_PATH)
if len(spectrograms) > 0:
    spectrograms_prepared = prepare_spectrograms_for_unet(spectrograms)
    print(f"Espectrogramas preparados: {spectrograms_prepared.shape}")
    
    # Simulación de etiquetas binarias
    labels = np.random.randint(0, 2, spectrograms_prepared.shape)
    labels = to_categorical(labels, num_classes=2)
    
    # División de datos
    X_train, X_val, y_train, y_val = train_test_split(spectrograms_prepared, labels, test_size=0.2, random_state=42)
         #Crear el modelo
    input_shape = (1025, None, 1)  # Dimensiones variables en el eje temporal
    model = build_unet(input_shape)
    # Construcción y entrenamiento del modelo
    model = build_unet((TARGET_FREQUENCY_BINS, TARGET_LENGTH, 1))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)
    # Resumen del modelo
    model.summary()



else:
    print("No se encontraron espectrogramas válidos.")

