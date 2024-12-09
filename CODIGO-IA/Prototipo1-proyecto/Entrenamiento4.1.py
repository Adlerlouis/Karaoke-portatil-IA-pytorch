#!/usr/bin/env python3
import pandas as pd
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model, Input

# ==== CONFIGURACIONES ====
TEXT_INPUT_PATH = Path("/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/txt_spect")  # Ruta a los archivos
TARGET_LENGTH = 654  # Longitud de frames en el espectrograma
TARGET_FREQUENCY_BINS = 1025  # Número de bins de frecuencia
BATCH_SIZE = 4  # Tamaño de los lotes reducido para Raspberry Pi
EPOCHS = 3  # Número de épocas de entrenamiento

# ==== FUNCIONES DE PROCESAMIENTO ====
def pad_or_trim_spectrogram(spectrogram, target_length=TARGET_LENGTH):
    """Recorta o rellena un espectrograma para que tenga una longitud uniforme."""
    if spectrogram.shape[1] > target_length:
        spectrogram = spectrogram[:, :target_length]
    elif spectrogram.shape[1] < target_length:
        padding = target_length - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant', constant_values=0)
    return spectrogram

def load_spectrogram_from_txt(freq_file, time_file, spec_file):
    """Carga espectrogramas desde archivos .txt."""
    freqs = np.loadtxt(freq_file)
    times = np.loadtxt(time_file)
    spectrogram_db = np.loadtxt(spec_file)

    spectrogram_db = pad_or_trim_spectrogram(spectrogram_db)

    # Validaciones
    if spectrogram_db.shape[1] != TARGET_LENGTH:
        print(f"Warning: Expected {TARGET_LENGTH} time bins, but got {spectrogram_db.shape[1]}")
    if freqs.shape[0] != TARGET_FREQUENCY_BINS:
        print(f"Warning: Expected {TARGET_FREQUENCY_BINS} frequency bins, but got {freqs.shape[0]}")

    return spectrogram_db

def load_batch_spectrograms(txt_folder, batch_start, batch_size):
    """Carga un lote de espectrogramas desde archivos .txt."""
    txt_files = list(txt_folder.glob("*.txt"))
    spectrograms = []
    batch_files = txt_files[batch_start:batch_start + batch_size]

    for txt_file in batch_files:
        if "frecuencias" in txt_file.name:
            base_name = txt_file.stem.split("_frecuencias")[0]
            freq_file = txt_folder / f"{base_name}_frecuencias.txt"
            time_file = txt_folder / f"{base_name}_tiempos.txt"
            spec_file = txt_folder / f"{base_name}_espectrograma_db.txt"

            if freq_file.exists() and time_file.exists() and spec_file.exists():
                spectrogram = load_spectrogram_from_txt(freq_file, time_file, spec_file)
                spectrograms.append(spectrogram)
            else:
                print(f"Archivos faltantes para: {base_name}")

    if len(spectrograms) > 0:
        spectrograms = np.array(spectrograms)
        spectrograms = np.expand_dims(spectrograms, axis=-1)  # Añadir canal
    return spectrograms

# ==== MODELO U-NET ====
def encoder_block(input_tensor, filters):
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, skip_tensor, num_filters):
    """
    Bloque de decodificación que incluye una capa de redimensionamiento para igualar las dimensiones antes de concatenar.
    """
    input_tensor = layers.Resizing(
        height=skip_tensor.shape[1],  # Altura del tensor del encoder
        width=skip_tensor.shape[2],   # Ancho del tensor del encoder
        interpolation="bilinear"
    )(input_tensor)

    x = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.Concatenate()([x, skip_tensor])  # Concatenar con el tensor del encoder
    x = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
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

    # Output (con 2 clases: voz e instrumento)
    outputs = layers.Conv2D(2, (1, 1), activation='softmax')(d4)

    return Model(inputs, outputs)

# ==== ENTRENAMIENTO ====
if __name__ == "__main__":
    total_files = len(list(TEXT_INPUT_PATH.glob("*.txt"))) // 3
    steps_per_epoch = total_files // BATCH_SIZE

    # Crear modelo
    model = build_unet((TARGET_FREQUENCY_BINS, TARGET_LENGTH, 1))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Generador de datos por lotes
    def batch_generator(txt_folder, batch_size, total_files):
        for batch_start in range(0, total_files, batch_size):
            spectrograms = load_batch_spectrograms(txt_folder, batch_start, batch_size)
            
            # Aquí deberías tener un sistema de etiquetado real para voz vs instrumentos
            labels = np.random.randint(0, 2, spectrograms.shape[:3])  # Etiquetas binarias para voz/instrumentos
            labels = to_categorical(labels, num_classes=2)  # Convertir a formato one-hot
            yield spectrograms, labels

    train_gen = batch_generator(TEXT_INPUT_PATH, BATCH_SIZE, total_files)

    # Entrenamiento del modelo
    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS
    )

    # Guardar el modelo
    model.save("unet_model_separacion_voz_instrumentos.h5")
    print("Entrenamiento completado y modelo guardado.")

