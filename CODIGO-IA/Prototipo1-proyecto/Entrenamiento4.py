#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model, Input
import librosa

# ==== CONFIGURACIONES ====
TEXT_INPUT_PATH = Path("/home/adler/PYTHON3INTELIGENCEARTIFICAL/samples")  # Ruta a los archivos de audio
TARGET_LENGTH = 654  # Longitud de frames en el espectrograma
TARGET_FREQUENCY_BINS = 1025  # Número de bins de frecuencia
BATCH_SIZE = 4  # Tamaño de los lotes
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

def load_spectrogram(file_path):
    """Carga un espectrograma desde un archivo .wav y lo convierte a escala logarítmica."""
    audio, sr = librosa.load(file_path, sr=22050)
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    magnitude, phase = np.abs(stft), np.angle(stft)
    log_magnitude = librosa.amplitude_to_db(magnitude)
    log_magnitude = pad_or_trim_spectrogram(log_magnitude)
    return log_magnitude, phase

def batch_generator(mix_path, batch_size, total_files):
    """Generador de lotes de entrenamiento."""
    mix_files = list(mix_path.glob("*.mp3"))  # Lista de archivos .mp3
    for batch_start in range(0, total_files, batch_size):
        batch_mix = mix_files[batch_start:batch_start + batch_size]
        mix_spectrograms = []
        for mix_file in batch_mix:
            # Procesar cada archivo sin verificar nombres específicos
            mix_spec, _ = load_spectrogram(mix_file)

            # Aquí puedes optar por lo que desees: usar la misma mezcla como objetivo de salida
            # Por ejemplo, simplemente podrías tratar de predecir la separación entre voz e instrumentos
            # para simplificar, se usa la misma mezcla como objetivo.
            mix_spectrograms.append(mix_spec)

        if mix_spectrograms:  # Solo devuelve lotes válidos
            mix_spectrograms = np.expand_dims(np.array(mix_spectrograms), axis=-1)
            # Asumimos que la salida es la mezcla misma como ejemplo
            targets = mix_spectrograms  # Aquí es donde se podría agregar un objetivo diferente
            yield mix_spectrograms, targets

# ==== MODELO U-NET ====
def encoder_block(input_tensor, filters):
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, skip_tensor, num_filters):
    input_tensor = layers.Resizing(skip_tensor.shape[1], skip_tensor.shape[2])(input_tensor)
    x = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.Concatenate()([x, skip_tensor])
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

    # Output para las fuentes (voz e instrumentos)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', name='source_outputs')(d4)

    return Model(inputs, outputs)

# ==== ENTRENAMIENTO ====
if __name__ == "__main__":
    total_files = len(list(TEXT_INPUT_PATH.glob("*.mp3")))
    if total_files == 0:
        raise ValueError(f"No se encontraron archivos en {TEXT_INPUT_PATH}. Verifica la ruta y el contenido.")

    steps_per_epoch = total_files // BATCH_SIZE

    # Crear modelo
    model = build_unet((TARGET_FREQUENCY_BINS, TARGET_LENGTH, 1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Generador de datos por lotes
    train_gen = batch_generator(TEXT_INPUT_PATH, BATCH_SIZE, total_files)

    # Validación inicial del generador
    try:
        sample_data = next(train_gen)
    except StopIteration:
        raise ValueError("El generador no está produciendo datos. Verifica las rutas y los archivos.")

    # Entrenamiento del modelo
    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS
    )

    # Guardar el modelo
    model.save("unet_model_audio_separation.h5")
    print("Entrenamiento completado y modelo guardado.")

