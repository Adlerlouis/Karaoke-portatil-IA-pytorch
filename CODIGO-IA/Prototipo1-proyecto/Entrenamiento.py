#!/usr/bin/env python3
import os
from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    Cropping2D,
    Lambda
)
from tensorflow.keras.models import Model

# Bloque de codificación
def encoder_block(inputs, filters):
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    p = MaxPooling2D((2, 2))(x)
    return x, p

# Bloque de decodificación con manejo dinámico de formas
def decoder_block(inputs, skip_connection, filters):
    x = UpSampling2D((2, 2))(inputs)
    x = Conv2D(filters, (2, 2), activation='relu', padding='same')(x)

    # Ajuste dinámico de las dimensiones
    def crop_to_match(skip, x):
        skip_shape = tf.shape(skip)
        x_shape = tf.shape(x)

        height_diff = skip_shape[1] - x_shape[1]
        width_diff = skip_shape[2] - x_shape[2]

        crop_height = (0, height_diff) if height_diff > 0 else (0, 0)
        crop_width = (0, width_diff) if width_diff > 0 else (0, 0)

        skip = tf.keras.layers.Cropping2D(cropping=(crop_height, crop_width))(skip)
        return skip

    # Calcular salida de la forma correcta para usar en Lambda
    def compute_output_shape(input_shapes):
        skip_shape, x_shape = input_shapes
        return x_shape

    # Usar Lambda para ajustar dinámicamente las dimensiones
    skip_connection = Lambda(
        lambda args: crop_to_match(*args),
        output_shape=lambda input_shapes: compute_output_shape(input_shapes)
    )([skip_connection, x])

    x = Concatenate()([x, skip_connection])
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    return x

# Arquitectura U-Net
def build_unet(input_shape):
    inputs = Input(shape=input_shape)

    # Codificador
    skip1, pool1 = encoder_block(inputs, 64)
    skip2, pool2 = encoder_block(pool1, 128)
    skip3, pool3 = encoder_block(pool2, 256)
    skip4, pool4 = encoder_block(pool3, 512)

    # Botella
    bottle = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    bottle = Conv2D(1024, (3, 3), activation='relu', padding='same')(bottle)

    # Decodificador
    up4 = decoder_block(bottle, skip4, 512)
    up3 = decoder_block(up4, skip3, 256)
    up2 = decoder_block(up3, skip2, 128)
    up1 = decoder_block(up2, skip1, 64)

    # Salida
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(up1)

    return Model(inputs, outputs)

# Crear el modelo
input_shape = (1025, None, 1)  # Dimensiones variables en el eje temporal
model = build_unet(input_shape)

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Resumen del modelo
model.summary()

