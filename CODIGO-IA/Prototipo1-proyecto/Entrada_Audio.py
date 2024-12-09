#!/usr/bin/env python3
from pathlib import Path
import sounddevice as sd
import numpy as np
import wave
import os
from pydub import AudioSegment

AUDIO_OUTPUT_PATH =Path("/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/AUDIOENT")
# Configuración del audio
DURATION = 10  # Duración de la grabación en segundos
RATE = 44100  # Frecuencia de muestreo
CHANNELS = 2  # Stereo
OUTPUT_FILENAME = "output.mp3" #tipo de archivo a guardar 

# Lista los dispositivos disponibles
print(sd.query_devices())

# Selecciona el dispositivo Bluetooth como entrada
input_device_index = int(input("AC:6C:90:96:A4:C6: "))

print("Grabando...")
audio_data = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=CHANNELS, dtype='int16',
                    device=input_device_index)
sd.wait()
print("Grabación finalizada.")

AUDIO_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)  # Crear la carpeta 
# Generar un nombre único incremental para el archivo MP3
existing_files = list(AUDIO_OUTPUT_PATH.glob("audio_*.mp3"))
file_count = len(existing_files) + 1  # Contamos cuántos archivos MP3 existen y sumamos 1
output_mp3_filename = f"audio_{file_count}.mp3"
output_mp3_path = AUDIO_OUTPUT_PATH / output_mp3_filename
# Guarda el audio en un archivo WAV
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)  # 16 bits = 2 bytes
    wf.setframerate(RATE)
    wf.writeframes(audio_data.tobytes())

# GUARDAMOS EL AUDIO EN LA CARPETA DE NUESTRO DISCO DURO DEFINIDA
#print(f"Archivo guardado como {OUTPUT_FILENAME}")

"""
# Generar un nombre único incremental para el archivo WAV
existing_files = list(AUDIO_OUTPUT_PATH.glob("audio_*.wav"))
file_count = len(existing_files) + 1
output_wav_filename = f"audio_{file_count}.wav"
output_wav_path = AUDIO_OUTPUT_PATH / output_wav_filename

# Guardar el audio como WAV
with wave.open(str(output_wav_path), 'wb') as wf:  # Convertimos Path a str
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)  # 16 bits = 2 bytes
    wf.setframerate(RATE)
    wf.writeframes(audio_data.tobytes())

print(f"Archivo WAV guardado en: {output_wav_path}")

# Convertir el archivo WAV a MP3
output_mp3_filename = f"audio_{file_count}.mp3"
output_mp3_path = AUDIO_OUTPUT_PATH / output_mp3_filename

audio = AudioSegment.from_wav(str(output_wav_path))
audio.export(str(output_mp3_path), format="mp3")

# Eliminar el archivo WAV si solo deseas conservar el MP3
output_wav_path.unlink()

print(f"Archivo MP3 guardado en: {output_mp3_path}")

""" 


# Guardar el archivo WAV temporal
output_wav_filename = f"audio_{file_count}.wav"  # Usamos el mismo contador para el archivo WAV
output_wav_path = AUDIO_OUTPUT_PATH / output_wav_filename

# Guardamos el archivo WAV temporal
with wave.open(str(output_wav_path), 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)  # 16 bits = 2 bytes
    wf.setframerate(RATE)
    wf.writeframes(audio_data.tobytes())  # Escribir los datos de audio

# Convertir el archivo WAV a MP3
audio = AudioSegment.from_wav(str(output_wav_path))
audio.export(str(output_mp3_path), format="mp3")

# Eliminar el archivo WAV después de la conversión a MP3
output_wav_path.unlink()

print(f"Archivo MP3 guardado en: {output_mp3_path}")

