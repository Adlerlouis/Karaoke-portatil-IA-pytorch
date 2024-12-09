#!/usr/bin/env python3
import os
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from pydub import AudioSegment


# Cargar el audio de referencia del entrenamiento
ref_filename = 'Cancion3.mp3'
est_rate = 'Cancion7.mp3'

# Apuntamos al directorio
ruta = f'/home/adler/PYTHON3INTELIGENCEARTIFICAL/samples/CANCIONES/{ref_filename}'
ruta2 = f'/home/adler/PYTHON3INTELIGENCEARTIFICAL/samples/CANCIONES/{est_rate}'

# Cargar el archivo de referencia
if os.path.exists(ruta):
    audio = AudioSegment.from_file(ruta)
    ref_audio = np.array(audio.get_array_of_samples())
    ref_rate = audio.frame_rate
    print(f"Archivo {ref_filename} cargado correctamente.")
else:
    print(f"El archivo {ref_filename} no existe.")

# Cargar el archivo de estimado
if os.path.exists(ruta2):
    audio = AudioSegment.from_file(ruta2)
    est_audio = np.array(audio.get_array_of_samples())
    est_rate = audio.frame_rate
    print(f"Archivo {est_rate} cargado correctamente.")
else:
    print(f"El archivo {est_rate} no existe.")


class Dataset_Handler:
    def __init__(self):
        # Definimos los directorios para la multimedia
        self.PATH_TO_DATASET_WAVS = "/home/adler/PYTHON3INTELIGENCEARTIFICAL/samples/CANCIONES"  # Ruta a tus archivos WAV si los tienes
        self.PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR = "/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/NPY"  # Ruta a los espectrogramas pre-generados
        self.IMG_OUTPUT_PATH = "/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/img"
        self.TXT_SPECT = "/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/txt_spect"
        self.PATH_TO_TRAIN_DATA_DIR = "./dataset/train_data"
    
    def validate_directories(self):
        """Verifica que las carpetas existan y tengan contenido."""
        directories = {
            "PATH_TO_DATASET_WAVS": self.PATH_TO_DATASET_WAVS,
            "PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR": self.PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR,
            "IMG_OUTPUT_PATH": self.IMG_OUTPUT_PATH,
            "TXT_SPECT": self.TXT_SPECT
        }

        all_valid = True
        for name, path in directories.items():
            print(f"Verificando: {name} -> {path}")
            if not os.path.exists(path):
                print(f"❌ Error: La carpeta {name} no existe en la ruta {path}")
                all_valid = False
            elif not os.listdir(path):
                print(f"⚠️ Advertencia: La carpeta {name} está vacía: {path}")
                all_valid = False
            else:
                print(f"✔ La carpeta {name} existe y tiene contenido.")
        return all_valid

    def store_new_wav(self, path, data, sample_rate=44100):
        """Guarda un archivo WAV en la ruta especificada."""
        if not path.endswith(".mp3"):
            path += ".mp3"
        sf.write(path, data, sample_rate)
        print(f"Archivo guardado en {path}")

    def cargar_espectrogramas_desde_directorio(self):
        """Carga espectrogramas pre-generados desde un directorio."""
        espectrogramas = {}
        for filename in os.listdir(self.PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR):
            if filename.endswith(".npy"):
                path = os.path.join(self.PATH_TO_TEMP_SPECTROGRAMS_FOR_TRAINING_DIR, filename)
                espectrogramas[filename] = np.load(path)
                print(f"Espectrograma cargado desde {path}")
        return espectrogramas

    def cargar_wavs_desde_directorio(self):
        """Carga archivos WAV desde un directorio."""
        wavs = {}
        for filename in os.listdir(self.PATH_TO_DATASET_WAVS):
            if filename.endswith(".wav"):  # Asegúrate de que sea un archivo WAV
                path = os.path.join(self.PATH_TO_DATASET_WAVS, filename)
                wavs[filename] = wavfile.read(path)
                print(f"Archivo WAV cargado desde {path}")
        return wavs

    def cargar_textos_desde_txt_spect(self):
        """Carga textos desde el directorio TXT_SPECT."""
        textos = {}
        for filename in os.listdir(self.TXT_SPECT):
            if filename.endswith(".txt"):
                path = os.path.join(self.TXT_SPECT, filename)
                with open(path, 'r', encoding='utf-8') as file:
                    textos[filename] = file.read()
                print(f"Texto cargado desde {path}")
        return textos

    def cargar_mp3_desde_directorio(self):
        """Carga archivos MP3 desde un directorio y los convierte a numpy arrays."""
        mp3_files = {}
        for filename in os.listdir(self.PATH_TO_DATASET_WAVS):
            if filename.endswith(".mp3"):  # Solo archivos MP3
                path = os.path.join(self.PATH_TO_DATASET_WAVS, filename)
                # Cargar MP3 usando pydub y convertirlo a numpy
                audio = AudioSegment.from_mp3(path)
                samples = np.array(audio.get_array_of_samples())
                mp3_files[filename] = (samples, audio.frame_rate)
                print(f"Archivo MP3 cargado desde {path}")
        return mp3_files

    def compute_difference(self, ref_rec: np.array, est_rec: np.array, weightage=[0.33, 0.33, 0.33]):
        """
        Calcula la diferencia entre dos señales en los dominios temporal y de frecuencia, 
        y sus respectivas diferencias de potencia.

        :param ref_rec: np.ndarray -> la grabación de referencia
        :param est_rec: np.ndarray -> la grabación estimada
        :param weightage: lista de ponderaciones para cada métrica.
        :return: la métrica total de diferencia ponderada.
        """
        ## Diferencia en el dominio del tiempo
        ref_time = np.correlate(ref_rec, ref_rec)
        inp_time = np.correlate(ref_rec, est_rec)
        diff_time = abs(ref_time - inp_time) / ref_time
        
        ## Diferencia en el dominio de frecuencia
        ref_freq = np.correlate(np.fft.fft(ref_rec), np.fft.fft(ref_rec))
        inp_freq = np.correlate(np.fft.fft(ref_rec), np.fft.fft(est_rec))
        diff_freq = complex(abs(ref_freq - inp_freq) / ref_freq).real

        ## Diferencia de potencia
        ref_power = np.sum(ref_rec ** 2)
        inp_power = np.sum(est_rec ** 2)
        diff_power = abs(ref_power - inp_power) / ref_power

        return float(
            weightage[0] * diff_time
            + weightage[1] * diff_freq
            + weightage[2] * diff_power
        )

    def cargar_segmento_audio(self, ruta, duracion_ms=10000):
        """
        Carga un segmento de un archivo de audio.
        :param ruta: Ruta del archivo de audio.
        :param duracion_ms: Duración del segmento en milisegundos.
        :return: Segmento de audio como numpy array y la tasa de muestreo.
        """
        audio = AudioSegment.from_file(ruta)
        segmento = audio[:duracion_ms]  # Cortar los primeros duracion_ms milisegundos
        muestras = np.array(segmento.get_array_of_samples())
        return muestras, segmento.frame_rate

    def comparar_audio(self, ref_filename, est_filename):
        """Compara dos archivos de audio MP3."""
        # Cargar segmentos de audio
        ref_audio, ref_rate = self.cargar_segmento_audio("Cancion3.mp3", duracion_ms=10000)
        est_audio, est_rate = self.cargar_segmento_audio("Cancion7.mp3", duracion_ms=10000)

        # Asegúrate de que las tasas de muestreo coincidan antes de comparar
        assert ref_rate == est_rate, "Las tasas de muestreo no coinciden."

        # Normalizar el tamaño si es necesario
        min_len = min(len(ref_audio), len(est_audio))
        ref_audio = ref_audio[:min_len]
        est_audio = est_audio[:min_len]

        # Comparar audios
        diferencia = self.compute_difference(ref_audio, est_audio)
        print(f"Diferencia entre audios: {diferencia}") 

        return diferencia

    def __create_train_data_dir(self):
        """Crea un directorio para datos de entrenamiento."""
        os.makedirs(self.PATH_TO_TRAIN_DATA_DIR, exist_ok=True)
        print(f"Directorio de datos de entrenamiento en {self.PATH_TO_TRAIN_DATA_DIR}")
        
        # Carga espectrogramas desde IMG_OUTPUT_PATH (si existe)
        espectrogramas = self.cargar_espectrogramas_desde_directorio()
        for nombre, espectrograma in espectrogramas.items():
            np.save(f'{self.PATH_TO_TRAIN_DATA_DIR}/{nombre}.npy', espectrograma)
        print("Espectrogramas guardados.")


