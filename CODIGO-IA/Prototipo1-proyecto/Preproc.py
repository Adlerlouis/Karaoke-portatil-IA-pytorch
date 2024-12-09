#!/usr/bin/env python3
import os
from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


# Configuración de rutas
DATASET_PATH = Path("/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/txt_spect")
NPY_PATH = Path("/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/NPY")
SPECTROGRAM_DIR = Path("/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/train_data")

class SpectrogramDatasetHandler:
    def __init__(self):
        self.dataset_path = DATASET_PATH
        self.npy_path = NPY_PATH
        self.spectrogram_dir = SPECTROGRAM_DIR

    def __check_npy_dataset_exists(self) -> bool:
        """
        Verifica si el dataset en formato NPY existe.
        """
        return self.npy_path.exists() and len(list(self.npy_path.glob("*.npy"))) > 0

    def __load_spectrogram_data(self, npy_file_path):
        """
        Carga los datos de espectrogramas desde un archivo .npy.
        """
        try:
            data = np.load(npy_file_path, allow_pickle=True).item()
            return data
        except Exception as e:
            print(f"Error al cargar {npy_file_path}: {e}")
            return None

    def __create_train_data_dir(self):
        """
        Crea un directorio con los datos de entrenamiento a partir de espectrogramas cargados.
        """
        if self.spectrogram_dir.exists():
            print(f"El directorio de datos de entrenamiento ya existe: {self.spectrogram_dir}")
            return

        self.spectrogram_dir.mkdir(parents=True, exist_ok=True)
        print("Creando datos de entrenamiento...")

        counter = 0
        for npy_file in self.npy_path.glob("*.npy"):
            data = self.__load_spectrogram_data(npy_file)
            if data is None:
                continue

            spectrogram_db = data.get("spectrogram_db")
            frequencies = data.get("frequencies")
            times = data.get("times")

            if spectrogram_db is None or frequencies is None or times is None:
                print(f"Datos faltantes en: {npy_file}")
                continue

            # Dividir en segmentos para entrenamiento
            num_segments = spectrogram_db.shape[1] // 100  # Ajustar según necesidad
            for segment_id in range(num_segments):
                start_idx = segment_id * 100
                end_idx = (segment_id + 1) * 100

                segment = spectrogram_db[:, start_idx:end_idx]
                file_name = f"{npy_file.stem}_segment_{segment_id}.npy"
                segment_path = self.spectrogram_dir / file_name
                np.save(segment_path, {"spectrogram": segment, "frequencies": frequencies, "times": times[start_idx:end_idx]})
                counter += 1

        print(f"Se generaron {counter} segmentos de entrenamiento.")

    def load_training_data(self):
        """
        Carga y genera datos de entrenamiento desde archivos NPY.
        """
        if not self.__check_npy_dataset_exists():
            print("No se encontraron datos NPY. Asegúrate de que existan espectrogramas generados.")
            return

        if not self.spectrogram_dir.exists():
            self.__create_train_data_dir()
        else:
            print(f"Datos de entrenamiento ya generados en: {self.spectrogram_dir}")


# Integración con el sistema original
def main():
    handler = SpectrogramDatasetHandler()
    handler.load_training_data()


if __name__ == "__main__":
    main()

