#!/usr/bin/env python3  
import os
import numpy as np
import soundfile as sf                                                          
from scipy.io import wavfile                                                
from pydub import AudioSegment    
class DatasetHandlerTxt:
    def __init__(self):
        self.PATH_TO_TXT_SPECT = "/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/txt_spect"
        self.PATH_TO_TXT_FROM_NPY = "/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/txt_from_npy"

    def cargar_txt_desde_directorio(self, directory):
        data = {}
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):  
                try:
                    path = os.path.join(directory, filename)
                    with open(path, "r") as file:
                        lines = file.readlines()

                    clean_data = []
                    for line in lines:
                        try:
                            row = [float(x) for x in line.split()]
                            clean_data.append(row)
                        except ValueError:
                            continue

                    if clean_data:
                        data_array = np.array(clean_data)
                        data[filename] = data_array
                        print(f"Cargado {filename}: {data_array.shape} elementos.")
                    else:
                        print(f"⚠️ Archivo {filename} no contiene datos válidos.")
                except Exception as e:
                    print(f"⚠️ Error cargando {filename}: {e}")
        return data

    def cargar_todos_los_txt(self):
        print("Cargando archivos desde txt_spect...")
        txt_spect_data = self.cargar_txt_desde_directorio(self.PATH_TO_TXT_SPECT)

        print("Cargando archivos desde txt_from_npy...")
        txt_from_npy_data = self.cargar_txt_desde_directorio(self.PATH_TO_TXT_FROM_NPY)

        return txt_spect_data, txt_from_npy_data

    def compute_difference(self, data1, data2):
        if data1.shape != data2.shape:
            raise ValueError("Los datos deben tener las mismas dimensiones para comparar.")
        return float(np.sum(np.abs(data1 - data2)))

    def comparar_archivos_txt(self):
        txt_spect_data, txt_from_npy_data = self.cargar_todos_los_txt()

        diferencias = {}
        for spect_name, spect_data in txt_spect_data.items():
            for npy_name, npy_data in txt_from_npy_data.items():
                try:
                    diferencia = self.compute_difference(spect_data, npy_data)
                    diferencias[(spect_name, npy_name)] = diferencia
                    print(f"Diferencia entre {spect_name} y {npy_name}: {diferencia}")
                except Exception as e:
                    print(f"⚠️ Error comparando {spect_name} y {npy_name}: {e}")

        return diferencias


if __name__ == "__main__":
    handler = DatasetHandlerTxt()
    diferencias = handler.comparar_archivos_txt()
    print(f"\nDiferencias calculadas:\n{diferencias}")

