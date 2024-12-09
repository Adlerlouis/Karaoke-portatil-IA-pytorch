#!/usr/bin/env python3
import os
from pathlib import Path
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Rutas de salida
IMG_OUTPUT_PATH = Path("/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/img")
TEXT_OUTPUT_PATH = Path("/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/txt_spect")
NPY_OUTPUT_PATH = Path("/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/NPY")
TXT_FROM_NPY_PATH = Path("/home/adler/PYTHON3INTELIGENCEARTIFICAL/DISC/txt_from_npy")

# Configuración de gráficos
SAVE_PARAMS = {"dpi": 300, "bbox_inches": "tight", "transparent": True}
TICKS = np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000])
TICK_LABELS = np.array(["31.25", "62.5", "125", "250", "500", "1k", "2k", "4k", "8k"])

def plot_spectrogram_and_save(signal, fs, output_path: Path, fft_size=2048, hop_size=512, window_size=2048):
    """
    Genera y guarda el espectrograma de una señal de audio como imagen, archivos .npy y .txt.
    """
    if not window_size:
        window_size = fft_size
    if not hop_size:
        hop_size = window_size // 4

    # Transformada de Fourier
    stft = librosa.stft(signal, n_fft=fft_size, hop_length=hop_size, win_length=window_size, center=False)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # Frecuencias y tiempos
    freqs = librosa.fft_frequencies(sr=fs, n_fft=fft_size)
    times = librosa.frames_to_time(np.arange(spectrogram_db.shape[1]), sr=fs, hop_length=hop_size)

    # Guardar archivos de texto
    TEXT_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    np.savetxt(TEXT_OUTPUT_PATH / f"{output_path.stem}_frecuencias.txt", freqs, header="Bins de frecuencia (Hz)")
    np.savetxt(TEXT_OUTPUT_PATH / f"{output_path.stem}_tiempos.txt", times, header="Bins de tiempo (s)")
    np.savetxt(TEXT_OUTPUT_PATH / f"{output_path.stem}_espectrograma_db.txt", spectrogram_db, header="Espectrograma en dB")

    # Guardar archivo .npy
    data_to_save = {"spectrogram_db": spectrogram_db, "frequencies": freqs, "times": times}
    NPY_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    np.save(NPY_OUTPUT_PATH / f"{output_path.stem}_datos.npy", data_to_save)

    # Generar y guardar imagen
    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(
        spectrogram_db, y_axis="log", x_axis="time", sr=fs, hop_length=hop_size, cmap="inferno"
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.yticks(TICKS, TICK_LABELS)
    plt.colorbar(img, format="%+2.f dBFS")
    plt.savefig(
        output_path.with_stem(
            f"{output_path.stem}_spectrogram_WINLEN={window_size}_HOPLEN={hop_size}_NFFT={fft_size}"
        ),
        **SAVE_PARAMS
    )
    plt.close()

def save_npy_as_txt(npy_file_path, txt_output_folder):
    """
    Lee un archivo .npy, muestra su contenido y guarda el contenido como archivo .txt.
    """
    try:
        # Leer el archivo .npy como un diccionario
        data = np.load(npy_file_path, allow_pickle=True).item()

        # Crear el archivo .txt con el mismo nombre base que el archivo .npy
        txt_output_folder.mkdir(parents=True, exist_ok=True)
        txt_output_path = txt_output_folder / f"{npy_file_path.stem}.txt"

        with open(txt_output_path, 'w') as txt_file:
            txt_file.write(f"Datos del archivo: {npy_file_path.name}\n")
            txt_file.write("=" * 50 + "\n\n")
            for key, value in data.items():
                txt_file.write(f"{key.upper()}:\n")
                if isinstance(value, np.ndarray):
                    txt_file.write(f"(Forma: {value.shape})\n")
                    np.savetxt(txt_file, value, fmt="%.5f")
                else:
                    txt_file.write(str(value) + "\n")
                txt_file.write("\n" + "-" * 50 + "\n\n")
        print(f"Datos guardados en: {txt_output_path}")
    except Exception as e:
        print(f"Error al procesar {npy_file_path}: {e}")

def process_audio_files(audio_folder_path):
    """
    Procesa archivos de audio para generar espectrogramas, archivos .npy y .txt.
    """
    for audio_file_path in audio_folder_path.glob("*.mp3"):
        if audio_file_path.is_file():
            try:
                signal, sample_rate = librosa.load(audio_file_path, sr=22050)
                print(f"Procesando archivo: {audio_file_path}, tasa de muestreo: {sample_rate}")

                output_image_path = IMG_OUTPUT_PATH / f"{audio_file_path.stem}_spectrogram.png"
                plot_spectrogram_and_save(signal, sample_rate, output_image_path)
            except Exception as e:
                print(f"Error procesando {audio_file_path}: {e}")

def process_npy_files(npy_folder):
    """
    Convierte todos los datos de los archivos .npy en un formato .txt organizado.
    """
    TXT_FROM_NPY_PATH.mkdir(parents=True, exist_ok=True)
    for npy_file in npy_folder.glob("*.npy"):
        if npy_file.is_file():
            save_npy_as_txt(npy_file, TXT_FROM_NPY_PATH)


def main():
    folder ='txt_spect'
    plt.rcParams.update({"font.size": 20})
    audio_folder_path =Path("/home/adler/PYTHON3INTELIGENCEARTIFICAL/samples")
    # Definir la ruta completa del archivo dentro de la carpeta
    file_path = os.path.join(folder,"spectogram.txt")
    
    print("Iniciando procesamiento de archivos de audio...")
    process_audio_files(audio_folder_path)
    
    print("Convirtiendo archivos .npy a formato .txt...")
    process_npy_files(NPY_OUTPUT_PATH)


#Se guardan las imagenes de los espectograma 
    for audio_file_path in audio_folder_path.glob("*.mp3"):
        if audio_file_path.is_file():  # Asegúrate de que sea un archivo
            try:
                signal, sample_rate = librosa.load(audio_file_path, sr=22050)
                print(f"Procesando archivo: {audio_file_path}, tasa de muestreo: {sample_rate}")
                
                # Nombre de salida de la imagen
                output_image_path = IMG_OUTPUT_PATH / f"{audio_file_path.stem}_spectrogram.png"
                
                # Genera y guarda el espectrograma
                plot_spectrogram_and_save(signal, sample_rate, output_image_path)
                print(f"Imagen guardada: {output_image_path}")
            except Exception as e:
                print(f"Error procesando {audio_file_path}: {e}")
        else:
            print(f"Omitido: {audio_file_path} no es un archivo de audio.")

            # Procesar archivos .npy y guardar contenido como .txt
    npy_folder = NPY_OUTPUT_PATH
    for npy_file in npy_folder.glob("*.npy"):
        save_npy_as_txt(npy_file, TXT_FROM_NPY_PATH)


    


if __name__ == "__main__":
    main()
