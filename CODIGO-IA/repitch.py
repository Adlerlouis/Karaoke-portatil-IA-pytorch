"""
Componentes principales:
1. Clase RepitchedWrapper:

    Proposito: Envolver un conjunto de datos para aplicar cambios aleatorios en el tono y el tempo de los datos de audio de manera "online" (al acceder a ellos).
    Parametros principales:
        dataset: El conjunto de datos original.
        proba: Probabilidad de aplicar cambios al audio.
        max_pitch: Máxima variacion en semitonos para el cambio de tono.
        max_tempo: Máxima variacion porcentual en el tempo.
        tempo_std: Desviación estándar para la variacion del tempo.
        vocals:     Indices de los canales que contienen voces, para tratarlos de forma especial.
        same: Si es True, todos los canales tendrán los mismos cambios de tono/tempo.
    Métodos:
        __len__: Devuelve la longitud del conjunto de datos.
        __getitem__: Aplica modificaciones aleatorias a las muestras de audio, ajustando tono y tempo según las probabilidades y configuraciones establecidas.

2. Función repitch:

    Proposito: Realizar cambios especificos en el tono y tempo de una señal de audio.
    Parámetros principales:
        wav: Señal de audio a procesar.
        pitch: Variacion en semitonos del tono.
        tempo: Variacion porcentual del tempo.
        voice: Si es True, aplica configuraciones específicas para voces.
        quick: Si es True, usa un modo rapido.
        samplerate: Frecuencia de muestreo del audio.
    Requisitos: Utiliza la herramienta externa soundstretch para realizar el procesamiento.
    Proceso:
        Guarda el audio original en un archivo temporal.
        Ejecuta el comando soundstretch con las opciones de tono y tempo.
        Carga el archivo procesado y verifica la frecuencia de muestreo.
        Devuelve la señal procesada.

3. Dependencias:

    Librerias:
        torch y torchaudio para trabajar con señales de audio como tensores.
        random para generar variaciones aleatorias.
        subprocess para ejecutar comandos externos.
        tempfile para gestionar archivos temporales.
    Herramienta externa:
        soundstretch para modificar tono y tempo de archivos WAV.
"""

import random
import subprocess as sp
import tempfile

import torch
import torchaudio as ta

from .audio import save_audio


class RepitchedWrapper:
    """
    Wrap a dataset to apply online change of pitch / tempo.
    """
    def __init__(self, dataset, proba=0.2, max_pitch=2, max_tempo=12,
                 tempo_std=5, vocals=[3], same=True):
        self.dataset = dataset
        self.proba = proba
        self.max_pitch = max_pitch
        self.max_tempo = max_tempo
        self.tempo_std = tempo_std
        self.same = same
        self.vocals = vocals

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        streams = self.dataset[index]
        in_length = streams.shape[-1]
        out_length = int((1 - 0.01 * self.max_tempo) * in_length)

        if random.random() < self.proba:
            outs = []
            for idx, stream in enumerate(streams):
                if idx == 0 or not self.same:
                    delta_pitch = random.randint(-self.max_pitch, self.max_pitch)
                    delta_tempo = random.gauss(0, self.tempo_std)
                    delta_tempo = min(max(-self.max_tempo, delta_tempo), self.max_tempo)
                stream = repitch(
                    stream,
                    delta_pitch,
                    delta_tempo,
                    voice=idx in self.vocals)
                outs.append(stream[:, :out_length])
            streams = torch.stack(outs)
        else:
            streams = streams[..., :out_length]
        return streams


def repitch(wav, pitch, tempo, voice=False, quick=False, samplerate=44100):
"""
El tempo es un delta relativo en porcentaje, por lo que tempo=10 significa tempo al 110 %.
El tono está en semitonos.
Requiere que `soundstretch` esté instalado, consulte
https://www.surina.net/soundtouch/soundstretch.html
"""
    infile = tempfile.NamedTemporaryFile(suffix=".wav")
    outfile = tempfile.NamedTemporaryFile(suffix=".wav")
    save_audio(wav, infile.name, samplerate, clip='clamp')
    command = [
        "soundstretch",
        infile.name,
        outfile.name,
        f"-pitch={pitch}",
        f"-tempo={tempo:.6f}",
    ]
    if quick:
        command += ["-quick"]
    if voice:
        command += ["-speech"]
    try:
        sp.run(command, capture_output=True, check=True)
    except sp.CalledProcessError as error:
        raise RuntimeError(f"Could not change bpm because {error.stderr.decode('utf-8')}")
    wav, sr = ta.load(outfile.name)
    assert sr == samplerate
    return wav
