"""
Características Destacadas
    Integracion con ffmpeg: Permite manejar una amplia variedad de formatos de audio y realizar operaciones avanzadas como resampleo y selección de streams
    Compatibilidad con Torch: Utiliza tensores para manipular datos de audio, facilitando su uso en pipelines de aprendizaje automatico.
    Soporte para multiples formatos: Lee y escribe formatos comunes como MP3, WAV y FLAC.
    Prevencion de Clipping: Ofrece métodos para evitar distorsión en el audio al guardar.
"""

import json
import subprocess as sp
from pathlib import Path

import lameenc
import julius
import numpy as np
import torch
import torchaudio as ta
import typing as tp

from .utils import temp_filenames


def _read_info(path):
    stdout_data = sp.check_output([
        'ffprobe', "-loglevel", "panic",
        str(path), '-print_format', 'json', '-show_format', '-show_streams'
    ])
    return json.loads(stdout_data.decode('utf-8'))


class AudioFile:
    """
    Permite leer audio desde cualquier formato compatible con ffmpeg, así como remuestrearlo o convertirlo a mono sobre la marcha.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self._info = None

    def __repr__(self):
        features = [("path", self.path)]
        features.append(("samplerate", self.samplerate()))
        features.append(("channels", self.channels()))
        features.append(("streams", len(self)))
        features_str = ", ".join(f"{name}={value}" for name, value in features)
        return f"AudioFile({features_str})"

    @property
    def info(self):
        if self._info is None:
            self._info = _read_info(self.path)
        return self._info

    @property
    def duration(self):
        return float(self.info['format']['duration'])

    @property
    def _audio_streams(self):
        return [
            index for index, stream in enumerate(self.info["streams"])
            if stream["codec_type"] == "audio"
        ]

    def __len__(self):
        return len(self._audio_streams)

    def channels(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['channels'])

    def samplerate(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['sample_rate'])

    def read(self,
             seek_time=None,
             duration=None,
             streams=slice(None),
             samplerate=None,
             channels=None):
        """
       Implementacion ligeramente más eficiente que stempeg,
       en particular, esto extraerá todos los stems a la vez
       en lugar de tener que recorrer un archivo varias veces
       para cada flujo.

       Argumentos:
       seek_time (float): tiempo de busqueda en segundos o None si no se necesita ninguna busqueda.
       duration (float): duracion en segundos para extraer o None para extraer hasta el final.
       streams (slices, int o lista): flujos para extraer, pueden ser un solo int, una lista o
       un segmento. Si es un segmento o una lista, la salida será de tamaño [S, C, T]
       con S el número de flujos, C el numero de canales y T el número de muestras.
       Si es un int, la salida será [C, T].
       samplerate (int): si se proporciona, se remuestreara sobre la marcha. Si es None, no se realizara ningun remuestreo. La frecuencia de muestreo original se puede obtener con :method:samplerate.
       channels (int): si es 1, se convertirá a mono. No dependemos de ffmpeg para eso, ya que ffmpeg escala automáticamente en +3dB para conservar el volumen cuando se reproduce en altavoces.
       Consulte https://sound.stackexchange.com/a/42710.
       Nuestra definicion de mono es simplemente el promedio de los dos canales. Cualquier otro valor será ignorado.
       """

        streams = np.array(range(len(self)))[streams]
        single = not isinstance(streams, np.ndarray)
        if single:
            streams = [streams]

        if duration is None:
            target_size = None
            query_duration = None
        else:
            target_size = int((samplerate or self.samplerate()) * duration)
            query_duration = float((target_size + 1) / (samplerate or self.samplerate()))

        with temp_filenames(len(streams)) as filenames:
            command = ['ffmpeg', '-y']
            command += ['-loglevel', 'panic']
            if seek_time:
                command += ['-ss', str(seek_time)]
            command += ['-i', str(self.path)]
            for stream, filename in zip(streams, filenames):
                command += ['-map', f'0:{self._audio_streams[stream]}']
                if query_duration is not None:
                    command += ['-t', str(query_duration)]
                command += ['-threads', '1']
                command += ['-f', 'f32le']
                if samplerate is not None:
                    command += ['-ar', str(samplerate)]
                command += [filename]

            sp.run(command, check=True)
            wavs = []
            for filename in filenames:
                wav = np.fromfile(filename, dtype=np.float32)
                wav = torch.from_numpy(wav)
                wav = wav.view(-1, self.channels()).t()
                if channels is not None:
                    wav = convert_audio_channels(wav, channels)
                if target_size is not None:
                    wav = wav[..., :target_size]
                wavs.append(wav)
        wav = torch.stack(wavs, dim=0)
        if single:
            wav = wav[0]
        return wav


def convert_audio_channels(wav, channels=2):
    """Convertir audio en en numero correspondiente de canales ."""
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Caso 1:
        #llama al canal 2 pero haz un stream multiple 
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # CasO 2:
        #llama al canal multiple pero haz que el archivo de entrada tenga un canal solo y replia el audio sobre el canal  
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Caso 3:
        #llama al canal multiple en diferentes canales donde el archivo de entrada tenga mas canales de los requeridos y en este caso retorna el primer canal
        wav = wav[..., :channels, :]
    else:
        # Caso 4: Que caso es el mas razonable aqui 
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


def convert_audio(wav, from_samplerate, to_samplerate, channels) -> torch.Tensor:
    """ Convierte el audi de forma la cual se samplea de forma que cada objetivo pueda caber en cada canal"""
    wav = convert_audio_channels(wav, channels)
    return julius.resample_frac(wav, from_samplerate, to_samplerate)


def i16_pcm(wav):
    """Convert audio to 16 bits integer PCM format."""

    """Convertir el audio a 16 bits de forma integral de formato PCM  """
    if wav.dtype.is_floating_point:
        return (wav.clamp_(-1, 1) * (2**15 - 1)).short()
    else:
        return wav


def f32_pcm(wav):
    """Convertir el audio  a formato float 32 pcm ."""
    if wav.dtype.is_floating_point:
        return wav
    else:
        return wav.float() / (2**15 - 1)


def as_dtype_pcm(wav, dtype):
    """Convertir audio de acuerdo a uno u otro."""
    if wav.dtype.is_floating_point:
        return f32_pcm(wav)
    else:
        return i16_pcm(wav)


def encode_mp3(wav, path, samplerate=44100, bitrate=320, quality=2, verbose=False):
    """Haz un enconder para audios tipo mp3 los cuales tiene un samplerate =44100 bitrate=320 ."""
    C, T = wav.shape
    wav = i16_pcm(wav)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(samplerate)
    encoder.set_channels(C)
    encoder.set_quality(quality)  # 2-alto 7-Super rapido 
    if not verbose:
        encoder.silence()
    wav = wav.data.cpu()
    wav = wav.transpose(0, 1).numpy()
    mp3_data = encoder.encode(wav.tobytes())
    mp3_data += encoder.flush()
    with open(path, "wb") as f:
        f.write(mp3_data)


def prevent_clip(wav, mode='rescale'):
    """
    estrategias para abordar el recorte de audio.
    """
    if mode is None or mode == 'none':
        return wav
    assert wav.dtype.is_floating_point, "demasido tarde para el recorte "
    if mode == 'rescale':
        wav = wav / max(1.01 * wav.abs().max(), 1)
    elif mode == 'clamp':
        wav = wav.clamp(-0.99, 0.99)
    elif mode == 'tanh':
        wav = torch.tanh(wav)
    else:
        raise ValueError(f"Invalid mode {mode}")
    return wav


def save_audio(wav: torch.Tensor,
               path: tp.Union[str, Path],
               samplerate: int,
               bitrate: int = 320,
               clip: tp.Literal["rescale", "clamp", "tanh", "none"] = 'rescale',
               bits_per_sample: tp.Literal[16, 24, 32] = 16,
               as_float: bool = False,
               preset: tp.Literal[2, 3, 4, 5, 6, 7] = 2):
        """Guardar archivo de audio, evitando automaticamente el recorte si es necesario
        segun la estrategia `clip` dada. Si la ruta termina en .mp3, esto
        se guardara como mp3 con el bitrate dado. Use preset para configurar la calidad del mp3:
        2 para la más alta calidad, 7 para la velocidad más rápida
        """
    wav = prevent_clip(wav, mode=clip)
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mp3":
        encode_mp3(wav, path, samplerate, bitrate, preset, verbose=True)
    elif suffix == ".wav":
        if as_float:
            bits_per_sample = 32
            encoding = 'PCM_F'
        else:
            encoding = 'PCM_S'
        ta.save(str(path), wav, sample_rate=samplerate,
                encoding=encoding, bits_per_sample=bits_per_sample)
    elif suffix == ".flac":
        ta.save(str(path), wav, sample_rate=samplerate, bits_per_sample=bits_per_sample)
    else:
        raise ValueError(f"Invalid suffix for path: {suffix}")
