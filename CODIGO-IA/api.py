#Descripcion lo que permite principalmente este codigo es crear una api la cual por medio de
#linea de comandos podramos llamar al modelo para la separacion de los audios sin correr todo el codigo 
import subprocess
import torch as th
import torchaudio as ta
from dora.log import fatal
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple, Union
from .apply import apply_model, _replace_dict
from .audio import AudioFile, convert_audio, save_audio
from .pretrained import get_model, _parse_remote_files, REMOTE_ROOT
from .repo import RemoteRepo, LocalRepo, ModelOnlyRepo, BagOnlyRepo


class LoadAudioError(Exception):
    pass


class LoadModelError(Exception):
    pass


class _NotProvided:
    pass


NotProvided = _NotProvided()


class Separator:
    def __init__(
        self,
        model: str = "htdemucs",
        repo: Optional[Path] = None,
        device: str = "cuda" if th.cuda.is_available() else "cpu",
        shifts: int = 1,
        overlap: float = 0.25,
        split: bool = True,
        segment: Optional[int] = None,
        jobs: int = 0,
        progress: bool = False,
        callback: Optional[Callable[[dict], None]] = None,
        callback_arg: Optional[dict] = None,
    ):
        """
       Esta clase es especialmente útil para procesar audios largos o modelos con alta demanda de memoria, como Tasnet, mientras permite optimizar la precisión y la velocidad mediante el paralelismo (jobs) y desplazamientos temporales (shifts
        """
        self._name = model
        self._repo = repo
        self._load_model()
        self.update_parameter(device=device, shifts=shifts, overlap=overlap, split=split,
                              segment=segment, jobs=jobs, progress=progress, callback=callback,
                              callback_arg=callback_arg)

    def update_parameter(
        self,
        device: Union[str, _NotProvided] = NotProvided,
        shifts: Union[int, _NotProvided] = NotProvided,
        overlap: Union[float, _NotProvided] = NotProvided,
        split: Union[bool, _NotProvided] = NotProvided,
        segment: Optional[Union[int, _NotProvided]] = NotProvided,
        jobs: Union[int, _NotProvided] = NotProvided,
        progress: Union[bool, _NotProvided] = NotProvided,
        callback: Optional[
            Union[Callable[[dict], None], _NotProvided]
        ] = NotProvided,
        callback_arg: Optional[Union[dict, _NotProvided]] = NotProvided,
    ):

"""
 La configuracion actualizada de Separator permite flexibilidad y precision en la separación de audio, mejorando la memoria y rendimiento mediante divisiones, superposicion, y cálculos paralelos. El sistema de callbacks facilita un monitoreo detallado y ajustable del progreso. Ideal para audios largos o modelos de alta complejidad como Tasnet
"""
        if not isinstance(device, _NotProvided):
            self._device = device
        if not isinstance(shifts, _NotProvided):
            self._shifts = shifts
        if not isinstance(overlap, _NotProvided):
            self._overlap = overlap
        if not isinstance(split, _NotProvided):
            self._split = split
        if not isinstance(segment, _NotProvided):
            self._segment = segment
        if not isinstance(jobs, _NotProvided):
            self._jobs = jobs
        if not isinstance(progress, _NotProvided):
            self._progress = progress
        if not isinstance(callback, _NotProvided):
            self._callback = callback
        if not isinstance(callback_arg, _NotProvided):
            self._callback_arg = callback_arg

    def _load_model(self):
        self._model = get_model(name=self._name, repo=self._repo)
        if self._model is None:
            raise LoadModelError("Failed to load model")
        self._audio_channels = self._model.audio_channels
        self._samplerate = self._model.samplerate

    def _load_audio(self, track: Path):
        errors = {}
        wav = None

        try:
            wav = AudioFile(track).read(streams=0, samplerate=self._samplerate,
                                        channels=self._audio_channels)
        except FileNotFoundError:
            errors["ffmpeg"] = "FFmpeg is not installed."
        except subprocess.CalledProcessError:
            errors["ffmpeg"] = "FFmpeg could not read the file."

        if wav is None:
            try:
                wav, sr = ta.load(str(track))
            except RuntimeError as err:
                errors["torchaudio"] = err.args[0]
            else:
                wav = convert_audio(wav, sr, self._samplerate, self._audio_channels)

        if wav is None:
            raise LoadAudioError(
                "\n".join(
                    "When trying to load using {}, got the following error: {}".format(
                        backend, error
                    )
                    for backend, error in errors.items()
                )
            )
        return wav

    def separate_tensor(
        self, wav: th.Tensor, sr: Optional[int] = None
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        """
      Separacion un tensor cargado.

     Parámetros
     ----------
     wav: Forma de onda del audio. Debe tener 2 dimensiones, la primera es cada canal de audio, 
     mientras que la segunda es la forma de onda de cada canal. El tipo debe ser float32. 
     p. ej. tupla(wav.shape) == (2, 884000) donde esto significa que el audio tiene 2 canales.
sr: Frecuencia de muestreo del audio original, la onda se volverá a muestrear si no coincide con el modelo 


      Nos devuelve 
     -------
     Una tupla, cuyo primer elemento es la onda original y el segundo elemento es un diccionario, cuyas claves
     son el nombre de los tallos y los valores son ondas separadas. La onda original ya se habrá
     remuestreado.

      Notas
     -----
      Utilizar con cuidado porque no se a verificado totalmente 
      """
        if sr is not None and sr != self.samplerate:
            wav = convert_audio(wav, sr, self._samplerate, self._audio_channels)
        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std() + 1e-8
        out = apply_model(
                self._model,
                wav[None],
                segment=self._segment,
                shifts=self._shifts,
                split=self._split,
                overlap=self._overlap,
                device=self._device,
                num_workers=self._jobs,
                callback=self._callback,
                callback_arg=_replace_dict(
                    self._callback_arg, ("audio_length", wav.shape[1])
                ),
                progress=self._progress,
            )
        if out is None:
            raise KeyboardInterrupt
        out *= ref.std() + 1e-8
        out += ref.mean()
        wav *= ref.std() + 1e-8
        wav += ref.mean()
        return (wav, dict(zip(self._model.sources, out[0])))

    def separate_audio_file(self, file: Path):
       """
        Separar un tensor cargado.

       Parametros
       ----------
       wav: Forma de onda del audio. Debe tener 2 dimensiones, la primera es cada canal de audio, 
       mientras que la segunda es la forma de onda de cada canal. El tipo debe ser float32. 
       p. ej. tupla(wav.shape) == (2, 884000) significa que el audio tiene 2 canales.
       sr: Frecuencia de muestreo del audio original, la onda se volverá a muestrear si no coincide con el modelo 
       .

Devuelve
-------
Una tupla, cuyo primer elemento es la onda original y el segundo elemento es un diccionario, cuyas claves
son el nombre de los tallos y los valores son ondas separadas. La onda original ya se habrá
remuestreado.

Notas
-----
Utilizar esta funcion con precaucion  con los datos correspondientes 
       
       """
        return self.separate_tensor(self._load_audio(file), self.samplerate)

    @property
    def samplerate(self):
        return self._samplerate

    @property
    def audio_channels(self):
        return self._audio_channels

    @property
    def model(self):
        return self._model


def list_models(repo: Optional[Path] = None) -> Dict[str, Dict[str, Union[str, Path]]]:
    """
 Enumera  los modelos disponibles. Recuerdemos que no todos los modelos son fidedignos

Parametros
----------
repositorio : El repositorio cuyos modelos se van a enumerar.

Devuelve
-------
Un diccionario con dos claves ("single" para modelos individuales y "bag" para un conjunto de modelos). Los valores son
listas cuyos componentes son cadenas. 
"""
"""
  """
    model_repo: ModelOnlyRepo
    if repo is None:
        models = _parse_remote_files(REMOTE_ROOT / 'files.txt')
        model_repo = RemoteRepo(models)
        bag_repo = BagOnlyRepo(REMOTE_ROOT, model_repo)
    else:
        if not repo.is_dir():
            fatal(f"{repo} must exist and be a directory.")
        model_repo = LocalRepo(repo)
        bag_repo = BagOnlyRepo(repo, model_repo)
    return {"single": model_repo.list_model(), "bag": bag_repo.list_model()}

#aqui ya se van a cargar todas las clases para este modelo 
if __name__ == "__main__":
    # Test API functions
    # two-stem not supported

    from .separate import get_parser

    args = get_parser().parse_args()
    separator = Separator(
        model=args.name,
        repo=args.repo,
        device=args.device,
        shifts=args.shifts,
        overlap=args.overlap,
        split=args.split,
        segment=args.segment,
        jobs=args.jobs,
        callback=print
    )
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    for file in args.tracks:
        separated = separator.separate_audio_file(file)[1]
        if args.mp3:
            ext = "mp3"
        elif args.flac:
            ext = "flac"
        else:
            ext = "wav"
        kwargs = {
            "samplerate": separator.samplerate,
            "bitrate": args.mp3_bitrate,
            "clip": args.clip_mode,
            "as_float": args.float32,
            "bits_per_sample": 24 if args.int24 else 16,
        }
        for stem, source in separated.items():
            stem = out / args.filename.format(
                track=Path(file).name.rsplit(".", 1)[0],
                trackext=Path(file).name.rsplit(".", 1)[-1],
                stem=stem,
                ext=ext,
            )
            stem.parent.mkdir(parents=True, exist_ok=True)
            save_audio(source, str(stem), **kwargs)
