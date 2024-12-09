
from collections import defaultdict
from concurrent.futures import CancelledError
from contextlib import contextmanager
import math
import os
import tempfile
import typing as tp

import torch
from torch.nn import functional as F
from torch.utils.data import Subset
"""
    Manipulacion de Tensores (PyTorch):
        unfold: Extrae subventanas (frames) de un tensor 1D utilizando un tamaño de kernel y un paso (stride), añadiendo relleno si es necesario.
        center_trim: Recorta un tensor a un tamaño de referencia en su última dimension, centrando el contenido.

    Gestión de Metricas:
        pull_metric: Extrae valores de metricas específicos de un historial almacenado como una lista de diccionarios.
        EMA: Calculo de Media Móvil Exponencial para métricas, permitiendo promedios ponderados con un parámetro beta.

    Herramientas Variadas:
        sizeof_fmt: Convierte un tamaño en bytes a una representación legible (e.g., KB, MB).
        temp_filenames: Genera nombres de archivos temporales, con opción de eliminarlos automáticamente.
        random_subset: Crea un subconjunto aleatorio de un conjunto de datos utilizando una semilla para reproducibilidad.

    Ejecución Simulada de Tareas:
        DummyPoolExecutor: Simula un ejecutor de tareas concurrentes con una implementación sencilla que puede cancelar tareas.


"""

def unfold(a, kernel_size, stride):
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1, 'data should be contiguous'
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)


def center_trim(tensor: torch.Tensor, reference: tp.Union[torch.Tensor, int]):
    """
    Centrar el "tensor" de recorte con respecto a la "referencia", a lo largo de la última dimension.
    La "referencia" tambien puede ser un numero, que representa la longitud a la que se debe recortar.
    Si la diferencia de tamaño es != 0 mod 2, la muestra adicional se elimina del lado derecho.
"""
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


def pull_metric(history: tp.List[dict], name: str):
    out = []
    for metrics in history:
        metric = metrics
        for part in name.split("."):
            metric = metric[part]
        out.append(metric)
    return out


def EMA(beta: float = 1):
    """
    Devolucion de llamada de media movil exponencial.
    Devuelve una unica funcion que se puede llamar para actualizar repetidamente la media movil exponencial
    con un diccionario de métricas. La devolución de llamada devolvera
    el nuevo diccionario de métricas promediado.

    Tenga en cuenta que para `beta=1`, esto es simplemente un promedio simple.
"""
f
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(metrics: dict, weight: float = 1) -> dict:
        nonlocal total, fix
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        return {key: tot / fix[key] for key, tot in total.items()}
    return _update


def sizeof_fmt(num: float, suffix: str = 'B'):




    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


@contextmanager
def temp_filenames(count: int, delete=True):
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)


def random_subset(dataset, max_samples: int, seed: int = 42):
    if max_samples >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=generator)
    return Subset(dataset, perm[:max_samples].tolist())


class DummyPoolExecutor:
    class DummyResult:
        def __init__(self, func, _dict, *args, **kwargs):
            self.func = func
            self._dict = _dict
            self.args = args
            self.kwargs = kwargs

        def result(self):
            if self._dict["run"]:
                return self.func(*self.args, **self.kwargs)
            else:
                raise CancelledError()

    def __init__(self, workers=0):
        self._dict = {"run": True}

    def submit(self, func, *args, **kwargs):
        return DummyPoolExecutor.DummyResult(func, self._dict, *args, **kwargs)

    def shutdown(self, *_, **__):
        self._dict["run"] = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return
