
"""
1. power_iteration

Es un metodo para aproximar el valor singular mas grande de una matriz cuadrada MM usando el metodo de potencia.

    Entrada:
        m: Una matriz cuadrada de tamaño n×nn×n.
        niters: Numero de iteraciones para refinar la aproximacion.
        bs: Tamaño del batch para usar multiples puntos iniciales en paralelo.

    Salida:
        El mayor valor singular aproximado de la matriz MM.
        2. svd_penalty

Aplica una penalización basada en los valores singulares de las capas del modelo.

    Parámetros principales:
        model: El modelo a evaluar.
        min_size: Tamaño mínimo de las capas (en MB) para ser consideradas.
        dim: Dimension de proyeccion para la descomposicion SVD de rango bajo.
        niters: Iteraciones para calcular el SVD de rango bajo.
        powm: Si es True, usa el método de potencia en lugar de SVD de rango bajo.
        convtr: Distingue entre capas convolucionales y convolucionales transpuestas.
        proba: Probabilidad de aplicar la penalización.
        conv_only: Si es True, aplica la penalización solo a capas convolucionales.
        exact: Si es True, utiliza SVD exacto (más lento, pero preciso).
        bs: Tamaño del batch para el método de potencia.

    Funcionamiento:
        Itera sobre las capas del modelo (model.modules()).
        Evalúa si la capa cumple ciertas condiciones (tamaño mínimo, tipo de capa, etc.).
        Calcula el mayor valor singular:
            Si exact es True, utiliza SVD exacto.
            Si powm es True, utiliza el método de potencia.
            Si no, utiliza SVD de rango bajo.
        Acumula los valores singulares al cuadrado (penalización total).

    Salida:
        El total de penalización normalizado por la probabilidad proba.
 Esto nos permite penalizar modelos al imponer restricciones en el valor singular más grande de las capas. Esto nos sirve para modelos complejos de nuestro sistema de redes neuronales
"""
import random
import torch


def power_iteration(m, niters=1, bs=1):
    """This is the power method. batch size is used to try multiple starting point in parallel."""
    assert m.dim() == 2
    assert m.shape[0] == m.shape[1]
    dim = m.shape[0]
    b = torch.randn(dim, bs, device=m.device, dtype=m.dtype)

    for _ in range(niters):
        n = m.mm(b)
        norm = n.norm(dim=0, keepdim=True)
        b = n / (1e-10 + norm)

    return norm.mean()


penalty_rng = random.Random(1234)


def svd_penalty(model, min_size=0.1, dim=1, niters=2, powm=False, convtr=True,
   """
Penalizacion sobre el valor singular mas grande de una capa.
Args:
- model: modelo a penalizar
- min_size: tamaño mínimo en MB de una capa a penalizar.
- dim: dimension de proyeccion para svd_lowrank. Cuanto mas alto, mejor, pero mas lento.
- niters: numero de iteraciones en el algoritmo utilizado por svd_lowrank.
- powm: usar el método de potencia en lugar de SVD de rango bajo, (No es recomendado usar debido a que es lento).
- convtr: cuando es verdadero, diferencia entre Conv y Conv transpuesto.
esto se mantiene por compatibilidad con experimentos más antiguos.
- proba: probabilidad de aplicar la penalizacion.
- conv_only: solo se aplica a conv y conv transpuesto, no a LSTM
(puede no ser confiable para otros modelos que no sean Demucs).
- exact: usar SVD exacto (lento pero útil en la validación).
- bs: batch_size para el método de potencia.
"""             proba=1, conv_only=False, exact=False, bs=1):
   total = 0
    if penalty_rng.random() > proba:
        return 0.

    for m in model.modules():
        for name, p in m.named_parameters(recurse=False):
            if p.numel() / 2**18 < min_size:
                continue
            if convtr:
                if isinstance(m, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d)):
                    if p.dim() in [3, 4]:
                        p = p.transpose(0, 1).contiguous()
            if p.dim() == 3:
                p = p.view(len(p), -1)
            elif p.dim() == 4:
                p = p.view(len(p), -1)
            elif p.dim() == 1:
                continue
            elif conv_only:
                continue
            assert p.dim() == 2, (name, p.shape)
            if exact:
                estimate = torch.svd(p, compute_uv=False)[1].pow(2).max()
            elif powm:
                a, b = p.shape
                if a < b:
                    n = p.mm(p.t())
                else:
                    n = p.t().mm(p)
                estimate = power_iteration(n, niters, bs)
            else:
                estimate = torch.svd_lowrank(p, dim, niters)[1][0].pow(2)
            total += estimate
    return total / proba
