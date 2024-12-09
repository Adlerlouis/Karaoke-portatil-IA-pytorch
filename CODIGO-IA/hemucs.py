"""
Este codigo contiene los espectogramas de forma hibrida para demuc
"""
from copy import deepcopy
import math
import typing as tp

from openunmix.filtering import wiener
import torch
from torch import nn
from torch.nn import functional as F

from .demucs import DConv, rescale_module
from .states import capture_init
from .spec import spectro, ispectro


def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'constant', value: float = 0.):
    """Pequeño contenedor alrededor de F.pad, solo para permitir el relleno de reflexión en entradas pequeñas.
Si este es el caso, insertamos un relleno de 0 adicional a la derecha antes de que se produzca la reflexión"."""
    x0 = x
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            extra_pad_right = min(padding_right, extra_pad)
            extra_pad_left = extra_pad - extra_pad_right
            paddings = (padding_left - extra_pad_left, padding_right - extra_pad_right)
            x = F.pad(x, (extra_pad_left, extra_pad_right))
    out = F.pad(x, paddings, mode, value)
    assert out.shape[-1] == length + padding_left + padding_right
    assert (out[..., padding_left: padding_left + length] == x0).all()
    return out


class ScaledEmbedding(nn.Module):
    """
Pequeño contenedor alrededor de F.pad, solo para permitir el relleno de reflexion en entradas pequeñas.
Si este es el caso, insertamos un relleno de 0 adicional a la derecha antes de que se produzca la reflexión 
   """
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 scale: float = 10., smooth=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            # al sumar gaussianos, la sobreescala aumenta como sqrt(n), por lo que normalizamos por eso 
            weight = weight / torch.arange(1, num_embeddings + 1).to(weight).sqrt()[:, None]
            self.embedding.weight.data[:] = weight
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        out = self.embedding(x) * self.scale
        return out


class HEncLayer(nn.Module):
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1, empty=False,
                 freq=True, dconv=True, norm=True, context=0, dconv_kw={}, pad=True,
                 rewrite=True):
      """Capa de codificador. Esta se utiliza tanto en la rama de tiempo como en la de frecuencia.

      Args:
      chin: numero de canales de entrada.
      chout: numero de canales de salida.
      norm_groups: numero de grupos para la norma de grupo.
      empty: se utiliza para crear una capa con solo la primera conv. Esto se utiliza
      antes de fusionar las ramas de tiempo y frecuencia.
      freq: actua sobre las frecuencias.
      dconv: inserta ramas residuales de DConv.
      norm: utiliza GroupNorm.
      context: tamaño del contexto para la conv. 1x1.
      dconv_kw: lista de kwargs para la clase DConv.
      pad: rellena la entrada. El relleno se realiza de modo que el tamaño de salida sea
      siempre el tamaño/paso de entrada.
      rewrite: agrega conv. 1x1 al final de la capa.
"""       super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
        klass = nn.Conv1d
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.norm = norm
        self.pad = pad
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad = [pad, 0]
            klass = nn.Conv2d
        self.conv = klass(chin, chout, kernel_size, stride, pad)
        if self.empty:
            return
        self.norm1 = norm_fn(chout)
        self.rewrite = None
        if rewrite:
            self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
            self.norm2 = norm_fn(2 * chout)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chout, **dconv_kw)

    def forward(self, x, inject=None):
     """
`inject` se utiliza para inyectar el resultado de la rama de tiempo en la rama de frecuencia,
cuando ambas tienen el mismo paso.
"""
        if not self.freq and x.dim() == 4:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)

        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = F.pad(x, (0, self.stride - (le % self.stride)))
        y = self.conv(x)
        if self.empty:
            return y
        if inject is not None:
            assert inject.shape[-1] == y.shape[-1], (inject.shape, y.shape)
            if inject.dim() == 3 and y.dim() == 4:
                inject = inject[:, :, None]
            y = y + inject
        y = F.gelu(self.norm1(y))
        if self.dconv:
            if self.freq:
                B, C, Fr, T = y.shape
                y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            if self.freq:
                y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        if self.rewrite:
            z = self.norm2(self.rewrite(y))
            z = F.glu(z, dim=1)
        else:
            z = y
        return z


class MultiWrap(nn.Module):
   """
Toma una capa y la replica N veces. Cada replica actuara
en una banda de frecuencia. Todo se hace de modo que si las N réplicas tienen los mismos pesos,
esto es exactamente equivalente a aplicar el módulo original en todas las frecuencias.

Esto esta un poco sobrediseñado para evitar artefactos en los bordes al dividir
las bandas de frecuencia, pero es posible que la implementación ingenua funcione también...
"""
    def __init__(self, layer, split_ratios):
     """
     Args:
     capa: módulo a clonar, debe ser HEncLayer o HDecLayer.
     split_ratios: lista de valores de punto flotante que indican qué proporción se debe mantener para cada banda.
    """
        super().__init__()
        self.split_ratios = split_ratios
        self.layers = nn.ModuleList()
        self.conv = isinstance(layer, HEncLayer)
        assert not layer.norm
        assert layer.freq
        assert layer.pad
        if not self.conv:
            assert not layer.context_freq
        for k in range(len(split_ratios) + 1):
            lay = deepcopy(layer)
            if self.conv:
                lay.conv.padding = (0, 0)
            else:
                lay.pad = False
            for m in lay.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            self.layers.append(lay)

    def forward(self, x, skip=None, length=None):
        B, C, Fr, T = x.shape

        ratios = list(self.split_ratios) + [1]
        start = 0
        outs = []
        for ratio, layer in zip(ratios, self.layers):
            if self.conv:
                pad = layer.kernel_size // 4
                if ratio == 1:
                    limit = Fr
                    frames = -1
                else:
                    limit = int(round(Fr * ratio))
                    le = limit - start
                    if start == 0:
                        le += pad
                    frames = round((le - layer.kernel_size) / layer.stride + 1)
                    limit = start + (frames - 1) * layer.stride + layer.kernel_size
                    if start == 0:
                        limit -= pad
                assert limit - start > 0, (limit, start)
                assert limit <= Fr, (limit, Fr)
                y = x[:, :, start:limit, :]
                if start == 0:
                    y = F.pad(y, (0, 0, pad, 0))
                if ratio == 1:
                    y = F.pad(y, (0, 0, 0, pad))
                outs.append(layer(y))
                start = limit - layer.kernel_size + layer.stride
            else:
                if ratio == 1:
                    limit = Fr
                else:
                    limit = int(round(Fr * ratio))
                last = layer.last
                layer.last = True

                y = x[:, :, start:limit]
                s = skip[:, :, start:limit]
                out, _ = layer(y, s, None)
                if outs:
                    outs[-1][:, :, -layer.stride:] += (
                        out[:, :, :layer.stride] - layer.conv_tr.bias.view(1, -1, 1, 1))
                    out = out[:, :, layer.stride:]
                if ratio == 1:
                    out = out[:, :, :-layer.stride // 2, :]
                if start == 0:
                    out = out[:, :, layer.stride // 2:, :]
                outs.append(out)
                layer.last = last
                start = limit
        out = torch.cat(outs, dim=2)
        if not self.conv and not last:
            out = F.gelu(out)
        if self.conv:
            return out
        else:
            return out, None


class HDecLayer(nn.Module):
    def __init__(self, chin, chout, last=False, kernel_size=8, stride=4, norm_groups=1, empty=False,
                 freq=True, dconv=True, norm=True, context=1, dconv_kw={}, pad=True,
                 context_freq=True, rewrite=True):
        """
        igual que HEnclayer pero para el decoder  para la documentacion.
        """
        super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
        self.pad = pad
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.norm = norm
        self.context_freq = context_freq
        klass = nn.Conv1d
        klass_tr = nn.ConvTranspose1d
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            klass = nn.Conv2d
            klass_tr = nn.ConvTranspose2d
        self.conv_tr = klass_tr(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)
        if self.empty:
            return
        self.rewrite = None
        if rewrite:
            if context_freq:
                self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
            else:
                self.rewrite = klass(chin, 2 * chin, [1, 1 + 2 * context], 1,
                                     [0, context])
            self.norm1 = norm_fn(2 * chin)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chin, **dconv_kw)

    def forward(self, x, skip, length):
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)

        if not self.empty:
            x = x + skip

            if self.rewrite:
                y = F.glu(self.norm1(self.rewrite(x)), dim=1)
            else:
                y = x
            if self.dconv:
                if self.freq:
                    B, C, Fr, T = y.shape
                    y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
                y = self.dconv(y)
                if self.freq:
                    y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        else:
            y = x
            assert skip is None
        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                z = z[..., self.pad:-self.pad, :]
        else:
            z = z[..., self.pad:self.pad + length]
            assert z.shape[-1] == length, (z.shape[-1], length)
        if not self.last:
            z = F.gelu(z)
        return z, y


class HDemucs(nn.Module):
   """
Espectrograma y modelo híbrido de Demucs.
El modelo de espectrograma tiene la misma estructura que Demucs, excepto que las primeras capas están sobre el eje de frecuencia, hasta que solo hay una frecuencia y luego pasa a convoluciones temporales.
Las capas de frecuencia aún pueden acceder a la información a través de pasos de tiempo gracias al residuo DConv.

El modelo híbrido tiene una rama temporal paralela. En alguna capa, la rama temporal tiene el mismo paso
que la rama de frecuencia y luego las dos se combinan. Lo opuesto sucede en el decodificador.

Los modelos pueden usar iSTFT ingenuo de enmascaramiento, filtrado de Wiener ([Ulhih et al. 2017]),
o complejos como canales (CaC) [Choi et al. 2020]. El filtrado de Wiener se basa en la implementación de Open Unmix [Stoter et al. 2019].

La pérdida siempre está en el dominio temporal, mediante la retropropagación a través de los métodos de salida anteriores y iSTFT. Esto permite definir modelos híbridos de manera clara. Sin embargo, esto rompe un poco el filtrado de Wiener, ya que hacer más iteraciones en el momento de la prueba cambiará la contribución del espectrograma, sin cambiar la de la forma de onda, lo que conducirá a un peor rendimiento. Intenté usar la opción residual en la implementación de Wiener de OpenUnmix, pero no mejoró.
Por otro lado, CaC proporciona un rendimiento similar para los modelos híbridos y funciona de manera natural con los modelos híbridos.

Este modelo también utiliza incrustaciones de frecuencia para mejorar la eficiencia en las convoluciones sobre el eje de frecuencia, siguiendo a [Isik et al. 2020] (https://arxiv.org/pdf/2008.04470.pdf).

"""
    @capture_init
    def __init__(self,
                 sources,
                 # Canales
                 audio_channels=2,
                 channels=48,
                 channels_time=None,
                 growth=2,
                 # transformada rapida de fourier 
                 nfft=4096,
                 wiener_iters=0,
                 end_iters=0,
                 wiener_residual=False,
                 cac=True,
                 # Estructura principal
                 depth=6,
                 rewrite=True,
                 hybrid=True,
                 hybrid_old=False,
                 # Brecha de muestreo 
                 multi_freqs=None,
                 multi_freqs_depth=2,
                 freq_emb=0.2,
                 emb_scale=10,
                 emb_smooth=True,
                 # Convoluciones 
                 kernel_size=8,
                 time_stride=2,
                 stride=4,
                 context=1,
                 context_enc=0,
                 # Normalizacion
                 norm_starts=4,
                 norm_groups=4,
                 # Brecha residual de convoluciones 
                 dconv_mode=1,
                 dconv_depth=2,
                 dconv_comp=4,
                 dconv_attn=4,
                 dconv_lstm=4,
                 dconv_init=1e-4,
                 # Peso incial 
                 rescale=0.1
                 # Metadatos 
                 samplerate=44100,
                 segment=4 * 10):
    """
Args:
sources (list[str]): lista de nombres de fuentes.
audio_channels (int): canales de audio de entrada/salida.
channels (int): numero inicial de canales ocultos.
channels_time: si no es None, use un valor channels diferente para la rama de tiempo.
growth: aumente el número de canales ocultos por este factor en cada capa.
nfft: numero de bins fft. Tenga en cuenta que cambiar esto requiere un cálculo cuidadoso de
varios parametros de forma y no funcionara de inmediato para modelos hibridos.
wiener_iters: cuando se usa el filtrado de Wiener, numero de iteraciones en el momento de la prueba.
end_iters: lo mismo pero en el momento del entrenamiento. Para un modelo hibrido, debe ser igual a wiener_iters.
wiener_residual: agregue la fuente residual antes del filtrado de Wiener.
cac: usa complejos como canales, es decir, los numeros complejos son 2 canales cada uno
en la entrada y la salida. No se realiza ningun procesamiento adicional antes de ISTFT.
depth (int): número de capas en el codificador y en el decodificador.
rewrite (bool): agrega una convolución 1x1 a cada capa.
hybrid (bool): crea un dominio hibrido de tiempo/frecuencia, de lo contrario solo frecuencia.
hybrid_old: algunos modelos entrenados para MDX tenían un error de relleno. Esto replica
este error para evitar volver a entrenarlos.
multi_freqs: lista de proporciones de frecuencia para dividir bandas de frecuencia con MultiWrap.
multi_freqs_depth: cuántas capas envolver con MultiWrap. Solo se envolverán las capas más externas.
freq_emb: agrega incrustación de frecuencia después de la primera capa de frecuencia si es > 0, el valor real controla el peso de la incrustacion.
emb_scale: equivalente a escalar la tasa de aprendizaje de incrustación
emb_smooth: inicializa la incrustación con una suave (con respecto a las frecuencias).
kernel_size: tamaño del nucleo para las capas del codificador y del decodificador.
stride: paso para las capas del codificador y del decodificador.
time_stride: paso para la capa de tiempo final, después de la fusion.
context: contexto para la conv 1x1 en el decodificador.
context_enc: contexto para la conv 1x1 en el codificador.
norm_starts: capa en la que se empieza a utilizar la norma de grupo.
las capas del decodificador se numeran en orden inverso.
norm_groups: numero de grupos para la norma de grupo.
dconv_mode: si es 1: dconv solo en el codificador, 2: solo en el decodificador, 3: ambos.
dconv_depth: profundidad de la rama DConv residual.
dconv_comp: compresión de la rama DConv.
dconv_attn: agrega capas de atención en la rama DConv comenzando en esta capa.
dconv_lstm: agrega una capa LSTM en la rama DConv comenzando en esta capa.
dconv_init: escala inicial para la rama DConv LayerScale.
rescale: truco de reescalado de peso

"""
        super().__init__()
        self.cac = cac
        self.wiener_residual = wiener_residual
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment

        self.nfft = nfft
        self.hop_length = nfft // 4
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None
        self.hybrid = hybrid
        self.hybrid_old = hybrid_old
        if hybrid_old:
            assert hybrid, "hybrid_old must come with hybrid=True"
        if hybrid:
            assert wiener_iters == end_iters

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        if hybrid:
            self.tencoder = nn.ModuleList()
            self.tdecoder = nn.ModuleList()

        chin = audio_channels
        chin_z = chin   #numero de canales para cada brecha de frecuencia 
        if self.cac:
            chin_z *= 2
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            lstm = index >= dconv_lstm
            attn = index >= dconv_attn
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                assert freqs == 1
                ker = time_stride * 2
                stri = time_stride

            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                'kernel_size': ker,
                'stride': stri,
                'freq': freq,
                'pad': pad,
                'norm': norm,
                'rewrite': rewrite,
                'norm_groups': norm_groups,
                'dconv_kw': {
                    'lstm': lstm,
                    'attn': attn,
                    'depth': dconv_depth,
                    'compress': dconv_comp,
                    'init': dconv_init,
                    'gelu': True,
                }
            }
            kwt = dict(kw)
            kwt['freq'] = 0
            kwt['kernel_size'] = kernel_size
            kwt['stride'] = stride
            kwt['pad'] = True
            kw_dec = dict(kw)
            multi = False
            if multi_freqs and index < multi_freqs_depth:
                multi = True
                kw_dec['context_freq'] = False

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayer(chin_z, chout_z,
                            dconv=dconv_mode & 1, context=context_enc, **kw)
            if hybrid and freq:
                tenc = HEncLayer(chin, chout, dconv=dconv_mode & 1, context=context_enc,
                                 empty=last_freq, **kwt)
                self.tencoder.append(tenc)

            if multi:
                enc = MultiWrap(enc, multi_freqs)
            self.encoder.append(enc)
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac:
                    chin_z *= 2
            dec = HDecLayer(chout_z, chin_z, dconv=dconv_mode & 2,
                            last=index == 0, context=context, **kw_dec)
            if multi:
                dec = MultiWrap(dec, multi_freqs)
            if hybrid and freq:
                tdec = HDecLayer(chout, chin, dconv=dconv_mode & 2, empty=last_freq,
                                 last=index == 0, context=context, **kwt)
                self.tdecoder.insert(0, tdec)
            self.decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z, smooth=emb_smooth, scale=emb_scale)
                self.freq_emb_scale = freq_emb

        if rescale:
            rescale_module(self, reference=rescale)

    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        x0 = x  # noqa

        if self.hybrid:
            # Rellenamos nuevamente la señal para mantener la propiedad
            # de que el tamaño de la salida es exactamente el tamaño de la entrada
            # dividido por el paso (aquí hop_length), cuando es divisible.
            # Esto se logra rellenando con 1/4 del tamaño del núcleo (aquí nfft).
            # lo cual no es compatible con torch.stft.
            # Hacer que todas las operaciones de convolucion sigan esta convención permite alinear fácilmente
            # las ramas de tiempo y frecuencia más adelante.
           assert hl == nfft // 4
            le = int(math.ceil(x.shape[-1] / hl))
            pad = hl // 2 * 3
            if not self.hybrid_old:
                x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode='reflect')
            else:
                x = pad1d(x, (pad, pad + le * hl - x.shape[-1]))

        z = spectro(x, nfft, hl)[..., :-1, :]
        if self.hybrid:
            assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
            z = z[..., 2:2+le]
        return z

    def _ispec(self, z, length=None, scale=0):
        hl = self.hop_length // (4 ** scale)
        z = F.pad(z, (0, 0, 0, 1))
        if self.hybrid:
            z = F.pad(z, (2, 2))
            pad = hl // 2 * 3
            if not self.hybrid_old:
                le = hl * int(math.ceil(length / hl)) + 2 * pad
            else:
                le = hl * int(math.ceil(length / hl))
            x = ispectro(z, hl, length=le)
            if not self.hybrid_old:
                x = x[..., pad:pad + length]
            else:
                x = x[..., :length]
        else:
            x = ispectro(z, hl, length)
        return x

    def _magnitude(self, z):
      # devuelve la magnitud del espectrograma, excepto cuando cac es Verdadero,
      # en cuyo caso simplemente movemos la dimensión compleja al canal uno.
        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m

    def _mask(self, z, m):
     # Aplicar enmascaramiento dado el espectrograma de mezcla `z` y la máscara estimada `m`.
     # Si `cac` es Verdadero, `m` es en realidad un espectrograma completo y `z` se ignora.
        niters = self.wiener_iters
        if self.cac:
            B, S, C, Fr, T = m.shape
            out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
            out = torch.view_as_complex(out.contiguous())
            return out
        if self.training:
            niters = self.end_iters
        if niters < 0:
            z = z[:, None]
            return z / (1e-8 + z.abs()) * m
        else:
            return self._wiener(m, z, niters)

    def _wiener(self, mag_out, mix_stft, niters):
     # aplicar filtrado de Wiener desde OpenUnmix.       
        init = mix_stft.dtype
        wiener_win_len = 300
        residual = self.wiener_residual

        B, S, C, Fq, T = mag_out.shape
        mag_out = mag_out.permute(0, 4, 3, 2, 1)
        mix_stft = torch.view_as_real(mix_stft.permute(0, 3, 2, 1))

        outs = []
        for sample in range(B):
            pos = 0
            out = []
            for pos in range(0, T, wiener_win_len):
                frame = slice(pos, pos + wiener_win_len)
                z_out = wiener(
                    mag_out[sample, frame], mix_stft[sample, frame], niters,
                    residual=residual)
                out.append(z_out.transpose(-1, -2))
            outs.append(torch.cat(out, dim=0))
        out = torch.view_as_complex(torch.stack(outs, 0))
        out = out.permute(0, 4, 3, 2, 1).contiguous()
        if residual:
            out = out[:, :-1]
        assert list(out.shape) == [B, S, C, Fq, T]
        return out.to(init)

    def forward(self, mix):
        x = mix
        length = x.shape[-1]

        z = self._spec(mix)
        mag = self._magnitude(z).to(mix.device)
        x = mag

        B, C, Fq, T = x.shape
      # a diferencia de los Demucs anteriores, siempre normalizamos porque es más facil.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        # x will be the freq. branch input.

        if self.hybrid:
            # Prepara el tiempo de brecha de entrada 
            xt = mix
            meant = xt.mean(dim=(1, 2), keepdim=True)
            stdt = xt.std(dim=(1, 2), keepdim=True)
            xt = (xt - meant) / (1e-5 + stdt)
        #omitimos esta seccion para el encoder 
        
        saved = []  
        saved_t = []  
        lengths = [] 
        lengths_t = []
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if self.hybrid and idx < len(self.tencoder):
                # we have not yet merged branches.
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    # save for skip connection
                    saved_t.append(xt)
                else:
                    # tenc contiene justamente la primera convolucion ahora lo haremos en timpo y frecuencia

                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                    # agregar incrustacion de frecuencia para permitir convoluciones no equivariantes               
                # sobre el axis de frecuencia.
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb

            saved.append(x)

        x = torch.zeros_like(x)
        if self.hybrid:
            xt = torch.zeros_like(x)
        # inicializa todo a cero (la señal pasara por los saltos de u-net).
        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
        # `pre` contiene la salida justo antes de la convolucion transpuesta final, 
        # que se utiliza cuando la frecuencia y la rama de tiempo se separan.

            if self.hybrid:
                offset = self.depth - len(self.tdecoder)
            if self.hybrid and idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        #Nos aseguramos de que sean almacenadas las conexiones 
        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
              
        x_is_mps = x.device.type == "mps"
        if x_is_mps:
            x = x.cpu()

        zout = self._mask(z, x)
        x = self._ispec(zout, length)

        if x_is_mps:
            x = x.to('mps')

        if self.hybrid:
            xt = xt.view(B, S, -1, length)
            xt = xt * stdt[:, None] + meant[:, None]
            x = xt + x
        return x
