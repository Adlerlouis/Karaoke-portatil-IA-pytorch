"""
1. Inicialización (__init__)

    Carga de componentes básicos:
        loaders: Manejadores de datos para el entrenamiento y la validacion.
        model: El modelo que se entrenara.
        optimizer: El optimizador que ajustara los pesos del modelo.
        args: Argumentos y configuraciones adicionales.

    EMA (Exponential Moving Average):
        Se utiliza para calcular una media exponencial de los parametros del modelo, ayudando a estabilizar el entrenamiento.
        Tiene dos tipos de EMA: por batch y por epoca.

    Augmentaciones de datos:
        Incluye desplazamiento, inversion de canales y escala, que son tecnicas para aumentar la diversidad de los datos de entrenamiento.

    Checkpoints:
        Se configuran para guardar el estado del modelo durante el entrenamiento.

2. Serializacion (_serialize)

Guarda el estado actual del modelo, el optimizador, el historial y los EMAs en un archivo de checkpoint. Ademas:

    Si el modelo mejora, guarda un "mejor modelo" en un archivo separado.
    Permite guardar checkpoints periódicamente basándose en las epocas.

3. Restablecimiento (_reset)

Restaura el estado del modelo y el optimizador desde un checkpoint, o bien, carga un modelo preentrenado si se especifica.
4. Entrenamiento (train)

El metodo principal que entrena el modelo a lo largo de multiples epocas:

    Entrenamiento por epoca:
        Ajusta los parametros del modelo usando el conjunto de entrenamiento.
    Validacion cruzada:
        Evalua el modelo en un conjunto de validación.
        Compara las metricas actuales con las mejores anteriores para identificar mejoras.
    Evaluacion final:
        Realiza una evaluacion en el conjunto de prueba al final del entrenamiento o en épocas especificadas.
    Guarda checkpoints después de cada epoca y actualiza el modelo "mejor" si las métricas mejoran.

5. Ejecutar una época (_run_one_epoch)

Realiza una pasada completa sobre el conjunto de datos (entrenamiento o validación):

    Aplica las augmentaciones si es un entrenamiento.
    Calcula las predicciones del modelo y las métricas.

6. Formateo de métricas (_format_train y _format_test)

Convierte las metricas de entrenamiento y prueba en un formato más legible para mostrarlas en los logs.
7. Logs

Se registran los resúmenes de cada época y métricas clave, incluyendo pérdidas, NSDR (una métrica común para evaluar separación de señales), y otras estadísticas.

"""
import logging

from dora import get_xp
from dora.utils import write_and_rename
from dora.log import LogProgress, bold
import torch
import torch.nn.functional as F

from . import augment, distrib, states, pretrained
from .apply import apply_model
from .ema import ModelEMA
from .evaluate import evaluate, new_sdr
from .svd import svd_penalty
from .utils import pull_metric, EMA

logger = logging.getLogger(__name__)


def _summary(metrics):
    return " | ".join(f"{key.capitalize()}={val}" for key, val in metrics.items())


class Solver(object):
    def __init__(self, loaders, model, optimizer, args):
        self.args = args
        self.loaders = loaders

        self.model = model
        self.optimizer = optimizer
        self.quantizer = states.get_quantizer(self.model, args.quant, self.optimizer)
        self.dmodel = distrib.wrap(model)
        self.device = next(iter(self.model.parameters())).device

        # Exponential moving average of the model, either updated every batch or epoch.
        # The best model from all the EMAs and the original one is kept based on the valid
        # loss for the final best model.
        self.emas = {'batch': [], 'epoch': []}
        for kind in self.emas.keys():
            decays = getattr(args.ema, kind)
            device = self.device if kind == 'batch' else 'cpu'
            if decays:
                for decay in decays:
                    self.emas[kind].append(ModelEMA(self.model, decay, device=device))

        # data augment
        augments = [augment.Shift(shift=int(args.dset.samplerate * args.dset.shift),
                                  same=args.augment.shift_same)]
        if args.augment.flip:
            augments += [augment.FlipChannels(), augment.FlipSign()]
        for aug in ['scale', 'remix']:
            kw = getattr(args.augment, aug)
            if kw.proba:
                augments.append(getattr(augment, aug.capitalize())(**kw))
        self.augment = torch.nn.Sequential(*augments)

        xp = get_xp()
        self.folder = xp.folder
        # Checkpoints
        self.checkpoint_file = xp.folder / 'checkpoint.th'
        self.best_file = xp.folder / 'best.th'
        logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.best_state = None
        self.best_changed = False

        self.link = xp.link
        self.history = self.link.history

        self._reset()

    def _serialize(self, epoch):
        package = {}
        package['state'] = self.model.state_dict()
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        for kind, emas in self.emas.items():
            for k, ema in enumerate(emas):
                package[f'ema_{kind}_{k}'] = ema.state_dict()
        with write_and_rename(self.checkpoint_file) as tmp:
            torch.save(package, tmp)

        save_every = self.args.save_every
        if save_every and (epoch + 1) % save_every == 0 and epoch + 1 != self.args.epochs:
            with write_and_rename(self.folder / f'checkpoint_{epoch + 1}.th') as tmp:
                torch.save(package, tmp)

        if self.best_changed:
            # Saving only the latest best model.
            with write_and_rename(self.best_file) as tmp:
                package = states.serialize_model(self.model, self.args)
                package['state'] = self.best_state
                torch.save(package, tmp)
            self.best_changed = False

    def _reset(self):
        """Reset state of the solver, potentially using checkpoint."""
        if self.checkpoint_file.exists():
            logger.info(f'Loading checkpoint model: {self.checkpoint_file}')
            package = torch.load(self.checkpoint_file, 'cpu')
            self.model.load_state_dict(package['state'])
            self.optimizer.load_state_dict(package['optimizer'])
            self.history[:] = package['history']
            self.best_state = package['best_state']
            for kind, emas in self.emas.items():
                for k, ema in enumerate(emas):
                    ema.load_state_dict(package[f'ema_{kind}_{k}'])
        elif self.args.continue_pretrained:
            model = pretrained.get_model(
                name=self.args.continue_pretrained,
                repo=self.args.pretrained_repo)
            self.model.load_state_dict(model.state_dict())
        elif self.args.continue_from:
            name = 'checkpoint.th'
            root = self.folder.parent
            cf = root / str(self.args.continue_from) / name
            logger.info("Loading from %s", cf)
            package = torch.load(cf, 'cpu')
            self.best_state = package['best_state']
            if self.args.continue_best:
                self.model.load_state_dict(package['best_state'], strict=False)
            else:
                self.model.load_state_dict(package['state'], strict=False)
            if self.args.continue_opt:
                self.optimizer.load_state_dict(package['optimizer'])

    def _format_train(self, metrics: dict) -> dict:
        """Formatting for train/valid metrics."""
        losses = {
            'loss': format(metrics['loss'], ".4f"),
            'reco': format(metrics['reco'], ".4f"),
        }
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], ".3f")
        if self.quantizer is not None:
            losses['ms'] = format(metrics['ms'], ".2f")
        if 'grad' in metrics:
            losses['grad'] = format(metrics['grad'], ".4f")
        if 'best' in metrics:
            losses['best'] = format(metrics['best'], '.4f')
        if 'bname' in metrics:
            losses['bname'] = metrics['bname']
        if 'penalty' in metrics:
            losses['penalty'] = format(metrics['penalty'], ".4f")
        if 'hloss' in metrics:
            losses['hloss'] = format(metrics['hloss'], ".4f")
        return losses

    def _format_test(self, metrics: dict) -> dict:
        """Formatting for test metrics."""
        losses = {}
        if 'sdr' in metrics:
            losses['sdr'] = format(metrics['sdr'], '.3f')
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], '.3f')
        for source in self.model.sources:
            key = f'sdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
            key = f'nsdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
        return losses

    def train(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            formatted = self._format_train(metrics['train'])
            logger.info(
                bold(f'Train Summary | Epoch {epoch + 1} | {_summary(formatted)}'))
            formatted = self._format_train(metrics['valid'])
            logger.info(
                bold(f'Valid Summary | Epoch {epoch + 1} | {_summary(formatted)}'))
            if 'test' in metrics:
                formatted = self._format_test(metrics['test'])
                if formatted:
                    logger.info(bold(f"Test Summary | Epoch {epoch + 1} | {_summary(formatted)}"))

        epoch = 0
        for epoch in range(len(self.history), self.args.epochs):
            # Train one epoch
            self.model.train()  # Turn on BatchNorm & Dropout
            metrics = {}
            logger.info('-' * 70)
            logger.info("Training...")
            metrics['train'] = self._run_one_epoch(epoch)
            formatted = self._format_train(metrics['train'])
            logger.info(
                bold(f'Train Summary | Epoch {epoch + 1} | {_summary(formatted)}'))

            # Cross validation
            logger.info('-' * 70)
            logger.info('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            with torch.no_grad():
                valid = self._run_one_epoch(epoch, train=False)
                bvalid = valid
                bname = 'main'
                state = states.copy_state(self.model.state_dict())
                metrics['valid'] = {}
                metrics['valid']['main'] = valid
                key = self.args.test.metric
                for kind, emas in self.emas.items():
                    for k, ema in enumerate(emas):
                        with ema.swap():
                            valid = self._run_one_epoch(epoch, train=False)
                        name = f'ema_{kind}_{k}'
                        metrics['valid'][name] = valid
                        a = valid[key]
                        b = bvalid[key]
                        if key.startswith('nsdr'):
                            a = -a
                            b = -b
                        if a < b:
                            bvalid = valid
                            state = ema.state
                            bname = name
                    metrics['valid'].update(bvalid)
                    metrics['valid']['bname'] = bname

            valid_loss = metrics['valid'][key]
            mets = pull_metric(self.link.history, f'valid.{key}') + [valid_loss]
            if key.startswith('nsdr'):
                best_loss = max(mets)
            else:
                best_loss = min(mets)
            metrics['valid']['best'] = best_loss
            if self.args.svd.penalty > 0:
                kw = dict(self.args.svd)
                kw.pop('penalty')
                with torch.no_grad():
                    penalty = svd_penalty(self.model, exact=True, **kw)
                metrics['valid']['penalty'] = penalty

            formatted = self._format_train(metrics['valid'])
            logger.info(
                bold(f'Valid Summary | Epoch {epoch + 1} | {_summary(formatted)}'))

            # Save the best model
            if valid_loss == best_loss or self.args.dset.train_valid:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_state = states.copy_state(state)
                self.best_changed = True

            # Eval model every `test.every` epoch or on last epoch
            should_eval = (epoch + 1) % self.args.test.every == 0
            is_last = epoch == self.args.epochs - 1
            if should_eval or is_last:
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                if self.args.test.best:
                    state = self.best_state
                else:
                    state = states.copy_state(self.model.state_dict())
                compute_sdr = self.args.test.sdr and is_last
                with states.swap_state(self.model, state):
                    with torch.no_grad():
                        metrics['test'] = evaluate(self, compute_sdr=compute_sdr)
                formatted = self._format_test(metrics['test'])
                logger.info(bold(f"Test Summary | Epoch {epoch + 1} | {_summary(formatted)}"))
            self.link.push_metrics(metrics)

            if distrib.rank == 0:
                # Save model each epoch
                self._serialize(epoch)
                logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())
            if is_last:
                break

    def _run_one_epoch(self, epoch, train=True):
        args = self.args
        data_loader = self.loaders['train'] if train else self.loaders['valid']
        if distrib.world_size > 1 and train:
            data_loader.sampler.set_epoch(epoch)

        label = ["Valid", "Train"][train]
        name = label + f" | Epoch {epoch + 1}"
        total = len(data_loader)
        if args.max_batches:
            total = min(total, args.max_batches)
        logprog = LogProgress(logger, data_loader, total=total,
                              updates=self.args.misc.num_prints, name=name)
        averager = EMA()

        for idx, sources in enumerate(logprog):
            sources = sources.to(self.device)
            if train:
                sources = self.augment(sources)
                mix = sources.sum(dim=1)
            else:
                mix = sources[:, 0]
                sources = sources[:, 1:]

            if not train and self.args.valid_apply:
                estimate = apply_model(self.model, mix, split=self.args.test.split, overlap=0)
            else:
                estimate = self.dmodel(mix)
            if train and hasattr(self.model, 'transform_target'):
                sources = self.model.transform_target(mix, sources)
            assert estimate.shape == sources.shape, (estimate.shape, sources.shape)
            dims = tuple(range(2, sources.dim()))

            if args.optim.loss == 'l1':
                loss = F.l1_loss(estimate, sources, reduction='none')
                loss = loss.mean(dims).mean(0)
                reco = loss
            elif args.optim.loss == 'mse':
                loss = F.mse_loss(estimate, sources, reduction='none')
                loss = loss.mean(dims)
                reco = loss**0.5
                reco = reco.mean(0)
            else:
                raise ValueError(f"Invalid loss {self.args.loss}")
            weights = torch.tensor(args.weights).to(sources)
            loss = (loss * weights).sum() / weights.sum()

            ms = 0
            if self.quantizer is not None:
                ms = self.quantizer.model_size()
            if args.quant.diffq:
                loss += args.quant.diffq * ms

            losses = {}
            losses['reco'] = (reco * weights).sum() / weights.sum()
            losses['ms'] = ms

            if not train:
                nsdrs = new_sdr(sources, estimate.detach()).mean(0)
                total = 0
                for source, nsdr, w in zip(self.model.sources, nsdrs, weights):
                    losses[f'nsdr_{source}'] = nsdr
                    total += w * nsdr
                losses['nsdr'] = total / weights.sum()

            if train and args.svd.penalty > 0:
                kw = dict(args.svd)
                kw.pop('penalty')
                penalty = svd_penalty(self.model, **kw)
                losses['penalty'] = penalty
                loss += args.svd.penalty * penalty

            losses['loss'] = loss

            for k, source in enumerate(self.model.sources):
                losses[f'reco_{source}'] = reco[k]

            # optimize model in training mode
            if train:
                loss.backward()
                grad_norm = 0
                grads = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm()**2
                        grads.append(p.grad.data)
                losses['grad'] = grad_norm ** 0.5
                if args.optim.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        args.optim.clip_grad)

                if self.args.flag == 'uns':
                    for n, p in self.model.named_parameters():
                        if p.grad is None:
                            print('no grad', n)
                self.optimizer.step()
                self.optimizer.zero_grad()
                for ema in self.emas['batch']:
                    ema.update()
            losses = averager(losses)
            logs = self._format_train(losses)
            logprog.update(**logs)
            # Just in case, clear some memory
            del loss, estimate, reco, ms
            if args.max_batches == idx:
                break
            if self.args.debug and train:
                break
            if self.args.flag == 'debug':
                break
        if train:
            for ema in self.emas['epoch']:
                ema.update()
        return distrib.average(losses, idx + 1)
