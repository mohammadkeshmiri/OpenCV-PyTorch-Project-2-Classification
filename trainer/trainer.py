import os
import datetime

from typing import Union, Callable
from pathlib import Path
from operator import itemgetter

import torch

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .hooks import test_hook_default, train_hook_default
from .visualizer import Visualizer

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loader_train: torch.utils.data.DataLoader,
        loader_test: torch.utils.data.DataLoader,
        loss_fn: Callable,
        metric_fn: Callable,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Callable,
        device: Union[torch.device, str] = "cuda",
        model_save_best: bool = True,
        model_saving_frequency: int = 1,
        save_dir: Union[str, Path] = "checkpoints",
        data_getter: Callable = itemgetter("image"),
        target_getter: Callable = itemgetter("target"),
        stage_progress: bool = True,
        visualizer: Union[Visualizer, None] = None,
        get_key_metric: Callable = itemgetter("top1"),
    ):
        self.model = model
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_save_best = model_save_best
        self.model_saving_frequency = model_saving_frequency
        self.save_dir = Path(save_dir)
        self.get_key_metric = get_key_metric
        self.stage_progress = stage_progress
        self.data_getter = data_getter
        self.target_getter = target_getter
        self.hooks = {}
        self.visualizer = visualizer
        self.metrics = {"epoch": [], "train_loss": [], "test_loss": [], "test_metric": []}       
        self._register_default_hooks()

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, epochs: int):
        previos_best =''
        iterator = tqdm(range(epochs), dynamic_ncols=True)
        for epoch in iterator:
            output_train = self.hooks["train"](
                self.model,
                self.loader_train,
                self.loss_fn,
                self.optimizer,
                self.device,
                prefix="[{}/{}]".format(epoch, epochs),
                stage_progress=self.stage_progress,
                data_getter=self.data_getter,
                target_getter=self.target_getter)
            output_test = self.hooks["test"](
                self.model,
                self.loader_test,
                self.loss_fn,
                self.metric_fn,
                self.device,
                prefix="[{}/{}]".format(epoch, epochs),
                stage_progress=self.stage_progress,
                data_getter=self.data_getter,
                target_getter=self.target_getter,
                get_key_metric=self.get_key_metric)

            if self.visualizer:
                self.visualizer.update_charts(
                    None, output_train['loss'], output_test['metric'], output_test['loss'],
                    self.optimizer.param_groups[0]['lr'], epoch
                )

            self.metrics['epoch'].append(epoch)
            self.metrics['train_loss'].append(output_train['loss'])
            self.metrics['test_loss'].append(output_test['loss'])
            self.metrics['test_metric'].append(output_test['metric'])

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(output_train['loss'])
                else:
                    self.lr_scheduler.step()
                
            if self.hooks["end_epoch"] is not None:
                self.hooks["end_epoch"](iterator, epoch, output_train, output_test)

            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
                }
            if self.model_save_best:
                best_acc = max([self.get_key_metric(item) for item in self.metrics['test_metric']])
                current_acc = self.get_key_metric(output_test['metric'])

                if current_acc >= best_acc:
                    os.makedirs(self.save_dir, exist_ok=True)
                    file_path = os.path.join(self.save_dir, self.model.__class__.__name__) + '_best_' + str(datetime.datetime.now()) + '.pth'
                    torch.save(checkpoint, file_path)

                    if os.path.exists(previos_best):
                        os.remove(previos_best)
                    previos_best = file_path
            else:
                if (epoch + 1) % self.model_saving_frequency == 0:
                    os.makedirs(self.save_dir, exist_ok=True)
                    torch.save(
                        checkpoint,
                        os.path.join(self.save_dir, self.model.__class__.__name__) + '_' +
                        str(datetime.datetime.now()) + '.pth'
                    )

        return self.metrics
    
    def register_hook(self, hook_type, hook_fn):

        self.hooks[hook_type] = hook_fn

    def _register_default_hooks(self):
        self.register_hook("train", train_hook_default)
        self.register_hook("test", test_hook_default)
        self.register_hook("end_epoch", None)

    
