import os
from operator import itemgetter

import torch
import torch.nn as nn
import torch.optim as optim

from trainer.datasets import KenyanFood13Dataset, get_data
from torch.optim.lr_scheduler import MultiStepLR

from trainer import trainer, hooks, configuration
from trainer.utils import setup_system, patch_configs
from trainer.metrics import AccuracyEstimator
from trainer.tensorboard_visualizer import TensorBoardVisualizer
from torchvision.transforms import v2 as transforms
from lenet import LeNet5
from resnet_model import pretrained_resnet18

import matplotlib.pyplot as plt

class Experiment:
    def __init__(
        self,
        system_config: configuration.SystemConfig = configuration.SystemConfig(),
        dataset_config: configuration.DatasetConfig = configuration.DatasetConfig(),
        dataloader_config: configuration.DataloaderConfig = configuration.DataloaderConfig(),
        optimizer_config: configuration.OptimizerConfig = configuration.OptimizerConfig()
    ):
        self.loader_train, self.loader_test = get_data(dataset_config, dataloader_config)

        self.model = LeNet5()
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric_fn = AccuracyEstimator(topk=(1, ))
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            momentum=optimizer_config.momentum
        )
        self.lr_scheduler = MultiStepLR(
            self.optimizer, milestones=optimizer_config.lr_step_milestones, gamma=optimizer_config.lr_gamma
        )
        self.lr_scheduler = None
        self.visualizer = TensorBoardVisualizer()
        setup_system(system_config)

    def run(self, trainer_config: configuration.TrainerConfig, check_point_name) -> dict:

        if check_point_name is not None:
            check_point_path = os.path.join(trainer_config.model_dir, check_point_name)
            self.model.load_state_dict(torch.load(check_point_path))

        device = torch.device(trainer_config.device)
        self.model = self.model.to(device)
        self.loss_fn = self.loss_fn.to(device)

        model_trainer = trainer.Trainer(
            model=self.model,
            loader_train=self.loader_train,
            loader_test=self.loader_test,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            device=device,
            data_getter=itemgetter(0),
            target_getter=itemgetter(1),
            stage_progress=trainer_config.progress_bar,
            get_key_metric=itemgetter("top1"),
            visualizer=self.visualizer,
            model_saving_frequency=trainer_config.model_saving_frequency,
            save_dir=trainer_config.model_dir
        )

        model_trainer.register_hook("end_epoch", hooks.end_epoch_hook_classification)
        self.metrics = model_trainer.fit(trainer_config.epoch_num)
        return self.metrics

    def test(self, rows, columns):
        fig, ax = plt.subplots(
            nrows=rows, ncols=columns, figsize=(10, 10), gridspec_kw={
                'wspace': 0,
                'hspace': 0.05
            }
        )

        dataset = self.loader_test.dataset
        dataset_lengh = len(dataset)
        for index, axi in enumerate(ax.flat):
            image, label = dataset[index]
            image = image.permute(1, 2, 0)
            axi.imshow(image)
            axi.axis('off')

        # labels = {}
        # for index in range(dataset_lengh):
        #     image, label = dataset[index]
        #     if label.item() not in labels.keys():
        #         labels[label.item()] = 1
        #     else:
        #         labels[label.item()] += 1

        fig.show()
        plt.pause(0)

# %%
def main():
    '''Run the experiment
    '''
    # patch configs depending on cuda availability
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    dataloader_config, trainer_config = patch_configs(epoch_num_to_set=1000, batch_size_to_set=32)
    dataset_config = configuration.DatasetConfig(root_dir="data", image_size=32)#224)
    dataset_config.train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    dataset_config.test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)

    ])
    experiment = Experiment(dataset_config=dataset_config, dataloader_config=dataloader_config)
    results = experiment.run(trainer_config)

    # experiment.test(10,10)


# %%
if __name__ == '__main__':
    main()