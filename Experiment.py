import os
import csv
from operator import itemgetter

import torch
import torch.nn as nn
import torch.optim as optim

from trainer.datasets import get_data, get_val_data
from torch.optim.lr_scheduler import MultiStepLR

from trainer import trainer, hooks, configuration
from trainer.utils import setup_system, patch_configs
from trainer.metrics import AccuracyEstimator
from trainer.tensorboard_visualizer import TensorBoardVisualizer
from torchvision.transforms import v2 as transforms
from resnet_model import pretrained_resnet50

import matplotlib.pyplot as plt

class Experiment:
    def __init__(
        self,
        system_config: configuration.SystemConfig = configuration.SystemConfig(),
        dataset_config: configuration.DatasetConfig = configuration.DatasetConfig(),
        dataloader_config: configuration.DataloaderConfig = configuration.DataloaderConfig(),
        optimizer_config: configuration.OptimizerConfig = configuration.OptimizerConfig()
    ):
        resnet50 = pretrained_resnet50(fix_feature_extractor=False, num_class=13)
        dataset_config.train_transforms = resnet50.get_train_transforms()
        dataset_config.test_transforms = resnet50.get_test_transforms()
        self.loader_train, self.loader_test, self.labels = get_data(dataset_config, dataloader_config)
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config
        
        self.model = resnet50.get_model()
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
        self.visualizer = TensorBoardVisualizer(resnet50.name)
        setup_system(system_config)

    def run(self, trainer_config: configuration.TrainerConfig, check_point_name = None) -> dict:

        if check_point_name is not None:
            check_point_path = os.path.join(trainer_config.model_dir, check_point_name)
            self.model.load_state_dict(torch.load(check_point_path)['state_dict'])

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

    def draw_test_set(self, rows, columns):
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

    def validate(self, trainer_config: configuration.TrainerConfig, check_point_name = "ResNet50_best_20240403_1155pm.pth"):

        check_point_path = os.path.join(trainer_config.model_dir, check_point_name)
        self.model.load_state_dict(torch.load(check_point_path)['state_dict'])

        device = torch.device(trainer_config.device)
        self.model = self.model.to(device)
        self.model = self.model.eval()
        
        dataloader_config = configuration.DataloaderConfig(batch_size=1, num_workers=1)
        val_loader = get_val_data(self.dataset_config, dataloader_config)
        
        submission_dict = []
        for sample in val_loader:
            
            inputs = sample[0].to(device)
            predict = self.model(inputs)
            predict = predict.softmax(dim=1).detach()
            predict = predict.argmax(dim=1).item()
            submission_dict.append([sample[1][0], str(self.labels[predict])])

        fields = ['id', 'class']
 
        # name of csv file
        filename = os.path.join(self.dataset_config.root_dir , "submission.csv")
        
        # writing to csv file
        with open(filename, 'w') as csvfile:
            # creating a csv dict writer object
            writer = csv.writer(csvfile)
        
            # writing headers (field names)
            writer.writerow(fields)
        
            # writing data rows
            writer.writerows(submission_dict)


# %%
def main():
    '''Run the experiment
    '''
    # patch configs depending on cuda availability
    
    dataloader_config, trainer_config = patch_configs(epoch_num_to_set=1000, batch_size_to_set=8)
    dataset_config = configuration.DatasetConfig()

    experiment = Experiment(dataset_config=dataset_config, dataloader_config=dataloader_config)
    results = experiment.run(trainer_config, check_point_name = None) #'ResNet_best_2024-03-30 04:12:20.291177.pth')


# %%
if __name__ == '__main__':
    main()