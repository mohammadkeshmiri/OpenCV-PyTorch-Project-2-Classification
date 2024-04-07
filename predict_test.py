from Experiment import Experiment
from trainer.utils import patch_configs

dataloader_config, trainer_config = patch_configs(epoch_num_to_set=1, batch_size_to_set=1)
experiment = Experiment(dataloader_config=dataloader_config)
results = experiment.validate(trainer_config=trainer_config)