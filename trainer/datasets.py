import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, Subset
from torchvision.transforms import functional as F
from sklearn.model_selection import train_test_split

from torchvision.transforms import ToTensor


import matplotlib.pyplot as plt

class KenyanFood13Dataset(Dataset):
    def __init__(self, root_dir, transform, image_size, is_validation=False):
        '''
        Args:
          root_dir: (str) ditectory to images.
          list_file: (str) path to index file.
          mode: (str) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root_dir = root_dir
        self.classes = []
        self.transform = transform
        self.image_size = image_size
        self.num_samples = 0
        self.is_validation = is_validation

        csv_path = os.path.join(root_dir, 'train.csv')
        if (is_validation):
            csv_path = os.path.join(root_dir, 'test.csv')
        self.train_df = pd.read_csv(csv_path) #, delimiter='*,*', engine='python')

        # initialize the train data dictionary
        self.data_dict = {
            'image_path': [],
            'class': []
        }

        self.labels = []

        self.image_dir = os.path.join(root_dir, 'images/images')

        for index, row in self.train_df.iterrows():
            self.data_dict['image_path'].append(str(row['id']))

            if not is_validation:
                if row['class'] not in self.labels:
                    self.labels.append(row['class'])
                
                self.data_dict['class'].append(self.labels.index(row['class']))

            self.num_samples += 1


    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          target: (tensor) target class.
        '''
        # Load image and boxes.
        image_name = self.data_dict['image_path'][idx]
        image_path = os.path.join(self.image_dir, image_name + ".jpg")
        img = cv2.imread(image_path)
        if img is None or np.prod(img.shape) == 0:
            print('cannot load image from path: ', path)
            sys.exit(-1)
        
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self.image_to_square(img)
        img = cv2.resize(img, (self.image_size, self.image_size) , interpolation= cv2.INTER_LINEAR)
        
        # F.resize(img, self.image_size)

        img = img[..., ::-1].copy()  # BGR to RGB

        if not self.is_validation:
            img_class = torch.tensor(self.data_dict['class'][idx])
        else:
            img_class = image_name

        if self.transform:
            # img = ToTensor()(img)
            img = self.transform(img)

        return img, img_class 

    def __len__(self):
        return self.num_samples
    
    def image_to_square(self, image):
        desired_size = max(image.shape[:2])
        old_size = image.shape[:2] # old_size is in (height, width) format

        delta_w = desired_size - old_size[1]
        delta_h = desired_size - old_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)

        return new_im

def get_data(dataset_config, dataloader_config):
    root_dir = dataset_config.root_dir
    train_transform= dataset_config.train_transforms
    test_transforms= dataset_config.test_transforms
    image_size = dataset_config.image_size
    batch_size = dataloader_config.batch_size
    train_dataset = KenyanFood13Dataset(root_dir, train_transform, image_size)
    test_dataset = KenyanFood13Dataset(root_dir, test_transforms, image_size)
    
    
    train_indices, val_indices = train_test_split(list(range(len(train_dataset))), test_size=0.2, random_state=42) # use 21 for altarnative test set
    train_dataset = Subset(train_dataset, train_indices)
    test_dataset = Subset(test_dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader, train_dataset.dataset.labels

def get_val_data(dataset_config, dataloader_config):
    root_dir = dataset_config.root_dir
    test_transforms= dataset_config.test_transforms
    image_size = dataset_config.image_size
    batch_size = dataloader_config.batch_size
    val_dataset = KenyanFood13Dataset(root_dir, test_transforms, image_size, is_validation=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return val_loader

def draw_image(image, class_name):
    fig1 = plt.figure("Figure 2")
    fig1.suptitle(class_name)
    plt.imshow(image)
    plt.show()