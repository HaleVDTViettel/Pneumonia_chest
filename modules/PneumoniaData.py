import os
import random
import numpy as np

import torch
from torchvision import transforms, datasets
import pytorch_lightning as pl
from torch.utils.data import WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .ImageFolder import CustomImageFolder

import yaml

class PneumoniaDataModule(pl.LightningDataModule):
    def __init__(self, h, data_dir):
        super().__init__()
        self.h = h
        self.data_dir = data_dir

    def setup(self, stage=None):
        h = self.h
        basic = self.h['basic']
        image_cropping = self.h['cropping']
        augmentations = self.h['augmentations']
        albumentations = self.h['albumentations']

        if image_cropping:
            data_transforms_train = transforms.Compose([
                transforms.Resize(size=(h["image_size"], h["image_size"])),
                transforms.CenterCrop(h["image_size"]),
                transforms.ToTensor(),
            ])
            data_transforms_val = transforms.Compose([
                transforms.Resize(size=(h["image_size"], h["image_size"])),
                transforms.CenterCrop(h["image_size"]),
                transforms.ToTensor(),
            ])
        elif augmentations:
            data_transforms_train = transforms.Compose([
                transforms.Resize(size=(h["image_size"], h["image_size"])),
                transforms.ToTensor(),
                transforms.RandomRotation(20),  # Randomly rotate the image within a range of (-20, 20) degrees
                transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally with 50% probability
                transforms.RandomResizedCrop((h["image_size"], h["image_size"]), scale=(0.8, 1.0)),  # Randomly crop the image and resize it
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Randomly change the brightness, contrast, saturation, and hue
                transforms.RandomApply([transforms.RandomAffine(0, translate=(0.1, 0.1))], p=0.5),  # Randomly apply affine transformations with translation
                transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2)], p=0.5),  # Randomly apply perspective transformations
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
            data_transforms_val = transforms.Compose([
                transforms.Resize(size=(h["image_size"], h["image_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        elif albumentations:
            data_transforms_train = A.Compose([
                A.Rotate(limit=20),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=1),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
                A.Perspective(scale=(0.05, 0.15), keep_size=True, p=0.5),
                A.Resize(height=h["image_size"], width=h["image_size"]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])        

            data_transforms_val = A.Compose([
                A.Resize(self.h["image_size"], self.h["image_size"]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            data_transforms_train = transforms.Compose([
                transforms.Resize(size=(h["image_size"], h["image_size"])),
                transforms.ToTensor(),
            ])
            data_transforms_val = transforms.Compose([
                transforms.Resize(size=(h["image_size"], h["image_size"])),
                transforms.ToTensor(),
            ])

        val_split = 0.2
        train_filenames, val_filenames = self._split_file_names(self.data_dir+"train/", val_split)
        
        # Load the datasets
        if albumentations:
            self.train_dataset = CustomImageFolder(h,
                                                self.data_dir+"train/",
                                                transform=data_transforms_train, 
                                                is_valid_file=lambda x: x in train_filenames)
            self.val_dataset = CustomImageFolder(h,
                                                self.data_dir+"train/",
                                                transform=data_transforms_val,
                                                is_valid_file=lambda x: x in val_filenames)    
            self.test_dataset = CustomImageFolder(h,
                                                self.data_dir+"test/",
                                                transform=data_transforms_val,
                                                is_valid_file=lambda x: self._is_image_file(x))
        else:
            self.train_dataset = datasets.ImageFolder(self.data_dir+"train/", 
                                                      transform=data_transforms_train,
                                                      is_valid_file=lambda x: x in train_filenames)
            self.val_dataset = datasets.ImageFolder(self.data_dir+"train/", 
                                                    transform=data_transforms_val,
                                                    is_valid_file=lambda x: x in val_filenames)
            self.test_dataset = datasets.ImageFolder(self.data_dir+"test/", 
                                                     transform=data_transforms_val,
                                                     is_valid_file=lambda x: self._is_image_file(x))

    
    def train_dataloader(self):
        if self.h["balance"]:
            sampler = self._create_weighted_sampler(self.train_dataset)
            return torch.utils.data.DataLoader(self.train_dataset, 
                                               batch_size=self.h["batch_size"], 
                                               sampler=sampler, 
                                               num_workers=4)
        else:
            return torch.utils.data.DataLoader(self.train_dataset, 
                                               batch_size=self.h["batch_size"], 
                                               shuffle=True, 
                                               num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, 
                                           batch_size=self.h["batch_size"], 
                                           num_workers=4)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, 
                                           batch_size=self.h["batch_size"], 
                                           num_workers=4)

    def _extract_patient_ids(self, filename):
        patient_id = filename.split('_')[0].replace("person", "")
        return patient_id

    def _is_image_file(self, file_path):
        return file_path.lower().endswith((".jpeg", ".jpg", ".png"))

    def _split_file_names(self, input_folder, val_split_perc):
        # Pneumonia files contain patient id, so we group split them by patient to avoid data leakage
        pneumonia_patient_ids = set([self._extract_patient_ids(fn) for fn in os.listdir(os.path.join(input_folder, 'PNEUMONIA'))])
        pneumonia_val_patient_ids = random.sample(list(pneumonia_patient_ids), int(val_split_perc * len(pneumonia_patient_ids)))

        pneumonia_val_filenames = []
        pneumonia_train_filenames = []

        for filename in os.listdir(os.path.join(input_folder, 'PNEUMONIA')):
            if self._is_image_file(filename):
                patient_id = self._extract_patient_ids(filename)
                if patient_id in pneumonia_val_patient_ids:
                    pneumonia_val_filenames.append(os.path.join(input_folder, 'PNEUMONIA', filename))
                else:
                    pneumonia_train_filenames.append(os.path.join(input_folder, 'PNEUMONIA', filename))

        # Normal (by file, no patient information in file names)
        normal_filenames  = [os.path.join(input_folder, 'NORMAL', fn) for fn in os.listdir(os.path.join(input_folder, 'NORMAL'))]
        normal_filenames = [filename for filename in normal_filenames if self._is_image_file(filename)]
        normal_val_filenames = random.sample(normal_filenames, int(val_split_perc * len(normal_filenames)))
        normal_train_filenames = list(set(normal_filenames)-set(normal_val_filenames))

        train_filenames = pneumonia_train_filenames + normal_train_filenames
        val_filenames = pneumonia_val_filenames + normal_val_filenames

        return train_filenames, val_filenames        


    def _create_weighted_sampler(self, dataset):
        targets = dataset.targets
        class_counts = np.bincount(targets)
        class_weights = 1.0 / class_counts
        weights = [class_weights[label] for label in targets]
        sampler = WeightedRandomSampler(weights, len(weights))
        return sampler