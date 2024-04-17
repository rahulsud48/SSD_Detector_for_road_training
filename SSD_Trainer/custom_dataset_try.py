import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

import matplotlib.pyplot as plt

from trainer.utils import resize, random_flip
from trainer.encoder import DataEncoder

import os
import numpy as np
import pandas as pd
import cv2

from PIL import Image
from albumentations import (
    CLAHE,
    Blur,
    OneOf,
    Compose,
    RGBShift,
    GaussNoise,
    RandomGamma,
    RandomContrast,
    RandomBrightness,
)

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import HueSaturationValue
from albumentations.augmentations.transforms import Normalize
from albumentations import Resize

import sys
from typing import Callable, Iterable
from dataclasses import dataclass

from torchvision.transforms import ToTensor


@dataclass
class SystemConfig:
    seed: int = 42  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = False  # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # make cudnn deterministic (reproducible training)


@dataclass
class DatasetConfig:
    root_dir: str = "data"  # dataset directory root
    train_transforms: Iterable[Callable] = (
        ToTensor(),
    )  # data transformation to use during training data preparation
    test_transforms: Iterable[Callable] = (
        ToTensor(),
    )  # data transformation to use during test data preparation


@dataclass
class DataloaderConfig:
    batch_size: int = 250  # amount of data to pass through the network at each forward-backward iteration
    num_workers: int = 5  # number of concurrent processes using to prepare data


@dataclass
class OptimizerConfig:
    learning_rate: float = 0.001  # determines the speed of network's weights update
    momentum: float = 0.9  # used to improve vanilla SGD algorithm and provide better handling of local minimas
    weight_decay: float = 0.0001  # amount of additional regularization on the weights values
    lr_step_milestones: Iterable = (
        30, 40
    )  # at which epoches should we make a "step" in learning rate (i.e. decrease it in some manner)
    lr_gamma: float = 0.1  # multiplier applied to current learning rate at each of lr_ctep_milestones


@dataclass
class TrainerConfig:
    model_dir: str = "checkpoints"  # directory to save model states
    model_save_best: bool = True  # save model with best accuracy
    model_saving_frequency: int = 1  # frequency of model state savings per epochs
    device: str = "cpu"  # device to use for training.
    epoch_num: int = 50  # number of times the whole dataset will be passed through the network
    progress_bar: bool = False  # enable progress bar visualization during train process


class ListDataset(Dataset):
    def __init__(self, root_dir, data_dir, list_file, classes, mode, transform, input_size):
        '''
        Args:
          root_dir: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.classes = classes
        self.mode = mode
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []

        list_file_path = os.path.join(root_dir, list_file)
        with open(list_file_path) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                xmin = splited[1 + 5 * i]
                ymin = splited[2 + 5 * i]
                xmax = splited[3 + 5 * i]
                ymax = splited[4 + 5 * i]
                class_label = splited[5 + 5 * i]
                box.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                label.append(int(class_label))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))


    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        path = os.path.join(self.root_dir, self.data_dir, self.fnames[idx])
        img = cv2.imread(path)
        if img is None or np.prod(img.shape) == 0:
            print('cannot load image from path: ', path)
            sys.exit(-1)

        img = img[..., ::-1]

        boxes = self.boxes[idx].clone()
        boxes = self.boxes[idx]
        labels = self.labels[idx]
        size = self.input_size

        # Resize & Flip
        img, boxes = resize(img, boxes, (size, size))
        # if self.mode == 'train':
        #     img, boxes = random_flip(img, boxes)
        # Data augmentation.
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, boxes, labels


    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, w, h)
        encoder = DataEncoder((w, h))
        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = encoder.encode(boxes[i], labels[i])
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples


def patch_configs(epoch_num_to_set=TrainerConfig.epoch_num, batch_size_to_set=DataloaderConfig.batch_size):
    """ Patches configs if cuda is not available

    Returns:
        returns patched dataloader_config and trainer_config

    """
    # default experiment params
    num_workers_to_set = DataloaderConfig.num_workers

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        batch_size_to_set = 16
        num_workers_to_set = 2
        epoch_num_to_set = 1

    dataloader_config = DataloaderConfig(batch_size=batch_size_to_set, num_workers=num_workers_to_set)
    trainer_config = TrainerConfig(device=device, epoch_num=epoch_num_to_set, progress_bar=True)
    return dataloader_config, trainer_config




def convert_img_tensor_to_numpy(torch_tensor):
    # Convert PyTorch tensor to NumPy array
    numpy_image = torch_tensor.numpy()

    # Convert to uint8 and transpose if necessary
    if numpy_image.dtype != np.uint8:
        numpy_image = (numpy_image * 255).astype(np.uint8)

    # If image is in CHW format (channels, height, width), transpose it to HWC format (height, width, channels)
    if numpy_image.shape[0] == 3:  # If channels are the first dimension
        numpy_image = numpy_image.transpose(1, 2, 0)
        
    return numpy_image


def collate(batch, input_size):


    imgs = [x[0] for x in batch]
    boxes = [x[1] for x in batch]
    labels = [x[2] for x in batch]

    h = w = input_size
    num_imgs = len(imgs)
    inputs = torch.zeros(num_imgs, 3, w, h)
    encoder = DataEncoder((w, h))
    loc_targets = []
    cls_targets = []
    for i in range(num_imgs):
        inputs[i] = imgs[i]
        loc_target, cls_target = encoder.encode(boxes[i], labels[i])
        loc_targets.append(loc_target)
        cls_targets.append(cls_target)
    return inputs, torch.stack(loc_targets), torch.stack(cls_targets)


def view_images(img_tensor, box, class_list):
    img = convert_img_tensor_to_numpy(img_tensor)
    box = box.numpy()
    class_list = class_list.numpy()
    for i in range(len(box)):
        x_min = int(box[i][0])
        y_min = int(box[i][1])
        x_max = int(box[i][2])
        y_max = int(box[i][3])
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        cv2.putText(img, classes_dict[class_list[i]], (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    plt.figure(figsize=(12, 18))
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":

    dataloader_config, trainer_config = patch_configs(epoch_num_to_set=10, batch_size_to_set=30)
    dataset_config = DatasetConfig(
        root_dir="../../Datasets/Road_Scene_Object_Detection",
        train_transforms=[
    #         Normalize(),
            Resize(height=300, width=300),
            ToTensorV2()
        ]
    )    

    dataset_train = ListDataset(
        root_dir=dataset_config.root_dir,
        data_dir = 'export',
        list_file='annots_transformed_train.txt',

        classes=[
            "__background__",
            "biker",
            "car",
            "pedestrian",
            "trafficLight",
            "trafficLight-Green",
            "trafficLight-GreenLeft",
            "trafficLight-Red",
            "trafficLight-RedLeft",
            "trafficLight-Yellow",
            "trafficLight-YellowLeft",
            "truck"
        ],
        mode='train',
        transform=Compose(dataset_config.train_transforms),
        input_size=300
    )

    loader_train = DataLoader(
        dataset=dataset_train,
        batch_size=dataloader_config.batch_size,
        shuffle=True,
    #     collate_fn=dataset_train.collate_fn,
        num_workers=dataloader_config.num_workers,
        pin_memory=True
    )

    classes_dict = {
            0:"biker",
            1:"car",
            2:"pedestrian",
            3:"trafficLight",
            4:"trafficLight-Green",
            5:"trafficLight-GreenLeft",
            6:"trafficLight-Red",
            7:"trafficLight-RedLeft",
            8:"trafficLight-Yellow",
            9:"trafficLight-YellowLeft",
            10:"truck",
    }


    img_id = 0

    dataset_train.__getitem__(img_id)

    dataset_train[img_id][0]



    # view_images(dataset_train[img_id][0],dataset_train[img_id][1],dataset_train[img_id][2])

    iterator = iter(loader_train)
    one_batch = next(iterator)

    print(one_batch)