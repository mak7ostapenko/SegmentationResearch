import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from rope import get_image_roi
import src.nn.data.transforms.segmentation as aug


class RopeRobot(Dataset):
    """
    Class for a dataloaders from PyTorch.
    Main purpose of the class is prepare data during training.
    For inference use RopeRobotInference.
    """
    def __init__(self, csv_file, num_classes, transform=None, target_type='gray_scale',
                 default_roi=None, adaptive_roi=False, model_type = None):
        """
        :param csv_file: path to .csv file with columns=['image', 'mask']
        :param transform: PyTorch transforms for preparing a input data.
        :param target_type: type of mask, could be one from {'binary_RGB', 'gray_scale', 'mult_class_multi_channel'}.
        :param default_roi: list or array, region of interest to crop a input image.
        :param adaptive_roi: boolean, if True then use adaptive roi of image from metadata
            else use default_roi.
        """
        self.path_frame = pd.read_csv(csv_file, header=0)
        self.num_classes = num_classes
        self.transform = transform
        self.len = len(self.path_frame)
        self.target_type = target_type

        self.adaptive_roi = adaptive_roi
        self.default_roi = default_roi

        self.model_type = model_type

        # WARNING: num of colors depends on number segmentation classes
        self.colors = self.get_rope_labels()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = Image.open(self.path_frame['image'][index]).convert('RGB')
        
        if self.target_type == 'binary_RGB':
            target = Image.open(self.path_frame['mask'][index]).convert('RGB')
            target = np.array(target)
            target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
            target = np.where(target == 255, 1.0, 0)
            target = Image.fromarray(target)

        elif self.target_type == 'gray_scale':
            target = Image.open(self.path_frame['mask'][index])
               
            # WARNING: you need open for hand and hand fixer segmentation
            target = np.array(target)

            kernel = np.ones((3,3),np.uint8)
            target = cv2.dilate(target, kernel, iterations=3)
            #target = cv2.erode(target, kernel, iterations=2)
            print('np arr uniq = ', np.unique(target))
            target = Image.fromarray(target)


        else:
            raise ValueError('ERROR: target_type {} is not defined.'.format(self.target_type))

        # WARNING: only for input mask with classes={backgroud, hand, num_box, dispenser}
        if self.model_type == "first_stage":
            target = np.array(target)
            # segment {hand, num_box, dispenser} as hand
            target[target != 0] = 1
            target = Image.fromarray(target)

        elif self.model_type == "second_stage":
            target = np.array(target)
            # segment only {num_box, dispenser}
            # NOTE: hand_color_ind = 2, depends on dataset
            target[target == 2] = 0
            target[target == 3] = 2
            target = Image.fromarray(target)
        
        # WARNING: if adaptive_roi is True then will not use the CropPIL transform
        if self.adaptive_roi:
            image, roi = get_image_roi(image, self.path_frame['image'][index], self.default_roi)

            # WARNING: works only in case of PIL images
            if self.target_type in ['gray_scale', 'binary_RGB']:
                target = target.crop(roi)

            else:
                raise Exception("ERORR: tensor cropping is not implemented.")
        
        if self.transform is not None:
            image, target = self.transform(image, target)
        
        return image, target

    def get_rope_labels(self):
        """
        Load the mapping that associates classes with label colors.

        :return: np.array with dimensions (num_classes, 3)
        """
        # WARNING: num of colors depends on number segmentation classes
        colors = np.array([
            [0, 0, 0],
            [255, 0, 0],
            [0, 0, 142],
            [128, 64, 188],
            [244, 35, 232],
            [70, 150, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32]
        ])
        return colors[0: self.num_classes, :]


class RopeRobotInference(Dataset):
    """
    Class for a dataloaders from PyTorch.
    Main purpose of the class is prepare data during inference.
    """
    def __init__(self, paths_list, num_classes, transform=None, default_roi=None, adaptive_roi=False):
        """
        :param paths_list: a list of paths to the input images.
        :param transform: PyTorch transforms for preparing a input data.
        :param default_roi: list or array, region of interest to crop a input image.
        :param adaptive_roi: boolean, if True then use adaptive roi of image from metadata
            else use default_roi.
        """
        self.paths_list = paths_list
        self.transform = transform
        self.len = len(self.paths_list)
        self.num_classes = num_classes

        self.adaptive_roi = adaptive_roi
        self.default_roi = default_roi

        # WARNING: num of colors depends on number segmentation classes
        self.colors = self.get_rope_labels()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_path = self.paths_list[index]
        image = Image.open(image_path).convert('RGB')

        # WARNING if adaptive_roi is True then will not use the CropPIL transform
        if self.adaptive_roi:
            image, roi = get_image_roi(image, image_path, self.default_roi)

        orig_image = image.copy()

        if self.transform is not None:
            image = self.transform(image)

        orig_image = torch.from_numpy(np.array(orig_image))

        return image, image_path, orig_image

    def get_rope_labels(self):
        """
        Load the mapping that associates classes with label colors.

        :return: np.array with dimensions (num_classes, 3)
        """
        # WARNING: num of colors depends on number segmentation classes
        colors = np.array([
            [0, 0, 0],
            [255, 0, 0],
            [0, 0, 142],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32]
        ])
        return colors[0: self.num_classes, :]



#############################
# Code for debugging purposes
#############################
from src.nn.utils.mask import decode_segmap


def rope_datagen_test():
    csv_file = '...'
    batch_size = 1

    transform = aug.Compose([
        aug.RandomCrop(500),
        aug.AdjustGamma(0.8),
        aug.AdjustSaturation(0.8),
        aug.AdjustHue(0.3),
        aug.AdjustBrightness(0.5),
        aug.AdjustContrast(0.9),
        aug.RandomRotate(45),
        aug.RandomHorizontallyFlip(),

        aug.ToTensor(),
        aug.Normalize([0, 0, 0], [1, 1, 1]),
    ])

    rope_dataset = RopeRobot(csv_file, transform=transform, target_type='gray_scale', default_roi=None, adaptive_roi=False)
    loader = DataLoader(rope_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    for i, data in enumerate(loader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])
        f, axarr = plt.subplots(1, 2)

        for j in range(batch_size):
            image = cv2.cvtColor(imgs[j], cv2.COLOR_BGR2RGB)
            mask = labels.numpy()[j]
            mask = decode_segmap(mask, loader.dataset.colors)

            axarr[0].imshow(image)
            axarr[1].imshow(mask)
            plt.show()
