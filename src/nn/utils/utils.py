import os

import torch
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from requests import get
import torch.nn.functional as F
# import pycocotools.mask as mask_utils
from sklearn.model_selection import train_test_split


def save_snapshot(model, optimizer, loss, epoch, train_history, snapshot_file):
    """
    Save snapshot of model.
    """
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'train_history': train_history.to_dict()
    }, snapshot_file)


def restore_snapshot(model, optimizer, snapshot_file):
    """
    Restore snapshot from file.
    """
    checkpoint = torch.load(snapshot_file)
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    model.load_state_dict(checkpoint['model'])

    #if optimizer is not None:
    #    optimizer.load_state_dict(checkpoint['optimizer'])

    train_history = pd.DataFrame.from_dict(checkpoint['train_history'])

    return model, start_epoch, train_history, best_loss


def load_checkpoint_enet(model, model_path):
    """Saves the model in a specified directory with a specified name.save
    Keyword arguments:
    - model (``nn.Module``): The stored model state is copied to this model
    instance.
    - optimizer (``torch.optim``): The stored optimizer state is copied to this
    optimizer instance.
    - folder_dir (``string``): The path to the folder where the saved model
    state is located.
    - filename (``string``): The model filename.
    Returns:
    The epoch, mean IoU, ``model``, and ``optimizer`` loaded from the
    checkpoint.
    """

    # Load the stored model parameters to the model instance
    checkpoint = torch.load(model_path)
    model.basemodel.load_state_dict(checkpoint['state_dict'])

    return model


def get_class_weights(labels):
    """Compute balanced class weights

    Arguments
        labels : list ar array
            Labels of training samples

    Returns
        weights_dict : dict
              Weight for each class label

    """
    unique, counts = np.unique(labels, return_counts=True)
    weights = counts / np.sum(counts)
    weights_dict = dict(zip(unique, weights))

    return weights_dict


def union_of_instance_masks(mask):
    """
    Function takes mask tensor (tensor.shape == (h, w, num_instance))
    transform it to matrix [h, w] by union of elements along 3rd dimension.
    
    :param mask: mask tensor [h, w, num_instance] 0 - background, 1 - foreground
    :return: matrix [h, w]

    """
    temp = np.zeros_like(mask[...,-1])
    
    for i in range(mask.shape[-1]):
        temp[mask[..., i] == 1] = 1
    
    return temp


# def mask_from_RLE(path_to_rle):
#     """
#     Function takes path to genereted by pycocotools.mask.frPyObjects RLE
#     transform it to matrix [h, w, 1] by union of all instance masks.
#
#     :param path_to_rle: obviously from the name;
#     :return: mask matrix [h, w, 1]4
#
#     """
#     with open(path_to_rle, 'rb') as f:
#         rle = pickle.load(f)
#
#     mask = mask_utils.decode(rle)
#     mask = union_of_instance_masks(mask)
#
#     return mask


def gen_train_test_frames(image_dir, mask_dir, test_size=0.2, output_dir='./dataset/'):
    """
    Example of usage:
        from utils.utils import gen_train_test_frames
        gen_train_test_frames(image_dir='img_with_mask/',
                              mask_dir='RLEs/',
                              output_dir='dataset/')

    :param image_dir:
    :param mask_dir:
    :param test_size:
    :param output_dir:
    :return:
    """
    images = os.listdir(image_dir)
    image_paths = [image_dir + image for image in images]
    
    mask_paths = list()
    result_image_paths = list()
    for image_path in image_paths:
        mask_path = mask_dir + image_path.split('/')[-1].split('.')[0] + '_RLE.pckl'
    
        if os.path.isfile(mask_path):
            mask_paths.append(mask_path)
            result_image_paths.append(image_path)
    
    image_paths = result_image_paths 
    image_mask_frame = pd.DataFrame({'image': image_paths, 'mask': mask_paths})

    image_mask_frame_train, image_mask_frame_test = train_test_split(image_mask_frame, test_size=test_size, shuffle=True)

    image_mask_frame_train.to_csv(output_dir+'segment_train.csv', index=False)
    image_mask_frame_test.to_csv(output_dir + 'segment_test.csv', index=False)


def load_png_masks(filename):
    """
    :return: a list of serialized PNG mask images
    """
    with open(filename, 'rb') as f:
        mask_list = pickle.load(f)
    return mask_list


def download(url, file_name):
    """Download file by its url.
    """
    with open(file_name, "wb") as file:
        response = get(url)
        file.write(response.content)


def interpolate_mask(outputs, targets, flatten=False):
    """

    :param outputs:
    :param targets:
    :param num_classes:
    :param flatten:
    :return:
    """
    n, c, h, w = outputs.size()
    nt, ht, wt = targets.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        outputs = F.interpolate(outputs, size=(ht, wt), mode="bilinear", align_corners=True)

    if flatten:
        outputs = outputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        targets = targets.view(-1)

    return outputs, targets


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return '%.3f' % self.avg


class AverageMeterDict(dict):
    def __init__(self, function_names):
        """
        :param function_names:
        """
        super(AverageMeterDict, self).__init__()
        self.avg_meter_dict = dict()

        for function_name in function_names:
            self.avg_meter_dict[function_name] = AverageMeter()

    def avg_update(self, values):
        assert len(values.keys()) == len(self.avg_meter_dict.keys())

        for value_name, value in values.items():
            if type(values) == torch.Tensor:
                value = value.cpu().item()

            self.avg_meter_dict[value_name].update(value)
