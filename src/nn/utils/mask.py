import torch
import numpy as np
import matplotlib.pyplot as plt

from src.nn.utils.utils import interpolate_mask


def generate_segm_class_colors(num_classes):
    """
    Generate colors for class each class for using in further visualization.

    :param num_classes: number of classes in segmentation dataset.
    :return:mask
    """
    # TODO: add better color setting, not random
    colors = np.random.choice(range(80, 255), size=(num_classes, 3))

    return colors


def decode_mask_batch(masks, dataloader):
    """

    :param masks:
    :param dataloader:
    :return:
    """
    masks = masks.detach().cpu().numpy()

    decoded_masks = list()
    for mask in masks:
        mask = decode_segmap(mask, dataloader.dataset.colors)

        decoded_masks.append(mask)

    decoded_masks = np.array(decoded_masks)
    decoded_masks = np.moveaxis(decoded_masks, -1, 1)

    decoded_masks = torch.from_numpy(decoded_masks)
    return decoded_masks


def get_prediction_mask(outputs, targets, binary=False):
    """

    :param targets: torch tensor of target masks batch with
        shape=(num_samples, num_classes, height, width).
    :param outputs: torch tensor of predicted masks batch with
        shape=(num_samples, num_classes, height, width).
    :return:
    """
    outputs, _ = interpolate_mask(outputs, targets, flatten=False)
    
    outputs = torch.sigmoid(outputs)
    if binary:
        outputs = outputs.detach().cpu().numpy()
        outputs[outputs > 0.5] = 1
        outputs[outputs <= 0.5] = 0

    else:
        outputs = outputs.data.max(1)[1].cpu().numpy()
    
    return outputs


def apply_mask_on_batch(images, targets, outputs, dataloader, color=None):
    """

    :param images: torch tensor of images batch with
        shape=(num_samples, num_channels, height, width).
    :param targets: torch tensor of target masks batch with
        shape=(num_samples, num_classes, height, width).
    :param outputs: torch tensor of predicted masks batch with
        shape=(num_samples, num_classes, height, width).
    :param dataloader: torch dataloader.
    :return:
    """
    images = images.cpu().numpy()
    images *= (255.0 / images.max())

    outputs = get_prediction_mask(outputs, targets)
    images = np.moveaxis(images, 1, -1)

    masked_images = list()
    for image, output in zip(images, outputs):
        if color is None:
            output = decode_segmap(output, dataloader.dataset.colors)
        else:
            output = decode_segmap(output, color)

        # masked_image = alpha_blend(image, output)
        masked_images.append(output) # TODO: mask

    masked_images = np.array(masked_images)
    masked_images = np.moveaxis(masked_images, -1, 1)
    masked_images = torch.from_numpy(masked_images)

    return masked_images


def encode_segmap(mask, mask_labels):
    """
    Encode segmentation label images as pascal classes.

    :param mask: np.array raw segmentation label image of dimension (M, N, 3),
        in which the dataset classes are encoded as colours.
    :param mask_labels: np.array with dimensions (num_classes, 3),
        where each row is RGB color for each class in dataset.
    :return label_mask: np.array class map with dimensions (M,N),
        where the value at a given location is the integer denoting
        the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)

    for ii, label in enumerate(mask_labels):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii

    label_mask = label_mask.astype(int)

    return label_mask


def decode_segmap(label_mask, mask_labels, plot=False):
    """
    Decode segmentation class labels into a color image.

    :param label_mask: np.array with shape (M,N) of integer values denoting
        the class label at each spatial location.
    :param mask_labels: np.array with dimensions (num_classes, 3),
        where each row is RGB color for each class in dataset.
    :param plot: bool, optional whether to show the resulting color image
        in a figure.
    :return: np.array, the resulting decoded color image.
    """
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()

    num_classes = mask_labels.shape[0]
    for ll in range(0, num_classes):
        r[label_mask == ll] = mask_labels[ll, 0]
        g[label_mask == ll] = mask_labels[ll, 1]
        b[label_mask == ll] = mask_labels[ll, 2]

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0

    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

