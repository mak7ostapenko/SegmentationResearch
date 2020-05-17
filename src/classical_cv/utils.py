import os

import cv2
import numpy as np
from requests import get
import matplotlib.pyplot as plt


def show_image(image, cmap=None, figsize=(10, 10)):
    """Plot image."""
    if cmap is 'bgr':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cmap = None

    plt.figure(figsize=figsize)
    plt.imshow(image, cmap)
    plt.show()


def remove_png(out_path = '.dataset/temp_matching_res/'):
    """Remove all .png images in directory"""
    for img_name in os.listdir(out_path):
        if 'png' in img_name:
            os.remove(os.path.join(out_path, img_name))


def show_2x3_image_grid(images, cmap='rgb', figsize=(20, 8)):
    """Plot grid of images."""
    if cmap is 'bgr':
        images = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                           for image in images])

    fig, axarr = plt.subplots(2, 3, figsize=figsize)
    axarr[0, 0].imshow(images[0])
    axarr[0, 1].imshow(images[1])
    axarr[0, 2].imshow(images[2])
    axarr[1, 0].imshow(images[3])
    axarr[1, 1].imshow(images[4])
    axarr[1, 2].imshow(images[5])

    plt.show()


def show_2x3_hist_grid(hists, figsize=(20, 8)):
    """Plot grid of histograms."""
    fig, axarr = plt.subplots(2, 3, figsize=figsize)
    axarr[0, 0].hist(hists[0].ravel(),256,[0,256])
    axarr[0, 1].hist(hists[1].ravel(),256,[0,256])
    axarr[0, 2].hist(hists[2].ravel(),256,[0,256])
    axarr[1, 0].hist(hists[3].ravel(),256,[0,256])
    axarr[1, 1].hist(hists[4].ravel(),256,[0,256])
    axarr[1, 2].hist(hists[5].ravel(),256,[0,256])

    plt.show()


def download(url, file_name):
    """Download file by its url."""
    with open(file_name, "wb") as file:
        response = get(url)
        file.write(response.content)