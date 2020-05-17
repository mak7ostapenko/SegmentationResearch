import os

import cv2
import numpy as np
import pandas as pd
from joblib import dump, load
from PIL import Image, ExifTags
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from src.classical_cv.segmentation import segment_by_color_in_HSV


def get_max_contour(mask):
    """
    Get contour with max length.

    :param mask: binary image.
    :return:
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = list()
    for i in range(0, len(contours)):
        areas.append(cv2.contourArea(contours[i]))
    
    if len(areas) == 0:
        return None

    max_contour_area = max(areas)
    max_contour_ind = areas.index(max_contour_area)

    contour = contours[max_contour_ind]
    return contour


def filter_contour_part(image, contour):
    """
    Filter contours.

    :param image: a input RGB image.
    :param contour: cv2 contour.
    :return:
    """
    contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, contour, -1, 255, 3)

    line_coords = np.array(np.where(contour_mask)).T

    y_min = np.min(line_coords[:, 0])
    x_min = np.min(line_coords[:, 1])
    x_max = np.max(line_coords[:, 1])

    contour_mask[y_min: y_min + 270, x_min: x_max] = 0
    return contour_mask


def cluster_contour_parts(contour_mask):
    """
    Cluster parts of contour in order to get two separate edges of blade.

    :param contour_mask: binary image.
    :return:
    """
    contour_points = np.array(np.where(contour_mask)).T

    # cluster coords
    X = contour_points[:, 1].reshape(-1, 1)
    k_means = KMeans(n_clusters=2, random_state=0)
    y = k_means.fit_predict(X)

    edge_inds1 = contour_points[y == 0]
    edge_inds2 = contour_points[y == 1]

    edge_one = np.zeros_like(contour_mask)
    edge_two = np.zeros_like(contour_mask)

    edge_one[tuple(edge_inds1.T)] = 1
    edge_two[tuple(edge_inds2.T)] = 1

    # find right and left lines
    if k_means.cluster_centers_[0] > k_means.cluster_centers_[1]:
        right_edge, right_inds = k_means.cluster_centers_[0], edge_inds1
        left_edge, left_inds = k_means.cluster_centers_[1], edge_inds2

    else:
        right_edge, right_inds = k_means.cluster_centers_[1], edge_inds2
        left_edge, left_inds = k_means.cluster_centers_[0], edge_inds1

    return right_edge, right_inds, left_edge, left_inds


def fit_lines(right_inds, left_inds):
    """
    Fit lines utilizing contour coordinates.

    :param right_inds: array of right point cloud.
    :param left_inds: array of left point cloud.
    :return:
    """
    x_right = right_inds[:, 1]
    y_right = right_inds[:, 0]

    x_left = left_inds[:, 1]
    y_left = left_inds[:, 0]

    right_fit = np.polyfit(x_right, y_right, 2)
    right_fit_fn = np.poly1d(right_fit)

    left_fit = np.polyfit(x_left, y_left, 2)
    left_fit_fn = np.poly1d(left_fit)

    right_curve_x = np.arange(np.min(x_right), np.max(x_right))
    right_curve_y = right_fit_fn(right_curve_x).astype(np.int16)

    left_curve_x = np.arange(np.min(x_left), np.max(x_left))
    left_curve_y = left_fit_fn(left_curve_x).astype(np.int16)

    right_line = np.vstack([right_curve_x, right_curve_y]).T
    left_line = np.vstack([left_curve_x, left_curve_y]).T

    return right_line, left_line, right_curve_x, right_curve_y, left_curve_x, left_curve_y


def get_image_meta_data(image_path):
    """
    Parse metadata from .jpg image.

    :param image_path: str, path to image.
    :return:
    """
    img = Image.open(image_path)
    img_exif = img._getexif()

    if img_exif:
        image_meta_data = {ExifTags.TAGS[k]: v for k, v in img_exif.items() if k in ExifTags.TAGS}
    else:
        image_meta_data = None

    return image_meta_data


def filter_meta_data(metadata):
    metadata = str(metadata['UserComment'])

    if 'Edge Dist' in metadata:
        dist_start_ind = metadata.find('Edge Dist')
        dist = float(metadata[dist_start_ind + 13: dist_start_ind + 20])
    else:
        dist = -1

    return dist


def train_scaler(dist_frame_path):
    dist_frame = pd.read_csv(dist_frame_path)
    comp_dists = dist_frame.comp_dist.values

    scaler = MinMaxScaler(feature_range=(0, 1))

    return scaler


def train_regressor(dist_frame_path, out_path='model_checkpoints/regressor.joblib'):
    """
    Train Linear Regression model for mapping pixel distance into miters.
    :param dist_frame_path:
    :return:
    """
    dist_frame = pd.read_csv(dist_frame_path)

    # filter frame
    dist_frame = dist_frame[dist_frame.true_dist >= 0]

    dist_frame = dist_frame.reset_index(drop=True)
    train = dist_frame[['true_dist', 'hausdorff_dists', 'dist_line_len']]

    # delete outliers
    # train = train[(np.abs(stats.zscore(train)) < 3).all(axis=1)]

    X, y = np.array(train[['hausdorff_dists', 'dist_line_len']]), np.array(train['true_dist'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr.fit(X_train, y_train)
    print('###NOTE###____ Regressor R2-score = ', gpr.score(X_test, y_test))

    dump(gpr, out_path)


    return gpr

