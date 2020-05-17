import cv2
import numpy as np


def segment_by_color_in_HSV(image, lower_thresh=[110, 50, 50], upper_thresh=[130, 255, 255]):
    """Simple color segmentation by range on color in HSV color space.

    NOTE that for robot segmentation the best params are next
    lower_thresh=[90, 80, 80], upper_thresh=[130, 180, 150].

    :param image: array of image in BGR color space.
    :return:
    """
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_thresh = np.array(lower_thresh)
    upper_thresh = np.array(upper_thresh)

    mask = cv2.inRange(hsv_img, lower_thresh, upper_thresh)
    segmented_obj = cv2.bitwise_and(image, image, mask=mask)

    return mask, segmented_obj


def histrogram_backprojection(image, mask=None, roi_hist=None):
    """Find regions on image with same histogram as mask has.

    :param image: array of image in BGR color space.
    :param image: array of mask in BGR color space. Includes ROI for detection.
    :return:
    """
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if roi_hist is None:
        hsv_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([hsv_mask], [0, 1], None, [1, 256], [0, 1, 0, 256])

    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsv_img], [0, 1], roi_hist, [0, 1, 0, 256], 1)

    # Now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(dst, -1, disc, dst)

    # threshold and binary AND
    ret, thresh = cv2.threshold(dst, 50, 255, 0)
    thresh = cv2.merge((thresh, thresh, thresh))
    res = cv2.bitwise_and(image, thresh)

    return image, thresh, res

