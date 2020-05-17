import cv2
import numpy as np
from PIL import Image


def get_gradients(image, scale=1, delta=0, x_grad_weight=0.5):
    """Compute gradients of image on X and Y directions then merge them.

    :param image: array of image could be Gray Scale and RGB.
    """
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    y_grad_weight = 1 - x_grad_weight
    grads = cv2.addWeighted(abs_grad_x, x_grad_weight, abs_grad_y, y_grad_weight, 0)

    return grads


def adjust_gamma(image, gamma=1.0):
    """Gamma correction.

    Could help to increase or decrease contrast of image.

    :param image: array of image in BGR color space.
    :param gamma:
    :return:

    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    result = cv2.LUT(image, table)

    return result


def rgb_hist_equalization(image):
    """Histogram equalization for increasing contrast of image.

    :param image: array of image in BGR color space.
    :return:
    """
    img_to_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_to_yuv[:, :, 0] = cv2.equalizeHist(img_to_yuv[:, :, 0])
    result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

    return result


def change_image_pallet(image, num_result_colors=10):
    """Change pallet of image for further segmentation.

    Could be used as segmentation.
    Changing image pallet has showed itself better in case of robot segmentation.

    :param image: array of image in BGR color space.
    :param num_result_colors: number of colors in resulting image pallet.
    :return:

    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image).convert('RGB')

    quant_pil_image = pil_image.quantize(colors=num_result_colors, method=0, kmeans=5, palette=None)
    result = np.array(quant_pil_image.convert('RGB'))
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    cv2.normalize(result, result, 0, 255, cv2.NORM_MINMAX)

    return result


