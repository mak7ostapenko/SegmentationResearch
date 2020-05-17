import cv2
from PIL import Image


class ResizeNP(object):
    """Resize np array."""
    def __init__(self, size, interpolation='bilinear'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        """
        :param image: np.array.
        :return:
        """
        image = cv2.resize(image, self.size)

        image = Image.fromarray(image)

        return image
