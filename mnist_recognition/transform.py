import numpy as np
from torchvision import transforms


class Invertor:
    def __call__(self, img):
        return transforms.functional.invert(img)


class Convertor:
    def __call__(self, img):
        return np.array(img)
