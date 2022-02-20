import os
import sys

import skimage.io as io

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
sys.path.append(PROJECT_ROOT)
from object_detection.segmentation.utils import choose_threshold


def test_choose_threshold_func(path):
    test_image = io.imread(path)
    mask = choose_threshold(test_image)
    return mask


path = os.path.join(PROJECT_ROOT, "test", "test_data", "img1.jpg")
result = test_choose_threshold_func(path)
print(result)
