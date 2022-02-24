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
    mask, thresholds = choose_threshold(test_image)
    return mask, thresholds


if __name__ == "__main__":
    path = os.path.join(PROJECT_ROOT, "test", "test_data", "img1.jpg")
    result, thres = test_choose_threshold_func(path)
    print(f"Result mask:\n{result},\n Limits Low & High:{thres}")
