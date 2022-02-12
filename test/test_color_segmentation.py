import os
import sys



PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

from object_detection.segmentation.color_segmentation import color_segmentation
import numpy as np


def test_color_segmentation_sample_1(test_data_1,test_masks_1,expected_result_1):
    assert np.all(color_segmentation(test_data_1,test_masks_1) == expected_result_1)


def test_color_segmentation_sample_2(test_data_2,test_masks_2,expected_result_2):
    assert np.all(color_segmentation(test_data_2,test_masks_2) == expected_result_2)


