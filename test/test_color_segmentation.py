import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
sys.path.append(PROJECT_ROOT)

import numpy as np

from mnist_recognition.segmentation.color_segmentation import color_segmentation


def test_color_segmentation_sample_1(
    test_data_1, test_masks_1, expected_result_1
):
    assert np.all(
        color_segmentation(test_data_1, test_masks_1) == expected_result_1
    )


def test_color_segmentation_sample_2(
    test_data_2, test_masks_2, expected_result_2
):
    assert np.all(
        color_segmentation(test_data_2, test_masks_2) == expected_result_2
    )


def test_color_segmentation_sample_with_real_img(
    test_data_image, test_mask_3, expected_data_3
):
    assert np.all(
        color_segmentation(test_data_image, [test_mask_3]) == expected_data_3
    )
