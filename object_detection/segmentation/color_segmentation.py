from typing import List

import cv2
import numpy as np


def color_segmentation(img: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:

    """
    Function use masks to create a segemnted image.

    Parameters:
    img (np.ndarray): Original image
    masks (List[np.ndarray]) : List of masks which is to be applied to image

    Returns:
    final_result (np.ndarray): image with color segmentetion
    """

    full_mask = 0  # Задаем переменную для будующей маски
    for i in range(
        len(masks)
    ):  # Проходимся по списку массок и объединяем их в общую маску
        full_mask += masks[i]
    final_result = cv2.bitwise_and(
        img, img, mask=full_mask
    )  # Создаем изображение с наложенной маской
    return final_result
