import albumentations as albu
import cv2
import numpy as np
from torchvision import transforms


class Invertor:
    def __call__(self, img):
        return transforms.functional.invert(img)


class Convertor:
    def __call__(self, img):
        return np.array(img)


class AlbuAugmentation:
    def __init__(self):
        sr_params = dict(
            border_mode=cv2.BORDER_CONSTANT,
            p=0.2,
        )

        self.description = [
            albu.OneOf(
                [
                    albu.ShiftScaleRotate(  # shift
                        shift_limit=0.2,
                        scale_limit=0,
                        rotate_limit=0,
                        interpolation=3,
                        border_mode=sr_params["border_mode"],
                        p=sr_params["p"],
                        value=255,  # white background for better representation
                    ),
                    albu.ShiftScaleRotate(  # scale
                        shift_limit=0,
                        scale_limit=0.25,
                        rotate_limit=0,
                        interpolation=3,
                        border_mode=sr_params["border_mode"],
                        p=sr_params["p"],
                        value=255,  # white background for better representation
                    ),
                    albu.ShiftScaleRotate(  # rotate
                        shift_limit=0,
                        scale_limit=0,
                        rotate_limit=20,
                        interpolation=3,
                        border_mode=sr_params["border_mode"],
                        p=sr_params["p"],
                        value=255,  # white background for better representation
                    ),
                    albu.ShiftScaleRotate(  # shift + scale + rotate
                        shift_limit=0.15,
                        scale_limit=0.2,
                        rotate_limit=20,
                        interpolation=3,
                        border_mode=sr_params["border_mode"],
                        p=sr_params["p"],
                        value=255,  # white background for better representation
                    ),
                    albu.RandomBrightnessContrast(  # Brightness + Contrast
                        brightness_limit=0.8,
                        contrast_limit=0.25,
                        brightness_by_max=True,
                        always_apply=False,
                        p=sr_params["p"],
                    ),
                    albu.GaussianBlur(blur_limit=0.1, p=sr_params["p"]),  # Blur
                ],
                p=0.2,
            )
        ]
        self.compose = albu.Compose(self.description, p=1)

    def __call__(self, img: np.ndarray) -> list:
        transformed = self.compose(image=img)
        return transformed["image"]
