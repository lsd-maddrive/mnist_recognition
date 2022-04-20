from torchvision import transforms


class Invertor:
    def __call__(self, img):
        return transforms.functional.invert(img)
