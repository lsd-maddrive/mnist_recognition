import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(ROOT_DIR)


import logging

import numpy as np
import torch

from mnist_recognition.models import MlpModel


class Inference:
    def __init__(self, model, device) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model = model
        self._device = device
        self._prepare_model()

    def _prepare_model(self):
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(self._device)

        self._model.eval()
        self._model = self._model.to(self._device)

    @classmethod
    def from_file(cls, fpath: str, device=None, **kwargs):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_chk = torch.load(fpath, map_location=device)

        return cls.from_checkpoint(model_chk, device=device, **kwargs)

    @classmethod
    def from_checkpoint(cls, checkpoint_state: dict, **kwargs):
        model_state = checkpoint_state["model_state"]

        model = MlpModel()
        model.load_state_dict(model_state)

        obj_ = cls(model=model, **kwargs)
        return obj_

    def get_prediction(self, image: np.ndarray):
        # переводим массив в тензор
        image = image.reshape(-1, 28 * 28)
        image_t = torch.from_numpy(image)
        image_t = image_t.float()
        image_t = image_t.to(self._device)

        with torch.no_grad():
            output = self._model(image_t)
            _, prediction = torch.max(output, 1)

        return prediction.tolist()[0]
