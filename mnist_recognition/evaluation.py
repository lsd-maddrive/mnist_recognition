import logging

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

from mnist_recognition import inference


class MnistEvaluator:
    def __init__(self, inference: inference.Inference, data: list):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._infer = inference
        self._data = data
        self._predictions = pd.DataFrame(
            [], columns=["true_label", "pred_label"]
        )

    def evaluate(self):
        pred = []
        labels = []
        for img, label in tqdm(self._data):
            img = np.array(img)
            pred.append(self._infer.get_prediction(img))
            labels.append(label)

        self._predictions[self._predictions.columns] = np.column_stack(
            (labels, pred)
        )
        return self._predictions

    def classification_report(self):
        if self._predictions.empty:
            raise ValueError(
                "Predictions were not made"
                "\n Method .evaluate() must be called first"
            )
        classification_r = classification_report(
            self._predictions["true_label"],
            self._predictions["pred_label"],
            output_dict=True,
        )

        classification_r = pd.DataFrame(classification_r).round(2).T
        classification_r.loc[
            ["accuracy"], ["precision", "recall", "support"]
        ] = " "
        return classification_r
