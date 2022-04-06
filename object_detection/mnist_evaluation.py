import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm


class MnistEvaluator:
    def __init__(self, inference: object, data: list):
        self._infer = inference
        self._data = data
        self._predictions = None

    def evaluate(self):
        pred = []
        labels = []
        for img, label in tqdm(self._data):
            img = np.array(img)
            pred.append(self._infer.get_prediction(img))
            labels.append(label)

        df_predicion = pd.DataFrame(labels, columns=["true_label"])
        df_predicion["pred_label"] = pred
        self._predictions = df_predicion
        return df_predicion

    def classification_report(self):
        if self._predictions is None:
            raise ValueError(
                "Predictions were not made"
                "\n Method .evaluate() must be called first"
            )
        classification_r = classification_report(
            self._predictions["true_label"], self._predictions["pred_label"]
        )
        return classification_r
