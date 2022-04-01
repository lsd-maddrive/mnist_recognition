import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from tqdm import tqdm


class MnistEvaluator:
    def __init__(self, inference: object, data_df: list):
        self._infer = inference
        self._data = data_df

    def evaluate(self):
        df_predicion = []
        labels = []
        for img, label in tqdm(self._data):
            img = np.array(img)
            df_predicion.append(self._infer.get_prediction(img))
            labels.append(label)

        df_predicion = pd.DataFrame(labels, columns=["true_label"])
        df_predicion["pred_label"] = df_predicion
        return df_predicion

    def classification_report(self, df_prediction):

        classification_report = []
        # matrix = create_CM(df_prediction["True labels"],df_prediction["Predictions"])
        classification_report.append(
            recall_score(
                df_prediction["true_label"], df_prediction["pred_label"]
            )
        )
        # classification_report = compute_classification_metrics(matrix)
        return classification_report
