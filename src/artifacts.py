import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib.pyplot as plt
from config import ARTIFACT_TRAIN_PATH, ARTIFACT_EVALUATION_PATH


class ComputeMatrics:
    def __init__(self, y_true, y_pred, y_prob):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob

    def init_comput_matrics(self) -> dict:

        self.metrics = {
            "Accuracy": round(float(accuracy_score(self.y_true, self.y_pred)), 4),
            "Precision": round(float(precision_score(self.y_true, self.y_pred)), 4),
            "Recall": round(float(recall_score(self.y_true, self.y_pred)), 4),
            "F1": round(float(f1_score(self.y_true, self.y_pred)), 4),
            "Roc_auc": round(float(roc_auc_score(self.y_true, self.y_prob)), 4),
        }

        return self.metrics


class Visualization:

    def __init__(
        self,
        y_true,
        y_prob,
        y_pred,
        run_name: str,
    ):
        self.y_true = y_true
        self.y_prob = y_prob
        self.y_pred = y_pred
        self.run_name = run_name

    def build_train_confusion_matrix(self):

        self.cm = confusion_matrix(self.y_true, self.y_pred)

        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.im = self.ax.imshow(self.cm, cmap="Blues")

        self.ax.set_xticks([0, 1])
        self.ax.set_yticks([0, 1])
        self.ax.set_xticklabels(["No Churn", "Churn"])
        self.ax.set_yticklabels(["No Churn", "Churn"])

        for i in range(2):
            for j in range(2):
                self.ax.text(j, i, self.cm[i, j], ha="center", va="center")

        self.ax.set_xlabel("Predicted")
        self.ax.set_ylabel("Actual")
        self.ax.set_title(f"Confusion Matrix – {self.run_name}")

        plt.colorbar(self.im)

        if not ARTIFACT_TRAIN_PATH.exists():
            ARTIFACT_TRAIN_PATH.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Artifacts directory not found: {ARTIFACT_TRAIN_PATH}")

        path = ARTIFACT_TRAIN_PATH / f"{self.run_name.replace(' ', '_')}_cm.png"
        self.fig.savefig(path, bbox_inches="tight")
        plt.close(self.fig)

        mlflow.log_artifact(str(path))

    def build_train_roc_curve(self, run_name: str):
        self.run_name = run_name

        self.fpr, self.tpr, _ = roc_curve(self.y_true, self.y_prob)

        self.fig, self.ax = plt.subplots()
        self.ax.plot(self.fpr, self.tpr)
        self.ax.plot([0, 1], [0, 1], "k--")

        self.ax.set_xlabel("FPR")
        self.ax.set_ylabel("TPR")
        self.ax.set_title(f"ROC Curve – {self.run_name}")

        if not ARTIFACT_TRAIN_PATH.exists():
            ARTIFACT_TRAIN_PATH.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Artifacts directory not found: {ARTIFACT_TRAIN_PATH}")

        self.path = ARTIFACT_TRAIN_PATH / f"{self.run_name.replace(' ', '_')}_roc.png"
        self.fig.savefig(self.path, bbox_inches="tight")
        plt.close(self.fig)

        mlflow.log_artifact(str(self.path))
