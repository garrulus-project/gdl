from torchmetrics.classification import (
    Accuracy,
    FBetaScore,
    JaccardIndex,
    Precision,
    Recall,
)
from torchmetrics import MetricCollection
from torchgeo.trainers import SemanticSegmentationTask


class GarrulusSemanticSegmentationTask(SemanticSegmentationTask):
    def __init__(self, *args, labels: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["num_classes"]

        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": Accuracy(
                    task="multiclass", num_classes=num_classes, average="micro"
                ),
                "OverallPrecision": Precision(
                    task="multiclass", num_classes=num_classes, average="micro"
                ),
                "OverallRecall": Recall(
                    task="multiclass", num_classes=num_classes, average="micro"
                ),
                "OverallF1Score": FBetaScore(
                    task="multiclass",
                    num_classes=num_classes,
                    beta=1.0,
                    average="micro",
                ),
                "MeanIoU": JaccardIndex(
                    num_classes=num_classes, task="multiclass", average="macro"
                ),
                # "ConfusionMatrix": ConfusionMatrix(
                #    task="multiclass", num_classes=num_classes
                # )
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
