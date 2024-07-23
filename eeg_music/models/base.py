from typing import Callable, Tuple

import lightning as L
import numpy as np
import torch
from braindecode.models.eegconformer import EEGConformer
from eeg_music.models.utils import transform_labels_classes
from eeg_music.parse_data import TARGET
from sklearn.metrics import accuracy_score, f1_score


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        # batch-aware op
        return x.view(x.size(0), *self.shape)


class LikertEmotionNet(L.LightningModule):
    def __init__(
        self,
        n_channels: int,
        input_window_samples: int,
        output_shape: Tuple[int] = (2, 5),  # 2 emotion dimensions, 5 point likert scale
        learning_rate: float = 1e-6,
        dropout: float = 0.5,
        use_three_class_problem: bool = False,
        eeg_model: torch.nn.Module = None,
    ):
        super(LikertEmotionNet, self).__init__()
        self.save_hyperparameters(ignore=["eeg_model"])
        self.lr = learning_rate
        self.output_shape = output_shape
        self.dropout = dropout
        self.input_window_samples = input_window_samples
        self.n_channels = n_channels
        self.use_three_class_problem = use_three_class_problem
        self.hidden_dim_clf = 128

        self.target = TARGET

        if self.use_three_class_problem:
            print(
                "Warning: overwriting second dim of output shape because of use_three_class_problem"
            )
            self.output_shape = (self.output_shape[0], 3)
            print(f"New output shape: {self.output_shape}")

        self.eeg_model = eeg_model

        classifier_input_dim = self.input_window_samples // 2
        if issubclass(type(self.eeg_model), EEGConformer):
            # number of hidden channels in the _FullyConnected module
            # are hardcoded for braindecode.EEGConformer
            classifier_input_dim = 32

        self.clf = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                classifier_input_dim,
                self.output_shape[0] * self.output_shape[1],
            ),
            Reshape(self.output_shape),
        )
        if not self.training:
            # CrossEntropyLoss does softmax internally
            self.clf.add_module(
                "softmax", torch.nn.Softmax(dim=2)
            )  # (batch, dim, classes)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.eeg_model(x)
        return self.clf(encoded)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def log_metric(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        mode: str,
        metric: Callable,
        metric_name: str,
        **kwargs,
    ) -> None:
        # this is  a multiclass-multi-label problem
        # dim -> multi-label
        # likert classes -> multi-class (3 or 5)
        # -> requires to calculate metrics by variable dim
        metric_by_dim = [
            metric(y_true[:, i], y_score[:, i], **kwargs)
            for i in range(self.output_shape[0])
        ]
        m = np.mean(metric_by_dim)

        self.log(f"{mode}_{metric_name}", m)
        for i, acc in enumerate(metric_by_dim):
            self.log(f"{mode}_{metric_name}_{self.target[i]}", acc)

    def step(self, batch, batch_idx, mode="train"):
        x, y = batch
        y_hat = self(x)

        if self.use_three_class_problem:
            y = transform_labels_classes(y)

        # "K-dimensional" case of CrossEntropyLoss -> 2D loss for arousal/pleasure
        y_hat = y_hat.permute(0, 2, 1)  # (batch, classes, dim) -> (batch, dim, classes)
        loss = self.criterion(y_hat, y)
        self.log(f"{mode}_loss", loss)

        # transform back to multi-label space from one-hot
        y_true = y.cpu().detach().numpy()
        y_hat = y_hat.permute(0, 2, 1)  # (batch, dim, classes) -> (batch, classes, dim)
        y_score = torch.argmax(y_hat, dim=2).cpu().detach().numpy()

        self.log_metric(y_true, y_score, mode, accuracy_score, "accuracy")
        self.log_metric(y_true, y_score, mode, f1_score, "f1", average="micro")

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        # TODO: confusion matrix
        return self.step(batch, batch_idx, "test")
