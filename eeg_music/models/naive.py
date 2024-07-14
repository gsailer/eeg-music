from datetime import datetime
import os
from typing import List, Literal, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
import torch
from torch.nn import Module, functional as F
from tqdm import tqdm
from eeg_music.dataset import TrainTestSplitStrategy, train_loader, test_loader


class EEGLikertConformer(Module):
    def __init__(
        self,
        batch_size: int,
        n_eeg_channels: int = 8,
        samples_per_window: int = 512,  # 2 seconds at 256 Hz
        output_size: int = 5,  # Number of classes for each Likert scale (0-4)
        nhead: int = 8,
        num_layers: int = 2,  # Number of transformer encoder layers
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
    ):
        super(EEGLikertConformer, self).__init__()
        self.n_eeg_channels = n_eeg_channels
        self.batch_size = batch_size
        self.samples_per_window = samples_per_window
        self.learning_rate = learning_rate
        self.device = device

        self.batchnorm = torch.nn.BatchNorm1d(n_eeg_channels)

        self.conv1 = torch.nn.Conv1d(
            in_channels=n_eeg_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=32, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        input_size = 32 * (samples_per_window // 2)
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, output_size * 2)

        self.dropout = torch.nn.Dropout(dropout)

        # Loss and optimizer
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x = self.batchnorm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.permute(2, 0, 1)

        encoded = self.transformer_encoder(x)
        encoded = encoded.permute(
            1, 0, 2
        )  # Reshape back to (batch_size, seq_len, feature_dim)
        encoded = encoded.contiguous().view(encoded.size(0), -1)

        encoded = F.relu(self.fc1(encoded))
        encoded = self.dropout(encoded)
        output = self.fc2(encoded)
        output = output.view(-1, 2, 5)  # Reshape to (batch_size, 2, 5)
        return F.softmax(output, dim=2)

    def _transform_labels_to_one_hot(self, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        # Normalize the labels to 0-based index
        labels = labels - 1
        transformed_labels = []
        for i in range(labels.shape[1]):
            raw_labels = labels
            one_hot_labels = torch.zeros(raw_labels.shape[0], 5).to(raw_labels.device)
            raw_labels_i = raw_labels[:, i].long()
            one_hot_labels[torch.arange(raw_labels_i.shape[0]), raw_labels_i] = 1
            transformed_labels.append(one_hot_labels)
        return tuple(transformed_labels)

    def train_step(self, eeg, labels):
        self.train()
        output = self.forward(eeg)  # batch, 2, 5

        output_arousal = output[:, 0, :]
        output_pleasure = output[:, 1, :]

        labels_arousal, labels_pleasure = self._transform_labels_to_one_hot(
            labels
        )  # (batch, 5; batch, 5)

        loss_arousal = self.loss(output_arousal, labels_arousal)
        loss_pleasure = self.loss(output_pleasure, labels_pleasure)
        loss = loss_pleasure + loss_arousal

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def predict(self, eeg):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output = self.forward(eeg)

            predicted_arousal = torch.argmax(output[:, 0, :], dim=1)
            predicted_pleasure = torch.argmax(output[:, 1, :], dim=1)
            return (
                torch.stack([predicted_arousal, predicted_pleasure], dim=1) + 1
            )  # Convert back to 1-based index


def calculate_metrics(labels, predictions):
    print(labels[:10])
    print("---")
    print(predictions[:10])
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    f1 = f1_score(labels, predictions, average="weighted")
    confusion = confusion_matrix(labels, predictions)

    return accuracy, precision, recall, f1, confusion


def trainer(
    split=TrainTestSplitStrategy.Participant,
    device="mps",
    batch=64,
    epochs=100,
    lr=1e-6,
    dropout=0.1,
):
    model = EEGLikertConformer(batch_size=batch, learning_rate=lr, dropout=dropout).to(
        device
    )
    print(model)
    data = train_loader(
        device,
        batch,
        split=split,
    )

    for _ in tqdm(range(epochs)):
        train_loss = []
        for eeg, labels in data:
            l = model.train_step(eeg, labels)
            train_loss.append(l.item())
        print(sum(train_loss) / len(train_loss))

    os.makedirs("checkpoints", exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(
            "checkpoints",
            f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_{split}_{batch}_{lr}_naive_model.pth",
        ),
    )
    evaluate(model, device=device, batch=batch, split=split)


def evaluation_plots(loss: List, metrics: dict, model_checkpoint: str, split: str):
    # scoped plotting dependencies
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    time = datetime.now().strftime("%Y-%m-%d_%H%M")
    os.makedirs("plots", exist_ok=True)

    with PdfPages(f"plots/{time}{model_checkpoint}-{split}.pdf") as pdf:
        # plot confusion matrix
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        for i, scale in enumerate(["Pleasure", "Arousal"]):
            disp = ConfusionMatrixDisplay(metrics[scale]["Confusion"])
            disp.plot(ax=ax[i])
            ax[i].set_title(f"{scale} Confusion Matrix")
        pdf.savefig(fig)
        plt.savefig(f"plots/{time}{model_checkpoint}-{split}-confusion.png")
        plt.close(fig)

        # plot loss histogram
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        sns.histplot(loss, ax=ax)
        ax.set_title("Loss Evaluation Distribution")
        pdf.savefig(fig)
        plt.close(fig)


def evaluate(
    model=None,
    device="mps",
    batch=64,
    checkpoint="2024-06-26_1705_256_1e-06_10000_gcp_naive_model.pth",
    split=TrainTestSplitStrategy.Participant,
):
    """
    Params
    ------
    model: EEGLikertConformer
        The model to evaluate. If None, a new model is created and the checkpoint is loaded.
        This requires checkpoint to be set.
    device: str
        The device to run the evaluation on.
    batch: int
        The batch size for evaluation.
    checkpoint: str
        The checkpoint to load. This is ignored if model is not None.
    split: TrainTestSplitStrategy
        The split strategy to use for evaluation.
    """
    if model is None:
        model = EEGLikertConformer(batch_size=batch, device=device).to(device)
        model.load_checkpoint(os.path.join("checkpoints", checkpoint))

    test_data = test_loader(device, batch, split=split)

    eval_loss = []
    l1_loss = []
    all_labels = []
    all_predictions = []

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for eeg, labels in test_data:
            logits = model.forward(eeg)

            logits_arousal = logits[:, 0, :]
            logits_pleasure = logits[:, 1, :]

            labels_arousal, labels_pleasure = model._transform_labels_to_one_hot(labels)

            # Compute loss
            loss_arousal = model.loss(logits_arousal, labels_arousal)
            loss_pleasure = model.loss(logits_pleasure, labels_pleasure)
            l1_loss.append(
                F.l1_loss(logits, torch.stack([labels_arousal, labels_pleasure], 1))
            )
            loss = loss_pleasure + loss_arousal
            eval_loss.append(loss.item())

            output = model.predict(eeg)
            predictions = output.cpu().numpy()

            labels_normalized = (
                torch.stack(
                    [torch.argmax(labels_arousal, 1), torch.argmax(labels_pleasure, 1)],
                    1,
                )
                + 1
            )

            all_labels.extend(labels_normalized.cpu().numpy())
            all_predictions.extend(predictions)

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        metrics = {"Pleasure": {}, "Arousal": {}}

        for i, scale in enumerate(["Pleasure", "Arousal"]):
            accuracy, precision, recall, f1, confusion = calculate_metrics(
                all_labels[:, i], all_predictions[:, i]
            )
            metrics[scale]["Accuracy"] = accuracy
            metrics[scale]["Precision"] = precision
            metrics[scale]["Recall"] = recall
            metrics[scale]["F1"] = f1
            metrics[scale]["Confusion"] = confusion

        for scale, values in metrics.items():
            print(
                f'{scale} - Accuracy: {values["Accuracy"]}, Precision: {values["Precision"]}, Recall: {values["Recall"]}, F1: {values["F1"]}'
            )
            print(values["Confusion"])

        evaluation_plots(eval_loss, metrics, checkpoint, split)

    print(f"Cross-Entropy loss: {sum(eval_loss) / len(eval_loss)}")
    print(f"L1 loss: {sum(l1_loss) / len(l1_loss)}")


if __name__ == "__main__":
    # trainer(
    #     split=TrainTestSplitStrategy.Track,
    #     batch=256,
    #     epochs=10000,
    #     lr=1e-6,
    #     device="cuda",
    # )
    # evaluate(
    #     device="mps",
    #     batch=16,
    #     checkpoint="2024-06-26_1705_256_1e-06_10000_gcp_naive_model.pth",
    #     split=TrainTestSplitStrategy.Participant,
    # )
    evaluate(
        device="mps",
        batch=16,
        checkpoint="2024-06-27_2202_track_256_1e-06_10000_gcp_naive_model.pth",
        split=TrainTestSplitStrategy.Track,
    )
    evaluate(
        device="mps",
        batch=16,
        checkpoint="2024-06-26_1705_participant_256_1e-06_10000_gcp_naive_model.pth",
        split=TrainTestSplitStrategy.Participant,
    )
    # trainer(
    #     device="mps",
    #     batch=256,
    #     epochs=15,
    #     lr=1e-6,
    #     split=TrainTestSplitStrategy.Participant,
    #     dropout=0.25,
    # )
