from typing import Tuple
import lightning as L
from braindecode.models.eegnet import EEGNetv4
import torch

from eeg_music.dataset import TrainTestSplitStrategy, train_loader, test_loader


def transform_labels_to_one_hot(
    labels: torch.Tensor,
    bins: int = 5,
) -> torch.Tensor:
    batch, dimensions = labels.shape
    labels = (labels - 1).to(torch.long)  # batch, dimensions

    transformed_labels = torch.zeros(batch, dimensions, bins).to(
        device=labels.device
    )  # batch, dimensions, likert
    indices = labels.unsqueeze(-1)
    return transformed_labels.scatter_(2, indices, 1)


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        # batch-aware op
        return x.view(x.size(0), *self.shape)


class LikertEmotionEEGNet(L.LightningModule):
    def __init__(
        self,
        n_channels: int,
        input_window_samples: int,
        output_shape: Tuple[int] = (2, 5),  # 2 emotion dimensions, 5 point likert scale
        learning_rate: float = 1e-6,
        pretrained_eegnet: bool = False,
        dropout: float = 0.5,
    ):
        super(LikertEmotionEEGNet, self).__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.output_shape = output_shape
        self.pretrained_eegnet = pretrained_eegnet
        self.dropout = dropout

        self.eeg_net = EEGNetv4(
            n_chans=n_channels,
            n_outputs=1,
            n_times=input_window_samples,
            final_conv_length="auto",
            drop_prob=self.dropout,
        )
        # drop classifier from eegnet
        del self.eeg_net.final_layer

        if self.pretrained_eegnet:
            # turn off grad calculations for eegnet and disable dropouts
            self.eeg_net.eval()
            for param in self.eeg_net.parameters():
                param.requires_grad = False

        self.clf = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_window_samples // 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_shape[0] * output_shape[1]),
            Reshape(output_shape),
            torch.nn.Softmax(dim=1),
        )

        self.criterion = torch.nn.CrossEntropyLoss()

    def _load_eegnet_weights(self, path: str):
        self.eeg_net.load_state_dict(torch.load(path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.eeg_net(x)
        return self.clf(encoded)

    def configure_optimizers(self):
        if self.pretrained_eegnet:
            # do not optimize eegnet if pretrained weights are used
            optimizer = torch.optim.Adam(self.clf.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def step(self, batch, batch_idx, mode="train"):
        x, y = batch
        y_hat = self(x)

        y = transform_labels_to_one_hot(y)
        # calculate cross-entropy for each dimension and sum
        loss = 0
        for i in range(self.output_shape[0]):
            loss += self.criterion(y_hat[i, :], y[i, :])
        self.log(f"{mode}_loss", loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")


if __name__ == "__main__":
    device = "mps"

    likert_emotions = LikertEmotionEEGNet(
        n_channels=8,
        input_window_samples=512,
        learning_rate=1e-6,
        pretrained_eegnet=False,
        dropout=0.5,
    ).to(device)
    train_set = train_loader(device, batch_size=256, split=TrainTestSplitStrategy.Track, normalize=True)
    test_set = test_loader(device, batch_size=8, split=TrainTestSplitStrategy.Track, normalize=True)

    trainer = L.Trainer(accelerator=device, max_epochs=10000, log_every_n_steps=10)
    trainer.fit(model=likert_emotions, train_dataloaders=train_set)

    trainer.test(model=likert_emotions, dataloaders=test_set)
