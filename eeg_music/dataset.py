import os
from typing import Literal, Tuple
from enum import Enum, auto
import numpy as np
import torch
from torch.utils.data import Dataset

BASE_PATH = os.path.join(os.path.dirname(__file__), "data", "processed")


class TrainTestSplitStrategy(Enum):
    Participant = auto()
    Track = auto()

    def to_directory(self) -> str:
        split_dir = {
            TrainTestSplitStrategy.Participant: "participant_holdout",
            TrainTestSplitStrategy.Track: "track_holdout",
        }.get(self)

        if split_dir is None:
            raise ValueError("Invalid train_test_split_strategy")
        return split_dir

    def __str__(self):
        return self.name.lower()


class EEGMusicDataset(Dataset):
    def __init__(
        self,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        mode: Literal["train", "test"] = "train",
        train_test_split_strategy: TrainTestSplitStrategy = TrainTestSplitStrategy.Participant,
    ):
        self.device = device
        self.dtype = torch.float32
        self.eeg_data, self.labels = self._load_windows(mode, train_test_split_strategy)

    def _load_windows(
        self,
        mode: Literal["train", "test"],
        train_test_split_strategy: TrainTestSplitStrategy,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eeg = torch.tensor([], dtype=self.dtype).to(self.device)
        labels = torch.tensor([], dtype=self.dtype).to(self.device)

        path = os.path.join(BASE_PATH, train_test_split_strategy.to_directory(), mode)
        for file in sorted(os.listdir(path)):
            if file.endswith("_eeg.npy"):
                participant_eeg = (
                    torch.from_numpy(np.load(os.path.join(path, file)))
                    .to(self.dtype)
                    .to(self.device)
                )
                eeg = torch.cat((eeg, participant_eeg))
            elif file.endswith("_labels.npy"):
                participant_labels = (
                    torch.from_numpy(np.load(os.path.join(path, file)))
                    .to(self.dtype)
                    .to(self.device)
                )
                labels = torch.cat((labels, participant_labels))
        return eeg, labels

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx]


def train_loader(
    device,
    batch_size,
    split: TrainTestSplitStrategy = TrainTestSplitStrategy.Participant,
):
    return torch.utils.data.DataLoader(
        EEGMusicDataset(device=device, mode="train", train_test_split_strategy=split),
        batch_size=batch_size,
        shuffle=True,
    )


def test_loader(
    device,
    batch_size,
    split: TrainTestSplitStrategy = TrainTestSplitStrategy.Participant,
):
    return torch.utils.data.DataLoader(
        EEGMusicDataset(device=device, mode="test", train_test_split_strategy=split),
        batch_size=batch_size,
        shuffle=False,
    )


if __name__ == "__main__":
    for eeg, labels in train_loader("cpu", 16):
        print(eeg.shape, labels.shape)
        break
