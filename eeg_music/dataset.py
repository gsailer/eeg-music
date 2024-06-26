import os
from typing import Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

BASE_PATH = os.path.join(os.path.dirname(__file__), "data", "processed")


class EEGMusicDataset(Dataset):
    def __init__(
        self,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        mode: Literal["train", "test"] = "train",
    ):
        self.device = device
        self.dtype = torch.float32
        self.eeg_data, self.labels = self._load_windows(mode)

    def _load_windows(
        self,
        mode: Literal["train", "test"],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eeg = torch.tensor([], dtype=self.dtype).to(self.device)
        labels = torch.tensor([], dtype=self.dtype).to(self.device)

        for file in sorted(os.listdir(f"{BASE_PATH}/{mode}")):
            if file.endswith("_eeg.npy"):
                participant_eeg = (
                    torch.from_numpy(np.load(f"{BASE_PATH}/{mode}/{file}"))
                    .to(self.dtype)
                    .to(self.device)
                )
                eeg = torch.cat((eeg, participant_eeg))
            elif file.endswith("_labels.npy"):
                participant_labels = (
                    torch.from_numpy(np.load(f"{BASE_PATH}/{mode}/{file}"))
                    .to(self.dtype)
                    .to(self.device)
                )
                labels = torch.cat((labels, participant_labels))
        return eeg, labels

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx]


train_loader = lambda device, batch_size: torch.utils.data.DataLoader(
    EEGMusicDataset(device=device, mode="train"), batch_size=batch_size, shuffle=True
)
test_loader = lambda device, batch_size: torch.utils.data.DataLoader(
    EEGMusicDataset(device=device, mode="test"), batch_size=batch_size, shuffle=False
)

if __name__ == "__main__":
    for eeg, labels in train_loader("cpu", 16):
        print(eeg.shape, labels.shape)
        break
