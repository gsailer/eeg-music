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
    # TODO: check if strat over track introduces strong bias to certain tracks
    StratifiedTrack = auto()  # not implemented

    def to_directory(self) -> str:
        split_dir = {
            TrainTestSplitStrategy.Participant: "participant_holdout",
            TrainTestSplitStrategy.Track: "track_holdout",
            TrainTestSplitStrategy.StratifiedTrack: "stratified_track_holdout",
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
        mode: Literal["train", "test", "val"] = "train",
        train_test_split_strategy: TrainTestSplitStrategy = TrainTestSplitStrategy.Participant,
        normalize: bool = False,
        load_participant_ids: bool = False,
    ):
        self.device = device
        self.dtype = torch.float32
        self.load_participant_ids = load_participant_ids

        if self.load_participant_ids:
            self.eeg_data, self.labels, self.participant_ids = self._load_windows(
                mode, train_test_split_strategy, normalize
            )
        else:
            self.eeg_data, self.labels = self._load_windows(
                mode, train_test_split_strategy, normalize
            )

    def _load_windows(
        self,
        mode: Literal["train", "test", "val"],
        train_test_split_strategy: TrainTestSplitStrategy,
        normalize: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eeg = torch.tensor([], dtype=self.dtype).to(self.device)
        labels = torch.tensor([], dtype=self.dtype).to(self.device)
        if self.load_participant_ids:
            participant_ids = torch.tensor([], dtype=torch.long).to(self.device)

        norm_path_part = "normalized" if normalize else "raw"
        path = os.path.join(
            BASE_PATH, train_test_split_strategy.to_directory(), norm_path_part, mode
        )
        for file in sorted(os.listdir(path)):
            dim = None
            if file.endswith("_eeg.npy"):
                participant_eeg = (
                    torch.from_numpy(np.load(os.path.join(path, file)))
                    .to(self.dtype)
                    .to(self.device)
                )
                eeg = torch.cat((eeg, participant_eeg))
                dim = participant_eeg.shape[0]
            elif file.endswith("_labels.npy"):
                participant_labels = (
                    torch.from_numpy(np.load(os.path.join(path, file)))
                    .to(self.dtype)
                    .to(self.device)
                )
                labels = torch.cat((labels, participant_labels))
                dim = participant_labels.shape[0]
            if self.load_participant_ids:
                participant_ids = torch.cat(
                    (
                        participant_ids,
                        # assume: P1_eeg.npy
                        torch.full(
                            (dim,),
                            int(file.split("_")[0][1:]),
                            dtype=torch.long,
                            device=self.device,
                        ),
                    )
                )
        if self.load_participant_ids:
            return eeg, labels, participant_ids
        return eeg, labels

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        if self.load_participant_ids:
            return self.eeg_data[idx], self.labels[idx], self.participant_ids[idx]
        return self.eeg_data[idx], self.labels[idx]


def loader(
    device,
    batch_size,
    split: TrainTestSplitStrategy = TrainTestSplitStrategy.Participant,
    normalize: bool = False,
    load_participant_ids: bool = False,
    mode: Literal["train", "test", "val"] = "train",
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        EEGMusicDataset(
            device=device,
            mode=mode,
            train_test_split_strategy=split,
            normalize=normalize,
            load_participant_ids=load_participant_ids,
        ),
        batch_size=batch_size,
        shuffle=False if mode == "test" else True,
    )


def train_loader(*args, **kwargs) -> torch.utils.data.DataLoader:
    return loader(*args, **kwargs, mode="train")


def test_loader(*args, **kwargs) -> torch.utils.data.DataLoader:
    return loader(*args, **kwargs, mode="test")


def validation_loader(*args, **kwargs) -> torch.utils.data.DataLoader:
    return loader(*args, **kwargs, mode="val")


if __name__ == "__main__":
    for eeg, labels in train_loader("cpu", 16):
        print(eeg.shape, labels.shape)
        break
