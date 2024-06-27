import os
import random
from typing import Generator, Tuple

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from eeg_music.dataset import TrainTestSplitStrategy
from eeg_music.utils import load_eeg

BASE_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "raw",
    "02_EYODF Pilot Data",
    "Pilot (Same Songs)",
)
OTREE_PATH = os.path.join(BASE_PATH, "all_participants_otree.csv")
EEG_PATH = os.path.join(BASE_PATH, "EEG")
EVENT_OFFSET_SECONDS = 3
MUSIC_DISCOVERY_DURATION_SECONDS = 20
TARGET = ["arousal", "pleasure"]
WINDOW_SIZE_SECONDS = 2
SAMPLE_RATE_HZ = 256

# denoise output
mne.set_log_level("ERROR")


def _load_by_chapter(otree: pd.Series, chapter: str) -> pd.DataFrame:
    otree = otree.reset_index()
    otree.columns = ["otree_variable", "value"]

    otree = otree.assign(
        mode=otree.apply(lambda row: row.otree_variable.split(".")[0], axis=1),
    )
    otree = otree.assign(
        dimension=otree.loc[otree["mode"] == chapter].apply(
            lambda row: row.otree_variable.split(".")[-1], axis=1
        ),
        track_id=(
            otree.loc[otree["mode"] == chapter].apply(
                lambda row: int(row.otree_variable.split(".")[1]), axis=1
            )
            if chapter in ["Intro", "Music_Discovery", "Outro"]
            else 1
        ),
    )

    tracks = otree.loc[otree["mode"] == chapter][
        ["value", "mode", "dimension", "track_id"]
    ].pivot(index="track_id", columns="dimension", values="value")

    if chapter == "Music_Discovery":
        # ASSUME: timestamps are in ms in UTC
        tracks = tracks.assign(
            time_start=pd.to_datetime(tracks.time_start, unit="ms"),
            time_add=pd.to_datetime(
                tracks.time_add.replace(0, np.nan), unit="ms", errors="coerce"
            ),
        )

    return tracks


def load_otree_data(path: str, participant: int) -> Tuple[pd.DataFrame]:
    df = pd.read_csv(path)
    otree_series = df.iloc[participant]
    return tuple(
        _load_by_chapter(otree_series, chapter)
        for chapter in ["participant", "Intro", "Music_Discovery", "Outro"]
    )


def preprocess_eeg(eeg: mne.io.RawArray) -> mne.io.RawArray:
    eeg.filter(l_freq=0.5, h_freq=60.0)
    eeg.notch_filter(freqs=50.0)
    return eeg


def extract_labels(
    music_discovery: pd.DataFrame, events: np.ndarray, event_id: dict
) -> np.ndarray:
    labels = np.zeros((events.shape[0], len(TARGET)))
    event_id = {v: k for k, v in event_id.items()}
    for i, event in enumerate(events):
        song = music_discovery.loc[music_discovery.current_song == event_id[event[2]]]
        labels[i] = song[TARGET].values[0]
    return labels


def load_eeg_to_mne(path: str) -> Generator[Tuple[int, mne.io.RawArray], None, None]:
    for file in os.listdir(path):
        participant, eeg, recording_start_time = load_eeg(os.path.join(path, file))
        ch_names = [c for c in eeg.columns if "EEG" in c]
        channel_names = ["F3", "Fz", "F4", "T3", "C3", "C4", "T4", "Pz"]

        info = mne.create_info(
            ch_names=channel_names,
            sfreq=SAMPLE_RATE_HZ,
            ch_types="eeg",
        )
        raw = mne.io.RawArray(eeg[ch_names].values.T, info)
        raw.set_montage("standard_1020")
        # annotate events
        p = participant - 1  # 0-indexed
        participant_meta, intro, music_discovery, outro = load_otree_data(OTREE_PATH, p)

        raw.set_annotations(
            mne.Annotations(
                onset=(music_discovery.time_start - recording_start_time)
                .dt.total_seconds()
                .values,
                duration=[0] * music_discovery.shape[0],  # ignored in events anyway
                description=music_discovery.current_song.values,
            )
        )
        raw = preprocess_eeg(raw)
        # create epochs from annotations
        epochs = mne.Epochs(
            raw=raw,
            tmin=EVENT_OFFSET_SECONDS,
            tmax=EVENT_OFFSET_SECONDS + MUSIC_DISCOVERY_DURATION_SECONDS,
            baseline=None,
            preload=True,
        )
        labels = extract_labels(music_discovery, epochs.events, epochs.event_id)

        # create sub-epochs of 2s
        sub_epochs_list = []
        SUBEPOCH_PER_EPOCH_COUNT = np.int64(
            MUSIC_DISCOVERY_DURATION_SECONDS / WINDOW_SIZE_SECONDS
        )
        sub_epoch_labels = np.zeros(
            (len(epochs) * SUBEPOCH_PER_EPOCH_COUNT, len(TARGET))
        )

        for i, epoch in enumerate(epochs):
            data = epoch.reshape(len(epochs.ch_names), -1)
            info = epochs.info
            temp_raw = mne.io.RawArray(data, info)
            sub_epochs = mne.make_fixed_length_epochs(
                temp_raw, duration=WINDOW_SIZE_SECONDS, overlap=0
            )
            sub_epochs_list.append(sub_epochs)
            sub_epoch_labels[
                i * SUBEPOCH_PER_EPOCH_COUNT : (i + 1) * SUBEPOCH_PER_EPOCH_COUNT
            ] = labels[i]
        all_sub_epochs = mne.concatenate_epochs(sub_epochs_list)

        print(all_sub_epochs.get_data().shape, sub_epoch_labels.shape)

        yield eeg.iloc[0].participant, all_sub_epochs, sub_epoch_labels


def _ensure_directories(split: TrainTestSplitStrategy) -> Tuple[str, str]:
    base = os.path.join(os.path.dirname(__file__), "data", "processed")
    train_path = os.path.join(base, split.to_directory(), "train")
    test_path = os.path.join(base, split.to_directory(), "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    return train_path, test_path


def write_base_dataset(
    participant: np.int64,
    eeg_epochs: mne.Epochs,
    labels: np.ndarray,
) -> None:
    train_path, test_path = _ensure_directories(TrainTestSplitStrategy.Participant)

    # train test split based on participant
    dest = ""
    # leave ~ 2 participants for test
    if random.random() < 0.25:
        dest = test_path
    else:
        dest = train_path
    np.save(os.path.join(dest, f"P{participant:.0f}_eeg.npy"), eeg_epochs.get_data())
    np.save(os.path.join(dest, f"P{participant:.0f}_labels.npy"), labels)


def write_track_holdout_dataset(
    participant: np.int64, eeg_epochs: mne.Epochs, labels: np.ndarray
) -> None:
    train_path, test_path = _ensure_directories(TrainTestSplitStrategy.Track)

    pass


if __name__ == "__main__":
    eeg_files = os.listdir(EEG_PATH)
    for p, eeg_epochs, labels in tqdm(load_eeg_to_mne(EEG_PATH), total=len(eeg_files)):
        write_track_holdout_dataset(p, eeg_epochs, labels)
