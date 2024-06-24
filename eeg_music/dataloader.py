import os
from typing import Generator, Tuple
import numpy as np
import pandas as pd
import mne
from eeg_music.utils import load_eeg


BASE_PATH = "./data/raw/02_EYODF Pilot Data/Pilot (Same Songs)"
OTREE_PATH = os.path.join(BASE_PATH, "all_participants_otree.csv")
EEG_PATH = os.path.join(BASE_PATH, "EEG")


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


def load_eeg_to_mne(path: str) -> Generator[Tuple[int, mne.io.RawArray], None, None]:
    for file in os.listdir(path):
        participant, eeg, recording_start_time = load_eeg(os.path.join(path, file))
        ch_names = [c for c in eeg.columns if "EEG" in c]
        channel_names = ["F3", "Fz", "F4", "T3", "C3", "C4", "T4", "Pz"]

        info = mne.create_info(
            ch_names=channel_names,
            sfreq=256,
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
                duration=[0] * music_discovery.shape[0],
                description=music_discovery.current_song.values,
            )
        )
        yield eeg.iloc[0].participant, preprocess_eeg(raw)


if __name__ == "__main__":
    p, eeg_data = next(load_eeg_to_mne(EEG_PATH))
    print(eeg_data.info)
