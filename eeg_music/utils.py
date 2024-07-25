import os
from typing import Tuple

import numpy as np
import pandas as pd


def parse_calibration_timestamps(raw_timestamps: str) -> pd.Series:
    elems = raw_timestamps.split(";")[1:]
    if len(elems) % 3 != 0:
        raise Exception(
            "number of elems in timestamp does not match expected format: name, time, timestamp"
        )
    times = []
    for name, time, timestamp in zip(*(iter(elems),) * 3):
        times.append((name, timestamp))
    s = pd.DataFrame.from_records(times, columns=["task", "timestamp"]).set_index(
        "task"
    )
    s = s.assign(
        timestamp=pd.to_datetime(pd.to_numeric(s.timestamp), unit="ms", origin="unix")
    )
    return s


def load_eeg(
    filename: str, sample_rate_hz=256, timezone="Europe/Berlin"
) -> Tuple[np.int64, pd.DataFrame, pd.Timestamp]:
    file = os.path.basename(filename).split(".")[0]
    assert file.startswith(
        "P"
    ), f"Expecting filename to be in the format P1_... but got {file}"
    participant = np.int64(file.split("_")[0][1:])  # remove the p from p1
    time = "".join(file.split("_")[2:])
    recording_start = pd.to_datetime(time, format="%Y%m%d%H%M%S")
    recording_start = recording_start.tz_localize(timezone)
    recording_start = recording_start.tz_convert("UTC")
    recording_start = recording_start.tz_localize(None)

    sample_rate_ms = 1000 / sample_rate_hz

    df = pd.read_csv(filename)
    df["timestamp"] = pd.date_range(
        start=recording_start, periods=df.shape[0], freq=f"{sample_rate_ms}ms"
    )
    df["participant"] = participant
    return participant, df.set_index("timestamp"), recording_start
