# EEG Music Modeling

This project was created within the context of a university seminar at the research group for Positive Information Systems at the KIT.

## Introduction

Making music or listening to it can enhance productivity and well-being by inducing flow states. This project explores using EEG data and deep learning to create adaptive music recommendations that help maintain mood stability for better focus. Our open-source library integrates experimental EEG data with emotion recognition models to support further research. Check out the code and contribute to enhancing emotion-based music recommendation systems!

## Setup and Getting Started

This project makes use of [Poetry](https://python-poetry.org/) for dependency management. To install the dependencies, run the following command:

```bash
poetry install
```

## Data

The data used for analysis in this project is not publicly available. In case you want to reproduce this work or use the code for your own data, use the following structure:

```
data
├── raw
│   ├── 02_EYODF\ Pilot\ Data
│   │   ├── EEG
│   │   │   ├── P1_Unicorn_20240515_094201.csv
│   |   ├── all_participants_otree.csv
```

The filename structure expects the following format: `P{participant_number}_{participant_name}_{date}_{time}.csv`, where the date is in Europe/Berlin timezone. In order to change this you can modify the timezone in utils.py:load_eeg.

## Preprocessing

EEG channel data is split into 2-second windows without overlap. The data is then normalized and transformed into a 2D matrix. The matrix is then split into 3D tensors of shape (n_windows, n_channels, n_samples).

## Emotion Recognition

Emotion is modeled according to the Circumplex model of affect and used as a direct classification target.
The model is jointly optimized for Arousal and Valence classification using a K-dimensional Cross-Entropy loss.

## Models

In the context of the experiment established pre-existing architectures were compared with the modified classification head.
The models used are implemented in the [braindecode](https://braindecode.org/) library and consist of one CNN-based and one CNN + Attention based architecture ([EEGNet](https://braindecode.org/stable/generated/braindecode.models.EEGNetv4.html#braindecode.models.EEGNetv4) and [EEGConformer](https://braindecode.org/stable/generated/braindecode.models.EEGConformer.html)).
