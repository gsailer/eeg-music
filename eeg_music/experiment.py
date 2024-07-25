import argparse
import os
from typing import Literal
import lightning as L
import torch
import pandas as pd

from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from eeg_music.dataset import (
    TrainTestSplitStrategy,
    test_loader,
    train_loader,
    validation_loader,
)
from eeg_music.models.base import LikertEmotionNet
from eeg_music.models.conformer import LikertEmotionConformer
from eeg_music.models.eegnet import LikertEmotionEEGNet


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEG Music")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Choose the device to train on",
    )
    parser.add_argument("--batch", type=int, default=256, help="Choose the batch size")
    return parser.parse_args()


def load_model(
    type: Literal["conformer", "eegnet"], three_class: bool
) -> LikertEmotionNet:
    """
    MUST use function to load model to ensure there is no accidental weight leaks
    """
    if type == "conformer":
        return LikertEmotionConformer(
            n_channels=8,
            input_window_samples=512,
            learning_rate=1e-6,
            dropout=0.5,
            use_three_class_problem=three_class,
        )
    elif type == "eegnet":
        return LikertEmotionEEGNet(
            n_channels=8,
            input_window_samples=512,
            learning_rate=1e-6,
            dropout=0.5,
            use_three_class_problem=three_class,
        )


def load_lightning_logs(logdir: str, skip_scalars=["hp_metric"]) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    scalar_data = {}
    for tag in event_acc.Tags()["scalars"]:
        if tag in skip_scalars:
            continue
        scalar_events = event_acc.Scalars(tag)
        values = [e.value for e in scalar_events]
        steps = [e.step for e in scalar_events]
        scalar_data[tag] = pd.DataFrame({"step": steps, "value": values})

    return scalar_data


def prepare_data_for_seaborn(scalar_data: dict) -> pd.DataFrame:
    dfs = []
    for tag, df in scalar_data.items():
        df["metric"] = tag
        dfs.append(df)
    combined_df = pd.concat(dfs)
    return combined_df


def smooth(values: pd.Series, weight=0.8) -> pd.Series:
    smoothed_values = []
    last = values[0]
    for value in values:
        smoothed_value = last * weight + (1 - weight) * value
        smoothed_values.append(smoothed_value)
        last = smoothed_value
    return smoothed_values


if __name__ == "__main__":
    args = _parse_args()

    for strategy in TrainTestSplitStrategy:
        for normalize in [True, False]:
            for three_class in [True, False]:
                for model_type in ["conformer", "eegnet"]:
                    model = load_model(model_type, three_class).to(args.device)
                    train_set = train_loader(
                        args.device,
                        batch_size=args.batch,
                        split=strategy,
                        normalize=normalize,
                    )
                    val_set = validation_loader(
                        args.device,
                        batch_size=args.batch,
                        split=strategy,
                        normalize=normalize,
                    )
                    test_set = test_loader(
                        args.device,
                        batch_size=args.batch,
                        split=strategy,
                        normalize=normalize,
                    )
                    run_name = f"{model_type}_{strategy}_{'norm' if normalize else 'raw'}_{'three' if three_class else 'likert'}"
                    trainer = L.Trainer(
                        accelerator=args.device,
                        max_epochs=300,
                        log_every_n_steps=10,
                        logger=TensorBoardLogger("lightning_logs", name=run_name),
                    )
                    trainer.fit(
                        model=model,
                        train_dataloaders=train_set,
                        val_dataloaders=val_set,
                    )
                    trainer.test(model=model, dataloaders=test_set)

    # Plotting
    scalars = pd.DataFrame()

    for logdir in os.listdir("lightning_logs"):
        scalar_data = prepare_data_for_seaborn(
            load_lightning_logs(f"lightning_logs/{logdir}/version_0")
        )
        scalar_data = scalar_data.assign(
            model=logdir.split("_")[0],
            strategy=logdir.split("_")[1],
            normalize=logdir.split("_")[2],
            classes=logdir.split("_")[3],
        )
        scalars = pd.concat([scalars, scalar_data])

    plots = [
        "train_loss",
        "val_loss",
        "train_accuracy",
        "train_accuracy_arousal",
        "train_accuracy_pleasure",
    ]

    import seaborn as sns
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)
    scalars = scalars.assign(
        smoothed_value=scalars.groupby(
            ["metric", "model", "strategy", "normalize", "classes"]
        )["value"].transform(lambda x: smooth(x))
    )

    for strategy in ["participant", "track"]:
        for norm in ["norm", "raw"]:
            for cls in ["likert", "three"]:
                for plot in plots:
                    plot_df = scalars.loc[
                        (scalars.strategy == strategy)
                        & (scalars.normalize == norm)
                        & (scalars.classes == cls)
                        & (scalars.metric == plot)
                    ]

                    plt.figure(figsize=(12, 8))

                    sns.lineplot(
                        data=plot_df,
                        x="step",
                        y="smoothed_value",
                        hue="model",
                        style="model",
                        markers=True,
                        dashes=False,
                        errorbar=None,
                    )
                    plt.legend(title="Model")
                    plt.xlabel("Step")
                    plt.ylabel("Value")
                    plt.tight_layout()
                    plt.savefig(f"plots/{strategy}_{norm}_{cls}_{plot}.png")

    # plot for class comparison
    normalize = True

    for cls in ["likert", "three"]:
        for plot in plots:
            plot_df = scalars.loc[
                (scalars.strategy == "participant")
                & (scalars.normalize == "norm")
                & (scalars.classes == cls)
                & (scalars.metric == plot)
            ]

            plt.figure(figsize=(12, 8))

            sns.lineplot(
                data=plot_df,
                x="step",
                y="smoothed_value",
                hue="model",
                style="model",
                markers=True,
                dashes=False,
                errorbar=None,
            )
            plt.legend(title="Model")
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.tight_layout()
            plt.savefig(f"plots/participant_{cls}_{plot}.png")
