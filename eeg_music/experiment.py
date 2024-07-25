import argparse
from typing import Literal
import lightning as L
import torch

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from eeg_music.dataset import (
    TrainTestSplitStrategy,
    test_loader,
    train_loader,
    validation_loader,
)
from eeg_music.models.conformer import LikertEmotionConformer
from eeg_music.models.eegnet import LikertEmotionEEGNet


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEG Music")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Choose the device to use",
    )
    parser.add_argument(
        "--batch", type=int, default=256, help="Choose the batch size to use"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Choose the patience for early stopping",
    )
    return parser.parse_args()


def load_model(
    type: Literal["conformer", "eegnet"], three_class: bool
) -> torch.nn.Module:
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
                        callbacks=[
                            EarlyStopping(
                                monitor="val_loss", mode="min", patience=args.patience
                            )
                        ],
                        logger=TensorBoardLogger("lightning_logs", name=run_name),
                    )
                    trainer.fit(
                        model=model,
                        train_dataloaders=train_set,
                        val_dataloaders=val_set,
                    )
                    trainer.test(model=model, dataloaders=test_set)
