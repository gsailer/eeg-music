import lightning as L
import torch
from braindecode.models.eegconformer import EEGConformer
from eeg_music.dataset import TrainTestSplitStrategy, test_loader, train_loader
from eeg_music.models.base import LikertEmotionNet
from eeg_music.parse_data import SAMPLE_RATE_HZ, WINDOW_SIZE_SECONDS


class EEGConformerWOClassifier(EEGConformer):
    def __init__(self, *args, **kwargs):
        super(EEGConformerWOClassifier, self).__init__(*args, **kwargs)
        del self.final_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # copied forward method from EEGConformer
        # and removed the final layer
        x = torch.unsqueeze(x, dim=1)  # add one extra dimension
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        # x = self.final_layer(x)
        return x


def LikertEmotionConformer(*args, **kwargs):
    conformer = EEGConformerWOClassifier(
        n_chans=kwargs.get("n_channels"),
        n_outputs=1,  # irrlevant because final_layer is cut
        n_times=kwargs.get("input_window_samples"),
        drop_prob=kwargs.get("dropout"),
        sfreq=SAMPLE_RATE_HZ,
        input_window_seconds=WINDOW_SIZE_SECONDS,
        final_fc_length="auto",
    )

    return LikertEmotionNet(
        eeg_model=conformer,
        *args,
        **kwargs,
    )


if __name__ == "__main__":
    device = "mps"

    likert_emotions = LikertEmotionConformer(
        n_channels=8,
        input_window_samples=512,
        learning_rate=1e-6,
        dropout=0.5,
        use_three_class_problem=True,
    ).to(device)
    train_set = train_loader(
        device, batch_size=256, split=TrainTestSplitStrategy.Track, normalize=False
    )
    test_set = test_loader(
        device, batch_size=8, split=TrainTestSplitStrategy.Track, normalize=False
    )

    trainer = L.Trainer(accelerator=device, max_epochs=300, log_every_n_steps=10)
    trainer.fit(model=likert_emotions, train_dataloaders=train_set)

    trainer.test(model=likert_emotions, dataloaders=test_set)
