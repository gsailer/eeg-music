import lightning as L
from braindecode.models.eegnet import EEGNetv4
from eeg_music.dataset import TrainTestSplitStrategy, test_loader, train_loader
from eeg_music.models.base import LikertEmotionNet
from eeg_music.parse_data import SAMPLE_RATE_HZ, WINDOW_SIZE_SECONDS


def LikertEmotionEEGNet(*args, **kwargs):
    eeg_net = EEGNetv4(
        n_chans=kwargs.get("n_channels"),
        n_outputs=1,  # irrlevant because final_layer is cut
        n_times=kwargs.get("input_window_samples"),
        final_conv_length="auto",
        drop_prob=kwargs.get("dropout"),
        sfreq=SAMPLE_RATE_HZ,
        input_window_seconds=WINDOW_SIZE_SECONDS,
    )
    # drop classifier from eegnet
    del eeg_net.final_layer
    return LikertEmotionNet(
        eeg_model=eeg_net,
        *args,
        **kwargs,
    )


if __name__ == "__main__":
    device = "mps"

    likert_emotions = LikertEmotionEEGNet(
        n_channels=8,
        input_window_samples=512,
        learning_rate=1e-6,
        dropout=0.5,  # param from Lawhern et al. 2018
        use_three_class_problem=True,
    ).to(device)
    train_set = train_loader(
        device, batch_size=256, split=TrainTestSplitStrategy.Track, normalize=True
    )
    test_set = test_loader(
        device, batch_size=8, split=TrainTestSplitStrategy.Track, normalize=True
    )

    trainer = L.Trainer(accelerator=device, max_epochs=300, log_every_n_steps=10)
    trainer.fit(model=likert_emotions, train_dataloaders=train_set)

    trainer.test(model=likert_emotions, dataloaders=test_set)
