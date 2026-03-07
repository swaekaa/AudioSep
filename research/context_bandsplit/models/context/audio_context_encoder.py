import torch
import torch.nn as nn
import torchaudio


class AudioContextEncoder(nn.Module):
    """
    Extracts global audio context features
    such as energy, spectral patterns, and temporal structure.
    """

    def __init__(self, n_mels=128, embedding_dim=256):
        super().__init__()

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, waveform):

        mel = self.melspec(waveform)
        mel = mel.unsqueeze(1)

        features = self.encoder(mel)

        features = features.view(features.size(0), -1)

        embedding = self.fc(features)

        return embedding