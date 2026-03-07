import torch
import torch.nn as nn
import torchaudio


class SpeakerEncoder(nn.Module):
    """
    Extracts speaker embeddings from audio.
    Can be trained using VoxCeleb.
    """

    def __init__(self, embedding_dim=256):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, waveform):

        if waveform.dim() == 3:
            waveform = waveform.mean(dim=1)

        waveform = waveform.unsqueeze(1)

        features = self.conv(waveform)

        features = features.squeeze(-1)

        embedding = self.fc(features)

        return embedding