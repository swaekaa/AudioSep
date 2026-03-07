import torch
import torch.nn as nn
import torchaudio


class BandSplitBackbone(nn.Module):
    """
    Splits spectrogram into frequency bands
    and processes each band independently.
    """

    def __init__(self, n_fft=1024, num_bands=4, hidden_dim=256):
        super().__init__()

        self.n_fft = n_fft
        self.num_bands = num_bands

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=n_fft // 4,
            power=None
        )

        self.band_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(n_fft // 2 // num_bands, hidden_dim, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.ReLU()
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, waveform):

        spec = self.stft(waveform)

        magnitude = torch.abs(spec)

        B, F, T = magnitude.shape

        band_size = F // self.num_bands

        band_outputs = []

        for i in range(self.num_bands):

            start = i * band_size
            end = start + band_size

            band = magnitude[:, start:end, :]

            encoded = self.band_encoders[i](band)

            band_outputs.append(encoded)

        output = torch.cat(band_outputs, dim=1)

        return output
    