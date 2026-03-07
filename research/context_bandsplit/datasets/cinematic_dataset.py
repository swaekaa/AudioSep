import os
import torch
import torchaudio
from torch.utils.data import Dataset


class CinematicDataset(Dataset):
    def __init__(self, root_dir, sample_rate=44100, segment_seconds=6):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.segment_length = sample_rate * segment_seconds

        self.audio_files = []
        
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".wav"):
                    self.audio_files.append(os.path.join(root, f))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        path = self.audio_files[idx]

        waveform, sr = torchaudio.load(path)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )

        if waveform.shape[1] > self.segment_length:
            waveform = waveform[:, :self.segment_length]

        return waveform