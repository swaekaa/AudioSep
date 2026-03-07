import torch
import torch.nn as nn

from .bandsplit.bandsplit_backbone import BandSplitBackbone
from .context.audio_context_encoder import AudioContextEncoder
from .context.speaker_encoder import SpeakerEncoder
from .fusion.film_fusion import FiLMFusion


class ContextAwareSeparator(nn.Module):

    def __init__(self):

        super().__init__()

        # Backbone
        self.backbone = BandSplitBackbone()

        # Context encoders
        self.audio_context = AudioContextEncoder()
        self.speaker_encoder = SpeakerEncoder()

        # Context dimension
        context_dim = 512
        feature_dim = 256

        # FiLM conditioning
        self.film = FiLMFusion(feature_dim, context_dim)

        # Mask decoder
        self.mask_decoder = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(feature_dim, 1, 1)
        )

    def forward(self, waveform):

        # Bandsplit backbone
        band_features = self.backbone(waveform)

        # Context features
        audio_ctx = self.audio_context(waveform)
        speaker_ctx = self.speaker_encoder(waveform)

        context = torch.cat([audio_ctx, speaker_ctx], dim=-1)

        # Apply FiLM conditioning
        conditioned = self.film(band_features, context)

        # Predict mask
        mask = self.mask_decoder(conditioned)

        return mask