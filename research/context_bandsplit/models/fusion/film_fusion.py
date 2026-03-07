import torch
import torch.nn as nn


class FiLMFusion(nn.Module):
    """
    FiLM conditioning layer
    Feature-wise Linear Modulation

    Modulates band features using context embeddings.
    """

    def __init__(self, feature_dim, context_dim):
        super().__init__()

        self.gamma = nn.Linear(context_dim, feature_dim)
        self.beta = nn.Linear(context_dim, feature_dim)

    def forward(self, features, context):

        gamma = self.gamma(context)
        beta = self.beta(context)

        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        modulated = gamma * features + beta

        return modulated