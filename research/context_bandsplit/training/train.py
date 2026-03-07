import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from research.context_bandsplit.datasets.cinematic_dataset import CinematicDataset
from research.context_bandsplit.models.separator import ContextAwareSeparator


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CinematicDataset(
        root_dir="datasets",
        sample_rate=44100,
        segment_seconds=6
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    model = ContextAwareSeparator().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    criterion = nn.L1Loss()

    epochs = 10

    for epoch in range(epochs):

        for batch in dataloader:

            batch = batch.to(device)

            optimizer.zero_grad()

            output = model(batch)

            loss = criterion(output, batch)

            loss.backward()

            optimizer.step()

        print(f"Epoch {epoch+1} Loss {loss.item()}")


if __name__ == "__main__":
    train()