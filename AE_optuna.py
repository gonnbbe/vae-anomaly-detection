import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import optuna


class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.view(-1, 1, 28, 28)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

full_train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_size = 50000
val_size = 10000

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size]
)


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    latent_dim = trial.suggest_categorical("latent_dim", [16, 32, 64, 128, 256, 512, 1024])
    batch_size = trial.suggest_categorical("batch_size", [ 64])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = Autoencoder(latent_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, _ in train_loader:
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    return val_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best parameters:")
print(study.best_params)

print("Best validation loss:")
print(study.best_value)