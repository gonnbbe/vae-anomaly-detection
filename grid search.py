import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import csv
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

EPOCHS = 10
LR = 0.0017
BATCH_SIZE = 64
LATENT_DIM = 64

# パラメータ数を変えるために width を変える
widths = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 元のAEクラスを貼る場所
# ============================================================
# ここでは width を変えることで総パラメータ数を変えます。
#
# 横軸は width ではなく param_count にします。
#
# つまり、
# widthを変える
# ↓
# モデルの総パラメータ数が変わる
# ↓
# param_count と test_loss の関係を見る
#
# という流れです。
# ============================================================

class Autoencoder(nn.Module):
    def __init__(self, width, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, 28, 28)
        return x


def count_parameters(model):
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()
    return total


def train_and_evaluate(width):
    model = Autoencoder(width=width, latent_dim=LATENT_DIM).to(device)

    param_count = count_parameters(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)

        print(
            f"width={width}, "
            f"params={param_count}, "
            f"Epoch [{epoch+1}/{EPOCHS}], "
            f"Train Loss: {train_loss:.6f}"
        )

    model.eval()
    train_loss_sum = 0.0
    test_loss_sum = 0.0

    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            train_loss_sum += loss.item() * images.size(0)

        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            test_loss_sum += loss.item() * images.size(0)

    final_train_loss = train_loss_sum / len(train_loader.dataset)
    final_test_loss = test_loss_sum / len(test_loader.dataset)
    elapsed = time.time() - start

    return final_train_loss, final_test_loss, param_count, elapsed


results = []

for width in widths:
    print("\n" + "=" * 60)
    print(f"Start training: width = {width}")
    print("=" * 60)

    train_loss, test_loss, param_count, elapsed = train_and_evaluate(width)

    results.append({
        "width": width,
        "param_count": param_count,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "time_sec": elapsed
    })

    print(
        f"width={width}, "
        f"params={param_count}, "
        f"train_loss={train_loss:.6f}, "
        f"test_loss={test_loss:.6f}, "
        f"time={elapsed:.2f} sec"
    )


with open("grid_search_parameter_count_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["width", "param_count", "train_loss", "test_loss", "time_sec"]
    )
    writer.writeheader()
    writer.writerows(results)


param_counts = [r["param_count"] for r in results]
train_losses = [r["train_loss"] for r in results]
test_losses = [r["test_loss"] for r in results]

plt.figure(figsize=(8, 5))
plt.plot(param_counts, train_losses, marker="o", label="Train Loss")
plt.plot(param_counts, test_losses, marker="s", label="Test Loss")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Number of Parameters")
plt.ylabel("MSE Loss")
plt.title("Double Descent Check by Parameter Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grid_search_parameter_count_plot.png", dpi=300)
plt.show()