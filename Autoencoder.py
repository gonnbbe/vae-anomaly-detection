import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
start_total = time.time()
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
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
model = Autoencoder().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



epochs = 10
loss_history = []


for epoch in range(epochs):
    start_time = time.time()
    
    total_loss = 0

    for images, _ in train_loader:
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    loss_history.append(avg_loss)

    end_time = time.time()
    if device.type == "cuda":
        torch.cuda.synchronize()
    epoch_time = end_time - start_time

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s")
# Lossグラフ（log10）
# =========================
log_loss = np.log10(loss_history)

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, epochs + 1),
    log_loss,
    marker='o',
    linewidth=2
)

# 値表示
for x, y, l in zip(range(1, epochs + 1), log_loss, loss_history):

    plt.text(
        x,
        y,
        f"{l:.4f}\n(log10={y:.2f})",
        fontsize=10,
        ha='center',
        va='bottom'
    )

plt.title("Training Loss (log scale)", fontsize=20)

plt.xlabel("Epoch", fontsize=16)

plt.ylabel("log10(Loss)", fontsize=16)

plt.grid(True)

plt.show(block=False)



# 学習後、1バッチ取得
images, _ = next(iter(train_loader))
images = images.to(device)

model.eval()

with torch.no_grad():
    outputs = model(images)
# 評価モード
model.eval()

with torch.no_grad():
    outputs = model(images)

# 表示枚数
n = 10

plt.figure(figsize=(15, 4))

# 上側タイトル
plt.suptitle("Top: Original Images / Bottom: Reconstructed Images", fontsize=14)

for i in range(n):

    # 元画像（上段）
    plt.subplot(2, n, i + 1)
    plt.imshow(images[i].detach().cpu().squeeze(), cmap="gray")
    plt.axis("off")

    # 左端だけラベル表示
    if i == 0:
        plt.title("Original")

    # 復元画像（下段）
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(outputs[i].detach().cpu().squeeze(), cmap="gray")
    plt.axis("off")

    # 左端だけラベル表示
    if i == 0:
        plt.title("Reconstructed")

plt.tight_layout()
plt.show()
model.eval()

test_loss = 0

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)

        outputs = model(images)

        loss = criterion(outputs, images)

        test_loss += loss.item()

test_loss /= len(test_loader)

print("Test Loss:", test_loss)
