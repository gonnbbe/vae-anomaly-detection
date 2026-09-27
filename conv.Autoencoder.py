import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 入力: [batch, 1, 28, 28]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # -> [16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> [32, 7, 7]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> [64, 7, 7]
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # -> [32, 7, 7]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),                # -> [16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1),                 # -> [1, 28, 28]
            #nn.Sigmoid() 消すとうまくいった
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

epochs = 10
loss_history = []

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
        avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)

    avg_loss = total_loss / len(train_loader)

    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    model.eval()
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

plt.show()
# 学習後、1バッチ取得
images, _ = next(iter(train_loader))
images = images.to(device)
model.eval()

with torch.no_grad():
    outputs = model(images)
    outputs = torch.clamp(outputs, 0, 1)

images = images.cpu()
outputs = outputs.cpu()

n = 10
plt.figure(figsize=(10, 4))

for i in range(n):
    # 上：元画像
    plt.subplot(2, n, i + 1)
    plt.imshow(images[i].detach().cpu().squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # 下：復元画像
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(outputs[i].detach().cpu().squeeze(), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

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

print("Test Loss:", f"{test_loss:.6f}")
print(len(train_dataset))