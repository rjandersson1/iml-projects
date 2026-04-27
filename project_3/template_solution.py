from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    train_data = np.load("train_data.npz")["data"]
    test_data_input = np.load("test_data.npz")["data"]

    train_data = torch.tensor(train_data, dtype=torch.float32) / 255.0
    test_data_input = torch.tensor(test_data_input, dtype=torch.float32) / 255.0

    train_data_label = train_data.clone()
    train_data_input = train_data.clone()

    train_data_input[:, :, 10:18, 10:18] = 0

    return train_data_input, train_data_label, test_data_input


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        pred = self.net(x)

        out = x.clone()
        out[:, :, 10:18, 10:18] = pred[:, :, 10:18, 10:18]

        return out


def training(train_data_input, train_data_label):
    model = Model().to(device)
    model.train()

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-6
    )

    batch_size = 256
    dataset = TensorDataset(train_data_input, train_data_label)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_epochs = 10

    for epoch in range(n_epochs):
        running_loss = 0.0

        for x, y in tqdm(data_loader, desc=f"Training Epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(x)

            loss = criterion(
                output[:, :, 10:18, 10:18],
                y[:, :, 10:18, 10:18]
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch} loss: {avg_loss:.6f}")

    return model


def testing(model, test_data_input):
    model.eval()
    model.to(device)

    with torch.no_grad():
        test_data_input = test_data_input.to(device)

        test_data_output = []
        batch_size = 128

        for i in tqdm(
            range(0, test_data_input.shape[0], batch_size),
            desc="Predicting test output",
        ):
            output = model(test_data_input[i:i + batch_size])
            test_data_output.append(output.cpu())

        test_data_output = torch.cat(test_data_output)

    assert test_data_output.shape == test_data_input.shape

    test_data_output = test_data_output.numpy() * 255.0

    save_data_clipped = np.clip(test_data_output, 0, 255)
    save_data_uint8 = save_data_clipped.astype(np.uint8)

    save_data = np.zeros_like(save_data_uint8)
    save_data[:, :, 10:18, 10:18] = save_data_uint8[:, :, 10:18, 10:18]

    np.savez_compressed("submit_this_test_data_output.npz", data=save_data)

    if True:
        Path("test_image_output").mkdir(exist_ok=True)

        for i in tqdm(range(20), desc="Plotting test images"):
            plt.subplot(1, 2, 1)
            plt.title("Test Input")
            plt.imshow(test_data_input[i].squeeze().cpu().numpy(), cmap="gray")

            plt.subplot(1, 2, 2)
            plt.title("Test Output")
            plt.imshow(test_data_output[i].squeeze(), cmap="gray")

            plt.savefig(f"test_image_output/image_{i}.png")
            plt.close()


def main():
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_data_input, train_data_label, test_data_input = load_data()

    model = training(train_data_input, train_data_label)

    testing(model, test_data_input)


if __name__ == "__main__":
    main()