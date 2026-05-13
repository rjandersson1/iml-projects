from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F

"""
README FIRST

The below code is a template for the solution. You can change the code according
to your preferences, but the testing function has to save the output of your 
model on the test data as it does in this template. This output must be submitted.

Replace the dummy code with your own code in the TODO sections.

We also encourage you to use tensorboard or wandb to log the training process
and the performance of your model. This will help you to debug your model and
to understand how it is performing. But the template does not include this
functionality.
Link for wandb:
https://docs.wandb.ai/quickstart/
Link for tensorboard: 
https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
"""

# The device is automatically set to GPU if available, otherwise CPU
# If you want to force the device to CPU, you can change the line to
# device = torch.device("cpu")

# If you have a Mac consult the following link:
# https://pytorch.org/docs/stable/notes/mps.html

# It is important that your model and all data are on the same device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(**kwargs):
    """
    Get the training and test data. The data files are assumed to be in the
    same directory as this script.

    Args:
    - kwargs: Additional arguments that you might find useful - not necessary

    Returns:
    - train_data_input: Tensor[N_train_samples, C, H, W]
    - train_data_label: Tensor[N_train_samples, C, H, W]
    - test_data_input: Tensor[N_test_samples, C, H, W]
    where N_train_samples is the number of training samples, N_test_samples is
    the number of test samples, C is the number of channels (1 for grayscale),
    H is the height of the image, and W is the width of the image.
    """
    # Load the training data
    train_data = np.load("train_data.npz")["data"]

    # Make the training data a tensor
    train_data = torch.tensor(train_data, dtype=torch.float32)

    # Load the test data
    test_data_input = np.load("test_data.npz")["data"]

    # Make the test data a tensor
    test_data_input = torch.tensor(test_data_input, dtype=torch.float32)

    ########################################
    train_data_label = train_data.clone()
    train_data_input = train_data.clone()

    # From task description: Given an image of dimensions (28, 28) we set the center 8x8 pixels to black, i.e. mask them
    # train_data_input: torch.Size([60000, 1, 28, 28])
    train_data_input[:, :, 10:18, 10:18] = 0

    # Visualize the training data if needed
    # Set to False if you don't want to save the images
    if True:
        # Create the output directory if it doesn't exist
        if not Path("train_image_output").exists():
            Path("train_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting train images"):
            # Show the training and the target image side by side
            plt.subplot(1, 2, 1)
            plt.imshow(train_data_input[i].squeeze(), cmap="gray")
            plt.title("Training Input")
            plt.subplot(1, 2, 2)
            plt.title("Training Label")
            plt.imshow(train_data_label[i].squeeze(), cmap="gray")

            plt.savefig(f"train_image_output/image_{i}.png")
            plt.close()

    return train_data_input, train_data_label, test_data_input

def training(train_data_input, train_data_label, **kwargs):
    model = Model()
    model.to(device) # Move once at the start

    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=5*1e-5, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=5*1e-4)

    # Dataset Splitting
    batch_size = 32
    total_samples = train_data_input.size(0)
    split_idx = total_samples - batch_size

    train_dataset = TensorDataset(train_data_input[:split_idx], train_data_label[:split_idx])
    val_dataset = TensorDataset(train_data_input[split_idx:], train_data_label[split_idx:])

    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Pre-prepare validation tensors on device to avoid doing it every epoch
    val_x = val_dataset.tensors[0].to(device)
    val_y = val_dataset.tensors[1].to(device)

    # Training history for plotting
    train_loss_history = []
    val_loss_history = []

    n_epochs = 20
    for epoch in range(n_epochs):
        model.train() # Set to train at start of epoch
        train_loss = 0.0
        
        for x, y in tqdm(data_loader, desc=f"Epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss = loss.item() # Keep track of the last training loss

        # 2. VALIDATION STEP
        model.eval() 
        with torch.no_grad():
            val_output = model(val_x)
            val_loss = criterion(val_output, val_y).item()

        print(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # 3. PLOT THE RESULTS
        title = "full_padding"
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_history, label='Training Loss', color='blue', linewidth=2)
        plt.plot(val_loss_history, label='Validation Loss', color='red', linestyle='--', linewidth=2)
        plt.title(f'Model Loss Progression (MSE) - {title}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.savefig(f'loss_plot_{title}.png')
    
    return model

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Feature Extraction (Padding)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=5),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.1)
        )

        # 2. Regression Layers (Input size is now 12*12 = 144)
        self.regressor = nn.Sequential(
            nn.Linear(324, 784) # Output stays 784 to reconstruct 28x28
        )
        
        self.regressor2 = nn.Sequential(
            nn.Linear(784, 784) # Additional layer for more complex transformations
        )

    def forward(self, x):
        x_old = x.clone()

        x = self.feature_extractor(x)

        # Flatten for Regression
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        x = x.view(x.size(0), 1, 28, 28)

        # 2. Transpose to prioritize columns
        x = x.transpose(2, 3).contiguous() 
        x = x.view(x.size(0), -1)
        x = self.regressor2(x)
        x = x.view(x.size(0), 1, 28, 28)
        x = x.transpose(2, 3).contiguous()

        x = x + x_old

        return x

def testing(model, test_data_input):
    """
    Uses your model to predict the ouputs for the test data. Saves the outputs
    as a binary file. This file needs to be submitted. This function does not
    need to be modified except for setting the batch_size value. If you choose
    to modify it otherwise, please ensure that the generating and saving of the
    output data is not modified.

    Args:
    - model: torch.nn.Module
    - test_data_input: Tensor
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        test_data_input = test_data_input.to(device)
        # Predict the output batch-wise to avoid memory issues
        test_data_output = []
        # TODO: You can increase or decrease this batch size depending on your
        # memory requirements of your computer / model
        # This will not affect the performance of the model and your score
        batch_size = 32
        for i in tqdm(
            range(0, test_data_input.shape[0], batch_size),
            desc="Predicting test output",
        ):
            output = model(test_data_input[i : i + batch_size])
            test_data_output.append(output.cpu())
        test_data_output = torch.cat(test_data_output)

    # Ensure the output has the correct shape
    assert test_data_output.shape == test_data_input.shape, (
        f"Expected shape {test_data_input.shape}, but got "
        f"{test_data_output.shape}."
        "Please ensure the output has the correct shape."
        "Without the correct shape, the submission cannot be evaluated and "
        "will hence not be valid."
    )

    # Save the output
    test_data_output = test_data_output.numpy()
    # Ensure all values are in the range [0, 255]
    save_data_clipped = np.clip(test_data_output, 0, 255)
    # Convert to uint8
    # Ensure your model outputs values in the [0, 255] range before this step! If you normalized your data to [0, 1], you must multiply by 255 before saving.
    save_data_uint8 = save_data_clipped.astype(np.uint8)
    # Loss is only computed on the masked area - so set the rest to 0 to save
    # space
    save_data = np.zeros_like(save_data_uint8)
    save_data[:, :, 10:18, 10:18] = save_data_uint8[:, :, 10:18, 10:18]

    np.savez_compressed(
        "submit_this_test_data_output.npz", data=save_data)

    # You can plot the output if you want
    # Set to False if you don't want to save the images
    if True:
        # Create the output directory if it doesn't exist
        if not Path("test_image_output").exists():
            Path("test_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting test images"):
            # Show the training and the target image side by side
            plt.subplot(1, 2, 1)
            plt.title("Test Input")
            plt.imshow(test_data_input[i].squeeze().cpu().numpy(), cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(test_data_output[i].squeeze(), cmap="gray")
            plt.title("Test Output")

            plt.savefig(f"test_image_output/image_{i}.png")
            plt.close()


def main():
    seed = 0
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # You don't need to change the code below
    # Load the data
    train_data_input, train_data_label, test_data_input = load_data()
    # Train the model
    model = training(train_data_input, train_data_label)

    # Test the model (this also generates the submission file)
    # The name of the submission file is submit_this_test_data_output.npz
    testing(model, test_data_input)

    return None


if __name__ == "__main__":
    main()
