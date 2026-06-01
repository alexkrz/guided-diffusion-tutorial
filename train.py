# %%
# Imports
import gc
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CustomMnistDataset
from src.model import UnetClassifier
from src.scheduler import GuidedDiffusionProcess


# %%
# Config
@dataclass
class CONFIG:
    # Save and Load Paths
    train_csv_path = "data/train.csv"
    test_csv_path = "data/test.csv"
    model_path = "checkpoints/openai_unet.pth"
    classifier_path = "results/openai_unet_classifier.pth"
    generated_csv_path = "results/mnist_generated_data.csv"

    # Training Hyperparams
    num_epochs = 50
    lr = 1e-4
    num_diffusion_timesteps = 1000
    batch_size = 128
    img_size = 28
    in_channels = 1
    num_classes = 10

    # Sampling Hyperparams
    num_img_to_generate = 256
    num_sampling_timesteps = 1000
    classifier_guidance = False
    classifier_scale = 1.0


# %%
# Train classifier
def train_classifier(cfg: CONFIG):

    # Dataset and Dataloader
    mnist_ds = CustomMnistDataset(cfg.train_csv_path)
    mnist_dl = DataLoader(mnist_ds, cfg.batch_size, shuffle=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Initiate Model
    model = UnetClassifier().to(device)

    # Initialize Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Initialize Diffusion Process
    diffusion_process = GuidedDiffusionProcess(
        num_timesteps=cfg.num_diffusion_timesteps, num_sampling_timesteps=cfg.num_diffusion_timesteps
    )

    criterion = nn.CrossEntropyLoss()

    # Best Loss
    best_eval_loss = float("inf")

    print("\n----------------------------------")
    print(f"\033[93mEpoch  Train-Loss   Accuracy\033[0m")
    print("----------------------------------")

    # Train
    for epoch in range(cfg.num_epochs // 2):
        # For Loss Tracking
        running_loss = 0.0
        correct = 0
        total = 0

        # Set model to train mode
        model.train()

        # Loop over dataloader
        for imgs, labels in tqdm(mnist_dl):
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Add noise
            noise = torch.randn_like(imgs).to(device)
            t = torch.randint(0, cfg.num_diffusion_timesteps, (imgs.shape[0],)).to(device)
            noisy_imgs = diffusion_process.add_noise(imgs, noise, t)

            # Avoid Gradient Accumulation
            optimizer.zero_grad()

            # Prediction
            outputs = model(noisy_imgs, t)
            loss = criterion(outputs, labels)

            # Backprop + Update model params
            loss.backward()
            optimizer.step()

            # Loss Tracking
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Mean Losses and Acc
        train_loss = running_loss / len(mnist_dl)
        accuracy = correct / total

        # Display
        print(f"{epoch + 1:<6} {train_loss:<10.4f}  {accuracy:<10.4f}", end="   ")

        # Save based on train-loss
        if train_loss < best_eval_loss:
            best_eval_loss = train_loss
            torch.save(model, cfg.classifier_path)

    print("----------------------------------")

    # Memory Management
    del model, imgs, noisy_imgs, labels, diffusion_process
    gc.collect()
    torch.cuda.empty_cache()


# Config
cfg = CONFIG()

# TRAIN
train_classifier(cfg)

# %%
