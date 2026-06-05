import gc

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import ConfigMNIST
from src.dataset import CustomMnistDataset
from src.model_tutorial import Unet, UnetClassifier
from src.scheduler import GuidedDiffusionProcess


def train_classifier(cfg: ConfigMNIST):

    # Dataset and Dataloader
    mnist_ds = CustomMnistDataset(cfg.train_csv_path)
    mnist_dl = DataLoader(mnist_ds, cfg.batch_size, shuffle=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initiate Model
    model = UnetClassifier().to(device)

    # Initialize Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Initialize Diffusion Process
    diffusion_process = GuidedDiffusionProcess(
        num_timesteps=cfg.num_diffusion_timesteps,
        num_sampling_timesteps=cfg.num_diffusion_timesteps,
    )

    criterion = nn.CrossEntropyLoss()

    # Best Loss
    best_eval_loss = float("inf")

    print("----------------------------------")
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
        print(f"{epoch + 1:<6} {train_loss:<10.4f}  {accuracy:<10.4f}")

        # Save based on train-loss
        if train_loss < best_eval_loss:
            best_eval_loss = train_loss
            torch.save(model.state_dict(), cfg.classifier_path)

    print("----------------------------------")

    # Memory Management
    del model, imgs, noisy_imgs, labels, diffusion_process
    gc.collect()
    torch.cuda.empty_cache()


def train(cfg: ConfigMNIST):

    # Dataset and Dataloader
    mnist_ds = CustomMnistDataset(cfg.train_csv_path)
    mnist_dl = DataLoader(mnist_ds, cfg.batch_size, shuffle=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initiate Model
    model = Unet().to(device)

    # Initialize Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Initialize Diffusion Process
    diffusion_process = GuidedDiffusionProcess(
        num_timesteps=cfg.num_diffusion_timesteps, num_sampling_timesteps=cfg.num_diffusion_timesteps
    )

    # Best Loss
    best_eval_loss = float("inf")

    print("--------------------------------------------")
    print(f"\033[93mEpoch  MSE-Loss   VLB-Loss   Total-Loss\033[0m")
    print("--------------------------------------------")

    # Train
    for epoch in range(cfg.num_epochs):
        # For Loss Tracking
        losses = []
        losses_mse = []
        losses_vlb = []

        # Set model to train mode
        model.train()

        # Loop over dataloader
        for imgs, labels in tqdm(mnist_dl):
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Generate noise and timestamps
            noise = torch.randn_like(imgs).to(device)
            t = torch.randint(0, cfg.num_diffusion_timesteps, (imgs.shape[0],)).to(device)

            # Avoid Gradient Accumulation
            optimizer.zero_grad()

            # Calculate training loss
            loss_dict = diffusion_process.training_losses(model, imgs, t, noise, labels)
            loss_mse = loss_dict["mse_loss"].mean()
            loss_vlb = loss_dict["vlb_loss"].mean()
            loss = loss_mse + loss_vlb

            losses.append(loss.item())
            losses_mse.append(loss_mse.item())
            losses_vlb.append(loss_vlb.item())

            # Backprop + Update model params
            loss.backward()
            optimizer.step()

        # Mean Losses
        mean_mse_loss = np.mean(losses_mse)
        mean_vlb_loss = np.mean(losses_vlb)
        mean_total_loss = np.mean(losses)

        # Display
        print(f"{epoch + 1:<6} {mean_mse_loss:<10.4f}  {mean_vlb_loss:<10.4f} {mean_total_loss:<8.4f}")

        # Save based on train-loss
        if mean_total_loss < best_eval_loss:
            best_eval_loss = mean_total_loss
            torch.save(model.state_dict(), cfg.model_path)

    print("--------------------------------------------")

    # Memory Management
    del model, imgs, labels, diffusion_process
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Load config
    cfg = ConfigMNIST()

    # Train the classifier
    print("Training the classifier:")
    train_classifier(cfg)

    # Train diffusion model
    print("Training the diffusion model:")
    train(cfg)
