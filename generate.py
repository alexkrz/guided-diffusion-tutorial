import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.config import ConfigImageNet, ConfigMNIST
from src.model_openai import EncoderUNetModel, UNetModel
from src.model_tutorial import Unet, UnetClassifier
from src.scheduler import GuidedDiffusionProcess


def generate(cfg, y):
    """
    Given Pretrained U-net model and label y, Generate Real-life
    Images conditioned on label y from noise by going backward step by step. i.e.,
    Mapping of Random Noise to Real-life images.
    """

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f'Device: {device}\n')

    # Initialize Diffusion Reverse Process
    diffusion_process = GuidedDiffusionProcess(
        num_timesteps=cfg.num_diffusion_timesteps,
        num_sampling_timesteps=cfg.num_sampling_timesteps,
        classifier_guidance=cfg.classifier_guidance,
        classifier_scale=cfg.classifier_scale,
    )

    # Classifier Guidance
    classifier = None
    if cfg.classifier_guidance:
        # Load Classifier Model
        if cfg.name == "openai":
            weights = torch.load(cfg.classifier_path, map_location="cpu", weights_only=True)
            classifier = EncoderUNetModel.from_config(cfg.classifier_config_path)
            classifier.load_state_dict(state_dict=weights)
        elif cfg.name == "tutorial":
            weights = torch.load(cfg.classifier_path, map_location="cpu", weights_only=True)
            model = UnetClassifier()
            model.load_state_dict(state_dict=weights)
        else:
            raise NotImplementedError()
        classifier.to(device)
        classifier.eval()

    # Load UNet Model
    if cfg.name == "openai":
        weights = torch.load(cfg.model_path, map_location="cpu", weights_only=True)
        # print("Number of weights.keys():", len(weights.keys()))
        model = UNetModel.from_config(cfg.model_config_path)
        model.load_state_dict(state_dict=weights)
    elif cfg.name == "tutorial":
        # Training script previously saved full module checkpoints
        weights = torch.load(cfg.model_path, map_location="cpu", weights_only=True)
        model = Unet()
        model.load_state_dict(state_dict=weights)
    else:
        raise NotImplementedError()
    model.to(device)
    model.eval()

    # Labels for class-conditional generation are expected as a 1-D [batch] tensor.
    return_single = np.asarray(y).ndim == 0
    y = torch.as_tensor(y, device=device, dtype=torch.long).view(-1)
    batch_size = y.shape[0]

    # Generate Noise sample from N(0, 1)
    xt = torch.randn(batch_size, cfg.in_channels, cfg.img_size, cfg.img_size).to(device)

    # Denoise step by step by going backward.
    num_sampling_timesteps = len(diffusion_process.use_timesteps)
    with torch.no_grad():
        for t in tqdm(reversed(range(num_sampling_timesteps))):
            timestep_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            xt = diffusion_process.p_sample(
                model,
                xt,
                timestep_batch,
                y,
                classifier,
            )["sample"]

    # Convert the image to proper scale
    xt = torch.clamp(xt, -1.0, 1.0).detach().cpu()
    xt = (xt + 1) / 2

    # Convert to uint8
    if xt.shape[1] == 1:
        xt = 255 * xt[:, 0].numpy()  # (B, H, W)
    else:
        xt = 255 * xt.permute(0, 2, 3, 1).numpy()  # (B, H, W, C)

    # Memory Management
    del model, diffusion_process
    gc.collect()
    torch.cuda.empty_cache()

    xt = xt.astype(np.uint8)
    if return_single:
        return xt[0]
    return xt


def run_generate(cfg, steps: int = 1000, guidance: bool = False):
    cfg.num_sampling_timesteps = steps
    cfg.classifier_guidance = guidance

    # Generate
    batch_size = 4
    num_rows = 1
    print(f"Generating {num_rows} label-batches x {batch_size} images with steps: {steps} and guidance: {guidance}")
    row_batches = []
    row_labels = []
    for _ in range(num_rows):
        label = np.random.randint(0, cfg.num_classes)
        y = np.full((batch_size,), label, dtype=np.int64)
        xt_batch = generate(cfg, y)
        row_batches.append(xt_batch)
        row_labels.append(label)

    # Visualize
    fig, axes = plt.subplots(num_rows, batch_size, figsize=(6, 1.5 * num_rows), squeeze=False)

    # Plot each batch in a row; show the conditioning label only on the far-left axis.
    for row_idx in range(num_rows):
        for col_idx in range(batch_size):
            ax = axes[row_idx, col_idx]
            img = row_batches[row_idx][col_idx]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[row_idx, 0].set_ylabel(
            f"Label: {row_labels[row_idx]}",
            # rotation=90,
            # labelpad=42,
            # va="center",
            # ha="right",
        )

    fig.tight_layout()  # Reserve left margin for row labels.
    fig.savefig(f"results/{cfg.name}_step-{steps}-guidance-{guidance}.png")
    # plt.show()


if __name__ == "__main__":
    cfg = ConfigImageNet()
    # Generate and plot results
    run_generate(cfg, steps=1000, guidance=False)
    # run_generate(250, guidance=False)
    # run_generate(1000, guidance=True)
    # run_generate(250, guidance=True)
