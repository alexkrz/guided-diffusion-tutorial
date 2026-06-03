import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.config import Config
from src.scheduler import GuidedDiffusionProcess


def generate(cfg: Config, y):
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
        classifier = torch.load(
            cfg.classifier_path,
            map_location="cpu",
            weights_only=False,
        ).to(device)
        classifier.eval()

    # Set model to eval mode
    model = torch.load(
        cfg.model_path,
        map_location="cpu",
        weights_only=False,  # Training script currently saves full module checkpoints
    ).to(device)
    model.eval()

    # Generate Noise sample from N(0, 1)
    xt = torch.randn(1, cfg.in_channels, cfg.img_size, cfg.img_size).to(device)

    # Denoise step by step by going backward.
    num_sampling_timesteps = len(diffusion_process.use_timesteps)
    with torch.no_grad():
        for t in tqdm(reversed(range(num_sampling_timesteps))):
            xt = diffusion_process.p_sample(
                model,
                xt,
                torch.as_tensor(t).unsqueeze(0).to(device),
                torch.as_tensor(y).to(device),
                classifier,
            )["sample"]

    # Convert the image to proper scale
    xt = torch.clamp(xt, -1.0, 1.0).detach().cpu()
    xt = (xt + 1) / 2

    # Convert to uint8
    xt = 255 * xt[0][0].numpy()

    # Memory Management
    del model, diffusion_process
    gc.collect()
    torch.cuda.empty_cache()

    return xt.astype(np.uint8)


def run_generate(steps: int = 1000, guidance: bool = False):
    cfg = Config()
    cfg.num_sampling_timesteps = steps
    cfg.classifier_guidance = guidance

    # Generate
    print(f"Generating 25 images with steps: {steps} and guidance: {guidance}")
    generated_imgs = []
    cond_labels = []
    for i in range(25):
        y = np.random.randint(0, 10)  # Randomly select a label bw 1 to 10.
        xt = generate(cfg, y)
        generated_imgs.append(xt)
        cond_labels.append(y)

    # Visualize
    fig, axes = plt.subplots(5, 5, figsize=(6, 6))
    axes: list[plt.Axes] = np.ravel(axes)

    # Plot each image in the corresponding subplot
    for i, ax in enumerate(axes):
        ax.imshow(generated_imgs[i], cmap="gray")  # You might need to adjust the colormap based on your images
        ax.set_title(f"Cond-Label: {cond_labels[i]}")
        ax.axis("off")  # Turn off axis labels

    fig.tight_layout()  # Adjust spacing between subplots
    fig.savefig(f"results/mnist_step-{steps}-guidance-{guidance}.png")
    # plt.show()


if __name__ == "__main__":
    # Generate and plot results
    run_generate(1000, guidance=False)
    run_generate(250, guidance=False)
    run_generate(1000, guidance=True)
    run_generate(250, guidance=True)
