from dataclasses import dataclass


@dataclass
class Config:
    # Save and Load Paths
    train_csv_path = "data/train.csv"
    test_csv_path = "data/test.csv"
    model_path = "results/openai_unet.pth"
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
