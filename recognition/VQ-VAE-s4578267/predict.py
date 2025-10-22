"""
predict.py contains a sample script for loading a trained model and plotting its outputs.
author: Garrett Bargewell, s4578267
"""

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim 

import modules
import dataset
from config import vqvae_config

import matplotlib.pyplot as plt

test_dir = "recognition/VQ-VAE-s4578267/data/keras_slices_data/keras_slices_test"

test_set = dataset.HipMRIDataset(X_dir=test_dir, earlyStop=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = modules.VQVAE(
    in_channels=vqvae_config.in_channels,
    out_channels=vqvae_config.out_channels, 
    latent_dim=vqvae_config.latent_dim, 
    kernel_size=vqvae_config.kernel_size,
    num_embeds=vqvae_config.num_embeds
    ).to(device)

path = "training/VQVAE_model.pth"

model_state_dict = torch.load(path, weights_only=True)
model.load_state_dict(model_state_dict)

# Test model
ssim_score = ssim(data_range=1.0).to(device)

num_images = 10
j = 0

model.eval()

# Generate images of predicted vs target
for i in range(num_images):
    imgs = next(iter(test_loader))
    images = imgs.to(device).float()

    # Forward
    output, _, _ = model(images)

    # Calculate Loss
    loss = model.loss_function(output, images, 1.0).item()

    # Calculate SSIM
    score = (ssim_score(output, images).item())

    # plot test input and reconstruction
    with torch.no_grad():
        fig, axes = plt.subplots(1,2)

        axes[0].imshow(images.cpu().squeeze())
        axes[0].set_title("Original")
        axes[0].axis('off')

        axes[1].imshow(output.cpu().squeeze())
        axes[1].set_title("Reconstruction")
        axes[1].axis('off')

        plt.suptitle(f"VQ-VAE Test Sample {j+1}")
        fig.text(0.5, 0.01, f"SSIM: {score:.3f}, Loss: {loss:.5f}", ha='center', va='bottom')
        plt.savefig(f"testing/test_images_{j+1}.png")
        plt.close()
        j += 1
