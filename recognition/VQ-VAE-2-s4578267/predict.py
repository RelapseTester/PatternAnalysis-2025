import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim 

import time

import modules
import dataset

import matplotlib.pyplot as plt

test_dir = "recognition/VQ-VAE-2-s4578267/data/keras_slices_data/keras_slices_test"

test_set = dataset.HipMRIDataset(X_dir=test_dir, earlyStop=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = modules.VQVAE(in_channels=1, out_channels=[128,256,512], latent_dim=64, kernel_size=3, image_size=test_set[0].shape[1:], num_embeds=32).to(device)

path = "training/VQVAE_model.pth"

model_state_dict = torch.load(path, weights_only=True)
model.load_state_dict(model_state_dict)

# Test model
ssim_score = ssim(data_range=1.0).to(device)
test_scores = []
test_losses = []

num_images = 10
j = 0

model.eval()
print("> Testing Started")
start_time = time.time()

for i, imgs in enumerate(test_loader):

    images = imgs.to(device).float()

    # Forward
    output = model(images)

    # Calculate Loss
    loss = model.loss_function(output, images, 1.0)
    test_losses.append(loss.item())

    # Calculate SSIM
    test_scores.append(ssim_score(output, images).item())

    # plot test input and reconstruction
    if (i+1) % (len(test_loader) // 10) == 0:
        with torch.no_grad():
            fig, axes = plt.subplots(1,2)

            axes[0].imshow(images.cpu().squeeze())
            axes[0].set_title("Original")
            axes[0].axis('off')

            axes[1].imshow(output.cpu().squeeze())
            axes[1].set_title("Reconstruction")
            axes[1].axis('off')

            plt.suptitle(f"VQ-VAE Test Sample {j+1}")
            fig.text(0.5, 0.01, f"SSIM: {test_scores[-1]:.3f}, Loss: {test_losses[-1]:.5f}", ha='center', va='bottom')
            plt.savefig(f"testing/test_images_{j+1}.png")
            plt.close()
            j += 1

end_time = time.time()
print(f"Testing took {(end_time - start_time):.3f} seconds")
print(f"Average loss: {(sum(test_losses) / len(test_losses)):.5f}, Average SSIM score: {(sum(test_scores) / len(test_scores)):.3f}")