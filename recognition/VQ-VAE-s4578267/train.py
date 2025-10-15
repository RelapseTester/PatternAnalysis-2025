import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim 

import time

import modules
import dataset
from config import vqvae_config

import matplotlib.pyplot as plt

# hyper-parameters
batch_size = 1
epochs = 1
learning_rate = 0.0005
beta = 1.0

val_batch_size = 1

train_dir = "recognition/VQ-VAE-s4578267/data/keras_slices_data/keras_slices_train"
val_dir = "recognition/VQ-VAE-s4578267/data/keras_slices_data/keras_slices_validate"
test_dir = "recognition/VQ-VAE-s4578267/data/keras_slices_data/keras_slices_test"

train_set = dataset.HipMRIDataset(X_dir=train_dir, earlyStop=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

val_set = dataset.HipMRIDataset(X_dir=val_dir)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, shuffle=True)

test_set = dataset.HipMRIDataset(X_dir=test_dir, earlyStop=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = modules.VQVAE(
    in_channels=vqvae_config.in_channels,
    out_channels=vqvae_config.out_channels, 
    latent_dim=vqvae_config.latent_dim, 
    kernel_size=vqvae_config.kernel_size, 
    image_size=train_set[0].shape[1:], 
    num_embeds=vqvae_config.num_embeds
    ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

###
###
###

# Train and validate model
model.train()
print("> Training Started")
start_time = time.time()

for epoch in range(epochs):

    train_losses = []
    val_losses = []
    
    print(f"Epoch [{epoch+1}/{epochs}]")
    
    for i, imgs in enumerate(train_loader):

        images = imgs.to(device).float()

        # Forward
        output = model(images)

        # Calculate loss
        loss = model.loss_function(output, images, beta)

        # Backwards and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Print progress every 10% of an epoch
        if (i+1) % (len(train_loader) // 10) == 0:
            print(f" - Batch [{i+1}/{len(train_loader)}]")
        
        # Test model using the validate dataset
        if (i+1 < len(val_loader)):
            model.eval()
            with torch.no_grad():
                val_images = next(iter(val_loader)).to(device).float()
                val_output = model(val_images)
                
                val_loss = model.loss_function(val_output, val_images, beta)
                val_losses.append(val_loss.item())

            model.train()
        
    print(f" - Avg training loss: {(sum(train_losses) / len(train_losses)):.5f}, Avg validation loss: {(sum(val_losses) / len(val_losses)):.5f}")

    # Plot first epoch training losses vs validation losses
    if epoch == 0:
        plt.plot(train_losses[:len(val_loader)], label="Training")
        plt.plot(val_losses, label="Validation")
        plt.legend()
        plt.title(f"VQ-VAE Epoch {epoch+1} Losses")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.savefig("training/vqvae_losses_plot.png")
        plt.close()

end_time = time.time()
print(f"Training took {(end_time - start_time):.3f} seconds")

###
###
###

# Test model
ssim_score = ssim(data_range=1.0).to(device)
test_scores = []
test_losses = []

model.eval()
print("\n> Testing Started")
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

end_time = time.time()
print(f"Testing took {(end_time - start_time):.3f} seconds")
print(f"Average loss: {(sum(test_losses) / len(test_losses)):.5f}, Average SSIM score: {(sum(test_scores) / len(test_scores)):.3f}")


path = "training/VQVAE_model.pth"

# Save model
model.eval()
torch.save(model.state_dict(), path)