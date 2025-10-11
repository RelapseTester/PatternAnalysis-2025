import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim 

import time

import os

import modules
import dataset

import matplotlib.pyplot as plt

# hyper-parameters
batch_size = 1
epochs = 1
learning_rate = 0.0005
beta = 1.0

val_batch_size = 1

# Set directory
#start_cwd = os.getcwd()
#os.chdir(os.path.join(start_cwd, "task4", "VAE"))
#cwd = os.getcwd()

# Load dataset
#data_dir = os.path.join(os.getcwd(), "..", "oasis_data")
#train_dir = os.path.join(data_dir, "keras_png_slices_train")
#test_dir = os.path.join(data_dir, "keras_png_slices_test")

train_dir = "recognition/VQ-VAE-2-s4578267/data/keras_slices_data/keras_slices_train"
val_dir = "recognition/VQ-VAE-2-s4578267/data/keras_slices_data/keras_slices_validate"

train_set = dataset.HipMRIDataset(X_dir=train_dir, earlyStop=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

val_set = dataset.HipMRIDataset(X_dir=val_dir)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = modules.VAE(in_channels=1, out_channels=[64,128,256], latent_dim=64, kernel_size=3, image_size=train_set[0].shape[1:]).to(device)

model = modules.VQVAE(in_channels=1, out_channels=[128,256,512], latent_dim=64, kernel_size=3, image_size=train_set[0].shape[1:], num_embeds=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
#scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1/3, total_iters=11)

ssim_score = ssim(data_range=1.0).to(device)
train_losses = []
val_losses = []

model.train()
print("> Training Started")
start_time = time.time()

for epoch in range(epochs):
    print(f"Epoch [{epoch+1}/{epochs}]")
    #total_loss = 0
    #total_bce = 0
    #total_kld = 0
    
    for i, imgs in enumerate(train_loader):

        images = imgs.to(device).float()

        # Forward
        output = model(images)

        #plt.imshow(images[0].cpu().squeeze())
        #plt.show()

        # Calculate loss
        loss = model.loss_function(output, images, beta)

        # Backwards and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #total_loss += loss.item()
        #total_bce += bce.item()
        #total_kld += kld.item()
        if epoch == 0:
            train_losses.append(loss.item())

        # test model every tenth of an epoch
        # Save output of test as png
        if (i+1) % (len(train_loader) // 10) == 0:
            print(f" - Batch [{i+1}/{len(train_loader)}]")

    model.eval()
    with torch.no_grad():
        for i, imgs in enumerate(val_loader):
            #val_images = next(iter(val_loader)).to(device).float()
            val_images = imgs.to(device).float()
            val_output = model(val_images)

            #fig, axes = plt.subplots(2, val_batch_size, figsize=(12, 4))
            #
            ## plot
            #for j in range(val_batch_size):
            #
            #    axes[0, j].imshow(val_images[j].cpu().squeeze())
            #    axes[0, j].axis('off')
            #
            #    axes[1, j].imshow(val_output[j].cpu().squeeze())
            #    axes[1, j].axis('off')
            #
            #plt.tight_layout()
            #plt.savefig(f"training/train_epoch_{epoch+1}.png")
            #
            #plt.close()
    
            #score = ssim_score(val_output, val_images).item()
            
            if epoch == 0:
                val_loss = model.loss_function(val_output, val_images, beta)
                val_losses.append(val_loss.item())
    #print(f"\tval_Loss: {val_loss:.5f} ssim: {score:.5f}")

    model.train()

    #avg_loss = total_loss / len(train_loader)

    #scheduler.step()

    #print(f"Epoch [{epoch+1}/{epochs}] avg: Loss: {avg_loss:.5f}")
end_time = time.time()
print(f"Training took {(end_time - start_time):.3f} seconds")

plt.plot(train_losses[:len(val_loader)], label="Training")
plt.plot(val_losses, label="Validation")
plt.legend()
plt.title("VQ-VAE Epoch 1, Training Losses")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.savefig("training/vqvae_losses_plot.png")

path = "training/VQVAE_model.pth"

# Save model
model.eval()
torch.save(model.state_dict(), path)