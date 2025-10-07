import torch

import time

import os

import modules
import dataset

import matplotlib.pyplot as plt

# hyper-parameters
batch_size = 64
epochs = 1
learning_rate = 0.001
beta = 1.0

val_batch_size = 8

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

model = modules.VAE(in_channels=1, out_channels=[64,128,256], latent_dim=64, kernel_size=3, image_size=train_set[0].shape[1:]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(model)

model.train()
print("> Training Started")
start_time = time.time()
for epoch in range(epochs):
    print(f"Epoch [{epoch+1}/{epochs}]")
    total_loss = 0
    total_bce = 0
    total_kld = 0
    
    for i, imgs in enumerate(train_loader):

        images = imgs.to(device).float()

        # Forward
        output, mu, logvar = model(images)

        #plt.imshow(images[0].cpu().squeeze())
        #plt.show()

        # Calculate loss
        loss, bce, kld = model.loss_function(output, images, mu, logvar, beta)

        # Backwards and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bce += bce.item()
        total_kld += kld.item()

        # test model every tenth of an epoch
        # Save output of test as png
        if (i+1) % (len(train_loader) // 10) == 0:
            model.eval()
            with torch.no_grad():
                test_images = next(iter(val_loader)).to(device).float()
                test_output, _, _ = model(test_images)

                fig, axes = plt.subplots(2, val_batch_size, figsize=(20, 4))

                # plot
                for j in range(val_batch_size):

                    axes[0, j].imshow(test_images[j].cpu().squeeze())
                    axes[0, j].axis('off')

                    axes[1, j].imshow(test_output[j].cpu().squeeze())
                    axes[1, j].axis('off')

                plt.tight_layout()
                plt.savefig(f"train_epoch_{epoch+1}_{i+1}.png")


            model.train()


    avg_loss = total_loss / len(train_loader.dataset)
    avg_bce = total_bce / len(train_loader.dataset)
    avg_kld = total_kld / len(train_loader.dataset)

    print(f"Epoch [{epoch+1}/{epochs}] avg: Loss: {avg_loss:.5f}, BCE: {avg_bce:.5f}, KLD: {avg_kld:.5f}")


path = "VAE_model.pth"

# Save model
model.eval()
torch.save(model.state_dict(), path)