# Question 10 VQ-VAE
Create a generative model of the HipMRI Study on Prostate Cancer using the processed 2D slices (2D
images) available here with the using a VQVAE [12] or VQVAE2 [13] that has a “reasonably clear image”
and a Structured Similarity (SSIM) of over 0.6. [Hard Difficulty]

## Hyper-Parameters
The hyper-parameters were chosen by training then evaluating the model at intervals within a range of reasonable values.

A batch size of 32 was used to take advantage of the available hardware. This was considered to be the maximum value that the GPU could handle. While this size decreased the training time per epoch, it also decreased the average SSIM score achieved when testing. This is likely due to the learning rate not being adjusted to accomodate the larger batch size. A batch size of 1 showed the greatest average SSIM when trained on a single epoch.

The training ran for 100 epochs. This value was used to increase confidence that the model had been trained to its utmost potential. In practice this model approached a plateau in its performance within 10 epochs.

The learning rate was set to 5e-3 to account for the batch size of 32. For a batch size of 1, a value of 5e-4 would be more appropriate given the greater number of iterations per epoch. Learning rates within the range of 1e-2 and 1e-6 were tested.

The commit loss was selected to be 0.25 as is considered the standard by other VQ-VAE implementations. Initially a value of 1.0 was used, however this did not show an observable difference during evaulation.

## Model Structure
The implemented VQ-VAE utilises components from a vanilla VAE, specifically the Encoder and Decoder. However, a Vector Quantisation module replaces the typical reparameterisation layer.

### Encoder
The Encoder structure is a standard down-sampling neural network. It is composed of blocks which are made up of a 2D Convolution, a 2D Batch Normalisation, a LeakyReLU, and a 2D Max Pooling layer. Each block halves the height and width dimensions of the input while increasing the number of channels.

### Decoder
The Decoder is a standard up-sampling neural network, desgined to reverse the down-sampling from the Encoder. The blocks consist of a 2D Transpose Convolution and a LeakyReLU activation function. These blocks perform the reverse of the Encoder blocks, decreasing the channels and doubling the height and width dimensions.

### Vector Quantisation


## Dependencies
- matplotlib - 3.10.7
- numpy - 1.26.4
- nibabel - 5.3.2
- torch - 2.2.2
- torchmetrics - 1.8.2
- torchvision - 0.17.2
- tqdm - 4.67.1