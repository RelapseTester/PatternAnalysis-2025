# Question 10 VQ-VAE
Create a generative model of the HipMRI Study on Prostate Cancer using the processed 2D slices (2D
images) available here with the using a VQVAE [12] or VQVAE2 [13] that has a “reasonably clear image”
and a Structured Similarity (SSIM) of over 0.6. [Hard Difficulty]


## Dependencies
- torch, torchvision - "pip3 install torch torchvision"
- torchmetrics - "pip3 install torchmetrics"
- matplotlib - "pip3 install matplotlib"

### dataset.py
- numpy - "pip3 install numpy"
- nibabel - "pip3 install nibabel"
- tqdm "pip3 install tqdm"