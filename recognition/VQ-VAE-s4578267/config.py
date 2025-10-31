"""
config.py contains the VQVAE parameters used in train.py and predict.py
author: Garrett Bargewell, s4578267
"""
class VQVAEConfig:
    def __init__(self) -> None:
        self.in_channels = 1
        self.out_channels = [128,256,512]
        self.latent_dim = 64
        self.kernel_size = 3
        self.num_embeds = 32
        self.commitment_cost = 1.0

vqvae_config = VQVAEConfig()