
class VQVAEConfig:
    def __init__(self) -> None:
        self.in_channels = 1
        self.out_channels = [128,256,512]
        self.latent_dim = 64
        self.kernel_size = 3
        self.num_embeds = 32

vqvae_config = VQVAEConfig()