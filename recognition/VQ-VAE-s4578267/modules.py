import torch
import torch.nn as nn
import torch.nn.functional as F

class EncodeBlock(nn.Module):
    """
    Basic Encode block used in an Encoder. Block constists of a Conv2d, BatchNorm2d, LeakyReLU, and MaxPool2d.
    Halves the Height and Width of the input using the MaxPool2d
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size=1, stride=1, padding=1):
        """
        Initialises The EncodeBlock with the given parameters.

        Args:
            in_channels (int): input channels to the Conv2d
            out_channels (int): output channels to the Conv2d
            kernel_size: kernel_size of the Conv2d
            stride: stride of the Conv2d
            padding: padding of the Conv2d
        """
        super(EncodeBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # halve height and width
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies this block to the input x.

        Args:
            x (torch.Tensor): input data to this block
        
        Returns:
            torch.Tensor: result of the input data parsed by this block
        """
        out = self.layer(x)
        return out

class Encoder(nn.Module):
    """
    Encoder composed of multiple EncodeBlocks. Does NOT flatten output.
    """

    def __init__(self, in_channels=1, out_channels=[64,128,256], kernel_size=3):
        """
        Initialises The Encoder with the given parameters. 

        in_channels (int): initial input channels to encoder.
        out_channels (list[int]): output channels of each layer. len(out_channels) is used to determine number of layers to create.
        kernel_size (int | (int,int)), default=3): kernel size for all layers. 
        """
        super(Encoder, self).__init__()

        self.layers = nn.Sequential()

        in_ch = in_channels
        for out_ch in out_channels:
            self.layers.append(EncodeBlock(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size))
            in_ch = out_ch
    
    def forward(self, x):
        """
        Applies all EncodeBlocks sequentially to the input x.

        Args:
            x (torch.Tensor): input data to this Encoder.
        
        Returns:
            torch.Tensor: result of the input data parsed by this Encoder. Return value is NOT flattened
        """
        output = self.layers(x)
        return output

class DecodeBlock(nn.Module):
    """
    Basic Decode block used in a Decoder. Block constists of a ConvTranspose2d and LeakyReLU.
    Doubles the Height and Width of the input.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size=1, stride=1, padding=1, out_padding=1):
        """
        Initialises The DecodeBlock with the given parameters.

        Args:
            in_channels (int): input channels to the ConvTranspose2d
            out_channels (int): output channels to the ConvTranspose2d
            kernel_size: kernel_size of the ConvTranspose2d
            stride: stride of the ConvTranspose2d
            padding: padding of the ConvTranspose2d
            out_padding: output_padding of the ConvTranspose2d
        """
        super(DecodeBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=out_padding),
            nn.LeakyReLU()
        )

    def forward(self, x):
        """
        Applies this block to the input x.

        Args:
            x (torch.Tensor): input data to this block
        
        Returns:
            torch.Tensor: result of the input data parsed by this block
        """
        output = self.layer(x)
        return output
    

class Decoder(nn.Module):
    """
    Decoder composed of multiple EncodeBlocks. Unflattening of inputs must be done before parsing to the Decoder.
    A final layer with a ConvTranspose2d with out_channels=1 and a Sigmoid function is appended to the end of the sequence.
    """

    def __init__(self, out_channels=[256,128,64], kernel_size=3):
        """
        Initialises The Decoder with the given parameters. 

        out_channels (list[int]): output channels of each layer. len(out_channels) is used to determine number of layers to create.
        kernel_size (int | (int,int)), default=3): kernel size for all layers. 
        """
        super(Decoder, self).__init__()

        self.layers = nn.Sequential()
        
        for i in range(len(out_channels)-1):
            self.layers.append(DecodeBlock(in_channels=out_channels[i], out_channels=out_channels[i+1], kernel_size=kernel_size, stride=2, padding=1, out_padding=1))

        self.layers.append(nn.ConvTranspose2d(in_channels=out_channels[-1], out_channels=1, kernel_size=kernel_size, stride=2, padding=1, output_padding=1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        """
        Applies all DecodeBlocks sequentially to the input x.

        Args:
            x (torch.Tensor): input data to this Decoder. Input must NOT be flattened.
        
        Returns:
            torch.Tensor: result of the input data parsed by this Decoder. Return value is in range [0,1)
        """
        return self.layers(x)

class VAE(nn.Module):
    """
    Vanilla Variational Auto-Encoder.
    Uses the Encoder, Decoder, and two linear layers for the distribution.
    """

    def __init__(self, in_channels=1, out_channels=[64,128,256], latent_dim=32, kernel_size=3, image_size=(256,128)):
        """
        Initialises the VAE and it's components.
        The Encoder and Decoder have the same number of EncodeBlocks and DecodeBlocks respectively.

        Args:
            in_channels (int): input channels to the Encoder.
            out_channels (list[int]): output channels of each layer. len(out_channels) is used to determine number of layers to create.
            latent_dim (int): Size of the latent dimension used in the reparameterisation of the VAE encoding.
            kernel_size: kernel_size for both the Encoder and Decoder.
            image_size (int,int): Height, Width dimensions of the input image.
        """
        super(VAE, self).__init__()

        self.encoder = Encoder(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

        self.encoder.layers.append(nn.Flatten())

        h, w = image_size[0] // pow(2, len(out_channels)), image_size[1] // pow(2, len(out_channels))
        self.latent_channels = out_channels[-1] * h * w

        self.fc_mu = nn.Linear(in_features=self.latent_channels, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=self.latent_channels, out_features=latent_dim)

        rev_out = out_channels
        rev_out.reverse()

        self.decoder = Decoder(out_channels=rev_out, kernel_size=kernel_size)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Performs the reparameterisation of the VAE latent space data.
        Samples from N(mu, var) using N(0,1)

        Args:
            mu (torch.Tensor): Sample means calculated from the fc_mu layer.
            logvar (torch.Tensor): Sample log variance calculated from the fc_logvar layer.

        Returns:
            torch.Tensor: During training, returns mu + torch.randn_like(std) * torch.exp(0.5 * logvar). Otherwise returns mu.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
        
    def forward(self, x: torch.Tensor):
        """
        Parses the input through this VAE, returning the reconstruction, mu, logvar.

        Args:
            x (torch.Tensor): input data to this VAE.

        Returns:
            torch.Tensor: result of the input data parsed by this VAE. Return value is in range [0,1)
        """
        output = self.encoder(x)
        mu = self.fc_mu(output)
        logvar = self.fc_logvar(output)
        z = self.reparameterize(mu,logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, predicted: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta=1.0):
        """
        VAE loss = Reconstruction loss + KL divergence
        beta = weight for KL divergence (beta-VAE)

        Args:
            predicted (torch.Tensor): Output from the VAE.
            target (torch.Tensor): True value of the input to the VAE.
            mu (torch.Tensor): Sample mean values generated by the input.
            logvar (torch.Tensor): Sample variance values generated by the input.
            beta (float): Weight for the KL divergence.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): binary_cross_entropy + beta * KL_divergence_loss, binary_cross_entropy, KL_divergence_loss
        """
        binary_cross_entropy = F.binary_cross_entropy(predicted, target, reduction='sum')

        KL_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return binary_cross_entropy + beta * KL_divergence_loss, binary_cross_entropy, KL_divergence_loss


class VectorQuantize(nn.Module):
    """
    Vector Quantisation module for use in a VQ-VAE model.
    """

    def __init__(self, num_embeds=2048, embed_dim=64):
        """
        Initialises the Vector Quantizer with the given parameters.

        Args:
            num_embeds (int): Code book size. The number of values in the embedding lookup table.
            embed_dim (int): Dimensionality of each vector in the codebook.
        """
        super(VectorQuantize, self).__init__()

        self.num_embeds = num_embeds
        self.embed_dim = embed_dim

        # initialize embedding lookup table
        self.embedding = nn.Embedding(num_embeddings=num_embeds, embedding_dim=embed_dim)
        self.embedding.weight.data.uniform_(-1/num_embeds, 1/num_embeds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parses the input through this Vector Quantiser module, returning the quantized outputs and the indices used.

        Args:
            x (torch.Tensor): input data to this Vector Quantizer.

        Returns:
            (torch.Tensor, torch.Tensor): Quantized outputs using the codebook lookup table, and the indices of each embedding used.
        
        REF: Inspired by VectorQuantizer in https://github.com/LukeDitria/pytorch_tutorials/blob/main/section07_autoencoders/solutions/Pytorch3_VQVAE.ipynb
        """

        # Permute x from (Batch_size, Channels, Height, Width) to (Batch_size, Height, Width, Channels)
        x = x.permute(0,2,3,1).contiguous()
        x_shape = x.shape

        # View x as a flattened tensor (Batch_size * Height * Width, 1, embed_dim)
        flatten = x.view(-1, 1, self.embed_dim)
        
        ## Subtraction Broadcast input vectors over embedding weights
        #   (Batch_size * Height * Width,          1, embed_dim)
        # - (                          1, num_embeds, embed_dim)
        # = (Batch_size * Height * Width, num_embeds, embed_dim)
        squared_diff = (flatten - self.embedding.weight.unsqueeze(0)).pow(2)

        ## Get vector norm of distances
        #   (Batch_size * Height * Width, num_embeds)
        distances = torch.linalg.vector_norm(squared_diff, dim=2)

        ## Find nearest codebook entries for each input vector
        #   (Batch_size * Height * Width, 1)
        nearest_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embedding(nearest_indices).view(x_shape)

        # Enable back-propagation
        if self.training:
            quantized = x + (quantized - x).detach()

        ## Return the quantised tensor with the same shape as the input, and the encoding indices
        #   (Batch_size, Channels, Height, Width), (Batch_size * Height * Width, 1)
        return quantized.permute(0, 3, 1, 2).contiguous(), nearest_indices

class VQVAE(nn.Module):
    """
    Vector Quantized Variational Auto-Encoder.
    Uses the Encoder, Decoder, and VectorQuantize modules.
    """

    def __init__(self, in_channels=1, out_channels=[64, 128, 256], latent_dim=32, kernel_size=3, num_embeds=64):
        """
        Initialises the VQVAE with the given parameters.

        Args:
            in_channels (int): input channels to the Encoder.
            out_channels (list[int]): output channels of each layer. len(out_channels) is used to determine number of layers to create.
            latent_dim (int): Dimensionality of the Vector Quantized embeddings.
            kernel_size: kernel_size for both the Encoder and Decoder.
            num_embeds (int): Code book size for the VectorQuantize module. The number of values in the embedding lookup table.
        """
        super(VQVAE, self).__init__()

        self.encoder = Encoder(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)

        self.vq = VectorQuantize(num_embeds, latent_dim)

        rev_out = out_channels
        rev_out.reverse()
        self.decoder = Decoder(out_channels=rev_out, kernel_size=kernel_size)

    def loss_function(self, predicted, target, commit_loss=1.0):
        """
        Computes both the encoding loss and the quantize loss.

        Args:
            predicted (torch.Tensor): Output of the VQVAE.
            target (torch.Tensor): True value of the input to the VQVAE.
            commit_loss (float): coefficient of the encode loss.

        Returns:
            torch.Tensor: encode_loss * commit_loss + quantize_loss
        """
        encode_loss = F.mse_loss(predicted.detach(), target)
        quantize_loss = F.mse_loss(predicted, target.detach())
        loss = encode_loss * commit_loss + quantize_loss
        return loss

    def forward(self, x):
        """
        Parses the input through this VQ-VAE, returning the reconstruction, quantized, and index Tensors.

        Args:
            x (torch.Tensor): input data to this VQ-VAE.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Reconstructed_x, quantized_x, indices_x
        """
        # Encode
        encoding = self.encoder(x)

        # Vector Quantize
        quantized, indices = self.vq(encoding)

        # Decode
        decoded = self.decoder(quantized)

        return decoded, quantized, indices

# Testing
if __name__ == "__main__":

    test_data = torch.randn((4,1,256,128))

    vqvae = VQVAE()
    #print(vqvae)
    print(test_data.shape)
    out = vqvae(test_data)
    print(out.shape)

