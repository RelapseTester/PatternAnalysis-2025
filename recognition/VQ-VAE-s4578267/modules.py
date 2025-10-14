import torch
import torch.nn as nn
import torch.nn.functional as F

class EncodeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1):
        super(EncodeBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # halve height and width
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class Encoder(nn.Module):

    def __init__(self, in_channels=1, out_channels=[64,128,256], latent_dim=64, image_size=(256,256), kernel_size=3):
        '''
        Encoder component of a VAE.

        in_channels (int, default=1): initial input channels to encoder.
        out_channels (list[int], default=[256,128,64]): output channels of each layer. len(out_channels) is used to determine number of layers to create.
        kernel_size (int | (int,int)), default=3): kernel size for all layers. 
        '''
        super(Encoder, self).__init__()

        self.layers = nn.Sequential()

        in_ch = in_channels
        for out_ch in out_channels:
            self.layers.append(EncodeBlock(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size))
            in_ch = out_ch
    
    def forward(self, x):
        '''
        '''
        output = self.layers(x)
        return output

class DecodeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, out_padding=1):
        super(DecodeBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=out_padding),
            nn.LeakyReLU()
        )

    def forward(self, x):
        out = self.layer(x)
        return out
    

class Decoder(nn.Module):

    def __init__(self, out_channels=[256,128,64], latent_dim=64, image_size=(256,256), kernel_size=3):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential()
        
        for i in range(len(out_channels)-1):
            self.layers.append(DecodeBlock(in_channels=out_channels[i], out_channels=out_channels[i+1], kernel_size=kernel_size, stride=2, padding=1, out_padding=1))

        self.layers.append(nn.ConvTranspose2d(in_channels=out_channels[-1], out_channels=1, kernel_size=kernel_size, stride=2, padding=1, output_padding=1))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        '''
        
        '''
        return self.layers(x)

class VAE(nn.Module):

    def __init__(self, in_channels=1, out_channels=[64,128,256], latent_dim=32, kernel_size=3, image_size=(256,128)):
        super(VAE, self).__init__()

        self.encoder = Encoder(in_channels=in_channels, out_channels=out_channels, latent_dim=latent_dim, kernel_size=kernel_size, image_size=image_size)

        self.encoder.layers.append(nn.Flatten())

        h, w = image_size[0] // pow(2, len(out_channels)), image_size[1] // pow(2, len(out_channels))
        self.latent_channels = out_channels[-1] * h * w

        self.fc_mu = nn.Linear(in_features=self.latent_channels, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=self.latent_channels, out_features=latent_dim)

        rev_out = out_channels
        rev_out.reverse()

        self.decoder = Decoder(latent_dim=latent_dim, out_channels=rev_out, kernel_size=kernel_size)

    def reparameterize(self, mu, logvar):
        """
        Sample from N(mu, var) using N(0,1)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
        
    def forward(self, x):
        output = self.encoder(x)
        mu = self.fc_mu(output)
        logvar = self.fc_logvar(output)
        z = self.reparameterize(mu,logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, reconstructed_x, x, mu, logvar, beta=1.0):
        """
        VAE loss = Reconstruction loss + KL divergence
        beta = weight for KL divergence (beta-VAE)
        """
        binary_cross_entropy = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')

        KL_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return binary_cross_entropy + beta * KL_divergence_loss, binary_cross_entropy, KL_divergence_loss


class VectorQuantize(nn.Module):

    def __init__(self, num_embeds=2048, embed_dim=64, commit_cost=1.0):
        super(VectorQuantize, self).__init__()

        self.num_embeds = num_embeds
        self.embed_dim = embed_dim
        self.commit_cost = commit_cost

        # initialize embedding lookup table
        self.embedding = nn.Embedding(num_embeddings=num_embeds, embedding_dim=embed_dim)
        self.embedding.weight.data.uniform_(-1/num_embeds, 1/num_embeds)

    def forward(self, x):
        """
        WIP
        """

        # Reshape x for distance calculations
        x = x.permute(0,2,3,1).contiguous()
        x_shape = x.shape
        flatten = x.view(-1, 1, self.embed_dim)

        # Calculate distances from inputs to embeddings
        distances = (flatten - self.embedding.weight.unsqueeze(0)).pow(2).mean(2)

        # Find nearest codebook entries
        nearest_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embedding(nearest_indices).view(x_shape)

        # Enable back-propagation
        if self.training:
            quantized = x + (quantized - x).detach()

        # Return in original shape
        return quantized.permute(0, 3, 1, 2).contiguous()


class VQVAE(nn.Module):

    def __init__(self, in_channels=1, out_channels=[64, 128, 256], latent_dim=32, kernel_size=3, image_size=(256, 128), num_embeds=64):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(in_channels=in_channels, out_channels=out_channels, latent_dim=latent_dim, kernel_size=kernel_size, image_size=image_size)

        self.vq = VectorQuantize(num_embeds, latent_dim)

        rev_out = out_channels
        rev_out.reverse()
        self.decoder = Decoder(out_channels=rev_out, latent_dim=latent_dim, image_size=image_size, kernel_size=kernel_size)

    def loss_function(self, predicted, target, commit_loss=1.0):
        """
        
        """
        encode_loss = F.mse_loss(predicted.detach(), target)
        quantize_loss = F.mse_loss(predicted, target.detach())
        loss = encode_loss * commit_loss + quantize_loss
        return loss

    def forward(self, x):

        # Encode
        encoding = self.encoder(x)

        # Vector Quantize
        quantized = self.vq(encoding)

        # Decode
        decoded = self.decoder(quantized)

        return decoded

# Testing
if __name__ == "__main__":

    test_data = torch.randn((4,1,256,128))

    vqvae = VQVAE()
    print(vqvae)
    print(test_data.shape)
    out = vqvae(test_data)
    print(out.shape)