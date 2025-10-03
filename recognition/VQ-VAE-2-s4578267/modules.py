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

    def forward(self, input):
        return self.layer(input)

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

        self.layers.append(nn.Flatten())

        h, w = image_size[0] // pow(2, len(out_channels)), image_size[1] // pow(2, len(out_channels))

        self.fc_mu = nn.Linear(in_features=in_ch * h * w, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=in_ch * h * w, out_features=latent_dim)
    
    def forward(self, input):
        '''
        '''
        output = self.layers(input)
        print(output.shape)
        mu = self.fc_mu(output)
        logvar = self.fc_logvar(output)
        return mu, logvar

class DecodeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1):
        super(DecodeBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU()
        )

    def forward(self, input):
        return self.layer(input)
    

class Decoder(nn.Module):

    def __init__(self, latent_dim, out_channels=[128,64], feature_map=(64,64), kernel_size=3):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=out_channels[0] * feature_map[0] * feature_map[1]),
            nn.LeakyReLU(),
            nn.Unflatten(1, (out_channels[0], feature_map[0], feature_map[1]))
        )

        for i in range(len(out_channels)-1):
            self.layers.append(DecodeBlock(in_channels=out_channels[i], out_channels=out_channels[i+1], kernel_size=kernel_size))

        self.layers.append(nn.ConvTranspose2d(in_channels=out_channels[-1], out_channels=1, kernel_size=kernel_size))
        self.layers.append(nn.Sigmoid())

    def forward(self, input):
        '''
        
        '''
        return self.layers(input)

class VAE(nn.Module):

    def __init__(self, in_channels=1, out_channels=[64,128,256], latent_dim=32, kernel_size=3):
        super(VAE, self).__init__()

        self.encoder = Encoder(in_channels=in_channels, out_channels=out_channels, latent_dim=latent_dim, kernel_size=kernel_size)

        rev_out = out_channels
        rev_out.reverse()

        self.decoder = Decoder(latent_dim=latent_dim, out_channels=rev_out[1:], feature_map=(latent_dim,latent_dim), kernel_size=kernel_size)

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
        mu, logvar = self.encoder(x)
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


class VectorQuantizer(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)



if __name__ == "__main__":
    #eb = EncodeBlock(1, 256)
    #print(eb)
    #ve = Encoder()
    #print(ve)
    #db = DecodeBlock(256, 128)
    #print(db)
    #vd = Decoder(64)
    #print(vd)

    #test_data = torch.randn((1,1,256,256))

    #print(test_data.shape)

    vae = VAE()
    print(vae)

    #vae(test_data)
    #print(test_data.shape)