import torch
import torch.nn as nn

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

    def __init__(self, in_channels=1, out_channels=[64,128,256], kernel_size=3):
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
    
    def forward(self, input):
        '''
        
        '''
        return self.layers(input)


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
            nn.LeakyReLU()
        )

        for i in range(len(out_channels)-1):
            self.layers.append(DecodeBlock(in_channels=out_channels[i], out_channels=out_channels[i+1], kernel_size=kernel_size))

        self.layers.append(nn.ConvTranspose2d(in_channels=out_channels[-1], out_channels=1, kernel_size=kernel_size))
        self.layers.append(nn.Sigmoid())

    def forward(self, input):
        '''
        
        '''
        return self.layers(input)

class VectorQuantizer(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)



if __name__ == "__main__":
    eb = EncodeBlock(1, 256)
    print(eb)

    ve = Encoder()
    print(ve)

    db = DecodeBlock(256, 128)
    print(db)

    vd = Decoder(64)
    print(vd)