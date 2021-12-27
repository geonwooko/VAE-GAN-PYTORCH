import torch
import torch.nn as nn



class VAE_GAN(nn.Module):
    def __init__(self, hyperpm):
        super(VAE_GAN, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()
        self.init_parameters()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    #init as original implementation
                    nn.init.xavier_normal_(m.weight)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    m.bias.data = torch.zeros_like(m.bias.data)

    def sample_with_reparmeterization(self, mean, logvar):
        std = torch.sqrt(torch.exp(logvar))
        gaussian_sample = torch.randn(std.shape).to(self.device)
        return mean + std * gaussian_sample
        
    def forward(self, X):
        mean, logvar = self.encoder(X)

        Z = self.sample_with_reparmeterization(mean, logvar)
        Z_prior = torch.randn(mean.shape).to(self.device) # sample prior ~ N(0,1)
        X_recon = self.decoder(Z)
        X_prior = self.decoder(Z_prior)

        disc_X_real = self.discriminator(X)
        disc_X_prior = self.discriminator(X_prior)

        sim_X_real = self.discriminator.similiarity_X(X)
        sim_X_recon = self.discriminator.similiarity_X(X_recon)

        return mean, logvar, X_recon, X_prior, disc_X_real, disc_X_prior, sim_X_real, sim_X_recon

class Encoder(nn.Module):
    def __init__(self, channel_in = 3, z_dim= 128):
        super(Encoder, self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9)

        self.conv2 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9)

        self.conv3 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(256, momentum=0.9)

        self.flatten = nn.Flatten(start_dim=1)
        self.mlp_mean = nn.Linear(256 * 16 * 16, 2048)
        self.bn4_mean = nn.BatchNorm1d(2048, momentum=0.9)


        self.mlp_logvar = nn.Linear(256 * 16 * 16, 2048)
        self.bn4_logvar = nn.BatchNorm1d(2048, momentum=0.9)

        self.norm_mean = nn.Linear(2048, z_dim)
        self.norm_logvar = nn.Linear(2048, z_dim)


    def forward(self, X):
        out = self.relu(self.bn1(self.conv1(X)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        mean = self.norm_mean(self.relu(self.bn4_mean(self.mlp_mean(self.flatten(out)))))
        logvar = self.norm_logvar(self.relu(self.bn4_logvar(self.mlp_logvar(self.flatten(out)))))
        # mean = self.norm_mean(out)
        # logvar = self.norm_logvar(out)

        return mean, logvar


class Decoder(nn.Module): # use normal distribution
    def __init__(self, channel_in = 3, z_dim = 128):
        super(Decoder, self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.upsample_mlp = nn.Linear(z_dim, 8 * 8 * 256)
        self.mlp_bn = nn.BatchNorm1d(8 * 8 * 256, momentum=0.9)

        self.deconv1 = nn.ConvTranspose2d(256, 256, 5, padding=2, stride=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(256, momentum=0.9)

        self.deconv2 = nn.ConvTranspose2d(256, 128, 5, padding=2, stride=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9)

        self.deconv3 = nn.ConvTranspose2d(128, 32, 5, padding=2, stride=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(32, momentum=0.9)

        self.conv = nn.ConvTranspose2d(32, 3, 5, padding=2, stride=2, output_padding=1)
        self.tanh = nn.Tanh()

    def forward(self, Z):
        out = self.relu(self.mlp_bn(self.upsample_mlp(Z))).reshape(Z.shape[0], 256, 8 ,8)
        out = self.relu(self.bn1(self.deconv1(out)))
        out = self.relu(self.bn2(self.deconv2(out)))
        out = self.relu(self.bn3(self.deconv3(out)))
        out = self.tanh(self.conv(out))

        return out

class Discriminator(nn.Module):
    def __init__(self, channel_in = 3):
        super(Discriminator, self).__init__()
        self.relu = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv2d(channel_in, 32, 5, padding=2, stride=2)

        self.conv2 = nn.Conv2d(32, 128, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9)

        self.conv3 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(256, momentum=0.9)

        self.conv4 = nn.Conv2d(256, 256, 5, padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(256, momentum=0.9)

        self.flatten = nn.Flatten(start_dim=1)
        self.mlp1 = nn.Linear(256 * 8 * 8, 512)
        self.mlp_bn = nn.BatchNorm1d(512, momentum=0.9)

        self.mlp2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):

        # Discriminate X as a probability
        out = self.relu(self.conv1(X))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        out = self.relu(self.mlp_bn(self.mlp1(self.flatten(out))))
        disc_prob = self.sigmoid(self.mlp2(out))

        return disc_prob

    def similiarity_X(self, X):

        # Make X as a similarity tensor
        out = self.relu(self.conv1(X))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))

        return out


