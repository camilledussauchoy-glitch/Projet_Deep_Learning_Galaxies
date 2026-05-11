import torch
import torch.nn as nn
import torch.nn.functional as F


class firstVAE(nn.Module):

    def __init__(self, latent_dim=64, img_size=128):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.feature_size = 128 * 16 * 16

        self.fc_mean = nn.Linear(self.feature_size, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_size, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.feature_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 3,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Sigmoid()
        )

    def encode(self, x):

        h = self.encoder(x)

        h = h.view(h.size(0), -1)

        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)

        return mean, logvar

    def reparameterization(self, mean, logvar):

        epsilon = torch.randn_like(logvar)

        z = mean + torch.exp(0.5 * logvar) * epsilon

        return z

    def decode(self, z):

        x = self.decoder_input(z)

        x = x.view(-1, 128, 16, 16)

        x_hat = self.decoder(x)

        return x_hat

    def forward(self, x):

        mean, logvar = self.encode(x)

        z = self.reparameterization(mean, logvar)

        x_hat = self.decode(z)

        return x_hat, mean, logvar
    
    def sample(self, num_samples, device):

        with torch.no_grad():

            z = torch.randn(num_samples, self.latent_dim).to(device)

            samples = self.decode(z)

        return samples

    def reconstruct(self, x):

        with torch.no_grad():

            mean, logvar = self.encode(x)

            x_hat = self.decode(mean)

        return x_hat