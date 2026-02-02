import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128*4*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128*4*4),
            nn.Unflatten(1, (128, 4, 4)),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=4, padding=1),
            nn.Conv2d(64, 128, (2, 4), stride=4, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def build_gan(
    noise_dim=100,
    lr=1e-4,
    betas=(0.5, 0.999),
    device="gpu"
):
    generator = Generator(noise_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()

    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=lr, betas=betas
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=betas
    )

    return {
        "G": generator,
        "D": discriminator,
        "opt_G": optimizer_g,
        "opt_D": optimizer_d,
        "criterion": criterion,
        "device": device
    }
