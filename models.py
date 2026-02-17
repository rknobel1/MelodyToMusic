import torch
import torch.nn as nn


class LSTMGenerator(nn.Module):
    def __init__(self, z_dim=100, hidden_size=256, num_layers=3, seq_len=64, feature_dim=512):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fc = nn.Linear(z_dim, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.output = nn.Linear(hidden_size, feature_dim)
        self.tanh = nn.Tanh()

    def forward(self, z):
        b = z.size(0)

        # per-step noise + global style
        x = torch.randn(b, self.seq_len, self.hidden_size, device=z.device)
        style = self.fc(z).view(b, 1, self.hidden_size)
        x = x + style

        out, _ = self.lstm(x)
        out = self.output(out)
        return self.tanh(out)  # [b, T, 128*C]


# Patch GAN
class Conv2DDiscriminator(nn.Module):
    def __init__(self, in_channels=4, use_mbstd=True):
        super().__init__()
        self.use_mbstd = use_mbstd

        def snconv(in_c, out_c, k=4, s=2, p=1):
            return nn.utils.spectral_norm(
                nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
            )

        c_in = in_channels + (1 if use_mbstd else 0)

        self.backbone = nn.Sequential(
            snconv(c_in, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            snconv(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            snconv(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            snconv(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Patch head: outputs a map of logits [B,1,H,W]
        self.head = nn.utils.spectral_norm(nn.Conv2d(512, 1, kernel_size=1))

    @staticmethod
    def minibatch_stddev(x):
        # x: [B,C,H,W]
        std = torch.std(x, dim=0, unbiased=False).mean()
        std_feat = std.expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std_feat], dim=1)

    def forward(self, x):
        # x: [B,T,128,C] -> [B,C,128,T]
        x = x.permute(0, 3, 2, 1).contiguous()

        if self.use_mbstd:
            x = self.minibatch_stddev(x)

        h = self.backbone(x)
        logits_map = self.head(h)                 # [B,1,H,W]
        return logits_map.mean(dim=(2, 3))        # [B,1]


def build_gan(
    noise_dim=100,
    lr_G=1e-4,
    lr_D=1e-4,
    betas=(0.0, 0.9),
    device="cuda",
    T=64,
    C=4,
    hidden_size=256,
    num_layers=3,
):
    feature_dim = 128 * C
    generator = LSTMGenerator(
        z_dim=noise_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        seq_len=T,
        feature_dim=feature_dim
    ).to(device)
    discriminator = Conv2DDiscriminator(in_channels=C).to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=betas)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=betas)

    return {"G": generator, "D": discriminator, "opt_G": optimizer_g, "opt_D": optimizer_d, "device": device}
