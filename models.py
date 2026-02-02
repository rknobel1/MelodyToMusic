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


class LSTMGenerator(nn.Module):
    def __init__(
        self,
        z_dim=100,
        hidden_size=256,
        num_layers=2,
        seq_len=100,
        feature_dim=4
    ):
        super().__init__()

        self.seq_len = seq_len

        # Project noise into initial hidden space
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
        # z: (batch, z_dim)
        batch_size = z.size(0)

        # Create repeated input sequence
        h0 = torch.tanh(self.fc(z))           # (batch, hidden)
        lstm_input = h0.unsqueeze(1).repeat(1, self.seq_len, 1)

        out, _ = self.lstm(lstm_input)
        out = self.output(out)

        return self.tanh(out)  # (batch, 100, 4)


class RNNDiscriminator(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, 4)
        out, (h_n, _) = self.rnn(x)

        # Take last hidden state from both directions
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)

        return self.sigmoid(self.fc(h_last))


def build_gan(
    noise_dim=100,
    lr=1e-4,
    betas=(0.5, 0.999),
    device="cuda"
):
    generator = LSTMGenerator(noise_dim).to(device)
    discriminator = RNNDiscriminator().to(device)

    criterion = nn.BCEWithLogitsLoss()

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
