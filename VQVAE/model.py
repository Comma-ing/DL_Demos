import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, inputs):
        x = self.relu(inputs)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + inputs

class VQVAE(nn.Module):
    def __init__(self, input_dim, dim, n_embedding):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.ReLU(), nn.Conv2d(dim, dim, 4, 2, 1),
            nn.ReLU(), nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim), ResidualBlock(dim)
        )

        self.vq_embedding = nn.Embedding(n_embedding, dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim), ResidualBlock(dim),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1)
        )
        self.n_downsample = 2

    def forward(self, x):
        z_e = self.encoder(x)

        embedding = self.vq_embedding.weight.data
        N, C, H, W = z_e.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = z_e.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
        nearest_neighbor = torch.argmin(distance, 1)

        z_q = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
        decoder_input = z_e + (z_q - z_e).detach()

        x_hat = self.decoder(decoder_input)
        return x_hat, z_e, z_q