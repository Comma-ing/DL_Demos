import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, pre_channel=3, img_length=64, hiddens=[16, 32, 64, 128, 256], latend_dim=128):
        super().__init__()

        modules = []
        for cur_channel in hiddens:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(pre_channel, cur_channel, 3, 2, 1),
                    nn.BatchNorm2d(cur_channel),
                    nn.ReLU()
                )
            )
            pre_channel = cur_channel
            img_length //= 2
        
        self.encoder = nn.Sequential(*modules)
        self.mean_linear = nn.Linear(pre_channel * img_length * img_length, latend_dim)
        self.var_linear = nn.Linear(pre_channel * img_length * img_length, latend_dim)
        self.latent_dim = latend_dim

        modules = []
        self.decoder_projection = nn.Linear(
            latend_dim, pre_channel * img_length * img_length
        )

        self.decoder_input_chw = (pre_channel, img_length, img_length)
        for i in range(len(hiddens)-1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i], hiddens[i-1], 3, 2, 1, 1),
                    nn.BatchNorm2d(hiddens[i-1]),
                    nn.ReLU()
                )
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hiddens[0], hiddens[0], 3, 2, 1, 1),
                nn.BatchNorm2d(hiddens[0]),
                nn.ReLU(),
                nn.Conv2d(hiddens[0], 3, 3, 2, 1),
                nn.BatchNorm2d(3),
                nn.ReLU()
                )
        )
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, 1)
        mean = self.mean_linear(encoded)
        logvar = self.var_linear(encoded)
        eps = torch.rand_like(logvar)
        std = torch.exp(logvar / 2)
        z = mean + eps * std
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)
        return decoded, mean, logvar

    def sample(self, n_sample, device="cuda"):
        z = torch.randn(n_sample, self.latent_dim).to(device)
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoder_input_chw))
        decoded = self.decoder(x)
        return decoded
