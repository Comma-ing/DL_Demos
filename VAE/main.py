from time import time

import torch
import torch.nn.functional as F

from load_celebA import get_dataloader
from model import VAE

# Hyperparameters
n_epochs = 10
kl_weight = 0.00025
lr = 0.005

def loss_fn(y, y_hat, mean, logvar):
    recon_loss = F.mse_loss(y, y_hat)
    kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)
    loss = recon_loss + kl_loss * kl_weight
    return loss

def train(device, dataloader, model, cpkt_pth):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    dataset_len = len(dataloader.dataset)
    model = model.to(device)

    begin_time = time()
    for i in range(n_epochs):
        loss_sum = 0
        for x in dataloader:
            x = x[0].to(device)
            y_hat, mean, logvar = model(x)
            loss = loss_fn(x, y_hat, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss
        loss_sum /= dataset_len
        training_time = time() - begin_time
        minute = int(training_time // 60)
        second = int(training_time % 60)
        print(f"epoch {i+1}: loss {loss_sum} {minute}:{second}")
        torch.save(model.state_dict(), cpkt_pth)

if __name__ == "__main__":

    device = "cuda:0"
    dataloader = get_dataloader(64)
    model = VAE()
    cpkt_pth = "model.pth"

    train(device, dataloader, model, cpkt_pth)

    def generate(device, model):
        model.eval()
        output = model.sample(device)
        output = output[0].detach().cpu()
        img = ToPILImage()(output)
        img.save('tmp.jpg')

    generate(device, model)
