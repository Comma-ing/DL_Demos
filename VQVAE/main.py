import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, ToTensor
import time, cv2, einops

from model import VQVAE

def get_dataloader(batch_size: int):
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = torchvision.datasets.MNIST(root='/home/siat502/siat_cxd/data/mnist',
                                         transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train(
    model: VQVAE,
    img_shape=None,
    device="cuda",
    ckpt_path="model.pth",
    batch_size=64,
    lr=1e-3,
    n_epochs=100,
    l_w_embedding=1,
    l_w_commitment=0.25):

    dataloader = get_dataloader(batch_size)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    mse_loss = nn.MSELoss()
    start = time.time()

    for e in range(n_epochs):
        total_loss = 0
        for x, _ in dataloader:
            cur_batch = x.shape[0]
            x = x.to(device)

            x_hat, ze, zq = model(x)
            loss_reconstruct = mse_loss(x, x_hat)
            loss_embedding = mse_loss(ze.detach(), zq)
            loss_commitment = mse_loss(ze, zq.detach())
            loss = loss_reconstruct + l_w_embedding * l_w_embedding + loss_commitment * l_w_commitment

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * cur_batch
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - start):.2f}s')
    print('Done')

def reconstruct(model, x, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x_hat, _, _ = model(x)
    
    n = x.shape[0]
    n1 = int(n**0.5)
    x_cat = torch.concat((x, x_hat), 3)
    x_cat = einops.rearrange(x_cat, '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=n1)
    x_cat = (x_cat.clip(0, 1) * 255).cpu().numpy().astype(np.uint8)
    cv2.imwrite(f'reconstruct.jpg', x_cat)

if __name__ == "__main__":    
    vqvae = VQVAE(1, 32, 32)
    train(vqvae)

    vqvae.load_state_dict(torch.load("model.pth"))
    dataloader = get_dataloader(64)
    img = next(iter(dataloader)).to(device)
    reconstruct(vqvae, img, "cuda")