import torch
import torch.nn as nn
from dataset import get_dataloader, get_img_shape
from ddpm import DDPM
from unet import (build_network, convnet_big_cfg,
                                  convnet_medium_cfg, convnet_small_cfg,
                                  unet_1_cfg, unet_res_cfg)
import cv2
import numpy as np
import einops

batch_size = 512
n_epochs = 100


def train(ddpm: DDPM, net, device, ckpt_pth):
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(batch_size)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    for e in range(n_epochs):
        for x, _ in dataloader:
            cur_batch = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, n_steps, (cur_batch, )).to(device)
            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x, t, eps)
            eps_theta = net(x_t, t)
            loss = loss_fn(eps, eps_theta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch: {e+1}")
    torch.save(net.state_dict(), ckpt_pth)

def sample_imgs(ddpm,
                net,
                output_path,
                n_sample=81,
                device='cuda',
                simple_var=True):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        shape = (n_sample, *get_img_shape())  # 1, 3, 28, 28
        imgs = ddpm.sample_backward(shape,
                                    net,
                                    device=device,
                                    simple_var=simple_var).detach().cpu()
        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)
        imgs = einops.rearrange(imgs,
                                '(b1 b2) c h w -> (b1 h) (b2 w) c',
                                b1=int(n_sample**0.5))

        imgs = imgs.numpy().astype(np.uint8)

        cv2.imwrite(output_path, imgs)

configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg
]

if __name__ == "__main__":
    n_steps = 1000
    config_id = 4
    device = "cuda:0"
    model_path = "/home/siat502/siat_cxd/DL_Demos/DDPM/model.pth"

    config = config = configs[config_id]
    net = build_network(config, n_steps)
    ddpm = DDPM(device, n_steps)

    # train(ddpm, net, device, path)


    net.load_state_dict(torch.load(model_path))
    sample_imgs(ddpm, net, '/home/siat502/siat_cxd/DL_Demos/DDPM/diffusion.jpg', device=device)
