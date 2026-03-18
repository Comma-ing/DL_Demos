import torch, sys, os
import torch.nn as nn
from ddim import DDIM

sys.path.append(os.path.dirname("__file__"))
from DDPM.dataset import get_img_shape
from DDPM.unet import (build_network, convnet_big_cfg,
                                  convnet_medium_cfg, convnet_small_cfg,
                                  unet_1_cfg, unet_res_cfg)
import cv2
import numpy as np
import einops

batch_size = 512
n_epochs = 100

def sample_imgs(ddim,
                net,
                output_path,
                n_sample=81,
                device='cuda',
                simple_var=False):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        shape = (n_sample, *get_img_shape())  # 1, 3, 28, 28
        imgs = ddim.sample_backward(shape,
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
    ddim = DDIM(n_steps, device)

    net.load_state_dict(torch.load(model_path))
    sample_imgs(ddim, net, '/home/siat502/siat_cxd/DL_Demos/DDIM/diffusion.jpg', device=device)
