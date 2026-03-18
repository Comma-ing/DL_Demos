import torch, sys, os
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname("__file__"))
from DDPM.ddpm import DDPM

class DDIM(DDPM):
    def __init__(self, n_steps: int, 
                 device, 
                 min_beta: float = 0.0001, 
                 max_beta: float = 0.02):
        super().__init__(device, n_steps, min_beta, max_beta)

    def sample_backward(self, img_or_shape, net, device, simple_var=True, ddim_step=20, eta=1):
        if simple_var:
            eta = 1
        ts = torch.linspace(self.n_steps, 0, (ddim_step + 1)).to(device).to(torch.long)
        if isinstance(img_or_shape, torch.Tensor):
            x = img_or_shape
        else:
            x = torch.randn(img_or_shape).to(device)

        batch_size = x.shape[0]
        for i in tqdm(range(1, ddim_step + 1), f'DDIM sampling with eta {eta} simple_var {simple_var}'):
            cur_t = ts[i - 1] - 1
            prev_t = ts[i] - 1

            cur_ab = self.alpha_bars[cur_t]
            prev_ab = self.alpha_bars[prev_t] if prev_t >= 0 else 1

            t_tensor = torch.tensor([cur_t] * batch_size, dtype=torch.long).to(device).unsqueeze(1)
            eps = net(x, t_tensor)
            var = eta * (1 - prev_ab) / (1 - cur_ab) * (1 - cur_ab / prev_ab)
            noise = torch.randn_like(x)

            first_term = torch.sqrt(prev_ab / cur_ab) * x
            second_term = (torch.sqrt(1 - prev_ab - var) - torch.sqrt(prev_ab * (1 - cur_ab) / cur_ab)) * eps
            if simple_var:
                third_norm = torch.sqrt(1 - cur_ab / prev_ab) * noise
            else:
                third_norm = torch.sqrt(var) * noise
            x = first_term + second_term + third_norm
        return x