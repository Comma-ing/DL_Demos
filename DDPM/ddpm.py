import torch
import torch.nn as nn

class DDPM():
    def __init__(self, device, n_steps: int, min_beta: float = 0.0001, max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        temp = 1
        for i, alpha in enumerate(alphas):
            temp *= alpha
            alpha_bars[i] = temp
        
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def sample_forward(self, x, t, eps=None):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps == None:
            eps = torch.randn(x.shape)
        x_t = torch.sqrt(alpha_bar) * x + eps * torch.sqrt(1 - alpha_bar)
        return x_t

    def sample_backward(self, img_shape, net, device, simple_var=True):
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps-1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var)
        return x
    
    def sample_backward_step(self, x, t, net, simple_var=True):
        n = x.shape[0]
        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(x.device).unsqueeze(1)
        eps = net(x, t_tensor)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t-1]) / (1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x) * torch.sqrt(var)

        mean = (x - (1 - self.alphas[t]) / torch.sqrt(1- self.alpha_bars[t]) * eps) / torch.sqrt(self.alphas[t])
        x_t = mean + noise
        return x_t 