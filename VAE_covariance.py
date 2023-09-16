import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from utils import create_cholesky_matrix, Squeeze


class VAE_covariance(nn.Module):
    def __init__(self,
                 in_dim: (int, int),
                 hidden_dim: int,
                 latent_dim: int,
                 output_dim: int) -> None:
        super(VAE_covariance, self).__init__()

        self.latent_dim = latent_dim
        self.mask_matrix = torch. torch.ones((latent_dim, latent_dim))

        self.encoder_mu = nn.Sequential(
            nn.Conv2d(1,hidden_dim, 3),
            nn.ReLU(),
            nn.Conv2d(hidden_dim,hidden_dim, 3),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 2),
            nn.ReLU(),
            nn.AvgPool2d((in_dim[0] - 5, in_dim[1] - 5)),
            Squeeze(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.encoder_sigma_h = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 2),
            nn.ReLU(),
            nn.AvgPool2d((in_dim[0] - 5, in_dim[1] - 5)),
            Squeeze()
        )

        self.encoder_log_sigma_sq = nn.Linear(hidden_dim, latent_dim)
        n_off_diagonal = int((latent_dim ** 2 - latent_dim) / 2)
        self.encoder_upper_L = nn.Linear(hidden_dim,  n_off_diagonal)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def encode(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        mu = self.encoder_mu(input)
        sigma_h = self.encoder_sigma_h(input)
        log_sigma_sq = self.encoder_log_sigma_sq(sigma_h)
        upper_L = self.encoder_upper_L(sigma_h)
        return mu, log_sigma_sq, upper_L

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, upper_L: torch.Tensor) -> torch.Tensor:
        L = create_cholesky_matrix(logvar, upper_L)
        eps = torch.randn_like(mu)
        return torch.squeeze(torch.bmm(L, torch.unsqueeze(eps, dim=-1))) + mu

    def forward(self, input: torch.Tensor, **kwargs) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        if len(input.shape) == 3:
            torch.unsqueeze(input, dim=0) # force batch_size of 1
        mu, log_var, upper_L = self.encode(input)
        z = self.reparameterize(mu, log_var, upper_L)
        return self.decode(z), mu, log_var, upper_L

    def loss_function(self, logits: torch.Tensor, target: torch.Tensor,
                      mu: torch.Tensor, log_var: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        class_loss = F.cross_entropy(logits, target)
        kld_loss = torch.mean(-0.5 * torch.sum(log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = class_loss + 0.001*kld_loss
        return loss, class_loss, kld_loss

    def sample(self,
               num_samples:int,
               current_device: int) -> torch.Tensor:
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def inference(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=0)
        mu, log_var, upper_L = self.encode(x)
        return self.decode(mu), mu, log_var, upper_L
