import torch
from torch import nn
from torch.nn import functional as F
from typing import List


class VAE(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 output_dim: int) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder_mu = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.encoder_log_sigma_sq = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def encode(self, input: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        mu = self.encoder_mu(input)
        log_sigma_sq = self.encoder_log_sigma_sq(input)
        return mu, log_sigma_sq

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, None

    def loss_function(self, logits: torch.Tensor, target: torch.Tensor,
                      mu: torch.Tensor, log_var: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        class_loss = F.cross_entropy(logits, target)
        kld_loss = torch.mean(-0.5 * torch.sum(log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = class_loss + 0.001*kld_loss
        return loss, class_loss, kld_loss

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def inference(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        mu, log_var = self.encode(x)
        return self.decode(mu), mu, log_var
