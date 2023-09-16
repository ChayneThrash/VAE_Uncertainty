import torch
import numpy as np


def create_cholesky_matrix(log_sigma_sq, upper_L):
    if len(log_sigma_sq.shape) == 1:
        log_sigma_sq = torch.unsqueeze(log_sigma_sq, dim=0)
        upper_L = torch.unsqueeze(upper_L, dim=0)
    latent_dim = log_sigma_sq.shape[-1]
    sigma = torch.exp(0.5 * log_sigma_sq)
    L = torch.diag_embed(sigma, offset=0, dim1=-2, dim2=-1)
    triu_indices = torch.triu_indices(latent_dim, latent_dim, offset=1)
    L[:, triu_indices[0], triu_indices[1]] = upper_L
    return L


def aggregate_variance(method, var):
    if method == 'max':
        return var.max()
    elif method == "mult":
        return np.prod(var.numpy(), axis=0)
    elif method == 'norm':
        return np.linalg.norm(var.numpy())
    elif method == 'sum':
        return np.sum(var.numpy())
    elif method == 'entropy':
        var_normalized = var / torch.sum(var)
        return entropy(var_normalized)
    elif method == 'neg_entropy':
        var_normalized = var / torch.sum(var)
        return -entropy(var_normalized)
    else:
        raise Exception("Invalid method: " + method)


def get_var_logits_entropy(model, device, x):
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        logits, mu, log_var, upper_L = model.inference(x)
        mu = mu.view(-1)
        L = create_cholesky_matrix(log_var, upper_L)
        var, directions = torch.linalg.eig(torch.bmm(L, torch.permute(L, (0, 2, 1))))
        var = torch.squeeze(var).real
        directions = torch.squeeze(directions).real
        normalized_var = var / torch.sum(var)
    model.zero_grad()
    mu.requires_grad = True
    logits = model.decode(mu).view(-1)
    grads = torch.zeros((10, 8))
    for i in range(0, 10):
        grads[i] = torch.autograd.grad(logits[i], mu, retain_graph=True)[0]

    norm_grads = grads / torch.unsqueeze(torch.linalg.norm(grads, dim=1), dim=1)
    grad_sim = torch.matmul(norm_grads, directions)
    weighted_grad_sim = grad_sim * torch.unsqueeze(normalized_var, dim=0)
    agg_grad_sim = torch.sum(weighted_grad_sim, dim=1)
    min_grad_sim = torch.min(agg_grad_sim)
    grad_sim_distribution = (agg_grad_sim - min_grad_sim) / torch.sum(agg_grad_sim - min_grad_sim)
    model.zero_grad()
    return entropy(grad_sim_distribution)


def entropy(x):
    return -torch.sum(x * torch.log(x + 1e-8))


class Squeeze(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
