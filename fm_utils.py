import torch


def compute_psi(x0, x1, t):
    """Compute the time dependent flow map, denoted as psi in the paper
    Args:
        x0 (torch.Tensor): samples frmo source distribution (batch_size, dim)
        x1 (torch.Tensor): samples from target distribution (batch_size, dim)
        t (float): Time parameter (batch_size, 1)

    Returns:
        torch.Tensor: The flow map (batch_size, dim)
    """
    return (1 - t) * x0 + t * x1


def sample_x0(n_samples):
    return torch.randn(n_samples, 2)


def sample_t(n_samples):
    """Sample time parameter t uniformly in [0, 1]"""
    return torch.rand(n_samples, 1)
