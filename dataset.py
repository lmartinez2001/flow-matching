import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset


def make_dataset(n_samples: int, noise: float, random_state: int = 0):
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    return dataset


def plot_dataset(dataset: TensorDataset, save_path: str = None):
    X = dataset.tensors[0]
    X = X.numpy()
    plt.scatter(*X.T, cmap="coolwarm", s=1)
    plt.title("Dataset")
    plt.axis("off")
    if save_path:
        plt.savefig(
            save_path, dpi=300, bbox_inches="tight", pad_inches=0.0, transparent=True
        )
    plt.show()
