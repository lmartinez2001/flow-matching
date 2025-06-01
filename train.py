import os
import torch
import numpy as np
from tqdm import tqdm
import fm_utils as utils
import torch.nn.functional as F


def train(model, loader, optimizer, epochs):
    batch_size = loader.batch_size
    losses = []

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        for x1 in loader:
            x1 = x1[0]
            t = utils.sample_t(batch_size)
            x0 = utils.sample_x0(batch_size)

            psi_x = utils.compute_psi(x0, x1, t)
            dx = x1 - x0

            model_input = torch.cat((psi_x, t), dim=1)
            preds = model(model_input)

            optimizer.zero_grad()
            loss = F.mse_loss(preds, dx)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return losses


if __name__ == "__main__":
    from model import MLP
    from torch.optim import AdamW
    from dataset import make_dataset
    from torch.utils.data import DataLoader

    lr = 1e-4
    n_samples = 1024 * 8
    epochs = 1000
    batch_size = 4096

    input_size = 3  # t (1) + psi_x (2)
    hidden_size = 512
    output_size = 2  # v_t (2)
    n_layers = 5

    output_root = "out"
    os.makedirs(output_root, exist_ok=True)

    dataset = make_dataset(n_samples=n_samples, noise=0.01)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLP(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        n_layers=n_layers,
    )
    optimizer = AdamW(model.parameters(), lr=lr)

    losses = train(model, loader, optimizer, epochs=epochs)
    np.save(os.path.join(output_root, "losses.npy"), np.array(losses))
    torch.save(model.state_dict(), os.path.join(output_root, "model.pth"))
    print("Training complete")
