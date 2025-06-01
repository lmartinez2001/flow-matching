import torch
import torch.nn as nn
import fm_utils as utils
from torchdiffeq import odeint


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int,
        checkpoint_path: str | None = None,
    ):
        super().__init__()

        self.layers = (
            [self._make_linear(input_size, hidden_size)]
            + [self._make_linear(hidden_size, hidden_size) for _ in range(n_layers - 2)]
            + [nn.Linear(hidden_size, output_size)]
        )

        self.net = nn.Sequential(*self.layers)

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(checkpoint)
            print(f"Model loaded from {checkpoint_path}")

    def _make_linear(self, input_size, output_size):
        return nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())

    def forward(self, x):
        return self.net(x)

    def vector_field(self, t, x):
        """Compute the flow vector field of x at time t"""
        t = t.unsqueeze(0).unsqueeze(1).repeat(x.size(0), 1)
        model_input = torch.cat((x, t), dim=1)
        with torch.no_grad():
            pred = self.forward(model_input)
        return pred

    def sample(self, n_samples: int, n_times: int):
        x0 = utils.sample_x0(n_samples)
        t = torch.linspace(0, 1, n_times)
        res = odeint(self.vector_field, x0, t)
        return res
