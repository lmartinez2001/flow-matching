import os
import torch
import numpy as np
from model import MLP
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def generate_animation(
    checkpoint_path: str,
    output_root: str,
    n_timesteps: int,
    grid_resolution: int,
    n_samples: int,
):
    input_size = 3  # t (1) + psi_x (2)
    hidden_size = 512
    output_size = 2  # v_t (2)
    n_layers = 5

    os.makedirs(output_root, exist_ok=True)

    model = MLP(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        n_layers=n_layers,
        checkpoint_path=checkpoint_path,
    )
    model.eval()

    all_t = torch.linspace(0, 1, n_timesteps)

    bounds = (-4, 4)
    # Prepare grid for vector field
    x = np.linspace(*bounds, grid_resolution)
    y = np.linspace(*bounds, grid_resolution)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    grid = torch.tensor(grid, dtype=torch.float32)

    # Pre-compute vector fields
    print("Computing vector fields...")
    vector_fields = []
    for idx, t in tqdm(
        enumerate(all_t), desc="Computing vector field", total=n_timesteps
    ):
        vfield = model.vector_field(t, grid)
        vf = vfield.reshape(grid_resolution, grid_resolution, 2)
        u = vf[:, :, 0].detach().numpy()
        v = vf[:, :, 1].detach().numpy()
        magnitude = np.sqrt(u**2 + v**2)
        vector_fields.append((u, v, magnitude))

    print("Computing flow map (this may take a while)")
    samples = model.sample(n_samples, n_timesteps)

    # Set up the figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    def animate(frame):
        ax1.clear()
        ax2.clear()

        time = all_t[frame].item()

        ax1.set_xlim(*bounds)
        ax1.set_ylim(*bounds)
        ax1.axis("off")
        ax1.set_title(f"Vector field (t = {time:.2f})", fontsize=18)

        ax2.set_xlim(*bounds)
        ax2.set_ylim(*bounds)
        ax2.axis("off")
        ax2.set_title(f"Flow map (t = {time:.2f})", fontsize=18)

        # Plot vector field
        u, v, magnitude = vector_fields[frame]
        ax1.quiver(
            xx,
            yy,
            u,
            v,
            magnitude,
            angles="xy",
            scale_units="xy",
            cmap="coolwarm",
            alpha=0.8,
            width=0.005,
        )

        # Plot flow map
        sample = samples[frame].numpy()
        ax2.hist2d(*sample.T, cmap="viridis", bins=300, range=[bounds, bounds])

    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.1)
    # Create animation
    print("Creating animation (this may take a while as well)")
    anim = animation.FuncAnimation(fig, animate, frames=n_timesteps, interval=50)

    # Save as gif
    gif_path = os.path.join(output_root, "flow_matching.gif")
    anim.save(gif_path, writer="pillow")
    print(f"Animation saved at {gif_path}")

    plt.show()


if __name__ == "__main__":
    generate_animation(
        checkpoint_path="out/model.pth",
        output_root="out/animations",
        n_timesteps=101,
        grid_resolution=15,
        n_samples=200_000,
    )
