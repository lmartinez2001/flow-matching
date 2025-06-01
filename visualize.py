import os
import torch
import numpy as np
from model import MLP
from tqdm import tqdm
import imageio.v3 as iio
import matplotlib.pyplot as plt


def make_gif(root: str, save: str, duration: int, loop: int = 0):
    imgs = []
    img_names = os.listdir(root)
    img_names = sorted(img_names)
    for img_name in img_names:
        img_path = os.path.join(root, img_name)
        imgs.append(iio.imread(img_path))

    if imgs:
        last_img = imgs[-1]
        for _ in range(15):
            imgs.append(last_img)
    iio.imwrite(save, imgs, duration=duration, loop=loop)
    print(f"Gif saved at {save}")


def generate_gifs(
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

    os.makedirs(f"{output_root}/vf", exist_ok=True)
    os.makedirs(f"{output_root}/fm", exist_ok=True)

    model = MLP(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        n_layers=n_layers,
        checkpoint_path=checkpoint_path,
    )
    model.eval()

    all_t = torch.linspace(0, 1, n_timesteps)

    # ==> Generate vector field
    x = np.linspace(-3, 3, grid_resolution)
    y = np.linspace(-3, 3, grid_resolution)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    grid = torch.tensor(grid, dtype=torch.float32)

    for idx, t in tqdm(
        enumerate(all_t), desc="Computing vector field", total=n_timesteps
    ):
        vfield = model.vector_field(t, grid)

        vf = vfield.reshape(grid_resolution, grid_resolution, 2)
        u = vf[:, :, 0].detach().numpy()
        v = vf[:, :, 1].detach().numpy()
        magnitude = np.sqrt(u**2 + v**2)
        plt.figure(figsize=(10, 10))
        plt.title(f"Vector field (t={t:.2f})")
        plt.quiver(xx, yy, u, v, magnitude, cmap="viridis", alpha=0.5)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.axis("off")
        plt.savefig(
            os.path.join(output_root, "vf", f"vf_{idx:04d}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print("Computing flow map")
    samples = model.sample(n_samples, n_timesteps)
    for idx in tqdm(range(n_timesteps), desc="Saving frames"):
        sample = samples[idx].numpy()
        time = all_t[idx].item()
        plt.figure(figsize=(10, 10), facecolor="black")
        plt.title(f"Distribution (t={time:.2f})", color="white")
        plt.gca().set_facecolor("black")
        plt.scatter(*sample.T, s=1, alpha=0.5, c="white", marker=".")
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.axis("off")
        plt.savefig(
            os.path.join(output_root, "fm", f"fm_{idx:04d}.png"),
            dpi=300,
            bbox_inches="tight",
            facecolor="black",
        )
        plt.close()

    print("Generating gifs")
    gif_duration = 120
    make_gif(os.path.join(output_root, "vf"), "res/vf.gif", gif_duration)
    make_gif(os.path.join(output_root, "fm"), "res/fm.gif", gif_duration)


if __name__ == "__main__":
    generate_gifs(
        checkpoint_path="out/model.pth",
        output_root="out/gifs",
        n_timesteps=100,
        grid_resolution=30,
        n_samples=50_000,
    )
