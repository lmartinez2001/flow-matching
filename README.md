# Flow Matching

<div align="center">
  <img src="teaser.gif" alt="Flow Matching Evolution" width="100%" />
</div>

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

1. Clone the repository:
```bash
git clone https://github.com/lmartinez2001/flow-matching
cd flow-matching
```

2. Install dependencies:
```bash
uv sync
```

## Usage

### Training the Model

Train a flow matching model on the two moons dataset:

```bash
uv run train.py
```

This will:
- Generate a synthetic two moons dataset
- Train an MLP to learn the vector field
- Save the trained model to `out/model.pth`
- Save training losses to `out/losses.npy`

Training parameters can be modified in [`train.py`](train.py):
- `lr`: Learning rate (default: 1e-4)
- `epochs`: Number of training epochs (default: 1000)
- `batch_size`: Batch size (default: 4096)
- `n_samples`: Number of training samples (default: 8192)

### Generating Visualizations

Create animated visualizations of the learned flow:

```bash
uv run visualize.py
```

This will generate:
- Vector field visualizations showing the learned flow at different time steps
- Flow evolution animations showing how samples are transported from noise to data
- GIFs saved to `res/vf.gif` (vector fields) and `res/fm.gif` (flow evolution)


## References

- **Paper**: [Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747) - Lipman et al., 2022
- **Video**: [Flow Matching Explained](https://www.youtube.com/watch?v=7cMzfkWFWhI&t=1094s)