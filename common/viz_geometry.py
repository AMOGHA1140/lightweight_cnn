"""Visualize the geometry a GAConv predicts.

Renders the per-location orientation and the two extents as heatmaps. Useful for
debugging (does the module learn sensible geometry?) and for paper figures.
"""

import math

import torch


@torch.no_grad()
def visualize_geometry(gaconv, x, batch_index=0, save_path=None, show=False):
    """Render theta / sigma_major / sigma_minor heatmaps for one input.

    Args:
        gaconv: a ``GAConv`` module.
        x: input tensor ``[B, C, H, W]`` (matching the module's channels).
        batch_index: which sample in the batch to plot.
        save_path: if given, the figure is written here.
        show: call ``plt.show()`` (off by default for headless use).
    Returns:
        the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    was_training = gaconv.training
    gaconv.eval()
    theta, sigma_maj, sigma_min = gaconv.geometry(x)
    gaconv.train(was_training)

    theta = theta[batch_index, 0].float().cpu().numpy()
    sigma_maj = sigma_maj[batch_index, 0].float().cpu().numpy()
    sigma_min = sigma_min[batch_index, 0].float().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        ("theta (rad)", theta, "twilight", -math.pi, math.pi),
        ("sigma_major", sigma_maj, "viridis", gaconv.sigma_min, gaconv.sigma_max),
        ("sigma_minor", sigma_min, "viridis", gaconv.sigma_min, gaconv.sigma_max),
    ]
    for ax, (title, data, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    from common.gaconv import GAConv

    gaconv = GAConv(64)
    x = torch.randn(1, 64, 48, 48)
    visualize_geometry(gaconv, x, save_path="gaconv_geometry.png")
    print("wrote gaconv_geometry.png")
