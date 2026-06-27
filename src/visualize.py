from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_figure(fig, save_path=None):
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")


def imshow(img, title=None):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    if title:
        plt.title(title, fontsize=9)


def plot_history(history, title="Training Curves", save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training Loss vs Epochs")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "r-o", markersize=3, label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Train & Val Accuracy vs Epochs")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_capacity_results(capacity_results, save_path=None):
    labels = [r["label"] for r in capacity_results]
    test_acc = [r["test_acc"] for r in capacity_results]
    val_acc = [r["val_acc"] for r in capacity_results]
    x = range(len(capacity_results))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, val_acc, "r-s", label="Val Acc", markersize=7)
    ax.plot(x, test_acc, "g-^", label="Test Acc", markersize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Architecture (increasing capacity →)")
    ax.set_title("MLP: Accuracy vs Model Capacity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def visualize_feature_maps(model, img_tensor, title="Feature Maps", save_path=None):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        inp = img_tensor.unsqueeze(0).to(device)
        _, (f1, f2, f3) = model(inp, return_features=True)

    feature_sets = [
        (f1.squeeze(0).cpu(), "Block 1 (32 ch, after pool)"),
        (f2.squeeze(0).cpu(), "Block 2 (64 ch, after pool)"),
        (f3.squeeze(0).cpu(), "Block 3 (128 ch, after pool)"),
    ]

    n_show = 8
    fig, axes = plt.subplots(3, n_show + 1, figsize=(16, 7))

    for row, (fmap, block_title) in enumerate(feature_sets):
        orig = img_tensor / 2 + 0.5
        axes[row, 0].imshow(np.transpose(orig.numpy(), (1, 2, 0)))
        axes[row, 0].set_title("Input", fontsize=8)
        axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(block_title, fontsize=8)

        for col in range(n_show):
            ch = fmap[col].numpy()
            axes[row, col + 1].imshow(ch, cmap="viridis")
            axes[row, col + 1].set_title(f"ch{col}", fontsize=7)
            axes[row, col + 1].axis("off")

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def visualize_misclassified(mis_imgs, mis_true, mis_pred, classes, title="Misclassified Images", save_path=None):
    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    for i, ax in enumerate(axes.flat):
        img = mis_imgs[i] / 2 + 0.5
        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        ax.set_title(f"T:{classes[mis_true[i]]}\nP:{classes[mis_pred[i]]}", fontsize=7)
        ax.axis("off")
    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig
