"""
plot_comparison.py
──────────────────
Load comparison_results.json and generate publication-quality comparison
visualizations between ResNet-18 and SE-ResNet-18.

Usage:
    python plot_comparison.py
    python plot_comparison.py --results comparison_results.json
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)


def plot_accuracy_comparison(results, save_path='comparison_accuracy.png'):
    """Side-by-side test accuracy curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'resnet18': '#2196F3', 'se_resnet18': '#FF5722'}
    labels = {'resnet18': 'ResNet-18', 'se_resnet18': 'SE-ResNet-18'}

    for key, r in results.items():
        epochs = r['epochs']
        accs = r['test_accuracies']
        ax.plot(epochs, accs, linewidth=2, color=colors.get(key, 'gray'),
                label=f"{labels.get(key, key)} (best: {max(accs):.2f}%)")

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_title('Test Accuracy: ResNet-18 vs SE-ResNet-18 on CIFAR-10',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()


def plot_loss_comparison(results, save_path='comparison_loss.png'):
    """Side-by-side training loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'resnet18': '#2196F3', 'se_resnet18': '#FF5722'}
    labels = {'resnet18': 'ResNet-18', 'se_resnet18': 'SE-ResNet-18'}

    for key, r in results.items():
        epochs = r['epochs']
        losses = r['train_losses']
        ax.plot(epochs, losses, linewidth=2, color=colors.get(key, 'gray'),
                label=labels.get(key, key))

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Training Loss', fontsize=13)
    ax.set_title('Training Loss: ResNet-18 vs SE-ResNet-18 on CIFAR-10',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()


def plot_per_class_comparison(results, save_path='comparison_per_class.png'):
    """Per-class accuracy grouped bar chart (final epoch)."""
    # Filter to models that have per-class data
    eligible = {k: r for k, r in results.items()
                if 'per_class_accuracies' in r and len(r['per_class_accuracies']) > 0}
    if not eligible:
        print("Skipped per-class plot: no per-class data available.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(CLASSES))
    n = len(eligible)
    width = 0.7 / max(n, 1)
    colors = {'resnet18': '#2196F3', 'se_resnet18': '#FF5722'}
    labels = {'resnet18': 'ResNet-18', 'se_resnet18': 'SE-ResNet-18'}

    bars = []
    for i, (key, r) in enumerate(eligible.items()):
        final_per_class = r['per_class_accuracies'][-1]
        accs = [final_per_class.get(c, 0) for c in CLASSES]
        offset = -width * n / 2 + i * width + width / 2
        b = ax.bar(x + offset, accs, width, label=labels.get(key, key),
                   color=colors.get(key, 'gray'), alpha=0.85)
        bars.append(b)

    ax.set_xlabel('Class', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Per-Class Accuracy at Final Epoch', fontsize=15,
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASSES], fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    plt.close()


def print_summary_table(results):
    """Print a formatted comparison table to terminal."""
    print(f"\n{'='*72}")
    print(f"{'COMPARISON TABLE':^72}")
    print(f"{'='*72}")
    print(f"{'Metric':<30} {'ResNet-18':>18} {'SE-ResNet-18':>18}")
    print(f"{'-'*72}")

    keys = list(results.keys())
    r1 = results[keys[0]]
    r2 = results[keys[1]] if len(keys) > 1 else None

    def row(metric, v1, v2=None):
        if v2 is not None:
            print(f"{metric:<30} {str(v1):>18} {str(v2):>18}")
        else:
            print(f"{metric:<30} {str(v1):>18}")

    row('Parameters', f"{r1['total_params']:,}",
        f"{r2['total_params']:,}" if r2 else None)
    row('Best Test Accuracy (%)', f"{max(r1['test_accuracies']):.2f}",
        f"{max(r2['test_accuracies']):.2f}" if r2 else None)
    row('Final Test Accuracy (%)', f"{r1['test_accuracies'][-1]:.2f}",
        f"{r2['test_accuracies'][-1]:.2f}" if r2 else None)
    row('Final Training Loss', f"{r1['train_losses'][-1]:.4f}",
        f"{r2['train_losses'][-1]:.4f}" if r2 else None)
    row('Training Time (s)', f"{r1['training_time_seconds']:.1f}",
        f"{r2['training_time_seconds']:.1f}" if r2 else None)
    row('Inference Time (ms)', f"{r1['inference_time_ms']:.3f}",
        f"{r2['inference_time_ms']:.3f}" if r2 else None)

    print(f"{'='*72}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='comparison_results.json',
                        help='Path to comparison results JSON')
    args = parser.parse_args()

    results = load_results(args.results)

    plot_accuracy_comparison(results)
    plot_loss_comparison(results)

    plot_per_class_comparison(results)

    if len(results) >= 2:
        print_summary_table(results)

    print("\n✓ All comparison plots generated.")


if __name__ == '__main__':
    main()
