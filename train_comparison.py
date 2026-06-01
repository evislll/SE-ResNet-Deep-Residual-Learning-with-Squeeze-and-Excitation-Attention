"""
train_comparison.py
───────────────────
Train both ResNet-18 and SE-ResNet-18 on CIFAR-10 under identical conditions
and save detailed per-epoch metrics to JSON for comparison.

Usage:
    python train_comparison.py                    # train both
    python train_comparison.py --model resnet     # train only ResNet-18
    python train_comparison.py --model se_resnet  # train only SE-ResNet-18
"""

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from resnet import resnet18
from se_resnet import se_resnet18

# ── CIFAR-10 class names ────────────────────────────────────────────────────
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

NUM_EPOCHS = 50
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4


# ── Helper functions ────────────────────────────────────────────────────────
def get_data_loaders():
    """Returns CIFAR-10 train and test data loaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True,
                                           transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=100,
                             shuffle=False, num_workers=2)
    return train_loader, test_loader


def count_parameters(model):
    """Total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch; return average loss."""
    model.train()
    total_loss = 0.0
    batches = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / batches


def evaluate(model, loader, device):
    """Return overall accuracy and per-class accuracy dict."""
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1

    overall_acc = 100.0 * correct / total
    per_class = {}
    for i, name in enumerate(CLASSES):
        if class_total[i] > 0:
            per_class[name] = round(100.0 * class_correct[i] / class_total[i], 2)
        else:
            per_class[name] = 0.0

    return overall_acc, per_class


def measure_inference_time(model, device, input_size=(1, 3, 32, 32), runs=100):
    """Average inference time in milliseconds over `runs` forward passes."""
    model.eval()
    dummy = torch.randn(*input_size).to(device)
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / runs * 1000  # ms
    return round(elapsed, 3)


# ── Main training pipeline ──────────────────────────────────────────────────
def train_model(model_name, model, train_loader, test_loader, device):
    """Full training loop. Returns a results dict."""
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {
        'model_name': model_name,
        'total_params': count_parameters(model),
        'epochs': [],
        'train_losses': [],
        'test_accuracies': [],
        'per_class_accuracies': [],
        'training_time_seconds': 0.0,
        'inference_time_ms': 0.0,
    }

    total_start = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.perf_counter()

        avg_loss = train_one_epoch(model, train_loader, criterion,
                                   optimizer, device)
        acc, per_class = evaluate(model, test_loader, device)
        scheduler.step()

        epoch_time = time.perf_counter() - epoch_start

        history['epochs'].append(epoch)
        history['train_losses'].append(round(avg_loss, 4))
        history['test_accuracies'].append(round(acc, 2))
        history['per_class_accuracies'].append(per_class)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  |  "
              f"Loss: {avg_loss:.4f}  |  "
              f"Acc: {acc:.2f}%  |  "
              f"Time: {epoch_time:.1f}s")

    history['training_time_seconds'] = round(
        time.perf_counter() - total_start, 1)
    history['inference_time_ms'] = measure_inference_time(model, device)

    print(f"\n✓ {model_name} finished — "
          f"Best acc: {max(history['test_accuracies']):.2f}%  |  "
          f"Total time: {history['training_time_seconds']:.1f}s")

    return history


def main():
    parser = argparse.ArgumentParser(description='ResNet vs SE-ResNet comparison')
    parser.add_argument('--model', choices=['resnet', 'se_resnet', 'both'],
                        default='both', help='Which model(s) to train')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders()

    results = {}

    if args.model in ('resnet', 'both'):
        model = resnet18(num_classes=10).to(device)
        results['resnet18'] = train_model('ResNet-18', model,
                                          train_loader, test_loader, device)

    if args.model in ('se_resnet', 'both'):
        model = se_resnet18(num_classes=10).to(device)
        results['se_resnet18'] = train_model('SE-ResNet-18', model,
                                             train_loader, test_loader, device)

    # Save results
    out_file = 'comparison_results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {out_file}")

    # Print summary table
    if len(results) == 2:
        print(f"\n{'='*60}")
        print(f"{'COMPARISON SUMMARY':^60}")
        print(f"{'='*60}")
        for key, r in results.items():
            best = max(r['test_accuracies'])
            print(f"  {r['model_name']:20s}  |  "
                  f"Params: {r['total_params']:>10,}  |  "
                  f"Best Acc: {best:.2f}%  |  "
                  f"Infer: {r['inference_time_ms']:.3f} ms")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
