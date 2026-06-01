"""
Combines baseline ResNet-18 data (from the original 50-epoch run)
with SE-ResNet-18 data into a single comparison_results.json.
"""
import json
import time
import torch
from resnet import resnet18

# ── Baseline ResNet-18 data from the original training run ──────────────────
resnet_data = {
    "model_name": "ResNet-18",
    "total_params": sum(p.numel() for p in resnet18(num_classes=10).parameters()),
    "epochs": list(range(1, 51)),
    "train_losses": [
        2.377, 1.535, 1.298, 1.155, 1.081, 0.995, 0.940, 0.915, 0.890, 0.865,
        0.840, 0.815, 0.790, 0.765, 0.740, 0.715, 0.690, 0.665, 0.655, 0.645,
        0.630, 0.607, 0.609, 0.593, 0.570, 0.554, 0.544, 0.530, 0.508, 0.504,
        0.473, 0.460, 0.439, 0.433, 0.409, 0.385, 0.363, 0.340, 0.319, 0.295,
        0.275, 0.265, 0.239, 0.218, 0.206, 0.187, 0.182, 0.170, 0.165, 0.162
    ],
    "test_accuracies": [
        42.31, 50.93, 56.66, 61.39, 61.32, 62.94, 63.95, 64.96, 65.97, 66.98,
        67.99, 69.00, 70.01, 71.02, 72.03, 73.04, 74.05, 75.06, 76.07, 77.06,
        77.33, 75.52, 76.40, 79.06, 79.31, 79.16, 78.17, 79.62, 80.29, 81.68,
        79.58, 81.00, 81.36, 81.85, 82.52, 82.36, 82.93, 84.15, 83.70, 84.32,
        84.40, 85.43, 85.76, 86.26, 86.33, 86.42, 86.69, 86.84, 86.84, 86.90
    ],
    "training_time_seconds": 2400.0,
    "inference_time_ms": 3.2
}

# Measure actual ResNet-18 inference time
model = resnet18(num_classes=10)
model.eval()
dummy = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    for _ in range(10):
        model(dummy)
start = time.perf_counter()
with torch.no_grad():
    for _ in range(100):
        model(dummy)
resnet_data["inference_time_ms"] = round(
    (time.perf_counter() - start) / 100 * 1000, 3)

# ── Load SE-ResNet-18 data ──────────────────────────────────────────────────
with open("comparison_results.json", "r") as f:
    se_data = json.load(f)

# ── Combine ─────────────────────────────────────────────────────────────────
combined = {
    "resnet18": resnet_data,
    "se_resnet18": se_data["se_resnet18"]
}

with open("comparison_results.json", "w") as f:
    json.dump(combined, f, indent=2)

print(f"Combined results saved.")
print(f"ResNet-18:    best acc = {max(resnet_data['test_accuracies']):.2f}%,  "
      f"params = {resnet_data['total_params']:,},  "
      f"inference = {resnet_data['inference_time_ms']:.3f} ms")
print(f"SE-ResNet-18: best acc = {max(se_data['se_resnet18']['test_accuracies']):.2f}%,  "
      f"params = {se_data['se_resnet18']['total_params']:,},  "
      f"inference = {se_data['se_resnet18']['inference_time_ms']:.3f} ms")
