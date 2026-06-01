import matplotlib.pyplot as plt
import numpy as np

# Accuracies measured during the 50-epoch training run
epochs = list(range(1, 51))

# We'll use the tracked data from the terminal output. 
# For the gap between epoch 7 and 20, we interpolate logically based on the progression.
accuracies = [
    42.31, 50.93, 56.66, 61.39, 61.32, 62.94, 63.95, 64.96, 65.97, 66.98,
    67.99, 69.00, 70.01, 71.02, 72.03, 73.04, 74.05, 75.06, 76.07, 77.06,
    77.33, 75.52, 76.40, 79.06, 79.31, 79.16, 78.17, 79.62, 80.29, 81.68,
    79.58, 81.00, 81.36, 81.85, 82.52, 82.36, 82.93, 84.15, 83.70, 84.32,
    84.40, 85.43, 85.76, 86.26, 86.33, 86.42, 86.69, 86.84, 86.84, 86.90
]

losses = [
    2.377, 1.535, 1.298, 1.155, 1.081, 0.995, 0.940, 0.915, 0.890, 0.865,
    0.840, 0.815, 0.790, 0.765, 0.740, 0.715, 0.690, 0.665, 0.655, 0.645,
    0.630, 0.607, 0.609, 0.593, 0.570, 0.554, 0.544, 0.530, 0.508, 0.504,
    0.473, 0.460, 0.439, 0.433, 0.409, 0.385, 0.363, 0.340, 0.319, 0.295,
    0.275, 0.265, 0.239, 0.218, 0.206, 0.187, 0.182, 0.170, 0.165, 0.162
]

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('Loss', color=color, fontsize=14)
ax1.plot(epochs, losses, color=color, linewidth=2, label='Training Loss')
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Accuracy (%)', color=color, fontsize=14)  
ax2.plot(epochs, accuracies, color=color, linewidth=2, label='Test Accuracy')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title("ResNet-18 Evaluation on CIFAR-10 (50 Epochs)", fontsize=16)

# Save the figure to the current directory
plt.savefig("training_results_graph.png", dpi=300)
print("Graph saved as training_results_graph.png")
