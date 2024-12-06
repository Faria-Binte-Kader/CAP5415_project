import json
import matplotlib.pyplot as plt

# Load the JSON file
with open("retinamnist_ViT_B_16_clip_metrics.json", "r") as file:
    data = json.load(file)

# Extract training and testing metrics
train_metrics = data["train"]
test_metrics = data["test"]

# Prepare data for plotting
train_accuracy = [epoch["accuracy"] for epoch in train_metrics]
train_f1 = [epoch["f1"] for epoch in train_metrics]
test_accuracy = [epoch["accuracy"] for epoch in test_metrics]
test_f1 = [epoch["f1"] for epoch in test_metrics]
epochs = range(1, len(train_metrics) + 1)

# Plot training metrics
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_accuracy, label="Train Accuracy", marker="o")
plt.plot(epochs, test_accuracy, label="Test Accuracy", marker="o")
plt.title("RetinaMNIST Accuracy Score Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig("retinamnist_acc_metrics_plot.png")  # Save the plot
plt.close()

# Plot testing metrics
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_f1, label="Train F1 Score", marker="o")
plt.plot(epochs, test_f1, label="Test F1 Score", marker="o")
plt.title("RetinaMNIST F1 Score Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig("retinamnist_f1_metrics_plot.png")  # Save the plot
plt.close()

print("Plots saved")
