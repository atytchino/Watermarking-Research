import matplotlib.pyplot as plt

# Epochs
epochs = list(range(1, 11))

# Metrics for C2
c2_wm_loss = [2.1111, 2.1884, 2.1554, 2.1137, 2.0859, 2.0494, 2.0222, 1.9882, 1.9507, 1.9091]
c2_clean_loss = [1.1129, 1.1011, 1.0910, 1.0747, 1.0644, 1.0528, 1.0398, 1.0295, 1.0145, 1.0043]

c2_acc_wm = [0.332, 0.397, 0.477, 0.532, 0.604, 0.664, 0.656, 0.677, 0.692, 0.732]
c2_acc_clean = [0.340, 0.407, 0.493, 0.551, 0.613, 0.656, 0.668, 0.669, 0.693, 0.709]

# Plot C2 Losses
plt.figure()
plt.plot(epochs, c2_wm_loss, marker='o', label='C2 WM Loss')
plt.plot(epochs, c2_clean_loss, marker='s', label='C2 Clean Loss')
plt.title("Epoch vs C2 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot C2 Accuracies
plt.figure()
plt.plot(epochs, c2_acc_wm, marker='o', label='C2 Acc WM')
plt.plot(epochs, c2_acc_clean, marker='s', label='C2 Acc Clean')
plt.title("Epoch vs C2 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()