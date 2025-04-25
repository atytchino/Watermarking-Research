import matplotlib.pyplot as plt

# Epochs
epochs = list(range(1, 11))

# Metrics from the log
g_loss = [16.4774, 16.3116, 16.0152, 15.6188, 15.0342, 14.7594, 14.5505, 13.7076, 14.0233, 14.0222]
c1_wm_loss = [0.0883, 0.0866, 0.0948, 0.1035, 0.1302, 0.1358, 0.1353, 0.1797, 0.1423, 0.1302]
c1_wm_acc = [0.973, 0.972, 0.961, 0.957, 0.948, 0.943, 0.943, 0.933, 0.939, 0.948]
c2_wm_loss = [2.1111, 2.1884, 2.1554, 2.1137, 2.0859, 2.0494, 2.0222, 1.9882, 1.9507, 1.9091]
c2_clean_loss = [1.1129, 1.1011, 1.0910, 1.0747, 1.0644, 1.0528, 1.0398, 1.0295, 1.0145, 1.0043]
c2_acc_wm = [0.332, 0.397, 0.477, 0.532, 0.604, 0.664, 0.656, 0.677, 0.692, 0.732]
c2_acc_clean = [0.340, 0.407, 0.493, 0.551, 0.613, 0.656, 0.668, 0.669, 0.693, 0.709]

# Dictionary of metrics for easy looping
metrics = {
    "G Loss": g_loss,
    "C1 WM Loss": c1_wm_loss,
    "C1 WM Acc": c1_wm_acc,
    "C2 WM Loss": c2_wm_loss,
    "C2 Clean Loss": c2_clean_loss,
    "C2 Acc WM": c2_acc_wm,
    "C2 Acc Clean": c2_acc_clean
}

# Generate separate plots for each metric
for name, values in metrics.items():
    plt.figure()
    plt.plot(epochs, values, marker='o')
    plt.title(f"{name} vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel(name)
    plt.xticks(epochs)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
