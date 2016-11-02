import numpy as np
import matplotlib.pyplot as plt

def plot_file(filename, color, label, linestyle):
    epochs = []
    scores = []
    std_devs = []
    print("Loading file")
    with open(filename, 'r') as f:
        for line in f:
            epoch = line.split()[0]
            score = line.split()[1]
            std_dev = line.split()[2]
            print("Epoch: " + epoch + ", Score: " + score + ", StdDev: " + std_dev)
            epochs.append(epoch)
            scores.append(score)
            std_devs.append(std_dev)

    e = np.array(epochs, dtype=float)
    s = np.array(scores, dtype=float)
    std = np.array(std_dev, dtype=float)

    plt.plot(e, s, color=color, label=label, linewidth=2.0, linestyle=linestyle)
    plt.fill_between(e, s-std, s+std, facecolor=color, alpha=0.25, linewidth=0.0)

plot_file('train_.dat', color="red", label="train", linestyle="dashed")
plot_file('test_.dat', color="blue", label="test", linestyle="solid")

plt.legend(loc='lower right', frameon=False)
plt.xlim([0, 20])
plt.ylim(ymax=100)

plt.axhline(0, color='black', linestyle="dashed")
plt.axvline(0, color='black', linestyle="dashed")

plt.ylabel('Score')
plt.xlabel('Epoch')

plt.show()