import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
folder = "death_match_killing_400_2/"

def plot_file(filename, color, label, linestyle):
    epochs = []
    scores = []
    std_devs = []
    abs_file_path = os.path.join(script_dir, folder + filename)
    print("Loading file " + abs_file_path)
    with open(abs_file_path, 'r') as f:
        for line in f:
            epoch = line.split()[0]
            score = line.split()[1]
            std_dev = line.split()[2]
            print("Epoch: " + epoch + ", Score: " + score + ", StdDev: " + std_dev)
            epochs.append(int(epoch) + 1)
            scores.append(score)
            std_devs.append(std_dev)

    e = np.array(epochs, dtype=float)
    s = np.array(scores, dtype=float)
    std = np.array(std_devs, dtype=float)

    plt.plot(e, s, color=color, label=label, linewidth=1.0, linestyle=linestyle)
    plt.fill_between(e, s-std, s+std, facecolor=color, alpha=0.10, linewidth=0.0)

plot_file('train.dat', color="red", label="train", linestyle="dashed")
plot_file('test.dat', color="blue", label="test", linestyle="solid")

plt.legend(loc='upper left', frameon=False)
plt.xlim([0, 400])
plt.ylim(ymin=-2)
plt.ylim(ymax=8)

plt.axhline(0, color='black', linestyle="dashed")
plt.axvline(0, color='black', linestyle="dashed")

plt.ylabel('Score')
plt.xlabel('Epoch')

plt.show()