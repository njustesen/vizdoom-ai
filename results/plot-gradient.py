import numpy as np
import matplotlib.pyplot as plt

def plot_file(filename):
    epochs = []
    train_scores = []
    test_scores = []
    losses = []
    with open(filename, 'r') as f:
        for line in f:
            print(line.split())
            epoch = line.split()[0]
            train_score = line.split()[1]
            test_score = line.split()[2]
            loss = line.split()[3]
            epochs.append(epoch)
            train_scores.append(train_score)
            test_scores.append(test_score)
            losses.append(loss)

    e = np.array(epochs, dtype=float)
    train = np.array(train_scores, dtype=float)
    test = np.array(test_scores, dtype=float)
    l = np.array(losses, dtype=float)

    plt.plot(e, train, color="red", label="train", linewidth=2.0, linestyle="dashed")
    plt.plot(e, test, color="blue", label="test", linewidth=2.0, linestyle="solid")
    plt.plot(e, l, color="green", label="loss", linewidth=2.0, linestyle="dotted")

plot_file('simple_basic_gradient_policy_theano/data.dat')

plt.legend(loc='upper right', frameon=True)
plt.xlim([0, 100])

plt.axhline(0, color='black', linestyle="dashed")
plt.axvline(0, color='black', linestyle="dashed")

plt.ylabel('Score')
plt.xlabel('Epoch')

plt.show()