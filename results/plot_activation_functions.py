import numpy as np
import matplotlib.pyplot as plt
import os


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


x = np.arange(-6, 6, .1)
plt.figure(figsize=(4,4))

plt.plot(x, sigmoid(x), linestyle="solid", label="sigmoid", linewidth=2.0)
plt.plot(x, relu(x), linestyle="solid", label="ReLU", linewidth=2.0)
plt.plot(x, tanh(x), linestyle="solid", label="tanh", linewidth=2.0)

plt.legend(loc='lower right', frameon=False)
size = 5
plt.xlim([-size, size])
plt.ylim([-2, 2])
plt.xticks(np.arange(-size, size+1, 1.0))

plt.axhline(0, color='black', linestyle="dashed")
plt.axvline(0, color='black', linestyle="dashed")

plt.ylabel('a')
plt.xlabel('Z')

plt.show()