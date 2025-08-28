import numpy as np
import matplotlib.pyplot as plt

a = 0
b = 5
N = 100
dx = (b - a) / N

X = []
F = []


def f(x):
    return (np.cos(x - 0.5)) ** 2 - x / (1 + x)


for i in range(N):
    X.append(i)
    F.append(f(i))


plt.plot(X, F)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()
