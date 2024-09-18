import numpy as np

def sigm(x):
    return 1 / (1 + np.exp(-x))

def dsigm(x):
    return sigm(x)*(1-sigm(x))

def fprop5(a1: np.ndarray, W2: np.ndarray, W3: np.ndarray, W4: np.ndarray, W5: np.ndarray, b2: np.ndarray, b3: np.ndarray, b4: np.ndarray, b5: np.ndarray):
    n2 = W2.T @ a1 + b2
    a2 = np.array(sigm(n2))
    n3 = W3.T @ a2 + b3
    a3 = np.array(sigm(n3))
    n4 = W4.T @ a3 + b4
    a4 = np.array(sigm(n4))
    n5 = W5.T @ a4 + b5
    return np.array(sigm(n5))

def fpropn(a1: np.ndarray, weights: list, biases: list):
    n = []
    a = [a1]
    for index, weight in enumerate(weights):
        n.append(weight.T @ a[index] + biases[index])
        a.append(np.array(sigm(n[-1])))

    return n, a, a[-1]