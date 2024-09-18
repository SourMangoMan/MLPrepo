import numpy as np
import random 
from module import sigm, dsigm, fpropn

random.seed(42)
np.random.seed(42)
# Training data
A_train = np.loadtxt("MNIST_train_1000.csv", delimiter=',').astype(int)

# Separating into features and labels
X_train = A_train[:, 1:].T/255
N_train = X_train.shape[1]

# ID numbers to check against
ID_numbers = [2, 3, 5, 9]

# One-hot encoding
Y_train = [1 if number in ID_numbers else 0 for number in A_train[:, 0]]

no_layers = 5
no_nodes = [784, 116, 86, 36, 1]
W = []
b = []

for layer in range(no_layers-1):
    W.append(0.5 - np.random.rand(no_nodes[layer], no_nodes[layer+1]))
    b.append(np.zeros((no_nodes[layer+1], 1)))

# print(W)
# print(b)

alpha = 0.009650
N_range = list(range(N_train))
errors = []
for epoch in range(50):
    rand_index = N_range
    random.shuffle(rand_index)
    errsum = 0
    for j in N_range:
        i = rand_index[j]

        a1 = X_train[:, i].reshape(-1,1)
        # print(f"a1 has shape {a1.shape}")
        n, a, y = fpropn(a1, W, b)

        error = Y_train[i] - y
        err = np.linalg.norm(error * error)
        errsum += err

        A = []
        for n_vec in n:
            A.append(np.diag(dsigm(n_vec.flatten())))
        
        S = [-2 * A[-1] @ error]
        for index, A_mat in enumerate(A[-2::-1]):
            S.insert(0, A_mat @ W[-index-1] @ S[0])

        for index in range(len(W)):
            # print([small_a.shape for small_a in a])
            W[index] -= alpha * a[index] @ S[index].T
            b[index] -= alpha * S[index]
    errors.append(errsum)
        

A_test = np.loadtxt("MNIST_test_100.csv", delimiter=',').astype(int)
X_test = A_test[:, 1:].T/255
Y_test = [1 if digit in ID_numbers else 0 for digit in A_test[:, 0]]
N_test = X_test.shape[1]
wins = 0
Y_pred = np.zeros((N_test, 1))

for i in range(N_test):
    a1 = X_test[:, i].reshape(-1,1)
    Y_pred[i] = fpropn(a1, W, b)[2]
    print(f"y_pred: {Y_pred[i]} y_test: {Y_test[i]}")

    if (Y_pred[i] >= 0.5 and Y_test[i] == 1) or (Y_pred[i] <= 0.5 and Y_test[i] == 0):
        wins += 1
    print(f"Testing Accuracy = {wins}/{N_test}")
    print()

print(f"Testing Accuracy = {wins}/{N_test} = {wins/N_test}")

        
        





        