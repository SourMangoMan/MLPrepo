import numpy as np
import random 
from module import sigm, dsigm

# Training data
A_train = np.loadtxt("MNIST_train_1000.csv", delimiter=',').astype(int)

# Separating into features and labels
X_train = A_train[:, 1:].T/255
N_train = X_train.shape[1]

# ID numbers to check against
ID_numbers = [2, 3, 5, 9]

# One-hot encoding
Y_train = [1 if number in ID_numbers else 0 for number in A_train[:, 0]]

# Initialize weights and biases
W2 = np.random.normal(size=(784, 139))
b2 = np.zeros((139, 1))
W3 = np.random.normal(size=(139, 1))
b3 = np.zeros((1, 1))

# Learning rate
alpha = 0.009650

# Training
N_range = list(range(N_train))
for epoch in range(50):
    rand_index = N_range
    random.shuffle(rand_index)
    for j in N_range:
        i = rand_index[j]

        # Forward propagation
        a1 = X_train[:, i].reshape(-1,1)
        n2 = W2.T @ a1 + b2
        # print((W2.T).shape)
        # print(a1.shape)
        # print(n2.shape)
        a2 = sigm(n2)

        n3 = W3.T @ a2 + b3
        y  = sigm(n3)
        
        # Error
        error = Y_train[i] - y
        A2 = np.diag(dsigm(n2.flatten()))
        A3 = np.diag(dsigm(n3))
        S3 = (-2 * A3 @ error)

        # Backpropagation
        S2 = ((A2 @ W3) @ S3).reshape(-1,1)

        # Update rule
        W3 -= alpha * a2 * S3
        W2 -= alpha * a1 @ S2.T
        b3 -= alpha * S3
        b2 -= alpha * S2

# Evaluate accuracy
A_test = np.loadtxt("MNIST_test_100.csv", delimiter=',').astype(int)
X_test = A_test[:,1:].T/255
Y_test = [1 if number in ID_numbers else 0 for number in A_test[:, 0]]
N_test = X_test.shape[1]
wins = 0
Y_pred = np.zeros((N_test, 1))

for i in range(N_test):
    a1 = X_test[:, i].reshape(-1,1)
    Y_pred[i] = sigm(W3.T @ sigm(W2.T @ a1 + b2) + b3)

    if (Y_test[i] == 1 and Y_pred[i] >= 0.5) or (Y_test[i] == 0 and Y_pred[i] <= 0.5):
        wins += 1

print(f"testing accuracy = {wins}/{N_test} = {wins/N_test}")



         