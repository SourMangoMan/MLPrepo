import numpy as np
import random 
import matplotlib.pyplot as plt
from module import sigm, dsigm, fprop5


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

W2 = 0.5 - np.random.rand(784, 116)
W3 = 0.5 - np.random.rand(116, 86)
W4 = 0.5 - np.random.rand(86, 36)
W5 = 0.5 - np.random.rand(36, 1)
b2 = np.zeros(shape=(116, 1))
b3 = np.zeros(shape=(86, 1)) 
b4 = np.zeros(shape=(36, 1))
b5 = np.zeros(shape=(1, 1))

alpha = 0.009650
N_range = list(range(N_train))
errors = []
for epoch in range(80):
    rand_index = N_range
    random.shuffle(rand_index)
    errsum = 0
    for j in N_range:
        i = rand_index[j]

        a1 = X_train[:, i].reshape(-1,1)

        # Forward propagation
        n2 = W2.T @ a1 + b2
        a2 = np.array(sigm(n2))
        n3 = W3.T @ a2 + b3
        a3 = np.array(sigm(n3))
        n4 = W4.T @ a3 + b4
        a4 = np.array(sigm(n4))
        n5 = W5.T @ a4 + b5
        y  = np.array(sigm(n5))


        error = Y_train[i] - y
        err = np.linalg.norm(error * error)
        errsum += err
        
        A5 = np.diag(dsigm(n5.flatten()))        
        A4 = np.diag(dsigm(n4.flatten()))
        A3 = np.diag(dsigm(n3.flatten()))
        A2 = np.diag(dsigm(n2.flatten()))

        S5 = (-2 * A5 @ error)
        S4 = ((A4 @ W5) @ S5)
        S3 = ((A3 @ W4) @ S4)
        S2 = ((A2 @ W3) @ S3)

        W2 -= alpha * a1 @ S2.T
        W3 -= alpha * a2 @ S3.T
        W4 -= alpha * a3 @ S4.T
        W5 -= alpha * a4 @ S5.T
        b2 -= alpha * S2
        b3 -= alpha * S3
        b4 -= alpha * S4
        b5 -= alpha * S5
    errors.append(errsum)



length = len(errors)
xvals = range(length)
plt.plot(xvals, errors)
plt.title('Error over updates')
plt.xlabel('update')
plt.ylabel('error')
plt.show()

A_test = np.loadtxt("MNIST_test_100.csv", delimiter=',').astype(int)
X_test = A_test[:, 1:].T/255
Y_test = [1 if digit in ID_numbers else 0 for digit in A_test[:, 0]]
N_test = X_test.shape[1]
wins = 0
Y_pred = np.zeros((N_test, 1))

for i in range(N_test):
    a1 = X_test[:, i].reshape(-1,1)
    Y_pred[i] = fprop5(a1,W2,W3,W4,W5,b2,b3,b4,b5)
    print(f"y_pred: {Y_pred[i]} y_test: {Y_test[i]}")

    if (Y_pred[i] >= 0.5 and Y_test[i] == 1) or (Y_pred[i] <= 0.5 and Y_test[i] == 0):
        wins += 1
    print(f"Testing Accuracy = {wins}/{N_test}")
    print()

print(f"Testing Accuracy = {wins}/{N_test} = {wins/N_test}")