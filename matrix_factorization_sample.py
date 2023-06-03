#!/usr/bin/env python3
import numpy as np

################################# Parameters ##################################
# Number of users
M = 5
# Number of items
N = 8
# Number of basis vectors
K = 2

# Rating matrix represented as a list: [[user_id,item_id,rating]]
# [
#  [4,4,5,?,2,?,1,?],
#  [5,4,?,4,?,?,?,1],
#  [?,?,?,2,?,4,4,5],
#  [2,?,1,1,?,4,5,4],
#  [4,?,?,5,?,?,4,4],
# ]
X = []
X.append([0,0,4])
X.append([0,1,4])
X.append([0,2,5])
X.append([0,4,2])
X.append([0,6,1])

X.append([1,0,5])
X.append([1,1,4])
X.append([1,3,4])
X.append([1,7,1])

X.append([2,3,2])
X.append([2,5,4])
X.append([2,6,4])
X.append([2,7,5])

X.append([3,0,2])
X.append([3,2,1])
X.append([3,3,1])
X.append([3,5,4])
X.append([3,6,5])
X.append([3,7,4])

X.append([4,0,4])
X.append([4,3,5])
X.append([4,6,4])
X.append([4,7,4])

############################ Matrix Factorization #############################
def matrix_factorization(X, U, V, K):
    alp = 0.001   # learning rate
    lam = 0.02    # regularization coefficient

    # SGD (Stochastic Gradient Descent)
    for itr in range(1000):
        # Update U and V
        np.random.shuffle(X)
        for (i,j,X_ij) in X:
            # X_ij - X_hat_ij (= U[i,:] V[:,j]) --> e_ij
            e_ij = X_ij - np.dot(U[i,:], V[:,j])
            for k in range(K):
                U[i,k] = U[i,k] + 2 * alp * (e_ij * V[k,j] - lam * U[i,k])
                V[k,j] = V[k,j] + 2 * alp * (e_ij * U[i,k] - lam * V[k,j])
        # Calculate the regularized squared error (sum of F_ij) --> F
        F = 0
        for (i,j,X_ij) in X:
            # 1st term of F_ij
            F_ij = (X_ij - np.dot(U[i,:], V[:,j]))**2
            # 2nd term of F_ij
            for k in range(K):
                F_ij = F_ij + lam * (U[i,k]**2 + V[k,j]**2)
            # Add F_ij to F
            F = F + F_ij
        # Convergence test
        if F < 0.01:
            break
        if (itr+1) % 100 == 0:
            print("#Iterations:", itr+1, "Regularized Squared Error:", F)
    # Sort X
    X.sort()

#################################### Main #####################################
# Fix a seed
np.random.seed(1)

U = np.random.rand(M,K)
V = np.random.rand(K,N)

# Matrix factorization --> U, V
matrix_factorization(X, U, V, K)

# Output U and V
print("U:")
print(U)
print("V:")
print(V)

# Prediction matrix (= UV) --> X_hat
X_hat = np.dot(U, V)
# Output X_hat
print("X_hat:")
print(X_hat)
