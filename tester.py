import scipy.sparse as sparse
from scipy import stats
import numpy as np
import pandas

np.random.seed(42)
A = sparse.csr_matrix(sparse.random(5, 2, density=0.5).toarray())
para = np.array([1,2,3,4,5])
y_hat = np.array([1,1,1,1,0])
y = np.array([0,1,0,1,1])

def gradient(X, y, y_hat):
    grad = np.dot(X.T, (y_hat-y))#X.T @(y_hat - y)
    grad /= X.shape[1]
    return grad

print(gradient(A,y,y_hat))


print(A.getcol(0).shape)
"""
A[2] = 2
print(A.todense())
print(A.getcol(0), A.getcol(0).toarray().flatten())

arr = np.array([0,1,2,4,0,5,6,0,11,0])

arr[arr == 0] = -1
print(arr)

print(A.getcol(0).toarray())
print(np.dot(A.getcol(0).toarray().flatten(),arr))
print(np.log(2))
"""