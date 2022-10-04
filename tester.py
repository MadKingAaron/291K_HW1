import scipy.sparse as sparse
from scipy import stats
import numpy as np
import pandas

np.random.seed(42)
A = sparse.csr_matrix(sparse.random(5, 2, density=0.5).toarray())
para = np.array([1,2,3,4,5])
y_hat = np.array([1,1,1,1,0])
y = np.array([0,1,0,1,1])

#print(y_hat-y)
def gradient(X, y, y_hat):
    #grad = np.dot(X.T, (y_hat-y))#X.T @(y_hat - y)
    grad = X.T @ (y_hat-y)
    grad /= X.shape[1]
    return grad
print('Grad:',gradient(A,y,y_hat))
#print(gradient(A,y,y_hat)[4], '\n\n' ,gradient(A,y,y_hat)[0])
def single_grad(X, y, y_hat):
    grad = X * (y_hat - y)
    # print("\n\nSingle Grad:", np.array(grad).flatten())
    return np.array(grad).flatten()
def grad_fixed(X, y, y_hat):
    grad = np.zeros_like(para, dtype=np.float64)

    for i in range(X.shape[1]):
        #print(X.getcol(i).shape)
        grad += single_grad(X.getcol(i).todense(), y[i], y_hat[i])
    
    grad /= X.shape[1]
    return grad

def score(X, w):
    return w @ X
def sigmoid(X, w):
    y = score(X,w)
    return np.exp(y)/(1+np.exp(y))
def predict(X,w):
    pred = sigmoid(X,w)
    for i in range(X.shape[1]):
        if pred[i] >= 0.5:
            pred[i] = 1.0
        else:
            pred[i] = 0.0
    return pred
def grad_decent(X, y_hat, para, lr=1, n_iter=10):
    para = para.astype(np.float64)
    for n in range(n_iter):
        #print(predict(X,para))
        #print(para)
        para -= lr*grad_fixed(X,predict(X, para),y_hat)
    return para
    

    


# print(A * 5, '\n\n', (A*5).todense())
print('Grad_Fixed:',grad_fixed(A,y,y_hat))
#print(A.getcol(0).shape)

#print(np.dot(para, A.getcol(0)))
# print(para.T @ A.getcol(0))

# print(sparse.csr_matrix(sparse.random(5, 0, density=0.5).toarray()))

# print((para @ A))
# print(sigmoid(A, para))

print(grad_decent(A,y,y_hat, para, n_iter=100), grad_decent(A,y,y_hat, para))
X = A
(d,m) = X.shape
s = np.zeros(shape=m, dtype=np.float64) # this is the desired type and shape for the output
        # TODO ======================== YOUR CODE HERE =====================================
s += (para.T @ X)
print(s)
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