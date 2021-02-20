import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

def f(x):
    if x == 0 or x == 1:
        return 0
    else:
        return 1

def u(x):
    return ((np.exp(x) - 1) / (1 - np.exp(1))) + x


def solve(n):
    h = 1.0 / (1 + n) 
    x = np.linspace(0, 1, n)
    b = np.array([f(i)*2*(h**2) for i in x])

    A = np.zeros((n,n))
    A[0][0] = 1; A[-1][-1] = 1
    for i in range(1, n-1):
        A[i][i-1] = - 2 - h
        A[i][i] = 4
        A[i][i+1] = h - 2

    p, res, rnk, s = lstsq(A, b)
    return p, x


E = []
for n in range(2, 9):
    p, x = solve(n)
    E_h = 0
    for i in range(n):
        if abs(p[i] - u(x[i])) > E_h:
            E_h = abs(p[i] - u(x[i]))
    print(f"Error for n = {n}, is {E_h}")
    E.append(E_h)

for i in range(len(E)):
    pass
alpha1 = ( np.log(E[2]) - np.log(E[6]) ) / np.log(2)
print(alpha1)
alpha2 = ( np.log(E[1]) - np.log(E[4]) ) / np.log(2)
print(alpha2)
print((alpha1 + alpha2) / 2)
"""
p, x = solve(2)
plt.plot(x, u(x), label="u(x)")
plt.plot(x, p, label="v")
plt.legend()
plt.show()
"""