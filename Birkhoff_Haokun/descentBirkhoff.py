# %%
import torch
import numpy as np
import scipy
from scipy.linalg import expm
import sinkhorn as sk

# %%
dim = 5

# %%
sk = sk.SinkhornKnopp()
seed = np.random.random((dim,dim))
seed = sk.fit(seed)

# %%
def generate_cities(n):
    """
    Generate an n*2 array of random number between 0 and d

    Returns
    -------
    array : cities positions

    """
    array = np.random.rand(n, 2) * 100
    return array

def generate_cost_array(n,cities):
    """
    Generate an n*n array of distance cost (2-norm)

    Returns
    -------
    array : cost (distance)

    """
    array = np.zeros((n, n))
    for var_1 in range(0, n):
        for var_2 in range(0, n):
            vector = cities[var_1] - cities[var_2]
            array[var_1, var_2] = np.linalg.norm(vector)
            if var_1 == var_2:
                array[var_1, var_2] = 0

    return array

# %%
def generate_default_solution(n):
    """
    Set the start array

    """
    array = np.zeros((n, n))
    array[n - 1, 0] = 1
    for var in range(0, n - 1):
        array[0 + var, 1 + var] = 1
    return array

def shuffle_perm(P):
    perm = np.random.permutation(dim)

    # Create an empty matrix
    res = np.zeros((dim,dim))

    # Assign 1 to the permuted indices
    for i in range(dim):
        res[i, perm[i]] = 1
    
    return res

# %%
A = generate_default_solution(dim)
M = shuffle_perm(dim)
# P = generate_default_solution(dim) + 0.01
P = (M+0.1)
# P = seed
Cts = generate_cities(dim)
C = generate_cost_array(dim,Cts)
L1 = np.array([100])
L2 = np.array([100])

# %%
np.round(P)

# %%
M

# %%
def calculate_cost(p,l1):
    cost = C.T @ p.T @ A @ p
    constraint_1 = np.trace(p.T @ p - np.identity(dim))
    return np.trace(cost) + (l1 * constraint_1)

# %%
def grad_P_his(p):
    delta = 0.000000001
    Gradient = P.copy()
    for i in range(len(P)):
        for j in range(len(P[i])):
            tmp1 = P.copy()
            tmp1[i][j] = tmp1[i][j]+delta
            tmp2 = P.copy()
            tmp2[i][j] = tmp2[i][j]-delta
            Gradient[i][j] = (calculate_cost(tmp1,0)-calculate_cost(tmp2,0))/(delta*2)

    return (Gradient)

# %%
def grad_P(p):
    return A @ P @ C.T + A.T @ P.T @ C

# %%
def procrustes(p):
    print(np.shape(p))
    u,s,w = np.linalg.svd(p)
    res_r = np.linalg.matrix_rank(s)
    first_ru = u[0:res_r,:]
    first_rw = w[:,0:res_r]
    res = u.T @ w
    print(np.shape(res))
    return res

# %%
def projection_DP(Y,X):
    alpha = np.linalg.pinv(1 - X @ X.T) @ (Y - X @ Y.T) @ np.ones(dim)
    beta = Y.T @ np.ones(dim) - X.T @ alpha
    res = Y - np.multiply((alpha @ alpha.T + np.ones(dim) @ beta),X)

    return res

# %%
def grad_DP(X):
    element = np.multiply(grad_P(X),X)
    return projection_DP(element,X)

# %%
def retraction_DP(p,xX):
    res = np.multiply(p,expm(np.divide(xX,p)))
    return res

# %%
def nearest_bistochastic(p):
        I = np.identity(dim)
        J = np.ones((dim,dim)) / dim
        W = I - J
        B = W @ p @ W + J
        return B

# %%
def calc_stepArmijo(p ,g ,tau, c):
    step = 1e-5

    t = c * 1
    j = 0
    max = 100
    while ((calculate_cost(p,L1) - calculate_cost(p+step*g,L1)) < step * t) and (j < max):
        step = tau*step
        j = j+1
    
    print(step)
    return step

# %%
cost_history = []
cost_history.append(calculate_cost(M,0))
P_history = []
P_history.append(M)
iter = 0
iter_max = 20

while iter < iter_max:
    iter = iter + 1
    xiX = -grad_DP(P)
    # step = calc_stepArmijo(P,xiX,0.9,0.9)
    step = 0.0000000000001
    print(retraction_DP(P,step * xiX))
    P = sk.fit(abs(retraction_DP(P,step * xiX)))
    print(sum(P))
    # P = procrustes(P)
    gL1 = np.trace(P.T @ P - np.identity(dim))
    L1 = L1 + 0.01 * gL1
    # gL2 = 1/3 * np.trace(P.T @ (P - (np.multiply(P,P))))
    # L2 = L2 + step * gL2
    cost_history.append(calculate_cost(P,0))
    P_history.append(P)

# %%
P.T @ P

# %%
print(retraction_DP(P,step * xiX))
print(step*xiX)


# %%
print("Original Solution Result: \t",calculate_cost(M,0))
print("Optimised Solution Result: \t",calculate_cost(np.round(P),0))
print("Identity Solution Result: \t",calculate_cost(np.identity(dim),0))

# %%
print(np.round(P))
print(P)
print(cost_history)


