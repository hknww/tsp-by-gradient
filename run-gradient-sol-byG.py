import numpy as np

# min d'une fonction f(x,y) = x + y s.t. x^2 + y^2 = 32

vec = np.ones((2,1))
mu = 0.01
Ak = np.zeros((2,1))
lambdak = 1
Akp1 = np.array(([1e4],[1e4]))

while (np.linalg.norm(vec + lambdak*Ak)) > 1e-8:
    Akp1 = Ak - mu*(vec + lambdak*Ak)
    lambdakp1 = lambdak + mu*(np.linalg.norm(Akp1)**2-32)
    Ak = Akp1
    lambdak = lambdakp1

print(Ak)
print(np.linalg.norm(vec + lambdak*Ak)**2)