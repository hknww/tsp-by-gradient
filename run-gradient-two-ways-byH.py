# Gradient Descent Test
from math import sqrt
# Parameters
step = 0.01
x = 0

# Constraint: x**2 + y**2 = 32
# To minimize: x + y

# Consider only y < 0
for i in range(3000):
    y = -sqrt(32-x**2)
    x = x - step*(1-0.5*(1/y)*2*x)

print("Methode 1 : x = " + str(x))

# With Penalty: cost function = x + y + (x^2+y^2-32)^2
[a,b] = [0,0]
for i in range(1000):
    a = a - (1+4*a*(a**2+b**2-32))*step
    b = b - (1+4*b*(a**2+b**2-32))*step

print("Method 2 : (x, y) = ", a, b)