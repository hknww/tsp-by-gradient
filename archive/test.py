import tsp_gradient_V2_4 as mat
import numpy as np

A = np.array([[1,2,3],[1,2,3],[4,3,2]])
B = np.array([[0,2,3],[1,0,3],[4,3,0]])

print(A)
print(B)
# print(np.dot(A,B))
# print(mat.std_lie_bracket(A,B))
# print(mat.gnr_lie_bracket(A,B))
# print(mat.indiv_product(A,B))
# print(mat.trace(A))
print(A@B)