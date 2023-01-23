#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:18:07 2022

@author: user
"""

from scipy.stats import ortho_group
import tsp_gradient_V2_5 as tsp

N = tsp.np.diag(list(range(7, 0, -1)))
P = ortho_group.rvs(7)
H = P.T @ N @ P
H_i = H

print("N : \n", N)
print("H : \n", H)
print(tsp.norm_fro(H))
print(tsp.norm_fro(N))

alpha = 1 / (4 * tsp.norm_fro(H) * tsp.norm_fro(N))
alpha_k = alpha

liste = [-1/2 * tsp.np.linalg.norm(H - N, "fro")**2]
liste_1 = [H[0,0]]
liste_2 = [H[1,1]]
liste_3 = [H[2,2]]
liste_4 = [H[3,3]]
liste_5 = [H[4,4]]
liste_6 = [H[5,5]]
liste_7 = [H[6,6]]

COUNT_LIMIT = 1e4
BREAKER_LIMIT = 1e-4

COUNT = 0
BREAKER = 1
while BREAKER > BREAKER_LIMIT and COUNT < COUNT_LIMIT:
    COUNT += 1

    # H_and_N = tsp.std_lie_bracket(H, N)
    H_and_N = tsp.std_lie_bracket(P.T @ H_i @ P, N)
    BREAKER = tsp.norm_fro(H_and_N)

    # H = H + alpha * tsp.std_lie_bracket(H, H_and_N)
    # H = tsp.exp_of_pade( -1 * alpha * H_and_N) @ H @ tsp.exp_of_pade( alpha * H_and_N)
    P = P @ tsp.exp_of_pade( -1 * alpha_k * H_and_N)
    H = P.T @ H_i @ P
    alpha_k = 1 / 2 / tsp.norm_fro(tsp.std_lie_bracket(H, N)) * tsp.np.log((tsp.norm_fro(tsp.std_lie_bracket(H, N))**2 / tsp.norm_fro(H_i) / tsp.norm_fro(tsp.std_lie_bracket(N, tsp.std_lie_bracket(H, N)))) + 1)

    # Graph
    liste += [-1/2 * tsp.np.linalg.norm(H - N, "fro")**2]
    liste_1 += [H[0,0]]
    liste_2 += [H[1,1]]
    liste_3 += [H[2,2]]
    liste_4 += [H[3,3]]
    liste_5 += [H[4,4]]
    liste_6 += [H[5,5]]
    liste_7 += [H[6,6]]



print(tsp.np.trunc(H*1000)/1000)

tsp.plt.subplot(211)
tsp.plt.plot(liste, label = "Fonction d'écart")
tsp.plt.legend()

tsp.plt.subplot(212)
tsp.plt.plot(liste_1, label = "Diag 1")
tsp.plt.plot(liste_2, label = "Diag 2")
tsp.plt.plot(liste_3, label = "Diag 3")
tsp.plt.plot(liste_4, label = "Diag 4")
tsp.plt.plot(liste_5, label = "Diag 5")
tsp.plt.plot(liste_6, label = "Diag 6")
tsp.plt.plot(liste_7, label = "Diag 7")
tsp.plt.legend()
# tsp.
tsp.plt.show()
