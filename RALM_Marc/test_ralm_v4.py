# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:20:28 2022

@author: WANG Haokun
@author: CHATROUX Marc
@author: LI Yuansheng
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from scipy import optimize
import tsp_gradient_outils as tsp

# %% [markdown]
# # Essayons de résoudre le TSP modélisé comme ci-dessous:
# ## Modèle des revues :
# Soit les entrées suivantes :
# $$ C\ :\ la\ matrice\ de\ coûts $$
# 
# min :
# $$
# trace_{H\ in\ T_n}(C^T*H)
# $$
# 
# 
# s.t.
# $$ G_1(H) = ‖H − H ◦ H ‖^2_F = 0 $$
# $$ G_2(H) = ‖ A · vec(H ) − B‖^2_2 = 0 $$

# %%
def get_m_a(size):
    """
    Génére la matrice A pour la contraite G_2
    """
    result = []

    # trace nul
    for i in range(size):
        new = [0] * size * size
        new[(size + 1) * i] = 1
        result += [new]
    # colonne == 2
    for i in range(size):
        new = [0] * size * size
        new[i * size:(i + 1) * size] = [1] * size
        result += [new]
    # ligne == 2
    for i in range(size):
        new = [0] * size
        new[i] = 1
        new = new * size
        result += [new]

    return np.array(result)

def get_m_b(size):
    """
    Génére la matrice B pour la contraite G_2
    """
    result = []

    # trace nul
    result += [[0]] * size
    # colonne == 2
    result += [[2]] * size
    # ligne == 2
    result += [[2]] * size

    return np.array(result)

def get_c(m_c):
    """
    Génére la matrice de coût pour la fonction de coût, avec une diagonale "forte"
    """
    result = m_c.copy()
    
    # NE PAS FAIRE CA !!! DONNE LES VALEURS NEGATIVE SUR LA DIAGONALE (EX : -0.24) SUREMENT POUR REDUIRE LE COUT

    # total = m_c.sum()    
    # for i in range(m_c.shape[0]):
    #     result[i, i] = total

    return result

def function_distance(x1, x2):
    """
    Retourne la distance (norme 2) entre deux vecteurs
    """
    vect = x1 - x2
    return np.sqrt(np.dot(vect, vect))

# %%
def the_big_solver(P_1, mute=False,
    n_iter = 100,
    Rho_var = 5e-1,
    Var_gamma_min = -10,
    Var_gamma_max = 1000,
    Var_gamma = 1,
    Epsilon_var = 1e-1,
    Epsilon_min = 1e-5,
    Theta_epsilon = 0.95,
    Theta_rho = 0.95,
    Theta_sigma = 0.99,
    D_min = 1e-3):
    """
    Réalise une simualtion avec les paramètres suivants :
        - n_iter : nombre d'itération maximal
        - Rho_var : Coefficient de pénalité initial
        - Var_gamma_min : Borne supérieur des coefficients de contraintes d'égalité
        - Var_gamma_max : Borne inférieur des coefficients de contraintes d'égalité
        - Var_gamma : Valeur initiales des coefficients de contraintes d'égalité
        - Epsilon_var : Valeur initiale de la précision
        - Epsilon_min : Valeur minimale de la précision
        - Theta_epsilon : Ratio pour l'évolution de Epsilon
        - Theta_rho : Ratio pour l'évolution de Rho
        - Theta_sigma : Ratio pour les critères d'évolution de Rho
        - D_min : Valeur du pas minimal
    
    On travail avec une forme vectoriel (dimension 1)
    """

    def function_f(vec_x):
        """
        Fonction de cout classique
        """
        n = np.int64(np.sqrt(vec_x.shape[0]))
        x = vec_x.reshape((n, n))
        return tsp.trace(C.T @ x)

    def function_h(vec_x):
        """
        Fonction de cout des contraintes d'égalités
        """        
        n = np.int64(np.sqrt(vec_x.shape[0]))
        x = vec_x.reshape((n, n))
        
        result = []

        # Contrainte de trace, la somme des ligne et la somme des colonne
        temp_vec = (M_a @ vec_x.reshape((n *n, 1)) - M_b).flatten()
        result += [[np.dot(temp_vec, temp_vec)]]
        # temp_vec = (M_a @ vec_x.reshape((n *n, 1)) - M_b)
        # result += [[np.linalg.norm(temp_vec, ord=2)**2]]

        # égalité : 0 = || H - H ° H ||_frov
        result += [[np.sum(np.square(np.abs(x - tsp.indiv_product(x, x))))]]

        return np.array(result)

    def function_global_cost(vec_x, gamma_value, rho_value):
        """
        Fonction de cout global
        """
        return function_f(vec_x) + rho_value / 2 * np.sum(np.square(function_h(vec_x) + gamma_value / rho_value))

    def function_eval(vec_x):
        """
        Evalue le respect des contraintes
        """
        return np.sum(function_h(vec_x))
    
    n_iter = np.int64(n_iter)

    # CONSTANTES
    n_size = P_1.get_number()
    C = get_c(P_1.get_cost_array())
    M_a = get_m_a(n_size)
    M_b = get_m_b(n_size)

    # VECTEUR INITIALE
    X_var = tsp.matrix_undir(n_size)
    # X_var[0, 1] = 0.8
    X_var = X_var.flatten()

    # INITIALISATION DES GAMMAS
    SIZE_GAMMA = len(function_h(X_var))
    Gamma_var = np.ones((SIZE_GAMMA, 1)) * Var_gamma
    Gamma_min = np.ones((SIZE_GAMMA, 1)) * Var_gamma_min
    Gamma_max = np.ones((SIZE_GAMMA, 1)) * Var_gamma_max

    # STOCKAGE DE VALEURS
    X_list, Gamma_list, Epsilon_list, Rho_list, H_list, D_list = [], [], [], [], [], []
    Cost_list, Global_cost_list, Eval_list = [], [], []
    X_list += [X_var]
    Gamma_list += [Gamma_var]
    Epsilon_list += [np.array([[Epsilon_var]])]
    Rho_list += [Rho_var]
    Cost_list += [function_f(X_var)]
    Global_cost_list += [function_global_cost(X_var, Gamma_var, Rho_var)]
    Eval_list += [function_eval(X_var)]
    H_list += [function_h(X_var)]

    # afficha (ou non) la barre de chargement
    iter_object = range(n_iter) if mute else tqdm(range(n_iter))

    # itération...
    for k in iter_object:

        # SAUVEGARDE TEMPORAIRE POUR CALCULS
        X_var = X_var.copy()
        Gamma_var = Gamma_var.copy()
        X_var_old = X_var.copy()

        # CALCUL DU NOUVEAU X
        function_to_solve = lambda X :  function_global_cost(X, Gamma_var, Rho_var)
        X_var = optimize.minimize(function_to_solve, X_var, tol=Epsilon_var).x
        distance = function_distance(X_var, X_var_old)

        # MODIF DES PARAMETRES DU RALM
        Gamma_var = np.clip(Gamma_var + Rho_var * function_h(X_var), Gamma_min, Gamma_max)
        Epsilon_var = max(Epsilon_min, Theta_epsilon * Epsilon_var)

        max_1 = function_h(X_var).max()
        max_2 = function_h(X_var_old).max()
        if k == 0 or max_1 <= Theta_sigma * max_2:
            Rho_var = Rho_var
        else:
            Rho_var = Theta_rho * Rho_var


        # STOCKAGE DE VALEURS
        X_list += [X_var]
        Gamma_list += [Gamma_var]
        Epsilon_list += [np.array([[Epsilon_var]])]
        Rho_list += [Rho_var]
        Cost_list += [function_f(X_var)]
        Global_cost_list += [function_global_cost(X_var, Gamma_var, Rho_var)]
        Eval_list += [function_eval(X_var)]
        H_list += [function_h(X_var)]
        D_list +=  [distance]

        # ARRET DE BOUCLE
        if distance <= D_min and Epsilon_var <= Epsilon_min:
            break

    # reshape vers carré pour analyse du résultat
    n = np.int64(np.sqrt(X_var.shape[0]))
    X_var = X_var.reshape((n, n))

    # INDICATEUR
    M_delta = P_1.lp_solution + P_1.lp_solution.T - np.round(X_var)
    Nb_delta = np.sum(M_delta != 0)
    Sm_delta = np.sum(np.abs(M_delta))
    Cost_result = function_f(np.round(X_var).flatten())
    True_cost_result = function_f(X_var.flatten())
    Cost_best = function_f((P_1.lp_solution + P_1.lp_solution.T).flatten())

    if not mute:
        # AFFICHAGE DU BILAN
        print(f"##### FIN !!! / DISTANCE = {distance} / EPSILON = {Epsilon_var} ###")

        # TOUT LES GRAPHES, OU LA MATRICE X SEUL
        ############################################################################################################################################
        ###                IL FAUT COMMENTER / DECOMMENTER UNE DES LIGNES SUIVANTES POUR NE PAS AFFICHER TOUS LES GRAPHES                        ###
        ###                le paramètre ungroup permet de regrouper les graphes sur une seule fenêtre si besoin                                  ###
        ############################################################################################################################################
        tsp.plot_array([X_list, H_list, Gamma_list, Epsilon_list, Rho_list, Cost_list, Global_cost_list, Eval_list, D_list], ["X_list", "H_list",  "Gamma_list", "Epsilon_list", "Rho_list", "Cost", "Global cost", "Eval", "D_list"], ungroup=True)
        # tsp.plot_array([X_list], ["X_list"], ungroup=True)

        print(" - Nombre de différence :", Nb_delta)
        print(" - Somme des différence :", Sm_delta)
        print(" - Différence de fonction objective :", np.round((Cost_result - Cost_best) / Cost_best * 10000) / 100, "%")
        print("Fonction f :", True_cost_result)
        print("Fonction f (arrondi) :", Cost_result)
        print("Excepted value : ", Cost_best)

        print("X :", np.round(X_var * 100) / 100)
        print("DELTA :", np.round(X_var) - P_1.lp_solution - P_1.lp_solution.T)
        
        print("Fonction h :", function_h(X_var.flatten()))
        print("Gamma :", Gamma_var)
        print("Rho :", Rho_var)
        print("Fonction L :", function_global_cost(X_var.flatten(), Gamma_var, Rho_var))

    return Nb_delta, Sm_delta, Cost_result, Cost_best, len(X_list), tsp.norm_fro(X_var - P_1.lp_solution + P_1.lp_solution.T)

# %% [markdown]
# # Réalisation de statistique :

# %%
def a_run(size=4, ratio=1, mute_stat=False, mute_opt=True, rounded=True, mute_run=False, **kargs):
    """
    Pour évaluer les écarts sur {ratio} % du dossier de taille {size}
    On peux rendre muet des parties :
        - mute_stat : les statistiques du résultat 
        - mute_opt : les informations de la fonction d'optimisation (graphiques, progression, et bilan)
        - mute_run : la barre de progression de cette fonction
    On peux passer des valeurs de simulation en paramètre (Epsilon_var, etc.)
    Le résultat prend en compte l'arrondi de la matrice final (rounded=True), ou la norme (fro) entre la matrice résultat et la matrice optinal
    """
    list_result = []

    the_path = f"data/size_{size:03d}/"
    list_of_file = sorted(os.listdir(the_path))
    nb_of_file = len(list_of_file)

    goal_rank = int(ratio / 100 * nb_of_file)

    P_1 = tsp.TSPProblem(6, 10)
    the_iterator = list_of_file[0:goal_rank] if mute_run else tqdm(list_of_file[0:goal_rank])

    for file_name in the_iterator:
        P_1.load(the_path + file_name)
        result_opt = the_big_solver(P_1, mute=mute_opt, **kargs)
        list_result += [abs(result_opt[3] - result_opt[2]) / result_opt[3] * 100 if rounded else result_opt[5]]

    if not mute_stat:
        print("  - Mean :", np.mean(list_result), "%")
        print("  - Std :", np.std(list_result), "%")
        print("  - Min :", np.min(list_result), "%")
        print("  - Max :", np.max(list_result), "%")
        print("  - List :", list_result)

    return np.mean(list_result)

# %% [markdown]
# # Simulation :

# %%
# On peux importer un fichier seul, afin de réaliser une résolution
# On peux même ne pas importer de fichiers, mais cela nécessite qu'un solveur (PULP et GUROBI) soit installé

P_1 = tsp.TSPProblem(6, 10)
# P_1.save("test.csv")
P_1.load("test.csv")

# P_1 = tsp.TSPProblem(8, 10)
# P_1.load("test_size_8.csv")

_ = P_1.lp_solve()

# %%
# On peux alors le solver avec les paramètres par défaut
print("Sans param (6)", the_big_solver(P_1))

# On peux aussi modifier les paramètres suivants
# - n_iter : nombre d'itération maximal
# - Rho_var : Coefficient de pénalité initial
# - Var_gamma_min : Borne supérieur des coefficients de contraintes d'égalité
# - Var_gamma_max : Borne inférieur des coefficients de contraintes d'égalité
# - Var_gamma : Valeur initiales des coefficients de contraintes d'égalité
# - Epsilon_var : Valeur initiale de la précision
# - Epsilon_min : Valeur minimale de la précision
# - Theta_epsilon : Ratio pour l'évolution de Epsilon
# - Theta_rho : Ratio pour l'évolution de Rho
# - Theta_sigma : Ratio pour les critères d'évolution de Rho
# - D_min : Valeur du pas minimal

# EX :
print("Sans param (6)", the_big_solver(P_1, n_iter=200))

# EX : Pondération forte sur les contraintes
print("Avec param (6)", the_big_solver(P_1, n_iter=200, Rho_var=10000))
# On ne varie pas pour ne pas s'éloigner de la contrainte

# EX : Pondération faible sur les contraintes
print("Avec param (6)", the_big_solver(P_1, n_iter=200, Rho_var=0.00001))
# On varie s'en se soucier de la contrainte

# %%
# On peut évaluer les simulations sur plusieurs instances 

# une seule instance (0.1% * 1000 = 1), rounded -> Le résultat est un pourcentage entre gurobi et le solveur RALM
print(a_run(size=6, mute_opt=False, ratio=0.1, rounded=True))

# 10 instances, not rounded -> Le résultat est la norme de frobenus entre gurobi et le solveur RALM
print(a_run(size=6, mute_opt=False, ratio=1.0, rounded=False))

# %%
# On peut analyser un paramètre sur un intervale (échelle log10)

# Borne inférieur (puissance 10)
borne_inf = -1
# Borne supérieur (puissance 10)
borne_sup = 1

# On réaliser un graphe pour Rho_var (échelle log), avec 20 instances de taille 6
val = []
ord = []
for i in np.logspace(borne_inf, borne_sup, 10):
    ord += [i]
    r = a_run(size=6, mute_opt=True, ratio=2, rounded=False, Rho_var=i, mute_stat=True)
    val += [r]
    print(i, ":", r)
plt.plot(ord, val)
plt.xscale("log")
plt.plot()

# On peut alors regarder les zones non pertinentes pour un paramètres (indépendament des autres)

# %%
print(a_run(size=6, ratio=1, rounded=False, mute_opt=False, mute_run=False))

# %%
print("Avec param (6)", the_big_solver(P_1, n_iter=200, Rho_var=10000, Theta_rho=0.80, Var_gamma_max=50))


