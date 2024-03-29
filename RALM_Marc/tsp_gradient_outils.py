# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:13:50 2022

@author: WANG Haokun
@author: CHATROUX Marc
@author: LI Yuansheng
"""

import csv
import time
from functools import partial
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import networkx as nx

from scipy.stats import ortho_group

try:
    from pulp import LpVariable, LpBinary, LpMinimize, LpProblem, lpSum, GUROBI_CMD, PULP_CBC_CMD, value
    import gurobi
except:
    print(" - PULP (ou GUROBI) non installé :")
    print("     La résolution optimale n'est pas disponible")
    print("     Les solutions optimales restent disponibles dans les fichiers d'instances\n")



## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ##
#                         FONCTION DE GRAPHIQUE AVANCE                        #
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ##

def plot_array(list_of_list_of_array, list_of_titre, ungroup=False):
    """
    Affiche l'évolution de n matrices (avec n titres) sur la même page (ou non)
    """
    n = len(list_of_list_of_array)
    n_1 = np.ceil(np.sqrt(n))
    n_2 = np.ceil(n / n_1)
    
    if ungroup:
        for rang, (list_of_array, titre) in enumerate(zip(list_of_list_of_array, list_of_titre)):
            fig = plt.figure(figsize=(12,6))
            dict_of_dict = {}
            for indice, element in enumerate(list_of_array):
                for position, valeur in np.ndenumerate(element):
                    cle = "Element " + str(position)
                    if cle not in dict_of_dict.keys():
                        dict_of_dict[cle] = {}
                    dict_of_dict[cle][indice] = valeur

            r_1 = rang // n_1
            r_2 = rang % n_1
            for cle, dictionnaire in dict_of_dict.items():
                lists = sorted(dictionnaire.items())
                x, y = zip(*lists)
                plt.plot(x, y, label=cle)
            plt.legend(loc=2, prop={"size":2})
            plt.title(titre)
            plt.show()
    else:
        fig, axs = plt.subplots(n_2, n_1, figsize=(15, 8))
        if n == 1:
            for rang, (list_of_array, titre) in enumerate(zip(list_of_list_of_array, list_of_titre)):
                dict_of_dict = {}
                for indice, element in enumerate(list_of_array):
                    for position, valeur in np.ndenumerate(element):
                        cle = "Element " + str(position)
                        if cle not in dict_of_dict.keys():
                            dict_of_dict[cle] = {}
                        dict_of_dict[cle][indice] = valeur

                r_1 = rang // n_1
                r_2 = rang % n_1
                for cle, dictionnaire in dict_of_dict.items():
                    lists = sorted(dictionnaire.items())
                    x, y = zip(*lists)
                    axs.plot(x, y, label=cle)
                axs.legend(loc=2, prop={"size":2})
                axs.set_title(titre)
        else:
            for rang, (list_of_array, titre) in enumerate(zip(list_of_list_of_array, list_of_titre)):
                dict_of_dict = {}
                for indice, element in enumerate(list_of_array):
                    for position, valeur in np.ndenumerate(element):
                        cle = "Element " + str(position)
                        if cle not in dict_of_dict.keys():
                            dict_of_dict[cle] = {}
                        dict_of_dict[cle][indice] = valeur

                r_1 = rang // n_1
                r_2 = rang % n_1
                for cle, dictionnaire in dict_of_dict.items():
                    lists = sorted(dictionnaire.items())
                    x, y = zip(*lists)
                    axs[r_1, r_2].plot(x, y, label=cle)
                axs[r_1, r_2].legend(loc=2, prop={"size":2})
                axs[r_1, r_2].set_title(titre)
        plt.show()



## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ##
#                            MATHEMATICAL TOOLS                               #
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ##

def std_lie_bracket(a_array, b_array):
    """
    Calculate a standard Lie bracket : [A, B] = A*B − B*A

    Parameters
    ----------
    a_array : a numpy array A
    b_array : a numpy array B

    Returns
    -------
    result : the standard Lie bracket of A and B

    """
    assert isinstance(a_array, np.ndarray), "Check the type of A"
    assert isinstance(b_array, np.ndarray), "Check the type of B"
    assert a_array.shape == b_array.shape, "Check the size link between A and B"
    result = a_array @ b_array - b_array @ a_array
    return result

def gnr_lie_bracket(a_array, b_array):
    """
    Calculate a generalized Lie bracket : {A, B} =  A^T * B − B^T * A

    Parameters
    ----------
    a_array : a numpy array A
    b_array : a numpy array B

    Returns
    -------
    result : the generalized Lie bracket of A and B

    """
    assert isinstance(a_array, np.ndarray), "Check the type of A"
    assert isinstance(b_array, np.ndarray), "Check the type of B"
    assert a_array.shape[0] == b_array.shape[0], "Check the size link between A and B"
    assert a_array.shape[1] == b_array.shape[1], "Check the size link between A and B"
    result = a_array.T @ b_array - b_array.T @ a_array
    return result

def indiv_product(a_array, b_array):
    """
    Calculate a term product of matrix : A_i,j * B_i,j for all i,j

    Parameters
    ----------
    a_array : a numpy array A
    b_array : a numpy array B

    Returns
    -------
    result : the term product of A and B matrix

    """
    assert isinstance(a_array, np.ndarray), "Check the type of A"
    assert isinstance(b_array, np.ndarray), "Check the type of B"
    assert a_array.shape[0] == b_array.shape[0], "Check the size link between A and B"
    assert a_array.shape[1] == b_array.shape[1], "Check the size link between A and B"
    result = np.multiply(a_array, b_array)
    return result

def trace(a_array):
    """
    Calculate a trace of matrix

    Parameters
    ----------
    a_array : a numpy array A

    Returns
    -------
    result : the trace of A

    """
    assert isinstance(a_array, np.ndarray), "Check the type of A"
    result = np.trace(a_array)
    return result

def exp_of_pade(a_array):
    """
    Calculate a Padé approximation to the matrix exponential
    exp(A) ~= (2*I - A) / (2*I + A)

    Parameters
    ----------
    a_array : a numpy array A

    Returns
    -------
    result : the Padé approximation of the exponential of A

    """
    assert isinstance(a_array, np.ndarray), "Check the type of A"
    assert a_array.shape[0] == a_array.shape[1  ], "Check the square size"
    identity = np.identity(a_array.shape[0])
    up_part = 2 * identity - a_array
    down_part = 2 * identity + a_array
    result = up_part @ np.linalg.inv(down_part)
    return result

def matrix_dir(size):
    """
    Return a direct size x size permutation matrix

    Parameters
    ----------
    size : integer
        the size of the matrix

    Returns
    -------
    the direct permutation matrix

    """
    assert isinstance(size, int), "Check the type of A"
    assert size > 0, "Check the type of A"
    array = np.zeros((size, size))
    array[size - 1, 0] = 1
    for var in range(0, size - 1):
        array[0 + var, 1 + var] = 1
    return array

def matrix_undir(size):
    """
    Return a undirect size x size permutation matrix

    Parameters
    ----------
    size : integer
        the size of the matrix

    Returns
    -------
    the undirect permutation matrix

    """
    assert isinstance(size, int), "Check the type of A"
    assert size > 0, "Check the type of A"
    array = np.zeros((size, size))
    array[size - 1, 0] = 1
    array[0, size - 1] = 1
    for var in range(0, size - 1):
        array[0 + var, 1 + var] = 1
        array[1 + var, 0 + var] = 1
    return array

def distance_to_permut_dir(array):
    """
    Return the distance of a matrix to the permutation matrix area

    Parameters
    ----------
    array : array
        a matrix

    Returns
    -------
    The distance

    """
    assert isinstance(array, np.ndarray), "Check the type of A"
    assert array.shape[0] == array.shape[1], "Check the size of A"
    f_array = array @ array.T
    f_array = f_array - np.identity(array.shape[0])
    return np.linalg.norm(f_array)

def distance_to_permut_undir(array):
    """
    Return the distance of a matrix to the permutation matrix area

    Parameters
    ----------
    array : array
        a matrix

    Returns
    -------
    The distance

    """
    assert isinstance(array, np.ndarray), "Check the type of A"
    assert array.shape[0] == array.shape[1], "Check the size of A"
    f_array = array @ array.T
    f_array = f_array - np.identity(array.shape[0])
    return np.linalg.norm(f_array)

def norm_fro(array):
    """
    Return the norm of frovenus

    Parameters
    ----------
    array : array
        a matrix

    Returns
    -------
    The norm

    """
    assert isinstance(array, np.ndarray), "Check the type of A"
    assert array.shape[0] == array.shape[1], "Check the size of A"
    return np.linalg.norm(array, "fro")



## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ##
#                                 TSP PROBLEM                                 #
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ##

class TSPProblem:
    """
    The class of a TSP Problem
    """

    def __init__(self, n, d):
        """
        Parameters
        ----------
        n : int
            number of cities.
        d : int
            size of the area.

        """
        assert isinstance(n, int), "Check the type of n (int)"
        assert isinstance(d, int), "Check the type of d (int)"
        # Save n and d
        self.__number = n
        self.__size = d
        # Generate (and save) an array of cities positions and an array of distances
        self.__cities_array = self.__generate_cities()
        # Matrice de coût
        self.__cost_array = self.__generate_cost_array()
        # Solution orienté
        self.lp_solution = None
        # Coût de la solution
        self.lp_solution_cost = None
        # Détails de la résolution (solveur et ttemps de résolution)
        self.lp_solution_details = None

    def __repr__(self):
        """
        Returns
        -------
        Return a representation of the problem

        """
        return str(self)

    def __str__(self):
        """
        Returns
        -------
        Return a representation of the problem in string format.

        """
        text = "TSP_Problem :\n"
        text += f"  - Number of cities : {self.__number}\n"
        text += f"  - Size of area : {self.__size}\n"
        text += f"  - Positions of cities : {self.__cities_array.shape}\n      "
        text += str(self.__cities_array).replace("\n", "\n      ") + "\n"
        text += f"  - Distances array :  {self.__cost_array.shape}\n      "
        text += str(self.__cost_array).replace("\n", "\n      ") + "\n"
        if not isinstance(self.lp_solution, type(None)):
            text += f"  - Solution array :  {self.lp_solution.shape}\n      "
            text += str(self.lp_solution).replace("\n", "\n      ") + "\n"
        else:
            text += f"  - Solution array :  {self.lp_solution}\n"
        text += f"  - Solution value :  {self.lp_solution_cost}\n"
        text += f"  - Solution details :  {self.lp_solution_details}\n      "
        return text

    def __generate_cities(self):
        """
        Generate an n*2 array of random number between 0 and d

        Returns
        -------
        array : cities positions

        """
        array = np.random.rand(self.__number, 2) * self.__size
        return array

    def __generate_cost_array(self):
        """
        Generate an n*n array of distance cost (2-norm)

        Returns
        -------
        array : cost (distance)

        """
        array = np.zeros((self.__number, self.__number))
        for var_1 in range(0, self.__number):
            for var_2 in range(0, self.__number):
                vector = self.__cities_array[var_1] - self.__cities_array[var_2]
                # Norme 2 par défaut
                array[var_1, var_2] = np.linalg.norm(vector)
        return array

    def load(self, filename):
        """
        Load a TSP Problem from a file

        Parameters
        ----------
        filename : the filename of the csv that we want to load

        """
        with open(filename, 'r', newline='', encoding="UTF-8") as csvfile:
            reader = list(csv.reader(csvfile, quoting=csv.QUOTE_MINIMAL))
            indice = 0

            # Le nombre de points
            self.__number = int(reader[indice][0])
            indice += 1

            # La taille de la zone
            self.__size = int(reader[indice][0])
            indice += 1

            # Les positions des points
            self.__cities_array = np.zeros([self.__number, 2])
            for row in range(0, self.__number):
                for column in range(0, 2):
                    self.__cities_array[row, column] = float(reader[indice][column])
                indice += 1

            # La matrice de coût
            self.__cost_array = np.zeros([self.__number, self.__number])
            for row in range(0, self.__number):
                for column in range(0, self.__number):
                    self.__cost_array[row, column] = float(reader[indice][column])
                indice += 1

            # On regarde si on dispose d'une solution optimale
            is_solved = bool(reader[indice])
            indice += 1

            if is_solved:
                # Solution optimal (orienté)
                self.lp_solution = np.zeros([self.__number, self.__number])
                for row in range(0, self.__number):
                    for column in range(0, self.__number):
                        self.lp_solution[row, column] = float(reader[indice][column])
                    indice += 1
                # Coût associé
                self.lp_solution_cost = float(reader[indice][0])
                indice += 1
                # Détails (solveur et temps de résolution)
                self.lp_solution_details = [reader[indice][0], float(reader[indice][1])]
                indice += 1
            else:
                # On purge les variables de solution
                self.lp_solution = None
                self.lp_solution_cost = None
                self.lp_solution_details = None

    def save(self, filename):
        """
        Save a TSP Problem on a file

        Parameters
        ----------
        filename : the filename of the csv where we want to save

        """
        with open(filename, 'w', newline='', encoding="UTF-8") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            # Le nombre de points
            writer.writerow([self.__number])
            # La taille de la zone
            writer.writerow([self.__size])
            # Les positions des points
            writer.writerows(self.__cities_array.tolist())
            # La matrice de coût
            writer.writerows(self.__cost_array.tolist())

            is_solved = not isinstance(self.lp_solution, type(None))
            # Booléen (True si on dispose d'une solution optimal)
            writer.writerow([is_solved])
            if is_solved:
                # La solution optimal (orienté)
                writer.writerows(self.lp_solution.tolist())
                # La coût (orienté)
                writer.writerow([self.lp_solution_cost])
                # Les détails (solveur, et temps de résolution)
                writer.writerow(self.lp_solution_details)
            


    def get_number(self):
        """
        Returns
        -------
        result : the number of cities

        """
        return self.__number

    def get_size(self):
        """
        Returns
        -------
        result : the size of the area

        """
        return self.__size

    def get_cities_array(self):
        """
        Returns
        -------
        result : the array of the positions of cities

        """
        return self.__cities_array

    def get_cost_array(self):
        """
        Returns
        -------
        result : the array of the distances between cities

        """
        return self.__cost_array

    def lp_solve(self, force=False):
        """
        Calculate the optimum solution with Linear programming
        Using an order constraint
        """
        # Uniquement si la solution n'existe pas encore
        # Ou que l'on force
        # Un "load" permet d'inporter l'ancienne solution  
        if (self.lp_solution is None) or force:
            t0 = time.time()

            C = self.get_cost_array()
            I = list(range(len(C)))

            x = {}
            for i in I:
                for j in I:
                    x[i, j] = LpVariable(f"x({i, j})", cat=LpBinary)

            u = {}
            for i in I:
                u[i] = LpVariable(f"u({i})")

            prob = LpProblem("ordo", LpMinimize)
            prob += lpSum(C[i, j] * x[i, j] for i in I for j in I)

            #C1
            for i in I:
                prob += lpSum([ x[(i,j)] for j in I])==1, f"C1_{i}"

            #C2
            for j in I:
                prob += lpSum([ x[(i,j)] for i in I])==1, f"C2_{j}"

            #C3
            for i in I[1:]:
                for j in I[1:]:
                    prob += u[i] - u[j] + I[-1] * x[i, j] <= I[-1] - 1, f"C3_{i, j}"

            # On enregistre les détails de la résolution (solveur et temps de résolution)
            try:
                prob.solve(GUROBI_CMD(msg=0))
                t_fin = time.time()
                self.lp_solution_details = ["GUROBI", t_fin - t0]
            except:
                prob.solve(PULP_CBC_CMD(msg=0))
                t_fin = time.time()
                self.lp_solution_details = ["PULP_CBC", t_fin - t0]
            # On enregistre le résultat (orienté) et le coût associé
            self.lp_solution = np.array([[x[i, j].varValue for i in I] for j in I])
            self.lp_solution_cost = value(prob.objective)
        return self.lp_solution


# ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ##
# #                                   SOLVER 1                                  #
# ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ##

# class OrientedSolver:
#     """
#     Solver from (12) of "Continuous relaxations for the traveling salesman problem"
#     """

#     def __init__(self, problem, pas = 0.00001):
#         """
#         Parameters
#         ----------
#         problem : TSPProblem
#             The TSP problem to solve

#         """

#         self.__problem = problem
#         # self.__pas = pas
#         # self.__number = problem.get_number()

#         self.__solution_history = []
#         self.__solution_turn_history = []
#         self.__lambda_history = []

#         self.grad_lbd_history = []
#         self.grad_history = []

#         self.__A = matrix_undir(problem.get_number())
#         self.__save_new_solution(ortho_group.rvs(problem.get_number()))

#         self.__N = None
#         self.base_change = None
#         self.get_N()

#         self.__alpha = None

#     def __str__(self):
#         text = "OrientedSolver :\n"
#         text += "  - Problem : "
#         text += ("\n" + str(self.__problem)).replace('\n', '\n            ') + "\n"
#         text += f"  - Actual solution array : {self.get_current_solution().shape}\n      "
#         text += str(self.get_current_solution()).replace("\n", "\n      ") + "\n"
#         text += f"  - Actual lambda : {self.get_current_lambda()}\n"
#         return text

#     # def __generate_default_solution(self):
#     #     """
#     #     Set the start array

#     #     """
#     #     array = np.zeros((self.__number, self.__number))
#     #     array[self.__number - 1, 0] = 1
#     #     for var in range(0, self.__number - 1):
#     #         array[0 + var, 1 + var] = 1
#     #     self.__save_new_solution(array)

#     # def __generate_default_lambda(self):
#     #     """
#     #     Set the start lambda

#     #     """
#     #     self.__save_new_lambda(1)

#     def get_current_solution(self):
#         """
#         Get the current solution i.e. the last solution of the history

#         """
#         return self.__solution_history[-1]

#     def get_solutions_history(self):
#         """
#         Get the history of the solution

#         """
#         return self.__solution_history

#     def __save_new_solution(self, p_array):
#         """
#         Save a new solution at the end of the history

#         """
#         self.__solution_history += [p_array]

#     def get_current_turn_solution(self):
#         """
#         Get the current solution i.e. the last solution of the history

#         """
#         return self.__solution_turn_history[-1]

#     def get_turn_solutions_history(self):
#         """
#         Get the history of the solution

#         """
#         return self.__solution_turn_history

#     def __save_new_turn_solution(self, turn_array):
#         """
#         Save a new solution at the end of the history

#         """
#         self.__solution_turn_history += [turn_array]

#     def get_current_lambda(self):
#         """
#         Get the current lambda i.e. the last lambda of the history

#         """
#         return self.__lambda_history[-1]

#     def get_solutions_lambda(self):
#         """
#         Get the history of the solution

#         """
#         return self.__lambda_history

#     def __save_new_lambda(self, lambda_var):
#         """
#         Save a new lambda at the end of the history

#         """
#         self.__lambda_history += [lambda_var]

#     # def new_solution(self, solution_var, lambda_var):
#     #     """
#     #     Calculate the futur solution
#     #         ̇ P = − P * ({ P^T * B * P, A} + { P^T B^T P, A^T })
#     #               − λ * P * ( (P ◦ P)^T * P − P^T * (P ◦ P) )

#     #     Parameters
#     #     ----------
#     #     solution_var : array
#     #     lambda_var : float

#     #     Returns
#     #     -------
#     #     an array
#     #         The new solution.

#     #     """
#     #     P = solution_var
#     #     lbd = lambda_var
#     #     A = self.__problem.get_cost_array()
#     #     B = self.__A

#     #     p_dot = -1 * P @ (gnr_lie_bracket(P.T @ B @ P, A) + gnr_lie_bracket(P.T @ B.T @ P, A.T))
#     #     p_dot = p_dot - lbd * P @ (indiv_product(P, P).T @ P - P.T @ indiv_product(P, P))
#     #     # print(p_dot)
#     #     self.grad_history += [np.linalg.norm(p_dot, 'fro')]
#     #     new_p = solution_var - self.__pas * p_dot

#     #     # print(new_p)
#     #     # return self.__solution_history[0] * lambda_var / 4
#     #     return new_p

#     # def new_lambda(self, solution_var, lambda_var):
#     #     """
#     #     Calculate the futur lambda
#     #         ̇ λ= 1/3 * tr( P^T * (P − (P ◦ P)) )

#     #     Parameters
#     #     ----------
#     #     solution_var : array
#     #     lambda_var : float

#     #     Returns
#     #     -------
#     #     an array
#     #         The new lambda.

#     #     """
#     #     P = solution_var

#     #     lbd_dot = 1/3 * trace(P.T @ (P - (indiv_product(P, P))))
#     #     new_lbd = lambda_var - self.__pas * lbd_dot
#     #     self.grad_lbd_history += [lbd_dot]

#     #     # print(new_lbd)
#     #     # return lambda_var + 0.01
#     #     return new_lbd

#     def get_N(self):
#         """


#         Returns
#         -------
#         the N array

#         """
#         if self.__N is None or self.base_change is None:
#             C = self.__problem.get_cost_array()

#             eigenValues, eigenVectors = np.linalg.eig(C)
#             # print(eigenValues)
#             # print(eigenVectors)

#             idx = eigenValues.argsort()[::1]
#             eigenValues = eigenValues[idx]
#             eigenVectors = eigenVectors[:,idx]
#             # print(eigenValues)
#             # print(eigenVectors)

#             self.__N = np.diag(eigenValues)
#             self.base_change = eigenVectors
#             print("\nN : \n", self.__N)
#             print("\nChangement de base : \n", self.base_change)
#         return self.__N

#     def get_alpha(self):
#         """


#         Returns
#         -------
#         alpha

#         """
#         if self.__alpha is None:
#             A = self.__A
#             N = self.get_N()
#             alpha = 1 / (4 * trace(A @ A.T)**1/2 * trace(N @ N.T)**1/2)
#             self.__alpha = alpha
#         return self.__alpha

#     # def new_solution_exp_old(self, solution_var, lambda_var):
#     #     """


#     #     Parameters
#     #     ----------
#     #     solution_var : array
#     #         the old array

#     #     Returns
#     #     -------
#     #     new_p : array
#     #         the new array

#     #     """
#     #     P = solution_var
#     #     H_zero = self.__A
#     #     N = self.get_N()
#     #     alpha = self.get_alpha()

#     #     lie_bracket = std_lie_bracket(P.T @ H_zero @ P, N)

#     #     new_p = P @ exp_of_pade( alpha * lie_bracket)

#     #     # print(new_p)
#     #     # return self.__solution_history[0] * lambda_var / 4
#     #     return new_p

#     # def new_lambda_exp_old(self, solution_var, lambda_var):
#     #     """


#     #     Parameters
#     #     ----------
#     #     solution_var : array
#     #         the old array

#     #     Returns
#     #     -------
#     #     new_p : array
#     #         the new array

#     #     """
#     #     P = solution_var

#     #     lbd_dot = 1/3 * trace(P.T @ (P - (indiv_product(P, P))))
#     #     new_lbd = lambda_var + self.__pas * lbd_dot

#     #     # print(new_lbd)
#     #     # return lambda_var + 0.01
#     #     return new_lbd

#     def new_solution_exp(self, solution_var):
#         """


#         Parameters
#         ----------
#         solution_var : array
#             the old array

#         Returns
#         -------
#         new_p : array
#             the new array

#         """

#         # P = self.base_change.T @ solution_var @ self.base_change
#         P_B = solution_var
#         P_D = self.base_change.T @ P_B @ self.base_change
#         H_zero_B = self.__A
#         H_zero_D = self.base_change.T @ H_zero_B @ self.base_change
#         H_D = P_D.T @ H_zero_D @ P_D
#         N_D = self.get_N()

#         # alpha = self.get_alpha()
#         alpha_k = 1 / 2 / norm_fro(std_lie_bracket(H_D, N_D))
#         alpha_k *= np.log((norm_fro(std_lie_bracket(H_D, N_D))**2 / norm_fro(P_D) / norm_fro(std_lie_bracket(N_D, std_lie_bracket(H_D, N_D)))) + 1)

#         H_and_N_D = std_lie_bracket(H_D, N_D)
#         new_p = P_D @ exp_of_pade( -1 * alpha_k * H_and_N_D)
        
#         self.grad_history += [norm_fro(H_and_N_D)]
#         self.__save_new_lambda(alpha_k)

#         # return new_p
#         return self.base_change @ new_p @ self.base_change.T

#     def breaker(self):
#         """
#         Calculate the breaker value

#         Parameters
#         ----------
#         solution_var : array
#         lambda_var : float

#         Returns
#         -------
#         a float

#         """
#         try:
#             return self.grad_history[-1]
#         except:
#             return 100

#     def lagrange(self, limit = 1e-8, n_max=100000):
#         """
#         Lagrange solving algorythme

#         """

#         count = 0
#         while self.breaker() > limit and count < n_max:
#             count += 1
#             ### Lagrange
#             # self.__save_new_solution(
#             #     self.new_solution(
#             #         self.get_current_solution(), self.get_current_lambda()))
#             # self.__save_new_turn_solution(
#             #     self.get_current_solution().T @ self.__A @ self.get_current_solution())
#             # self.__save_new_lambda(
#             #     self.new_lambda(
#             #         self.get_current_solution(), self.get_current_lambda()))
#             ### Exponential 1
#             # self.__save_new_solution(
#             #     self.new_solution_exp_old(
#             #         self.get_current_solution(), self.get_current_lambda()))
#             # self.__save_new_turn_solution(
#             #     self.get_current_solution().T @ self.__A @ self.get_current_solution())
#             # self.__save_new_lambda(
#             #     self.new_lambda_exp_old(
#             #         self.get_current_solution(), self.get_current_lambda()))
#             ### Exponential de tour
#             self.__save_new_solution(
#                 self.new_solution_exp(
#                     self.get_current_solution()))
#             self.__save_new_turn_solution(
#                 self.get_current_solution().T @ self.__A @ self.get_current_solution())
#             # print(self.cost(-1))

#     def solve(self):
#         """
#         Resolve the TSP Problem

#         """
#         self.lagrange(limit=1e-8, n_max = 1e5)
#         self.__save_new_turn_solution(self.__problem.lp_solve())
#         print("\n - Current solution : \n", self.get_current_solution())
#         print("\n - Current turn solution : \n", self.get_turn_solutions_history()[-2])
#         print("\n - Correct turn solution : \n", self.get_turn_solutions_history()[-1])

#     def plot(self):
#         """
#         Plot the solution evolution

#         """

#         fig = plt.figure()
#         # For slider and buttons
#         area_01 = fig.add_axes([0.02, 0.20, 0.09, 0.72])
#         area_02 = fig.add_axes([0.07, 0.04, 0.04, 0.08])
#         area_03 = fig.add_axes([0.02, 0.04, 0.04, 0.08])
#         # For networks
#         area_11 = fig.add_axes([0.14, 0.04, 0.25, 0.42])
#         area_12 = fig.add_axes([0.14, 0.54, 0.25, 0.42])
#         # For plotting values variations
#         area_21 = fig.add_axes([0.44, 0.04, 0.25, 0.42])
#         area_21.set_title('Lambda history')
#         area_22 = fig.add_axes([0.44, 0.54, 0.25, 0.42])
#         area_22.set_title('P_dot history')
#         area_31 = fig.add_axes([0.74, 0.04, 0.23, 0.42])
#         area_31.set_title('Cost history')
#         area_32 = fig.add_axes([0.74, 0.54, 0.23, 0.42])
#         area_32.set_title('Lambda_dot history')

#         list_distance = []
#         list_norm = []
#         list_cost = []
#         for i in range(len(self.get_turn_solutions_history()) - 1):
#             list_distance += [distance_to_permut_undir(self.__solution_turn_history[i])]
#             list_norm += [norm_fro(self.__solution_turn_history[i])]
#             list_cost += [self.cost(i)]

#         area_21.plot(self.__lambda_history)
#         area_31.plot(list_cost)
#         area_31.plot(
#             [0, len(self.get_turn_solutions_history()) - 2], [2*self.cost(-1), 2*self.cost(-1)])
#         area_22.plot(self.grad_history)
#         area_32.plot(self.grad_lbd_history)
#         area_32.plot(list_norm)
#         area_32.twinx().plot(list_distance)

#         area_31.annotate('%0.2f' % list_cost[-1], xy=(0.98, list_cost[-1]), xytext=(8, 0),
#                          xycoords=('axes fraction', 'data'), textcoords='offset points')
#         area_31.annotate('%0.2f' % (2 * self.cost(-1)), xy=(0.98, 2 * self.cost(-1)), xytext=(8, 0),
#                          xycoords=('axes fraction', 'data'), textcoords='offset points')
#         area_22.annotate('{:.5E}'.format(self.grad_history[-1]), xy=(0.95, self.grad_history[-1]), xytext=(8, 0),
#                          xycoords=('axes fraction', 'data'), textcoords='offset points')


#         intervale_min = 0
#         intervale_max = len(self.__solution_turn_history) - 2
#         intervale_init = intervale_max


#         list_pos = {}
#         for i in range(self.__problem.get_number()):
#             list_pos[i] = np.array(
#                 [self.__problem.get_cities_array()[i, 0],
#                  self.__problem.get_cities_array()[i, 1]])

#         def get_edge(index_solution):
#             edge = []
#             array = self.__solution_turn_history[index_solution]
#             for i in range(self.__problem.get_number()):
#                 for j in range(self.__problem.get_number()):
#                     if array[i, j] > 0.01:
#                         edge += [(i, j, {"weight":int(array[i, j] * 100) / 50, "color":'black'})]
#                     if array[i, j] < -0.01:
#                         edge += [(i, j, {"weight":int(array[i, j] * 100) / 50, "color":'r'})]
#             # print(edge)
#             return edge


#         def graph_network(graph, v_list_pos, v_index, v_ax):
#             graph.clear_edges()
#             graph.update(edges=get_edge(v_index))
#             nx.draw_networkx_nodes(graph, v_list_pos, node_size=150, ax=v_ax)
#             widthlist = list(nx.get_edge_attributes(graph,'weight').values())
#             nx.draw_networkx_edges(graph, v_list_pos, width=widthlist, ax=v_ax)
#             colors = nx.get_edge_attributes(graph,'color').values()
#             nx.draw_networkx_edges(graph, v_list_pos, edge_color=colors, ax=v_ax)
#             nx.draw_networkx_labels(graph, v_list_pos, font_size=10, ax=v_ax)
#             v_ax.set_title('Flow')

#         network_graph = nx.DiGraph()
#         network_graph_2 = nx.DiGraph()
#         graph_network(network_graph, list_pos, intervale_init, area_11)
#         graph_network(network_graph_2, list_pos, intervale_init + 1, area_12)
#         area_12.set_title('Solution')


#         slider = Slider(ax=area_01,
#                         label="Itération",valmin=intervale_min,
#                         valmax=intervale_max,
#                         valinit=intervale_init,
#                         orientation="vertical",
#                         valstep=1)
#         b_add = Button(area_02, '+')
#         b_sub = Button(area_03, '-')

#         def update(v_fig, v_ax, v_graph, v_list_pos, val):
#             v_ax.cla()
#             graph_network(v_graph, v_list_pos, val, v_ax)
#             v_fig.canvas.draw_idle()

#         def act_add(v_fig, v_ax, v_graph, v_list_pos, v_slider, val):
#             v_slider.set_val(min(v_slider.val + 1, v_slider.valmax))
#             update(v_fig, v_ax, v_graph, v_list_pos, v_slider.val)

#         def act_sub(v_fig, v_ax, v_graph, v_list_pos, v_slider, val):
#             v_slider.set_val(max(v_slider.val - 1, v_slider.valmin))
#             update(v_fig, v_ax, v_graph, v_list_pos, v_slider.val)

#         slider.on_changed(partial(update, fig, area_11, network_graph, list_pos))
#         b_add.on_clicked(partial(act_add, fig, area_11, network_graph, list_pos, slider))
#         b_sub.on_clicked(partial(act_sub, fig, area_11, network_graph, list_pos, slider))

#         # manager = plt.get_current_fig_manager()
#         # manager.full_screen_toggle()
#         plt.show()

#     def cost(self, v_index):
#         """
#         Calculate the cost value of a solution in history

#         Parameters
#         ----------
#         v_index : integer

#         Returns
#         -------
#         a float
#             The cost value of the v_index solution of history

#         """
#         # P = self.get_solutions_history()[v_index]
#         # result = trace(self.__problem.get_cost_array().T @ P.T @ self.__A @ P)
#         P = self.get_turn_solutions_history()[v_index]
#         result = trace(self.__problem.get_cost_array().T @ P)
#         return result

#     def all_cost_plot(self):
#         """
#         Generete the cost plot of all doubly stochastic matrix

#         """
#         costs = []
#         for element in list(permutations(list(range(self.__problem.get_number())))):
#             array = np.zeros((self.__problem.get_number(), self.__problem.get_number()))
#             # print(array)
#             for x, y in enumerate(element):
#                 array[x, y] = 1
#             costs += [trace(self.__problem.get_cost_array().T @ array)]
#         plt.plot(costs)
#         the_cost = trace(self.__problem.lp_solve() @ self.__problem.get_cost_array().T)
#         plt.plot([0, len(costs) - 1], [the_cost, the_cost])
#         plt.show()


## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ##

if __name__ == "__main__":

    P_1 = TSPProblem(6, 100)

    # P_1.save("test.csv")
    P_1.load("test.csv")
    P_1.lp_solve()
    print(P_1)

    # P_2 = TSPProblem(50, 200)
    # print(P_2)

    # P_3 = TSPProblem(500, 2000)
    # print(P_3)

    # a = OrientedSolver(P_1)
    # # print(a)
    # a.solve()
    # # print(a)
    # a.plot()
    # print("for debug")
    # # a.all_cost_plot()
