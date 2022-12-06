# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:13:50 2022

@author: WANG Haokun
@author: CHATROUX Marc
@author: LI Yuansheng
"""

import csv
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import networkx as nx
from pulp import LpVariable, LpBinary, LpMinimize, LpProblem, lpSum, GUROBI_CMD
# import gurobi



## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ##
#                            MATHEMATICAL TOOLS                               #
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ##

def std_lie_bracket(a_array, b_array):
    """
    Calculate a standard Lie bracket : [A, B] = A*B − B*A

    Parameters
    ----------
    a_array : a numpy array
    b_array : a numpy array

    Returns
    -------
    result : the standard Lie bracket

    """
    assert isinstance(a_array, np.ndarray), "Check the type of A"
    assert isinstance(b_array, np.ndarray), "Check the type of B"
    assert a_array.shape == b_array.shape, "Check the size link between A and B"
    result = np.dot(a_array, b_array) - np.dot(b_array, a_array)
    return result


def gnr_lie_bracket(a_array, b_array):
    """
    Calculate a generalized Lie bracket : {A, B} =  A^T * B − B^T * A

    Parameters
    ----------
    a_array : a numpy array
    b_array : a numpy array

    Returns
    -------
    result : the generalized Lie bracket

    """
    assert isinstance(a_array, np.ndarray), "Check the type of A"
    assert isinstance(b_array, np.ndarray), "Check the type of B"
    assert a_array.shape[0] == b_array.shape[0], "Check the size link between A and B"
    assert a_array.shape[1] == b_array.shape[1], "Check the size link between A and B"
    result = np.dot(a_array.transpose(), b_array) - np.dot(b_array.transpose(), a_array)
    return result


def indiv_product(a_array, b_array):
    """
    Calculate a term product of matrix : a_i,j * b_i,j for all i,j

    Parameters
    ----------
    a_array : a numpy array
    b_array : a numpy array

    Returns
    -------
    result : the term product of matrix

    """
    assert isinstance(a_array, np.ndarray), "Check the type of A"
    assert isinstance(b_array, np.ndarray), "Check the type of B"
    assert a_array.shape[0] == b_array.shape[0], "Check the size link between A and B"
    assert a_array.shape[1] == b_array.shape[1], "Check the size link between A and B"
    result = np.multiply(a_array, b_array)
    return result


def trace(a_array):
    """
    Calculate a trace of matrix : a_i,j * b_i,j for all i,j

    Parameters
    ----------
    a_array : a numpy array

    Returns
    -------
    result : the trace of a_array

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
    a_array : a numpy array

    Returns
    -------
    result : the Padé approximation to the matrix exponential

    """
    assert isinstance(a_array, np.ndarray), "Check the type of A"
    assert a_array.shape[0] == a_array.shape[1  ], "Check the square size"
    identity = np.identity(a_array.shape[0])
    up = 2 * identity - a_array
    down = 2 * identity + a_array
    result = up @ np.linalg.inv(down)
    return result



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
        self.__cost_array = self.__generate_cost_array()
        self.lp_solution = None

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
        text += str(self.__cost_array).replace("\n", "\n      ")
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
                array[var_1, var_2] = np.linalg.norm(vector)
        return array

    def load(self, filename):
        """
        Load a TSP Problem from a file

        Parameters
        ----------
        filename : the filename of the csv that we want to load

        """
        with open(filename, 'r', newline='') as csvfile:
            reader = list(csv.reader(csvfile, quoting=csv.QUOTE_MINIMAL))
            self.__number = int(reader[0][0])
            self.__size = int(reader[1][0])
            self.__cities_array = np.zeros([self.__number, 2])
            for row in range(0, self.__number):
                for column in range(0, 2):
                    self.__cities_array[row, column] = float(reader[2 + row][column])
            self.__cost_array = np.zeros([self.__number, self.__number])
            for row in range(0, self.__number):
                for column in range(0, self.__number):
                    self.__cost_array[row, column] = float(reader[2 + self.__number + row][column])
            self.lp_solution = None

    def save(self, filename):
        """
        Save a TSP Problem on a file

        Parameters
        ----------
        filename : the filename of the csv where we want to save

        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow([self.__number])
            writer.writerow([self.__size])
            writer.writerows(self.__cities_array.tolist())
            writer.writerows(self.__cost_array.tolist())

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

    def lp_solve(self):
        """
        Calculate the optimum solution with Linear programming

        """
        if self.lp_solution is None:
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
                    print(i, j)
                    prob += u[i] - u[j] + I[-1] * x[i, j] <= I[-1] - 1, f"C3_{i, j}"
                    
            # print(prob)
            prob.solve(GUROBI_CMD(msg=1))
            self.lp_solution = np.array([[x[i, j].varValue for i in I] for j in I])
        return self.lp_solution


## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ##
#                                   SOLVER 1                                  #
## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ##

class OrientedSolver:
    """
    Solver from (12) of "Continuous relaxations for the traveling salesman problem"
    """

    def __init__(self, problem, pas = 0.0001):
        """
        Parameters
        ----------
        problem : TSPProblem
            The TSP problem to solve

        """

        self.__problem = problem
        self.__pas = pas
        self.__number = problem.get_number()
        self.__solution_history = []
        self.__solution_turn_history = []
        self.__lambda_history = []
        self.__A = None
        self.__generate_default_solution()
        self.__generate_default_lambda()
        
        self.__N = None
        self.__alpha = None

    def __str__(self):
        text = "OrientedSolver :\n"
        text += "  - Problem : "
        text += ("\n" + str(self.__problem)).replace('\n', '\n            ') + "\n"
        text += f"  - Actual solution array : {self.get_current_solution().shape}\n      "
        text += str(self.get_current_solution()).replace("\n", "\n      ") + "\n"
        text += f"  - Actual lambda : {self.get_current_lambda()}\n"
        return text

    def __generate_default_solution(self):
        """
        Set the start array

        """
        array = np.zeros((self.__number, self.__number))
        array[self.__number - 1, 0] = 1
        for var in range(0, self.__number - 1):
            array[0 + var, 1 + var] = 1
        self.__A = array
        self.__save_new_solution(array)
        self.__save_new_turn_solution(array.T @ self.__A @ array)

    def __generate_default_lambda(self):
        """
        Set the start lambda

        """
        self.__save_new_lambda(1)

    def get_current_solution(self):
        """
        Get the current solution i.e. the last solution of the history

        """
        return self.__solution_history[-1]

    def get_solutions_history(self):
        """
        Get the history of the solution

        """
        return self.__solution_history

    def __save_new_solution(self, p_array):
        """
        Save a new solution at the end of the history

        """
        self.__solution_history += [p_array]

    def get_current_turn_solution(self):
        """
        Get the current solution i.e. the last solution of the history

        """
        return self.__solution_turn_history[-1]

    def get_turn_solutions_history(self):
        """
        Get the history of the solution

        """
        return self.__solution_turn_history

    def __save_new_turn_solution(self, turn_array):
        """
        Save a new solution at the end of the history

        """
        self.__solution_turn_history += [turn_array]

    def get_current_lambda(self):
        """
        Get the current lambda i.e. the last lambda of the history

        """
        return self.__lambda_history[-1]

    def get_solutions_lambda(self):
        """
        Get the history of the solution

        """
        return self.__lambda_history

    def __save_new_lambda(self, lambda_var):
        """
        Save a new lambda at the end of the history

        """
        self.__lambda_history += [lambda_var]

    def new_solution(self, solution_var, lambda_var):
        """
        Calculate the futur solution
            ̇ P = −P * ({ P^T * B * P, A} + { P^T B^T P, A^T }) − λ * P * ( (P ◦ P)^T * P − P^T * (P ◦ P) )

        Parameters
        ----------
        solution_var : array
        lambda_var : float

        Returns
        -------
        an array
            The new solution.

        """
        P = solution_var
        lbd = lambda_var
        A = self.__problem.get_cost_array()
        B = self.__A

        p_dot = -1 * P @ (gnr_lie_bracket(P.T @ B @ P, A) + gnr_lie_bracket(P.T @ B.T @ P, A.T))
        p_dot = p_dot - lbd * P @ (indiv_product(P, P).T @ P - P.T @ indiv_product(P, P))
        new_p = solution_var - self.__pas * p_dot

        # print(new_p)
        # return self.__solution_history[0] * lambda_var / 4
        return new_p

    def new_lambda(self, solution_var, lambda_var):
        """
        Calculate the futur lambda
            ̇ λ= 1/3 * tr( P^T * (P − (P ◦ P)) )

        Parameters
        ----------
        solution_var : array
        lambda_var : float

        Returns
        -------
        an array
            The new lambda.

        """
        P = solution_var

        lbd_dot = 1/3 * trace(P.T @ (P - (indiv_product(P, P))))
        new_lbd = lambda_var - self.__pas * lbd_dot

        # print(new_lbd)
        # return lambda_var + 0.01
        return new_lbd

    def get_N(self):
        """
        

        Returns
        -------
        None.

        """
        if self.__N is None:
            C = self.__problem.get_cost_array()
            self.__N = C
        return self.__N
        
    def get_alpha(self):
        """
        
        
        Returns
        -------
        None.

        """
        if self.__alpha is None:
            A = self.__A
            N = self.get_N()
            alpha = 1 / (4 * trace(A @ A.T)**1/2 * trace(N @ N.T)**1/2)
            self.__alpha = alpha
        return self.__alpha
        

    def new_solution_exp(self, solution_var, lambda_var):
        """
        Calculate the futur solution
            ̇ P = −P * ({ P^T * B * P, A} + { P^T B^T P, A^T }) − λ * P * ( (P ◦ P)^T * P − P^T * (P ◦ P) )

        Parameters
        ----------
        solution_var : array
        lambda_var : float

        Returns
        -------
        an array
            The new solution.

        """
        lbd = lambda_var
        
        P = solution_var
        H_zero = self.__A
        N = self.get_N()
        alpha = self.get_alpha()

        lie_bracket = std_lie_bracket(P.T @ H_zero @ P, N)
        
        new_p = P @ exp_of_pade( alpha * lie_bracket)

        # print(new_p)
        # return self.__solution_history[0] * lambda_var / 4
        return new_p

    def new_lambda_exp(self, solution_var, lambda_var):
        """
        Calculate the futur lambda
            ̇ λ= 1/3 * tr( P^T * (P − (P ◦ P)) )

        Parameters
        ----------
        solution_var : array
        lambda_var : float

        Returns
        -------
        an array
            The new lambda.

        """
        P = solution_var

        lbd_dot = 1/3 * trace(P.T @ (P - (indiv_product(P, P))))
        new_lbd = lambda_var + self.__pas * lbd_dot

        # print(new_lbd)
        # return lambda_var + 0.01
        return new_lbd

    def breaker(self, solution_var, lambda_var):
        """
        Calculate the breaker value

        Parameters
        ----------
        solution_var : array
        lambda_var : float

        Returns
        -------
        a float

        """
        return 100

    def lagrange(self, limit = 1e-8, n_max=100000):
        """
        Lagrange solving algorythme

        """

        count = 0
        while self.breaker(self.get_current_solution(), self.get_current_lambda()) > limit and count < n_max:
            count += 1
            ### Lagrange
            self.__save_new_solution(self.new_solution(self.get_current_solution(), self.get_current_lambda()))
            self.__save_new_turn_solution(self.get_current_solution().T @ self.__A @ self.get_current_solution())
            self.__save_new_lambda(self.new_lambda(self.get_current_solution(), self.get_current_lambda()))
            ### Exponential
            # self.__save_new_solution(self.new_solution_exp(self.get_current_solution(), self.get_current_lambda()))
            # self.__save_new_turn_solution(self.get_current_solution().T @ self.__A @ self.get_current_solution())
            # self.__save_new_lambda(self.new_lambda_exp(self.get_current_solution(), self.get_current_lambda()))
            # print(self.cost(-1))

    def solve(self):
        """
        Resolve the TSP Problem

        """
        self.lagrange(n_max = 10000)
        self.__save_new_turn_solution(self.__problem.lp_solve())
        print(self.__solution_history[-1])
        plt.plot(self.__lambda_history)
        plt.show()
        plt.plot([self.cost(i) for i in range(len(self.get_turn_solutions_history()))])
        plt.show()

    def plot(self):
        """
        Plot the solution evolution

        """

        intervale_min = 0
        intervale_max = len(self.__solution_turn_history) - 1
        intervale_init = intervale_max

        list_pos = {}
        for i in range(self.__problem.get_number()):
            list_pos[i] = np.array([self.__problem.get_cities_array()[i, 0], self.__problem.get_cities_array()[i, 1]])

        def get_edge(index_solution):
            edge = []
            array = self.__solution_turn_history[index_solution]
            for i in range(self.__problem.get_number()):
                for j in range(self.__problem.get_number()):
                    if array[i, j] > 0.05:
                        edge += [(i, j, {"weight":int(array[i, j] * 100) / 50})]
            # print(edge)
            return edge

        def graph_network(graph, v_list_pos, v_index, v_ax):
            graph.clear_edges()
            graph.update(edges=get_edge(v_index))
            nx.draw_networkx_nodes(graph, v_list_pos, node_size=150, ax=v_ax)
            widthlist = list(nx.get_edge_attributes(graph,'weight').values())
            nx.draw_networkx_edges(graph, v_list_pos, width=widthlist, ax=v_ax)
            nx.draw_networkx_labels(graph, v_list_pos, font_size=10, font_family="sans-serif", ax=v_ax)
            # edge_labels = nx.get_edge_attributes(graph, "weight")
            # nx.draw_networkx_edge_labels(graph, v_list_pos, edge_labels, ax=v_ax)

        fig, axe = plt.subplots()
        network_graph = nx.DiGraph()
        graph_network(network_graph, list_pos, intervale_init, axe)
        fig.subplots_adjust(left=0.25)
        axamp = fig.add_axes([0.1, 0.1, 0.0225, 0.8])
        amp_slider = Slider(ax=axamp,
                            label="Itération",valmin=intervale_min,
                            valmax=intervale_max,
                            valinit=intervale_init,
                            orientation="vertical",
                            valstep=1)

        def update(v_fig, v_ax, v_graph, v_list_pos, val):
            v_ax.cla()
            # print(val)
            # print(self.get_solutions_history()[val])
            graph_network(v_graph, v_list_pos, val, v_ax)
            v_fig.canvas.draw_idle()

        amp_slider.on_changed(partial(update, fig, axe, network_graph, list_pos))

        plt.show()

    def cost(self, v_index):
        """
        Calculate the cost value of a solution in history

        Parameters
        ----------
        v_index : integer

        Returns
        -------
        a float
            The cost value of the v_index solution of history

        """
        # P = self.get_solutions_history()[v_index]
        # result = trace(self.__problem.get_cost_array().T @ P.T @ self.__A @ P)
        P = self.get_turn_solutions_history()[v_index]
        result = trace(self.__problem.get_cost_array().T @ P)
        return result

## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ##

if __name__ == "__main__":

    P_1 = TSPProblem(4, 100)

    # P_1.save("test.csv")
    # P_1.load("test.csv")
    # P_1.lp_solve()
    # print(P_1)

    # P_2 = Problem(50, 200)
    # print(P_2)

    # P_3 = Problem(500, 2000)
    # print(P_3)

    a = OrientedSolver(P_1)
    # print(a)
    a.solve()
    # print(a)
    a.plot()
