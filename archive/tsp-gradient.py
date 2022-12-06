# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:13:50 2022

@author: WANG Haokun
@author: CHATROUX Marc
@author: LI Yuansheng
"""

import numpy as np
import matplotlib.pyplot as plt

# def generate_random_cities(n = 6, d = 2):
#     return np.random.rand(n,d)*100

# def dist(a, b):
#     return np.linalg.norm(a - b)

# def calculate_distance(loc_cities):
#     city_nbr = len(loc_cities)
#     mat_dist = np.zeros((city_nbr,city_nbr))
#     for i in range(city_nbr):
#         for j in range(city_nbr):
#             mat_dist[i][j] = dist(loc_cities[i],loc_cities[j])
#     return mat_dist

## +++++++++++++++++++++++++++++++++++ ##

class Problem:
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

        Returns
        -------
        None.

        """
        self.__number = n
        self.__size = d
        self.__cities = self.__generate_cities()
        self.__cost_array = self.__generate_cost_array()


    def __generate_cities(self):
        """
        Returns
        -------
        Generate n differents cities on an area of d * d
        """
        return np.random.rand(self.__number, 2) * self.__size

    def __generate_cost_array(self):
        """
        Returns the array of cost (2-norm)
        """
        array = np.zeros((self.__number, self.__number))
        for var_1 in range(0, self.__number):
            for var_2 in range(0, self.__number):
                array[var_1, var_2] = np.linalg.norm(self.__cities[var_1] - self.__cities[var_2])
        return array


    def load(self, filename):
        """
        Load a TSP Problem from a file
        """
        pass

    def save(self, filename):
        """
        Save a TSP Problem on a file
        """
        pass


    def get_number(self):
        """ Returns the number of cities """
        return self.__number

    def get_size(self):
        """ Returns the size of the area """
        return self.__size

    def get_cities(self):
        """ Returns the positions of cities """
        return self.__cities

    def get_cost_array(self):
        """ Returns the cost array """
        return self.__cost_array

## +++++++++++++++++++++++++++++++++++ ##

class Solveur_non_orienté:
    """
    """

    def __init__(self, problem):
        """
        """

        self.__problem = problem
        self.__number = problem.get_number()
        self.__solution = self.__generate_default_solution()

    def __generate_default_solution(self):
        """
        """
        array = np.zeros((self.__number, self.__number))
        array[0, self.__number - 1] = 1
        array[self.__number - 1, 0] = 1
        for var in range(0, self.__number - 1):
            array[0 + var, 1 + var] = 1
            array[1 + var, 0 + var] = 1
        return array

    def get_current_solution(self):
        """ Returns the current solution """
        return self.__solution


## +++++++++++++++++++++++++++++++++++ ##

if __name__ == "__main__":

    P_1 = Problem(5, 20)
    # P_2 = Problem(50, 200)
    # P_3 = Problem(500, 2000)

    print("matrice de coût 1 : ", P_1.get_cost_array())
    # print("matrice de coût 2 : ", P_2.get_cost_array())
    # print("matrice de coût 3 : ", P_3.get_cost_array())

    a = Solveur_non_orienté(P_1)
    print(a.get_current_solution())
