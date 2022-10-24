import numpy as np

def generate_random_cities(n = 6, d = 2):
    return np.random.rand(n,d)*100

def dist(a, b):
    return np.linalg.norm(a - b)

def calculate_distance(loc_cities):
    city_nbr = len(loc_cities)
    mat_dist = np.zeros((city_nbr,city_nbr))
    for i in range(city_nbr):
        for j in range(city_nbr):
            mat_dist[i][j] = dist(loc_cities[i],loc_cities[j])
    return mat_dist
    
## +++++++++++++++++++++++++++++++++++ ##
