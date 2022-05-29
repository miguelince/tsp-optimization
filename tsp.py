import json
from charles.charles import Population, Individual
from copy import deepcopy
import numpy as np
from charles.selection import *
from charles.mutation import *
from charles.crossover import *

# import data

def import_data(file, single=True, metric=1):
    
    """ Function to import the data for the desired problem
    
    Atributes:
        - filname: name of json file to import
        - single: True for single-objective, False for multi-objective optimization
        - metric: 0 for distance, 1 for duration
    """
    
    # loads the data for optimization
    def read_jsons(file):
        with open(file) as json_file:
            data = json.load(json_file)
        return data
    
    # load data
    distance_matrix = read_jsons(file)['distances']
    duration_matrix = read_jsons(file)['durations']
    
    if single:
        if metric == 0:
            return distance_matrix
        else:
            return duration_matrix
    else:
        # creates a 2d list for multiobjective optimization
        twod_matrix = []
        for i in range(len(distance_matrix)):
            vec = []
            for j in range(len(distance_matrix)):
                vec.append([distance_matrix[i][j], duration_matrix[i][j]])
            twod_matrix.append(vec)
        return twod_matrix
    
    
matrix = import_data('distance_matrix.json', single=False)


#Fitness Single Objective
def get_fitness_single(self):

    fitness = 0
    for i in range(len(self.representation)):
        fitness += matrix[self.representation[i - 1]][self.representation[i]]
    return int(fitness)

    
def get_fitness_multi(self):

    fitness = np.array([0, 0])
    for i in range(len(self.representation)):
        fitness[0] += int(matrix[self.representation[i - 1]][self.representation[i]][0])
        fitness[1] += int(matrix[self.representation[i - 1]][self.representation[i]][1])
    return fitness
        
    
def get_neighbours(self):
    """A neighbourhood function for the TSP problem. Switches
    indexes around in pairs.

    Returns:
        list: a list of individuals
    """
    n = [deepcopy(self.representation) for i in range(len(self.representation) - 1)]

    for count, i in enumerate(n):
        i[count], i[count + 1] = i[count + 1], i[count]

    n = [Individual(i) for i in n]
    return n


# Monkey patching
# Individual.get_fitness = get_fitness_single # single objective
Individual.get_fitness = get_fitness_multi # multi objective
Individual.get_neighbours = get_neighbours


# run experiments 

pop = Population(
size=20,
sol_size=len(matrix[0]),
valid_set=[i for i in range(len(matrix[0]))],
replacement=False,
optim="min",
)
pop.evolve(
    gens=100,
    select=pareto_selection,
    crossover=cycle_co,
    mutate=swap_mutation,
    co_p=0.9,
    mu_p=0.1,
    elitism=True,
    single = False, 
    optimal_fitness = [0,0]
)


