from random import uniform, choice, choices
from operator import attrgetter
from copy import deepcopy

import numpy as np


def fps(population):
    """Fitness proportionate selection implementation.

    Args:
        population (Population): The population we want to select from.

    Returns:
        Individual: selected individual.
    """

    # Sum total fitness
    total_fitness = sum([i.fitness for i in population])
    # Get a 'position' on the wheel
    spin = uniform(0, total_fitness)
    position = 0

    if population.optim == "max":

        # Find individual in the position of the spin
        for individual in population:
            position += individual.fitness
            if position > spin:
                return individual

    elif population.optim == "min":
        # Sum total fitness
        total_fitness = sum([1/(i.fitness) for i in population])
        # Get a 'position' on the wheel
        spin = uniform(0, total_fitness)
        position = 0
        # Find individual in the position of the spin
        for individual in population:
            position += 1/individual.fitness
            if position > spin:
                return individual
    else:
        raise Exception("No optimization specified (min or max).")


def tournament(population, size=5):
    """Tournament selection implementation.

    Args:
        population (Population): The population we want to select from.
        size (int): Size of the tournament.

    Returns:
        Individual: Best individual in the tournament.
    """

    # Select individuals based on tournament size
    tournament = [choice(population.individuals) for i in range(size)]

    # Check if the problem is max or min
    if population.optim == 'max':
        return max(tournament, key=attrgetter("fitness"))
    elif population.optim == 'min':
        return min(tournament, key=attrgetter("fitness"))
    else:
        raise Exception("No optimization specified (min or max).")
        
        

def ranking(population):
    
    n = len(population)
    
    # sums the rankings
    #total_ranks = n * (n + 1) / 2

    ranks = [i for i in range(1,n+1)]

    # get the ordered list of the individual by their fitness
    if population.optim == 'max':
        sorted_pop = deepcopy(sorted(population, key=attrgetter('fitness')))
        # passing ranks the 1st element of the sorted list will have 1/total_ranks probability of being selected 
        rank = choices(sorted_pop, weights=ranks)
        return max(rank, key=attrgetter("fitness"))
        
    elif population.optim == 'min':
        sorted_pop = deepcopy(sorted(population, key=attrgetter('fitness'), reverse=True))
        rank = choices(sorted_pop, weights=ranks)
        return min(rank, key=attrgetter("fitness"))
        
    else:
        raise Exception("No optimization specified (min or max).")

def pareto_selection(population):
    """Receives: Population
    Returns: Individual based on Bakers method for selection in Pareto
    based multiobjective GA's
    """
    # creates a fitness array for each individual of the population
    fitness = np.array([i.get_fitness() for i in population])
    # creates a new fitness array used for the calculations bellow
    fitness_calc = fitness.copy()
    # array for the flags of each individual
    flags = np.zeros(len(population), dtype=int)
    flag_count = 1

    # repeat until there are iindividuals without a flag
    while len(fitness_calc) > 0:
        # points to be deleted after each iteration
        points_to_delete = np.array([], dtype=int)
        # calculating non_dominated points
        pareto_points = pareto_front(fitness_calc, optim= population.optim)

        # iterating over those points
        for i, point in enumerate(pareto_points):
            # if is non_dominated
            if point == 1:
                # calculate the index array using the 1st fitness array
                # there can be multiple individuals with the same fitness
                indexs = set(np.where(fitness == fitness_calc[i])[0])

                #iterating over each index
                for index in indexs:
                    # atribute the flag to the correct index
                    flags[index] = flag_count
                    # append that index to the points to be deleted
                points_to_delete = np.append(points_to_delete, i)

        # eliminates all non_dominated points in each iteration
        fitness_calc = np.delete(fitness_calc, points_to_delete, axis=0)
        # updates the value of the flag
        flag_count += 1

    #inverse flags
    sum_flags_inverted = sum([1/flag for flag in flags])
    spin = uniform(0, sum_flags_inverted)
    position = 0

    for flag, individual in zip(flags, population):
        position += 1/flag
        if position > spin:
            return individual


def pareto_front(fitness, optim = None):
    """Code inpired by Eyal Kazin
       Link: https://github.com/elzurdo/multi_objective_optimisation/blob/master/01_knapsack%202D_exhaustive.ipynb

       Receives a list of fitness values
       Return: The points that are non-dominated or bellow the pareto front"""
    #starts with an empty array
    pareto_front = np.array([], dtype = int)

    #repeates for every value of fitness
    for i, fitness_array in enumerate(fitness):

        #until otherwise the point non-dominated
        is_pareto = True

        distance = fitness_array[0]
        duration = fitness_array[1]

        #creates a new fitness array without the point selected
        new_fitness = fitness.copy()
        new_fitness = np.delete(new_fitness, i, axis = 0)

        for new_fitness_array in new_fitness:

            new_distance = new_fitness_array[0]
            new_duration = new_fitness_array[1]

            #if any point dominates the previous point
            if optim == 'min' and (new_distance < distance and new_duration < duration):
                #it becomes dominated
                is_pareto = False
                pareto_front = np.append(pareto_front, 0)
                break
            if optim == 'max' and (new_distance > distance and new_duration > duration):
                is_pareto = False
                pareto_front = np.append(pareto_front, 0)
                break

        #if no point dominates the previous point
        if is_pareto != False:
            #it remains non-dominated
            pareto_front = np.append(pareto_front, 1)

    #returns a list of 0 - dominated , 1 - non-dominated
    return pareto_front

