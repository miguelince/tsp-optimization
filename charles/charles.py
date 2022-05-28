from random import shuffle, choice, sample, random
from charles.selection import pareto_front
import numpy as np
from operator import attrgetter
from copy import deepcopy


class Individual:
    def __init__(
            self,
            representation=None,
            size=None,
            replacement=True,
            valid_set=None,
    ):
        if representation is None:  # creating representation for the first time
            if replacement == True:
                self.representation = [choice(valid_set) for i in range(size)]
            elif replacement == False:
                self.representation = sample(valid_set, size)
        else:
            self.representation = representation

        self.fitness = self.get_fitness()

    def get_fitness(self, *kwargs):
        raise Exception("You need to monkey patch the fitness path.")

    def get_neighbours(self, func, **kwargs):
        raise Exception("You need to monkey patch the neighbourhood function.")

    def index(self, value):
        return self.representation.index(value)

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
        return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
        return f"Individual(size={len(self.representation)}); Fitness: {self.fitness}"


class Population:
    def __init__(self, size, optim, **kwargs):
        self.individuals = []
        self.size = size
        self.optim = optim
        for i in range(size):
            self.individuals.append(
                Individual(
                    size=kwargs["sol_size"],
                    replacement=kwargs["replacement"],
                    valid_set=kwargs["valid_set"],
                )
            )

    def evolve(self, gens, select, crossover, mutate, co_p, mu_p, elitism, single = True, optimal_fitness = None):
        for gen in range(gens):
            new_pop = []

            if elitism == True:
                if single==False:
                    fitness_values = [individ.fitness for individ in self.individuals]
                    pareto_points = pareto_front(fitness_values)
                    elites = np.array([], dtype = int)
                    for i, point in enumerate(pareto_points):

                        if point == 1:
                            elites = np.append(elites, i)

                else:
                    if self.optim == "max":
                        elite = deepcopy(max(self.individuals, key=attrgetter("fitness")))
                    elif self.optim == "min":
                        elite = deepcopy(min(self.individuals, key=attrgetter("fitness")))

            # selection of the parents
            while len(new_pop) < self.size:
                parent1, parent2 = select(self), select(self)  # select parents based on selection method
                # Crossover
                if random() < co_p:
                    offspring1, offspring2 = crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2
                # Mutation
                if random() < mu_p:
                    offspring1 = mutate(offspring1)
                if random() < mu_p:
                    offspring2 = mutate(offspring2)

                new_pop.append(Individual(representation=offspring1))

                if len(new_pop) < self.size:
                    new_pop.append(Individual(representation=offspring2))

            if elitism == True:

                if single==False: # only for minimization
                    fitness_values = [i.fitness for i in self]
                    pareto_points = pareto_front(fitness_values)
                    for i, point in enumerate(pareto_points):
                        if len(elites) == 0:
                            break
                        elif point == 0:
                            new_pop.pop(i)
                            new_pop.append(self[elites[0]])
                            elites = np.delete(elites, 0)

                else:
                    if self.optim == "max":
                        least = min(new_pop, key=attrgetter("fitness"))
                    elif self.optim == "min":
                        least = max(new_pop, key=attrgetter("fitness"))

                    new_pop.pop(new_pop.index(least))
                    new_pop.append(elite)


            self.individuals = new_pop

            if single == False:
                # for minimization
                fitness_values = [i.fitness for i in self]
                distance_vector = np.array([])
                for fit in fitness_values:
                    distance = np.linalg.norm(optimal_fitness - fit)
                    distance_vector = np.append(distance_vector, distance)
                
                if self.optim == "min":
                    # prints the fitness of the individual closer to the origin
                    print(f'Best Individual: {self[np.argmin(distance_vector)].fitness[0]} : {self[np.argmin(distance_vector)].fitness[1]}')
                    

            else:
                if self.optim == "max":
                    print(f'Best Individual: {max(self, key=attrgetter("fitness"))}')
                elif self.optim == "min":
                    print(f'Best Individual: {min(self, key=attrgetter("fitness"))}')
            
                

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]

    def __repr__(self):
        return f"Population(size={len(self.individuals)}, individual_size={len(self.individuals[0])})"
