"""Generic genetic algorithm framework.

Provides an abstract base for implementing genetic algorithm optimizers.
Subclasses must implement ``generate_individual``, ``fitness``, and
``mutation``.
"""

import random
from abc import ABC, abstractmethod


class GeneticAlgorithm(ABC):
    """Abstract genetic algorithm framework.

    Parameters
    ----------
    population_size : int
        Number of individuals in the population.
    n_crossover : int
        Number of crossover points for recombination.
    mutation_prob : float
        Probability of mutating each gene.
    maximize : bool
        If True, maximize fitness. If False, minimize.
    """

    def __init__(self, population_size=15, n_crossover=3, mutation_prob=0.05,
                 maximize=True):
        self.population_size = population_size
        self.n_crossover = n_crossover
        self.mutation_prob = mutation_prob
        self.maximize = maximize
        self.fitness_history = []
        self.population = [self.generate_individual() for _ in range(self.population_size)]

    @abstractmethod
    def generate_individual(self):
        """Generate a random individual."""
        pass

    @abstractmethod
    def fitness(self, individual):
        """Return the fitness of an individual."""
        pass

    @abstractmethod
    def mutation(self, individual, prob):
        """Mutate an individual with given probability per gene."""
        pass

    def crossover(self, x, y, k):
        """K-point crossover between two parents.

        Parameters
        ----------
        x, y : list
            Parent individuals.
        k : int
            Number of crossover points.

        Returns
        -------
        tuple of (list, list)
            Two offspring.
        """
        if len(x) <= 1 or k <= 0:
            return x[:], y[:]

        points = sorted(random.sample(range(1, len(x)), min(k, len(x) - 1)))
        points = [0] + points + [len(x)]

        x_new, y_new = x[:], y[:]
        for i in range(1, len(points), 2):
            start = points[i - 1]
            end = points[i] if i < len(points) else len(x)
            x_new[start:end], y_new[start:end] = y[start:end], x[start:end]

        return x_new, y_new

    def solve(self, max_generations, goal_fitness=1.0):
        """Run the genetic algorithm.

        Parameters
        ----------
        max_generations : int
            Maximum number of generations to run.
        goal_fitness : float
            Stop early if this fitness is reached.

        Returns
        -------
        object
            The best individual found.
        """
        for _ in range(max_generations):
            # Sort population by fitness
            self.population.sort(
                key=lambda x: self.fitness(x), reverse=self.maximize
            )

            best_fitness = self.fitness(self.population[0])
            self.fitness_history.append(best_fitness)

            if self.maximize and best_fitness >= goal_fitness:
                return self.population[0]
            if not self.maximize and best_fitness <= goal_fitness:
                return self.population[0]

            # Keep top half
            half = max(len(self.population) // 2, 1)
            survivors = self.population[:half]

            # Create children through crossover and mutation
            children = []
            for _ in range(half):
                parent_a = random.choice(survivors)
                parent_b = random.choice(survivors)
                child_a, child_b = self.crossover(
                    parent_a, parent_b, self.n_crossover
                )
                children.append(self.mutation(child_a, self.mutation_prob))
                children.append(self.mutation(child_b, self.mutation_prob))

            # Keep best half of children
            children.sort(
                key=lambda x: self.fitness(x), reverse=self.maximize
            )
            children = children[:half]

            self.population = survivors + children

        # Final sort
        self.population.sort(
            key=lambda x: self.fitness(x), reverse=self.maximize
        )
        self.fitness_history.append(self.fitness(self.population[0]))
        return self.population[0]
