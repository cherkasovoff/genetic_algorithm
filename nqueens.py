# -*- coding: utf-8 -*-
import random


class Solver_8_queens:
    cells_count = 64
    best_chromosome = []
    best_fitness = 0

    def __init__(self, pop_size=300, cross_prob=0.5, mut_prob=0.5):
        self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob

    def solve(self, min_fitness=1, max_epochs=None):
        current_population = []
        current_fitness = []
        all_fitness = 0
        add_population = []
        new_population = []
        self.best_chromosome = []

        finding_solution = True
        epoch_num = 0

        current_population = list(self.generate_chromosomes(self.pop_size))
        current_fitness, all_fitness = self.set_fitness(current_population)
        self.set_best_value(current_population, current_fitness)

        while finding_solution:
            epoch_num += 1
            new_population.clear()
            add_population.clear()

            add_population = list(self.generate_chromosomes(self.pop_size // 2))
            add_population += self.get_add_population(current_population, current_fitness, all_fitness)

            new_population = list(self.get_new_generation(current_population, add_population))

            current_population = list(new_population)
            current_fitness.clear()
            all_fitness = 0

            current_fitness, all_fitness = self.set_fitness(current_population)
            self.set_best_value(current_population, current_fitness)

            if min_fitness is not None and max_epochs is not None:
                if self.best_fitness >= min_fitness or epoch_num >= max_epochs:
                    finding_solution = False
            elif min_fitness is not None:
                if self.best_fitness >= min_fitness:
                    finding_solution = False
            elif max_epochs is not None:
                if epoch_num >= max_epochs:
                    finding_solution = False

        best_fit = self.best_fitness
        visualization = self.get_solution()
        return best_fit, epoch_num, visualization

    def generate_chromosomes(self, pop_size):
        pop_generated = []

        for i in range(pop_size):
            chromosome = []
            for n in range(self.cells_count):
                chromosome.append(0)
            for n in range(8):
                rnd = random.randint(n * 8, 7 + 8 * n)
                chromosome[rnd] = 1
            pop_generated.append(chromosome)
        return pop_generated

    def set_fitness(self, population):
        fitness = []
        fitness_sum = 0

        for i in population:
            fitness_value = self.ffunc(i)
            fitness.append(fitness_value)
            fitness_sum += fitness_value
        return fitness, fitness_sum

    def set_best_value(self, population, fitness):
        self.best_fitness = max(fitness)
        self.best_chromosome = list(population[fitness.index(self.best_fitness)])

    def get_add_population(self, current_population, current_fitness, all_fitness):
        parents = []
        add_population = []

        for i in range(self.pop_size):
            parent_chromosome = self.roulette(current_fitness, all_fitness)
            parents.append(current_population[parent_chromosome])

        add_population = list(self.get_cross_chromosomes(parents))
        return add_population

    def get_cross_chromosomes(self, parents):
        cross_population = []

        for i in range(0, self.pop_size, 2):
            parent_a = parents[random.randint(1, len(parents) - 1)]
            parent_b = parents[random.randint(1, len(parents) - 1)]
            cut_point = random.randint(1, self.cells_count)
            ran_cross = random.random()
            if ran_cross < self.cross_prob:
                chromosome_ab = parent_a[:cut_point] + parent_b[cut_point:]
                ran_mut = random.random()
                if ran_mut < self.mut_prob:
                    gene_position = random.randint(0, self.cells_count - 1)
                    chromosome_ab = self.mutate_gene(chromosome_ab, gene_position)
                cross_population.append(chromosome_ab)

            ran_cross = random.random()
            if ran_cross < self.cross_prob:
                chromosome_ba = parent_b[:cut_point] + parent_a[cut_point:]
                ran_mut = random.random()
                if ran_mut < self.mut_prob:
                    gene_position = random.randint(0, self.cells_count - 1)
                    chromosome_ba = self.mutate_gene(chromosome_ba, gene_position)
                cross_population.append(chromosome_ba)
        return cross_population

    def get_new_generation(self, first_population, second_population):
        new_generation = []
        first_population_fitness = []
        second_population_fitness = []

        for i in range(len(first_population)):
            first_population_fitness.append(self.ffunc(first_population[i]))
        for i in range(len(second_population)):
            second_population_fitness.append(self.ffunc(second_population[i]))
        for i in range(self.pop_size):
            best_g1 = max(first_population_fitness) if len(first_population_fitness) > 0 else 0
            best_g2 = max(second_population_fitness) if len(second_population_fitness) > 0 else 0
            if best_g2 > best_g1:
                chromosome_index = second_population_fitness.index(best_g2)
                new_chromosome = second_population[chromosome_index]
                new_generation.append(new_chromosome)
                second_population.remove(new_chromosome)
                second_population_fitness.remove(second_population_fitness[chromosome_index])
            else:
                chromosome_index = first_population_fitness.index(best_g1)
                new_chromosome = first_population[chromosome_index]
                new_generation.append(new_chromosome)
                first_population.remove(new_chromosome)
                first_population_fitness.remove(first_population_fitness[chromosome_index])
        return new_generation

    def roulette(self, values, fitness_sum):
        random_point = random.random() * fitness_sum
        cur_sum = 0
        i = 0

        for i in range(len(values)):
            cur_sum += values[i]
            if cur_sum >= random_point:
                break
        return i

    def mutate_gene(self, chromosome, pos):
        if chromosome[pos] == 0:
            chromosome[pos] = 1
        elif chromosome[pos] == 1:
            chromosome[pos] = 0
        return chromosome

    def ffunc(self, chromosome):
        queens_x = []
        queens_y = []
        mistakes = 0
        fitness = 0
        queens_number = 0
        need_queens = 8

        for i in range(len(chromosome)):
            if chromosome[i] == 1:
                queens_number += 1
                need_queens -= 1
                queens_x.append(i % 8)
                queens_y.append(i // 8)
        if queens_number > 1:
            for x1 in range(0, queens_number - 1):
                for x2 in range(x1 + 1, queens_number):
                    if queens_x[x1] == queens_x[x2]:
                        mistakes += 1
                    if queens_y[x1] == queens_y[x2]:
                        mistakes += 1
                    if abs(queens_x[x2] - queens_x[x1]) == abs(queens_y[x2] - queens_y[x1]):
                        mistakes += 1
        mistakes += abs(need_queens)
        fitness = 1 / (1 + mistakes)
        return fitness

    def get_solution(self):
        solution = list(self.best_chromosome)

        for i in range(7):
            solution.insert(i + 8 * (i + 1), '\n')
        for i in range(len(solution)):
            if solution[i] == 0:
                solution[i] = '+'
            elif solution[i] == 1:
                solution[i] = 'Q'
        solution = ''.join(solution)
        return solution