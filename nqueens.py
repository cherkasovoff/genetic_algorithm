# -*- coding: utf-8 -*-
import random

class Solver_8_queens:

    def __init__(self, pop_size=300, cross_prob=0.5, mut_prob=0.5):
        self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob

    def solve(self, min_fitness=1, max_epochs=None):
        population = []
        pop_fitness = []
        parents = []
        all_fitness = 0
        n = 64
        best_ind = []
        best_fitness = 0
        new_pop = []

        gen1_f = []
        gen2_f = []
        next_gen = []

        algo_act = True
        geners = 0

        for x in range(self.pop_size):
            ind = []
            for y1 in range(n):
                ind.append(0)
            for y2 in range(8):
                rnd = random.randint(y2 * 8, 7 + 8 * y2)
                ind[rnd] = 1
            population.append(ind)

        for p in range(len(population)):
            po = population[p]
            fitness = self.ffunc(po)
            pop_fitness.append(fitness)
            all_fitness += fitness
        best_fitness = max(pop_fitness)
        best_ind = population[pop_fitness.index(max(pop_fitness))]

        while algo_act:

            geners += 1

            for x in range(self.pop_size // 2):
                ind = []
                for y1 in range(n):
                    ind.append(0)
                for y2 in range(8):
                    rnd = random.randint(y2 * 8, 7 + 8 * y2)
                    ind[rnd] = 1
                new_pop.append(ind)

            for x in range(self.pop_size):
                ind_sel = self.roulette(pop_fitness, all_fitness)
                parents.append(population[ind_sel])

            for x in range(0, self.pop_size, 2):
                parent_a = parents[random.randint(1, len(parents) - 1)]
                parent_b = parents[random.randint(1, len(parents) - 1)]
                cut_point = random.randint(1, n)

                ran_cross = random.random()
                if ran_cross < self.cross_prob:
                    ind_ab = parent_a[:cut_point] + parent_b[cut_point:]

                    ran_mut = random.random()
                    if ran_mut < self.mut_prob:
                        gene_position = random.randint(0, n - 1)
                        ind_mut = self.mutate(ind_ab, gene_position)
                        ind_ab = ind_mut
                        new_pop.append(ind_ab)
                        fitness = self.ffunc(ind_ab)

                ran_cross = random.random()
                if ran_cross < self.cross_prob:
                    ind_ba = parent_b[:cut_point] + parent_a[cut_point:]

                    ran_mut = random.random()
                    if ran_mut < self.mut_prob:
                        gene_position = random.randint(0, n - 1)
                        ind_mut = self.mutate(ind_ba, gene_position)
                        ind_ba = ind_mut
                        new_pop.append(ind_ba)
                        fitness = self.ffunc(ind_ba)

            gen1 = list(population)
            gen1_f.clear()
            gen2 = list(new_pop)
            gen2_f.clear()
            next_gen.clear()
            for x in range(len(gen1)):
                gen1_f.append(self.ffunc(gen1[x]))
            for x in range(len(gen2)):
                gen2_f.append(self.ffunc(gen2[x]))
            for x in range(self.pop_size):
                best_g1 = max(gen1_f) if len(gen1_f) > 0 else 0
                best_g2 = max(gen2_f) if len(gen2_f) > 0 else 0
                if best_g2 > best_g1:
                    ind_index = gen2_f.index(best_g2)
                    next_ind = gen2[ind_index]
                    next_gen.append(next_ind)
                    gen2.remove(next_ind)
                    gen2_f.remove(gen2_f[ind_index])
                else:
                    ind_index = gen1_f.index(best_g1)
                    next_ind = gen1[ind_index]
                    next_gen.append(next_ind)
                    gen1.remove(next_ind)
                    gen1_f.remove(gen1_f[ind_index])

            population.clear()
            population = list(next_gen)
            new_pop.clear()
            pop_fitness.clear()
            parents.clear()
            all_fitness = 0
            for p in range(len(population)):
                po = population[p]
                fitness = self.ffunc(po)
                pop_fitness.append(fitness)
                all_fitness += fitness
            best_fitness = max(pop_fitness)
            best_ind = population[pop_fitness.index(max(pop_fitness))]

            if min_fitness is not None and max_epochs is not None:
                if best_fitness >= min_fitness or geners >= max_epochs:
                    algo_act = False
            elif min_fitness is not None:
                if best_fitness >= min_fitness:
                    algo_act = False
            elif max_epochs is not None:
                if geners >= max_epochs:
                    algo_act = False

        for x in range(7):
            best_ind.insert(x + 8 * (x + 1), '\n')
        for x in range(len(best_ind)):
            if best_ind[x] == 0:
                best_ind[x] = '+'
            elif best_ind[x] == 1:
                best_ind[x] = 'Q'
        best_ind = ''.join(best_ind)

        best_fit = best_fitness
        epoch_num = geners
        visualization = best_ind
        return best_fit, epoch_num, visualization

    def generation(self):
        pass

    def roulette(self, values, fitness):
        n_rand = random.random() * fitness
        sum_fit = 0
        i = 0
        for i in range(len(values)):
            sum_fit += values[i]
            if sum_fit >= n_rand:
                break
        return i

    def mutate(self, ind, pos):
        if ind[pos] == 0:
            ind[pos] = 1
        elif ind[pos] == 1:
            ind[pos] = 0
        return ind

    def ffunc(self, ind):
        allx = []
        ally = []
        ms = 0
        fitness = 0
        cont = 0
        num = 8
        for z in range(len(ind)):
            if ind[z] == 1:
                cont += 1
                num -= 1
                allx.append(z % 8)
                ally.append(z // 8)
        if cont > 1:
            for x1 in range(0, cont - 1):
                for x2 in range(x1 + 1, cont):
                    if allx[x1] == allx[x2]:
                        ms += 1
                    if ally[x1] == ally[x2]:
                        ms += 1
                    if abs(allx[x2] - allx[x1]) == abs(ally[x2] - ally[x1]):
                        ms += 1
        ms += abs(num)
        fitness = 1 / (1 + ms)
        return fitness