import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

width = 200
height = 16

# Use design elements
class Individual_DE(object):
    __slots__ = ["genome", "_fitness", "_level"]

    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m], coefficients))
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, genome):
        if len(genome) == 0:
            return genome
        index = random.randint(0, len(genome) - 1)
        de = genome[index]
        x = de[0]
        de_type = de[1]
        new_de = de

        if de_type in ["4_block", "5_qblock"]:
            y = de[2]
            special = de[3]
            if random.random() < 0.5:
                y = max(1, min(height - 3, y + random.choice([-1, 1])))
            else:
                special = not special
            new_de = (x, de_type, y, special)

        elif de_type == "3_coin":
            y = de[2]
            y = max(1, min(height - 3, y + random.choice([-1, 1])))
            new_de = (x, de_type, y)

        elif de_type == "7_pipe":
            h = de[2]
            h = max(2, min(6, h + random.choice([-1, 1])))
            new_de = (x, de_type, h)

        elif de_type == "0_hole":
            w = de[2]
            w = max(1, min(6, w + random.choice([-1, 1])))
            new_de = (x, de_type, w)

        elif de_type == "6_stairs":
            h = de[2]
            dx = de[3]
            h = max(2, min(6, h + random.choice([-1, 1])))
            dx = -dx if random.random() < 0.5 else dx
            new_de = (x, de_type, h, dx)

        elif de_type == "1_platform":
            w = de[2]
            y = de[3]
            mat = de[4]
            y = max(2, min(height - 4, y + random.choice([-1, 1])))
            new_de = (x, de_type, w, y, mat)

        genome[index] = new_de
        return genome

    def generate_children(self, other):
        a = self.genome
        b = other.genome
        pa = random.randint(0, len(a))
        pb = random.randint(0, len(b))
        child1 = a[:pa] + b[pb:]
        child2 = b[:pb] + a[pa:]
        return Individual_DE(self.mutate(child1)), Individual_DE(self.mutate(child2))

    def to_level(self):
        if self._level is None:
            base = [["-" for _ in range(width)] for _ in range(height)]
            for x in range(width):
                base[height - 1][x] = "X"
            base[14][0] = "m"
            base[7][-1] = "v"
            for row in range(8, 14):
                base[row][-1] = "f"
            for row in range(14, 16):
                base[row][-1] = "X"

            for de in sorted(self.genome, key=lambda d: (d[0], d[1])):
                x = clip(1, de[0], width - 2)
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    block = "B" if de[3] else "X"
                    base[y][x] = block
                elif de_type == "5_qblock":
                    y = de[2]
                    block = "M" if de[3] else "?"
                    base[y][x] = block
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    for dy in range(h):
                        base[height - 1 - dy][x] = "|"
                    base[height - h - 1][x] = "T"
                elif de_type == "0_hole":
                    w = de[2]
                    for dx in range(w):
                        base[height - 1][clip(1, x + dx, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]
                    for step in range(h):
                        for dy in range(step + 1):
                            bx = clip(1, x + step * dx, width - 2)
                            by = height - 1 - dy
                            base[by][bx] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    y = height - 1 - de[3]
                    mat = de[4]
                    for dx in range(w):
                        base[y][clip(1, x + dx, width - 2)] = mat
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(cls):
        return cls([])

    @classmethod
    def random_individual(cls):
        count = random.randint(20, 60)
        elements = []
        for _ in range(count):
            x = random.randint(1, width - 2)
            choice = random.random()
            if choice < 0.1:
                elements.append((x, "0_hole", random.randint(1, 4)))
            elif choice < 0.2:
                elements.append((x, "1_platform", random.randint(3, 7), random.randint(3, 10), random.choice(["X", "B", "?"])))
            elif choice < 0.35:
                elements.append((x, "2_enemy"))
            elif choice < 0.5:
                elements.append((x, "3_coin", random.randint(3, 10)))
            elif choice < 0.65:
                elements.append((x, "4_block", random.randint(3, 10), random.choice([True, False])))
            elif choice < 0.8:
                elements.append((x, "5_qblock", random.randint(3, 10), random.choice([True, False])))
            elif choice < 0.9:
                elements.append((x, "6_stairs", random.randint(2, 6), random.choice([-1, 1])))
            else:
                elements.append((x, "7_pipe", random.randint(2, 6)))
        return cls(elements)

Individual = Individual_DE

def clip(lo, val, hi):
    return max(lo, min(val, hi))

def generate_successors(population):
    next_gen = []
    elite_count = int(0.1 * len(population))
    tournament_size = 5
    mutation_rate = 0.1

    population.sort(key=Individual.fitness, reverse=True)
    elites = population[:elite_count]
    next_gen.extend(elites)

    while len(next_gen) < len(population):
        parent1 = max(random.sample(population, tournament_size), key=Individual.fitness)
        parent2 = max(random.sample(population, tournament_size), key=Individual.fitness)
        children = parent1.generate_children(parent2)
        for child in children:
            if random.random() < mutation_rate:
                child.genome = child.mutate(child.genome)
                child._fitness = None
            next_gen.append(child)
            if len(next_gen) >= len(population):
                break
    return next_gen

def ga():
    pop_limit = 480
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        population = [Individual.random_individual() if random.random() < 0.9 else Individual.empty_individual() for _ in range(pop_limit)]
        population = pool.map(Individual.calculate_fitness, population, batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")
        try:
            while True:
                now = time.time()
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")
                generation += 1
                stop_condition = False
                if stop_condition:
                    break
                gentime = time.time()
                next_population = generate_successors(population)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                next_population = pool.map(Individual.calculate_fitness, next_population, batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pass
    return population

if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")
