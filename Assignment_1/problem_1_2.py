import random
import statistics
import matplotlib.pyplot as plt
import time


class NQueenGA:
    def __init__(self, board_dim) -> None:
        self.size = board_dim

    def generate_population(self, k_states):
        population_list = []
        for _ in range(k_states):
            k = [random.randint(0, self.size-1) for _ in range(self.size)]
            population_list.append(k)

        return population_list

    def fitness(self, k_state):
        max_non_atk = self.size * (self.size - 1)//2
        atk = 0
        for i in range(self.size):
            for j in range(i+1, self.size):
                if k_state[i] == k_state[j] \
                    or abs(k_state[i] - k_state[j]) == abs(i - j):
                    atk += 1

        return max_non_atk - atk

    def selection(self, fit_pop):
        """Returns pairs from top 50% population"""
        x = random.choice(fit_pop[:len(fit_pop)//2])
        y = random.choice(fit_pop[:len(fit_pop)//2])
        return (x[0], y[0])
        
    def reproduce(self, p_1, p_2):
        """
        Returns a child based on parents
        Crossover
        """
        cut_off = random.randint(1, len(p_1))
        return p_1[:cut_off] + p_2[cut_off:]

    def mutate(self, child):
        element = random.randint(0, len(child)-1)
        child[element] = random.randint(0, self.size)
        return child

    def solve(self, mutate_times=1, pop_size=10):
        pop = self.generate_population(pop_size)
        gen = 0
        gen_thresh = 1000
        max_non_atk = self.size * (self.size - 1)//2

        while True:
            if gen > gen_thresh:
                print("Generation Threshold Exceeded!")
                break

            fit_list = [self.fitness(k) for k in pop]
            best_fit = max(fit_list)
            if best_fit == max_non_atk:
                # print("Result Achieved!")
                break

            fit_pop = list(zip(pop, fit_list))
            fit_pop.sort(key=lambda x:x[1], reverse=True)
            new_pop = [x[0] for x in fit_pop[:int(len(fit_pop)*0.1)]]  # Top 10% population continued to next

            for _ in range(len(pop)):
                p_1, p_2 = self.selection(fit_pop)  # Selects 2 parents randomly
                child = self.reproduce(p_1, p_2)
                for _ in range(mutate_times):
                    mutated_child = self.mutate(child)
                    child = mutated_child

                new_pop.append(child)

            pop = new_pop
            gen += 1

        fit_list = [self.fitness(k) for k in pop]
        best_fit = max(fit_list)
        best_child = pop[fit_list.index(best_fit)]

        return gen, best_fit, best_child

def run_batch_experiments(show_plot=True):
    """Run GA 5 times for each n = 5 to 10 and record stats"""
    results = {
        1: [],  # heuristic 1 results: list of tuples (n, avg_steps, avg_nodes, avg_time)
        3: []   # heuristic 2 results
    }

    for mutations in results.keys():
        for n in range(5, 11):
            s_time = time.time()
            print(f"Board Size: {n}")
            steps_list = []
            fitness_list = []

            for _ in range(5):  # 5 runs per n
                ga = NQueenGA(n)
                steps, final_fitness, _ = ga.solve(mutations, pop_size=100)
                steps_list.append(steps)
                fitness_list.append(final_fitness)

            avg_steps = round(statistics.mean(steps_list), 2)
            avg_fit = round(statistics.mean(fitness_list), 2)
            results[mutations].append((n, avg_steps, avg_fit))
            print(f"n={n}: Avg Steps = {avg_steps}, Avg Fitness = {avg_fit}")
            if time.time() - s_time > 600:
                print("Took longer than 10 mins")
                break
            

    if show_plot:
        for m in results.keys():
            ns = [r[0] for r in results[m]]
            step_vals = [r[1] for r in results[m]]
            plt.plot(ns, step_vals, marker='o', label=f"NQueens GCA Test Mutaions: {m}")

        plt.xlabel("Board Size (n)")
        plt.ylabel("Avg Generations to Solve")
        plt.title("Genetic Algorithm: Steps vs Board Size")
        plt.grid(True)
        plt.legend()

    return results

if __name__ == "__main__":
    std_results = run_batch_experiments(show_plot=True)
    plt.show()
