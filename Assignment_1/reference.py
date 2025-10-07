import random
import time
import sys
import matplotlib.pyplot as plt
import pandas as pd

class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state      
        self.parent = parent
        self.g = g              
        self.h = h              
        self.f = g + h        

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(self.state)

def misplaced_tiles(state, goal):
    return sum(
        1
        for i in range(len(state))
        for j in range(len(state))
        if state[i][j] != 0 and state[i][j] != goal[i][j]
    )

def manhattan_distance(state, goal):
    n = len(state)
    # map tile -> goal position
    pos = {goal[i][j]: (i, j) for i in range(n) for j in range(n)}
    dist = 0
    for i in range(n):
        for j in range(n):
            val = state[i][j]
            if val != 0:
                x, y = pos[val]
                dist += abs(i - x) + abs(j - y)
    return dist

def get_neighbors(state):
    n = len(state)
    state_list = [list(row) for row in state]
    # find blank
    i, j = next((r, c) for r in range(n) for c in range(n) if state[r][c] == 0)
    moves = []
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < n and 0 <= nj < n:
            new_state = [row[:] for row in state_list]
            new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
            moves.append(tuple(tuple(row) for row in new_state))
    return moves

def generate_random_state(goal, moves=1000):
    state = [list(row) for row in goal]
    n = len(state)
    for _ in range(moves):
        i, j = next((r, c) for r in range(n) for c in range(n) if state[r][c] == 0)
        choices = []
        if i > 0: choices.append((i-1, j))
        if i < n-1: choices.append((i+1, j))
        if j > 0: choices.append((i, j-1))
        if j < n-1: choices.append((i, j+1))
        ni, nj = random.choice(choices)
        state[i][j], state[ni][nj] = state[ni][nj], state[i][j]
    return tuple(tuple(row) for row in state)

def astar(start, goal, heuristic_fn):
    open_list = []
    closed_set = set()

    start_node = Node(start, None, 0, heuristic_fn(start, goal))
    open_list.append(start_node)

    nodes_expanded = 0

    while open_list:
        # choose node with smallest f
        current = min(open_list, key=lambda n: n.f)
        open_list.remove(current)
        closed_set.add(current)

        if current.state == goal:
            # reconstruct path
            path = []
            node = current
            while node:
                path.append(node.state)
                node = node.parent
            return path[::-1], nodes_expanded

        # expand neighbors
        for neighbor_state in get_neighbors(current.state):
            g_cost = current.g + 1
            h_cost = heuristic_fn(neighbor_state, goal)
            neighbor = Node(neighbor_state, current, g_cost, h_cost)

            if neighbor in closed_set:
                continue

            existing = next((n for n in open_list if n.state == neighbor.state and n.g <= neighbor.g), None)
            if existing:
                continue

            open_list.append(neighbor)

        nodes_expanded += 1

    return None, nodes_expanded

def run_experiments():
    ks = [3, 8, 15]  # puzzle sizes
    results = []

    for K in ks:
        n = int((K + 1) ** 0.5)  # board dimension
        # goal state
        goal = [[(i*n + j + 1) % (n*n) for j in range(n)] for i in range(n)]
        goal = tuple(tuple(row) for row in goal)

        print(f"\n=== {n}x{n} Puzzle (K={K}) ===")
        for trial in range(1, 6):
            start = generate_random_state(goal, moves=40)

            print(f"\nTrial {trial}, Initial State:")
            for row in start:
                print(row)

            for h_name, h_fn in [("h1 (misplaced)", misplaced_tiles),
                                 ("h2 (manhattan)", manhattan_distance)]:
                start_time = time.time()
                path, expanded = astar(start, goal, h_fn)
                elapsed = time.time() - start_time

                if path:
                    depth = len(path) - 1
                    print(f"{h_name}: Path length={depth}, Expanded={expanded}, Time={elapsed:.4f}s")
                    results.append((K, n, h_name, depth, expanded))
                else:
                    print(f"{h_name}: No solution found (Expanded={expanded}, Time={elapsed:.4f}s)")

    return results

if __name__ == "__main__":
    data = run_experiments()

    # organize results
    df = pd.DataFrame(data, columns=["K","n","heuristic","depth","expanded"])
    summary = df.groupby(["K","n","heuristic"]).mean().reset_index()

    # compute b*
    summary["b*"] = (summary["expanded"] / summary["depth"]) ** (1/summary["depth"])

    print("\n=== Summary ===")
    print(summary)

    # Plot 1: puzzle size vs avg nodes generated
    plt.figure(figsize=(7,5))
    for h in summary["heuristic"].unique():
        sub = summary[summary["heuristic"]==h]
        plt.plot(sub["n"], sub["expanded"], marker='o', label=h)
    plt.xlabel("Puzzle size n (n x n)")
    plt.ylabel("Avg nodes generated (expanded)")
    plt.title("Puzzle size vs avg nodes generated")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: b* vs total nodes generated
    plt.figure(figsize=(7,5))
    for h in summary["heuristic"].unique():
        sub = summary[summary["heuristic"]==h]
        plt.plot(sub["expanded"], sub["b*"], marker='o', label=h)
    plt.xlabel("Total nodes generated (expanded)")
    plt.ylabel("Effective branching factor b*")
    plt.title("b* vs total nodes generated")
    plt.legend()
    plt.grid(True)
    plt.show()
