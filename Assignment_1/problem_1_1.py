"""
CS 534 Assignment-1.1: A-Star Algorithm.

Author: Samar Kale
Date: 2025-10-06
"""

import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import random
import time
import heapq
import statistics
import matplotlib.pyplot as plt

class SlidingPuzzle:
    def __init__(self, size: int = 3):
        """
        Initialize sliding puzzle game
        
        Args:
            size (int): Size of the grid (size x size)
        """
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.blank_pos = (0, 0)  # Position of blank space (0) at top-left
        self.moves = 0
        
        # Initialize solved state
        self.reset_puzzle()
        
        # GUI elements
        self.root = tk.Tk()
        self.root.title(f"Sliding Puzzle ({size}x{size})")
        self.root.resizable(False, False)
        
        self.buttons = []
        self.moves_label = None
        self.setup_gui()
        
    def reset_puzzle(self):
        """Reset puzzle to solved state"""
        # Fill grid with 0 (blank) at top-left, then numbers 1 to size*size-1
        num = 1
        for i in range(self.size):
            for j in range(self.size):
                if i == 0 and j == 0:
                    self.grid[i][j] = 0  # Blank space at top-left
                else:
                    self.grid[i][j] = num
                    num += 1
        
        self.blank_pos = (0, 0)
        self.moves = 0
        
    def get_state(self) -> np.ndarray:
        """
        Return current state of the grid as numpy array
        
        Returns:
            np.ndarray: Current grid state with 0 representing blank space
        """
        return self.grid.copy()
    
    def set_state(self, input_grid, blank_tile):
        """
        Return current state of the grid as numpy array
        
        Returns:
            np.ndarray: Current grid state with 0 representing blank space
        """
        self.grid = input_grid
        self.blank_pos = blank_tile
        self.moves = 0
        self.update_display()
    
    def get_goal_state(self) -> np.ndarray:
        """
        Return the goal/solved state of the grid as numpy array
        
        Returns:
            np.ndarray: Goal state with 0 at top-left and numbers 1 to size²-1
        """
        goal = np.zeros((self.size, self.size), dtype=int)
        num = 1
        for i in range(self.size):
            for j in range(self.size):
                if i == 0 and j == 0:
                    goal[i][j] = 0  # Blank space at top-left
                else:
                    goal[i][j] = num
                    num += 1
        return goal
    
    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if moving tile at (row, col) is valid"""
        blank_row, blank_col = self.blank_pos
        
        # Check if the clicked tile is adjacent to blank space
        return ((abs(row - blank_row) == 1 and col == blank_col) or 
                (abs(col - blank_col) == 1 and row == blank_row))
    
    def move_tile(self, row: int, col: int) -> bool:
        """
        Move tile at (row, col) if valid move
        
        Returns:
            bool: True if move was successful
        """
        if not self.is_valid_move(row, col):
            print("Exiting False")
            return False
        
        blank_row, blank_col = self.blank_pos
        
        # Swap tile with blank space
        self.grid[blank_row][blank_col] = self.grid[row][col]
        self.grid[row][col] = 0
        
        # Update blank position
        self.blank_pos = (row, col)
        self.moves += 1

        return True
    
    def is_solved(self) -> bool:
        """Check if puzzle is in solved state"""
        num = 1
        for i in range(self.size):
            for j in range(self.size):
                if i == 0 and j == 0:
                    if self.grid[i][j] != 0:
                        return False
                else:
                    if self.grid[i][j] != num:
                        return False
                    num += 1
        return True

    def shuffle(self, num_moves: int = 75):
        """Shuffle the puzzle by making random valid moves"""
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
        
        for _ in range(num_moves):
            # Get current blank position
            blank_row, blank_col = self.blank_pos
            
            # Find valid moves
            valid_moves = []
            for dr, dc in moves:
                new_row, new_col = blank_row + dr, blank_col + dc
                if (0 <= new_row < self.size and 0 <= new_col < self.size):
                    valid_moves.append((new_row, new_col))
            
            # Make a random valid move
            if valid_moves:
                move_row, move_col = random.choice(valid_moves)
                self.move_tile(move_row, move_col)
        
        self.moves = 0  # Reset move counter after shuffling
    
    def setup_gui(self):
        """Setup the GUI"""
        # Main frame for the puzzle
        puzzle_frame = tk.Frame(self.root, bg='gray', padx=5, pady=5)
        puzzle_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        
        # Create buttons for tiles
        self.buttons = []
        for i in range(self.size):
            button_row = []
            for j in range(self.size):
                btn = tk.Button(
                    puzzle_frame,
                    text="",
                    width=4,
                    height=2,
                    font=('Arial', 16, 'bold'),
                    command=lambda r=i, c=j: self.on_tile_click(r, c)
                )
                btn.grid(row=i, column=j, padx=2, pady=2)
                button_row.append(btn)
            self.buttons.append(button_row)
        
        # Control buttons and labels
        control_frame = tk.Frame(self.root)
        control_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        self.moves_label = tk.Label(control_frame, text="Moves: 0", font=('Arial', 12))
        self.moves_label.grid(row=0, column=0, columnspan=3, pady=5)
        
        shuffle_btn = tk.Button(control_frame, text="Shuffle", command=self.shuffle_puzzle,
                               font=('Arial', 10), bg='lightblue')
        shuffle_btn.grid(row=1, column=0, padx=5)
        
        reset_btn = tk.Button(control_frame, text="Reset", command=self.reset_and_update,
                             font=('Arial', 10), bg='lightgreen')
        reset_btn.grid(row=1, column=1, padx=5)
        
        resize_btn = tk.Button(control_frame, text="Resize", command=self.resize_puzzle,
                              font=('Arial', 10), bg='lightyellow')
        resize_btn.grid(row=1, column=2, padx=5)
        
        state_btn = tk.Button(control_frame, text="Show State", command=self.show_state,
                             font=('Arial', 10), bg='lightcoral')
        state_btn.grid(row=2, column=0, columnspan=3, pady=5)
        
        self.update_display()
    
    def on_tile_click(self, row: int, col: int):
        """Handle tile click"""
        if self.move_tile(row, col):
            self.update_display()
            if self.is_solved():
                messagebox.showinfo("Congratulations!", 
                                  f"Puzzle solved in {self.moves} moves!")
    
    def update_display(self):
        """Update the visual display"""
        for i in range(self.size):
            for j in range(self.size):
                value = self.grid[i][j]
                btn = self.buttons[i][j]
                
                if value == 0:  # Blank space
                    btn.config(text="", state="disabled", bg='gray')
                else:
                    btn.config(text=str(value), state="normal", bg='white')
        
        # Update moves counter
        if self.moves_label:
            self.moves_label.config(text=f"Moves: {self.moves}")
    
    def shuffle_puzzle(self):
        """Shuffle puzzle and update display"""
        self.shuffle()
        # self.update_display()
    
    def reset_and_update(self):
        """Reset puzzle and update display"""
        self.reset_puzzle()
        self.update_display()
    
    def resize_puzzle(self):
        """Resize the puzzle"""
        new_size = simpledialog.askinteger("Resize Puzzle", 
                                          "Enter new size (3-6):", 
                                          minvalue=3, maxvalue=6)
        if new_size and new_size != self.size:
            self.root.destroy()
            new_puzzle = SlidingPuzzle(new_size)
            new_puzzle.run()
    
    def show_state(self):
        """Show current state as numpy array"""
        state = self.get_state()
        goal = self.get_goal_state()
        state_str = ("Current Grid State (0 = blank):\n\n" + str(state) + 
                    "\n\nGoal State:\n\n" + str(goal))
        messagebox.showinfo("Grid State", state_str)
    
    def run(self):
        """Start the game"""
        self.shuffle_puzzle()  # Start with shuffled puzzle
        # self.root.mainloop()


class Node:
    def __init__(self, grid, parent, g_cost, h_cost, blank_pos):
        self.parent = parent
        self.grid = grid
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.total_cost = g_cost + h_cost
        self.blank_pos = blank_pos


class SlidingAStar:
    def __init__(self, size=3, heuristic=2):
        self.size = size
        self.puzzle = SlidingPuzzle(size)
        self.node_list = []  # Stores all explored nodes
        self.counter = 0  # Tie-breaker for heapq
        self.visted_node_list = set()  # Stores all visited nodes
        self.heuristic = heuristic
        self.goal_state = ()
        goal = self.puzzle.get_goal_state()
        self.goal_pos_dict = {goal[x][y]: (x, y) for x in range(self.size) for y in range(self.size)}
        self.open_set = {}

    def heuristic_function(self, c_state):
        """
        Calculates the heuristic cost for a given grid.
        Args:
            c_state (tuple): Grid for which heuristic is
                to be calculated
        
        Returns:
            int: heuristic value
        """
        if self.heuristic == 1:
            h_val = 0
            for row in range(self.size):
                for col in range(self.size):
                    if c_state[row][col] != self.goal_state[row][col]:
                        h_val += 1
            return h_val

        curr_pos_dict = {c_state[x][y]: (x, y) for x in range(self.size) for y in range(self.size)}
        h_val = 0
        for i in range(1, self.size**2):
            g_val = self.goal_pos_dict[i]
            c_val = curr_pos_dict[i]
            h_val += abs(g_val[0] - c_val[0]) + abs(g_val[1] - c_val[1])  # Manhattan Distance

        return h_val

    def get_best_node(self) -> Node:
        """Returns the node with least total cost"""
        node = heapq.heappop(self.node_list)[-1]  # retrieve node with least total cost
        self.open_set.pop(node.grid, None)  # Remove node from open set
        self.visted_node_list.add(node.grid)  # Add node to visited as it is expanded

        return node

    def explore_moves(self, node: Node):
        """Explores new nodes around current node"""
        c_state = node.grid
        row, col = node.blank_pos

        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < self.size) or not (0 <= nc < self.size):
                continue

            # Update move to c_state and save in n_grid
            n_grid = [list(row) for row in c_state]
            n_grid[row][col], n_grid[nr][nc] = n_grid[nr][nc], n_grid[row][col]
            n_grid = tuple(tuple(row) for row in n_grid)

            # If n_grid already visited skip
            if n_grid in self.visted_node_list:
                continue

            # If n_grid already as an optimal parent skip
            new_hcost = self.heuristic_function(n_grid)
            new_tcost = node.g_cost + 1 + new_hcost
            if n_grid in self.open_set and self.open_set[n_grid] < new_tcost:
                continue

            new_node = Node(
                n_grid,
                node,
                node.g_cost + 1,
                new_hcost,
                (nr, nc)
            )

            self.counter += 1
            heapq.heappush(self.node_list, (new_node.total_cost, self.counter, new_node))
            self.open_set[new_node.grid] = new_node.total_cost

    def solve(self, state=None, blank_space=None):
        """Main loop which executes the puzzle and A-Star"""
        if state is None:
            self.puzzle.run()

        else:
            self.puzzle.set_state(state.copy(), blank_space)

        # Read start and goal stae of puzzle into tuples
        start_state = tuple(tuple(int(x) for x in row) for row in self.puzzle.get_state())
        self.goal_state = tuple(tuple(int(x) for x in row) for row in self.puzzle.get_goal_state())

        print("Initial state:")
        print(start_state)
        print()

        # Start timer
        s_time = time.time()

        # Initialize the nodes
        blank_pos = next((x, y) for x in range(self.size) for y in range(self.size) if start_state[x][y] == 0)
        c_node = Node(start_state, -1, 0, self.heuristic_function(start_state), blank_pos)
        heapq.heappush(self.node_list, (c_node.total_cost, self.counter, c_node))
        print("Timer Started, Searching for solution")

        # Main solution loop
        while True:
            if len(self.node_list) == 0:
                stats = {
                    "nodes_expanded": len(self.visted_node_list),
                    "path_length": -1,
                    "time_taken": time.time() - s_time
                }
                print("No path found")
                return False, stats
            best_node = self.get_best_node()
            if best_node.grid == self.goal_state:
                break

            self.explore_moves(best_node)  # Explores possible nodes w.r.t possible moves

        move_list = []
        curr_node = best_node

        # Loop to find path
        while True:
            if curr_node.parent == -1:
                break
            move_list.append(curr_node.blank_pos)
            curr_node = curr_node.parent

        stats = {
            "nodes_expanded": len(self.visted_node_list),
            "path_length": len(move_list),
            "time_taken": time.time() - s_time
        }
        print(f"Total nodes explored: {stats['nodes_expanded']}")
        print(f"Number of moves to solve: {stats['path_length']}")
        print(f"Time to find solution: {stats['time_taken']}s")

        # To simulate path in GUI
        # for move in move_list[::-1]:
        #     self.puzzle.root.update_idletasks()
        #     self.puzzle.move_tile(move[0], move[1])
        #     self.puzzle.update_display()
            # time.sleep(1)
        # input("Press enter to exit")

        return True, stats

def run_batch_experiments(show_plot=True):
    """
    Run SlidingAStar solver for heuristics 1 and 2 over puzzle sizes 2 to 5,
    each 5 runs per size, collect stats, and plot results.
    """
    heuristic_map = {
        1: "Misplaced Tiles (Hamming Distance)",
        2: "Manhattan Distance"
    }
    results = {
        1: [],  # heuristic 1 results: list of tuples (n, avg_steps, avg_nodes, avg_time)
        2: []   # heuristic 2 results
    }

    for heuristic in [1, 2]:
        print(f"Running experiments for Heuristic {heuristic}")
        for n in range(2, 5):
            s_time = time.time()
            print(f"Board Size: {n}")
            steps_list = []
            nodes_expanded = []
            time_list = []

            for _ in range(5):
                solver = SlidingAStar(n, heuristic)
                _, stats = solver.solve()
                steps_list.append(stats["path_length"])
                nodes_expanded.append(stats["nodes_expanded"])
                time_list.append(stats["time_taken"])

            avg_steps = round(statistics.mean(steps_list), 2)
            avg_nodes = round(statistics.mean(nodes_expanded), 2)
            avg_time = round(statistics.mean(time_list), 2)
            results[heuristic].append((n, avg_steps, avg_nodes, avg_time))
            print(f"n={n}: Avg Steps = {avg_steps}, Avg Nodes Expanded = {avg_nodes}, Avg Time: {avg_time}")

            if time.time() - s_time > 600:
                print("Took longer than 10 mins, stopping...")
                break

    if show_plot:
        # Plot 1: Board size vs Avg nodes expanded (for both heuristics)
        plt.figure(figsize=(12, 7))
        for h in [1, 2]:
            ns = [r[0] for r in results[h]]
            avg_nodes = [r[2] for r in results[h]]
            plt.plot(ns, avg_nodes, marker='o', label=f"Heuristic {heuristic_map[h]}")
        plt.xlabel("Board Size (n)")
        plt.ylabel("Average Nodes Expanded")
        plt.title("Puzzle Size vs Average Nodes Expanded")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot 2: Effective branching factor vs total nodes expanded
        plt.figure(figsize=(12, 7))
        for h in [1, 2]:
            avg_steps = [r[1] for r in results[h]]
            avg_nodes = [r[2] for r in results[h]]

            effective_branching = []
            for N, d in zip(avg_nodes, avg_steps):
                if d > 0:
                    b_f = (N / d) ** (1 / d)
                else:
                    b_f = 0
                effective_branching.append(b_f)

            plt.plot(effective_branching, avg_nodes, marker='o', label=f"Heuristic {heuristic_map[h]}")

        plt.xlabel("Effective Branching Factor")
        plt.ylabel("Average Nodes Expanded")
        plt.title("Effective Branching Factor vs Nodes Expanded")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    return results

# Example usage and demonstration
if __name__ == "__main__":
    stats = run_batch_experiments(True)
