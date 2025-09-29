import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import random
from typing import Tuple, List

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
        print(f"Blank Pos: {blank_row, blank_col}")
        print(f"Moving Tile: {row, col}")
        self.grid[blank_row][blank_col] = self.grid[row][col]
        self.grid[row][col] = 0
        
        # Update blank position
        self.blank_pos = (row, col)
        self.moves += 1

        print("Exiting True")
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
    
    def shuffle(self, num_moves: int = 1000):
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
        self.update_display()
    
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


class SlidingAStar:
    def __init__(self, size=3) -> None:
        self.size = size
        self.puzzle = SlidingPuzzle(size)
        self.goal = self.puzzle.get_goal_state()
        # Example of using get_state() method

    def cost_function(self, g_cost):
        return g_cost + 1

    def get_manhattan_dist(self, s_point, g_point):
        return abs((g_point[0] - s_point[0]) + (g_point[1] - s_point[1]))

    def heuristic_function(self, node):
        """
        Calculates the heuristic cost, if the node is moved
        to empty_space.

        Steps:
            1. Find empty space (element==0)
            2. Swap node with empty space
            3. Calculate heuristic between goal and swapper matrix
            4. Return heuristic
        """
        p_state = self.puzzle.get_state()
        row, col = np.where(p_state == 0)
        mod_pose = p_state.copy()
        mod_pose[row, col], mod_pose[node[0], node[1]] = mod_pose[node[0], node[1]], mod_pose[row, col]

        h_val = 0
        goal_pos = self.puzzle.get_goal_state()
        for i in range(self.size):
            g_val = np.where(goal_pos == i)
            c_val = np.where(p_state == i)
            h_val += self.get_manhattan_dist(c_val, g_val)

        return h_val

    def total_cost(self, node, curr_pos, goal_pos, g_cost):
        """
        Calculates the total cost for a given node

            h(n) = g(n) + f(n)
        
        Args:
            node: node whose cost is to be found
            curr_pos: current position of target tile
            goal_pos: goal position of target tile
            g_cost: g(n), number of moves
        
        Returns:
            float: cost of the node
        """
        return self.cost_function(g_cost) + self.heuristic_function(node)

    def get_best_node(self, node_list, target, g_cost) -> np.ndarray:
        """
        Calculates the total cost of each node and returns the
        one with least value

        Args:
            node_list: List of available nodes to explore cost for
            target: current element to explore for
            g_cost: current traversal cost

        Returns:

            np.ndarray size(2): location of node with least cost
        """
        p_state = self.puzzle.get_state()
        curr_pos = np.where(p_state == target)
        goal_pos = np.where(self.goal == target)

        cost_arr = np.zeros(len(node_list))
        for i, node in enumerate(node_list):
            cost_arr[i] = self.total_cost(node, curr_pos, goal_pos, g_cost)

        node_id = np.argmin(cost_arr)
        return node_list[node_id]

    def explore_nodes(self):
        p_state = self.puzzle.get_state()
        row, col = np.where(p_state == 0)

        up_node = (row - 1, col) if row > 0 else (row, col)
        right_node = (row , col + 1) if col < self.size-1 else (row, col)
        down_node = (row + 1, col) if row < self.size-1 else (row, col)
        left_node = (row , col - 1) if col > 0 else (row, col)

        return (up_node, right_node, down_node, left_node)

    def solve(self):
        """
        Main loop which executes the puzzle and A-Star

        A-Star:
            1. Initialise, g, h, f, r to 0
            2. Initialize i as i to size^2 -1
            3. Get current state
            4. Explore nodes and calculate their h
            5. Select the one with least h as r.
            6. Pass r to puzzle to make a move, repeat loop.
            7. If f(i) = 0, update i by decrementing by 1.
            8. Exit loop when i = 0 or time exceeds time_thresh.
        """
        self.puzzle.run()
        self.puzzle.shuffle()
        print("Initial state:")
        print(self.puzzle.get_state())
        print()

        g_cost = f_cost = h_cost = root = 0
        i_tile = self.size**2 - 1

        while True:
            nodes = self.explore_nodes()
            least_h = self.get_best_node(nodes, i_tile, g_cost)
            print(f"Least H: {least_h}")
            input("Press enter to continue: ")
            ret = self.puzzle.move_tile(least_h[0].item(), least_h[1].item())
            print(f"Move Tile: {ret}")
            break





# Example usage and demonstration
if __name__ == "__main__":
    solver = SlidingAStar()
    solver.solve()
