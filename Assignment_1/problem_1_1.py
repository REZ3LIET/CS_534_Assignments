import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import random
import time

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


class Node:
    def __init__(self, grid, parent, g_cost, h_cost, move):
        self.parent = parent
        self.grid = grid
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.total_cost = g_cost + h_cost
        self.move = move

class SlidingAStar:
    def __init__(self, size=3):
        self.size = size
        self.puzzle = SlidingPuzzle(size)
        self.goal = self.puzzle.get_goal_state()
        self.node_list = []  # Stores all explored nodes
        self.visted_node_list = []  # Stores all visited nodes

    def cost_function(self, g_cost):
        return g_cost + 1

    def get_move_weight(self, tile_curr, tile_g, blank):
        # Manhatten to goal
        d_tile = self.get_manhattan_dist(tile_curr, tile_g)

        # Blocked moves = available moves - legal moves
        b_m = 4
        if tile_curr[0] == 0:  # 0th row so no down moves
            b_m -= 1
        if tile_curr[0] == self.size: # last row so no up moves
            b_m -= 1
        if tile_curr[1] == 0:  # 0th column so no left moves
            b_m -= 1
        if tile_curr[1] == self.size:  # last column so no roght moves
            b_m -= 1

        if self.get_manhattan_dist(tile_curr, blank) == 1:  # If there is legal move
            b_m -= 1 

        # Calculate number of obstacles (d_tile - 1 if blank_space)
        # check for blank_space between tile_curr and tile_g
        # d_tile = dist(tile_curr, blank) + dist(blank, tile_g) if on path
        curr_to_blank_dist = self.get_manhattan_dist(tile_curr, blank)
        blank_to_goal_dist = self.get_manhattan_dist(blank, tile_g)
        if d_tile == (curr_to_blank_dist + blank_to_goal_dist):
            n_obj = d_tile - 1
        else:
            n_obj = d_tile


        # final h = dist(c, g) + b_m + n_obj + n_obj*4
        final_h = d_tile + b_m + n_obj * 5
        return final_h

    def get_hamming_dist(self, curr):
         # Flatten the arrays for easy comparison
        current_flat = curr.flatten()
        goal_flat = self.puzzle.get_goal_state().flatten()
        
        # Ignore the blank (0) when comparing
        mismatch = (current_flat != goal_flat) & (goal_flat != 0)
        
        return np.sum(mismatch)

    def get_manhattan_dist(self, s_point, g_point):
        return abs(g_point[0] - s_point[0]) + abs(g_point[1] - s_point[1])

    def heuristic_function(self, c_state):
        """
        Calculates the heuristic cost, if the node is moved
        to empty_space.

        Steps:
            1. Find empty space (element==0)
            2. Swap node with empty space
            3. Calculate heuristic between goal and swapper matrix
            4. Return heuristic
        """
        # goal_pos = self.puzzle.get_goal_state()
        # blank_r = np.where(c_state == 0)[0].item()
        # blank_c = np.where(c_state == 0)[1].item()

        # h_val = 0
        # for i in range(0, self.size**2):
        #     g_val = np.where(goal_pos == i)
        #     c_val = np.where(c_state == i)
            # h_val += self.get_manhattan_dist(c_val, g_val)
            # h_val += self.get_move_weight(c_val, g_val, (blank_r, blank_c))

        return self.get_hamming_dist(c_state)
        # return h_val

    def get_best_node(self) -> Node:
        """
        Calculates the total cost of each node and returns the
        one with least value

        Args:
            node_list: List of available nodes to explore cost for
            target: current element to explore for
            g_cost: current traversal cost

        Returns:
            move, leading to least cost node location of node with least cost
        """
        self.node_list = sorted(self.node_list, key=lambda i: i.total_cost)
        node = self.node_list.pop(0)

        self.visted_node_list.append(node)
        return node

    def is_explored(self, grid):
        """Checks if given node is explored or visited"""
        in_nodes = any(np.array_equal(grid, x.grid) for x in self.node_list)
        in_visited = any(np.array_equal(grid, x.grid) for x in self.visted_node_list)
        if in_nodes or in_visited:
            return True
        return False


    def explore_moves(self, node: Node):
        """Explores nodes with cost around blank space"""
        c_state = node.grid
        row = np.where(c_state == 0)[0].item()  # find empty space
        col = np.where(c_state == 0)[1].item()  # find empty space

        # Get possible moves
        up_node = (row - 1, col) if row > 0 else (-1, -1)
        right_node = (row, col + 1) if col < self.size-1 else (-1, -1)
        down_node = (row + 1, col) if row < self.size-1 else (-1, -1)
        left_node = (row, col - 1) if col > 0 else (-1, -1)

        explored_nodes = []
        for element in (up_node, right_node, down_node, left_node):
            if any(np.array_equal(element, x) for x in explored_nodes):
                continue

            if element[0] == element[1] == -1:
                continue

            n_grid = c_state.copy()
            n_grid[row][col], n_grid[element[0]][element[1]] = n_grid[element[0]][element[1]], n_grid[row][col]

            if self.is_explored(n_grid):
                continue

            new_node = Node(
                n_grid,
                node,
                node.g_cost + 1,
                self.heuristic_function(n_grid),
                element
            )

            explored_nodes.append(element)
            self.node_list.append(new_node)

    def solve(self):
        """
        Main loop which executes the puzzle and A-Star

        A-Star:
            1. Initialise, g, h, f, r to 0
            2. Initialize i as i to size^2 -1
            3. Get current state
            4. Explore nodes and calculate their h
            5. Select the one with least h as r.
            8. Exit loop when i = 0 or time exceeds time_thresh.
        """
        # self.puzzle.run()
        self.puzzle.set_state(np.array(
            [[3, 4, 8],
             [7, 2, 1],
             [0, 5, 6]]
        ),
        (2, 0))
        print("Initial state:")
        print(self.puzzle.get_state())
        print()

        start_state = self.puzzle.get_state()
        goal_state = self.puzzle.get_goal_state()
        c_node = Node(start_state, -1, 0, self.heuristic_function(start_state), (-1, -1))
        input("Press enter to continue: ")
        while True:
            self.explore_moves(c_node)  # Explores possible nodes w.r.t possible moves
            best_node = self.get_best_node()
            # print(f"node gcost, total cost: {best_node.g_cost, best_node.total_cost}")

            if np.array_equal(best_node.grid, goal_state):
                break

            c_node = best_node

        print(f"Goal g_cost, total_cost: {best_node.g_cost, best_node.total_cost}")
        move_list = []
        curr_node = best_node
        while True:
            if curr_node.parent == -1:
                break
            move_list.append(curr_node.move)
            curr_node = curr_node.parent

        num_nodes = len(self.node_list) + len(self.visted_node_list)
        num_moves = len(move_list)
        print(f"Total nodes explored: {num_nodes}")
        print(f"Number of moves to solve: {num_moves}")
        for move in move_list[::-1]:
            self.puzzle.root.update_idletasks()
            self.puzzle.move_tile(move[0], move[1])
            self.puzzle.update_display()
            time.sleep(1)
        input("Press enter to exit")


# Example usage and demonstration
if __name__ == "__main__":
    solver = SlidingAStar()
    solver.solve()
