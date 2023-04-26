import random
import numpy as np
import numba # type: ignore
import cv2 # type: ignore

def add_terrain(maze, average_radius=60, colors=((255,0,0), (0,255,0), (0,0,255))):
    height, width, channels = maze.shape
    # size length^2 / average radius
    amount_of_circles = int((height / average_radius)**2)
    color_texture = np.zeros_like(maze)

    for _ in range(amount_of_circles):
        # randomizes coordinates
        random_x = random.randint(1, width)
        random_y = random.randint(1, height)
        random_color_index = np.random.randint(len(colors))
        random_color = colors[random_color_index]

        radius = int(random.normalvariate(average_radius, 5))
        center = (random_x, random_y)

        # draws the circle
        cv2.circle(color_texture, center, radius, random_color, -1)

    # Interpolate the color texture to the maze
    kernel_size = (average_radius // 2) + (average_radius % 2 + 1)
    cv2.blur(color_texture, (kernel_size, kernel_size), color_texture)
    
    # adds color to the maze
    maze = cv2.add(maze, color_texture)
    
    return  maze
    
# Maze Generation Logic from https://inventwithpython.com/recursion/chapter11.html

@numba.njit(cache=True)
def generate_maze(width: int, height: int):
    """Generate a maze using the depth-first search algorithm."""
    
    # Must be odd to prevent double walls on far edges
    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1
    
    # Define terrian types
    EMPTY, WALL = 0, 1
    
    # Create the grid filled with walls
    grid = np.ones((height, width), dtype=np.uint8)
    
    # Create a mask that filters invalid neighbor positions
    def in_range(positions):
        return (0 < positions[:, 0]) * (positions[:, 0] < width-1) * (0 < positions[:, 1]) * (positions[:, 1] < height-1)
    
    # Define an iterator of relative neighbor positions
    neighbor_deltas = np.array(((0, -2), (0, 2), (-2, 0), (2, 0)))
    
    # DFS starting with the top left corner
    stack = []
    stack.append(np.array((1, 1)))
    grid[1, 1] = EMPTY
    
    while stack:  # Pythonic check for non-empty list
        # Peek at the top of the stack
        current = stack[-1] 
        # Compute neighbor positions
        neighbors = neighbor_deltas + current 
        # Filter invalid positions
        neighbors = neighbors[in_range(neighbors)] 
        # Filter visited positions
        unvisited_neighbors = [n for n in neighbors if grid[n[1], n[0]] == WALL]
        
        # If there are unvisited neighbors, choose one at random
        if len(unvisited_neighbors) > 0:
            neighbor_choice = np.random.choice(len(unvisited_neighbors))
            neighbor_pos = unvisited_neighbors[neighbor_choice]
            wall_pos = current + (neighbor_pos - current) // 2
            # Remove the wall between the current cell and the neighbor
            grid[wall_pos[1], wall_pos[0]] = EMPTY
            # Remove the wall also mark the neighbor as visited
            grid[neighbor_pos[1], neighbor_pos[0]] = EMPTY
            # Add the neighbor to the stack
            stack.append(neighbor_pos)
        else:
            # If there are no unvisited neighbors, backtrack
            stack.pop()
    
    return grid

if __name__ == '__main__':
    WIDTH = 317
    HEIGHT = 317
    
    maze = generate_maze(WIDTH, HEIGHT)
    
    # Convert to BGR
    maze = cv2.cvtColor((maze * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # Open center
    maze[HEIGHT//2-2:HEIGHT//2+3, WIDTH//2-2:WIDTH//2+3] = [0, 0, 0]
    
    GRAY = (63, 63, 63)
    YELLOW = (0, 63, 63)
    CYAN = (63, 63, 0)
    MAGENTA = (63, 0, 63)
    RED = (0, 0, 63)
    GREEN = (0, 63, 0)
    BLUE = (63, 0, 0)    
    
    AVERAGE_RADIUS = 40

    cv2.imwrite("generated/maze_gray.png", cv2.resize(maze, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST))
    
    maze = add_terrain(maze, AVERAGE_RADIUS, (GRAY, YELLOW, CYAN, MAGENTA, RED, GREEN, BLUE))

    cv2.imwrite("generated/maze.png", maze)
