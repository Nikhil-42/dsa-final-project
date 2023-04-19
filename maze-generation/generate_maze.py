import random
import numpy as np
import numba
import cv2

# Maze Generation Logic from https://inventwithpython.com/recursion/chapter11.html
def add_terrain(maze):
    # picks average radius
    average_radius = 60
    height, width, channels = maze.shape
    # size length^2 / average radius
    amount_of_circles = int(height**2 / average_radius**2)
    color_texture = np.zeros_like(maze)


    # changes colors
    green = (0, 102, 0)
    blue = (0, 0, 204)
    yellow = (204, 204, 0)
    colors = [green, blue,  yellow]

    for _ in range(amount_of_circles):
        # randomizes coordinates
        random_x = random.randint(1, width)
        random_y = random.randint(1, height)
        random_color = random.choice(colors)

        radius = int(random.normalvariate(average_radius, 5))
        center = (random_x, random_y)

        # draws the circle
        cv2.circle(color_texture, center, radius, random_color, -1)

    # adds color to the maze
    maze = cv2.add(maze, color_texture)

    
    #cv2.imshow('Circles', maze)
    #cv2.waitKey(0)

    return  maze
    
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
    width = 317
    height = 317
    
    maze = generate_maze(width, height)
    
    # # Print the maze to the console
    # for i in range(maze.shape[0]):
    #     for j in range(maze.shape[1]):
    #         if maze[i, j] == 0:
    #             print('  ', end='')
    #         else:
    #             print('##', end='')
    #     print()
    
    # Export the maze as an image
    from PIL import Image
    # Scale up to help with interpolation
    Image.fromarray(maze*255).resize((width*16, height*16), Image.NEAREST).save('maze.png')

    maze = cv2.imread("maze.png")
    maze = cv2.resize(maze, (maze.shape[0] // 16, maze.shape[1] // 16), interpolation=cv2.INTER_NEAREST)
    maze = add_terrain(maze)
    cv2.imwrite("maze.png", maze)
    