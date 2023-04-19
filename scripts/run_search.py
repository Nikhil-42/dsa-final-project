import cv2
import math
import numba
import numpy as np
from searches import *
from time import sleep

@numba.njit(cache=True, parallel=True)
def build_adj_list(maze: np.ndarray):
    """Build an adjacency list from a maze."""
    
    # Create a mask that filters invalid neighbor positions
    def in_range(x, y):
        return (0 < x) * (x < maze.shape[1]-1) * (0 < y) * (y < maze.shape[0]-1)
    
    # Define an iterator of relative neighbor positions
    neighbor_deltas = np.array(((0, -1), (0, 1), (-1, 0), (1, 0)))
    
    # Conversion factor from pixel height delta. Maps [-256,256] to (0, inf)
    remap = lambda x: math.exp(x/16.0)
    
    adj_list = (np.empty((maze.shape[0] * maze.shape[1], 4), dtype=np.float32), np.empty((maze.shape[0] * maze.shape[1], 4), dtype=np.int32))
    for i in numba.prange(adj_list[0].shape[0]):
        x, y = i % maze.shape[1], i // maze.shape[1]
        
        # Compute neighbor positions
        neighbors = neighbor_deltas + np.array((x, y))

        adj_list[1][i] = np.array([nx + ny * maze.shape[1] for nx, ny in neighbors])
        adj_list[0][i] = np.array([remap(maze[ny, nx] - maze[y, x]) if in_range(nx, ny) else np.inf for nx, ny in neighbors])
    
    return adj_list

def animate_agents(maze: np.ndarray):
    """Animate the agents moving through the maze. Maze should be a grayscale image."""

    # BGR image of the maze
    frame = np.empty((maze.shape[0], maze.shape[1], 3), dtype=np.uint8)
    frame[:, :, 0] = maze
    frame[:, :, 1] = maze
    frame[:, :, 2] = maze
    
    video_writer = cv2.VideoWriter('maze.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame.shape[1], frame.shape[0]))

    try:
        # Define the agent's starting position
        runner_pos = np.random.randint(0, len(maze) // 2, size=(2,)) * 2 + 1
        chaser_pos = np.random.randint(1, len(maze) // 2, size=(2,)) * 2 + 1
        
        # Define the adjacency list for the maze
        adj_list = build_adj_list(maze)

        # Make readonly positions for the runner and chaser positions to prevent the search algorithms from modifying them
        runner_pos_view = np.array(runner_pos[0] + runner_pos[1] * maze.shape[1])
        chaser_pos_view = np.array(chaser_pos[0] + chaser_pos[1] * maze.shape[1])

        # Instantiate the agents' search algorithms
        chaser_pathing = dijkstra(adj_list, chaser_pos[0] + chaser_pos[1] * maze.shape[1], runner_pos_view)
        runner_pathing = bfs(adj_list, runner_pos[0] + runner_pos[1] * maze.shape[1], chaser_pos_view)
        
        # Run the search algorithms until they meet
        while any(chaser_pos != runner_pos):
            try:
                chaser_search_pos = next(chaser_pathing)
                frame[chaser_search_pos // len(maze), chaser_search_pos % len(maze), 0] = 255
            except StopIteration as e:
                if np.array_equal(chaser_pos, runner_pos):
                    break
                else:
                    output_table = e.value
                    chaser_pathing = dijkstra(adj_list, chaser_pos[0] + chaser_pos[1] * maze.shape[1], runner_pos_view)
            
            try:
                runner_search_pos = next(runner_pathing)
                frame[runner_search_pos // len(maze), runner_search_pos % len(maze), 2] = 255
            except StopIteration as e:
                if np.array_equal(runner_pos, chaser_pos):
                    break
                else:
                    output_table = e.value
                    runner_pathing = bfs(adj_list, runner_pos[0] + runner_pos[1] * maze.shape[1], chaser_pos_view)
            
            video_writer.write(frame)
    except Exception as e:
        video_writer.release()
        raise e
    
if __name__ == '__main__':
    maze = cv2.imread("maze.png", cv2.IMREAD_GRAYSCALE)
    maze = cv2.resize(maze, (maze.shape[0] // 16, maze.shape[1] // 16), interpolation=cv2.INTER_NEAREST)
    animate_agents(maze)