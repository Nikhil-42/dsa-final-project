import cv2
import math
import numba
import numpy as np
from searches import *
import time

@numba.njit(cache=True, parallel=True)
def build_adj_list(maze: np.ndarray):
    """Build an adjacency list from a maze."""
    
    # Create a mask that filters invalid neighbor positions
    def in_range(x, y):
        return (0 < x) * (x < maze.shape[1]-1) * (0 < y) * (y < maze.shape[0]-1)
    
    # Define an iterator of relative neighbor positions
    neighbor_deltas = np.array(((0, -1), (0, 1), (-1, 0), (1, 0)))
    
    # Conversion factor from pixel height delta. Maps [-256,256] to (0, inf)
    # Divide by 88.7228390619 = ln(floatmax) | 256 / 88.7228390619 = 2.88539
    remap = lambda x: math.exp(x/2.85)
    
    adj_list = (np.empty((maze.shape[0] * maze.shape[1], 4), dtype=np.float32), np.empty((maze.shape[0] * maze.shape[1], 4), dtype=np.int32))
    
    for i in numba.prange(adj_list[0].shape[0]):
        x, y = i % maze.shape[1], i // maze.shape[1]
        
        # Compute neighbor positions
        neighbors = neighbor_deltas + np.array((x, y))

        adj_list[1][i] = np.array([nx + ny * maze.shape[1] for nx, ny in neighbors])
        adj_list[0][i] = np.array([remap(maze[ny, nx] - maze[y, x]) if in_range(nx, ny) else np.inf for nx, ny in neighbors])
    
    return adj_list

# this is for our A* algorithm
def manhattan_distance(source, stop_node, maze_width):
    source_x = source % maze_width
    source_y = source // maze_width
    goal_x = stop_node % maze_width
    goal_y = stop_node // maze_width
    return np.abs(source_x - goal_x) + np.abs(source_y - goal_y)

def animate_agents(maze: np.ndarray):
    """Animate the agents moving through the maze. Maze should be a grayscale image."""

    # BGR image of the maze
    frame = np.empty((maze.shape[0], maze.shape[1], 3), dtype=np.uint8)
    frame[:, :, 0] = maze
    frame[:, :, 1] = maze
    frame[:, :, 2] = maze
    
    video_writer = cv2.VideoWriter('generated/maze.avi', cv2.VideoWriter_fourcc(*'XVID'), 300, (frame.shape[1], frame.shape[0]))

    try:
        # Define the agent's starting position
        bfs_runner_pos = [1, 1]
        dijkstra_runner_pos = [313, 313]
        a_star_runner_pos = [1, 313]
        bellman_ford_pos = [313, 313]
        center = (maze.shape[1] * maze.shape[0]) // 2
        
        # Define the adjacency list for the maze
        adj_list = build_adj_list(maze)

        # Instantiate the agents' search algorithms
        bfs_runner_pathing = bfs(adj_list, bfs_runner_pos[0] + bfs_runner_pos[1] * maze.shape[1], center)
        dijkstra_runner_pathing = dijkstra(adj_list, dijkstra_runner_pos[0] + dijkstra_runner_pos[1] * maze.shape[1], center)
        a_star_runner_pathing = a_star(adj_list, a_star_runner_pos[0] + a_star_runner_pos[1] * maze.shape[1],maze.shape[1], manhattan_distance ,center)
        bellman_ford_pathing = bellman_ford(adj_list, bellman_ford_pos[0] + bellman_ford_pos[1] * maze.shape[1], center)

        
        # Run the search algorithms until they meet
        bfs_found = False
        dijkstra_found = False
        a_star_found = False
        bellman_ford_found = False

        start_time = time.time()
        bfs_time = 0
        dijkstra_time = 0
        a_star_time = 0
        bellman_ford_time = 0
        while not (bfs_found and dijkstra_found and a_star_found and bellman_ford_found):
            # BFS
            if not bfs_found:
                try:
                    bfs_runner_search_pos = next(bfs_runner_pathing)
                    frame[bfs_runner_search_pos // len(maze), bfs_runner_search_pos % len(maze), 2] = 255
                except StopIteration as e:
                    if bfs_runner_search_pos == center:
                        end_time = time.time()
                        bfs_time = end_time - start_time
                        bfs_found = True
            
            # Dijkstra's
            if not dijkstra_found:
                try:
                    dijkstra_runner_search_pos = next(dijkstra_runner_pathing)
                    frame[dijkstra_runner_search_pos // len(maze), dijkstra_runner_search_pos % len(maze), 0] = 255 
                except StopIteration as e:
                    if dijkstra_runner_search_pos == center:
                        end_time = time.time()
                        dijkstra_time= end_time - start_time
                        dijkstra_found = True
            
            # A_Star
            if not a_star_found:
                try:
                    a_star_runner_search_pos = next(a_star_runner_pathing)
                    frame[a_star_runner_search_pos // len(maze), a_star_runner_search_pos % len(maze), 1] = 255
                except StopIteration as e:
                    if a_star_runner_search_pos == center:
                        end_time = time.time()
                        a_star_time= end_time - start_time
                        a_star_found = True
            
            # Bellman Ford
            if not bellman_ford_found:
                try:
                    bellman_ford_runner_search_pos = next(bellman_ford_pathing)
                    frame[bellman_ford_runner_search_pos // len(maze), bellman_ford_runner_search_pos % len(maze), 1] = 120
                except StopIteration as e:
                    if bellman_ford_runner_search_pos == center:
                        end_time = time.time()
                        bellman_ford_time = end_time - start_time
                        bellman_ford_found = True

            video_writer.write(frame)
        print("BFS time: ", bfs_time)
        print("Dijkstra time: ", dijkstra_time)
        print("A* time: ", a_star_time)
        print("Bellman Ford time: ", bellman_ford_time)
    except Exception as e:
        video_writer.release()
        raise e
    
if __name__ == '__main__':
    maze = cv2.imread("generated/maze.png", cv2.IMREAD_GRAYSCALE)
    animate_agents(maze)