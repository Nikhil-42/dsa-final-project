from enum import Enum
import cv2
import math
import numba
import ffmpeg
import numpy as np
from searches import *
import time

class VideoWriter:
    def __init__(self, filepath, fps, shape, input_args: dict = None, output_args: dict = None):
        if input_args is None:
            input_args = {}
        if output_args is None:
            output_args = {}
        input_args['framerate'] = fps
        input_args['pix_fmt'] = 'bgr24'
        input_args['s'] = '{}x{}'.format(*shape)
        self.filepath = filepath
        self.shape = shape
        self.input_args = input_args
        self.output_args = output_args
        self.process = (
            ffmpeg
                .input('pipe:', format='rawvideo', **input_args)
                .filter('fps', fps=30, round='up')
                .output(self.filepath, **output_args)
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )
    
    def write(self, frame): 
        self.process.stdin.write(
            frame.astype(np.uint8).tobytes()
        )

    def release(self):
        self.process.stdin.close()
        self.process.wait()


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
    remap = lambda x: x if x != 255 else np.inf
    
    adj_list = (np.empty((maze.shape[0] * maze.shape[1], 4), dtype=np.float32), np.empty((maze.shape[0] * maze.shape[1], 4), dtype=np.int32))
    
    for i in numba.prange(adj_list[0].shape[0]):
        x, y = i % maze.shape[1], i // maze.shape[1]
        
        # Compute neighbor positions
        neighbors = neighbor_deltas + np.array((x, y))

        adj_list[1][i] = np.array([nx + ny * maze.shape[1] for nx, ny in neighbors])
        adj_list[0][i] = np.array([0.5 * remap(maze[ny, nx]) + 0.5 * remap(maze[y, x]) if in_range(nx, ny) else np.inf for nx, ny in neighbors])
    
    return adj_list


def solve_path(backlink_table: np.ndarray, start_node: int, stop_node: int):
    """Solve the path from the distance table."""
    path = [stop_node]
    while path[-1] != start_node:
        path.append(backlink_table[path[-1]])
    for node in path[::-1]:
        yield node

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
    background = maze
    paths = np.zeros_like(maze)
    maze = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY)
    
    # ffmpeg -i generated/video/maze_%02d.png -r 360 maze.mp4 to covert folder of pngs to video
    video_writer = VideoWriter('generated/maze.mp4', 360, (paths.shape[1], paths.shape[0]))

    try:
        # Define the agent's starting position
        last_y = maze.shape[0] - 2
        last_x = maze.shape[1] - 2
        bfs_runner_pos = [1, 1]
        dijkstra_runner_pos = [last_x, last_y]
        a_star_runner_pos = [1, last_y]
        bellman_ford_pos = [last_x, last_y]
        center = (maze.shape[1] * maze.shape[0]) // 2
        
        # Define the adjacency list for the maze
        adj_list = build_adj_list(maze)

        # Instantiate the agents' search algorithms
        bfs_runner_pathing = bfs(adj_list, bfs_runner_pos[0] + bfs_runner_pos[1] * maze.shape[1], center)
        dijkstra_runner_pathing = dijkstra(adj_list, dijkstra_runner_pos[0] + dijkstra_runner_pos[1] * maze.shape[1], center)
        a_star_runner_pathing = a_star(adj_list, a_star_runner_pos[0] + a_star_runner_pos[1] * maze.shape[1], center, lambda node: manhattan_distance(node, center, maze.shape[1]))
        bellman_ford_pathing = bellman_ford(adj_list, bellman_ford_pos[0] + bellman_ford_pos[1] * maze.shape[1], center)

        
        # Run the search algorithms until they meet
        SEARCHING, BACKTRACKING, WALKING, DONE = 0, 1, 2, 3
        bfs_status = SEARCHING
        dijkstra_status = SEARCHING
        a_star_status = SEARCHING
        bellman_ford_status = DONE 

        bfs_time = 0.0
        dijkstra_time = 0.0
        a_star_time = 0.0
        bellman_ford_time = 0.0
        while not (bfs_status == DONE and dijkstra_status == DONE and a_star_status == DONE and bellman_ford_status == DONE):
            # BFS
            if bfs_status == SEARCHING:
                try:
                    start_time = time.time()
                    bfs_runner_search_pos = next(bfs_runner_pathing)
                    bfs_time += time.time() - start_time
                    paths[bfs_runner_search_pos // len(maze), bfs_runner_search_pos % len(maze), 2] = 255
                    background[bfs_runner_search_pos // len(maze), bfs_runner_search_pos % len(maze)] = paths[bfs_runner_search_pos // len(maze), bfs_runner_search_pos % len(maze)]
                except StopIteration as e:
                    bfs_status = BACKTRACKING if bfs_runner_search_pos == center else DONE
                    if bfs_status == BACKTRACKING:
                        bfs_output_table = e.value
                        bfs_bacttrack = solve_path(bfs_output_table[1], bfs_runner_pos[0] + bfs_runner_pos[1] * maze.shape[1], center)
                        print('BFS found in {} seconds'.format(bfs_time))
            elif bfs_status == BACKTRACKING:
                try:
                    next_node = next(bfs_bacttrack)
                    paths[next_node // maze.shape[1], next_node % maze.shape[1], :] = 0
                    background[next_node // maze.shape[1], next_node % maze.shape[1]] = paths[next_node // maze.shape[1], next_node % maze.shape[1]]
                except StopIteration as e:
                    bfs_status = DONE
            
                
            # Dijkstra's
            if dijkstra_status == SEARCHING:
                try:
                    start_time = time.time()
                    dijkstra_runner_search_pos = next(dijkstra_runner_pathing)
                    dijkstra_time += time.time() - start_time
                    paths[dijkstra_runner_search_pos // len(maze), dijkstra_runner_search_pos % len(maze), 0] = 255 
                    background[dijkstra_runner_search_pos // len(maze), dijkstra_runner_search_pos % len(maze)] = paths[dijkstra_runner_search_pos // len(maze), dijkstra_runner_search_pos % len(maze)]
                except StopIteration as e:
                    dijkstra_status = BACKTRACKING if dijkstra_runner_search_pos == center else DONE
                    if dijkstra_status == BACKTRACKING:
                        dijkstra_output_table = e.value
                        dijkstra_bacttrack = solve_path(dijkstra_output_table[1], dijkstra_runner_pos[0] + dijkstra_runner_pos[1] * maze.shape[1], center)
                        print('dijkstra found in {} seconds'.format(dijkstra_time))
            elif dijkstra_status == BACKTRACKING:
                try:
                    next_node = next(dijkstra_bacttrack)
                    paths[next_node // maze.shape[1], next_node % maze.shape[1], :] = 0
                    background[next_node // maze.shape[1], next_node % maze.shape[1]] = paths[next_node // maze.shape[1], next_node % maze.shape[1]]
                except StopIteration as e:
                    dijkstra_status = DONE
            
            # A_Star
            if a_star_status == SEARCHING:
                try:
                    start_time = time.time()
                    a_star_runner_search_pos = next(a_star_runner_pathing)
                    a_star_time += time.time() - start_time
                    paths[a_star_runner_search_pos // len(maze), a_star_runner_search_pos % len(maze), 1] = 255
                    background[a_star_runner_search_pos // len(maze), a_star_runner_search_pos % len(maze)] = paths[a_star_runner_search_pos // len(maze), a_star_runner_search_pos % len(maze)]
                except StopIteration as e:
                    a_star_status = BACKTRACKING if a_star_runner_search_pos == center else DONE
                    if a_star_status == BACKTRACKING:
                        a_star_output_table = e.value
                        a_star_bacttrack = solve_path(a_star_output_table[1], a_star_runner_pos[0] + a_star_runner_pos[1] * maze.shape[1], center)
                        print('a_star found in {} seconds'.format(a_star_time))
            elif a_star_status == BACKTRACKING:
                try:
                    next_node = next(a_star_bacttrack)
                    paths[next_node // maze.shape[1], next_node % maze.shape[1], :] = 0
                    background[next_node // maze.shape[1], next_node % maze.shape[1]] = paths[next_node // maze.shape[1], next_node % maze.shape[1]]
                except StopIteration as e:
                    a_star_status = DONE
            
            # Bellman Ford
            if bellman_ford_status == SEARCHING:
                try:
                    start_time = time.time()
                    bellman_ford_runner_search_pos = next(bellman_ford_pathing)
                    bellman_ford_time += time.time() - start_time
                    paths[bellman_ford_runner_search_pos // len(maze), bellman_ford_runner_search_pos % len(maze), 1] = 255
                    background[bellman_ford_runner_search_pos // len(maze), bellman_ford_runner_search_pos % len(maze)] = paths[bellman_ford_runner_search_pos // len(maze), bellman_ford_runner_search_pos % len(maze)]
                except StopIteration as e:
                    bellman_ford_status = BACKTRACKING if bellman_ford_runner_search_pos == center else DONE
                    bellman_ford_output_table = e.value
                    bellman_ford_bacttrack = solve_path(bellman_ford_output_table[1], bellman_ford_runner_pos[0] + bellman_ford_runner_pos[1] * maze.shape[1], center)
                    print('bellman_ford found in {} seconds'.format(bellman_ford_time))
            elif bellman_ford_status == BACKTRACKING:
                try:
                    next_node = next(bellman_ford_bacttrack)
                    paths[next_node // maze.shape[1], next_node % maze.shape[1], :] = 0
                    background[next_node // maze.shape[1], next_node % maze.shape[1]] = paths[next_node // maze.shape[1], next_node % maze.shape[1]]
                except StopIteration as e:
                    bellman_ford_status = DONE

            video_writer.write(background)
    except Exception as e:
        video_writer.release()
        raise e
    finally:
        time.sleep(1)
        print()
        print("BFS time: ", bfs_time)
        print("Dijkstra time: ", dijkstra_time)
        print("A* time: ", a_star_time)
        print("Bellman Ford time: ", bellman_ford_time)
        


if __name__ == '__main__':
    maze = cv2.imread("generated/maze.png")
    animate_agents(maze)