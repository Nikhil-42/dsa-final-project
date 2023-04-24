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
        input_args['pix_fmt'] = 'rgba'
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
    frame = np.zeros((maze.shape[0], maze.shape[1], 4), dtype=np.uint8)
    
    # ffmpeg -i generated/video/maze_%02d.png -r 360 maze.mp4 to covert folder of pngs to video
    video_writer = VideoWriter('generated/maze.mp4', 360, (frame.shape[1], frame.shape[0]))

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
        bfs_found = False
        dijkstra_found = False
        a_star_found = False
        bellman_ford_found = True

        start_time = time.time()
        bfs_time = 0.0
        dijkstra_time = 0.0
        a_star_time = 0.0
        bellman_ford_time = 0.0
        while not (bfs_found and dijkstra_found and a_star_found and bellman_ford_found):
            # BFS
            if not bfs_found:
                try:
                    start_time = time.time()
                    bfs_runner_search_pos = next(bfs_runner_pathing)
                    bfs_time += time.time() - start_time
                    frame[bfs_runner_search_pos // len(maze), bfs_runner_search_pos % len(maze), 2] = 255
                except StopIteration as e:
                    bfs_found = (bfs_runner_search_pos == center)
            
            # Dijkstra's
            if not dijkstra_found:
                try:
                    start_time = time.time()
                    dijkstra_runner_search_pos = next(dijkstra_runner_pathing)
                    dijkstra_time += time.time() - start_time
                    frame[dijkstra_runner_search_pos // len(maze), dijkstra_runner_search_pos % len(maze), 0] = 255 
                except StopIteration as e:
                    dijkstra_found = dijkstra_runner_search_pos == center
            
            # A_Star
            if not a_star_found:
                try:
                    start_time = time.time()
                    a_star_runner_search_pos = next(a_star_runner_pathing)
                    a_star_time += time.time() - start_time
                    frame[a_star_runner_search_pos // len(maze), a_star_runner_search_pos % len(maze), 1] = 255
                except StopIteration as e:
                    a_star_found = (a_star_runner_search_pos == center)
            
            # Bellman Ford
            if not bellman_ford_found:
               try:
                   start_time = time.time()
                   bellman_ford_runner_search_pos = next(bellman_ford_pathing)
                   bellman_ford_time += time.time() - start_time
                   frame[bellman_ford_runner_search_pos // len(maze), bellman_ford_runner_search_pos % len(maze), 1] = 120
               except StopIteration as e:
                   bellman_ford_found = (bellman_ford_runner_search_pos == center)

            video_writer.write(frame)
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
    maze = cv2.imread("generated/maze.png", cv2.IMREAD_GRAYSCALE)
    animate_agents(maze)