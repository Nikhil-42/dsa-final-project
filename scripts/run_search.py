import cv2
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


def idx_to_pos(index, maze_width):
    return index % maze_width, index // maze_width

def pos_to_idx(pos, maze_width):
    return pos[0] + pos[1] * maze_width 


def search(start_idx, end_idx, pathing, paths: np.ndarray, color):
    """Search for a path from start_pos to end_pos using pathing."""   
    SEARCHING, BACKTRACKING, WALKING, DONE = range(4)
    WIDTH = paths.shape[1]
    time_taken = 0.0
    status = SEARCHING    
    while not status == DONE:
        if status == SEARCHING:
            try:
                start_time = time.time()
                search_idx = next(pathing)
                time_taken += time.time() - start_time
                search_pos = idx_to_pos(search_idx, WIDTH)
                paths[search_pos[::-1]] += color
                yield search_pos
            except StopIteration as e:
                status = BACKTRACKING if search_idx == end_idx else DONE
                if status == BACKTRACKING:
                    output_table = e.value
                    backtrack = solve_path(output_table[1], start_idx, end_idx)
        elif status == BACKTRACKING:
            try:
                next_node = next(backtrack)
                next_node_pos = idx_to_pos(next_node, WIDTH)
                paths[next_node_pos[::-1]] -= color
                yield next_node_pos
            except StopIteration as e:
                status = DONE
    return time_taken


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
    HEIGHT, WIDTH = maze.shape[:2]

    # BGR image of the maze
    background = maze
    paths = np.zeros_like(maze, dtype=np.uint8)
    maze = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY)
    
    # ffmpeg -i generated/video/maze_%02d.png -r 360 maze.mp4 to covert folder of pngs to video
    video_writer = VideoWriter('generated/maze.mp4', 360, (paths.shape[1], paths.shape[0]))

    # Define the adjacency list for the maze
    adj_list = build_adj_list(maze)
    
    # Define the agent's starting position
    last_y = maze.shape[0] - 2
    last_x = maze.shape[1] - 2
    
    bfs_runner_pos = [1, 1]
    dijkstra_runner_pos = [last_x, last_y]
    a_star_runner_pos = [1, last_y]
    dfs_runner_pos = [last_x, last_y]
    
    center_idx = (maze.shape[1] * maze.shape[0]) // 2

    # Instantiate the agents' search algorithms
    bfs_runner_pathing = bfs(adj_list, pos_to_idx(bfs_runner_pos, WIDTH), center_idx)
    dijkstra_runner_pathing = dijkstra(adj_list, pos_to_idx(dijkstra_runner_pos, WIDTH), center_idx)
    a_star_runner_pathing = a_star(adj_list, pos_to_idx(a_star_runner_pos, WIDTH), center_idx, lambda node: manhattan_distance(node, center_idx, maze.shape[1]))
    dfs_pathing = dfs(adj_list, pos_to_idx(dfs_runner_pos, WIDTH), center_idx)
    
    # Instantiate the agents' search runners
    bfs_search = search(pos_to_idx(bfs_runner_pos, WIDTH), center_idx, bfs_runner_pathing, paths, np.array([255, 0, 0], dtype=np.uint8))
    dijkstra_search = search(pos_to_idx(dijkstra_runner_pos, WIDTH), center_idx, dijkstra_runner_pathing, paths, np.array([0, 255, 0], dtype=np.uint8))
    a_star_search = search(pos_to_idx(a_star_runner_pos, WIDTH), center_idx, a_star_runner_pathing, paths, np.array([0, 0, 255], dtype=np.uint8))
    dfs_search = search(pos_to_idx(dfs_runner_pos, WIDTH), center_idx, dfs_pathing, paths, np.array([255, 255, 0], dtype=np.uint8))
    
    bfs_done = False
    dijkstra_done = False
    a_star_done = False
    dfs_done = True

    times = {
        "BFS": 0.0,
        "Dijkstra's": 0.0,
        "A*": 0.0,
        "DFS": 0.0,
    }
    
    try:        
        # Run the search algorithms until they meet in the middle
        while not (bfs_done and dijkstra_done and a_star_done and dfs_done):
            # BFS
            if not bfs_done:
                try:
                    next_pos = next(bfs_search)
                    background[next_pos[::-1]] = paths[next_pos[::-1]]
                except StopIteration as e:
                    bfs_done = True
                    time["BFS:"] = e.value
                    print(f"BFS took {e.value} seconds")
            
            # Dijkstra's
            if not dijkstra_done:
                try:
                    next_pos = next(dijkstra_search)
                    background[next_pos[::-1]] = paths[next_pos[::-1]]
                except StopIteration as e:
                    dijkstra_done = True
                    times["Dijkstra's"] = e.value
                    print(f"Dijkstra's took {e.value} seconds")
                
            # A*
            if not a_star_done:
                try:
                    next_pos = next(a_star_search)
                    background[next_pos[::-1]] = paths[next_pos[::-1]]
                except StopIteration as e:
                    a_star_done = True
                    times["A*"] = e.value
                    print(f"A* took {e.value} seconds")
            
            # Bellman Ford
            if not dfs_done:
                try:
                    next_pos = next(dfs_search)
                    background[next_pos[::-1]] = paths[next_pos[::-1]]
                except StopIteration as e:
                    dfs_done = True
                    times["DFS"] = e.value
                    print(f"Bellman Ford took {e.value} seconds")

            video_writer.write(background)
    except Exception as e:
        video_writer.release()
        raise e
    return times
        

if __name__ == '__main__':
    maze = cv2.imread("generated/maze.png")
    animate_agents(maze)