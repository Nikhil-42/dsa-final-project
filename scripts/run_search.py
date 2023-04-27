import cv2 # type: ignore
import numba # type: ignore
import numpy as np
from searches import dijkstra, astar, bfs, dfs, bellman_ford
from utils import VideoWriter, idx_to_pos, pos_to_idx, build_adj_list
import time
from typing import Any
import itertools
import json
import argparse
import sys


def solve_path(backlink_table: np.ndarray, start_node: int, stop_node: int):
    """Solve the path from the distance table."""
    path = [stop_node]
    while path[-1] != start_node:
        path.append(int(backlink_table[path[-1]]))
    for node in path[::-1]:
        yield node
    return path[::-1]


def search(start_idx, end_idx, pathing, paths: np.ndarray, color):
    """Search for a path from start_pos to end_pos using pathing."""   
    SEARCHING, BACKTRACKING, WALKING, DONE = range(4)
    WIDTH = paths.shape[1]
    time_taken = 0.0
    path_length = 0.0
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
                path = e.value
                status = DONE
    return time_taken, path 


# this is for our A* algorithm
@numba.njit(cache=True)
def manhattan_distance(source, stop_node, maze_width):
    source_x = source % maze_width
    source_y = source // maze_width
    goal_x = stop_node % maze_width
    goal_y = stop_node // maze_width
    return np.abs(source_x - goal_x) + np.abs(source_y - goal_y)


def animate_agents(maze: np.ndarray, search_agents, starting_positions, video_writer: VideoWriter):
    """Animate the agents moving through the maze. Maze should be a grayscale image.
        agent = (name: str, pathing_func, pos: np.ndarray[int], color: np.ndarray[np.uint8])
    """
    HEIGHT, WIDTH = maze.shape[:2]

    # BGR image of the maze
    background = maze.copy()
    paths = np.zeros_like(maze, dtype=np.uint8)
    maze = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY)

    # Define the adjacency list for the maze
    adj_list = build_adj_list(maze)
    
    # Label the center index of the maze
    center_idx = (maze.shape[1] * maze.shape[0]) // 2    

    # Instantiate the agents' search algorithms
    pathings = [pathing_func(adj_list, pos_to_idx(pos, WIDTH), center_idx) for (_, pathing_func, _), pos in zip(search_agents, starting_positions)]
    # Instantiate the agents' search runners
    searches = [search(pos_to_idx(pos, WIDTH), center_idx, pathing, paths, color) for pathing, (_, _, color), pos in zip(pathings, search_agents, starting_positions)]
    done = [False] * len(search_agents)
    
    # Run the search algorithms until they meet in the middle
    while not all(done):
        for i, (agent, _, _) in enumerate(search_agents):
            if done[i]:
                continue
            try:
                next_pos = next(searches[i])
                background[next_pos[::-1]] = paths[next_pos[::-1]]
            except StopIteration as e:
                time_taken, path = e.value
                done[i] = True
                output_data[agent]['times'].append(time_taken)
                output_data[agent]['paths'].append(path)
                print()
                print(f"{agent} took {time_taken} seconds")
                print(f"{agent} path length: {sum(maze[tuple(idx_to_pos(node, WIDTH)[::-1])] for node in path)}")
                output_data[agent]['finish_frames'].append(video_writer.frame_count)

        video_writer.write(background)
    return background

# sets up arguments
def getArguments(): 
    parser = argparse.ArgumentParser(description='Run a search algorithm on a maze') 
    parser.add_argument("-d",'--dfs', help='Depth First Search', default=False, action='store_true')
    parser.add_argument("-b",'--bfs', help='Breadth First Search', default=False, action='store_true')
    parser.add_argument("-dj",'--dijkstras', help='Dijkstras Algorithm', default=False,  action='store_true')
    parser.add_argument("-a",'--astar', help='A* Algorithm', default=False, action='store_true')
    #parser.add_argument("-bf",'--bellman_ford', help='Bellman Ford Algorithm', default=False, action='store_true')
    return parser.parse_args()
    

if __name__ == '__main__':
    FPS = 3600

    def astar_pathing(*args, **kwargs):
        """A* pathing algorithm."""
        kwargs['heuristic'] = lambda node: manhattan_distance(node, center_idx, maze.shape[1])
        return astar(*args, **kwargs)


    agents = [
    ]

    args = getArguments() # a boolean for each algorithm to run
    if len(sys.argv) == 1:
        raise ValueError("Please enter an algorithm to run") # no arguments
    elif len(sys.argv) > 4:
        raise ValueError("Please enter a maximum of four algorithms to run")

    if (args.dijkstras):
        agents.append(("Dijkstra", dijkstra, np.array((0, 0, 128), dtype=np.uint8)))
    elif (args.astar):
        agents.append(("A*", astar_pathing, np.array((0, 128, 0), dtype=np.uint8)))
    elif (args.bfs):
        agents.append(("BFS", bfs, np.array((128, 0, 0), dtype=np.uint8)))
    elif (args.dfs):
        agents.append(("DFS", dfs, np.array((127, 127, 0), dtype=np.uint8)))
    #elif (args.bellman_ford):
    #    agents.append(("Bellman Ford", bellman_ford, np.array((127, 127, 127), dtype=np.uint8)))
        

    
     # initial maze
    maze = cv2.imread("generated/maze.png")
    
    # Define the agent's starting position
    last_y = maze.shape[0] - 2
    last_x = maze.shape[1] - 2
    starting_positions = [(1, 1), (1, last_y), (last_x, last_y), (last_x, 1)]
    center_idx = (maze.shape[1] * maze.shape[0]) // 2


    output_data: dict[str, dict[str, Any]] = {}

    # ffmpeg -i generated/video/maze_%02d.png -r 360 maze.mp4 to covert folder of pngs to video
    with VideoWriter('generated/maze.mpg', FPS, (maze.shape[1], maze.shape[0]), output_args={'s': "512x512"}) as video_writer:
        for name, _, color in agents:
            output_data[name] = {}
            output_data[name]['times'] = []
            output_data[name]['paths'] = []
            output_data[name]['finish_frames'] = []
            output_data[name]['fps'] = FPS
            output_data[name]['color'] = color.tolist()

        for rotation in range(4):
            current_starting_positions = list(itertools.islice(itertools.cycle(starting_positions), rotation, rotation + len(agents)))
            cv2.imwrite(f"generated/solved_maze_{rotation}.png", animate_agents(maze, agents, current_starting_positions, video_writer))
        
        for name, _, color in agents:
            output_data[name]['finish_timestamps'] = [frame / FPS for frame in output_data[name]['finish_frames']]
    
    with open("generated/output.json", "w") as f:
        json.dump(output_data, f)
