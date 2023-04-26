import cv2
import numba
import numpy as np
from searches import dijkstra, astar, bfs, dfs
from utils import VideoWriter, idx_to_pos, pos_to_idx, build_adj_list
import time
import csv
from typing import Any
import itertools

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
                status = DONE
    return time_taken, output_table 


def solve_path(backlink_table: np.ndarray, start_node: int, stop_node: int):
    """Solve the path from the distance table."""
    path = [stop_node]
    while path[-1] != start_node:
        path.append(backlink_table[path[-1]])
    for node in path[::-1]:
        yield node


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
                time_taken, output_table = e.value
                done[i] = True
                output_data[agent].append({'time': time_taken, 'output_table': output_table})
                print()
                print(f"{agent} took {time_taken} seconds")
                print(f"{agent} path length: {output_table[0][center_idx]}")

        video_writer.write(background)

def writeDict(d, name):
    # outputs the times as a csv
    with open('generated/' + name + '.csv', mode = 'w', newline='') as file:
        writer = csv.writer(file)
        for key, value in d.items():
            writer.writerow([key,value])

if __name__ == '__main__':
    # initial maze
    maze = cv2.imread("generated/maze.png")
    
    # Define the agent's starting position
    last_y = maze.shape[0] - 2
    last_x = maze.shape[1] - 2
    starting_positions = [(1, 1), (1, last_y), (last_x, last_y), (last_x, 1)]
    center_idx = (maze.shape[1] * maze.shape[0]) // 2
    
    agents = [
        ("Dijkstra", dijkstra, np.array((0, 0, 255), dtype=np.uint8)),
        ("A*", lambda *args, **kwargs: astar(*args, heuristic=lambda node: manhattan_distance(node, center_idx, maze.shape[1]), **kwargs), np.array((0, 255, 0), dtype=np.uint8)),
        ("BFS", bfs, np.array((255, 0, 0), dtype=np.uint8)),
        ("DFS", dfs, np.array((255, 255, 0), dtype=np.uint8)),
    ]

    output_data: dict[str, list[dict[str, Any]]] = {}

    # ffmpeg -i generated/video/maze_%02d.png -r 360 maze.mp4 to covert folder of pngs to video
    with VideoWriter('generated/maze.mpg', 3600, (maze.shape[1], maze.shape[0])) as video_writer:
        for name, _, _ in agents:
            output_data[name] = []

        for rotation in range(4):
            current_starting_positions = list(itertools.islice(itertools.cycle(starting_positions), rotation, rotation + len(agents)))
            animate_agents(maze, agents, current_starting_positions, video_writer)

