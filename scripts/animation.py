import cv2  # type: ignore
import numpy as np
from utils import VideoWriter, idx_to_pos, pos_to_idx, build_adj_list
import time
from typing import Any

output_data: dict[str, dict[str, Any]] = {}


def solve_path(backlink_table: np.ndarray, start_node: int, stop_node: int):
    """Solve the path from the distance table."""
    path = [stop_node]
    while path[-1] != start_node:
        path.append(int(backlink_table[path[-1]]))
    for node in path[::-1]:
        yield node
    return path[::-1]


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
    pathings = [pathing_func(adj_list, pos_to_idx(pos, WIDTH), center_idx) for (
        _, pathing_func, _), pos in zip(search_agents, starting_positions)]
    # Instantiate the agents' search runners
    searches = [search(pos_to_idx(pos, WIDTH), center_idx, pathing, paths, color)
                for pathing, (_, _, color), pos in zip(pathings, search_agents, starting_positions)]
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
                output_data[agent]['path_lengths'].append(
                    int(sum(maze[idx_to_pos(node, WIDTH)[::-1]] for node in path)))
                print()
                print(f"{agent} took {time_taken} seconds")
                print(
                    f"{agent} path length: {sum(maze[tuple(idx_to_pos(node, WIDTH)[::-1])] for node in path)}")
                output_data[agent]['finish_frames'].append(
                    video_writer.frame_count)

        video_writer.write(background)
    return background


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
