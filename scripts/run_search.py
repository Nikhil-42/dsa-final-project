import cv2  # type: ignore
import numpy as np
from searches import dijkstra, astar, bfs, dfs, manhattan_distance
from utils import VideoWriter
from animation import animate_agents, output_data
from typing import Any
import itertools
import json
import argparse
import sys

# sets up arguments


def getArguments():
    parser = argparse.ArgumentParser(
        description='Run a search algorithm on a maze')
    parser.add_argument("-d", '--dfs', help='Depth First Search',
                        default=False, action='store_true')
    parser.add_argument("-b", '--bfs', help='Breadth First Search',
                        default=False, action='store_true')
    parser.add_argument("-dj", '--dijkstras', help='Dijkstras Algorithm',
                        default=False,  action='store_true')
    parser.add_argument("-a", '--astar', help='A* Algorithm',
                        default=False, action='store_true')
    # parser.add_argument("-bf",'--bellman_ford', help='Bellman Ford Algorithm', default=False, action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    # this is for our heuristic functions
    def astar_pathing(*args, **kwargs):
        """A* pathing algorithm."""
        kwargs['heuristic'] = lambda node: manhattan_distance(
            node, center_idx, maze.shape[1])
        return astar(*args, **kwargs)

    agents = [
    ]

    args = getArguments()  # a boolean for each algorithm to run
    if len(sys.argv) == 1:
        args.dijkstras = True
        args.astar = True
        args.bfs = True
        args.dfs = True
    elif len(sys.argv) > 4:
        raise ValueError("Please enter a maximum of four algorithms to run")
        
    if (args.dijkstras):
        print("added dikjstras")
        agents.append(
            ("Dijkstra", dijkstra, np.array((0, 0, 128), dtype=np.uint8)))
    if (args.astar):
        print("added astar")
        agents.append(
            ("A*", astar_pathing, np.array((0, 128, 0), dtype=np.uint8)))
    if (args.bfs):
        print("added bfs")
        agents.append(("BFS", bfs, np.array((128, 0, 0), dtype=np.uint8)))
    if (args.dfs):
        print("added dfs")
        agents.append(("DFS", dfs, np.array((127, 127, 0), dtype=np.uint8)))
    # if (args.bellman_ford):
    #    agents.append(("Bellman Ford", bellman_ford, np.array((127, 127, 127), dtype=np.uint8)))

     # initial maze
    maze = cv2.imread("generated/maze.png")

    # Define the agent's starting position
    last_y = maze.shape[0] - 2
    last_x = maze.shape[1] - 2
    starting_positions = [(1, 1), (1, last_y), (last_x, last_y), (last_x, 1)]
    center_idx = (maze.shape[1] * maze.shape[0]) // 2

    # ffmpeg -i generated/video/maze_%02d.png -r 360 maze.mp4 to covert folder of pngs to video
    FPS = 3600

    with VideoWriter('generated/maze.mpg', FPS, (maze.shape[1], maze.shape[0]), filters=[
            ('fps', {'fps':60, 'round':'up'}),
            ('pad', {'width': 512, 'height': 512, 'x': 0, 'y': 0, 'color': 'black'}),
        ], 
                     output_args={'vcodec': 'libx264','crf': 0}) as video_writer:
        for name, _, color in agents:
            output_data[name] = {}
            output_data[name]['times'] = []
            output_data[name]['paths'] = []
            output_data[name]['finish_frames'] = []
            output_data[name]['fps'] = FPS
            output_data[name]['color'] = color.tolist()

        # Pad with zeros to make the maze a power of 2
        tmp = np.zeros((512, 512, 3))
        for rotation in range(4):
            current_starting_positions = list(itertools.islice(
                itertools.cycle(starting_positions), rotation, rotation + len(agents)))
            tmp[:maze.shape[1], :maze.shape[0]] = animate_agents(maze, agents, current_starting_positions, video_writer)
            cv2.imwrite(f"generated/solved_maze_{rotation}.png", tmp)

        for name, _, color in agents:
            output_data[name]['finish_timestamps'] = [
                frame / FPS for frame in output_data[name]['finish_frames']]



    with open("generated/output.json", "w") as f:
        json.dump(output_data, f)
