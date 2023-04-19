from queue import PriorityQueue, Queue
from sys import maxsize
import numpy as np

def dijkstra(adj_list: tuple[np.ndarray, np.ndarray], source: int, stop_node: int=-1):
    """Takes an adjacency list of (weight_array, destination_array) edge entries where the index is the from node
    as well as a source and returns a table embedding the shortest path for each destination node"""
    
    output_table = (-np.ones(len(adj_list[0]), dtype=np.float32), -np.ones(len(adj_list[0]), dtype=np.int32))
    """A list of (distance_array, source_array) nodes that stores the shortest paths for each node"""
    
    path_queue: PriorityQueue[tuple[float, int, int]] = PriorityQueue()
    """A queue of the possible paths to explore sorted by path length"""
    
    for edge_weight, destination in zip(adj_list[0][source], adj_list[1][source]):
        path_queue.put((edge_weight, destination, source))
    
    while path_queue:
        current_distance, current_node, source_node = path_queue.get()
        if output_table[1][current_node] == -1:
            output_table[0][current_node] = current_distance + edge_weight
            output_table[1][current_node] = source_node
            yield current_node
            if current_node == stop_node:
                return output_table
            else:
                for edge_weight, destination in zip(adj_list[0][current_node], adj_list[1][current_node]):
                    path_queue.put((current_distance + edge_weight, destination, current_node))

    return output_table

def bfs(adj_list: tuple[np.ndarray, np.ndarray], source: int, stop_node=None):
    """Takes an adjacency list of (weight_array, destination_array) edge entries where the index is the from node
    as well as a source and returns a table embedding the shortest path for each destination node"""
    
    output_table = (-np.ones(len(adj_list[0]), dtype=np.float32), -np.ones(len(adj_list[0]), dtype=np.int32))
    """A list of (distance_array, source_array) nodes that stores the shortest paths for each node"""
    
    path_queue: Queue[tuple[float, int, int]] = Queue()
    """A queue of the possible paths to explore sorted by path length"""
    
    for edge_weight, destination in zip(adj_list[0][source], adj_list[1][source]):
        path_queue.put((edge_weight, destination, source))
    
    while path_queue:
        current_distance, current_node, source_node = path_queue.get()
        if output_table[1][current_node] == -1:
            output_table[0][current_node] = current_distance + edge_weight
            output_table[1][current_node] = source_node
            yield current_node
            if current_node == stop_node:
                return output_table
            else:
                for edge_weight, destination in zip(adj_list[0][current_node], adj_list[1][current_node]):
                    if edge_weight < 100:
                        path_queue.put((current_distance + edge_weight, destination, current_node))

    return output_table
