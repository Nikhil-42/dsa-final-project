from queue import PriorityQueue, Queue
from collections import deque
import heapq
from sys import maxsize
import numpy as np

def dijkstra(adj_list: tuple[np.ndarray, np.ndarray], source: int, stop_node: int=-1):
    """Takes an adjacency list of (weight_array, destination_array) edge entries where the index is the from node
    as well as a source and returns a table embedding the shortest path for each destination node"""
    
    output_table = (-np.ones(len(adj_list[0]), dtype=np.float32), -np.ones(len(adj_list[0]), dtype=np.int32))
    """A list of (distance_array, source_array) nodes that stores the shortest paths for each node"""
    
    path_queue: PriorityQueue[tuple[float, int, int]] = PriorityQueue()
    """A queue of the possible paths to explore sorted by path length"""
    
    output_table[0][source] = 0
    output_table[1][source] = source
    
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

def bfs(adj_list, start_node, target_node):
    visited = set()
    queue = Queue()
    start_node = (start_node, [])
    queue.put(start_node)

    while queue:
        current_node, path = queue.get()
        yield current_node
        if current_node == target_node:
            return current_node
        if current_node not in visited:
            visited.add(current_node)
            for neighbor, weight in zip(adj_list[1][current_node], adj_list[0][current_node]):
                if neighbor not in visited and weight < np.inf:
                    queue.put((neighbor, path + [current_node]))
    return None


# Bellman Ford pseudocode from https://www.geeksforgeeks.org/bellman-ford-algorithm-dp-23/
def bellman_ford(adj_list: tuple[np.ndarray, np.ndarray], source: int, stop_node=None):
    """Takes an adjacency list of (weight_array, destination_array) edge entries where the index is the from node
    as well as a source and returns a table embedding the shortest path for each destination node"""
    
    output_table = (np.full(len(adj_list[0]), np.inf, dtype=np.float32), -np.ones(len(adj_list[0]), dtype=np.int32))
    """A list of (distance_array, source_array) nodes that stores the shortest paths for each node"""
    
    output_table[0][source] = 0
    output_table[1][source] = source
    
    for _ in range(len(adj_list[0]) - 1):
        for node_index, (weight_array, destination_array) in enumerate(adj_list):
            for edge_weight, destination in zip(weight_array, destination_array):
                if output_table[0][destination] > output_table[0][node_index] + edge_weight:
                    output_table[0][destination] = output_table[0][node_index] + edge_weight
                    output_table[1][destination] = node_index
                    yield destination
                    if destination == stop_node:
                        return output_table
    
    return output_table


def a_star(adj_list: tuple[np.ndarray, np.ndarray], source: int, width: int,heuristic, stop_node=None):
    visited = set()
    queue = PriorityQueue()
    start_node = (heuristic(source, stop_node, width), source, [])
    queue.put(start_node)

    while not queue.empty():
        distance, current_node, path = queue.get()
        yield current_node
        if current_node == stop_node:
            optimal_path = [current_node]
            for parent in reversed(path):
                optimal_path.append(parent)
            return optimal_path
        if current_node not in visited:
            visited.add(current_node)
            for neighbor, weight in zip(adj_list[1][current_node], adj_list[0][current_node]):
                if neighbor not in visited and weight < np.inf:
                    queue.put((heuristic(neighbor, stop_node, width), neighbor, path + [current_node]))
        return None
    
