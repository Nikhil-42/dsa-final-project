from queue import PriorityQueue, Queue
from collections import deque
import heapq
from sys import maxsize
import numpy as np


def bfs(adj_list: tuple[np.ndarray, np.ndarray], source: int, stop_node: int=-1):    
    """Takes an adjacency list of (weight_array, destination_array) edge entries where the index is the from node
    as well as a source and returns a table embedding the shortest path for each destination node"""
    
    output_table = (-np.ones(len(adj_list[0]), dtype=np.float32), -np.ones(len(adj_list[0]), dtype=np.int32))
    """A list of (distance_array, source_array) nodes that stores the shortest paths for each node"""
    
    path_queue: Queue[tuple[float, int]] = PriorityQueue()
    """A queue of the possible paths to explore sorted by path length"""
    
    output_table[0][source] = 0.0
    
    path_queue.put((0.0, source))
    
    while path_queue:
        current_distance, current_node = path_queue.get()
        # print("BFS", current_node)
        yield current_node
        for edge_weight, destination in zip(adj_list[0][current_node], adj_list[1][current_node]):
            dest_distance = current_distance + edge_weight
            if edge_weight < np.inf and (output_table[1][destination] == -1 or output_table[0][destination] > dest_distance):
                output_table[0][destination] = dest_distance
                output_table[1][destination] = current_node
                if destination == stop_node:
                    yield destination
                    return output_table
                else:
                    path_queue.put((dest_distance, destination))

    return output_table

def a_star(adj_list: tuple[np.ndarray, np.ndarray], source: int, stop_node=None, heuristic=lambda node: 0):
    """Takes an adjacency list of (weight_array, destination_array) edge entries where the index is the from node
    as well as a source and returns a table embedding the shortest path for each destination node"""
    
    output_table = (-np.ones(len(adj_list[0]), dtype=np.float32), -np.ones(len(adj_list[0]), dtype=np.int32))
    """A list of (distance_array, source_array) nodes that stores the shortest paths for each node"""
    
    path_queue: PriorityQueue[tuple[float, float, int]] = PriorityQueue()
    """A queue of the possible paths to explore sorted by path length"""
    
    output_table[0][source] = 0.0
    
    path_queue.put((heuristic(source), 0.0, source))
    
    while path_queue:
        _, current_distance, current_node = path_queue.get()
        # print("A*", current_node)
        yield current_node
        for edge_weight, destination in zip(adj_list[0][current_node], adj_list[1][current_node]):
            dest_distance = current_distance + edge_weight
            if output_table[1][destination] == -1 or output_table[0][destination] > dest_distance:
                output_table[0][destination] = dest_distance
                output_table[1][destination] = current_node
                if destination == stop_node:
                    yield destination
                    return output_table
                else:
                    path_queue.put((heuristic(destination) + dest_distance, dest_distance, destination))

    return output_table

def dijkstra(adj_list: tuple[np.ndarray, np.ndarray], source: int, stop_node: int=-1):
    return a_star(adj_list, source, stop_node, lambda node: 0.0)

def dfs(adj_list: tuple[np.ndarray, np.ndarray], source: int, stop_node: int=-1):
    """Takes an adjacency list of (weight_array, destination_array) edge entries where the index is the from node
    as well as a source and returns a table embedding the shortest path for each destination node"""

    output_table = (-np.ones(len(adj_list[0]), dtype=np.float32), -np.ones(len(adj_list[0]), dtype=np.int32))
    """A list of (distance_array, source_array) nodes that stores the shortest paths for each node"""

    path_stack = []
    """A stack of the possible paths to explore"""

    visited = set()
    """A set to keep track of visited nodes"""

    output_table[0][source] = 0.0

    path_stack.append((0.0, source))

    while path_stack:
        current_distance, current_node = path_stack.pop()
        
        if current_node not in visited:
            visited.add(current_node)
            yield current_node
            
            if current_node == stop_node:
                return output_table

            for edge_weight, destination in zip(adj_list[0][current_node], adj_list[1][current_node]):
                dest_distance = current_distance + edge_weight
                if edge_weight < np.inf and (output_table[1][destination] == -1 or output_table[0][destination] > dest_distance):
                    output_table[0][destination] = dest_distance
                    output_table[1][destination] = current_node
                    path_stack.append((dest_distance, destination))

    return output_table

# Bellman Ford pseudocode from https://www.geeksforgeeks.org/bellman-ford-algorithm-dp-23/
def bellman_ford(adj_list: tuple[np.ndarray, np.ndarray], source: int, stop_node=None):
 # initialize the output table
    output_table = (np.full(len(adj_list[0]), np.inf, dtype=np.float32), -np.ones(len(adj_list[0]), dtype=np.int32))

    # set the distance from the source node to itself to 0
    output_table[0][source] = 0
    output_table[1][source] = source

    # iterate over all edges and relax them
    for node_index in range(len(adj_list[0]) - 1):
        for i, (weight_array, destination_array) in enumerate(adj_list):
            for edge_weight, destination in zip(weight_array, destination_array):
                if output_table[0][destination] > output_table[0][node_index] + edge_weight:
                    output_table[0][destination] = output_table[0][node_index] + edge_weight
                    output_table[1][destination] = node_index
                    yield destination
                    if destination == stop_node:
                        return output_table

    # check for negative weight cycles
    for node_index, (weight_array, destination_array) in enumerate(adj_list):
        for edge_weight, destination in zip(weight_array, destination_array):
            if output_table[0][destination] > output_table[0][node_index] + edge_weight:
                raise ValueError("Graph contains negative weight cycle")

    return output_table