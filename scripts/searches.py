from queue import PriorityQueue, Queue
from sys import maxsize

def dijkstra(adjList: list[list[tuple[float, int]]], source: int, stop_node=None):
    """Takes an adjacency list of (weight, destination) edge entries as well as a source
        and returns a table embedding the shortest path for each destination node"""
    
    output_table: list[tuple[float, int]] = [(-1.0, None)] * len(adjList)
    """A list of (distance, source) nodes that stores the shortest paths for each node"""
    
    path_queue: PriorityQueue[tuple[float, int]] = PriorityQueue()
    """A queue of the possible paths to explore sorted by path length"""
    
    for edge in adjList[source]:
        path_queue.put(edge)
    
    while not path_queue.empty():
        current_distance, current_node = path_queue.pop()
        for edge_weight, dest in adjList[source]:
            if output_table[dest] == (-1.0, None):
                output_table[dest] = (current_distance + edge_weight, current_node)
                path_queue.put((current_distance + edge_weight, dest))
                if dest == stop_node:
                    return output_table
    
    return output_table

def bfs(adjList: list[list[(float, int)]], source: int, stop_node=None):
    output_table: list[tuple[float, int]] = [(-1.0, None)] * len(adjList)
    """A list of (distance, source) nodes that stores the shortest paths for each node"""
    
    path_queue: Queue[tuple[float, int]] = Queue()
    """A queue of the possible paths to explore sorted by path length"""
    
    for edge in adjList[source]:
        path_queue.put(edge)
    
    while not path_queue.empty():
        current_distance, current_node = path_queue.pop()
        for edge_weight, dest in adjList[source]:
            if output_table[dest] == (-1.0, None):
                output_table[dest] = (current_distance + edge_weight, current_node)
                path_queue.put((current_distance + edge_weight, dest))
                if dest == stop_node:
                    return output_table
    
    return output_table
    