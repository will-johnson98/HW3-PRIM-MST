import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')


    def construct_mst(self):
        """
        Implements Prim's algorithm to construct a minimum spanning tree.
        
        The algorithm works by:
        1. Starting from an arbitrary vertex
        2. Finding the minimum weight edge that connects the current tree to an unvisited vertex
        3. Adding that edge to the MST
        4. Repeating until all vertices are visited
        
        Uses a min-heap to efficiently find the minimum weight edge at each step.
        """
        n = len(self.adj_mat)
        visited = [False] * n
        self.mst = np.zeros((n, n))
        
        # Start from vertex 0
        visited[0] = True
        edges = []
        
        # Add all edges from starting vertex to heap
        for j in range(n):
            if self.adj_mat[0,j] > 0:  # Only add existing edges
                heapq.heappush(edges, (self.adj_mat[0,j], 0, j))
        
        while edges:
            weight, u, v = heapq.heappop(edges)
            
            if visited[v]:
                continue
                
            # Add edge to MST
            visited[v] = True
            self.mst[u,v] = weight
            self.mst[v,u] = weight  # Maintain symmetry for undirected graph
            
            # Add new edges from vertex v
            for w in range(n):
                if not visited[w] and self.adj_mat[v,w] > 0:
                    heapq.heappush(edges, (self.adj_mat[v,w], v, w))
