import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    Validates the mts/Graph class's ability to initiate a minimum spanning tree.
    
    Specific checks:
    1. Total weight matches expected weight
    2. MST has exactly n-1 edges where n is number of vertices
    3. MST is symmetric (undirected)
    4. MST is connected
    5. All edge weights in MST exist in original graph
    """
    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    n = len(adj_mat)
    
    assert np.allclose(mst, mst.T), 'MST must be symmetric'
    
    edge_count = np.sum(mst > 0) / 2
    assert edge_count == n - 1, f'MST must have exactly {n-1} edges, found {edge_count}'
    
    for i in range(n):
        for j in range(i):
            if mst[i,j] > 0:
                assert approx_equal(mst[i,j], adj_mat[i,j]), \
                    f'Edge weight {mst[i,j]} not found in original graph at position ({i},{j})'
    
    def is_connected(graph):
        visited = [False] * n
        queue = [0]
        visited[0] = True
        
        while queue:
            v = queue.pop(0)
            for w in range(n):
                if graph[v,w] > 0 and not visited[w]:
                    visited[w] = True
                    queue.append(w)
        
        return all(visited)
    
    assert is_connected(mst), 'MST must be connected'
    
    total = np.sum(mst) / 2
    assert approx_equal(total, expected_weight), f'Expected weight {expected_weight}, got {total}'
    

def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    Tests MST construction on a simple cyclic graph.
    The graph is a cycle with 4 vertices:
    
    A --1-- B
    |       |
    4       2
    |       |
    D --3-- C
    
    The MST should exclude the highest weight edge (4).
    I used Claude to come up with the idea of using a cyclic graph, and to ensure the drawn 
    graph above matched the np array below.
    """
    cycle = np.array([
        [0, 1, 0, 4],
        [1, 0, 2, 0],
        [0, 2, 0, 3],
        [4, 0, 3, 0]
    ])
    
    g = Graph(cycle)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 6)  # Total weight should be 1 + 2 + 3 = 6
