import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


# def check_mst(adj_mat: np.ndarray, 
#               mst: np.ndarray, 
#               expected_weight: int, 
#               allowed_error: float = 0.0001):
#     """
    
#     Helper function to check the correctness of the adjacency matrix encoding an MST.
#     Note that because the MST of a graph is not guaranteed to be unique, we cannot 
#     simply check for equality against a known MST of a graph. 

#     Arguments:
#         adj_mat: adjacency matrix of full graph
#         mst: adjacency matrix of proposed minimum spanning tree
#         expected_weight: weight of the minimum spanning tree of the full graph
#         allowed_error: allowed difference between proposed MST weight and `expected_weight`

#     TODO: Add additional assertions to ensure the correctness of your MST implementation. For
#     example, how many edges should a minimum spanning tree have? Are minimum spanning trees
#     always connected? What else can you think of?

#     """

#     def approx_equal(a, b):
#         return abs(a - b) < allowed_error

#     total = 0
#     for i in range(mst.shape[0]):
#         for j in range(i+1):
#             total += mst[i, j]
#     assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    Validates the correctness of a minimum spanning tree.
    
    Key properties checked:
    1. Total weight matches expected weight
    2. MST has exactly n-1 edges where n is number of vertices
    3. MST is symmetric (undirected)
    4. MST is connected
    5. All edge weights in MST exist in original graph
    """
    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    n = len(adj_mat)
    
    # Check symmetry
    assert np.allclose(mst, mst.T), 'MST must be symmetric'
    
    # Count edges and verify n-1 property
    edge_count = np.sum(mst > 0) / 2  # Divide by 2 since matrix is symmetric
    assert edge_count == n - 1, f'MST must have exactly {n-1} edges, found {edge_count}'
    
    # Verify edge weights exist in original graph
    for i in range(n):
        for j in range(i):
            if mst[i,j] > 0:
                assert approx_equal(mst[i,j], adj_mat[i,j]), \
                    f'Edge weight {mst[i,j]} not found in original graph at position ({i},{j})'
    
    # Check connectivity using BFS
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
    
    # Check total weight
    total = np.sum(mst) / 2  # Divide by 2 since matrix is symmetric
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
    Tests MST construction on a simple cycle graph.
    The graph is a cycle with 4 vertices:
    
    A --1-- B
    |       |
    4       2
    |       |
    D --3-- C
    
    The MST should exclude the highest weight edge (4).
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
