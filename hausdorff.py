# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:48:22 2026

@author: kerem
"""

import numpy as np
import sympy as sp
import re
from skimage import measure
from scipy.spatial.distance import cdist

def get_shape_points(equation_str, resolution=200, grid_bounds=(-6, 6)):
    """
    Parses equations, inequalities, and domain restrictions to extract 2D shape points.
    
    This function evaluates a given mathematical string over a 2D grid using boolean masking.
    It supports inequalities (e.g., solid disks) and logical AND operations (e.g., line segments).
    For exact equations (e.g., x = y), it applies a dynamic tolerance to capture the line 
    on the discrete grid.
    
    Parameters
    ----------
    equation_str : str
        The mathematical condition (e.g., "x**2 + y**2 <= 4" or "y = 0 & x >= -5 & x <= 5").
    resolution : int, optional
        The grid density for evaluating the condition (default is 200).
    grid_bounds : tuple of float, optional
        The (min, max) coordinate limits for the evaluation grid (default is (-6, 6)).
        
    Returns
    -------
    numpy.ndarray
        An (N, 2) array containing the (X, Y) coordinates of the points satisfying the condition.
    """
    vals = np.linspace(grid_bounds[0], grid_bounds[1], resolution)
    X, Y = np.meshgrid(vals, vals)
    
    tol = (grid_bounds[1] - grid_bounds[0]) / resolution * 1.5
    
    eq_str = equation_str.replace(' and ', ' & ').replace(' or ', ' | ')
    tokens = re.split(r'(&|\|)', eq_str)
    processed_tokens = []
    
    for token in tokens:
        if token in ('&', '|'):
            processed_tokens.append(token)
            continue
            
        t = token.strip()
        if not t: continue
            
        if any(op in t for op in ['<=', '>=', '<', '>']):
            processed_tokens.append(f"({t})")
        elif '==' in t:
            lhs, rhs = t.split('==')
            processed_tokens.append(f"(np.abs(({lhs}) - ({rhs})) < {tol})")
        elif '=' in t:
            lhs, rhs = t.split('=')
            processed_tokens.append(f"(np.abs(({lhs}) - ({rhs})) < {tol})")
        else:
            processed_tokens.append(f"(np.abs({t}) < {tol})")
            
    final_expression = " ".join(processed_tokens)
    
    env = {
        "x": X, "y": Y, 
        "np": np, "abs": np.abs, "tol": tol,
        "sin": np.sin, "cos": np.cos, "sqrt": np.sqrt
    }
    
    try:
        mask = eval(final_expression, {"__builtins__": {}}, env)
        points = np.column_stack((X[mask], Y[mask]))
        return points
    except Exception as e:
        return np.array([])

def get_hausdorff_details(A, B, metric_type="euclidean"):
    """
    Calculates the Hausdorff distance between two sets of points and identifies the critical points.
    
    This function computes the directed Hausdorff distance from set A to B, and from B to A. 
    It determines the bidirectional Hausdorff metric by taking the maximum of these two directed 
    distances. Crucially for visualization, it extracts the exact pair of points that form 
    this maximum distance.
    
    Parameters
    ----------
    A : numpy.ndarray
        An (N, 2) array of coordinates representing the first compact set.
    B : numpy.ndarray
        An (M, 2) array of coordinates representing the second compact set.
    metric_type: str, optional
        Optional metric type. (default: euclidean)
        
    Returns
    -------
    tuple
        A tuple containing 4 elements:
        - h (float): The final bidirectional Hausdorff distance.
        - pt_source (numpy.ndarray): The 1D array (x, y) of the source point causing the max distance.
        - pt_target (numpy.ndarray): The 1D array (x, y) of the nearest point in the other set.
        - direction (str): Indicates which directed distance was larger ("A -> B" or "B -> A").
        
    External Dependencies & Documentation
    -------------------------------------
    - scipy.spatial.distance.cdist: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    - numpy.min: https://numpy.org/doc/stable/reference/generated/numpy.min.html
    - numpy.argmin: https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
    - numpy.argmax: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
    """
    dists_A_to_B = np.min(cdist(A, B, metric=metric_type), axis=1)
    idx_max_A = np.argmax(dists_A_to_B)
    d_AB = dists_A_to_B[idx_max_A]
    pt_A = A[idx_max_A]
    pt_B_near_A = B[np.argmin(cdist([pt_A], B, metric=metric_type))]
    
    dists_B_to_A = np.min(cdist(B, A, metric=metric_type), axis=1)
    idx_max_B = np.argmax(dists_B_to_A)
    d_BA = dists_B_to_A[idx_max_B]
    pt_B = B[idx_max_B]
    pt_A_near_B = A[np.argmin(cdist([pt_B], A, metric=metric_type))]
    
    h = max(d_AB, d_BA)
    
    return h, d_AB, pt_A, pt_B_near_A, d_BA, pt_B, pt_A_near_B
    
