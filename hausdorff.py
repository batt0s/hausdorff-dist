# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:48:22 2026

@author: kerem
"""

import numpy as np
import sympy as sp
from skimage import measure
from scipy.spatial.distance import cdist



def get_boundary_points(equation_str, resolution=200, grid_bounds=(-6, 6)):
    """
    Parses a mathematical equation string and extracts the 2D boundary points of the shape.
    
    This function converts a string equation (implicit or explicit) into a level-set surface 
    (Z = f(x, y)) over a defined grid. It then uses the marching squares algorithm to find 
    the zero-contour (where Z = 0), representing the actual boundary of the geometric shape.
    
    Parameters
    ----------
    equation_str : str
        The mathematical equation of the set (e.g., "x**2 + y**2 = 4" or "abs(x) + abs(y) = 1").
    resolution : int, optional
        The grid density for evaluating the equation (default is 200). Higher values yield 
        smoother boundaries but increase computation time.
    grid_bounds : tuple of float, optional
        The (min, max) coordinate limits for both the X and Y axes used to generate 
        the evaluation grid (default is (-6, 6)). It defines the spatial bounding box 
        within which the algorithm searches for the geometric shape's contour.
        
    Returns
    -------
    numpy.ndarray
        An (N, 2) array containing the (X, Y) coordinates of the points lying on the boundary.
        Returns an empty array if no boundary is found.
        
    External Dependencies & Documentation
    -------------------------------------
    - sympy.symbols: https://docs.sympy.org/latest/modules/core.html#sympy.core.symbol.symbols
    - sympy.parse_expr: https://docs.sympy.org/latest/modules/parsing.html#sympy.parsing.sympy_parser.parse_expr
    - sympy.lambdify: https://docs.sympy.org/latest/modules/utilities/lambdify.html
    - numpy.linspace: https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
    - numpy.meshgrid: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    - skimage.measure.find_contours: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.find_contours
    - numpy.zeros_like: https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html
    - numpy.vstack: https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
    """
    x_sym, y_sym = sp.symbols('x y')
    lhs, rhs = equation_str.split('=')
    expr = sp.parse_expr(lhs) - sp.parse_expr(rhs)
    f = sp.lambdify((x_sym, y_sym), expr, 'numpy')
    
    vals = np.linspace(grid_bounds[0], grid_bounds[1], resolution)
    X, Y = np.meshgrid(vals, vals)
    Z = f(X, Y)
    
    contours = measure.find_contours(Z, 0)
    
    all_points = []
    for contour in contours:
        actual_points = np.zeros_like(contour)
        actual_points[:, 0] = vals[0] + contour[:, 1] * (vals[1] - vals[0]) # X
        actual_points[:, 1] = vals[0] + (resolution - 1 - contour[:, 0]) * (vals[1] - vals[0]) # Y (Y ekseni ters)
        all_points.append(actual_points)
    
    return np.vstack(all_points) if all_points else np.array([])

def get_hausdorff_details(A, B):
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
    dists_A_to_B = np.min(cdist(A, B), axis=1)
    idx_max_A = np.argmax(dists_A_to_B)
    d_AB = dists_A_to_B[idx_max_A]
    pt_A = A[idx_max_A]
    pt_B_near_A = B[np.argmin(cdist([pt_A], B))]
    
    dists_B_to_A = np.min(cdist(B, A), axis=1)
    idx_max_B = np.argmax(dists_B_to_A)
    d_BA = dists_B_to_A[idx_max_B]
    pt_B = B[idx_max_B]
    pt_A_near_B = A[np.argmin(cdist([pt_B], A))]
    
    h = max(d_AB, d_BA)
    
    if d_AB >= d_BA:
        return h, pt_A, pt_B_near_A, "A -> B"
    else:
        return h, pt_B, pt_A_near_B, "B -> A"
    
