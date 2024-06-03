## This script is adopted from https://github.com/ika-rwth-aachen/MultiCorrupt
## Thanks to the authors of multicorrupt
import os
import copy
import math
from typing import Dict, List, Tuple
from copy import deepcopy
import numpy as np
import pickle5 as pickle
from pathlib import Path
import yaml
import itertools
from scipy.stats import linregress
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PI = np.pi

seed = 1000
np.random.seed(seed)
RNG = np.random.default_rng(seed)

""" spatial misalignment """
def transform_points(points, severity):
    """
    Rotate and translate a set of points.
    
    Parameters:
    points (numpy.ndarray): A 2D array where each row represents a point (x, y, z, ...).
    angle_degree (float): The rotation angle in degrees.
    
    Returns:
    numpy.ndarray: The transformed points.
    """
    s = [(0.2, 1), (0.4, 2), (0.6, 3)][severity - 1]
    
    
    # Convert the angle from degrees to radians
    rand_num = np.random.rand()
    
    # Check if the random number is less than the given probability
    if rand_num < s[0]:
        
        # Convert the angle from degrees to radians
        theta = np.radians(s[1])

        # Define the rotation matrix for rotation around the x-axis
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta), np.sin(theta)],
            [0, -np.sin(theta), np.cos(theta)]
        ])

        # Define the rotation matrix for rotation around the y-axis
        rotation_matrix_y = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)]
        ])

        # Define the rotation matrix for rotation around the z-axis
        rotation_matrix_z = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        # Combine the three rotations
        rotation_matrix = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z

                
        # Extract the x, y, z coordinates of the points
        xyz_points = points[:, :3]
        
        # Apply the rotation matrix to the x, y, z coordinates
        rotated_xyz = np.dot(xyz_points, rotation_matrix.T)
        
        # Define the translation vector (2 units along the x-axis)
        translation_vector = np.array([2, 0, 0])
        
        # Apply the translation to the rotated x, y, z coordinates
        translated_xyz = rotated_xyz + translation_vector
        
        # Concatenate the translated x, y, z coordinates with the other properties of the points
        transformed_points = np.hstack((translated_xyz, points[:, 3:]))
        
        return transformed_points
    
    else:
        # If the random number is not less than the given probability, return the original points
        return points