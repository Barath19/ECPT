import numpy as np


def generate_perturbation_matrix(rotation=(2,2,2), translation=(2,0,0)):
    rx, ry, rz = np.radians(rotation)
    
    # Rotation matrices around x, y, and z axes
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    R_y = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    R_z = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = R_z @ R_y @ R_x
    
    # Translation vector
    t = np.array(translation).reshape(3, 1)
    
    # Construct the 4x4 transformation matrix
    perturbation_matrix = np.eye(4)
    perturbation_matrix[:3, :3] = R
    perturbation_matrix[:3, 3] = t.flatten()
    
    return perturbation_matrix