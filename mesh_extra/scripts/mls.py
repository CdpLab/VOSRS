import open3d as o3d
import numpy as np
from scipy.spatial import KDTree

kdtree = KDTree(points)

def get_neighbors(point, radius):
    indices = kdtree.query_ball_point(point, radius)
    return points[indices]

def fit_plane(neighbors):
    centroid = np.mean(neighbors, axis=0)
    cov_matrix = np.cov(neighbors - centroid, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal = eigenvectors[:, 0]
    return centroid, normal

def project_to_plane(point, centroid, normal):
    vec = point - centroid
    distance = np.dot(vec, normal)
    projection = point - distance * normal
    return projection

def mls_smoothing(points, radius):
    smoothed_points = []
    for point in points:
        neighbors = get_neighbors(point, radius)
        if len(neighbors) < 3:
            smoothed_points.append(point)
            continue
        centroid, normal = fit_plane(neighbors)
        projection = project_to_plane(point, centroid, normal)
        smoothed_points.append(projection)
    return np.array(smoothed_points)
