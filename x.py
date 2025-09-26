import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load a PLY file
def load_ply_file(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points

# Project 3D points onto a 2D plane
def project_to_2d(points, plane='xy'):
    if plane == 'xy':
        return points[:, [0, 1]]  # Keep x, y coordinates
    elif plane == 'xz':
        return points[:, [0, 2]]  # Keep x, z coordinates
    elif plane == 'yz':
        return points[:, [1, 2]]  # Keep y, z coordinates
    else:
        raise ValueError("Plane must be 'xy', 'xz', or 'yz'")

# Visualize 3D and 2D points
def visualize_points(points_3d, points_2d, plane='xy'):
    fig = plt.figure(figsize=(12, 5))
    
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='.', s=1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Point Cloud')
    
    # 2D plot
    ax2 = fig.add_subplot(122)
    ax2.scatter(points_2d[:, 0], points_2d[:, 1], c='r', marker='.', s=1)
    ax2.set_xlabel(plane[0].upper())
    ax2.set_ylabel(plane[1].upper())
    ax2.set_title(f'2D Projection on {plane.upper()} Plane')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Specify the path to your PLY file
    ply_file_path = "logs/dsail_mary_desk.ply"  # Update with your desired PLY file
    
    # Load points from PLY file
    points_3d = load_ply_file(ply_file_path)
    
    # Project onto XY plane
    points_2d = project_to_2d(points_3d, plane='xy')
    
    # Visualize
    visualize_points(points_3d, points_2d, plane='xy')