import cv2
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.linalg import null_space

from sklearn.neighbors import NearestNeighbors

import hashlib
import os
import time

## UTIL ----------------------------------------------

def make_image_pair(imgs):
    for img in imgs:
        assert imgs[0].shape[0] == img.shape[0]
        assert imgs[0].ndim == img.ndim
    
    img = np.hstack(imgs)
    
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

def visualize_img_pair(img1, img2, filename=None):
    plt.figure(figsize=(20, 10))
    plt.imshow(make_image_pair((img1, img2)), cmap='gray')
    plt.axis('off')

    if filename:   
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
        # plt.draw()

def visualize_correspondence(img1, img2, correspondence, filename=None):
    plt.figure(figsize=(20, 10))
    plt.imshow(make_image_pair((img1, img2)), cmap='gray')
    
    x1, x2 = correspondence
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'

    cmap = plt.get_cmap('tab10') 
    colors = [cmap(i % 10) for i in range(x1.shape[0])]

    for i in range(x1.shape[0]):
        # ensure these are the same color
        plt.plot([x1[i, 0], x2[i, 0] + img1.shape[1]], [x1[i, 1], x2[i, 1]], color=colors[i])
        plt.plot([x1[i, 0], x2[i, 0] + img1.shape[1]], [x1[i, 1], x2[i, 1]], 'o', color=colors[i])
        
    plt.axis('off')

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
        # plt.draw()

def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = (F @ np.array([[p[0], p[1], 1]]).T).flatten()
    p1, p2 = (0, int(-el[2] / el[1])), (img.shape[1], int((-img_width * el[0] - el[2]) / el[1]))
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2

def visualize_epipolar_lines(img1, img2, correspondence, F, filename=None):
    plt.figure(figsize=(20, 10))
    plt.imshow(make_image_pair((img1, img2)), cmap='gray')
    
    pts1, pts2 = correspondence
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'

    cmap = plt.get_cmap('tab10')  # You can choose another colormap if you like
    colors = [cmap(i % 10) for i in range(pts1.shape[0])]

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        plt.scatter(x1, y1, s=5, color=colors[i])

        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        plt.plot([p1[0] + img1.shape[1], p2[0] + img1.shape[1]], [p1[1], p2[1]], linewidth=0.5, color=colors[i])
    
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        plt.scatter(x2 + img1.shape[1], y2, s=5, color=colors[i])

        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5, color=colors[i])

    plt.axis('off')

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)
    C = C.flatten()  # (3, 1) -> (3,)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')

def visualize_camera_poses(Rs, Cs, filename=None):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure(figsize=(20, 10))
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
        ax.title.set_text('Configuration {}'.format(i))
    
    fig.tight_layout()
    
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds, filename=None):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure(figsize=(20, 20))
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
        ax.title.set_text('Configuration {}'.format(i))
    fig.tight_layout()

    if filename:
        
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

def visualize_camera_pose_with_pts(R, C, pts3D, filename=None):
    fig = plt.figure(figsize=(20, 20))

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    R1, C1 = np.eye(3), np.zeros((3, 1))
    draw_camera(ax, R1, C1, 5)
    draw_camera(ax, R, C, 5)

    ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')

    set_axes_equal(ax)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')

    ax.view_init(azim=-90, elev=0)
    ax.title.set_text('Camera Pose with 3D Points')

    fig.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

def visualize_disparity_map(disparity, filename=None):
    disparity[disparity > 150] = 150
    plt.figure(figsize=(20, 10))
    plt.imshow(disparity, cmap='jet')

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
