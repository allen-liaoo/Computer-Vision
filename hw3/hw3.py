import os
import cv2
import numpy as np

from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from scipy.linalg import null_space

from sklearn.neighbors import NearestNeighbors
from util import *

### F ---------------------------------------------------

def compute_F(correspondence, max_iter=5000, eps=5e-1):
    """
    Compute the fundamental matrix using RANSAC for robust estimation.

    Args:
    - correspondence (tuple): Two np.ndarrays (pts1, pts2), each of shape (N, 2).
    - max_iter (int): Maximum number of RANSAC iterations.
    - eps (float): Threshold for determining inliers.
    
    Returns:
    - best_F (np.ndarray): Estimated fundamental matrix of shape (3, 3).
    """
    bestF = None
    maxNumInliers = 0
    pts1, pts2 = correspondence
    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    indexes = pts1.shape[0]

    for _ in range(max_iter):
        F = None
        choices = np.random.choice(indexes, 8, replace=True)

        # Solve for Ax=0, where x is the F's terms in a vector form
        # And each row of A is a combination of coords from a pair of correpondance
        A = []
        for i in choices:
            ux, uy, *_ = pts1[i]
            vx, vy, *_ = pts2[i]
            A.append([ux*vx, uy*vx, vx, ux*vy, uy*vy, vy, ux, uy, 1])
        A = np.array(A)
        [U,S,V] = np.linalg.svd(A)
        xf = V.T[:, -1]
        F = xf.reshape((3,3)) # F is the current computed matrix
        [U,S,V] = np.linalg.svd(F) # SVD cleanup: since F is rank 2... ?
        S[2] = 0
        F = U @ np.diag(S) @ V

        # calculate error
        # Using the fact that v.T (Fu) = 0; BAD because the value cab be influenced by scaling F
        # numInliers = 0
        # for j in range(pts1_h.shape[0]):
        #     expect_0 = pts2_h[j] @ F @ pts1_h[j].T
        #     if abs(expect_0) < eps:
        #         numInliers += 1

        # Calculate error by using epipolar lines (Iu, Iv)
        # And finding the distance between the epipolar line and the actual point
        # normalizing by the line equation
        Iu = (F @ pts1_h.T).T
        Iv = (F.T @ pts2_h.T).T
        du = np.abs(np.sum(pts2_h * Iu, 1)) / np.linalg.norm(Iu[:, :2], 2, 1) 
        dv = np.abs(np.sum(pts1_h * Iv, 1)) / np.linalg.norm(Iv[:, :2], 2, 1)
        numInliers = np.sum((du < eps) & (dv < eps))

        if numInliers > maxNumInliers:
            bestF = F
            maxNumInliers = numInliers
    return bestF

## Triangulation --------------------------------------

def triangulation(P1, P2, correspondence):
    """
    Triangulate the 3D positions of each correspondence between two images.

    Args:
    - P1 (np.ndarray): Projection matrix of the first camera (3x4).
    - P2 (np.ndarray): Projection matrix of the second camera (3x4).
    - correspondence (tuple): Two np.ndarrays (pts1, pts2), each of shape (N, 2), containing corresponding points between the two views.
    
    Returns:
    - pts3D (np.ndarray): Array of shape (N, 3) representing the 3D coordinates of the points.
    """
    def skewMat(pt): 
        return np.array([[0, -pt[2], pt[1]],
                         [pt[2], 0, -pt[0]],
                         [-pt[1], pt[0], 0]])

    pts1, pts2 = correspondence
    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    # linear triangulation method
    # A [pts3d; 1] = 0, we stack A up
    all_Xs = None
    for i in range(pts1_h.shape[0]):
        stack1 = skewMat(pts1_h[i]) @ P1
        stack2 = skewMat(pts2_h[i]) @ P2
        A = np.vstack((stack1[:2], stack2[:2]))
        [U,S,V] = np.linalg.svd(A)
        X = V.T[:, -1]
        X = (X / X[-1])[:-1] # normalize X, discard last element (make it not homogenous)

        if i == 0:
            all_Xs = X
        else:
            all_Xs = np.vstack((all_Xs, X))

    return all_Xs

def disambiguate_pose(Rs, Cs, pts3Ds, K=None, screen_size=None):
    """
    Find the best relative camera pose based on the most valid 3D points visible on the screen.

    Args:
    - Rs (list): List of np.ndarrays, each of shape (3, 3), representing possible rotation matrices.
    - Cs (list): List of np.ndarrays, each of shape (3, 1), representing camera centers.
    - pts3Ds (list): List of np.ndarrays, each of shape (N, 3), representing possible 3D points corresponding to different poses.
    - K (np.ndarray): Intrinsic camera matrix of shape (3, 3). <- It is optional to use this as an input. ie its possible without it but easier with it.
    - screen_size (tuple): Screen dimensions as (height, width). <- It is optional to use this as an input. ie its possible without it but easier with it.
    
    Returns:
    - R (np.ndarray): The best rotation matrix of shape (3, 3).
    - C (np.ndarray): The best camera center of shape (3, 1).
    - pts3D (np.ndarray): The best set of 3D points of shape (N, 3).
    """
    bestSetInd = None
    maxNumFittingPoints = 0
    for i in range(len(Rs)):
        R, C, PT = Rs[i], Cs[i], pts3Ds[i]
        r3 = R[2, :]

        # Calculate dot product of r3 and (X-C), get count of then > 0
        # vectorized
        errs = r3 @ (PT - C.T).T
        numFittingPoints = np.sum(errs > 0)

        # non-vectorized
        # numFittingPoints = 0
        # for x in PT:
        #     if np.sum(r3 * (x[:3] - C.T)) > 0:
        #         numFittingPoints += 1
        if numFittingPoints > maxNumFittingPoints:
            bestSetInd = i
            maxNumFittingPoints = numFittingPoints
    return Rs[bestSetInd], Cs[bestSetInd], pts3Ds[bestSetInd]

# Disparity MAP --------------------------------------

def compute_rectification(K, R, C):
    """
    Compute the rectification homographies for both left and right images.

    Args:
    - K (np.ndarray): Intrinsic camera matrix (3x3).
    - R (np.ndarray): Rotation matrix (3x3) of the second camera.
    - C (np.ndarray): Camera center of the second camera (3x1).
    
    Returns:
    - H1 (np.ndarray): Homography for the left image (3x3).
    - H2 (np.ndarray): Homography for the right image (3x3).
    """
    rx = C / np.linalg.norm(C)
    rx = rx.flatten()
    rzhat = np.array([0,0,1])
    rz = (rzhat - (rzhat.T @ rx) * rx)
    rz = rz / np.linalg.norm(rz)
    rz = rz.flatten()
    ry = np.cross(rz, rx)
    ry /= np.linalg.norm(ry)
    ry = ry.flatten()
    R_rect = np.array([rx,ry,rz])

    H1 = K @ R_rect @ np.linalg.inv(K)
    H2 = K @ (R_rect @ R.T) @ np.linalg.inv(K)
    return (H1, H2)

def dense_match(img1, img2, descriptors1, descriptors2):
    """
    Estimate disparity by finding dense correspondences between two images.
    
    Args:
    - img1 (np.ndarray): First image, typically a grayscale image, of shape (H, W).
    - img2 (np.ndarray): Second image, typically a grayscale image, of shape (H, W).
    - descriptors1 (np.ndarray): Feature descriptors for the first image, shape (H, W, D), where D is the descriptor dimension.
    - descriptors2 (np.ndarray): Feature descriptors for the second image, shape (H, W, D).
    
    Returns:
    - disparity_map (np.ndarray): Disparity map of shape (H, W), representing the pixel-wise disparity between img1 and img2.
    """
    h, w = img1.shape
    disparity_map = np.zeros((h,w))

    for y in range(h):
        for x in range(w):
            if (img1[y,x] == 0).all():
                continue
            diffs = descriptors1[y,x] - descriptors2[y,:x+1]
            dnorm = np.linalg.norm(diffs, axis=-1)
            disparity_map[y,x] = x - np.argmin(dnorm)
    return disparity_map

## MAIN

submission = False

if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread(os.path.join('assets', 'left.bmp'), 1)
    img_right = cv2.imread(os.path.join('assets', 'right.bmp'), 1)
    if (submission): visualize_img_pair(img_left, img_right)

    H, W, _ = img_right.shape

    # Step 0: get correspondences between image pair in assets/correspondence.npz
    correspondence = np.load('assets/correspondence.npz') #(np.zeros((10, 2)), np.zeros((10, 2))) # TODO load from assets/correspondence.npz
    correspondence = (correspondence['pts1'], correspondence['pts2'])
    if (submission): visualize_correspondence(img_right, img_right, correspondence)

    # Step 1: compute fundamental matrix and recover four sets of camera poses
    F = compute_F(correspondence)
    if (submission): visualize_epipolar_lines(img_left, img_right, correspondence, F)

    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    if (submission): visualize_camera_poses(Rs, Cs)

    # Step 2: triangulation
    pts3Ds = []
    # Todo compute P1: hint its the identity projection matrix
    I = np.eye(3,3)
    P1 = K @ np.hstack((I, np.zeros((3,1))))
    for i in range(len(Rs)):
        # Todo compute P2: hint its relative to P1 using R and C
        P2 = K @ Rs[i] @ np.hstack((I, -Cs[i]))
        pts3D = triangulation(P1, P2, correspondence)
        pts3Ds.append(pts3D)

    if (submission): visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 3: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds, K, (H, W))
    if (submission): visualize_camera_pose_with_pts(R, C, pts3D)

    # Step 4: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0])) # Todo compute warped img left: Hint warp img_left using H1 
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0])) # Todo compute warped img right: Hint warp img_left using H1 
    if (submission): visualize_img_pair(img_left_w, img_right_w)

    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)

    # Todo Load data from assets/dsift_descriptor.npz: hint the keys are descriptors1, descriptors2
    desps = np.load('assets/dsift_descriptor.npz')
    desp1, desp2 = desps['descriptors1'], desps['descriptors2']

    disparity = dense_match(img_left_w, img_right_w, desp1, desp2)
    visualize_disparity_map(disparity)