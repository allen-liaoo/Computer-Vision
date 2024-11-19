
from hw3 import *
from util import *

import os
import glob
import shutil
import zipfile

assets_dir = 'assets'

def run_hw3():
    # Construct file paths for images and assets
    img_left_path = os.path.join(assets_dir, 'left.bmp')
    img_right_path = os.path.join(assets_dir, 'right.bmp')
    ans0_path = os.path.join(assets_dir, 'ans0.png')
    ans1_path = os.path.join(assets_dir, 'ans1.png')
    ans2_path = os.path.join(assets_dir, 'ans2.png')
    ans3_path = os.path.join(assets_dir, 'ans3.png')
    ans4_path = os.path.join(assets_dir, 'ans4.png')
    ans5_path = os.path.join(assets_dir, 'ans5.png')
    ans6_path = os.path.join(assets_dir, 'ans6.png')
    ans7_path = os.path.join(assets_dir, 'ans7.png')
    correspondence_path = os.path.join(assets_dir, 'correspondence.npz')
    dsift_descriptor_path = os.path.join(assets_dir, 'dsift_descriptor.npz')

    # Read in left and right images as RGB images
    img_left = cv2.imread(img_left_path, 1)
    img_right = cv2.imread(img_right_path, 1)
    visualize_img_pair(img_left, img_right, ans0_path)

    H, W, _ = img_right.shape

    # Step 0: get correspondences between image pair
    data = np.load(correspondence_path)
    correspondence = (data['pts1'], data['pts2'])
    visualize_correspondence(img_right, img_right, correspondence, ans1_path)

    # Step 1: compute fundamental matrix and recover four sets of camera poses
    F = compute_F(correspondence)
    visualize_epipolar_lines(img_left, img_right, correspondence, F, ans2_path)

    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs, ans3_path)

    # Step 2: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, correspondence)
        pts3Ds.append(pts3D)

    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds, ans4_path)

    # Step 3: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds, K, (H, W))
    visualize_camera_pose_with_pts(R, C, pts3D, ans5_path)

    # Step 4: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w, ans6_path)

    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    data = np.load(dsift_descriptor_path, allow_pickle=True)
    desp1, desp2 = data['descriptors1'], data['descriptors2']

    disparity = dense_match(img_left_w, img_right_w, desp1, desp2)
    visualize_disparity_map(disparity, ans7_path)

if __name__ == '__main__':
    run_hw3()

    source_pattern = os.path.join(assets_dir, 'ans*.png')
    destination = os.getcwd()

    # Move the matching files from 'assets/' to CWD
    for file in glob.glob(source_pattern):
        shutil.move(file, destination)

    # Create a zip file
    zip_filename = 'submission.zip'
    files_to_zip = glob.glob('ans*.png') + ['hw3.py']

    # Zip the files
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in files_to_zip:
            zipf.write(file)

    # Remove the answer files from the current directory
    for file in glob.glob('ans*.png'):
        os.remove(file)

    print(f"Created {zip_filename}. Check results in assets.")