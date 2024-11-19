import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm

from util import *

def get_differential_filter():
    # Returns Sobel-like differential filters in x and y directions for edge detection.
    # Output:
    #   filter_x (3x3 ndarray): filter for x-direction.
    #   filter_y (3x3 ndarray): filter for y-direction (transpose of filter_x).

    filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return filter_x, filter_y


def filter_image(im, filter):
    # Applies a specified filter to an image using manual convolution.
    # Input:
    #   im (HxW ndarray): grayscale image.
    #   filter (kxk ndarray): filter to be applied.
    # Output:
    #   im_filtered (HxW ndarray): filtered image.
    # Hint: Pad the Image before applying the convolution.

    # pad the image
    npads = filter.shape[0]//2
    newim = np.pad(im, (npads, npads), constant_values=(0,))

    # convolution
    im_windows = np.lib.stride_tricks.sliding_window_view(newim, filter.shape)
    conv_t = filter * im_windows
    im_filtered = np.sum(conv_t, (2,3))

    # H, W = filter.shape
    # im_filtered = np.zeros(im.shape)
    # for i in range(newim.shape[0]-H+1):
    #     for j in range(newim.shape[1]-W+1):
    #         window = newim[i:i+H, j:j+W]
    #         conv_t = filter * window
    #         im_filtered[i][j] = np.sum(conv_t)
    return im_filtered


def get_gradient(im_dx, im_dy):
    # Computes gradient magnitude and direction from x and y gradient images.
    # Input:
    #   im_dx (HxW ndarray): gradient image in x-direction.
    #   im_dy (HxW ndarray): gradient image in y-direction.
    # Output:
    #   grad_mag (HxW ndarray): gradient magnitudes.
    #   grad_angle (HxW ndarray): gradient angles, shifted to non-negative values.

    grad_mag = np.sqrt(im_dx ** 2 + im_dy ** 2)#np.square(im_dx) + np.square(im_dy))
    grad_angle = np.arctan2(im_dy, im_dx) #np.atan(im_dy / (im_dx + 1e-10))
    # make grad_angle positive
    from math import pi as PI
    grad_angle[grad_angle < 0] += PI
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size, num_bins=6):
    # Builds a histogram of gradient orientations within cells.
    # Input:
    #   grad_mag (HxW ndarray): gradient magnitudes.
    #   grad_angle (HxW ndarray): gradient angles.
    #   cell_size (int): size of each cell.
    #   num_bins (int): number of bins for teh histogram
    # Output:
    #   ori_histo (H'xW'xB ndarray): orientation histogram with B bins for each cell.

    from math import pi as PI

    histos = []
    shifted_grad_angle = (grad_angle + (PI / (num_bins*2))) % PI

    # angleRange = (0, PI) # hardcoded :(
    # for i in range(0, grad_mag.shape[0]-cell_size+1, cell_size):
    #     for j in range(0, grad_mag.shape[1]-cell_size+1, cell_size):
    #         hist = [0] * num_bins
    #         for k in range(cell_size):
    #             for l in range(cell_size):
    #                 angle = shifted_grad_angle[i+k][j+l]
    #                 mag = grad_mag[i+k][j+l]

    #                 # calculate index to add to histogram based on range and num_bins
    #                 index = int((angle - angleRange[0]) // ((angleRange[1] - angleRange[0]) / num_bins))
    #                 hist[index] += mag
    #         histos.append(hist)

    for i in range(0, grad_mag.shape[0]-cell_size+1, cell_size):
        for j in range(0, grad_mag.shape[1]-cell_size+1, cell_size):
            hist, bins = np.histogram(shifted_grad_angle[i:i+cell_size, j:j+cell_size].flatten(),
                                   bins= num_bins,
                                   range= (0, PI),
                                   weights= grad_mag[i:i+cell_size, j:j+cell_size].flatten())
            histos.append(hist)

    H, W = grad_mag.shape[0] // cell_size, grad_mag.shape[1] // cell_size
    ori_histo = np.array(histos).reshape((H, W, num_bins))
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # Normalizes histograms within blocks of cells.
    # Input:
    #   ori_histo (CxCxB ndarray): cell histograms.
    #   block_size (int): block size (in cells).
    # Output:
    #   ori_histo_normalized (CxCx(BxB) ndarray): normalized block histograms.
    # Hint: Each block contains (block_size)^2 cells.
    M, N = ori_histo.shape[0] - block_size + 1, ori_histo.shape[1] - block_size + 1

    # ori_histo_n = []
    # for i in range(M):
    #     for j in range(N):
    #         block = []
    #         for k in range(block_size):
    #             for l in range(block_size):
    #                 block.append(ori_histo[i+k,j+l])
    #         block = np.array(block).flatten()
    #         ori_histo_n.append(block / np.sqrt(np.sum(np.square(block)) + 1e-5))
    # ori_histo_normalized = np.array(ori_histo_n).reshape((M, N, -1), order='C')
    # return ori_histo_normalized

    block_windows = np.lib.stride_tricks.sliding_window_view(ori_histo, (block_size, block_size, 6), axis=(0,1,2))
    BH, BW = block_windows.shape[0], block_windows.shape[1]

    ori_histo_n = []
    for i in range(BH):
        for j in range(BW):
            block = block_windows[i,j,:].flatten()
            ori_histo_n.append(block / np.sqrt(np.sum(np.square(block)) + 1e-5))
    ori_histo_normalized = np.array(ori_histo_n).reshape((BH, BW, -1), order='C')
    return ori_histo_normalized

def extract_hog(im, cell_size=8, block_size=2, num_bins=6, show=True):
    # Extracts HOG (Histogram of Oriented Gradients) features from an image.
    # Input:
    #   im (HxW ndarray): grayscale image.
    #   cell_size (int): cell size in pixels.
    #   block_size (int): block size in cells.
    #   num_bins (int): number of histogram bins
    # Output:
    #   hog (1D ndarray): flattened HOG feature vector.
    # Hint you just need to combine get_gradient, build_histogram, and get_block_descriptor

    # im = im.astype('float') / 255.0

    # get gradients
    dx, dy = get_differential_filter() # get sobel filters
    im_dx, im_dy = filter_image(im, dx), filter_image(im, dy) # apply filters
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)

    hist = build_histogram(grad_mag, grad_angle, cell_size, num_bins)
    histo_normd = get_block_descriptor(hist, block_size)
    return histo_normd.flatten()


def compute_iou(box1, box2, w, h):
    # Computes the Intersection over Union (IoU) for two bounding boxes.
    # Input:
    #   box1, box2 (np.array): bounding box coordinates [x, y].
    #   w, h (int): width and height of the bounding boxes.
    # Output:
    #   iou (float): Intersection over Union score.

    dx = abs(box1[0] - box2[0])
    dy = abs(box1[1] - box2[1])
    i_area = (w-dx) * (h-dy)
    u_area = (w*h) * 2 - i_area
    iou = i_area / u_area
    return iou


def non_max_suppression(boxes, threshold, w, h):
    # Filters overlapping bounding boxes using non-maximum suppression (NMS).
    # Input:
    #   boxes (Nx3 ndarray): list of bounding boxes with confidence scores.
    #   threshold (float): IoU threshold for filtering.
    #   w, h (int): width and height of bounding boxes.
    # Output:
    #   keep (Nx3 ndarray): filtered bounding boxes after NMS.

    confs = boxes[:, 2] # confidences
    ord = np.argsort(confs)
    ord = np.flip(ord) # ascending to descending order
    order = ord.tolist()

    keep = []
    while len(order) != 0:
        box1 = boxes[order.pop(0)]
        keep.append(box1)

        i = 0
        while len(order) != 0 and i < len(order):
            box2 = boxes[order[i]]
            if compute_iou(box1[:2], box2[:2], w, h) >= threshold:
                order.pop(i)
            else:
                i += 1
    return np.array(keep)


def face_recognition(I_target, I_template, thr_diff=.45, thr_iou=.5, stride=2):
    # Performs template matching using HOG features to locate a face in the target image.
    # Input:
    #   I_target (HxW ndarray): target image.
    #   I_template (hxw ndarray): face template.
    #   thr_diff (float): threshold for normalized cross-correlation.
    #   thr_iou (float): IoU threshold for NMS.
    #   stride (int): step size for sliding window.
    # Output:
    #   bounding_boxes (list): list of bounding boxes for matched faces.
    # Hint compute the hog for the template, move a sliding window across the image and apply hog to each window then compare with cosine similarity

    # I_template = I_template.astype('float') / 255.0
    # I_target = I_target.astype('float') / 255.0
    hog = extract_hog(I_template)
    hog_mean = np.mean(hog)

    winH, winW = I_template.shape
    H, W = I_target.shape

    boxes = []
    for i in range(0, H - winH, stride):
        for j in range(0, W - winW, stride):
            win = I_target[i:i+winH, j:j+winW]
            win_hog = extract_hog(win)
            win_hog_mean = np.mean(win_hog)

            a = hog - hog_mean
            b = win_hog - win_hog_mean
            conf = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            if (conf >= thr_diff):
                boxes.append([j, i, conf])

    bounding_boxes = non_max_suppression(np.array(boxes), thr_iou, winW, winH)
    return bounding_boxes


def find_match(img1, img2, dist_thr=.7):
    # Finds matching keypoints between two images using SIFT and nearest neighbor matching.
    # Input:
    #   img1, img2 (HxW ndarray): grayscale images.
    #   dist_thr (float): distance threshold for filtering matches.
    # Output:
    #   x1, x2 (Nx2 ndarray): matched keypoints in each image.
    # Hint: Good article on the topic https://datahacker.rs/feature-matching-methods-comparison-in-opencv/
    # You are allowed to use: cv2.SIFT_create, sift.detectAndCompute, cv2.KeyPoint_convert, NearestNeighbors, nbrs2.kneighbors, and all numpy functions

    sift = cv2.SIFT.create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    nn = NearestNeighbors()
    nn = nn.fit(des2)
    dists, inds = nn.kneighbors(des1, n_neighbors=2, return_distance= True)

    kp1Matches = []
    kp2Matches = []
    for i in range(dists.shape[0]):
        if dists[i][0] < dist_thr * dists[i][1]:
            kp1Matches.append(i)
            kp2Matches.append(inds[i][0])

    x1 = cv2.KeyPoint.convert(kp1, np.array(kp1Matches))
    x2 = cv2.KeyPoint.convert(kp2, np.array(kp2Matches))
    return x1, x2

def compute_A(x1, x2):
    # Computes the affine transformation matrix between two sets of points using RANSAC.
    # 
    # Input:
    #    x1 (Nx2 ndarray): Keypoints from the first image.
    #    x2 (Nx2 ndarray): Keypoints from the second image.
    #
    # Output:
    #    A (3x3 ndarray): Affine transformation matrix.

    A, _ = cv2.estimateAffine2D(x2, x1)
    A = np.vstack((A, np.array([0,0,1])))
    return A

def compute_warped_image(target, A, width, height):
    # Warps the target image using the provided affine transformation matrix.
    #
    # Input:
    #     target (HxW ndarray): Image to be warped.
    #     A (3x3 ndarray): Affine transformation matrix.
    #     width (int): Width of the output image.
    #     height (int): Height of the output image.
    #
    # Output:
    #     warped_image (HxW ndarray): Warped image.

    img_warped = cv2.warpPerspective(target, A, (width, height))
    return img_warped

if __name__ == '__main__':
    ##### HOG #####
    im = cv2.imread('assets/cameraman.tif', 0)
    res = np.load('assets/correct_results.npz')

    # step one calculate gradients with convolutions
    dx, dy = get_differential_filter() # get sobel filters
    im_dx, im_dy = filter_image(im, dx), filter_image(im, dy) # apply filters
    grad_mag, grad_angle = get_gradient(im_dx, im_dy) # compute mag and dir

    visualize_gradients(im_dx, im_dy, grad_mag, grad_angle)

    assert im_dx.shape == res["dx"].shape, "Dims wrong" # Make sure you pad
    assert im_dy.shape == res["dy"].shape, "Dims wrong" # Make sure you pad
    assert vector_diff(im_dx, res["dx"]) > .9, "Image DX gradient is not close enough" # Hint if you vector_diff between our answer and yours is clos to -1 ie -.98 then your gradient is backwards
    assert vector_diff(im_dy, res["dy"]) > .9, "Image DX gradient is not close enough" # Hint if you vector_diff between our answer and yours is clos to -1 ie -.98 then your gradient is backwards

    # step two get hog features and visualize
    hog = extract_hog(im, cell_size=8, block_size=2)
    visualize_hog(im.astype('float') / 255.0, hog, cell_size=8, block_size=2)

    assert res["hog"].shape == hog.shape, "Make sure to flatten and use cell_size=8, block_size=2, bin_size=6"
    assert vector_diff(res["hog"], hog) > .6, "Your HOG features do not resemble out answer close enough"

    I_target = cv2.imread('assets/target.png', 0) # MxN image
    I_template = cv2.imread('assets/template.png', 0) # mxn  face template

    bounding_boxes = face_recognition(I_target, I_template)

    I_target_c = cv2.imread('assets/target.png') # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0]) # this is visualization code.

    assert res["bb"].shape == bounding_boxes.shape, "You Didn't find all the faces"
    assert np.max(np.linalg.norm(res["bb"][:, :2] - bounding_boxes[:, :2], axis=1)) < 12.5, "Bounding Boxes Not close enough" 

    ##### SIFT #####
    template = cv2.imread('assets/sift_template.jpg', 0)  # read as grey scale image
    target = cv2.imread('assets/sift_target.jpg', 0)  # read as grey scale image
    H_temp, W_temp = template.shape

    visualize_sift(target)

    x1, x2 = find_match(template, target)
    visualize_find_match(template, target, x1, x2)

    # TODO: specify parameters.
    ransac_thr = 3
    A = compute_A(x1, x2)
    visualize_align_image_using_feature(template, target, x1, x2, A, ransac_thr)

    # TODO: Warp the image using the computed homography.
    img_warped = compute_warped_image(target, A, W_temp, H_temp)
    visualize_warp_image(img_warped, template)

    hog_template = extract_hog(template, cell_size=8, block_size=2, show=False)
    hog_warped = extract_hog(img_warped, cell_size=8, block_size=2, show=False)

    assert vector_diff(hog_template, hog_warped) > .7, "Warped Image is not similar enough to the template"
