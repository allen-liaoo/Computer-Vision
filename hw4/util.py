import cv2
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import hsv_to_rgb

def vector_diff(v1, v2):
    v1 = v1.flatten()
    v2 = v2.flatten()

    v1 = v1 - np.mean(v1)
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)

    v2 = v2 - np.mean(v2)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)

    return v1 @ v2

def visualize_gradients(d_dx, d_dy, grad_mag, grad_angle, filename=None): 
    gm_norm = (grad_mag - np.min(grad_mag)) / (np.max(grad_mag) - np.min(grad_mag))
    ga_norm = (grad_angle - np.min(grad_angle)) / (np.max(grad_angle) - np.min(grad_angle))
    gs = np.dstack((ga_norm, np.ones_like(grad_angle), gm_norm))
    rgb = hsv_to_rgb(gs)
    
    plt.figure(figsize=(10, 8))

    # Plot each subplot
    plt.subplot(2, 2, 1)
    plt.imshow(d_dx, cmap='viridis')
    plt.title('d / du Image')

    plt.subplot(2, 2, 2)
    plt.imshow(d_dy, cmap='viridis')
    plt.title('d / dv Image')

    plt.subplot(2, 2, 3)
    plt.imshow(grad_mag, cmap='viridis')
    plt.title('Gradient Magnitude')

    plt.subplot(2, 2, 4)
    plt.imshow(rgb)
    plt.title('Gradient Angle')
    
    plt.tight_layout()
    if filename:   
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

def visualize_heatmap(heatmap, filename=None):
    plt.imshow(heatmap)
    if filename:   
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size, filename=None):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size ** 2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized ** 2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi / num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size * num_cell_w: cell_size],
                                 np.r_[cell_size: cell_size * num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
            color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    if filename:   
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def visualize_face_detection(I_target, bounding_boxes, box_size, filename=None):
    hh, ww, cc = I_target.shape

    fimg = I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii, 0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1 < 0:
            x1 = 0
        if x1 > ww - 1:
            x1 = ww - 1
        if x2 < 0:
            x2 = 0
        if x2 > ww - 1:
            x2 = ww - 1
        if y1 < 0:
            y1 = 0
        if y1 > hh - 1:
            y1 = hh - 1
        if y2 < 0:
            y2 = 0
        if y2 > hh - 1:
            y2 = hh - 1
        fimg = cv2.rectangle(fimg, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f" % bounding_boxes[ii, 2], (int(x1) + 1, int(y1) + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2, cv2.LINE_AA)

    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    if filename:   
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def visualize_sift(img, filename=None):
    sift = cv2.SIFT_create()
    kp = sift.detect(img, None)
    img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    if filename:   
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def visualize_find_match(img1, img2, x1, x2, img_h=500, filename=None):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h / img1.shape[0]
    scale_factor2 = img_h / img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    if filename:   
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def visualize_align_image_using_feature(img1, img2, x1, x2, A, ransac_thr, img_h=500, filename=None):
    x2_t = np.hstack((x1, np.ones((x1.shape[0], 1)))) @ A.T
    errors = np.sqrt(np.sum(np.square(x2_t[:, :2] - x2), axis=1))
    mask_inliers = errors < ransac_thr
    boundary_t = np.hstack((np.array(
        [[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]], [0, img1.shape[0]], [0, 0]]),
                            np.ones((5, 1)))) @ A[:2, :].T

    scale_factor1 = img_h / img1.shape[0]
    scale_factor2 = img_h / img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    boundary_t = boundary_t * scale_factor2
    boundary_t[:, 0] += img1_resized.shape[1]
    plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'y', linewidth=3)
    for i in range(x1.shape[0]):
        if mask_inliers[i]:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'g')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'go')
        else:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'r')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'ro')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    if filename:   
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def visualize_warp_image(img_warped, img, filename=None):
    plt.subplot(131)
    plt.imshow(img_warped, cmap='gray')
    plt.title('Warped image')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(img, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(np.abs(img_warped - img), cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    if filename:   
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()
