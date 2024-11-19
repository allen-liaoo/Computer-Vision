from hw4 import *
from util import *

import os
import glob
import shutil
import zipfile

assets_dir = 'assets'

def run_hw4():
    # Read in left and right images as RGB images
    im = cv2.imread(os.path.join(assets_dir, 'cameraman.tif'), 0)

    # step one calculate gradients with convolutions
    dx, dy = get_differential_filter() # get sobel filters
    im_dx, im_dy = filter_image(im, dx), filter_image(im, dy) # apply filters
    grad_mag, grad_angle = get_gradient(im_dx, im_dy) # compute mag and dir

    visualize_gradients(im_dx, im_dy, grad_mag, grad_angle, filename=os.path.join(assets_dir, 'ans0.png'))

    # step two get hog features and visualize
    hog = extract_hog(im, cell_size=8, block_size=2)
    visualize_hog(im.astype('float') / 255.0, hog, cell_size=8, block_size=2, filename=os.path.join(assets_dir, 'ans1.png'))

    I_target = cv2.imread(os.path.join(assets_dir, 'target.png'), 0) # MxN image
    I_template = cv2.imread(os.path.join(assets_dir, 'template.png'), 0) # mxn  face template

    bounding_boxes = face_recognition(I_target, I_template, stride=10)

    I_target_c = cv2.imread(os.path.join(assets_dir, 'target.png')) # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0], filename=os.path.join(assets_dir, 'ans2.png')) # this is visualization code.

    ##### SIFT #####
    template = cv2.imread(os.path.join(assets_dir, 'sift_template.jpg'), 0)  # read as grey scale image
    target = cv2.imread(os.path.join(assets_dir, 'sift_target.jpg'), 0)  # read as grey scale image
    H_temp, W_temp = template.shape

    x1, x2 = find_match(template, target)
    visualize_find_match(template, target, x1, x2, filename=os.path.join(assets_dir, 'ans3.png'))

    A = compute_A(x1, x2)
    img_warped = compute_warped_image(target, A, W_temp, H_temp)
    ransac_thr = 3

    visualize_align_image_using_feature(template, target, x1, x2, A, ransac_thr, filename=os.path.join(assets_dir, 'ans4.png'))
    visualize_warp_image(img_warped, template, filename=os.path.join(assets_dir, 'ans5.png'))


if __name__ == '__main__':
    try:
        run_hw4()
    except Exception as e:
        print(e)
        print("Hw4 is not complete, but you can still submit it")

    source_pattern = os.path.join(assets_dir, 'ans*.png')
    cwd = os.getcwd()

    for file_path in glob.glob(source_pattern):
        shutil.move(file_path, cwd)

    # move files out of assets to cwd

    # Create a zip file
    zip_filename = 'submission.zip'
    files_to_zip = glob.glob('ans*.png') + ['hw4.py']

    # Zip the files
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in files_to_zip:
            zipf.write(file)
    
    # move files back
    for file_path in glob.glob('ans*.png'):
        shutil.move(file_path, assets_dir)

    print(f"Created {zip_filename}. Check results in assets.")
