import os
import numpy as np
import cv2
from skimage import io


IMAGE_FILE = "/home/riware/Documents/Mittal_Lab/kidney_project/VGG16_model_data/model18/altered ROIs/16 A1-9-TRI-K - 2022-03-29 10.12.59_1 - Copy_ndpi_pixel(138000, 24000)_size15000_redpertscale1.0bias-0.5.tiff"
DATA_DIRECTORY = "/home/riware/Documents/Mittal_Lab/cycle_gan/dataset_test"
x_tile_size = 768 #patch size
y_tile_size = 768 #patch size
threshold = 25  # allows for 25% background
color_delta = 50  # defines distance from 255 considered background
overlap = .40 # .40 indicates 60% overlap

def percent_background(roi_array, color_delta):
    roi_width = roi_array.shape[0]
    roi_height = roi_array.shape[1]
    white = np.array([255, 255, 255])
    threshold = np.subtract(white, color_delta)
    sub_threshold = np.subtract(roi_array, threshold)
    sub_threshold = np.where(sub_threshold < 0, 0, sub_threshold)
    num_background_pix = np.count_nonzero(np.all(sub_threshold, axis=2))
    total_pixels = roi_width * roi_height
    percent_background = num_background_pix / total_pixels * 100
    return percent_background

# Create folder for case, if doesn't exist
case = os.path.basename(IMAGE_FILE)
case = os.path.splitext(case)[0]
CASE_DIRECTORY = os.path.join(DATA_DIRECTORY, case)
existing_cases = os.listdir(DATA_DIRECTORY)
if os.path.basename(CASE_DIRECTORY) in existing_cases:
    pass
else:
    os.mkdir(CASE_DIRECTORY)

# Create a folder in the case folder for this patch size
PATCH_DIRECTORY = os.path.join(CASE_DIRECTORY, f"patch_{x_tile_size}")
os.mkdir(PATCH_DIRECTORY)

# Open tiff with openslide and select ROI
tiff_file = cv2.imread(IMAGE_FILE)
tiff_file = cv2.cvtColor(tiff_file, cv2.COLOR_BGR2RGB)
roi_array = np.asarray(tiff_file)
roi_size_x = roi_array.shape[1]
roi_size_y = roi_array.shape[0]

# Calc number of tiles to create per ndpi tile.  Cuts off last row/column of image if ROI not divisible by tile size
x_tile_num = int(np.floor(roi_size_x / (x_tile_size * overlap)))
y_tile_num = int(np.floor(roi_size_y / (y_tile_size * overlap)))

# Left to right and top to bottom
for iy in range(y_tile_num):
    for ix in range(x_tile_num):
        # Coordinates of the upper left corner of each image
        start_x = int(ix * x_tile_size * overlap)
        start_y = int(iy * y_tile_size * overlap)
        end_x = start_x + x_tile_size
        end_y = start_y + y_tile_size
        if end_x > roi_size_x or end_y > roi_size_y: 
            pass
        else:
            # Grab patch and only make image if the threshold for black pixels is satisfied
            cur_tile = roi_array[
                start_y:end_y,
                start_x:end_x,
            ]
            per_background = percent_background(cur_tile, color_delta)
            # py.imshow(cur_tile)
            # py.show()
            if per_background < threshold:
                patch_savename = f"{case}_{iy}-{ix}_patch_{x_tile_size}.png"
                io.imsave(os.path.join(PATCH_DIRECTORY, patch_savename), cur_tile)
            else:
                print(f"{case}_{iy}-{ix} has too much background: {per_background}")
