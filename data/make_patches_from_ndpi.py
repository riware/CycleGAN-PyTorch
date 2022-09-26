import os
import numpy as np
import openslide
from skimage import io

############### Inputs ###############

# File to extract patches from
IMAGE_FILE = "/home/riware/Documents/Mittal_Lab/kidney_project/all_model_data/already extracted ndpis/5 A1-9-TRI - 2022-04-04 20.35.20_1 - Copy.ndpi"

# Directory to place patches
DATA_DIRECTORY = "/home/riware/Documents/Mittal_Lab/cycle_gan/dataset_test"

# Details of extraction
pyramid_tier = 0
ndpi_x_tile_size = 10000 #size of ndpi in active memory for extraction - choose based off of available memory
ndpi_y_tile_size = 10000 #size of ndpi in active memory for extraction - choose based off of available memory
x_tile_size = 512 #patch size
y_tile_size = 512 #patch size
threshold = 25  # allows for 25% background
color_delta = 50  # defines distance from 255 considered background
overlap = .40 # .40 indicates 60% overlap

############### Percent Background Calc ###############
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

############### Extraction ###############

# Create folder for case, if doesn't exist
case = os.path.basename(IMAGE_FILE)
case = os.path.splitext(case)[0]
CASE_DIRECTORY = os.path.join(DATA_DIRECTORY, case)
existing_cases = os.listdir(DATA_DIRECTORY)
if os.path.basename(CASE_DIRECTORY) in existing_cases:
    pass
else:
    os.mkdir(CASE_DIRECTORY)

# Create a folder in the case folder for this patch size, if it doesn't exist
existing_patch_sizes = os.listdir(CASE_DIRECTORY)
patch_size = f"{x_tile_size}x{y_tile_size}"
PATCH_DIRECTORY = os.path.join(CASE_DIRECTORY, patch_size)
if patch_size in existing_cases:
    pass
else:
    os.mkdir(PATCH_DIRECTORY)

# Open ndpi with openslide and select ROI
ndpi = openslide.OpenSlide(IMAGE_FILE)
dims = ndpi.level_dimensions
level_dims = dims[pyramid_tier]

# Calc number of ndpi tiles to create. Cuts off last row/column of image if ROI not divisible by tile size
ndpi_x_tile_num = int(np.floor(level_dims[0] / ndpi_x_tile_size))
ndpi_y_tile_num = int(np.floor(level_dims[1] / ndpi_y_tile_size))

# Calc number of tiles to create per ndpi tile.  Cuts off last row/column of image if ROI not divisible by tile size
x_tile_num = int(np.floor(ndpi_x_tile_size / (x_tile_size * overlap)))
y_tile_num = int(np.floor(ndpi_y_tile_size / (y_tile_size * overlap)))

for iy in range(ndpi_y_tile_num):
    for ix in range(ndpi_x_tile_num):
        ndpi_start_x = int(ix * ndpi_x_tile_size)
        ndpi_start_y = int(iy * ndpi_y_tile_size)
        start_pixel = (ndpi_start_x, ndpi_start_y)
        roi = ndpi.read_region(start_pixel, level=pyramid_tier, size=(ndpi_x_tile_size, ndpi_y_tile_size))
        roi_array = np.array(roi)[:, :, 0:3]
        
        # Left to right and top to bottom
        for jy in range(y_tile_num):
            for jx in range(x_tile_num):
                # Coordinates of the upper left corner of each image
                start_x = int(jx * x_tile_size * overlap)
                start_y = int(jy * y_tile_size * overlap)
                end_x = start_x + x_tile_size
                end_y = start_y + y_tile_size

                if end_x > ndpi_x_tile_size or end_y > ndpi_y_tile_size:
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
                        patch_savename = f"{case}_{iy}-{ix}_{jy}-{jx}.png"
                        io.imsave(os.path.join(PATCH_DIRECTORY, patch_savename), cur_tile)
                    else:
                        print(f"{case}_{iy}-{ix}_{jy}-{jx} has too much background: {per_background}")
