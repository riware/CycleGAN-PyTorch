import os
import random
import shutil

A_DATAPATH = "/home/riware/Documents/Mittal_Lab/cycle_gan/dataset_test/16 A1-9-TRI-K - 2022-03-29 10.12.59_1 - Copy_ndpi_pixel(138000, 24000)_size15000/patch_512"
B_DATAPATH = "/home/riware/Documents/Mittal_Lab/cycle_gan/dataset_test/16 A1-9-TRI-K - 2022-03-29 10.12.59_1 - Copy_ndpi_pixel(138000, 24000)_size15000_redpertscale1.0bias-0.5/patch_512"

A_CYCLEGAN_TRAIN = "/home/riware/Documents/Mittal_Lab/cycle_gan/CycleGAN-PyTorch/data/stain2newstaintest/train/A"
B_CYCLEGAN_TRAIN = "/home/riware/Documents/Mittal_Lab/cycle_gan/CycleGAN-PyTorch/data/stain2newstaintest/train/B"

A_CYCLEGAN_TEST = "/home/riware/Documents/Mittal_Lab/cycle_gan/CycleGAN-PyTorch/data/stain2newstaintest/test/A"
B_CYCLEGAN_TEST = "/home/riware/Documents/Mittal_Lab/cycle_gan/CycleGAN-PyTorch/data/stain2newstaintest/test/B"

def select_cases(A_DATAPATH, B_DATAPATH, A_CYCLEGAN_TRAIN, B_CYCLEGAN_TRAIN, A_CYCLEGAN_TEST, B_CYCLEGAN_TEST, num_train_patches, num_test_patches):
    # Get list of cases
    a_patches = [file for file in os.listdir(A_DATAPATH) if os.path.isfile(os.path.join(A_DATAPATH, file))]
    b_patches = [file for file in os.listdir(B_DATAPATH) if os.path.isfile(os.path.join(B_DATAPATH, file))]
    a_test = []
    b_test = []

    # Randomly select cases and exclude num_test_cases from selection
    a_train = random.sample(a_patches, num_train_patches)
    b_train = random.sample(b_patches, num_train_patches)

    for a_file in a_patches:
        if a_file in a_train:
            shutil.copy(os.path.join(A_DATAPATH, a_file), A_CYCLEGAN_TRAIN)
        else:
            a_test.append(a_file)

    for b_file in b_patches:
        if b_file in b_train:
            shutil.copy(os.path.join(B_DATAPATH, b_file), B_CYCLEGAN_TRAIN)
        else:
            b_test.append(b_file)

    a_test = random.sample(a_test, num_test_patches)
    b_test = random.sample(b_test, num_test_patches)

    for a_file, b_file in zip(a_test, b_test):
        shutil.copy(os.path.join(A_DATAPATH, a_file), A_CYCLEGAN_TEST)
        shutil.copy(os.path.join(B_DATAPATH, b_file), B_CYCLEGAN_TEST)
    return

test = select_cases(A_DATAPATH, B_DATAPATH, A_CYCLEGAN_TRAIN, B_CYCLEGAN_TRAIN, A_CYCLEGAN_TEST, B_CYCLEGAN_TEST, num_train_patches=3000, num_test_patches=100)