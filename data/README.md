# Data
Location for all datasets to be utilized.  Naming of outputs/results depends on dataset name, so if the same dataset name is kept, results will be overwritten. 

## make_patches_from_ndpi.py
Extracts patches from region of an ndpi file as defined in inputs (includes background threshold) and creates a patch folder for that file.  

## make_patches_from_tiff.py
Extracts patches from entire tiff file as defined in inputs (includes background threshold) and creates a patch folder for that file.  

## make_dataset.py
Define a folder for style A and style B patches.  These folders may be populated with patches created from the make_patches_from_ndpi.py or make_patches_from_tiff.py scripts.