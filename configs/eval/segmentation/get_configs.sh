#!/bin/bash

BASE_URL=https://dl.fbaipublicfiles.com/dinov2
DATASET=ade20k

# Missing: dinov2_vits14_reg dinov2_vitb14_reg dinov2_vitl14_reg dinov2_vitg14_reg
# Meta does not support segmentation for models with registers yet

for arch in dinov2_vits14 dinov2_vitb14 dinov2_vitl14 dinov2_vitg14
do
    # Getting the file
    FILENAME=${arch}_${DATASET}_linear_config.py
    wget ${BASE_URL}/${arch}/${FILENAME}

    # Updating work_dir and data_root
    # sed -i "s|work_dir = '.*'|work_dir = '/srv/beegfs/scratch/users/p/pulfer/robustness_tokens/results/${arch}/segmentation'|g" ${FILENAME}
    # sed -i "s|data_root'.*'|data_root = '/srv/beegfs/scratch/users/p/pulfer/datasets/ADE20K_2021_17_01'|g" ${FILENAME}
done
