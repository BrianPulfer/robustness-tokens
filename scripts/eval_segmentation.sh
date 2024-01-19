#!/bin/sh

## No registers, no robustness tokens
CPUS_PER_TASK=6 GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh tools/slurm_train.sh shared-gpu vits-seg ../../configs/eval/segmentation/without_rtokens/dinov2_vits14_ade20k_linear_config.py
CPUS_PER_TASK=6 GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh tools/slurm_train.sh shared-gpu vitb-seg ../../configs/eval/segmentation/without_rtokens/dinov2_vitb14_ade20k_linear_config.py
CPUS_PER_TASK=6 GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh tools/slurm_train.sh shared-gpu vitl-seg ../../configs/eval/segmentation/without_rtokens/dinov2_vitl14_ade20k_linear_config.py
CPUS_PER_TASK=6 GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh tools/slurm_train.sh shared-gpu vitg-seg ../../configs/eval/segmentation/without_rtokens/dinov2_vitg14_ade20k_linear_config.py

# Registers, no robustness tokens
CPUS_PER_TASK=6 GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh tools/slurm_train.sh shared-gpu vits-seg ../../configs/eval/segmentation/without_rtokens/dinov2_vits14_reg_ade20k_linear_config.py
CPUS_PER_TASK=6 GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh tools/slurm_train.sh shared-gpu vitb-seg ../../configs/eval/segmentation/without_rtokens/dinov2_vitb14_reg_ade20k_linear_config.py
CPUS_PER_TASK=6 GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh tools/slurm_train.sh shared-gpu vitl-seg ../../configs/eval/segmentation/without_rtokens/dinov2_vitl14_reg_ade20k_linear_config.py
CPUS_PER_TASK=6 GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh tools/slurm_train.sh shared-gpu vitg-seg ../../configs/eval/segmentation/without_rtokens/dinov2_vitg14_reg_ade20k_linear_config.py

# No registers, Robustness tokens
CPUS_PER_TASK=6 GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh tools/slurm_train.sh shared-gpu vits-seg ../../configs/eval/segmentation/dinov2_vits14_ade20k_linear_config.py
CPUS_PER_TASK=6 GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh tools/slurm_train.sh shared-gpu vitb-seg ../../configs/eval/segmentation/dinov2_vitb14_ade20k_linear_config.py
CPUS_PER_TASK=6 GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh tools/slurm_train.sh shared-gpu vitl-seg ../../configs/eval/segmentation/dinov2_vitl14_ade20k_linear_config.py
CPUS_PER_TASK=6 GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh tools/slurm_train.sh shared-gpu vitg-seg ../../configs/eval/segmentation/dinov2_vitg14_ade20k_linear_config.py

# Registers and Robustness tokens
CPUS_PER_TASK=6 GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh tools/slurm_train.sh shared-gpu vits-seg ../../configs/eval/segmentation/dinov2_vits14_reg_ade20k_linear_config.py
CPUS_PER_TASK=6 GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh tools/slurm_train.sh shared-gpu vitb-seg ../../configs/eval/segmentation/dinov2_vitb14_reg_ade20k_linear_config.py
CPUS_PER_TASK=6 GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh tools/slurm_train.sh shared-gpu vitl-seg ../../configs/eval/segmentation/dinov2_vitl14_reg_ade20k_linear_config.py
CPUS_PER_TASK=6 GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh tools/slurm_train.sh shared-gpu vitg-seg ../../configs/eval/segmentation/dinov2_vitg14_reg_ade20k_linear_config.py
