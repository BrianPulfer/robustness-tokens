# Single GPU
# Training
# python src/mmsegmentation/tools/train.py configs/eval/segmentation/dinov2_vits14_ade20k_linear_config.py
# python src/mmsegmentation/tools/train.py configs/eval/segmentation/dinov2_vitb14_ade20k_linear_config.py
# python src/mmsegmentation/tools/train.py configs/eval/segmentation/dinov2_vitl14_ade20k_linear_config.py
# python src/mmsegmentation/tools/train.py configs/eval/segmentation/dinov2_vitg14_ade20k_linear_config.py

# Testing
# python src/mmsegmentation/tools/test.py configs/eval/segmentation/dinov2_vits14_ade20k_linear_config.py results/dinov2_vits14/segmentation/latest.pth
# python src/mmsegmentation/tools/test.py configs/eval/segmentation/dinov2_vits14_ade20k_linear_config.py results/dinov2_vitb14/segmentation/latest.pth
# python src/mmsegmentation/tools/test.py configs/eval/segmentation/dinov2_vits14_ade20k_linear_config.py results/dinov2_vitl14/segmentation/latest.pth
# python src/mmsegmentation/tools/test.py configs/eval/segmentation/dinov2_vits14_ade20k_linear_config.py results/dinov2_vitg14/segmentation/latest.pth


# Multi-GPU
# Training
GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh src/mmsegmentation/tools/slurm_train.sh shared-gpu vits-seg configs/eval/segmentation/dinov2_vits14_ade20k_linear_config.py
GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh src/mmsegmentation/tools/slurm_train.sh shared-gpu vitb-seg configs/eval/segmentation/dinov2_vitb14_ade20k_linear_config.py
GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh src/mmsegmentation/tools/slurm_train.sh shared-gpu vitl-seg configs/eval/segmentation/dinov2_vitl14_ade20k_linear_config.py
GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--time 12:00:00" sh src/mmsegmentation/tools/slurm_train.sh shared-gpu vitg-seg configs/eval/segmentation/dinov2_vitg14_ade20k_linear_config.py
