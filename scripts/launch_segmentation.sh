# SLURM commands
# Training
# sh src/mmsegmentation/tools/slurm_train.sh private-cui-gpu s_train_seg configs/eval/segmentation/dinov2_vits14_ade20k_linear_config.py # --work-dir results/dinov2_vits14/segmentation

# Testing
# sh src/mmsegmentation/tools/slurm_test.sh private-cui-gpu s_test_seg configs/eval/segmentation/dinov2_vits14_ade20k_linear_config.py

# Non-SLURM
# Training
python src/mmsegmentation/tools/train.py configs/eval/segmentation/dinov2_vits14_ade20k_linear_config.py
# python src/mmsegmentation/tools/train.py configs/eval/segmentation/dinov2_vitb14_ade20k_linear_config.py
# python src/mmsegmentation/tools/train.py configs/eval/segmentation/dinov2_vitl14_ade20k_linear_config.py
# python src/mmsegmentation/tools/train.py configs/eval/segmentation/dinov2_vitg14_ade20k_linear_config.py

# Testing
# python src/mmsegmentation/tools/test.py configs/eval/segmentation/dinov2_vits14_ade20k_linear_config.py results/dinov2_vits14/segmentation/latest.pth
# python src/mmsegmentation/tools/test.py configs/eval/segmentation/dinov2_vits14_ade20k_linear_config.py results/dinov2_vitb14/segmentation/latest.pth
# python src/mmsegmentation/tools/test.py configs/eval/segmentation/dinov2_vits14_ade20k_linear_config.py results/dinov2_vitl14/segmentation/latest.pth
# python src/mmsegmentation/tools/test.py configs/eval/segmentation/dinov2_vits14_ade20k_linear_config.py results/dinov2_vitg14/segmentation/latest.pth
