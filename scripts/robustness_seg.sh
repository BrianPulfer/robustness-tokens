#!/bin/sh

# No registers, no robustness tokens
PYTHONPATH=src python src/robustness/seg.py --config configs/robustness/segmentation/small.yaml
PYTHONPATH=src python src/robustness/seg.py --config configs/robustness/segmentation/base.yaml
PYTHONPATH=src python src/robustness/seg.py --config configs/robustness/segmentation/large.yaml
PYTHONPATH=src python src/robustness/seg.py --config configs/robustness/segmentation/giant.yaml

# Registers, no robustness tokens
PYTHONPATH=src python src/robustness/seg.py --config configs/robustness/segmentation/small-reg.yaml
PYTHONPATH=src python src/robustness/seg.py --config configs/robustness/segmentation/base-reg.yaml
PYTHONPATH=src python src/robustness/seg.py --config configs/robustness/segmentation/large-reg.yaml
PYTHONPATH=src python src/robustness/seg.py --config configs/robustness/segmentation/giant-reg.yaml

# No registers, Robustness tokens
PYTHONPATH=src python src/robustness/seg.py --config configs/robustness/segmentation/small-rob.yaml
PYTHONPATH=src python src/robustness/seg.py --config configs/robustness/segmentation/base-rob.yaml
PYTHONPATH=src python src/robustness/seg.py --config configs/robustness/segmentation/large-rob.yaml
PYTHONPATH=src python src/robustness/seg.py --config configs/robustness/segmentation/giant-rob.yaml

# Registers and Robustness tokens
PYTHONPATH=src python src/robustness/seg.py --config configs/robustness/segmentation/small-reg-rob.yaml
PYTHONPATH=src python src/robustness/seg.py --config configs/robustness/segmentation/base-reg-rob.yaml
PYTHONPATH=src python src/robustness/seg.py --config configs/robustness/segmentation/large-reg-rob.yaml
PYTHONPATH=src python src/robustness/seg.py --config configs/robustness/segmentation/giant-reg-rob.yaml
