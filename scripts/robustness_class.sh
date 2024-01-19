#!/bin/sh

# No registers, no robustness tokens
PYTHONPATH=src python src/robustness/class.py --config configs/robustness/classification/small.yaml
PYTHONPATH=src python src/robustness/class.py --config configs/robustness/classification/base.yaml
PYTHONPATH=src python src/robustness/class.py --config configs/robustness/classification/large.yaml
PYTHONPATH=src python src/robustness/class.py --config configs/robustness/classification/giant.yaml

# Registers, no robustness tokens
PYTHONPATH=src python src/robustness/class.py --config configs/robustness/classification/small-reg.yaml
PYTHONPATH=src python src/robustness/class.py --config configs/robustness/classification/base-reg.yaml
PYTHONPATH=src python src/robustness/class.py --config configs/robustness/classification/large-reg.yaml
PYTHONPATH=src python src/robustness/class.py --config configs/robustness/classification/giant-reg.yaml

# No registers, Robustness tokens
PYTHONPATH=src python src/robustness/class.py --config configs/robustness/classification/small-rob.yaml
PYTHONPATH=src python src/robustness/class.py --config configs/robustness/classification/base-rob.yaml
PYTHONPATH=src python src/robustness/class.py --config configs/robustness/classification/large-rob.yaml
PYTHONPATH=src python src/robustness/class.py --config configs/robustness/classification/giant-rob.yaml

# Registers and Robustness tokens
PYTHONPATH=src python src/robustness/class.py --config configs/robustness/classification/small-reg-rob.yaml
PYTHONPATH=src python src/robustness/class.py --config configs/robustness/classification/base-reg-rob.yaml
PYTHONPATH=src python src/robustness/class.py --config configs/robustness/classification/large-reg-rob.yaml
PYTHONPATH=src python src/robustness/class.py --config configs/robustness/classification/giant-reg-rob.yaml
