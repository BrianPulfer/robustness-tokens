#!/bin/sh

# No registers, no robustness tokens
PYTHONPATH=src python src/robustness/feat.py --config configs/robustness/features/small.yaml
PYTHONPATH=src python src/robustness/feat.py --config configs/robustness/features/base.yaml
PYTHONPATH=src python src/robustness/feat.py --config configs/robustness/features/large.yaml
PYTHONPATH=src python src/robustness/feat.py --config configs/robustness/features/giant.yaml

# Registers, no robustness tokens
PYTHONPATH=src python src/robustness/feat.py --config configs/robustness/features/small-reg.yaml
PYTHONPATH=src python src/robustness/feat.py --config configs/robustness/features/base-reg.yaml
PYTHONPATH=src python src/robustness/feat.py --config configs/robustness/features/large-reg.yaml
PYTHONPATH=src python src/robustness/feat.py --config configs/robustness/features/giant-reg.yaml

# No registers, Robustness tokens
PYTHONPATH=src python src/robustness/feat.py --config configs/robustness/features/small-rob.yaml
PYTHONPATH=src python src/robustness/feat.py --config configs/robustness/features/base-rob.yaml
PYTHONPATH=src python src/robustness/feat.py --config configs/robustness/features/large-rob.yaml
PYTHONPATH=src python src/robustness/feat.py --config configs/robustness/features/giant-rob.yaml

# Registers and Robustness tokens
PYTHONPATH=src python src/robustness/feat.py --config configs/robustness/features/small-reg-rob.yaml
PYTHONPATH=src python src/robustness/feat.py --config configs/robustness/features/base-reg-rob.yaml
PYTHONPATH=src python src/robustness/feat.py --config configs/robustness/features/large-reg-rob.yaml
PYTHONPATH=src python src/robustness/feat.py --config configs/robustness/features/giant-reg-rob.yaml
