# Robustness Tokens

## Set-up
### Conda environment
Create a conda environment with the required dependencies with either:

```bash
conda create --file environment.yml
conda activate rtokens
```

or

```bash
conda create -n rtokens python=3.11
conda activate rtokens
pip install -r requirements.txt
```

### Dotenv file
Create a `.env` file in the root directory of the project with the following content:

```bash
PYTHONPATH=src
```

## Training
To train robustness tokens for a pre-trained DinoV2 model, run:

```bash
PYTHONPATH=src/ python src/train.py --config $path_to_file
```

An example of training configuration file can be found in [`configs/train/default.yaml`](configs/train/default.yaml).

## Evaluation
(⚠️ **Warning**: this is still work in progress and won't currently work ⚠️)

To evaluate robustness tokens, run:
```bash
# Evaluate classification performance
PYTHONPATH=src/ python src/eval/classification.py --config $path_to_file
# Evaluate segmentation performance
PYTHONPATH=src/ python src/eval/segmentation.py --config $path_to_file
# Evaluate depth estimation performance
PYTHONPATH=src/ python src/eval/depth.py --config $path_to_file
```

##  License
The code is released with the [MIT license](LICENSE).
