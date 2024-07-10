# Robustness Tokens
![Training schema for Robustness Tokens](assets/training.png)

Official implementation of the paper <i>"Robustness Tokens: Towards Adversarial Robustness of Transformers"</i> (ECCV 2024).

## Set-up
We build on top of the official [DinoV2](https://github.com/facebookresearch/dinov2) implementation. In particular, our code:
  - Allows training robustness tokens
  - Implements the PGD adversarial attack
  - Converts obtained robustness tokens into a valid checkpoint for evaluation with the DinoV2 codebase

We use the official [DinoV2](https://github.com/facebookresearch/dinov2) codebase for evaluation of robustness tokens on downstream tasks:
  - Classification
  - Semantic segmentation

For each task, we evaluate the performances of the original model and of the model with robustness tokens, both on clean and adversarial examples.


### Conda environment
Create the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate rtokens
```

Alternatively, you can install the required packages using pip:

```bash
conda create -n rtokens python=3.11
conda activate rtokens

# Install PyTorch with the desired version, e.g.:
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

# Install the remaining requirements
pip install -r requirements.txt
```

### Python-Dotenv
Create a `.env` file in the root directory of the project with a variable `IMAGENET_DIR` pointing to the ImageNet dataset directory.

### Datasets
We use the [ImageNet](https://image-net.org/) dataset for training robustness tokens and to evaluate linear classification capabilities. For segmentation, we use the [ADE20k 2016](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) dataset. You can get the test set from [here](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip).

For the **ImageNet** dataset, pre-process the dataset as described in the [DinoV2](https://github.com/facebookresearch/dinov2/blob/main/README.md) codebase:

```python
from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root=your_imagenet_dir, extra=your_extra_dir)
    dataset.dump_extra()
```

you will need to use the [labels.txt](labels.txt) file to map the ImageNet labels to the class indices used in the codebase.

## Training robustness tokens
To train robustness tokens for a pre-trained DinoV2 model, run:

```bash
PYTHONPATH=src/ python src/train.py --config $path_to_file
```

Examples of training configurations can be found in [`configs/train/`](configs/train/).

## Evaluating robustness
You can evaluate the robustness of features extracted by models with or without robustness tokens to adversarial attacks.

```bash
PYTHONPATH=src/ python src/robustness/feat.py --config $path_to_file
```

Examples of training configurations can be found in [`configs/robustness/features/`](configs/robustness/features/).

The same can be done to evaluate robustness in the case of classification and segmentation with the scripts [`src/robustness/class.py`](src/robustness/class.py) and [`src/robustness/seg.py`](src/robustness/seg.py), respectively.


## Evaluating downstream performance
We verify that downstream performance is not affected by the addition of robustness tokens.

For evaluation, we convert our checkpoints into a valid checkpoint for the DinoV2 codebase.

```bash
PYTHONPATH=src/ python src/eval/convert.py --checkpoint $path_to_file --output $path_to_file
```

The robustness tokens are converted into DinoV2 *register* tokens and appended before patch tokens. Please refer to the [DinoV2](https://github.com/facebookresearch/dinov2) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) codebases for more details on how to evaluate the downstream performance.

##  License
The code is released with the [MIT license](LICENSE).
