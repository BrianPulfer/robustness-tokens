# Robustness Tokens

## Set-up
We build on top of the official [DinoV2](https://github.com/facebookresearch/dinov2) implementation. In particular, our code:
  - Allows training robustness tokens
  - Implements adversarial attacks (PGD)
  - Converts obtained robustness tokens into a valid checkpoint for evaluation with the DinoV2 codebase

We use the official [DinoV2](https://github.com/facebookresearch/dinov2) codebase for evaluation of robustness tokens on downstream tasks:
  - Classification
  - Semantic segmentation
  - Depth estimation

For each task, we evaluate the performances of the original model and of the model with robustness tokens, both on clean and adversarial examples.


### Conda environment
Create the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate rtokens
```

### Clone DinoV2
Clone the [DinoV2](https://github.com/facebookresearch/dinov2) repository under `/src`:

```bash
cd src/
git clone -b main https://github.com/facebookresearch/dinov2.git
```


### Datasets
We use the [ImageNet](https://image-net.org/) dataset for training robustness tokens and to evaluate linear classification capabilities. For segmentation, we use the [ADE20k dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/) dataset, whereas for depth estimation, we use the [NYU Depth v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset.

For the **ImageNet** dataset, pre-process the dataset as described in the [DinoV2](https://github.com/facebookresearch/dinov2/blob/main/README.md) codebase:
```python
from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root=your_imagenet_dir, extra=your_extra_dir)
    dataset.dump_extra()
```


## Training robustness tokens
To train robustness tokens for a pre-trained DinoV2 model, run:

```bash
PYTHONPATH=src/ python src/train/robustness.py --config $path_to_file
```

An example of training configuration file can be found in [`configs/train/default.yaml`](configs/train/default.yaml). This will result in a checkpoint file containing the robustness tokens.

## Evaluating robustness tokens
You can evaluate the robustness of features extracted by models with or without robustness tokens to adversarial attacks.

```bash
PYTHONPATH=src/ python src/eval/robustness.py --config $path_to_file
```

An example of training configuration file can be found in [`configs/eval/default.yaml`](configs/eval/default.yaml).

## Evaluation on downstream tasks
For evaluation, we convert our checkpoints into a valid checkpoint for the DinoV2 codebase.

```bash
PYTHONPATH=src/ python src/eval/convert.py --checkpoint $path_to_file --output $path_to_file
```

The robustness tokens are converted into DinoV2 *register* tokens and appended before patch tokens. Refer to the [DinoV2](https://github.com/facebookresearch/dinov2) codebase for more details.

##  License
The code is released with the [MIT license](LICENSE).
