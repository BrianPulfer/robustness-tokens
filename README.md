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
IMAGENET_DIR=$path_to_imagenet_dataset
ADE20K_DIR=$path_to_ade20k_dataset
```

## Training
To train robustness tokens for a pre-trained DinoV2 model, run:

```bash
PYTHONPATH=src/ python src/train.py --config $path_to_file
```

An example of training configuration file can be found in [`configs/train/default.yaml`](configs/train/default.yaml).

## Evaluation

### Classification

```bash
# Evaluate classification performance
PYTHONPATH=src/ python src/eval/classification.py --config $path_to_file
```


### Segmentation
To evaluate segmentation capabilities, first install [MMSegmentation and MMCV](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation):

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

then download the [ADE20k dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/). Finally run:

```bash
# Evaluate segmentation performance
PYTHONPATH=src/ python src/eval/segmentation.py --config $path_to_file
```

### Depth estimation

(⚠️ **Warning**: this is still work in progress and won't currently work ⚠️)

```bash
# Evaluate depth estimation performance
PYTHONPATH=src/ python src/eval/depth.py --config $path_to_file
```

##  License
The code is released with the [MIT license](LICENSE).
