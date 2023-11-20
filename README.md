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

To be able to train and test segmentation and depth estimation models, install [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation):

```bash
pip install -U openmim
mim install mmengine
mim install mmcv
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

### Dotenv file
Create a `.env` file in the root directory of the project with the following content:

```bash
PYTHONPATH=src
IMAGENET_DIR=$path_to_imagenet_dataset
ADE20K_DIR=$path_to_ade20k_dataset
NYUD_DIR=$path_to_nyud_dataset
```

## Training
To train robustness tokens for a pre-trained DinoV2 model, run:

```bash
PYTHONPATH=src/ python src/train/robustness.py --config $path_to_file
```

An example of training configuration file can be found in [`configs/train/robustness.yaml`](configs/train/robustness.yaml).
Downstream task models can be obtained starting from the original model (without robustness tokens), with

## Evaluation

### Classification

```bash
# Evaluate classification performance
PYTHONPATH=src/ python src/eval/classification.py --config $path_to_file
```


### Segmentation
First download the [ADE20k dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/). Then run:

```bash
# Evaluate segmentation performance
PYTHONPATH=src/ python src/eval/segmentation.py --config $path_to_file
```

### Depth estimation

(⚠️ **Warning**: this is still work in progress and won't currently work ⚠️)

Get the [nyu dataset](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#nyu) and convert it using the script under `mmsegmentation` :

```bash
python tools/dataset_converters/nyu.py nyu.zip
```

then, run:

```bash
# Evaluate depth estimation performance
PYTHONPATH=src/ python src/eval/depth.py --config $path_to_file
```

##  License
The code is released with the [MIT license](LICENSE).
