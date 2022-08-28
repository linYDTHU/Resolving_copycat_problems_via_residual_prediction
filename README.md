# Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction

This is the code of the paper [Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction](https://arxiv.org/abs/2207.09705). You can use this repo to reproduce the results of our method in CARLA.

## Environment

### Requirements

- Hardware: A computer with a dedicated GPU capable of running Unreal Engine.
- OS: Ubuntu also compatible with CARLA

### Installation

To run the code, we provide a conda environment requirements file. Start by cloning the requirement on the same folder and then, to install, just run:

```
conda env create -f requirements.yaml
```
### Setting Environment/ Getting Data
The first thing you need to do is define the datasets folder.
This is the folder that will contain your training and validation datasets

    export COIL_DATASET_PATH=<Path to where your dataset folders are>

Download a sample dataset pack, with one training and two validations, by running

```
python3 tools/get_sample_datasets.py
```

The datasets (CoILTrain, CoILVal1 and CoILVal2) will be stored at the COIL_DATASET_PATH folder.

The dataset used in our paper is CARLA100, which can be downloaded from the original [github repo](https://github.com/felipecode/coiltraine/blob/master/docs/exploring_limitations.md) of Felipe Codevilla. Download the .zip files and extract the dataset into $COIL_DATASET_PATH/CoilTrain100.

To collect other datasets please check the data collector repository. https://github.com/carla-simulator/data-collector

## How to use our code?

1. Create a folder containing the configurations of model in .yaml (can refer to configs/action_residual_prediction)
2. Use main.py to train and evaluate
```
python3 main.py --folder action_residual_prediction --gpus 0 1 2 -de NocrashTrainingDense_Town01 --docker carlasim/carla:0.8.4
```

3. The training checkpoints would be saved in _logs

4. The driving results would be saved in _benchmarks_results, and you can print detailed results by using tools/print_metrics.py

```
python3 tools/print_metrics.py --path=<Path to where your results folders are> 
```

## Citations

Please consider citing our paper in your publications if it helps. Here is the bibtex:

```
@misc{https://doi.org/10.48550/arxiv.2207.09705,
  doi = {10.48550/ARXIV.2207.09705},
  
  url = {https://arxiv.org/abs/2207.09705},
  
  author = {Chuang, Chia-Chi and Yang, Donglin and Wen, Chuan and Gao, Yang},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

