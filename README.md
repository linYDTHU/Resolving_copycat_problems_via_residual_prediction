# Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction

## Environment

#### Requirements

- Hardware: A computer with a dedicated GPU capable of running Unreal Engine.
- OS: Ubuntu also compatible with CARLA (16.04)

#### Installation

To run the code, we provide a conda environment requirements file. Start by cloning the requirement on some folder and then, to install, just run:

```
conda env create -f requirements.yaml
```
#### Setting Environment/ Getting Data
The first thing you need to do is define the datasets folder.
This is the folder that will contain your training and validation datasets

    export COIL_DATASET_PATH=<Path to where your dataset folders are>

To collect datasets please check the data collector repository https://github.com/carla-simulator/data-collector.

## Data Collection

The data collection can refer to https://github.com/carla-simulator/data-collector

## How to use our code?

1. Create a folder containing the configurations of model in .yaml (can refer to configs/action_residual_prediction)
2. Use main.py to train and evaluate
```
  python3 main.py --folder action_residual_prediction --gpus 0 1 2 -de NocrashTrainingDense_Town01
```
3. The training checkpoint would be saved in _logs
4. The driving results would be saved in _benchmarks_results
