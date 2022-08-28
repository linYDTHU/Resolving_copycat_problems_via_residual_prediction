# Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction

## Environment

The environment setup can refer to https://github.com/felipecode/coiltraine

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