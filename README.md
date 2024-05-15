# Blackbox Adapter for Prompted Segmentation (BAPS)
This repository contains the code for "Blackbox Adaptation for Medical Image Segmentation". This work has been accepted at MICCAI 2024****

## Environment File
Create a new conda environment with the config file given in the repository as follows:
```
conda env create --file=baps.yaml
conda activate baps
```

## General file descriptions
- data_transforms/*.py - data transforms defined here for different datasets.
- datasets/*.py - dataset class defined here.
- data_utils.py - functions to load different datasets
- modelling - model architectures for encoders, IP decoder and blackbox foundation models defined here
- trainer.py - code for general training, common to all datasets
- eval/generate_predictions.py - code for generating results for a trained model
- configs/default.yml - config file for defining various settings

## Link to model checkpoints
Coming Soon

## Data Structure
The data structure for different datasets is defined by datasets/< dataset_name.py >. In general, the structure is as follows:
```
root
|--train
    |--images
    |--masks
|--val
    |--images
    |--masks
|--test
    |--images
    |--masks
```
There are minor differences which vary from dataset to dataset. Please refer to the files in datasets folder.

For training on Custom Datastes: Create a dataset class and add it to the datasets folder. Add an additional 'if' block in data_utils.py

## Example Usage for Training
```
python main.py --config configs/default.yml --save_path "<path to save model>>" --device cuda:0
```
To resume training from a pretrained checkpoint, use the pretrained_path flag as follows:
```
python main.py --config configs/default.yml --pretrained_path "<path to model checkpoint>" --save_path "<path to save model>>" --device cuda:0
```
## Example Usage for Evaluation
```
cd eval

python generate_predictions.py -config configs/default.yml --pretrained_path <path to model checkpoint> --save_path "<folder to save the results>" --device cuda:0
```

## Citation
```
To be added
```
