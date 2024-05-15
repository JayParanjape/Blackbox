# Blackbox Adapter for Prompted Segmentation (BAPS)
This repository contains the code for "Blackbox Adaptation for Medical Image Segmentation". This work has been accepted at MICCAI 2024****

## Abstract
In recent years, various large foundation models have been proposed for image segmentation. There models are often trained on large amounts of data corresponding to general computer vision taks.  Hence, these models do not perform well on medical data. There have been some attempts in the literature to perform parameter-efficient finetuning of such foundation models for medical image segmentation. However, these approaches assume that all the parameters of the model are available for adaptation. But, in many cases, these models are released as APIs or blackboxes, with no or limited access to the model parameters and data. In addition, finetuning methods also require a significant amount of compute, which may not be available for the downstream task. At the same time, medical data can't be shared with third-party agents for finetuning due to privacy reasons. To tackle these challenges, we pioneer a blackbox adaptation technique for prompted medical image segmentation, called BAPS. BAPS has two components - (i) An Image-Prompt decoder (IP decoder) module that generates visual prompts given an image and a prompt, and (ii) A Zero Order Optimization (ZOO) Method, called SPSA-GC that is used to update the IP decoder without the need for backpropagating through the foundation model. Thus, our method does not require any knowledge about the foundation model's weights or gradients. We test BAPS on four different modalities and show that our method can improve the original model's performance by around 4%.

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
