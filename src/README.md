## Introduction
The following directory contains my implementation for my master thesis.

## Overview
The directory contains the following directories: 
* [./data/](data/): contains scripts for downloading the pretraining-datasets.
* [./features/](features/): contains scripts for preprocessing the downloaded pretraining-datasets.
* [./models/](models/): contains scripts for training and evaluating the various models.
* [./visualization/](visualization): contains scripts for various visualizations.

## Guide 
To generate the used data, do the following
1. Run [./data/make_dataset.py](./data/make_dataset.py) to download the data. 
2. Run [./features/build_features.py](features/build_features.py) to prepare/preprocess the data.

To pretrain the models run the following four scripts
1. [./models/train_baseline.py](./models/train_baseline.py) to train 3DConv
2. [./models/train_deciwatch.py](./models/train_deciwatch.py) to train DeciWatch
3. [./models/train_unipose.py](./models/train_unipose.py) to train bi-ConvLSTM Model S
4. [./models/train_unipose2.py](./models/train_unipose2.py) to train bi-ConvLSTM Model C

To finetune the pretrained models run [./models/finetune_models.py](./models/finetune_models.py).