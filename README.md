# Introduction
This repository contains my master thesis *Temporal Smoothing in 2D Human Pose EStimation for Bouldering*.

## Abstract
In this thesis we implement four architectures for extending an already developed keypoint detector for bouldering. The three architectures consist of (1) a single 3-dimensional convolutional layer followed by the ReLU activation function, (2) DeciWatch by Zeng Et al., and (3) two kinds of bidirectional convolutional LSTMs inspired by Unipise-LSTM by Artacho and Savakis, where the difference between the two architectures lies in how they combine the two processing directions. The models are pretrained on the BRACE dataset and parts of the Penn Action dataset, and further finetuned on a dataset for bouldering. The keypoint detector and the finetuning dataset are both provided by ClimbAlong at NorthTech ApS. We perform various experiments to find the optimal setting of the four models. Finally, we conclude, that DeciWatch by Zeng Et al. yields the most accurate results, one of the bidirectional convolutional LSTMs yields the best rough estimations, and the simple 3-dimensional convolutional layer yields the best results when also considering in the size and prediction time of the models.

## Overview
The repository contains the following directories:
* [./data/](data/): empty directory of where the data will be stored
* [./finetuned_models/](finetuned_models/): empty directory of where the finetuned models will be stored
* [./other/](other/): directory of various notes I have noted through the execution of the project
* [./presentation/](presentation/): the presentation used by me during the oral examination
* [./pretrained_models/](pretrained_models/): empty directory of where the pretrained modelss will be stored
* [./report/](report/): directory of the thesis
* [./src/](src/): directory of the implementation
