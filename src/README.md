## Introduction
The following directory contains my implementation for my master thesis.

## Overview
The directory contains the following directories: 
* [./data/](data/): contains scripts for downloading the BRACE-dataset.
* [./features/](features/): contains scripts for preprocessing the downloaded BRACE-dataset.
* [./visualization/](visualization): contains scripts for visualizations.

## Guide 
To generate the used data, do the following
1. Run [./data/make_dataset.py](./data/make_dataset.py) to download the data. 
2. Run [./features/build_features.py](features/build_features.py) to prepare/preprocess the data.