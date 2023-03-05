# RT1_TFT

Repo for colaborative work of Benedikt Rein, Yulia and Alican


This repository provides the code, nessesary for the linked paper.

We are trying to reproduce the performance of the TFT model, shown in the Google paper.

Once a MAE is reached that is less than a magnitude higher than this of the Google paper, we will start implementing other statistical or ML models, once those perform in a comparable magnitude we will implement a third dataset.


Run the setup.sh scrit to:
  -create needed directory structure
  -download the needed datasets
  -setup a conda enviroment
  -run a script for each dataset
  -compare the performance of the models
  

electricity dataset:
https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip

retail dataset: https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/7391/44328/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1676046400&Signature=sErgEXSRVP6IRWbBu3Mmu68le1PSV2gv2lRR9Ys55tfQaxPVG4Bty0fHBIrTyhh95nZWyqxR%2Fo1pr13Y5jPum2zpB3QhgCDlW1HYLEcSTSxvpj%2FeAvhSTHMv%2ByvcFaA3sPeu1WSX7S6Y9lFQtM2%2BGeA9GCI%2Bf3lPXbghpGXRqfvhVJS5%2BGgUzIuq1GPwUAiFDmDOEPiWMxCbPavFSFfBILXbgU1PmjWsFcW9EMQNGMeATjg5tgw%2FrFUpiSCl3kquaUhzJJoJLwBnPzG2taAo%2BX8Fqm0tBBkHuZFxcFC29PXCxY4vhseA6wHbxiZI%2BTVVymqbFDv53k3%2BWgDkW%2Fe81g%3D%3D&response-content-disposition=attachment%3B+filename%3Dfavorita-grocery-sales-forecasting.zip







prelimenary code layout:

run setup.py to create venv, folderstructure, download datasets

use notebooks for EDA, loading trained models for validation run and visualization / explainability

use model_dataset_hypertuning.py to find optimal hyperparameters for the model-dataset combination

use model_dataset.py to make full training run - best model can be loaded into notebook and used exploratory