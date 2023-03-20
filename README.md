# Evaluation of selected DL algorithmic methods for interpretable time series forecasting

The main goal of this research is to reproduce results shown in the paper
[Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/pdf/1912.09363.pdf).

This repository provides code for replicating the experiments described in our paper. 

It was developed on Linux but should also run on other machines with minor changes.

Read the Electricity_EDA_Eval.ipynp alongside the paper.
In the notebook you can find some exploratory analysis of the electricity dataset and we present 
our best performing NeuralProphet and TemporalFusionTransformer models alongside our baseline.


Run `bash setup.sh` to:  
  -download the needed dataset  
  -setup a conda enviroment  

  
  
With an activated virtual enviroment run:

`python3 tft_electricity_hypertuning.py` to selecting optimal hyperparameters  

`python3 tft_electricity_google_normalizer.py` to run our TFT implementation with already tuned hyperparameters and copied normalization from the TFT paper  

`python3 tft_electricity_build_in_normalizer.py` to run our TFT implementation with already tuned hyperparameters and let the TFT module take care of normalization  

`python3 neuralprophet_electricity.py` to run our NeuralProphet implementation without hyperparameter tuning and let the NeuralProphet module take care of normalization  

`python3 arima_electricity.py` to create an ARIMA model for every local timeseries and safe the predictions to a csv, models are not saved.  


[Download the electricity dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip).
