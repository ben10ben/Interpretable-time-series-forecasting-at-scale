# Evaluation of selected DL algorithmic methods for interpretable time series forecasting
by Alican Gündogdu, Benedikt Rein and Yuliya Vandysheva

Humboldt-University of Berlin  
Chair of Information Systems  
Course: Information Systems  
Professor: Stephan Lessmann  

The main goal of this research is to reproduce results shown in the paper
[Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/pdf/1912.09363.pdf).

This repository provides code for replicating the experiments described in the paper, which you can find under:
Interpretable_timeseries_forecasting_Information_Science_Seminar.pdf


It was developed on Linux but should also run on other machines with minor changes.

Read the Electricity_EDA_Eval.ipynp alongside the paper.
In the notebook you can find some exploratory analysis of the electricity dataset and we present 
our best performing NeuralProphet and TemporalFusionTransformer models alongside our baseline.


Run `bash setup.sh` to:  
  -download the needed dataset  
  -setup a conda enviroment  
  
 We cannot guarantee, that virtual enviroment setup works on every machnine, if you prefer venv or the setup fails, use the `requirement.txt` so setup the enviroment.

  
  
With an activated virtual enviroment and `requirements.txt` installed run:

`python3 tft_electricity_hypertuning.py` to selecting optimal hyperparameters  

`python3 tft_electricity_google_normalizer.py` to run our TFT implementation with already tuned hyperparameters and copied normalization from the TFT paper  

`python3 tft_electricity_build_in_normalizer.py` to run our TFT implementation with already tuned hyperparameters and let the TFT module take care of normalization  

`python3 neuralprophet_electricity_build_in_normalizer.py` to run our NeuralProphet implementation without hyperparameter tuning and let the NeuralProphet module take care of normalization  

`python3 neuralprophet_electricity_google_normalizer.py` to run our NeuralProphet implementation without hyperparameter tuning and copied normalization from the TFT paper

`python3 arima_electricity.py` to create an ARIMA model for every local timeseries with copied normalization from the TFT paper and safe the predictions to a csv, models are not saved.  


[Download the electricity dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip).
