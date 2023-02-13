import neuralprophet
import sys
import optuna
import wget
import os
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
import pickle
import tensorflow as tf
import tensorboard as tb
import time
import matplotlib.pyplot as plt
import pyunpack
import glob
import gc
import sklearn.preprocessing
from neuralprophet import NeuralProphet, set_log_level
from pathlib import Path
from pyunpack import Archive
from torch import nn
from datetime import timedelta
from dataloading_helpers import electricity_dataloader, retail_dataloader
from config import *

# create dataset for 
path = CONFIG_DICT["datasets"]["electricity"] / "LD2011_2014.txt"

df = pd.read_csv(path, index_col=0, sep=';', decimal=',')
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)

# Used to determine the start and end dates of a series
output = df.resample('1h').mean().replace(0., np.nan)

earliest_time = output.index.min()


df_list = []
for label in output:
    srs = output[label]

    start_date = min(srs.fillna(method='ffill').dropna().index)
    end_date = max(srs.fillna(method='bfill').dropna().index)

    active_range = (srs.index >= start_date) & (srs.index <= end_date)
    srs = srs[active_range].fillna(0.)

    tmp = pd.DataFrame({'y': srs})
    date = tmp.index
    #tmp['time_idx'] = (date - earliest_time).seconds / 60 / 60 + (date - earliest_time).days * 24
    #tmp['days_from_start'] = (date - earliest_time).days
    #tmp['categorical_id'] = label
    #tmp['date'] = date
    tmp['ID'] = label
    tmp['hour'] = date.hour
    tmp['day'] = date.day
    tmp['day_of_week'] = date.dayofweek
    tmp['month'] = date.month
    tmp["ds"] = tmp.index
    df_list.append(tmp)

output = pd.concat(df_list, axis=0, join='outer').reset_index(drop=True)
#output['categorical_id'] = output['id'].copy()
#output['hours_from_start'] = output['time_idx']
#output['categorical_day_of_week'] = output['day_of_week'].copy()
#output['categorical_hour'] = output['hour'].copy()
#output = output.rename(columns = {"id":"ID", "power_usage":"y"})

model = NeuralProphet(
    trend_global_local="global", 
    season_global_local="global", 
    #n_lags=24,  #autoregressor on last 24h
    n_lags=7*24,
    n_forecasts=24,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=Truet,
    growth="off",
    learning_rate=0.01,
    loss_func="MSE",
   
    )

output_small = output[0:10000]
#output_small = output_small.rename(columns = {"ids":"ID"})

output_small['ID'] = output_small['ID'].astype('string').astype('category')


columnsToKeep = ['ds','y', 'ID']
regressorsList = ['hour', 'day', 'day_of_week', 'month']

# now the cool one liner..
# this removes all columns not 'ds','y' or any regressors
df_regressors_only = output_small[output_small.columns.intersection(np.concatenate((columnsToKeep, regressorsList)))]

model = model.add_lagged_regressor(names=regressorsList)

df_train, df_test = m.split_df(df_regressors_only, freq='auto', valid_p=0.10, local_split=False)

metrics = model.fit(df_train, freq='auto', validation_df=df_test, progress="plot")