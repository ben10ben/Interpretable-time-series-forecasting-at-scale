import pandas as pd
import numpy as np
from pathlib import Path
from config import *
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, TorchNormalizer, EncoderNormalizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dataloading_helpers import electricity_formatter 
# set path in config.py
txt_file = CONFIG_DICT["datasets"]["electricity"] / "LD2011_2014.txt"
csv_file = CONFIG_DICT["datasets"]["electricity"] / "LD2011_2014.csv"
csv_file_normalized = CONFIG_DICT["datasets"]["electricity"] / "LD2011_2014_normalized.csv"



"""
prep_electricity_data function copied from google paper:
https://github.com/google-research/google-research/blob/master/tft/script_download_data.py

args:
  -txt_file: path to .txt document containg raw electricity dataset
      
  -output_path: path to save/load prepared csv to/from

output: electricity_dataset_dict
  -training dataset
  -training dataloader
  -validation dataloader
  -validation dataset

"""

def prep_electricity_data(txt_file):
    df = pd.read_csv(txt_file, index_col=0, sep=';', decimal=',')
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

      tmp = pd.DataFrame({'power_usage': srs})
      date = tmp.index
      tmp['time_idx'] = (date - earliest_time).seconds / 60 / 60 + (
          date - earliest_time).days * 24
      tmp['days_from_start'] = (date - earliest_time).days
      tmp['categorical_id'] = label
      tmp['date'] = date
      tmp['id'] = label
      tmp['hour'] = date.hour
      tmp['day'] = date.day
      tmp['day_of_week'] = date.dayofweek
      tmp['month'] = date.month

      df_list.append(tmp)

    output = pd.concat(df_list, axis=0, join='outer').reset_index(drop=True)

    output['categorical_id'] = output['id'].copy()
    output['hours_from_start'] = output['time_idx']
  
    # Filter to match range used by other academic papers
    output = output[(output['days_from_start'] >= 1096) & (output['days_from_start'] < 1346)].copy()

    output.to_csv(csv_file)
    return output


def create_electricity_timeseries_tft():

    try:
        electricity = pd.read_csv(csv_file, index_col=0)    
    except:
        electricity = prep_electricity_data(txt_file)

    electricity['time_idx'] = electricity['time_idx'].astype('int')

    standardizer = electricity_formatter.ElectricityFormatter()
    train, test, validation = standardizer.split_data(df=electricity)

    train["categorical_id"] = train['categorical_id'].astype('string').astype("category")
    test["categorical_id"] = test['categorical_id'].astype('string').astype("category")
    validation["categorical_id"] = validation['categorical_id'].astype('string').astype("category") 
        
    max_prediction_length = 24
    max_encoder_length = 168
    
    training = TimeSeriesDataSet(
      train,
      time_idx="time_idx",
      target="power_usage",
      group_ids=["id"],
      min_encoder_length=max_encoder_length,
      max_encoder_length=max_encoder_length,
      min_prediction_length=max_prediction_length,
      max_prediction_length=max_prediction_length,
      static_categoricals=["categorical_id"],
      static_reals=[],
      time_varying_known_categoricals=[],
      time_varying_known_reals=["time_idx", "hour", "day_of_week"],
      time_varying_unknown_categoricals=[],
      time_varying_unknown_reals=[],
      target_normalizer=None,
      categorical_encoders=[],
      add_relative_time_idx=False,
      add_target_scales=False,
      add_encoder_length=False,
    )

    model_parameters = training.get_parameters()

    testing = TimeSeriesDataSet.from_parameters(parameters=model_parameters, data=test, predict=True, stop_randomization=True)
    validating = TimeSeriesDataSet.from_parameters(parameters=model_parameters, data=validation, predict=True, stop_randomization=True)
    

    # create dataloaders for model
    batch_size = 64
    
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=10, pin_memory=True)
    test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size, num_workers=10, pin_memory=True)
    val_dataloader = validating.to_dataloader(train=False, batch_size=batch_size, num_workers=10, pin_memory=True)       
  
    # output data as dict for easier modularity
    return {"training_dataset": training, 
          "train_dataloader": train_dataloader,
          "val_dataloader": val_dataloader, 
          "validation_dataset": validating,
          "test_dataset": testing,
          "test_dataloader": test_dataloader,
           }
  
  
def create_electricity_timeseries_np():
    try:
        electricity = pd.read_csv(csv_file, index_col=0)    
    except:
        electricity = prep_electricity_data(txt_file)

    electricity['time_idx'] = electricity['time_idx'].astype('int')

    standardizer = electricity_formatter.ElectricityFormatter()
    train, test, validation = standardizer.split_data(df=electricity)

    train["categorical_id"] = train['categorical_id'].astype('string').astype("category")
    test["categorical_id"] = test['categorical_id'].astype('string').astype("category")
    validation["categorical_id"] = validation['categorical_id'].astype('string').astype("category") 
    
    return train, test, validation