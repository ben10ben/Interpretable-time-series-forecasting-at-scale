import pandas as pd
import numpy as np
from pathlib import Path
from config import *
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, TorchNormalizer, EncoderNormalizer
from sklearn.preprocessing import StandardScaler

# set path in config.py
txt_file = CONFIG_DICT["datasets"]["electricity"] / "LD2011_2014.txt"
csv_file = CONFIG_DICT["datasets"]["electricity"] / "LD2011_2014.csv"


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
    #output['categorical_day_of_week'] = output['day_of_week'].copy()
    #output['categorical_hour'] = output['hour'].copy()

    # Filter to match range used by other academic papers
    output = output[(output['days_from_start'] >= 1096)
                    & (output['days_from_start'] < 1346)].copy()

    output.to_csv(csv_file)
    return output


def create_electricity_timeseries_tft():
    """  
    Google input definitions:
    
      #('id', DataTypes.REAL_VALUED, InputTypes.ID),
      #('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
      #('power_usage', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      #('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      #('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),

    """
    
    try:
        electricity_data = pd.read_csv(csv_file, index_col=0)    
    except:
        electricity_data = prep_electricity_data(txt_file)


    electricity_data['time_idx'] = electricity_data['time_idx'].astype('int')

    #electricity_data['categorical_day_of_week'] = electricity_data['categorical_day_of_week'].astype('string').astype('category')
    #electricity_data['categorical_hour'] = electricity_data['categorical_hour'].astype('string').astype('category')
    electricity_data['categorical_id'] = electricity_data['categorical_id'].astype('category')
    #electricity_data['month'] = electricity_data['month'].astype('string').astype('category')
  
    max_prediction_length = 24
    max_encoder_length = 168
    training_cutoff = electricity_data["time_idx"].max() - max_prediction_length

    # NOT USED
    # split data like google
    #index = df['days_from_start']
    #train = df.loc[index < valid_boundary]
    #valid = df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
    #test = df.loc[index >= test_boundary - 7]
    print(electricity_data["power_usage"].mean())
    #target_normalizer = TorchNormalizer()
    #electricity_data["power_usage"] = target_normalizer.fit_transform(electricity_data["power_usage"])
    #print(electricity_data["power_usage"].mean())
    #    static = ["categorical_id"]
    #numeric = ["power_usage", "hour", "day_of_week", "hours_from_start"]
    #categorical = ["categorical_id"]
    
    #StandardScaler.fit_transform(electricity_data, electricity_data["power_usage"])
    #print(electricity_data["power_usage"].mean())
    
    training = TimeSeriesDataSet(
      electricity_data[lambda x: x.time_idx <= training_cutoff],
      time_idx="time_idx",
      target="power_usage",
      group_ids=["id"],
      min_encoder_length=max_encoder_length,# // 2,  # keep encoder length long (as it is in the validation set)
      max_encoder_length=max_encoder_length,
      min_prediction_length=max_prediction_length,
      max_prediction_length=max_prediction_length,
      static_categoricals=["categorical_id"],
      static_reals=[],
      time_varying_known_categoricals=[],
      #variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
      time_varying_known_reals=["time_idx", "hour", "day_of_week"],
      time_varying_unknown_categoricals=[],
      time_varying_unknown_reals=[],
      target_normalizer=EncoderNormalizer(),#(groups=["power_usage", "hour", "day_of_week", "time_idx"]),
      #target_normalizer=GroupNormalizer(groups=["categorical_id"])
      add_relative_time_idx=False,
      add_target_scales=True,
      add_encoder_length=False, #
    )
    
    
    """
    old timeseries dataset definition:
    
    
    training = TimeSeriesDataSet(
      electricity_data[lambda x: x.time_idx <= training_cutoff],
      time_idx="time_idx",
      target="power_usage",
      group_ids=["categorical_id", "id"],
      min_encoder_length=max_encoder_length,# // 2,  # keep encoder length long (as it is in the validation set)
      max_encoder_length=max_encoder_length,
      min_prediction_length=max_prediction_length,
      max_prediction_length=max_prediction_length,
      static_categoricals=["categorical_id", "id"],
      static_reals=[],
      time_varying_known_categoricals=["categorical_day_of_week", "categorical_hour", "month"],
      #variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
      time_varying_known_reals=["time_idx", "hours_from_start", "days_from_start"],
      time_varying_unknown_categoricals=[],
      time_varying_unknown_reals=["power_usage"],
      target_normalizer=GroupNormalizer(
          groups=["categorical_id", "id"], transformation="softplus"
      ),  # use softplus and normalize by group
      add_relative_time_idx=True,
      add_target_scales=False,
      add_encoder_length=False, #
    )
    """

  # create validation set (predict=True) which means to predict the last max_prediction_length points in time
  # for each series
    validation = TimeSeriesDataSet.from_dataset(training, electricity_data, predict=True, stop_randomization=True)

  # create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=45, pin_memory=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 20, num_workers=30, pin_memory=True)


# output data as dict for easier modularity
    return {"training_dataset": training, 
          "train_dataloader": train_dataloader,
          "val_dataloader": val_dataloader, 
          "validaton_dataset": validation}



def create_electricity_timeseries_nhits():

    try:
        electricity_data = pd.read_csv(csv_file, index_col=0)    
    except:
        electricity_data = prep_electricity_data(txt_file)


    electricity_data['time_idx'] = electricity_data['time_idx'].astype('int')

    electricity_data['categorical_day_of_week'] = electricity_data['categorical_day_of_week'].astype('string').astype('category')
    electricity_data['categorical_hour'] = electricity_data['categorical_hour'].astype('string').astype('category')
    electricity_data['categorical_id'] = electricity_data['categorical_id'].astype('category')
    electricity_data['month'] = electricity_data['month'].astype('string').astype('category')
  
    max_prediction_length = 24
    max_encoder_length = 168
    training_cutoff = electricity_data["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
      electricity_data[lambda x: x.time_idx <= training_cutoff],
      time_idx="time_idx",
      target="power_usage",
      group_ids=["categorical_id", "id"],
      min_encoder_length=max_encoder_length,# // 2,  # keep encoder length long (as it is in the validation set)
      max_encoder_length=max_encoder_length,
      min_prediction_length=max_prediction_length,
      max_prediction_length=max_prediction_length,
      static_categoricals=["categorical_id", "id"],
      static_reals=[],
      time_varying_known_categoricals=["categorical_day_of_week", "categorical_hour", "month"],
      #variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
      time_varying_known_reals=["time_idx", "hours_from_start", "days_from_start"],
      time_varying_unknown_categoricals=[],
      time_varying_unknown_reals=["power_usage"],
      target_normalizer=GroupNormalizer(
          groups=["categorical_id", "id"], transformation="softplus"
      ),  # use softplus and normalize by group
      add_relative_time_idx=False,
      add_target_scales=False,
      add_encoder_length=False, #
  )

  # create validation set (predict=True) which means to predict the last max_prediction_length points in time
  # for each series
    validation = TimeSeriesDataSet.from_dataset(training, electricity_data, predict=True, stop_randomization=True)

  # create dataloaders for model
    batch_size =128  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=14)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=14)


# output data as dict for easier modularity
    return {"training_dataset": training, 
          "train_dataloader": train_dataloader,
          "val_dataloader": val_dataloader, 
          "validaton_dataset": validation}






def create_electricity_timeseries_deepar():

    try:
        electricity_data = pd.read_csv(csv_file, index_col=0)    
    except:
        electricity_data = prep_electricity_data(txt_file)

    # create dataset and dataloaders
    max_encoder_length = 168
    max_prediction_length = 24

    training_cutoff = data["time_idx"].max() - max_prediction_length

    context_length = max_encoder_length
    prediction_length = max_prediction_length

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="power_usage",
        categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
        group_ids=["series"],
        static_categoricals=[
            "series"
        ],  # as we plan to forecast correlations, it is important to use series characteristics (e.g. a series identifier)
        time_varying_unknown_reals=["value"],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
    )

    validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)
    batch_size = 128
    # synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=20, batch_sampler="synchronized")
    
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=10, batch_sampler="synchronized")
    
    # output data as dict for easier modularity
    return {"training_dataset": training, 
          "train_dataloader": train_dataloader,
          "val_dataloader": val_dataloader, 
          "validaton_dataset": validation}

