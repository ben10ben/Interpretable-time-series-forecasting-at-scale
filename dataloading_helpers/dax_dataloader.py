import pandas as pd
import numpy as np
from pathlib import Path
from config import *
from pytorch_forecasting import TimeSeriesDataSet

# set path in config.py
csv_file = CONFIG_DICT["datasets"]["stocks"] / "DAX.csv"


def load_and_cast_stocks_df():
    stocks_df = pd.read_csv(csv_file)
    stocks_df.fillna(method='ffill', inplace=True)
    stocks_df["Date"] = pd.to_datetime(stocks_df["Date"])
    stocks_df.sort_values(by="Date", inplace=True)
    stocks_df["time_idx"] = stocks_df.index
    #stocks_df.set_index("time_idx", inplace = True)

    stocks_df.rename(columns={"Adj Close":"adj_close"}, inplace=True)

    stocks_df['day'] = stocks_df.Date.dt.day
    stocks_df['day_of_week'] = stocks_df.Date.dt.dayofweek
    stocks_df['month'] = stocks_df.Date.dt.month

    stocks_df['categorical_day_of_week'] = stocks_df.day_of_week.astype('string').astype("category")
    stocks_df['categorical_day'] = stocks_df['day'].astype('string').astype("category")
    stocks_df['categorical_month'] = stocks_df['month'].astype('string').astype("category")
    stocks_df["group_id"] = 1

    return stocks_df


def create_stocks_timeseries():
    
    stocks_df = load_and_cast_stocks_df()
    
    max_prediction_length = 24
    max_encoder_length = 168
    training_cutoff = stocks_df["time_idx"].max() - max_prediction_length
    
    training = TimeSeriesDataSet(
      stocks_df[lambda x: x.time_idx <= training_cutoff],
      time_idx="time_idx",
      target="High",
      group_ids=["group_id"],
      min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
      max_encoder_length=max_encoder_length,
      min_prediction_length=1,
      max_prediction_length=max_prediction_length,
      static_categoricals=[],
      static_reals=[],
      time_varying_known_categoricals=["categorical_day_of_week", "categorical_day", "categorical_month"],
      time_varying_known_reals=["time_idx", "day_of_week", "day", "month"],
      time_varying_unknown_categoricals=[],
      time_varying_unknown_reals=["Open", "High", "Low", "Close", "adj_close"],
      target_normalizer=None,
      add_relative_time_idx=True,
      add_target_scales=True,
      add_encoder_length=False, 
      )

  # create validation set (predict=True) which means to predict the last max_prediction_length points in time
  # for each series
    validation = TimeSeriesDataSet.from_dataset(training, stocks_df, predict=True, stop_randomization=True)

  # create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)


# output data as dict for easier modularity
    return {"training_dataset": training, 
          "train_dataloader": train_dataloader,
          "val_dataloader": val_dataloader, 
          "validaton_dataset": validation}