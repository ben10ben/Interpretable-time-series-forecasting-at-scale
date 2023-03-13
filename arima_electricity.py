print("Importing modules.")

import time
import pandas as pd
from dataloading_helpers import electricity_dataloader
from config import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as Arima


if __name__ == '__main__': 
  # load data
  print("Loading dataset.")
  train_data, test_data, val_data = electricity_dataloader.create_electricity_timeseries_np()

  #test_ids = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
  global_preds = pd.Series()
  global_y = pd.Series()

  for id_string in train_data.categorical_id.unique():
      start_time = time.time()

      # select one id, define x/y train dataframe and fit model
      train_data_id = train_data[train_data["categorical_id"] == id_string]
      train_data_id_y = train_data_id["power_usage"]
      train_data_id_x = train_data[["categorical_hour", "time_idx", "categorical_day_of_week"]].copy()

      model = Arima.auto_arima(train_data_id_y, exogenous=train_data_id_x, stepwise=True, seasonal=True, m=24, maxiter=15)


      # refit model on test_data and predict on last 24 timesteps
      test_data_id = test_data[test_data["categorical_id"] == id_string]
      train_limit = (len(test_data_id)) - 24
      test_data_id_update, test_data_id_predict = train_test_split(test_data_id, train_size=train_limit)

      test_data_id_update_y = test_data_id_update["power_usage"]
      test_data_id_predict_y = test_data_id_predict["power_usage"]

      test_data_id_update_x = test_data_id_update[["categorical_hour", "time_idx", "categorical_day_of_week"]].copy()
      test_data_id_predict_x = test_data_id_predict[["categorical_hour", "time_idx", "categorical_day_of_week"]].copy()

      model.update(test_data_id_update_y, test_data_id_update_x)


      # make your forecasts
      forecasts, confidence = model.predict(test_data_id_predict_y.shape[0], return_conf_int=True, exogenous=test_data_id_predict_x)

      global_preds = np.concatenate((global_preds, forecasts), axis=None)
      global_y = np.concatenate((global_y, test_data_id_predict_y), axis=None)

      print(f"Mean absolute error for id {id_string} is: {mean_absolute_error(test_data_id_predict_y, forecasts)}")

      end_time = time.time()
      print("This iteration took: ", (end_time-start_time)/60 )

  print(f"Mean absolute error over all ids is: {mean_absolute_error(global_preds, global_y)}")