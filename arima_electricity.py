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

  print("Loading dataset.")
  train_data, test_data, val_data = electricity_dataloader.create_electricity_timeseries_np()

  predictions_path = CONFIG_DICT["models"]["electricity"] / "arima" / "global_preds_arima.csv"

  try:
      global_preds_arima = pd.read_csv(predictions_path) 
  except:
      print("No predictions available.")
      global_preds_arima = pd.DataFrame(columns=["id_nr", "forecasts", "test_data_id_predict_y", "global_confidence_upper", "global_confidence_lower"])


  # iterate over all ids
  for id_string in test_data.categorical_id.unique():
      start_time = time.time()

      # skip over ids that have predictions present, id=15 throws "LinAlgError: LU decomposition error."
      if int(id_string) in global_preds_arima["id_nr"].values or id_string=="15":
          continue

      # select one id, define x/y train dataframe and fit model
      train_data_id = train_data[train_data["categorical_id"] == id_string]
      train_data_id = train_data_id[int(len(train_data_id) * 0.8) :]
      train_data_id_y = train_data_id["power_usage"]
      train_data_id_x = train_data[["categorical_hour", "time_idx", "categorical_day_of_week"]].copy()

      model = Arima.auto_arima(train_data_id_y, exogenous=train_data_id_x, stepwise=True, seasonal=True, m=24, maxiter=5)


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
      forecasts, confidence = model.predict(test_data_id_predict_y.shape[0], return_conf_int=True, exogenous=test_data_id_predict_x, alpha=0.1)

      end_time = time.time()
      print(f"Mean absolute error for id {id_string} is: {(mean_absolute_error(test_data_id_predict_y, forecasts)):.3f}. This iteration took: {((end_time-start_time)/60):.2f} minutes.")

      id_list = [id_string] * 24

      # join relevent values to dataframe
      output = pd.DataFrame({
                          'id_nr': id_list,
                          'forecasts': forecasts,
                          'test_data_id_predict_y': test_data_id_predict_y,
                          'global_confidence_upper': confidence[:,1],
                          'global_confidence_lower': confidence[:,0]
                          })

      # join to present predictions
      global_preds_arima = pd.concat([global_preds_arima, output], axis=0)

      global_preds_arima.to_csv(predictions_path, index=False)

  print(f"Mean absolute error: {mean_absolute_error(global_preds_arima.forecasts , global_preds_arima.test_data_id_predict_y)}")