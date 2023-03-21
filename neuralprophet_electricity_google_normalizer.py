if __name__ == '__main__': 
  print("Importing modules...")
  import pandas as pd
  import matplotlib.pyplot as plt
  import time
  from neuralprophet import NeuralProphet, set_log_level, set_random_seed, save
  from config import *
  from dataloading_helpers import electricity_dataloader

  set_log_level("ERROR")
  print("Defining functions.")
  
  # we did not find the correct way for changing output_dict, thus lightning_logs are in the main folder
  # run this for using googles normalization
  # reshape data to needed input
  train, val, test = electricity_dataloader.create_electricity_timeseries_np()

  train['date'] =  pd.to_datetime(train['date'], format='%Y-%m-%d %H:%M:%S.%f')
  train.rename(columns={"power_usage": "y", "date": "ds", "id": "ID"}, inplace = True)

  test['date'] =  pd.to_datetime(test['date'], format='%Y-%m-%d %H:%M:%S.%f')
  test.rename(columns={"power_usage": "y", "date": "ds", "id": "ID"}, inplace = True)
  #test = test[test["ds"] >= "2014-09-01"]
  test = test[test["ds"] >= "2014-08-31"] #use this for 7 days lookback


  val['date'] =  pd.to_datetime(val['date'], format='%Y-%m-%d %H:%M:%S.%f')
  val.rename(columns={"power_usage": "y", "date": "ds", "id": "ID"}, inplace = True)

  # specify input variables
  input_columns = ["ID", "y","ds"]                                  # index + target + datetime

  future_regressors = []
  lagged_regressors = ['hour', 'day', 'day_of_week', 'month'] 
  events = [] 

 
  df_train = train[input_columns + lagged_regressors]    # with regressors
  test = test[input_columns + lagged_regressors] 
  df_val = val[input_columns + lagged_regressors] 

  # uncomment to use timeseries up till num_id
  #num_id = 1
  #df_train = df_train[df_train.ID.isin(df_train.ID.unique()[:num_id])]
  #test = test[test.ID.isin(test.ID.unique()[:num_id])]
  #df_val = df_val[df_val.ID.isin(df_val.ID.unique()[:num_id])]


  np_model = NeuralProphet(
          growth = "off",                    # no trend
          trend_global_local = "global",
          season_global_local = "global",                
          n_lags = 7*24,                      # autoregressor on last 24h x 7 days
          n_forecasts = 24,                   # forecast horizon
          yearly_seasonality = True,
          weekly_seasonality = True,
          daily_seasonality = True,
          learning_rate = 0.05,
          loss_func = "MSE",
          quantiles = [0.1, 0.5, 0.9],
          normalize="off"
      )

  np_model = np_model.highlight_nth_step_ahead_of_each_forecast(step_number = np_model.n_forecasts)
  np_model = np_model.add_lagged_regressor(names = lagged_regressors, normalize="off") 

    
  metrics = np_model.fit(
          df = df_train, 
          validation_df = df_val,
          freq='H', 
          progress="print",
          num_workers = 40,
          early_stopping=False,
          learning_rate=0.05,
          epochs=30,
          batch_size=64
        )                      

  
  print("Training done. Saving metrics.")
  metrics.to_csv(CONFIG_DICT["models"]["electricity"] / "neuralprophet" / "np_metrics_week.csv")

  print("Predicting on test dataset.")

  predictions = np_model.predict(test)
    
  predictions.to_csv(CONFIG_DICT["models"]["electricity"] / "neuralprophet" / "np_preds_week.csv")
  
  predictions_decomposed = np_model.predict(test, decompose=True)
  
  predictions.to_csv(CONFIG_DICT["models"]["electricity"] / "neuralprophet" / "np_preds_week_decomposed.csv")
