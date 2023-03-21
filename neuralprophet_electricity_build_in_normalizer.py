if __name__ == '__main__': 
  print("Importing modules...")
  import pandas as pd
  import matplotlib.pyplot as plt
  import time
  from neuralprophet import NeuralProphet, set_log_level, set_random_seed, save
  from config import *
  import contextlib

  set_log_level("ERROR")
  print("Defining functions.")
  
  
  def get_model():
      np_model = NeuralProphet(
          growth = "linear",                    # no trend
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
          normalize="standardize",
      )

      np_model = np_model.highlight_nth_step_ahead_of_each_forecast(step_number = np_model.n_forecasts)

      np_model = np_model.add_lagged_regressor(names = lagged_regressors)   # , only_last_value=True)

      return np_model
  
  
  def split_train_test(df, np_model, num_id=0, valid_p=0.2):
      '''
      to ran only on part of data (for first # id) :   specify parameter num_id, e.g. num_id=5 (for first 5 ids)
      '''
      if num_id==0:
          df = df
      else:
          df = df[df.ID.isin(df.ID.unique()[:num_id])]

      df_train, df_test = np_model.split_df( 
          df,    
          freq='H',
          valid_p = valid_p,         
          local_split = True
      )

      return df_train, df_test
    
    
  
  print("Loading dataset and doing train-test split.")
  csv_file = CONFIG_DICT["datasets"]["electricity"] / "LD2011_2014.csv"
  electricity = pd.read_csv(csv_file, index_col=0)

  electricity['date'] =  pd.to_datetime(electricity['date'], format='%Y-%m-%d %H:%M:%S.%f')
  electricity.rename(columns={"power_usage": "y", "date": "ds", "id": "ID"}, inplace = True)

  test_boundary=1339
  index = electricity['days_from_start']
  train = electricity.loc[(index >= 1100) & (index < test_boundary)]
  test = electricity.loc[(index >= test_boundary -1)]


  # specify input variables
  input_columns = ["ID", "y","ds"]                    # index + target + datetime

  future_regressors = []
  lagged_regressors = ['hour', 'day', 'day_of_week', 'month'] 
  events = [] 

  train = train[input_columns + lagged_regressors]    # with regressors
  test = test[input_columns + lagged_regressors] 


  # loading and fitting model
  print("Loading and fitting model. Warning: No dependable print-outs during training.")
  np_model = get_model()

  df_train, df_val = split_train_test(train, np_model, num_id=0) #num_id=0 -> all ids
  
  metrics = np_model.fit(
          df = df_train, 
          validation_df = df_val,
          freq='H', 
          progress="print",
          num_workers = 40,
          early_stopping=False,
          learning_rate=0.1,
          epochs=15,
          batch_size=64
        )                      

  
  print("Training done. Saving model.")
  metrics.to_csv(CONFIG_DICT["models"]["electricity"] / "neuralprophet" / "np_metrics_real.csv")

  print("Predicting on test dataset.")

  predictions = np_model.predict(test)
  
  print("Saving predictions to file: np_preds_real.csv")
  
  predictions.to_csv(CONFIG_DICT["models"]["electricity"] / "neuralprophet" / "np_preds_real.csv")

    
  