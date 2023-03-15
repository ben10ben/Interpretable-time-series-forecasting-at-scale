if __name__ == '__main__': 
  print("Importing modules...")
  import pandas as pd
  import matplotlib.pyplot as plt
  import time
  from neuralprophet import NeuralProphet, set_log_level
  from neuralprophet import set_random_seed
  from config import *
  import pickle
  from pytorch_lightning.loggers import TensorBoardLogger
  import pytorch_lightning as pl

  set_log_level("ERROR")
  print("Defining functions.")
  logger = TensorBoardLogger(CONFIG_DICT["models"]["electricity"] / "neuralprophet") 
  trainer = pl.Trainer(default_root_dir=CONFIG_DICT["models"]["electricity"] / "neuralprophet",
                       logger=logger
                      )
  
  
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
    
    
  def fit_model(np_model, df_train, df_val, num_epochs, batch_size, learning_rate, num_workers):
      start_time = time.perf_counter()

      metrics = np_model.fit(
          df = df_train, 
          validation_df = df_val,
          freq='H', 
          progress="print",
          num_workers = num_workers,
          #early_stopping=True,
          learning_rate=learning_rate,
          epochs=num_epochs,
          batch_size=batch_size
      )                                            
      end_time = time.perf_counter()
      total_time = end_time - start_time
      print(f'Training Took {total_time:.4f} seconds')

      return metrics, np_model 
  
  
  print("Loading dataset and doing train-test split.")
  csv_file = CONFIG_DICT["datasets"]["electricity"] / "LD2011_2014.csv"
  electricity = pd.read_csv(csv_file, index_col=0)

  electricity['date'] =  pd.to_datetime(electricity['date'], format='%Y-%m-%d %H:%M:%S.%f')
  electricity.rename(columns={"power_usage": "y", "date": "ds", "id": "ID"}, inplace = True)

  test_boundary=1339
  index = electricity['days_from_start']

  train = electricity.loc[(index >= 1100) & (index < test_boundary)]
  test = electricity.loc[index >= test_boundary]


  # specify input variables
  input_columns = ["ID", "y","ds"]                    # index + target + datetime

  future_regressors = []
  lagged_regressors = ['hour', 'day', 'day_of_week', 'month'] 
  events = [] 

  train = train[input_columns + lagged_regressors]    # with regressors
  test = test[input_columns + lagged_regressors] 


  # loading and fitting model
  print("Loading and fitting model.")
  np_model = get_model()

  df_train, df_val = split_train_test(train, np_model, num_id=0) #num_id=0 -> all ids

  metrics, np_model = fit_model(np_model, df_train=df_train, df_val=df_val, num_epochs=20, batch_size=64, learning_rate=0.05, num_workers=30)

  #save model for later use
  with open(CONFIG_DICT["models"]["electricity"] / "neuralprophet" / "neuralprophet_model.pkl", "wb") as f:
      pickle.dump(np_model, f)

  # create a future data frame consisting of the time steps into the future that we need to forecast
  future = np_model.make_future_dataframe(test, n_historic_predictions=True)

  forecast = np_model.predict(future)
  
  with open(CONFIG_DICT["models"]["electricity"] / "neuralprophet" / "neuralprophet_predictions.pkl", "wb") as f:
    pickle.dump(forecast, f)
  
  
  print("Trainign metrics: ", metrics)
  print(forecast)