if __name__ == '__main__': 
  print("Importing modules...")
  

  import pandas as pd
  import matplotlib.pyplot as plt
  import time

  from neuralprophet import NeuralProphet, set_log_level
  from neuralprophet import set_random_seed
  from config import *
  import pickle

  csv_file = CONFIG_DICT["datasets"]["electricity"] / "LD2011_2014.csv"
  electricity = pd.read_csv(csv_file, index_col=0)

  electricity['date'] =  pd.to_datetime(electricity['date'], format='%Y-%m-%d %H:%M:%S.%f')
  electricity.rename(columns={"power_usage": "y", "date": "ds", "id": "ID"}, inplace = True)

  test_boundary=1339
  index = electricity['days_from_start']

  train = electricity.loc[(index >= 1250) & (index < test_boundary)]
  test = electricity.loc[index >= test_boundary]



  # specify input variables
  input_columns = ["ID", "y","ds"]                                  # index + target + datetime

  future_regressors = []
  lagged_regressors = ['hour', 'day', 'day_of_week', 'month'] 
  events = [] 

  train = train[input_columns + lagged_regressors]    # with regressors
  test = test[input_columns + lagged_regressors] 



  def get_model():
      m = NeuralProphet(
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
          quantiles = [0.1, 0.5, 0.9]   
      )

      m = m.highlight_nth_step_ahead_of_each_forecast(step_number = m.n_forecasts)

      m = m.add_lagged_regressor(names = lagged_regressors)   # , only_last_value=True)

      return m


  m = get_model()



  def split_train_test(df, model, num_id=0, valid_p=0.2):
      '''
      to ran only on part of data (for first # id) :   specify parameter num_id, e.g. num_id=5 (for first 5 ids)
      '''
      if num_id==0:
          df = df
      else:
          df = df[df.ID.isin(df.ID.unique()[:num_id])]

      df_train, df_test = model.split_df( 
          df,    
          freq='H',
          valid_p = valid_p,         
          local_split = True
      )

      return df_train, df_test

  df_train, df_val = split_train_test(train, m, num_id=0)


  def fit_model(m, df_train, df_val, num_epochs, batch_size, learning_rate, num_workers):
      start_time = time.perf_counter()

      metrics = m.fit(
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

      return metrics   


  metrics = fit_model(m, df_train=df_train, df_val=df_val, num_epochs=10, batch_size=128, learning_rate=0.05, num_workers=15)


  with open('models/electricity/neuralprophet_model.pkl', "wb") as f:
      # dump information to that file
      pickle.dump(metrics, f)


  # create a future data frame consisting of the time steps into the future that we need to forecast
  future = m.make_future_dataframe(
                                  test,                          
                                  n_historic_predictions = True
                                  )

  forecast = m.predict(future)