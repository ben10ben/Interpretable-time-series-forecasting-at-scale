"""
THIS SHOULD GO INTO SUBFOLDER "dataloading_helpers" AS A .py FILE

prep_electricity_data copied from google paper:
https://github.com/google-research/google-research/blob/master/tft/script_download_data.py

whole cell is for the electricity dataset

args:
  -txt_file: path to .txt document containg raw electricity dataset
      
  -output_path: path to save prepared csv to

output: electricity_dataset_dict
  -training dataset
  -training dataloader
  -validation dataloader
  -validation dataset

"""

def prep_electricity_data(txt_file, output_path):
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
    output['categorical_day_of_week'] = output['day_of_week'].copy()
    output['categorical_hour'] = output['hour'].copy()

    # Filter to match range used by other academic papers
    output = output[(output['days_from_start'] >= 1096)
                    & (output['days_from_start'] < 1346)].copy()

    output.to_csv(output_path)
    return output

def load_electricity_dataset(output_path):
  # define src and dst for electricity data prep
  electricity_dir = output_path + "/electricity/"
  output_file = output_path + "/electricity/LD2011_2014.csv"
  ziped_data = output_path + "/electricity/LD2011_2014.zip"
  txt_file = "/content/drive/MyDrive/data/electricity/LD2011_2014.txt"


  # read or prepare data
  if os.path.exists(output_file):
    return pd.read_csv(output_file, index_col=0)
  else:  
    electricity_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'
    print("Downloading file...")
    ziped_data = wget.download(electricity_url)
    print("Unpacking file...")  
    pyunpack.Archive(ziped_data).extractall(electricity_dir)
    print("preparing data...")
    return prep_electricity_data(txt_file, output_file)


def create_timeseries_electricity():
  electricity_data = load_electricity_dataset(output_path)
  electricity_data['time_idx'] = electricity_data['time_idx'].astype('int')

  electricity_data['categorical_day_of_week'] = electricity_data['categorical_day_of_week'].astype('string')
  electricity_data['categorical_day_of_week'] = electricity_data['categorical_day_of_week'].astype('category')

  electricity_data['categorical_hour'] = electricity_data['categorical_hour'].astype('string')
  electricity_data['categorical_hour'] = electricity_data['categorical_hour'].astype('category')

  electricity_data['categorical_id'] = electricity_data['categorical_id'].astype('category')

  electricity_data['month'] = electricity_data['month'].astype('string')
  electricity_data['month'] = electricity_data['month'].astype('category')
  
  max_prediction_length = 24
  max_encoder_length = 168
  training_cutoff = electricity_data["time_idx"].max() - max_prediction_length

  training = TimeSeriesDataSet(
      electricity_data[lambda x: x.time_idx <= training_cutoff],
      time_idx="time_idx",
      target="power_usage",
      group_ids=["categorical_id", "id"],
      min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
      max_encoder_length=max_encoder_length,
      min_prediction_length=1,
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
      add_target_scales=True,
      add_encoder_length=False, #
  )

  # create validation set (predict=True) which means to predict the last max_prediction_length points in time
  # for each series
  validation = TimeSeriesDataSet.from_dataset(training, electricity_data, predict=True, stop_randomization=True)

  # create dataloaders for model
  batch_size = 128  # set this between 32 to 128
  train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
  val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)


# output data as dict for easier modularity
  return {"training_dataset": training, 
          "train_dataloader": train_dataloader,
          "val_dataloader": val_dataloader, 
          "validaton_dataset": validation}