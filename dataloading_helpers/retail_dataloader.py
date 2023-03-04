import pandas as pd
import numpy as np
from pathlib import Path
from config import *
from pytorch_forecasting import TimeSeriesDataSet
from dataloading_helpers import google_helpers
import sklearn.preprocessing
import os
import pyunpack
import glob
import wget
import gc

csv_file = CONFIG_DICT["datasets"]["retail"] / "retail.csv"


DataTypes = google_helpers.DataTypes
InputTypes = google_helpers.InputTypes


class FavoritaFormatter(google_helpers.GenericDataFormatter):
    """Defines and formats data for the Favorita dataset.
    Attributes:
        column_definition: Defines input and data type of column used in the
        experiment.
    identifiers: Entity identifiers used in experiments.
    """

    _column_definition = [
        ('traj_id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('date', DataTypes.DATE, InputTypes.TIME),
        ('log_sales', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('onpromotion', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('transactions', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('oil', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('day_of_month', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('month', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('national_hol', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('regional_hol', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('local_hol', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('open', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('item_nbr', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('store_nbr', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('city', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('state', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('type', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('cluster', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('family', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('class', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('perishable', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
    ]

    def __init__(self):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None

    def split_data(self, df, valid_boundary=None, test_boundary=None):
        """Splits data frame into training-validation-test data frames.
        This also calibrates scaling object, and transforms data for each split.
        Args:
          df: Source data frame to split.
          valid_boundary: Starting year for validation data
          test_boundary: Starting year for test data
        Returns:
          Tuple of transformed (train, valid, test) data.
        """

        print('Formatting train-valid-test splits.')

        if valid_boundary is None:
            valid_boundary = pd.datetime(2015, 12, 1)

        fixed_params = self.get_fixed_params()
        time_steps = fixed_params['total_time_steps']
        lookback = fixed_params['num_encoder_steps']
        forecast_horizon = time_steps - lookback

        df['date'] = pd.to_datetime(df['date'])
        df_lists = {'train': [], 'valid': [], 'test': []}
        for _, sliced in df.groupby('traj_id'):
            index = sliced['date']
            train = sliced.loc[index < valid_boundary]
            train_len = len(train)
            valid_len = train_len + forecast_horizon
            valid = sliced.iloc[train_len - lookback:valid_len, :]
            test = sliced.iloc[valid_len - lookback:valid_len + forecast_horizon, :]

            sliced_map = {'train': train, 'valid': valid, 'test': test}

            for k in sliced_map:
                item = sliced_map[k]

                if len(item) >= time_steps:
                    df_lists[k].append(item)

        dfs = {k: pd.concat(df_lists[k], axis=0) for k in df_lists}

        train = dfs['train']
        self.set_scalers(train, set_real=True)

        # Use all data for label encoding  to handle labels not present in training.
        self.set_scalers(df, set_real=False)

        # Filter out identifiers not present in training (i.e. cold-started items).
    def filter_ids(frame):
        identifiers = set(self.identifiers)
        index = frame['traj_id']
        return frame.loc[index.apply(lambda x: x in identifiers)]

        valid = filter_ids(dfs['valid'])
        test = filter_ids(dfs['test'])

        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df, set_real=True):
        """Calibrates scalers using the data supplied.
        Label encoding is applied to the entire dataset (i.e. including test),
        so that unseen labels can be handled at run-time.
        Args:
          df: Data to use to calibrate scalers.
          set_real: Whether to fit set real-valued or categorical scalers
        """
        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        if set_real:
            # Extract identifiers in case required
            self.identifiers = list(df[id_column].unique())

            # Format real scalers
            self._real_scalers = {}
            for col in ['oil', 'transactions', 'log_sales']:
                self._real_scalers[col] = (df[col].mean(), df[col].std())

            self._target_scaler = (df[target_column].mean(), df[target_column].std())

        else:
            # Format categorical scalers
            categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

            categorical_scalers = {}
            num_classes = []
            if self.identifiers is None:
                raise ValueError('Scale real-valued inputs first!')
            id_set = set(self.identifiers)
            valid_idx = df['traj_id'].apply(lambda x: x in id_set)
            for col in categorical_inputs:
                # Set all to str so that we don't have mixed integer/string columns
                srs = df[col].apply(str).loc[valid_idx]
                categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs.values)

            num_classes.append(srs.nunique())

            # Set categorical scaler outputs
            self._cat_scalers = categorical_scalers
            self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        """Performs feature transformations.
        This includes both feature engineering, preprocessing and normalisation.
        Args:
          df: Data frame to transform.
        Returns:
          Transformed data frame.
        """
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        column_definitions = self.get_column_definition()

        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Format real inputs
        for col in ['log_sales', 'oil', 'transactions']:
            mean, std = self._real_scalers[col]
            output[col] = (df[col] - mean) / std

            if col == 'log_sales':
                output[col] = output[col].fillna(0.)  # mean imputation

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output
  
  
def download_from_url(url, output_path):
  """Downloads a file froma url."""

  print('Pulling data from {} to {}'.format(url, output_path))
  wget.download(url, output_path)
  print('done')


def recreate_folder(path):
  """Deletes and recreates folder."""

  shutil.rmtree(path)
  os.makedirs(path)


def unzip(zip_path, output_file, data_folder):
  """Unzips files and checks successful completion."""

  print('Unzipping file: {}'.format(zip_path))
  pyunpack.Archive(zip_path).extractall(data_folder)

  # Checks if unzip was successful
  if not os.path.exists(output_file):
    raise ValueError(
        'Error in unzipping process! {} not found.'.format(output_file))


def download_and_unzip(url, zip_path, csv_path, data_folder):
  """Downloads and unzips an online csv file.
  Args:
    url: Web address
    zip_path: Path to download zip file
    csv_path: Expected path to csv file
    data_folder: Folder in which data is stored.
  """

  download_from_url(url, zip_path)

  unzip(zip_path, csv_path, data_folder)

  print('Done.')
  

def process_favorita():
    """Processes Favorita dataset.
    Makes use of the raw files should be manually downloaded from Kaggle @
    https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data
    Args:
      config: Default experiment config for Favorita
    """

    url = 'https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data'
    data_folder = CONFIG_DICT['datasets']['retail']
 
    # Save manual download to root folder to avoid deleting when re-processing.
    zip_file = os.path.join(data_folder, 'favorita-grocery-sales-forecasting.zip')

    if not os.path.exists(zip_file):
        raise ValueError(
            'Favorita zip file not found in {}!'.format(zip_file) +
            ' Please manually download data from Kaggle @ {}'.format(url))

    # Unpack main zip file
    outputs_file = os.path.join(data_folder, 'train.csv.7z')
    unzip(zip_file, outputs_file, data_folder)

    # Unpack individually zipped files
    for file in glob.glob(os.path.join(data_folder, '*.7z')):
        csv_file = file.replace('.7z', '')
        unzip(file, csv_file, data_folder)

    print('Unzipping complete, commencing data processing...')

    # Extract only a subset of data to save/process for efficiency
    start_date = pd.datetime(2015, 1, 1)
    end_date = pd.datetime(2016, 6, 1)

    print('Regenerating data...')

    # load temporal data
    temporal = pd.read_csv(os.path.join(data_folder, 'train.csv'), index_col=0)

    store_info = pd.read_csv(os.path.join(data_folder, 'stores.csv'), index_col=0)
    oil = pd.read_csv(os.path.join(data_folder, 'oil.csv'), index_col=0).iloc[:, 0]
    holidays = pd.read_csv(os.path.join(data_folder, 'holidays_events.csv'))
    items = pd.read_csv(os.path.join(data_folder, 'items.csv'), index_col=0)
    transactions = pd.read_csv(os.path.join(data_folder, 'transactions.csv'))

    # Take first 6 months of data
    temporal['date'] = pd.to_datetime(temporal['date'])

    # Filter dates to reduce storage space requirements
    if start_date is not None:
        temporal = temporal[(temporal['date'] >= start_date)]
    if end_date is not None:
        temporal = temporal[(temporal['date'] < end_date)]

    dates = temporal['date'].unique()

    # Add trajectory identifier
    temporal['traj_id'] = temporal['store_nbr'].apply(str) + '_' + temporal['item_nbr'].apply(str)
    temporal['unique_id'] = temporal['traj_id'] + '_' + temporal['date'].apply(str)

    # Remove all IDs with negative returns
    print('Removing returns data')
    min_returns = temporal['unit_sales'].groupby(temporal['traj_id']).min()
    valid_ids = set(min_returns[min_returns >= 0].index)
    selector = temporal['traj_id'].apply(lambda traj_id: traj_id in valid_ids)
    new_temporal = temporal[selector].copy()
    del temporal
    gc.collect()
    temporal = new_temporal
    temporal['open'] = 1

    # Resampling
    print('Resampling to regular grid')
    resampled_dfs = []
    for traj_id, raw_sub_df in temporal.groupby('traj_id'):
       # print('Resampling', traj_id)
        sub_df = raw_sub_df.set_index('date', drop=True).copy()
        sub_df = sub_df.resample('1d').last()
        sub_df['date'] = sub_df.index
        sub_df[['store_nbr', 'item_nbr', 'onpromotion']] = sub_df[['store_nbr', 'item_nbr', 'onpromotion']].fillna(method='ffill')
        sub_df['open'] = sub_df['open'].fillna(0)  # flag where sales data is unknown
        sub_df['log_sales'] = np.log(sub_df['unit_sales'])
        resampled_dfs.append(sub_df.reset_index(drop=True))

    new_temporal = pd.concat(resampled_dfs, axis=0)
    del temporal
    gc.collect()
    temporal = new_temporal

    print('Adding oil')
    oil.name = 'oil'
    oil.index = pd.to_datetime(oil.index)
    
    #fill oil dates and add nan, was handled differently in older pandas version
    idx = pd.date_range('01-01-2013', '08-30-2017')
    oil = oil.reindex(idx, fill_value="NaN")
    
    temporal = temporal.join(oil.loc[dates].fillna(method='ffill'), on='date', how='left')
    temporal['oil'] = temporal['oil'].fillna(-1)

    
    print('Adding store info')
    temporal = temporal.join(store_info, on='store_nbr', how='left')

    print('Adding item info')
    temporal = temporal.join(items, on='item_nbr', how='left')

    transactions['date'] = pd.to_datetime(transactions['date'])
    temporal = temporal.merge(
      transactions,
      left_on=['date', 'store_nbr'],
      right_on=['date', 'store_nbr'],
      how='left')
    temporal['transactions'] = temporal['transactions'].fillna(-1)

    # Additional date info
    temporal['day_of_week'] = pd.to_datetime(temporal['date'].values).dayofweek
    temporal['day_of_month'] = pd.to_datetime(temporal['date'].values).day
    temporal['month'] = pd.to_datetime(temporal['date'].values).month

    # Add holiday info
    print('Adding holidays')
    holiday_subset = holidays[holidays['transferred'].apply(lambda x: not x)].copy()
    holiday_subset.columns = [s if s != 'type' else 'holiday_type' for s in holiday_subset.columns]
    holiday_subset['date'] = pd.to_datetime(holiday_subset['date'])
    local_holidays = holiday_subset[holiday_subset['locale'] == 'Local']
    regional_holidays = holiday_subset[holiday_subset['locale'] == 'Regional']
    national_holidays = holiday_subset[holiday_subset['locale'] == 'National']

    temporal['national_hol'] = temporal.merge(national_holidays, left_on=['date'], right_on=['date'], how='left')['description'].fillna('')
    temporal['regional_hol'] = temporal.merge(regional_holidays, left_on=['state', 'date'], right_on=['locale_name', 'date'], how='left')['description'].fillna('')
    temporal['local_hol'] = temporal.merge(local_holidays, left_on=['city', 'date'], right_on=['locale_name', 'date'], how='left')['description'].fillna('')

    temporal.sort_values('unique_id', inplace=True)

    print('Saving processed file to {}'.format(CONFIG_DICT['datasets']['retail']))
    temporal.to_csv(CONFIG_DICT['datasets']['retail'] / "retail.csv")
    
    
 
    
def create_retail_timeseries_tft():
    try:
        retail_data = pd.read_csv(csv_file, index_col=0)
        retail_data_small = retail_data[retail_data["store_nbr"] < 3]
        retail_data_small.to_csv(CONFIG_DICT['datasets']['retail'] / "retail_small.csv")

        #retail_data = pd.read_csv(CONFIG_DICT["datasets"]["retail"] / "retail_small.csv", index_col=0)   
    except FileNotFoundError:
        process_favorita()
        retail_data = pd.read_csv(csv_file, index_col=0)    
        retail_small = retail_data[:999999]
        retail_small.to_csv(CONFIG_DICT['datasets']['retail'] / "retail_small.csv")
    
    return retail_data_small
    
