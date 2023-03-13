if __name__ == '__main__': 
  print("Importing modules...")

  
  
  import torch  
  import pytorch_lightning as pl
  import tensorboard as tb
  import json
  import time
  import warnings
  import pandas as pd
  from torch import nn
  from pytorch_lightning.accelerators import *
  from torch.utils.tensorboard import SummaryWriter
  from lightning.pytorch.accelerators import find_usable_cuda_devices
  from pytorch_lightning.loggers import TensorBoardLogger
  from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
  from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
  from pytorch_forecasting.data.encoders import GroupNormalizer, TorchNormalizer, NaNLabelEncoder
  from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
  from torch.optim import Adam
  from torch.optim.lr_scheduler import ReduceLROnPlateau
  from dataloading_helpers import electricity_dataloader
  from config import *
  from pathlib import Path
  from sklearn.preprocessing import StandardScaler, LabelEncoder



  csv_file = CONFIG_DICT["datasets"]["electricity"] / "LD2011_2014.csv"
  electricity = pd.read_csv(csv_file)

  config_name_string = "electricity"
  model_dir = CONFIG_DICT["models"][config_name_string]


  electricity['time_idx'] = electricity['time_idx'].astype('int')
  electricity["categorical_id"] = electricity['categorical_id'].astype('string').astype("category")



  max_prediction_length = 24
  max_encoder_length = 168

  valid_boundary=1315
  test_boundary=1339

  index = electricity['days_from_start']
  train = electricity.loc[index < valid_boundary]
  valid = electricity.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
  test = electricity.loc[index >= test_boundary - 7]

  training_cutoff = electricity["time_idx"].max() - max_prediction_length

  training = TimeSeriesDataSet(
      train[lambda x: x.time_idx <= training_cutoff],
      time_idx="time_idx",
      target="power_usage",
      group_ids=["id"],
      min_encoder_length=max_encoder_length,
      max_encoder_length=max_encoder_length,
      min_prediction_length=max_prediction_length,
      max_prediction_length=max_prediction_length,
      static_categoricals=["categorical_id"],
      static_reals=[],
      time_varying_known_categoricals=[],
      time_varying_known_reals=["time_idx", "hour", "day_of_week"],
      time_varying_unknown_categoricals=[],
      time_varying_unknown_reals=["power_usage"],
      #target_normalizer=TorchNormalizer(method="standard").fit(train["power_usage"]),
      target_normalizer=GroupNormalizer(groups=["id"]),
      scalers = {"time_idx" : StandardScaler(), "hour" : StandardScaler(), "day_of_week": StandardScaler()},
      categorical_encoders= {'__group_id__id': NaNLabelEncoder(add_nan=False, warn=True), 'categorical_id': NaNLabelEncoder(add_nan=False, warn=True)},
      add_relative_time_idx=False,
      add_target_scales=True,
      add_encoder_length=False,
  #    lags= {"power_usage": [(24)]}
  )

  # get parameters from train dataset to create val/test
  model_parameters = training.get_parameters()

  testing = TimeSeriesDataSet.from_parameters(parameters=model_parameters, data=test, predict=True, stop_randomization=True)
  validating = TimeSeriesDataSet.from_parameters(parameters=model_parameters, data=valid, predict=True,stop_randomization=True)


  # create dataloaders for model
  batch_size = 64

  train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=10, pin_memory=True)
  test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size, num_workers=10, pin_memory=True)
  val_dataloader = validating.to_dataloader(train=False, batch_size=batch_size, num_workers=10, pin_memory=True)       



  if torch.cuda.is_available():
      accelerator = "gpu"
      devices = find_usable_cuda_devices(1)
  else:
      accelerator = "cpu"
      devices = None

  checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="val_loss", mode="min",
            dirpath=CONFIG_DICT["models"]["electricity"] / "checkpoint_callback_logs",
            filename="sample-mnist-{epoch:02d}-{val_loss:.2f}")

  writer = SummaryWriter(log_dir = CONFIG_DICT["models"]["electricity"] / "writer_logs" )
  early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=False, mode="min")
  lr_logger = LearningRateMonitor(logging_interval='epoch') 
  logger = TensorBoardLogger(CONFIG_DICT["models"]["electricity"]) 

    # best parameters estimated by hypertuning and manually rounded
  hyper_dict = {
                  'gradient_clip_val': 0.052, 
                  'hidden_size': 128, 
                  'dropout': 0.15, 
                  'hidden_continuous_size': 32, 
                  'attention_head_size': 2, 
                  'learning_rate': 0.007,
               }

  # uncomment to read hyperparamters from hyper-tuning script
  #hyper_dict = pd.read_pickle(CONFIG_DICT["models"]["electricity"] / "tuning_logs" / "hypertuning_electricity.pkl")

  trainer = pl.Trainer(
        default_root_dir=model_dir,
        max_epochs=10,
        devices=devices,
        accelerator=accelerator,
        enable_model_summary=True,
        gradient_clip_val=hyper_dict["gradient_clip_val"],
        fast_dev_run=False,  
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        log_every_n_steps=1,
        logger=logger,
        profiler="simple",
      )

  print("Definining TFT...")

  tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=hyper_dict["learning_rate"],
        hidden_size=hyper_dict["hidden_size"],
        attention_head_size=hyper_dict["attention_head_size"],
        dropout=hyper_dict["dropout"],
        hidden_continuous_size=hyper_dict["hidden_continuous_size"],
        output_size= 3,
        loss=QuantileLoss([0.1, 0.5, 0.9]),
        log_interval=1,
        reduce_on_plateau_patience=4,
        optimizer="adam"
      )

  trainer.optimizer = Adam(tft.parameters(), lr=hyper_dict["learning_rate"])
  scheduler = ReduceLROnPlateau(trainer.optimizer, factor=0.2)  

  print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
  print("Training model...")

    # fit network
  trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        #ckpt="~/RT1_TFT/models/electricity/lightning_logs/version_28/checkpoints/"
  )

    # safe model for later use
  torch.save(tft.state_dict(), CONFIG_DICT["models"]["electricity"] / "tft_model")

  print("trainging done. Evaluating...")

  output = trainer.test(model=tft, dataloaders=test_dataloader , ckpt_path="best")

  with open(CONFIG_DICT["models"]["electricity"] / "tuning_logs" / "tft_electricity_test_output_internal_scaling.pkl", "wb") as fout:
      pickle.dump(output, fout)

  print("Done.")