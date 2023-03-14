if __name__ == '__main__': 
  print("Importing modules...")

  import torch
  import pytorch_lightning as pl
  import tensorboard as tb
  import json
  import time
  import warnings
  from torch import nn
  from pytorch_lightning.accelerators import *
  from torch.utils.tensorboard import SummaryWriter
  from lightning.pytorch.accelerators import find_usable_cuda_devices
  from pytorch_lightning.loggers import TensorBoardLogger
  from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
  from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
  from pytorch_forecasting.data.encoders import GroupNormalizer
  from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
  from torch.optim import Adam
  from torch.optim.lr_scheduler import ReduceLROnPlateau
  from dataloading_helpers import electricity_dataloader
  from config import *

  print("Preparing dataset...") 
  
  electricity = electricity_dataloader.create_electricity_timeseries_tft()
  timeseries_dict =  electricity
  config_name_string = "electricity"
  parameters = []
  model_dir = CONFIG_DICT["models"][config_name_string]
  
  print("Checking for device...")

  if torch.cuda.is_available():
      accelerator = "gpu"
      devices = find_usable_cuda_devices(1)
  else:
      accelerator = None
      devices = 'cpu'

  checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="val_loss", mode="min",
          dirpath=CONFIG_DICT["models"]["electricity"] / "checkpoint_callback_logs",
          filename="sample-mnist-{epoch:02d}-{val_loss:.2f}")
  
  writer = SummaryWriter(log_dir = CONFIG_DICT["models"]["electricity"] / "writer_logs" )
  early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=False, mode="min")
  lr_logger = LearningRateMonitor(logging_interval='epoch') 
  logger = TensorBoardLogger(CONFIG_DICT["models"]["electricity"]) 
  
  # best parameters estimated by hypertuning and manually rounded
  # with those we achieved train_loss: 0.18, val_loss: 0.22 test_loss: 0.211, MAE/P50: 0.33
  hyper_dict = {
                'gradient_clip_val': 0.052, 
                'hidden_size': 128, 
                'dropout': 0.15, 
                'hidden_continuous_size': 32, 
                'attention_head_size': 2, 
                'learning_rate': 0.007,
               }
  
  # uncomment to read hyperparamters from hyper-tuning script
  #hyper_dict = pd.read_pickle(CONFIG_DICT["models"]["electricity"] / "tuning_logs" / "tft_hypertuning_electricity.pkl")
  
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
      timeseries_dict["training_dataset"],
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

  # connect optimizer with trainer
  trainer.optimizer = Adam(tft.parameters(), lr=hyper_dict["learning_rate"])
  scheduler = ReduceLROnPlateau(trainer.optimizer, factor=0.2)  
  
  print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
  print("Training model...")
  
  # fit network
  trainer.fit(
      tft,
      train_dataloaders=timeseries_dict["train_dataloader"],
      val_dataloaders=timeseries_dict["val_dataloader"],
      #ckpt_path=""
  )

  # safe model for later use
  torch.save(tft.state_dict(), CONFIG_DICT["models"]["electricity"] / "tft_model_google_normalizer")
  
  print("trainging done. Evaluating...")

  output = trainer.test(model=tft, dataloaders=electricity["test_dataloader"], ckpt_path="best")

  with open(CONFIG_DICT["models"]["electricity"] / "tuning_logs" / "tft_electricity_test_output_google_normalizer.pkl", "wb") as fout:
      pickle.dump(output, fout)

  print("Done.")
    