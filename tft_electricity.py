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
  from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, DeviceStatsMonitor
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

  
  print(timeseries_dict)
  
  print("Checking for device...")

  if torch.cuda.is_available():
      accelerator = "gpu"
      devices = find_usable_cuda_devices(1)
  else:
      accelerator = None
      devices = 'cpu'

  print("Training on ", accelerator, "on device: ", devices, ". \nDefining Trainer...") 

  checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="val_loss", mode="min",
          dirpath=CONFIG_DICT["models"]["electricity"] / "checkpoint_callback_logs",
          filename="sample-mnist-{epoch:02d}-{val_loss:.2f}")
  
  writer = SummaryWriter(log_dir = CONFIG_DICT["models"]["electricity"] / "logs" )
  early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
  lr_logger = LearningRateMonitor(logging_interval='epoch') 
  logger = TensorBoardLogger(CONFIG_DICT["models"]["electricity"]) 

  trainer = pl.Trainer(
      default_root_dir=model_dir,
      max_epochs=10,
      devices=devices,
      accelerator=accelerator,
      enable_model_summary=True,
      gradient_clip_val=0.01,
      fast_dev_run=False,  
      callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
      log_every_n_steps=1,
      logger=logger,
      profiler="simple",
    )

  print("Definining TFT...")
  
  warnings.filterwarnings("error") # supress UserWarning
  
  tft = TemporalFusionTransformer.from_dataset(
      timeseries_dict["training_dataset"],
      learning_rate=0.001,
      hidden_size=160,
      attention_head_size=4,
      dropout=0.1,
      hidden_continuous_size=80,
      output_size= 3,
      loss=QuantileLoss([0.1, 0.5, 0.9]),
      log_interval=1,
      reduce_on_plateau_patience=4
      optimizer="adam"
    )

  warnings.resetwarnings()
  trainer.optimizer = Adam(tft.parameters(), lr=0.001)
  scheduler = ReduceLROnPlateau(trainer.optimizer, factor=0.2)  
  
  print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
  print("Training model")
  
  # fit network
  trainer.fit(
      tft,
      train_dataloaders=timeseries_dict["train_dataloader"],
      val_dataloaders=timeseries_dict["test_dataloader"],
      #ckpt="~/RT1_TFT/models/electricity/lightning_logs/version_28/checkpoints/"
  )

        
  print("trainging done. Evaluating...")

  
  ## evaluate
  best_model_path = trainer.checkpoint_callback.best_model_path
  best_tft = tft.load_from_checkpoint(best_model_path)
  actuals = torch.cat([y[0] for x, y in iter(timeseries_dict["val_dataloader"])])
  predictions = best_tft.predict(timeseries_dict["val_dataloader"])
  print("Best model MAE: ",(actuals - predictions).abs().mean().item())



  output_dict = {
                'model_path': best_model_path,
                'MAE'       : (actuals - predictions).abs().mean().item(),
                'device'    : devices,
                'dataset'   : "electricity",
                }

  with open('output.txt', 'w') as convert_file:
       convert_file.write(json.dumps(output_dict))

  print("Done.")
    