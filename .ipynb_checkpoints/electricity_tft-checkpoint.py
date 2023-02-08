"""
to find out:

-is every epoch different data?
-1 gpu = 4 its/sec
-3 gpu = 2 its/sec

bottelneck or thats how it is?

"""

if __name__ == '__main__': 
  print("Importing modules...")

  import torch
  import pytorch_lightning as pl
  import tensorboard as tb
  import json
  import time
  from torch import nn
  from pytorch_lightning.accelerators import *
  from torch.utils.tensorboard import SummaryWriter
  from lightning.pytorch.accelerators import find_usable_cuda_devices
  from pytorch_lightning.loggers import TensorBoardLogger
  from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
  from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
  from pytorch_forecasting.data.encoders import GroupNormalizer
  from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, DeviceStatsMonitor

  from pytorch_lightning.strategies.ddp import DDPStrategy  
  from torch.nn.parallel import DistributedDataParallel as DDP
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
      devices = find_usable_cuda_devices(2)
  else:
      accelerator = None
      devices = 'cpu'

  print("Training mode ", accelerator, "on device: ", devices, ". \nDefining Trainer...") 

  writer = SummaryWriter(log_dir = CONFIG_DICT["models"]["electricity"] / "logs" )
  early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
  lr_logger = LearningRateMonitor() 
  logger = TensorBoardLogger(CONFIG_DICT["models"]["electricity"]) 
  DeviceStatsMonitor = DeviceStatsMonitor()

  trainer = pl.Trainer(
      default_root_dir=model_dir,
      max_epochs=20,
      devices=devices,
      accelerator=accelerator,
      enable_model_summary=False,
      gradient_clip_val=0.01,
      limit_train_batches=0.1, 
      fast_dev_run=False,  
      callbacks=[lr_logger, early_stop_callback, DeviceStatsMonitor],
      log_every_n_steps=5,
      logger=logger,
      profiler="simple",
      #strategy= DDPStrategy(find_unused_parameters=False),
      strategy="ddp",
    )

  print("Definining TFT...")
  tft = TemporalFusionTransformer.from_dataset(
      timeseries_dict["training_dataset"],
      learning_rate=0.01,
      hidden_size=160,
      attention_head_size=4,
      dropout=0.1,
      hidden_continuous_size=80,
      output_size= 3,  # 7 quantiles by default
      loss=QuantileLoss([0.1, 0.5, 0.9]),
      log_interval=40,
      reduce_on_plateau_patience=4,
  )
  print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

  print("Training model")
  # fit network
  trainer.fit(
      tft,
      train_dataloaders=timeseries_dict["train_dataloader"],
      val_dataloaders=timeseries_dict["val_dataloader"],
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
                'dataset'   : "electricity"
                }

  with open('output.txt', 'w') as convert_file:
       convert_file.write(json.dumps(output_dict))

  print("Done.")