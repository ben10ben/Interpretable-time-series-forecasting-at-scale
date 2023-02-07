if __name__ == '__main__': 
  print("Importing modules...")

  import torch
  import pytorch_lightning as pl
  import tensorboard as tb
  import matplotlib.pyplot as plt
  import json
  import gc
  from pytorch_lightning.accelerators import *
  from torch.utils.tensorboard import SummaryWriter
  from torch import nn
  from lightning.pytorch.accelerators import find_usable_cuda_devices
  from pytorch_lightning.loggers import TensorBoardLogger
  from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
  from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
  from pytorch_forecasting.data.encoders import GroupNormalizer
  from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, DeviceStatsMonitor
  from dataloading_helpers import retail_dataloader
  from config import *

  print("Preparing dataset...") 
  retail = retail_dataloader.create_retail_timeseries()
  timeseries_dict =  retail
  config_name_string = "retail"
  parameters = []