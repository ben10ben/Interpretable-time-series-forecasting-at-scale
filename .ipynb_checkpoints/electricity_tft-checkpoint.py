print("Importing modules...")

import torch
import pytorch_lightning as pl
#import tensorflow as tf
import tensorboard as tb
import matplotlib.pyplot as plt
import json
import time
from pytorch_lightning.accelerators import *

from pytorch_lightning.loggers import TensorBoardLogger
#from neuralprophet import NeuralProphet, set_log_level
#from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from torch import nn
from dataloading_helpers import electricity_dataloader
from config import *


print("Preparing dataset") 
# load dataset
electricity = electricity_dataloader.create_electricity_timeseries_tft()
timeseries_dict =  electricity
config_name_string = "electricity"
parameters = []
model_dir = CONFIG_DICT["models"][config_name_string]


# if possible use GPU
if torch.cuda.is_available():
    accelerator = "auto"
    devices = torch.cuda.current_device()
else:
    accelerator = None
    devices = None
    
print("Training on: ", accelerator, devices) 

print("Defining model")
# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger(CONFIG_DICT["models"]["electricity"])  # logging results to a tensorboard


trainer = pl.Trainer(
    default_root_dir=model_dir,
    max_epochs=20,
    gpus=1,
    auto_select_gpus=True,
    #devices=devices,
    #accelerator=accelerator,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=0.4, 
    fast_dev_run=False,  
    callbacks=[lr_logger, early_stop_callback],
    log_every_n_steps=5,
    logger=logger,
)

#cuda_instance = pl.accelerators.CUDAAccelerator()
#cuda_instance = CUDAAccelerator()
#cuda_instance.setup(trainer)


tft = TemporalFusionTransformer.from_dataset(
    timeseries_dict["training_dataset"],
    learning_rate=0.01,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=64,
    output_size= 3,  # 7 quantiles by default
    loss=QuantileLoss([0.1, 0.5, 0.9]),
    log_interval=5,
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

timeseries_dict['val_dataloader'].to(devices)


print("Training model")
# fit network
trainer.fit(
    tft,
    train_dataloaders=timeseries_dict["train_dataloader"],
    val_dataloaders=timeseries_dict["val_dataloader"],
)

print("trainging done. Evaluating...")


# evaluate
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = tft.load_from_checkpoint(best_model_path)
actuals = torch.cat([y[0] for x, y in iter(timeseries_dict["val_dataloader"])])
predictions = best_tft.predict(timeseries_dict["val_dataloader"])
print("Best model MAE: ",(actuals - predictions).abs().mean().item())



output_dict = {'model_path': best_model_path,
               'MAE'       : (actuals - predictions).abs().mean().item(),
               'device'    : devices}

with open('output.txt', 'w') as convert_file:
     convert_file.write(json.dumps(details))