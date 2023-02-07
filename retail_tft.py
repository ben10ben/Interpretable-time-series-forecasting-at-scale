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
from dataloading_helpers import retail_dataloader
from config import *


from pytorch_lightning.callbacks import DeviceStatsMonitor