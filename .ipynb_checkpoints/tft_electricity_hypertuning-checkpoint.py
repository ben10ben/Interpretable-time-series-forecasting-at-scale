if __name__ == '__main__': 
  print("Importing modules...")

  from torch.cuda import is_available
  from lightning.pytorch.accelerators import find_usable_cuda_devices
  from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
  from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
  from pytorch_forecasting.data.encoders import GroupNormalizer
  from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, DeviceStatsMonitor
  from torch.optim import Adam
  from torch.optim.lr_scheduler import ReduceLROnPlateau
  from dataloading_helpers import electricity_dataloader
  from config import *
  import pickle
  from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

  print("Preparing dataset...") 
  
  electricity = electricity_dataloader.create_electricity_timeseries_tft()
  timeseries_dict =  electricity
  config_name_string = "electricity"
  parameters = []
  model_dir = CONFIG_DICT["models"][config_name_string]


  if is_available():
      accelerator = "gpu"
      devices = find_usable_cuda_devices(1)
  else:
      accelerator = "cpu"
      devices = None

  # create study
  study = optimize_hyperparameters(
    electricity["train_dataloader"],
    electricity["val_dataloader"],
    model_path=CONFIG_DICT["models"]["electricity"],
    n_trials=100,
    max_epochs=20,
    gradient_clip_val_range=(0.01, 0.2),
    hidden_size_range=(16, 160),
    hidden_continuous_size_range=(8, 64),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.0005, 0.01),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=100, max_epochs=20, log_every_n_steps=5, accelerator=accelerator, devices=1),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
  )

  # save study results - also we can resume tuning at a later point in time
  with open("hypertuning_electricity.pkl", "wb") as fout:
      pickle.dump(study, fout)

  # show best hyperparameters
  print(study.best_trial.params)