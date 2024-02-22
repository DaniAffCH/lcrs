from calvin_agent.utils.utils import print_system_env_info, get_last_checkpoint, format_sftp_path
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.cloud_io import load as pl_load
import hydra
import logging
from pathlib import Path
import sys
from typing import List
import lcrs
logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config")
def validate(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)
    datamodule = hydra.utils.instantiate(cfg.datamodule, training_repo_root=Path(lcrs.__file__).parents[1])
    model = hydra.utils.instantiate(cfg.model)
    chk = get_last_checkpoint(Path.cwd())

    if chk:
        pretrain_chk = pl_load(format_sftp_path(Path(cfg.pretrain_chk)), map_location=lambda storage, loc: storage)
        model.load_state_dict(pretrain_chk["state_dict"], strict=False)
        log_rank_0("LOADED PREVIOUS TRAINING")

    log_rank_0(print_system_env_info())

    train_logger = setup_logger(cfg, model)
    callbacks = setup_callbacks(cfg.callbacks)
    lr_logger = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_logger)

    trainer_args = {
        **cfg.trainer,
        "logger": train_logger,
        "callbacks": callbacks,
        "benchmark": False,
    }

    trainer = Trainer(**trainer_args)

    trainer.validate(model=model, datamodule=datamodule)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)