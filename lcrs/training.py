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
def train(cfg: DictConfig) -> None:
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

    trainer.fit(model, datamodule=datamodule)


def setup_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """
    Instantiate all training callbacks.

    Args:
        callbacks_cfg: DictConfig with all callback params

    Returns:
        List of instantiated callbacks.
    """
    callbacks = [hydra.utils.instantiate(cb) for cb in callbacks_cfg.values()]
    return callbacks


def setup_logger(cfg: DictConfig, model: LightningModule) -> Logger:
    """
    Set up the logger (tensorboard or wandb) from hydra config.

    Args:
        cfg: Hydra config
        model: LightningModule

    Returns:
        logger
    """
    pathlib_cwd = Path.cwd()
    if "group" in cfg.logger:
        cfg.logger.group = pathlib_cwd.parent.name
        cfg.logger.name = pathlib_cwd.parent.name + "/" + pathlib_cwd.name
        cfg.logger.id = cfg.logger.name.replace("/", "_")
        train_logger = hydra.utils.instantiate(cfg.logger)
    else:
        train_logger = hydra.utils.instantiate(cfg.logger)
    return train_logger


def modify_argv_hydra() -> None:
    """
    To make hydra work with pytorch-lightning and ddp, we modify sys.argv for the child processes spawned with ddp.
    """
    cwd = Path.cwd().as_posix()
    cwd = f'"{cwd}"'
    sys.argv = sys.argv[:1]
    sys.argv.extend(
        [
            f"hydra.run.dir={cwd}",
            "hydra/hydra_logging=disabled",
            "hydra/job_logging=disabled",
        ]
    )
    overrides = OmegaConf.load(".hydra/overrides.yaml")
    for o in overrides:
        if "hydra/sweeper" in o:
            continue

        if "hydra/launcher" in o:
            continue

        sys.argv.append(o)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)


if __name__ == "__main__":
    train()
