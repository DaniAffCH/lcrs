from calvin_agent.utils.utils import print_system_env_info
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf
import hydra
from datetime import timedelta
import logging
from pathlib import Path
import sys
from typing import List, Union
import lcrs

from lightning_lite.accelerators.cuda import num_cuda_devices
from pytorch_lightning.strategies import DDPStrategy


logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    This is called to start a training.

    Args:
        cfg: hydra config
    """
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(cfg.seed, workers=True)
    datamodule = hydra.utils.instantiate(cfg.datamodule, training_repo_root=Path(lcrs.__file__).parents[1])
    # chk = get_last_checkpoint(Path.cwd())

    # Load Model
    # if chk is not None:
    # model = getattr(models_m, cfg.model["_target_"].split(".")[-1]).load_from_checkpoint(chk.as_posix())
    # else:
    model = hydra.utils.instantiate(cfg.model)
    # if "pretrain_chk" in cfg:
    # initialize_pretrained_weights(model, cfg)

    log_rank_0(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
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

    # Start training
    # trainer.fit(model, datamodule=datamodule, ckpt_path=chk)  # type: ignore
    trainer.fit(model, datamodule=datamodule)  # type: ignore


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
        # train_logger.watch(model)
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
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


if __name__ == "__main__":
    train()
