import logging
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

from calvin_agent.models.calvin_base_model import CalvinBaseModel

from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import hydra


logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)


class Lcrs(pl.LightningModule, CalvinBaseModel):

    def __init__(
        self,
        perceptual_encoder: DictConfig,
        plan_proposal: DictConfig,
        plan_recognition: DictConfig,
        language_goal: DictConfig,
        visual_goal: DictConfig,
        action_decoder: DictConfig,
        kl_beta: float,
        kl_balancing_mix: float,
        state_recons: bool,
        state_recon_beta: float,
        use_bc_z_auxiliary_loss: bool,
        bc_z_auxiliary_loss_beta: float,
        use_mia_auxiliary_loss: bool,
        mia_auxiliary_loss_beta: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        distribution: DictConfig,
        val_instructions: DictConfig,
        use_clip_auxiliary_loss: bool,
        clip_auxiliary_loss_beta: float,
        replan_freq: int = 30,
        bc_z_lang_decoder: Optional[DictConfig] = None,
        mia_lang_discriminator: Optional[DictConfig] = None,
        proj_vis_lang: Optional[DictConfig] = None,
    ):
        super(Lcrs, self).__init__()
        self.perceptual_encoder = hydra.utils.instantiate(perceptual_encoder)

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        '''

        batch structure:
        - vis:
            'robot_obs', 'rgb_obs', 'depth_obs', 'actions', 'state_info', 'lang', 'idx'

        - lang:
            'robot_obs', 'rgb_obs', 'depth_obs', 'actions', 'state_info', 'use_for_aux_lang_loss', 'lang', 'idx'

        '''
        encodingLoss = torch.tensor(0.0).to(self.device)

        for self.modality_scope, dataset_batch in batch.items():
            rgbs = dataset_batch["rgb_obs"]["rgb_static"]
            b, s, c, h, w = rgbs.shape

            visual_features = self.perceptual_encoder(rgbs.reshape(-1, c, h, w))
            visual_features = visual_features.reshape(b, s, -1)
            
            encodingLoss = encodingLoss + self.perceptual_encoder.getLoss(visual_features, dataset_batch["robot_obs"])

        loss = encodingLoss
        return loss

    def configure_optimizers(self):
        pass

    def step(self, obs, goal):
        """
        Do one step of inference with the model.
        """
        pass
