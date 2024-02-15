import logging
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

from calvin_agent.models.calvin_base_model import CalvinBaseModel

from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info
import torch
import hydra


logger = logging.getLogger(__name__)


class Lcrs(pl.LightningModule, CalvinBaseModel):

    def __init__(
        self,
        perceptual_encoder: DictConfig,
        plan_proposal: DictConfig,
        plan_recognition: DictConfig,
        language_goal: DictConfig,
        visual_goal: DictConfig,
        action_decoder: DictConfig,
        state_reconstruction_weight: float,
        plan_weight: float,
        action_gripper_weight: float,
        action_joints_weight: float,
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
        self.language_encoder = hydra.utils.instantiate(language_goal)
        self.distribution = hydra.utils.instantiate(distribution)
        self.plan_proposal = hydra.utils.instantiate(plan_proposal, dist=self.distribution)
        self.plan_recognition = hydra.utils.instantiate(plan_recognition, dist=self.distribution)
        self.action_decoder = hydra.utils.instantiate(action_decoder)

        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler

        self.state_reconstruction_weight = state_reconstruction_weight
        self.plan_weight = plan_weight
        self.action_gripper_weight = action_gripper_weight
        self.action_joints_weight = action_joints_weight

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    def logUpdate(self, encodingLoss, planLoss, actionLoss, gripperLoss) -> None:
        self.log(
            "train/encoding_loss",
            encodingLoss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/plan_loss",
            planLoss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/action_loss",
            actionLoss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/gripper_loss",
            gripperLoss,
            on_step=False,
            on_epoch=True,
        )
        tot = encodingLoss + actionLoss + gripperLoss + planLoss
        self.log(
            "train/total_loss",
            tot,
            on_step=False,
            on_epoch=True,
        )

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        '''

        batch structure:
        - vis:
            'robot_obs', 'rgb_obs', 'depth_obs', 'actions', 'state_info', 'lang', 'idx'

        - lang:
            'robot_obs', 'rgb_obs', 'depth_obs', 'actions', 'state_info', 'use_for_aux_lang_loss', 'lang', 'idx'

        '''
        encodingLoss = torch.tensor(0.0).to(self.device)
        actionLoss = torch.tensor(0.0).to(self.device)
        gripperLoss = torch.tensor(0.0).to(self.device)
        planLoss = torch.tensor(0.0).to(self.device)

        for modalityScope, dataset_batch in batch.items():
            if modalityScope != "lang":  # skip visual goal
                continue
            static = dataset_batch["rgb_obs"]["rgb_static"]
            gripper = dataset_batch["rgb_obs"]["rgb_gripper"]
            language = dataset_batch["lang"]
            obs_gt = dataset_batch["robot_obs"]
            proprioceptive = dataset_batch["state_info"]["robot_obs"]
            actions_gt = dataset_batch["actions"]

            bs, ss, cs, hs, ws = static.shape
            bg, sg, cg, hg, wg = gripper.shape

            assert bs == bg and ss == sg and cs == cg

            # VISUAL FEATURES EMBEDDING
            visualFeatures = self.perceptual_encoder(static.reshape(-1, cs, hs, ws), gripper.reshape(-1, cg, hg, wg))
            visualFeatures = visualFeatures.reshape(bs, ss, -1)

            encodingLoss = encodingLoss + self.perceptual_encoder.getLoss(visualFeatures, obs_gt)

            # LANGUAGE EMBEDDING
            languageFeatures = self.language_encoder(language)

            # PLAN PROPOSAL
            planProposalState = self.plan_proposal(visual=visualFeatures[:, 0], language=languageFeatures)

            # PLAN RECOGNITION
            planRecognitionState = self.plan_recognition(visualFeatures)
            planRecognitionDist = self.distribution.get_dist(planRecognitionState)
            sampled_plan = torch.flatten(planRecognitionDist.rsample(), start_dim=-2, end_dim=-1)

            planLoss = planLoss + self.plan_proposal.getLoss(planProposalState, planRecognitionState)

            # ACTION GENERATION
            pi, mu, sigma, gripperAct = self.action_decoder(sampled_plan, visualFeatures, languageFeatures)

            logistics_loss, gripper_act_loss = self.action_decoder.getLoss(
                actions_gt, proprioceptive, pi, mu, sigma, gripperAct)
            actionLoss = actionLoss + logistics_loss
            gripperLoss = gripperLoss + gripper_act_loss

        encodingLoss = encodingLoss * self.state_reconstruction_weight
        planLoss = planLoss * self.plan_weight
        actionLoss = actionLoss * self.action_joints_weight
        gripperLoss = gripperLoss * self.action_gripper_weight

        self.logUpdate(encodingLoss, planLoss, actionLoss, gripperLoss)

        loss = encodingLoss + planLoss + actionLoss + gripperLoss
        return loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def step(self, obs, goal):
        """
        Do one step of inference with the model.
        """
        pass
