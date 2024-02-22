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
        action_decoder: DictConfig,
        state_reconstruction_weight: float,
        language_weight: float,
        plan_weight: float,
        action_gripper_weight: float,
        action_joints_weight: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        distribution: DictConfig,

        val_instructions: DictConfig,
        replan_freq: int = 30,

    ):
        super(Lcrs, self).__init__()
        self.perceptual_encoder = hydra.utils.instantiate(perceptual_encoder)
        self.distribution = hydra.utils.instantiate(distribution)
        self.language_encoder = hydra.utils.instantiate(language_goal, dist=self.distribution)
        self.plan_proposal = hydra.utils.instantiate(plan_proposal, dist=self.distribution)
        self.plan_recognition = hydra.utils.instantiate(plan_recognition, dist=self.distribution)
        self.action_decoder = hydra.utils.instantiate(action_decoder, dist=self.distribution)

        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler

        self.state_reconstruction_weight = state_reconstruction_weight
        self.language_weight = language_weight
        self.plan_weight = plan_weight
        self.action_gripper_weight = action_gripper_weight
        self.action_joints_weight = action_joints_weight

        self.save_hyperparameters()

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    def logUpdate(self, encodingLoss, languageLoss, planLoss, actionLoss, gripperLoss) -> None:
        self.log(
            "train/encoding_loss",
            encodingLoss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/language_loss",
            languageLoss,
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
        tot = encodingLoss + actionLoss + gripperLoss + planLoss + languageLoss
        self.log(
            "train/total_loss",
            tot,
            on_step=False,
            on_epoch=True,
        )

    def forward(self, static, gripper, language, isTraining=False):
        '''
        Output structure:
        - features:
            - visual
            - language
        - plan:
            - proposal:
                - state
                - sampled
            - recognition: (defined only if isTraining is True)
                - state
                - sampled
        - action:
            - pi
            - mu
            - sigma
            - gripper
        '''

        bs, ss, cs, hs, ws = static.shape
        bg, sg, cg, hg, wg = gripper.shape

        assert bs == bg and ss == sg and cs == cg

        plan = None
        planRecognitionState = None
        sampledRecognitionPlan = None

        # VISUAL FEATURES EMBEDDING
        visualFeatures = self.perceptual_encoder(static.reshape(-1, cs, hs, ws), gripper.reshape(-1, cg, hg, wg))
        visualFeatures = visualFeatures.reshape(bs, ss, -1)

        # LANGUAGE EMBEDDING
        languageFeatures = self.language_encoder(language)

        # PLAN PROPOSAL
        # TODO: check if the dimensions match both in the validation and in plain inference
        planProposalState = self.plan_proposal(visual=visualFeatures[:, 0], language=languageFeatures)

        # PLAN PROPOSAL SAMPLING
        planProposalDist = self.distribution.get_dist(planProposalState)
        sampledProposalPlan = torch.flatten(planProposalDist.rsample(), start_dim=-2, end_dim=-1)
        plan = sampledProposalPlan

        if isTraining:
            # PLAN RECOGNITION
            planRecognitionState, planFeatures = self.plan_recognition(visualFeatures)

            # PLAN RECOGNITION SAMPLING
            planRecognitionDist = self.distribution.get_dist(planRecognitionState)
            sampledRecognitionPlan = torch.flatten(planRecognitionDist.rsample(), start_dim=-2, end_dim=-1)

            plan = sampledRecognitionPlan

        # ACTION GENERATION
        pi, mu, sigma, gripperAct = self.action_decoder(plan, visualFeatures, languageFeatures)

        return {"features":
                {"visual": visualFeatures,
                 "language": languageFeatures},
                "plan":
                    {"proposal":
                        {"state": planProposalState,
                         "sampled": sampledProposalPlan},
                     "recognition":
                        {"state": planRecognitionState,
                         "sampled": sampledRecognitionPlan}},
                "action":
                    {"pi": pi,
                     "mu": mu,
                     "sigma": sigma,
                     "gripper": gripperAct}}

    def training_step(self, batch: Dict[str, Dict], batch_idx: int) -> torch.Tensor:
        '''

        batch structure:
        - vis:
            'robot_obs', 'rgb_obs', 'depth_obs', 'actions', 'state_info', 'lang', 'idx'

        - lang:
            'robot_obs', 'rgb_obs', 'depth_obs', 'actions', 'state_info', 'use_for_aux_lang_loss', 'lang', 'idx'

        '''
        encodingLoss = torch.tensor(0.0).to(self.device)
        languageLoss = torch.tensor(0.0).to(self.device)
        actionLoss = torch.tensor(0.0).to(self.device)
        gripperLoss = torch.tensor(0.0).to(self.device)
        planLoss = torch.tensor(0.0).to(self.device)

        batchSize = 1
        auxSize = 1

        for modalityScope, dataset_batch in batch.items():
            if modalityScope != "lang":  # skip visual goal
                continue
            static = dataset_batch["rgb_obs"]["rgb_static"]
            gripper = dataset_batch["rgb_obs"]["rgb_gripper"]
            language = dataset_batch["lang"]
            obs_gt = dataset_batch["robot_obs"]
            proprioceptive = dataset_batch["state_info"]["robot_obs"]
            actions_gt = dataset_batch["actions"]
            aux_lang = dataset_batch["use_for_aux_lang_loss"]

            batchSize = dataset_batch["actions"].shape[0]
            auxSize = max(1.0, torch.sum(dataset_batch["use_for_aux_lang_loss"]).detach())

            out = self(static, gripper, language, True)

            # LOSSES

            encodingLoss = encodingLoss + self.perceptual_encoder.getLoss(out["features"]["visual"], obs_gt)

            planLoss = planLoss + \
                self.plan_proposal.getLoss(out["plan"]["proposal"]["state"], out["plan"]["recognition"]["state"])

            languageLoss = languageLoss + self.language_encoder.getLoss(
                out["plan"]["recognition"]["state"].logit, out["features"]["language"], aux_lang)

            logistics_loss, gripper_act_loss = self.action_decoder.getLoss(
                actions_gt, proprioceptive, out["action"]["pi"], out["action"]["mu"], out["action"]["sigma"], out["action"]["gripper"])
            actionLoss = actionLoss + logistics_loss
            gripperLoss = gripperLoss + gripper_act_loss

        encodingLoss = encodingLoss * self.state_reconstruction_weight
        encodingLoss /= batchSize
        languageLoss = languageLoss * self.language_weight
        languageLoss /= auxSize
        planLoss = planLoss * self.plan_weight
        planLoss /= batchSize
        actionLoss = actionLoss * self.action_joints_weight
        actionLoss /= batchSize
        gripperLoss = gripperLoss * self.action_gripper_weight
        gripperLoss /= batchSize

        self.logUpdate(encodingLoss, languageLoss, planLoss, actionLoss, gripperLoss)

        loss = encodingLoss + planLoss + actionLoss + gripperLoss + languageLoss
        return loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:
        '''
        todo
        '''
        output = {}
        val_total_act_loss = torch.tensor(0.0).to(self.device)
        for modalityScope, dataset_batch in batch.items():
            perceptual_emb = self.perceptual_encoder(dataset_batch["rgb_obs"], dataset_batch["depth_obs"], dataset_batch["robot_obs"])
            # if self.state_recons: #state_reconstruction_weight?
            #     state_recon_loss = self.preceptual_encoder.state_reconstruction_loss()
            #     self.log(f"val/proprio_loss_{self.modality_scope}", state_recon_loss, sync_dist=True)
            # if "lang" in self.modality_scope:
                # latent_goal = self.language_goal(dataset_batch["lang"])
            # else:
                # latent_goal = self.visual_goal(perceptual_emb[:, -1])
            
            if modalityScope != "lang":
                continue #skip visual goal 
            # Variable mapping
            static = dataset_batch["rgb_obs"]["rgb_static"]
            gripper = dataset_batch["rgb_obs"]["rgb_gripper"]
            language = dataset_batch["lang"]
            # obs_gt = dataset_batch["robot_obs"]
            proprioceptive = dataset_batch["state_info"]["robot_obs"]
            actions_gt = dataset_batch["actions"]
            # aux_lang = dataset_batch["use_for_aux_lang_loss"]

            # batchSize = dataset_batch["actions"].shape[0]
            # auxSize = max(1.0, torch.sum(dataset_batch["use_for_aux_lang_loss"]).detach())

            # out = self(static, gripper, language, True)

            # Forward
            sample_features, sample_plan, sample_act = self.forward(static=static, gripper=gripper, language=language, isTraining=false)
            # TODO: Get loss
            empty_plan = torch.empty((dataset_batch["actions"].shape[0]), 0).to(self.device)
            

            action_loss, sample_act = self.action_decoder.loss_and_act(  # type:  ignore
                empty_plan, perceptual_emb, language, actions_gt, proprioceptive
            )

            # Calculate metrics
            mae = torch.nn.functional.l1_loss(
                sample_act[..., :-1], actions_gt[..., :-1], reduction="none"
            )  # (batch, seq, 6)
            mae = torch.mean(mae, 1)  # (batch, 6)
            # gripper action
            gripper_discrete = sample_act[..., -1]
            gt_gripper_act = actions_gt[..., -1]
            m = gripper_discrete > 0
            gripper_discrete[m] = 1
            gripper_discrete[~m] = -1
            gripper_sr = torch.mean((gt_gripper_act == gripper_discrete).float())
            # _, seq_feat = self.plan_recognition(perceptual_emb)

            # if "lang" in modalityScope:
            #     if self.use_bc_z_auxiliary_loss:
            #         val_pred_lang_loss = self.bc_z_auxiliary_loss(
            #             seq_feat, dataset_batch["lang"], dataset_batch["use_for_aux_lang_loss"]
            #         )
            #         self.log("val/lang_pred_loss", val_pred_lang_loss, sync_dist=True)
                # if self.use_clip_auxiliary_loss:
                #     val_pred_clip_loss = self.clip_auxiliary_loss(
                #         seq_feat, latent_goal, dataset_batch["use_for_aux_lang_loss"]
                #     )
                #     self.log("val/val_pred_clip_loss", val_pred_clip_loss, sync_dist=True)
                #     self.clip_groundtruth(seq_feat, dataset_batch["idx"], dataset_batch["use_for_aux_lang_loss"])
                # if self.use_mia_auxiliary_loss:
                #     val_pred_contrastive_loss = self.mia_auxiliary_loss(
                #         seq_feat, latent_goal, dataset_batch["use_for_aux_lang_loss"]
                #     )
                    # self.log("val/lang_contrastive_loss", val_pred_contrastive_loss, sync_dist=True)
            val_total_act_loss += action_loss
            mae_mean = mae.mean()
            pos_mae = mae[..., :3].mean()
            orn_mae = mae[..., 3:6].mean()

            # Logging
            self.log(f"val_total_mae/{modalityScope}_total_mae", mae_mean, sync_dist=True)
            self.log(f"val_pos_mae/{modalityScope}_pos_mae", pos_mae, sync_dist=True)
            self.log(f"val_orn_mae/{modalityScope}_orn_mae", orn_mae, sync_dist=True)
            self.log(f"val_act/{modalityScope}_act_loss", action_loss, sync_dist=True)
            self.log(f"val_grip/{modalityScope}_grip_sr", gripper_sr, sync_dist=True)
            self.log(
                "val_act/action_loss",
                val_total_act_loss / len(self.trainer.datamodule.modalities),  # type:ignore
                sync_dist=True,
            )
            output[f"idx_{modalityScope}"] = dataset_batch["idx"]

        return output

            




            



        return super().validation_step(*args, **kwargs)
    def step(self, obs, goal):
        """
        Do one step of inference with the model.
        """
        pass
