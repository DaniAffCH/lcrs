import logging
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

from calvin_agent.models.calvin_base_model import CalvinBaseModel

from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info
import torch
import hydra
import numpy as np

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

        self.replan_freq = replan_freq

        self.rolloutPlan = None
        self.rolloutStep = 0
        self.rolloutGoal = None

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self) -> None:
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_start(self) -> None:
        logger.info(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def logUpdate(self, modality, losses):
        totalLoss = 0
        for k, v in losses.items():
            v = v.cpu().detach()
            self.log(
                f"{modality}/{k}",
                v,
                on_step=False,
                on_epoch=True,
            )
            if k.endswith("_loss"):
                totalLoss += v

        self.log(
            f"{modality}/total_loss",
            totalLoss,
            on_step=False,
            on_epoch=True,
        )

    def forward(self, static, gripper, language, recognizePlan=False, useRecognition=False):
        '''
        Input:
        recognizePlan: use the recognition network to sample a plan
        useRecognition: use the plan from the recognition network to sample an action (otherwise the proposal network is used)

        Output structure:
        - features:
            - visual
            - language
        - plan:
            - proposal:
                - state
                - sampled
            - recognition: (defined only if recognizePlan is True)
                - state
                - sampled
        - action:
            - pi
            - mu
            - sigma
            - gripper
        '''

        # Use recognition => recognizePlan
        assert not useRecognition or recognizePlan

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

        if recognizePlan:
            # PLAN RECOGNITION
            planRecognitionState, planFeatures = self.plan_recognition(visualFeatures)

            # PLAN RECOGNITION SAMPLING
            planRecognitionDist = self.distribution.get_dist(planRecognitionState)
            sampledRecognitionPlan = torch.flatten(planRecognitionDist.rsample(), start_dim=-2, end_dim=-1)

        plan = sampledRecognitionPlan if useRecognition else sampledProposalPlan

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
        losses = dict()

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

            out = self(static, gripper, language, True, True)

            # losses

            losses["encoding_loss"] = self.perceptual_encoder.getLoss(out["features"]["visual"], obs_gt)

            losses["plan_loss"] = self.plan_proposal.getLoss(
                out["plan"]["proposal"]["state"], out["plan"]["recognition"]["state"])

            losses["language_loss"] = self.language_encoder.getLoss(
                out["plan"]["recognition"]["state"].logit, out["features"]["language"], aux_lang)

            logistics_loss, gripper_act_loss = self.action_decoder.getLoss(
                actions_gt, proprioceptive, out["action"]["pi"], out["action"]["mu"], out["action"]["sigma"], out["action"]["gripper"])
            losses["action_loss"] = logistics_loss
            losses["gripper_loss"] = gripper_act_loss

        losses["encoding_loss"] = losses["encoding_loss"] * self.state_reconstruction_weight
        losses["encoding_loss"] /= batchSize
        losses["language_loss"] = losses["language_loss"] * self.language_weight
        losses["language_loss"] /= auxSize
        losses["plan_loss"] = losses["plan_loss"] * self.plan_weight
        losses["plan_loss"] /= batchSize
        losses["action_loss"] = losses["action_loss"] * self.action_joints_weight
        losses["action_loss"] /= batchSize
        losses["gripper_loss"] = losses["gripper_loss"] * self.action_gripper_weight
        losses["gripper_loss"] /= batchSize

        self.logUpdate("training", losses)

        loss = sum(list(losses.values()))
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Dict], batch_idx: int) -> Dict[str, torch.Tensor]:

        losses = dict()
        out = None
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

            # proposal feedforward
            out = self(static, gripper, language, True, False)

            # plan independent metrics
            losses["encoding_loss"] = self.perceptual_encoder.getLoss(out["features"]["visual"], obs_gt)

            losses["plan_loss"] = self.plan_proposal.getLoss(
                out["plan"]["proposal"]["state"], out["plan"]["recognition"]["state"])

            losses["language_loss"] = self.language_encoder.getLoss(
                out["plan"]["recognition"]["state"].logit, out["features"]["language"], aux_lang)

            # plan proposal metrics
            logistics_loss, gripper_act_loss = self.action_decoder.getLoss(
                actions_gt, proprioceptive, out["action"]["pi"], out["action"]["mu"], out["action"]["sigma"], out["action"]["gripper"])

            losses["action_proposal_loss"] = logistics_loss
            losses["gripper_proposal_loss"] = gripper_act_loss
            sampledJointAction, sampledGripperConverted = self.split_sampled_actions(sampleAction=self.action_decoder.sample(**out["action"]))
            losses["gripper_proposal_accuracy"] = torch.mean((actions_gt[:, :, -1] == sampledGripperConverted).float())
            losses["action_proposal_mse"] = torch.nn.functional.mse_loss(sampledJointAction, actions_gt[:, :, :-1])
           
            # recognition feedforward (terribly unefficient but needed for keeping the code clean :) )
            out = self(static, gripper, language, True, True)

            # plan recognition metrics
            logistics_loss, gripper_act_loss = self.action_decoder.getLoss(
                actions_gt, proprioceptive, out["action"]["pi"], out["action"]["mu"], out["action"]["sigma"], out["action"]["gripper"])
            losses["action_recognition_loss"] = logistics_loss
            losses["gripper_recognition_loss"] = gripper_act_loss
            sampledJointAction, sampledGripperConverted = self.split_sampled_actions(sampleAction=self.action_decoder.sample(**out["action"]))
            losses["gripper_recognition_accuracy"] = torch.mean((actions_gt[:, :, -1] == sampledGripperConverted).float())
            losses["action_recognition_mse"] = torch.nn.functional.mse_loss(sampledJointAction, actions_gt[:, :, :-1])

        losses["encoding_loss"] = losses["encoding_loss"] * self.state_reconstruction_weight
        losses["encoding_loss"] /= batchSize
        losses["language_loss"] = losses["language_loss"] * self.language_weight
        losses["language_loss"] /= auxSize
        losses["plan_loss"] = losses["plan_loss"] * self.plan_weight
        losses["plan_loss"] /= batchSize
        losses["action_proposal_loss"] = losses["action_proposal_loss"] * self.action_joints_weight
        losses["action_proposal_loss"] /= batchSize
        losses["action_recognition_loss"] = losses["action_recognition_loss"] * self.action_joints_weight
        losses["action_recognition_loss"] /= batchSize
        losses["gripper_proposal_loss"] = losses["gripper_proposal_loss"] * self.action_gripper_weight
        losses["gripper_proposal_loss"] /= batchSize
        losses["gripper_recognition_loss"] = losses["gripper_recognition_loss"] * self.action_gripper_weight
        losses["gripper_recognition_loss"] /= batchSize

        self.logUpdate("eval", losses)

        return {
            "sampled_plan_pp_lang": out["plan"]["proposal"]["sampled"],
            "sampled_plan_pr_lang": out["plan"]["recognition"]["sampled"],
            "idx_val": batch_idx
        }

    def split_sampled_actions(sampledAction):
        sampledJointAction = sampledAction[:, :, :-1]
        sampledGripperAction = sampledAction[:, :, -1]
        sampledGripperConverted = torch.where(sampledGripperAction > 0, 1, -1)
        return sampledJointAction, sampledGripperConverted

    # Required for CalvinBaseModel rollout
    def load_lang_embeddings(self, embeddings_path):
        embeddings = np.load(embeddings_path, allow_pickle=True).item()
        # lang_embedding is a dictionary string:embedding(int)
        self.lang_embeddings = {v["ann"][0]: v["emb"] for _, v in embeddings.items()}

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def reset(self):
        self.rolloutPlan = None
        self.rolloutStep = 0
        self.rolloutGoal = None

    @torch.no_grad()
    def step(self, obs, goal):
        """
        Do one step of inference with the model.
        """
        if self.rolloutGoal is None:

            language = torch.from_numpy(self.lang_embeddings[goal]).to(self.device).squeeze(0).float()
            self.rolloutGoal = self.language_encoder(language)

        static = obs["rgb_obs"]["rgb_static"]
        gripper = obs["rgb_obs"]["rgb_gripper"]

        bs, ss, cs, hs, ws = static.shape
        bg, sg, cg, hg, wg = gripper.shape

        visualFeatures = self.perceptual_encoder(static.reshape(-1, cs, hs, ws), gripper.reshape(-1, cg, hg, wg))
        visualFeatures = visualFeatures.reshape(bs, ss, -1)

        if self.rolloutStep % self.replan_freq == 0:

            planProposalState = self.plan_proposal(visual=visualFeatures[:, 0], language=self.rolloutGoal)
            planProposalDist = self.distribution.get_dist(planProposalState)
            self.rolloutPlan = torch.flatten(planProposalDist.rsample(), start_dim=-2, end_dim=-1)

        pi, mu, sigma, gripperAct = self.action_decoder(self.rolloutPlan, visualFeatures, self.rolloutGoal)

        sampledAction = self.action_decoder.sample(pi, mu, sigma, gripperAct)

        self.rolloutStep += 1

        return sampledAction
