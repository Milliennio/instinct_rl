import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from instinct_rl.modules.all_mixer import EncoderMoEActorCritic
from instinct_rl.modules.actor_critic import get_activation
from instinct_rl.utils.utils import get_subobs_by_components


class TerrainAuxEncoderMoEActorCritic(EncoderMoEActorCritic):
    """Encoder MoE actor-critic with a training-only terrain reconstruction head."""

    def __init__(
        self,
        *args,
        terrain_aux_group_name="terrain_aux",
        terrain_aux_latent_component_name="parallel_latent_0_depth_encoder",
        terrain_aux_output_shape=(99,),
        terrain_aux_hidden_dims=(128,),
        terrain_aux_activation="elu",
        terrain_aux_loss_func="smooth_l1",
        terrain_aux_smooth_l1_beta=0.05,
        **kwargs,
    ):
        self.terrain_aux_group_name = terrain_aux_group_name
        self.terrain_aux_latent_component_name = terrain_aux_latent_component_name
        self.terrain_aux_output_shape = tuple(terrain_aux_output_shape)
        self.terrain_aux_loss_func = terrain_aux_loss_func
        self.terrain_aux_smooth_l1_beta = terrain_aux_smooth_l1_beta
        super().__init__(*args, **kwargs)

        latent_shape = self.encoders.output_segment[self.terrain_aux_latent_component_name]
        latent_dim = math.prod(latent_shape)
        output_dim = math.prod(self.terrain_aux_output_shape)
        activation = get_activation(terrain_aux_activation)
        layers = []
        in_dim = latent_dim
        for hidden_dim in terrain_aux_hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.terrain_aux_head = nn.Sequential(*layers)

    def compute_auxiliary_losses(self, observations, aux_obs):
        if self.terrain_aux_group_name not in aux_obs:
            return {}, {}

        encoded_obs = self.encoders(observations)
        terrain_latent = get_subobs_by_components(
            encoded_obs,
            [self.terrain_aux_latent_component_name],
            self.encoders.output_segment,
        )
        prediction = self.terrain_aux_head(terrain_latent)
        target = aux_obs[self.terrain_aux_group_name].reshape_as(prediction)

        if self.terrain_aux_loss_func == "smooth_l1":
            loss = F.smooth_l1_loss(
                prediction,
                target,
                beta=self.terrain_aux_smooth_l1_beta,
            )
        elif self.terrain_aux_loss_func == "mse":
            loss = F.mse_loss(prediction, target)
        elif self.terrain_aux_loss_func == "l1":
            loss = F.l1_loss(prediction, target)
        else:
            raise ValueError(f"Unsupported terrain_aux_loss_func: {self.terrain_aux_loss_func}")

        stats = {
            "terrain_aux_abs_error": torch.mean(torch.abs(prediction.detach() - target)),
            "terrain_aux_target_std": torch.std(target.detach()),
        }
        return {"terrain_reconstruction_loss": loss}, stats
