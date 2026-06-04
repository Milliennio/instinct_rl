import torch
import torch.nn as nn
import torch.nn.functional as F

from .actor_critic import ActorCritic, get_activation
from .moe import MoeLayer


class MoEActorCritic(ActorCritic):
    def __init__(
        self,
        obs_format,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        num_rewards=1,
        mu_activation=None,
        num_moe_experts=8,
        moe_gate_hidden_dims=[],
        **kwargs,
    ):
        self.num_moe_experts = num_moe_experts
        self.moe_gate_hidden_dims = moe_gate_hidden_dims
        super().__init__(
            obs_format,
            num_actions,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            init_noise_std,
            num_rewards,
            mu_activation,
            **kwargs,
        )

    def _build_actor(self, num_actions):
        moe = MoeLayer(
            self.mlp_input_dim_a,
            self.num_moe_experts,
            output_dim=num_actions,
            activation=self.activation,
            expert_hidden_dims=self.actor_hidden_dims,
            gate_hidden_dims=self.moe_gate_hidden_dims,
        )
        if self.mu_activation:
            return nn.Sequential(moe, get_activation(self.mu_activation))
        return moe

    def _build_critic(self, num_values=1):
        return MoeLayer(
            self.mlp_input_dim_c,
            self.num_moe_experts,
            output_dim=num_values,
            activation=self.activation,
            expert_hidden_dims=self.critic_hidden_dims,
            gate_hidden_dims=self.moe_gate_hidden_dims,
        )

    def _find_moe_layer(self, module):
        if isinstance(module, MoeLayer):
            return module
        for child in module.children():
            moe_layer = self._find_moe_layer(child)
            if moe_layer is not None:
                return moe_layer
        return None

    def _encode_actor_gate_observations(self, observations):
        if hasattr(self, "encoders"):
            return self.encoders(observations)
        return observations

    def _encode_critic_gate_observations(self, critic_observations):
        if hasattr(self, "critic_encoders") and self.critic_encoders is not None:
            return self.critic_encoders(critic_observations)
        return critic_observations

    @torch.no_grad()
    def get_moe_gate_weights(self, observations=None, critic_observations=None):
        """Return current actor and critic MoE gate weights for logging."""
        gate_weights = {}

        if observations is not None:
            actor_moe = self._find_moe_layer(self.actor)
            if actor_moe is not None:
                actor_observations = self._encode_actor_gate_observations(observations)
                gate_weights["actor"] = actor_moe.gate_weights(actor_observations).detach()

        if critic_observations is not None:
            critic_modules = list(self.critics) if hasattr(self, "critics") else [self.critic]
            if isinstance(critic_observations, list):
                critic_inputs = [
                    self._encode_critic_gate_observations(critic_obs) for critic_obs in critic_observations
                ]
            else:
                critic_obs = self._encode_critic_gate_observations(critic_observations)
                critic_inputs = [critic_obs for _ in critic_modules]

            for i, (critic, critic_obs) in enumerate(zip(critic_modules, critic_inputs)):
                critic_moe = self._find_moe_layer(critic)
                if critic_moe is not None:
                    gate_name = "critic" if len(critic_modules) == 1 else f"critic_{i}"
                    gate_weights[gate_name] = critic_moe.gate_weights(critic_obs).detach()

        return gate_weights
