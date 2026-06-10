import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from instinct_rl.utils.utils import get_subobs_indexing_by_components, get_subobs_size

from .actor_critic import ActorCritic, get_activation
from .moe import MoeLayer


class MoEGateExportWrapper(nn.Module):
    def __init__(self, moe_layer: MoeLayer):
        super().__init__()
        self.moe_layer = moe_layer

    def forward(self, observations):
        return self.moe_layer.gate_weights(observations)


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
        moe_actor_gate_component_names=None,
        moe_critic_gate_component_names=None,
        **kwargs,
    ):
        self.num_moe_experts = num_moe_experts
        self.moe_gate_hidden_dims = moe_gate_hidden_dims
        self.moe_actor_gate_component_names = moe_actor_gate_component_names
        self.moe_critic_gate_component_names = moe_critic_gate_component_names
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

    def _get_moe_gate_input_spec(self, obs_segments, component_names, gate_name):
        if component_names is None:
            return None, None
        if isinstance(component_names, str):
            component_names = [component_names]
        if len(component_names) == 0:
            raise ValueError(f"{gate_name} MoE gate component list must not be empty.")

        missing_names = [name for name in component_names if name not in obs_segments]
        if missing_names:
            raise ValueError(
                f"{gate_name} MoE gate component(s) {missing_names} are not in the available observation segments "
                f"{list(obs_segments.keys())}."
            )

        gate_input_dim = get_subobs_size(obs_segments, component_names)
        gate_input_indices = get_subobs_indexing_by_components(obs_segments, component_names)
        return int(gate_input_dim), gate_input_indices

    def _get_actor_mlp_obs_segments(self):
        return getattr(self, "_ActorCritic__obs_segments", self.obs_segments)

    def _get_critic_mlp_obs_segments(self):
        return getattr(self, "_ActorCritic__critic_obs_segments", self.critic_obs_segments)

    def _build_actor(self, num_actions):
        gate_input_dim, gate_input_indices = self._get_moe_gate_input_spec(
            self._get_actor_mlp_obs_segments(),
            self.moe_actor_gate_component_names,
            "actor",
        )
        moe = MoeLayer(
            self.mlp_input_dim_a,
            self.num_moe_experts,
            output_dim=num_actions,
            activation=self.activation,
            expert_hidden_dims=self.actor_hidden_dims,
            gate_hidden_dims=self.moe_gate_hidden_dims,
            gate_input_dim=gate_input_dim,
            gate_input_indices=gate_input_indices,
        )
        if self.mu_activation:
            return nn.Sequential(moe, get_activation(self.mu_activation))
        return moe

    def _build_critic(self, num_values=1):
        gate_input_dim, gate_input_indices = self._get_moe_gate_input_spec(
            self._get_critic_mlp_obs_segments(),
            self.moe_critic_gate_component_names,
            "critic",
        )
        return MoeLayer(
            self.mlp_input_dim_c,
            self.num_moe_experts,
            output_dim=num_values,
            activation=self.activation,
            expert_hidden_dims=self.critic_hidden_dims,
            gate_hidden_dims=self.moe_gate_hidden_dims,
            gate_input_dim=gate_input_dim,
            gate_input_indices=gate_input_indices,
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

    def export_as_onnx(self, observations, filedir):
        """Export actor action and actor MoE gate weights as separate ONNX models."""
        super().export_as_onnx(observations, filedir)
        actor_moe = self._find_moe_layer(self.actor)
        if actor_moe is None:
            return

        self.eval()
        gate_model = MoEGateExportWrapper(actor_moe)
        gate_model.eval()
        with torch.no_grad():
            torch.onnx.export(
                gate_model,
                observations,
                os.path.join(filedir, "actor_moe_gate.onnx"),
                input_names=["input"],
                output_names=["gate_weights"],
                opset_version=12,
            )
        print(f"Exported actor MoE gate model to {os.path.join(filedir, 'actor_moe_gate.onnx')}")
