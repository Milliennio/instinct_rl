# Instinct RL (from Rsl-RL code base, but with lots of modificaitons)

## Warning
This codebase is under [CC BY-NC 4.0 license](LICENSE), with inherited license in IsaacLab. You may not use the material for commercial purposes, e.g., to make demos to advertise your commercial products or wrap the code for your own commercial purposes.

## Contributing
See our [Contributor Agreement](CONTRIBUTOR_AGREEMENT.md) for contribution guidelines. By contributing or submitting a pull request, you agree to transfer copyright ownership of your contributions to the project maintainers.

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for a list of acknowledged contributors.

---

## Install

- Clone this repository separately from the Project Instinct installation:

  ```bash
  # Option 1: HTTPS
  git clone https://github.com/project-instinct/instinct_rl.git
  # Option 2: SSH
  git clone git@github.com:project-instinct/instinct_rl.git
  ```

- Install using any python interpreter

  ```bash
  python -m pip install -e instinct_rl
  ```

---

## Important New Concepts

### General Observation format

- **obs_format**: An OrderedDict specifying the format of input observations.
    key: obs_segment name, specifically for algorithm requirements, as obs_group in ManagerBasedRlEnv in IsaacLab
    value: an obs_segment object (see below)

- **obs_segment**: An OrderedDict specifying the segment of the observation.
    key: obs term name, as obs_term_name in ManagerBasedRlEnv in IsaacLab
    value: the shape of each part of the observation.

- **obs_pack**: The dictionary of observations following the definition of obs_format.
    key: obs_segment name as in observation_manager in ManagerBasedRlEnv in IsaacLab
    value: A flattened tensor/vector of each observation (required by the algorithm).

- **obs_component**: As obs_term in ManagerBasedRlEnv in IsaacLab.

---

## How to use the swappable algorithms and network modules design

In `on_policy_runner.py`, we have the following code:

```python
actor_critic = modules.build_actor_critic(
    self.policy_cfg.pop("class_name"),
    self.policy_cfg,
    obs_format,
    num_actions=env.num_actions,
    num_rewards=env.num_rewards,
).to(self.device)
```

This code is used to build the actor critic network. The `class_name` is the name of the actor critic class, and the `policy_cfg` is the configuration of the actor critic. The `obs_format` is the observation format, and the `num_actions` is the number of actions, and the `num_rewards` is the number of rewards.

The `modules.build_actor_critic` function is used to build the actor critic network. It is a factory function that builds the actor critic network based on the `class_name` and `policy_cfg`.

The `class_name` is the name of the actor critic class, which can be a class implemented in this repository, or a full import path to your own actor critic class in `module_name:class_name` pattern.

---

## Algorithms

### PPO

Standard Proximal Policy Optimization (PPO) algorithm for reinforcement learning. Implements the PPO objective with clipped surrogate loss, value function loss, and entropy regularization. Supports multiple learning epochs and mini-batch updates per environment step.

### State Estimator

Algorithm for learning state representations and estimation from partial observations. Combines reconstruction and prediction objectives to learn meaningful latent state representations that can be used for control.

### AMP (WASABI)

Implementation of WASABI, starting from the algorithm framework of AMP. Uses adversarial training with a discriminator to learn motion priors from expert demonstrations. Supports multiple discriminator architectures, including BCE loss, Wasserstein loss, and MSELoss. Includes gradient penalty and various discriminator architectures.

### Distillation and DAgger (TPPO)

Teacher-student distillation framework that extends PPO with teacher network guidance. Implements Dataset Aggregation (DAgger) and knowledge distillation techniques. Allows learning from expert demonstrations while maintaining PPO's stability guarantees. Supports various teacher action selection probabilities and distillation loss coefficients.

### VAE Distillation

Teacher-student distillation framework that extends TPPO with a VAE student network. Uses a VAE to encode and decode to generate the student's action. It generates additional latent distribution separate from the action distribution, which enables the potential of VAE-distillation combined with PPO.

---

## Network modules

### Actor Critic

Standard actor-critic architecture with separate policy (actor) and value (critic) networks. Supports configurable hidden dimensions, activation functions, and observation processing. Implements both continuous and discrete action spaces with proper probability distributions.

### MoE Actor Critic

Mixture of Experts (MoE) extension of the actor-critic architecture. Uses multiple expert networks with a gating mechanism to specialize different experts on different parts of the state/action space. Can improve performance on complex tasks by allowing specialization while maintaining parameter efficiency.

### VAE Actor Critic

Variational Autoencoder (VAE) extension of the actor-critic architecture. Uses a VAE to learn a latent representation of the observation, and then uses the actor to predict the action from the latent representation. Can be used for dimensionality reduction, feature learning, or as part of a larger reinforcement learning pipeline.

### Encoder related modules

Collection of encoder architectures for processing different input modalities:

- **Encoder Actor Critic**: Actor-critic networks with encoder backbones for processing high-dimensional inputs like images or complex state representations
- **All Mixer**: Multi-modal input processing that combines different observation types
- **VQ-VAE**: Vector Quantized Variational Autoencoder for discrete latent representations
- **State Estimator**: Networks for estimating hidden states from partial observations




# 0414

python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=20260409_125742

/home/you/instinct_rl/instinctlab/logs/0414/model_30000.pt

python scripts/rsl_rl/play.py --num_envs 64 --task Unitree-G1-29dof-Velocity --checkpoint /home/you/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/5090logs/0408/model_94999.pt

python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-v0 --checkpoint /home/you/instinct_rl/instinctlab/logs/0414/model_30000.pt --num_envs 64


python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py \
--task=Instinct-Parkour-Target-Amp-G1-v0 --num_envs 1 \
--load_run 0414 \
--checkpoint model_30000.pt

[INFO] Generating terrains based on curriculum took : 27.150010 seconds
Detected 43103 edges after processing.
Generate virtual obstacle edges : 64.027296 seconds
[INFO]: Time taken for scene creation : 120.910859 seconds
[INFO]: Scene manager:  <class InteractiveScene>
        Number of environments: 4096
        Environment spacing   : 2.5
        Source prim name      : /World/envs/env_0
        Global prim paths     : ['/World/ground']
        Replicate physics     : True
[INFO]: Starting the simulation. This may take a few seconds. Please wait...
Killed

export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=https://127.0.0.1:7897


yzp1-495012


python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-Play-v0 --load_run=20260430_153307_from20260409_125742

python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-Play-v0 --load_run=0414

python scripts/instinct_rl/collect_depth_data.py \
    --task Instinct-Parkour-Target-Amp-G1-Play-v0 \
    --load_run 0414 \
    --collect_steps 5000 \
    --num_envs 128 \
    --save_dir ./perception_dataset

0514

基于落脚点的控制？
控制目标为脚落在某个位置？


python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-Play-v0 --load_run=0414 --useonnx --num_envs 1

/home/you/instinct_rl/instinctlab/logs/0414/exported/actor.onnx

QFontDatabase: Cannot find font directory /home/you/miniconda3/envs/isaaclab/lib/python3.11/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/you/miniconda3/envs/isaaclab/lib/python3.11/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/you/miniconda3/envs/isaaclab/lib/python3.11/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/you/miniconda3/envs/isaaclab/lib/python3.11/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/you/miniconda3/envs/isaaclab/lib/python3.11/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
[INFO] Observation Manager: <ObservationManager> contains 4 groups.

# 0522
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-Play-v0 --load_run=0519_att --exportonnx


python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=0519_att --exportonnx



python source/instinctlab/instinctlab/tasks/parkour/scripts/collect_and_vis_depth.py --task Instinct-Parkour-Target-Amp-G1-Play-v0 --load_run 0519_att