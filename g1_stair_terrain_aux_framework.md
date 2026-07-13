# Instinct-Parkour-Target-Amp-G1-Stair-TerrainAux-v0 框架梳理

本文档基于当前代码中的配置文件整理：

- 任务入口：`instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/__init__.py`
- 环境增量配置：`instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_terrain_aux_cfg.py`
- G1 / AMP 基础配置：`instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py`
- 楼梯地形配置：`instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_stair_cfg.py`
- 基础 Parkour MDP：`instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`
- 训练配置：`instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg_terrain_aux.py`
- 辅助头实现：`instinct_rl/modules/terrain_aux_actor_critic.py`

## 1. 总体定位

`Instinct-Parkour-Target-Amp-G1-Stair-TerrainAux-v0` 是一个面向 Unitree G1 29 DoF 机器人的目标点驱动 parkour 任务。它在原始 Target AMP G1 任务上叠加了两个关键约束：

1. 只使用偏楼梯/粗糙平地的地形子集训练，而不是完整 rough terrain 集合。
2. 增加一个训练期地形辅助监督任务：从深度图编码得到的视觉 latent 重建局部高度图。

因此当前任务可以概括为：

> G1 机器人在楼梯类地形上，根据地形采样出的目标点生成速度命令，用 PPO + AMP/WASABI 学习运动控制，同时让视觉深度编码器通过 `terrain_aux` 辅助损失学习局部地形结构。

## 2. 任务注册与继承链路

Gym 任务注册如下：

```python
gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Stair-TerrainAux-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    kwargs={
        "env_cfg_entry_point": "...g1_parkour_target_amp_terrain_aux_cfg:G1ParkourStairTerrainAuxEnvCfg",
        "instinct_rl_cfg_entry_point": "...agents.instinct_rl_amp_cfg_terrain_aux:G1ParkourTerrainAuxPPORunnerCfg",
    },
)
```

环境配置继承关系：

```text
ParkourEnvCfg
  -> G1ParkourRoughEnvCfg
      -> G1ParkourStairEnvCfg
          -> G1ParkourStairTerrainAuxEnvCfg
```

其中：

- `ParkourEnvCfg` 定义通用 MDP：scene、观测、动作、命令、奖励、终止、事件、课程。
- `G1ParkourRoughEnvCfg` 绑定 G1 机器人、粗糙地形、运动参考。
- `G1ParkourStairEnvCfg` 将地形裁剪为楼梯相关子集，并同步裁剪命令速度范围。
- `G1ParkourStairTerrainAuxEnvCfg` 通过 `TerrainAuxConfigMixin` 增加局部高度图 ray scanner 和 `terrain_aux` 观测组。

训练配置继承关系：

```text
G1ParkourTerrainAuxPPORunnerCfg
  policy    -> TerrainAuxMoEPolicyCfg
  algorithm -> TerrainAuxAmpAlgoCfg -> AmpAlgoCfg
```

## 3. 机器人与仿真设置

机器人主体是 G1 29 DoF torso-base popsicle 版本，并替换为带鞋 URDF：

- 基础资产：`G1_29DOF_TORSOBASE_POPSICLE_CFG`
- 当前任务实际使用：`g1_29dof_torsoBase_popsicle_with_shoe.urdf`
- 初始根位置：`(0.0, 0.0, 0.9)`
- `merge_fixed_joints = True`
- 使用 BeyondMimic 风格的 delayed actuators：`beyondmimic_g1_29dof_delayed_actuators`
- 鞋底配置会把脚底相关高度偏置改为 `0.058`

动作空间：

- 类型：`JointPositionActionCfg`
- 控制对象：所有关节 `[".*"]`
- 动作维度：G1 29 DoF，对应 29 维关节位置 action
- action scale：`beyondmimic_action_scale`
- 使用默认关节位置作为 offset：`use_default_offset=True`

仿真主参数：

| 项目 | 当前值 |
| --- | --- |
| 并行环境数 | `4096` |
| 环境间距 | `2.5` |
| episode 长度 | `20.0 s` |
| sim dt | `0.005 s` |
| decimation | `4` |
| 控制周期 | `0.02 s` / 50 Hz |
| render interval | `4` |
| contact sensor update period | `0.005 s` |

## 4. 楼梯地形配置

基础 rough terrain 生成器定义在 `ROUGH_TERRAINS_CFG`，原始包含平地、gap、楼梯、box、slope 等多类地形。当前 Stair 任务只保留以下子地形：

```text
perlin_rough
perlin_rough_stand
pyramid_stairs
pyramid_stairs_high
pyramid_stairs_inv
pyramid_stairs_inv_high
```

训练地形网格：

- `STAIR_TRAIN_NUM_ROWS = 6`
- `STAIR_TRAIN_NUM_COLS = 12`
- 每块 terrain size：`8.0 m x 8.0 m`
- horizontal scale：`0.05`
- vertical scale：`0.005`
- curriculum：继承基础 rough terrain 的 curriculum 设置

楼梯子类的关键范围：

| 地形 | step height | step width | platform width |
| --- | --- | --- | --- |
| `pyramid_stairs` | `0.05 - 0.23` | `0.3` | `2.5` |
| `pyramid_stairs_high` | `0.05 - 0.45` | `1.5` | `4.0` |
| `pyramid_stairs_inv` | `0.05 - 0.23` | `0.3` | `2.5` |
| `pyramid_stairs_inv_high` | `0.05 - 0.45` | `1.5` | `4.0` |

每个子地形都配置了 `flat_patch_sampling["target"]`，用于后续目标点命令采样。命令生成器会从这些 flat patches 中采样目标点。

Play 版本补充：

- `STAIR_PLAY_NUM_ROWS = 4`
- `STAIR_PLAY_NUM_COLS = 12`
- 墙概率清零
- `num_envs = 16`
- episode 长度改为 `10 s`
- terrain curriculum 关闭

当前文档主任务是训练版 `...-v0`，不是 `...-Play-v0`。

## 5. 目标点速度命令

命令项是 `base_velocity = PoseVelocityCommandCfg(...)`，但它不是简单随机速度命令，而是基于地形 flat patch 的目标点命令。

流程：

1. 从当前环境所在 terrain level/type 的 `flat_patches["target"]` 中采样目标点。
2. 计算机器人根节点到目标点的相对位置。
3. 用相对位置生成 base frame 线速度命令。
4. 用目标方向和机器人当前 heading 的误差生成 yaw 角速度命令。
5. 当目标距离小于 `target_dis_threshold=0.4` 时，速度命令置零。

关键参数：

| 参数 | 当前值 |
| --- | --- |
| resampling time | `8.0 - 12.0 s` |
| velocity_control_stiffness | `2.0` |
| heading_control_stiffness | `2.0` |
| rel_standing_envs | `0.05` |
| only_positive_lin_vel_x | `True` |
| lin_vel_threshold | `0.0` |
| ang_vel_threshold | `0.0` |
| target_dis_threshold | `0.4` |

Stair 任务会把速度范围裁剪到保留地形子集。例如：

- `perlin_rough`：`lin_vel_x=(0.45, 1.0)`，`ang_vel_z=(-1.0, 1.0)`
- `perlin_rough_stand`：所有速度为 0
- 楼梯类地形：`lin_vel_x=(0.45, 0.8)`，`ang_vel_z=(-1.0, 1.0)`

## 6. 传感器与观测来源

### 6.1 深度相机

深度相机挂在 `torso_link`，使用 `NoisyGroupedRayCasterCameraCfg`：

- 原始分辨率：`64 x 36`
- FOV 近似：水平 `89.51 deg`，垂直 `58.29 deg`
- data type：`distance_to_image_plane`
- update period：`0.02 s`
- min distance：`0.1`
- mesh prims：地形 `/World/ground` + G1 各 link

噪声/预处理流水线：

1. `CropAndResizeCfg(crop_region=(18, 0, 16, 16))`
   - 上方裁掉 18 行，左右各裁掉 16 列
   - 原始 `36 x 64` 变成 `18 x 32`
   - 没有额外 resize
2. `GaussianBlurNoiseCfg(kernel_size=3, sigma=1)`
3. `DepthNormalizationCfg(depth_range=(0.0, 2.5), output_range=(0.0, 1.0))`

相机维护历史：

- `data_histories["distance_to_image_plane_noised"] = 37`
- policy/critic 观测取 `num_output_frames=8`
- `history_skip_frames=5`
- `delayed_frame_ranges=(0, 1)`

因此深度观测形状为：

```text
depth_image: (8, 18, 32)
```

### 6.2 足端高度扫描

用于 `feet_at_plane` 奖励：

- `left_height_scanner` 挂在 `left_ankle_roll_link`
- `right_height_scanner` 挂在 `right_ankle_roll_link`
- offset：`(0.04, 0.0, 20.0)`
- grid pattern：`resolution=0.12, size=[0.12, 0.0]`
- update period：`0.02 s`

### 6.3 接触与腿部体积点

接触传感器：

- prim path：`{ENV_REGEX_NS}/Robot/.*`
- history length：`3`
- `track_air_time=True`

腿部体积点：

- prim path：`.*_ankle_roll_link`
- 原始范围：`x=-0.025..0.12`，`y=-0.03..0.03`，`z=-0.04..0.0`
- 网格：`10 x 5 x 2`
- 鞋底配置后 z 范围调整为：`z=-0.063..-0.023`

### 6.4 TerrainAux 局部高度图扫描

这是当前任务相对 Stair AMP 的核心增量。

`TerrainAuxConfigMixin.apply_terrain_aux_config()` 新增：

```python
self.scene.terrain_height_map_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    offset=RayCasterCfg.OffsetCfg(pos=(0.45, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(
        resolution=0.12,
        size=[1.2, 0.96],
    ),
    mesh_prim_paths=["/World/ground"],
    update_period=0.02,
)
```

高度图参数：

| 项目 | 当前值 |
| --- | --- |
| 挂载 link | `torso_link` |
| 前向偏移 | `x=0.45 m` |
| ray 对齐 | yaw 对齐 |
| 覆盖范围 | `1.2 m x 0.96 m` |
| 分辨率 | `0.12 m` |
| 输出维度 | `99`，对应约 `11 x 9` 网格 |
| update period | `0.02 s` |

高度图观测由 `mdp.local_terrain_height_map` 计算：

1. 取 scanner 的 ray hit 世界高度 `ray_hits_w[..., 2]`。
2. 取左右 `.*_ankle_roll_link` 中更低的足端高度。
3. 支撑高度为：`min_foot_height - support_height_offset`。
4. 当前 `support_height_offset = 0.058`，对应带鞋底偏置。
5. 输出相对支撑平面的局部地形高度。
6. 非有限值置 0。
7. 裁剪到 `[-0.5, 0.8]`。

该观测组设置：

```text
terrain_aux.height_map: (99,)
enable_corruption = False
concatenate_terms = True
```

## 7. 观测组结构

当前任务实际有 5 组观测：

```text
policy
critic
amp_policy
amp_reference
terrain_aux
```

### 7.1 Policy 观测

Policy 观测 `concatenate_terms=False`，每个 component 保留名字，后续由 encoder/MoE 按 component 处理。

| component | 来源 | 历史 | 噪声/scale | 形状 |
| --- | --- | --- | --- | --- |
| `base_ang_vel` | base 角速度 | 8 | uniform `[-0.2, 0.2]`, scale `0.25` | `24` |
| `projected_gravity` | 重力投影 | 8 | uniform `[-0.05, 0.05]` | `24` |
| `velocity_commands` | base velocity command | 8 | 无 | `24` |
| `joint_pos` | 相对默认关节位置 | 8 | uniform `[-0.01, 0.01]` | `232` |
| `joint_vel` | 关节速度 | 8 | uniform `[-0.5, 0.5]`, scale `0.05` | `232` |
| `actions` | 上一次 action | 8 | 无 | `232` |
| `depth_image` | 延迟深度图历史 | 8 | 相机噪声管线 | `(8,18,32)` |

原始 policy flatten 总维度：

```text
24 + 24 + 24 + 232 + 232 + 232 + 8*18*32 = 5376
```

深度图经过 encoder 后被替换为 96 维 latent，所以送入 MoE actor 的编码后维度为：

```text
24 + 24 + 24 + 232 + 232 + 232 + 96 = 864
```

### 7.2 Critic 观测

Critic 相比 policy 多了 privileged 的 `base_lin_vel`，并且不开 corruption。

| component | 来源 | 历史 | scale | 形状 |
| --- | --- | --- | --- | --- |
| `base_lin_vel` | base 线速度 | 8 | 无 | `24` |
| `base_ang_vel` | base 角速度 | 8 | `0.25` | `24` |
| `projected_gravity` | 重力投影 | 8 | 无 | `24` |
| `velocity_commands` | base velocity command | 8 | 无 | `24` |
| `joint_pos` | 相对默认关节位置 | 8 | 无 | `232` |
| `joint_vel` | 关节速度 | 8 | `0.05` | `232` |
| `actions` | 上一次 action | 8 | 无 | `232` |
| `depth_image` | 延迟深度图历史 | 8 | 相机噪声管线 | `(8,18,32)` |

原始 critic flatten 总维度：

```text
24 + 24 + 24 + 24 + 232 + 232 + 232 + 4608 = 5400
```

深度 encoder 后 critic 输入维度为：

```text
24 + 24 + 24 + 24 + 232 + 232 + 232 + 96 = 888
```

### 7.3 AMP 观测

AMP 使用两组观测：

- `amp_policy`：当前机器人状态序列
- `amp_reference`：motion reference 状态序列

两者结构对齐，供 discriminator 区分 policy rollout 和 reference motion。

包含项：

| component | 历史长度 | 说明 |
| --- | --- | --- |
| `projected_gravity` | 10 | 重力投影 |
| `joint_pos_rel` | 10 | 相对默认关节位置 |
| `joint_vel` | 10 | 关节速度，scale `0.05` |
| `base_lin_vel` | 10 | base 线速度 |
| `base_ang_vel` | 10 | base 角速度 |

如果按 G1 29 DoF 展平，AMP 状态约为：

```text
projected_gravity 30
joint_pos_rel     290
joint_vel         290
base_lin_vel      30
base_ang_vel      30
total             670
```

### 7.4 TerrainAux 观测

`terrain_aux` 是额外的监督标签组，不直接作为 actor/critic 输入。它会被 rollout storage 保存，在 PPO update 时传给 actor-critic 的 `compute_auxiliary_losses()`。

```text
terrain_aux.height_map: (99,)
```

## 8. 视觉编码器与 MoE Actor-Critic

当前 policy class：

```python
class_name = "instinct_rl.modules.terrain_aux_actor_critic:TerrainAuxEncoderMoEActorCritic"
```

它的主体是：

```text
Parallel depth encoder
  -> EncoderMoEActorCritic
      -> MoE actor
      -> MoE critic
      -> terrain reconstruction auxiliary head
```

### 8.1 深度图编码器

`DepthEncoderTemporalTerrainCfg` 使用 `ConvTemporalTransformerHeadModel`：

| 参数 | 当前值 |
| --- | --- |
| 输入 component | `depth_image` |
| 输入形状 | `(8, 18, 32)` |
| output size | `96` |
| CNN channels | `[16, 32, 64, 128]` |
| CNN kernels | `[3, 3, 3, (3, 4)]` |
| CNN strides | `[2, 2, 2, 1]` |
| CNN paddings | `[1, 1, 1, 0]` |
| Transformer d_model | `128` |
| heads | `4` |
| layers | `1` |
| FFN dim | `256` |
| dropout | `0.1` |
| activation | `relu` |
| norm_first | `True` |
| temporal_pool | `latest` |
| temporal position embedding | `True` |

编码器的输出 component 名称为：

```text
parallel_latent_0_depth_encoder: (96,)
```

由于 `takeout_input_components=True` 是默认值，`depth_image` 会从后续 MLP/MoE 输入中移除，替换为这 96 维视觉 latent。

Actor 和 critic 各自有一套 encoder：

```python
encoder_configs = EncoderConfigs()
critic_encoder_configs = EncoderConfigs()
```

也就是说当前配置没有共享 actor/critic 视觉编码器参数。

### 8.2 MoE Actor-Critic

`TerrainAuxMoEPolicyCfg` 的 MoE 设置：

| 参数 | 当前值 |
| --- | --- |
| num_moe_experts | `4` |
| actor hidden dims | `[256, 128, 64]` |
| critic hidden dims | `[256, 128, 64]` |
| activation | `elu` |
| init noise std | `1.0` |
| MoE gate hidden dims | `[128]` |

Actor：

```text
encoded policy obs (864)
  -> MoE gate
  -> 4 expert actor MLPs [256,128,64]
  -> action mean (29)
```

Critic：

```text
encoded critic obs (888)
  -> MoE gate
  -> 4 expert critic MLPs [256,128,64]
  -> value
```

当前代码状态需要特别注意：

```python
moe_actor_gate_component_names = None
moe_critic_gate_component_names = None
```

这意味着在 `TerrainAuxMoEPolicyCfg` 中，MoE gate 默认使用完整的编码后 actor/critic 输入：

- actor gate 输入：`864`
- critic gate 输入：`888`

文件里虽然导入了：

```python
from .gate_slice import ACTOR_MOE_GATE_COMPONENT_NAMES, CRITIC_MOE_GATE_COMPONENT_NAMES
```

但当前 `TerrainAux` policy 没有使用它们。`GateSeparatedMoEPolicyCfg` 才使用这些切片。如果以后把 `TerrainAuxMoEPolicyCfg` 改成 gate-separated，则 gate 输入会变成：

- actor gate：`projected_gravity + velocity_commands + base_ang_vel + depth_latent = 24+24+24+96 = 168`
- critic gate：`base_lin_vel + base_ang_vel + projected_gravity + velocity_commands + depth_latent = 24+24+24+24+96 = 192`

这点和 checkpoint 兼容性直接相关。

## 9. TerrainAux 辅助学习目标

`TerrainAuxEncoderMoEActorCritic` 在常规 actor-critic 之外增加一个训练期辅助头：

```text
depth latent (96)
  -> Linear(96, 96)
  -> ELU
  -> Linear(96, 99)
  -> predicted local height map
```

监督目标：

```text
target = aux_obs["terrain_aux"].reshape_as(prediction)
```

损失：

```python
F.smooth_l1_loss(prediction, target, beta=0.05)
```

配置参数：

| 参数 | 当前值 |
| --- | --- |
| terrain_aux_group_name | `terrain_aux` |
| terrain_aux_latent_component_name | `parallel_latent_0_depth_encoder` |
| terrain_aux_output_shape | `(99,)` |
| terrain_aux_hidden_dims | `[96]` |
| terrain_aux_activation | `elu` |
| terrain_aux_loss_func | `smooth_l1` |
| smooth_l1 beta | `0.05` |
| terrain_reconstruction_loss_coef | `0.1` |

这个辅助任务的意义：

- 强迫深度编码器的 96 维 latent 保留局部地形结构。
- 标签来自仿真 raycast 的局部高度图，不需要人工标注。
- 该损失只在训练 update 阶段加入总 loss；推理时不需要 `terrain_aux` 标签。
- 辅助头依赖 actor encoder 的 `parallel_latent_0_depth_encoder`，所以主要正则 actor 侧视觉表征。

PPO update 阶段的总损失大致为：

```text
total_loss =
  surrogate_loss
  + value_loss_coef * value_loss
  - entropy_coef * entropy
  + terrain_reconstruction_loss_coef * terrain_reconstruction_loss
```

AMP discriminator 另有独立优化步骤。

## 10. AMP / WASABI 训练框架

当前算法类：

```python
class_name = "WasabiPPO"
```

它在 PPO 外叠加 discriminator：

1. PPO 用环境 reward + discriminator auxiliary reward 更新 policy。
2. Discriminator 用 `amp_policy` 与 `amp_reference` 两组状态序列做二分类/打分。
3. 当前 discriminator loss 使用 `MSELoss`，reward 类型使用 `quad`，更偏 WASABI 风格。

Discriminator 配置：

| 参数 | 当前值 |
| --- | --- |
| hidden sizes | `[1024, 512]` |
| nonlinearity | `ReLU` |
| discriminator_reward_coef | `0.25` |
| reward type | `quad` |
| loss func | `MSELoss` |
| gradient penalty coef | `5.0` |
| optimizer | `AdamW` |
| discriminator lr | `1e-4` |
| weight decay coef | `3e-4` |
| logit weight decay coef | `0.04` |

PPO 配置：

| 参数 | 当前值 |
| --- | --- |
| rollout steps per env | `24` |
| max iterations | `30000` |
| save interval | `5000` |
| experiment name | `g1_parkour_terrain_aux` |
| empirical normalization | `False` |
| value loss coef | `1.0` |
| clip param | `0.2` |
| entropy coef | `0.006` |
| learning epochs | `5` |
| mini batches | `4` |
| actor-critic lr | `1e-3` |
| schedule | `adaptive` |
| gamma | `0.99` |
| lambda | `0.95` |
| desired KL | `0.01` |
| max grad norm | `1.0` |

## 11. 运动参考数据

AMP reference 来自 AMASS motion buffer：

```python
path = "/home/you/instinct_rl/instinctlab/data/parkour_motion_reference"
filtered_motion_selection_filepath = ".../parkour_motion_without_run.yaml"
motion_start_from_middle_range = [0.0, 0.9]
frame_interval_s = 0.02
update_period = 0.02
num_frames = 10
```

当前 motion buffer 名称：

```python
motion_buffers = {
    "run_walk": AmassMotionCfg(),
}
```

关注 links：

```text
pelvis
torso_link
left/right_shoulder_roll_link
left/right_elbow_link
left/right_wrist_yaw_link
left/right_hip_roll_link
left/right_knee_link
left/right_ankle_roll_link
```

还配置了左右对称增强：

- link mapping
- joint mapping
- joint sign reverse buffer

## 12. 奖励结构

奖励使用 `MultiRewardCfg`，当前只有一个 group：`rewards: G1Rewards`。主要奖励项如下。

任务相关：

| reward | weight | 作用 |
| --- | --- | --- |
| `track_lin_vel_xy_exp` | `2.0` | 跟踪 XY 线速度命令 |
| `track_ang_vel_z_exp` | `2.0` | 跟踪 yaw 角速度命令 |
| `heading_error` | `-1.0` | 惩罚 yaw 命令/朝向误差相关项 |
| `dont_wait` | `-0.5` | 有前进命令时惩罚不动 |
| `is_alive` | `3.0` | 存活奖励 |
| `stand_still` | `-0.3` | 无速度命令时约束回默认姿态 |

步态与足端：

| reward | weight | 作用 |
| --- | --- | --- |
| `volume_points_penetration` | `-4.0` | 惩罚腿部体积点穿入虚拟障碍 |
| `feet_air_time` | `0.5` | 鼓励合理摆腿/单脚支撑 |
| `feet_slide` | `-0.4` | 惩罚接触时足端滑动 |
| `feet_flat_ori` | `-0.4` | 惩罚接触足端姿态不平 |
| `feet_at_plane` | `-0.1` | 惩罚接触足端高于地面扫描平面 |
| `feet_close_xy` | `0.4` | 控制两脚横向距离过近问题 |

姿态/能耗/关节正则：

| reward | weight | 作用 |
| --- | --- | --- |
| `joint_deviation_hip` | `-0.5` | 惩罚 hip yaw/roll 偏离 |
| `ang_vel_xy_l2` | `-0.05` | 惩罚 roll/pitch 角速度 |
| `dof_torques_l2` | `-1.5e-7` | 腿部 torque 正则 |
| `dof_acc_l2` | `-1.25e-7` | 关节加速度正则 |
| `dof_vel_l2` | `-1e-4` | 关节速度正则 |
| `action_rate_l2` | `-0.005` | action rate 正则 |
| `flat_orientation_l2` | `-3.0` | 惩罚 base 姿态倾斜 |
| `pelvis_orientation_l2` | `-3.0` | 惩罚 pelvis 姿态倾斜 |
| `energy` | `-5e-5` | 电机功率正则 |
| `freeze_upper_body` | `-0.004` | 约束上肢和腰部偏离 |

安全项：

| reward | weight | 作用 |
| --- | --- | --- |
| `dof_pos_limits` | `-1.0` | 关节位置限制 |
| `dof_vel_limits` | `-1.0` | 关节速度限制，soft ratio `0.9` |
| `torque_limits` | `-0.01` | torque ratio 超限惩罚，limit ratio `0.8` |
| `undesired_contacts` | `-1.0` | 非足端接触惩罚 |

## 13. 终止、事件与课程

终止条件：

| termination | 参数 | 说明 |
| --- | --- | --- |
| `time_out` | time out | episode 到时 |
| `terrain_out_bound` | `distance_buffer=2.0` | 离开 terrain 边界 |
| `base_contact` | torso contact threshold `1.0` | 躯干接触终止 |
| `bad_orientation` | `limit_angle=1.0` | 姿态过差 |
| `root_height` | `minimum_height=0.5` | 根高度过低 |
| `dataset_exhausted` | motion reference | reference 数据耗尽 |

事件：

| event | mode | 说明 |
| --- | --- | --- |
| `physics_material` | startup | 随机化机器人刚体摩擦/反弹 |
| `reset_base` | reset | 根位置和速度随机重置 |
| `register_virtual_obstacles` | startup | 给腿部体积点传感器注册虚拟障碍 |
| `reset_robot_joints` | reset | 关节位置随机 offset `[-0.15, 0.15]` |

课程：

```python
terrain_levels = CurrTerm(
    func=mdp.tracking_exp_vel,
    params={"lin_vel_threshold": (0.3, 0.6), "ang_vel_threshold": (0.0, 0.0)},
)
```

也就是说 terrain level curriculum 主要根据速度跟踪表现推进。

## 14. 当前框架的数据流

```text
IsaacLab simulation
  |
  |-- G1 proprioception/history
  |-- target-point velocity command
  |-- delayed normalized depth history (8,18,32)
  |-- AMP robot/reference state history
  |-- TerrainAux local height map label (99)
  v
Observation groups
  |
  |-- policy obs -> depth ConvTemporalTransformer -> 96d latent
  |                 + proprio/action/command history
  |                 -> 864d encoded actor input
  |                 -> 4-expert MoE actor -> 29d action
  |
  |-- critic obs -> separate depth ConvTemporalTransformer -> 96d latent
  |                 + privileged base_lin_vel
  |                 -> 888d encoded critic input
  |                 -> 4-expert MoE critic -> value
  |
  |-- amp_policy / amp_reference -> discriminator -> AMP/WASABI reward + discriminator loss
  |
  |-- terrain_aux height_map -> auxiliary reconstruction loss from actor depth latent
```

训练目标：

```text
Policy learning:
  environment rewards
  + discriminator reward
  + PPO clipped objective
  + value learning
  + entropy regularization
  + 0.1 * terrain height-map reconstruction loss

Discriminator learning:
  distinguish amp_policy state sequence from amp_reference state sequence
  + gradient penalty
  + weight decay regularization
```

## 15. 与相邻任务的区别

| 任务 | 地形 | TerrainAux | 训练配置 |
| --- | --- | --- | --- |
| `Instinct-Parkour-Target-Amp-G1-v0` | 完整 rough terrain | 无 | `G1ParkourPPORunnerCfg` |
| `Instinct-Parkour-Target-Amp-G1-Stair-v0` | 楼梯子集 | 无 | `G1ParkourPPORunnerCfg` |
| `Instinct-Parkour-Target-Amp-G1-TerrainAux-v0` | 完整 rough terrain | 有 | `G1ParkourTerrainAuxPPORunnerCfg` |
| `Instinct-Parkour-Target-Amp-G1-Stair-TerrainAux-v0` | 楼梯子集 | 有 | `G1ParkourTerrainAuxPPORunnerCfg` |

当前任务的独特组合是：

```text
Stair terrain specialization + depth temporal encoder + local terrain reconstruction auxiliary loss + AMP/WASABI motion prior
```

## 16. 当前代码状态与注意点

1. `TerrainAux` 环境配置很薄，真正的环境主体在 `ParkourEnvCfg` 和 `G1ParkourStairEnvCfg`。`g1_parkour_target_amp_terrain_aux_cfg.py` 只负责新增 height-map scanner 和 `terrain_aux` 观测组。

2. `TERRAIN_AUX_OUTPUT_SHAPE = (99,)` 在环境配置和训练配置中各定义了一份。后续如果改 grid size/resolution，需要同时保持两处一致，否则辅助头输出和标签 shape 会错。

3. 当前 `TerrainAuxMoEPolicyCfg` 没有启用 gate-separated 切片。虽然导入了 `gate_slice`，但 actor/critic gate component names 均为 `None`。这意味着当前 checkpoint 结构应匹配 full-gate：

```text
actor gate input  = 864
critic gate input = 888
```

4. 如果改成：

```python
moe_actor_gate_component_names = ACTOR_MOE_GATE_COMPONENT_NAMES
moe_critic_gate_component_names = CRITIC_MOE_GATE_COMPONENT_NAMES
```

则 gate 结构会变成：

```text
actor gate input  = 168
critic gate input = 192
```

这会影响旧 checkpoint 加载和 ONNX 导出。

5. `terrain_aux` 是训练期监督信号，不是 actor 推理输入。部署时需要深度 encoder 和 actor 主干，但不需要 height-map scanner 标签。

6. 高度图标签相对脚底支撑平面，而不是绝对世界 z。这个设计能减少地形整体高度/机器人当前台阶高度对监督目标的干扰，更直接表达“脚附近/前方的相对可通行形状”。
