# Instinct-Parkour-Target-Amp-G1-TerrainAux-v0 框架梳理

本文档基于当前代码配置整理，参考上一份 `g1_stair_terrain_aux_framework.md` 的结构，但本任务不是 Stair 专用版本，而是完整 rough terrain + TerrainAux 版本。

主要涉及文件：

- 任务注册：`instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/__init__.py`
- TerrainAux 增量配置：`instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_terrain_aux_cfg.py`
- G1 Target AMP 基础配置：`instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py`
- 通用 Parkour MDP：`instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`
- TerrainAux 训练配置：`instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg_terrain_aux.py`
- TerrainAux actor-critic 实现：`instinct_rl/modules/terrain_aux_actor_critic.py`

## 1. 一句话概览

`Instinct-Parkour-Target-Amp-G1-TerrainAux-v0` 是一个完整 rough terrain 上的 G1 目标点 parkour 任务。它使用目标点生成速度命令，用 PPO + AMP/WASABI 学习运动控制，同时通过 `terrain_aux` 辅助监督让深度图编码器重建机器人前方/脚附近的局部相对高度图。

和 `Instinct-Parkour-Target-Amp-G1-Stair-TerrainAux-v0` 相比：

- 本任务使用完整 `ROUGH_TERRAINS_CFG`，包含平地、gap、楼梯、box、slope 等 10 类地形。
- 不经过 `G1ParkourStairEnvCfg`，因此没有 Stair 版本的地形子集裁剪。
- TerrainAux scanner、深度编码器、MoE policy、AMP/WASABI 训练配置与 Stair-TerrainAux 版本一致。

## 2. 任务注册与继承链路

注册入口：

```python
gym.register(
    id="Instinct-Parkour-Target-Amp-G1-TerrainAux-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "...g1_parkour_target_amp_terrain_aux_cfg:G1ParkourTerrainAuxEnvCfg",
        "instinct_rl_cfg_entry_point": "...agents.instinct_rl_amp_cfg_terrain_aux:G1ParkourTerrainAuxPPORunnerCfg",
    },
)
```

环境继承关系：

```text
ParkourEnvCfg
  -> G1ParkourRoughEnvCfg
      -> G1ParkourEnvCfg
          -> G1ParkourTerrainAuxEnvCfg
```

每层含义：

| 层级 | 作用 |
| --- | --- |
| `ParkourEnvCfg` | 定义通用 scene、观测、动作、命令、奖励、终止、事件、课程 |
| `G1ParkourRoughEnvCfg` | 绑定 G1 机器人、完整 rough terrain、深度相机碰撞对象、motion reference |
| `G1ParkourEnvCfg` | 通过 `ShoeConfigMixin` 替换为带鞋 URDF，并修正脚底高度偏置 |
| `G1ParkourTerrainAuxEnvCfg` | 增加局部地形高度图 scanner 和 `terrain_aux` 观测组 |

训练配置：

```text
G1ParkourTerrainAuxPPORunnerCfg
  policy    = TerrainAuxMoEPolicyCfg
  algorithm = TerrainAuxAmpAlgoCfg
```

## 3. 机器人与动作空间

机器人配置来自 `g1_parkour_target_amp_cfg.py`：

- 基础资产：`G1_29DOF_TORSOBASE_POPSICLE_CFG`
- 当前任务使用带鞋 URDF：
  - `g1_29dof_torsoBase_popsicle_with_shoe.urdf`
- 初始 base 高度：
  - `G1_CFG.init_state.pos = (0.0, 0.0, 0.9)`
- `merge_fixed_joints = True`
- actuators：
  - `beyondmimic_g1_29dof_delayed_actuators`

`ShoeConfigMixin` 的影响：

- `self.scene.robot` 替换为 `G1_with_shoe_CFG`
- `leg_volume_points` 的 z 范围改为 `[-0.063, -0.023]`
- `feet_at_plane` 奖励的 `height_offset` 改为 `0.058`

动作配置：

```python
joint_pos = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=[".*"],
    scale=beyondmimic_action_scale,
    use_default_offset=True,
)
```

因此：

- 动作类型：全关节 position target offset
- 动作维度：G1 29 DoF，对应 29 维 action
- action scale：按 BeyondMimic actuator effort/stiffness 计算
- 使用默认姿态作为 action offset

## 4. 仿真与场景参数

`ParkourEnvCfg.__post_init__()` 中的核心参数：

| 参数 | 当前值 |
| --- | --- |
| 并行环境数 | `4096` |
| env spacing | `2.5` |
| episode length | `20.0 s` |
| sim dt | `0.005 s` |
| decimation | `4` |
| 控制周期 | `0.02 s` / 50 Hz |
| render interval | `4` |
| contact sensor update period | `0.005 s` |
| PhysX gpu max rigid patch count | `10 * 2**15` |
| PhysX gpu collision stack size | `2**29` |

地形 scene：

- terrain prim path：`/World/ground`
- terrain type：generator
- collision group：`-1`
- static/dynamic friction：`1.0`
- visual material：IsaacLab tiles material
- virtual obstacles：
  - `GreedyconcatEdgeCylinderCfg`
  - cylinder radius：`0.05`

## 5. 完整 Rough Terrain 配置

本任务使用 `ROUGH_TERRAINS_CFG`，没有 Stair 任务中的 `_build_stair_terrain_cfg()` 裁剪。

Terrain generator 总参数：

| 参数 | 当前值 |
| --- | --- |
| seed | `0` |
| tile size | `(8.0, 8.0)` |
| border width | `3` |
| num rows | `10` |
| num cols | `20` |
| horizontal scale | `0.05` |
| vertical scale | `0.005` |
| slope threshold | `1.0` |
| curriculum | `True` |
| use cache | `False` |

完整子地形列表：

| 子地形 | proportion | 主要特点 |
| --- | --- | --- |
| `perlin_rough` | `0.05` | Perlin 粗糙平地，噪声 `0.0-0.1` |
| `perlin_rough_stand` | `0.05` | 用于站立/原地转向的粗糙平地 |
| `square_gaps` | `0.10` | 方形 gap，距离 `0.1-0.7`，深度 `0.4-0.6` |
| `pyramid_stairs` | `0.15` | 正向金字塔楼梯，高度 `0.05-0.23`，step width `0.3` |
| `pyramid_stairs_high` | `0.10` | 高楼梯，高度 `0.05-0.45`，step width `1.5` |
| `pyramid_stairs_inv` | `0.15` | 反向/下行金字塔楼梯，高度 `0.05-0.23` |
| `pyramid_stairs_inv_high` | `0.10` | 高版本反向/下行楼梯，高度 `0.05-0.45` |
| `boxes` | `0.10` | 离散 box 障碍，20 个 obstacle |
| `mesh_boxes` | `0.10` | mesh 随机多 box，box mean 高度 `0.1-0.4` |
| `hf_pyramid_slope_inv` | `0.10` | 反向坡面，slope range `0.0-0.7` |

共同地形特征：

- 多数地形带墙概率：
  - `wall_prob=[0.3, 0.3, 0.3, 0.3]`
  - `wall_height=5.0`
  - `wall_thickness=0.05`
- 每类地形都有 `flat_patch_sampling["target"]`
  - 通常 `num_patches=50`
  - `max_height_diff=0.05`
  - 用于目标点命令采样

和 Stair 版本的核心差异：

```text
TerrainAux-v0:
  ROUGH_TERRAINS_CFG
  num_rows = 10
  num_cols = 20
  sub_terrains = 10 类完整地形

Stair-TerrainAux-v0:
  STAIR_TERRAINS_CFG
  num_rows = 6
  num_cols = 12
  sub_terrains = 6 类楼梯/平地子集
```

## 6. 目标点速度命令

命令项：

```python
base_velocity = mdp.PoseVelocityCommandCfg(...)
```

它不是简单采样速度，而是从地形 flat patch 中采样目标点，然后把目标点相对位置转换成 base frame 速度命令。

流程：

1. 从当前 terrain level/type 的 `flat_patches["target"]` 随机采样目标点。
2. 计算目标点相对机器人 root 的位置。
3. 用位置误差乘 `velocity_control_stiffness=2.0` 得到 XY 速度。
4. 用目标方向与当前 heading 的误差乘 `heading_control_stiffness=2.0` 得到 yaw 角速度。
5. 限幅到对应地形的速度范围。
6. 若目标距离小于 `target_dis_threshold=0.4`，命令置零。

通用参数：

| 参数 | 当前值 |
| --- | --- |
| resampling time | `8.0 - 12.0 s` |
| rel_standing_envs | `0.05` |
| only_positive_lin_vel_x | `True` |
| lin_vel_threshold | `0.0` |
| ang_vel_threshold | `0.0` |
| target_dis_threshold | `0.4` |

各地形速度范围：

| 地形 | lin_vel_x | lin_vel_y | ang_vel_z |
| --- | --- | --- | --- |
| `perlin_rough` | `0.45 - 1.0` | `0.0` | `-1.0 - 1.0` |
| `perlin_rough_stand` | `0.0` | `0.0` | `0.0` |
| `square_gaps` | `0.45 - 0.8` | `0.0` | `-1.0 - 1.0` |
| `pyramid_stairs` | `0.45 - 0.8` | `0.0` | `-1.0 - 1.0` |
| `pyramid_stairs_high` | `0.45 - 0.8` | `0.0` | `-1.0 - 1.0` |
| `pyramid_stairs_inv` | `0.45 - 0.8` | `0.0` | `-1.0 - 1.0` |
| `pyramid_stairs_inv_high` | `0.45 - 0.8` | `0.0` | `-1.0 - 1.0` |
| `boxes` | `0.45 - 0.8` | `0.0` | `-1.0 - 1.0` |
| `mesh_boxes` | `0.45 - 0.8` | `0.0` | `-1.0 - 1.0` |
| `hf_pyramid_slope_inv` | `0.45 - 0.8` | `0.0` | `-1.0 - 1.0` |

补充：

- `random_velocity_terrain=["perlin_rough_stand"]`
- 对 `perlin_rough_stand`，线速度保持 0，但 yaw 可从全局 `ranges.ang_vel_z=(-1.0, 1.0)` 采样，小幅 yaw 会被阈值逻辑置零。

## 7. 传感器系统

### 7.1 深度相机

深度相机是 policy/critic 的主要外感知输入。

配置：

- 类型：`NoisyGroupedRayCasterCameraCfg`
- prim path：`{ENV_REGEX_NS}/Robot/torso_link`
- mesh prim paths：
  - `/World/ground`
  - G1 所有主要 link
- ray alignment：`yaw`
- 原始分辨率：`64 x 36`
- FOV：
  - horizontal `89.51 deg`
  - vertical `58.29 deg`
- data type：`distance_to_image_plane`
- update period：`0.02 s`
- depth clipping behavior：`max`
- min distance：`0.1`

噪声和预处理：

```text
CropAndResize(crop_region=(18, 0, 16, 16))
  -> 36x64 裁成 18x32
GaussianBlur(kernel_size=3, sigma=1)
DepthNormalization(depth_range=(0.0, 2.5), output_range=(0.0, 1.0))
```

深度历史：

- sensor history：`37`
- observation 输出帧数：`8`
- `history_skip_frames=5`
- `delayed_frame_ranges=(0, 1)`

因此深度观测形状：

```text
depth_image: (8, 18, 32)
```

### 7.2 足端高度扫描

用于 `feet_at_plane` 奖励：

| scanner | 挂载 link | offset | pattern |
| --- | --- | --- | --- |
| `left_height_scanner` | `left_ankle_roll_link` | `(0.04, 0.0, 20.0)` | `resolution=0.12, size=[0.12, 0.0]` |
| `right_height_scanner` | `right_ankle_roll_link` | `(0.04, 0.0, 20.0)` | `resolution=0.12, size=[0.12, 0.0]` |

update period：`0.02 s`

### 7.3 接触与腿部体积点

接触传感器：

- prim path：`{ENV_REGEX_NS}/Robot/.*`
- history length：`3`
- `track_air_time=True`

腿部体积点：

- prim path：`{ENV_REGEX_NS}/Robot/.*_ankle_roll_link`
- 点数：`10 x 5 x 2`
- x 范围：`-0.025 .. 0.12`
- y 范围：`-0.03 .. 0.03`
- 鞋底修正后 z 范围：`-0.063 .. -0.023`

### 7.4 TerrainAux 局部高度图扫描

`TerrainAuxConfigMixin` 新增 scanner：

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

参数总结：

| 参数 | 当前值 |
| --- | --- |
| scanner 名称 | `terrain_height_map_scanner` |
| 挂载 link | `torso_link` |
| 前向偏移 | `0.45 m` |
| 覆盖区域 | `1.2 m x 0.96 m` |
| 分辨率 | `0.12 m` |
| 输出维度 | `99`，约 `11 x 9` grid |
| ray alignment | yaw |
| update period | `0.02 s` |

高度图观测项：

```python
height_map = ObsTerm(
    func=mdp.local_terrain_height_map,
    params={
        "sensor_cfg": SceneEntityCfg("terrain_height_map_scanner"),
        "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
        "support_height_offset": 0.058,
        "min_height": -0.5,
        "max_height": 0.8,
    },
)
```

`local_terrain_height_map` 计算逻辑：

1. 读取 scanner ray hit 的世界高度。
2. 取左右 ankle roll link 的较低高度。
3. 支撑平面高度为 `min_foot_height - 0.058`。
4. ray 高度减支撑平面高度，得到相对高度。
5. 非有限值替换为 0。
6. clamp 到 `[-0.5, 0.8]`。

该设计让辅助标签表达“相对脚底支撑面的局部地形形状”，而不是世界绝对高度。

## 8. 观测组与维度

当前任务观测组：

```text
policy
critic
amp_policy
amp_reference
terrain_aux
```

### 8.1 Policy 观测

`policy` 组用于 actor，开启 corruption，不把各项直接 concat 成匿名向量，而是保留 component 名称供 encoder/MoE 使用。

| component | 历史 | 噪声/scale | 维度 |
| --- | --- | --- | --- |
| `base_ang_vel` | 8 | uniform `[-0.2, 0.2]`, scale `0.25` | `24` |
| `projected_gravity` | 8 | uniform `[-0.05, 0.05]` | `24` |
| `velocity_commands` | 8 | 无 | `24` |
| `joint_pos` | 8 | uniform `[-0.01, 0.01]` | `232` |
| `joint_vel` | 8 | uniform `[-0.5, 0.5]`, scale `0.05` | `232` |
| `actions` | 8 | 无 | `232` |
| `depth_image` | 8 | 相机噪声/延迟 | `(8,18,32)` |

原始 flatten 维度：

```text
24 + 24 + 24 + 232 + 232 + 232 + 8*18*32 = 5376
```

深度图经 encoder 替换为 96 维 latent 后，actor MoE 输入维度：

```text
24 + 24 + 24 + 232 + 232 + 232 + 96 = 864
```

### 8.2 Critic 观测

`critic` 组不开 corruption，并额外包含 privileged `base_lin_vel`。

| component | 历史 | scale | 维度 |
| --- | --- | --- | --- |
| `base_lin_vel` | 8 | 无 | `24` |
| `base_ang_vel` | 8 | `0.25` | `24` |
| `projected_gravity` | 8 | 无 | `24` |
| `velocity_commands` | 8 | 无 | `24` |
| `joint_pos` | 8 | 无 | `232` |
| `joint_vel` | 8 | `0.05` | `232` |
| `actions` | 8 | 无 | `232` |
| `depth_image` | 8 | 相机噪声/延迟 | `(8,18,32)` |

原始 flatten 维度：

```text
24 + 24 + 24 + 24 + 232 + 232 + 232 + 4608 = 5400
```

深度图经 encoder 替换为 96 维 latent 后，critic MoE 输入维度：

```text
24 + 24 + 24 + 24 + 232 + 232 + 232 + 96 = 888
```

### 8.3 AMP 观测

AMP discriminator 使用：

- `amp_policy`：机器人当前 rollout 的状态序列
- `amp_reference`：运动参考中的 expert 状态序列

两组结构对齐：

| component | 历史长度 | 说明 |
| --- | --- | --- |
| `projected_gravity` | 10 | 重力投影 |
| `joint_pos_rel` | 10 | 相对默认关节位置 |
| `joint_vel` | 10 | 关节速度，scale `0.05` |
| `base_lin_vel` | 10 | base 线速度 |
| `base_ang_vel` | 10 | base 角速度 |

按 29 DoF 展平，单组 AMP state 约为：

```text
30 + 290 + 290 + 30 + 30 = 670
```

### 8.4 TerrainAux 观测

`terrain_aux` 不作为 actor 推理输入，而是训练期辅助监督标签：

```text
terrain_aux.height_map: (99,)
enable_corruption = False
concatenate_terms = True
```

PPO rollout storage 会额外保存该组，因为算法配置中有：

```python
auxiliary_observation_group_names = ["terrain_aux"]
```

## 9. 网络结构

当前 policy class：

```python
class_name = "instinct_rl.modules.terrain_aux_actor_critic:TerrainAuxEncoderMoEActorCritic"
```

整体结构：

```text
policy obs / critic obs
  -> ParallelLayer depth encoder
  -> encoded observation
  -> MoE Actor-Critic
  -> action / value

encoded policy depth latent
  -> terrain_aux reconstruction head
  -> 99d local height map prediction
```

### 9.1 深度 temporal encoder

`DepthEncoderTemporalTerrainCfg`：

| 参数 | 当前值 |
| --- | --- |
| class | `ConvTemporalTransformerHeadModel` |
| input component | `depth_image` |
| input shape | `(8, 18, 32)` |
| output size | `96` |
| CNN channels | `[16, 32, 64, 128]` |
| CNN kernels | `[3, 3, 3, (3, 4)]` |
| CNN strides | `[2, 2, 2, 1]` |
| CNN paddings | `[1, 1, 1, 0]` |
| d_model | `128` |
| num heads | `4` |
| num layers | `1` |
| FFN dim | `256` |
| dropout | `0.1` |
| activation | `relu` |
| nonlinearity | `ReLU` |
| norm_first | `True` |
| temporal_pool | `latest` |
| temporal pos embedding | `True` |

输出 component：

```text
parallel_latent_0_depth_encoder: (96,)
```

由于 ParallelLayer 默认 `takeout_input_components=True`，原始 `depth_image` 会从后续 MLP/MoE 输入中移除，替换为 96 维 latent。

Actor 和 critic 分别拥有独立 encoder：

```python
encoder_configs = EncoderConfigs()
critic_encoder_configs = EncoderConfigs()
```

### 9.2 MoE Actor-Critic

`TerrainAuxMoEPolicyCfg`：

| 参数 | 当前值 |
| --- | --- |
| num_moe_experts | `4` |
| actor hidden dims | `[256, 128, 64]` |
| critic hidden dims | `[256, 128, 64]` |
| activation | `elu` |
| init_noise_std | `1.0` |
| moe_gate_hidden_dims | `[128]` |

Actor 数据流：

```text
encoded policy obs (864)
  -> MoE gate
  -> 4 expert actor MLPs
  -> action mean (29)
```

Critic 数据流：

```text
encoded critic obs (888)
  -> MoE gate
  -> 4 expert critic MLPs
  -> value
```

当前代码状态：

```python
moe_actor_gate_component_names = ACTOR_MOE_GATE_COMPONENT_NAMES
moe_critic_gate_component_names = CRITIC_MOE_GATE_COMPONENT_NAMES
```

因此当前 TerrainAux 任务默认启用 gate-slice，gate 不再看完整编码后观测，而只看任务相关的低维组件：

```text
actor gate components:
  projected_gravity
  velocity_commands
  base_ang_vel
  parallel_latent_0_depth_encoder

critic gate components:
  base_lin_vel
  base_ang_vel
  projected_gravity
  velocity_commands
  parallel_latent_0_depth_encoder
```

对应 gate 输入维度为：

```text
actor gate input  = 168
critic gate input = 192
```

CLI 仍保留 `--gate_slice` 和 `--no_gate_slice`。默认不传参数时使用配置文件中的 gate-slice；若显式加入 `--no_gate_slice`，会把 actor/critic gate component names 覆盖为 `None`，切回 full-gate：

```text
actor gate input  = 864
critic gate input = 888
```

这会改变 checkpoint 中 gate 层权重形状。因此训练、测试、导出 ONNX 时必须保持一致：默认 gate-slice checkpoint 不应在测试时额外加 `--no_gate_slice`；如果用 `--no_gate_slice` 训练了 full-gate checkpoint，测试和导出时也应继续加 `--no_gate_slice`。

## 10. TerrainAux 辅助头

辅助头实现于 `TerrainAuxEncoderMoEActorCritic.compute_auxiliary_losses()`。

输入：

```text
parallel_latent_0_depth_encoder: 96d
```

网络：

```text
Linear(96, 96)
ELU
Linear(96, 99)
```

目标：

```text
aux_obs["terrain_aux"] -> reshape 成 prediction 形状
```

损失：

```python
F.smooth_l1_loss(prediction, target, beta=0.05)
```

配置：

| 参数 | 当前值 |
| --- | --- |
| terrain_aux_group_name | `terrain_aux` |
| terrain_aux_latent_component_name | `parallel_latent_0_depth_encoder` |
| terrain_aux_output_shape | `(99,)` |
| terrain_aux_hidden_dims | `[96]` |
| terrain_aux_activation | `elu` |
| terrain_aux_loss_func | `smooth_l1` |
| terrain_aux_smooth_l1_beta | `0.05` |
| terrain_reconstruction_loss_coef | `0.1` |

记录的辅助统计：

- `terrain_aux_abs_error`
- `terrain_aux_target_std`

这个头只在训练 update 时使用。推理/部署时，actor 不需要 `terrain_aux` 标签，也不需要输出高度图。

## 11. PPO + AMP/WASABI 算法

Runner：

```python
class G1ParkourTerrainAuxPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 5000
    experiment_name = "g1_parkour_terrain_aux"
    empirical_normalization = False
```

Algorithm：

```python
class TerrainAuxAmpAlgoCfg(AmpAlgoCfg):
    auxiliary_observation_group_names = ["terrain_aux"]
    terrain_reconstruction_loss_coef = 0.1
```

PPO 主参数：

| 参数 | 当前值 |
| --- | --- |
| class_name | `WasabiPPO` |
| value_loss_coef | `1.0` |
| use_clipped_value_loss | `True` |
| clip_param | `0.2` |
| entropy_coef | `0.006` |
| num_learning_epochs | `5` |
| num_mini_batches | `4` |
| learning_rate | `1e-3` |
| schedule | `adaptive` |
| gamma | `0.99` |
| lam | `0.95` |
| desired_kl | `0.01` |
| max_grad_norm | `1.0` |

AMP/WASABI discriminator：

| 参数 | 当前值 |
| --- | --- |
| hidden sizes | `[1024, 512]` |
| nonlinearity | `ReLU` |
| discriminator_reward_coef | `0.25` |
| discriminator_reward_type | `quad` |
| discriminator_loss_func | `MSELoss` |
| discriminator_gradient_penalty_coef | `5.0` |
| optimizer | `AdamW` |
| discriminator lr | `1e-4` |
| weight decay coef | `3e-4` |
| logit weight decay coef | `0.04` |

训练中有两类优化：

```text
PPO / actor-critic update:
  clipped surrogate loss
  + value loss
  + entropy regularization
  + terrain reconstruction auxiliary loss

AMP / discriminator update:
  discriminator actor/reference loss
  + gradient penalty
  + weight decay
  + logit weight decay
```

PPO 环境 reward 还会被加上 discriminator auxiliary reward：

```text
transition reward += 0.25 * discriminator_reward
```

## 12. Motion Reference

运动参考用于 AMP discriminator 的 expert/reference 状态。

配置：

| 项目 | 当前值 |
| --- | --- |
| motion path | `/home/you/instinct_rl/instinctlab/data/parkour_motion_reference` |
| filtered selection | `parkour_motion_without_run.yaml` |
| motion_start_from_middle_range | `[0.0, 0.9]` |
| motion_start_height_offset | `0.0` |
| ensure_link_below_zero_ground | `False` |
| frame_interval_s | `0.02` |
| update_period | `0.02` |
| num_frames | `10` |
| motion buffer | `"run_walk": AmassMotionCfg()` |
| interpolation | `motion_interpolate_bilinear` |
| velocity estimation | `frontward` |

关注 link：

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

同时配置了左右对称增强：

- link mapping
- joint mapping
- joint reverse sign buffer

## 13. 奖励结构

奖励使用 `MultiRewardCfg`，当前只有一个 reward group：`G1Rewards`。

任务奖励：

| reward | weight | 说明 |
| --- | --- | --- |
| `track_lin_vel_xy_exp` | `2.0` | 跟踪 XY 线速度 |
| `track_ang_vel_z_exp` | `2.0` | 跟踪 yaw 角速度 |
| `heading_error` | `-1.0` | 惩罚朝向/yaw 相关误差 |
| `dont_wait` | `-0.5` | 有前进命令时惩罚停滞 |
| `is_alive` | `3.0` | 存活奖励 |
| `stand_still` | `-0.3` | 零命令时约束回默认姿态 |

步态和足端奖励：

| reward | weight | 说明 |
| --- | --- | --- |
| `volume_points_penetration` | `-4.0` | 惩罚腿部体积点穿入虚拟障碍 |
| `feet_air_time` | `0.5` | 鼓励合理摆腿/单脚支撑 |
| `feet_slide` | `-0.4` | 惩罚接触足端滑动 |
| `feet_flat_ori` | `-0.4` | 惩罚接触足端姿态不平 |
| `feet_at_plane` | `-0.1` | 接触足高度与地面扫描平面约束 |
| `feet_close_xy` | `0.4` | 约束左右脚横向距离过近 |

姿态、能耗和关节正则：

| reward | weight |
| --- | --- |
| `joint_deviation_hip` | `-0.5` |
| `ang_vel_xy_l2` | `-0.05` |
| `dof_torques_l2` | `-1.5e-7` |
| `dof_acc_l2` | `-1.25e-7` |
| `dof_vel_l2` | `-1e-4` |
| `action_rate_l2` | `-0.005` |
| `flat_orientation_l2` | `-3.0` |
| `pelvis_orientation_l2` | `-3.0` |
| `energy` | `-5e-5` |
| `freeze_upper_body` | `-0.004` |

安全项：

| reward | weight | 说明 |
| --- | --- | --- |
| `dof_pos_limits` | `-1.0` | 关节位置限制 |
| `dof_vel_limits` | `-1.0` | 关节速度限制，soft ratio `0.9` |
| `torque_limits` | `-0.01` | torque ratio 超限惩罚，limit ratio `0.8` |
| `undesired_contacts` | `-1.0` | 非足端接触惩罚 |

## 14. 终止、事件与课程

终止条件：

| termination | 参数 | 说明 |
| --- | --- | --- |
| `time_out` | time out | episode 到时 |
| `terrain_out_bound` | `distance_buffer=2.0` | 离开 terrain 边界 |
| `base_contact` | torso contact threshold `1.0` | 躯干接触终止 |
| `bad_orientation` | `limit_angle=1.0` | 姿态过差 |
| `root_height` | `minimum_height=0.5` | root 高度过低 |
| `dataset_exhausted` | motion reference | reference 数据耗尽 |

事件：

| event | mode | 参数/说明 |
| --- | --- | --- |
| `physics_material` | startup | 摩擦/反弹随机化 |
| `reset_base` | reset | root pose/velocity 随机重置 |
| `register_virtual_obstacles` | startup | 给 leg volume points 注册虚拟障碍 |
| `reset_robot_joints` | reset | joint position offset `[-0.15, 0.15]`，velocity 0 |

课程：

```python
terrain_levels = CurrTerm(
    func=mdp.tracking_exp_vel,
    params={"lin_vel_threshold": (0.3, 0.6), "ang_vel_threshold": (0.0, 0.0)},
)
```

地形难度随速度跟踪表现推进。

## 15. 数据流框架

```text
Full rough terrain generator
  -> 10 类 terrain / curriculum levels / target flat patches
  -> PoseVelocityCommand 采样目标点并生成速度命令

Simulation step
  -> proprioception history
  -> delayed depth image history
  -> command history
  -> AMP policy/reference state history
  -> TerrainAux local height map label

Policy path
  depth_image (8,18,32)
    -> ConvTemporalTransformer
    -> 96d depth latent
  proprio/action/command history + depth latent
    -> 864d encoded actor input
    -> 4-expert MoE actor
    -> 29d joint position action

Critic path
  privileged base_lin_vel + policy-like observations
    -> separate depth encoder
    -> 888d encoded critic input
    -> 4-expert MoE critic
    -> value

AMP path
  amp_policy vs amp_reference
    -> discriminator
    -> discriminator reward + discriminator loss

TerrainAux path
  actor depth latent
    -> reconstruction head
    -> predicted 99d local height map
    -> SmoothL1 against raycast height map
```

## 16. 与 Stair-TerrainAux 版本的关键区别

| 项目 | TerrainAux-v0 | Stair-TerrainAux-v0 |
| --- | --- | --- |
| env cfg | `G1ParkourTerrainAuxEnvCfg` | `G1ParkourStairTerrainAuxEnvCfg` |
| 父类主体 | `G1ParkourEnvCfg` | `G1ParkourStairEnvCfg` |
| terrain cfg | `ROUGH_TERRAINS_CFG` | `STAIR_TERRAINS_CFG` |
| terrain rows/cols | `10 x 20` | `6 x 12` |
| terrain 类型 | 10 类完整 rough terrain | 6 类楼梯/平地子集 |
| velocity ranges | 包含 gap、box、slope 等全部地形 | 仅保留 stair 子集对应范围 |
| TerrainAux scanner | 相同 | 相同 |
| depth encoder | 相同 | 相同 |
| MoE policy | 相同 | 相同 |
| AMP runner | 相同 | 相同 |

可以把当前任务理解为：

```text
基础 G1 Target AMP parkour
  + 带鞋 G1
  + 完整 rough terrain curriculum
  + 深度 temporal encoder
  + TerrainAux local height-map auxiliary reconstruction
```

## 17. 当前代码注意点

1. `g1_parkour_target_amp_terrain_aux_cfg.py` 同时定义了 full rough terrain 的 TerrainAux 类和 Stair TerrainAux 类。本任务使用的是 `G1ParkourTerrainAuxEnvCfg`，不是 `G1ParkourStairTerrainAuxEnvCfg`。

2. `TERRAIN_AUX_OUTPUT_SHAPE = (99,)` 在环境配置和训练配置中各有一份。若以后改 `TERRAIN_AUX_GRID_RESOLUTION` 或 `TERRAIN_AUX_GRID_SIZE`，需要同步修改 policy 辅助头输出 shape。

3. 当前 TerrainAux policy 的默认 MoE gate 已经启用 gate-slice：

```text
actor gate input  = 168
critic gate input = 192
```

如果命令行加入 `--no_gate_slice`，`instinctlab/scripts/instinct_rl/cli_args.py` 会把 `moe_actor_gate_component_names` / `moe_critic_gate_component_names` 覆盖为 `None`，gate 输入切回 actor `864`、critic `888`。

4. gate-slice 是 checkpoint 结构相关配置。训练、play、ONNX 导出时应保持一致：默认 gate-slice checkpoint 不加 `--no_gate_slice`；full-gate checkpoint 需要继续加 `--no_gate_slice`。否则 actor/critic gate 权重形状会不匹配。

5. `terrain_aux` 是训练期辅助标签，不是部署时必需输入。部署 actor 时需要深度图输入和 encoder，但不需要 height-map scanner 标签。

6. 本任务由于保留完整 rough terrain，比 Stair-TerrainAux 覆盖更宽：包括 gap、box、slope、上下楼梯和平地。因此训练到的视觉 latent 更偏通用地形表征，而不是楼梯专用表征。
