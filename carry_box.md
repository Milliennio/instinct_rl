# Carry Box Parkour MVP

本文记录 G1 Parkour Stair TerrainAux 持箱遮挡 MVP 的实际改动、任务入口和验证方式。

## 目标

在不修改 PPO、TerrainAux loss、depth encoder 输入维度和 observation 结构的前提下，新增一个胸前固定箱子的 G1 楼梯 TerrainAux 任务：

- policy depth camera 会 raycast 到机器人自身和 carry box，产生真实几何遮挡。
- carry 机器人使用单独的上肢默认姿态，使双臂呈现捧箱姿态。
- `terrain_aux` height map scanner 仍然只 raycast `/World/ground`，作为 clean 地形监督。
- 训练配置继续复用 `G1ParkourTerrainAuxPPORunnerCfg`。

## 已改动文件

### URDF

新增：

```text
instinctlab/source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe_carry_box.urdf
```

另新增一个仅用于静态检查捧箱姿态的 preview URDF：

```text
instinctlab/source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe_carry_box_hold_preview.urdf
```

preview URDF 复制自 carry-box URDF，但将肩、肘、腕的捧箱关节改为 fixed，并把目标关节角近似 bake 到各 joint origin 的 `rpy` 中。它用于单独加载 URDF 检查外观，不用于训练、play 任务或 checkpoint 推理。

该文件复制自现有 shoe 版本：

```text
instinctlab/source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf
```

新增固定 link：

```xml
<link name="carry_box_link">
  ...
</link>

<joint name="carry_box_fixed_joint" type="fixed">
  <origin xyz="0.26 0.0 0.18" rpy="0 0 0"/>
  <parent link="torso_link"/>
  <child link="carry_box_link"/>
</joint>
```

箱体参数：

- size: `0.14 0.16 0.10`
- mass: `0.2`
- inertia: `ixx=0.00059, iyy=0.00049, izz=0.00075`
- fixed joint origin: `0.26 0.0 0.18`

### Robot cfg

修改：

```text
instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py
```

新增 `G1_with_shoe_carry_box_CFG`，复用 shoe 机器人配置并替换 URDF 路径：

```python
G1_with_shoe_carry_box_CFG = copy.deepcopy(G1_with_shoe_CFG)
G1_with_shoe_carry_box_CFG.spawn.asset_path = os.path.abspath(
    f"{__file_dir__}/../../urdf/g1_29dof_torsoBase_popsicle_with_shoe_carry_box.urdf"
)
G1_with_shoe_carry_box_CFG.spawn.merge_fixed_joints = False
```

`merge_fixed_joints = False` 用于保留 `carry_box_link`，保证 camera raycaster 能通过 link prim 命中箱子 visual。

carry 任务同时覆盖上肢默认关节，使机器人从捧箱姿态 reset。由于动作配置使用 `use_default_offset=True`，上肢 freeze reward 也会围绕这个默认姿态约束：

```python
G1_with_shoe_carry_box_CFG.init_state.joint_pos.update(
    {
        "left_shoulder_pitch_joint": -0.191986,   # -11 deg
        "right_shoulder_pitch_joint": -0.191986,  # -11 deg (mirrored)
        "left_shoulder_roll_joint": 0.15708,      # 9 deg
        "right_shoulder_roll_joint": -0.15708,    # -9 deg (mirrored)
        "left_shoulder_yaw_joint": -0.279253,     # -16 deg
        "right_shoulder_yaw_joint": 0.279253,     # 16 deg (mirrored)
        "left_elbow_joint": 0.0349066,            # 2 deg
        "right_elbow_joint": 0.0349066,           # 2 deg (same)
        "left_wrist_roll_joint": -1.41372,        # -81 deg
        "right_wrist_roll_joint": 1.41372,        # 81 deg (mirrored)
        "left_wrist_pitch_joint": 0.15708,        # 9 deg
        "right_wrist_pitch_joint": 0.15708,       # 9 deg (same)
        "left_wrist_yaw_joint": -0.715585,        # -41 deg
        "right_wrist_yaw_joint": 0.715585,        # 41 deg (mirrored)
    }
)
```

当前实现仍然是视觉捧箱加 torso fixed box，不是双手闭链抓握。这样能保持 locomotion 训练稳定，同时提供更真实的胸前遮挡和上肢外观。

### Preview URDF 可视化检查

新增 preview URDF：

```text
instinctlab/source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe_carry_box_hold_preview.urdf
```

该文件复制自 carry-box URDF，将所有捧箱关节的角度直接 bake 进各 joint `origin rpy` 中。所有关节保持 `type="revolute"`，不锁关节，仅用于单独加载 URDF 检查静态捧箱外观。

### Carry TerrainAux cfg

新增：

```text
instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_carry_terrain_aux_cfg.py
```

新增类：

- `G1ParkourStairCarryTerrainAuxEnvCfg`
- `G1ParkourStairCarryTerrainAuxEnvCfg_PLAY`

核心逻辑：

- 将 `self.scene.robot` 替换为 `G1_with_shoe_carry_box_CFG`。
- 将 `carry_box_link` 加入 `self.scene.camera.mesh_prim_paths`。
- 不修改 `terrain_height_map_scanner`，因此 TerrainAux target 仍只来自 `/World/ground`。
- Play 版本打开 camera debug 和 policy depth debug，便于确认遮挡。

### 任务注册

修改：

```text
instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/__init__.py
```

新增任务：

- `Instinct-Parkour-Target-Amp-G1-Stair-Carry-TerrainAux-v0`
- `Instinct-Parkour-Target-Amp-G1-Stair-Carry-TerrainAux-Play-v0`

两个任务都复用：

```text
instinctlab.tasks.parkour.config.g1.agents.instinct_rl_amp_cfg_terrain_aux:G1ParkourTerrainAuxPPORunnerCfg
```

## 验证命令

先用单环境 play 检查深度图遮挡、URDF prim 和仿真稳定性：

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py \
  --task=Instinct-Parkour-Target-Amp-G1-Stair-Carry-TerrainAux-Play-v0 \
  --num_envs=1 \
  --terrain_name=pyramid_stairs_inv \
  --terrain_level=2
```

示例：

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py \
  --task=Instinct-Parkour-Target-Amp-G1-Stair-Carry-TerrainAux-Play-v0 \
  --num_envs=1 \
  --terrain_name=pyramid_stairs_inv \
  --terrain_level=2 \
  --load_run=0612_5090
```

小规模训练跑通：

```bash
python instinctlab/scripts/instinct_rl/train.py \
  --headless \
  --task=Instinct-Parkour-Target-Amp-G1-Stair-Carry-TerrainAux-v0 \
  --num_envs=256
```

## 注意事项

- 如果 raycaster 报找不到 `carry_box_link/visuals`，优先确认 `G1_with_shoe_carry_box_CFG.spawn.merge_fixed_joints = False` 是否生效。
- 如果深度图里箱子遮挡不明显，优先调整 URDF 中 `carry_box_fixed_joint` 的 `origin xyz`。
- 如果双臂没有贴合箱子，优先微调 carry cfg 中的 shoulder/elbow/wrist 默认关节值。
- 如果只是想静态检查捧箱外观，可以加载 `g1_29dof_torsoBase_popsicle_with_shoe_carry_box_hold_preview.urdf`。不要把该 preview URDF 直接接到训练任务里。
- 如果仿真初始阶段爆炸或 contact 异常，检查 carry box collision 是否和 torso/arms 初始姿态重叠。
- 当前 MVP 不包含 occlusion mask、随机物体、自由抓握、闭链约束或 world model。
