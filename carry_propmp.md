按 2026-06-25 的现状看，我的判断很明确：**方向 3 更适合作为主线，方向 4 更适合作为增强模块或第二篇延展**。

**总体结论**

你现在已经有了一个很好的起点：`Hiking in the Wild` 式上楼梯任务 + 深度 Transformer + 训练期地形重建辅助头。下一步最有价值的不是泛泛加一个“大世界模型”，而是把问题收窄成：

> **持物/手臂自遮挡下，人形机器人如何利用历史视觉记忆重建脚下地形，并稳定上楼梯。**

这条线同时具备实用性、仿真可复现性、论文故事完整度，而且能直接继承你现有的辅助地形重建头。

**现状脉络**

近两年人形感知运动已经明显往“深度图/高程图重建”走：

- `Hiking in the Wild` 已经把 raw depth + proprioception 到动作的端到端人形 parkour 做到了很强的基线，强调不依赖外部状态估计。([arxiv.org](https://arxiv.org/abs/2601.07718))
- `DPL` 明确提出用 Cross-Attention Transformer 从 noisy depth 重建结构化地形表示，并做 self-occlusion-aware ray casting 和噪声建模，报告地形重建误差降低 30% 以上。([arxiv.org](https://arxiv.org/abs/2510.07152))
- `Gait-Adaptive Perceptive Humanoid Locomotion` 用下视深度相机 + compact U-Net 实时重建脚周围 height map，能做上下楼梯和 46 cm gap。([arxiv.org](https://arxiv.org/abs/2512.07464))
- `Omnidirectional Humanoid Locomotion on Stairs` 用 rolling point-cloud elevation map、时空置信度衰减和 self-protection zone，说明“时序地图/记忆”在楼梯任务上已经是强方向。([arxiv.org](https://arxiv.org/abs/2603.07928))
- 更关键的是，`RPL` 已经正面提到 upper-body motion 带来的 dynamic self-occlusion 和 payload 场景，并用多深度相机、dynamic robot mesh ray casting、random side masking 来增强鲁棒性，实机展示了带 2 kg payload 上下复杂地形。([arxiv.org](https://arxiv.org/abs/2602.03002))

所以方向 3 的空白不在“是否有人做地形重建”，而在更具体的切口：

> **单/少量胸前相机在持物时产生任务相关盲区，策略如何用历史深度记忆和语义遮挡 mask 恢复当前脚下/脚前地形。**

这个点仍然很有空间，尤其适合 G1 持箱上楼梯。

**方向 3 评估**

推荐指数：**9/10，建议作为主论文主线。**

优点：

- 和你当前实现高度连续：你已有地形重建辅助头，只需要从“当前帧重建”扩展到“遮挡区域 + 历史记忆重建”。
- 仿真很容易做出干净实验：双手 attach box/cylinder/basin，随机尺寸、材质、位置，让它真实挡住胸前深度。
- 评价指标很明确：上楼梯成功率、foot edge/unsafe contact、遮挡区域 height RMSE、盲区占比、泛化到未见物体尺寸/楼梯高度。
- 论文故事好讲：现有 perceptive locomotion 假设视野可用；真实搬运场景视野被自己和物体破坏；我们提出 occlusion-aware temporal terrain reconstruction。

风险：

- 需要和 `RPL` 区分。RPL 已经有 payload、自遮挡、多深度和 masking。你的差异应放在：**历史记忆重建盲区**，而不是只做随机 mask 或多相机。
- 如果只做“深度图随机遮挡增强”，新意会弱。必须把遮挡机制做成“由手臂/物体几何真实产生”，并显式报告 blind-zone reconstruction。

建议命名方向：

> **Occlusion-Aware Memory Terrain Reconstruction for Object-Carrying Humanoid Stair Locomotion**

核心实验组合：

1. 无持物，正常上楼梯 baseline。
2. 持物但不加遮挡鲁棒训练。
3. 持物 + random depth dropout/mask。
4. 持物 + semantic hand/object mask。
5. 持物 + semantic mask + temporal memory。
6. 持物 + semantic mask + temporal memory + terrain reconstruction auxiliary loss。

最后一组应该是你的主方法。

**方向 4 评估**

推荐指数：**6.5/10，适合做辅助，不建议现在作为主线。**

World Model 确实热。`DreamerV3` 证明了用 learned world model 想象未来场景可跨很多任务稳定学习。([arxiv.org](https://arxiv.org/abs/2301.04104)) `DayDreamer` 也展示了真实机器人上直接用 Dreamer 学习，包括四足机器人一小时内学会翻身、站立和行走。([arxiv.org](https://arxiv.org/abs/2206.14176)) 人形方面也已有 `World Model Reconstruction`，用 estimator 显式重建 world state，并让策略输入重建信息，在真实复杂地形完成 3.2 km 行走。([arxiv.org](https://arxiv.org/abs/2502.16230))

但对你当前任务，完整 Dreamer/World Model 主线有几个问题：

- 工程量大，会偏离你已有 PPO/teacher-student/辅助头框架。
- 对高频人形楼梯控制，world model 的 latent rollout 稳定性、延迟、训练调参都很重。
- 如果只是“预测下一帧深度图/未来脚端位置”，容易被审稿人认为是 auxiliary loss，而不是完整 world model。

更好的用法是把方向 4 收进方向 3：

> 在遮挡重建网络里加入 **future terrain / future foot placement auxiliary prediction**，让 memory latent 不只补全当前盲区，也具备短时前瞻能力。

具体可以加两个轻量头：

- `current_height_recon_head`：预测当前脚下局部 height map。
- `future_foot_contact_head`：预测未来 0.3-0.8 s 左右左右脚落点或 contact heatmap。
- 可选 `future_height_head`：预测 ego-motion 对齐后的下一帧/未来 N 帧局部 height map。

训练时用，推理时可以去掉，或者只保留 latent 给 policy。这样你能借 world model 的学术味道，但不会把项目拖进高风险大坑。

**我建议的最终路线**

主线选方向 3，标题和贡献围绕：

1. **Object-carrying induced occlusion benchmark**：G1 持物上楼梯，真实几何遮挡深度图。
2. **Semantic occlusion mask**：区分 terrain depth 与 robot/object interference。
3. **Temporal terrain memory**：用 GRU/Temporal Transformer/history buffer 从过去可见地形重建当前盲区。
4. **Auxiliary terrain and foothold prediction**：训练期用 privileged height map / foot placement 监督，推理去头或轻量保留。

方向 4 不单独开大坑，而是作为一句很漂亮的贡献：**predictive terrain belief / future foothold auxiliary learning**。这样最稳，也最容易从你现在的代码长出来。

请验证以下方案是否可行：
可以。**最小可行修改 MVP** 的目标应该非常窄：

> 在不改算法、不改网络、不改 TerrainAux 训练逻辑的前提下，让 G1 在楼梯 TerrainAux 任务中“胸前持一个固定箱子”，使 policy 深度图真实出现遮挡，而 TerrainAux 仍然用 clean 地形高度图监督。

也就是先证明：**同一个 Stair-TerrainAux 框架，在持物遮挡下能不能学会用历史深度恢复脚下地形。**

**MVP 不做什么**

第一版先不要做这些：

- 不做自由抓握。
- 不做双手闭链约束。
- 不做随机物体。
- 不改 depth encoder 输入通道。
- 不加 occlusion mask。
- 不加 world model。
- 不改 PPO / TerrainAux loss。

这样最小改动能最快得到可训练任务。

**改动 1：新增一个 carry-box URDF**

建议复制当前楼梯任务实际使用的 shoe URDF：

```text
instinctlab/source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf
```

新增：

```text
instinctlab/source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe_carry_box.urdf
```

在 URDF 里加一个固定 link，先挂到 `torso_link`，不要挂双手。位置可以先设在胸前偏下，后面用 depth 可视化微调：

```xml
<link name="carry_box_link">
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia ixx="0.0208" ixy="0" ixz="0" iyy="0.0168" iyz="0" izz="0.0342"/>
  </inertial>

  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.36 0.42 0.28"/>
    </geometry>
    <material name="carry_box_mat">
      <color rgba="0.7 0.5 0.25 1"/>
    </material>
  </visual>

  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.36 0.42 0.28"/>
    </geometry>
  </collision>
</link>

<joint name="carry_box_fixed_joint" type="fixed">
  <origin xyz="0.34 0.0 0.12" rpy="0 0 0"/>
  <parent link="torso_link"/>
  <child link="carry_box_link"/>
</joint>
```

注意：如果后面 raycaster 找不到 `carry_box_link/visuals`，就把该 asset 的 `merge_fixed_joints` 设成 `False`。固定关节被 merge 后，link 名可能会消失。

**改动 2：新增 carry robot cfg**

在：

```text
instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py
```

现有 `G1_with_shoe_CFG` 后面新增一个 cfg：

```python
G1_with_shoe_carry_box_CFG = copy.deepcopy(G1_with_shoe_CFG)
G1_with_shoe_carry_box_CFG.spawn.asset_path = os.path.abspath(
    f"{__file_dir__}/../../urdf/g1_29dof_torsoBase_popsicle_with_shoe_carry_box.urdf"
)
G1_with_shoe_carry_box_CFG.spawn.merge_fixed_joints = False
```

这里复用 shoe 版本，是因为当前 `ShoeConfigMixin` 会调整脚底 volume points 和 `feet_at_plane` 高度。不要突然切回普通 G1，否则楼梯表现会混入脚底模型变化。

**改动 3：新增 carry stair terrain aux 环境配置**

新建文件：

```text
instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_carry_terrain_aux_cfg.py
```

内容结构：

```python
from isaaclab.utils import configclass

from instinctlab.sensors import get_link_prim_targets
from instinctlab.assets.unitree_g1 import G1_29DOF_LINKS

from .g1_parkour_target_amp_cfg import G1_with_shoe_carry_box_CFG
from .g1_parkour_target_amp_terrain_aux_cfg import (
    G1ParkourStairTerrainAuxEnvCfg,
    G1ParkourStairTerrainAuxEnvCfg_PLAY,
)


CARRY_OBJECT_LINKS = ["carry_box_link"]


class CarryBoxConfigMixin:
    def apply_carry_box_config(self):
        self.scene.robot = G1_with_shoe_carry_box_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # The base G1 cfg already appends normal robot links.
        # We only need to add the carried object link.
        self.scene.camera.mesh_prim_paths.extend(
            get_link_prim_targets(CARRY_OBJECT_LINKS)
        )


@configclass
class G1ParkourStairCarryTerrainAuxEnvCfg(
    G1ParkourStairTerrainAuxEnvCfg,
    CarryBoxConfigMixin,
):
    def __post_init__(self):
        super().__post_init__()
        self.apply_carry_box_config()


@configclass
class G1ParkourStairCarryTerrainAuxEnvCfg_PLAY(
    G1ParkourStairTerrainAuxEnvCfg_PLAY,
    CarryBoxConfigMixin,
):
    def __post_init__(self):
        super().__post_init__()
        self.apply_carry_box_config()
        self.scene.camera.debug_vis = True
        self.observations.policy.depth_image.params["debug_vis"] = True
```

核心点是：**policy camera 要打到 carry box；TerrainAux 的 `terrain_height_map_scanner` 不改，继续只打 `/World/ground`。**

**改动 4：注册新任务**

在：

```text
instinctlab/source/instinctlab/instinctlab/tasks/parkour/config/g1/__init__.py
```

新增两个任务：

```python
gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Stair-Carry-TerrainAux-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{task_entry}.g1_parkour_target_amp_carry_terrain_aux_cfg:"
            "G1ParkourStairCarryTerrainAuxEnvCfg"
        ),
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_terrain_aux:"
            "G1ParkourTerrainAuxPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Stair-Carry-TerrainAux-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{task_entry}.g1_parkour_target_amp_carry_terrain_aux_cfg:"
            "G1ParkourStairCarryTerrainAuxEnvCfg_PLAY"
        ),
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_terrain_aux:"
            "G1ParkourTerrainAuxPPORunnerCfg"
        ),
    },
)
```

第一版直接复用现有 `G1ParkourTerrainAuxPPORunnerCfg`，因为 observation 结构没变，还是原来的 depth image + proprioception + TerrainAux target。

**改动 5：验证命令**

先 play 单环境看深度图有没有被箱子挡住：

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py \
  --task=Instinct-Parkour-Target-Amp-G1-Stair-Carry-TerrainAux-Play-v0 \
  --num_envs=1 \
  --terrain_name=pyramid_stairs_inv \
  --terrain_level=2
```

如果 raycaster 报找不到 `carry_box_link/visuals`，优先检查：

```python
G1_with_shoe_carry_box_CFG.spawn.merge_fixed_joints = False
```

如果深度图没有明显遮挡，调 URDF 里的 joint origin：

```xml
<origin xyz="0.34 0.0 0.12" rpy="0 0 0"/>
```

一般可以试：

```text
x: 0.28 - 0.45
z: 0.02 - 0.20
box size: 0.30-0.45, 0.35-0.55, 0.20-0.35
```

**改动 6：训练命令**

小规模先跑通：

```bash
python instinctlab/scripts/instinct_rl/train.py \
  --headless \
  --task=Instinct-Parkour-Target-Amp-G1-Stair-Carry-TerrainAux-v0 \
  --num_envs=256
```

确认无问题后再放大到 1024/4096。

**MVP 成功标准**

最小版本只看三件事：


1. 深度图里确实出现胸前物体/手臂遮挡。
2. `terrain_aux_abs_error` 正常下降，不 NaN。
3. carry 任务相比 clean stair 任务成功率下降，但 TerrainAux 版本能部分恢复。

这版跑通后，再做第二阶段：加 `terrain_camera` 生成 occlusion mask，升级 depth encoder 支持 `depth + mask` 双通道。当前先别加，MVP 就让任务站起来。



left_shoulder_pitch _joint -11
left_shoulder_rolljoint 9
left_shoulder_yaw_joint -16
left_elbow_joint 2
left_wrist_rolljoint -81
left_wrist_pitch_joint 9
left_wrist_yaw_joint -41
