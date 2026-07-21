# TerrainAux 深度编码器改进方案

本文档围绕当前 `Instinct-Parkour-Target-Amp-G1-TerrainAux-v0` 的多帧深度编码器，整理两条可推进路线，并明确当前优先实现的方案 2：用当前帧深度 latent 与 command 作为 query，对 8 帧深度 memory 做 cross-attention。

## 1. 当前基线

当前深度编码器为：

```text
depth_image: (8, 18, 32)
  -> shared frame-wise CNN
  -> 8 tokens, each 128-d
  -> learned temporal positional embedding
  -> 1-layer temporal Transformer encoder
  -> latest token pooling
  -> 96-d depth latent
```

它已经是时序模型，并且已经有注意力机制。注意力发生在 8 个时间 token 之间，是 temporal self-attention；空间结构主要由每帧 CNN 压缩，不是 ViT 式 spatial attention。

当前优点：

- 轻量，适合高并行 RL 训练。
- Transformer 能让最新 token 读取历史帧信息。
- 输出 96 维 latent，可直接接入 MoE actor/critic 和 TerrainAux head。

当前限制：

- 每帧最终被压缩成单个 token，空间细节较早丢失。
- temporal position embedding 只表示第几帧，不表示该帧对应的机器人姿态、相机运动或速度命令。
- `latest pooling` 隐式地让最后一帧读取历史，但 query 不显式包含当前 command，因此不能主动按控制意图检索历史地形 memory。

## 2. 路线一：Ego-Motion / Proprioception Conditioned Tokens

核心想法：给每张深度图对应的 token 加入该时刻的机器人状态编码。这里不建议叫传统 position encoding，更准确是：

```text
ego-motion conditioning
robot-state conditioning
```

候选结构：

```text
For each frame i:
  d_i = CNN(D_i)
  r_i = MLP(robot_state_i)
  x_i = Linear([d_i, r_i, temporal_pos_i])

x_{1:T} -> temporal Transformer -> z_d
```

可选 robot-state：

- `projected_gravity_i`
- `base_ang_vel_i`
- `velocity_command_i`
- `joint_pos_i`
- `joint_vel_i`
- `previous_action_i`

建议从最小状态开始：

```text
projected_gravity + base_ang_vel + velocity_command
```

优点：

- 减少 egocentric depth 的几何歧义。
- 对 pitch/roll、转向、速度命令变化更敏感。
- 改动较小，训练风险低。

风险：

- 如果加入过多本体状态，depth latent 可能偷懒，用 proprio 拟合 TerrainAux label，而不是学习真正地形结构。
- 如果 actor 和 TerrainAux head 都过度依赖 proprio，视觉表征的泛化性可能下降。

建议作为后续 ablation：

```text
V0: depth + temporal position
V1: depth + projected_gravity
V2: depth + projected_gravity + base_ang_vel
V3: depth + projected_gravity + base_ang_vel + command
```

## 3. 路线二：Query-Based Temporal Terrain Memory

核心想法：先用 8 帧深度图构建 temporal memory，再用当前帧深度 latent 与当前 command 作为 query，主动从历史 memory 中检索与当前控制相关的地形信息。

目标结构：

```text
Depth frames D_{t-7:t}
  -> shared CNN
  -> depth tokens x_{t-7:t}
  -> temporal position embedding
  -> temporal Transformer encoder
  -> memory M

Current query:
  q = MLP([CNN(D_t), command_t])

Cross attention:
  z = CrossAttention(Q=q, K=M, V=M)

Output:
  z -> 96-d depth latent
```

本路线和当前 `latest pooling` 的关系：

- 当前基线：最后一帧 token 在 self-attention 后被取出，属于隐式 current-token pooling。
- 新方案：显式构造 current query，再对 self-attention memory 做 cross-attention。
- 新方案更可解释，也更容易加入 command，使视觉编码按当前运动意图读取历史。

为什么 query 使用“当前帧 depth latent + command”：

- 当前帧 depth latent 表示机器人此刻可见的局部几何。
- command 表示当前控制意图，例如前进、转向、站立。
- 历史 memory 中包含过去视角看到、当前可能不可见但与落脚相关的地形信息。
- Cross-attention 可以让当前控制意图主动检索历史地形记忆。

## 4. 本次实现方案

新增 encoder：

```text
DepthCommandCrossAttentionHeadModel
```

输入 component：

```text
velocity_commands
depth_image
```

其中：

- `depth_image`: `(8, 18, 32)`
- `velocity_commands`: 8 帧历史，展平后为 24 维
- 当前 command 取最后一帧的 3 维命令

模型流程：

```text
depth_image
  -> shared CNN tokenizer
  -> depth tokens, shape (B, 8, 128)
  -> temporal position embedding
  -> temporal Transformer self-attention
  -> memory, shape (B, 8, 128)

latest depth token + latest command
  -> query MLP
  -> query, shape (B, 1, 128)

query cross-attends memory
  -> cross-attended terrain latent
  -> output MLP
  -> 96-d latent
```

输出 component 仍保持：

```text
parallel_latent_0_depth_encoder: (96,)
```

为了保留旧功能：

- 旧任务 `Instinct-Parkour-Target-Amp-G1-TerrainAux-v0` 不变。
- 旧 agent config `instinct_rl_amp_cfg_terrain_aux.py` 不变。
- 新增 agent config：
  - `instinct_rl_amp_cfg_terrain_aux_cross_attn.py`
- 新增任务入口：
  - `Instinct-Parkour-Target-Amp-G1-TerrainAux-CrossAttn-v0`
  - `Instinct-Parkour-Target-Amp-G1-TerrainAux-CrossAttn-Play-v0`

## 5. 重要实现细节

### 5.1 保留 command 主干输入

Cross-attention encoder 需要读取 `velocity_commands`，但 actor/critic 主干也仍然应该看到原始 command。因此新增 `takeout_component_names` 支持：

```text
component_names = ["velocity_commands", "depth_image"]
takeout_component_names = ["depth_image"]
```

这表示：

- encoder 使用 command 和 depth。
- 后续 encoded obs 中只移除 `depth_image`。
- `velocity_commands` 仍保留在 actor/critic 输入里。

因此 actor/critic 编码后维度仍保持：

```text
actor input  = 864
critic input = 888
```

### 5.2 与默认 gate-slice 的关系

TerrainAux 系列配置默认启用 gate-slice。

因为新 encoder 仍输出同名 latent：

```text
parallel_latent_0_depth_encoder
```

所以 gate-slice 组件表无需修改：

```text
actor gate:
  projected_gravity
  velocity_commands
  base_ang_vel
  parallel_latent_0_depth_encoder

critic gate:
  base_lin_vel
  base_ang_vel
  projected_gravity
  velocity_commands
  parallel_latent_0_depth_encoder
```

默认 gate 输入维度仍为：

```text
actor gate  = 168
critic gate = 192
```

CLI 仍保留 `--gate_slice` 和 `--no_gate_slice`。默认不传参数时使用 gate-slice；只有需要兼容 full-gate checkpoint 时才使用 `--no_gate_slice`。

### 5.3 TerrainAux 辅助头保持不变

TerrainAux head 仍从：

```text
parallel_latent_0_depth_encoder
```

重建：

```text
local terrain height map, 99 dims
```

所以新方案可以直接和旧 TerrainAux loss 对比，不需要改辅助监督标签。

## 6. 推荐实验

最小 ablation：

```text
Baseline:
  Instinct-Parkour-Target-Amp-G1-TerrainAux-v0

CrossAttn:
  Instinct-Parkour-Target-Amp-G1-TerrainAux-CrossAttn-v0

Stair CrossAttn:
  Instinct-Parkour-Target-Amp-G1-Stair-TerrainAux-CrossAttn-v0

Full-gate compatibility:
  Any TerrainAux task + --no_gate_slice
```

重点观察：

- terrain reconstruction loss / abs error
- rough terrain success rate
- stairs / gaps / boxes 分地形成功率
- MoE gate entropy
- expert usage balance
- ONNX 导出后的推理稳定性

## 7. 论文表述建议

可以将方案命名为：

```text
Command-Queried Temporal Depth Encoder
```

或：

```text
Query-Based Temporal Terrain Memory
```

一句话方法描述：

> We first encode the depth history into a temporal terrain memory using self-attention. A command-conditioned current-depth query then cross-attends to this memory, producing a compact terrain latent for MoE locomotion control and auxiliary terrain reconstruction.
