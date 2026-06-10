这是我现在使用的网络（底层数据流匹配（自动完成）：根据代码库中 ParallelLayer 的逻辑，当把 class_name 指定为 TransformerHeadModel 时，temporal=True 会被触发，底层会自动将原本展平的 1D 历史观测数据恢复成 (Batch, Sequence_Length, Feature_Size) 的 3D 张量，你不需要去手动修改底层的观测张量重组代码。），训练好之后在仿真环境查看发现效果比之前的CNNencoder好，但是部署到实机后发现模型遇到台阶没有跨越行为，我我现在希望更改网络结构，使得我的网络输入和之前完全一样，需要使用CNN提取特征之后再展平输入transformer，而不是使用mlp。
以下是我网络更改前后的结构，请你首先输出我需要的网络结构的设计思路和网络结构详情，供我判断和确认
transformer encoder:instinct_rl_amp_cfg_transformer

Actor Encoder: ParallelLayer(1 blocks): ModuleDict(
  (depth_encoder): TransformerHeadModel(
    (input_layer): MlpModel(
      (model): Sequential(
        (0): Linear(in_features=576, out_features=256, bias=True)
        (1): ReLU()
      )
    )
    (tf_encoder): TransformerEncoder(
      (layers): ModuleList(
        (0): TransformerEncoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=512, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
      )
      (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    )
    (output_layer): MlpModel(
      (model): Sequential(
        (0): Linear(in_features=256, out_features=128, bias=True)
        (1): ReLU()
      )
    )
  )
)


之前的模型：cnn_encoder:instinct_rl_amp_cfg_cnn.py

输入(B,8,18,32)
Actor Encoder: ParallelLayer(1 blocks): ModuleDict(
  (depth_encoder): Conv2dHeadModel(
    (conv): Conv2dModel(
      (conv): Sequential(
        (0): Conv2d(8, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
      )
    )
    (head): MlpModel(
      (model): Sequential(
        (0): Linear(in_features=2304, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=128, bias=True)
        (5): ReLU()
      )
    )
  )
)
输出（B，128）

import torch
import torch.nn as nn

class ConvTransformerHeadModel(nn.Module):
    def __init__(self):
        super(ConvTransformerHeadModel, self).__init__()
        
        # 1. input_layer (FCN Tokenizer)
        # 完全使用卷积进行降维，最终输出单像素 256 通道的特征图，并将其展平
        self.input_layer = nn.Sequential(
            # Conv1: 输入通道 1, 输出通道 32, (18, 32) -> (9, 16)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Conv2: 输入通道 32, 输出通道 64, (9, 16) -> (5, 8)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Conv3: 输入通道 64, 输出通道 128, (5, 8) -> (3, 4)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # Conv4: 输入通道 128, 输出通道 256, 空间尺寸 (3, 4) -> (1, 1)
            nn.Conv2d(128, 256, kernel_size=(3, 4), stride=1, padding=0),
            nn.ReLU(),
            
            # Flatten: 展平 C, H, W 维度。此时特征图为 (256, 1, 1)，展平后刚好 256 维。
            nn.Flatten(start_dim=1)
        )
        
        # 2. tf_encoder (Transformer 时序处理)
        # 保持与标准一致的 1 层 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, 
            nhead=8, 
            dim_feedforward=512, 
            dropout=0.1, 
            batch_first=True
        )
        self.tf_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 3. output_layer (输出头)
        self.output_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward 数据流
        """
        # 1. 输入接收: x 的 shape 为 (B, T, 576)
        B, T, _ = x.shape
        
        # 2. 时空融合变形: 将 x reshape 为 (B * T, 1, 18, 32)
        # 这里把 batch_size 和 seq_len 融合到一起，这样我们可以用标准的 2D 卷积处理每一帧的深度图
        x = x.view(B * T, 1, 18, 32)
        
        # 3. 空间特征提取: 将变形后的张量送入 self.input_layer
        # 预期经过所有卷积和 Flatten 后，输出张量的 shape 为 (B * T, 256)
        x = self.input_layer(x)
        
        # 4. 时序展开变形: 将张量重新 reshape 为 (B, T, 256)
        # 恢复时间维度，以供后续 Transformer 时序序列处理使用
        x = x.view(B, T, 256)
        
        # 5. 时序特征提取: 送入 self.tf_encoder，输出 shape 依然为 (B, T, 256)
        x = self.tf_encoder(x)
        
        # 6. 输出映射: 送入 self.output_layer，最终返回 shape 为 (B, T, 128)
        x = self.output_layer(x)
        
        return x

# ============== 测试代码 (可选验证) ==============
if __name__ == "__main__":
    B, T = 4, 10
    # 模拟输入 (Batch_Size, Sequence_Length, 576)
    dummy_input = torch.randn(B, T, 576)
    
    model = ConvTransformerHeadModel()
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (B, T, 128), "Output shape does not match (B, T, 128)"



    # 角色设定
你是一个资深的 PyTorch 和强化学习（RL）算法工程师，精通基于视觉的机器人控制策略网络构建，特别擅长处理 Sim2Real（仿真到实机）的特征提取器重构。

# 任务目标
请参考我现有框架中/home/you/instinct_rl/instinct_rl/modules/transformer.py和/home/you/instinct_rl/instinct_rl/modules/conv2d.py文件，的帮我重构代码库中的 `Actor Encoder` 网络。当前，我们在训练 Unitree G1 人形机器人跨越台阶。由于原始网络在处理深度图时使用了过多的 Linear 层，导致特征缺乏空间平移不变性，实机部署时跨越行为丢失。
现在，我们需要将原本基于 MlpModel 的特征提取替换为**多层全卷积网络（Deep FCN）**，直接将空间特征压缩为 256 维向量，再交由现有的 TransformerEncoder 进行时序处理。

# 输入数据与架构上下文
1. 底层引擎已开启 `temporal=True`，送入该网络的原始张量形状为 `(Batch_Size, Sequence_Length, 576)`。
2. 这 576 维实际上是被展平的 `(18, 32)` 单通道深度图（18 宽，32 高，即 18*32=576）。
3. 网络的目标是：提取空间特征 -> 提取时序特征 -> 输出 `(Batch_Size, Sequence_Length, 128)` 给后续网络。

# 新网络结构设计规格 (ConvTransformerHeadModel)

请创建一个名为 `ConvTransformerHeadModel` 的 `nn.Module`，包含以下三个子模块：

## 1. input_layer (FCN Tokenizer)
**约束：不准使用任何 `nn.Linear`，全部使用卷积降维！**
这是一个 `nn.Sequential`，包含 4 层卷积，每一层后接 `nn.ReLU()`。
*   **Conv1**: 输入通道 1，输出通道 32，kernel_size=3，stride=2，padding=1
*   **Conv2**: 输入通道 32，输出通道 64，kernel_size=3，stride=2，padding=1
*   **Conv3**: 输入通道 64，输出通道 128，kernel_size=3，stride=2，padding=1
*   **Conv4**: 输入通道 128，输出通道 256，kernel_size=(3, 4)，stride=1，padding=0
*   **Flatten**: 展平最后两个维度（使用 `nn.Flatten(start_dim=1)`）。最终输出特征应刚好为 256 维。

## 2. tf_encoder (Transformer 时序处理)
请保持与标准 TransformerEncoder 一致（`d_model=256`）。
*   可直接实例化一个包含 1 层 `nn.TransformerEncoderLayer` 的 `nn.TransformerEncoder`。
*   `nhead=8`，`dim_feedforward=512`，`dropout=0.1`，`batch_first=True`。

## 3. output_layer (输出头)
*   这是一个简单的 MLP，包含一层 `nn.Linear(256, 128)` 后接 `nn.ReLU()`。

# Forward 函数数据流逻辑要求 (极其重要)
在编写 `forward(self, x)` 时，请严格按照以下张量变形逻辑实现，并加上详细的注释：

1. **输入接收**: `x` 的 shape 为 `(B, T, 576)`。
2. **时空融合变形**: 获取 `B` 和 `T`，将 `x` reshape 为 `(B * T, 1, 18, 32)`。
3. **空间特征提取**: 将变形后的张量送入 `self.input_layer`。
    * 预期经过所有卷积和 Flatten 后，输出张量的 shape 为 `(B * T, 256)`。
4. **时序展开变形**: 将张量重新 reshape 为 `(B, T, 256)`。
5. **时序特征提取**: 送入 `self.tf_encoder`，输出 shape 依然为 `(B, T, 256)`。
6. **输出映射**: 送入 `self.output_layer`，最终返回 shape 为 `(B, T, 128)` 的张量。

# 输出要求
请直接在instinct_rl_amp_cfg.py中修改出完整的符合以上架构的配置文件，确保代码风格规范，并在 forward 函数中用注释标明每一步的 tensor shape 变化。