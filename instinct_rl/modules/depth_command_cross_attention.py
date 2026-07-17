from __future__ import annotations

from typing import Sequence

import math
import torch
import torch.nn as nn

from .mlp import MlpModel


class DepthCommandCrossAttentionHeadModel(nn.Module):
    """Temporal depth memory queried by the latest depth token and velocity command."""

    def __init__(
        self,
        input_shapes: Sequence[torch.Size],
        output_size: int,
        cnn_channels: Sequence[int] = (16, 32, 64, 128),
        cnn_kernel_sizes: Sequence[int | tuple[int, int]] = (3, 3, 3, (3, 4)),
        cnn_strides: Sequence[int | tuple[int, int]] = (2, 2, 2, 1),
        cnn_paddings: Sequence[int | tuple[int, int]] = (1, 1, 1, 0),
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        nonlinearity: str = "ReLU",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        command_hidden_sizes: Sequence[int] = (),
        query_hidden_sizes: Sequence[int] = (),
        output_hidden_sizes: Sequence[int] = (),
        use_temporal_pos_embedding: bool = True,
    ):
        super().__init__()
        assert len(input_shapes) == 2, "DepthCommandCrossAttentionHeadModel expects command and depth components."
        assert batch_first, "DepthCommandCrossAttentionHeadModel expects batch_first=True."
        assert len(cnn_channels) == len(cnn_kernel_sizes) == len(cnn_strides) == len(cnn_paddings)
        assert cnn_channels[-1] == d_model, "The last CNN channel must equal d_model."

        self.input_shapes = [tuple(int(v) for v in shape) for shape in input_shapes]
        self.output_size = output_size
        self.d_model = d_model
        self.use_temporal_pos_embedding = use_temporal_pos_embedding

        depth_indices = [i for i, shape in enumerate(self.input_shapes) if len(shape) in (3, 4)]
        command_indices = [i for i, shape in enumerate(self.input_shapes) if i not in depth_indices]
        if len(depth_indices) != 1 or len(command_indices) != 1:
            raise ValueError(
                "Expected exactly one depth component shaped (T,H,W)/(T,C,H,W) and one command component."
            )
        self.depth_index = depth_indices[0]
        self.command_index = command_indices[0]
        self.depth_shape = self.input_shapes[self.depth_index]
        self.command_shape = self.input_shapes[self.command_index]
        self.depth_flat_size = math.prod(self.depth_shape)
        self.command_flat_size = math.prod(self.command_shape)
        self.sequence_length, self.frame_channels, self.frame_height, self.frame_width = self._parse_depth_shape(
            self.depth_shape
        )
        self.command_frame_dim = self._infer_command_frame_dim()

        if isinstance(nonlinearity, str):
            nonlinearity_cls = getattr(nn, nonlinearity)
        else:
            nonlinearity_cls = nonlinearity

        in_channels = [self.frame_channels] + list(cnn_channels[:-1])
        cnn_layers = []
        for in_channel, out_channel, kernel_size, stride, padding in zip(
            in_channels, cnn_channels, cnn_kernel_sizes, cnn_strides, cnn_paddings
        ):
            cnn_layers.append(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            cnn_layers.append(nonlinearity_cls())
        self.input_layer = nn.Sequential(*cnn_layers, nn.Flatten(start_dim=1))
        self.token_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        if self.use_temporal_pos_embedding:
            self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, self.sequence_length, d_model))
            nn.init.normal_(self.temporal_pos_embedding, mean=0.0, std=0.02)
        else:
            self.register_parameter("temporal_pos_embedding", None)

        tf_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.memory_encoder = nn.TransformerEncoder(
            encoder_layer=tf_encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps),
        )

        self.command_layer = MlpModel(
            input_size=self.command_frame_dim,
            hidden_sizes=list(command_hidden_sizes) + [d_model],
            nonlinearity=nonlinearity_cls,
        )
        self.query_layer = MlpModel(
            input_size=2 * d_model,
            hidden_sizes=list(query_hidden_sizes) + [d_model],
            nonlinearity=nonlinearity_cls,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.cross_dropout = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nonlinearity_cls(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.output_layer = MlpModel(
            input_size=d_model,
            hidden_sizes=list(output_hidden_sizes) + [output_size],
            nonlinearity=nonlinearity_cls,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_dim = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        pieces = torch.split(
            flat,
            [math.prod(shape) for shape in self.input_shapes],
            dim=-1,
        )
        depth = pieces[self.depth_index].reshape(
            -1,
            self.sequence_length,
            self.frame_channels,
            self.frame_height,
            self.frame_width,
        )
        command = pieces[self.command_index].reshape(-1, self.command_flat_size)

        depth_tokens = self._encode_depth_frames(depth)
        memory = depth_tokens
        if self.temporal_pos_embedding is not None:
            memory = memory + self.temporal_pos_embedding[:, : memory.shape[1], :]
        memory = self.memory_encoder(memory)

        latest_depth_token = depth_tokens[:, -1, :]
        latest_command = self._latest_command(command)
        command_token = self.command_layer(latest_command)
        query = self.query_layer(torch.cat([latest_depth_token, command_token], dim=-1)).unsqueeze(1)

        attended, _ = self.cross_attention(query, memory, memory, need_weights=False)
        query = self.cross_norm(query + self.cross_dropout(attended))
        latent = query.squeeze(1)
        latent = self.ffn_norm(latent + self.ffn(latent))
        output = self.output_layer(latent)
        return output.reshape(leading_dim + (self.output_size,))

    def _encode_depth_frames(self, depth: torch.Tensor) -> torch.Tensor:
        depth = depth.reshape(-1, self.frame_channels, self.frame_height, self.frame_width)
        tokens = self.input_layer(depth)
        tokens = tokens.reshape(-1, self.sequence_length, self.d_model)
        return self.token_norm(tokens)

    def _latest_command(self, command: torch.Tensor) -> torch.Tensor:
        if len(self.command_shape) >= 2 and self.command_shape[0] == self.sequence_length:
            return command.reshape(command.shape[0], self.sequence_length, -1)[:, -1, :]
        if self.command_flat_size % self.sequence_length == 0:
            return command.reshape(command.shape[0], self.sequence_length, -1)[:, -1, :]
        return command

    def _infer_command_frame_dim(self) -> int:
        if len(self.command_shape) >= 2 and self.command_shape[0] == self.sequence_length:
            return math.prod(self.command_shape[1:])
        if self.command_flat_size % self.sequence_length == 0:
            return self.command_flat_size // self.sequence_length
        return self.command_flat_size

    @staticmethod
    def _parse_depth_shape(depth_shape: Sequence[int]) -> tuple[int, int, int, int]:
        if len(depth_shape) == 3:
            sequence_length, frame_height, frame_width = depth_shape
            return sequence_length, 1, frame_height, frame_width
        if len(depth_shape) == 4:
            sequence_length, frame_channels, frame_height, frame_width = depth_shape
            return sequence_length, frame_channels, frame_height, frame_width
        raise ValueError(
            "DepthCommandCrossAttentionHeadModel expects depth shaped (T,H,W) or (T,C,H,W), "
            f"but got {tuple(depth_shape)}."
        )

    @property
    def output_shape(self):
        return (self.output_size,)
