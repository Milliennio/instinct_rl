from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn as nn

from .mlp import MlpModel


class ConvTemporalTransformerHeadModel(nn.Module):
    """Frame-wise CNN tokenizer followed by a temporal Transformer encoder.

    The expected depth input is ``(..., T, H, W)`` where ``T`` is the history length.
    For compatibility with temporal observation extraction, ``(..., T, H * W)`` and
    fully flattened ``(..., T * H * W)`` inputs are also accepted.
    """

    def __init__(
        self,
        input_shapes: Sequence[torch.Size],
        output_size: int,
        cnn_channels: Sequence[int] = (32, 64, 128, 256),
        cnn_kernel_sizes: Sequence[int | tuple[int, int]] = (3, 3, 3, (3, 4)),
        cnn_strides: Sequence[int | tuple[int, int]] = (2, 2, 2, 1),
        cnn_paddings: Sequence[int | tuple[int, int]] = (1, 1, 1, 0),
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
        nonlinearity: str = "ReLU",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        temporal_pool: Literal["latest", "mean", "max"] = "latest",
        use_temporal_pos_embedding: bool = True,
        output_hidden_sizes: Sequence[int] = (),
    ):
        super().__init__()
        assert len(input_shapes) == 1, "ConvTemporalTransformerHeadModel only accepts one image-history component."
        assert batch_first, "ConvTemporalTransformerHeadModel expects batch_first=True."
        assert len(cnn_channels) == len(cnn_kernel_sizes) == len(cnn_strides) == len(cnn_paddings)
        assert cnn_channels[-1] == d_model, "The last CNN channel must equal d_model."

        self.input_shapes = input_shapes
        self.output_size = output_size
        self.sequence_length, self.frame_channels, self.frame_height, self.frame_width = self._parse_input_shape(
            input_shapes[0]
        )
        self.frame_feature_size = self.frame_channels * self.frame_height * self.frame_width
        self.d_model = d_model
        self.temporal_pool = temporal_pool
        self.use_temporal_pos_embedding = use_temporal_pos_embedding
        self.layer_norm_eps = layer_norm_eps

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
        self.tf_encoder = nn.TransformerEncoder(
            encoder_layer=tf_encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps),
        )
        self.output_layer = MlpModel(
            input_size=d_model,
            hidden_sizes=list(output_hidden_sizes) + [output_size],
            nonlinearity=nonlinearity_cls,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_dim, x = self._reshape_to_image_sequence(x)

        # (..., T, C, H, W) -> (N * T, C, H, W), where N is the flattened leading dimension.
        x = x.reshape(-1, self.frame_channels, self.frame_height, self.frame_width)
        # (N * T, C, H, W) -> (N * T, d_model)
        x = self.input_layer(x)
        # (N * T, d_model) -> (N, T, d_model)
        x = x.reshape(-1, self.sequence_length, self.d_model)
        x = self.token_norm(x)

        if self.temporal_pos_embedding is not None:
            x = x + self.temporal_pos_embedding[:, : x.shape[1], :]

        # (N, T, d_model) -> (N, T, d_model)
        x = self.tf_encoder(x)

        # Select a single temporal embedding for the downstream actor.
        if self.temporal_pool == "latest":
            x = x[:, -1, :]
        elif self.temporal_pool == "mean":
            x = x.mean(dim=1)
        elif self.temporal_pool == "max":
            x = x.amax(dim=1)
        else:
            raise ValueError(f"Unsupported temporal_pool: {self.temporal_pool}")

        # (N, d_model) -> (N, output_size) -> (..., output_size)
        x = self.output_layer(x)
        return x.reshape(leading_dim + (self.output_size,))

    def _reshape_to_image_sequence(self, x: torch.Tensor) -> tuple[torch.Size, torch.Tensor]:
        """Normalize supported input layouts to ``(N, T, C, H, W)``."""
        if x.shape[-3:] == (self.sequence_length, self.frame_height, self.frame_width):
            leading_dim = x.shape[:-3]
            x = x.reshape(*leading_dim, self.sequence_length, 1, self.frame_height, self.frame_width)
        elif x.shape[-4:] == (
            self.sequence_length,
            self.frame_channels,
            self.frame_height,
            self.frame_width,
        ):
            leading_dim = x.shape[:-4]
        elif x.shape[-2:] == (self.sequence_length, self.frame_feature_size):
            leading_dim = x.shape[:-2]
            x = x.reshape(
                *leading_dim,
                self.sequence_length,
                self.frame_channels,
                self.frame_height,
                self.frame_width,
            )
        elif x.shape[-1] == self.sequence_length * self.frame_feature_size:
            leading_dim = x.shape[:-1]
            x = x.reshape(
                *leading_dim,
                self.sequence_length,
                self.frame_channels,
                self.frame_height,
                self.frame_width,
            )
        else:
            raise ValueError(
                "Unsupported input shape for ConvTemporalTransformerHeadModel: "
                f"{tuple(x.shape)}. Expected (..., {self.sequence_length}, {self.frame_height}, {self.frame_width}), "
                f"(..., {self.sequence_length}, {self.frame_feature_size}), or "
                f"(..., {self.sequence_length * self.frame_feature_size})."
            )
        return leading_dim, x

    @staticmethod
    def _parse_input_shape(input_shape: torch.Size | Sequence[int]) -> tuple[int, int, int, int]:
        shape = tuple(int(v) for v in input_shape)
        if len(shape) == 3:
            sequence_length, frame_height, frame_width = shape
            frame_channels = 1
        elif len(shape) == 4:
            sequence_length, frame_channels, frame_height, frame_width = shape
        else:
            raise ValueError(
                "ConvTemporalTransformerHeadModel expects an image-history component shaped "
                f"(T, H, W) or (T, C, H, W), but got {shape}."
            )
        return sequence_length, frame_channels, frame_height, frame_width

    @property
    def output_shape(self):
        return (self.output_size,)
