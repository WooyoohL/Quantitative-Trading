from __future__ import annotations

import torch
import torch.nn as nn


class AlphaRecurrentEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        rnn_type: str = "gru",
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        rnn_type = str(rnn_type).lower()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}.get(rnn_type)
        if rnn_cls is None:
            raise ValueError(f"Unsupported rnn_type={rnn_type}. Expected one of: gru, lstm.")

        self.rnn_type = rnn_type
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.rnn = rnn_cls(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [batch, seq_len, feature_dim], got {tuple(x.shape)}")

        # 输入固定为 [batch, seq_len, feature_dim]，先映射到隐藏维度，再做时序编码。
        encoded = self.input_projection(x)
        sequence_output, _ = self.rnn(encoded)
        last_hidden = sequence_output[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


class AlphaAttentionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        ff_multiplier: int = 2,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for attention encoder.")

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [batch, seq_len, feature_dim], got {tuple(x.shape)}")
        seq_len = int(x.size(1))
        if seq_len > self.position_embedding.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.position_embedding.size(1)} for attention encoder."
            )

        # 轻量 attention 版本只编码最近这一段时间序列，不做跨股票注意力，避免参数和噪声失控。
        encoded = self.input_projection(x) + self.position_embedding[:, :seq_len, :]
        sequence_output = self.encoder(encoded)
        last_hidden = sequence_output[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


def build_model(input_dim: int, model_cfg: dict) -> tuple[nn.Module, dict]:
    model_name = str(model_cfg.get("name", "gru")).lower()
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    num_layers = int(model_cfg.get("num_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.1))

    # 当前样本量在几万级，优先保留小模型：GRU/LSTM baseline 或轻量 attention。
    if model_name in {"gru", "lstm"}:
        model = AlphaRecurrentEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=model_name,
        )
        resolved = {
            "name": model_name,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        return model, resolved

    if model_name in {"attention", "transformer"}:
        num_heads = int(model_cfg.get("attention_heads", 4))
        ff_multiplier = int(model_cfg.get("ff_multiplier", 2))
        max_seq_len = int(model_cfg.get("max_seq_len", 64))
        model = AlphaAttentionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
            ff_multiplier=ff_multiplier,
            max_seq_len=max_seq_len,
        )
        resolved = {
            "name": "attention",
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "attention_heads": num_heads,
            "ff_multiplier": ff_multiplier,
            "max_seq_len": max_seq_len,
        }
        return model, resolved

    raise ValueError(f"Unsupported model.name={model_name}. Expected one of: gru, lstm, attention.")


# 兼容旧引用，默认别名指向当前推荐的时序 baseline。
AlphaSequenceEncoder = AlphaRecurrentEncoder
