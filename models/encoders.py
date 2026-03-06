from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:

    class _TorchRegressor(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)


class SimpleAlphaModel:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        epochs: int = 40,
        lr: float = 1e-3,
        dropout: float = 0.1,
        l2: float = 1e-2,
        use_torch: bool = True,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.l2 = l2
        # 若本机无 torch，则自动退回线性岭回归，保证流程可运行。
        self.use_torch = bool(use_torch and torch is not None and nn is not None)

        self._weights: np.ndarray | None = None
        self._model = None
        self.backend = "torch" if self.use_torch else "linear_fallback"
        self.fit_info: dict[str, float | int | str] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.fit_info = {
            "backend": self.backend,
            "n_samples": int(x.shape[0]),
            "n_features": int(x.shape[1]),
        }

        if self.use_torch:
            self._fit_torch(x, y)
            return

        xtx = x.T @ x
        ridge = self.l2 * np.eye(xtx.shape[0], dtype=np.float32)
        self._weights = np.linalg.solve(xtx + ridge, x.T @ y)
        self.fit_info["epochs"] = 1

    def _fit_torch(self, x: np.ndarray, y: np.ndarray) -> None:
        model = _TorchRegressor(self.input_dim, self.hidden_dim, self.dropout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.l2)
        criterion = nn.MSELoss()

        x_t = torch.tensor(x)
        y_t = torch.tensor(y)
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = model(x_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()
        self._model = model.eval()
        self.fit_info["epochs"] = int(self.epochs)
        self.fit_info["last_train_loss"] = float(loss.item())

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.use_torch and self._model is not None:
            with torch.no_grad():
                pred = self._model(torch.tensor(x))
            return pred.numpy()

        if self._weights is None:
            raise RuntimeError("Model has not been fitted yet.")
        return x @ self._weights
