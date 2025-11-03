from __future__ import annotations
from pathlib import Path
import sys, numpy as np, pandas as pd, torch

THIS = Path(__file__).resolve()
FORECASTING_DIR = THIS.parent
PROJECT_ROOT = THIS.parents[1]
for p in (str(FORECASTING_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import Config
from src.encoders import TemporalEncoder1D
from src.tcgc import TCGCBlock
from src.ar_head import ARHead
from src.rl_dqn import DQNAgent, DQNConfig, build_adj_from_pairs

class LiveForecaster:
    def __init__(self, ckpt_path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ckpt_path = Path(ckpt_path)
        self.ckpt = torch.load(self.ckpt_path, map_location=self.device)

        meta = self.ckpt["meta"]
        self.cols = meta["cols"]
        self.window = meta["window"]
        self.N = meta["num_nodes"]

        self.mu = np.array(meta.get("mu") or meta.get("mean") or meta.get("x_mean") or [0]*self.N, dtype=float)
        self.sigma = np.array(meta.get("sigma") or meta.get("std") or meta.get("x_std") or [1]*self.N, dtype=float)
        self.sigma = np.where(self.sigma == 0.0, 1.0, self.sigma)

        cfg = Config()
        self.temp = TemporalEncoder1D(d_out=cfg.d_temporal, k=cfg.temporal_kernel, n_layers=cfg.temporal_layers).to(self.device)
        self.tcgc = TCGCBlock(d_in=cfg.d_temporal, d_gcn=cfg.d_gcn, d_fused=cfg.d_fused, use_rl_channel=True).to(self.device)
        self.ar   = ARHead(d_in=cfg.d_fused).to(self.device)

        if "temp" in self.ckpt: self.temp.load_state_dict(self.ckpt["temp"])
        if "tcgc" in self.ckpt: self.tcgc.load_state_dict(self.ckpt["tcgc"])
        if "ar"   in self.ckpt: self.ar.load_state_dict(self.ckpt["ar"])

        self.A_static = self.ckpt.get("A_static")
        if self.A_static is None:
            self.A_static = torch.eye(self.N)
        self.A_static = self.A_static.to(self.device)

        self.agent = None
        if "agent" in self.ckpt:  # RL checkpoint
            top_k_val = getattr(cfg, "dqn_top_k", 2)
            dqn_cfg = DQNConfig(eps_start=0.0, eps_end=0.0, eps_decay_steps=1, top_k=top_k_val)
            self.agent = DQNAgent(d_in=cfg.d_temporal, N_nodes=self.N, cfg=dqn_cfg).to(self.device)
            self.agent.load_state_dict(self.ckpt["agent"])
            self.agent.eval()

        for m in (self.temp, self.tcgc, self.ar): m.eval()
        self.buffer = pd.DataFrame(columns=self.cols)

    def _standardize(self, X):
        return (X - self.mu.reshape(1,-1)) / self.sigma.reshape(1,-1)

    def _destandardize(self, y):
        return y * self.sigma.reshape(1,-1) + self.mu.reshape(1,-1)

    def update_with_reading(self, reading: dict[str, float] | pd.Series) -> dict[str, float] | None:
        row = {c: float(reading.get(c, np.nan)) for c in self.cols}
        self.buffer = pd.concat([self.buffer, pd.DataFrame([row])], ignore_index=True).ffill().bfill()
        if len(self.buffer) < self.window:
            return None

        X_TN = self.buffer.iloc[-self.window:].to_numpy(dtype=float)
        X_std = self._standardize(X_TN)
        X = torch.tensor(X_std.T[None, ...], dtype=torch.float32, device=self.device)  # (1,N,T)

        with torch.no_grad():
            O = self.temp(X)
            A_b = self.A_static.unsqueeze(0).expand(1, self.N, self.N)

            if self.agent is not None:
                _, picks, _ = self.agent.select_pairs(O, self.device)  # B=1
                A_rl = build_adj_from_pairs(self.N, picks[0], self.device).unsqueeze(0)
            else:
                A_rl = None

            H, _ = self.tcgc(O, A_b, A_rl=A_rl)
            Yhat = self.ar(H)  # (1,N)

        y_real = self._destandardize(Yhat.detach().cpu().numpy())  # (1,N)
        return {c: float(y_real[0, i]) for i, c in enumerate(self.cols)}
