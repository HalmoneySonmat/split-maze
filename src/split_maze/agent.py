"""IMPALA-CNN agent for procgen maze (PPO).

Matches the standard procgen IMPALA-CNN used by joonleesky/train-procgen-pytorch
and the Langosco goal-misgeneralization line of work:
- input: (B, 3, 64, 64) uint8 in [0,255] or float (auto-normalized to [0,1])
- 3 IMPALA blocks with channels (16, 32, 32)
- each block: Conv3x3 → MaxPool3x3(stride 2) → ResBlock → ResBlock
- each ResBlock: ReLU → Conv3x3 → ReLU → Conv3x3 + skip
- ReLU → flatten → Linear(2048→256) → ReLU = h_agent  (PLAN §3.4 추출 지점)
- policy head: Linear(256 → 15)
- value head : Linear(256 → 1)

forward() returns AgentOutput(logits, value, h_agent). h_agent is the 256-d
embedding used for ACC reconstruction in Phase 3 — exposed cleanly so the
(C-thin) detach can be applied at the call site.

Weight init follows the standard PPO-with-IMPALA practice:
- conv / hidden Linear: orthogonal, gain √2
- policy head: orthogonal, gain 0.01  (small initial action prefs)
- value head : orthogonal, gain 1.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


D_A: int = 256          # PLAN §3.4: 에이전트 추출 차원 (IMPALA embedding)
NUM_ACTIONS: int = 15   # procgen Discrete(15)


class _ResBlock(nn.Module):
    """IMPALA residual block: ReLU → Conv → ReLU → Conv, plus skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        return x + h


class _ImpalaBlock(nn.Module):
    """One IMPALA block: Conv (channel change) → MaxPool (down 2×) → 2 ResBlocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = _ResBlock(out_channels)
        self.res2 = _ResBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


@dataclass(frozen=True)
class AgentOutput:
    """One forward pass result. h_agent is the 256-d PLAN §3.4 추출 지점."""
    logits: torch.Tensor   # (B, num_actions)
    value: torch.Tensor    # (B,)
    h_agent: torch.Tensor  # (B, D_A) — for ACC, expose cleanly


class ImpalaAgent(nn.Module):
    """IMPALA-CNN agent for 64×64×3 procgen observations."""

    def __init__(self,
                 num_actions: int = NUM_ACTIONS,
                 d_a: int = D_A,
                 channels: tuple[int, ...] = (16, 32, 32)):
        super().__init__()
        self.num_actions = num_actions
        self.d_a = d_a
        self.channels = channels

        layers = []
        in_ch = 3
        for ch in channels:
            layers.append(_ImpalaBlock(in_ch, ch))
            in_ch = ch
        self.blocks = nn.Sequential(*layers)

        # After 3 maxpool(stride 2): 64 → 32 → 16 → 8 spatial.
        # With channels[-1]=32: feature map (B, 32, 8, 8) → flatten 2048.
        flat_dim = channels[-1] * (64 // (2 ** len(channels))) ** 2
        self.embed = nn.Linear(flat_dim, d_a)
        self.policy = nn.Linear(d_a, num_actions)
        self.value = nn.Linear(d_a, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Standard PPO+IMPALA orthogonal init."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Override heads (small policy, unit value) per PPO convention.
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, obs: torch.Tensor) -> AgentOutput:
        """Forward pass.

        Args:
            obs: (B, 3, 64, 64). uint8 in [0,255] is auto-normalized; float
                 is assumed pre-normalized (caller responsibility).
        """
        if obs.dim() != 4 or obs.shape[-3:] != (3, 64, 64):
            raise ValueError(f"expected (B, 3, 64, 64), got {tuple(obs.shape)}")
        if obs.dtype == torch.uint8:
            x = obs.float() / 255.0
        else:
            x = obs

        x = self.blocks(x)
        x = F.relu(x)
        x = x.flatten(start_dim=1)
        h = F.relu(self.embed(x))   # (B, D_A) — h_agent
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return AgentOutput(logits=logits, value=value, h_agent=h)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
