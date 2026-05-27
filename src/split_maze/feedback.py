"""Phase-6 R2 feedback path (lm → agent) — PREREG §1.

The R2 ("online bidirectional / corpus-callosum") regime injects the
interpreter's reading back into the decider's next decision:

    h' = h + λ · Wᵀ · LN( LM_summary( W · LN(h_agent) ) )

i.e. the agent hidden is bridged into LM space (W·LN), the LM *reads and
summarizes* it (its interpretation), then the summary is bridged back to
agent space (Wᵀ·LN) and added — scaled by the fixed gate λ — to the agent's
hidden at the NEXT step (the gate lives in ``ImpalaAgent.forward(inject=...)``).

This module is the atomic feedback computation the live regime loop calls.
It takes the ACC bridge and the MazeLM as arguments (no import coupling) and
makes no assumption about grad: the caller decides (detach for the agent's
pure-RL PPO update — the (C-thin) boundary).

Echo guard (PREREG §0.5 "메아리 체크"): if the LM merely mirrors the agent,
the feedback degenerates into re-injecting the agent's own hidden. ``echo_ratio``
compares the full (LM-processed) feedback to the bare bridge round-trip
(LM skipped); ~1 means the LM added nothing (echo), < 1 means its
interpretation diverges from a pass-through.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_inject(acc, lm, h_agent: torch.Tensor, lam: float) -> torch.Tensor:
    """λ · Wᵀ · LN( LM_summary( W · LN(h_agent) ) )  →  (B, d_agent).

    Args:
        acc: an :class:`split_maze.acc.ACC` (provides ``ln_agent``, ``ln_lm``,
             ``predict_lm_from_agent`` = W·, ``predict_agent_from_lm`` = Wᵀ·).
        lm:  a :class:`split_maze.lm.MazeLM` (provides ``summarize_vector``).
        h_agent: (B, d_agent) agent hidden (pre-injection h).
        lam: fixed gate λ (≥ 0). λ = 0 returns zeros (≡ R0/R1).
    Returns:
        (B, d_agent) additive injection for ``ImpalaAgent.forward(inject=...)``.
    """
    n_agent = acc.ln_agent(h_agent)                 # LN(h_agent)
    hhat_lm = acc.predict_lm_from_agent(n_agent)    # W·ñ_agent   (agent → lm)
    h_lm = lm.summarize_vector(hhat_lm)             # LM reads & summarizes
    n_lm = acc.ln_lm(h_lm)                          # LN(h_lm)
    inject = acc.predict_agent_from_lm(n_lm)        # Wᵀ·ñ_lm     (lm → agent)
    return lam * inject


@torch.no_grad()
def echo_ratio(acc, lm, h_agent: torch.Tensor) -> torch.Tensor:
    """Per-sample cosine between the LM-processed feedback and the bare bridge
    round-trip (LM skipped). ~1 ⇒ feedback is an echo of the agent's own
    hidden; < 1 ⇒ the LM's interpretation diverges. Returns (B,).
    """
    n_agent = acc.ln_agent(h_agent)
    hhat_lm = acc.predict_lm_from_agent(n_agent)
    with_lm = acc.predict_agent_from_lm(acc.ln_lm(lm.summarize_vector(hhat_lm)))
    no_lm = acc.predict_agent_from_lm(acc.ln_lm(hhat_lm))   # skip LM processing
    return F.cosine_similarity(with_lm, no_lm, dim=-1)
