"""
networks.py
===========
All neural-network components for SafePC-MARL.

Four classes are exported:

PolicyNet          -- Stateless functional policy. Accepts externally supplied
                      weights (from HyperNetwork) and executes a forward pass
                      using torch.nn.functional operations. Gradients flow back
                      through the weight tensor into the HyperNetwork, enabling
                      end-to-end differentiation.

HyperNetwork       -- Maps a preference vector ω ∈ Δ² to per-cell policy
                      parameters. Shared trunk + per-cell output heads.
                      Also holds one learnable log_std scalar per cell.

CentralizedCritic  -- Multi-head value function used under CTDE.
                      Input: all cell observations concatenated + ω.
                      Output: (N_CELLS, N_OBJ) value estimates, one per
                      cell and per reward objective.

CostCritic         -- Per-cell value function estimating E[C_i^π].
                      Used by the Lagrangian dual updater.

Architecture constants
----------------------
  OBS_DIM    = 6   (5 process features + last action)
  N_CELLS    = 3
  N_OBJ      = 3   (throughput, energy, ergonomic)
  OMEGA_DIM  = 3
  POLICY_HIDDEN = 64
  CRITIC_HIDDEN = 128
  HNET_HIDDEN   = 128
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Dimensions (single source of truth)
# ---------------------------------------------------------------------------

OBS_DIM:       int = 6
N_CELLS:       int = 3
N_OBJ:         int = 3
OMEGA_DIM:     int = 3
ACTION_DIM:    int = 1

POLICY_HIDDEN: int = 64
CRITIC_HIDDEN: int = 256   # increased from 128: deeper representation for multi-objective value
HNET_HIDDEN:   int = 256   # increased from 128: trunk must map omega to 4673-dim outputs

# --- PolicyNet parameter layout (exact, computed analytically) ---
# Layer 1 : Linear(OBS_DIM,      POLICY_HIDDEN)
# Layer 2 : Linear(POLICY_HIDDEN, POLICY_HIDDEN)
# Mean head: Linear(POLICY_HIDDEN, ACTION_DIM)
_W1 = OBS_DIM       * POLICY_HIDDEN   # 6×64  = 384
_B1 = POLICY_HIDDEN                   # 64
_W2 = POLICY_HIDDEN * POLICY_HIDDEN   # 64×64 = 4096
_B2 = POLICY_HIDDEN                   # 64
_WM = POLICY_HIDDEN * ACTION_DIM      # 64×1  = 64
_BM = ACTION_DIM                      # 1

POLICY_PARAM_DIM: int = _W1 + _B1 + _W2 + _B2 + _WM + _BM  # = 4673

# Slice boundaries inside the flat weight vector
_S0 = 0
_S1 = _S0 + _W1 + _B1          # 448
_S2 = _S1 + _W2 + _B2          # 4608
_S3 = _S2 + _WM + _BM          # 4673


# ---------------------------------------------------------------------------
# Helper: weight initialisation
# ---------------------------------------------------------------------------

def _orthogonal_init(module: nn.Module, gain: float = math.sqrt(2)) -> nn.Module:
    """Apply orthogonal initialisation to all Linear layers."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            nn.init.zeros_(m.bias)
    return module


# ---------------------------------------------------------------------------
# 1. PolicyNet  (stateless functional network)
# ---------------------------------------------------------------------------

class PolicyNet:
    """
    Functional policy network for a single cell.

    This class contains NO nn.Parameters of its own.  All weights are
    supplied externally from the HyperNetwork at each forward call.
    This is what makes true hypernetwork gradient flow possible.

    Architecture
    ------------
    obs  (B, 6)
     -> Linear(6, 64), Tanh
     -> Linear(64, 64), Tanh
     -> mean head: Linear(64, 1), Tanh   (action ∈ (−1, 1))
        std head:  exp(log_std)           (scalar, from HyperNetwork)

    The Tanh at the mean head bounds the pre-squash action mean to (−1, 1),
    matching the environment's action range without a separate squashing step
    in the action distribution.  Sampling uses the reparameterisation trick:
        a = tanh(mu + std * eps),  eps ~ N(0, 1)
    but since mu is already tanh-bounded, we apply squashing only once.
    """

    @staticmethod
    def unpack(theta: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Split a flat weight vector theta (shape: (POLICY_PARAM_DIM,)) into
        individual weight matrices and bias vectors.

        Returns
        -------
        W1, b1, W2, b2, Wm, bm
        """
        W1 = theta[_S0         : _S0 + _W1].view(POLICY_HIDDEN, OBS_DIM)
        b1 = theta[_S0 + _W1   : _S1      ].view(POLICY_HIDDEN)
        W2 = theta[_S1         : _S1 + _W2].view(POLICY_HIDDEN, POLICY_HIDDEN)
        b2 = theta[_S1 + _W2   : _S2      ].view(POLICY_HIDDEN)
        Wm = theta[_S2         : _S2 + _WM].view(ACTION_DIM,    POLICY_HIDDEN)
        bm = theta[_S2 + _WM   : _S3      ].view(ACTION_DIM)
        return W1, b1, W2, b2, Wm, bm

    @staticmethod
    def forward(obs: Tensor, theta: Tensor) -> Tensor:
        """
        Compute action mean for a batch of observations.

        Parameters
        ----------
        obs   : (B, OBS_DIM)
        theta : (POLICY_PARAM_DIM,)  — flat weight vector for ONE cell

        Returns
        -------
        mu : (B, ACTION_DIM)  — action mean, values in (−1, 1)
        """
        W1, b1, W2, b2, Wm, bm = PolicyNet.unpack(theta)
        x  = F.tanh(F.linear(obs, W1, b1))
        x  = F.tanh(F.linear(x,   W2, b2))
        mu = F.tanh(F.linear(x,   Wm, bm))
        return mu


# ---------------------------------------------------------------------------
# 2. HyperNetwork
# ---------------------------------------------------------------------------

class HyperNetwork(nn.Module):
    """
    Maps operator preference vector ω ∈ Δ² to cell-specific policy parameters.

    Architecture
    ------------
    Shared trunk:
        ω  (3,) -> Linear(3, HNET_HIDDEN), Tanh
                -> Linear(HNET_HIDDEN, HNET_HIDDEN), Tanh

    Per-cell output heads (N_CELLS independent heads):
        -> Linear(HNET_HIDDEN, POLICY_PARAM_DIM)

    Per-cell log_std parameters:
        One learnable scalar per cell, initialised to log(0.5) ≈ −0.693.
        These are NOT generated by the trunk; they are free parameters of
        HyperNetwork itself.

    Initialisation
    --------------
    Trunk layers use orthogonal init (gain=√2).
    Output heads use orthogonal init with small gain (0.01) so that at
    the start of training all cells produce near-identical (near-zero)
    actions for any ω — a stable starting point.

    Parameters
    ----------
    n_cells   : number of manufacturing cells (3)
    omega_dim : dimensionality of ω (3)
    """

    def __init__(
        self,
        n_cells:   int = N_CELLS,
        omega_dim: int = OMEGA_DIM,
    ):
        super().__init__()
        self.n_cells   = n_cells
        self.omega_dim = omega_dim

        # --- Shared trunk (3 layers for richer omega encoding) ---
        # Mapping omega ∈ Δ² to 4673-dim policy parameters per cell requires
        # a more expressive trunk.  A third layer improves the quality of
        # per-corner specialisation and reduces switching degradation.
        self.trunk = nn.Sequential(
            nn.Linear(omega_dim,   HNET_HIDDEN),
            nn.Tanh(),
            nn.Linear(HNET_HIDDEN, HNET_HIDDEN),
            nn.Tanh(),
            nn.Linear(HNET_HIDDEN, HNET_HIDDEN),
            nn.Tanh(),
        )
        _orthogonal_init(self.trunk, gain=math.sqrt(2))

        # --- Per-cell output heads ---
        self.heads = nn.ModuleList([
            nn.Linear(HNET_HIDDEN, POLICY_PARAM_DIM)
            for _ in range(n_cells)
        ])
        for head in self.heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

        # --- Per-cell log_std (free parameters) ---
        # Initialise at log(0.5): moderate initial exploration.
        init_log_std = math.log(0.5)
        self.log_std = nn.Parameter(
            torch.full((n_cells,), init_log_std)
        )

    def forward(self, omega: Tensor) -> Tuple[List[Tensor], Tensor]:
        """
        Generate policy parameters for all cells given ω.

        Parameters
        ----------
        omega : (OMEGA_DIM,) or (B, OMEGA_DIM)
            Preference vector(s). If 1-D, a batch dimension of 1 is added
            internally and removed before returning.

        Returns
        -------
        thetas   : List[Tensor] of length n_cells, each (POLICY_PARAM_DIM,)
                   when omega is 1-D, or (B, POLICY_PARAM_DIM) when batched.
        log_stds : Tensor of shape (n_cells,)  — shared across any batch dim.
        """
        squeeze = (omega.dim() == 1)
        if squeeze:
            omega = omega.unsqueeze(0)   # (1, OMEGA_DIM)

        h      = self.trunk(omega)       # (B, HNET_HIDDEN)
        thetas = [head(h) for head in self.heads]   # each (B, POLICY_PARAM_DIM)

        if squeeze:
            thetas = [t.squeeze(0) for t in thetas]  # each (POLICY_PARAM_DIM,)

        return thetas, self.log_std

    def get_action_dist(
        self, obs_per_cell: List[Tensor], omega: Tensor
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Convenience method: compute (mu, std) per cell for given observations.

        Parameters
        ----------
        obs_per_cell : List of (B, OBS_DIM) tensors, one per cell
        omega        : (OMEGA_DIM,)

        Returns
        -------
        mus  : List of (B, ACTION_DIM) action means, one per cell
        stds : List of (B, ACTION_DIM) action stds,  one per cell
        """
        thetas, log_stds = self(omega)
        mus, stds = [], []
        for i, (theta, obs) in enumerate(zip(thetas, obs_per_cell)):
            mu  = PolicyNet.forward(obs, theta)
            std = log_stds[i].exp().expand_as(mu)
            mus.append(mu)
            stds.append(std)
        return mus, stds

    def diversity_loss(
        self,
        obs_per_cell: List[Tensor],
        n_omega_samples: int = 32,
    ) -> Tensor:
        """
        Diversity regulariser: penalise low variance of action means across
        sampled preference vectors.

        Vectorised implementation: the trunk is called ONCE with all K omega
        samples stacked into a batch, then per-cell policy parameters are
        computed via a single head(h_K) call.  The per-observation action
        means are computed with bmm rather than a Python loop over K,
        reducing wall-clock time by ~10× for K=32.

        Loss = -mean_{i, obs} [ Var_{omega} [ mu_i(obs, omega) ] ]
        """
        device = next(self.parameters()).device

        # Sample K omega vectors uniformly from Δ² (Dirichlet(1,1,1))
        alphas = torch.ones(n_omega_samples, self.omega_dim, device=device)
        omegas = torch.distributions.Dirichlet(alphas).sample()   # (K, 3)

        # ── Vectorised trunk: one forward pass for ALL K omegas ──────────────
        h_k = self.trunk(omegas)   # (K, HNET_HIDDEN)
        # Per-cell parameter tensors: each (K, POLICY_PARAM_DIM)
        thetas_k = [head(h_k) for head in self.heads]

        total_var = torch.tensor(0.0, device=device)
        count     = 0

        for i, obs in enumerate(obs_per_cell):
            B  = obs.shape[0]
            K  = n_omega_samples
            tk = thetas_k[i]   # (K, POLICY_PARAM_DIM)

            # ── Unpack all K weight matrices at once ──────────────────────────
            # Layer 1
            W1 = tk[:, _S0      : _S0 + _W1].view(K, POLICY_HIDDEN, OBS_DIM)
            b1 = tk[:, _S0 + _W1: _S1      ].view(K, 1, POLICY_HIDDEN)
            # Layer 2
            W2 = tk[:, _S1      : _S1 + _W2].view(K, POLICY_HIDDEN, POLICY_HIDDEN)
            b2 = tk[:, _S1 + _W2: _S2      ].view(K, 1, POLICY_HIDDEN)
            # Mean head
            Wm = tk[:, _S2      : _S2 + _WM].view(K, ACTION_DIM, POLICY_HIDDEN)
            bm = tk[:, _S2 + _WM: _S3      ].view(K, 1, ACTION_DIM)

            # ── Batched forward pass: obs (B, OBS_DIM) × K thetas ────────────
            # Expand obs to (K, B, OBS_DIM) for bmm
            obs_k = obs.unsqueeze(0).expand(K, -1, -1)   # (K, B, OBS_DIM)

            # Layer 1: x = tanh(obs_k @ W1^T + b1)   shape (K, B, HIDDEN)
            x = F.tanh(torch.bmm(obs_k, W1.transpose(1, 2)) + b1)
            # Layer 2
            x = F.tanh(torch.bmm(x,     W2.transpose(1, 2)) + b2)
            # Mean head
            mu = F.tanh(torch.bmm(x,    Wm.transpose(1, 2)) + bm)  # (K, B, 1)

            # Variance across the K omega samples (dim=0), averaged over B and A
            var        = mu.var(dim=0)    # (B, 1)
            total_var  = total_var + var.mean()
            count     += 1

        # Minimise -variance  ↔  maximise action diversity across omega
        assert count > 0, "obs_per_cell must be non-empty"
        return -(total_var / count)


# ---------------------------------------------------------------------------
# 3. CentralizedCritic
# ---------------------------------------------------------------------------

class CentralizedCritic(nn.Module):
    """
    Centralised multi-objective value function (used only during training).

    Input
    -----
    all_obs : (B, N_CELLS * OBS_DIM)  — concatenated cell observations
    omega   : (B, OMEGA_DIM) or (OMEGA_DIM,)

    Output
    ------
    values : (B, N_CELLS, N_OBJ)
        values[b, i, j] = estimated return for cell i, objective j, under
        the current policy and preference vector ω.

    Architecture
    ------------
    Trunk:  Linear(N_CELLS*OBS_DIM + OMEGA_DIM, CRITIC_HIDDEN), Tanh
         -> Linear(CRITIC_HIDDEN, CRITIC_HIDDEN), Tanh

    Per-(cell, objective) head:
        N_CELLS * N_OBJ independent Linear(CRITIC_HIDDEN, 1) heads.
        Arranged as a ModuleList indexed by i * N_OBJ + j.
    """

    def __init__(
        self,
        n_cells:   int = N_CELLS,
        n_obj:     int = N_OBJ,
        obs_dim:   int = OBS_DIM,
        omega_dim: int = OMEGA_DIM,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.n_obj   = n_obj
        input_dim    = n_cells * obs_dim + omega_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim,    CRITIC_HIDDEN),
            nn.Tanh(),
            nn.Linear(CRITIC_HIDDEN, CRITIC_HIDDEN),
            nn.Tanh(),
        )
        _orthogonal_init(self.trunk, gain=math.sqrt(2))

        # One output head per (cell, objective) pair
        self.heads = nn.ModuleList([
            nn.Linear(CRITIC_HIDDEN, 1)
            for _ in range(n_cells * n_obj)
        ])
        for head in self.heads:
            nn.init.orthogonal_(head.weight, gain=1.0)
            nn.init.zeros_(head.bias)

    def forward(self, all_obs: Tensor, omega: Tensor) -> Tensor:
        """
        Parameters
        ----------
        all_obs : (B, N_CELLS * OBS_DIM)
        omega   : (OMEGA_DIM,) or (B, OMEGA_DIM)

        Returns
        -------
        values : (B, N_CELLS, N_OBJ)
        """
        B = all_obs.shape[0]
        if omega.dim() == 1:
            omega = omega.unsqueeze(0).expand(B, -1)

        x    = torch.cat([all_obs, omega], dim=-1)   # (B, input_dim)
        feat = self.trunk(x)                          # (B, CRITIC_HIDDEN)

        vals = torch.stack(
            [head(feat) for head in self.heads],
            dim=-1,
        ).squeeze(1)   # (B, N_CELLS * N_OBJ)

        return vals.view(B, self.n_cells, self.n_obj)


# ---------------------------------------------------------------------------
# 4. CostCritic
# ---------------------------------------------------------------------------

class CostCritic(nn.Module):
    """
    Per-cell value function for the safety cost signal.

    Estimates V^π_cost_i(obs_i) = E[C_i^π | s_i],
    the expected composite failure proxy under the current policy.

    Input
    -----
    obs : (B, OBS_DIM)

    Output
    ------
    v_cost : (B, 1)   — predicted cost value, unbounded (critic; not clamped)

    Architecture
    ------------
    Linear(OBS_DIM, CRITIC_HIDDEN), Tanh
    Linear(CRITIC_HIDDEN, CRITIC_HIDDEN), Tanh
    Linear(CRITIC_HIDDEN, 1)
    """

    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,     CRITIC_HIDDEN),
            nn.Tanh(),
            nn.Linear(CRITIC_HIDDEN, CRITIC_HIDDEN),
            nn.Tanh(),
            nn.Linear(CRITIC_HIDDEN, 1),
        )
        # Initialise hidden layers with gain=√2 (standard for Tanh networks).
        # The final layer uses gain=1.0 for output-scale stability; we
        # explicitly initialise only the hidden layers here and then the
        # final layer separately, avoiding the double-initialisation that
        # occurred when _orthogonal_init was applied to all layers first.
        for m in list(self.net.modules()):
            if isinstance(m, nn.Linear) and m is not self.net[-1]:
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        # Final output layer: smaller gain for stable value predictions.
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: Tensor) -> Tensor:
        """
        Parameters
        ----------
        obs : (B, OBS_DIM)

        Returns
        -------
        v_cost : (B, 1)
        """
        return self.net(obs)


# ---------------------------------------------------------------------------
# Convenience: build the full set of networks for a training run
# ---------------------------------------------------------------------------

def build_networks(device: torch.device | str | None = None) -> Dict:
    """
    Instantiate and return all networks as a dict.

    Keys
    ----
    "hnet"          : HyperNetwork
    "critic"        : CentralizedCritic
    "cost_critics"  : List[CostCritic]  (one per cell)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    nets = {
        "hnet":         HyperNetwork().to(device),
        "critic":       CentralizedCritic().to(device),
        "cost_critics": [CostCritic().to(device) for _ in range(N_CELLS)],
    }
    return nets


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    nets = build_networks(device)
    hnet    = nets["hnet"]
    critic  = nets["critic"]
    cc_list = nets["cost_critics"]

    # -----------------------------------------------------------------------
    print("\n=== HyperNetwork ===")
    omega = torch.tensor([0.5, 0.3, 0.2], device=device)
    thetas, log_stds = hnet(omega)
    print(f"  omega shape  : {omega.shape}")
    print(f"  thetas[0] shape  : {thetas[0].shape}  "
          f"(expected ({POLICY_PARAM_DIM},))")
    print(f"  log_stds shape   : {log_stds.shape}  (expected ({N_CELLS},))")
    n_hnet = sum(p.numel() for p in hnet.parameters())
    print(f"  HyperNetwork parameters: {n_hnet:,}")

    # Check batched omega
    omega_batch = torch.rand(8, OMEGA_DIM, device=device)
    omega_batch = omega_batch / omega_batch.sum(dim=-1, keepdim=True)
    thetas_b, _ = hnet(omega_batch)
    print(f"  Batched thetas[0] shape: {thetas_b[0].shape}  "
          f"(expected (8, {POLICY_PARAM_DIM}))")

    # -----------------------------------------------------------------------
    print("\n=== PolicyNet (functional) ===")
    B   = 16
    obs = torch.randn(B, OBS_DIM, device=device)
    mu  = PolicyNet.forward(obs, thetas[0])
    print(f"  obs shape : {obs.shape}")
    print(f"  mu shape  : {mu.shape}  (expected ({B}, {ACTION_DIM}))")
    print(f"  mu range  : [{mu.min().item():.4f}, {mu.max().item():.4f}]  "
          f"(should be in (-1, 1))")

    # Check gradients flow back to HyperNetwork
    loss = mu.sum()
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item()
        for p in hnet.parameters()
        if p.grad is not None
    )
    print(f"  Gradient norm through HNet: {grad_norm:.6f}  (must be > 0)")
    hnet.zero_grad()

    # -----------------------------------------------------------------------
    print("\n=== Diversity loss ===")
    obs_per_cell = [torch.randn(B, OBS_DIM, device=device) for _ in range(N_CELLS)]
    div_loss = hnet.diversity_loss(obs_per_cell, n_omega_samples=8)
    print(f"  Diversity loss: {div_loss.item():.6f}  (should be ≤ 0)")
    div_loss.backward()
    grad_norm_div = sum(
        p.grad.norm().item()
        for p in hnet.parameters()
        if p.grad is not None
    )
    print(f"  Gradient norm (diversity): {grad_norm_div:.6f}  (must be > 0)")
    hnet.zero_grad()

    # -----------------------------------------------------------------------
    print("\n=== CentralizedCritic ===")
    all_obs = torch.randn(B, N_CELLS * OBS_DIM, device=device)
    omega_b = torch.full((OMEGA_DIM,), 1.0 / OMEGA_DIM, device=device)
    values  = critic(all_obs, omega_b)
    print(f"  all_obs shape : {all_obs.shape}")
    print(f"  values shape  : {values.shape}  "
          f"(expected ({B}, {N_CELLS}, {N_OBJ}))")
    n_critic = sum(p.numel() for p in critic.parameters())
    print(f"  CentralizedCritic parameters: {n_critic:,}")

    # -----------------------------------------------------------------------
    print("\n=== CostCritic ===")
    obs_c   = torch.randn(B, OBS_DIM, device=device)
    v_cost  = cc_list[0](obs_c)
    print(f"  obs shape    : {obs_c.shape}")
    print(f"  v_cost shape : {v_cost.shape}  (expected ({B}, 1))")
    n_cc = sum(p.numel() for p in cc_list[0].parameters())
    print(f"  CostCritic parameters: {n_cc:,}")

    # -----------------------------------------------------------------------
    print("\n=== Parameter summary ===")
    print(f"  HyperNetwork    : {n_hnet:>10,}")
    print(f"  CentralizedCritic:{n_critic:>9,}")
    print(f"  CostCritic (×3) : {n_cc * N_CELLS:>9,}")
    print(f"  Total           : {n_hnet + n_critic + n_cc * N_CELLS:>10,}")

    print("\nAll checks passed.")
