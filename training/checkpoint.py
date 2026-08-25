"""
Saving/loading trained network weights. Lives here rather than in
agents/nn_agent/network.py because it's a training/iteration-bookkeeping
concern (optimizer state, iteration count) - network.py stays purely
architectural.
"""

import torch

from agents.nn_agent.network import build_network


def save_checkpoint(path, network, optimizer=None, extra=None):
    """Saves network.state_dict() plus enough constructor arguments to
    rebuild an identically-shaped PolicyNetwork later (num_hexes isn't
    among them - see build_network's docstring, torch Linear layers
    don't depend on it). `extra`: an optional dict of additional
    metadata to stash alongside (e.g. {"iteration": i})."""
    payload = {
        "model_state_dict": network.state_dict(),
        "num_factions": network.num_factions,
        "hidden_dim": network.hidden_dim,
        "num_mp_rounds": len(network.mp_rounds),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device=None, load_optimizer_into=None):
    """Returns (network, payload) - payload is the full saved dict (so
    callers can read back `extra` metadata like iteration count).
    `load_optimizer_into`: an existing optimizer instance to restore
    saved optimizer state into, if the checkpoint has any."""
    payload = torch.load(path, map_location=device or "cpu")
    network = build_network(
        payload["num_factions"], hidden_dim=payload["hidden_dim"],
        num_mp_rounds=payload["num_mp_rounds"], device=device,
    )
    network.load_state_dict(payload["model_state_dict"])
    if load_optimizer_into is not None and "optimizer_state_dict" in payload:
        load_optimizer_into.load_state_dict(payload["optimizer_state_dict"])
    return network, payload
