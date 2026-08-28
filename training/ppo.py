"""
The actual PPO update: clipped-surrogate policy loss + value loss +
entropy bonus, computed by re-running the network on stored gae.
TrainingSample data under the *current* weights and comparing against
the log-prob recorded at collection time.

Unlike an earlier version of this file, this batches: every sample in a
minibatch (per_hex/global_feats/battle_hex_index) is stacked into one
real batch and run through network(...) exactly once per minibatch, not
once per sample - network.py's forward() is natively batched for
exactly this reason (see that module's docstring). The forward pass
itself doesn't care what decision_kind produced a given sample (every
kind runs through the same shared trunk); only turning each sample's
slice of the batched output into a log-prob/entropy needs to know its
kind, and even that is vectorized: samples are grouped by decision_kind
(mask/action shapes are uniform within a kind) and each group's log-
prob/entropy is computed in one call rather than looping per sample.
"""

import random

import numpy as np
import torch

from .buffer import BUY, CAVALRY, DRAFT, MOVEMENT, PLACEMENT, RECTIFY, SWAP, TARGET

_HEAD_BY_KIND = {
    MOVEMENT: "movement",
    CAVALRY: "cavalry",
    TARGET: "target",
    RECTIFY: "rectify",
    PLACEMENT: "capital_pref",
    DRAFT: "capital_pref",
    SWAP: "capital_pref",
}

_MASK_SENTINEL = -1e9  # see _masked_log_probs_and_entropy - a large finite
# negative number, not -inf, specifically to keep gradients finite


def _masked_log_probs_and_entropy(logits, mask_t):
    """logits, mask_t: same shape, last dim is the categorical axis (any
    number of leading batch/group dims). Returns (log_probs, entropy) -
    entropy only reduces the last (categorical) axis, so leading dims are
    preserved; a caller with an extra grouping axis beyond the
    categorical one (buy's per-hex choices) reduces that separately.

    Uses a large *finite* negative sentinel for masked-out entries, not
    literal -inf - this is the standard masked-softmax trick for exactly
    this reason: torch.where(mask, f(x), g(x)) evaluates BOTH branches in
    the forward pass, and autograd's backward rule for where computes
    mask * d(f)/dx + (1-mask) * d(g)/dx at every position, even ones the
    forward output doesn't select. If the masked branch's own inputs
    contain a literal -inf, d(f)/dx can itself be NaN at that position
    (from probs=0 times log_probs=-inf inside the entropy formula), and
    "not selected" doesn't save it: (1-mask)=0 times a NaN is still NaN,
    not 0. A finite sentinel this large still underflows to an exact 0.0
    probability after softmax (same effective masking as -inf, since
    exp(-1e9 - anything) underflows completely in float32) but every
    intermediate value along the way stays finite, so no NaN ever enters
    the gradient. actions.py's sampling-time masking can safely keep
    using literal -inf instead - it only ever runs under
    torch.inference_mode() (no autograd graph is built there at all), and
    produces numerically identical log-probs for the legal/chosen entries
    either way, so old_log_prob (from actions.py) and new_log_prob (from
    here) stay on a consistent basis for PPO's ratio."""
    masked_logits = torch.where(mask_t, logits, torch.full_like(logits, _MASK_SENTINEL))
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    probs = log_probs.exp()
    entropy = -torch.where(mask_t, probs * log_probs, torch.zeros_like(probs)).sum(dim=-1)
    return log_probs, entropy


def _recompute_batch(out, batch, device):
    """out: the dict network(...) returned for this whole minibatch (every
    tensor's leading dim is len(batch)). batch: list[gae.TrainingSample],
    same order as the batch dim. Groups samples by decision_kind (mask/
    action shapes are uniform within a kind) and computes each group's
    new log-prob/entropy in one vectorized call.

    Returns (perm, new_log_probs, entropy): new_log_probs/entropy are
    [len(batch)] tensors, but in *grouped* order, not original order -
    `perm` is the list of original batch indices in that same grouped
    order, so the caller must gather old_log_prob/advantage/return/value
    using this same perm to keep every per-sample tensor aligned (cheaper
    than sorting the group outputs back to original order)."""
    groups = {}
    for i, s in enumerate(batch):
        groups.setdefault(s.decision_kind, []).append(i)

    perm = []
    log_prob_parts = []
    entropy_parts = []

    for kind, idxs in groups.items():
        group = [batch[i] for i in idxs]
        idx_t = torch.tensor(idxs, device=device)

        if kind == BUY:
            logits = out["buy"][idx_t]  # [G, num_hexes, 5]
            mask_t = torch.as_tensor(np.stack([s.mask for s in group]), dtype=torch.bool, device=device)
            log_probs, pos_entropy = _masked_log_probs_and_entropy(logits, mask_t)  # pos_entropy: [G, num_hexes]
            choice_t = torch.as_tensor(np.stack([s.action_repr for s in group]), dtype=torch.long, device=device)
            lp = log_probs.gather(2, choice_t.unsqueeze(-1)).squeeze(-1).sum(dim=1)  # [G] - joint over hexes
            ent = pos_entropy.sum(dim=1)  # [G]
        else:
            head = _HEAD_BY_KIND[kind]
            logits = out[head][idx_t]
            if kind in (MOVEMENT, CAVALRY):
                logits = logits.reshape(len(idxs), -1)  # [G, num_hexes*6]
                mask_np = np.stack([np.asarray(s.mask).reshape(-1) for s in group])
            else:
                mask_np = np.stack([s.mask for s in group])
            mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
            log_probs, ent = _masked_log_probs_and_entropy(logits, mask_t)  # ent: [G] already
            action_t = torch.as_tensor([s.action_repr for s in group], dtype=torch.long, device=device)
            lp = log_probs.gather(1, action_t.unsqueeze(-1)).squeeze(-1)  # [G]

        perm.extend(idxs)
        log_prob_parts.append(lp)
        entropy_parts.append(ent)

    return perm, torch.cat(log_prob_parts), torch.cat(entropy_parts)


def update(network, optimizer, samples, neighbor_table_t, device, clip_eps=0.2, epochs=4,
           minibatch_size=256, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, verbose=False):
    """One PPO update over `samples` (a flat list of gae.TrainingSample),
    `epochs` passes with freshly-shuffled minibatches each pass. Each
    minibatch gets exactly one network(...) forward call (see module
    docstring) and one backward()/optimizer.step(). `verbose`: print one
    line per epoch with the combined loss and its policy/value/entropy
    components separately - the combined number alone can look flat/stuck
    while actually being dominated by one term (e.g. an unnormalized
    value loss, since rewards are raw VP-delta swings), so the breakdown
    is what actually shows which piece is or isn't improving."""
    network.train()
    indices = list(range(len(samples)))
    for epoch in range(epochs):
        random.shuffle(indices)
        epoch_losses = []
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []
        for start in range(0, len(indices), minibatch_size):
            batch_idx = indices[start:start + minibatch_size]
            batch = [samples[i] for i in batch_idx]

            per_hex_t = torch.from_numpy(np.stack([s.per_hex for s in batch])).to(device)
            global_feats_t = torch.from_numpy(np.stack([s.global_feats for s in batch])).to(device)
            battle_hex_t = torch.tensor([s.battle_hex_index for s in batch], dtype=torch.long, device=device)

            out = network(per_hex_t, global_feats_t, neighbor_table_t, battle_hex_t)

            perm, new_log_probs, entropy = _recompute_batch(out, batch, device)
            perm_t = torch.tensor(perm, device=device)

            old_log_probs = torch.tensor([batch[i].old_log_prob for i in perm], device=device)
            advantages = torch.tensor([batch[i].advantage for i in perm], device=device)
            returns = torch.tensor([batch[i].return_ for i in perm], device=device)
            values = out["value"][perm_t]

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2)
            value_loss = (values - returns) ** 2
            loss = (policy_loss + value_coef * value_loss - entropy_coef * entropy).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_policy_losses.append(policy_loss.mean().item())
            epoch_value_losses.append(value_loss.mean().item())
            epoch_entropies.append(entropy.mean().item())

        if verbose:
            def _mean(xs):
                return sum(xs) / len(xs) if xs else float("nan")
            print(f"  ppo epoch {epoch + 1}/{epochs}: loss={_mean(epoch_losses):.4f} "
                  f"(policy={_mean(epoch_policy_losses):.4f}, "
                  f"value={_mean(epoch_value_losses):.4f}, "
                  f"entropy={_mean(epoch_entropies):.4f})")
    network.eval()
