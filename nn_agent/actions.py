"""
Masking, sampling, and decoding for each of the five decision types -
turns raw network logits + an engine_v2 legality mask into an actual
action in engine_v2's expected format (or None/[] when nothing's legal
or nothing needs deciding).

Movement/cavalry masks come straight from engine_v2.movement (already
exactly [num_hexes, 6] - no translation needed). Buy, target, and
rectification don't have an existing fixed-shape mask in engine_v2 (v1's
buy/target/rectification representations are variable-length action
lists, ported as-is - see those modules' docstrings), so building one is
part of this file:

  - Buy: [num_hexes, 4] - per hex, one of {do nothing, buy infantry,
    convert to cavalry, convert to archers}. Unlike movement (pick ONE
    hex+direction per step), buy allows multiple simultaneous purchases,
    so every hex gets its OWN independent categorical choice rather than
    one flat sample over the whole board.
  - Target: [num_factions + 1] - one of the alive rival factions, or an
    explicit abstain slot (last index). Already small/fixed since
    num_factions is fixed per game.
  - Rectification: [num_hexes], masked to the winner's own contribution
    origin hexes for that battle. SIMPLIFICATION: picks ONE hex and
    sends the entire overflow there (infantry -> cavalry -> archers
    priority, same cascade used everywhere else in the rules), rather
    than v1's freedom to split overflow across multiple origins. Still
    fully rulebook-legal (nothing requires splitting); revisit if this
    turns out to matter.
"""

import jax
import jax.numpy as jnp
import numpy as np

from engine_v2.battle import faction_totals
from engine_v2.state import MAX_STACK_SIZE


def _masked_categorical(rng_key, logits, mask):
    """logits, mask: same-shaped arrays; samples one index along the
    last axis from the masked (illegal -> -inf) distribution."""
    masked_logits = jnp.where(jnp.asarray(mask), logits, -jnp.inf)
    return jax.random.categorical(rng_key, masked_logits)


def decode_movement_or_cavalry(rng_key, logits, mask):
    """logits, mask: [num_hexes, 6]. Returns (hex_index, direction) or
    None if nothing's legal this step (mirrors RandomAgent's "pass")."""
    if not bool(np.any(mask)):
        return None
    flat_logits = logits.reshape(-1)
    flat_mask = np.asarray(mask).reshape(-1)
    idx = int(_masked_categorical(rng_key, flat_logits, flat_mask))
    hex_index, direction = divmod(idx, 6)
    return hex_index, direction


def buy_action_mask(state, legal_buy_actions):
    """[num_hexes, 4] bool, columns = (no-op, buy_infantry,
    convert_to_cavalry, convert_to_archers). Built from
    get_legal_buy_actions' output (engine_v2.buy) - no-op is always
    legal (a faction never has to spend everything)."""
    mask = np.zeros((state.num_hexes, 4), dtype=bool)
    mask[:, 0] = True
    for a in legal_buy_actions:
        if a["type"] == "buy_infantry":
            mask[a["city_hex"], 1] = True
        else:
            mask[a["hex"], 2 if a["unit_type"] == "cavalry" else 3] = True
    return mask


def decode_buy(rng_key, buy_logits, mask):
    """buy_logits, mask: [num_hexes, 4]. Every hex gets its own
    independent sampled choice (see module docstring) - one
    vectorized jax.random.categorical call over the whole board rather
    than num_hexes separate calls."""
    masked_logits = jnp.where(jnp.asarray(mask), buy_logits, -jnp.inf)
    choices = np.asarray(jax.random.categorical(rng_key, masked_logits, axis=-1))

    actions = []
    for hex_index, choice in enumerate(choices):
        if choice == 0:
            continue
        if choice == 1:
            actions.append({"type": "buy_infantry", "city_hex": int(hex_index)})
        else:
            unit_type = "cavalry" if choice == 2 else "archers"
            actions.append({"type": "convert_to_special", "hex": int(hex_index), "unit_type": unit_type})
    return actions


def target_mask(num_factions, legal_targets):
    """[num_factions + 1] bool - last slot (abstain) is always legal."""
    mask = np.zeros(num_factions + 1, dtype=bool)
    mask[-1] = True
    for f in legal_targets:
        mask[f] = True
    return mask


def decode_target(rng_key, target_logits, mask):
    """target_logits, mask: [num_factions + 1]. Returns a faction id, or
    None (abstain)."""
    choice = int(_masked_categorical(rng_key, target_logits, mask))
    abstain_index = target_logits.shape[0] - 1
    return None if choice == abstain_index else choice


def rectification_origin_mask(state, hex_index, winner_faction):
    """[num_hexes] bool - hexes that are actually one of winner_faction's
    own contribution origins in this battle (the only legal
    destinations for sent-back units)."""
    mask = np.zeros(state.num_hexes, dtype=bool)
    for k in range(state.battle_faction.shape[1]):
        if state.battle_faction[hex_index, k] == winner_faction:
            mask[state.battle_origin[hex_index, k]] = True
    return mask


def decode_rectification(rng_key, rectify_logits, state, hex_index, winner_faction):
    """Returns engine_v2.battle.rectify_overflow's send_back list: []
    if the winner isn't actually over the stack cap, otherwise one
    entry sending the whole overflow to a single chosen origin hex
    (see module docstring)."""
    totals = faction_totals(state, hex_index)[winner_faction]
    overflow = int(totals.sum()) - MAX_STACK_SIZE
    if overflow <= 0:
        return []

    mask = rectification_origin_mask(state, hex_index, winner_faction)
    if not np.any(mask):
        return []
    chosen_hex = int(_masked_categorical(rng_key, rectify_logits, mask))

    units = [0, 0, 0]
    remaining = overflow
    for ut in range(3):
        take = min(int(totals[ut]), remaining)
        units[ut] = take
        remaining -= take
        if remaining <= 0:
            break

    return [{"origin_hex": chosen_hex, "units": units}]
