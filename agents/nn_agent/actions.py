"""
Masking, sampling, and decoding for each of the eight decision types -
turns raw network logits + an engine legality mask into an actual action
in engine's expected format (or None/[] when nothing's legal or nothing
needs deciding).

Movement/cavalry masks come straight from engine.movement (already
exactly [num_hexes, 6] - no translation needed). Buy, target, and
rectification don't have an existing fixed-shape mask in engine (v1's
buy/target/rectification representations are variable-length action
lists, ported as-is - see those modules' docstrings), so building one is
part of this file:

  - Buy: [num_hexes, 5] - per hex, one of {do nothing, buy infantry,
    convert to cavalry, convert to archers, build an outpost}. Unlike
    movement (pick ONE hex+direction per step), buy allows multiple
    simultaneous purchases, so every hex gets its OWN independent
    categorical choice rather than one flat sample over the whole board.
    build_outpost's *unit-type* choice (which of infantry/cavalry/
    archers standing on that hex gets sacrificed) is deliberately NOT
    part of the action space - resolved at decode time via a fixed
    sacrifice priority (infantry first), the same simplification
    agents/greedy_agent.py's greedy_buy already uses for the identical
    problem (see _OUTPOST_UNIT_PRIORITY/_outpost_unit_type_by_hex
    below). NOT COVERED YET: engine.buy's upgrade_outpost action (see
    rulebook.md's Outpost Upgrades) has no column here, so this agent
    can never propose one - get_legal_buy_actions may return
    upgrade_outpost entries, but buy_action_mask below simply doesn't
    set a column for them, so they're never selected.
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
  - Placement/draft/swap (see engine/placement.py's run_city_setup):
    all three are fundamentally "how much do I want this hex as my
    capital," just over different candidate subsets of hexes - so all
    three share one decode step (decode_capital_choice) reading the
    network's capital_pref head, masked to whatever that decision's
    candidates are (capital_choice_mask_from_pool for draft's
    legal_pool list, capital_choice_mask_pair for swap's exactly-two-
    hexes choice; placement's legal_mask is already the right shape).

Sampling uses torch.multinomial (not torch.distributions.Categorical) on
masked-then-softmaxed logits specifically because multinomial accepts an
explicit torch.Generator - Categorical.sample() doesn't - and this
package threads an explicit, seeded, per-agent-set generator throughout
(see agent.py) rather than touching torch's global RNG state, matching
this codebase's convention everywhere else of explicit RNG threading.

Every mask this file ever samples from is guaranteed non-empty by
construction: buy's no-op column is always legal; decode_movement_or_
cavalry/decode_rectification explicitly check for an all-False mask and
return early (None/[]) before ever sampling; engine/placement.py
guarantees legal_mask/legal_pool/the swap pair are always non-empty.
"""

import numpy as np
import torch

from engine.battle import faction_totals

# Which unit type to sacrifice first when a hex offers a choice of more
# than one for building an outpost there - mirrors greedy_agent.py's
# constant of the same name/purpose (duplicated rather than imported: a
# 3-tuple literal is cheap enough that duplicating it beats reaching into
# a sibling scripted-agent module's private constant).
_OUTPOST_UNIT_PRIORITY = ("infantry", "cavalry", "archers")


def _masked_categorical(rng_gen, logits, mask):
    """logits: torch tensor (1-D). mask: bool array-like, same length.
    Samples one index from the masked (illegal -> -inf) distribution
    using rng_gen (a torch.Generator on the same device as logits -
    required by torch.multinomial)."""
    mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=logits.device)
    masked_logits = torch.where(mask_t, logits, torch.full_like(logits, float("-inf")))
    probs = torch.softmax(masked_logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=rng_gen).item())


def decode_movement_or_cavalry(rng_gen, logits, mask):
    """logits: [num_hexes, 6] torch tensor. mask: [num_hexes, 6] bool.
    Returns (hex_index, direction) or None if nothing's legal this step
    (mirrors RandomAgent's "pass")."""
    if not bool(np.any(mask)):
        return None
    flat_logits = logits.reshape(-1)
    flat_mask = np.asarray(mask).reshape(-1)
    idx = _masked_categorical(rng_gen, flat_logits, flat_mask)
    hex_index, direction = divmod(idx, 6)
    return hex_index, direction


def buy_action_mask(state, legal_buy_actions):
    """[num_hexes, 5] bool, columns = (no-op, buy_infantry,
    convert_to_cavalry, convert_to_archers, build_outpost). Built from
    get_legal_buy_actions' output (engine.buy) - no-op is always legal
    (a faction never has to spend everything). upgrade_outpost entries
    in legal_buy_actions are ignored here (see module docstring)."""
    mask = np.zeros((state.num_hexes, 5), dtype=bool)
    mask[:, 0] = True
    for a in legal_buy_actions:
        if a["type"] == "buy_infantry":
            mask[a["city_hex"], 1] = True
        elif a["type"] == "convert_to_special":
            mask[a["hex"], 2 if a["unit_type"] == "cavalry" else 3] = True
        elif a["type"] == "build_outpost":
            mask[a["hex"], 4] = True
    return mask


def _outpost_unit_type_by_hex(legal_buy_actions):
    """{hex_index: unit_type} - the highest-priority unit type actually
    available to sacrifice for a build_outpost at each hex offering one
    (mirrors greedy_agent.py's greedy_buy `by_hex` resolution).
    get_legal_buy_actions emits one build_outpost entry per (hex,
    unit_type present) - which one to actually use is resolved here,
    deterministically, from that real data rather than re-deriving
    buy.py's own legality logic."""
    by_hex = {}
    for a in legal_buy_actions:
        if a["type"] == "build_outpost":
            by_hex.setdefault(a["hex"], set()).add(a["unit_type"])
    return {
        hex_index: next(ut for ut in _OUTPOST_UNIT_PRIORITY if ut in options)
        for hex_index, options in by_hex.items()
    }


def decode_buy(rng_gen, buy_logits, mask, legal_buy_actions):
    """buy_logits, mask: [num_hexes, 5]. Every hex gets its own
    independent sampled choice (see module docstring) - one vectorized
    torch.multinomial call over the whole board rather than num_hexes
    separate calls. legal_buy_actions: the same list buy_action_mask was
    built from - needed again here to resolve build_outpost's sacrificed
    unit_type (see _outpost_unit_type_by_hex)."""
    mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=buy_logits.device)
    masked_logits = torch.where(mask_t, buy_logits, torch.full_like(buy_logits, float("-inf")))
    probs = torch.softmax(masked_logits, dim=-1)
    choices = torch.multinomial(probs, 1, generator=rng_gen).squeeze(-1).tolist()
    outpost_unit_type = _outpost_unit_type_by_hex(legal_buy_actions)

    actions = []
    for hex_index, choice in enumerate(choices):
        if choice == 0:
            continue
        if choice == 1:
            actions.append({"type": "buy_infantry", "city_hex": int(hex_index)})
        elif choice in (2, 3):
            unit_type = "cavalry" if choice == 2 else "archers"
            actions.append({"type": "convert_to_special", "hex": int(hex_index), "unit_type": unit_type})
        else:  # choice == 4
            unit_type = outpost_unit_type.get(hex_index)
            if unit_type is not None:  # always set given mask[hex, 4] - defensive
                actions.append({"type": "build_outpost", "hex": int(hex_index), "unit_type": unit_type})
    return actions


def target_mask(num_factions, legal_targets):
    """[num_factions + 1] bool - last slot (abstain) is always legal."""
    mask = np.zeros(num_factions + 1, dtype=bool)
    mask[-1] = True
    for f in legal_targets:
        mask[f] = True
    return mask


def decode_target(rng_gen, target_logits, mask):
    """target_logits, mask: [num_factions + 1]. Returns a faction id, or
    None (abstain)."""
    choice = _masked_categorical(rng_gen, target_logits, mask)
    abstain_index = target_logits.shape[0] - 1
    return None if choice == abstain_index else choice


def rectification_origin_mask(state, hex_index, winner_faction):
    """[num_hexes] bool - hexes that are actually one of winner_faction's
    own contribution origins in this battle (the only legal destinations
    for sent-back units)."""
    mask = np.zeros(state.num_hexes, dtype=bool)
    for k in range(state.battle_faction.shape[1]):
        if state.battle_faction[hex_index, k] == winner_faction:
            mask[state.battle_origin[hex_index, k]] = True
    return mask


def decode_rectification(rng_gen, rectify_logits, state, hex_index, winner_faction, cap):
    """Returns engine.battle.rectify_overflow's send_back list: [] if
    the winner isn't actually over `cap`, otherwise one entry sending
    the whole overflow to a single chosen origin hex (see module
    docstring). `cap` is normally MAX_STACK_SIZE, but is 0 when the
    winner just won a battle on a foreign capital (see turn.py's
    _run_battle_phase) - capitals are uncapturable, so the winner has to
    send everything back."""
    totals = faction_totals(state, hex_index)[winner_faction]
    overflow = int(totals.sum()) - cap
    if overflow <= 0:
        return []

    mask = rectification_origin_mask(state, hex_index, winner_faction)
    if not np.any(mask):
        return []
    chosen_hex = _masked_categorical(rng_gen, rectify_logits, mask)

    units = [0, 0, 0]
    remaining = overflow
    for ut in range(3):
        take = min(int(totals[ut]), remaining)
        units[ut] = take
        remaining -= take
        if remaining <= 0:
            break

    return [{"origin_hex": chosen_hex, "units": units}]


def capital_choice_mask_from_pool(num_hexes, legal_pool):
    """[num_hexes] bool - used for decide_draft's legal_pool (a plain
    list of hex indices, guaranteed non-empty by run_city_setup)."""
    mask = np.zeros(num_hexes, dtype=bool)
    mask[legal_pool] = True
    return mask


def capital_choice_mask_pair(num_hexes, hex_a, hex_b):
    """[num_hexes] bool, exactly two hexes True - used for decide_swap's
    choice between the leftover city and what the placer already
    drafted."""
    mask = np.zeros(num_hexes, dtype=bool)
    mask[hex_a] = True
    mask[hex_b] = True
    return mask


def decode_capital_choice(rng_gen, capital_pref_logits, mask):
    """capital_pref_logits: [num_hexes]. mask: [num_hexes] bool. Shared
    decode step for decide_placement/decide_draft/decide_swap (see
    agent.py) - all three sample from the same per-hex "how much do I
    want this hex" score, masked to that decision's candidate set."""
    return _masked_categorical(rng_gen, capital_pref_logits, mask)
