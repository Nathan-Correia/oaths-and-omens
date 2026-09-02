"""
Observation encoding: turns an engine ArrayState into the numeric
arrays a neural net actually reads.

Two things a raw ArrayState doesn't have that a network needs:
  1. Everything as floats, categoricals one-hot'd - terrain/faction ids
     are just integer codes right now, meaningless as raw numbers to a
     network (terrain index 4 isn't "twice as much terrain" as index 2).
  2. "Mine vs. theirs" instead of raw faction ids. This is what makes
     self-play with ONE shared network work: the network is called once
     per faction's decision, each time re-encoding the SAME board from
     that faction's own point of view, so it never has to learn
     "faction 3 behaves like X" - only "my stuff vs. everyone else's."

Per-hex features (NUM_TERRAIN_TYPES + 13 numbers per hex, computed as
PER_HEX_FEATURES below rather than hardcoded - terrain types have
changed before, e.g. forest's removal, and will again):
  terrain one-hot (NUM_TERRAIN_TYPES) + city ownership relative to the
  acting faction (3: none / mine / enemy's - capital vs. outpost isn't
  distinguished here, just like real vs. structural-shot-only defense
  isn't; revisit if that turns out to matter) + colourless-city-placement
  relative to the acting faction (3: none / placed by me / placed by
  someone else - see engine/placement.py's run_city_setup and
  engine/state.py's city_placer field; this is what gives
  decide_placement/decide_draft/decide_swap any visibility into the
  board at all during setup, since city_owner is NO_FACTION for
  everyone until the draft finalizes) + my army units here (3,
  infantry/cavalry/archers, /6) + enemy army units here (3, /6) +
  locked-in-battle flag (1). "Army units here" merges the peaceful army
  and battle-contribution cases (a hex is exactly one or the other,
  never both) into a single pair of features rather than carrying
  separate always-partially-zero columns for each.

  city_placer stays populated (but static and no longer authoritative)
  for the rest of the game once setup finishes - see its own docstring
  in engine/state.py - so this feature block is harmless outside setup
  too: at worst the network learns to treat it as a fixed per-hex bias
  during turn-phase decisions, never a source of live/confusing signal,
  the same precedent as state.py's `alive` field.

Global features (6 numbers), not tied to any one hex: my gold, my
kill-XP (both normalized against a rough scale, not a hard cap - values
can exceed 1), fraction of OTHER factions still alive (vestigial now
that elimination is gone - see engine/state.py's alive field docstring;
always 1.0 in practice), how far through the game we are (turn_number /
max_turns), my victory points, and the leading rival's victory points
(both / VP_TO_WIN) - the actual win condition, so the network needs to
see how close everyone is to it.

NOT COVERED YET: the new resources/outpost-upgrade rules (see
rulebook.md) aren't reflected in this observation - resources/upgrades
exist in engine.state but this network can't see them. Revisit once the
NN agent's observation/action space is actually being worked on.

SIMPLIFICATION worth knowing about: enemy presence is aggregated across
all non-self factions into one "the enemy" signal, per hex - the network
can't tell WHICH opponent is where, just that someone hostile is. Fine
for a first version; revisit if that turns out to matter once training
is actually running.
"""

import numpy as np

from engine.collect import VP_TO_WIN
from engine.state import NO_FACTION, TERRAIN_TYPES

NUM_TERRAIN_TYPES = len(TERRAIN_TYPES)
PER_HEX_FEATURES = NUM_TERRAIN_TYPES + 3 + 3 + 3 + 3 + 1
GLOBAL_FEATURES = 6

GOLD_SCALE = 100.0
KILL_XP_SCALE = 20.0
UNIT_SCALE = 6.0  # MAX_STACK_SIZE
VP_SCALE = float(VP_TO_WIN)


def _per_hex_army_features(state, faction):
    """(my_units, enemy_units): each [num_hexes, 3], merging the
    peaceful-army and locked-battle cases (mutually exclusive per hex -
    see module docstring)."""
    n = state.num_hexes
    my_units = np.zeros((n, 3), dtype=np.float32)
    enemy_units = np.zeros((n, 3), dtype=np.float32)

    peaceful = ~state.locked
    is_mine = peaceful & (state.army_faction == faction)
    is_enemy = peaceful & (state.army_faction != faction) & (state.army_faction != NO_FACTION)
    my_units[is_mine] = state.army_units[is_mine]
    enemy_units[is_enemy] = state.army_units[is_enemy]

    if np.any(state.locked):
        mine_mask = state.battle_faction == faction
        enemy_mask = (state.battle_faction != faction) & (state.battle_faction != NO_FACTION)
        my_units += np.where(mine_mask[:, :, None], state.battle_units, 0).sum(axis=1)
        enemy_units += np.where(enemy_mask[:, :, None], state.battle_units, 0).sum(axis=1)

    return my_units, enemy_units


def encode_observation(state, faction, max_turns=100):
    """Returns (per_hex: float32[num_hexes, PER_HEX_FEATURES],
    global_feats: float32[GLOBAL_FEATURES]), plain numpy arrays, from
    `faction`'s point of view. Framework-agnostic on purpose - callers
    (agent.py) own converting these to torch tensors and placing them on
    whatever device the network lives on."""
    n = state.num_hexes

    terrain_onehot = np.eye(NUM_TERRAIN_TYPES, dtype=np.float32)[state.terrain]

    city_rel = np.zeros((n, 3), dtype=np.float32)
    city_rel[state.city_owner == NO_FACTION, 0] = 1.0
    city_rel[state.city_owner == faction, 1] = 1.0
    city_rel[(state.city_owner != NO_FACTION) & (state.city_owner != faction), 2] = 1.0

    placer_rel = np.zeros((n, 3), dtype=np.float32)
    placer_rel[state.city_placer == NO_FACTION, 0] = 1.0
    placer_rel[state.city_placer == faction, 1] = 1.0
    placer_rel[(state.city_placer != NO_FACTION) & (state.city_placer != faction), 2] = 1.0

    my_units, enemy_units = _per_hex_army_features(state, faction)

    locked_flag = state.locked.astype(np.float32)[:, None]

    per_hex = np.concatenate(
        [terrain_onehot, city_rel, placer_rel, my_units / UNIT_SCALE, enemy_units / UNIT_SCALE, locked_flag],
        axis=1,
    )

    alive_others = int(np.sum(state.alive)) - (1 if state.alive[faction] else 0)
    total_others = max(1, state.num_factions - 1)

    rival_vp = [int(state.victory_points[f]) for f in range(state.num_factions) if f != faction]
    best_rival_vp = max(rival_vp) if rival_vp else 0

    global_feats = np.array(
        [
            float(state.gold[faction]) / GOLD_SCALE,
            float(state.kill_xp[faction]) / KILL_XP_SCALE,
            alive_others / total_others,
            float(state.turn_number) / max_turns,
            float(state.victory_points[faction]) / VP_SCALE,
            float(best_rival_vp) / VP_SCALE,
        ],
        dtype=np.float32,
    )

    return per_hex, global_feats
