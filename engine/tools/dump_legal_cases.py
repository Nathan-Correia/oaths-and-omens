"""
Dumps legal-action generation (buy lists, movement/cavalry masks) for comparison.

WHY THIS EXISTS SEPARATELY from the turn traces: those replay the decisions an
agent MADE, so they verify the engine's response to a chosen action but never
check the menu it was chosen from. get_legal_buy_actions and the movement masks
are the agent-facing API - every one of the twelve agents indexes into them - and
before this file they were entirely untested. Their ORDER is part of the contract
too, so these are compared element by element, not as sets.

Usage:  python engine/tools/dump_legal_cases.py > engine/tests/data/legal_cases.txt

Format, per case:
  LEGAL_CASE <name>
  <state>
  FACTION <f>
  BUY <count> [<type> <hex> <unit> <upgrade>]*
  MOVEMASK <count> [<hex> <dir>]*      (set cells only)
  CAVMASK  <count> [<hex> <dir>]*
  ... one FACTION block per faction
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402

_bootstrap.install()

import numpy as np  # noqa: E402

from engine.buy import get_legal_buy_actions  # noqa: E402
from engine.movement import legal_cavalry_mask, legal_movement_mask  # noqa: E402

from dump_phase_cases import SCENARIOS, _clone, _perturb, harvest  # noqa: E402
from state_io import write_state  # noqa: E402

UNIT_INDEX = {"infantry": 0, "cavalry": 1, "archers": 2}
UPGRADE_INDEX = {"barracks": 0, "workshop": 1, "temple": 2}
BUY_TYPE_INDEX = {
    "buy_infantry": 0,
    "convert_to_special": 1,
    "build_outpost": 2,
    "upgrade_outpost": 3,
}


def buy_tokens(action):
    kind = BUY_TYPE_INDEX[action["type"]]
    if action["type"] == "buy_infantry":
        return f"{kind} {action['city_hex']} 0 0"
    if action["type"] == "convert_to_special":
        return f"{kind} {action['hex']} {UNIT_INDEX[action['unit_type']]} 0"
    if action["type"] == "build_outpost":
        return f"{kind} {action['hex']} {UNIT_INDEX[action['unit_type']]} 0"
    return f"{kind} {action['hex']} 0 {UPGRADE_INDEX[action['upgrade']]}"


def mask_tokens(mask):
    rows, cols = np.nonzero(mask)
    return len(rows), " ".join(f"{int(r)} {int(c)}" for r, c in zip(rows, cols))


def main():
    out = sys.stdout
    rng = random.Random(4242)
    cases = []

    for radius, num_factions, seed, turns, agent_key in SCENARIOS:
        for i, snap in enumerate(harvest(radius, num_factions, seed, turns, agent_key)):
            cases.append((f"real/{agent_key}-r{radius}f{num_factions}s{seed}t{i}", _clone(snap)))
            # Perturbed states matter as much here as for the phases: they are what
            # put outposts, upgrades and scattered armies on the board, and so what
            # makes the build_outpost and upgrade branches produce anything at all.
            cases.append((f"perturbed/{agent_key}-r{radius}f{num_factions}s{seed}t{i}",
                          _perturb(snap, rng)))

    out.write(f"LEGAL_CASES {len(cases)}\n")
    for name, state in cases:
        out.write(f"LEGAL_CASE {name}\n")
        write_state(out, state)
        for faction in range(state.num_factions):
            out.write(f"FACTION {faction}\n")
            actions = get_legal_buy_actions(state, faction)
            tokens = " ".join(buy_tokens(a) for a in actions)
            out.write(f"BUY {len(actions)}" + (" " + tokens if tokens else "") + "\n")
            for label, mask in (("MOVEMASK", legal_movement_mask(state, faction)),
                                ("CAVMASK", legal_cavalry_mask(state, faction))):
                count, toks = mask_tokens(mask)
                out.write(f"{label} {count}" + (" " + toks if toks else "") + "\n")


if __name__ == "__main__":
    main()
