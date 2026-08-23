"""
Runs a game with random agents end-to-end, writing board_state.json in
a diff-based format: terrain is stored once (it never changes), and
each turn is stored as a sparse keyframe (state at turn start) plus
the actual chosen actions and resulting board deltas for each of that
turn's 9 phases (buy, 3 movement steps, 4 cavalry steps, battle).

This keeps file size proportional to how much actually happens in the
game, rather than to board size or how many turns have passed -
see the conversation this was designed in for the reasoning.

Agent mix: every faction is a HeuristicAgent. Faction id -> color is
fixed (see hex_visualizer.py's FACTION_COLORS: 0=red, 1=blue, 2=green,
3=purple, 4=orange, 5=brown, 6=pink, 7=grey).

File shape:
{
  "radius": int, "num_factions": int,
  "terrain": {"q_r_s": terrain_str, ...},
  "turns": [turn_record, ...]   # see engine_v2/turn.py:run_turn_and_log
}
"""

import json
import random
import time

from engine_v2.setup import create_initial_state
from engine_v2.state import TERRAIN_TYPES
from engine_v2.turn import run_turn_and_log, check_game_end
from engine_v2.agents.heuristic_agent import make_heuristic_agents

OUTPUT_FILE = "board_state.json"
RADIUS = 8
NUM_FACTIONS = 8
MAX_TURNS = 100

# Seeded from the system clock so every run produces a different game.
# Printed at the end so a specific run can still be reproduced later by
# hardcoding this value back in, if that's ever useful for debugging.
SEED = int(time.time() * 1000) % (2 ** 31)


def main():
    rng = random.Random(SEED)
    state = create_initial_state(radius=RADIUS, num_factions=NUM_FACTIONS, seed=SEED)
    decide_buy, decide_movement, decide_cavalry, decide_target, decide_rectification = make_heuristic_agents(
        NUM_FACTIONS, seed=SEED,
    )

    terrain_map = {
        f"{q}_{r}_{s}": TERRAIN_TYPES[int(t)]
        for (q, r, s), t in zip(state.grid.coords, state.terrain)
    }

    turns = []
    while not check_game_end(state, max_turns=MAX_TURNS):
        state, turn_record = run_turn_and_log(
            state, decide_buy, decide_movement, decide_cavalry, decide_target, decide_rectification, rng=rng,
        )
        turns.append(turn_record)

    json_dict = {
        "radius": RADIUS,
        "num_factions": NUM_FACTIONS,
        "terrain": terrain_map,
        "turns": turns,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(json_dict, f)

    print(f"Ran {len(turns)} turns (seed={SEED}), wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()