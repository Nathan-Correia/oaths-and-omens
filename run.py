"""
Runs a game with random agents end-to-end, writing board_state.json in
a diff-based format: terrain is stored once (it never changes), and
each turn is stored as a sparse keyframe (state at turn start) plus
the actual chosen actions and resulting board deltas for each of that
turn's 9 phases (buy, 3 movement steps, 4 cavalry steps, battle).

This keeps file size proportional to how much actually happens in the
game, rather than to board size or how many turns have passed -
see the conversation this was designed in for the reasoning.

Agent mix: factions 0-2 are RandomAgent, 3-5 are SmartRandomAgent,
6-7 are HeuristicAgent. Faction id -> color is fixed (see
hex_visualizer.py's FACTION_COLORS: 0=red, 1=blue, 2=green, 3=purple,
4=orange, 5=brown, 6=pink, 7=grey), so with this assignment: red/
blue/green are pure random, purple/orange/brown are smart-random,
pink/grey are the heuristic agent.

File shape:
{
  "radius": int, "num_factions": int,
  "terrain": {"q_r_s": terrain_str, ...},
  "turns": [turn_record, ...]   # see engine/turn.py:run_turn_and_log
}
"""

import json
import random
import time

from engine.setup import create_initial_state
from engine.turn import run_turn_and_log, check_game_end
from engine.agents.random_agent import RandomAgent
from engine.agents.smart_random_agent import SmartRandomAgent
from engine.agents.heuristic_agent import HeuristicAgent

OUTPUT_FILE = "board_state.json"
RADIUS = 8
NUM_FACTIONS = 8
MAX_TURNS = 100

# Seeded from the system clock so every run produces a different game.
# Printed at the end so a specific run can still be reproduced later by
# hardcoding this value back in, if that's ever useful for debugging.
SEED = int(time.time() * 1000) % (2 ** 31)

AGENT_ASSIGNMENT = {
    0: HeuristicAgent,
    1: HeuristicAgent,
    2: HeuristicAgent,
    3: HeuristicAgent,
    4: HeuristicAgent,
    5: HeuristicAgent,
    6: HeuristicAgent,
    7: HeuristicAgent,
}


def main():
    rng = random.Random(SEED)
    state = create_initial_state(radius=RADIUS, num_factions=NUM_FACTIONS, seed=SEED)
    agents = {f: AGENT_ASSIGNMENT[f](f, rng=random.Random(SEED + 1000 + f)) for f in state.players}

    terrain_map = {f"{c[0]}_{c[1]}_{c[2]}": h.terrain for c, h in state.board.items()}

    turns = []
    while not check_game_end(state, max_turns=MAX_TURNS):
        state, turn_record = run_turn_and_log(state, agents, rng=rng)
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