"""
Runs a game with random agents end-to-end, capturing a board snapshot
after every movement and cavalry step (no battle/buy/income visuals
for v1 - see conversation) and writing them to board_state.json in
the same format hex_visualizer.py already reads.

engine/state.py's HexState.to_dict() emits the visualizer's exact
field names directly (terrain/city/troops) - there's no separate
translation step. This script just calls state.to_dict()["hexes"]
after each step and appends it to the states list.
"""

import json
import random

from engine.setup import create_initial_state
from engine.turn import run_turn, check_game_end
from engine.agents.random_agent import RandomAgent
from engine.agents.smart_random_agent import SmartRandomAgent

OUTPUT_FILE = "board_state.json"
RADIUS = 8
NUM_FACTIONS = 8
MAX_TURNS = 20   # kept short for a first playable log; bump up once this is verified
SEED = 42


def main():
    rng = random.Random(SEED)
    state = create_initial_state(radius=RADIUS, num_factions=NUM_FACTIONS, seed=SEED)
    agents = {f: SmartRandomAgent(f, rng=random.Random(1000 + f)) for f in state.players}

    states = [state.to_dict()["hexes"]]  # initial state before any turn runs

    def on_step(s):
        states.append(s.to_dict()["hexes"])

    turn_count = 0
    while not check_game_end(state, max_turns=MAX_TURNS):
        state = run_turn(state, agents, rng=rng, on_step=on_step)
        turn_count += 1

    json_dict = {"radius": RADIUS, "num_factions": NUM_FACTIONS, "states": states}
    with open(OUTPUT_FILE, "w") as f:
        json.dump(json_dict, f, indent=2)
    print(f"Ran {turn_count} turns, logged {len(states)} states to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()