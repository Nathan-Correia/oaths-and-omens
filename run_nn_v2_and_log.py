"""
Plays a full engine_v2 self-play game (one shared, randomly-initialized
JAX policy network controlling every faction, same as run_nn_v2_demo.py)
and writes board_state.json in the same diff-viewer-compatible shape
run.py writes for v1 - so hex_visualizer.py can load and replay it
unchanged. See engine_v2/turn.py's run_turn_and_log docstring for how
this log format differs from v1's (independent sparse snapshots per
checkpoint, not incremental diffs).

Kept deliberately small (radius/faction count/turn count) - the network
isn't jitted yet (see nn_agent/network.py's docstring), so each decision
is an eager JAX forward pass; a radius-8/8-faction/100-turn game like
run.py's would take a long time right now.

Run with: python run_nn_v2_and_log.py
"""

import json
import random

import jax

from engine_v2.setup import create_initial_state
from engine_v2.state import TERRAIN_TYPES
from engine_v2.turn import check_game_end, run_turn_and_log

from nn_agent.agent import make_nn_agents
from nn_agent.network import init_params

OUTPUT_FILE = "board_state.json"
RADIUS = 8
NUM_FACTIONS = 8
MAX_TURNS = 100
SEED = 1


def main():
    state = create_initial_state(radius=RADIUS, num_factions=NUM_FACTIONS, seed=SEED)

    terrain_map = {
        f"{q}_{r}_{s}": TERRAIN_TYPES[int(t)]
        for (q, r, s), t in zip(state.grid.coords, state.terrain)
    }

    rng_key = jax.random.PRNGKey(SEED)
    network, params = init_params(rng_key, state.num_hexes, NUM_FACTIONS)
    decide_buy, decide_movement, decide_cavalry, decide_target, decide_rectification = make_nn_agents(
        network, params, NUM_FACTIONS, seed=SEED, max_turns=MAX_TURNS,
    )

    rng = random.Random(SEED)

    turns = []
    while not check_game_end(state, max_turns=MAX_TURNS):
        state, turn_record = run_turn_and_log(
            state, decide_buy, decide_movement, decide_cavalry, decide_target, decide_rectification, rng=rng,
        )
        turns.append(turn_record)
        print(f"turn {state.turn_number:3d}  alive={int(state.alive.sum())}  "
              f"battles_this_turn={len(turn_record['battle_events'])}")

    json_dict = {
        "radius": RADIUS,
        "num_factions": NUM_FACTIONS,
        "terrain": terrain_map,
        "turns": turns,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(json_dict, f)

    print(f"\nRan {len(turns)} turns (seed={SEED}), wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
