"""
Runs a game end-to-end, writing board_state.json in the checkpoint-log
format engine/turn.py's run_turn_and_log produces: terrain is stored
once (it never changes), and each turn is stored as a sparse snapshot
at each of its 10 checkpoints (start, buy, 3 movement steps, 4 cavalry
steps, battle) - see that module's docstring for why these are
independent snapshots rather than incremental diffs.

Agent mix is configurable per faction via AGENT_ASSIGNMENT below - any
mix of "random"/"smart_random"/"heuristic"/"nn" (a randomly-initialized,
untrained JAX policy network; see agents/nn_agent/). Faction id -> color
is fixed (see hex_visualizer.py's FACTION_COLORS: 0=red, 1=blue,
2=green, 3=purple, 4=orange, 5=brown, 6=pink, 7=grey).

File shape:
{
  "radius": int, "num_factions": int,
  "terrain": {"q_r_s": terrain_str, ...},
  "turns": [turn_record, ...]   # see engine/turn.py:run_turn_and_log
}

Also writes terrain_gen_log.json - every individual hex placement made
while generating the board's terrain, in order (see
engine/setup.py:generate_terrain), for hex_gen.py to step through.
File shape: {"radius": int, "steps": [{"q","r","s","terrain","round"}, ...]}
"""

import json
import random
import time

from agents import compose_agents
from agents.heuristic_agent import make_heuristic_agents
from agents.random_agent import make_random_agents
from agents.smart_random_agent import make_smart_random_agents
from engine.setup import create_initial_state
from engine.state import TERRAIN_TYPES
from engine.turn import run_turn_and_log, check_game_end

OUTPUT_FILE = "board_state.json"
TERRAIN_LOG_FILE = "terrain_gen_log.json"
RADIUS = 7
NUM_FACTIONS = 8
MAX_TURNS = 100

# Seeded from the system clock so every run produces a different game.
# Printed at the end so a specific run can still be reproduced later by
# hardcoding this value back in, if that's ever useful for debugging.
SEED = int(time.time() * 1000) % (2 ** 31)

# Per-faction agent choice - any of "random", "smart_random", "heuristic", "nn".
AGENT_ASSIGNMENT = {f: "random" for f in range(NUM_FACTIONS)}


def _build_nn_agents(state):
    """Lazily pulls in jax/agents.nn_agent - only paid for if AGENT_ASSIGNMENT
    actually uses "nn" for at least one faction."""
    import jax

    from agents.nn_agent.agent import make_nn_agents
    from agents.nn_agent.network import init_params

    rng_key = jax.random.PRNGKey(SEED)
    network, params = init_params(rng_key, state.num_hexes, NUM_FACTIONS)
    return make_nn_agents(network, params, NUM_FACTIONS, seed=SEED, max_turns=MAX_TURNS)


def main():
    rng = random.Random(SEED)
    terrain_log = []
    state = create_initial_state(radius=RADIUS, num_factions=NUM_FACTIONS, seed=SEED, terrain_log=terrain_log)

    with open(TERRAIN_LOG_FILE, "w") as f:
        json.dump({"radius": RADIUS, "steps": terrain_log}, f)

    build_fns = {
        "random": lambda: make_random_agents(NUM_FACTIONS, seed=SEED),
        "smart_random": lambda: make_smart_random_agents(NUM_FACTIONS, seed=SEED),
        "heuristic": lambda: make_heuristic_agents(NUM_FACTIONS, seed=SEED),
        "nn": lambda: _build_nn_agents(state),
    }
    decide_buy, decide_movement, decide_cavalry, decide_target, decide_rectification = compose_agents(
        AGENT_ASSIGNMENT, build_fns,
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

    print(f"Ran {len(turns)} turns (seed={SEED}), wrote {OUTPUT_FILE} and {TERRAIN_LOG_FILE}")


if __name__ == "__main__":
    main()
