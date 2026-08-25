"""
Runs a game end-to-end, writing board_state.json in the checkpoint-log
format engine/turn.py's run_turn_and_log produces: terrain is stored
once (it never changes), and each turn is stored as a sparse snapshot
at each of its 8 checkpoints (start, buy, 3 movement steps, 2 cavalry
steps, battle) - see that module's docstring for why these are
independent snapshots rather than incremental diffs.

Agent mix is configurable per faction via AGENT_ASSIGNMENT below - any
mix of "random"/"greedy"/"nn" (a randomly-initialized, untrained PyTorch
policy network; see agents/nn_agent/). Faction id -> color
is fixed (see hex_visualizer.py's FACTION_COLORS: 0=red, 1=blue,
2=green, 3=purple, 4=orange, 5=brown, 6=pink, 7=grey).

Before any turns run, engine/placement.py's run_city_setup plays out
colourless city placement and the capital draft - every agent kind,
including "nn", implements all eight decision points (the five
turn-phase ones plus placement/draft/swap), so no per-faction fallback
wiring is needed here.

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

Also writes city_placement_log.json - every individual placement/draft
step made while running engine/placement.py:run_city_setup, in order,
for city_placement_visualizer.py to step through. File shape:
{"radius": int, "num_factions": int, "terrain": {"q_r_s": terrain_str, ...},
 "steps": [{"type","faction","q","r","s", ...}, ...]}  # see run_city_setup's docstring
"""

import json
import random
import time

from agents import compose_agents
from agents.greedy_agent import make_greedy_agents
from agents.random_agent import make_random_agents
from engine.placement import run_city_setup
from engine.setup import create_initial_state
from engine.state import TERRAIN_TYPES
from engine.turn import run_turn_and_log, check_game_end

OUTPUT_FILE = "board_state.json"
TERRAIN_LOG_FILE = "terrain_gen_log.json"
PLACEMENT_LOG_FILE = "city_placement_log.json"
RADIUS = 7
NUM_FACTIONS = 8
MAX_TURNS = 100

# Seeded from the system clock so every run produces a different game.
# Printed at the end so a specific run can still be reproduced later by
# hardcoding this value back in, if that's ever useful for debugging.
SEED = int(time.time() * 1000) % (2 ** 31)

# Per-faction agent choice - any of "random", "greedy", "nn".
AGENT_ASSIGNMENT = {
    0: "nn",
    1: "nn",
    2: "nn",
    3: "nn",
    4: "greedy",
    5: "greedy",
    6: "greedy",
    7: "greedy",
}

# Path to a training/checkpoint.py checkpoint (e.g. "checkpoints/iter_190.pt")
# to load "nn" faction(s) from - the trained weights actually play, instead
# of a fresh random-init network. None (the default) plays random-init, same
# as before checkpoints existed. The checkpoint's own saved num_factions/
# hidden_dim/num_mp_rounds are used to rebuild the network (see
# training/checkpoint.py's load_checkpoint) - NUM_FACTIONS above must match
# what it was trained with, or loading its state_dict will fail with a shape
# mismatch; RADIUS doesn't have to match (the network is board-size-agnostic
# - see network.py - but matching whatever it actually trained on is the
# fairest/most representative game to watch).
NN_CHECKPOINT = "checkpoints/iter_190.pt"


def _build_nn_agents():
    """Lazily pulls in torch/agents.nn_agent - only paid for if
    AGENT_ASSIGNMENT actually uses "nn" for at least one faction."""
    from agents.nn_agent.agent import make_nn_agents

    if NN_CHECKPOINT is not None:
        from training.checkpoint import load_checkpoint
        network, payload = load_checkpoint(NN_CHECKPOINT)
        print(f"Loaded {NN_CHECKPOINT} (trained for {payload.get('iteration', '?')} iterations)")
    else:
        from agents.nn_agent.network import build_network
        network = build_network(NUM_FACTIONS, seed=SEED)

    return make_nn_agents(network, NUM_FACTIONS, seed=SEED, max_turns=MAX_TURNS)


def main():
    rng = random.Random(SEED)
    terrain_log = []
    state = create_initial_state(radius=RADIUS, num_factions=NUM_FACTIONS, seed=SEED, terrain_log=terrain_log)

    with open(TERRAIN_LOG_FILE, "w") as f:
        json.dump({"radius": RADIUS, "steps": terrain_log}, f)

    terrain_map = {
        f"{q}_{r}_{s}": TERRAIN_TYPES[int(t)]
        for (q, r, s), t in zip(state.grid.coords, state.terrain)
    }

    build_fns = {
        "random": lambda: make_random_agents(NUM_FACTIONS, seed=SEED),
        "greedy": lambda: make_greedy_agents(NUM_FACTIONS, seed=SEED),
        "nn": lambda: _build_nn_agents(),
    }
    (decide_buy, decide_movement, decide_cavalry, decide_target, decide_rectification,
     decide_placement, decide_draft, decide_swap) = compose_agents(AGENT_ASSIGNMENT, build_fns)

    placement_log = []
    state = run_city_setup(state, decide_placement, decide_draft, decide_swap, rng, log=placement_log)

    with open(PLACEMENT_LOG_FILE, "w") as f:
        json.dump({
            "radius": RADIUS, "num_factions": NUM_FACTIONS,
            "terrain": terrain_map, "steps": placement_log,
        }, f)

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

    print(f"Ran {len(turns)} turns (seed={SEED}), wrote {OUTPUT_FILE}, {TERRAIN_LOG_FILE}, and {PLACEMENT_LOG_FILE}")


if __name__ == "__main__":
    main()
