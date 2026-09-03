"""
Runs a game end-to-end, writing board_state.json in the checkpoint-log
format engine/turn.py's run_turn_and_log produces: terrain is stored
once (it never changes), and each turn is stored as a sparse snapshot
at each of its 8 checkpoints (start, buy, 3 movement steps, 2 cavalry
steps, battle) - see that module's docstring for why these are
independent snapshots rather than incremental diffs.

NOTE: hex_visualizer.py (board_state.json's original viewer) is
deprecated - board_state.json is still written for parity/potential
other consumers, but its battle_events field is now always empty (see
run_turn_and_log's docstring for why: that field's only consumer was
hex_visualizer.py's battle animation).

Agent mix is configurable per faction via AGENT_ASSIGNMENT below - any
mix of "random", "greedy", "heuristic", "turtle", "denier", "vanguard",
"warlord", "legion", "hussar", "sentinel", "marshal", "tactician" (see
agents/*.py - tactician_agent's docstring in particular for the
strongest of these and why, though it costs meaningfully more compute
per game than the others, being a search agent). Faction id -> color is
fixed (see hex_visualizer.py's FACTION_COLORS: 0=red, 1=blue, 2=green,
3=purple, 4=orange, 5=brown, 6=pink, 7=grey). See tournament.py for
headless many-game comparisons between these agents instead of a single
logged game.

Before any turns run, engine/placement.py's run_city_setup plays out
colourless city placement and the capital draft - every agent kind
implements all eight decision points (the five turn-phase ones plus
placement/draft/swap), so no per-faction fallback wiring is needed here.

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

import torch

from agents import compose_agents
from agents.greedy_agent import make_greedy_agents
from agents.random_agent import make_random_agents
from agents.heuristic_agent import make_heuristic_agents
from agents.turtle_agent import make_turtle_agents
from agents.denier_agent import make_denier_agents
from agents.vanguard_agent import make_vanguard_agents
from agents.warlord_agent import make_warlord_agents
from agents.legion_agent import make_legion_agents
from agents.hussar_agent import make_hussar_agents
from agents.sentinel_agent import make_sentinel_agents
from agents.marshal_agent import make_marshal_agents
from agents.tactician_agent import make_tactician_agents
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

# Per-faction agent choice - any of "random", "greedy", "heuristic", "turtle", "denier",
# "vanguard", "warlord", "legion", "hussar", "sentinel", "marshal", "tactician".
AGENT_ASSIGNMENT = {
    0: "tactician",
    1: "tactician",
    2: "tactician",
    3: "tactician",
    4: "tactician",
    5: "tactician",
    6: "tactician",
    7: "tactician",
}


def main():
    setup_rng = random.Random(SEED)
    gen = torch.Generator()
    gen.manual_seed(SEED)

    terrain_log = []
    state = create_initial_state(radius=RADIUS, num_factions=NUM_FACTIONS, seed=SEED, terrain_log=terrain_log)

    with open(TERRAIN_LOG_FILE, "w") as f:
        json.dump({"radius": RADIUS, "steps": terrain_log}, f)

    terrain_map = {
        f"{q}_{r}_{s}": TERRAIN_TYPES[int(t)]
        for (q, r, s), t in zip(state.grid.coords, state.terrain[0].tolist())
    }

    build_fns = {
        "random": lambda: make_random_agents(NUM_FACTIONS, seed=SEED),
        "greedy": lambda: make_greedy_agents(NUM_FACTIONS, seed=SEED),
        "heuristic": lambda: make_heuristic_agents(NUM_FACTIONS, seed=SEED),
        "turtle": lambda: make_turtle_agents(NUM_FACTIONS, seed=SEED),
        "denier": lambda: make_denier_agents(NUM_FACTIONS, seed=SEED),
        "vanguard": lambda: make_vanguard_agents(NUM_FACTIONS, seed=SEED),
        "warlord": lambda: make_warlord_agents(NUM_FACTIONS, seed=SEED),
        "legion": lambda: make_legion_agents(NUM_FACTIONS, seed=SEED),
        "hussar": lambda: make_hussar_agents(NUM_FACTIONS, seed=SEED),
        "sentinel": lambda: make_sentinel_agents(NUM_FACTIONS, seed=SEED),
        "marshal": lambda: make_marshal_agents(NUM_FACTIONS, seed=SEED),
        "tactician": lambda: make_tactician_agents(NUM_FACTIONS, seed=SEED),
    }
    (decide_buy, decide_movement, decide_cavalry, decide_target, decide_rectification, decide_resource_choice,
     decide_placement, decide_draft, decide_swap) = compose_agents(AGENT_ASSIGNMENT, build_fns)

    placement_log = []
    state = run_city_setup(state, decide_placement, decide_draft, decide_swap, setup_rng, log=placement_log)

    with open(PLACEMENT_LOG_FILE, "w") as f:
        json.dump({
            "radius": RADIUS, "num_factions": NUM_FACTIONS,
            "terrain": terrain_map, "steps": placement_log,
        }, f)

    decide_buy_list = [decide_buy]
    decide_movement_list = [decide_movement]
    decide_cavalry_list = [decide_cavalry]
    decide_target_list = [decide_target]
    decide_rectification_list = [decide_rectification]
    decide_resource_choice_list = [decide_resource_choice]

    turns = []
    while not bool(check_game_end(state, max_turns=MAX_TURNS)[0]):
        state, turn_record = run_turn_and_log(
            state, decide_buy_list, decide_movement_list, decide_cavalry_list, decide_target_list,
            decide_rectification_list, decide_resource_choice_list, gen,
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
