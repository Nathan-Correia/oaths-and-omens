"""
Demo: a randomly-initialized JAX policy network plays engine_v2 via
self-play (one shared network/params controls every faction), driven
through engine_v2.turn.run_turn exactly like any other agent would be.
No training - this just proves the network <-> engine wiring (masking,
action decoding, per-faction perspective encoding) actually works end
to end.

Run with: python run_nn_v2_demo.py
"""

import random

import jax

from engine.setup import create_initial_state
from engine_v2.state import from_v1
from engine_v2.turn import check_game_end, run_turn, tally_final_score

from nn_agent.agent import make_nn_agents
from nn_agent.network import init_params

RADIUS = 5
NUM_FACTIONS = 6
MAX_TURNS = 30
SEED = 1


def main():
    v1_state = create_initial_state(radius=RADIUS, num_factions=NUM_FACTIONS, seed=SEED)
    state = from_v1(v1_state, NUM_FACTIONS)

    rng_key = jax.random.PRNGKey(SEED)
    network, params = init_params(rng_key, state.num_hexes, NUM_FACTIONS)

    decide_buy, decide_movement, decide_cavalry, decide_target, decide_rectification = make_nn_agents(
        network, params, NUM_FACTIONS, seed=SEED, max_turns=MAX_TURNS,
    )

    rng = random.Random(SEED)
    while not check_game_end(state, max_turns=MAX_TURNS):
        run_turn(state, decide_buy, decide_movement, decide_cavalry, decide_target, decide_rectification, rng=rng)
        alive = int(state.alive.sum())
        total_units = int(state.army_units.sum()) + int(state.battle_units.sum())
        print(f"turn {state.turn_number:3d}  alive={alive}  units_on_board_or_in_battle={total_units}")

    print()
    print("final scores:", tally_final_score(state))


if __name__ == "__main__":
    main()
