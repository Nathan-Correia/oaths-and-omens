"""
Evaluation: plays the current network against greedy_agent and reports a
win rate - the actual "is this working" signal, since raw self-play
reward can look fine even while the network just gets better at
exploiting itself rather than genuinely improving (see the training-
strategy discussion this package came out of).

No recorder, no gradients - reuses run_turn/run_city_setup/check_game_end
directly, the same shape as run.py's driver loop.
"""

import random

from agents import compose_agents
from agents.greedy_agent import make_greedy_agents
from agents.nn_agent.agent import make_nn_agents
from engine.placement import run_city_setup
from engine.setup import create_initial_state
from engine.turn import check_game_end, get_game_winner, run_turn, tally_final_score


def run_evaluation(network, num_games, num_factions, radius, max_turns, base_seed, device):
    """Rotates which seat the trained network occupies across games so
    the result isn't biased by a fixed board position. Returns
    {"win_rate": float, "num_games": int}."""
    wins = 0
    for game_index in range(num_games):
        nn_faction = game_index % num_factions
        seed = base_seed * 1_000_003 + game_index
        seat_assignment = {f: ("nn" if f == nn_faction else "greedy") for f in range(num_factions)}
        build_fns = {
            "nn": lambda: make_nn_agents(network, num_factions, seed=seed, max_turns=max_turns, device=device),
            "greedy": lambda: make_greedy_agents(num_factions, seed=seed),
        }
        agents = compose_agents(seat_assignment, build_fns)
        rng = random.Random(seed)
        state = create_initial_state(radius=radius, num_factions=num_factions, seed=seed)
        state = run_city_setup(state, agents[5], agents[6], agents[7], rng)

        while not check_game_end(state, max_turns=max_turns):
            state = run_turn(state, *agents[:5], rng=rng)

        winner = get_game_winner(state)
        if winner is None:  # truncated without a VP win - fall back to the highest-VP faction
            scores = tally_final_score(state)
            winner = max(scores, key=lambda f: scores[f])
        if winner == nn_faction:
            wins += 1

    return {"win_rate": wins / num_games, "num_games": num_games}
