"""
Profiles a full 100-turn, 8-faction game two ways:
  1. plain run_turn() - the training-shaped path, no logging at all
  2. run_turn_and_log() - the replay-logging path used for visualization

Prints the top 25 functions by time-spent for each, plus total wall
time for both, so it's easy to see both where the time goes and how
much the logging machinery itself costs.

Agent mix is configurable per faction via AGENT_ASSIGNMENT below, same
as run.py.

Run with: python profile_engine.py
"""

import cProfile
import pstats
import random
import time

from agents import compose_agents
from agents.greedy_agent import make_greedy_agents
from agents.random_agent import make_random_agents
from engine.placement import run_city_setup
from engine.setup import create_initial_state
from engine.turn import run_turn, run_turn_and_log, check_game_end

RADIUS = 8
NUM_FACTIONS = 8
MAX_TURNS = 100
SEED = 1

# Per-faction agent choice - any of "random", "greedy", "nn".
AGENT_ASSIGNMENT = {f: "greedy" for f in range(NUM_FACTIONS)}


def _build_nn_agents():
    from agents.nn_agent.agent import make_nn_agents
    from agents.nn_agent.network import build_network

    network = build_network(NUM_FACTIONS, seed=SEED)
    return make_nn_agents(network, NUM_FACTIONS, seed=SEED, max_turns=MAX_TURNS)


def make_game():
    state = create_initial_state(radius=RADIUS, num_factions=NUM_FACTIONS, seed=SEED)
    build_fns = {
        "random": lambda: make_random_agents(NUM_FACTIONS, seed=SEED),
        "greedy": lambda: make_greedy_agents(NUM_FACTIONS, seed=SEED),
        "nn": lambda: _build_nn_agents(),
    }
    agents = compose_agents(AGENT_ASSIGNMENT, build_fns)
    rng = random.Random(SEED + 1)
    state = run_city_setup(state, *agents[5:], rng)
    return state, agents[:5], rng


def run_plain():
    state, agents, rng = make_game()
    while not check_game_end(state, max_turns=MAX_TURNS):
        state = run_turn(state, *agents, rng=rng)
    return state


def run_logged():
    state, agents, rng = make_game()
    while not check_game_end(state, max_turns=MAX_TURNS):
        state, _record = run_turn_and_log(state, *agents, rng=rng)
    return state


def profile_and_report(fn, label):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")

    start = time.perf_counter()
    profiler = cProfile.Profile()
    profiler.enable()
    fn()
    profiler.disable()
    elapsed = time.perf_counter() - start

    print(f"Wall time: {elapsed:.3f}s\n")
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats(25)


if __name__ == "__main__":
    profile_and_report(run_plain, f"PLAIN run_turn (no logging) - {MAX_TURNS} turns, {NUM_FACTIONS} factions, radius {RADIUS}")
    profile_and_report(run_logged, f"run_turn_and_log (full replay logging) - {MAX_TURNS} turns, {NUM_FACTIONS} factions, radius {RADIUS}")