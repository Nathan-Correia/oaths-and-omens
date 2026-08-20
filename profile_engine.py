"""
Profiles a full 100-turn, 8-faction game two ways:
  1. plain run_turn() - the training-shaped path, no logging at all
  2. run_turn_and_log() - the replay-logging path used for visualization

Prints the top 25 functions by time-spent for each, plus total wall
time for both, so it's easy to see both where the time goes and how
much the logging machinery itself costs.

Run with: python profile_engine.py
"""

import cProfile
import pstats
import random
import time

from engine.setup import create_initial_state
from engine.turn import run_turn, run_turn_and_log, check_game_end
from engine.agents.heuristic_agent import HeuristicAgent

RADIUS = 8
NUM_FACTIONS = 8
MAX_TURNS = 100
SEED = 1


def make_game():
    state = create_initial_state(radius=RADIUS, num_factions=NUM_FACTIONS, seed=SEED)
    agents = {f: HeuristicAgent(f, rng=random.Random(100 + f)) for f in state.players}
    rng = random.Random(SEED + 1)
    return state, agents, rng


def run_plain():
    state, agents, rng = make_game()
    while not check_game_end(state, max_turns=MAX_TURNS):
        state = run_turn(state, agents, rng=rng)
    return state


def run_logged():
    state, agents, rng = make_game()
    while not check_game_end(state, max_turns=MAX_TURNS):
        state, _record = run_turn_and_log(state, agents, rng=rng)
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