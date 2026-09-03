"""
Headless tournament harness for engine agents.

Unlike run.py, this never builds replay/visualization logs (per-checkpoint
board snapshots, battle event logs) - that bookkeeping is run.py's
dominant cost, and tournament-scale comparisons only care about the final
winner/VP, not watching the game play out. play_game here just drives
engine.turn.run_turn directly, turn after turn, with no logging.

AGENT_BUILDERS: {agent_key: make_X_agents} - add an entry here for every
new agents/*.py module (each already exposes a make_X_agents(num_factions,
seed) -> 9-tuple, the same shape run.py's build_fns use).

Two levels of comparison:
  - run_matchup: challenger vs a single baseline agent, filling every
    other seat. The challenger's seat is rotated across every faction
    index as `num_games` increases, canceling out the fact that
    placement/draft order (and therefore which capital you land, and
    capital_settle_order's tie-break) isn't perfectly symmetric across
    seats. Reports win rate, average final VP, and average rank.
  - run_free_for_all: one seat per distinct agent kind in `agent_keys`
    (cycling through the list if num_factions > len(agent_keys)), rotated
    across seats the same way, for genuine many-strategy melees rather
    than 1-vs-N-clones.

Both take a `sizes` list of (radius, num_factions) pairs and aggregate
per size - a strategy that only wins on the default 169-hex/8-faction
board shouldn't get mistaken for a generally good one (see SIZE_SWEEP).
"""

import random
import time
from collections import defaultdict

from agents import compose_agents
from agents.random_agent import make_random_agents
from agents.greedy_agent import make_greedy_agents
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
from engine.turn import check_game_end, get_game_winner, run_turn

AGENT_BUILDERS = {
    "random": make_random_agents,
    "greedy": make_greedy_agents,
    "heuristic": make_heuristic_agents,
    "turtle": make_turtle_agents,
    "denier": make_denier_agents,
    "vanguard": make_vanguard_agents,
    "warlord": make_warlord_agents,
    "legion": make_legion_agents,
    "hussar": make_hussar_agents,
    "sentinel": make_sentinel_agents,
    "marshal": make_marshal_agents,
    "tactician": make_tactician_agents,
}

DEFAULT_SIZE = (7, 8)  # radius, num_factions - the actual game's default
SIZE_SWEEP = [
    (7, 8),  # default
    (5, 6),
    (4, 4),
    (8, 8),  # bigger board, same faction count - radius 9+ hits a real
             # engine bug: generate_terrain's BAG_COUNTS totals 250 hexes,
             # less than a radius-9 board's 271, so terrain generation
             # crashes (rng.choices on an empty type list) past radius 8.
]
MAX_TURNS = 200


def play_game(assignment, radius, num_factions, seed, max_turns=MAX_TURNS):
    """assignment: {faction: agent_key}. Returns {"winner", "turns",
    "vp": {faction: vp}}. `winner` is None only if max_turns was hit
    before anyone reached VP_TO_WIN."""
    rng = random.Random(seed)
    state = create_initial_state(radius=radius, num_factions=num_factions, seed=seed, terrain_log=None)

    keys = set(assignment.values())
    build_fns = {k: (lambda k=k: AGENT_BUILDERS[k](num_factions, seed=seed)) for k in keys}
    decide = compose_agents(assignment, build_fns)
    (decide_buy, decide_movement, decide_cavalry, decide_target, decide_rectification, decide_resource_choice,
     decide_placement, decide_draft, decide_swap) = decide

    state = run_city_setup(state, decide_placement, decide_draft, decide_swap, rng)

    turns = 0
    while not check_game_end(state, max_turns=max_turns):
        state = run_turn(state, decide_buy, decide_movement, decide_cavalry, decide_target,
                          decide_rectification, decide_resource_choice, rng=rng)
        turns += 1

    return {
        "winner": get_game_winner(state),
        "turns": turns,
        "vp": {f: int(state.victory_points[f]) for f in range(num_factions)},
    }


def _rank_of(vp_by_faction, faction):
    v = vp_by_faction[faction]
    return 1 + sum(1 for f, other in vp_by_faction.items() if other > v)


def run_matchup(challenger, baseline, num_games, radius=None, num_factions=None, base_seed=0, max_turns=MAX_TURNS):
    """challenger occupies one seat (rotated across every faction index in
    turn), baseline fills every other seat."""
    radius = DEFAULT_SIZE[0] if radius is None else radius
    num_factions = DEFAULT_SIZE[1] if num_factions is None else num_factions

    wins = vp_sum = rank_sum = turns_sum = no_winner = 0
    for g in range(num_games):
        seed = base_seed + g
        seat = g % num_factions
        assignment = {f: baseline for f in range(num_factions)}
        assignment[seat] = challenger
        result = play_game(assignment, radius, num_factions, seed, max_turns)
        vp_sum += result["vp"][seat]
        rank_sum += _rank_of(result["vp"], seat)
        turns_sum += result["turns"]
        if result["winner"] == seat:
            wins += 1
        if result["winner"] is None:
            no_winner += 1

    return {
        "challenger": challenger, "baseline": baseline,
        "games": num_games, "radius": radius, "num_factions": num_factions,
        "win_rate": wins / num_games,
        "avg_vp": vp_sum / num_games,
        "avg_rank": rank_sum / num_games,
        "avg_turns": turns_sum / num_games,
        "no_winner": no_winner,
    }


def run_matchup_sweep(challenger, baseline, games_per_size, sizes=None, base_seed=0, max_turns=MAX_TURNS):
    sizes = sizes or SIZE_SWEEP
    return [
        run_matchup(challenger, baseline, games_per_size, radius, num_factions, base_seed, max_turns)
        for radius, num_factions in sizes
    ]


def run_free_for_all(agent_keys, num_games, radius=None, num_factions=None, base_seed=0, max_turns=MAX_TURNS):
    """One seat per distinct kind in agent_keys (cycling through the list
    if num_factions > len(agent_keys)); seat assignment rotates by game so
    each kind samples every seat roughly evenly. Returns {agent_key:
    stats} aggregated across every seat that key held, plus "_avg_turns"."""
    radius = DEFAULT_SIZE[0] if radius is None else radius
    num_factions = DEFAULT_SIZE[1] if num_factions is None else num_factions

    per_agent = defaultdict(lambda: {"seats": 0, "wins": 0, "vp_sum": 0, "rank_sum": 0})
    turns_sum = 0
    for g in range(num_games):
        rotation = g % len(agent_keys)
        assignment = {f: agent_keys[(f + rotation) % len(agent_keys)] for f in range(num_factions)}
        seed = base_seed + g
        result = play_game(assignment, radius, num_factions, seed, max_turns)
        turns_sum += result["turns"]
        for f, key in assignment.items():
            stat = per_agent[key]
            stat["seats"] += 1
            stat["vp_sum"] += result["vp"][f]
            stat["rank_sum"] += _rank_of(result["vp"], f)
            if result["winner"] == f:
                stat["wins"] += 1

    out = {
        key: {
            "seats": s["seats"],
            "win_rate": s["wins"] / s["seats"],
            "avg_vp": s["vp_sum"] / s["seats"],
            "avg_rank": s["rank_sum"] / s["seats"],
        }
        for key, s in per_agent.items()
    }
    out["_avg_turns"] = turns_sum / num_games
    return out


def format_matchup(stats):
    return (f"{stats['challenger']:>14} vs {stats['baseline']:<10} "
            f"[r={stats['radius']:>2},f={stats['num_factions']}] "
            f"win_rate={stats['win_rate']:6.1%} avg_vp={stats['avg_vp']:5.1f} "
            f"avg_rank={stats['avg_rank']:.2f} avg_turns={stats['avg_turns']:5.1f}"
            + (f" no_winner={stats['no_winner']}" if stats["no_winner"] else ""))


if __name__ == "__main__":
    t0 = time.time()
    stats = run_matchup("greedy", "random", 100)
    print(format_matchup(stats), f"  ({time.time() - t0:.1f}s)")
