"""
Self-play rollout collection: plays num_games full games with the given
network wired into a RolloutBuffer via agents/nn_agent/agent.py's
`recorder` param, and returns the filled buffer for training/gae.py to
consume.

Builds the instrumented nn agent set ONCE per collect() call (one
weight-sync), reused across every game in that call - agent.py's
torch.Generator is a long-lived, continuously-advancing object; calling
make_nn_agents fresh per game would reseed it identically every time and
produce the exact same trajectory every game, killing exploration
diversity entirely.

Reward is Δ(own VP) - Δ(leading rival's VP), computed once per turn from
state.victory_points and attributed to the most recent decision that
faction made this turn (see buffer.RolloutBuffer.mark_end_of_turn) -
GAE/the value function handle backward credit propagation to the turn's
earlier decisions, not this module. Only "nn"-assigned seats get reward
computed/attributed at all - a "greedy"/"random" seat's closures never
call record_step, so there's nothing in the buffer to attribute to.
"""

import random

import torch

from agents import compose_agents
from agents.greedy_agent import make_greedy_agents
from agents.nn_agent.agent import make_nn_agents
from agents.nn_agent.encode import encode_observation
from agents.random_agent import make_random_agents
from engine.placement import run_city_setup
from engine.setup import create_initial_state
from engine.turn import get_game_winner, run_turn

from .buffer import RolloutBuffer


def _leading_rival_vp(vp_array, faction):
    others = [int(vp_array[f]) for f in range(len(vp_array)) if f != faction]
    return max(others) if others else 0


def _estimate_bootstrap_values(network, state, factions, max_turns, device, neighbor_table_t):
    """Value-only forward pass at a max_turns truncation cutoff - no
    sampling, no recording, just the value head's read of the final
    state, one call per nn-controlled faction that needs a bootstrap
    (see gae.py for why a truncation bootstraps from this instead of
    being treated as a true terminal). network(...) always expects a
    batch dimension (see network.py) - wrapped as a batch of 1 here,
    same as agent.py's forward()."""
    values = {}
    with torch.inference_mode():
        for faction in factions:
            per_hex, global_feats = encode_observation(state, faction, max_turns=max_turns)
            out = network(
                torch.from_numpy(per_hex).to(device).unsqueeze(0),
                torch.from_numpy(global_feats).to(device).unsqueeze(0),
                neighbor_table_t,
                torch.zeros(1, dtype=torch.long, device=device),
            )
            values[faction] = float(out["value"].squeeze(0).item())
    return values


def collect(network, num_factions, radius, max_turns, num_games, base_seed, device,
            neighbor_table_t, seat_assignment=None, verbose=False):
    """Plays num_games full games. seat_assignment: {faction: "nn"|
    "greedy"|"random"}, defaulting to pure self-play (every seat "nn").
    Returns a filled RolloutBuffer. `verbose`: print one line per
    completed game (turns played, decisions recorded, outcome) - this is
    usually the slowest, least-visible part of a training iteration
    (unbatched, one game at a time - see module docstring), so a caller
    like train.py wants some sign of life during it rather than silence
    until the whole thing finishes."""
    seat_assignment = seat_assignment or {f: "nn" for f in range(num_factions)}
    nn_factions = {f for f, kind in seat_assignment.items() if kind == "nn"}

    buf = RolloutBuffer()

    build_fns = {
        "nn": lambda: make_nn_agents(network, num_factions, seed=base_seed, max_turns=max_turns,
                                      device=device, recorder=buf),
        "greedy": lambda: make_greedy_agents(num_factions, seed=base_seed),
        "random": lambda: make_random_agents(num_factions, seed=base_seed),
    }
    (decide_buy, decide_movement, decide_cavalry, decide_target, decide_rectification,
     decide_placement, decide_draft, decide_swap) = compose_agents(seat_assignment, build_fns)

    for game_index in range(num_games):
        game_seed = base_seed * 1_000_003 + game_index
        turn_rng = random.Random(game_seed)
        state = create_initial_state(radius=radius, num_factions=num_factions, seed=game_seed)

        game_id = buf.begin_game()
        state = run_city_setup(state, decide_placement, decide_draft, decide_swap, turn_rng)

        while True:
            vp_before = state.victory_points.copy()  # must copy - run_turn mutates in place
            state = run_turn(state, decide_buy, decide_movement, decide_cavalry,
                              decide_target, decide_rectification, rng=turn_rng)

            for faction in nn_factions:
                reward = float(
                    (int(state.victory_points[faction]) - int(vp_before[faction]))
                    - (_leading_rival_vp(state.victory_points, faction) - _leading_rival_vp(vp_before, faction))
                )
                buf.mark_end_of_turn(faction, reward)

            winner = get_game_winner(state)
            if winner is not None:
                buf.end_game({f: None for f in nn_factions})  # true terminal - no bootstrap
                outcome = f"faction {winner} won"
                break
            if state.turn_number >= max_turns:
                bootstrap = _estimate_bootstrap_values(network, state, nn_factions, max_turns,
                                                        device, neighbor_table_t)
                buf.end_game(bootstrap)  # truncation - bootstrap from the value head
                outcome = "truncated (turn cap)"
                break

        if verbose:
            print(f"  game {game_index + 1}/{num_games}: {state.turn_number} turns, "
                  f"{buf.steps_in_game(game_id)} decisions, {outcome}")

    return buf
