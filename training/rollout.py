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

collect_parallel() below fans collect() out across several OS processes
(see train.py's NUM_WORKERS) - a single-game collect() call is dominated
by thousands of tiny batch-of-1 network forward passes, and those are
actually *cheaper* per-call on CPU than CUDA (kernel-launch/dispatch
overhead dwarfs the compute for a network this small at batch size 1), so
workers run entirely on CPU and never touch the GPU at all - only
ppo.py's already-batched update phase benefits from CUDA. Each worker
rebuilds the network from a plain state_dict (never a live CUDA module)
and its own CPU neighbor_table, both cheap to reconstruct, so nothing
GPU-resident ever needs to cross the process boundary.
"""

import random

import numpy as np
import torch

from agents import compose_agents
from agents.greedy_agent import make_greedy_agents
from agents.nn_agent.agent import make_nn_agents
from agents.nn_agent.encode import encode_observation
from agents.nn_agent.network import build_network
from agents.random_agent import make_random_agents
from engine.geometry import HexGrid
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
            neighbor_table_t, seat_assignment=None, verbose=False, log_prefix=""):
    """Plays num_games full games. seat_assignment: {faction: "nn"|
    "greedy"|"random"}, defaulting to pure self-play (every seat "nn").
    Returns a filled RolloutBuffer. `verbose`: print one line per
    completed game (turns played, decisions recorded, outcome) - this is
    usually the slowest, least-visible part of a training iteration
    (one game at a time within a single call - see module docstring), so
    a caller like train.py wants some sign of life during it rather than
    silence until the whole thing finishes. `log_prefix`: prepended to
    each printed line - collect_parallel gives each worker a distinct
    prefix (e.g. "[w2] ") since several workers print concurrently and
    their lines can otherwise interleave without any way to tell them
    apart."""
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
            print(f"{log_prefix}game {game_index + 1}/{num_games}: {state.turn_number} turns, "
                  f"{buf.steps_in_game(game_id)} decisions, {outcome}")

    return buf


def _worker_collect(state_dict, num_factions, hidden_dim, num_mp_rounds, radius, max_turns,
                     num_games, base_seed, seat_assignment, verbose, worker_id):
    """Top-level (picklable) target for one collect_parallel worker
    process - must stay a plain module-level function, not a closure or
    lambda, since multiprocessing locates it by import path in the child
    process (Windows always uses "spawn", which has no fork-style shared
    memory - every argument here crosses the process boundary via
    pickling). Runs entirely on CPU: rebuilds the network fresh from
    `state_dict` rather than receiving the live (possibly CUDA-resident)
    module object, and rebuilds its own neighbor_table from `radius`
    rather than receiving a tensor - both are cheap and deterministic, so
    there's no reason to ship GPU-adjacent objects across the boundary at
    all. Returns one RolloutBuffer, covering just this worker's
    `num_games` games; collect_parallel is responsible for combining
    several workers' buffers back into one training batch."""
    network = build_network(num_factions, hidden_dim=hidden_dim, num_mp_rounds=num_mp_rounds, device="cpu")
    network.load_state_dict(state_dict)
    network.eval()

    grid = HexGrid(radius)
    neighbor_table_t = torch.from_numpy(grid.neighbor_table.astype(np.int64))

    return collect(network, num_factions, radius, max_turns, num_games, base_seed,
                    device="cpu", neighbor_table_t=neighbor_table_t,
                    seat_assignment=seat_assignment, verbose=verbose, log_prefix=f"[w{worker_id}] ")


def collect_parallel(pool, network, num_factions, radius, max_turns, num_games, base_seed,
                      num_workers, seat_assignment=None, verbose=False):
    """Splits num_games as evenly as possible across num_workers CPU
    worker processes (via the given, already-running multiprocessing.Pool
    - see train.py, which creates it once and keeps it alive for the
    whole run rather than paying Windows' spawn/torch-import startup cost
    every iteration) and collects them in parallel, entirely on CPU (see
    module docstring for why CPU, not CUDA, is the right device for
    collection specifically).

    `network`'s current weights are copied to CPU once per call and
    shipped to every worker as a plain state_dict - small for this
    network's size, no need for shared memory. Each worker gets a
    disjoint seed range (offset by a large per-worker constant) so games
    across workers never collide despite each worker's own game_index
    restarting at 0 internally.

    Returns a list[RolloutBuffer], one per worker that actually ran any
    games - callers pass itertools.chain.from_iterable(b.trajectories()
    for b in buffers) to gae.compute_gae rather than merging the buffers
    themselves (see gae.py's docstring for why trajectories() is the
    correct merge point, not the buffers' internal dicts)."""
    seat_assignment = seat_assignment or {f: "nn" for f in range(num_factions)}
    state_dict = {k: v.detach().cpu() for k, v in network.state_dict().items()}
    hidden_dim = network.hidden_dim
    num_mp_rounds = len(network.mp_rounds)

    counts = [num_games // num_workers] * num_workers
    for i in range(num_games % num_workers):
        counts[i] += 1

    args = [
        (state_dict, num_factions, hidden_dim, num_mp_rounds, radius, max_turns,
         count, base_seed + worker_id * 1_000_000_007, seat_assignment, verbose, worker_id)
        for worker_id, count in enumerate(counts) if count > 0
    ]

    return pool.starmap(_worker_collect, args)
