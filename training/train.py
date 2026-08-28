"""
Top-level PPO self-play training loop: collect self-play games with the
current network -> compute GAE advantages -> PPO update -> periodically
evaluate against greedy_agent and checkpoint -> repeat.

Hyperparameters are module-level constants (matching run.py's convention
elsewhere in this repo - no CLI-args framework). RADIUS/GAMES_PER_ITERATION
default small: a full-size board (radius 7-8, 8 factions) collects a LOT
of per-decision data per iteration (per_hex is a float32[num_hexes,
PER_HEX_FEATURES] array *per recorded decision*, and a game has roughly
9 recorded decisions per faction per turn) - the buffer is fully
discarded after every update (no cross-iteration replay, pure on-policy),
but a single iteration's live memory footprint still scales with
RADIUS^2 * GAMES_PER_ITERATION * turns-per-game. Scale these up once the
loop is confirmed working.

Run with: python -m training.train

Rollout collection runs across NUM_WORKERS CPU worker processes (see
training/rollout.py's collect_parallel) rather than on the main process's
GPU: a batch-of-1 forward pass (one decision at a time, which is all a
single game ever needs) pays CUDA's per-call dispatch overhead without
enough compute to amortize it, so collection is CPU-bound work spread
across cores, while ppo.update()'s already-batched minibatch forward
passes are the part that actually benefits from the GPU. The worker pool
is created once, here, and kept alive for the whole run - Windows'
mandatory "spawn" start method re-imports torch in every worker process,
which costs real time (seconds), so paying it once up front rather than
once per iteration matters.
"""

import itertools
import multiprocessing
import os

import numpy as np
import torch

from agents.nn_agent.network import build_network
from engine.geometry import HexGrid

from . import checkpoint, evaluate, gae, ppo, rollout

RADIUS = 7
NUM_FACTIONS = 8
MAX_TURNS = 100
SEED = 42

HIDDEN_DIM = 128
NUM_MP_ROUNDS = 2

NUM_WORKERS = 6  # 5600x: 6 physical cores - leaves the main process free
# to run the GPU-bound PPO update without contending with collection
# workers for CPU time, and avoids hyperthreading's diminishing returns
# for this kind of workload (see training-loop discussion this came out of)

GAMES_PER_ITERATION = 24
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP_EPS = 0.2
PPO_EPOCHS = 4
MINIBATCH_SIZE = 256
LEARNING_RATE = 3e-4
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

NUM_ITERATIONS = 100
EVAL_EVERY_N_ITERATIONS = 10
EVAL_GAMES = 20
CHECKPOINT_EVERY_N_ITERATIONS = 10
CHECKPOINT_DIR = "checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    network = build_network(NUM_FACTIONS, hidden_dim=HIDDEN_DIM, num_mp_rounds=NUM_MP_ROUNDS,
                             seed=SEED, device=DEVICE)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

    grid = HexGrid(RADIUS)
    neighbor_table_t = torch.from_numpy(grid.neighbor_table.astype(np.int64)).to(DEVICE)

    # Created once and kept alive for the whole run - see module docstring
    # for why (Windows' spawn start method pays real per-process startup
    # cost, so a fresh pool every iteration would waste most of the win).
    pool = multiprocessing.Pool(NUM_WORKERS)
    try:
        for iteration in range(NUM_ITERATIONS):
            print(f"iter {iteration}: collecting {GAMES_PER_ITERATION} game(s) "
                  f"across {NUM_WORKERS} worker processes...")
            buffers = rollout.collect_parallel(
                pool, network, NUM_FACTIONS, RADIUS, MAX_TURNS, GAMES_PER_ITERATION,
                base_seed=SEED * 1_000_003 + iteration, num_workers=NUM_WORKERS, verbose=True,
            )
            samples = gae.compute_gae(
                itertools.chain.from_iterable(b.trajectories() for b in buffers),
                gamma=GAMMA, lam=GAE_LAMBDA,
            )
            print(f"iter {iteration}: running PPO update over {len(samples)} decisions...")
            ppo.update(
                network, optimizer, samples, neighbor_table_t, DEVICE,
                clip_eps=PPO_CLIP_EPS, epochs=PPO_EPOCHS, minibatch_size=MINIBATCH_SIZE,
                value_coef=VALUE_LOSS_COEF, entropy_coef=ENTROPY_COEF, max_grad_norm=MAX_GRAD_NORM,
                verbose=True,
            )
            num_samples = len(samples)
            del buffers, samples  # on-policy - nothing carries over to the next iteration

            print(f"iter {iteration}: collected {num_samples} decisions")

            if iteration % EVAL_EVERY_N_ITERATIONS == 0:
                result = evaluate.run_evaluation(
                    network, EVAL_GAMES, NUM_FACTIONS, RADIUS, MAX_TURNS, base_seed=SEED, device=DEVICE,
                )
                print(f"iter {iteration}: win_rate vs greedy_agent = {result['win_rate']:.3f} "
                      f"({result['num_games']} games)")

            if iteration % CHECKPOINT_EVERY_N_ITERATIONS == 0:
                path = os.path.join(CHECKPOINT_DIR, f"iter_{iteration}.pt")
                checkpoint.save_checkpoint(path, network, optimizer, extra={"iteration": iteration})
                print(f"iter {iteration}: saved checkpoint to {path}")

        # The loop's last iteration only lands on a checkpoint boundary above
        # if NUM_ITERATIONS - 1 happens to be a multiple of
        # CHECKPOINT_EVERY_N_ITERATIONS - otherwise its training (and every
        # iteration since the last periodic save) would silently never be
        # written to disk once this process exits. Always save the final
        # state unconditionally so a completed run never loses its last
        # stretch of progress.
        final_iteration = NUM_ITERATIONS - 1
        if final_iteration % CHECKPOINT_EVERY_N_ITERATIONS != 0:
            path = os.path.join(CHECKPOINT_DIR, f"iter_{final_iteration}.pt")
            checkpoint.save_checkpoint(path, network, optimizer, extra={"iteration": final_iteration})
            print(f"iter {final_iteration}: saved final checkpoint to {path}")
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
