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
"""

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

GAMES_PER_ITERATION = 16
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP_EPS = 0.2
PPO_EPOCHS = 4
MINIBATCH_SIZE = 256
LEARNING_RATE = 3e-4
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

NUM_ITERATIONS = 200
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

    for iteration in range(NUM_ITERATIONS):
        print(f"iter {iteration}: collecting {GAMES_PER_ITERATION} game(s)...")
        buf = rollout.collect(
            network, NUM_FACTIONS, RADIUS, MAX_TURNS, GAMES_PER_ITERATION,
            base_seed=SEED * 1_000_003 + iteration, device=DEVICE, neighbor_table_t=neighbor_table_t,
            verbose=True,
        )
        samples = gae.compute_gae(buf, gamma=GAMMA, lam=GAE_LAMBDA)
        print(f"iter {iteration}: running PPO update over {len(samples)} decisions...")
        ppo.update(
            network, optimizer, samples, neighbor_table_t, DEVICE,
            clip_eps=PPO_CLIP_EPS, epochs=PPO_EPOCHS, minibatch_size=MINIBATCH_SIZE,
            value_coef=VALUE_LOSS_COEF, entropy_coef=ENTROPY_COEF, max_grad_norm=MAX_GRAD_NORM,
            verbose=True,
        )
        num_samples = len(samples)
        del buf, samples  # on-policy - nothing carries over to the next iteration

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


if __name__ == "__main__":
    main()
