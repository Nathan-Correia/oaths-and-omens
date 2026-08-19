"""
Builds the initial GameState for a new game: random terrain, city
placement (reusing the same farthest-point-spread + 2-hex-away second
city logic from map_generator.py), and starting player resources per
the rulebook (50 silver, 2 kill-XP, no starting armies - units are
bought during the first buy phase).
"""

import random

from .geometry import cube_hexes_in_radius, hex_distance, CUBE_DIRECTIONS
from .state import GameState, HexState, PlayerState, TERRAIN_TYPES

STARTING_SILVER = 50
STARTING_KILL_XP = 2


def create_initial_state(radius=8, num_factions=8, seed=42):
    rng = random.Random(seed)
    all_coords = cube_hexes_in_radius(radius)

    board = {}
    for coord in all_coords:
        terrain = rng.choice(TERRAIN_TYPES)
        board[coord] = HexState(terrain=terrain)

    def ensure_passable(coord):
        if board[coord].terrain in ("mountain", "lake"):
            board[coord].terrain = rng.choice([t for t in TERRAIN_TYPES if t not in ("mountain", "lake")])

    home_coords = []
    first_home = rng.choice(all_coords)
    home_coords.append(first_home)
    for _ in range(num_factions - 1):
        best_coord = max(
            all_coords,
            key=lambda c: min(hex_distance(c, h) for h in home_coords) + rng.random() * 0.01,
        )
        home_coords.append(best_coord)

    players = {}
    for faction, home in enumerate(home_coords):
        ensure_passable(home)
        board[home].city_owner = faction

        directions = list(CUBE_DIRECTIONS)
        rng.shuffle(directions)
        second_city = None
        for dq, dr, ds in directions:
            cand = (home[0] + 2 * dq, home[1] + 2 * dr, home[2] + 2 * ds)
            if cand in board and board[cand].city_owner is None:
                second_city = cand
                break
        if second_city is not None:
            ensure_passable(second_city)
            board[second_city].city_owner = faction

        players[faction] = PlayerState(
            faction=faction,
            silver=STARTING_SILVER,
            kill_xp_bank=[{"unit_type": "infantry"} for _ in range(STARTING_KILL_XP)],
        )

    return GameState(board=board, players=players, battles={}, turn_number=0, radius=radius)
