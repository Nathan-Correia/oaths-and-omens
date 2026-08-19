"""
Board state generator.

Generates a hexagon-shaped board with random terrain and a starting
layout for each faction (2 cities + a starting army), then writes a
sequence of 10 states to board_state.json in the same directory.

For now the sequence is a placeholder for exercising the visualizer's
timeline: state 0 is the initial board, state 1 removes all troops,
state 2 restores them, and so on alternating. Once the real game
engine exists, generate_state_sequence() is what gets replaced with
actual turn-by-turn simulation output.
"""

import json
import random

from hex_common import (
    RADIUS, NUM_FACTIONS, TERRAIN_TYPES, CUBE_DIRECTIONS,
    cube_hexes_in_radius, hex_neighbors, hex_distance,
)

OUTPUT_FILE = "board_state.json"


def generate_board(radius, num_factions=8, seed=42):
    """
    Returns a dict keyed by (q, r, s) cube-coord tuples:
      {
        "terrain": str,
        "city": None or faction_id (int),
        "troops": None or {"faction": int, "infantry": int, "cavalry": int, "archers": int}
      }

    Layout logic:
      - Random terrain everywhere, except city hexes are nudged to
        non-ocean so starting positions are always on land.
      - Home hex (1st city) placement: player 1 gets a random hex, then
        each following player's home is chosen to be far away from all
        previously-placed homes (farthest-point placement), so starting
        positions spread out across the board rather than clustering.
      - 2nd city: placed exactly 2 hexes from the home city, in a
        random direction (falls back to another direction if that spot
        is off the board or already taken).
      - Starting army (4 infantry, 2 cavalry, 2 archers): each unit
        *type* independently picks a random hex from "in or adjacent
        to" either of the faction's 2 cities - so the 3 unit types may
        end up stacked together on one hex, split across two, or
        spread across three, depending on the random draws.
    """
    rng = random.Random(seed)
    all_coords = cube_hexes_in_radius(radius)

    board = {}
    for coord in all_coords:
        terrain = rng.choice(TERRAIN_TYPES)
        board[coord] = {"terrain": terrain, "city": None, "troops": None}

    def ensure_land(coord):
        if board[coord]["terrain"] == "ocean":
            board[coord]["terrain"] = rng.choice([t for t in TERRAIN_TYPES if t != "ocean"])

    # --- Place home hexes: player 1 random, each subsequent player as
    # far as possible from all already-placed homes.
    home_coords = []
    first_home = rng.choice(all_coords)
    home_coords.append(first_home)

    for _ in range(num_factions - 1):
        best_coord = max(
            all_coords,
            key=lambda c: min(hex_distance(c, h) for h in home_coords)
                          + rng.random() * 0.01,  # tiny jitter to break ties randomly
        )
        home_coords.append(best_coord)

    for faction_id, home in enumerate(home_coords):
        ensure_land(home)
        board[home]["city"] = faction_id

        # 2nd city: exactly 2 hexes away, random direction, with fallback
        # to another direction if that spot is invalid/taken.
        directions = list(CUBE_DIRECTIONS)
        rng.shuffle(directions)
        second_city = None
        for dq, dr, ds in directions:
            cand = (home[0] + 2 * dq, home[1] + 2 * dr, home[2] + 2 * ds)
            if cand in board and board[cand]["city"] is None:
                second_city = cand
                break
        if second_city is not None:
            ensure_land(second_city)
            board[second_city]["city"] = faction_id

        # --- Starting army: each unit type independently picks a random
        # hex from "in or adjacent to" either city.
        city_hexes = [home] + ([second_city] if second_city is not None else [])
        candidate_hexes = set(city_hexes)
        for c in city_hexes:
            candidate_hexes.update(hex_neighbors(c, radius))
        candidate_hexes = list(candidate_hexes)

        unit_counts = {"infantry": 4, "cavalry": 2, "archers": 2}
        for unit_type, count in unit_counts.items():
            target = rng.choice(candidate_hexes)
            if board[target]["troops"] is None:
                board[target]["troops"] = {
                    "faction": faction_id,
                    "infantry": 0,
                    "cavalry": 0,
                    "archers": 0,
                }
            board[target]["troops"][unit_type] += count

    return board


def board_to_hexes_list(board):
    """Flatten a (q,r,s)-keyed board dict into a list of hex dicts."""
    hexes = []
    for (q, r, s), data in board.items():
        hexes.append({
            "q": q, "r": r, "s": s,
            "terrain": data["terrain"],
            "city": data["city"],
            "troops": data["troops"],
        })
    return hexes


def make_troops_removed_copy(board):
    """Same terrain/cities, but every hex's troops cleared."""
    copy = {}
    for coord, data in board.items():
        copy[coord] = {"terrain": data["terrain"], "city": data["city"], "troops": None}
    return copy


def generate_state_sequence(radius, num_factions, seed=42, num_states=10):
    """
    Builds a simple multi-state sequence for exercising the timeline:
      state 0: initial board (terrain + cities + starting armies)
      state 1: same board, all troops removed
      state 2: troops back (same as state 0)
      ... alternating, for `num_states` states total.

    Terrain and city ownership never change across states in this demo -
    only troop presence toggles. Once the real game engine exists, this
    function is what gets replaced with actual turn-by-turn simulation
    output.
    """
    base_board = generate_board(radius, num_factions, seed=seed)
    troopless_board = make_troops_removed_copy(base_board)

    states = []
    for i in range(num_states):
        board = base_board if i % 2 == 0 else troopless_board
        states.append(board_to_hexes_list(board))
    return states


def main():
    states = generate_state_sequence(RADIUS, NUM_FACTIONS, seed=42, num_states=10)
    json_dict = {"radius": RADIUS, "num_factions": NUM_FACTIONS, "states": states}
    with open(OUTPUT_FILE, "w") as f:
        json.dump(json_dict, f, indent=2)
    print(f"Wrote {len(states)} states ({len(states[0])} hexes each) to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()