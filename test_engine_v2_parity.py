"""
Parity tests for engine_v2 against engine/ (v1) - the "reference oracle"
approach agreed for building the GPU-batchable rewrite: engine_v2 doesn't
need its own from-scratch test suite, it needs to be checked against the
known-good v1 engine on matching inputs.

Covers geometry indexing, v1<->v2 state conversion, movement/cavalry
legality masks, and apply_movement_step - the slice of engine_v2 built so
far (see engine_v2/ module docstrings for what's deliberately deferred:
battle resolution, buy, income, terrain effects).

Run with: python test_engine_v2_parity.py
"""

import random
from collections import defaultdict

import numpy as np

from engine.agents.base_agent import BaseAgent as BaseAgentV1
from engine.battle import (
    get_winner as get_winner_v1,
    rectify_overflow as rectify_overflow_v1,
    resolve_full_battle as resolve_full_battle_v1,
)
from engine.buy import apply_buy_phase as apply_buy_phase_v1, get_legal_buy_actions as get_legal_buy_actions_v1
from engine.geometry import hex_neighbors
from engine.income import apply_income_phase as apply_income_phase_v1
from engine.movement import (
    apply_movement_step as apply_movement_step_v1,
    get_legal_cavalry_actions,
    get_legal_movement_actions,
)
from engine.setup import create_initial_state
from engine.state import IMPASSABLE_TERRAIN, Battle
from engine.terrain import apply_terrain_effects as apply_terrain_effects_v1
from engine.turn import get_legal_target_actions as get_legal_target_actions_v1
from engine.turn import run_turn as run_turn_v1

from engine_v2.battle import (
    faction_totals as faction_totals_v2,
    get_legal_target_actions as get_legal_target_actions_v2,
    get_winner as get_winner_v2,
    rectify_overflow as rectify_overflow_v2,
    resolve_full_battle as resolve_full_battle_v2,
)
from engine_v2.buy import apply_buy_phase as apply_buy_phase_v2, get_legal_buy_actions as get_legal_buy_actions_v2
from engine_v2.geometry import HexGrid
from engine_v2.income import apply_income_phase as apply_income_phase_v2
from engine_v2.movement import (
    apply_movement_step as apply_movement_step_v2,
    legal_cavalry_mask,
    legal_movement_mask,
)
from engine_v2.state import NO_FACTION, NO_ORIGIN, from_v1, state_diff
from engine_v2.terrain import apply_terrain_effects as apply_terrain_effects_v2
from engine_v2.turn import run_turn as run_turn_v2


def random_v1_state_with_armies(radius, num_factions, seed, army_fraction=0.35, freeze_chance=0.15):
    """A v1 GameState with real terrain/cities (from create_initial_state)
    plus randomly scattered armies - create_initial_state alone places no
    armies (they only ever appear via the buy phase, which isn't part of
    this milestone), so tests need their own way to get armies onto the
    board."""
    rng = random.Random(seed)
    state = create_initial_state(radius=radius, num_factions=num_factions, seed=seed)

    coords = list(state.board.keys())
    rng.shuffle(coords)
    num_armies = int(len(coords) * army_fraction)
    for coord in coords[:num_armies]:
        h = state.board[coord]
        if h.terrain in IMPASSABLE_TERRAIN:
            continue
        faction = rng.randrange(num_factions)
        total = rng.randint(1, 6)
        i = rng.randint(0, total)
        c = rng.randint(0, total - i)
        a = total - i - c
        h.army = {
            "faction": faction, "infantry": i, "cavalry": c, "archers": a,
            "frozen": rng.random() < freeze_chance,
        }

    # Occasionally pre-seed an in-progress battle too, to exercise the
    # "reinforcing an already-locked hex" path from turn 1 of the test,
    # not just battles that happen to form organically during it.
    if rng.random() < 0.4:
        battle_coord = rng.choice(coords)
        h = state.board[battle_coord]
        h.army = None
        h.locked = True
        contributions = []
        for _ in range(2):
            origin = rng.choice(coords)
            contributions.append({
                "faction": rng.randrange(num_factions), "origin_hex": origin,
                "infantry": rng.randint(0, 3), "cavalry": rng.randint(0, 2), "archers": rng.randint(0, 2),
            })
        state.battles[battle_coord] = Battle(hex_coord=battle_coord, contributions=contributions)

    return state


def randomize_players_and_cities(state, num_factions, rng):
    """Mutates a v1 GameState's player resources and city ownership so
    income-phase tests actually exercise varied city counts (0, 1, 2, 3+
    per faction) and some dead factions, not just whatever
    create_initial_state happens to produce (always exactly 2 cities per
    faction, everyone alive)."""
    coords = list(state.board.keys())
    rng.shuffle(coords)
    num_cities = int(len(coords) * 0.15)
    for coord in coords:
        state.board[coord].city_owner = None
    for coord in coords[:num_cities]:
        state.board[coord].city_owner = rng.randrange(num_factions)

    for faction, player in state.players.items():
        player.silver = rng.randint(0, 50)
        player.kill_xp = rng.randint(0, 10)
        player.alive = rng.random() > 0.1

    return state


def random_v1_battle_scenario(radius, num_factions, seed):
    """A v1 GameState with one rich, ready-to-resolve locked battle: 2-4
    factions, multiple contributions each (mixed infantry/cavalry/
    archers so archer abilities and cavalry dismounts both get
    exercised), some contributions originating from forest hexes (tests
    the forest terrain bonus), and the battle hex sometimes a city owned
    by one of the battling factions (tests the city terrain bonus)."""
    rng = random.Random(seed)
    state = create_initial_state(radius=radius, num_factions=num_factions, seed=seed)
    coords = list(state.board.keys())

    battle_coord = rng.choice(coords)
    h = state.board[battle_coord]
    h.army = None
    h.locked = True
    battle_factions = rng.sample(range(num_factions), min(rng.randint(2, 4), num_factions))
    h.city_owner = rng.choice(battle_factions) if rng.random() < 0.5 else None

    forest_coords = [c for c, hh in state.board.items() if hh.terrain == "forest"] or coords

    contributions = []
    for faction in battle_factions:
        for _ in range(rng.randint(1, 2)):
            origin = rng.choice(forest_coords) if rng.random() < 0.4 else rng.choice(coords)
            infantry, cavalry, archers = rng.randint(0, 4), rng.randint(0, 3), rng.randint(0, 3)
            if infantry + cavalry + archers == 0:
                infantry = 1
            contributions.append({
                "faction": faction, "origin_hex": origin,
                "infantry": infantry, "cavalry": cavalry, "archers": archers,
            })
    rng.shuffle(contributions)

    state.battles[battle_coord] = Battle(hex_coord=battle_coord, contributions=contributions)
    return state, battle_coord


def random_decide_rectification_v1(battle, winner_faction, rng):
    """Mirrors engine/agents/random_agent.py's RandomAgent.decide_rectification."""
    totals = battle.faction_totals()[winner_faction]
    overflow = sum(totals.values()) - 6
    if overflow <= 0:
        return []
    origins = [c["origin_hex"] for c in battle.contributions if c["faction"] == winner_faction]
    if not origins:
        return []
    send_back = []
    remaining = dict(totals)
    for ut in ("infantry", "cavalry", "archers"):
        while overflow > 0 and remaining[ut] > 0:
            send_back.append({"origin_hex": rng.choice(origins), "units": {ut: 1}})
            remaining[ut] -= 1
            overflow -= 1
    return send_back


def random_decide_rectification_v2(state, hex_index, winner_faction, rng):
    totals = faction_totals_v2(state, hex_index)[winner_faction]
    overflow = int(totals.sum()) - 6
    if overflow <= 0:
        return []
    origins = [
        int(state.battle_origin[hex_index, k]) for k in range(state.battle_faction.shape[1])
        if state.battle_faction[hex_index, k] == winner_faction
    ]
    if not origins:
        return []
    send_back = []
    remaining = [int(x) for x in totals]
    for ut in (0, 1, 2):
        while overflow > 0 and remaining[ut] > 0:
            units = [0, 0, 0]
            units[ut] = 1
            send_back.append({"origin_hex": rng.choice(origins), "units": units})
            remaining[ut] -= 1
            overflow -= 1
    return send_back


def buy_action_to_v2(action, grid):
    if action["type"] == "buy_infantry":
        return {"type": "buy_infantry", "city_hex": grid.index_of(tuple(action["city_hex"]))}
    return {"type": "convert_to_special", "hex": grid.index_of(tuple(action["hex"])), "unit_type": action["unit_type"]}


def v1_action_to_v2(action, grid):
    from_idx = grid.index_of(tuple(action["from_hex"]))
    to_idx = grid.index_of(tuple(action["to_hex"]))
    direction = grid.direction_between(from_idx, to_idx)
    assert direction is not None, f"{action['from_hex']} -> {action['to_hex']} aren't neighbors"
    return from_idx, direction


def mask_to_pairs(mask):
    rows, cols = np.nonzero(mask)
    return set(zip(rows.tolist(), cols.tolist()))


def test_geometry_parity():
    print("geometry parity...")
    for radius in (2, 4, 8):
        grid = HexGrid(radius)
        for i, coord in enumerate(grid.coords):
            v1_neighbors = hex_neighbors(coord, radius)
            v2_neighbors = [grid.coord_of(int(j)) for j in grid.neighbor_table[i] if j >= 0]
            assert v1_neighbors == v2_neighbors, (
                f"radius={radius} coord={coord}: v1={v1_neighbors} v2={v2_neighbors}"
            )
    print("  OK")


def test_state_conversion(seeds, radius=5, num_factions=6):
    print("v1->v2 state conversion...")
    grid = HexGrid(radius)
    for seed in seeds:
        v1_state = random_v1_state_with_armies(radius, num_factions, seed)
        v2_state = from_v1(v1_state, num_factions, grid=grid)
        mismatches = state_diff(v1_state, v2_state)
        assert not mismatches, f"seed={seed}: {mismatches}"
    print(f"  OK ({len(seeds)} seeds)")


def test_legality_mask_parity(seeds, radius=5, num_factions=6):
    print("movement/cavalry legality mask parity...")
    grid = HexGrid(radius)
    for seed in seeds:
        v1_state = random_v1_state_with_armies(radius, num_factions, seed)
        v2_state = from_v1(v1_state, num_factions, grid=grid)

        for faction in range(num_factions):
            v1_actions = get_legal_movement_actions(v1_state, faction)
            expected = {v1_action_to_v2(a, grid) for a in v1_actions}
            actual = mask_to_pairs(legal_movement_mask(v2_state, faction))
            assert expected == actual, (
                f"seed={seed} faction={faction} movement mask mismatch: "
                f"expected-actual={expected - actual} actual-expected={actual - expected}"
            )

            v1_cav_actions = get_legal_cavalry_actions(v1_state, faction)
            expected_cav = {v1_action_to_v2(a, grid) for a in v1_cav_actions}
            actual_cav = mask_to_pairs(legal_cavalry_mask(v2_state, faction))
            assert expected_cav == actual_cav, (
                f"seed={seed} faction={faction} cavalry mask mismatch: "
                f"expected-actual={expected_cav - actual_cav} actual-expected={actual_cav - expected_cav}"
            )
    print(f"  OK ({len(seeds)} seeds x {num_factions} factions)")


def test_apply_movement_step_parity(seeds, radius=5, num_factions=6, num_steps=10, move_chance=0.7):
    print("apply_movement_step parity across randomized steps...")
    grid = HexGrid(radius)
    total_steps = 0
    for seed in seeds:
        rng = random.Random(seed * 7919 + 1)
        v1_state = random_v1_state_with_armies(radius, num_factions, seed)
        v2_state = from_v1(v1_state, num_factions, grid=grid)

        mismatches = state_diff(v1_state, v2_state)
        assert not mismatches, f"seed={seed} initial conversion mismatch: {mismatches}"

        for step in range(num_steps):
            use_cavalry = step % 2 == 1
            actions_v1 = {}
            actions_v2 = {}
            for faction in range(num_factions):
                legal = get_legal_cavalry_actions(v1_state, faction) if use_cavalry \
                    else get_legal_movement_actions(v1_state, faction)
                if not legal or rng.random() > move_chance:
                    continue
                chosen = rng.choice(legal)
                actions_v1[faction] = [chosen]
                actions_v2[faction] = v1_action_to_v2(chosen, grid)

            apply_movement_step_v1(v1_state, actions_v1)
            apply_movement_step_v2(v2_state, actions_v2, cavalry_only=use_cavalry)
            total_steps += 1

            mismatches = state_diff(v1_state, v2_state)
            assert not mismatches, (
                f"seed={seed} step={step} cavalry={use_cavalry}\n"
                f"actions_v1={actions_v1}\nmismatches:\n" + "\n".join(mismatches)
            )
    print(f"  OK ({len(seeds)} seeds x {num_steps} steps = {total_steps} steps total)")


def test_income_parity(seeds, radius=5, num_factions=6):
    print("income phase parity...")
    grid = HexGrid(radius)
    for seed in seeds:
        rng = random.Random(seed * 104729 + 3)
        v1_state = random_v1_state_with_armies(radius, num_factions, seed)
        randomize_players_and_cities(v1_state, num_factions, rng)
        v2_state = from_v1(v1_state, num_factions, grid=grid)

        apply_income_phase_v1(v1_state)
        apply_income_phase_v2(v2_state)

        mismatches = state_diff(v1_state, v2_state)
        assert not mismatches, f"seed={seed}:\n" + "\n".join(mismatches)
    print(f"  OK ({len(seeds)} seeds)")


def test_buy_parity(seeds, radius=5, num_factions=6, max_actions=6):
    print("buy phase parity...")
    grid = HexGrid(radius)
    for seed in seeds:
        rng = random.Random(seed * 613 + 5)
        v1_state = random_v1_state_with_armies(radius, num_factions, seed)
        randomize_players_and_cities(v1_state, num_factions, rng)
        v2_state = from_v1(v1_state, num_factions, grid=grid)

        mismatches = state_diff(v1_state, v2_state)
        assert not mismatches, f"seed={seed} initial mismatch: {mismatches}"

        actions_v1_by_faction = {}
        actions_v2_by_faction = {}
        for faction in range(num_factions):
            legal_v1 = get_legal_buy_actions_v1(v1_state, faction)

            # cross-check the legal-action sets themselves, not just the
            # end result of applying a chosen subset
            expected = {
                (a["type"], grid.index_of(tuple(a.get("city_hex", a.get("hex")))), a.get("unit_type"))
                for a in legal_v1
            }
            legal_v2 = get_legal_buy_actions_v2(v2_state, faction)
            actual = {(a["type"], a.get("city_hex", a.get("hex")), a.get("unit_type")) for a in legal_v2}
            assert expected == actual, (
                f"seed={seed} faction={faction} legal buy actions differ: "
                f"expected-actual={expected - actual} actual-expected={actual - expected}"
            )

            if not legal_v1:
                continue
            chosen = [rng.choice(legal_v1) for _ in range(rng.randint(1, max_actions))]
            actions_v1_by_faction[faction] = chosen
            actions_v2_by_faction[faction] = [buy_action_to_v2(a, grid) for a in chosen]

        apply_buy_phase_v1(v1_state, actions_v1_by_faction)
        apply_buy_phase_v2(v2_state, actions_v2_by_faction)

        mismatches = state_diff(v1_state, v2_state)
        assert not mismatches, f"seed={seed}:\n" + "\n".join(mismatches)
    print(f"  OK ({len(seeds)} seeds)")


def test_battle_parity(seeds, radius=6, num_factions=6, target_abstain_chance=0.15):
    print("battle resolution parity...")
    grid = HexGrid(radius)
    for seed in seeds:
        v1_state, battle_coord = random_v1_battle_scenario(radius, num_factions, seed)
        v2_state = from_v1(v1_state, num_factions, grid=grid)
        hex_index = grid.index_of(battle_coord)

        mismatches = state_diff(v1_state, v2_state)
        assert not mismatches, f"seed={seed} initial mismatch: {mismatches}"

        target_rng_v1 = random.Random(seed * 31 + 1)
        target_rng_v2 = random.Random(seed * 31 + 1)
        battle_rng_v1 = random.Random(seed * 17 + 2)
        battle_rng_v2 = random.Random(seed * 17 + 2)

        def v1_target_fn(battle, faction, _rng=target_rng_v1):
            legal = get_legal_target_actions_v1(battle, faction)
            if not legal or _rng.random() < target_abstain_chance:
                return None
            return _rng.choice(legal)

        def v2_target_fn(state, hidx, faction, _rng=target_rng_v2):
            legal = get_legal_target_actions_v2(state, hidx, faction)
            if not legal or _rng.random() < target_abstain_chance:
                return None
            return _rng.choice(legal)

        battle_v1 = v1_state.battles[battle_coord]
        resolve_full_battle_v1(battle_v1, v1_target_fn, v1_state, rng=battle_rng_v1)
        resolve_full_battle_v2(v2_state, hex_index, v2_target_fn, battle_rng_v2)

        mismatches = state_diff(v1_state, v2_state)
        assert not mismatches, f"seed={seed} post-resolution mismatch:\n" + "\n".join(mismatches)

        winner_v1 = get_winner_v1(battle_v1)
        winner_v2 = get_winner_v2(v2_state, hex_index)
        assert winner_v1 == winner_v2, f"seed={seed} winner mismatch v1={winner_v1} v2={winner_v2}"

        if winner_v1 is None:
            h = v1_state.board[battle_coord]
            h.army = None
            h.locked = False
            del v1_state.battles[battle_coord]

            v2_state.army_faction[hex_index] = NO_FACTION
            v2_state.army_units[hex_index] = 0
            v2_state.locked[hex_index] = False
            v2_state.battle_faction[hex_index] = NO_FACTION
            v2_state.battle_origin[hex_index] = NO_ORIGIN
            v2_state.battle_units[hex_index] = 0
            v2_state.battle_order.remove(hex_index)
        else:
            rect_rng_v1 = random.Random(seed * 53 + 3)
            rect_rng_v2 = random.Random(seed * 53 + 3)
            send_back_v1 = random_decide_rectification_v1(battle_v1, winner_v1, rect_rng_v1)
            send_back_v2 = random_decide_rectification_v2(v2_state, hex_index, winner_v2, rect_rng_v2)
            rectify_overflow_v1(v1_state, battle_coord, winner_v1, send_back_v1)
            rectify_overflow_v2(v2_state, hex_index, winner_v2, send_back_v2)

        mismatches = state_diff(v1_state, v2_state)
        assert not mismatches, f"seed={seed} post-rectify mismatch:\n" + "\n".join(mismatches)

    print(f"  OK ({len(seeds)} seeds)")


def test_terrain_parity(seeds, radius=5, num_factions=6):
    print("terrain effects parity...")
    grid = HexGrid(radius)
    for seed in seeds:
        v1_state = random_v1_state_with_armies(radius, num_factions, seed)
        v2_state = from_v1(v1_state, num_factions, grid=grid)

        apply_terrain_effects_v1(v1_state)
        apply_terrain_effects_v2(v2_state)

        mismatches = state_diff(v1_state, v2_state)
        assert not mismatches, f"seed={seed}:\n" + "\n".join(mismatches)
    print(f"  OK ({len(seeds)} seeds)")


def random_buy_choice(legal, rng, max_n=3):
    if not legal:
        return []
    n = rng.randint(0, min(max_n, len(legal)))
    return rng.sample(legal, n) if n > 0 else []


def random_whole_stack_choice(legal, rng, skip_chance=0.5):
    """Picks a random legal movement/cavalry action without partial-
    splitting it (unlike RandomAgent's real decide_movement/
    decide_cavalry) - engine_v2's action representation can only move a
    whole army/cavalry contingent (see movement.py's SCOPE decision), so
    driving v1 with a sub-split choice here would make the two engines'
    inputs genuinely different, not just differently-represented."""
    if not legal or rng.random() < skip_chance:
        return None
    return rng.choice(legal)


def test_full_turn_parity(seeds, radius=5, num_factions=5, num_turns=4):
    """Drives v1's real run_turn with a recording agent (random but
    v2-representable decisions - see random_whole_stack_choice), then
    replays translated versions of the exact same decisions into v2's
    run_turn, turn by turn. This is the piece that validates the
    orchestration itself (phase order, turn_number, elimination, and -
    the part that actually surfaced a design gap - battle processing
    order/shared infantry cap when a turn has more than one battle),
    on top of the mechanics each phase's own dedicated test already
    covers.
    """
    print("full turn orchestration parity...")
    grid = HexGrid(radius)
    for seed in seeds:
        v1_state = random_v1_state_with_armies(radius, num_factions, seed, army_fraction=0.5)
        randomize_players_and_cities(v1_state, num_factions, random.Random(seed * 91 + 1))
        v2_state = from_v1(v1_state, num_factions, grid=grid)

        mismatches = state_diff(v1_state, v2_state)
        assert not mismatches, f"seed={seed} initial mismatch: {mismatches}"

        for turn_idx in range(num_turns):
            log_buy, log_movement, log_cavalry = {}, {}, {}
            log_target, log_rectify = defaultdict(list), defaultdict(list)

            class RecordingAgent(BaseAgentV1):
                def __init__(self, faction, rng):
                    super().__init__(faction)
                    self.rng = rng

                def decide_buy(self, state, faction, legal):
                    chosen = random_buy_choice(legal, self.rng)
                    log_buy[faction] = chosen
                    return chosen

                def decide_movement(self, state, faction, step, legal):
                    chosen = random_whole_stack_choice(legal, self.rng)
                    log_movement[(faction, step)] = chosen
                    return [chosen] if chosen else []

                def decide_cavalry(self, state, faction, step, legal):
                    chosen = random_whole_stack_choice(legal, self.rng)
                    log_cavalry[(faction, step)] = chosen
                    return [chosen] if chosen else []

                def decide_target(self, state, battle, faction, legal_targets):
                    if not legal_targets or self.rng.random() < 0.15:
                        target = None
                    else:
                        target = self.rng.choice(legal_targets)
                    log_target[faction].append(target)
                    return target

                def decide_rectification(self, state, battle, winning_faction):
                    send_back = random_decide_rectification_v1(battle, winning_faction, self.rng)
                    log_rectify[winning_faction].append(send_back)
                    return send_back

            agents_v1 = {
                f: RecordingAgent(f, random.Random(seed * 1013 + turn_idx * 97 + f))
                for f in range(num_factions)
            }
            dice_rng_v1 = random.Random(seed * 2027 + turn_idx)
            dice_rng_v2 = random.Random(seed * 2027 + turn_idx)

            run_turn_v1(v1_state, agents_v1, rng=dice_rng_v1)

            decide_buy_v2, decide_movement_v2, decide_cavalry_v2 = {}, {}, {}
            decide_target_v2, decide_rectification_v2 = {}, {}
            for faction in range(num_factions):
                def db(state, faction, legal, _f=faction):
                    return [buy_action_to_v2(a, grid) for a in log_buy.get(_f, [])]

                def dm(state, faction, step, legal, _f=faction):
                    chosen = log_movement.get((_f, step))
                    return v1_action_to_v2(chosen, grid) if chosen else None

                def dc(state, faction, step, legal, _f=faction):
                    chosen = log_cavalry.get((_f, step))
                    return v1_action_to_v2(chosen, grid) if chosen else None

                def dt(state, hex_index, faction, _f=faction):
                    return log_target[_f].pop(0)

                def dr(state, hex_index, winner, _f=faction):
                    return [
                        {
                            "origin_hex": grid.index_of(tuple(e["origin_hex"])),
                            "units": [e["units"].get(ut, 0) for ut in ("infantry", "cavalry", "archers")],
                        }
                        for e in log_rectify[_f].pop(0)
                    ]

                decide_buy_v2[faction] = db
                decide_movement_v2[faction] = dm
                decide_cavalry_v2[faction] = dc
                decide_target_v2[faction] = dt
                decide_rectification_v2[faction] = dr

            run_turn_v2(
                v2_state, decide_buy_v2, decide_movement_v2, decide_cavalry_v2,
                decide_target_v2, decide_rectification_v2, rng=dice_rng_v2,
            )

            mismatches = state_diff(v1_state, v2_state)
            assert not mismatches, f"seed={seed} turn={turn_idx}:\n" + "\n".join(mismatches)

        assert v1_state.turn_number == v2_state.turn_number == num_turns

    print(f"  OK ({len(seeds)} seeds x {num_turns} turns)")


def main():
    seeds = list(range(1, 41))

    test_geometry_parity()
    test_state_conversion(seeds)
    test_legality_mask_parity(seeds)
    test_apply_movement_step_parity(seeds)
    test_income_parity(seeds)
    test_terrain_parity(seeds)
    test_buy_parity(seeds)
    test_battle_parity(seeds)
    test_full_turn_parity(seeds)

    print("\nAll engine_v2 parity tests passed.")


if __name__ == "__main__":
    main()
