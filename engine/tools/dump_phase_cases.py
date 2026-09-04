"""
Generates (before, after) state pairs for the M2 phases, for C++ parity testing.

Two sources of states, because neither alone is enough:

  1. REAL GAMES. States harvested from random-agent games driven by engine_old.
     These are guaranteed reachable and exercise whatever the engine actually
     produces - but they under-cover rare configurations. Outpost upgrades barely
     appear, and a faction with zero cities never appears at all (capitals are
     uncapturable), so apply_gold_income's no-city branch would go untested.

  2. PERTURBED STATES. A harvested state with its armies, upgrades, terrain
     occupancy and per-faction economy randomized into a different but still legal
     configuration. This is what actually covers the branches: every upgrade type,
     armies parked on desert and marsh, frozen stacks, and factions stripped of
     every city.

Both engines then run the same phase on the same "before" state and the results are
compared field by field (tests/test_phases.cpp).

The resource-choice callback is a fixed deterministic policy - see CHOICE_POLICY
below - mirrored exactly in the C++ test. It is only consulted for an outpost
adjacent to both a mountain and a lake, and alternating on (faction + hex) makes
sure both the iron and fish branches are taken.

Usage:  python engine/tools/dump_phase_cases.py > engine/tests/data/phase_cases.txt
"""

import copy
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: E402

_bootstrap.install()

from engine import collect as collect_mod  # noqa: E402
from engine import terrain as terrain_mod  # noqa: E402
from engine.placement import run_city_setup  # noqa: E402
from engine.setup import create_initial_state  # noqa: E402
from engine.turn import run_turn  # noqa: E402
from agents import compose_agents  # noqa: E402
from agents.greedy_agent import make_greedy_agents  # noqa: E402
from agents.random_agent import make_random_agents  # noqa: E402

from state_io import write_state  # noqa: E402

NO_FACTION = -1
NO_UPGRADE = -1


def choice_policy(faction, hex_index):
    """MUST match test_phases.cpp's choice_policy()."""
    return "iron" if (faction + hex_index) % 2 == 0 else "fish"


def _clone(state):
    new = copy.copy(state)
    for field in ("terrain", "city_owner", "is_capital", "outpost_upgrade", "city_placer",
                  "capital_settle_order", "army_faction", "army_units", "frozen", "locked",
                  "battle_faction", "battle_origin", "battle_units", "battle_moved",
                  "battle_round", "gold", "resources", "kill_xp", "victory_points", "alive"):
        setattr(new, field, getattr(state, field).copy())
    new.battle_order = list(state.battle_order)
    return new


def _perturb(state, rng):
    """Randomize into a different but still LEGAL configuration.

    This does the heavy lifting for coverage. A first version only re-rolled
    upgrades on cities that already existed and left everything else to real
    games - which measured out at ZERO outposts and zero upgrades of any type, so
    apply_resource_income and apply_victory_points changed nothing in 100% of
    cases and were effectively untested. It now BUILDS outposts rather than
    waiting for them.

    Outposts are placed without regard to the buy-phase placement rules (distance
    from capitals, spacing between outposts). That is fine and deliberate: none of
    the phases under test here consult those rules, and ignoring them reaches
    board configurations that a legal game would take far longer to produce.
    """
    s = _clone(state)
    n = s.num_hexes
    passable = [h for h in range(n) if s.terrain[h] not in (1, 2)]  # not mountain/lake

    # Fresh outposts, spread over every terrain type and every upgrade slot value.
    # Placed on hexes with no city and no pending battle.
    free_for_city = [h for h in passable if s.city_owner[h] == NO_FACTION and not s.locked[h]]
    rng.shuffle(free_for_city)
    n_outposts = rng.randint(0, min(len(free_for_city), 4 * s.num_factions))
    for h in free_for_city[:n_outposts]:
        s.city_owner[h] = rng.randrange(s.num_factions)
        s.is_capital[h] = False
        s.outpost_upgrade[h] = rng.choice([NO_UPGRADE, 0, 1, 2])
    # And re-roll whatever upgrades were already there.
    for h in range(n):
        if s.city_owner[h] != NO_FACTION and not s.is_capital[h]:
            s.outpost_upgrade[h] = rng.choice([NO_UPGRADE, 0, 1, 2])

    # Re-scatter armies. Hexes locked in a battle keep whatever they have, so the
    # battle invariants stay consistent.
    for h in range(n):
        if not s.locked[h]:
            s.army_faction[h] = NO_FACTION
            s.army_units[h] = 0
            s.frozen[h] = False
    free = [h for h in passable if not s.locked[h]]
    rng.shuffle(free)
    for h in free[: rng.randint(0, max(1, len(free) // 3))]:
        s.army_faction[h] = rng.randrange(s.num_factions)
        total = rng.randint(1, 6)
        units = [0, 0, 0]
        for _ in range(total):
            units[rng.randrange(3)] += 1
        s.army_units[h] = units
        s.frozen[h] = rng.random() < 0.3

    # A synthetic pending battle. None of the M2 phases read battle storage, which
    # is exactly why this is worth including: it proves they leave it untouched,
    # and it exercises the sparse battle round-trip in state_io on both sides.
    if rng.random() < 0.4:
        candidates = [h for h in passable if not s.locked[h]]
        for hex_index in rng.sample(candidates, min(len(candidates), rng.randint(1, 3))):
            s.army_faction[hex_index] = NO_FACTION
            s.army_units[hex_index] = 0
            s.frozen[hex_index] = False
            s.locked[hex_index] = True
            s.battle_round[hex_index] = rng.randint(0, 4)
            for slot in range(rng.randint(2, 5)):
                s.battle_faction[hex_index, slot] = rng.randrange(s.num_factions)
                s.battle_origin[hex_index, slot] = rng.choice(passable)
                s.battle_units[hex_index, slot] = [rng.randint(0, 4) for _ in range(3)]
                s.battle_moved[hex_index, slot] = rng.random() < 0.5
            s.battle_order.append(hex_index)

    # Occasionally strip a faction of every city, to reach apply_gold_income's
    # no-city branch (unreachable in a real game - capitals cannot be taken).
    if rng.random() < 0.25:
        victim = rng.randrange(s.num_factions)
        for h in range(n):
            if s.city_owner[h] == victim:
                s.city_owner[h] = NO_FACTION
                s.is_capital[h] = False
                s.outpost_upgrade[h] = NO_UPGRADE

    for f in range(s.num_factions):
        s.gold[f] = rng.randint(0, 60)
        s.kill_xp[f] = rng.randint(0, 20)
        s.victory_points[f] = rng.randint(0, 55)
        for r in range(4):
            s.resources[f, r] = rng.randint(0, 12)
    return s


PHASES = {
    "terrain": lambda s: terrain_mod.apply_terrain_effects(s),
    "gold_income": lambda s: collect_mod.apply_gold_income(s),
    "resource_income": lambda s: collect_mod.apply_resource_income(
        s, {f: (lambda st, fa, hx: choice_policy(fa, hx)) for f in range(s.num_factions)}),
    "victory_points": lambda s: collect_mod.apply_victory_points(s),
    "collect": lambda s: collect_mod.apply_collect_phase(
        s, {f: (lambda st, fa, hx: choice_policy(fa, hx)) for f in range(s.num_factions)}),
}


def harvest(radius, num_factions, seed, max_turns, agent_key):
    """Play a game, yielding the state at the end of every turn.

    `agent_key` matters for coverage: random agents essentially never build
    outposts, so greedy games are what supply real (as opposed to synthesized)
    outpost and upgrade configurations.
    """
    builders = {"random": make_random_agents, "greedy": make_greedy_agents}
    rng = random.Random(seed)
    state = create_initial_state(radius=radius, num_factions=num_factions, seed=seed)
    decide = compose_agents(
        {f: agent_key for f in range(num_factions)},
        {agent_key: lambda: builders[agent_key](num_factions, seed=seed)})
    (d_buy, d_move, d_cav, d_target, d_rect, d_res, d_place, d_draft, d_swap) = decide
    state = run_city_setup(state, d_place, d_draft, d_swap, rng)
    yield _clone(state)
    for _ in range(max_turns):
        state = run_turn(state, d_buy, d_move, d_cav, d_target, d_rect, d_res, rng=rng)
        yield _clone(state)


SCENARIOS = [
    # radius, factions, seed, turns, agent
    (4, 4, 1, 8, "random"),
    (5, 6, 2, 8, "random"),
    (7, 8, 3, 8, "random"),
    (8, 8, 4, 6, "random"),
    (5, 10, 5, 6, "random"),
    # Greedy games actually build outposts and buy upgrades - the states that
    # make apply_resource_income and apply_victory_points do anything at all.
    (7, 8, 11, 14, "greedy"),
    (5, 6, 12, 14, "greedy"),
    (8, 8, 13, 12, "greedy"),
    (4, 4, 14, 12, "greedy"),
]


def main():
    out = sys.stdout
    rng = random.Random(20260903)
    cases = []

    for radius, num_factions, seed, turns, agent_key in SCENARIOS:
        for i, snap in enumerate(harvest(radius, num_factions, seed, turns, agent_key)):
            for phase, fn in PHASES.items():
                # Straight from a real game.
                before = _clone(snap)
                after = _clone(before)
                fn(after)
                cases.append((f"{phase}/real/{agent_key}-r{radius}f{num_factions}s{seed}t{i}", before, after))
                # And a perturbed variant, for the branches real games miss.
                before = _perturb(snap, rng)
                after = _clone(before)
                fn(after)
                cases.append((f"{phase}/perturbed/{agent_key}-r{radius}f{num_factions}s{seed}t{i}", before, after))

    out.write(f"CASES {len(cases)}\n")
    for name, before, after in cases:
        out.write(f"CASE {name}\n")
        write_state(out, before)
        write_state(out, after)


if __name__ == "__main__":
    main()
