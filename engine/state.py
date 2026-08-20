"""
Core state data structures. No game logic lives here - just the shape
of a GameState snapshot and JSON (de)serialization.

Design notes:
  - A hex holds at most one "settled" army. While a battle is pending
    on a hex, that hex is locked and its combatants live in
    state.battles[hex] instead, as a list of per-faction contributions
    (each contribution remembers which hex it moved in from, since the
    winner needs that to send overflow units back after the fight).
  - kill_xp is tracked as a plain integer (currency), not literal unit
    tokens - it can only ever be spent to convert infantry into
    cavalry/archers, so there's no need to model it as physical pieces.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional

UNIT_TYPES = ["infantry", "cavalry", "archers"]
TERRAIN_TYPES = ["plains", "forest", "mountain", "lake", "desert", "marsh"]
IMPASSABLE_TERRAIN = {"mountain", "lake"}

MAX_STACK_SIZE = 6
SPAWN_CAPS = {"infantry": 24, "cavalry": 12, "archers": 12}


def empty_army(faction):
    return {"faction": faction, "infantry": 0, "cavalry": 0, "archers": 0, "frozen": False}


def army_total(army):
    if army is None:
        return 0
    return army["infantry"] + army["cavalry"] + army["archers"]


@dataclass
class HexState:
    terrain: str
    city_owner: Optional[int] = None
    army: Optional[dict] = None          # empty_army() shape, or None
    locked: bool = False                 # True while a battle is pending here

    def to_dict(self):
        """Emits the same shape the visualizer reads directly - no
        translation layer between engine and visualizer. `troops` is a
        shallow copy since army dict values get mutated in place turn
        to turn; without copying, every logged snapshot would end up
        pointing at the same live dict and all show the final state."""
        return {
            "terrain": self.terrain,
            "city": self.city_owner,
            "troops": dict(self.army) if self.army is not None else None,
            "locked": self.locked,
        }

    @staticmethod
    def from_dict(d):
        return HexState(terrain=d["terrain"], city_owner=d.get("city"),
                         army=d.get("troops"), locked=d.get("locked", False))


@dataclass
class PlayerState:
    faction: int
    silver: int = 0
    kill_xp: int = 0                     # plain currency count, not literal tokens
    voting_tokens: int = 1
    alive: bool = True

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return PlayerState(faction=d["faction"], silver=d["silver"],
                            kill_xp=d.get("kill_xp", 0),
                            voting_tokens=d.get("voting_tokens", 1),
                            alive=d.get("alive", True))


@dataclass
class Battle:
    """A pending or in-progress fight on one hex.

    contributions: list of {"faction", "origin_hex", "infantry", "cavalry", "archers"}
    Multiple contributions can share a faction (reinforcements arriving
    in later movement steps) - they're kept separate so rectification
    knows which origin hex each unit can be sent back to.
    round_number: int, starts at 0 (pre-round / archer-ability stage)
    """
    hex_coord: tuple
    contributions: list = field(default_factory=list)
    round_number: int = 0

    def to_dict(self):
        return {"hex": list(self.hex_coord), "contributions": self.contributions,
                "round_number": self.round_number}

    @staticmethod
    def from_dict(d):
        return Battle(hex_coord=tuple(d["hex"]), contributions=d["contributions"],
                      round_number=d.get("round_number", 0))

    def faction_totals(self):
        """Sum contributions per faction -> {faction: {"infantry":n, "cavalry":n, "archers":n}}"""
        totals = {}
        for c in self.contributions:
            t = totals.setdefault(c["faction"], {"infantry": 0, "cavalry": 0, "archers": 0})
            for ut in UNIT_TYPES:
                t[ut] += c[ut]
        return totals

    def factions(self):
        return list(self.faction_totals().keys())


@dataclass
class GameState:
    board: dict            # {(q,r,s): HexState}
    players: dict           # {faction_id: PlayerState}
    battles: dict            # {(q,r,s): Battle}
    turn_number: int = 0
    radius: int = 8

    def to_dict(self):
        return {
            "radius": self.radius,
            "turn_number": self.turn_number,
            "hexes": [
                {"q": c[0], "r": c[1], "s": c[2], **h.to_dict()}
                for c, h in self.board.items()
            ],
            "players": [p.to_dict() for p in self.players.values()],
            "battles": [b.to_dict() for b in self.battles.values()],
        }

    @staticmethod
    def from_dict(d):
        board = {}
        for entry in d["hexes"]:
            coord = (entry["q"], entry["r"], entry["s"])
            board[coord] = HexState.from_dict(entry)
        players = {p["faction"]: PlayerState.from_dict(p) for p in d["players"]}
        battles = {}
        for bd in d["battles"]:
            b = Battle.from_dict(bd)
            battles[b.hex_coord] = b
        return GameState(board=board, players=players, battles=battles,
                          turn_number=d.get("turn_number", 0), radius=d.get("radius", 8))


def count_units_in_play(state, faction, unit_type):
    """How many of `faction`'s `unit_type` units currently exist, on the
    board or mid-battle. This is the live, concurrent cap check
    ("physical pieces in the bag") - NOT a lifetime production count.
    A unit that dies frees up capacity immediately; kill-XP is spent
    as currency and has no bearing on this count.

    NOTE: this does a full board+battles scan every call. Profiling
    showed callers that need this repeatedly within one phase (buy,
    battle dismounts) re-scanning on every check/application, which
    dominated total engine runtime on large/long games. Those callers
    now take one snapshot via count_all_units_in_play() and maintain a
    local running tally instead of calling this in a loop - see
    buy.py and turn.py's _run_battle_phase. Prefer that pattern for
    any new hot-path caller; this function is fine for one-off checks."""
    total = 0
    for h in state.board.values():
        if h.army and h.army["faction"] == faction:
            total += h.army[unit_type]
    for b in state.battles.values():
        for c in b.contributions:
            if c["faction"] == faction:
                total += c[unit_type]
    return total


def count_all_units_in_play(state, faction):
    """Same data as three count_units_in_play() calls, but in a single
    pass over the board/battles - the snapshot callers take once before
    maintaining their own running tally (see count_units_in_play's note)."""
    counts = {"infantry": 0, "cavalry": 0, "archers": 0}
    for h in state.board.values():
        if h.army and h.army["faction"] == faction:
            for ut in UNIT_TYPES:
                counts[ut] += h.army[ut]
    for b in state.battles.values():
        for c in b.contributions:
            if c["faction"] == faction:
                for ut in UNIT_TYPES:
                    counts[ut] += c[ut]
    return counts