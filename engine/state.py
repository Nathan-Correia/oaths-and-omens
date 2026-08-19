"""
Core state data structures. No game logic lives here - just the shape
of a GameState snapshot and JSON (de)serialization.

Design notes:
  - A hex holds at most one "settled" army. While a battle is pending
    on a hex, that hex is locked and its combatants live in
    state.battles[hex] instead, as a list of per-faction contributions
    (each contribution remembers which hex it moved in from, since the
    winner needs that to send overflow units back after the fight).
  - kill_xp_bank is a list of unit tokens (not just a count), because
    the "swap a piece from someone else's bank" rule needs to know
    which player each banked unit conceptually belongs to.
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
    kill_xp_bank: list = field(default_factory=list)   # list of {"unit_type": str}
    spawn_counts: dict = field(default_factory=lambda: {"infantry": 0, "cavalry": 0, "archers": 0})
    voting_tokens: int = 1
    pending_free_infantry: int = 0   # queued by the cavalry death ability, resolved next buy phase
    alive: bool = True

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return PlayerState(faction=d["faction"], silver=d["silver"],
                            kill_xp_bank=d["kill_xp_bank"], spawn_counts=d["spawn_counts"],
                            voting_tokens=d.get("voting_tokens", 1),
                            pending_free_infantry=d.get("pending_free_infantry", 0),
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