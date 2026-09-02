"""
Hexagon-shaped hex grid visualizer (pygame).

Reads board_state.json (the diff-based format written by
run.py / engine/turn.py:run_turn_and_log) and reconstructs
every turn's checkpoint states (start-of-turn keyframe, then after
buy, each of the 3 movement steps, each of the 2 cavalry steps, and
after battle - see engine/turn.py's CHECKPOINT_LABELS) by replaying that
turn's sparse deltas on top of a running full board. A single slider
scrubs through every checkpoint of every turn as one continuous
timeline (turn * len(CHECKPOINT_LABELS) + checkpoint), since every turn
always produces the same fixed number of checkpoints.

Each hex can show:
  - terrain (washed-out fill color, background context only)
  - a capital (top-center, square+triangle "building" glyph, faction-colored)
    or an outpost (top-center, small triangle, faction-colored) - a hex
    never has both, since city_owner is one-or-the-other per hex
  - EITHER a peaceful troop row (center, up to 3 shapes: circle=infantry,
    square=cavalry, triangle=archers, each showing its count) OR, if the
    hex is currently locked in a pending battle, a stack of small
    faction-colored rectangles - one per faction contributing to the
    fight, each showing "infantry cavalry archers" as plain numbers.

All colors are defined as named constants up top so they're easy to
retune without touching any drawing logic.
"""

import json
import math
import os
import sys
import pygame

from hex_common import (
    hex_to_pixel, hex_corner, compute_hex_size,
)

try:
    from engine.turn import CHECKPOINT_LABELS
    from engine.collect import VP_TO_WIN
except ImportError:
    CHECKPOINT_LABELS = ["Start", "Buy", "Move 1", "Move 2", "Move 3",
                          "Cav 1", "Cav 2", "Battle"]
    VP_TO_WIN = 50

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOARD_AREA_W = 1000  # width reserved for the hex board itself (unchanged from before)
SIDEBAR_WIDTH = 340   # extra width added for the always-visible player info panel
WINDOW_W = BOARD_AREA_W + SIDEBAR_WIDTH
WINDOW_H = 1010
MARGIN = 20  # pixels of padding around the board

SLIDER_BAND_HEIGHT = 70  # reserved band at the bottom for the slider
SLIDER_TRACK_COLOR = (70, 70, 78)
SLIDER_FILL_COLOR = (120, 160, 220)
SLIDER_HANDLE_COLOR = (230, 230, 235)
SLIDER_HANDLE_RADIUS = 9
SLIDER_LABEL_COLOR = (220, 220, 225)

# --- Score ticker: a vertical panel on the side, one column per faction
# (so close/tied scores don't just overlap into an unreadable blob) - each
# column is its own 0-VP_TO_WIN vertical track (0 at the bottom, climbing
# toward VP_TO_WIN at the top, like a thermometer), a dot marking that
# faction's current score, swatch + numeric score below the track.
SCORE_TICKER_COLUMN_WIDTH = 40
SCORE_TICKER_SIDE_MARGIN = 20   # left/right padding within the ticker panel
SCORE_TICKER_TOP_PADDING = 30   # room for the title label above the tracks
SCORE_TICKER_BOTTOM_PADDING = 46  # room for the swatch + score number below each track
SCORE_TICKER_TRACK_COLOR = (70, 70, 78)
SCORE_TICKER_WIN_LINE_COLOR = (200, 200, 80)
SCORE_TICKER_HANDLE_RADIUS = 6

SIDEBAR_BG_COLOR = (24, 24, 30)
SIDEBAR_DIVIDER_COLOR = (50, 50, 58)
SIDEBAR_DEAD_TEXT_COLOR = (110, 110, 115)

# --- Battle log popup: a table, one column per faction, one row per
# round. Sized so 3 columns fit comfortably; fewer factions just means
# a narrower popup (not padded out to look like 3 always).
BATTLE_POPUP_BG_COLOR = (20, 20, 26)
BATTLE_POPUP_BORDER_COLOR = (90, 90, 100)
BATTLE_COLUMN_WIDTH = 260
BATTLE_LABEL_COL_WIDTH = 110
BATTLE_ROW_HEIGHT = 70
BATTLE_HEADER_HEIGHT = 40
BATTLE_POPUP_PADDING = 16
BATTLE_CELL_TEXT_COLOR = (225, 225, 230)
BATTLE_CELL_DIM_COLOR = (130, 130, 138)

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "board_state.json")

# --- Terrain colors: washed-out / pastel so they read as background
# texture rather than competing with faction colors. Edit freely.
# Matches engine/state.py's TERRAIN_TYPES exactly.
TERRAIN_COLORS = {
    "plains":   (210, 232, 197),  # pale light green
    "forest":   (176, 204, 176),  # pale dark green
    "mountain": (205, 205, 205),  # pale grey
    "lake":     (190, 213, 232),  # pale blue
    "desert":   (235, 222, 168),  # pale yellow (previously "hills")
    "marsh":    (191, 173, 150),  # dull washed brown
}

# --- Faction colors: one qualitative, colorblind-conscious palette
# (Dark2/Set1-style), used consistently for troop shapes, city icons,
# AND battle-contribution rectangles. Index = faction id (0-7).
FACTION_COLORS = [
    (228, 26, 28),    # 0 red
    (55, 126, 184),   # 1 blue
    (77, 175, 74),    # 2 green
    (152, 78, 163),   # 3 purple
    (255, 127, 0),    # 4 orange
    (166, 86, 40),    # 5 brown
    (247, 129, 191),  # 6 pink
    (153, 153, 153),  # 7 grey
]

# --- Structural / neutral colors
HEX_OUTLINE_COLOR = (90, 90, 90)
BG_COLOR = (15, 15, 20)
SHAPE_OUTLINE_COLOR = (30, 30, 30)
TEXT_COLOR = (255, 255, 255)

TROOP_TYPES = ["infantry", "cavalry", "archers"]  # circle, square, triangle

# Troop shape sizing (radius/half-width in px) - fixed, large enough to hold a number
SHAPE_SIZE = 11
SQUARE_SIZE = SHAPE_SIZE - 1  # cavalry square rendered a touch smaller than the others

# City icon sizing (no outline on these, kept separate from troop shapes)
CITY_ICON_SIZE = 9  # half-width of the square base

# Outpost icon sizing - a small outlined shape, distinct from (and
# smaller than) a capital's building icon. An unupgraded outpost is a
# plain triangle; each upgrade swaps in its own silhouette instead (see
# _outpost_icon_points) so upgrades are readable by shape alone, at a
# glance, without relying on color (which is already spoken for by
# faction ownership).
OUTPOST_ICON_SIZE = 6

# Battle-contribution rectangle sizing (stacked when a hex is locked in a fight)
BATTLE_RECT_WIDTH = 40
BATTLE_RECT_HEIGHT = 13
BATTLE_RECT_GAP = 2


# ---------------------------------------------------------------------------
# Loading + reconstructing the log - two formats, both producing the same
# turn_checkpoints shape the rest of this file consumes:
#
# v1 (a since-removed dict-of-objects engine): a turn is
#   {"keyframe": [...], "deltas": {...}} - a sparse keyframe once per
#   turn, then a sparse diff per phase-step, reconstructed incrementally
#   on a running board. Built to keep file size proportional to how much
#   actually happens, for large/long games.
#
#   current engine (engine/turn.py's run_turn_and_log): a turn is
#   {"checkpoints": [...]} - one independent sparse snapshot per
#   checkpoint (not a diff against the previous one), each rebuilt fresh
#   from empty every time. Simpler (no incremental reconstruction, no
#   diffing to get wrong) at the cost of a bit more file size - a
#   deliberate tradeoff, not a mistake; see that module's docstring.
#
# Which format a given turn uses is detected per-turn (via which of
# "keyframe"/"checkpoints" is present), so a file could in principle mix
# them, though in practice a whole file will come from one engine version.
# ---------------------------------------------------------------------------

def _empty_hex():
    return {"city": None, "troops": None, "battle": None}


def _apply_delta(current, delta_entries):
    for e in delta_entries:
        coord = (e["q"], e["r"], e["s"])
        current[coord] = {"city": e["city"], "troops": e["troops"], "battle": e["battle"]}


def _checkpoint_from_sparse(terrain_map, sparse_entries):
    """v2-format checkpoint: starts from an all-empty board (independent
    of any other checkpoint) and overlays just the occupied/city/battle
    hexes - see the format comment above."""
    board = {coord: _empty_hex() for coord in terrain_map}
    _apply_delta(board, sparse_entries)
    return board


def load_game(path):
    """Returns (radius, num_factions, terrain_map, turn_checkpoints,
    turn_player_stats, turn_battles_by_hex).

    terrain_map: {(q,r,s): terrain_str}
    turn_checkpoints: list (one per turn) of lists of 10 dense board
    states, each a dict {(q,r,s): {"city","troops","battle"}}.
    turn_player_stats: list (one per turn) of lists of 10 dicts
    {faction: {"gold","resources","kill_xp","alive"}} - one per checkpoint,
    straight from the log (already a full snapshot each time, no
    reconstruction needed).
    turn_battles_by_hex: list (one per turn) of {(q,r,s): battle_event}
    - a battle belongs to the turn it resolved in, not to any specific
    checkpoint, so clicking a hex looks this up independent of which
    checkpoint is currently displayed.
    """
    with open(path, "r") as f:
        data = json.load(f)

    terrain_map = {}
    for key, terrain in data["terrain"].items():
        q, r, s = (int(x) for x in key.split("_"))
        terrain_map[(q, r, s)] = terrain

    current = {coord: _empty_hex() for coord in terrain_map}  # only used by the v1 (diff) path
    turn_checkpoints = []
    turn_player_stats = []
    turn_battles_by_hex = []

    for turn in data["turns"]:
        if "checkpoints" in turn:
            checkpoints = [_checkpoint_from_sparse(terrain_map, sparse) for sparse in turn["checkpoints"]]
        else:
            checkpoints = []

            _apply_delta(current, turn["keyframe"])
            checkpoints.append(dict(current))

            _apply_delta(current, turn["deltas"]["buy"])
            checkpoints.append(dict(current))

            for step_delta in turn["deltas"]["movement"]:
                _apply_delta(current, step_delta)
                checkpoints.append(dict(current))

            for step_delta in turn["deltas"]["cavalry"]:
                _apply_delta(current, step_delta)
                checkpoints.append(dict(current))

            _apply_delta(current, turn["deltas"]["battle"])
            checkpoints.append(dict(current))

        turn_checkpoints.append(checkpoints)

        # player_stats keys arrive as JSON strings ("0","1",...) - convert to int
        stats_per_checkpoint = [
            {int(f): v for f, v in snapshot.items()} for snapshot in turn["player_stats"]
        ]
        turn_player_stats.append(stats_per_checkpoint)

        battles_by_hex = {}
        for event in turn.get("battle_events", []):
            battles_by_hex[tuple(event["hex"])] = event
        turn_battles_by_hex.append(battles_by_hex)

    return (data["radius"], data["num_factions"], terrain_map,
            turn_checkpoints, turn_player_stats, turn_battles_by_hex)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _draw_count_text(surface, center, count, font):
    text_surf = font.render(str(count), True, TEXT_COLOR)
    text_rect = text_surf.get_rect(center=center)
    surface.blit(text_surf, text_rect)


def draw_circle_shape(surface, center, size, color, count, font):
    pygame.draw.circle(surface, color, center, size)
    pygame.draw.circle(surface, SHAPE_OUTLINE_COLOR, center, size, width=1)
    _draw_count_text(surface, center, count, font)


def draw_square_shape(surface, center, size, color, count, font):
    rect = pygame.Rect(0, 0, size * 2, size * 2)
    rect.center = center
    pygame.draw.rect(surface, color, rect)
    pygame.draw.rect(surface, SHAPE_OUTLINE_COLOR, rect, width=1)
    _draw_count_text(surface, center, count, font)


def draw_triangle_shape(surface, center, size, color, count, font):
    cx, cy = center
    points = [
        (cx, cy - size),
        (cx - size, cy + size * 0.7),
        (cx + size, cy + size * 0.7),
    ]
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, SHAPE_OUTLINE_COLOR, points, width=1)
    _draw_count_text(surface, (cx, cy + size * 0.15), count, font)


def draw_city_icon(surface, center, faction_color):
    """Small building glyph: square base + triangle roof, top-center of hex. No outlines.

    The roof's points are derived from the base rect's actual (rounded)
    pixel coordinates rather than the raw float center - this avoids a
    1px rounding seam between the two shapes where they meet.
    """
    cx, cy = center
    s = CITY_ICON_SIZE

    base_rect = pygame.Rect(0, 0, s * 4 / 3, s)
    base_rect.center = (round(cx), round(cy))
    pygame.draw.rect(surface, faction_color, base_rect)

    roof_points = [
        (base_rect.centerx, base_rect.top - s * 1),
        (base_rect.left, base_rect.top),
        (base_rect.right, base_rect.top),
    ]
    pygame.draw.polygon(surface, faction_color, roof_points)


def _star_points(cx, cy, outer, inner, num_points=5):
    """Alternating outer/inner vertices around a circle, first point
    straight up - the standard way to build an n-pointed star polygon."""
    points = []
    for i in range(num_points * 2):
        radius = outer if i % 2 == 0 else inner
        angle = math.pi / 2 + i * math.pi / num_points
        points.append((cx + radius * math.cos(angle), cy - radius * math.sin(angle)))
    return points


def _outpost_icon_points(cx, cy, s, upgrade):
    """Vertex list for an outpost's icon, shaped by its upgrade
    (None = plain triangle, matching the pre-upgrades look) so an
    upgraded outpost is distinguishable by SILHOUETTE, not just color -
    color is already spoken for by faction ownership, and a colored ring
    around one fixed shape read as too subtle at OUTPOST_ICON_SIZE."""
    if upgrade == "barracks":
        # Square - a blocky fortification.
        return [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)]
    if upgrade == "workshop":
        # Diamond - a distinct silhouette from both the triangle and the square.
        return [(cx, cy - s * 1.15), (cx + s * 1.15, cy), (cx, cy + s * 1.15), (cx - s * 1.15, cy)]
    if upgrade == "temple":
        # 5-pointed star - reads as "special"/sacred at a glance.
        return _star_points(cx, cy, outer=s * 1.15, inner=s * 0.45)
    return [(cx, cy - s), (cx - s, cy + s * 0.7), (cx + s, cy + s * 0.7)]


def draw_outpost_icon(surface, center, faction_color, upgrade=None):
    """Small outlined icon, top-center of hex - marks an outpost,
    distinct from a capital's building icon (draw_city_icon). Its shape
    depends on `upgrade` (None/"barracks"/"workshop"/"temple" - see
    _outpost_icon_points): an unupgraded outpost is the original plain
    triangle, each upgrade gets its own silhouette. Always drawn in the
    owning faction's color, same as before - only the shape changes."""
    cx, cy = center
    points = _outpost_icon_points(cx, cy, OUTPOST_ICON_SIZE, upgrade)
    pygame.draw.polygon(surface, faction_color, points)
    pygame.draw.polygon(surface, SHAPE_OUTLINE_COLOR, points, width=1)


def _shape_positions(present, cx, cy, size):
    spacing = size * 0.78
    n = len(present)

    if n == 3:
        positions = {}
        positions["archers"] = (cx, cy - size * 0.38)
        positions["infantry"] = (cx - spacing * 0.55, cy + size * 0.32)
        positions["cavalry"] = (cx + spacing * 0.55, cy + size * 0.32)
        return positions

    start_x = cx - spacing * (n - 1) / 2
    return {ttype: (start_x + i * spacing, cy + size * 0.05) for i, ttype in enumerate(present)}


def draw_troop_row(surface, center, size, troops, font):
    cx, cy = center
    faction_color = FACTION_COLORS[troops["faction"]]
    present = [t for t in TROOP_TYPES if troops.get(t, 0) > 0]
    if not present:
        return
    positions = _shape_positions(present, cx, cy, size)
    for ttype in present:
        shape_center = positions[ttype]
        count = troops[ttype]
        if ttype == "infantry":
            draw_circle_shape(surface, shape_center, SHAPE_SIZE, faction_color, count, font)
        elif ttype == "cavalry":
            draw_square_shape(surface, shape_center, SQUARE_SIZE, faction_color, count, font)
        elif ttype == "archers":
            draw_triangle_shape(surface, shape_center, SHAPE_SIZE, faction_color, count, font)


def draw_battle_rectangles(surface, center, contributions, battle_font):
    """One small faction-colored rectangle per contributing faction,
    stacked vertically with a bit of padding, each showing plain
    "infantry cavalry archers" numbers. Contributions are merged per
    faction before this is called. Overflow above/below the hex if
    there are many contributors is an accepted tradeoff, not handled."""
    cx, cy = center
    n = len(contributions)
    total_h = n * BATTLE_RECT_HEIGHT + (n - 1) * BATTLE_RECT_GAP
    start_y = cy - total_h / 2 + BATTLE_RECT_HEIGHT / 2

    for i, c in enumerate(contributions):
        y = start_y + i * (BATTLE_RECT_HEIGHT + BATTLE_RECT_GAP)
        rect = pygame.Rect(0, 0, BATTLE_RECT_WIDTH, BATTLE_RECT_HEIGHT)
        rect.center = (round(cx), round(y))
        color = FACTION_COLORS[c["faction"]]
        pygame.draw.rect(surface, color, rect)

        text = f'{c["infantry"]} {c["cavalry"]} {c["archers"]}'
        text_surf = battle_font.render(text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=rect.center)
        surface.blit(text_surf, text_rect)


def draw_hex(surface, center, size, terrain, hex_data, font, battle_font):
    points = [hex_corner(center, size, i) for i in range(6)]
    pygame.draw.polygon(surface, TERRAIN_COLORS[terrain], points)
    pygame.draw.polygon(surface, HEX_OUTLINE_COLOR, points, width=1)

    cx, cy = center

    if hex_data["city"] is not None:
        city = hex_data["city"]
        city_center = (cx, cy - size * 0.62 + 4)
        if city["is_capital"]:
            draw_city_icon(surface, city_center, FACTION_COLORS[city["faction"]])
        else:
            draw_outpost_icon(surface, city_center, FACTION_COLORS[city["faction"]], city.get("upgrade"))

    if hex_data["battle"] is not None:
        draw_battle_rectangles(surface, center, hex_data["battle"]["contributions"], battle_font)
    elif hex_data["troops"]:
        draw_troop_row(surface, center, size, hex_data["troops"], font)


# ---------------------------------------------------------------------------
# Slider widget (reused for both the turn slider and the checkpoint slider)
# ---------------------------------------------------------------------------

def compute_unit_counts(board_state, num_factions):
    """{faction: {"infantry","cavalry","archers"}} - sums troops sitting
    peacefully on the board AND units currently locked in a pending
    battle (both are equally "alive", just in different data shapes)."""
    counts = {f: {"infantry": 0, "cavalry": 0, "archers": 0} for f in range(num_factions)}
    for data in board_state.values():
        troops = data.get("troops")
        if troops:
            f = troops["faction"]
            for ut in TROOP_TYPES:
                counts[f][ut] += troops.get(ut, 0)
        battle = data.get("battle")
        if battle:
            for c in battle["contributions"]:
                f = c["faction"]
                for ut in TROOP_TYPES:
                    counts[f][ut] += c.get(ut, 0)
    return counts


def draw_sidebar(surface, x0, y0, height, num_factions, player_stats, unit_counts, header_font, row_font):
    """Always-visible per-faction panel: color swatch, alive/dead state,
    gold, kill-XP, resources, and each unit type's count separately."""
    pygame.draw.rect(surface, SIDEBAR_BG_COLOR, pygame.Rect(x0, y0, SIDEBAR_WIDTH, height))

    row_h = height / num_factions
    pad = 14

    for faction in range(num_factions):
        row_top = y0 + faction * row_h
        stats = player_stats.get(faction, {})
        alive = stats.get("alive", True)
        text_color = TEXT_COLOR if alive else SIDEBAR_DEAD_TEXT_COLOR

        if faction > 0:
            pygame.draw.line(surface, SIDEBAR_DIVIDER_COLOR,
                              (x0 + pad, row_top), (x0 + SIDEBAR_WIDTH - pad, row_top), width=1)

        # swatch always shows the faction's real color, even when eliminated -
        # dimming it too would make a naturally-grey faction (index 7) visually
        # indistinguishable from the "eliminated" state. Only the text dims.
        swatch = pygame.Rect(0, 0, 22, 22)
        swatch.topleft = (x0 + pad, row_top + 10)
        pygame.draw.rect(surface, FACTION_COLORS[faction], swatch)
        if not alive:
            pygame.draw.line(surface, SIDEBAR_DEAD_TEXT_COLOR, swatch.topleft,
                              (swatch.right, swatch.bottom), width=2)
            pygame.draw.line(surface, SIDEBAR_DEAD_TEXT_COLOR, (swatch.left, swatch.bottom),
                              (swatch.right, swatch.top), width=2)

        header = f"Faction {faction}" + ("" if alive else "  (eliminated)")
        header_surf = header_font.render(header, True, text_color)
        surface.blit(header_surf, (x0 + pad + 32, row_top + 8))

        gold = stats.get("gold", 0)
        kill_xp = stats.get("kill_xp", 0)
        resources = stats.get("resources", {})
        counts = unit_counts.get(faction, {"infantry": 0, "cavalry": 0, "archers": 0})

        line1 = f"Gold: {gold}    Kill XP: {kill_xp}"
        line2 = f"Infantry: {counts['infantry']}   Cavalry: {counts['cavalry']}   Archers: {counts['archers']}"
        line3 = (
            f"Wood: {resources.get('wood', 0)}  Iron: {resources.get('iron', 0)}  "
            f"Clay: {resources.get('clay', 0)}  Fish: {resources.get('fish', 0)}"
        )

        line1_surf = row_font.render(line1, True, text_color)
        line2_surf = row_font.render(line2, True, text_color)
        line3_surf = row_font.render(line3, True, text_color)
        surface.blit(line1_surf, (x0 + pad + 32, row_top + 34))
        surface.blit(line2_surf, (x0 + pad + 32, row_top + 54))
        surface.blit(line3_surf, (x0 + pad + 32, row_top + 74))


def _empty_totals():
    return {"infantry": 0, "cavalry": 0, "archers": 0}


def compute_battle_table(battle_event):
    """Turns one battle_event log into a table: {"factions": [...], "rows": [...]}.
    Each row is {"label": str, "cells": {faction: {"text": str, "totals": {...}}}}.
    Rows are: Archer Phase, Round 1, Round 2, ..., Result. Totals are a
    running per-faction unit count, ticking down (or occasionally back
    up, from a dismount) row by row - so scanning down one column shows
    that faction's strength over the course of the fight."""
    totals = {}
    for c in battle_event["contributions_start"]:
        t = totals.setdefault(c["faction"], _empty_totals())
        for ut in TROOP_TYPES:
            t[ut] += c.get(ut, 0)
    factions = sorted(totals.keys())

    rows = []

    # --- Archer Phase row ---
    dealt_by_faction = {f: 0 for f in factions}
    for entry in battle_event["archer_phase"]:
        totals[entry["faction"]][entry["unit_type"]] -= entry["count"]
        dealt_by_faction[entry["killer"]] = dealt_by_faction.get(entry["killer"], 0) + entry["count"]

    archer_cells = {}
    for f in factions:
        dealt = dealt_by_faction.get(f, 0)
        text = f"Dealt {dealt} kill{'s' if dealt != 1 else ''}" if dealt > 0 else "-"
        archer_cells[f] = {"text": text, "totals": dict(totals[f])}
    rows.append({"label": "Archer Phase", "cells": archer_cells})

    # --- one row per round ---
    for round_idx, round_log in enumerate(battle_event["rounds"], start=1):
        cells = {}
        deaths_by_faction = {}
        for d in round_log["deaths"]:
            deaths_by_faction.setdefault(d["faction"], []).append(d)
        dismounts_by_faction = {}
        for d in round_log["dismounts"]:
            dismounts_by_faction.setdefault(d["faction"], []).append(d)

        for f in factions:
            submitted = round_log["target_choices_submitted"].get(f)
            resolved = round_log["resolved_targets"].get(f)
            roll = round_log["rolls"].get(f)
            kills_dealt = round_log["kills_dealt"].get(f, 0)

            lines = []
            if submitted is None:
                lines.append("-")
            elif resolved is not None:
                roll_text = f" (roll {roll})" if roll is not None else ""
                lines.append(f"-> Faction {resolved}{roll_text}")
                if kills_dealt:
                    lines.append(f"{kills_dealt} kill{'s' if kills_dealt != 1 else ''} dealt")
            else:
                lines.append(f"wanted Faction {submitted}, outnumbered")

            for d in deaths_by_faction.get(f, []):
                totals[f][d["unit_type"]] -= d["count"]
                lines.append(f"lost {d['count']} {d['unit_type']}")

            for d in dismounts_by_faction.get(f, []):
                if d["success"]:
                    totals[f]["infantry"] += 1
                    lines.append("cavalry dismounted -> +1 infantry")
                elif d.get("reason") == "cap":
                    lines.append("dismount blocked (unit cap)")
                else:
                    lines.append("dismount roll failed")

            cells[f] = {"text": "\n".join(lines), "totals": dict(totals[f])}

        rows.append({"label": f"Round {round_idx}", "cells": cells})

    # --- Result row ---
    winner = battle_event["winner"]
    result_cells = {}
    for f in factions:
        if winner is None:
            text = "Mutual wipeout"
        elif f == winner:
            text = "WINNER"
        else:
            text = "Defeated"
        result_cells[f] = {"text": text, "totals": dict(totals[f])}
    rows.append({"label": "Result", "cells": result_cells})

    return {"factions": factions, "rows": rows}


def find_hex_at_pixel(centers, hex_size, pos):
    """Nearest hex to a click position, or None if nothing's close enough
    (guards against picking a far-off hex when clicking empty space)."""
    px, py = pos
    best_coord, best_dist_sq = None, None
    for coord, (cx, cy) in centers.items():
        d = (cx - px) ** 2 + (cy - py) ** 2
        if best_dist_sq is None or d < best_dist_sq:
            best_dist_sq, best_coord = d, coord
    if best_dist_sq is not None and best_dist_sq <= (hex_size * 1.05) ** 2:
        return best_coord
    return None


def draw_battle_popup(surface, window_w, window_h, table, header_font, cell_font, label_font):
    factions = table["factions"]
    rows = table["rows"]
    num_factions = max(1, len(factions))

    popup_w = BATTLE_LABEL_COL_WIDTH + BATTLE_COLUMN_WIDTH * num_factions + BATTLE_POPUP_PADDING * 2
    available_h = window_h - 40
    row_h = min(BATTLE_ROW_HEIGHT, max(28, (available_h - BATTLE_HEADER_HEIGHT - BATTLE_POPUP_PADDING * 2) / max(1, len(rows))))
    popup_h = min(available_h, BATTLE_HEADER_HEIGHT + row_h * len(rows) + BATTLE_POPUP_PADDING * 2)

    popup_x = max(0, (window_w - popup_w) // 2)
    popup_y = max(0, (window_h - popup_h) // 2)

    popup_rect = pygame.Rect(popup_x, popup_y, popup_w, popup_h)
    pygame.draw.rect(surface, BATTLE_POPUP_BG_COLOR, popup_rect)
    pygame.draw.rect(surface, BATTLE_POPUP_BORDER_COLOR, popup_rect, width=2)

    col_x0 = popup_x + BATTLE_POPUP_PADDING + BATTLE_LABEL_COL_WIDTH
    header_y = popup_y + BATTLE_POPUP_PADDING

    for i, f in enumerate(factions):
        cx = col_x0 + i * BATTLE_COLUMN_WIDTH
        swatch = pygame.Rect(cx, header_y, 16, 16)
        pygame.draw.rect(surface, FACTION_COLORS[f], swatch)
        text_surf = header_font.render(f"Faction {f}", True, BATTLE_CELL_TEXT_COLOR)
        surface.blit(text_surf, (cx + 22, header_y))

    body_y0 = header_y + BATTLE_HEADER_HEIGHT

    for r, row in enumerate(rows):
        row_top = body_y0 + r * row_h
        if r > 0:
            pygame.draw.line(surface, BATTLE_POPUP_BORDER_COLOR,
                              (popup_x + 4, row_top), (popup_x + popup_w - 4, row_top), width=1)

        label_surf = label_font.render(row["label"], True, BATTLE_CELL_TEXT_COLOR)
        surface.blit(label_surf, (popup_x + BATTLE_POPUP_PADDING, row_top + 6))

        for i, f in enumerate(factions):
            cx = col_x0 + i * BATTLE_COLUMN_WIDTH
            cell = row["cells"].get(f)
            if not cell:
                continue
            for line_idx, line in enumerate(cell["text"].split("\n")):
                line_surf = cell_font.render(line, True, BATTLE_CELL_TEXT_COLOR)
                surface.blit(line_surf, (cx, row_top + 6 + line_idx * 15))

            t = cell["totals"]
            totals_text = f"{t['infantry']} {t['cavalry']} {t['archers']}"
            totals_surf = cell_font.render(totals_text, True, BATTLE_CELL_DIM_COLOR)
            surface.blit(totals_surf, (cx, row_top + row_h - 18))

    hint_surf = label_font.render("(click anywhere to close)", True, BATTLE_CELL_DIM_COLOR)
    surface.blit(hint_surf, (popup_x + BATTLE_POPUP_PADDING, popup_y + popup_h - 22))


class ScoreTicker:
    """Always-visible victory-points readout, a vertical panel with one
    column per faction (so close/tied scores don't just overlap into an
    unreadable blob): each column is its own 0-VP_TO_WIN vertical track
    (0 at the bottom, VP_TO_WIN at the top), with a dot marking that
    faction's current score and a swatch + numeric score below the
    track. Purely a display (no drag/click handling, unlike Slider) -
    just call draw() with whatever player_stats dict is current."""

    def __init__(self, x0, y0, height, num_factions, vp_to_win=VP_TO_WIN):
        self.x0 = x0
        self.y0 = y0
        self.height = height
        self.num_factions = num_factions
        self.vp_to_win = vp_to_win
        self.track_y1 = y0 + SCORE_TICKER_TOP_PADDING       # VP_TO_WIN (top)
        self.track_y2 = y0 + height - SCORE_TICKER_BOTTOM_PADDING  # 0 (bottom)

    @property
    def width(self):
        return SCORE_TICKER_SIDE_MARGIN * 2 + self.num_factions * SCORE_TICKER_COLUMN_WIDTH

    def _column_x(self, faction):
        return self.x0 + SCORE_TICKER_SIDE_MARGIN + faction * SCORE_TICKER_COLUMN_WIDTH + SCORE_TICKER_COLUMN_WIDTH / 2

    def draw(self, surface, player_stats, label_font, row_font):
        title_surf = label_font.render("Victory", True, SLIDER_LABEL_COLOR)
        surface.blit(title_surf, title_surf.get_rect(centerx=self.x0 + self.width / 2, top=self.y0))
        win_surf = row_font.render(f"Points (first to {self.vp_to_win})", True, SLIDER_LABEL_COLOR)
        surface.blit(win_surf, win_surf.get_rect(centerx=self.x0 + self.width / 2, top=self.y0 + 16))

        pygame.draw.line(surface, SCORE_TICKER_WIN_LINE_COLOR,
                          (self.x0 + SCORE_TICKER_SIDE_MARGIN / 2, self.track_y1),
                          (self.x0 + self.width - SCORE_TICKER_SIDE_MARGIN / 2, self.track_y1), width=2)

        for faction in range(self.num_factions):
            x = self._column_x(faction)
            color = FACTION_COLORS[faction]

            pygame.draw.line(surface, SCORE_TICKER_TRACK_COLOR, (x, self.track_y1), (x, self.track_y2), width=2)

            vp = player_stats.get(faction, {}).get("victory_points", 0)
            frac = max(0.0, min(1.0, vp / self.vp_to_win))
            handle_y = self.track_y2 - frac * (self.track_y2 - self.track_y1)
            pygame.draw.circle(surface, color, (int(x), int(handle_y)), SCORE_TICKER_HANDLE_RADIUS)
            pygame.draw.circle(surface, SHAPE_OUTLINE_COLOR, (int(x), int(handle_y)), SCORE_TICKER_HANDLE_RADIUS, width=1)

            swatch = pygame.Rect(0, 0, 12, 12)
            swatch.center = (int(x), int(self.track_y2 + 16))
            pygame.draw.rect(surface, color, swatch)

            score_surf = row_font.render(str(vp), True, TEXT_COLOR)
            surface.blit(score_surf, score_surf.get_rect(centerx=int(x), top=int(self.track_y2 + 28)))


class Slider:
    """A click/drag scrubber along one horizontal track at a fixed y.
    `label_fn(index) -> str` builds the label text shown above the track,
    so the turn slider and checkpoint slider can each show different info."""

    def __init__(self, window_w, track_y, num_states, label_fn, x_pad=40):
        self.num_states = num_states
        self.track_x1 = x_pad
        self.track_x2 = window_w - x_pad
        self.track_width = self.track_x2 - self.track_x1
        self.track_y = track_y
        self.label_fn = label_fn

    def _x_for_index(self, index):
        if self.num_states <= 1:
            return self.track_x1
        t = index / (self.num_states - 1)
        return self.track_x1 + t * self.track_width

    def index_at(self, mouse_x):
        t = (mouse_x - self.track_x1) / self.track_width
        t = max(0.0, min(1.0, t))
        return round(t * (self.num_states - 1))

    def hit_test(self, pos):
        x, y = pos
        return (self.track_x1 - 15 <= x <= self.track_x2 + 15
                and self.track_y - 15 <= y <= self.track_y + 15)

    def draw(self, surface, current_index, font):
        pygame.draw.line(surface, SLIDER_TRACK_COLOR,
                          (self.track_x1, self.track_y), (self.track_x2, self.track_y), width=4)

        handle_x = self._x_for_index(current_index)
        pygame.draw.line(surface, SLIDER_FILL_COLOR,
                          (self.track_x1, self.track_y), (handle_x, self.track_y), width=4)

        for i in range(self.num_states):
            tx = self._x_for_index(i)
            pygame.draw.circle(surface, SLIDER_TRACK_COLOR, (int(tx), self.track_y), 3)

        pygame.draw.circle(surface, SLIDER_HANDLE_COLOR, (int(handle_x), self.track_y), SLIDER_HANDLE_RADIUS)
        pygame.draw.circle(surface, SHAPE_OUTLINE_COLOR, (int(handle_x), self.track_y), SLIDER_HANDLE_RADIUS, width=1)

        label = self.label_fn(current_index)
        text_surf = font.render(label, True, SLIDER_LABEL_COLOR)
        text_rect = text_surf.get_rect(center=((self.track_x1 + self.track_x2) // 2, self.track_y - 18))
        surface.blit(text_surf, text_rect)


def main():
    if not os.path.exists(STATE_FILE):
        print(f"Could not find {STATE_FILE}")
        print("Run run_game_and_log.py first to create it.")
        sys.exit(1)

    radius, num_factions, terrain_map, turn_checkpoints, turn_player_stats, turn_battles_by_hex = load_game(STATE_FILE)
    num_turns = len(turn_checkpoints)

    board_area_h = WINDOW_H - SLIDER_BAND_HEIGHT
    score_ticker = ScoreTicker(WINDOW_W, y0=0, height=board_area_h, num_factions=num_factions)
    window_w = WINDOW_W + score_ticker.width

    pygame.init()
    screen = pygame.display.set_mode((window_w, WINDOW_H))
    pygame.display.set_caption("Hex Board Visualizer")
    clock = pygame.time.Clock()

    size = compute_hex_size(radius, BOARD_AREA_W, board_area_h, MARGIN)
    font = pygame.font.SysFont("arial", max(9, int(SHAPE_SIZE * 1.1)), bold=True)
    battle_font = pygame.font.SysFont("arial", 10, bold=True)
    label_font = pygame.font.SysFont("arial", 16)
    sidebar_header_font = pygame.font.SysFont("arial", 15, bold=True)
    sidebar_row_font = pygame.font.SysFont("arial", 14)
    battle_header_font = pygame.font.SysFont("arial", 15, bold=True)
    battle_cell_font = pygame.font.SysFont("arial", 12)
    battle_label_font = pygame.font.SysFont("arial", 13, bold=True)

    raw_centers = {coord: hex_to_pixel(coord[0], coord[1], size) for coord in terrain_map}
    xs = [p[0] for p in raw_centers.values()]
    ys = [p[1] for p in raw_centers.values()]
    board_cx = (min(xs) + max(xs)) / 2
    board_cy = (min(ys) + max(ys)) / 2
    offset_x = BOARD_AREA_W / 2 - board_cx
    offset_y = board_area_h / 2 - board_cy
    centers = {coord: (p[0] + offset_x, p[1] + offset_y) for coord, p in raw_centers.items()}

    # Single continuous timeline: every turn contributes the same
    # len(CHECKPOINT_LABELS) checkpoints (run_turn_and_log always builds
    # that many, one per phase), so a flat step index maps to
    # (turn, checkpoint) via divmod - no separate turn/checkpoint sliders
    # to keep in sync.
    steps_per_turn = len(CHECKPOINT_LABELS)
    total_steps = num_turns * steps_per_turn

    def label_fn(i):
        turn_idx, checkpoint_idx = divmod(i, steps_per_turn)
        return (f"Turn {turn_idx + 1} / {num_turns}   "
                f"Phase: {CHECKPOINT_LABELS[checkpoint_idx]}   (drag, or use ←/→)")

    timeline_slider = Slider(window_w, WINDOW_H - 28, total_steps, label_fn=label_fn)

    current_step = 0
    open_battle_table = None  # None, or a table dict from compute_battle_table

    running = True
    dragging = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if open_battle_table is not None:
                        open_battle_table = None
                    else:
                        running = False
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    current_step = max(0, current_step - 1)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    current_step = min(total_steps - 1, current_step + 1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if open_battle_table is not None:
                    # any click closes the popup, per the "(click anywhere to close)" hint
                    open_battle_table = None
                elif timeline_slider.hit_test(event.pos):
                    dragging = True
                    current_step = timeline_slider.index_at(event.pos[0])
                elif event.pos[0] < BOARD_AREA_W and event.pos[1] < board_area_h:
                    clicked_hex = find_hex_at_pixel(centers, size, event.pos)
                    if clicked_hex is not None:
                        current_turn, _ = divmod(current_step, steps_per_turn)
                        battle_event = turn_battles_by_hex[current_turn].get(clicked_hex)
                        if battle_event is not None:
                            open_battle_table = compute_battle_table(battle_event)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                current_step = timeline_slider.index_at(event.pos[0])

        current_turn, current_checkpoint = divmod(current_step, steps_per_turn)
        board_state = turn_checkpoints[current_turn][current_checkpoint]
        player_stats = turn_player_stats[current_turn][current_checkpoint]
        unit_counts = compute_unit_counts(board_state, num_factions)

        screen.fill(BG_COLOR)
        score_ticker.draw(screen, player_stats, label_font, sidebar_row_font)

        for coord, terrain in terrain_map.items():
            draw_hex(screen, centers[coord], size, terrain, board_state[coord], font, battle_font)

        draw_sidebar(screen, BOARD_AREA_W, 0, board_area_h, num_factions,
                     player_stats, unit_counts, sidebar_header_font, sidebar_row_font)

        timeline_slider.draw(screen, current_step, label_font)

        if open_battle_table is not None:
            draw_battle_popup(screen, window_w, WINDOW_H, open_battle_table,
                               battle_header_font, battle_cell_font, battle_label_font)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()