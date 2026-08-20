"""
Hexagon-shaped hex grid visualizer (pygame).

Reads board_state.json (the diff-based format written by
run_game_and_log.py / engine/turn.py:run_turn_and_log) and reconstructs
every turn's 10 checkpoint states (start-of-turn keyframe, then after
buy, each of the 3 movement steps, each of the 4 cavalry steps, and
after battle) by replaying that turn's sparse deltas on top of a
running full board. Two sliders let you scrub by turn and by
checkpoint-within-turn independently.

Each hex can show:
  - terrain (washed-out fill color, background context only)
  - a city icon (top-center, square+triangle "building" glyph, faction-colored)
  - EITHER a peaceful troop row (center, up to 3 shapes: circle=infantry,
    square=cavalry, triangle=archers, each showing its count) OR, if the
    hex is currently locked in a pending battle, a stack of small
    faction-colored rectangles - one per faction contributing to the
    fight, each showing "infantry cavalry archers" as plain numbers.

All colors are defined as named constants up top so they're easy to
retune without touching any drawing logic.
"""

import json
import os
import sys
import pygame

from hex_common import (
    hex_to_pixel, hex_corner, compute_hex_size,
)

try:
    from engine.turn import CHECKPOINT_LABELS
except ImportError:
    CHECKPOINT_LABELS = ["Start", "Buy", "Move 1", "Move 2", "Move 3",
                          "Cav 1", "Cav 2", "Cav 3", "Cav 4", "Battle"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOW_W, WINDOW_H = 1000, 1010
MARGIN = 20  # pixels of padding around the board

SLIDER_BAND_HEIGHT = 110  # reserved band at the bottom for both sliders
SLIDER_TRACK_COLOR = (70, 70, 78)
SLIDER_FILL_COLOR = (120, 160, 220)
SLIDER_HANDLE_COLOR = (230, 230, 235)
SLIDER_HANDLE_RADIUS = 9
SLIDER_LABEL_COLOR = (220, 220, 225)

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

# Battle-contribution rectangle sizing (stacked when a hex is locked in a fight)
BATTLE_RECT_WIDTH = 40
BATTLE_RECT_HEIGHT = 13
BATTLE_RECT_GAP = 2


# ---------------------------------------------------------------------------
# Loading + reconstructing the diff-based log
# ---------------------------------------------------------------------------

def _empty_hex():
    return {"city": None, "troops": None, "battle": None}


def _apply_delta(current, delta_entries):
    for e in delta_entries:
        coord = (e["q"], e["r"], e["s"])
        current[coord] = {"city": e["city"], "troops": e["troops"], "battle": e["battle"]}


def load_game(path):
    """Returns (radius, num_factions, terrain_map, turn_checkpoints).

    terrain_map: {(q,r,s): terrain_str}
    turn_checkpoints: list (one per turn) of lists of 10 dense board
    states, each a dict {(q,r,s): {"city","troops","battle"}}. Built by
    replaying each turn's sparse keyframe + deltas on a running board.
    """
    with open(path, "r") as f:
        data = json.load(f)

    terrain_map = {}
    for key, terrain in data["terrain"].items():
        q, r, s = (int(x) for x in key.split("_"))
        terrain_map[(q, r, s)] = terrain

    current = {coord: _empty_hex() for coord in terrain_map}
    turn_checkpoints = []

    for turn in data["turns"]:
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

    return data["radius"], data["num_factions"], terrain_map, turn_checkpoints


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
        city_center = (cx, cy - size * 0.62 + 4)
        draw_city_icon(surface, city_center, FACTION_COLORS[hex_data["city"]])

    if hex_data["battle"] is not None:
        draw_battle_rectangles(surface, center, hex_data["battle"]["contributions"], battle_font)
    elif hex_data["troops"]:
        draw_troop_row(surface, center, size, hex_data["troops"], font)


# ---------------------------------------------------------------------------
# Slider widget (reused for both the turn slider and the checkpoint slider)
# ---------------------------------------------------------------------------

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

    radius, num_factions, terrain_map, turn_checkpoints = load_game(STATE_FILE)
    num_turns = len(turn_checkpoints)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Hex Board Visualizer")
    clock = pygame.time.Clock()

    board_area_h = WINDOW_H - SLIDER_BAND_HEIGHT
    size = compute_hex_size(radius, WINDOW_W, board_area_h, MARGIN)
    font = pygame.font.SysFont("arial", max(9, int(SHAPE_SIZE * 1.1)), bold=True)
    battle_font = pygame.font.SysFont("arial", 10, bold=True)
    label_font = pygame.font.SysFont("arial", 16)

    raw_centers = {coord: hex_to_pixel(coord[0], coord[1], size) for coord in terrain_map}
    xs = [p[0] for p in raw_centers.values()]
    ys = [p[1] for p in raw_centers.values()]
    board_cx = (min(xs) + max(xs)) / 2
    board_cy = (min(ys) + max(ys)) / 2
    offset_x = WINDOW_W / 2 - board_cx
    offset_y = board_area_h / 2 - board_cy
    centers = {coord: (p[0] + offset_x, p[1] + offset_y) for coord, p in raw_centers.items()}

    turn_slider = Slider(
        WINDOW_W, WINDOW_H - 78, num_turns,
        label_fn=lambda i: f"Turn {i + 1} / {num_turns}   (drag, or use ↑/↓)",
    )
    checkpoint_slider = Slider(
        WINDOW_W, WINDOW_H - 28, len(CHECKPOINT_LABELS),
        label_fn=lambda i: f"Phase: {CHECKPOINT_LABELS[i]}   (drag, or use ←/→)",
    )

    current_turn = 0
    current_checkpoint = 0

    running = True
    dragging = None  # None | "turn" | "checkpoint"
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    current_checkpoint = max(0, current_checkpoint - 1)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    current_checkpoint = min(len(CHECKPOINT_LABELS) - 1, current_checkpoint + 1)
                elif event.key in (pygame.K_UP, pygame.K_w):
                    current_turn = min(num_turns - 1, current_turn + 1)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    current_turn = max(0, current_turn - 1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if turn_slider.hit_test(event.pos):
                    dragging = "turn"
                    current_turn = turn_slider.index_at(event.pos[0])
                elif checkpoint_slider.hit_test(event.pos):
                    dragging = "checkpoint"
                    current_checkpoint = checkpoint_slider.index_at(event.pos[0])
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = None
            elif event.type == pygame.MOUSEMOTION and dragging:
                if dragging == "turn":
                    current_turn = turn_slider.index_at(event.pos[0])
                elif dragging == "checkpoint":
                    current_checkpoint = checkpoint_slider.index_at(event.pos[0])

        board_state = turn_checkpoints[current_turn][current_checkpoint]

        screen.fill(BG_COLOR)
        for coord, terrain in terrain_map.items():
            draw_hex(screen, centers[coord], size, terrain, board_state[coord], font, battle_font)

        turn_slider.draw(screen, current_turn, label_font)
        checkpoint_slider.draw(screen, current_checkpoint, label_font)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()