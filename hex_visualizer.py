"""
Hexagon-shaped hex grid visualizer (pygame).

Reads board_state.json (written by map_generator.py, expected to sit
in the same directory as this file) and renders it.

Each hex can show:
  - terrain (washed-out fill color, background context only)
  - a city icon (top-center, square+triangle "building" glyph, faction-colored)
  - a troop row (center, up to 3 shapes: circle=infantry, square=cavalry,
    triangle=archers), faction-colored, each shape showing its count

All colors are defined as named constants up top so they're easy to
retune without touching any drawing logic.

Next step (not yet implemented): board_state.json will become a list
of states instead of a single board, and this file will grow a
timeline/scrubber to step through them.
"""

import json
import os
import sys
import pygame

from hex_common import (
    hex_to_pixel, hex_corner, compute_hex_size,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOW_W, WINDOW_H = 1000, 960
MARGIN = 20  # pixels of padding around the board

SLIDER_HEIGHT = 60  # reserved band at the bottom of the window for the timeline
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
# (Dark2/Set1-style), used consistently for troop shapes AND city icons.
# Index = faction id (0-7). Edit freely / reorder as needed.
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
CITY_ICON_SIZE = 7  # half-width of the square base


# ---------------------------------------------------------------------------
# Loading board state
# ---------------------------------------------------------------------------

def load_states(path):
    """Read board_state.json and return (radius, num_factions, list_of_board_dicts).

    Each board dict is keyed by (q, r, s) cube-coord tuples, matching
    the in-memory shape the generator uses internally.
    """
    with open(path, "r") as f:
        data = json.load(f)

    states = []
    for hexes in data["states"]:
        board = {}
        for entry in hexes:
            coord = (entry["q"], entry["r"], entry["s"])
            board[coord] = {
                "terrain": entry["terrain"],
                "city": entry["city"],
                "troops": entry["troops"],
            }
        states.append(board)
    return data["radius"], data["num_factions"], states


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
    # nudge text down slightly since the triangle's visual center sits high
    _draw_count_text(surface, (cx, cy + size * 0.15), count, font)


def draw_city_icon(surface, center, faction_color):
    """Small building glyph: square base + triangle roof, top-center of hex. No outlines.

    The roof's points are derived from the base rect's actual (rounded)
    pixel coordinates rather than the raw float center - this avoids a
    1px rounding seam between the two shapes where they meet.
    """
    cx, cy = center
    s = CITY_ICON_SIZE

    base_rect = pygame.Rect(0, 0, s * 1.6, s)
    base_rect.center = (round(cx), round(cy))
    pygame.draw.rect(surface, faction_color, base_rect)

    roof_points = [
        (base_rect.centerx, base_rect.top - s * 1.4),
        (base_rect.left, base_rect.top - s * 0.5),
        (base_rect.right, base_rect.top - s * 0.5),
    ]
    pygame.draw.polygon(surface, faction_color, roof_points)


def _shape_positions(present, cx, cy, size):
    """
    Return {troop_type: (x, y)} for the shapes present in this hex.
    - 1 or 2 types: simple horizontal row, centered.
    - all 3 types: triangle formation, archers on top (matches the
      triangle icon sitting naturally at the apex).
    """
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


def draw_hex(surface, center, size, hex_data, font):
    points = [hex_corner(center, size, i) for i in range(6)]
    pygame.draw.polygon(surface, TERRAIN_COLORS[hex_data["terrain"]], points)
    pygame.draw.polygon(surface, HEX_OUTLINE_COLOR, points, width=1)

    cx, cy = center

    if hex_data["city"] is not None:
        city_center = (cx, cy - size * 0.62 + 4)
        draw_city_icon(surface, city_center, FACTION_COLORS[hex_data["city"]])

    troops = hex_data["troops"]
    if troops:
        faction_color = FACTION_COLORS[troops["faction"]]
        present = [t for t in TROOP_TYPES if troops.get(t, 0) > 0]
        if present:
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


class Slider:
    """A click/drag timeline scrubber pinned to the bottom of the window.

    Draws a horizontal track spanning most of the window width, a filled
    portion up to the current state, a draggable handle, and a "state
    X / N" label. `index_at(x)` converts a mouse x-coordinate into the
    nearest state index, snapped to the discrete number of states.
    """

    def __init__(self, window_w, window_h, band_height, num_states):
        self.num_states = num_states
        pad = 40
        self.track_y = window_h - band_height // 2 - 8
        self.track_x1 = pad
        self.track_x2 = window_w - pad
        self.track_width = self.track_x2 - self.track_x1
        self.label_y = window_h - band_height + 12

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

        # tick marks for each discrete state
        for i in range(self.num_states):
            tx = self._x_for_index(i)
            pygame.draw.circle(surface, SLIDER_TRACK_COLOR, (int(tx), self.track_y), 3)

        pygame.draw.circle(surface, SLIDER_HANDLE_COLOR, (int(handle_x), self.track_y), SLIDER_HANDLE_RADIUS)
        pygame.draw.circle(surface, SHAPE_OUTLINE_COLOR, (int(handle_x), self.track_y), SLIDER_HANDLE_RADIUS, width=1)

        label = f"State {current_index + 1} / {self.num_states}   (drag the handle, or use ←/→)"
        text_surf = font.render(label, True, SLIDER_LABEL_COLOR)
        text_rect = text_surf.get_rect(center=((self.track_x1 + self.track_x2) // 2, self.label_y))
        surface.blit(text_surf, text_rect)


def main():
    if not os.path.exists(STATE_FILE):
        print(f"Could not find {STATE_FILE}")
        print("Run map_generator.py first to create it.")
        sys.exit(1)

    radius, num_factions, states = load_states(STATE_FILE)
    num_states = len(states)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Hex Board Visualizer")
    clock = pygame.time.Clock()

    board_area_h = WINDOW_H - SLIDER_HEIGHT
    size = compute_hex_size(radius, WINDOW_W, board_area_h, MARGIN)
    font = pygame.font.SysFont("arial", max(9, int(SHAPE_SIZE * 1.1)), bold=True)
    label_font = pygame.font.SysFont("arial", 16)

    # centers are computed once (terrain/city/troop *positions* never move,
    # only the data shown at each hex changes between states)
    any_board = states[0]
    raw_centers = {coord: hex_to_pixel(coord[0], coord[1], size) for coord in any_board}
    xs = [p[0] for p in raw_centers.values()]
    ys = [p[1] for p in raw_centers.values()]
    board_cx = (min(xs) + max(xs)) / 2
    board_cy = (min(ys) + max(ys)) / 2
    offset_x = WINDOW_W / 2 - board_cx
    offset_y = board_area_h / 2 - board_cy
    centers = {coord: (p[0] + offset_x, p[1] + offset_y) for coord, p in raw_centers.items()}

    slider = Slider(WINDOW_W, WINDOW_H, SLIDER_HEIGHT, num_states)
    current_state = 0

    running = True
    dragging = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    current_state = max(0, current_state - 1)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    current_state = min(num_states - 1, current_state + 1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if slider.hit_test(event.pos):
                    dragging = True
                    current_state = slider.index_at(event.pos[0])
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                current_state = slider.index_at(event.pos[0])

        screen.fill(BG_COLOR)
        for coord, data in states[current_state].items():
            draw_hex(screen, centers[coord], size, data, font)

        slider.draw(screen, current_state, label_font)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()