"""
Step-by-step visualizer for engine/setup.py's round-based terrain
generation (pygame).

Reads terrain_gen_log.json (written by run.py), which is a flat, ordered
list of every individual hex placement made while generating the board.
A single slider scrubs through that list one placement at a time; hexes
not yet reached are drawn blank. The label above the slider shows which
round (and terrain type) the current placement belongs to, so you can
watch each round's blob grow and see where the next round picks up.

A sidebar (same layout pattern as hex.py's player panel) shows each
terrain type's remaining count in the generation "bag" (engine/setup.py's
BAG_COUNTS), ticking down as the slider advances, plus a checkbox to
render lake/mountain hexes as plain background instead of their terrain
color - handy for eyeballing the passable-terrain shape mountains and
lakes carve out.
"""

import collections
import json
import os
import sys
import pygame

from hex_common import hex_to_pixel, hex_corner, compute_hex_size, cube_hexes_in_radius
from hex import (
    TERRAIN_COLORS, HEX_OUTLINE_COLOR, BG_COLOR, TEXT_COLOR,
    SIDEBAR_BG_COLOR, SIDEBAR_DIVIDER_COLOR, Slider,
)
from engine.setup import BAG_COUNTS

BOARD_AREA_W = 900
SIDEBAR_WIDTH = 260
WINDOW_W = BOARD_AREA_W + SIDEBAR_WIDTH
WINDOW_H = 900
MARGIN = 20
SLIDER_BAND_HEIGHT = 70

UNSET_COLOR = (40, 40, 46)  # not-yet-placed hex

CHECKBOX_SIZE = 18
CHECKBOX_BOX_COLOR = (90, 90, 100)

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terrain_gen_log.json")

_HIDDEN_TERRAIN = ("lake", "mountain")


def load_log(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["radius"], data["steps"]


def draw_hex(surface, center, size, color, outline_color=HEX_OUTLINE_COLOR):
    points = [hex_corner(center, size, i) for i in range(6)]
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, outline_color, points, width=1)


class Checkbox:
    def __init__(self, x, y, label, checked):
        self.rect = pygame.Rect(x, y, CHECKBOX_SIZE, CHECKBOX_SIZE)
        self.label = label
        self.checked = checked

    def hit_test(self, pos):
        return self.rect.collidepoint(pos)

    def toggle(self):
        self.checked = not self.checked

    def draw(self, surface, font):
        pygame.draw.rect(surface, CHECKBOX_BOX_COLOR, self.rect, width=2)
        if self.checked:
            inner = self.rect.inflate(-6, -6)
            pygame.draw.rect(surface, TEXT_COLOR, inner)
        label_surf = font.render(self.label, True, TEXT_COLOR)
        surface.blit(label_surf, (self.rect.right + 10, self.rect.top + 1))


def draw_counts_sidebar(surface, x0, y0, width, height, counts_used, checkbox, header_font, row_font):
    pygame.draw.rect(surface, SIDEBAR_BG_COLOR, pygame.Rect(x0, y0, width, height))

    pad = 16
    header_surf = header_font.render("Terrain Bag", True, TEXT_COLOR)
    surface.blit(header_surf, (x0 + pad, y0 + pad))

    row_h = 34
    list_top = y0 + pad + 34
    for i, (terrain, total) in enumerate(BAG_COUNTS.items()):
        row_top = list_top + i * row_h
        used = counts_used.get(terrain, 0)
        remaining = total - used

        swatch = pygame.Rect(x0 + pad, row_top + 4, 16, 16)
        pygame.draw.rect(surface, TERRAIN_COLORS[terrain], swatch)

        text = f"{terrain.capitalize()}: {remaining} / {total}"
        text_surf = row_font.render(text, True, TEXT_COLOR)
        surface.blit(text_surf, (swatch.right + 10, row_top))

    divider_y = list_top + len(BAG_COUNTS) * row_h + 8
    pygame.draw.line(surface, SIDEBAR_DIVIDER_COLOR,
                      (x0 + pad, divider_y), (x0 + width - pad, divider_y), width=1)

    checkbox.rect.topleft = (x0 + pad, divider_y + 20)
    checkbox.draw(surface, row_font)


def main():
    if not os.path.exists(STATE_FILE):
        print(f"Could not find {STATE_FILE}")
        print("Run run.py first to create it.")
        sys.exit(1)

    radius, steps = load_log(STATE_FILE)
    coords = cube_hexes_in_radius(radius)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Terrain Generation Visualizer")
    clock = pygame.time.Clock()

    board_area_h = WINDOW_H - SLIDER_BAND_HEIGHT
    size = compute_hex_size(radius, BOARD_AREA_W, board_area_h, MARGIN)
    label_font = pygame.font.SysFont("arial", 16)
    sidebar_header_font = pygame.font.SysFont("arial", 17, bold=True)
    sidebar_row_font = pygame.font.SysFont("arial", 14)

    raw_centers = {coord: hex_to_pixel(coord[0], coord[1], size) for coord in coords}
    xs = [p[0] for p in raw_centers.values()]
    ys = [p[1] for p in raw_centers.values()]
    board_cx = (min(xs) + max(xs)) / 2
    board_cy = (min(ys) + max(ys)) / 2
    offset_x = BOARD_AREA_W / 2 - board_cx
    offset_y = board_area_h / 2 - board_cy
    centers = {coord: (p[0] + offset_x, p[1] + offset_y) for coord, p in raw_centers.items()}

    def label_fn(i):
        if i == 0:
            return f"Step 0 / {len(steps)}   (empty board)   (drag, or use ←/→)"
        step = steps[i - 1]
        return f"Step {i} / {len(steps)}   Round {step['round'] + 1}: {step['terrain']}   (drag, or use ←/→)"

    slider = Slider(WINDOW_W, WINDOW_H - 30, len(steps) + 1, label_fn)
    hide_checkbox = Checkbox(0, 0, "Hide lakes/mountains", checked=True)

    current_step = len(steps)  # start fully generated; scrub back to replay
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
                    current_step = max(0, current_step - 1)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    current_step = min(len(steps), current_step + 1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if hide_checkbox.hit_test(event.pos):
                    hide_checkbox.toggle()
                elif slider.hit_test(event.pos):
                    dragging = True
                    current_step = slider.index_at(event.pos[0])
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                current_step = slider.index_at(event.pos[0])

        placed_so_far = steps[:current_step]
        terrain_by_coord = {
            (step["q"], step["r"], step["s"]): step["terrain"]
            for step in placed_so_far
        }
        counts_used = collections.Counter(step["terrain"] for step in placed_so_far)

        screen.fill(BG_COLOR)
        for coord in coords:
            terrain = terrain_by_coord.get(coord)
            outline_color = HEX_OUTLINE_COLOR
            if terrain is None:
                color = UNSET_COLOR
            elif hide_checkbox.checked and terrain in _HIDDEN_TERRAIN:
                color = BG_COLOR
                outline_color = BG_COLOR  # no border between hidden hexes and the background
            else:
                color = TERRAIN_COLORS[terrain]
            draw_hex(screen, centers[coord], size, color, outline_color)

        draw_counts_sidebar(screen, BOARD_AREA_W, 0, SIDEBAR_WIDTH, board_area_h,
                             counts_used, hide_checkbox, sidebar_header_font, sidebar_row_font)

        slider.draw(screen, current_step, label_font)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
