// Writers for the three replay files (PLAN.md §1.3). Output must be
// byte-identical to Python's json.dump - see json.cpp's header for the rules.

#pragma once

#include "oo/log.hpp"
#include "oo/placement.hpp"
#include "oo/setup.hpp"
#include "oo/state.hpp"

#include <string>
#include <vector>

namespace oo {

// board_state.json - consumed by web_visualizer.html.
void write_board_state_json(std::string& out, const GameState& state, int radius, int num_factions,
                            const std::vector<TurnRecord>& turns);

// terrain_gen_log.json - every terrain placement in generation order.
void write_terrain_log_json(std::string& out, int radius,
                            const std::vector<TerrainLogEntry>& steps);

// city_placement_log.json - every placement/draft/swap step in order.
void write_placement_log_json(std::string& out, const GameState& state, int radius,
                              int num_factions, const std::vector<PlacementLogEntry>& steps);

}  // namespace oo
