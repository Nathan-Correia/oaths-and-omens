// Whole-game driver - the native equivalent of tournament.py's play_game.
//
// Deliberately kept free of I/O and of any per-turn logging: this is the path
// tournaments and (later) self-play take, and §7's thread pool will call it once
// per game with nothing shared but the immutable HexGrid.

#pragma once

#include "oo/agent.hpp"
#include "oo/state.hpp"

namespace oo {

struct GameResult {
    int winner = -1;  // -1 if max_turns was hit before anyone reached kVpToWin
    int turns = 0;
    int32_t victory_points[MAX_FACTIONS] = {};
};

// Plays one game to completion. `seed` drives terrain generation, the turn RNG
// and every agent's generator, exactly as tournament.py does - so a game is fully
// reproducible from (agents, radius, num_factions, seed) alone.
GameResult play_game(const AgentSet& agents, int radius, int num_factions, int64_t seed,
                     int max_turns);

}  // namespace oo
