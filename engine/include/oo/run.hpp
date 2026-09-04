// Parallel batch driver (PLAN.md §7).
//
// Parallelism sits at the GAME level, not inside a turn. Games are perfectly
// independent, run for about a millisecond, and share exactly one thing: the
// immutable HexGrid. Threading inside a turn would be synchronizing over a
// 217-hex board and would lose to its own overhead.
//
// DETERMINISM IS THE CONTRACT. out[i] depends only on specs[i] and cfg - never on
// num_threads, never on which worker picked the game up, never on scheduling.
// That falls out of seeding per GAME rather than per thread: a game's seed drives
// terrain, the turn RNG and every agent's generator, so a game is a pure function
// of its spec. Run the same batch on 1 thread or 12 and the results are
// byte-identical, which is what keeps parity testing and tournament
// reproducibility intact under parallelism.

#pragma once

#include "oo/agent.hpp"
#include "oo/game.hpp"

#include <cstdint>

namespace oo {

struct RunConfig {
    int radius = 7;
    int num_factions = 8;
    int max_turns = 100;
};

// One game to play: its seed and who sits in each seat. Seats are explicit rather
// than derived from a rotation rule so that the driver, not this layer, owns how
// assignments are chosen.
struct GameSpec {
    int64_t seed = 0;
    AgentKind seats[MAX_FACTIONS] = {};
};

// Plays specs[0..num_games) and writes outcomes into out[0..num_games).
//
// num_threads <= 0 means "one per hardware thread". num_threads == 1 runs inline
// with no threads spawned at all, which keeps single-game debugging and the
// replay path free of any concurrency.
//
// Grids for cfg.radius are warmed before any worker starts, so no worker races to
// build one and none of them contend on the cache lock in steady state.
void run_games(const RunConfig& cfg, const GameSpec* specs, GameResult* out, int num_games,
               int num_threads);

// Hardware threads available, or 1 if the platform will not say. Exposed so a
// driver can report what it is about to use.
int default_thread_count();

}  // namespace oo
