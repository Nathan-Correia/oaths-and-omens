#include "oo/game.hpp"

#include "oo/placement.hpp"
#include "oo/setup.hpp"
#include "oo/turn.hpp"

#include <memory>

namespace oo {

GameResult play_game(const AgentSet& agents, int radius, int num_factions, int64_t seed,
                     int max_turns) {
    // tournament.py threads ONE random.Random(seed) through setup and every turn,
    // while create_initial_state separately builds its own Random(seed) for
    // terrain. Both are reproduced here: `rng` below is the shared one, and
    // create_initial_state seeds its own internally.
    auto state = std::make_unique<GameState>();
    create_initial_state(*state, radius, num_factions, seed);

    Rng rng(seed);
    SetupDecisions sd = make_setup_decisions(agents);
    run_city_setup(*state, sd, rng);

    TurnDecisions td = make_turn_decisions(agents);
    GameResult result;
    while (!check_game_end(*state, max_turns)) {
        run_turn(*state, td, rng);
        ++result.turns;
    }

    result.winner = get_game_winner(*state);
    for (int f = 0; f < num_factions; ++f) result.victory_points[f] = state->victory_points[f];
    return result;
}

}  // namespace oo
