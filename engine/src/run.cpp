#include "oo/run.hpp"

#include "oo/grid.hpp"

#include <atomic>
#include <thread>
#include <vector>

namespace oo {

int default_thread_count() {
    const unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : static_cast<int>(n);
}

namespace {

// Plays one spec. Everything it touches is either on its own stack or reached
// through the immutable grid, so this is safe to call from any number of threads
// at once.
void play_one(const RunConfig& cfg, const GameSpec& spec, GameResult& out) {
    AgentSet agents;
    build_mixed_agents(agents, spec.seats, cfg.num_factions, spec.seed);
    out = play_game(agents, cfg.radius, cfg.num_factions, spec.seed, cfg.max_turns);
}

}  // namespace

void run_games(const RunConfig& cfg, const GameSpec* specs, GameResult* out, int num_games,
               int num_threads) {
    if (num_games <= 0) return;
    if (num_threads <= 0) num_threads = default_thread_count();
    if (num_threads > num_games) num_threads = num_games;

    // Built before any worker exists. HexGrid::shared is mutex-guarded so a race
    // here would be safe anyway, but warming it means workers never serialize on
    // that lock and never pay for construction inside a timed region.
    (void)HexGrid::shared(cfg.radius);

    if (num_threads == 1) {
        for (int i = 0; i < num_games; ++i) play_one(cfg, specs[i], out[i]);
        return;
    }

    // A shared cursor rather than a static range split. Games vary a lot in cost -
    // a tactician game can run 40x a random one, and turn counts differ by 3x
    // within one agent - so fixed slices would leave threads idle waiting on
    // whichever slice drew the long games. Pulling work on demand self-balances,
    // and because each game writes only out[i], the ORDER games are claimed in has
    // no effect on the results.
    std::atomic<int> next{0};
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(num_threads));
    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&] {
            for (;;) {
                const int i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= num_games) return;
                play_one(cfg, specs[i], out[i]);
            }
        });
    }
    for (std::thread& w : workers) w.join();
}

}  // namespace oo
