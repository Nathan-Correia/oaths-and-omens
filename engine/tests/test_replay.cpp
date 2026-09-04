// Replay-file parity: the three JSON files must come out byte-identical to
// Python's (PLAN.md §1.3).
//
// Compares SHA-256 of the bytes against goldens produced by the Python pipeline.
// Hashing rather than checking in the files themselves is only about repo size -
// the check is still byte-exact, which matters because web_visualizer.html is the
// one viewer that survives Python removal.
//
// Usage: test_replay <path-to-replay_hashes.txt>

#include "oo/agent.hpp"
#include "oo/json.hpp"
#include "oo/log.hpp"
#include "oo/placement.hpp"
#include "oo/setup.hpp"
#include "oo/turn.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Minimal SHA-256 - a few dozen lines and no dependency, versus pulling in a
// crypto library for one test.
struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint64_t len = 0;
    uint8_t buf[64];
    size_t buf_len = 0;

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t* p) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(p[i * 4]) << 24) | (uint32_t(p[i * 4 + 1]) << 16) |
                   (uint32_t(p[i * 4 + 2]) << 8) | uint32_t(p[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const uint8_t* p, size_t n) {
        len += n;
        while (n > 0) {
            const size_t take = std::min(n, size_t(64) - buf_len);
            std::memcpy(buf + buf_len, p, take);
            buf_len += take;
            p += take;
            n -= take;
            if (buf_len == 64) {
                block(buf);
                buf_len = 0;
            }
        }
    }

    std::string hex() {
        const uint64_t bits = len * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        pad = 0;
        while (buf_len != 56) update(&pad, 1);
        uint8_t tail[8];
        for (int i = 0; i < 8; ++i) tail[i] = uint8_t(bits >> (56 - i * 8));
        update(tail, 8);
        std::ostringstream os;
        for (int i = 0; i < 8; ++i) {
            os << std::hex << std::setw(8) << std::setfill('0') << h[i];
        }
        return os.str();
    }
};

std::string sha256(const std::string& s) {
    Sha256 sha;
    sha.update(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    return sha.hex();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_replay <replay_hashes.txt>\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 2;
    }

    std::string tag;
    int total = 0;
    in >> tag >> total;
    if (tag != "REPLAY_HASHES") {
        std::cerr << "malformed file: expected REPLAY_HASHES header\n";
        return 2;
    }

    // Rows for the same game are contiguous; regenerate a game once and check all
    // three of its files.
    std::string last_key;
    std::string board_json, terrain_json, placement_json;

    int passed = 0, failures = 0;
    for (int i = 0; i < total; ++i) {
        std::string agent, file, want_hash;
        int radius, factions, max_turns;
        long long seed;
        size_t want_size;
        in >> agent >> radius >> factions >> seed >> max_turns >> file >> want_size >> want_hash;

        std::ostringstream keyss;
        keyss << agent << radius << factions << seed << max_turns;
        if (keyss.str() != last_key) {
            last_key = keyss.str();
            oo::AgentKind kind;
            if (!oo::agent_kind_from_name(agent.c_str(), kind)) {
                std::cerr << "unknown agent " << agent << "\n";
                return 2;
            }
            auto state = std::make_unique<oo::GameState>();
            std::vector<oo::TerrainLogEntry> terrain_log;
            oo::create_initial_state(*state, radius, factions, seed, &terrain_log);

            oo::AgentSet agents;
            oo::build_agents(agents, kind, factions, seed);
            oo::Rng rng(seed);
            std::vector<oo::PlacementLogEntry> placement_log;
            oo::SetupDecisions sd = oo::make_setup_decisions(agents);
            oo::run_city_setup(*state, sd, rng, &placement_log);

            oo::TurnDecisions td = oo::make_turn_decisions(agents);
            std::vector<oo::TurnRecord> turns;
            while (!oo::check_game_end(*state, max_turns)) {
                oo::TurnRecord record;
                oo::run_turn_and_log(*state, td, rng, record);
                turns.push_back(std::move(record));
            }

            oo::write_terrain_log_json(terrain_json, radius, terrain_log);
            oo::write_placement_log_json(placement_json, *state, radius, factions, placement_log);
            oo::write_board_state_json(board_json, *state, radius, factions, turns);
        }

        const std::string& got = file == "board_state.json"      ? board_json
                                 : file == "terrain_gen_log.json" ? terrain_json
                                                                  : placement_json;
        const std::string got_hash = sha256(got);
        if (got.size() == want_size && got_hash == want_hash) {
            ++passed;
        } else if (++failures <= 10) {
            std::cerr << "FAIL " << agent << "-r" << radius << "f" << factions << "s" << seed
                      << " " << file << ": " << got.size() << " bytes / " << got_hash
                      << ", want " << want_size << " bytes / " << want_hash << "\n";
        }
    }

    std::printf("test_replay: %d/%d replay files byte-identical to Python, %d failures\n", passed,
                total, failures);
    return failures == 0 ? 0 : 1;
}
