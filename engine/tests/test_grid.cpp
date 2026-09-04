// Checks oo::HexGrid against engine_old/geometry.py, plus self-consistency
// properties that need no reference at all.
//
// Usage: test_grid <path-to-grid_golden.txt>

#include "oo/grid.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void fail(const std::string& what) {
    if (++g_failures <= 20) std::cerr << "FAIL: " << what << "\n";
}

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) fail(what);
}

bool read_named(std::istream& in, const char* name, std::vector<long long>& out) {
    std::string tag;
    long long n = 0;
    if (!(in >> tag >> n) || tag != name) {
        std::cerr << "expected " << name << ", got " << tag << "\n";
        return false;
    }
    out.resize(static_cast<size_t>(n));
    for (long long i = 0; i < n; ++i) {
        if (!(in >> out[static_cast<size_t>(i)])) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_grid <grid_golden.txt>\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 2;
    }

    std::string tag;
    int radii = 0;
    while (in >> tag) {
        if (tag != "GRID") {
            std::cerr << "expected GRID, got " << tag << "\n";
            return 2;
        }
        int radius = 0, num_hexes = 0;
        in >> radius >> num_hexes;
        ++radii;

        const oo::HexGrid& g = oo::HexGrid::shared(radius);
        check(g.num_hexes() == num_hexes, "num_hexes for radius " + std::to_string(radius));

        std::vector<long long> v;

        // The index -> (q, r, s) assignment. Everything else depends on this.
        if (!read_named(in, "COORDS", v)) return 2;
        for (int i = 0; i < num_hexes; ++i) {
            const oo::HexCoord& c = g.coord_of(i);
            const bool ok = c.q == v[static_cast<size_t>(i) * 3 + 0] &&
                            c.r == v[static_cast<size_t>(i) * 3 + 1] &&
                            c.s == v[static_cast<size_t>(i) * 3 + 2];
            check(ok, "coord mismatch at radius " + std::to_string(radius) + " index " +
                          std::to_string(i));
        }

        if (!read_named(in, "NEIGHBOURS", v)) return 2;
        for (int i = 0; i < num_hexes; ++i) {
            for (int d = 0; d < oo::NUM_DIRECTIONS; ++d) {
                check(g.neighbour(i, d) ==
                          v[static_cast<size_t>(i) * oo::NUM_DIRECTIONS + static_cast<size_t>(d)],
                      "neighbour mismatch at radius " + std::to_string(radius) + " index " +
                          std::to_string(i) + " dir " + std::to_string(d));
            }
        }

        if (!read_named(in, "EDGE", v)) return 2;
        for (int i = 0; i < num_hexes; ++i) {
            check(g.is_edge(i) == (v[static_cast<size_t>(i)] != 0),
                  "is_edge mismatch at radius " + std::to_string(radius) + " index " +
                      std::to_string(i));
        }

        if (!read_named(in, "DISTSUM", v)) return 2;
        for (int i = 0; i < num_hexes; ++i) {
            long long sum = 0;
            for (int j = 0; j < num_hexes; ++j) sum += g.distance(i, j);
            check(sum == v[static_cast<size_t>(i)],
                  "distance row sum mismatch at radius " + std::to_string(radius) + " index " +
                      std::to_string(i));
        }

        while (in >> tag && tag == "DISTROW") {
            int row_index = 0;
            in >> row_index;
            long long n = 0;
            in >> n;
            for (long long j = 0; j < n; ++j) {
                long long want = 0;
                in >> want;
                check(g.distance(row_index, static_cast<int>(j)) == want,
                      "distance mismatch at radius " + std::to_string(radius) + " [" +
                          std::to_string(row_index) + "][" + std::to_string(j) + "]");
            }
        }
        if (tag != "ENDGRID") {
            std::cerr << "expected ENDGRID, got " << tag << "\n";
            return 2;
        }

        // --- self-consistency, no reference needed ---------------------------
        for (int i = 0; i < num_hexes; ++i) {
            check(g.index_of(g.coord_of(i)) == i, "index_of/coord_of round trip");
            check(g.distance(i, i) == 0, "distance to self is zero");
            for (int d = 0; d < oo::NUM_DIRECTIONS; ++d) {
                const int j = g.neighbour(i, d);
                if (j < 0) continue;
                // Adjacency is symmetric, exactly one step apart, and
                // direction_between inverts the neighbour lookup.
                check(g.distance(i, j) == 1, "neighbour is at distance 1");
                check(g.direction_between(i, j) == d, "direction_between inverts neighbour");
                bool mutual = false;
                for (int d2 = 0; d2 < oo::NUM_DIRECTIONS; ++d2) {
                    if (g.neighbour(j, d2) == i) mutual = true;
                }
                check(mutual, "adjacency is symmetric");
            }
        }
        // --- ball() is exactly the set the distance table says it is ---------
        // eligible_outpost_mask now trusts ball() to be the complete radius-r
        // neighbourhood. If it ever under-reports, hexes that should be banned
        // silently become legal outpost sites - a rules change that no parity
        // test would necessarily catch, because it only shows up once a game
        // reaches that board configuration. So check it exhaustively against
        // distance(), which the golden data above already pins.
        for (int i = 0; i < num_hexes; ++i) {
            for (int r = 0; r <= oo::kMaxBallRadius; ++r) {
                const int16_t* ball = g.ball(i, r);
                const int m = g.ball_size(i, r);

                bool in_ball[oo::MAX_HEXES] = {};
                for (int k = 0; k < m; ++k) {
                    const int j = ball[k];
                    check(j >= 0 && j < num_hexes, "ball entry in range");
                    check(!in_ball[j], "ball has no duplicates");
                    in_ball[j] = true;
                    check(g.distance(i, j) <= r, "ball entry is within r");
                }
                // ...and nothing within r is missing.
                int expected = 0;
                for (int j = 0; j < num_hexes; ++j) {
                    if (g.distance(i, j) > r) continue;
                    ++expected;
                    check(in_ball[j], "ball omits a hex within r");
                }
                check(m == expected, "ball size matches the distance table");
                check(m >= 1 && ball[0] == i, "ball starts with the centre itself");
            }
            // Prefix property: the radius-r ball is the first ball_size(i, r)
            // entries of the SAME array. eligible_outpost_mask depends on this -
            // ball() ignores its r argument entirely.
            for (int r = 1; r <= oo::kMaxBallRadius; ++r) {
                check(g.ball_size(i, r) >= g.ball_size(i, r - 1),
                      "ball sizes are non-decreasing in r");
                check(g.ball(i, r) == g.ball(i, r - 1), "balls share one backing array");
            }
        }

        // Off-board coordinates must be rejected rather than aliasing onto a hex.
        check(g.index_of(oo::HexCoord{static_cast<int8_t>(radius + 1), 0,
                                      static_cast<int8_t>(-(radius + 1))}) == -1,
              "index_of rejects off-board coord");
        check(g.index_of(oo::HexCoord{1, 1, 1}) == -1, "index_of rejects non-cube coord");
    }

    std::printf("test_grid: %d checks across %d radii, %d failures\n", g_checks, radii, g_failures);
    return g_failures == 0 ? 0 : 1;
}
