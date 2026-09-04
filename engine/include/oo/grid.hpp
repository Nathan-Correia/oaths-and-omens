// Immutable per-radius hex geometry - the port of engine_old/geometry.py.
//
// A HexGrid is built once per board radius, never mutated afterwards, and shared
// by every GameState and every thread by const reference (PLAN.md §4.2). That
// immutability is the multicore contract: nothing here may ever gain mutable
// state.
//
// The index assignment order is copied verbatim from engine_old/geometry.py's
// cube_hexes_in_radius, because index i must correspond to the same (q, r, s) the
// Python engine uses - every parity comparison depends on it.
//
// THE ONE REAL ADDITION over the Python version is `dist`, a precomputed
// [num_hexes][num_hexes] distance table. In the profile of a tactician game,
// hex_distance was 434k calls (0.37 s cumulative) and min_hex_distance_to_any
// materialized an [N, k, 3] numpy temporary on every call (0.36 s). Both collapse
// to a table lookup here. At radius 10 the table is 331^2 = 110 KB - trivially
// cache-resident for the radii actually played (169^2 = 28 KB at radius 7).

#pragma once

#include "oo/config.hpp"

#include <cstdint>
#include <vector>

namespace oo {

// engine_old/geometry.py: CUBE_DIRECTIONS. Order matters - a movement action is
// (hex_index, direction), so direction indices are part of the action space.
inline constexpr int8_t kCubeDirections[NUM_DIRECTIONS][3] = {
    {1, 0, -1}, {1, -1, 0}, {0, -1, 1}, {-1, 0, 1}, {-1, 1, 0}, {0, 1, -1},
};

struct HexCoord {
    int8_t q = 0, r = 0, s = 0;
    friend bool operator==(const HexCoord& a, const HexCoord& b) {
        return a.q == b.q && a.r == b.r && a.s == b.s;
    }
};

// engine_old/geometry.py: hex_distance. Only for coordinates not on the board
// (or during grid construction) - prefer HexGrid::distance, which is a lookup.
inline int hex_distance(const HexCoord& a, const HexCoord& b) {
    const int dq = a.q - b.q, dr = a.r - b.r, ds = a.s - b.s;
    const int aq = dq < 0 ? -dq : dq;
    const int ar = dr < 0 ? -dr : dr;
    const int as = ds < 0 ? -ds : ds;
    return aq > ar ? (aq > as ? aq : as) : (ar > as ? ar : as);
}

class HexGrid {
public:
    explicit HexGrid(int radius);

    int radius() const { return radius_; }
    int num_hexes() const { return num_hexes_; }

    const HexCoord& coord_of(int index) const { return coords_[index]; }

    // -1 if `coord` is off the board. engine_old's coord_to_index dict.
    int index_of(const HexCoord& coord) const;

    // neighbour(i, d) is hex i's neighbour in direction kCubeDirections[d], or
    // -1 if that direction falls off the board. The array-table equivalent of
    // engine_old/geometry.py's cached hex_neighbors().
    int16_t neighbour(int index, int direction) const {
        return neighbours_[index * NUM_DIRECTIONS + direction];
    }
    const int16_t* neighbours_of(int index) const { return &neighbours_[index * NUM_DIRECTIONS]; }

    // Precomputed cube distance. See the header comment for why this exists.
    uint8_t distance(int a, int b) const { return dist_[a * num_hexes_ + b]; }

    // engine_old/geometry.py: is_edge - on the board's outer ring.
    bool is_edge(int index) const { return is_edge_[index]; }

    // engine_old/geometry.py: direction_between. -1 if not neighbours.
    int direction_between(int from_index, int to_index) const;

    // Grids are immutable and a few hundred KB; build each radius once and share.
    // Not thread-safe to call concurrently for a radius not yet built - build the
    // radii you need before starting a thread pool.
    static const HexGrid& shared(int radius);

private:
    int radius_ = 0;
    int num_hexes_ = 0;
    std::vector<HexCoord> coords_;
    std::vector<int16_t> neighbours_;  // [num_hexes * 6]
    std::vector<uint8_t> dist_;        // [num_hexes * num_hexes]
    std::vector<uint8_t> is_edge_;
    std::vector<int> col_base_;        // prefix sum of column lengths, for index_of
};

}  // namespace oo
