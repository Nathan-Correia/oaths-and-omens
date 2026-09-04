#include "oo/grid.hpp"

#include <cassert>
#include <map>
#include <memory>
#include <mutex>

namespace oo {

HexGrid::HexGrid(int radius) : radius_(radius) {
    assert(radius >= 0 && radius <= MAX_RADIUS);

    // Identical enumeration order to engine_old/geometry.py's
    // cube_hexes_in_radius. THIS ORDER IS THE INDEX ASSIGNMENT used by every
    // array in GameState, so it must match exactly or nothing lines up with the
    // Python engine.
    for (int q = -radius; q <= radius; ++q) {
        const int r1 = (-radius > -q - radius) ? -radius : -q - radius;
        const int r2 = (radius < -q + radius) ? radius : -q + radius;
        for (int r = r1; r <= r2; ++r) {
            coords_.push_back(HexCoord{static_cast<int8_t>(q), static_cast<int8_t>(r),
                                       static_cast<int8_t>(-q - r)});
        }
    }
    num_hexes_ = static_cast<int>(coords_.size());
    assert(num_hexes_ == 3 * radius * (radius + 1) + 1);
    assert(num_hexes_ <= MAX_HEXES);

    // Prefix sum of column lengths, so index_of is O(1) rather than a scan.
    // Built before the neighbour table because that table is filled via index_of.
    col_base_.resize(static_cast<size_t>(2 * radius + 2));
    int running = 0;
    for (int q = -radius; q <= radius; ++q) {
        col_base_[static_cast<size_t>(q + radius)] = running;
        const int a = (-radius > -q - radius) ? -radius : -q - radius;
        const int b = (radius < -q + radius) ? radius : -q + radius;
        running += b - a + 1;
    }
    col_base_[static_cast<size_t>(2 * radius + 1)] = running;

    neighbours_.assign(static_cast<size_t>(num_hexes_) * NUM_DIRECTIONS, -1);
    for (int i = 0; i < num_hexes_; ++i) {
        const HexCoord& c = coords_[static_cast<size_t>(i)];
        for (int d = 0; d < NUM_DIRECTIONS; ++d) {
            const HexCoord cand{static_cast<int8_t>(c.q + kCubeDirections[d][0]),
                                static_cast<int8_t>(c.r + kCubeDirections[d][1]),
                                static_cast<int8_t>(c.s + kCubeDirections[d][2])};
            neighbours_[static_cast<size_t>(i) * NUM_DIRECTIONS + d] =
                static_cast<int16_t>(index_of(cand));
        }
    }

    dist_.resize(static_cast<size_t>(num_hexes_) * static_cast<size_t>(num_hexes_));
    for (int i = 0; i < num_hexes_; ++i) {
        for (int j = i; j < num_hexes_; ++j) {
            const uint8_t d = static_cast<uint8_t>(
                hex_distance(coords_[static_cast<size_t>(i)], coords_[static_cast<size_t>(j)]));
            dist_[static_cast<size_t>(i) * num_hexes_ + j] = d;
            dist_[static_cast<size_t>(j) * num_hexes_ + i] = d;
        }
    }

    is_edge_.resize(static_cast<size_t>(num_hexes_));
    for (int i = 0; i < num_hexes_; ++i) {
        const HexCoord& c = coords_[static_cast<size_t>(i)];
        const int aq = c.q < 0 ? -c.q : c.q;
        const int ar = c.r < 0 ? -c.r : c.r;
        const int as = c.s < 0 ? -c.s : c.s;
        const int m = aq > ar ? (aq > as ? aq : as) : (ar > as ? ar : as);
        is_edge_[static_cast<size_t>(i)] = (m == radius) ? 1u : 0u;
    }
}

int HexGrid::index_of(const HexCoord& coord) const {
    // Closed-form inverse of the enumeration above (O(1) via col_base_), rather
    // than engine_old's coord_to_index dict. Within a column q the rows are
    // contiguous starting at r1 = max(-radius, -q - radius).
    const int q = coord.q, r = coord.r;
    if (coord.q + coord.r + coord.s != 0) return -1;
    if (q < -radius_ || q > radius_) return -1;
    const int r1 = (-radius_ > -q - radius_) ? -radius_ : -q - radius_;
    const int r2 = (radius_ < -q + radius_) ? radius_ : -q + radius_;
    if (r < r1 || r > r2) return -1;
    return col_base_[static_cast<size_t>(q + radius_)] + (r - r1);
}

int HexGrid::direction_between(int from_index, int to_index) const {
    for (int d = 0; d < NUM_DIRECTIONS; ++d) {
        if (neighbour(from_index, d) == to_index) return d;
    }
    return -1;
}

const HexGrid& HexGrid::shared(int radius) {
    static std::mutex mu;
    static std::map<int, std::unique_ptr<HexGrid>> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(radius);
    if (it == cache.end()) {
        it = cache.emplace(radius, std::make_unique<HexGrid>(radius)).first;
    }
    return *it->second;
}

}  // namespace oo
