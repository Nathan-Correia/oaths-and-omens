// Action and decision types shared by buy/movement/battle/turn.
//
// Everything here is fixed-capacity and heap-free (PLAN.md §4.5). SmallVec exists
// so legal-action generation can hand back a list without allocating in the turn
// loop.

#pragma once

#include "oo/config.hpp"

#include <cassert>
#include <cstdint>

namespace oo {

template <class T, int Capacity>
struct SmallVec {
    T items[Capacity];
    int count = 0;

    void clear() { count = 0; }
    bool empty() const { return count == 0; }
    int size() const { return count; }
    static constexpr int capacity() { return Capacity; }

    void push_back(const T& value) {
        assert(count < Capacity && "SmallVec overflow - raise the capacity");
        items[count++] = value;
    }
    T& operator[](int i) { return items[i]; }
    const T& operator[](int i) const { return items[i]; }
    const T* begin() const { return items; }
    const T* end() const { return items + count; }
};

// --- movement ---------------------------------------------------------------

// A movement action: move the phase-determined subset of the army at `hex` one
// step in direction `dir`. The whole army during the movement phase, or exactly
// the cavalry count during the cavalry phase - never an arbitrary split. See
// engine_old/movement.py's SCOPE note.
struct Move {
    int16_t hex;
    int8_t dir;
};

// Per-faction movement actions for one simultaneous step. `has[f]` false means
// that faction is not moving this step (Python's None).
//
// SUBMISSION ORDER IS PART OF THE STATE, not just which factions moved. Movement
// groups simultaneous arrivals by destination in first-arrival order, and that
// grouping order becomes battle CREATION order - which is load-bearing, since the
// dismount infantry-cap tally is shared across a turn's battles.
//
// engine_old carries this implicitly: it iterates `actions_by_faction`, a dict, in
// insertion order. run_turn inserts ascending so it never shows... except in
// tactician_agent's rollout, which builds `{faction: first_action}` and only then
// adds everyone else, submitting the searching faction FIRST. An array indexed by
// faction cannot express that, and silently reordered the battles. Hence `order`.
struct MoveActions {
    Move move[MAX_FACTIONS];
    bool has[MAX_FACTIONS];
    int8_t order[MAX_FACTIONS];  // factions in submission order
    int n_order;

    void clear() {
        for (int f = 0; f < MAX_FACTIONS; ++f) has[f] = false;
        n_order = 0;
    }
    void set(int faction, int hex, int dir) {
        if (!has[faction]) order[n_order++] = static_cast<int8_t>(faction);
        move[faction] = Move{static_cast<int16_t>(hex), static_cast<int8_t>(dir)};
        has[faction] = true;
    }
};

// legal_mask[h][d]: may this faction's army at h move to h's neighbour in
// direction d this step?
struct LegalMask {
    bool cell[MAX_HEXES][NUM_DIRECTIONS];

    bool any_for_hex(int h) const {
        for (int d = 0; d < NUM_DIRECTIONS; ++d) {
            if (cell[h][d]) return true;
        }
        return false;
    }
};

// --- buy --------------------------------------------------------------------

enum class BuyType : uint8_t {
    kBuyInfantry = 0,
    kConvertToSpecial = 1,
    kBuildOutpost = 2,
    kUpgradeOutpost = 3,
};

struct BuyAction {
    BuyType type;
    int16_t hex;       // city_hex for kBuyInfantry, hex for the rest
    int8_t unit_type;  // kConvertToSpecial: kCavalry/kArchers. kBuildOutpost: unit consumed.
    int8_t upgrade;    // kUpgradeOutpost: kBarracks/kWorkshop/kTemple

    friend bool operator==(const BuyAction& a, const BuyAction& b) {
        if (a.type != b.type || a.hex != b.hex) return false;
        switch (a.type) {
            case BuyType::kConvertToSpecial:
            case BuyType::kBuildOutpost:
                return a.unit_type == b.unit_type;
            case BuyType::kUpgradeOutpost:
                return a.upgrade == b.upgrade;
            default:
                return true;
        }
    }
};

// A faction's legal buy list can get long: convert and build_outpost each offer
// several actions per army hex, so on a big board with many armies this runs to
// several hundred. Sized with headroom; SmallVec asserts if it is ever wrong.
inline constexpr int kMaxLegalBuy = 4096;
inline constexpr int kMaxChosenBuy = 512;

using LegalBuyActions = SmallVec<BuyAction, kMaxLegalBuy>;
using ChosenBuyActions = SmallVec<BuyAction, kMaxChosenBuy>;

// --- battle rectification ---------------------------------------------------

// "Send these units back to this origin hex." engine_old/battle.py's
// rectify_overflow send_back entries.
struct SendBackEntry {
    int32_t origin_hex;  // NO_ORIGIN, or an out-of-range value, means the units are lost
    int16_t units[NUM_UNIT_TYPES];
};

inline constexpr int kMaxSendBack = 64;
using SendBack = SmallVec<SendBackEntry, kMaxSendBack>;

}  // namespace oo
