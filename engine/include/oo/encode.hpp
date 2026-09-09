// GameState -> network input, in the compact wire form (PLAN.md §10.4, §10.8).
//
// This is NOT the tensor the network consumes. §10.4 describes ~55 features per
// hex; expanding to that on the CPU and shipping it would be 11.8 KB per
// position, and at batch 1024 that is 12 MB over PCIe per decision point - which
// caps throughput at ~2.1M evals/s regardless of how fast the GPU is. For a net
// this small the BUS is the bottleneck, not the maths.
//
// So the CPU writes a dense byte record - 12 bytes per hex - and a CUDA kernel
// expands it to planes on the far side. That is ~2.6 KB per position at radius
// 8, a 4.5x reduction, and it hands the bottleneck back to compute.
//
// EVERYTHING IS RELATIVE TO THE ACTING FACTION. Faction f appears at index
// `(f - acting + num_factions) % num_factions`, so slot 0 is always "me". This
// is the cheapest sample-efficiency win available and it has to be here from the
// start: adding it later invalidates every checkpoint (§12.4).
//
// The layout is versioned. Any change to these structs must bump
// kEncodingVersion, because a checkpoint trained against one layout is silently
// wrong against another - it will happily load and play like noise.

#pragma once

#include "oo/config.hpp"
#include "oo/policy.hpp"
#include "oo/state.hpp"

#include <cstdint>

namespace oo {

inline constexpr int kEncodingVersion = 1;

// Per-hex flags.
enum HexFlag : uint8_t {
    kHexPresent = 1 << 0,   // false for padding past num_hexes - NOT the same as
                            // "plains", which is what a zeroed terrain byte means
    kHexFrozen = 1 << 1,
    kHexBattle = 1 << 2,
    kHexEdge = 1 << 3,
    kHexQuery = 1 << 4,      // the hex this decision is ABOUT (target, rectify,
                             // resource) - what lets one trunk serve every kind
    kHexQuery2 = 1 << 5,     // the second hex of a swap
    kHexPassable = 1 << 6,
};

#pragma pack(push, 1)
struct HexRecord {
    uint8_t terrain;      // index into TERRAIN_TYPES
    uint8_t city;         // 0 = none, else 1 + relative owner
    uint8_t city_flags;   // bit0 is_capital; bits1-2 = upgrade + 1 (0 = none)
    uint8_t army;         // 0 = none, else 1 + relative owner
    uint8_t units[NUM_UNIT_TYPES];  // saturated at 255
    uint8_t flags;        // HexFlag
    uint8_t battle_round;
    uint8_t pad;
    uint16_t battle_factions;  // multi-hot over relative faction index
};

struct FactionRecord {
    uint8_t active;       // f < num_factions
    uint8_t is_me;
    uint8_t outposts;
    uint8_t units[NUM_UNIT_TYPES];  // in play, by type, saturated
    int8_t settle_order;
    uint8_t pad;
    int16_t gold;
    int16_t resources[NUM_RESOURCES];
    int16_t victory_points;
    int16_t kill_xp;
    int16_t vp_behind_leader;
};

struct HeaderRecord {
    uint8_t version;
    uint8_t radius;
    uint8_t num_factions;
    uint8_t decision_kind;  // DecisionKind
    uint8_t step;           // movement / cavalry sub-step
    uint8_t pad;
    int16_t num_hexes;
    int16_t turn_number;
    int16_t turns_remaining;  // to the cap; negative means uncapped
    int16_t max_vp;           // highest VP on the board
    int16_t vp_to_win;
};
#pragma pack(pop)

static_assert(sizeof(HexRecord) == 12, "HexRecord layout is the wire format");
static_assert(sizeof(FactionRecord) == 24, "FactionRecord layout is the wire format");
static_assert(sizeof(HeaderRecord) == 16, "HeaderRecord layout is the wire format");

// One encoded position: a header, every faction slot, and every hex up to
// MAX_HEXES. Fixed size so a batch is a flat array with no offset table - the
// GPU indexes it arithmetically.
#pragma pack(push, 1)
struct EncodedPosition {
    HeaderRecord header;
    FactionRecord factions[MAX_FACTIONS];
    HexRecord hexes[MAX_HEXES];
};
#pragma pack(pop)

static_assert(sizeof(EncodedPosition) == 16 + 24 * MAX_FACTIONS + 12 * MAX_HEXES,
              "EncodedPosition must stay tightly packed");

// Bytes actually written for a board of `num_hexes`. A batch bucketed by radius
// (§10.8) transfers only this prefix per position rather than the MAX_HEXES
// worth, which is where most of the r5 saving comes from.
constexpr int encoded_bytes(int num_hexes) {
    return static_cast<int>(sizeof(HeaderRecord) + sizeof(FactionRecord) * MAX_FACTIONS +
                            sizeof(HexRecord) * static_cast<size_t>(num_hexes));
}

// Encodes one position. `request` supplies the decision kind, the step, and
// which hexes the question is about; everything else comes from the state.
void encode_position(const GameState& state, const DecisionRequest& request, EncodedPosition& out);

// Encodes a whole batch into a flat buffer. `out` must have room for
// `n * sizeof(EncodedPosition)` bytes.
void encode_batch(const DecisionRequest* requests, int n, EncodedPosition* out);

}  // namespace oo
