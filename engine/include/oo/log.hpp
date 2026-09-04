// Replay logging - the data behind board_state.json (PLAN.md §1.3).
//
// These structures exist to be serialized, so they mirror the Python dicts
// engine_old/turn.py produces exactly, INCLUDING KEY ORDER. json.cpp writes them
// in declaration order and the output must come out byte-identical to what
// json.dump produced, because web_visualizer.html is the one viewer that
// survives Python removal.
//
// Unlike everything else in the engine these use std::vector. That is deliberate:
// logging only runs for a single logged game via `oo_run`, never in the
// tournament/self-play path, so the no-heap-in-the-turn-loop rule (§4.5) does not
// apply here - and a fixed-capacity round log would have to be sized for the
// 50-round safety cap it will never reach.

#pragma once

#include "oo/agent.hpp"
#include "oo/state.hpp"

#include <string>
#include <vector>

namespace oo {

// engine_old/turn.py: CHECKPOINT_LABELS. 1 start + 1 buy + 3 move + 2 cavalry + 1
// battle. Not serialized (the viewer has its own copy), but the count defines how
// many snapshots a turn produces.
inline constexpr int kCheckpointsPerTurn = 8;

struct DeathEntry {
    int8_t faction;
    int8_t unit_type;  // index into UNIT_TYPES; written as a name
    int16_t count;
    int8_t killer;
};

struct DismountEntry {
    int8_t faction;
    bool success;
    bool capped;  // adds "reason": "cap" - only ever present when success is false
};

struct RoundLog {
    // Dicts keyed by faction. Stored as parallel vectors so insertion order,
    // which json.dump preserves and the viewer relies on, survives.
    std::vector<int8_t> choice_faction;
    std::vector<int8_t> choice_target;  // -1 means Python's None
    std::vector<int8_t> resolved_attacker;
    std::vector<int8_t> resolved_target;
    std::vector<int8_t> roll_faction;
    std::vector<int16_t> roll_value;
    std::vector<int16_t> kills_dealt;  // parallel to roll_faction
    std::vector<DeathEntry> deaths;
    std::vector<DismountEntry> dismounts;
};

struct BattleLog {
    std::vector<DeathEntry> structure_phase;
    std::vector<DismountEntry> structure_dismounts;
    std::vector<DeathEntry> archer_phase;
    std::vector<DismountEntry> archer_dismounts;
    std::vector<RoundLog> rounds;
};

struct ContributionStart {
    int8_t faction;
    int32_t origin_hex;
    int16_t units[NUM_UNIT_TYPES];
};

struct SendBackLog {
    int32_t origin_hex;
    int16_t units[NUM_UNIT_TYPES];
};

struct BattleEvent {
    int8_t q, r, s;  // "hex": [q, r, s]
    std::vector<ContributionStart> contributions_start;
    BattleLog log;
    int winner;  // -1 means Python's None
    std::vector<SendBackLog> rectification;
};

// One hex in a checkpoint snapshot. `has_city` / `has_troops` / `has_battle`
// stand in for Python's None.
struct HexSnapshot {
    int8_t q, r, s;

    bool has_city = false;
    int8_t city_faction = 0;
    bool is_capital = false;
    int8_t upgrade = NO_UPGRADE;  // written as a name, or null

    bool has_troops = false;
    int8_t troop_faction = 0;
    int16_t troops[NUM_UNIT_TYPES] = {};
    bool frozen = false;

    bool has_battle = false;
    std::vector<ContributionStart> contributions;  // origin_hex unused here
};

struct PlayerStats {
    int32_t gold;
    int32_t resources[NUM_RESOURCES];
    int32_t kill_xp;
    int32_t victory_points;
    bool alive;
};

struct TurnRecord {
    int32_t turn_number;
    // kCheckpointsPerTurn entries, each already filtered to occupied hexes.
    std::vector<std::vector<HexSnapshot>> checkpoints;
    std::vector<BattleEvent> battle_events;
    std::vector<std::vector<PlayerStats>> player_stats;
};

// Same as run_turn, but also fills `out` with everything needed to replay the
// turn. Deliberately a separate entry point so the unlogged path stays free of
// any logging cost at all (logging is run.py's dominant expense).
void run_turn_and_log(GameState& state, const TurnDecisions& decisions, Rng& rng, TurnRecord& out);

// Occupied hexes only - engine_old's sparse_hexes(snapshot_hexes(state)).
void snapshot_sparse(const GameState& state, std::vector<HexSnapshot>& out);
void snapshot_player_stats(const GameState& state, std::vector<PlayerStats>& out);

}  // namespace oo
