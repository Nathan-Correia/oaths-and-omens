// Native JSON writer for the three replay files (PLAN.md §1.3).
//
// THE OUTPUT MUST BE BYTE-IDENTICAL to what Python's json.dump produced, because
// web_visualizer.html is the one viewer that survives Python removal and the
// files are compared against Python-generated ones as the M6c gate. That means
// matching json.dump's defaults exactly:
//
//   - separators ", " between items and ": " after a key
//   - no indentation, no trailing newline inside the document
//   - null / true / false lowercase
//   - dict keys in INSERTION order (never sorted - sort_keys defaults to False)
//   - integer dict keys become strings ("0": ...), which is what json.dump does
//
// Every value in these files is an int, string, bool or null - there are no
// floats anywhere, which removes the only genuinely hard formatting question.

#include "oo/json.hpp"

#include <cstdio>

namespace oo {

namespace {

const char* kTerrainNames[NUM_TERRAIN_TYPES] = {"plains", "mountain", "lake", "desert", "marsh"};
const char* kUnitNames[NUM_UNIT_TYPES] = {"infantry", "cavalry", "archers"};
const char* kUpgradeNames[NUM_UPGRADE_TYPES] = {"barracks", "workshop", "temple"};
const char* kResourceNames[NUM_RESOURCES] = {"wood", "iron", "clay", "fish"};

void put_int(std::string& o, long long v) {
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%lld", v);
    o += buf;
}

void key(std::string& o, const char* k, bool first = false) {
    if (!first) o += ", ";
    o += '"';
    o += k;
    o += "\": ";
}

void write_deaths(std::string& o, const std::vector<DeathEntry>& deaths) {
    o += '[';
    for (size_t i = 0; i < deaths.size(); ++i) {
        if (i) o += ", ";
        o += '{';
        key(o, "faction", true);
        put_int(o, deaths[i].faction);
        key(o, "unit_type");
        o += '"';
        o += kUnitNames[deaths[i].unit_type];
        o += '"';
        key(o, "count");
        put_int(o, deaths[i].count);
        key(o, "killer");
        put_int(o, deaths[i].killer);
        o += '}';
    }
    o += ']';
}

void write_dismounts(std::string& o, const std::vector<DismountEntry>& d) {
    o += '[';
    for (size_t i = 0; i < d.size(); ++i) {
        if (i) o += ", ";
        o += '{';
        key(o, "faction", true);
        put_int(o, d[i].faction);
        key(o, "success");
        o += d[i].success ? "true" : "false";
        // "reason" is only present on a cap-blocked failure, matching engine_old.
        if (d[i].capped) {
            key(o, "reason");
            o += "\"cap\"";
        }
        o += '}';
    }
    o += ']';
}

// A dict keyed by faction id. json.dump renders int keys as strings.
void write_int_keyed(std::string& o, const std::vector<int8_t>& keys,
                     const std::vector<int16_t>& values) {
    o += '{';
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i) o += ", ";
        o += '"';
        put_int(o, keys[i]);
        o += "\": ";
        put_int(o, values[i]);
    }
    o += '}';
}

void write_int_keyed_i8(std::string& o, const std::vector<int8_t>& keys,
                        const std::vector<int8_t>& values, bool null_for_negative) {
    o += '{';
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i) o += ", ";
        o += '"';
        put_int(o, keys[i]);
        o += "\": ";
        if (null_for_negative && values[i] < 0) {
            o += "null";
        } else {
            put_int(o, values[i]);
        }
    }
    o += '}';
}

void write_units_inline(std::string& o, const int16_t units[NUM_UNIT_TYPES]) {
    key(o, "infantry");
    put_int(o, units[kInfantry]);
    key(o, "cavalry");
    put_int(o, units[kCavalry]);
    key(o, "archers");
    put_int(o, units[kArchers]);
}

void write_snapshot(std::string& o, const std::vector<HexSnapshot>& hexes) {
    o += '[';
    for (size_t i = 0; i < hexes.size(); ++i) {
        const HexSnapshot& e = hexes[i];
        if (i) o += ", ";
        o += '{';
        key(o, "q", true);
        put_int(o, e.q);
        key(o, "r");
        put_int(o, e.r);
        key(o, "s");
        put_int(o, e.s);

        key(o, "city");
        if (!e.has_city) {
            o += "null";
        } else {
            o += '{';
            key(o, "faction", true);
            put_int(o, e.city_faction);
            key(o, "is_capital");
            o += e.is_capital ? "true" : "false";
            key(o, "upgrade");
            if (e.upgrade == NO_UPGRADE) {
                o += "null";
            } else {
                o += '"';
                o += kUpgradeNames[e.upgrade];
                o += '"';
            }
            o += '}';
        }

        key(o, "troops");
        if (!e.has_troops) {
            o += "null";
        } else {
            o += '{';
            key(o, "faction", true);
            put_int(o, e.troop_faction);
            write_units_inline(o, e.troops);
            key(o, "frozen");
            o += e.frozen ? "true" : "false";
            o += '}';
        }

        key(o, "battle");
        if (!e.has_battle) {
            o += "null";
        } else {
            o += '{';
            key(o, "contributions", true);
            o += '[';
            for (size_t c = 0; c < e.contributions.size(); ++c) {
                if (c) o += ", ";
                o += '{';
                key(o, "faction", true);
                put_int(o, e.contributions[c].faction);
                write_units_inline(o, e.contributions[c].units);
                o += '}';
            }
            o += ']';
            o += '}';
        }
        o += '}';
    }
    o += ']';
}

void write_battle_event(std::string& o, const BattleEvent& ev) {
    o += '{';
    key(o, "hex", true);
    o += '[';
    put_int(o, ev.q);
    o += ", ";
    put_int(o, ev.r);
    o += ", ";
    put_int(o, ev.s);
    o += ']';

    key(o, "contributions_start");
    o += '[';
    for (size_t i = 0; i < ev.contributions_start.size(); ++i) {
        if (i) o += ", ";
        o += '{';
        key(o, "faction", true);
        put_int(o, ev.contributions_start[i].faction);
        key(o, "origin_hex");
        put_int(o, ev.contributions_start[i].origin_hex);
        write_units_inline(o, ev.contributions_start[i].units);
        o += '}';
    }
    o += ']';

    key(o, "structure_phase");
    write_deaths(o, ev.log.structure_phase);
    key(o, "structure_phase_dismounts");
    write_dismounts(o, ev.log.structure_dismounts);
    key(o, "archer_phase");
    write_deaths(o, ev.log.archer_phase);
    key(o, "archer_phase_dismounts");
    write_dismounts(o, ev.log.archer_dismounts);

    key(o, "rounds");
    o += '[';
    for (size_t i = 0; i < ev.log.rounds.size(); ++i) {
        const RoundLog& r = ev.log.rounds[i];
        if (i) o += ", ";
        o += '{';
        key(o, "target_choices_submitted", true);
        write_int_keyed_i8(o, r.choice_faction, r.choice_target, /*null_for_negative=*/true);
        key(o, "resolved_targets");
        write_int_keyed_i8(o, r.resolved_attacker, r.resolved_target, false);
        key(o, "rolls");
        write_int_keyed(o, r.roll_faction, r.roll_value);
        key(o, "kills_dealt");
        write_int_keyed(o, r.roll_faction, r.kills_dealt);
        key(o, "deaths");
        write_deaths(o, r.deaths);
        key(o, "dismounts");
        write_dismounts(o, r.dismounts);
        o += '}';
    }
    o += ']';

    key(o, "winner");
    if (ev.winner < 0) {
        o += "null";
    } else {
        put_int(o, ev.winner);
    }

    key(o, "rectification");
    o += '[';
    for (size_t i = 0; i < ev.rectification.size(); ++i) {
        if (i) o += ", ";
        o += '{';
        key(o, "origin_hex", true);
        put_int(o, ev.rectification[i].origin_hex);
        key(o, "units");
        o += '[';
        put_int(o, ev.rectification[i].units[0]);
        o += ", ";
        put_int(o, ev.rectification[i].units[1]);
        o += ", ";
        put_int(o, ev.rectification[i].units[2]);
        o += ']';
        o += '}';
    }
    o += ']';
    o += '}';
}

void write_player_stats(std::string& o, const std::vector<PlayerStats>& stats) {
    o += '{';
    for (size_t f = 0; f < stats.size(); ++f) {
        if (f) o += ", ";
        o += '"';
        put_int(o, static_cast<long long>(f));
        o += "\": {";
        key(o, "gold", true);
        put_int(o, stats[f].gold);
        key(o, "resources");
        o += '{';
        for (int r = 0; r < NUM_RESOURCES; ++r) {
            if (r) o += ", ";
            o += '"';
            o += kResourceNames[r];
            o += "\": ";
            put_int(o, stats[f].resources[r]);
        }
        o += '}';
        key(o, "kill_xp");
        put_int(o, stats[f].kill_xp);
        key(o, "victory_points");
        put_int(o, stats[f].victory_points);
        key(o, "alive");
        o += stats[f].alive ? "true" : "false";
        o += '}';
    }
    o += '}';
}

void write_terrain_map(std::string& o, const GameState& state) {
    // Keyed "q_r_s", in grid-coordinate order - the same insertion order run.py's
    // dict comprehension produces.
    o += '{';
    for (int h = 0; h < state.num_hexes; ++h) {
        if (h) o += ", ";
        const HexCoord& c = state.grid->coord_of(h);
        o += '"';
        put_int(o, c.q);
        o += '_';
        put_int(o, c.r);
        o += '_';
        put_int(o, c.s);
        o += "\": \"";
        o += kTerrainNames[state.terrain[h]];
        o += '"';
    }
    o += '}';
}

}  // namespace

void write_board_state_json(std::string& o, const GameState& state, int radius, int num_factions,
                            const std::vector<TurnRecord>& turns) {
    o.clear();
    o += '{';
    key(o, "radius", true);
    put_int(o, radius);
    key(o, "num_factions");
    put_int(o, num_factions);
    key(o, "terrain");
    write_terrain_map(o, state);

    key(o, "turns");
    o += '[';
    for (size_t t = 0; t < turns.size(); ++t) {
        const TurnRecord& tr = turns[t];
        if (t) o += ", ";
        o += '{';
        key(o, "turn_number", true);
        put_int(o, tr.turn_number);

        key(o, "checkpoints");
        o += '[';
        for (size_t c = 0; c < tr.checkpoints.size(); ++c) {
            if (c) o += ", ";
            write_snapshot(o, tr.checkpoints[c]);
        }
        o += ']';

        key(o, "battle_events");
        o += '[';
        for (size_t b = 0; b < tr.battle_events.size(); ++b) {
            if (b) o += ", ";
            write_battle_event(o, tr.battle_events[b]);
        }
        o += ']';

        key(o, "player_stats");
        o += '[';
        for (size_t p = 0; p < tr.player_stats.size(); ++p) {
            if (p) o += ", ";
            write_player_stats(o, tr.player_stats[p]);
        }
        o += ']';
        o += '}';
    }
    o += ']';
    o += '}';
}

void write_terrain_log_json(std::string& o, int radius, const std::vector<TerrainLogEntry>& steps) {
    o.clear();
    o += '{';
    key(o, "radius", true);
    put_int(o, radius);
    key(o, "steps");
    o += '[';
    for (size_t i = 0; i < steps.size(); ++i) {
        if (i) o += ", ";
        o += '{';
        key(o, "q", true);
        put_int(o, steps[i].q);
        key(o, "r");
        put_int(o, steps[i].r);
        key(o, "s");
        put_int(o, steps[i].s);
        key(o, "terrain");
        o += '"';
        o += kTerrainNames[steps[i].terrain];
        o += '"';
        key(o, "round");
        put_int(o, steps[i].round);
        o += '}';
    }
    o += ']';
    o += '}';
}

void write_placement_log_json(std::string& o, const GameState& state, int radius, int num_factions,
                              const std::vector<PlacementLogEntry>& steps) {
    static const char* kKinds[] = {"place", "draft", "draft_auto", "keep", "swap"};
    o.clear();
    o += '{';
    key(o, "radius", true);
    put_int(o, radius);
    key(o, "num_factions");
    put_int(o, num_factions);
    key(o, "terrain");
    write_terrain_map(o, state);
    key(o, "steps");
    o += '[';
    for (size_t i = 0; i < steps.size(); ++i) {
        const PlacementLogEntry& e = steps[i];
        if (i) o += ", ";
        o += '{';
        key(o, "type", true);
        o += '"';
        o += kKinds[e.kind];
        o += '"';
        key(o, "faction");
        put_int(o, e.faction);
        key(o, "q");
        put_int(o, e.q);
        key(o, "r");
        put_int(o, e.r);
        key(o, "s");
        put_int(o, e.s);
        // The placer_* keys exist only on a swap, matching run_city_setup.
        if (e.kind == PlacementLogEntry::kSwap) {
            key(o, "placer_faction");
            put_int(o, e.placer_faction);
            key(o, "placer_q");
            put_int(o, e.placer_q);
            key(o, "placer_r");
            put_int(o, e.placer_r);
            key(o, "placer_s");
            put_int(o, e.placer_s);
        }
        o += '}';
    }
    o += ']';
    o += '}';
}

}  // namespace oo
