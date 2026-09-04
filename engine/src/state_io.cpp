#include "oo/state_io.hpp"

#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace oo {

namespace {

template <class T>
void write_array(std::ostream& out, const char* name, const T* values, int n) {
    out << name << ' ' << n;
    for (int i = 0; i < n; ++i) out << ' ' << static_cast<long long>(values[i]);
    out << '\n';
}

bool expect_token(std::istream& in, const char* want, std::string& error) {
    std::string got;
    if (!(in >> got) || got != want) {
        error = std::string("expected '") + want + "', got '" + got + "'";
        return false;
    }
    return true;
}

// Reads "<NAME> <count> v0 v1 ..." and stores into `out`, checking the count.
template <class T>
bool read_array(std::istream& in, const char* name, T* out, int expected_n, std::string& error) {
    if (!expect_token(in, name, error)) return false;
    int n = 0;
    if (!(in >> n) || n != expected_n) {
        std::ostringstream os;
        os << name << ": expected count " << expected_n << ", got " << n;
        error = os.str();
        return false;
    }
    for (int i = 0; i < n; ++i) {
        long long v = 0;
        if (!(in >> v)) {
            std::ostringstream os;
            os << name << ": ran out of values at index " << i;
            error = os.str();
            return false;
        }
        out[i] = static_cast<T>(v);
    }
    return true;
}

}  // namespace

void write_state(std::ostream& out, const GameState& state) {
    const int n = state.num_hexes;
    const int f = state.num_factions;

    out << "STATE 1\n";
    out << "RADIUS " << state.grid->radius() << '\n';
    out << "NUM_FACTIONS " << f << '\n';
    out << "NUM_HEXES " << n << '\n';
    out << "TURN " << state.turn_number << '\n';

    write_array(out, "TERRAIN", state.terrain, n);
    write_array(out, "CITY_OWNER", state.city_owner, n);
    {
        std::vector<int> tmp(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) tmp[static_cast<size_t>(i)] = state.is_capital[i] ? 1 : 0;
        write_array(out, "IS_CAPITAL", tmp.data(), n);
    }
    write_array(out, "OUTPOST_UPGRADE", state.outpost_upgrade, n);
    write_array(out, "CITY_PLACER", state.city_placer, n);
    write_array(out, "ARMY_FACTION", state.army_faction, n);
    {
        std::vector<int> tmp(static_cast<size_t>(n) * NUM_UNIT_TYPES);
        for (int i = 0; i < n; ++i) {
            for (int u = 0; u < NUM_UNIT_TYPES; ++u) {
                tmp[static_cast<size_t>(i) * NUM_UNIT_TYPES + u] = state.army_units[i][u];
            }
        }
        write_array(out, "ARMY_UNITS", tmp.data(), n * NUM_UNIT_TYPES);
    }
    {
        std::vector<int> tmp(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) tmp[static_cast<size_t>(i)] = state.frozen[i] ? 1 : 0;
        write_array(out, "FROZEN", tmp.data(), n);
        for (int i = 0; i < n; ++i) tmp[static_cast<size_t>(i)] = state.locked[i] ? 1 : 0;
        write_array(out, "LOCKED", tmp.data(), n);
    }
    write_array(out, "BATTLE_ROUND", state.battle_round, n);

    // Sparse: only occupied contribution slots.
    int slot_count = 0;
    for (int h = 0; h < n; ++h) {
        for (int k = 0; k < MAX_BATTLE_CONTRIB; ++k) {
            if (state.battle_faction[h][k] != NO_FACTION) ++slot_count;
        }
    }
    out << "BATTLE_SLOTS " << slot_count << '\n';
    for (int h = 0; h < n; ++h) {
        for (int k = 0; k < MAX_BATTLE_CONTRIB; ++k) {
            if (state.battle_faction[h][k] == NO_FACTION) continue;
            out << h << ' ' << k << ' ' << static_cast<int>(state.battle_faction[h][k]) << ' '
                << state.battle_origin[h][k] << ' ' << state.battle_units[h][k][kInfantry] << ' '
                << state.battle_units[h][k][kCavalry] << ' ' << state.battle_units[h][k][kArchers]
                << ' ' << (state.battle_moved[h][k] ? 1 : 0) << '\n';
        }
    }

    write_array(out, "BATTLE_ORDER", state.battle_order, state.num_battles);
    write_array(out, "CAPITAL_SETTLE_ORDER", state.capital_settle_order, f);
    write_array(out, "GOLD", state.gold, f);
    {
        std::vector<int> tmp(static_cast<size_t>(f) * NUM_RESOURCES);
        for (int i = 0; i < f; ++i) {
            for (int r = 0; r < NUM_RESOURCES; ++r) {
                tmp[static_cast<size_t>(i) * NUM_RESOURCES + r] = state.resources[i][r];
            }
        }
        write_array(out, "RESOURCES", tmp.data(), f * NUM_RESOURCES);
    }
    write_array(out, "KILL_XP", state.kill_xp, f);
    write_array(out, "VICTORY_POINTS", state.victory_points, f);
    {
        std::vector<int> tmp(static_cast<size_t>(f));
        for (int i = 0; i < f; ++i) tmp[static_cast<size_t>(i)] = state.alive[i] ? 1 : 0;
        write_array(out, "ALIVE", tmp.data(), f);
    }
    out << "END\n";
}

bool read_state(std::istream& in, GameState& state, std::string& error) {
    if (!expect_token(in, "STATE", error)) return false;
    int version = 0;
    if (!(in >> version) || version != 1) {
        error = "unsupported STATE version";
        return false;
    }

    int radius = 0, num_factions = 0, num_hexes = 0, turn = 0;
    if (!expect_token(in, "RADIUS", error) || !(in >> radius)) return false;
    if (!expect_token(in, "NUM_FACTIONS", error) || !(in >> num_factions)) return false;
    if (!expect_token(in, "NUM_HEXES", error) || !(in >> num_hexes)) return false;
    if (!expect_token(in, "TURN", error) || !(in >> turn)) return false;

    const HexGrid& grid = HexGrid::shared(radius);
    if (grid.num_hexes() != num_hexes) {
        error = "NUM_HEXES disagrees with RADIUS";
        return false;
    }
    new_empty(state, grid, num_factions);
    state.turn_number = turn;

    const int n = num_hexes;
    const int f = num_factions;
    std::vector<int> tmp;

    if (!read_array(in, "TERRAIN", state.terrain, n, error)) return false;
    if (!read_array(in, "CITY_OWNER", state.city_owner, n, error)) return false;
    tmp.assign(static_cast<size_t>(n), 0);
    if (!read_array(in, "IS_CAPITAL", tmp.data(), n, error)) return false;
    for (int i = 0; i < n; ++i) state.is_capital[i] = tmp[static_cast<size_t>(i)] != 0;
    if (!read_array(in, "OUTPOST_UPGRADE", state.outpost_upgrade, n, error)) return false;
    if (!read_array(in, "CITY_PLACER", state.city_placer, n, error)) return false;
    if (!read_array(in, "ARMY_FACTION", state.army_faction, n, error)) return false;

    tmp.assign(static_cast<size_t>(n) * NUM_UNIT_TYPES, 0);
    if (!read_array(in, "ARMY_UNITS", tmp.data(), n * NUM_UNIT_TYPES, error)) return false;
    for (int i = 0; i < n; ++i) {
        for (int u = 0; u < NUM_UNIT_TYPES; ++u) {
            state.army_units[i][u] =
                static_cast<int16_t>(tmp[static_cast<size_t>(i) * NUM_UNIT_TYPES + u]);
        }
    }

    tmp.assign(static_cast<size_t>(n), 0);
    if (!read_array(in, "FROZEN", tmp.data(), n, error)) return false;
    for (int i = 0; i < n; ++i) state.frozen[i] = tmp[static_cast<size_t>(i)] != 0;
    if (!read_array(in, "LOCKED", tmp.data(), n, error)) return false;
    for (int i = 0; i < n; ++i) state.locked[i] = tmp[static_cast<size_t>(i)] != 0;

    if (!read_array(in, "BATTLE_ROUND", state.battle_round, n, error)) return false;

    if (!expect_token(in, "BATTLE_SLOTS", error)) return false;
    int slot_count = 0;
    if (!(in >> slot_count)) {
        error = "BATTLE_SLOTS: missing count";
        return false;
    }
    for (int i = 0; i < slot_count; ++i) {
        int h, k, faction, origin, inf, cav, arc, moved;
        if (!(in >> h >> k >> faction >> origin >> inf >> cav >> arc >> moved)) {
            error = "BATTLE_SLOTS: truncated row";
            return false;
        }
        state.battle_faction[h][k] = static_cast<int8_t>(faction);
        state.battle_origin[h][k] = origin;
        state.battle_units[h][k][kInfantry] = static_cast<int16_t>(inf);
        state.battle_units[h][k][kCavalry] = static_cast<int16_t>(cav);
        state.battle_units[h][k][kArchers] = static_cast<int16_t>(arc);
        state.battle_moved[h][k] = moved != 0;
        if (k + 1 > state.battle_nslots[h]) state.battle_nslots[h] = static_cast<uint8_t>(k + 1);
    }

    if (!expect_token(in, "BATTLE_ORDER", error)) return false;
    int num_battles = 0;
    if (!(in >> num_battles)) {
        error = "BATTLE_ORDER: missing count";
        return false;
    }
    state.num_battles = static_cast<int16_t>(num_battles);
    for (int i = 0; i < num_battles; ++i) {
        int h = 0;
        if (!(in >> h)) {
            error = "BATTLE_ORDER: truncated";
            return false;
        }
        state.battle_order[i] = static_cast<int16_t>(h);
    }

    if (!read_array(in, "CAPITAL_SETTLE_ORDER", state.capital_settle_order, f, error)) return false;
    if (!read_array(in, "GOLD", state.gold, f, error)) return false;
    tmp.assign(static_cast<size_t>(f) * NUM_RESOURCES, 0);
    if (!read_array(in, "RESOURCES", tmp.data(), f * NUM_RESOURCES, error)) return false;
    for (int i = 0; i < f; ++i) {
        for (int r = 0; r < NUM_RESOURCES; ++r) {
            state.resources[i][r] = tmp[static_cast<size_t>(i) * NUM_RESOURCES + r];
        }
    }
    if (!read_array(in, "KILL_XP", state.kill_xp, f, error)) return false;
    if (!read_array(in, "VICTORY_POINTS", state.victory_points, f, error)) return false;
    tmp.assign(static_cast<size_t>(f), 0);
    if (!read_array(in, "ALIVE", tmp.data(), f, error)) return false;
    for (int i = 0; i < f; ++i) state.alive[i] = tmp[static_cast<size_t>(i)] != 0;

    return expect_token(in, "END", error);
}

namespace {

template <class T>
bool diff_array(const T* a, const T* b, int n, const char* name, std::string& diff) {
    for (int i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            std::ostringstream os;
            os << name << "[" << i << "]: got " << static_cast<long long>(a[i]) << ", want "
               << static_cast<long long>(b[i]);
            diff = os.str();
            return false;
        }
    }
    return true;
}

}  // namespace

bool compare_states(const GameState& a, const GameState& b, std::string& diff) {
    if (a.num_hexes != b.num_hexes || a.num_factions != b.num_factions) {
        diff = "shape mismatch (num_hexes / num_factions)";
        return false;
    }
    const int n = a.num_hexes;
    const int f = a.num_factions;

    if (!diff_array(a.terrain, b.terrain, n, "terrain", diff)) return false;
    if (!diff_array(a.city_owner, b.city_owner, n, "city_owner", diff)) return false;
    if (!diff_array(a.is_capital, b.is_capital, n, "is_capital", diff)) return false;
    if (!diff_array(a.outpost_upgrade, b.outpost_upgrade, n, "outpost_upgrade", diff)) return false;
    if (!diff_array(a.city_placer, b.city_placer, n, "city_placer", diff)) return false;
    if (!diff_array(a.army_faction, b.army_faction, n, "army_faction", diff)) return false;
    for (int u = 0; u < NUM_UNIT_TYPES; ++u) {
        for (int i = 0; i < n; ++i) {
            if (a.army_units[i][u] != b.army_units[i][u]) {
                std::ostringstream os;
                os << "army_units[" << i << "][" << u << "]: got " << a.army_units[i][u]
                   << ", want " << b.army_units[i][u];
                diff = os.str();
                return false;
            }
        }
    }
    if (!diff_array(a.frozen, b.frozen, n, "frozen", diff)) return false;
    if (!diff_array(a.locked, b.locked, n, "locked", diff)) return false;
    if (!diff_array(a.battle_round, b.battle_round, n, "battle_round", diff)) return false;

    for (int h = 0; h < n; ++h) {
        for (int k = 0; k < MAX_BATTLE_CONTRIB; ++k) {
            if (a.battle_faction[h][k] != b.battle_faction[h][k] ||
                a.battle_origin[h][k] != b.battle_origin[h][k] ||
                a.battle_moved[h][k] != b.battle_moved[h][k] ||
                a.battle_units[h][k][0] != b.battle_units[h][k][0] ||
                a.battle_units[h][k][1] != b.battle_units[h][k][1] ||
                a.battle_units[h][k][2] != b.battle_units[h][k][2]) {
                std::ostringstream os;
                os << "battle slot [" << h << "][" << k << "]: got (f=" << int(a.battle_faction[h][k])
                   << " o=" << a.battle_origin[h][k] << " u=" << a.battle_units[h][k][0] << "/"
                   << a.battle_units[h][k][1] << "/" << a.battle_units[h][k][2]
                   << " moved=" << a.battle_moved[h][k] << "), want (f="
                   << int(b.battle_faction[h][k]) << " o=" << b.battle_origin[h][k]
                   << " u=" << b.battle_units[h][k][0] << "/" << b.battle_units[h][k][1] << "/"
                   << b.battle_units[h][k][2] << " moved=" << b.battle_moved[h][k] << ")";
                diff = os.str();
                return false;
            }
        }
    }

    if (a.num_battles != b.num_battles) {
        std::ostringstream os;
        os << "num_battles: got " << a.num_battles << ", want " << b.num_battles;
        diff = os.str();
        return false;
    }
    if (!diff_array(a.battle_order, b.battle_order, a.num_battles, "battle_order", diff)) return false;

    if (!diff_array(a.capital_settle_order, b.capital_settle_order, f, "capital_settle_order", diff))
        return false;
    if (!diff_array(a.gold, b.gold, f, "gold", diff)) return false;
    for (int r = 0; r < NUM_RESOURCES; ++r) {
        for (int i = 0; i < f; ++i) {
            if (a.resources[i][r] != b.resources[i][r]) {
                std::ostringstream os;
                os << "resources[" << i << "][" << r << "]: got " << a.resources[i][r] << ", want "
                   << b.resources[i][r];
                diff = os.str();
                return false;
            }
        }
    }
    if (!diff_array(a.kill_xp, b.kill_xp, f, "kill_xp", diff)) return false;
    if (!diff_array(a.victory_points, b.victory_points, f, "victory_points", diff)) return false;
    if (!diff_array(a.alive, b.alive, f, "alive", diff)) return false;

    if (a.turn_number != b.turn_number) {
        diff = "turn_number";
        return false;
    }
    return true;
}

bool validate_state(const GameState& state, std::string& problem) {
    const int n = state.num_hexes;
    std::ostringstream os;

    for (int h = 0; h < n; ++h) {
        if (state.army_faction[h] != NO_FACTION) {
            if (state.army_faction[h] < 0 || state.army_faction[h] >= state.num_factions) {
                os << "hex " << h << ": army_faction out of range";
                problem = os.str();
                return false;
            }
            if (state.units_at(h) == 0) {
                os << "hex " << h << ": army_faction set but no units";
                problem = os.str();
                return false;
            }
            if (!state.passable(h)) {
                os << "hex " << h << ": army standing on impassable terrain";
                problem = os.str();
                return false;
            }
            // The 6-unit limit is strict outside battle. A locked hex is exempt:
            // engine_old's _revert_departure can legitimately recreate a peaceful
            // army on a hex that an unrelated battle locked this same step.
            if (!state.locked[h] && state.units_at(h) > MAX_STACK_SIZE) {
                os << "hex " << h << ": peaceful stack of " << state.units_at(h) << " exceeds "
                   << MAX_STACK_SIZE;
                problem = os.str();
                return false;
            }
        } else if (state.units_at(h) != 0) {
            os << "hex " << h << ": units present with no owning faction";
            problem = os.str();
            return false;
        }

        // Occupied battle slots must be contiguous from 0 - engine_old allocates
        // via _first_empty_battle_slot and never frees an individual slot, so a
        // gap means something cleared a slot it should not have.
        bool seen_empty = false;
        int occupied = 0;
        for (int k = 0; k < MAX_BATTLE_CONTRIB; ++k) {
            if (state.battle_faction[h][k] == NO_FACTION) {
                seen_empty = true;
            } else {
                if (seen_empty) {
                    os << "hex " << h << ": battle slot " << k << " occupied after an empty slot";
                    problem = os.str();
                    return false;
                }
                ++occupied;
            }
        }
        if (occupied != state.battle_nslots[h]) {
            os << "hex " << h << ": battle_nslots=" << int(state.battle_nslots[h]) << " but "
               << occupied << " slots occupied";
            problem = os.str();
            return false;
        }
        if ((occupied > 0) != state.locked[h]) {
            os << "hex " << h << ": locked=" << state.locked[h] << " but " << occupied
               << " battle slots occupied";
            problem = os.str();
            return false;
        }
    }

    // battle_order must list exactly the locked hexes, without duplicates.
    if (state.num_battles < 0 || state.num_battles > MAX_ACTIVE_BATTLES) {
        problem = "num_battles out of range";
        return false;
    }
    int locked_total = 0;
    for (int h = 0; h < n; ++h) locked_total += state.locked[h] ? 1 : 0;
    if (locked_total != state.num_battles) {
        os << "num_battles=" << state.num_battles << " but " << locked_total << " hexes locked";
        problem = os.str();
        return false;
    }
    for (int i = 0; i < state.num_battles; ++i) {
        const int h = state.battle_order[i];
        if (h < 0 || h >= n || !state.locked[h]) {
            os << "battle_order[" << i << "]=" << h << " is not a locked hex";
            problem = os.str();
            return false;
        }
        for (int j = 0; j < i; ++j) {
            if (state.battle_order[j] == h) {
                os << "battle_order contains hex " << h << " twice";
                problem = os.str();
                return false;
            }
        }
    }
    return true;
}

}  // namespace oo
