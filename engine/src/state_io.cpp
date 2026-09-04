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
        // LOCKED is derived from battle_index now, but stays in the format: the
        // Python reference still has it as a field, and writing it lets read_state
        // cross-check the two representations for free.
        for (int i = 0; i < n; ++i) tmp[static_cast<size_t>(i)] = state.locked(i) ? 1 : 0;
        write_array(out, "LOCKED", tmp.data(), n);
    }
    {
        std::vector<int> rounds(static_cast<size_t>(n), 0);
        for (int h = 0; h < n; ++h) {
            const Battle* b = state.battle_at(h);
            rounds[static_cast<size_t>(h)] = b != nullptr ? b->round : 0;
        }
        write_array(out, "BATTLE_ROUND", rounds.data(), n);
    }

    // Written in HEX order, which is what the Python side produces by scanning its
    // dense arrays. The file format predates the sparse layout and is kept
    // unchanged so existing goldens stay valid.
    int slot_count = 0;
    for (int h = 0; h < n; ++h) {
        if (const Battle* b = state.battle_at(h)) slot_count += b->nslots;
    }
    out << "BATTLE_SLOTS " << slot_count << '\n';
    for (int h = 0; h < n; ++h) {
        const Battle* b = state.battle_at(h);
        if (b == nullptr) continue;
        for (int k = 0; k < b->nslots; ++k) {
            const BattleSlot& sl = b->slots[k];
            out << h << ' ' << k << ' ' << static_cast<int>(sl.faction) << ' ' << sl.origin << ' '
                << sl.units[kInfantry] << ' ' << sl.units[kCavalry] << ' ' << sl.units[kArchers]
                << ' ' << (sl.moved ? 1 : 0) << '\n';
        }
    }

    {
        std::vector<int> order(static_cast<size_t>(state.num_battles));
        for (int i = 0; i < state.num_battles; ++i) {
            order[static_cast<size_t>(i)] = state.battles[i].hex;
        }
        write_array(out, "BATTLE_ORDER", order.data(), state.num_battles);
    }
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
    std::vector<int> want_locked(static_cast<size_t>(n));
    if (!read_array(in, "LOCKED", want_locked.data(), n, error)) return false;

    std::vector<int> want_round(static_cast<size_t>(n));
    if (!read_array(in, "BATTLE_ROUND", want_round.data(), n, error)) return false;

    if (!expect_token(in, "BATTLE_SLOTS", error)) return false;
    int slot_count = 0;
    if (!(in >> slot_count)) {
        error = "BATTLE_SLOTS: missing count";
        return false;
    }
    // Slots arrive in hex order; the battles themselves are created below in
    // BATTLE_ORDER, which is the creation order and the one that affects outcomes.
    struct PendingSlot {
        int h, k, faction, origin, inf, cav, arc, moved;
    };
    std::vector<PendingSlot> pending(static_cast<size_t>(slot_count));
    for (int i = 0; i < slot_count; ++i) {
        PendingSlot& ps = pending[static_cast<size_t>(i)];
        if (!(in >> ps.h >> ps.k >> ps.faction >> ps.origin >> ps.inf >> ps.cav >> ps.arc >>
              ps.moved)) {
            error = "BATTLE_SLOTS: truncated row";
            return false;
        }
    }

    if (!expect_token(in, "BATTLE_ORDER", error)) return false;
    int num_battles = 0;
    if (!(in >> num_battles)) {
        error = "BATTLE_ORDER: missing count";
        return false;
    }
    for (int i = 0; i < num_battles; ++i) {
        int h = 0;
        if (!(in >> h)) {
            error = "BATTLE_ORDER: truncated";
            return false;
        }
        Battle& b = state.new_battle(h);
        b.round = static_cast<int16_t>(want_round[static_cast<size_t>(h)]);
    }
    for (const PendingSlot& ps : pending) {
        Battle* b = state.battle_at(ps.h);
        if (b == nullptr) {
            error = "BATTLE_SLOTS: slot on a hex missing from BATTLE_ORDER";
            return false;
        }
        BattleSlot& sl = b->slots[ps.k];
        sl.faction = static_cast<int8_t>(ps.faction);
        sl.origin = ps.origin;
        sl.units[kInfantry] = static_cast<int16_t>(ps.inf);
        sl.units[kCavalry] = static_cast<int16_t>(ps.cav);
        sl.units[kArchers] = static_cast<int16_t>(ps.arc);
        sl.moved = ps.moved != 0;
        if (ps.k + 1 > b->nslots) b->nslots = static_cast<uint8_t>(ps.k + 1);
    }

    // Free cross-check: the file's LOCKED must agree with the derived value.
    for (int i = 0; i < n; ++i) {
        if (state.locked(i) != (want_locked[static_cast<size_t>(i)] != 0)) {
            std::ostringstream os;
            os << "LOCKED disagrees with battle storage at hex " << i;
            error = os.str();
            return false;
        }
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

    if (a.num_battles != b.num_battles) {
        std::ostringstream os;
        os << "num_battles: got " << a.num_battles << ", want " << b.num_battles;
        diff = os.str();
        return false;
    }
    // Compared in creation order, which is the order that affects outcomes.
    for (int i = 0; i < a.num_battles; ++i) {
        const Battle& ba = a.battles[i];
        const Battle& bb = b.battles[i];
        if (ba.hex != bb.hex || ba.round != bb.round || ba.nslots != bb.nslots) {
            std::ostringstream os;
            os << "battle[" << i << "]: got (hex=" << ba.hex << " round=" << ba.round
               << " nslots=" << int(ba.nslots) << "), want (hex=" << bb.hex << " round=" << bb.round
               << " nslots=" << int(bb.nslots) << ")";
            diff = os.str();
            return false;
        }
        for (int k = 0; k < ba.nslots; ++k) {
            const BattleSlot& sa = ba.slots[k];
            const BattleSlot& sb = bb.slots[k];
            if (sa.faction != sb.faction || sa.origin != sb.origin || sa.moved != sb.moved ||
                sa.units[0] != sb.units[0] || sa.units[1] != sb.units[1] ||
                sa.units[2] != sb.units[2]) {
                std::ostringstream os;
                os << "battle hex " << ba.hex << " slot " << k << ": got (f=" << int(sa.faction)
                   << " o=" << sa.origin << " u=" << sa.units[0] << "/" << sa.units[1] << "/"
                   << sa.units[2] << " moved=" << sa.moved << "), want (f=" << int(sb.faction)
                   << " o=" << sb.origin << " u=" << sb.units[0] << "/" << sb.units[1] << "/"
                   << sb.units[2] << " moved=" << sb.moved << ")";
                diff = os.str();
                return false;
            }
        }
    }

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
            if (!state.locked(h) && state.units_at(h) > MAX_STACK_SIZE) {
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

        // battle_index must point at a battle that agrees it lives on this hex.
        // With the sparse layout this replaces the old "occupied slots are
        // contiguous" check - a gap is now structurally impossible, but a stale
        // or crossed index is the new failure mode worth guarding.
        const int bi = state.battle_index[h];
        if (bi != -1) {
            if (bi < 0 || bi >= state.num_battles) {
                os << "hex " << h << ": battle_index " << bi << " out of range (num_battles="
                   << state.num_battles << ")";
                problem = os.str();
                return false;
            }
            if (state.battles[bi].hex != h) {
                os << "hex " << h << ": battle_index points at a battle on hex "
                   << state.battles[bi].hex;
                problem = os.str();
                return false;
            }
        }
    }

    if (state.num_battles < 0 || state.num_battles > MAX_ACTIVE_BATTLES) {
        problem = "num_battles out of range";
        return false;
    }
    for (int i = 0; i < state.num_battles; ++i) {
        const Battle& b = state.battles[i];
        if (b.hex < 0 || b.hex >= n) {
            os << "battles[" << i << "] has hex " << b.hex << " out of range";
            problem = os.str();
            return false;
        }
        if (state.battle_index[b.hex] != i) {
            os << "battles[" << i << "] on hex " << b.hex << " but battle_index says "
               << state.battle_index[b.hex];
            problem = os.str();
            return false;
        }
        if (b.nslots == 0 || b.nslots > MAX_BATTLE_CONTRIB) {
            os << "battles[" << i << "] has " << int(b.nslots) << " slots";
            problem = os.str();
            return false;
        }
        for (int k = 0; k < b.nslots; ++k) {
            if (b.slots[k].faction < 0 || b.slots[k].faction >= state.num_factions) {
                os << "battles[" << i << "] slot " << k << ": faction out of range";
                problem = os.str();
                return false;
            }
        }
    }
    return true;
}

}  // namespace oo
