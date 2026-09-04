// Buy-phase parity: legal-action generation, and the per-turn batch rules.
//
// Two files, two different holes each closes (see the dumpers for the full why):
//   legal_cases.txt    - get_legal_buy_actions and the movement/cavalry masks,
//                        compared element by element. Turn traces replay what an
//                        agent CHOSE and never check the menu it chose from.
//   buy_scenarios.txt  - hand-built action lists that force the per-turn batch
//                        caps to bite. Mutation testing showed removing the
//                        one-outpost-action-per-turn cap broke nothing in 180
//                        traced turns, because no real agent ever proposes two.
//
// Usage: test_buy <legal_cases.txt> <buy_scenarios.txt>

#include "oo/buy.hpp"
#include "oo/movement.hpp"
#include "oo/state_io.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void fail(const std::string& what) {
    if (++g_failures <= 20) std::cerr << "FAIL: " << what << "\n";
}

const char* buy_type_name(oo::BuyType t) {
    switch (t) {
        case oo::BuyType::kBuyInfantry: return "buy_infantry";
        case oo::BuyType::kConvertToSpecial: return "convert";
        case oo::BuyType::kBuildOutpost: return "build_outpost";
        default: return "upgrade_outpost";
    }
}

std::string describe(const oo::BuyAction& a) {
    std::ostringstream os;
    os << buy_type_name(a.type) << "(hex=" << a.hex << ",unit=" << int(a.unit_type)
       << ",up=" << int(a.upgrade) << ")";
    return os.str();
}

oo::BuyAction read_action(std::istream& in) {
    int type, hex, unit, upgrade;
    in >> type >> hex >> unit >> upgrade;
    oo::BuyAction a{};
    a.type = static_cast<oo::BuyType>(type);
    a.hex = static_cast<int16_t>(hex);
    a.unit_type = static_cast<int8_t>(unit);
    a.upgrade = static_cast<int8_t>(upgrade);
    return a;
}

int run_legal_cases(const char* path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "cannot open " << path << "\n";
        return -1;
    }
    std::string tag;
    int total = 0;
    in >> tag >> total;
    if (tag != "LEGAL_CASES") {
        std::cerr << "malformed legal case file\n";
        return -1;
    }

    auto state = std::make_unique<oo::GameState>();
    oo::LegalBuyActions legal;
    oo::LegalMask mask;
    int checked = 0;

    for (int i = 0; i < total; ++i) {
        std::string name;
        in >> tag >> name;
        std::string error;
        if (!oo::read_state(in, *state, error)) {
            std::cerr << "legal case " << name << ": " << error << "\n";
            return -1;
        }
        for (int f = 0; f < state->num_factions; ++f) {
            int faction = 0;
            in >> tag >> faction;

            int n = 0;
            in >> tag >> n;  // BUY
            oo::get_legal_buy_actions(*state, faction, legal);
            if (legal.size() != n) {
                std::ostringstream os;
                os << name << " f" << faction << ": legal buy count " << legal.size() << ", want "
                   << n;
                fail(os.str());
            }
            for (int a = 0; a < n; ++a) {
                const oo::BuyAction want = read_action(in);
                if (a < legal.size() && !(legal[a] == want)) {
                    std::ostringstream os;
                    os << name << " f" << faction << ": legal buy[" << a << "] = "
                       << describe(legal[a]) << ", want " << describe(want);
                    fail(os.str());
                }
            }
            ++checked;

            for (int which = 0; which < 2; ++which) {
                int count = 0;
                in >> tag >> count;  // MOVEMASK / CAVMASK
                if (which == 0) {
                    oo::legal_movement_mask(*state, faction, mask);
                } else {
                    oo::legal_cavalry_mask(*state, faction, mask);
                }
                std::vector<std::pair<int, int>> want;
                want.reserve(static_cast<size_t>(count));
                for (int c = 0; c < count; ++c) {
                    int h, d;
                    in >> h >> d;
                    want.emplace_back(h, d);
                }
                std::vector<std::pair<int, int>> got;
                for (int h = 0; h < state->num_hexes; ++h) {
                    for (int d = 0; d < oo::NUM_DIRECTIONS; ++d) {
                        if (mask.cell[h][d]) got.emplace_back(h, d);
                    }
                }
                if (got != want) {
                    std::ostringstream os;
                    os << name << " f" << faction << ": " << (which == 0 ? "movement" : "cavalry")
                       << " mask has " << got.size() << " cells, want " << want.size();
                    fail(os.str());
                }
                ++checked;
            }
        }
    }
    std::printf("  legal_cases: %d states, %d comparisons\n", total, checked);
    return total;
}

int run_buy_scenarios(const char* path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "cannot open " << path << "\n";
        return -1;
    }
    std::string tag;
    int total = 0;
    in >> tag >> total;
    if (tag != "BUY_SCENARIOS") {
        std::cerr << "malformed buy scenario file\n";
        return -1;
    }

    auto before = std::make_unique<oo::GameState>();
    auto expected = std::make_unique<oo::GameState>();

    for (int i = 0; i < total; ++i) {
        std::string name;
        in >> tag >> name;
        int n_factions = 0;
        in >> tag >> n_factions;

        oo::ChosenBuyActions chosen[oo::MAX_FACTIONS];
        for (int f = 0; f < oo::MAX_FACTIONS; ++f) chosen[f].clear();
        for (int k = 0; k < n_factions; ++k) {
            int faction = 0, count = 0;
            in >> faction >> count;
            for (int a = 0; a < count; ++a) chosen[faction].push_back(read_action(in));
        }

        std::string error;
        if (!oo::read_state(in, *before, error) || !oo::read_state(in, *expected, error)) {
            std::cerr << "buy scenario " << name << ": " << error << "\n";
            return -1;
        }

        oo::apply_buy_phase(*before, chosen);

        std::string diff;
        if (!oo::compare_states(*before, *expected, diff)) {
            fail(name + ": " + diff);
        }
    }
    std::printf("  buy_scenarios: %d scenarios\n", total);
    return total;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test_buy <legal_cases.txt> <buy_scenarios.txt>\n";
        return 2;
    }
    const int legal = run_legal_cases(argv[1]);
    const int scenarios = run_buy_scenarios(argv[2]);
    if (legal < 0 || scenarios < 0) return 2;

    std::printf("test_buy: %d legal-action states + %d buy scenarios, %d failures\n", legal,
                scenarios, g_failures);
    return g_failures == 0 ? 0 : 1;
}
