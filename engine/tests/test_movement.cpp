// Targeted movement scenarios - the edge cases full-turn traces barely reach.
//
// See tools/dump_movement_scenarios.py for what each case is and why it matters.
// Usage: test_movement <path-to-movement_scenarios.txt>

#include "oo/movement.hpp"
#include "oo/state_io.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_movement <movement_scenarios.txt>\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 2;
    }

    std::string tag;
    int total = 0;
    in >> tag >> total;
    if (tag != "SCENARIOS") {
        std::cerr << "malformed file: expected SCENARIOS header\n";
        return 2;
    }

    auto before = std::make_unique<oo::GameState>();
    auto expected = std::make_unique<oo::GameState>();

    int passed = 0, failures = 0;
    for (int i = 0; i < total; ++i) {
        std::string case_name;
        long long seed = 0;
        int cavalry_only = 0, n_actions = 0;
        in >> tag >> case_name;
        if (tag != "SCENARIO") {
            std::cerr << "malformed file at case " << i << "\n";
            return 2;
        }
        in >> tag >> seed;
        in >> tag >> cavalry_only;
        in >> tag >> n_actions;

        // Actions are read in the order the file lists them, which is the order
        // Python submitted them - and submission order decides battle creation
        // order (see MoveActions).
        oo::MoveActions actions;
        actions.clear();
        for (int a = 0; a < n_actions; ++a) {
            int faction, hex_index, direction;
            in >> faction >> hex_index >> direction;
            actions.set(faction, hex_index, direction);
        }

        std::string error;
        if (!oo::read_state(in, *before, error)) {
            std::cerr << "case " << case_name << " (before): " << error << "\n";
            return 2;
        }
        if (!oo::read_state(in, *expected, error)) {
            std::cerr << "case " << case_name << " (after): " << error << "\n";
            return 2;
        }

        oo::Rng rng(seed);
        oo::apply_movement_step(*before, actions, rng, cavalry_only != 0);

        std::string diff;
        if (!oo::compare_states(*before, *expected, diff)) {
            if (++failures <= 20) std::cerr << "FAIL " << case_name << ": " << diff << "\n";
            continue;
        }
        // Deliberately NOT validating the output state here: several of these
        // scenarios exercise engine_old's documented _revert_departure quirks,
        // which can legitimately leave a peaceful army on a locked hex. Matching
        // the reference is the point; tidiness is tracked separately in PLAN.md §9.
        ++passed;
    }

    std::printf("test_movement: %d/%d scenarios passed, %d failures\n", passed, total, failures);
    return failures == 0 ? 0 : 1;
}
