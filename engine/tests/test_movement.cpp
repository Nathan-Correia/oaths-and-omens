// Targeted movement scenarios - the edge cases full-turn traces barely reach.
//
// The cases were built by tools/dump_movement_scenarios.py, deleted with the rest
// of the Python at M8; the file itself is now the only record of them. Case names
// still say what each one is for.
//
// Usage: test_movement <path-to-movement_scenarios.txt> [--rewrite <out>]

#include "rewrite.hpp"

#include "oo/movement.hpp"
#include "oo/state_io.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <array>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string rewrite_path;
    const bool rewriting = oo_test::take_rewrite_flag(argc, argv, rewrite_path);

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
    // apply_movement_step mutates `before` in place, so a rewrite needs the
    // original back to re-emit it as the case's input.
    auto original = std::make_unique<oo::GameState>();

    oo_test::Rewriter rw;
    if (rewriting) {
        if (!rw.open(rewrite_path)) return 2;
        rw.out << "SCENARIOS " << total << "\n";
    }

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
        std::vector<std::array<int, 3>> raw_actions;
        for (int a = 0; a < n_actions; ++a) {
            int faction, hex_index, direction;
            in >> faction >> hex_index >> direction;
            actions.set(faction, hex_index, direction);
            raw_actions.push_back({faction, hex_index, direction});
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

        *original = *before;
        oo::Rng rng(seed);
        oo::apply_movement_step(*before, actions, rng, cavalry_only != 0);

        if (rw) {
            rw.out << "SCENARIO " << case_name << "\n"
                   << "SEED " << seed << "\n"
                   << "CAVALRY_ONLY " << cavalry_only << "\n"
                   << "ACTIONS " << n_actions << "\n";
            for (const auto& a : raw_actions) {
                rw.out << a[0] << ' ' << a[1] << ' ' << a[2] << "\n";
            }
            oo::write_state(rw.out, *original);
            oo::write_state(rw.out, *before);
            ++passed;
            continue;
        }

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

    if (rw) {
        std::printf("test_movement: rewrote %d scenarios to %s\n", passed, rewrite_path.c_str());
        return 0;
    }
    std::printf("test_movement: %d/%d scenarios passed, %d failures\n", passed, total, failures);
    return failures == 0 ? 0 : 1;
}
