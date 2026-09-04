// M2 phase parity: runs each ported phase on a Python-produced "before" state and
// compares against the Python-produced "after" state.
//
// Cases come from tools/dump_phase_cases.py - both real random-agent game states
// and deliberately perturbed ones (see that file for why both are needed).
//
// Usage: test_phases <path-to-phase_cases.txt>

#include "oo/collect.hpp"
#include "oo/state_io.hpp"
#include "oo/terrain.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace {

// MUST match dump_phase_cases.py's choice_policy(). Only consulted for an outpost
// adjacent to both a mountain and a lake; alternating on (faction + hex) makes sure
// both branches are exercised.
oo::Resource choice_policy(const oo::GameState&, int faction, int hex_index, void*) {
    return ((faction + hex_index) % 2 == 0) ? oo::kIron : oo::kFish;
}

bool run_phase(const std::string& name, oo::GameState& s) {
    if (name == "terrain") {
        oo::apply_terrain_effects(s);
    } else if (name == "gold_income") {
        oo::apply_gold_income(s);
    } else if (name == "resource_income") {
        oo::apply_resource_income(s, &choice_policy, nullptr);
    } else if (name == "victory_points") {
        oo::apply_victory_points(s);
    } else if (name == "collect") {
        oo::apply_collect_phase(s, &choice_policy, nullptr);
    } else {
        return false;
    }
    return true;
}

std::string phase_of(const std::string& case_name) {
    const size_t slash = case_name.find('/');
    return slash == std::string::npos ? case_name : case_name.substr(0, slash);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_phases <phase_cases.txt>\n";
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
    if (tag != "CASES") {
        std::cerr << "malformed case file: expected CASES header\n";
        return 2;
    }

    // GameState is ~65 KB; heap-allocate rather than blow the stack with three.
    auto before = std::make_unique<oo::GameState>();
    auto expected = std::make_unique<oo::GameState>();

    int passed = 0, failures = 0, validated = 0;
    for (int i = 0; i < total; ++i) {
        std::string case_tag, case_name;
        if (!(in >> case_tag >> case_name) || case_tag != "CASE") {
            std::cerr << "malformed case file at case " << i << "\n";
            return 2;
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

        // Every Python-produced state must satisfy our own invariants too - if it
        // does not, either the invariants are wrong or the reader is.
        std::string problem;
        if (!oo::validate_state(*before, problem)) {
            std::cerr << "FAIL " << case_name << ": input state invalid: " << problem << "\n";
            ++failures;
            continue;
        }
        ++validated;

        const std::string phase = phase_of(case_name);
        if (!run_phase(phase, *before)) {
            std::cerr << "unknown phase '" << phase << "' in case " << case_name << "\n";
            return 2;
        }

        std::string diff;
        if (!oo::compare_states(*before, *expected, diff)) {
            if (++failures <= 15) {
                std::cerr << "FAIL " << case_name << ": " << diff << "\n";
            }
            continue;
        }
        if (!oo::validate_state(*before, problem)) {
            std::cerr << "FAIL " << case_name << ": output state invalid: " << problem << "\n";
            ++failures;
            continue;
        }
        ++passed;
    }

    std::printf("test_phases: %d/%d cases passed (%d inputs validated), %d failures\n", passed,
                total, validated, failures);
    return failures == 0 ? 0 : 1;
}
