// Shared plumbing for the golden-file `--rewrite` mode (PLAN.md §3.4).
//
// The seed-derived golden files were produced by the Python reference, which no
// longer exists (M8). When the engine's behaviour changes deliberately — the RNG
// swap, the movement clamp (§11) — those files have to be reblessed, and there
// is nothing left to regenerate them from.
//
// The trick is that the INPUTS are already stored in the files: seeds, agent
// assignments, before-states, action lists, recorded decision traces. So a
// rewrite does not need to reproduce the Python scenario *generators* (~2 000
// lines). It parses a file with the reader the test already has, keeps every
// input field byte-for-byte, recomputes only the expected outputs, and writes
// the file back.
//
// THIS VALIDATES ITSELF, which is the whole reason it is safe. Run --rewrite
// with the engine UNCHANGED and the output must be byte-identical to the input.
// Any difference is a bug in the rewrite path, not a change in the engine. Only
// once that passes should the engine change and the files be reblessed for real.
// `tools/rewrite_goldens.ps1` runs exactly that check.

#pragma once

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

namespace oo_test {

// Pulls `<flag> <path>` out of argv. Returns true if present, and shrinks argc
// so the caller's positional-argument handling is unaffected.
inline bool take_flag(int& argc, char** argv, const char* flag, std::string& out_path) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], flag) != 0) continue;
        if (i + 1 >= argc) {
            std::cerr << flag << " needs an output path\n";
            std::exit(2);
        }
        out_path = argv[i + 1];
        for (int j = i; j + 2 < argc; ++j) argv[j] = argv[j + 2];
        argc -= 2;
        return true;
    }
    return false;
}

inline bool take_rewrite_flag(int& argc, char** argv, std::string& out_path) {
    return take_flag(argc, argv, "--rewrite", out_path);
}

// An output file that is only opened when rewriting, so every test can write
// `if (rw) rw.out << ...` without branching on a mode flag everywhere.
struct Rewriter {
    bool active = false;
    std::ofstream out;

    // Binary mode: text mode on Windows turns '\n' into "\r\n" and the file
    // would no longer be byte-comparable with the checked-in golden.
    bool open(const std::string& path) {
        out.open(path, std::ios::binary);
        if (!out) {
            std::cerr << "cannot write " << path << "\n";
            return false;
        }
        active = true;
        return true;
    }

    explicit operator bool() const { return active; }
};

}  // namespace oo_test
