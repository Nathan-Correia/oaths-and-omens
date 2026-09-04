// High-volume RNG parity: many seeds x many draws, compared by hash.
//
// cycle() below MUST stay identical to dump_rng_stress.py's stress_hash(). See
// that file for why this duplication is accepted, and use the self-describing
// golden trace (test_rng.cpp) to diagnose any mismatch this reports - a hash
// tells you THAT something diverged, never what.
//
// Usage: test_rng_stress <path-to-rng_stress.txt>

#include "oo/rng.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

namespace {

constexpr uint64_t kFnvOffset = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

inline uint64_t fnv1a_u64(uint64_t h, uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        h ^= (value >> shift) & 0xFFull;
        h *= kFnvPrime;
    }
    return h;
}

uint64_t cycle(int64_t seed, int ops) {
    oo::Rng rng(seed);
    uint64_t h = kFnvOffset;
    for (int i = 0; i < ops; ++i) {
        switch (i % 7) {
            case 0:
                h = fnv1a_u64(h, rng.getrandbits(1 + (i % 64)));
                break;
            case 1:
                h = fnv1a_u64(h, rng.randbelow(static_cast<uint64_t>(1 + (i % 1000))));
                break;
            case 2:
                h = fnv1a_u64(h, static_cast<uint64_t>(rng.randint(1, 20)));
                break;
            case 3: {
                const double d = rng.random();
                uint64_t bits;
                std::memcpy(&bits, &d, sizeof(bits));
                h = fnv1a_u64(h, bits);
                break;
            }
            case 4:
                h = fnv1a_u64(h, static_cast<uint64_t>(rng.randrange(1, 2 + (i % 100))));
                break;
            case 5:
                h = fnv1a_u64(h, static_cast<uint64_t>(rng.choice_index(
                                     static_cast<size_t>(1 + (i % 50)))));
                break;
            default: {
                const int n = 5 + (i % 300);
                const int k = 1 + (i % 4);
                for (int v : rng.sample_indices(n, k)) {
                    h = fnv1a_u64(h, static_cast<uint64_t>(v));
                }
                break;
            }
        }
    }
    return h;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_rng_stress <rng_stress.txt>\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 2;
    }

    std::string tag;
    int seeds = 0, ops = 0;
    in >> tag >> seeds >> ops;
    if (tag != "CONFIG") {
        std::cerr << "malformed stress file: expected CONFIG header\n";
        return 2;
    }

    int checked = 0, failures = 0;
    int64_t seed;
    std::string want_hex;
    while (in >> seed >> want_hex) {
        const uint64_t want = std::stoull(want_hex, nullptr, 16);
        const uint64_t got = cycle(seed, ops);
        ++checked;
        if (got != want) {
            if (++failures <= 10) {
                std::fprintf(stderr, "FAIL seed %lld: got %016llx, want %016llx\n",
                             static_cast<long long>(seed),
                             static_cast<unsigned long long>(got),
                             static_cast<unsigned long long>(want));
            }
        }
    }

    if (checked != seeds) {
        std::fprintf(stderr, "WARNING: header promised %d seeds, read %d\n", seeds, checked);
    }
    std::printf("test_rng_stress: %d seeds x %d ops (~%lld draws), %d failures\n",
                checked, ops, static_cast<long long>(checked) * ops, failures);
    return failures == 0 ? 0 : 1;
}
