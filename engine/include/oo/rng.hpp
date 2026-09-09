// xoshiro256++ with SplitMix64 seeding.
//
// This replaced a bit-compatible reimplementation of CPython's `random.Random`
// at M8b (PLAN.md §3.4). That reimplementation existed for one reason: to make a
// C++ game and a Python game, from the same seed, produce byte-identical states
// after every phase. With the Python engine deleted at M8 there is nothing left
// to be identical to, and the cost of keeping it was real:
//
//   - 2.5 KB of state (624 words plus an index) per generator. Every agent owns
//     one, tactician copies a whole Rng per rollout candidate, and §10 wants
//     thousands of games in flight each holding one. xoshiro256++ is 32 bytes.
//   - ~4.5 % of runtime (measured: Rng::seed + genrand_uint32 + randbelow were
//     0.85 s of 18.5 s engine / 0.84 s of 20.0 s tactician).
//   - The fiddliest code in the engine. `_randbelow_with_getrandbits`, sample's
//     setsize heuristic, `genrand_res53`, and bisect's `hi = n - 1` were all
//     bug-for-bug reproductions of CPython, and every one of them was a trap for
//     anyone reading this file expecting ordinary code.
//
// WHAT IS STILL LOAD-BEARING. Determinism did not stop mattering, it only
// stopped being cross-language. A game must still be a pure function of its seed
// (PLAN.md §7's threading contract depends on it), and reproducibility should
// survive a change of compiler or platform. So:
//
//   Do NOT use <random>. std::mt19937 seeds differently per implementation, and
//   std::uniform_int_distribution and std::shuffle are explicitly
//   implementation-defined - MSVC and libstdc++ produce different sequences from
//   the same seed. Everything below is written out for that reason, not from
//   not-invented-here.
//
// Algorithms: xoshiro256++ 1.0 and SplitMix64, both by Blackman and Vigna, both
// public domain. xoshiro256++ has a 2^256-1 period, passes BigCrush, and is a
// few instructions per draw.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace oo {

class Rng {
public:
    Rng() { seed(0); }
    explicit Rng(int64_t s) { seed(s); }

    // SplitMix64 expands the 64-bit seed into 256 bits of state. Seeding a
    // xoshiro generator directly from a small integer is the classic way to get
    // correlated streams out of nearby seeds; SplitMix64 is the author's
    // prescribed fix. It matters here because seeds ARE nearby - `oo_run` uses
    // seed+0, seed+1, seed+2..., and agents are seeded seed*1000003 + faction.
    void seed(int64_t s) {
        uint64_t z = static_cast<uint64_t>(s);
        for (int i = 0; i < 4; ++i) {
            z += 0x9e3779b97f4a7c15ull;
            uint64_t x = z;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
            s_[i] = x ^ (x >> 31);
        }
    }

    // xoshiro256++ next().
    uint64_t next() {
        const uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform in [0, 1). 53 significant bits, which is every double the interval
    // can represent at full precision; the multiply is exact.
    double random() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }

    // Low k bits. Kept because it is a natural primitive, though nothing in the
    // engine needs it any more now that randbelow does not go through it.
    uint64_t getrandbits(int k) {
        assert(k >= 0 && k <= 64);
        if (k == 0) return 0;
        if (k == 64) return next();
        return next() >> (64 - k);
    }

    // Uniform in [0, n), unbiased, and portable: no 128-bit multiply, so it
    // behaves identically on every compiler.
    //
    // `2^64 mod n` is the size of the short tail at the bottom of the range that
    // would otherwise be over-represented by a plain modulo. Rejecting draws that
    // land in it leaves an exact multiple of n, so the modulo is uniform. The
    // rejection probability is under n/2^64 - unmeasurable for any n here.
    uint64_t randbelow(uint64_t n) {
        assert(n > 0);
        if (n <= 1) return 0;
        const uint64_t reject_below = (~n + 1u) % n;  // 2^64 mod n
        uint64_t r;
        do {
            r = next();
        } while (r < reject_below);
        return r % n;
    }

    int64_t randrange(int64_t stop) {
        assert(stop > 0);
        return static_cast<int64_t>(randbelow(static_cast<uint64_t>(stop)));
    }
    int64_t randrange(int64_t start, int64_t stop) {
        assert(stop > start);
        return start + static_cast<int64_t>(randbelow(static_cast<uint64_t>(stop - start)));
    }

    // Inclusive of both ends, like Python's randint.
    int64_t randint(int64_t a, int64_t b) { return randrange(a, b + 1); }

    template <class T>
    const T& choice(const std::vector<T>& seq) {
        assert(!seq.empty());
        return seq[static_cast<size_t>(randbelow(seq.size()))];
    }

    int choice_index(int n) {
        assert(n > 0);
        return static_cast<int>(randbelow(static_cast<uint64_t>(n)));
    }

    // Fisher-Yates, descending. Written out rather than std::shuffle because
    // std::shuffle's draw pattern is implementation-defined.
    template <class T>
    void shuffle(T* x, size_t n) {
        for (int64_t i = static_cast<int64_t>(n) - 1; i >= 1; --i) {
            const int64_t j = static_cast<int64_t>(randbelow(static_cast<uint64_t>(i + 1)));
            T tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
    }

    template <class T>
    void shuffle(std::vector<T>& x) {
        shuffle(x.data(), x.size());
    }

    // k distinct indices from [0, n), in selection order.
    //
    // Two strategies, chosen by density rather than by CPython's old `setsize`
    // heuristic: a partial Fisher-Yates over a scratch pool is O(n) in time and
    // memory, which is the wrong trade when k is tiny and n is large (the random
    // agent samples 3 of several hundred legal actions). Above that, retry-until-
    // unseen would spin.
    std::vector<int> sample_indices(int n, int k) {
        assert(k >= 0 && k <= n);
        std::vector<int> result(static_cast<size_t>(k));

        if (static_cast<int64_t>(k) * 3 >= n) {
            std::vector<int> pool(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) pool[static_cast<size_t>(i)] = i;
            for (int i = 0; i < k; ++i) {
                const int j = static_cast<int>(randbelow(static_cast<uint64_t>(n - i)));
                result[static_cast<size_t>(i)] = pool[static_cast<size_t>(j)];
                pool[static_cast<size_t>(j)] = pool[static_cast<size_t>(n - i - 1)];
            }
        } else {
            // Sparse: draw and retry. With k*3 < n the expected retries per pick
            // are below 0.5, so this terminates quickly.
            std::vector<char> taken(static_cast<size_t>(n), 0);
            for (int i = 0; i < k; ++i) {
                int j = static_cast<int>(randbelow(static_cast<uint64_t>(n)));
                while (taken[static_cast<size_t>(j)]) {
                    j = static_cast<int>(randbelow(static_cast<uint64_t>(n)));
                }
                taken[static_cast<size_t>(j)] = 1;
                result[static_cast<size_t>(i)] = j;
            }
        }
        return result;
    }

    // One index, chosen with probability proportional to its weight.
    //
    // Straight binary search now, without CPython's `hi = n - 1` quirk. Rounding
    // in the accumulation can leave `x` fractionally above the final cumulative
    // total, so the result is clamped rather than allowed off the end.
    int choices_index(const std::vector<double>& weights) {
        assert(!weights.empty());
        std::vector<double> cum(weights.size());
        double running = 0.0;
        for (size_t i = 0; i < weights.size(); ++i) {
            running += weights[i];
            cum[i] = running;
        }
        const double total = cum.back();
        assert(total > 0.0);
        const double x = random() * total;

        int lo = 0;
        int hi = static_cast<int>(cum.size());
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (x < cum[static_cast<size_t>(mid)]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        const int last = static_cast<int>(cum.size()) - 1;
        return lo > last ? last : lo;
    }

private:
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    uint64_t s_[4] = {};
};

static_assert(sizeof(Rng) == 32, "Rng is copied per rollout candidate and lives per game; "
                                 "keep it small (PLAN.md §3.4)");

}  // namespace oo
