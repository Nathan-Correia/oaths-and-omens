// Bit-compatible reimplementation of CPython's `random.Random`.
//
// WHY THIS EXISTS (see engine/PLAN.md §3.1): the Python engine threads one
// random.Random through a whole turn and consumes it in a fixed order that
// engine_old/battle.py's docstring calls out as load-bearing. Reproducing that
// stream exactly is what lets a C++ game and a Python game, given the same seed,
// produce byte-identical states after every phase - which turns "is the port
// correct?" from an argument into a test.
//
// Every method below mirrors a specific piece of CPython, and the reference is
// named in a comment so divergences can be checked against the source:
//   - Modules/_randommodule.c  for the Mersenne Twister core, seeding, random(),
//     and getrandbits()
//   - Lib/random.py            for _randbelow, randrange, randint, choice,
//     sample, shuffle and choices
//
// Do NOT "improve" any of this. Every apparent oddity (bisect's `hi = n - 1`,
// sample's setsize heuristic, getrandbits' word order) is load-bearing for parity.
// It is also fast enough to keep permanently: a game consumes a few hundred draws,
// so there is no reason to swap in PCG64 later and invalidate the golden traces.

#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace oo {

class Rng {
public:
    Rng() { seed(0); }
    explicit Rng(int64_t s) { seed(s); }

    // random.Random.seed(int) -> _randommodule.c:random_seed.
    // CPython takes abs(n), splits it into 32-bit little-endian words, and feeds
    // those to init_by_array. int64 covers every seed this codebase produces
    // (the largest is `seed * 1_000_003 + f`, well under 2^63).
    void seed(int64_t s) {
        // abs() written to stay defined at INT64_MIN.
        uint64_t n = (s < 0) ? (~static_cast<uint64_t>(s) + 1u) : static_cast<uint64_t>(s);
        uint32_t key[2];
        int key_length;
        if (n == 0) {
            key[0] = 0;
            key_length = 1;  // bits == 0 -> keyused = 1
        } else {
            key[0] = static_cast<uint32_t>(n & 0xffffffffu);
            key[1] = static_cast<uint32_t>(n >> 32);
            key_length = (key[1] != 0) ? 2 : 1;  // == (bit_length(n) - 1) / 32 + 1
        }
        init_by_array(key, key_length);
    }

    // _randommodule.c:genrand_uint32 - the standard MT19937 next-word step.
    uint32_t genrand_uint32() {
        if (index_ >= kN) {
            static constexpr uint32_t mag01[2] = {0x0u, kMatrixA};
            uint32_t y;
            int kk = 0;
            for (; kk < kN - kM; ++kk) {
                y = (mt_[kk] & kUpperMask) | (mt_[kk + 1] & kLowerMask);
                mt_[kk] = mt_[kk + kM] ^ (y >> 1) ^ mag01[y & 0x1u];
            }
            for (; kk < kN - 1; ++kk) {
                y = (mt_[kk] & kUpperMask) | (mt_[kk + 1] & kLowerMask);
                mt_[kk] = mt_[kk + (kM - kN)] ^ (y >> 1) ^ mag01[y & 0x1u];
            }
            y = (mt_[kN - 1] & kUpperMask) | (mt_[0] & kLowerMask);
            mt_[kN - 1] = mt_[kM - 1] ^ (y >> 1) ^ mag01[y & 0x1u];
            index_ = 0;
        }
        uint32_t y = mt_[index_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

    // _randommodule.c:random_random - genrand_res53, two draws per double.
    // Written in CPython's exact form; the reciprocal is a power of two so the
    // multiply is exact, but keep it verbatim rather than dividing.
    double random() {
        uint32_t a = genrand_uint32() >> 5;
        uint32_t b = genrand_uint32() >> 6;
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    // _random_Random_getrandbits_impl. Words are filled least-significant first,
    // so the FIRST draw becomes the LOW word and only the last word is truncated.
    // Capped at 64 bits, which is all this codebase ever asks for.
    uint64_t getrandbits(int k) {
        assert(k >= 0 && k <= 64);
        if (k == 0) return 0;
        if (k <= 32) return genrand_uint32() >> (32 - k);
        const int words = (k - 1) / 32 + 1;
        uint64_t result = 0;
        int remaining = k;
        for (int i = 0; i < words; ++i, remaining -= 32) {
            uint32_t r = genrand_uint32();
            if (remaining < 32) r >>= (32 - remaining);  // drop least significant bits
            result |= static_cast<uint64_t>(r) << (32 * i);
        }
        return result;
    }

    // random.py:_randbelow_with_getrandbits. Note `bit_length(n)`, not
    // `bit_length(n - 1)` - CPython uses the former so that n == 1 works.
    uint64_t randbelow(uint64_t n) {
        if (n == 0) return 0;
        const int k = std::bit_width(n);
        uint64_t r = getrandbits(k);
        while (r >= n) r = getrandbits(k);
        return r;
    }

    // random.py:randrange(stop) and randrange(start, stop).
    int64_t randrange(int64_t stop) {
        assert(stop > 0);
        return static_cast<int64_t>(randbelow(static_cast<uint64_t>(stop)));
    }
    int64_t randrange(int64_t start, int64_t stop) {
        assert(stop > start);
        return start + static_cast<int64_t>(randbelow(static_cast<uint64_t>(stop - start)));
    }

    // random.py:randint(a, b) == randrange(a, b + 1). Inclusive both ends.
    int64_t randint(int64_t a, int64_t b) { return randrange(a, b + 1); }

    // random.py:choice - seq[_randbelow(len(seq))].
    template <class T>
    const T& choice(const std::vector<T>& seq) {
        assert(!seq.empty());
        return seq[static_cast<size_t>(randbelow(seq.size()))];
    }
    // Index-only form, for callers holding something other than a vector.
    size_t choice_index(size_t n) {
        assert(n > 0);
        return static_cast<size_t>(randbelow(n));
    }

    // random.py:shuffle - Fisher-Yates walking downward, `j = randbelow(i + 1)`.
    template <class T>
    void shuffle(std::vector<T>& x) {
        for (int64_t i = static_cast<int64_t>(x.size()) - 1; i >= 1; --i) {
            const int64_t j = static_cast<int64_t>(randbelow(static_cast<uint64_t>(i + 1)));
            std::swap(x[static_cast<size_t>(i)], x[static_cast<size_t>(j)]);
        }
    }

    // random.py:sample, for a population of `n` items, returning the chosen
    // INDICES (equivalently, the result of sample(range(n), k)).
    //
    // Both of CPython's branches are implemented because both are genuinely
    // reachable here: placement.py's sample(range(num_factions), num_factions)
    // takes the pool branch, while random_agent's sample(legal, k<=3) over a
    // few hundred legal actions takes the selected-set branch. They consume
    // different numbers of draws, so picking the wrong one silently desyncs
    // everything downstream.
    std::vector<int> sample_indices(int n, int k) {
        assert(k >= 0 && k <= n);
        std::vector<int> result(static_cast<size_t>(k));

        int setsize = 21;  // size of a small set minus size of an empty list
        if (k > 5) {
            setsize += static_cast<int>(
                std::pow(4.0, std::ceil(std::log(static_cast<double>(k) * 3.0) / std::log(4.0))));
        }

        if (n <= setsize) {
            std::vector<int> pool(static_cast<size_t>(n));
            std::iota(pool.begin(), pool.end(), 0);
            for (int i = 0; i < k; ++i) {
                const int j = static_cast<int>(randbelow(static_cast<uint64_t>(n - i)));
                result[static_cast<size_t>(i)] = pool[static_cast<size_t>(j)];
                pool[static_cast<size_t>(j)] = pool[static_cast<size_t>(n - i - 1)];
            }
        } else {
            // CPython uses a set purely for membership - iteration order is never
            // observed, so a flat bitmap reproduces it exactly.
            std::vector<char> selected(static_cast<size_t>(n), 0);
            for (int i = 0; i < k; ++i) {
                int j = static_cast<int>(randbelow(static_cast<uint64_t>(n)));
                while (selected[static_cast<size_t>(j)]) {
                    j = static_cast<int>(randbelow(static_cast<uint64_t>(n)));
                }
                selected[static_cast<size_t>(j)] = 1;
                result[static_cast<size_t>(i)] = j;
            }
        }
        return result;
    }

    // random.py:choices(population, weights=..., k=1), returning the chosen index.
    // One random() draw, then bisect_right over the accumulated weights - note
    // CPython passes `hi = n - 1`, not `n`, which genuinely changes the result
    // for a draw landing in the last bucket.
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
        int hi = static_cast<int>(cum.size()) - 1;  // CPython's bisect hi
        while (lo < hi) {
            const int mid = (lo + hi) / 2;  // Python's (lo + hi) // 2
            if (x < cum[static_cast<size_t>(mid)]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

private:
    static constexpr int kN = 624;
    static constexpr int kM = 397;
    static constexpr uint32_t kMatrixA = 0x9908b0dfu;
    static constexpr uint32_t kUpperMask = 0x80000000u;
    static constexpr uint32_t kLowerMask = 0x7fffffffu;

    // _randommodule.c:init_genrand
    void init_genrand(uint32_t s) {
        mt_[0] = s;
        for (int i = 1; i < kN; ++i) {
            mt_[i] = (1812433253u * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + static_cast<uint32_t>(i));
        }
        index_ = kN;
    }

    // _randommodule.c:init_by_array
    void init_by_array(const uint32_t* init_key, int key_length) {
        init_genrand(19650218u);
        int i = 1;
        int j = 0;
        int k = std::max(kN, key_length);
        for (; k; --k) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525u)) +
                     init_key[j] + static_cast<uint32_t>(j);
            ++i;
            ++j;
            if (i >= kN) {
                mt_[0] = mt_[kN - 1];
                i = 1;
            }
            if (j >= key_length) j = 0;
        }
        for (k = kN - 1; k; --k) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941u)) -
                     static_cast<uint32_t>(i);
            ++i;
            if (i >= kN) {
                mt_[0] = mt_[kN - 1];
                i = 1;
            }
        }
        mt_[0] = 0x80000000u;  // MSB is 1; assuring non-zero initial array
        index_ = kN;
    }

    uint32_t mt_[kN] = {};
    int index_ = kN;
};

}  // namespace oo
