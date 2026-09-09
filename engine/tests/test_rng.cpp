// Properties of the xoshiro256++ generator (PLAN.md §3.4).
//
// This replaced a golden-trace test that compared several thousand draws against
// CPython's `random.Random`, byte for byte. That test was exactly right while the
// Python engine existed and completely pointless afterwards: it pinned
// compatibility with something that is no longer in the repo.
//
// What still matters is different, so this test is different. Nothing here has a
// reference file. It asserts the properties the ENGINE actually relies on:
// reproducibility from a seed, independence of nearby seeds, correct ranges, and
// enough uniformity that the game's dice are not visibly loaded. A golden trace
// would only say "the generator still does what it did yesterday", which the
// reblessed corpus already covers.
//
// Deliberately NOT a statistical test suite. xoshiro256++ passes BigCrush; that
// is the authors' evidence, not something to re-derive here. These are smoke
// tests for wiring mistakes - a shift in the wrong direction, an off-by-one in a
// range, a seeding path that collapses.

#include "oo/rng.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++g_failures;
    }
}

}  // namespace

int main() {
    // --- reproducibility, which is the whole contract (§7) -------------------
    {
        oo::Rng a(12345), b(12345);
        bool same = true;
        for (int i = 0; i < 10000; ++i) same = same && (a.next() == b.next());
        check(same, "the same seed gives the same stream");

        oo::Rng c(12345);
        for (int i = 0; i < 500; ++i) c.next();
        oo::Rng d = c;  // tactician copies a seeded Rng per rollout candidate
        bool copy_same = true;
        for (int i = 0; i < 1000; ++i) copy_same = copy_same && (c.next() == d.next());
        check(copy_same, "a copied Rng continues the identical stream");

        check(sizeof(oo::Rng) == 32, "Rng stays 32 bytes");
    }

    // --- ADJACENT seeds must not be correlated -------------------------------
    // This is the property SplitMix64 seeding exists for, and the one that would
    // actually bite: oo_run runs game g with seed+g, and agents are seeded
    // seed*1000003 + faction. If nearby seeds produced nearby streams, a batch of
    // games would share dice and every aggregate result would be quietly wrong.
    {
        int collisions = 0;
        for (int64_t s = 1000; s < 1100; ++s) {
            oo::Rng x(s), y(s + 1);
            int matches = 0;
            for (int i = 0; i < 8; ++i) {
                if (x.next() == y.next()) ++matches;
            }
            if (matches > 1) ++collisions;
        }
        check(collisions == 0, "seeds s and s+1 produce unrelated streams");

        std::set<uint64_t> firsts;
        for (int64_t s = 0; s < 2000; ++s) firsts.insert(oo::Rng(s).next());
        check(firsts.size() == 2000, "2000 consecutive seeds give 2000 distinct first draws");

        // Seed 0 must not degenerate - xoshiro is all-zero-absorbing, so a naive
        // seeding path would produce a stuck generator here.
        oo::Rng zero(0);
        bool nonzero = false;
        for (int i = 0; i < 10; ++i) nonzero = nonzero || (zero.next() != 0);
        check(nonzero, "seed 0 does not collapse to an all-zero state");
    }

    // --- random() in [0, 1) --------------------------------------------------
    {
        oo::Rng r(7);
        bool in_range = true;
        double lo = 1.0, hi = 0.0;
        for (int i = 0; i < 200000; ++i) {
            const double v = r.random();
            in_range = in_range && (v >= 0.0 && v < 1.0);
            lo = v < lo ? v : lo;
            hi = v > hi ? v : hi;
        }
        check(in_range, "random() stays in [0, 1)");
        check(lo < 0.001 && hi > 0.999, "random() spans the interval");
    }

    // --- randbelow: range, and uniformity good enough for dice ---------------
    {
        oo::Rng r(99);
        for (uint64_t n : {1ull, 2ull, 3ull, 6ull, 7ull, 100ull, 1000ull}) {
            bool in_range = true;
            for (int i = 0; i < 5000; ++i) in_range = in_range && (r.randbelow(n) < n);
            check(in_range, "randbelow(" + std::to_string(n) + ") stays below n");
        }

        // A d6, which is what the battle dice actually are.
        int counts[6] = {};
        const int rolls = 600000;
        for (int i = 0; i < rolls; ++i) ++counts[r.randbelow(6)];
        const double expected = rolls / 6.0;
        double chi2 = 0.0;
        for (int i = 0; i < 6; ++i) {
            const double d = counts[i] - expected;
            chi2 += d * d / expected;
        }
        // 5 df, p=0.001 critical value is 20.5. A fair die clears this
        // essentially always; a broken one (a stuck bit, a bad modulo) does not.
        check(chi2 < 20.5, "randbelow(6) is uniform (chi2 = " + std::to_string(chi2) + ")");
    }

    // --- randint is inclusive at BOTH ends ------------------------------------
    {
        oo::Rng r(3);
        bool saw_lo = false, saw_hi = false, in_range = true;
        for (int i = 0; i < 20000; ++i) {
            const int64_t v = r.randint(3, 9);
            in_range = in_range && v >= 3 && v <= 9;
            saw_lo = saw_lo || v == 3;
            saw_hi = saw_hi || v == 9;
        }
        check(in_range, "randint stays within [a, b]");
        check(saw_lo && saw_hi, "randint reaches both endpoints");
        check(oo::Rng(1).randint(5, 5) == 5, "randint(n, n) is n");
    }

    // --- shuffle is a permutation, and mixes -----------------------------------
    {
        oo::Rng r(11);
        bool permutation = true;
        std::vector<int> position_hits(8, 0);
        for (int trial = 0; trial < 20000; ++trial) {
            std::vector<int> v(8);
            for (int i = 0; i < 8; ++i) v[static_cast<size_t>(i)] = i;
            r.shuffle(v);
            std::set<int> seen(v.begin(), v.end());
            permutation = permutation && seen.size() == 8;
            // Track where element 0 lands, to catch a shuffle that never moves it.
            for (int i = 0; i < 8; ++i) {
                if (v[static_cast<size_t>(i)] == 0) ++position_hits[static_cast<size_t>(i)];
            }
        }
        check(permutation, "shuffle always yields a permutation");
        bool spread = true;
        for (int i = 0; i < 8; ++i) spread = spread && position_hits[static_cast<size_t>(i)] > 2000;
        check(spread, "shuffle sends element 0 to every position");

        std::vector<int> one{42};
        r.shuffle(one);
        check(one.size() == 1 && one[0] == 42, "shuffle of one element is a no-op");
        std::vector<int> none;
        r.shuffle(none);  // must not read off the end
        check(none.empty(), "shuffle of an empty vector is safe");
    }

    // --- sample_indices: both branches -----------------------------------------
    {
        oo::Rng r(5);
        // Dense (k*3 >= n): the partial Fisher-Yates path. This is what
        // placement.cpp uses, with k == n, i.e. a full permutation.
        for (int trial = 0; trial < 2000; ++trial) {
            const std::vector<int> s = r.sample_indices(8, 8);
            std::set<int> seen(s.begin(), s.end());
            check(s.size() == 8 && seen.size() == 8 && *seen.begin() == 0 && *seen.rbegin() == 7,
                  "sample(n, n) is a permutation of [0, n)");
            if (g_failures) break;
        }
        // Sparse (k*3 < n): the retry path, which is what the random agent uses.
        for (int trial = 0; trial < 2000; ++trial) {
            const std::vector<int> s = r.sample_indices(500, 3);
            std::set<int> seen(s.begin(), s.end());
            bool ok = s.size() == 3 && seen.size() == 3;
            for (int v : s) ok = ok && v >= 0 && v < 500;
            check(ok, "sparse sample gives k distinct indices in range");
            if (g_failures) break;
        }
        check(r.sample_indices(5, 0).empty(), "sample of 0 is empty");
    }

    // --- choices_index respects the weights -------------------------------------
    {
        oo::Rng r(13);
        // A zero-weight bucket must NEVER be chosen. This is the property
        // setup.cpp's terrain bag depends on: an exhausted type has weight 0 and
        // picking it would place terrain that is not available.
        const std::vector<double> w{3.0, 0.0, 1.0, 0.0};
        int counts[4] = {};
        const int draws = 200000;
        for (int i = 0; i < draws; ++i) ++counts[r.choices_index(w)];
        check(counts[1] == 0 && counts[3] == 0, "zero-weight buckets are never chosen");
        const double ratio = double(counts[0]) / double(counts[2]);
        check(ratio > 2.9 && ratio < 3.1,
              "weights are respected 3:1 (got " + std::to_string(ratio) + ")");
        check(oo::Rng(1).choices_index({1.0}) == 0, "a single bucket is always index 0");
        // The last bucket must be reachable - CPython's bisect quirk (hi = n - 1)
        // used to make this subtly wrong, and it is why the quirk was dropped.
        const std::vector<double> tail{1.0, 1.0, 1.0};
        int saw_last = 0;
        oo::Rng t(17);
        for (int i = 0; i < 10000; ++i) {
            if (t.choices_index(tail) == 2) ++saw_last;
        }
        check(saw_last > 3000, "the last bucket is reachable");
    }

    std::printf("test_rng: %d property checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
