// Replays the golden trace from tools/dump_rng_reference.py against oo::Rng.
//
// The trace is self-describing (each line is an operation plus CPython's result),
// so this file contains no test battery of its own - extending the Python dumper
// automatically extends what is checked here. See engine/PLAN.md §3.1.
//
// Usage: test_rng <path-to-rng_golden.txt>

#include "oo/rng.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;
std::string g_line;
int g_line_no = 0;

void fail(const std::string& what) {
    if (++g_failures <= 20) {
        std::cerr << "FAIL line " << g_line_no << ": " << what << "\n"
                  << "      " << g_line << "\n";
    }
}

template <class T>
void expect_eq(const T& got, const T& want, const char* what) {
    ++g_checks;
    if (!(got == want)) {
        std::ostringstream os;
        os << what << ": got " << got << ", want " << want;
        fail(os.str());
    }
}

// random() is compared bit-exactly. The golden file stores CPython's float.hex(),
// which strtod parses back to the identical double - no decimal round-trip.
void expect_double_bits(double got, double want) {
    ++g_checks;
    if (std::memcmp(&got, &want, sizeof(double)) != 0) {
        std::ostringstream os;
        os.precision(17);
        os << "random(): got " << got << ", want " << want;
        fail(os.str());
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_rng <rng_golden.txt>\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 2;
    }

    oo::Rng rng;
    int seeds_seen = 0;

    while (std::getline(in, g_line)) {
        ++g_line_no;
        if (g_line.empty()) continue;
        std::istringstream ls(g_line);
        std::string op;
        ls >> op;

        if (op == "SEED") {
            int64_t s;
            ls >> s;
            rng.seed(s);
            ++seeds_seen;

        } else if (op == "RANDOM") {
            std::string hexfloat;
            ls >> hexfloat;
            const double want = std::strtod(hexfloat.c_str(), nullptr);
            expect_double_bits(rng.random(), want);

        } else if (op == "GETRANDBITS") {
            int k;
            uint64_t want;
            ls >> k >> want;
            expect_eq(rng.getrandbits(k), want, "getrandbits");

        } else if (op == "RANDBELOW") {
            uint64_t n, want;
            ls >> n >> want;
            expect_eq(rng.randbelow(n), want, "randbelow");

        } else if (op == "RANDINT") {
            int64_t a, b, want;
            ls >> a >> b >> want;
            expect_eq(rng.randint(a, b), want, "randint");

        } else if (op == "RANDRANGE1") {
            int64_t stop, want;
            ls >> stop >> want;
            expect_eq(rng.randrange(stop), want, "randrange");

        } else if (op == "RANDRANGE2") {
            int64_t start, stop, want;
            ls >> start >> stop >> want;
            expect_eq(rng.randrange(start, stop), want, "randrange");

        } else if (op == "CHOICE") {
            size_t n, want;
            ls >> n >> want;
            expect_eq(rng.choice_index(n), want, "choice");

        } else if (op == "SHUFFLE") {
            int n;
            ls >> n;
            std::vector<int> want(static_cast<size_t>(n));
            for (int& v : want) ls >> v;
            std::vector<int> got(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) got[static_cast<size_t>(i)] = i;
            rng.shuffle(got);
            expect_eq(got == want, true, "shuffle");

        } else if (op == "SAMPLE") {
            int n, k;
            ls >> n >> k;
            std::vector<int> want(static_cast<size_t>(k));
            for (int& v : want) ls >> v;
            expect_eq(rng.sample_indices(n, k) == want, true, "sample");

        } else if (op == "CHOICES") {
            int n;
            ls >> n;
            std::vector<double> weights(static_cast<size_t>(n));
            for (double& w : weights) ls >> w;
            int want;
            ls >> want;
            expect_eq(rng.choices_index(weights), want, "choices");

        } else {
            fail("unknown op '" + op + "'");
        }
    }

    std::printf("test_rng: %d checks across %d seeds, %d failures\n",
                g_checks, seeds_seen, g_failures);
    if (g_failures > 20) {
        std::printf("  (%d further failures not shown)\n", g_failures - 20);
    }
    return g_failures == 0 ? 0 : 1;
}
