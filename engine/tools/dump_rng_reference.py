"""
Emits a golden trace of CPython random.Random calls for the C++ Rng to replay.

See engine/PLAN.md §3.1. The output is deliberately self-describing: each line is
an operation plus the result CPython produced, so tests/test_rng.cpp doesn't
duplicate the battery - it parses the file, performs the same op on oo::Rng, and
compares. Adding a case here automatically tests it on the C++ side.

Usage:  python engine/tools/dump_rng_reference.py > engine/tests/data/rng_golden.txt

Line formats (all whitespace-separated):
  SEED        <int64>                    reseed the generator
  RANDOM      <hexfloat>                 random()
  GETRANDBITS <k> <result>               getrandbits(k)
  RANDBELOW   <n> <result>               _randbelow(n)
  RANDINT     <a> <b> <result>           randint(a, b)
  RANDRANGE1  <stop> <result>            randrange(stop)
  RANDRANGE2  <start> <stop> <result>    randrange(start, stop)
  CHOICE      <n> <result>               choice(range(n))
  SHUFFLE     <n> <perm...>              shuffle(list(range(n)))
  SAMPLE      <n> <k> <picked...>        sample(range(n), k)
  CHOICES     <n> <w...> <result>        choices(range(n), weights=w, k=1)[0]

random() is written as a hex float so the comparison is exact rather than
round-trip-through-decimal.
"""

import random
import sys

# Seeds worth covering: zero, small, a value crossing the 32-bit boundary (two
# init_by_array key words instead of one), a negative one (CPython seeds on
# abs(n)), and the exact shapes this codebase generates - `seed * 1_000_003 + f`
# from every make_X_agents, and `seed + 999_983` from tactician's opponent model.
SEEDS = [
    0, 1, 2, 42, 12345, 999_983,
    2 ** 31 - 1, 2 ** 31, 2 ** 32 - 1, 2 ** 32, 2 ** 32 + 1,
    -1, -42, -(2 ** 32),
    42 * 1_000_003 + 0, 42 * 1_000_003 + 7,
    1234567 * 1_000_003 + 3,
    2 ** 53, 2 ** 62,
]


def emit(line):
    sys.stdout.write(line + "\n")


def battery(rng):
    """One pass of mixed operations. Interleaved on purpose: a bug that consumes
    the wrong NUMBER of draws (rather than the wrong value) only shows up when a
    later op reads from a desynced stream position."""

    for _ in range(20):
        emit(f"RANDOM {rng.random().hex()}")

    # Every width, including the k <= 32 fast path, the exact 32 boundary, and
    # multi-word fills where CPython's least-significant-word-first ordering and
    # last-word truncation both matter.
    for k in list(range(1, 65)):
        emit(f"GETRANDBITS {k} {rng.getrandbits(k)}")

    # _randbelow's rejection loop: values just under, at, and just over powers of
    # two exercise the retry path at very different rates. n == 1 is the case
    # that forces bit_length(n) rather than bit_length(n - 1).
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 20, 31, 32, 33,
              100, 127, 128, 129, 255, 256, 1000, 65535, 65536,
              2 ** 31 - 1, 2 ** 31, 2 ** 32 + 1, 2 ** 40]:
        emit(f"RANDBELOW {n} {rng._randbelow(n)}")

    # The engine's actual battle roll, plus asymmetric ranges.
    for _ in range(20):
        emit(f"RANDINT 1 20 {rng.randint(1, 20)}")
    for a, b in [(0, 0), (0, 1), (-5, 5), (3, 3), (-100, -50), (0, 2 ** 31 - 1)]:
        emit(f"RANDINT {a} {b} {rng.randint(a, b)}")

    # tactician_agent's per-decision rollout seed.
    for _ in range(5):
        emit(f"RANDRANGE1 {2 ** 31} {rng.randrange(2 ** 31)}")
    for stop in [1, 2, 7, 169, 217, 331]:
        emit(f"RANDRANGE1 {stop} {rng.randrange(stop)}")
    for start, stop in [(0, 6), (1, 21), (-3, 4)]:
        emit(f"RANDRANGE2 {start} {stop} {rng.randrange(start, stop)}")

    for n in [1, 2, 6, 8, 169, 217, 331]:
        emit(f"CHOICE {n} {rng.choice(range(n))}")

    # greedy_agent / hussar_agent shuffle small candidate lists.
    for n in [0, 1, 2, 3, 6, 8, 20]:
        x = list(range(n))
        rng.shuffle(x)
        emit(f"SHUFFLE {n} " + " ".join(map(str, x)))

    # Both of sample's branches must be covered - they consume different numbers
    # of draws, so choosing wrongly desyncs everything after it.
    #   n <= setsize  -> pool branch    (placement.py: sample(range(f), f))
    #   n >  setsize  -> selected branch (random_agent: sample(legal, k<=3))
    # setsize is 21 for k <= 5, and grows by 4**ceil(log(3k, 4)) beyond that.
    sample_cases = [
        (4, 4), (6, 6), (8, 8), (10, 10),      # pool branch, k == n
        (21, 3), (22, 3), (21, 1), (22, 1),    # exactly astride the setsize edge
        (100, 0), (100, 1), (100, 2), (100, 3),  # selected branch
        (400, 3), (169, 2),
        (30, 6), (30, 10), (85, 6), (86, 6),   # k > 5 -> setsize formula in play
        (200, 8), (500, 12),
    ]
    for n, k in sample_cases:
        picked = rng.sample(range(n), k)
        emit(f"SAMPLE {n} {k} " + " ".join(map(str, picked)))

    # setup.py's terrain-type draw. Includes a single-element list (where
    # bisect's hi = n - 1 makes the search degenerate), heavily skewed weights,
    # and equal weights.
    choices_cases = [
        [1],
        [1, 1],
        [1, 1, 1, 1, 1],
        [120, 25, 25, 40, 40],   # BAG_COUNTS' starting distribution
        [1, 1000],
        [1000, 1],
        [3, 1, 4, 1, 5, 9, 2, 6],
    ]
    for weights in choices_cases:
        for _ in range(8):
            got = rng.choices(range(len(weights)), weights=weights, k=1)[0]
            emit(f"CHOICES {len(weights)} " + " ".join(map(str, weights)) + f" {got}")


def main():
    for seed in SEEDS:
        emit(f"SEED {seed}")
        battery(random.Random(seed))


if __name__ == "__main__":
    main()
