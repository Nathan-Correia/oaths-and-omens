"""
High-volume RNG parity check: many seeds x many draws, compared by hash.

The golden trace from dump_rng_reference.py is exhaustive about EDGE CASES but
covers only a handful of seeds, and writing out 10^7 individual results would
make an unusably large file. This instead runs a fixed, mixed operation cycle for
`ops` iterations per seed and emits one 64-bit FNV-1a hash of every result, so
huge volumes compress to one line per seed.

The tradeoff: the operation cycle below is duplicated in tests/test_rng_stress.cpp
and the two MUST be kept in lockstep. It is deliberately short and mechanical for
that reason. A mismatch shows up as a differing hash with no indication of which
op diverged - when that happens, narrow it down with the self-describing golden
trace from dump_rng_reference.py, which is built for diagnosis rather than volume.

Usage:  python engine/tools/dump_rng_stress.py [seeds] [ops] > engine/tests/data/rng_stress.txt
"""

import random
import struct
import sys

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def fnv1a_u64(h, value):
    """Fold one uint64 into the running hash, byte by byte, little-endian."""
    for shift in range(0, 64, 8):
        h ^= (value >> shift) & 0xFF
        h = (h * FNV_PRIME) & MASK64
    return h


def stress_hash(seed, ops):
    """MUST stay identical to test_rng_stress.cpp's cycle()."""
    rng = random.Random(seed)
    h = FNV_OFFSET
    for i in range(ops):
        t = i % 7
        if t == 0:
            h = fnv1a_u64(h, rng.getrandbits(1 + (i % 64)))
        elif t == 1:
            h = fnv1a_u64(h, rng._randbelow(1 + (i % 1000)))
        elif t == 2:
            h = fnv1a_u64(h, rng.randint(1, 20))
        elif t == 3:
            # Hash the double's raw bits so the comparison is exact.
            h = fnv1a_u64(h, struct.unpack("<Q", struct.pack("<d", rng.random()))[0])
        elif t == 4:
            h = fnv1a_u64(h, rng.randrange(1, 2 + (i % 100)))
        elif t == 5:
            h = fnv1a_u64(h, rng.choice(range(1 + (i % 50))))
        else:
            n = 5 + (i % 300)          # straddles sample's setsize branch point
            k = 1 + (i % 4)
            for v in rng.sample(range(n), k):
                h = fnv1a_u64(h, v)
    return h


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    ops = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    sys.stdout.write(f"CONFIG {seeds} {ops}\n")
    for seed in range(seeds):
        sys.stdout.write(f"{seed} {stress_hash(seed, ops):016x}\n")


if __name__ == "__main__":
    main()
