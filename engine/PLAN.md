# C++ Engine Port — Plan

Frozen Python reference lives in `engine/engine_old/` (copied from the repo-root
`engine_old/`, which is itself the package that used to be `engine/`). It is the
specification: anything the C++ engine does that `engine_old` doesn't do the same
way is a bug until deliberately decided otherwise.

---

## 0. Where we are

**Baseline measurements** (radius 7 / 169 hexes, 8 factions, this machine —
Ryzen 5600X, Python 3.12, numpy 2.5.2):

| game | wall clock | turns |
|---|---|---|
| random x8 | 0.15 s | 45 |
| greedy x8 | 0.18 s | 14 |
| marshal x8 | 0.26 s | 14 |
| 1 tactician vs 7 marshal | 0.41 s | 16 |
| tactician x8 | 2.08 s | 18 |

**Profile of a tactician x8 game** (3.3 s under cProfile, 6.1M calls). Nothing is
compute-bound; it is all per-element interpreter and numpy-dispatch overhead on
tiny arrays:

- `numpy.ufunc.reduce` 176 878 calls — 0.41 s
- `movement._legal_mask` 23 652 calls — 0.37 s
- `geometry.hex_distance` 434 199 calls — 0.37 s cumulative
- `geometry.min_hex_distance_to_any` 7 569 calls — 0.36 s cumulative
- agent-side generator expressions (`heuristic`, `marshal`, `vanguard`) — ~0.9 s combined
- `builtins.abs` 1 303 104 calls, `builtins.max` 440 657 calls

Roughly **40–50 % of the time is inside `engine_old`, 50–60 % inside `agents/`**.
That ratio drives the phase ordering below: porting only the engine buys ~2x;
the 100x+ needs the agents native too.

**The core structural insight:** the entire game state is *tiny*. A radius-8 board is
217 hexes; ≤10 factions. Held as a flat POD struct with no heap allocation, a whole
`GameState` is a few tens of KB — so cloning a state (which `tactician_agent`'s
rollouts do constantly, and which any future MCTS/self-play will do far more) becomes
a `memcpy` instead of a dozen `np.copy` calls. That single property is worth more
than any micro-optimization.

**Toolchain: installed and verified** — see §2. M0 is already green.

---

## 1. Target layout and Python independence

### 1.1 Layout

```
engine/
  PLAN.md                 <- this file
  CMakeLists.txt
  include/oo/
    config.hpp            compile-time caps (MAX_HEXES, MAX_FACTIONS, ...)
    rng.hpp               CPython-compatible Mersenne Twister
    grid.hpp              HexGrid: coords, neighbour table, distances
    state.hpp             GameState POD + helpers
    terrain.hpp  movement.hpp  battle.hpp  buy.hpp  collect.hpp
    placement.hpp  setup.hpp  turn.hpp
    agent.hpp             decision interface (§6)
    json.hpp              replay-log writer (board_state.json etc.)
  src/                    one .cpp per header
  agents/                 the twelve agents, in C++ (§6)
  apps/
    oo_run.cpp            replaces run.py       -> board_state.json
    oo_tournament.cpp     replaces tournament.py
  tests/
    test_*.cpp            unit + invariant tests
    parity/               golden-trace comparison against engine_old
  bench/
    bench_main.cpp
  --- transitional, deleted at M8 (§1.2) ---
  engine_old/             frozen Python reference (edited only per §3.2)
  bindings/module.cpp     pybind11 -> `oo_engine`, behind -DOO_BUILD_PYTHON
  __init__.py turn.py ... Python shims so `from engine.turn import ...` still works
```

### 1.2 Python independence — the endgame

**No. The engine core has zero Python dependency, by construction.** Stating the
rule explicitly, because it is easy to erode by accident:

> Nothing under `include/oo/`, `src/`, `agents/` or `apps/` may include
> `Python.h`, `pybind11`, or link against a Python library. Ever.

`bindings/` is the *only* place Python appears, it is a leaf that depends on the core
and nothing depends on it, and it sits behind a CMake option:

```cmake
option(OO_BUILD_PYTHON "Build the pybind11 module" ON)   # flip OFF at M8
```

With it `OFF`, the build produces a standalone static library plus the `oo_run` and
`oo_tournament` executables, and Python is not involved at any stage — not at build
time, not at run time. That target is testable from day one; CI should build both
configurations so the dependency can never creep back in.

**Three decisions follow from this**, and they differ from what an
engine-that-lives-inside-Python would do:

1. **JSON is written from C++, not Python** (§4.4 item 11). The replay logs
   (`board_state.json`, `terrain_gen_log.json`, `city_placement_log.json`) are the
   decoupling interface, and `oo_run` must emit them natively. This is the single
   most important one — routing log construction through the binding layer, which is
   the obvious shortcut, would make replay permanently Python-dependent.
2. **`run.py` and `tournament.py` become C++ executables**, not scripts that call in.
3. **The binding layer is transitional scaffolding, not a product.** It earns its
   keep between M5 and M6b as the cheapest possible full-system integration test —
   twelve existing agents exercising the new engine, with `run.py`/`tournament.py`
   unmodified — and is then deleted. Do not gold-plate it.

**What Python removal actually costs you**, checked against the repo:

| file | fate |
|---|---|
| `web_visualizer.html` | **survives untouched** — pure JS, loads `board_state.json` from a file picker. No Python anywhere in it. The only visualizer going forward. |
| `hex_visualizer.py`, `city_placement_visualizer.py`, `hex_gen_visualizer.py` | **deprecated and deleted** (commit `1e21ffd`). Not ported. |
| `hex_common.py` | orphaned by that deletion — nothing imports it any more. Delete. |
| ~~`run.py`~~ | **deleted at M6c** - replaced by `oo_run` |
| `tournament.py` | still drives the M5 gate; goes at M8, replaced by `oo_tournament` |
| `engine_old/`, `agents/` | parity oracles; deleted or archived at M8 |
| numpy | only ever reached through the bindings — gone with them |

### 1.3 The three JSON files are a hard requirement

`oo_run` must keep writing **all three**, natively, in the current format:

| file | written by | consumed by |
|---|---|---|
| `board_state.json` | `run_turn_and_log` per-turn checkpoints | `web_visualizer.html` |
| `terrain_gen_log.json` | `generate_terrain`'s placement log | the deleted `hex_gen_visualizer` — **keep emitting anyway** |
| `city_placement_log.json` | `run_city_setup`'s step log | the deleted `city_placement_visualizer` — **keep emitting anyway** |

Two of the three no longer have a Python consumer, and they are kept deliberately:
they are the highest-resolution record of exactly what terrain generation and the
capital draft did, which makes them the natural diffing surface for parity work
(§3.3) and for any future replacement viewer. Dropping them would throw away the
easiest way to see *where* a C++ map generator diverged from the Python one.

So: the log-emitting code paths stay fully populated even though only
`board_state.json` currently has a reader. Verify all three byte-for-byte against
Python-produced files at M6c.

The web visualizer covering the main use case is what makes "delete all Python"
genuinely cheap. Keeping these formats byte-identical is therefore a hard requirement
of the port, not a nicety.

---

## 2. Step 1 — toolchain — **DONE, M0 green**

Installed and verified on this machine:

| | |
|---|---|
| Visual Studio Build Tools 2022 | 17.14.39, MSVC toolset **14.44.35207** (v143) |
| install root | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` |
| CMake | 3.x, bundled at `Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin` |
| Ninja | bundled alongside it |
| pybind11 | 3.1.0 (pip) |
| Python | 3.12.4, **64-bit**, built with MSC v.1940 — v143 ABI, matches |

**Gotcha — the `cl.exe` on `PATH` is the 32-bit `HostX86\x86` one.** Python is
64-bit, so building against it directly produces a link failure. Always enter the
environment through

```
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

which puts `HostX64\x64\cl.exe` first. (`vcvars64` prints a harmless
`'vswhere.exe' is not recognized` warning here; ignore it.) CMake with the Ninja
generator must likewise be invoked from a `vcvars64` shell, or given
`-DCMAKE_C_COMPILER`/`-DCMAKE_CXX_COMPILER` pointing at the x64 `cl.exe`.

**Smoke test passed.** Built and imported an extension exercising the exact
technique §5 depends on — a `py::array_t` view onto a C++ POD struct's memory,
`base` set to the owning object:

```
sizeof(Box) = 3972
view shape (331, 3) int32  owns_data False      <- zero-copy, no allocation
write-through: 42                                <- numpy writes hit C++ memory
numpy ops on the view: 1                         <- (b.a[:,1] == 42).sum() works
```

So the binding design is validated, not just assumed: existing agent code doing
`state.army_faction == faction` will operate directly on engine memory with no copy.

Build config: C++20, `/O2 /fp:fast /arch:AVX2 /GL` (LTCG) for release,
`/Od /Zi /RTC1` plus assertions for debug. Keep both configured from day one — the
debug build with bounds-checked accessors is what makes the parity work tractable.

`engine/dev.bat` wraps `vcvars64` so the x64 compiler is used; everything goes
through it:

```
engine\dev.bat cmake -S engine -B engine\build -G Ninja -DCMAKE_BUILD_TYPE=Release -DOO_BUILD_PYTHON=OFF
engine\dev.bat cmake --build engine\build
engine\dev.bat ctest --test-dir engine\build --output-on-failure
```

Confirmed picking up `Hostx64\x64\cl.exe` (MSVC 19.44.35228.0).

---

## 3. Step 2 — bit-exact parity infrastructure (build this *before* the engine)

The single highest-leverage thing in this whole port. If a C++ game and a Python
game, given the same seed, produce byte-identical states after every phase, then
correctness is *proved* rather than argued, and every later optimization is free to
be aggressive.

### 3.1 CPython-compatible RNG

`engine_old` threads one `random.Random` through the turn and consumes it in a fixed
order that `battle.py`'s docstring already calls out as load-bearing. To match it,
`rng.hpp` must reimplement CPython's `_randommodule.c` exactly:

- MT19937 with `init_by_array` seeding from an integer seed (CPython converts the
  int to a little-endian `uint32` key array).
- `random()` = `genrand_res53`: `((a >> 5) * 67108864.0 + (b >> 6)) / 9007199254740992.0`
- `getrandbits(k)` — CPython's word order for k > 32.
- `_randbelow_with_getrandbits(n)` — rejection sampling with `k = n.bit_length()`.
- `randint(a, b)` = `a + _randbelow(b - a + 1)`
- `choice(seq)` = `seq[_randbelow(len(seq))]`
- `sample(pop, k)` — both the set-based and selection-based branches (CPython picks
  by `k` vs `n` ratio); `run_city_setup` calls it with `k == n`.
- `choices(pop, weights, k)` — cumulative weights + `bisect` on `random() * total`.

**DONE — M1 green.** `include/oo/rng.hpp`, verified two ways:

- **`tests/data/rng_golden.txt`** (from `tools/dump_rng_reference.py`) — a
  self-describing trace: every line is an operation plus the result CPython
  produced, so `tests/test_rng.cpp` parses and replays it rather than duplicating
  the battery. Extending the Python dumper automatically extends the C++ test.
  4 617 checks across 19 seeds, covering `getrandbits` at every width 1–64,
  `_randbelow`'s rejection loop around powers of two, both of `sample`'s branches,
  and `choices` with skewed weights. **0 failures.**
- **`tests/data/rng_stress.txt`** (from `tools/dump_rng_stress.py`) — 10 000 seeds ×
  1 000 mixed ops ≈ **10⁷ draws**, compared as one FNV-1a hash per seed so the
  volume stays a 10 001-line file. **0 failures**, in 1.2 s (Python needs 18 s just
  to generate it).

Negative control run on both: corrupting a golden value is detected, including a
**single-ULP** change to a `random()` result — `random()` is compared by raw bits
via `float.hex()`, not through a decimal round-trip.

The golden files are checked in, so `ctest` needs no Python (§1.2). Python is only
required to *regenerate* them, when the battery changes.

Two things worth keeping in mind:

- `tools/dump_rng_stress.py`'s op cycle is **duplicated** in
  `tests/test_rng_stress.cpp` and the two must stay in lockstep. Accepted
  deliberately — a hash tells you *that* something diverged, never what, so the
  self-describing golden trace is the diagnosis tool and this is only the volume
  check.
- `sample()` has two branches with **different draw counts** (pool vs. selected-set,
  split at `setsize`). Both are reachable here — `placement.py` takes the pool
  branch, `random_agent`'s `sample(legal, k<=3)` over hundreds of legal actions
  takes the other — so picking one would silently desync everything downstream.
  Both are implemented and both are covered astride the boundary.

This is a *parity* requirement, not a permanent constraint — but there is no reason
to swap it out later either. RNG draws are nowhere near a bottleneck (a few hundred
per game), MT19937 is perfectly adequate for self-play, and keeping it means the
golden traces stay valid forever. Keep it after M8; don't churn to PCG64 for
imagined speed.

### 3.2 One `engine_old` change: make terrain generation order-portable

Everything in the turn loop is already order-deterministic and portable —
`arrivals_by_dest`, `by_target`, `cav_died_by_faction`, `enemy_adjacent_cache`,
`actions_by_faction` are all insertion-ordered dicts, and the one set iterated for a
value (`arrival_factions`) is only read when it holds exactly one element.

The exception is `setup.py`'s terrain generation, which does
`candidates = set(); ... rng.choice(candidates)` in `_place_round` and
`unset`-set iteration in `generate_terrain`. CPython's small-int set ordering is
deterministic but is an artifact of hash-table layout, and emulating it in C++ is
fragile busywork.

**DONE — applied and verified.** Three `sorted(...)` wrappers in
`engine/engine_old/setup.py`: the candidate list in `_place_round`, and `edge_hexes` +
the round-start candidate list in `generate_terrain`. The whole pipeline, map
generation included, is now bit-parity testable.

Measured effect, 300 seeds per radius, original vs. modified:

| radius | plains | mountain | lake | desert | marsh | disconnected |
|---|---|---|---|---|---|---|
| r7 orig | 55.90 % | 7.74 % | 9.04 % | 13.67 % | 13.65 % | 0/300 |
| r7 sorted | 55.82 % | 7.78 % | 8.91 % | 13.73 % | 13.76 % | 0/300 |
| r8 orig | 52.01 % | 8.66 % | 9.56 % | 15.05 % | 14.72 % | 0/300 |
| r8 sorted | 52.07 % | 8.81 % | 9.50 % | 14.87 % | 14.75 % | 0/300 |

Distributions match within sampling noise (≤0.15 pp on every category at r7/r8) and
the connectivity invariant holds in both. r4/r5 likewise.

**One consequence worth knowing:** a given seed now produces a *different* map than
it did before — the rules and the distribution are unchanged, but the seed→map
mapping is not. Old seeds no longer reproduce their old boards, so any saved
`board_state.json` or hardcoded debugging seed refers to a map that no longer
regenerates.

### 3.3a Phase parity harness — DONE (built during M2)

The infrastructure §3.3 calls for, in its first working form:

- **`src/state_io.cpp` + `tools/state_io.py`** — one canonical text serialization of
  a whole state, written by both engines. Battle contributions are stored sparsely
  (occupied slots only), since a dense [331][16] dump would be almost all padding.
  Text rather than binary because these files get read by a human the moment
  anything disagrees.
- **`compare_states`** reports the *first* differing field with its index
  (`army_units[15][0]: got 1, want 2`) rather than a bare boolean.
- **`validate_state`** checks the invariants from §4.5 independently of any
  comparison: battle storage agrees with `battle_order`/`locked`, occupied battle
  slots are contiguous from 0, no army stands on impassable terrain, peaceful
  stacks respect the 6-unit cap. Every Python-produced input state is run through
  it too — if the reference violates our invariants, either the invariants or the
  reader is wrong, and either way it needs knowing.
- **`tools/dump_phase_cases.py`** emits (before, after) pairs per phase.
- **`tools/_bootstrap.py`** aliases `engine/engine_old/` back to `engine.*` so the
  frozen reference and `agents/` import unmodified. Order matters — aliasing
  before importing the submodules sends the import machinery into infinite
  recursion.

**Two lessons worth carrying into M3, both learned the hard way here:**

1. **Measure what the cases actually exercise; do not trust a green result.** The
   first version of this suite reported 350/350 passing. It was nearly worthless:
   random-agent games build no outposts, so `resource_income` and
   `victory_points` changed *nothing* in 100% of cases, and not one outpost
   upgrade of any type appeared anywhere in the corpus. The fix was to generate
   from greedy games as well (which do build) and to have the perturbation *build*
   outposts rather than only re-roll upgrades on ones that already existed.
   Coverage now, per phase, as a fraction of cases where the phase changes state:
   collect 100%, gold_income 100%, resource_income 74%, terrain 67%,
   victory_points 70% — with 7 014 upgrade slots across all three types, 541
   armies on desert, 435 frozen stacks, 180 states holding a pending battle, and
   103 faction-instances holding zero cities.
2. **Mutation-test the suite.** Four plausible bugs were introduced one at a time
   and every one was caught by a wide margin: flat per-outpost VP instead of
   `max(0, n-1)` → 287 failures; mountains yielding clay instead of iron → 231;
   the Barracks gold bonus ignored → 221; desert attrition ignoring the
   city exemption → 306. A suite that has never been seen to fail is not evidence.

### 3.3b Turn traces and scenario suites — DONE (built during M3)

**Decision-trace replay** turned out to be the right shape for §3.3, and it is
better than the "mirror each policy in C++" alternative: `tools/dump_turn_traces.py`
records what engine_old's agents actually *decided*, in order, and
`tests/test_turn.cpp` replays those decisions through the C++ engine. No policy is
duplicated, so any agent can be a trace source — including tactician, whose search
would be miserable to mirror by hand.

It also checks more than the resulting state: the replaying provider asserts the
C++ engine asks for decisions in the same **order** and with the same
**arguments**. A battle running an extra round, or a movement step querying the
wrong faction, is reported where it happens rather than as a mystery state diff
several phases later. Each turn gets a fresh RNG seeded from (game seed, turn
number) so every case is independently replayable.

**Three suites, because each closes a hole the others cannot see:**

| suite | what it covers | why it is needed |
|---|---|---|
| `turn_traces` | 180 full turns, 9 064 decisions, 6 agent kinds, radii 4–8, 4–10 factions | ordinary play end to end |
| `movement_scenarios` | 27 hand-built boards | the rare paths — line-battle tie-breaks, overstack reverts, both `_revert_departure` quirks, undefended structures, 6 factions into one hex |
| `legal_cases` + `buy_scenarios` | 194 states × every faction; 23 hand-built buy batches | legal-action generation, and the per-turn batch caps |

**The lesson from M2 repeated itself, and mutation testing is what caught it.**
The first M3 suites reported 180/180 and 27/27 on the very first run. Thirteen
plausible bugs were then introduced one at a time:

- Eleven were caught by wide margins — inverted line-battle tie-break (8 turns),
  overstack off-by-one (40), kill threshold 15→14 (18), lone attacker scoring 2
  (37), dismount threshold (12), inverted target-conflict tie-break (3), capital
  defence shots (15), stationary archers firing (45), capital not evicting (9), no
  VP for destroying an outpost (61), Barracks not lifting the recruit cap (9).
- **Marsh no longer freezing broke nothing across 180 turns** — caught only by a
  movement scenario. Within a turn the effect is invisible unless an army enters
  a marsh and then tries to move again, and terrain effects clear `frozen` before
  the after-state snapshot.
- **Removing the one-outpost-action-per-turn cap entirely broke nothing anywhere.**
  The rule is only observable when an agent proposes two outpost actions in one
  turn, and no real agent ever does. That is what `buy_scenarios` exists for; the
  same mutation now fails 3 scenarios.

Two structural gaps came out of the same exercise and are now closed:
`get_legal_buy_actions` and the movement/cavalry masks were **entirely untested** —
turn traces replay what an agent *chose* and never check the menu it chose from,
even though all twelve agents index into it. Their order is part of the contract,
so `legal_cases` compares element by element.

### 3.3c Full-pipeline parity — DONE (M4)

**§3.2's `sorted()` change paid off exactly as intended.** Terrain generation is
pure — a radius and a seed in, a map out, no decisions anywhere — so there is
nothing to replay: the C++ engine regenerates from the same seed and the map must
be identical hex for hex, plus the generation log in the same order.

**320 maps across radii 1–8, 29 120 hexes, 0 failures.** That is only possible if
the RNG, the bag weighting *order*, the blob shape rules, the island check and the
round structure all agree exactly.

The trap worth recording: `BAG_COUNTS` is a dict in the order **plains, lake,
mountain, desert, marsh** — *not* terrain-index order — and that order decides
which weight pairs with which type inside `rng.choices`. Using the "obvious"
index order produces a perfectly plausible-looking map that diverges from
Python's. Mutation testing scored it at **274 of 320 maps wrong**.

Setup cases go further: each carries `PARAMS <radius> <factions> <seed>`, so the C++
side *builds* the starting state from the seed alone via `create_initial_state` and
checks it against Python's before replaying the placement decisions. Each case is
therefore a genuine from-seed pipeline check, not just a placement replay.

**Mutation testing found the same class of gap for the third time.** Eleven
mutations; ten caught immediately, and one — "the placement mask always relaxes
past tier 1", i.e. the 5–7 player edge ban silently dropped — **caught nothing**.
The reason is the same one that hid the buy-action list in M3: the trace recorded
the agent's *choice*, and a mask that is merely LARGER than the correct one still
validates that choice. Comparing effects is not comparing the menu.

Fixed by recording the legal placement mask and the draft pool themselves and
comparing them element by element. The edge-ban mutation now fails 6 cases, and
"capital spacing 3 → 2" went from 2 failures to 19 — every case.

> **The rule this establishes for M5 onward:** any function that hands an agent a
> set of options is part of the contract and must be compared directly. Testing
> what the agent did with the options is not enough, three times over now.

### 3.3 Golden traces

- `tools/dump_trace.py` — runs `engine_old` with a given seed and agent assignment,
  writing after every phase boundary (buy, each of 3 move steps, each of 2 cavalry
  steps, battle, terrain, collect): a canonical serialization of every state array,
  plus the running count of RNG draws consumed.
- `tests/parity/` — the C++ build replays the same seed and asserts equality at every
  boundary, reporting the first divergence with the phase, hex, and field.
- Drive it with `random_agent` first (it fuzzes every code path — that is its stated
  purpose), then `greedy`, then `marshal`, then `tactician`. A few thousand seeds
  across radii 4–8 and 4–10 factions.

The RNG-draw counter matters as much as the state: two engines can agree on state
while having consumed a different number of rolls, and that divergence surfaces
several turns later somewhere unrelated.

### 3.4 Retiring the parity corpus (decided, do at M8)

Once Python is gone the CPython-compatible RNG (§3.1) stops earning its keep,
and it is replaced with a native generator — recommended: **xoshiro256++ seeded
through SplitMix64**.

**What this buys.** Speed is the smallest part: `Rng::seed` + `genrand_uint32` +
`randbelow` measure 0.85 s of 18.5 s (engine profile) and 0.84 s of 20.0 s
(tactician), so ~4.5%. The two real gains:

- **State size: 2.5 KB -> 32 bytes.** MT19937 carries 624 words plus an index.
  Every agent owns one, tactician copies a whole `Rng` per rollout candidate
  (§7.2 item 3), and §10 wants thousands of games in flight each holding one.
- **Deleting the intricate part.** `_randbelow_with_getrandbits`, the
  dual-branch `sample`, `genrand_res53`, `choices` with `hi = n - 1` — the
  fiddliest code in the engine, and all of it exists only to match CPython.

**This is a one-way door.** The golden corpus is what proves the C++ engine
implements the same game as the Python original; change the RNG and it can
never be re-derived. So: **tag the last Python-verifiable commit first** (the
same tag that archives `agents/` and `engine_old/`, §9), and do the RNG swap as
the final act of M8, after everything else is green.

The corpus splits cleanly:

| survives untouched — no RNG, stays a Python-verified oracle | regenerate from C++ — downgrades to a regression test |
|---|---|
| `grid_golden`, `legal_cases`, `buy_scenarios`, `movement_scenarios`, `phase_cases` | `turn_traces`, `agent_games`, `replay_hashes`, `setup_cases` |

The left column is pure geometry, explicit states, or hand-built action lists —
no seed is involved, so those files stay valid forever and keep some
Python-derived ground truth in the repo. The right column is generated from
seeds; regenerate each file once from the C++ engine in the same commit as the
swap. Those tests still catch "I broke something today"; they stop proving
"matches Python". That is the correct trade at M8 and not before.

`rng_golden` and `rng_stress` become meaningless outright — their entire purpose
is CPython bit-compatibility. Replace them with basic property tests on the new
generator (range, period, uniformity of `randbelow` across moduli, `shuffle`
producing every permutation).

**Do not use `std::uniform_int_distribution`, `std::shuffle`, or
`std::mt19937`.** All three are implementation-defined or
implementation-seeded, so MSVC and libstdc++ produce different sequences from
the same seed. Cross-platform reproducibility is weaker than the CPython parity
being given up, but it is still worth keeping, and it costs one hand-written
bounded-integer draw.

---

## 4. Step 3 — the core engine

### 4.1 `config.hpp`

```cpp
inline constexpr int MAX_RADIUS         = 10;
inline constexpr int MAX_HEXES          = 331;  // 3*r*(r+1)+1 at r=10
inline constexpr int MAX_FACTIONS       = 10;
inline constexpr int MAX_BATTLE_CONTRIB = 16;   // == engine_old's
inline constexpr int NUM_UNIT_TYPES     = 3;
inline constexpr int NUM_RESOURCES      = 4;
inline constexpr int MAX_STACK_SIZE     = 6;
```

Compile-time caps, not runtime — that is what keeps `GameState` a POD.

### 4.2 `grid.hpp`

Immutable per-radius data, built once and shared by every state and thread
(`const HexGrid&`, never copied into the state):

- `coords[MAX_HEXES]` as `int8 q, r, s`
- `neighbour[MAX_HEXES][6]`, `int16`, `-1` off-board
- `is_edge[MAX_HEXES]` precomputed
- **Precompute the full `dist[MAX_HEXES][MAX_HEXES]` table** (`uint8`, 331² = 110 KB).
  `hex_distance` is 434 k calls per game and `min_hex_distance_to_any` materializes an
  `[N, k, 3]` temporary each time — both collapse to a table lookup. This alone
  removes most of what the profile shows above.
- `direction_between`, `index_of`, `coord_of` for the binding layer.

Cache built grids in a `map<radius, HexGrid>` behind a mutex, or build eagerly for
the radii in use. Never mutable after construction — that is the multicore contract.

**Done (M2).** `index_of` is a closed-form inverse of the enumeration via a
column-offset prefix sum, so it is O(1) rather than engine_old's dict lookup.
Checked against Python at every radius 0–10: coordinate assignment, neighbour
table, edge ring, full distance-matrix row sums plus sampled exact rows — 43 105
checks. Plus reference-free properties: `index_of`/`coord_of` round-trip,
adjacency symmetric, every neighbour at distance exactly 1, `direction_between`
inverting the neighbour lookup, and off-board/non-cube coordinates rejected rather
than aliasing onto a real hex.

### 4.3 `state.hpp`

Straight transcription of `ArrayState`, fixed-size and trivially copyable:

```cpp
struct GameState {
  int8_t   terrain[MAX_HEXES];
  int8_t   city_owner[MAX_HEXES];
  bool     is_capital[MAX_HEXES];
  int8_t   outpost_upgrade[MAX_HEXES];
  int8_t   city_placer[MAX_HEXES];
  int32_t  capital_settle_order[MAX_FACTIONS];
  int8_t   army_faction[MAX_HEXES];
  int16_t  army_units[MAX_HEXES][3];
  bool     frozen[MAX_HEXES];
  bool     locked[MAX_HEXES];
  int8_t   battle_faction[MAX_HEXES][MAX_BATTLE_CONTRIB];
  int32_t  battle_origin [MAX_HEXES][MAX_BATTLE_CONTRIB];
  int16_t  battle_units  [MAX_HEXES][MAX_BATTLE_CONTRIB][3];
  bool     battle_moved  [MAX_HEXES][MAX_BATTLE_CONTRIB];
  int16_t  battle_round[MAX_HEXES];
  int16_t  battle_order[MAX_ACTIVE_BATTLES];
  uint8_t  num_battles;
  int32_t  gold[MAX_FACTIONS];
  int32_t  resources[MAX_FACTIONS][NUM_RESOURCES];
  int32_t  kill_xp[MAX_FACTIONS];
  int32_t  victory_points[MAX_FACTIONS];
  bool     alive[MAX_FACTIONS];
  int32_t  turn_number;
  int32_t  num_factions;
  int32_t  num_hexes;
  const HexGrid* grid;   // not owned
};
```

Notes:

- Keep the **dense** battle arrays for phase A. `agents/` reads
  `state.battle_faction[hex]`, `state.battle_units[hex, k]`, `state.battle_origin`
  and `.shape[1]` directly, so the layout is part of the public contract until
  the agents are native. ~65 KB per state, and a 65 KB `memcpy` is still ~2 µs —
  far cheaper than what `_clone_state` costs today.
- Also add a per-hex `uint8 battle_nslots` so internal loops iterate used slots
  rather than scanning all 16. Pure win, no layout change.
- `battle_order` removal must be an **order-preserving compacting erase**, not a
  swap-erase — `list.remove` keeps order, and battle processing order is
  outcome-affecting near the dismount cap (see `state.py`'s docstring).
- **The dense-layout constraint expires at M6b.** Once every agent is native and the
  bindings are gone, collapse the battle arrays into a sparse side table
  (`battle_index[MAX_HEXES]` + `Battle battles[32]`). Drops `GameState` to ~10 KB,
  which is what makes deep search trees viable — scheduled as M6d.

### 4.4 Module-by-module port order

Each step ends with its own parity test green before starting the next.

1. **`rng`** — §3.1. Standalone; test against Python directly.
2. **`grid`** — cheap, no rules, unblocks everything.
3. **`state`** — struct, `new_empty`, `count_units_in_play`, `count_all_units_in_play`.
4. **`terrain`** — smallest rules module (`apply_terrain_effects`), good warm-up:
   desert attrition, unfreeze, and the deliberate quirk that a hex emptied by desert
   loss is not touched by the unfreeze pass.
5. **`collect`** — gold, resources, VP. Pure arithmetic; no RNG. One callback
   (`decide_resource_choice`).
6. **`buy`** — `eligible_outpost_mask` becomes three passes over the precomputed
   distance table. `get_legal_buy_actions` returns a fixed-capacity
   `small_vector<BuyAction, N>` — no heap. Preserve the per-turn-batch caps
   (1 recruit/turn/outpost unless Barracks; 1 outpost action/turn/faction) exactly
   where `apply_buy_phase` enforces them, and preserve the stack-cap-before-gold
   ordering fix that its docstring calls out.
7. **`movement`** — the trickiest module. Port literally, quirks included:
   **Done (M3)**, quirks preserved and directly covered by `movement_scenarios`;
   both branches of the exact-tie coin flip are exercised.
   - pass 1 line-battle/swap detection, with the smaller-army tiebreak and the
     `rng.random() < 0.5` coin flip on exact ties (**this RNG draw's position in the
     sequence is load-bearing**);
   - pass 2 destination grouping: reinforce-locked / hostile / multi-faction /
     foreign-structure -> battle, else capped peaceful merge;
   - `_revert_departure` including its two documented edge cases (origin claimed by
     another faction -> starts a battle there; origin locked -> recreates a peaceful
     army on a locked hex). These are *ported bugs*. Keep them, note them, decide
     later.
   - `battle_moved` flags exactly as set today — they gate the archer ability and are
     deliberately not derivable from `battle_origin`.
8. **`battle`** — port literally, preserving the **exact order of every
   `randint(1, 20)`**: structure defence shots, then archer abilities, then per round
   (targeting conflicts -> rolls in `resolved_targets` insertion order -> simultaneous
   kill application -> dismount rolls in `cav_died_by_faction` insertion order).
   `_battle_faction_order` is first-appearance-by-slot; keep it. `rectify_overflow`
   with the `cap` parameter (0 for capital eviction).
9. **`turn`** — `run_turn`, `_run_battle_phase` (shared `infantry_counts` tally across
   every battle in the turn — that sharing is why battle order matters),
   `get_game_winner` with the `capital_settle_order` tiebreak, `check_game_end`.
10. **`setup`** / **`placement`** — terrain generation and the placement/draft/swap
    setup, subject to §3.2.
11. **Logging** — `run_turn_and_log`, `snapshot_hexes`, `sparse_hexes`, plus a native
    JSON writer (`json.hpp`). **Emit the replay logs from C++, not from the binding
    layer** (§1.2): `board_state.json`, `terrain_gen_log.json` and
    `city_placement_log.json` must come out byte-identical to what `run.py` writes
    today, since that format is what keeps `web_visualizer.html` working after the
    Python is gone. Verify by diffing against a Python-produced file for the same
    post-setup state. No JSON dependency needed — the schema is small and fixed;
    hand-roll it or vendor a single header. Logging is `run.py`'s dominant cost and
    must be *fully* skippable (`run_turn` does no logging at all).

### 4.5 General C++ rules for this codebase

- **No heap allocation in the turn loop.** Fixed arrays and `small_vector` only.
- **No globals, no function-local statics, no mutable shared state.** RNG is always
  an explicit `Rng&` parameter. This is the multicore prerequisite and is far cheaper
  to honour from the start than to retrofit.
- Engine functions are free functions taking `(GameState&, ...)` — no methods that
  hide state, nothing that reads a config singleton.
- `assert`-heavy debug build: stack ≤ 6 outside battle, unit counts within
  `SPAWN_CAPS`, no army on impassable terrain, `battle_order` consistent with
  `locked`, no army on a locked hex.

---

## 5. Step 4 — Python bindings (`oo_engine`) — *transitional, deleted at M8*

Scoped as an integration harness, not a product (§1.2). It exists so twelve working
agents can exercise the new engine before any of them are ported, which is a far
stronger test than anything hand-written. Build it behind `-DOO_BUILD_PYTHON=ON`;
nothing in the core may include it.

- `GameState` exposed as an opaque object whose array fields are `py::array_t`
  **views** onto the C++ memory (`base` = the owning state object, so lifetime is
  safe and there is no copy). Existing agent code like
  `state.army_faction == faction` or `state.army_units[hex, 1]` then works unmodified.
- `state.grid` exposed with `coords`, `coords_array`, `neighbor_table`, `num_hexes`,
  `coord_of`, `index_of`, `is_edge`, `direction_between`.
- Free functions bound one-to-one with the `engine_old` module functions.
- Thin Python shims — `engine/__init__.py`, `engine/turn.py`, `engine/state.py`, etc.
  — re-export from `oo_engine` so `from engine.buy import OUTPOST_CAP` and friends
  keep resolving. All module-level constants (`SPAWN_CAPS`, `UPGRADE_COSTS`,
  `TERRAIN_TYPES`, `MOVEMENT_STEPS`, ...) re-exported with identical values.
- Callbacks stay Python `{faction: callable}` dicts in this phase. Release the GIL
  around pure-C++ stretches, reacquire to call back.
- **Acceptance gate:** `run.py` and `tournament.py` run unmodified, all 12 agents
  work, and `tournament.run_matchup("greedy", "random", 100)` produces the *same
  results* as the Python engine for the same seeds.

Expected speedup at this gate: ~2x on tactician games, more on cheap-agent games
where the engine's share of the runtime is higher.

---

### 5.1 M5 result — DONE

`tournament.py` and all twelve agents run **unmodified** on the C++ engine.

**Parity: 100/100 games identical** — 20 (agent × board-size) combinations × 5
seeds, comparing winner, turn count and every faction's victory points.
`tools/compare_engines.py` runs the matrix under each engine (two processes; only
one can own the `engine` module name) and diffs.

**Speed**, same games both sides (`tools/bench_engines.py`, 5 games each):

| agents | board | Python s/game | C++ s/game | speedup |
|---|---|---|---|---|
| random | r7f8 | 0.150 | 0.0094 | **15.9x** |
| greedy | r7f8 | 0.190 | 0.035 | 5.5x |
| heuristic | r7f8 | 0.211 | 0.065 | 3.2x |
| vanguard | r7f8 | 0.240 | 0.119 | 2.0x |
| marshal | r7f8 | 0.267 | 0.158 | 1.7x |
| tactician | r7f8 | 1.851 | 0.875 | 2.1x |
| tactician | r5f6 | 0.799 | 0.357 | 2.2x |

This is §0's prediction landing exactly: the speedup tracks how much of a game was
engine rather than agent. `random_agent` barely thinks, so it gets 16x; marshal and
tactician are agent-dominated and get ~2x. **The remaining time is now almost
entirely Python agent code**, which is what M6 is for.

**Two things found that the C++ test suite could not have caught**, both worth
recording:

1. **`MoveActions` lost submission order** — a real API gap, not a binding slip.
   `engine_old` iterates `actions_by_faction`, a dict, in *insertion* order, and
   that order decides how simultaneous arrivals are grouped by destination, which
   becomes battle *creation* order, which is load-bearing for the shared dismount
   cap. `run_turn` always inserts ascending, so an array indexed by faction
   matched everywhere the trace tests looked. But `tactician_agent`'s rollout
   builds `{faction: first_action}` and only then adds the others — submitting the
   searching faction **first**. The two engines resolved the same two battles in
   opposite order, which drifted the rollout opponents' RNGs, which eventually
   changed a real move. `MoveActions` now carries an explicit `order[]`.
2. **RNG bridging works by state transfer, not callbacks.** `run_turn` is handed a
   live `random.Random`; because `oo::Rng` is bit-compatible (§3.1) we borrow via
   `getstate()`, run the phase natively, and write back with `setstate()`. Verified
   directly (`oo_engine._rng_draw`): zero-draw borrows leave the generator
   untouched, native draws match Python's exactly, repeated one-draw borrows match,
   and 2 000 draws across the 624-word regeneration boundary match.

**Deliberately deferred: `run.py`.** It needs `run_turn_and_log`, which builds the
per-checkpoint replay record. That belongs in C++ (§1.3 — the JSON must be written
natively or replay stays Python-dependent forever), so building a Python-side
version now would be precisely the throwaway work §1.2 warns against. `run.py`
therefore does **not** work at M5; `oo_run` covers it at M6c. `tournament.py`,
which drives `run_turn` with no logging, is the M5 integration test and works fully.

Also verified: `-DOO_BUILD_PYTHON=OFF` still configures, builds and passes all 8
tests, producing no `.pyd` — the §1.2 guarantee holds.

---

## 6. Step 5 — rewrite the agents in C++ (where the real speedup is)

50–60 % of current runtime is in `agents/`, and every Python callback additionally
forces the binding layer back across the GIL. Phase A alone therefore caps out around
2x; **this phase is where the order-of-magnitude lives**, and it is also the
prerequisite for §7's multicore and for any future self-play, since neither is worth
much if a Python callback sits in the inner loop.

### 6.1 It is much smaller than it looks

The files are docstring-heavy — those long design journals are most of the bytes.
Actual code:

| file | code lines | file | code lines |
|---|---|---|---|
| `greedy_agent` | 171 | `denier_agent` | 93 |
| `tactician_agent` | 166 | `hussar_agent` | 90 |
| `vanguard_agent` | 130 | `random_agent` | 84 |
| `heuristic_agent` | 123 | `warlord_agent` | 82 |
| `marshal_agent` | 76 | `legion_agent` | 70 |
| `sentinel_agent` | 61 | `turtle_agent` | 48 |
| | | **total** | **~1 200** |

~1 200 lines of real logic across all twelve. **Keep the docstrings** — port them
across as comments. They record which ideas were tried and rejected and with what
measured win rates, and that history is worth more than the code it annotates.

### 6.2 Dependency graph — port bottom-up

Verified by reading every import. Nothing is cyclic; the leaves carry most of the
shared logic:

```
random_agent      (random_rectification, random_target, movement/buy/setup policies)
  └─ greedy_agent (greedy_buy, _move_toward, greedy_placement/draft/swap/resource_choice)
       └─ heuristic_agent (heuristic_target, _resource_bonus, _best_expansion_target)
            ├─ vanguard_agent (_all_targets, _best_direction, _ranked_*, rank_score)
            │    ├─ marshal_agent (_greedy_match, marshal_move)
            │    │    └─ tactician_agent (search + rollouts)
            │    ├─ sentinel_agent, warlord_agent, legion_agent, hussar_agent
            │    └─ denier_agent (_current_leader)  ─┘ (warlord also uses this)
            └─ turtle_agent
```

Port order: **random → greedy → heuristic → vanguard → marshal → tactician**, then
the six thin leaves (turtle, denier, warlord, legion, hussar, sentinel) which are
mostly wiring over helpers already ported by then. Each agent gets its own parity
test against the Python original before the next one starts.

`tactician_agent` goes last and is the payoff: its rollouts then run entirely in C++
on `memcpy`-cloned states, replacing today's `_clone_state` (a dozen `np.copy` calls
per rollout, up to `MAX_CANDIDATES = 10` rollouts per turn per faction).

### 6.3 Interface

```cpp
struct Agent {
  virtual BuyActions decide_buy(const GameState&, int faction, const LegalBuy&) = 0;
  virtual std::optional<Move> decide_movement(const GameState&, int faction, int step,
                                              const LegalMask&) = 0;
  virtual std::optional<Move> decide_cavalry(...) = 0;
  virtual int  decide_target(const GameState&, int hex, int faction) = 0;
  virtual SendBack decide_rectification(const GameState&, int hex, int winner, int cap) = 0;
  virtual Resource decide_resource_choice(const GameState&, int faction, int hex) = 0;
  virtual int  decide_placement(const GameState&, int faction, const HexMask&) = 0;
  virtual int  decide_draft(const GameState&, int faction, const HexList&) = 0;
  virtual bool decide_swap(const GameState&, int faction, int leftover, int placer, int placer_hex) = 0;
  Rng rng;                     // per-faction, persistent for the whole game
};
```

One instance per faction per game, owning its own `Rng` — mirroring today's
`rngs = {f: random.Random(seed * 1_000_003 + f) ...}` in every `make_X_agents`.
Keep the Python-callback `Agent` subclass alive alongside the native ones so a future
NN policy plugs in on the Python side, and so mixed native/Python line-ups work.

**Agents are stateful across turns** — this is easy to miss. `legion_agent` closes
over a per-faction `claimed` set that persists for the entire game (`_prune_claims`
releases entries), and every agent's RNG stream is persistent. Agent objects are
therefore per-game, not shared singletons, and must be constructible cheaply
(§7 wants one full set per thread).

### 6.4 Parity hazards specific to the agents

Good news first: **all twelve agents are already order-portable**, unlike
`setup.py` (§3.2). I checked every `set`, `dict` and `sorted` use:

- `marshal._greedy_match`'s `used_origins`/`used_targets` and `legion`'s `claimed`
  are membership-only — order never observed.
- Every dict iterated for values (`distances.items()` in heuristic/denier) is built
  by `zip` of lists, so insertion order is deterministic and portable.
- Ranking is all `sorted(...)` / `min(key=)` / `max(key=)` over **lists**.

That last point is the one real trap: **Python's `sorted` is stable and its
`min`/`max` return the first extremum**, and several agents lean on that deliberately.
`_greedy_match` sorts `((dist, o, t) for o in origins for t in targets)` **by distance
only**, so every distance tie is broken by generator order — origin-major, then target
order. Reproduce with `std::stable_sort` on an identically-ordered input, never
`std::sort`. Same rule for `vanguard._best_direction`, whose docstring explicitly
records that a naive `min()`-over-direction-index tiebreak was a measured regression.
Use `std::min_element` (returns first) and never `std::sort` on a partial key.

**Tactician's RNG discipline** — verified, and better behaved than expected:
`_search_first_move` draws exactly one `rngs[faction].randrange(2**31)` from the
agent's own RNG, then gives **every candidate rollout a fresh `random.Random(seed)`**
with that same seed (deliberately — it makes candidates comparable). So rollouts are
hermetic with respect to the real game RNG: the search never perturbs the actual dice,
no matter how many candidates it evaluates. Two things must still be replicated
exactly:

1. `randrange(2**31)` and re-seeding a fresh MT from an `int` — both already covered
   by §3.1.
2. The rollouts' **opponent** agents are one shared `make_random_agents(...)` instance
   whose per-faction RNGs *are* mutated by rollouts and persist across the game. Their
   stream state is a function of how many rollouts ran. Model it explicitly.

### 6.6 M6a result — DONE

`random`, `greedy`, `heuristic`, `vanguard` and `marshal` are native, along with
`agent_util` (the shared helpers), `game.cpp`'s `play_game`, and an `oo_tournament`
app. A game can now run with **no Python in the loop at all**.

**Parity, two independent gates:**

- **Decision level** (`tools/compare_agents.py`): the game is driven by the PYTHON
  agents so it follows the reference trajectory exactly, and at every decision
  point the native agent is asked about the same state and its answer compared.
  **25 255 decisions, 0 differences**, across 3 board sizes x 3 seeds x 12 turns
  per agent. This pinpoints a divergence at the decision that causes it rather
  than leaving a diverging final score to be reverse-engineered.
- **Whole game** (`test_agents`, in ctest): nothing replayed, nothing fed in - a
  (agent, radius, factions, seed) tuple in, and the native agents must reach the
  identical winner, turn count and per-faction VP the Python agents did.
  **80/80 games.** The golden file is checked in, so this needs no Python.

**Speed** — same games all three ways (turn counts identical, confirming parity):

| agents | Python engine | C++ engine (M5) | native (M6a) | total |
|---|---|---|---|---|
| random | 0.150 s | 0.0092 s | 0.0027 s | **56x** |
| greedy | 0.174 s | 0.0317 s | 0.0030 s | **58x** |
| heuristic | 0.214 s | 0.0606 s | 0.0034 s | **63x** |
| vanguard | 0.241 s | 0.1173 s | 0.0037 s | **65x** |
| marshal | 0.270 s | 0.1573 s | 0.0048 s | **56x** |

The M5 column is what the binding layer bought; the gap between it and M6a is
exactly the Python agent overhead this milestone removed.

**A second reference normalization was needed, and it is the §3.2 decision again.**
`greedy_rush_move` and `heuristic_move` ranked mobile armies with
`np.argsort(-sizes)` — and **numpy's default argsort is quicksort, whose order
among equal keys is an unreproducible implementation artifact**. Army sizes are
small integers so ties are the common case; measured, the default differs from a
stable sort in **62 % of random 8-element arrays and ~100 % by 20 elements**. Which
of two equally sized armies moved first was therefore being decided by numpy
internals.

My first attempt assumed numpy's sub-16-element insertion-sort path made it
stable in practice. That was simply wrong, and measuring it is what showed so.

Both call sites now pass `kind="stable"`, making the tie-break plain ascending hex
order. Same reasoning as §3.2's `sorted()` fix: normalize the reference rather than
bake a library implementation detail into the permanent C++ engine forever. It does
change Python agent behaviour on ties, so pre-M6a game outcomes are not comparable.

**Design notes worth keeping:**

- `Agent` carries a no-op `decide_play_cards` from the start (§9), so adding action
  cards later does not have to touch every agent.
- Agents are per-game objects seeded `seed * 1_000_003 + faction`, matching every
  `make_X_agents`; §7's thread pool wants one full set per thread, so construction
  is deliberately cheap.
- `NativeAgentSet` is exposed through the bindings purely so decisions can be
  compared side by side in one process. It goes with the rest of `bindings/` at M8.

### 6.7 M6b result — DONE

All **twelve** agents are native: turtle, denier, warlord, legion, hussar,
sentinel, and `tactician` — the search agent. `agents/` is now purely a parity
oracle.

**Parity, same two gates as M6a:**

- **Decision level**: 62 786 decisions at the default sweep, and **202 921
  decisions over a deeper one** (30 turns × 4 seeds × 3 board sizes × 12 agents),
  **0 differences**.
- **Whole game**: **155/155** games reaching the identical winner, turn count and
  per-faction VP.

**Speed** — tactician is the headline, since it runs the engine thousands of extra
times per turn:

| agents | Python | native | speedup |
|---|---|---|---|
| tactician r7f8 | 1.838 s | 0.0329 s | **56x** |
| tactician r5f6 | 0.792 s | 0.0108 s | **73x** |
| every other agent | 0.15–0.27 s | 0.0026–0.0062 s | 45–65x |

`tactician`'s rollouts are the payoff for the POD layout: cloning a state is now a
`memcpy` where the Python original ran a dozen `np.copy` calls per rollout, up to
`MAX_CANDIDATES = 10` times per turn per faction.

**The bug worth recording.** `legion` failed 1 of 155 games — diverging at turn 32
of one game while its first 12 turns were byte-identical. It is the only agent
carrying non-RNG state: a `claimed` set of objectives that persists for the whole
game. Two things were wrong, and only the second mattered:

1. Python's `claimed` is a **set**, so re-claiming a target is idempotent; a
   `CoordList` appending unconditionally accumulated duplicates. Real bug, but not
   this one.
2. **`legion_move` bails out on "no mobile armies" BEFORE pruning claims.** I
   pruned first. Pruning drops claims on hexes that are no longer objectives, so an
   extra prune on a step where the faction cannot move at all permanently forgets a
   claim Python still holds — surfacing ~20 turns later as a different target and a
   different move.

This is the case for keeping **both** gates. The decision comparator's default
12-turn sweep showed legion clean; only the whole-game test, running to 60 turns,
caught it — and then the comparator, pointed at that specific seed and run to 40
turns, located the exact decision. Neither alone would have been enough: one has
the reach, the other has the resolution.

### 6.8 M6c result — DONE

`oo_run` and `oo_tournament` are the native replacements for `run.py` and
`tournament.py`, and all three replay files are written from C++. **`run.py` is
deleted** — it had been non-functional since M5 (it needs `run_turn_and_log`,
which §1.2 deliberately kept out of the binding layer) and `oo_run` supersedes it.
`tournament.py` stays for now: it is what `compare_engines.py` drives for the M5
gate, and it goes at M8 with the rest.

**The gate: 120/120 files byte-identical** — 5 configurations × 8 seeds × 3 files,
compared byte for byte against Python's `json.dump` output, not "semantically
equal". Byte-identity is the only standard that can still be checked once Python
is gone, and it catches key-order and separator drift a dict comparison would
accept. Matching it meant reproducing json.dump's defaults exactly: `", "` and
`": "` separators, no indent, insertion-ordered keys (never sorted), and integer
dict keys rendered as strings.

`test_replay` keeps this in ctest without needing Python, by goldening SHA-256
hashes rather than the files (one `board_state.json` is ~270 KB).

**Coverage is reported, not assumed.** A byte-identical result over a corpus that
never produces a `"reason": "cap"` dismount or a temple upgrade would not have
tested those paths at all — the §3.3a trap. `compare_replay_json.py` therefore
prints what the compared corpus exercised, and warns about anything it never hit:

| branch | count | branch | count |
|---|---|---|---|
| battle events | 1 941 | rounds | 2 771 |
| deaths | 4 234 | dismounts | 1 057 |
| **cap-blocked dismounts** | 7 | **winner: null** | 293 |
| structure-defence kills | 591 | archer kills | 180 |
| frozen troops | 3 996 | rectifications | 239 |
| barracks / workshop / temple | 10 025 / 1 235 / 137 | placement swap / keep / draft_auto | 26 / 8 / 6 |

One branch is never exercised and cannot be: `target_choices_submitted` holding a
`null`. `resolve_full_battle` only asks a faction for a target while two or more
are still alive, so `get_legal_target_actions` is never empty at that point and no
current agent can abstain. It would take an agent that deliberately declines to
attack. Flagged rather than papered over.

**The native tournament reproduces the recorded Python findings**, which is a
useful independent check that the agents really are the same players:

| matchup | native (300 games) | recorded in `agents/*.py` |
|---|---|---|
| tactician vs greedy | 45.0 % | 40–58 %, ~45 % combined |
| tactician vs marshal | 18.0 % | ~16.7 % over 120 games |
| 12-agent free-for-all | tactician 1st (29.5 %), marshal 2nd | tactician 1st (31.2 %), marshal 2nd |

400 free-for-all games run in 3.0 s; 300 tactician-vs-greedy games in 2.2 s.

### 6.5 Native game driver

Once agents are native, expose `play_game(config, seeds, assignment)` and
`run_games(...)` so `tournament.py` crosses the language boundary **once per batch**
instead of thousands of times per game. That is also the natural seam for §7's thread
pool. `run.py` keeps the Python-callback path for logged/visualized single games,
where crossing the boundary costs nothing relative to the JSON writing.

**Expected at this gate: 100x+ over today on tactician games** — the ~2x from the
engine, multiplied by removing the agent-side interpreter overhead and the per-decision
GIL round trips, and with `_clone_state` collapsed to a `memcpy`.

---

### 6.8a Running the engine

`run.bat` at the repo root wraps `engine/build/oo_run.exe` so the build path does
not have to be remembered; every argument passes straight through.

```
run                                              one logged game, r7 f8, all tactician
run --list-agents
run --radius 5 --factions 4 --agents tactician,greedy,greedy,random
run --games 200 --agents tactician,marshal --rotate
run --games 50 --agent marshal --replay 7        summary, plus game 7's replay
```

| option | meaning |
|---|---|
| `--radius N` | board radius, 1–8 (radius 9+ cannot be filled — §9) |
| `--factions N` | 1–10 |
| `--games N` | default 1 |
| `--agent NAME` | one kind for every seat |
| `--agents A,B,…` | per-seat kinds, **cycled** if fewer names than factions |
| `--seed N` | base seed; game *g* uses `seed + g` (default: clock) |
| `--rotate` | shift the assignment one seat per game, cancelling seat bias |
| `--replay K` | with `--games > 1`, also write game K's replay files |
| `--max-turns N`, `--out-dir DIR` | safety cap (not a rule) and output location |

A single game also writes the three replay files, so `web_visualizer.html` can
open the result immediately; a batch prints a summary instead, broken down **both**
by agent kind and by seat. The per-seat table is worth having: it shows whether
position mattered, which is exactly what `--rotate` is there to neutralise.

### 6.9 M6d result — DONE

`GameState` is **70 200 -> 18 128 bytes**, a 3.9x reduction, with every gate still
green.

Battle storage was **65 207 of the original 70 200 bytes - 93 % of the state** -
for something empty on almost every hex almost all the time. It is now a small
side table: `battle_index[MAX_HEXES]` plus `Battle battles[64]`, where a `Battle`
is 200 bytes (16 slots of 12). Total battle cost: 13 464 bytes.

`MAX_ACTIVE_BATTLES` dropped from `MAX_HEXES` (331) to **64**, and that bound is
what makes the change a win rather than a loss - at 331 entries the sparse table
would have cost 87 KB, *more* than the dense arrays it replaced. The bound is
provable: a movement step gives each faction at most one move, so at most
`num_factions` new battles per step, over 3 movement + 2 cavalry steps = 5 x 10 =
**50 worst case**. Measured across 15 radius-8 / 10-faction games, the observed
peak was **8**.

Because `assert` compiles out under NDEBUG and running off the end of `battles`
would silently corrupt the whole state, `new_battle` and the slot append also
carry a release-safe guard. They run once per battle, so the check costs nothing.

**`locked` is gone.** A hex is locked exactly when it has a battle, so a separate
bool was a second source of truth that could desync; it is now
`battle_index[h] >= 0`. `state_io` still writes the derived value, which gives
`read_state` a free cross-check that the two representations agree.

**No speedup, and that is the expected result.** Tactician's rollouts went 0.0329
-> 0.0323 s/game, inside noise. The memcpy was never the bottleneck at one state
per rollout. The win is footprint, and it pays off where footprint matters:

| | dense | sparse |
|---|---|---|
| one state | 68.6 KB | 17.7 KB |
| 12 threads (§7) | 823 KB | 212 KB |
| a 10 000-state batch | 700 MB | 181 MB |
| a 1 M-node search tree | 70 GB | 18 GB |

That last row is the actual motivation - it is the difference between a search
tree fitting in memory and not.

**Gates after the refactor:** all 10 ctests; 62 786 agent decisions and a deeper
127 447-decision sweep, 0 differences; 100/100 M5 games; 120/120 replay files
byte-identical.

## 7. Step 6 — multicore (design for it now, build it later)

Not to be built yet, but the constraints above exist to make it a small change:

- Parallelism belongs at the **game** level, not inside a turn. Games are perfectly
  independent, run for milliseconds, and share only the immutable `HexGrid`.
  Intra-turn threading on a 217-hex board would lose to its own synchronization.
- `run_games(config, seeds[], n_threads)` -> results, with a simple work queue.
  Each thread owns its `GameState`, its agents, and its RNGs; nothing is shared
  mutably. On this machine (6 cores / 12 threads) that is a clean ~10x on top of
  everything above.
- Determinism: seed per *game*, never per thread, so results are independent of
  thread count and scheduling. Non-negotiable — it is what keeps parity testing and
  tournament reproducibility working under parallelism.
- Later, for MCTS / NN self-play: a batched `step(states[], actions[])` over an
  SoA layout, so many positions advance together and observations can be encoded
  straight into a tensor. The sparse-battle refactor from §4.3 matters here —
  it is the difference between 65 KB and 10 KB per node.

### 7.1 Result (M7)

Built as designed: `run_games` in `include/oo/run.hpp`, a work queue over a
shared cursor, `--threads N` on `oo_run` (default: all hardware threads, `1`
spawns nothing).

Determinism verified at 1, 2, 3, 5, 7, 12 and 32 threads — byte-identical
summaries. `oo_run` tallies afterwards on one thread in game order, so even the
floating-point sums cannot depend on completion order. `tests/test_run.cpp`
pins the invariant with no golden data and is mutation-tested: seeding per
thread is caught (146 failures), writing results in completion order is caught
(731).

Measured on a 5600X (6 cores / 12 threads), r7 f8:

| threads | greedy games/s | tactician games/s |
|--------:|---------------:|------------------:|
| 1  |  900 |  52.0 |
| 2  | 1774 | 102.0 |
| 4  | 3332 | 189.0 |
| 6  | 4353 | 247.7 |
| 12 | 5732 | 344.4 |

**6.4x on 12 threads, not the ~10x predicted above** — that estimate treated 12
hardware threads as 12 cores. It is 6 physical cores plus SMT: near-linear to 4
threads (3.7x), 4.8x at 6, and SMT adds the remaining ~32 %.

A stale comment on `HexGrid::shared` claiming it was not thread-safe is
corrected; it is mutex-guarded, entries are never removed, and grids are
heap-allocated, so returned references stay valid for the process lifetime.
`run_games` still warms the radius before spawning, to keep the lock off the
hot path.

### 7.2 Profiling (AMD uProf)

Profiled before M7 on the real `/O2 /GL` binary (time-based sampling, 1 kHz).
Release now emits PDBs — `/OPT:REF` and `/OPT:ICF` are restated explicitly
because `/DEBUG` otherwise flips them off, which would change the binary being
measured. `tools/profile.ps1` captures both workloads.

Three hotspots, all fixed, all output-identical (2.7x engine, 1.6x tactician):

| | share before | after |
|---|---|---|
| `eligible_outpost_mask` | 63 % of a greedy game | 11x less time |
| `legal_mask_impl` | 26 % of a tactician game | 2.2x less time |
| `Rng::seed` (tactician rollouts) | 4.5 % | off the chart |

The first was the big one: it swept all 169 hexes per city to clear a ban
radius of 2, which covers 7. `HexGrid` now precomputes neighbourhood (ball)
lists, making the cost independent of board size; `test_grid` gained 346k
checks pinning `ball()` against the golden distance table, because an
under-reporting ball would silently make banned hexes legal — a rules change
no existing parity test would necessarily reach.

**Next tier, not yet done.** The three current leaders are the same shape: a
full-board scan to find the few hexes a faction occupies (`legal_mask_impl`
line 50, `mobile_hexes`, `random_movement`'s cell enumeration). A per-faction
occupied-hex list in `GameState` would collapse all three, at the cost of an
invariant every army mutation must preserve.

---

## 8. Milestones

| # | Deliverable | Gate |
|---|---|---|
| ~~M0~~ | ~~Toolchain, pybind11 zero-copy view smoke test~~ | **done — §2** |
| ~~M1~~ | ~~`rng.hpp` matches CPython~~ | **done — 4 617 edge-case checks + 10⁷ draws over 10⁴ seeds, 0 failures (§3.1)** |
| ~~M2~~ | ~~Grid + state + terrain + collect~~ | **done — 43 105 grid checks over 11 radii; 970 phase cases, 0 failures; 4/4 mutations caught (§3.3a)** |
| ~~M3~~ | ~~buy + movement + battle + turn~~ | **done — 180 turns / 9 064 decisions, 27 movement scenarios, 194 legal-action states, 23 buy scenarios; 13/13 mutations caught (§3.3b)** |
| ~~M4~~ | ~~setup + placement~~ | **done — 320 maps / 29 120 hexes regenerated from seed alone, 19 setup cases; 11/11 mutations caught (§3.3c)** |
| ~~M5~~ | ~~Bindings; `tournament.py` unmodified~~ | **done — 100/100 games identical; 1.7x–15.9x by agent (§5.1). `run.py` deferred to M6c with the native JSON writer.** |
| ~~M6a~~ | ~~Native random/greedy/heuristic/vanguard/marshal~~ | **done — 25 255 decisions and 80/80 whole games identical; 56–65x end to end (§6.6)** |
| ~~M6b~~ | ~~Native tactician + the six leaf agents~~ | **done — 202 921 decisions and 155/155 games identical; tactician 56–73x (§6.7)** |
| ~~M6c~~ | ~~`oo_run` / `oo_tournament` + native JSON~~ | **done — 120/120 files byte-identical; `run.py` deleted (§6.8)** |
| ~~M6d~~ | ~~Sparse battle storage~~ | **done — 68.6 KB -> 17.7 KB; all gates green (§6.9)** |
| M7 | `run_games` thread pool | **DONE** — 6.4x on 12 threads (6 cores + SMT), deterministic per seed |
| M8 | **Python removed** | `-DOO_BUILD_PYTHON=OFF` builds and passes everything; `bindings/`, shims, `engine_old/`, `agents/` deleted. Finish by swapping the CPython RNG for a native one and regenerating the seed-derived corpus (§3.4) — tag first, it is a one-way door |
| M8b | Rules + cleanup window (§11) | auto-clamp merges, RNG swap, `alive[]` dropped; corpus regenerated once. **Must close before M9 generates training data** |
| M9 | Neural policy (§10) | resumable `play_game`, batched encoder, TensorRT inference; a learned policy that beats `tactician` head to head |

M1–M4 are where nearly all the risk lives. M5 is mechanical. M6 is the largest
*volume* of work (~1 200 lines, §6.1) but low risk, since each agent is parity-tested
against its Python original independently and the dependency graph is a clean tree.

---

## 9. Open questions

- ~~**§3.2** — modify `setup.py`'s set iterations for full-pipeline bit parity?~~
  **Resolved: yes. Applied and verified — see §3.2.**
- ~~The pygame visualizers~~ **Resolved: deprecated and deleted** (`1e21ffd`).
  `web_visualizer.html` is the only visualizer going forward. All three JSON files
  are still emitted regardless — see §1.3.
- ~~Duplicate `engine_old/` at the repo root~~ **Resolved: removed; the canonical
  copy is `engine/engine_old/`.**
- ~~`hex_common.py` is orphaned~~ **Resolved: deleted** (`1467630`).
- ~~**Action cards and trading are in the rulebook but absent from the engine.**~~
  **Resolved: dropped from scope.** The no-op `decide_play_cards` hook is already
  in `agent.hpp` and costs nothing, so it stays; no `decide_trade` hook is added
  and no card design work is planned. If cards are ever revived, §10.6's
  autoregressive set-decode for `decide_buy` is the template to copy.
- ~~**The two `_revert_departure` edge cases**~~ **Resolved by §11.1.** Both are
  reached only through the peaceful-merge overstack revert, which is
  `revert_departure`'s single call site. Clamping the merge at collect time
  means nothing is ever sent back, so the lambda becomes dead code and both
  quirks are deleted rather than fixed — and `validate_state` regains the strict
  6-unit check it had to give up for quirk 2. Kept here as the description of
  what is being removed:

  *Quirk 1 — a refused move could start a battle on the mover's own origin.* If
  another faction peacefully moved into that origin during the same step, the
  revert fell to the `else` branch and started a battle there, with the
  reverting army marked `moved = false` (its move was voided, so by end of step
  it never left) — and `moved` gates the Archers ability, so it changed combat.

  *Quirk 2 — a peaceful army could be recreated on a locked hex.*
  `revert_departure` tested `army_faction[origin]` but never `locked(origin)`,
  so an unrelated battle on the origin (which leaves `NO_FACTION`) let the first
  branch write a peaceful army onto a hex with a pending battle. `engine_old`'s
  docstring calls this "a latent quirk present in v1 too", so it predates the
  port.
- ~~`radius >= 9` crashes terrain generation (`BAG_COUNTS` totals 250 hexes vs a
  radius-9 board's 271). Fix, or keep bug-compatible?~~ **Resolved: boards of
  radius 9+ are out of scope.** `oo_run` rejects `--radius > 8` with an
  explanatory error and that stands. Radii 1–8 is therefore a hard bound, which
  §10.3 already assumes — the network's distance-bias buckets never need to
  cover more than distance 16.
- ~~`alive[]` is vestigial (always true, never set false) — drop it?~~
  **Resolved: yes, in the §11.2 window.** Not done now because `state_io`
  serializes an `ALIVE` row, so every state-bearing golden file carries one;
  removing the field today would cost either a fake constant row or a corpus
  regeneration, to save 10 bytes of an 18 128-byte `GameState`.
- Once all twelve agents are native (M6b), `agents/` and `engine_old/` are the parity
  oracles and nothing else. They must be deleted at M8 to hit "no Python in the repo"
  — worth being deliberate that this trades away the fastest place to prototype a new
  heuristic. ~~Archive them on a branch or tag first?~~ **Resolved: no tag. Git
  history is sufficient** — the deleting commit is the boundary, and `git show`
  against any earlier commit recovers the Python oracle if it is ever needed.

---

## 10. Step 7 — a neural policy (design)

Target hardware: RTX 3080 Ti (GA102, 12 GB, ~68 TFLOP/s fp16 tensor with fp32
accumulate, ~20 TFLOP/s realistically achievable on our shapes) plus the 5600X
already driving the engine. CUDA 12.6 is installed.

### 10.1 Scope: policy-only, no search

The first network replaces an agent's decision functions with a learned policy.
No MCTS. That is a deliberate ordering, not timidity:

- The hard part of this milestone is **not the network**, it is restructuring
  `play_game` so many games can be in flight at once (§10.9). Search adds a
  second hard part on top. Do them one at a time.
- Training data is nearly free: the engine produces 5 732 greedy games/s and
  344 tactician games/s on 12 threads. Behaviour-cloning tactician gives a
  working policy and validates the entire pipeline — encoder, batching,
  inference, masking — before any of it has to also be correct under search.
- A policy-only net at the sizes below runs at tens of games/s against
  tactician's 344. It is *slower*. The point is strength and the ability to
  improve by training rather than by hand-editing heuristics.

Search comes after this works.

### 10.2 Why a transformer, and why the usual argument is the weak one

The common case for attention is "long-range information". That is true here
but not decisive: a radius-7 board is 14 hexes across and a hex-conv's
receptive field grows one hex per layer, so ~14 conv layers already see
everything. AlphaZero-scale ResNets are deeper than that.

The real argument is that this game's difficulty is **assignment**, not
perception. Which army goes to which objective, given what the other armies are
doing and where the enemy stacks are, is pairwise relational reasoning over
entities. Every hand-written agent is built from exactly this —
`ranked_attack_targets`, `move_toward(nearest enemy capital)`,
`mobile_hexes_by_size_desc` are all global matching or ranking. Attention
computes that natively; convolution has to launder it through many layers of
local mixing.

The economics happen to be unusually favourable. Per transformer layer with a
4x FFN, cost is `8nd^2` (projections) + `4n^2 d` (attention) + `16nd^2` (FFN).
At d=128 and n=180 that is 70.8 + 16.6 MFLOP = **87 MFLOP, of which attention is
only 19%**. A 128-channel hex-conv residual block is 77.6 MFLOP. So:

> A transformer layer costs about what a residual conv block costs at the same
> width, and has a global receptive field from layer one.

n is small enough that the quadratic term never dominates — 19% of the layer at
r7, 23% at r8, 31% at r10.

### 10.3 Multi-size is a requirement

Decided: **one network across radii 1–8 and 1–10 factions**, not a net per
configuration. `oo_run` exposes radius and faction count as first-class knobs
and an agent that only works at r7/f8 would quietly retire them.

This costs roughly 15% throughput (padding waste, masking) and buys a single net
trainable on pooled data from every configuration — worth it given how cheaply
games are generated.

Everything below is size-agnostic by construction. The three things that make it
so, each of which has a fixed-size alternative that would have been simpler:

1. **Relative distance bias, not learned absolute position embeddings**
   (§10.5). Absolute embeddings would pin the net to one board.
2. **Per-token heads** (§10.6). The movement action space is `n x 6 + 1`, a
   linear `d -> 6` applied to every hex token — it scales with the board rather
   than being a fixed 1015 logits.
3. **Faction tokens with masking** (§10.4), sized `MAX_FACTIONS`, not 8.

**The game is not scale-invariant, though, and that is a separate problem.**
Measured turn counts: r3 = 83.8 turns/game with 19/40 hitting the 100-turn cap,
r5 = 23.3, r7 = 16.0, r8 = 14.9. A radius-3 game is a different game — dense,
all contact, VP accrual barely outrunning the cap. A size-agnostic architecture
will happily run on both and play one of them badly. So `radius`, `num_hexes`
and `num_factions` are fed explicitly into the CLS token, and training must mix
the sizes we care about. Conditioning is cheap; assuming invariance is the
mistake.

### 10.4 Token encoding

Sequence = `n_hex` hex tokens + `MAX_FACTIONS` faction tokens + 1 CLS.
At r7/f8 that is 169 + 10 + 1 = 180 tokens; at r8, 228.

**Everything is encoded relative to the acting faction.** Faction *f*'s relative
index is `(f - acting + num_factions) % num_factions`, so slot 0 is always "me".
This is the single cheapest sample-efficiency win available and it must be in
the encoder from day one — retrofitting it invalidates every checkpoint.

**Hex token** (~55 dims, projected to d):

| feature | dims | encoding |
|---|---:|---|
| terrain | 5 | one-hot (plains/mountain/lake/desert/marsh) |
| passable | 1 | derived, saves the net learning it |
| is_edge | 1 | from `HexGrid::is_edge` |
| city owner | 11 | one-hot: none, then relative faction 0–9 |
| is_capital | 1 | |
| outpost upgrade | 4 | one-hot: none/barracks/workshop/temple |
| army owner | 11 | one-hot: none, then relative faction 0–9 |
| army units | 3 | `log1p` of infantry/cavalry/archers |
| army total | 1 | `log1p`, and `/MAX_STACK_SIZE` |
| frozen | 1 | |
| has_battle | 1 | |
| battle round | 1 | scaled |
| battle participants | 10 | multi-hot over relative faction index |
| is_query_hex | 1 | see below |
| is_query_secondary | 1 | see below |

The two query flags are what let one trunk serve the hex-specific decisions.
`decide_target`, `decide_rectification` and `decide_resource_choice` all ask
about a particular `hex_index`; `decide_swap` involves two hexes (the leftover
and the placer's). Marking them in the token stream means the trunk sees *which*
hex is being asked about, instead of needing a separate model per decision type.

`city_placer` is deliberately excluded — it is setup-only bookkeeping and
`city_owner`/`is_capital` are authoritative afterwards (see `state.hpp`).
`alive[]` is excluded because it is vestigial (§9).

**Faction token** (~25 dims):

| feature | dims |
|---|---:|
| is_me | 1 |
| relative index | 10 (one-hot) |
| is_active | 1 (`f < num_factions`; padding mask) |
| gold | 1 (`log1p`) |
| resources | 4 (`log1p`: wood/iron/clay/fish) |
| victory_points | 1 (`/kVpToWin`) |
| VP gap to the current leader | 1 |
| kill_xp | 1 (`log1p`) |
| outpost count | 1 (`/kOutpostCap`) |
| capital_settle_order | 1 (normalised; it is the VP tie-break) |
| units in play by type | 3 (`log1p`) |

Putting faction state in tokens rather than broadcasting it across 169 hex
planes is a transformer-native advantage with no clean conv equivalent: hexes
attend to faction state and factions attend to the board, and the VP race
becomes directly visible to the policy.

**CLS token** (~18 dims): turn number, turns remaining to the cap, radius,
`num_hexes`, `num_factions`, max VP on the board, movement/cavalry step index,
and a 9-way one-hot for **which decision is being asked**. The phase one-hot
matters even with separate heads — it lets the trunk allocate attention
differently for a buy decision than for a battle target.

### 10.5 Positional information: relative hex-distance bias

No absolute position embeddings. Instead, add a learned bias to the attention
logits indexed by hex distance:

```
logits[head][i][j] += bias[head][bucket(dist(i, j))]
```

`HexGrid` already precomputes the full distance table (§4.2), so this is a table
lookup we have already paid for.

**Bucket logarithmically** — `{0, 1, 2, 3, 4, 5, 6-7, 8-11, 12+}`, 9 buckets.
Two reasons, and the second is the one that matters:

- Max hex distance is `2 x radius`: 14 at r7, 16 at r8, 20 at r10. Train at r7
  and play at r8 and the un-bucketed bins 15–16 are untrained noise. Bucketing
  makes far-apart hexes share a bin, so distances never seen still work.
- It is 9 parameters per head instead of 21, which is a rounding error either
  way, but the prior is better: the model should care a lot about the difference
  between distance 1 and 2, and almost nothing about 15 vs 16.

Richer variant worth trying once the basic version trains: index by
`(distance bucket, direction sector)` over 6 sectors, which captures
directionality — useful because movement is directional and the ban radii in
`eligible_outpost_mask` are not.

Faction and CLS tokens sit outside the hex metric; give them their own bias
buckets (hex-faction, hex-CLS, faction-faction, and so on) rather than a fake
distance.

### 10.6 Trunk and heads

**Trunk:** 2 hex-conv layers as a stem (7-tap, over `HexGrid::neighbours_of`),
then 4–6 pre-LN transformer blocks with the distance bias. The stem is cheap and
encodes adjacency directly, so attention does not have to rediscover "who is
next to me" — which is precisely what the movement action space is defined over.

**One trunk, nine heads.** Not nine networks.

| decision | head | output |
|---|---|---|
| `decide_movement` / `decide_cavalry` | per-hex linear `d->6`, plus a pass logit from CLS | `n x 6 + 1` |
| `decide_placement` | per-hex linear `d->1` | `n` |
| `decide_draft` | same per-hex head, restricted to the pool | `pool_size` |
| `decide_swap` | CLS `d->2` (query hexes are marked in the tokens) | 2 |
| `decide_resource_choice` | query-hex token `d->2` (iron / fish) | 2 |
| `decide_target` | score faction tokens, `d->1`, masked to battle participants | <=10 |
| `decide_rectification` | see below | autoregressive |
| `decide_buy` | see below | autoregressive |
| `decide_play_cards` | none yet (§9) | — |

The movement head falling out of the token layout for free is the strongest
practical argument for one-token-per-hex: the action space *is* the sequence.

**`decide_buy` is a set, not a choice.** `LegalBuyActions` runs to hundreds of
entries (sized 4096) and `ChosenBuyActions` holds up to 512 — the agent picks a
*subset*, under a gold/resource budget and the per-turn batch caps. Proposal:
score each legal action as `MLP([action embedding ; trunk output at action.hex ;
CLS])`, where the action embedding is `(BuyType one-hot 4, unit_type one-hot 3,
upgrade one-hot 3)`, then **decode autoregressively** — pick the best action,
append it, re-score with the chosen set folded into the CLS, repeat until a
learned STOP wins. Budget feasibility is enforced by masking, exactly as
`get_legal_buy_actions` already computes it. Typically 2–5 passes per buy
decision.

**`decide_rectification` is an assignment.** The winner's stack exceeds `cap`
(`MAX_STACK_SIZE = 6`, or 0 when evicted from an enemy capital) and units must
be sent back to the origins recorded in the battle's slots, or lost. Model it
over the battle's `BattleSlot`s (at most `MAX_BATTLE_CONTRIB` = 16), scoring
`(slot, unit_type)` pairs and decoding one unit at a time until the stack is
under cap. Slots are the natural entities; a flat "how many of each type" head
cannot express *which origin* they return to, and `SendBackEntry` requires it.

**Value head:** predict final VP (or win probability) from CLS. Include it even
though this milestone is policy-only — it costs almost nothing, reliably
improves the trunk's representation as an auxiliary loss, and is needed anyway
the moment search is added.

### 10.7 Masking

Legality masking is not optional and the engine already computes every mask
needed: `LegalMask` for movement/cavalry, `legal[MAX_HEXES]` for placement,
`LegalBuyActions` for buys, the battle's participant list for targets. Set
illegal logits to `-inf` before the softmax.

Three padding hazards, all of which fail silently:

- **Hex padding.** Boards are padded to the bucket's token count. Padded tokens
  must be masked as **both queries and keys** — mask them only as keys and the
  padding still attends outward and contributes gradient noise.
- **Faction padding.** Same, for slots `>= num_factions`.
- **`n_hex` vs `MAX_HEXES`.** `legal_mask_impl` already zeroes the
  `n..MAX_HEXES` tail deterministically, so the mask is safe to read at full
  width — but the encoder must not treat those hexes as real terrain, since
  `terrain[]` past `num_hexes` is zero, which decodes as *plains*, not *absent*.

### 10.8 Batching, and a free 8x

`run_turn` takes every faction's decision against the **same `const` state** and
only then applies:

```cpp
for (int f = 0; f < state.num_factions; ++f) {
    legal_movement_mask(state, f, legal);
    decisions.movement(state, f, step, legal, chosen, decisions.ctx);
}
apply_movement_step(state, actions, rng, false);
```

The buy phase has the identical shape. So **all factions' evaluations at a given
step are one batch** — a single game yields batch-8 with no restructuring at
all, and ~128 games in flight gives batch ~1024 naturally.

**Bucket by radius.** Padding r5 (102 tokens) up to r8 (228) wastes 2.6x. Since
the driver chooses which games are in flight, group them by radius and run each
batch at one size:

| board | tokens | MFLOP/layer @ d=128 | padded to r8 |
|---|---:|---:|---:|
| r5 | 102 | 45.4 | 116.3 (2.6x waste) |
| r7 | 180 | 87.4 | 116.3 (1.3x) |
| r8 | 228 | 116.3 | — |

**Send bytes, not planes.** A ~55-plane fp16 encoding is 11.8 KB/position; at
batch 1024 that is 12 MB per batch, 0.48 ms over PCIe 4.0 — which caps
throughput at ~2.1M evals/s regardless of how fast the GPU is. For small nets
the bus, not the GPU, is the bottleneck. Instead transfer a compact byte form
(terrain, city owner, flags, army faction, three unit counts, about 7 B/hex,
~1.3 KB per position) and **expand to planes in a CUDA kernel**. 9x less
traffic, and the bottleneck goes back to compute where it belongs.

### 10.9 Engine work required

The network is the easy part. This is the milestone's real cost:

1. **`play_game` must become resumable.** Today it is a blocking loop; batched
   inference needs N games in flight, each advancing to its next decision point
   and parking while one batched evaluation serves all of them. A state machine
   or coroutine over the phase sequence. This is a larger change than M7.
   Storage is not a concern — 2048 games x 17.7 KB = 36 MB, which is the M6d
   sparse-battle refactor paying off again.
2. **Batched encoder** writing directly into a pinned host buffer in the compact
   byte format, one row per (game, faction) pair.
3. **Inference thread**, double-buffered H2D -> infer -> D2H so CPU and GPU
   overlap. CUDA graphs to kill per-kernel launch overhead, which at ~50 kernels
   x ~5 us is otherwise 250 us per batch.
4. **Determinism, decided: reproducible at a fixed batch size.** M7's contract
   (§7) is that a game's result depends only on its seed, independent of thread
   count. A GPU policy weakens that, because fp16 reductions are not
   batch-invariant — the same position can produce fractionally different logits
   depending on how many other positions shared its batch, and an argmax near a
   tie will then flip.

   The decision is to **accept batch-size-dependent reproducibility** rather
   than pay for deterministic reductions or maintain a parallel CPU inference
   path. Consequences to build around:

   - A replay must record the batch size it ran at, and reproduces only at that
     size. `GameSpec` (§7) gains a batch-size field for NN runs.
   - The M7 `test_run` invariant (thread count cannot change results) still
     holds for scripted agents and must keep running for them. It cannot cover
     NN agents; do not weaken it to accommodate them.
   - Batch composition must therefore be a deterministic function of the seed
     set, not of completion order — so the resumable driver has to fill batches
     in a fixed order, which is a constraint on §10.9 item 1, not a free choice.

### 10.10 Sizes and expected throughput

Evaluations per game, at r7/f8 with the net making *every* decision (15.9
turns/game measured):

| decision | evals/game |
|---|---:|
| movement (3 steps x 8 factions) | 381 |
| cavalry (2 x 8) | 254 |
| buy (8/turn x ~3 autoregressive passes) | 381 |
| resource choice | ~254 |
| target + rectification | ~127 |
| setup (placement, draft, swap) | ~24 |
| **total** | **~1 420** |

That is 2.2x the 636 decisions tactician searches, because tactician only
searches movement and cavalry. **Worth considering: leave the low-stakes
decisions to heuristics.** Dropping resource choice (a binary iron/fish pick) to
the existing greedy rule saves ~18% of all evaluations for almost no strategic
loss.

At ~20 TFLOP/s effective, r7/f8, ~1 420 evals/game:

| config | MFLOP/eval | params | evals/s | games/s |
|---|---:|---:|---:|---:|
| d=128 x 4 blocks | ~370 | ~0.9M | 54k | **38** |
| d=128 x 6 blocks | ~550 | ~1.3M | 36k | 26 |
| d=192 x 6 blocks | ~1 105 | ~2.9M | 18k | 13 |

Against tactician's 344 games/s, so 9–26x slower. That is the expected trade.

**Do not go below d=128.** Tensor cores want K, N >= 128; a d=64 net will not
approach 20 TFLOP/s and buys far less real speed than its FLOP count suggests.
Parameter counts are small by board-game standards (AlphaZero chess was ~40M) —
start at ~1M and grow only when data volume justifies it.

Inference stack: **TensorRT** (or LibTorch), fp16, CUDA graphs. Both are pure
C++, so §1.2's "no Python at run time" survives intact. Training is the one
place Python is hard to avoid; recommendation is a single offline trainer script
kept physically outside `engine/`, since the constraint that matters is the
*engine* not depending on Python, not the research tooling.

### 10.11 Training

**Stage 1 — behaviour cloning.** Log `(state, legal mask, chosen action)` from
tactician self-play; train cross-entropy on the chosen action, plus the value
head on final VP. This validates the entire pipeline against a known-good
teacher and yields an agent that should approach tactician's strength at a
fraction of its per-decision search.

**Stage 2 — policy improvement.** Self-play with the trained policy, keeping
what wins. Without search this is closer to iterated behaviour cloning / league
play than to AlphaZero, and it is where search would take over.

**D6 augmentation: don't bother.** The hex board has 12-fold symmetry (6
rotations x reflection), so any training example `(S, A)` yields 11 more as
`(g.S, g.A)` — two permutation tables, hex index and direction index, both
cheap for `HexGrid` to precompute. AlphaGo used the square board's 8-fold
symmetry exactly this way.

It is not worth it here, for a reason specific to this project: **augmentation
stretches scarce data, and our data is not scarce.** The engine generates 5 732
greedy and 344 tactician games/s. Twelve rotations of one game carry strictly
less information than twelve fresh games, which bring new terrain, new openings
and new positions. Generate more games instead.

There is also a second-order wrinkle worth recording, because it will show up
in metrics whether or not augmentation is used. The teacher's tie-breaks are
enumeration-order artifacts: tactician keeps the first maximum
(`score > best_score`, strict), and candidate order comes from hex and
direction index order — the same reason the port needed `std::stable_sort` and
`std::min_element` throughout (§4). Rotate the board and the index order
changes, so a *different* one of several equally-scored moves comes first.
`g.A` is therefore still a tied-optimal move, just not the one the teacher
would have picked. Consequences:

- "Top-1 agreement with tactician" reads lower than the policy deserves on
  tied positions. Do not read that as weakness.
- Hard cross-entropy cannot reach zero on ties; the net learns to spread mass
  across tied moves, which is arguably better play than cloning an index-order
  artifact.

The symmetry remains useful in two other forms: as an architectural prior (an
equivariant trunk gets it for free rather than learning it), and as test-time
augmentation — average the policy over all 12 rotations for a smoother output,
at 12x inference cost.

### 10.12 Open questions

- **Which decisions get a network at all.** Full coverage is ~1 420 evals/game;
  movement + cavalry + buy only is ~1 020. Recommend starting narrow and
  widening once each head is shown to beat its heuristic.
- **Board-size curriculum.** Train on pooled r5–r8, or start at r7 and widen?
  Pooled is more honest but slower to converge; r3 in particular is nearly a
  different game and may deserve exclusion rather than inclusion.
- **`decide_play_cards` / `decide_trade`** (§9) — if action cards ever land, the
  buy head's autoregressive set-decode is the natural template, which is a
  further argument for building it that way rather than as a fixed-width head.

---

## 11. Rules and cleanup changes — the post-M8 window

Everything here **changes game outcomes or the golden state format**, so all of it
lands in one window, after M8, and the seed-derived corpus (§3.4) is regenerated
exactly once rather than once per change. Doing any of it earlier means
maintaining parity against a Python engine that plays a different game.

**Ordering constraint:** this window must close *before* §10 generates training
data. Encoder features, net width, even multi-size support can all be revised
incrementally; the **action space cannot** — changing it invalidates every
checkpoint and forces retraining from scratch.

Contents of the window: the RNG swap (§3.4), the movement clamp (§11.1), and
`alive[]` (§11.2).

### 11.1 Peaceful merges auto-clamp to the stack cap

**Decided.** Today, a move whose peaceful merge would exceed `MAX_STACK_SIZE`
is submitted, accepted as legal, silently refused at resolution, and reverted —
which is where both `_revert_departure` quirks (§9) come from. The new rule:

> A move carries at most as many units as the destination can hold. If the
> destination already holds `MAX_STACK_SIZE` of your own units, the move is
> illegal.

So a 5-stack moving onto your own 2-stack moves 4 and leaves 1 behind; onto your
own 6-stack, the move is not offered at all.

**The clamp goes at COLLECT time, not at resolution time.** This is the whole
decision, and picking the other placement loses most of the benefit:

- *Collect time* — `units_to_move` clamps against the destination's occupancy in
  the **pre-step** state. The remainder never leaves the origin, so nothing ever
  has to be sent back, `revert_departure` becomes dead code, and **both §9
  quirks are deleted rather than fixed**. `validate_state` can then go back to
  enforcing the 6-unit cap strictly, dropping the locked-hex exemption added
  purely to tolerate quirk 2.
- *Resolution time* — clamping inside the peaceful branch and returning the
  excess keeps the revert path alive, so both quirks survive, merely with fewer
  units involved. Cheaper to write, fixes nothing.

Collect-time clamping is also the better **design**, not just the cheaper one.
The clamp is computed against the board as the player sees it when deciding, so
how many units will move is knowable at decision time. Resolution-time clamping
would depend on what opponents did simultaneously, which the player cannot see.

*Accepted cost.* Because the clamp reads the pre-step board, an opponent moving
into the same destination turns the merge into a battle — and battles are not
stack-capped, so the clamped units would ideally all have joined. You arrive with
fewer than the uncapped rules would allow. This is deliberate: the rule is "a
move carries what fits at its destination", evaluated once, deterministically.

*Legal-mask change.* Only the fully-blocked case becomes illegal — destination
holds `MAX_STACK_SIZE` of your own units. This is exactly maskable with no
simultaneity hazard: `MoveActions` holds one move per faction per step, so for a
peaceful merge both stacks are yours and the sitting one cannot also be moving.
The condition is a pure function of the acting faction's own pre-step state:

```
units_to_move(from) + units_at(neighbour(from, dir)) > MAX_STACK_SIZE   // clamp
units_at(neighbour(from, dir)) >= MAX_STACK_SIZE                        // illegal
```

Masking the blocked case matters because a faction gets **one move per step** —
offering a move that carries zero units would waste it.

*Scope.* Clamping applies only when the destination holds the mover's own army.
Empty destinations and enemy-held destinations are unaffected: the latter start
battles, which are uncapped by design and reconciled afterwards by
`rectify_overflow`.

*Cavalry steps* already move a partial stack (`units_to_move(cavalry_only)` takes
cavalry and leaves infantry and archers behind), so they clamp the same way with
no additional machinery.

**Explicit unit selection was considered and rejected for now.** Letting the
player name which units move is the richer game, and the engine would take it
more easily than expected — partial-stack movement already exists for cavalry,
and multi-slot-per-faction battles are already supported
(`MAX_BATTLE_CONTRIB` = 16 against `MAX_FACTIONS` = 10). The cost is elsewhere:
the action space grows from `(hex, dir)` = 1015 to ~84 000 (83 non-empty splits
of a 6-stack across 3 unit types), and all twelve scripted agents need a split
policy. If it is ever revisited, the §10 policy head extends cleanly — add a
second per-hex head emitting 83 split logits (169 x 83 = 14k outputs, riding the
same forward pass), decode `(hex, dir)` first and read that hex's split
distribution. Note it would have to happen in this same window, before training
data exists.

### 11.2 Remove `alive[]`

**Decided: remove, but here rather than now.** The field is vestigial — always
true, never set false (§9).

It is not as free as it looks. `state_io` serializes an `ALIVE` row into the
canonical state format, so every state-bearing golden file carries one (970
phase cases, plus turn traces, movement scenarios, setup cases). Removing the
field today would mean either emitting a fake constant row purely to keep the
corpus valid, or regenerating the corpus — to save 10 bytes of an 18 128-byte
`GameState`. Neither is worth doing on its own; in this window the corpus is
regenerated anyway.

Three call sites to settle when it happens:

- `state_io` — drop the `ALIVE` row from both writer and reader, and from
  `compare_states`.
- `json.cpp` — `board_state.json` emits `"alive"`. `web_visualizer.html` already
  reads a missing value as alive (`stats.alive !== false`, line 850), so it can
  be dropped from the JSON too; that is a one-line viewer-format change, and the
  replay hashes are being regenerated regardless.
- `log.hpp` / `turn_log.cpp` — `RoundLog`'s per-faction snapshot carries it.

