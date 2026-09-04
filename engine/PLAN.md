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
| `run.py`, `tournament.py` | replaced by `oo_run` / `oo_tournament` |
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

---

## 8. Milestones

| # | Deliverable | Gate |
|---|---|---|
| ~~M0~~ | ~~Toolchain, pybind11 zero-copy view smoke test~~ | **done — §2** |
| ~~M1~~ | ~~`rng.hpp` matches CPython~~ | **done — 4 617 edge-case checks + 10⁷ draws over 10⁴ seeds, 0 failures (§3.1)** |
| ~~M2~~ | ~~Grid + state + terrain + collect~~ | **done — 43 105 grid checks over 11 radii; 970 phase cases, 0 failures; 4/4 mutations caught (§3.3a)** |
| ~~M3~~ | ~~buy + movement + battle + turn~~ | **done — 180 turns / 9 064 decisions, 27 movement scenarios, 194 legal-action states, 23 buy scenarios; 13/13 mutations caught (§3.3b)** |
| ~~M4~~ | ~~setup + placement~~ | **done — 320 maps / 29 120 hexes regenerated from seed alone, 19 setup cases; 11/11 mutations caught (§3.3c)** |
| M5 | Bindings; `run.py` / `tournament.py` unmodified | identical results, ~2x |
| M6a | Native random/greedy/heuristic/vanguard/marshal | per-agent parity vs Python |
| M6b | Native tactician + the six leaf agents | per-agent parity; all 12 native |
| M6c | `oo_run` / `oo_tournament` executables + native JSON | all three JSON files byte-identical (§1.3); `web_visualizer.html` loads it; 100x+ |
| M6d | Sparse battle storage | `GameState` ~10 KB, parity holds |
| M7 | `run_games` thread pool | ~10x on 12 threads, deterministic per seed |
| M8 | **Python removed** | `-DOO_BUILD_PYTHON=OFF` builds and passes everything; `bindings/`, shims, `engine_old/`, `agents/` deleted |

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
- `hex_common.py` is now orphaned — the deleted visualizers were its only importers,
  and `web_visualizer.html` has its geometry inlined already ("ported from
  hex_common.py"). Safe to delete; say the word.
- **Action cards and trading are in the rulebook but entirely absent from the engine**
  — no deck, no hands, no per-faction trade actions; `action_cards.md` says the pool
  is still undesigned. Not a reason to delay the port, but worth one cheap decision
  now: **add a no-op `decide_play_cards` to the `Agent` interface (§6.3) up front.**
  Adding a field to a POD `GameState` later is trivial; adding a tenth decision
  method to twelve already-written C++ agents is not. One virtual with a default
  empty implementation costs nothing and removes the only expensive part of adding
  cards later. Same argument for a `decide_trade` hook, though trading between
  scripted agents is close to meaningless until there is a policy that can value an
  offer.
- The two `_revert_departure` edge cases are ported bugs. Keep bug-compatible for
  now — parity is worth more than tidiness — but they should be revisited on their
  own once the port is green.
- `radius >= 9` currently crashes terrain generation (`BAG_COUNTS` totals 250 hexes
  vs a radius-9 board's 271, per `tournament.py`'s comment). Fix during the port, or
  keep bug-compatible? Recommend fixing, and recording it as an intentional divergence.
- `alive[]` is vestigial (always true, never set false). Now safely droppable from
  `GameState`: nothing in `engine_old` or `agents/` reads it except the log snapshot
  and `tactician._clone_state`, and `web_visualizer.html` tests it defensively
  (`stats.alive !== false`, line 850) so a missing field reads as alive. Recommend
  dropping the field and emitting `"alive": true` in the log for format
  compatibility — or dropping it from the log too and letting the viewer's default
  handle it, if we accept a one-line viewer-format change.
- Once all twelve agents are native (M6b), `agents/` and `engine_old/` are the parity
  oracles and nothing else. They must be deleted at M8 to hit "no Python in the repo"
  — worth being deliberate that this trades away the fastest place to prototype a new
  heuristic. Archiving them on a branch or tag costs nothing and keeps that option.
