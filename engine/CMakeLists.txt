cmake_minimum_required(VERSION 3.20)
project(oaths_and_omens_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(MSVC)
  add_compile_options(/W4 /permissive-)
  # Release: full optimization + AVX2 + link-time codegen. /fp:fast is safe here
  # because no float value is ever compared for parity - random() is built from
  # exact integer arithmetic on powers of two (see rng.hpp), and the evaluation
  # heuristics are ranked, not equality-tested.
  add_compile_options($<$<CONFIG:Release>:/O2>
                      $<$<CONFIG:Release>:/fp:fast>
                      $<$<CONFIG:Release>:/arch:AVX2>
                      $<$<CONFIG:Release>:/GL>)
  # Symbols in the OPTIMIZED build. A sampling profiler (AMD uProf, Very Sleepy,
  # WPA) can only name a hot address if there is a PDB for it, and profiling a
  # separate unoptimized build would measure a program we do not ship.
  #
  # /Zi does not change codegen. /DEBUG does change the LINKER: it flips the
  # defaults to /OPT:NOREF,NOICF, which would leave dead code in and stop
  # identical-function folding - a different binary from the one we benchmark.
  # So both are re-stated explicitly, and the executable is byte-equivalent to
  # what it was before, plus a .pdb beside it.
  add_compile_options($<$<CONFIG:Release>:/Zi>)
  add_link_options($<$<CONFIG:Release>:/LTCG>
                   $<$<CONFIG:Release>:/DEBUG>
                   $<$<CONFIG:Release>:/OPT:REF>
                   $<$<CONFIG:Release>:/OPT:ICF>)
  # Debug keeps assertions on - the bounds-checked, invariant-asserting build is
  # what makes the parity work tractable (PLAN.md §4.5).
  add_compile_options($<$<CONFIG:Debug>:/Od> $<$<CONFIG:Debug>:/RTC1>)
else()
  add_compile_options(-Wall -Wextra)
  add_compile_options($<$<CONFIG:Release>:-O3>)
endif()

# --- core library -----------------------------------------------------------
# Sources get added here as modules land. rng.hpp stays header-only (it is small
# and hot enough to want inlined).
add_library(oo_core STATIC
  src/grid.cpp
  src/state.cpp
  src/state_io.cpp
  src/terrain.cpp
  src/collect.cpp
  src/buy.cpp
  src/movement.cpp
  src/battle.cpp
  src/turn.cpp
  src/setup.cpp
  src/placement.cpp
  src/agent_util.cpp
  src/agents.cpp
  src/game.cpp
  src/run.cpp
  src/turn_log.cpp
  src/json.cpp
)
target_include_directories(oo_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# std::thread for the §7 game-level pool. A no-op on MSVC; the explicit link is
# what makes the same source build on a POSIX toolchain.
find_package(Threads REQUIRED)
target_link_libraries(oo_core PUBLIC Threads::Threads)

# --- apps (the eventual replacements for run.py / tournament.py) ------------
add_executable(oo_run apps/oo_run.cpp)
target_link_libraries(oo_run PRIVATE oo_core)

add_executable(oo_tournament apps/oo_tournament.cpp)
target_link_libraries(oo_tournament PRIVATE oo_core)

# --- tests ------------------------------------------------------------------
enable_testing()

add_executable(test_rng tests/test_rng.cpp)
target_link_libraries(test_rng PRIVATE oo_core)

# The golden traces under tests/data/ were generated from the Python reference
# before it was deleted at M8. They are checked in and are now the only record
# of it. Nothing can regenerate them from Python any more; PLAN.md §3.4
# describes the --rewrite path that reblesses the seed-derived ones from the
# C++ engine when the RNG is swapped at M8b.
add_test(NAME rng_parity
         COMMAND test_rng ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/rng_golden.txt)

add_executable(test_rng_stress tests/test_rng_stress.cpp)
target_link_libraries(test_rng_stress PRIVATE oo_core)

add_test(NAME rng_stress
         COMMAND test_rng_stress ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/rng_stress.txt)

add_executable(test_grid tests/test_grid.cpp)
target_link_libraries(test_grid PRIVATE oo_core)
add_test(NAME grid_parity
         COMMAND test_grid ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/grid_golden.txt)

add_executable(test_phases tests/test_phases.cpp)
target_link_libraries(test_phases PRIVATE oo_core)
add_test(NAME phase_parity
         COMMAND test_phases ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/phase_cases.txt)

add_executable(test_replay tests/test_replay.cpp)
target_link_libraries(test_replay PRIVATE oo_core)
add_test(NAME replay_parity
         COMMAND test_replay ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/replay_hashes.txt)

add_executable(test_agents tests/test_agents.cpp)
target_link_libraries(test_agents PRIVATE oo_core)
add_test(NAME agent_parity
         COMMAND test_agents ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/agent_games.txt)

add_executable(test_setup tests/test_setup.cpp)
target_link_libraries(test_setup PRIVATE oo_core)
add_test(NAME setup_parity
         COMMAND test_setup ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/setup_cases.txt)

add_executable(test_buy tests/test_buy.cpp)
target_link_libraries(test_buy PRIVATE oo_core)
add_test(NAME buy_parity
         COMMAND test_buy ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/legal_cases.txt
                          ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/buy_scenarios.txt)

add_executable(test_movement tests/test_movement.cpp)
target_link_libraries(test_movement PRIVATE oo_core)
add_test(NAME movement_scenarios
         COMMAND test_movement ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/movement_scenarios.txt)

# No golden data: this one asserts an invariant (thread count cannot change
# results), so it stays valid as the rules evolve.
add_executable(test_run tests/test_run.cpp)
target_link_libraries(test_run PRIVATE oo_core)
add_test(NAME run_determinism COMMAND test_run)

add_executable(test_turn tests/test_turn.cpp)
target_link_libraries(test_turn PRIVATE oo_core)
add_test(NAME turn_parity
         COMMAND test_turn ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/turn_traces.txt)
