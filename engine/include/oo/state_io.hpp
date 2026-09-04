// Canonical text serialization of a GameState, shared with Python.
//
// This is the backbone of the parity work (PLAN.md §3.3): tools/state_io.py writes
// the identical format from engine_old's ArrayState, so a C++ phase can be run
// against a Python-produced "before" state and its result compared to the
// Python-produced "after" state, field by field.
//
// Text rather than binary on purpose. These files are read by a human the moment
// anything disagrees, and the volumes involved (a few hundred small states) make
// compactness irrelevant next to being able to eyeball a diff.
//
// Battle contributions are stored SPARSELY (only occupied slots) because battles
// are rare - a dense [331][16] dump would be almost entirely padding.

#pragma once

#include "oo/state.hpp"

#include <iosfwd>
#include <string>

namespace oo {

// Writes one STATE ... END block.
void write_state(std::ostream& out, const GameState& state);

// Reads one STATE ... END block, resolving the grid via HexGrid::shared(radius).
// Returns false and fills `error` if the stream does not hold a well-formed block.
bool read_state(std::istream& in, GameState& state, std::string& error);

// Field-by-field comparison. Returns true if identical; otherwise `diff` describes
// the FIRST difference found, naming the field and index. Order of checks is
// deliberately board-first so a divergence reports the most specific location
// available rather than a downstream symptom.
bool compare_states(const GameState& a, const GameState& b, std::string& diff);

// Debug-build invariant check (PLAN.md §4.5). Returns true if `state` is
// self-consistent; otherwise `problem` says what is wrong. Checks that battle
// storage agrees with battle_order/locked, that occupied battle slots are
// contiguous from 0, that no army sits on impassable terrain, and that peaceful
// stacks respect MAX_STACK_SIZE.
bool validate_state(const GameState& state, std::string& problem);

}  // namespace oo
