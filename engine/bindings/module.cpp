// pybind11 bindings - TRANSITIONAL SCAFFOLDING, deleted at M8 (PLAN.md §1.2, §5).
//
// This exists for one purpose: to let the twelve existing Python agents and
// run.py / tournament.py drive the C++ engine UNMODIFIED. That is the cheapest
// strong integration test available - twelve independently written policies
// hammering every engine entry point - and it is worth building even though it
// gets thrown away.
//
// It is deliberately not polished. Do not grow features here; anything the final
// engine needs belongs in src/ behind a plain C++ API.
//
// THE ONE NON-OBVIOUS PIECE is RNG sharing. run_turn and friends are handed a live
// Python random.Random. Calling back into it per die roll would work but is slow;
// instead, because oo::Rng is bit-compatible with CPython's Mersenne Twister
// (§3.1), we borrow the generator's state via getstate(), run the whole phase
// natively, and write the advanced state back with setstate(). The stream stays
// exactly shared with Python, at native speed.
//
// The other thing worth knowing: every array is exposed as a numpy VIEW onto the
// C++ struct's memory, sized to num_hexes / num_factions rather than the fixed
// MAX_HEXES capacity. Sizing matters - a view over the padding would make
// `state.city_owner == NO_FACTION` report hundreds of off-board hexes as free.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "oo/agent.hpp"
#include "oo/battle.hpp"
#include "oo/buy.hpp"
#include "oo/collect.hpp"
#include "oo/movement.hpp"
#include "oo/placement.hpp"
#include "oo/setup.hpp"
#include "oo/state.hpp"
#include "oo/terrain.hpp"
#include "oo/turn.hpp"

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace oo;

namespace {

// --- RNG bridging -----------------------------------------------------------

// Borrows a Python random.Random's internal state. getstate() returns
// (version, tuple_of_625_ints, gauss_next).
Rng rng_from_python(py::object py_rng) {
    Rng rng;
    if (py_rng.is_none()) return rng;
    py::tuple st = py_rng.attr("getstate")();
    py::tuple words = st[1].cast<py::tuple>();
    uint32_t raw[625];
    for (int i = 0; i < 625; ++i) raw[i] = words[static_cast<size_t>(i)].cast<uint32_t>();
    rng.set_state(raw);
    return rng;
}

void rng_to_python(py::object py_rng, const Rng& rng) {
    if (py_rng.is_none()) return;
    uint32_t raw[625];
    rng.get_state(raw);
    py::tuple words(625);
    for (int i = 0; i < 625; ++i) words[static_cast<size_t>(i)] = raw[i];
    py::tuple st(3);
    st[0] = 3;
    st[1] = words;
    st[2] = py::none();
    py_rng.attr("setstate")(st);
}

// RAII: borrow on entry, hand back on scope exit, so every early return still
// leaves the Python generator correctly advanced.
struct BorrowedRng {
    py::object py_rng;
    Rng rng;
    explicit BorrowedRng(py::object o) : py_rng(o), rng(rng_from_python(o)) {}
    ~BorrowedRng() { rng_to_python(py_rng, rng); }
};

// --- array views -------------------------------------------------------------

template <class T>
py::array_t<T> view1(py::object self, const T* data, int n) {
    return py::array_t<T>({static_cast<py::ssize_t>(n)},
                          {static_cast<py::ssize_t>(sizeof(T))}, data, self);
}

template <class T, int D1>
py::array_t<T> view2(py::object self, const T (*data)[D1], int n) {
    return py::array_t<T>(
        {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(D1)},
        {static_cast<py::ssize_t>(sizeof(T) * D1), static_cast<py::ssize_t>(sizeof(T))},
        &data[0][0], self);
}

template <class T, int D1, int D2>
py::array_t<T> view3(py::object self, const T (*data)[D1][D2], int n) {
    return py::array_t<T>({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(D1),
                           static_cast<py::ssize_t>(D2)},
                          {static_cast<py::ssize_t>(sizeof(T) * D1 * D2),
                           static_cast<py::ssize_t>(sizeof(T) * D2),
                           static_cast<py::ssize_t>(sizeof(T))},
                          &data[0][0][0], self);
}

// --- buy action <-> dict ------------------------------------------------------

const char* kUnitNames[NUM_UNIT_TYPES] = {"infantry", "cavalry", "archers"};
const char* kUpgradeNames[NUM_UPGRADE_TYPES] = {"barracks", "workshop", "temple"};

py::dict buy_action_to_dict(const BuyAction& a) {
    py::dict d;
    switch (a.type) {
        case BuyType::kBuyInfantry:
            d["type"] = "buy_infantry";
            d["city_hex"] = static_cast<int>(a.hex);
            break;
        case BuyType::kConvertToSpecial:
            d["type"] = "convert_to_special";
            d["hex"] = static_cast<int>(a.hex);
            d["unit_type"] = kUnitNames[a.unit_type];
            break;
        case BuyType::kBuildOutpost:
            d["type"] = "build_outpost";
            d["hex"] = static_cast<int>(a.hex);
            d["unit_type"] = kUnitNames[a.unit_type];
            break;
        case BuyType::kUpgradeOutpost:
            d["type"] = "upgrade_outpost";
            d["hex"] = static_cast<int>(a.hex);
            d["upgrade"] = kUpgradeNames[a.upgrade];
            break;
    }
    return d;
}

int unit_index_from_name(const std::string& name) {
    if (name == "infantry") return kInfantry;
    if (name == "cavalry") return kCavalry;
    return kArchers;
}

int upgrade_index_from_name(const std::string& name) {
    if (name == "barracks") return kBarracks;
    if (name == "workshop") return kWorkshop;
    return kTemple;
}

// Returns false for an unrecognised dict, which the caller drops - engine_old's
// _apply_one likewise just returns False for anything it does not understand.
bool buy_action_from_dict(const py::dict& d, BuyAction& out) {
    const std::string type = d["type"].cast<std::string>();
    out = BuyAction{};
    if (type == "buy_infantry") {
        out.type = BuyType::kBuyInfantry;
        out.hex = d["city_hex"].cast<int16_t>();
    } else if (type == "convert_to_special") {
        out.type = BuyType::kConvertToSpecial;
        out.hex = d["hex"].cast<int16_t>();
        out.unit_type = static_cast<int8_t>(unit_index_from_name(d["unit_type"].cast<std::string>()));
    } else if (type == "build_outpost") {
        out.type = BuyType::kBuildOutpost;
        out.hex = d["hex"].cast<int16_t>();
        out.unit_type = static_cast<int8_t>(unit_index_from_name(d["unit_type"].cast<std::string>()));
    } else if (type == "upgrade_outpost") {
        out.type = BuyType::kUpgradeOutpost;
        out.hex = d["hex"].cast<int16_t>();
        out.upgrade = static_cast<int8_t>(upgrade_index_from_name(d["upgrade"].cast<std::string>()));
    } else {
        return false;
    }
    return true;
}

// --- decision dispatch --------------------------------------------------------

// Holds the {faction: callable} dicts run_turn is given and routes each C++
// callback to the right Python object. `state_obj` is the already-constructed
// Python wrapper, reused for every call so agents see one stable object.
struct PyDecisions {
    py::object state_obj;
    py::dict buy, movement, cavalry, target, rectification, resource_choice;
};

void py_buy(const GameState&, int faction, const LegalBuyActions& legal, ChosenBuyActions& out,
            void* ctx) {
    PyDecisions& d = *static_cast<PyDecisions*>(ctx);
    py::list legal_list;
    for (int i = 0; i < legal.size(); ++i) legal_list.append(buy_action_to_dict(legal[i]));
    py::object chosen = d.buy[py::int_(faction)](d.state_obj, faction, legal_list);
    out.clear();
    if (chosen.is_none()) return;
    for (py::handle h : chosen.cast<py::sequence>()) {
        BuyAction action{};
        if (buy_action_from_dict(h.cast<py::dict>(), action)) out.push_back(action);
    }
}

py::array_t<bool> mask_to_numpy(const LegalMask& mask, int num_hexes) {
    py::array_t<bool> arr({static_cast<py::ssize_t>(num_hexes),
                           static_cast<py::ssize_t>(NUM_DIRECTIONS)});
    auto m = arr.mutable_unchecked<2>();
    for (int h = 0; h < num_hexes; ++h) {
        for (int dir = 0; dir < NUM_DIRECTIONS; ++dir) m(h, dir) = mask.cell[h][dir];
    }
    return arr;
}

bool py_move_impl(py::dict& table, const GameState& state, py::object state_obj, int faction,
                  int step, const LegalMask& legal, Move& out) {
    py::object got = table[py::int_(faction)](state_obj, faction, step,
                                              mask_to_numpy(legal, state.num_hexes));
    if (got.is_none()) return false;
    py::sequence pair = got.cast<py::sequence>();
    out.hex = pair[0].cast<int16_t>();
    out.dir = pair[1].cast<int8_t>();
    return true;
}

bool py_movement(const GameState& state, int faction, int step, const LegalMask& legal, Move& out,
                 void* ctx) {
    PyDecisions& d = *static_cast<PyDecisions*>(ctx);
    return py_move_impl(d.movement, state, d.state_obj, faction, step, legal, out);
}

bool py_cavalry(const GameState& state, int faction, int step, const LegalMask& legal, Move& out,
                void* ctx) {
    PyDecisions& d = *static_cast<PyDecisions*>(ctx);
    return py_move_impl(d.cavalry, state, d.state_obj, faction, step, legal, out);
}

int py_target(const GameState&, int hex_index, int faction, void* ctx) {
    PyDecisions& d = *static_cast<PyDecisions*>(ctx);
    py::object got = d.target[py::int_(faction)](d.state_obj, hex_index, faction);
    return got.is_none() ? -1 : got.cast<int>();
}

void py_rectification(const GameState&, int hex_index, int winner, int cap, SendBack& out,
                      void* ctx) {
    PyDecisions& d = *static_cast<PyDecisions*>(ctx);
    py::object got = d.rectification[py::int_(winner)](d.state_obj, hex_index, winner, cap);
    out.clear();
    if (got.is_none()) return;
    for (py::handle h : got.cast<py::sequence>()) {
        py::dict entry = h.cast<py::dict>();
        SendBackEntry e{};
        py::object origin = entry["origin_hex"];
        e.origin_hex = origin.is_none() ? NO_ORIGIN : origin.cast<int32_t>();
        py::sequence units = entry["units"].cast<py::sequence>();
        for (int t = 0; t < NUM_UNIT_TYPES; ++t) {
            e.units[t] = units[static_cast<size_t>(t)].cast<int16_t>();
        }
        out.push_back(e);
    }
}

Resource py_resource_choice(const GameState&, int faction, int hex_index, void* ctx) {
    PyDecisions& d = *static_cast<PyDecisions*>(ctx);
    py::object got = d.resource_choice[py::int_(faction)](d.state_obj, faction, hex_index);
    return got.cast<std::string>() == "iron" ? kIron : kFish;
}

// --- setup decision dispatch --------------------------------------------------

struct PySetupDecisions {
    py::object state_obj;
    py::dict placement, draft, swap;
};

int py_placement(const GameState& state, int faction, const bool* legal, void* ctx) {
    PySetupDecisions& d = *static_cast<PySetupDecisions*>(ctx);
    py::array_t<bool> mask({static_cast<py::ssize_t>(state.num_hexes)});
    auto m = mask.mutable_unchecked<1>();
    for (int i = 0; i < state.num_hexes; ++i) m(i) = legal[i];
    py::object got = d.placement[py::int_(faction)](d.state_obj, faction, mask);
    return got.is_none() ? -1 : got.cast<int>();
}

int py_draft(const GameState&, int faction, const int16_t* pool, int pool_size, void* ctx) {
    PySetupDecisions& d = *static_cast<PySetupDecisions*>(ctx);
    py::list options;
    for (int i = 0; i < pool_size; ++i) options.append(static_cast<int>(pool[i]));
    py::object got = d.draft[py::int_(faction)](d.state_obj, faction, options);
    return got.is_none() ? -1 : got.cast<int>();
}

bool py_swap(const GameState&, int faction, int leftover, int placer, int placer_hex, void* ctx) {
    PySetupDecisions& d = *static_cast<PySetupDecisions*>(ctx);
    py::object got = d.swap[py::int_(faction)](d.state_obj, faction, leftover, placer, placer_hex);
    return !got.is_none() && got.cast<bool>();
}

// Copies a numpy array's first `n` elements into a C++ buffer.
template <class Dst, class Src>
void copy_in(py::object src, Dst* dst, int n) {
    auto arr = src.cast<py::array_t<Src, py::array::c_style | py::array::forcecast>>();
    auto r = arr.template unchecked<1>();
    for (int i = 0; i < n; ++i) dst[i] = static_cast<Dst>(r(i));
}

}  // namespace

PYBIND11_MODULE(oo_engine, m) {
    m.doc() = "Transitional pybind11 bindings for the C++ Oaths & Omens engine (PLAN.md §5)";

    // --- HexGrid ------------------------------------------------------------
    // nodelete holder: grids are immutable, cached statics shared by every state
    // and thread (PLAN.md §4.2), so Python must never take ownership of one.
    py::class_<HexGrid, std::unique_ptr<HexGrid, py::nodelete>>(m, "HexGrid")
        .def(py::init([](int radius) {
            return const_cast<HexGrid*>(&HexGrid::shared(radius));
        }))
        .def_property_readonly("radius", &HexGrid::radius)
        .def_property_readonly("num_hexes", &HexGrid::num_hexes)
        .def_property_readonly("coords",
                               [](const HexGrid& g) {
                                   py::list out;
                                   for (int i = 0; i < g.num_hexes(); ++i) {
                                       const HexCoord& c = g.coord_of(i);
                                       out.append(py::make_tuple(c.q, c.r, c.s));
                                   }
                                   return out;
                               })
        .def_property_readonly("coords_array",
                               [](const HexGrid& g) {
                                   py::array_t<int32_t> arr(
                                       {static_cast<py::ssize_t>(g.num_hexes()),
                                        static_cast<py::ssize_t>(3)});
                                   auto r = arr.mutable_unchecked<2>();
                                   for (int i = 0; i < g.num_hexes(); ++i) {
                                       const HexCoord& c = g.coord_of(i);
                                       r(i, 0) = c.q;
                                       r(i, 1) = c.r;
                                       r(i, 2) = c.s;
                                   }
                                   return arr;
                               })
        .def_property_readonly("neighbor_table",
                               [](const HexGrid& g) {
                                   py::array_t<int32_t> arr(
                                       {static_cast<py::ssize_t>(g.num_hexes()),
                                        static_cast<py::ssize_t>(NUM_DIRECTIONS)});
                                   auto r = arr.mutable_unchecked<2>();
                                   for (int i = 0; i < g.num_hexes(); ++i) {
                                       for (int d = 0; d < NUM_DIRECTIONS; ++d) {
                                           r(i, d) = g.neighbour(i, d);
                                       }
                                   }
                                   return arr;
                               })
        .def_property_readonly("coord_to_index",
                               [](const HexGrid& g) {
                                   py::dict out;
                                   for (int i = 0; i < g.num_hexes(); ++i) {
                                       const HexCoord& c = g.coord_of(i);
                                       out[py::make_tuple(c.q, c.r, c.s)] = i;
                                   }
                                   return out;
                               })
        .def("coord_of",
             [](const HexGrid& g, int i) {
                 const HexCoord& c = g.coord_of(i);
                 return py::make_tuple(c.q, c.r, c.s);
             })
        .def("index_of",
             [](const HexGrid& g, py::sequence coord) {
                 return g.index_of(HexCoord{coord[0].cast<int8_t>(), coord[1].cast<int8_t>(),
                                            coord[2].cast<int8_t>()});
             })
        .def("is_edge", &HexGrid::is_edge)
        .def("direction_between", [](const HexGrid& g, int a, int b) -> py::object {
            const int d = g.direction_between(a, b);
            return d < 0 ? py::none() : py::cast(d);
        });

    // --- GameState (named ArrayState for drop-in compatibility) --------------
    py::class_<GameState>(m, "ArrayState")
        .def(py::init([](py::kwargs kw) {
            // tactician_agent._clone_state builds a state directly from arrays.
            // engine_old shares terrain / city_placer / capital_settle_order by
            // reference there and copies the rest; copying everything is
            // equivalent, because nothing in the simulated phase range writes to
            // the shared ones.
            auto s = std::make_unique<GameState>();
            const HexGrid& grid = kw["grid"].cast<const HexGrid&>();
            const int nf = kw["num_factions"].cast<int>();
            new_empty(*s, grid, nf);
            const int n = grid.num_hexes();

            copy_in<int8_t, int8_t>(kw["terrain"], s->terrain, n);
            copy_in<int8_t, int8_t>(kw["city_owner"], s->city_owner, n);
            copy_in<bool, bool>(kw["is_capital"], s->is_capital, n);
            copy_in<int8_t, int8_t>(kw["outpost_upgrade"], s->outpost_upgrade, n);
            copy_in<int8_t, int8_t>(kw["city_placer"], s->city_placer, n);
            copy_in<int8_t, int8_t>(kw["army_faction"], s->army_faction, n);
            copy_in<bool, bool>(kw["frozen"], s->frozen, n);
            copy_in<bool, bool>(kw["locked"], s->locked, n);
            copy_in<int16_t, int16_t>(kw["battle_round"], s->battle_round, n);
            copy_in<int32_t, int32_t>(kw["capital_settle_order"], s->capital_settle_order, nf);
            copy_in<int32_t, int32_t>(kw["gold"], s->gold, nf);
            copy_in<int32_t, int32_t>(kw["kill_xp"], s->kill_xp, nf);
            copy_in<int32_t, int32_t>(kw["victory_points"], s->victory_points, nf);
            copy_in<bool, bool>(kw["alive"], s->alive, nf);

            {
                auto arr = kw["army_units"]
                               .cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
                auto r = arr.unchecked<2>();
                for (int i = 0; i < n; ++i) {
                    for (int t = 0; t < NUM_UNIT_TYPES; ++t) s->army_units[i][t] = r(i, t);
                }
            }
            {
                auto arr = kw["resources"]
                               .cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
                auto r = arr.unchecked<2>();
                for (int i = 0; i < nf; ++i) {
                    for (int t = 0; t < NUM_RESOURCES; ++t) s->resources[i][t] = r(i, t);
                }
            }
            {
                auto bf = kw["battle_faction"]
                              .cast<py::array_t<int8_t, py::array::c_style | py::array::forcecast>>();
                auto bo = kw["battle_origin"]
                              .cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
                auto bm = kw["battle_moved"]
                              .cast<py::array_t<bool, py::array::c_style | py::array::forcecast>>();
                auto bu = kw["battle_units"]
                              .cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
                auto rf = bf.unchecked<2>();
                auto ro = bo.unchecked<2>();
                auto rm = bm.unchecked<2>();
                auto ru = bu.unchecked<3>();
                for (int i = 0; i < n; ++i) {
                    int used = 0;
                    for (int k = 0; k < MAX_BATTLE_CONTRIB; ++k) {
                        s->battle_faction[i][k] = rf(i, k);
                        s->battle_origin[i][k] = ro(i, k);
                        s->battle_moved[i][k] = rm(i, k);
                        for (int t = 0; t < NUM_UNIT_TYPES; ++t) {
                            s->battle_units[i][k][t] = ru(i, k, t);
                        }
                        if (rf(i, k) != NO_FACTION) used = k + 1;
                    }
                    s->battle_nslots[i] = static_cast<uint8_t>(used);
                }
            }
            {
                py::sequence order = kw["battle_order"].cast<py::sequence>();
                s->num_battles = 0;
                for (py::handle h : order) {
                    s->battle_order[s->num_battles++] = h.cast<int16_t>();
                }
            }
            s->turn_number = kw["turn_number"].cast<int32_t>();
            return s;
        }))
        .def_property_readonly("grid",
                               [](const GameState& s) { return const_cast<HexGrid*>(s.grid); },
                               py::return_value_policy::reference)
        .def_property_readonly("num_hexes", [](const GameState& s) { return s.num_hexes; })
        .def_readwrite("num_factions", &GameState::num_factions)
        .def_readwrite("turn_number", &GameState::turn_number)
        // Views are sized to num_hexes / num_factions, NOT the fixed capacity -
        // a view over the padding would report off-board hexes as free ground.
        .def_property_readonly("terrain", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view1(self, s.terrain, s.num_hexes);
        })
        .def_property_readonly("city_owner", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view1(self, s.city_owner, s.num_hexes);
        })
        .def_property_readonly("is_capital", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view1(self, s.is_capital, s.num_hexes);
        })
        .def_property_readonly("outpost_upgrade", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view1(self, s.outpost_upgrade, s.num_hexes);
        })
        .def_property_readonly("city_placer", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view1(self, s.city_placer, s.num_hexes);
        })
        .def_property_readonly("army_faction", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view1(self, s.army_faction, s.num_hexes);
        })
        .def_property_readonly("army_units", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view2<int16_t, NUM_UNIT_TYPES>(self, s.army_units, s.num_hexes);
        })
        .def_property_readonly("frozen", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view1(self, s.frozen, s.num_hexes);
        })
        .def_property_readonly("locked", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view1(self, s.locked, s.num_hexes);
        })
        .def_property_readonly("battle_faction", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view2<int8_t, MAX_BATTLE_CONTRIB>(self, s.battle_faction, s.num_hexes);
        })
        .def_property_readonly("battle_origin", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view2<int32_t, MAX_BATTLE_CONTRIB>(self, s.battle_origin, s.num_hexes);
        })
        .def_property_readonly("battle_units", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view3<int16_t, MAX_BATTLE_CONTRIB, NUM_UNIT_TYPES>(self, s.battle_units,
                                                                      s.num_hexes);
        })
        .def_property_readonly("battle_moved", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view2<bool, MAX_BATTLE_CONTRIB>(self, s.battle_moved, s.num_hexes);
        })
        .def_property_readonly("battle_round", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view1(self, s.battle_round, s.num_hexes);
        })
        // A plain list, matching ArrayState - agents only ever read it.
        .def_property_readonly("battle_order",
                               [](const GameState& s) {
                                   py::list out;
                                   for (int i = 0; i < s.num_battles; ++i) {
                                       out.append(static_cast<int>(s.battle_order[i]));
                                   }
                                   return out;
                               })
        .def_property_readonly("capital_settle_order", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view1(self, s.capital_settle_order, s.num_factions);
        })
        .def_property_readonly("gold", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view1(self, s.gold, s.num_factions);
        })
        .def_property_readonly("resources", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view2<int32_t, NUM_RESOURCES>(self, s.resources, s.num_factions);
        })
        .def_property_readonly("kill_xp", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view1(self, s.kill_xp, s.num_factions);
        })
        .def_property_readonly("victory_points", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view1(self, s.victory_points, s.num_factions);
        })
        .def_property_readonly("alive", [](py::object self) {
            GameState& s = self.cast<GameState&>();
            return view1(self, s.alive, s.num_factions);
        });

    // --- geometry -----------------------------------------------------------
    m.def("hex_distance", [](py::sequence a, py::sequence b) {
        return hex_distance(
            HexCoord{a[0].cast<int8_t>(), a[1].cast<int8_t>(), a[2].cast<int8_t>()},
            HexCoord{b[0].cast<int8_t>(), b[1].cast<int8_t>(), b[2].cast<int8_t>()});
    });

    // --- state helpers ------------------------------------------------------
    m.def("count_units_in_play", &count_units_in_play);
    m.def("count_all_units_in_play", [](const GameState& s, int faction) {
        int32_t counts[NUM_UNIT_TYPES];
        count_all_units_in_play(s, faction, counts);
        py::array_t<int32_t> arr({static_cast<py::ssize_t>(NUM_UNIT_TYPES)});
        auto r = arr.mutable_unchecked<1>();
        for (int t = 0; t < NUM_UNIT_TYPES; ++t) r(t) = counts[t];
        return arr;
    });
    m.def("_outpost_count", &outpost_count);

    // --- terrain / collect --------------------------------------------------
    m.def("apply_terrain_effects", [](GameState& s) {
        apply_terrain_effects(s);
        return &s;
    }, py::return_value_policy::reference);

    m.def("apply_gold_income", [](GameState& s) {
        apply_gold_income(s);
        return &s;
    }, py::return_value_policy::reference);

    m.def("apply_victory_points", [](GameState& s) {
        apply_victory_points(s);
        return &s;
    }, py::return_value_policy::reference);

    m.def("apply_resource_income", [](py::object state_obj, py::dict choose) {
        GameState& s = state_obj.cast<GameState&>();
        PyDecisions d;
        d.state_obj = state_obj;
        d.resource_choice = choose;
        apply_resource_income(s, &py_resource_choice, &d);
        return state_obj;
    });

    m.def("apply_collect_phase", [](py::object state_obj, py::dict choose) {
        GameState& s = state_obj.cast<GameState&>();
        PyDecisions d;
        d.state_obj = state_obj;
        d.resource_choice = choose;
        apply_collect_phase(s, &py_resource_choice, &d);
        return state_obj;
    });

    // --- buy ----------------------------------------------------------------
    m.def("get_legal_buy_actions", [](const GameState& s, int faction) {
        LegalBuyActions legal;
        get_legal_buy_actions(s, faction, legal);
        py::list out;
        for (int i = 0; i < legal.size(); ++i) out.append(buy_action_to_dict(legal[i]));
        return out;
    });

    m.def("apply_buy_phase", [](py::object state_obj, py::dict actions_by_faction) {
        GameState& s = state_obj.cast<GameState&>();
        ChosenBuyActions chosen[MAX_FACTIONS];
        for (int f = 0; f < MAX_FACTIONS; ++f) chosen[f].clear();
        for (auto item : actions_by_faction) {
            const int faction = item.first.cast<int>();
            if (faction < 0 || faction >= MAX_FACTIONS) continue;
            for (py::handle h : item.second.cast<py::sequence>()) {
                BuyAction a{};
                if (buy_action_from_dict(h.cast<py::dict>(), a)) chosen[faction].push_back(a);
            }
        }
        apply_buy_phase(s, chosen);
        return state_obj;
    });

    m.def("eligible_outpost_mask", [](const GameState& s, int faction) {
        bool mask[MAX_HEXES];
        eligible_outpost_mask(s, faction, mask);
        py::array_t<bool> arr({static_cast<py::ssize_t>(s.num_hexes)});
        auto r = arr.mutable_unchecked<1>();
        for (int i = 0; i < s.num_hexes; ++i) r(i) = mask[i];
        return arr;
    });
    m.def("_can_build_outpost", &can_build_outpost);

    // --- movement -----------------------------------------------------------
    m.def("legal_movement_mask", [](const GameState& s, int faction) {
        LegalMask mask;
        legal_movement_mask(s, faction, mask);
        return mask_to_numpy(mask, s.num_hexes);
    });
    m.def("legal_cavalry_mask", [](const GameState& s, int faction) {
        LegalMask mask;
        legal_cavalry_mask(s, faction, mask);
        return mask_to_numpy(mask, s.num_hexes);
    });

    m.def("apply_movement_step",
          [](py::object state_obj, py::dict actions_by_faction, py::object py_rng,
             bool cavalry_only) {
              GameState& s = state_obj.cast<GameState&>();
              MoveActions actions;
              actions.clear();
              for (auto item : actions_by_faction) {
                  const int faction = item.first.cast<int>();
                  if (faction < 0 || faction >= MAX_FACTIONS) continue;
                  if (item.second.is_none()) continue;
                  py::sequence pair = item.second.cast<py::sequence>();
                  actions.set(faction, pair[0].cast<int>(), pair[1].cast<int>());
              }
              BorrowedRng rng(py_rng);
              apply_movement_step(s, actions, rng.rng, cavalry_only);

              py::set out;
              for (int i = 0; i < s.num_hexes; ++i) {
                  if (s.locked[i]) out.add(py::int_(i));
              }
              return out;
          },
          py::arg("state"), py::arg("actions_by_faction"), py::arg("rng"),
          py::arg("cavalry_only") = false);

    // --- battle -------------------------------------------------------------
    m.def("faction_totals", [](const GameState& s, int hex_index) {
        FactionTotals totals;
        faction_totals(s, hex_index, totals);
        py::dict out;
        for (int i = 0; i < totals.count; ++i) {
            py::array_t<int32_t> arr({static_cast<py::ssize_t>(NUM_UNIT_TYPES)});
            auto r = arr.mutable_unchecked<1>();
            for (int t = 0; t < NUM_UNIT_TYPES; ++t) r(t) = totals.units[i][t];
            out[py::int_(totals.faction[i])] = arr;
        }
        return out;
    });

    m.def("get_legal_target_actions", [](const GameState& s, int hex_index, int faction) {
        SmallVec<int8_t, MAX_FACTIONS> legal;
        get_legal_target_actions(s, hex_index, faction, legal);
        py::list out;
        for (int i = 0; i < legal.size(); ++i) out.append(static_cast<int>(legal[i]));
        return out;
    });

    m.def("get_winner", [](const GameState& s, int hex_index) -> py::object {
        const int w = get_winner(s, hex_index);
        return w < 0 ? py::none() : py::cast(w);
    });
    m.def("is_battle_over", &is_battle_over);

    // --- turn ---------------------------------------------------------------
    m.def("run_turn",
          [](py::object state_obj, py::dict buy, py::dict movement, py::dict cavalry,
             py::dict target, py::dict rectification, py::dict resource_choice, py::object py_rng) {
              GameState& s = state_obj.cast<GameState&>();
              PyDecisions d;
              d.state_obj = state_obj;
              d.buy = buy;
              d.movement = movement;
              d.cavalry = cavalry;
              d.target = target;
              d.rectification = rectification;
              d.resource_choice = resource_choice;

              TurnDecisions td;
              td.buy = &py_buy;
              td.movement = &py_movement;
              td.cavalry = &py_cavalry;
              td.target = &py_target;
              td.rectification = &py_rectification;
              td.resource_choice = &py_resource_choice;
              td.ctx = &d;

              BorrowedRng rng(py_rng);
              run_turn(s, td, rng.rng);
              return state_obj;
          },
          py::arg("state"), py::arg("decide_buy"), py::arg("decide_movement"),
          py::arg("decide_cavalry"), py::arg("decide_target"), py::arg("decide_rectification"),
          py::arg("decide_resource_choice"), py::arg("rng") = py::none());

    m.def("_run_battle_phase",
          [](py::object state_obj, py::dict target, py::dict rectification, py::object py_rng) {
              GameState& s = state_obj.cast<GameState&>();
              PyDecisions d;
              d.state_obj = state_obj;
              d.target = target;
              d.rectification = rectification;

              TurnDecisions td;
              td.target = &py_target;
              td.rectification = &py_rectification;
              td.ctx = &d;

              BorrowedRng rng(py_rng);
              run_battle_phase(s, td, rng.rng);
              // The structured per-battle event log lands with the native JSON
              // writer at M6c; nothing in agents/ reads it.
              return py::list();
          });

    m.def("get_game_winner", [](const GameState& s) -> py::object {
        const int w = get_game_winner(s);
        return w < 0 ? py::none() : py::cast(w);
    });
    m.def("check_game_end",
          [](const GameState& s, py::object max_turns) {
              return check_game_end(s, max_turns.is_none() ? -1 : max_turns.cast<int>());
          },
          py::arg("state"), py::arg("max_turns") = py::none());
    m.def("tally_final_score", [](const GameState& s) {
        py::dict out;
        for (int f = 0; f < s.num_factions; ++f) out[py::int_(f)] = s.victory_points[f];
        return out;
    });

    // --- setup / placement --------------------------------------------------
    m.def("create_initial_state",
          [](int radius, int num_factions, int64_t seed, py::object terrain_log) {
              auto s = std::make_unique<GameState>();
              std::vector<TerrainLogEntry> log;
              create_initial_state(*s, radius, num_factions, seed,
                                   terrain_log.is_none() ? nullptr : &log);
              if (!terrain_log.is_none()) {
                  py::list out = terrain_log.cast<py::list>();
                  const char* names[] = {"plains", "mountain", "lake", "desert", "marsh"};
                  for (const TerrainLogEntry& e : log) {
                      py::dict d;
                      d["q"] = e.q;
                      d["r"] = e.r;
                      d["s"] = e.s;
                      d["terrain"] = names[e.terrain];
                      d["round"] = e.round;
                      out.append(d);
                  }
              }
              return s;
          },
          py::arg("radius") = 8, py::arg("num_factions") = 8, py::arg("seed") = 42,
          py::arg("terrain_log") = py::none());

    m.def("legal_placement_mask", [](const GameState& s, int /*num_factions*/) {
        bool mask[MAX_HEXES];
        legal_placement_mask(s, mask);
        py::array_t<bool> arr({static_cast<py::ssize_t>(s.num_hexes)});
        auto r = arr.mutable_unchecked<1>();
        for (int i = 0; i < s.num_hexes; ++i) r(i) = mask[i];
        return arr;
    });

    m.def("run_city_setup",
          [](py::object state_obj, py::dict placement, py::dict draft, py::dict swap,
             py::object py_rng, py::object log_obj) {
              GameState& s = state_obj.cast<GameState&>();
              PySetupDecisions d;
              d.state_obj = state_obj;
              d.placement = placement;
              d.draft = draft;
              d.swap = swap;

              SetupDecisions sd;
              sd.placement = &py_placement;
              sd.draft = &py_draft;
              sd.swap = &py_swap;
              sd.ctx = &d;

              std::vector<PlacementLogEntry> log;
              BorrowedRng rng(py_rng);
              run_city_setup(s, sd, rng.rng, log_obj.is_none() ? nullptr : &log);

              if (!log_obj.is_none()) {
                  const char* kinds[] = {"place", "draft", "draft_auto", "keep", "swap"};
                  py::list out = log_obj.cast<py::list>();
                  for (const PlacementLogEntry& e : log) {
                      py::dict d2;
                      d2["type"] = kinds[e.kind];
                      d2["faction"] = e.faction;
                      d2["q"] = e.q;
                      d2["r"] = e.r;
                      d2["s"] = e.s;
                      if (e.kind == PlacementLogEntry::kSwap) {
                          d2["placer_faction"] = e.placer_faction;
                          d2["placer_q"] = e.placer_q;
                          d2["placer_r"] = e.placer_r;
                          d2["placer_s"] = e.placer_s;
                      }
                      out.append(d2);
                  }
              }
              return state_obj;
          },
          py::arg("state"), py::arg("decide_placement"), py::arg("decide_draft"),
          py::arg("decide_swap"), py::arg("rng"), py::arg("log") = py::none());

    // --- native agents, exposed for side-by-side decision comparison --------
    // Lets one process ask the Python agent and the native agent for a decision
    // about the SAME state and diff them. That is how M6a parity is verified:
    // comparing decisions directly rather than inferring from game outcomes, the
    // same "compare the menu, not the effect" lesson as §3.3c.
    py::class_<AgentSet>(m, "NativeAgentSet")
        .def(py::init([](const std::string& kind, int num_factions, int64_t seed) {
            AgentKind k;
            if (!agent_kind_from_name(kind.c_str(), k)) {
                throw std::runtime_error("unknown native agent: " + kind);
            }
            auto set = std::make_unique<AgentSet>();
            build_agents(*set, k, num_factions, seed);
            return set;
        }))
        .def("decide_buy",
             [](AgentSet& self, const GameState& s, int faction) {
                 LegalBuyActions legal;
                 get_legal_buy_actions(s, faction, legal);
                 ChosenBuyActions chosen;
                 chosen.clear();
                 self.get(faction)->decide_buy(s, faction, legal, chosen);
                 py::list out;
                 for (int i = 0; i < chosen.size(); ++i) out.append(buy_action_to_dict(chosen[i]));
                 return out;
             })
        .def("decide_movement",
             [](AgentSet& self, const GameState& s, int faction, int step) -> py::object {
                 LegalMask legal;
                 legal_movement_mask(s, faction, legal);
                 Move mv{};
                 if (!self.get(faction)->decide_movement(s, faction, step, legal, mv)) {
                     return py::none();
                 }
                 return py::make_tuple(static_cast<int>(mv.hex), static_cast<int>(mv.dir));
             })
        .def("decide_cavalry",
             [](AgentSet& self, const GameState& s, int faction, int step) -> py::object {
                 LegalMask legal;
                 legal_cavalry_mask(s, faction, legal);
                 Move mv{};
                 if (!self.get(faction)->decide_cavalry(s, faction, step, legal, mv)) {
                     return py::none();
                 }
                 return py::make_tuple(static_cast<int>(mv.hex), static_cast<int>(mv.dir));
             })
        .def("decide_target",
             [](AgentSet& self, const GameState& s, int hex_index, int faction) -> py::object {
                 const int t = self.get(faction)->decide_target(s, hex_index, faction);
                 return t < 0 ? py::none() : py::cast(t);
             })
        .def("decide_rectification",
             [](AgentSet& self, const GameState& s, int hex_index, int winner, int cap) {
                 SendBack sb;
                 sb.clear();
                 self.get(winner)->decide_rectification(s, hex_index, winner, cap, sb);
                 py::list out;
                 for (int i = 0; i < sb.size(); ++i) {
                     py::dict d;
                     d["origin_hex"] = sb[i].origin_hex;
                     d["units"] = py::make_tuple(sb[i].units[0], sb[i].units[1], sb[i].units[2]);
                     out.append(d);
                 }
                 return out;
             })
        .def("decide_resource_choice",
             [](AgentSet& self, const GameState& s, int faction, int hex_index) {
                 return self.get(faction)->decide_resource_choice(s, faction, hex_index) == kIron
                            ? "iron"
                            : "fish";
             })
        .def("decide_placement",
             [](AgentSet& self, const GameState& s, int faction) {
                 bool legal[MAX_HEXES];
                 legal_placement_mask(s, legal);
                 return self.get(faction)->decide_placement(s, faction, legal);
             })
        .def("decide_draft",
             [](AgentSet& self, const GameState& s, int faction, py::sequence pool) {
                 SmallVec<int16_t, MAX_HEXES> p;
                 p.clear();
                 for (py::handle h : pool) p.push_back(h.cast<int16_t>());
                 return self.get(faction)->decide_draft(s, faction, p.items, p.size());
             })
        .def("decide_swap",
             [](AgentSet& self, const GameState& s, int faction, int leftover, int placer,
                int placer_hex) {
                 return self.get(faction)->decide_swap(s, faction, leftover, placer, placer_hex);
             });

    // Debug helper for verifying the getstate/setstate RNG bridge (see the file
    // header). Draws n values natively through the borrow path, so a Python
    // caller can check the generator advanced exactly as if it had drawn them.
    m.def("_rng_draw", [](py::object py_rng, int n) {
        BorrowedRng rng(py_rng);
        py::list out;
        for (int i = 0; i < n; ++i) out.append(rng.rng.random());
        return out;
    });

    // --- constants, exported so the shims have a single source of truth -----
    m.attr("MAX_STACK_SIZE") = MAX_STACK_SIZE;
    m.attr("MAX_BATTLE_CONTRIB") = MAX_BATTLE_CONTRIB;
    m.attr("NO_FACTION") = NO_FACTION;
    m.attr("NO_ORIGIN") = NO_ORIGIN;
    m.attr("NO_UPGRADE") = NO_UPGRADE;
    m.attr("INFANTRY_COST") = kInfantryCost;
    m.attr("OUTPOST_COST") = kOutpostCost;
    m.attr("OUTPOST_CAP") = kOutpostCap;
    m.attr("VP_TO_WIN") = kVpToWin;
    m.attr("OUTPOST_DESTROY_VP") = kOutpostDestroyVp;
    m.attr("MOVEMENT_STEPS") = kMovementSteps;
    m.attr("CAVALRY_STEPS") = kCavalrySteps;
    m.attr("STARTING_GOLD") = kStartingGold;
    m.attr("STARTING_KILL_XP") = kStartingKillXp;
    m.attr("CAPITAL_MIN_DIST") = kCapitalMinDist;
}
