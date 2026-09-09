// Full-turn parity: replays Python decision traces through the C++ engine.
//
// Each case is a before-state, the exact sequence of decisions engine_old's agents
// made, and the after-state. The replaying decision provider asserts that the C++
// engine asks for decisions in the same ORDER and with the same ARGUMENTS - so a
// battle that runs an extra round, or a movement step that queries the wrong
// faction, is reported at the point it happens rather than as a mystery state diff
// several phases later.
//
// Usage: test_turn <path-to-turn_traces.txt> [--rewrite <out>]

#include "rewrite.hpp"

#include "oo/agent.hpp"
#include "oo/game.hpp"
#include "oo/setup.hpp"
#include "oo/placement.hpp"
#include "oo/state_io.hpp"
#include "oo/turn.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Decision {
    char kind = '?';  // B buy, M move, C cavalry, T target, R rectify, P resource
    int a = 0, b = 0, c = 0, d = 0;
    std::vector<int> extra;
};

// Serves recorded decisions in order, refusing any request that does not match
// what Python was asked at the same point.
struct Replay {
    const std::vector<Decision>* decisions = nullptr;
    size_t cursor = 0;
    std::string error;

    const Decision* next(char kind, const char* what) {
        if (!error.empty()) return nullptr;
        if (cursor >= decisions->size()) {
            std::ostringstream os;
            os << "ran out of decisions: C++ asked for " << what << " (" << kind
               << ") after Python made only " << decisions->size();
            error = os.str();
            return nullptr;
        }
        const Decision& dec = (*decisions)[cursor];
        if (dec.kind != kind) {
            std::ostringstream os;
            os << "decision " << cursor << ": C++ asked for " << what << " (" << kind
               << "), Python recorded '" << dec.kind << "'";
            error = os.str();
            return nullptr;
        }
        ++cursor;
        return &dec;
    }

    void mismatch(size_t index, const std::string& detail) {
        if (error.empty()) {
            std::ostringstream os;
            os << "decision " << index << ": " << detail;
            error = os.str();
        }
    }
};

void replay_buy(const oo::GameState&, int faction, const oo::LegalBuyActions&,
                oo::ChosenBuyActions& out, void* ctx) {
    Replay& r = *static_cast<Replay*>(ctx);
    const size_t index = r.cursor;
    const Decision* dec = r.next('B', "a buy decision");
    if (!dec) return;
    if (dec->a != faction) {
        r.mismatch(index, "buy asked for faction " + std::to_string(faction) + ", Python recorded " +
                              std::to_string(dec->a));
        return;
    }
    out.clear();
    for (int i = 0; i < dec->b; ++i) {
        oo::BuyAction action{};
        action.type = static_cast<oo::BuyType>(dec->extra[static_cast<size_t>(i) * 4 + 0]);
        action.hex = static_cast<int16_t>(dec->extra[static_cast<size_t>(i) * 4 + 1]);
        action.unit_type = static_cast<int8_t>(dec->extra[static_cast<size_t>(i) * 4 + 2]);
        action.upgrade = static_cast<int8_t>(dec->extra[static_cast<size_t>(i) * 4 + 3]);
        out.push_back(action);
    }
}

bool replay_move_impl(char kind, int faction, int step, oo::Move& out, void* ctx) {
    Replay& r = *static_cast<Replay*>(ctx);
    const size_t index = r.cursor;
    const Decision* dec = r.next(kind, kind == 'M' ? "a movement decision" : "a cavalry decision");
    if (!dec) return false;
    if (dec->a != faction || dec->b != step) {
        std::ostringstream os;
        os << "movement asked for faction " << faction << " step " << step
           << ", Python recorded faction " << dec->a << " step " << dec->b;
        r.mismatch(index, os.str());
        return false;
    }
    if (dec->c < 0) return false;  // Python returned None
    out.hex = static_cast<int16_t>(dec->c);
    out.dir = static_cast<int8_t>(dec->d);
    return true;
}

bool replay_movement(const oo::GameState&, int faction, int step, const oo::LegalMask&,
                     oo::Move& out, void* ctx) {
    return replay_move_impl('M', faction, step, out, ctx);
}

bool replay_cavalry(const oo::GameState&, int faction, int step, const oo::LegalMask&,
                    oo::Move& out, void* ctx) {
    return replay_move_impl('C', faction, step, out, ctx);
}

int replay_target(const oo::GameState&, int hex_index, int faction, void* ctx) {
    Replay& r = *static_cast<Replay*>(ctx);
    const size_t index = r.cursor;
    const Decision* dec = r.next('T', "a battle target");
    if (!dec) return -1;
    if (dec->a != hex_index || dec->b != faction) {
        std::ostringstream os;
        os << "target asked at hex " << hex_index << " for faction " << faction
           << ", Python recorded hex " << dec->a << " faction " << dec->b;
        r.mismatch(index, os.str());
        return -1;
    }
    return dec->c;
}

void replay_rectification(const oo::GameState&, int hex_index, int winner, int cap,
                          oo::SendBack& out, void* ctx) {
    Replay& r = *static_cast<Replay*>(ctx);
    const size_t index = r.cursor;
    const Decision* dec = r.next('R', "a rectification");
    if (!dec) return;
    if (dec->a != hex_index || dec->b != winner || dec->c != cap) {
        std::ostringstream os;
        os << "rectification asked at hex " << hex_index << " winner " << winner << " cap " << cap
           << ", Python recorded hex " << dec->a << " winner " << dec->b << " cap " << dec->c;
        r.mismatch(index, os.str());
        return;
    }
    out.clear();
    for (int i = 0; i < dec->d; ++i) {
        oo::SendBackEntry entry{};
        entry.origin_hex = dec->extra[static_cast<size_t>(i) * 4 + 0];
        for (int t = 0; t < oo::NUM_UNIT_TYPES; ++t) {
            entry.units[t] =
                static_cast<int16_t>(dec->extra[static_cast<size_t>(i) * 4 + 1 + static_cast<size_t>(t)]);
        }
        out.push_back(entry);
    }
}

oo::Resource replay_resource(const oo::GameState&, int faction, int hex_index, void* ctx) {
    Replay& r = *static_cast<Replay*>(ctx);
    const size_t index = r.cursor;
    const Decision* dec = r.next('P', "a resource choice");
    if (!dec) return oo::kFish;
    if (dec->a != faction || dec->b != hex_index) {
        std::ostringstream os;
        os << "resource choice asked for faction " << faction << " hex " << hex_index
           << ", Python recorded faction " << dec->a << " hex " << dec->b;
        r.mismatch(index, os.str());
        return oo::kFish;
    }
    return dec->c == 1 ? oo::kIron : oo::kFish;
}

bool parse_decision(const std::string& line, Decision& out) {
    std::istringstream ls(line);
    std::string kind;
    if (!(ls >> kind) || kind.size() != 1) return false;
    out.kind = kind[0];
    out.extra.clear();
    switch (out.kind) {
        case 'B': {  // B <faction> <count> [<type> <hex> <unit> <upgrade>]*
            ls >> out.a >> out.b;
            int v;
            while (ls >> v) out.extra.push_back(v);
            return static_cast<int>(out.extra.size()) == out.b * 4;
        }
        case 'M':
        case 'C':  // <faction> <step> <hex> <dir>
            ls >> out.a >> out.b >> out.c >> out.d;
            return true;
        case 'T':  // <hex> <faction> <target>
            ls >> out.a >> out.b >> out.c;
            return true;
        case 'R': {  // R <hex> <winner> <cap> <count> [<origin> <i> <c> <a>]*
            ls >> out.a >> out.b >> out.c >> out.d;
            int v;
            while (ls >> v) out.extra.push_back(v);
            return static_cast<int>(out.extra.size()) == out.d * 4;
        }
        case 'P':  // <faction> <hex> <iron?>
            ls >> out.a >> out.b >> out.c;
            return true;
        default:
            return false;
    }
}

}  // namespace

// --- corpus recording (--record) --------------------------------------------
//
// --rewrite keeps the recorded decision traces and recomputes outcomes, which
// works only while the engine still ASKS for the same decisions. A change to the
// dice or the movement rules alters battle round counts, so traces get dropped
// and the corpus erodes - the RNG swap alone would have taken it from 176 cases
// to 80. Recording regenerates the whole corpus from seeds instead, so the
// deepest test in the suite survives deliberate change intact.
//
// The game keys below are the ones the Python dumper used, so coverage stays
// comparable: a spread of agents, radii 4-8 and 4-10 factions.
namespace {

struct GameKey {
    const char* agent;
    int radius;
    int factions;
    int64_t seed;
};

constexpr GameKey kGameKeys[] = {
    {"random", 4, 4, 101},   {"random", 5, 6, 102},     {"random", 7, 8, 103},
    {"random", 8, 8, 104},   {"random", 5, 10, 105},    {"greedy", 7, 8, 201},
    {"greedy", 5, 6, 202},   {"greedy", 4, 4, 203},     {"greedy", 8, 8, 204},
    {"heuristic", 7, 8, 301},{"vanguard", 7, 8, 302},   {"marshal", 7, 8, 401},
    {"marshal", 5, 6, 402},  {"tactician", 7, 8, 501},
};

constexpr int kRecordMaxTurns = 15;

// Wraps a real AgentSet, writing down every decision in the exact line format
// parse_decision reads back.
struct Recorder {
    const oo::AgentSet* agents = nullptr;
    std::vector<std::string> lines;
};

void rec_buy(const oo::GameState& s, int faction, const oo::LegalBuyActions& legal,
             oo::ChosenBuyActions& out, void* ctx) {
    Recorder& r = *static_cast<Recorder*>(ctx);
    r.agents->get(faction)->decide_buy(s, faction, legal, out);
    std::ostringstream os;
    os << "B " << faction << ' ' << out.size();
    for (int i = 0; i < out.size(); ++i) {
        os << ' ' << int(out[i].type) << ' ' << out[i].hex << ' ' << int(out[i].unit_type) << ' '
           << int(out[i].upgrade);
    }
    r.lines.push_back(os.str());
}

bool rec_move_impl(char kind, const oo::GameState& s, int faction, int step,
                   const oo::LegalMask& legal, oo::Move& out, void* ctx) {
    Recorder& r = *static_cast<Recorder*>(ctx);
    const bool moved = kind == 'M'
                           ? r.agents->get(faction)->decide_movement(s, faction, step, legal, out)
                           : r.agents->get(faction)->decide_cavalry(s, faction, step, legal, out);
    std::ostringstream os;
    os << kind << ' ' << faction << ' ' << step << ' ' << (moved ? out.hex : -1) << ' '
       << (moved ? int(out.dir) : -1);
    r.lines.push_back(os.str());
    return moved;
}

bool rec_movement(const oo::GameState& s, int faction, int step, const oo::LegalMask& legal,
                  oo::Move& out, void* ctx) {
    return rec_move_impl('M', s, faction, step, legal, out, ctx);
}
bool rec_cavalry(const oo::GameState& s, int faction, int step, const oo::LegalMask& legal,
                 oo::Move& out, void* ctx) {
    return rec_move_impl('C', s, faction, step, legal, out, ctx);
}

int rec_target(const oo::GameState& s, int hex_index, int faction, void* ctx) {
    Recorder& r = *static_cast<Recorder*>(ctx);
    const int t = r.agents->get(faction)->decide_target(s, hex_index, faction);
    std::ostringstream os;
    os << "T " << hex_index << ' ' << faction << ' ' << t;
    r.lines.push_back(os.str());
    return t;
}

void rec_rectification(const oo::GameState& s, int hex_index, int winner, int cap,
                       oo::SendBack& out, void* ctx) {
    Recorder& r = *static_cast<Recorder*>(ctx);
    r.agents->get(winner)->decide_rectification(s, hex_index, winner, cap, out);
    std::ostringstream os;
    os << "R " << hex_index << ' ' << winner << ' ' << cap << ' ' << out.size();
    for (int i = 0; i < out.size(); ++i) {
        os << ' ' << out[i].origin_hex << ' ' << out[i].units[0] << ' ' << out[i].units[1] << ' '
           << out[i].units[2];
    }
    r.lines.push_back(os.str());
}

oo::Resource rec_resource(const oo::GameState& s, int faction, int hex_index, void* ctx) {
    Recorder& r = *static_cast<Recorder*>(ctx);
    const oo::Resource res = r.agents->get(faction)->decide_resource_choice(s, faction, hex_index);
    std::ostringstream os;
    os << "P " << faction << ' ' << hex_index << ' ' << (res == oo::kIron ? 1 : 0);
    r.lines.push_back(os.str());
    return res;
}

int record_corpus(const std::string& path) {
    std::ostringstream body;
    int cases = 0;

    for (const GameKey& key : kGameKeys) {
        oo::AgentKind kind;
        if (!oo::agent_kind_from_name(key.agent, kind)) {
            std::cerr << "unknown agent " << key.agent << "\n";
            return -1;
        }
        auto state = std::make_unique<oo::GameState>();
        oo::create_initial_state(*state, key.radius, key.factions, key.seed);

        oo::AgentSet agents;
        oo::build_agents(agents, kind, key.factions, key.seed);
        oo::Rng master(key.seed);
        oo::SetupDecisions sd = oo::make_setup_decisions(agents);
        oo::run_city_setup(*state, sd, master);

        Recorder rec;
        rec.agents = &agents;
        oo::TurnDecisions td;
        td.buy = &rec_buy;
        td.movement = &rec_movement;
        td.cavalry = &rec_cavalry;
        td.target = &rec_target;
        td.rectification = &rec_rectification;
        td.resource_choice = &rec_resource;
        td.ctx = &rec;

        auto before = std::make_unique<oo::GameState>();
        for (int t = 0; !oo::check_game_end(*state, kRecordMaxTurns); ++t) {
            // Each turn gets its own generator, seeded from the master, so a case
            // is replayable standalone from the SEED recorded with it.
            const int64_t turn_seed = master.randrange(static_cast<int64_t>(1) << 31);
            *before = *state;
            rec.lines.clear();
            oo::Rng turn_rng(turn_seed);
            oo::run_turn(*state, td, turn_rng);

            body << "TURN_CASE " << key.agent << "-r" << key.radius << "f" << key.factions << "s"
                 << key.seed << "t" << t << "\n"
                 << "SEED " << turn_seed << "\n";
            oo::write_state(body, *before);
            body << "DECISIONS " << rec.lines.size() << "\n";
            for (const std::string& l : rec.lines) body << l << "\n";
            oo::write_state(body, *state);
            ++cases;
        }
    }

    oo_test::Rewriter rw;
    if (!rw.open(path)) return -1;
    rw.out << "TURN_CASES " << cases << "\n" << body.str();
    return cases;
}

}  // namespace

int main(int argc, char** argv) {
    std::string record_path;
    if (oo_test::take_flag(argc, argv, "--record", record_path)) {
        const int n = record_corpus(record_path);
        if (n < 0) return 2;
        std::printf("test_turn: recorded %d cases to %s\n", n, record_path.c_str());
        return 0;
    }

    std::string rewrite_path;
    const bool rewriting = oo_test::take_rewrite_flag(argc, argv, rewrite_path);

    if (argc < 2) {
        std::cerr << "usage: test_turn <turn_traces.txt>\n";
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 2;
    }

    std::string tag;
    int total = 0;
    in >> tag >> total;
    if (tag != "TURN_CASES") {
        std::cerr << "malformed trace file: expected TURN_CASES header\n";
        return 2;
    }

    auto before = std::make_unique<oo::GameState>();
    auto expected = std::make_unique<oo::GameState>();
    // run_turn mutates `before`, so a rewrite needs the original to re-emit.
    auto original = std::make_unique<oo::GameState>();

    // Buffered, because a case whose decision sequence no longer replays cannot
    // be rewritten at all and has to be DROPPED - which changes the case count in
    // the header. Buffering lets the real count be written once at the end.
    std::ostringstream rw_body;
    int rw_cases = 0, rw_dropped = 0;

    int passed = 0, failures = 0;
    long long total_decisions = 0;
    for (int i = 0; i < total; ++i) {
        std::string case_tag, case_name, seed_tag;
        long long seed = 0;
        if (!(in >> case_tag >> case_name >> seed_tag >> seed) || case_tag != "TURN_CASE") {
            std::cerr << "malformed trace file at case " << i << "\n";
            return 2;
        }
        std::string error;
        if (!oo::read_state(in, *before, error)) {
            std::cerr << "case " << case_name << " (before): " << error << "\n";
            return 2;
        }

        std::string dec_tag;
        int n_decisions = 0;
        in >> dec_tag >> n_decisions;
        if (dec_tag != "DECISIONS") {
            std::cerr << "case " << case_name << ": expected DECISIONS\n";
            return 2;
        }
        std::getline(in, tag);  // consume rest of line
        std::vector<Decision> decisions;
        // Kept verbatim rather than re-serialized from the parsed struct, so a
        // rewrite reproduces the recorded input exactly instead of round-tripping
        // it through a formatter that might normalise something.
        std::vector<std::string> raw_decisions;
        decisions.reserve(static_cast<size_t>(n_decisions));
        for (int d = 0; d < n_decisions; ++d) {
            std::string line;
            std::getline(in, line);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            Decision dec;
            if (!parse_decision(line, dec)) {
                std::cerr << "case " << case_name << ": bad decision line: " << line << "\n";
                return 2;
            }
            decisions.push_back(std::move(dec));
            raw_decisions.push_back(std::move(line));
        }
        total_decisions += n_decisions;

        if (!oo::read_state(in, *expected, error)) {
            std::cerr << "case " << case_name << " (after): " << error << "\n";
            return 2;
        }

        std::string problem;
        if (!oo::validate_state(*before, problem)) {
            std::cerr << "FAIL " << case_name << ": input state invalid: " << problem << "\n";
            ++failures;
            continue;
        }

        Replay replay;
        replay.decisions = &decisions;
        oo::TurnDecisions td;
        td.buy = &replay_buy;
        td.movement = &replay_movement;
        td.cavalry = &replay_cavalry;
        td.target = &replay_target;
        td.rectification = &replay_rectification;
        td.resource_choice = &replay_resource;
        td.ctx = &replay;

        *original = *before;
        oo::Rng rng(seed);
        oo::run_turn(*before, td, rng);

        if (rewriting) {
            // A case whose recorded decision sequence no longer matches what the
            // engine asks for cannot be reblessed - the trace is the input, and
            // it is now wrong. Drop it loudly rather than writing a broken case.
            if (!replay.error.empty() || replay.cursor != decisions.size()) {
                ++rw_dropped;
                std::cerr << "DROPPED " << case_name << ": "
                          << (replay.error.empty() ? "decision count changed" : replay.error)
                          << "\n";
                continue;
            }
            rw_body << "TURN_CASE " << case_name << "\n" << "SEED " << seed << "\n";
            oo::write_state(rw_body, *original);
            rw_body << "DECISIONS " << n_decisions << "\n";
            for (const std::string& line : raw_decisions) rw_body << line << "\n";
            oo::write_state(rw_body, *before);
            ++rw_cases;
            ++passed;
            continue;
        }

        if (!replay.error.empty()) {
            if (++failures <= 15) std::cerr << "FAIL " << case_name << ": " << replay.error << "\n";
            continue;
        }
        if (replay.cursor != decisions.size()) {
            if (++failures <= 15) {
                std::cerr << "FAIL " << case_name << ": C++ consumed " << replay.cursor
                          << " decisions, Python made " << decisions.size() << "\n";
            }
            continue;
        }

        std::string diff;
        if (!oo::compare_states(*before, *expected, diff)) {
            if (++failures <= 15) std::cerr << "FAIL " << case_name << ": " << diff << "\n";
            continue;
        }
        if (!oo::validate_state(*before, problem)) {
            std::cerr << "FAIL " << case_name << ": output state invalid: " << problem << "\n";
            ++failures;
            continue;
        }
        ++passed;
    }

    if (rewriting) {
        oo_test::Rewriter rw;
        if (!rw.open(rewrite_path)) return 2;
        rw.out << "TURN_CASES " << rw_cases << "\n" << rw_body.str();
        std::printf("test_turn: rewrote %d cases to %s (%d dropped)\n", rw_cases,
                    rewrite_path.c_str(), rw_dropped);
        return 0;
    }
    std::printf("test_turn: %d/%d turns passed (%lld decisions replayed), %d failures\n", passed,
                total, total_decisions, failures);
    return failures == 0 ? 0 : 1;
}
