// Full-turn parity: replays Python decision traces through the C++ engine.
//
// Each case is a before-state, the exact sequence of decisions engine_old's agents
// made, and the after-state. The replaying decision provider asserts that the C++
// engine asks for decisions in the same ORDER and with the same ARGUMENTS - so a
// battle that runs an extra round, or a movement step that queries the wrong
// faction, is reported at the point it happens rather than as a mystery state diff
// several phases later.
//
// Usage: test_turn <path-to-turn_traces.txt>

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

int main(int argc, char** argv) {
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
        decisions.reserve(static_cast<size_t>(n_decisions));
        for (int d = 0; d < n_decisions; ++d) {
            std::string line;
            std::getline(in, line);
            Decision dec;
            if (!parse_decision(line, dec)) {
                std::cerr << "case " << case_name << ": bad decision line: " << line << "\n";
                return 2;
            }
            decisions.push_back(std::move(dec));
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

        oo::Rng rng(seed);
        oo::run_turn(*before, td, rng);

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

    std::printf("test_turn: %d/%d turns passed (%lld decisions replayed), %d failures\n", passed,
                total, total_decisions, failures);
    return failures == 0 ? 0 : 1;
}
