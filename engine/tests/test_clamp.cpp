// The movement stack clamp (PLAN.md §11.1).
//
// Needs its own test because the golden corpus cannot provide one. Those files
// were reblessed from this engine, so they now agree with whatever it does by
// construction - they catch regressions but they cannot say the rule is RIGHT.
// This asserts the rule directly, from states built by hand.
//
// It also pins the deletion of `revert_departure`: the clamp is only safe if a
// peaceful merge can never overflow, so the last two cases check that the
// situations which used to trigger a revert now cannot arise.

#include "oo/movement.hpp"
#include "oo/state_io.hpp"

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++g_failures;
    }
}

// A board with no cities and no battles, so only movement is in play.
std::unique_ptr<oo::GameState> board(int radius, int num_factions) {
    auto s = std::make_unique<oo::GameState>();
    oo::new_empty(*s, oo::HexGrid::shared(radius), num_factions);
    for (int h = 0; h < s->num_hexes; ++h) s->terrain[h] = oo::kPlains;
    return s;
}

void put(oo::GameState& s, int hex, int faction, int inf, int cav, int arc) {
    s.army_faction[hex] = static_cast<int8_t>(faction);
    s.army_units[hex][oo::kInfantry] = static_cast<int16_t>(inf);
    s.army_units[hex][oo::kCavalry] = static_cast<int16_t>(cav);
    s.army_units[hex][oo::kArchers] = static_cast<int16_t>(arc);
}

// Two adjacent hexes on the board, with the direction from the first to the
// second. Hex 0 always has neighbours at radius >= 1.
struct Pair {
    int from, to, dir;
};

Pair adjacent(const oo::GameState& s) {
    for (int d = 0; d < oo::NUM_DIRECTIONS; ++d) {
        const int j = s.grid->neighbour(0, d);
        if (j >= 0) return Pair{0, j, d};
    }
    std::abort();
}

bool mask_allows(const oo::GameState& s, int faction, const Pair& p, bool cavalry) {
    oo::LegalMask m;
    if (cavalry) {
        oo::legal_cavalry_mask(s, faction, m);
    } else {
        oo::legal_movement_mask(s, faction, m);
    }
    return m.cell[p.from][p.dir];
}

void step(oo::GameState& s, int faction, const Pair& p, bool cavalry = false) {
    oo::MoveActions a;
    a.clear();
    a.set(faction, p.from, p.dir);
    oo::Rng rng(1234);
    oo::apply_movement_step(s, a, rng, cavalry);
}

}  // namespace

int main() {
    // --- 1. a merge that would overflow carries only what fits ---------------
    {
        auto s = board(4, 2);
        const Pair p = adjacent(*s);
        put(*s, p.from, 0, 3, 1, 1);  // 5 units moving
        put(*s, p.to, 0, 2, 0, 0);    // 2 already there -> capacity 4
        step(*s, 0, p);

        check(s->units_at(p.to) == oo::MAX_STACK_SIZE,
              "destination fills exactly to the cap, not past it");
        check(s->units_at(p.from) == 1, "the one unit that did not fit stays behind");
        // Infantry are shed first (rectify_overflow's order), so the unit left
        // behind is infantry and the cavalry/archers went forward.
        check(s->army_units[p.from][oo::kInfantry] == 1, "infantry is what stays behind");
        check(s->army_units[p.to][oo::kCavalry] == 1 && s->army_units[p.to][oo::kArchers] == 1,
              "cavalry and archers are carried in preference to infantry");
        check(s->army_faction[p.from] == 0, "the origin is still owned, not emptied");
    }

    // --- 2. a full destination makes the move ILLEGAL, not a no-op -----------
    // This is the half of §11.1 that is about not wasting the faction's single
    // move for the step, so it has to be in the mask, not just in the resolver.
    {
        auto s = board(4, 2);
        const Pair p = adjacent(*s);
        put(*s, p.from, 0, 3, 0, 0);
        put(*s, p.to, 0, 6, 0, 0);  // already at the cap
        check(!mask_allows(*s, 0, p, false), "moving into our own full stack is masked out");

        // And if submitted anyway it must change nothing at all.
        const auto before = *s;
        step(*s, 0, p);
        std::string diff;
        auto copy = std::make_unique<oo::GameState>(before);
        check(oo::compare_states(*s, *copy, diff), "a submitted illegal merge is a no-op: " + diff);
    }

    // --- 3. a merge that fits is not clamped and stays legal -----------------
    {
        auto s = board(4, 2);
        const Pair p = adjacent(*s);
        put(*s, p.from, 0, 2, 0, 0);
        put(*s, p.to, 0, 3, 0, 0);  // 2 + 3 = 5, fits
        check(mask_allows(*s, 0, p, false), "a merge that fits stays legal");
        step(*s, 0, p);
        check(s->units_at(p.to) == 5, "a merge that fits moves everything");
        check(s->army_faction[p.from] == oo::NO_FACTION, "the origin empties");
    }

    // --- 4. an ENEMY destination is never clamped ---------------------------
    // Battles are not stack-capped; clamping here would silently weaken attacks.
    {
        auto s = board(4, 2);
        const Pair p = adjacent(*s);
        put(*s, p.from, 0, 6, 0, 0);
        put(*s, p.to, 1, 6, 0, 0);  // enemy, 12 units total once joined
        check(mask_allows(*s, 0, p, false), "attacking a full enemy stack stays legal");
        step(*s, 0, p);
        const oo::Battle* b = s->battle_at(p.to);
        check(b != nullptr, "attacking starts a battle");
        if (b) {
            int total = 0;
            for (int i = 0; i < b->nslots; ++i) {
                for (int t = 0; t < oo::NUM_UNIT_TYPES; ++t) total += b->slots[i].units[t];
            }
            check(total == 12, "all 12 units reach the battle - no clamp on a fight");
        }
    }

    // --- 5. the cavalry sub-phase clamps the same way -----------------------
    {
        auto s = board(4, 2);
        const Pair p = adjacent(*s);
        put(*s, p.from, 0, 0, 4, 0);  // 4 cavalry want to move
        put(*s, p.to, 0, 4, 0, 0);    // capacity 2
        step(*s, 0, p, /*cavalry=*/true);
        check(s->units_at(p.to) == oo::MAX_STACK_SIZE, "cavalry merge fills to the cap");
        check(s->army_units[p.to][oo::kCavalry] == 2, "only the cavalry that fit moved");
        check(s->army_units[p.from][oo::kCavalry] == 2, "the rest stayed");
    }

    // --- 6. no peaceful stack anywhere ever exceeds the cap -----------------
    // The invariant validate_state used to have to relax for _revert_departure's
    // second quirk. With the clamp it holds again, so assert it directly.
    {
        auto s = board(4, 3);
        const Pair p = adjacent(*s);
        put(*s, p.from, 0, 6, 0, 0);
        put(*s, p.to, 0, 5, 0, 0);
        step(*s, 0, p);
        for (int h = 0; h < s->num_hexes; ++h) {
            if (s->army_faction[h] == oo::NO_FACTION || s->locked(h)) continue;
            check(s->units_at(h) <= oo::MAX_STACK_SIZE, "no peaceful stack over the cap");
        }
        std::string problem;
        check(oo::validate_state(*s, problem), "state stays valid: " + problem);
    }

    std::printf("test_clamp: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
