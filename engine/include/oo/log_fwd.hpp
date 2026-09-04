// Forward declarations, so battle.hpp can take an optional log pointer without
// pulling in log.hpp (which depends on agent.hpp, which depends on battle.hpp).
#pragma once

namespace oo {
struct BattleLog;
}  // namespace oo
