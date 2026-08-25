"""
Rollout recording: what agents/nn_agent/agent.py's `recorder` parameter
pushes decisions into during self-play, and what training/gae.py reads
back out to compute advantages.

Decision-kind constants - plain strings, matching exactly what agent.py's
decide_* closures pass as `decision_kind` (agent.py uses string literals
directly rather than importing these, to avoid agents/ gaining a
dependency on training/ - see that module's docstring). Defined here so
this file's own callers (gae.py, ppo.py) don't retype string literals.
"""

MOVEMENT = "movement"
CAVALRY = "cavalry"
BUY = "buy"
TARGET = "target"
RECTIFY = "rectify"
PLACEMENT = "placement"
DRAFT = "draft"
SWAP = "swap"


class TrajectoryStep:
    """One recorded decision. `per_hex`/`global_feats` are exactly the
    arrays that were fed to the network for this decision (freshly
    allocated by encode_observation each call, never aliased to the live
    ArrayState - safe to retain across a whole game). `action_repr` is an
    int for every decision_kind except BUY, where it's an int64[num_hexes]
    choice vector (see actions.py's buy_action_to_choice_vector) - buy's
    "action" is a joint draw over every hex's independent choice, not a
    single index. `reward` starts at 0 and is filled in by
    RolloutBuffer.mark_end_of_turn once the turn that produced this step
    actually resolves."""

    __slots__ = ("faction", "decision_kind", "per_hex", "global_feats", "battle_hex_index",
                 "mask", "action_repr", "old_log_prob", "value", "reward")

    def __init__(self, faction, decision_kind, per_hex, global_feats, battle_hex_index,
                 mask, action_repr, old_log_prob, value):
        self.faction = faction
        self.decision_kind = decision_kind
        self.per_hex = per_hex
        self.global_feats = global_feats
        self.battle_hex_index = battle_hex_index
        self.mask = mask
        self.action_repr = action_repr
        self.old_log_prob = old_log_prob
        self.value = value
        self.reward = 0.0


class RolloutBuffer:
    """Collects TrajectoryStep records across possibly many self-play
    games, keyed by (game_id, faction) so every faction's trajectory
    within every game stays separate. One buffer is built per rollout-
    collection call (training/rollout.py's collect()) and fully discarded
    after the PPO update that consumes it - see that module's memory-
    sizing note; there's no cross-iteration replay."""

    def __init__(self):
        self._trajectories = {}       # (game_id, faction) -> [TrajectoryStep, ...]
        self._pending_reward = {}     # (game_id, faction) -> float, only used if record_step
                                       # hasn't been called yet this turn when a reward arrives
        self._bootstrap = {}          # (game_id, faction) -> float or None (None = true terminal)
        self._next_game_id = 0
        self._current_game_id = None

    def begin_game(self):
        """Starts a new game's bookkeeping; returns its id (not usually
        needed by the caller - trajectories() reconstructs everything by
        (game_id, faction) internally)."""
        self._current_game_id = self._next_game_id
        self._next_game_id += 1
        return self._current_game_id

    def record_step(self, faction, decision_kind, per_hex, global_feats, battle_hex_index,
                     mask, action_repr, log_prob, value):
        key = (self._current_game_id, faction)
        pending = self._pending_reward.pop(key, 0.0)
        step = TrajectoryStep(faction, decision_kind, per_hex, global_feats, battle_hex_index,
                               mask, action_repr, log_prob, value)
        step.reward = pending
        self._trajectories.setdefault(key, []).append(step)

    def mark_end_of_turn(self, faction, reward):
        """Attributes `reward` (this turn's Δown-VP - Δleading-rival-VP)
        to the most recent decision this faction made this game - GAE
        propagates credit backward from there through the turn's earlier
        decisions (buy, movement steps, any battle target/rectification
        calls) via the value function, rather than this needing to
        hand-split reward by decision type. If nothing's been recorded
        for this (game, faction) yet (shouldn't normally happen - buy's
        no-op column is always legal, so every faction records at least
        one step essentially every turn - but defensively handled rather
        than silently dropping reward), the reward carries forward and
        lands on the next step actually recorded."""
        key = (self._current_game_id, faction)
        steps = self._trajectories.get(key)
        if steps:
            steps[-1].reward += reward
        else:
            self._pending_reward[key] = self._pending_reward.get(key, 0.0) + reward

    def end_game(self, bootstrap_values):
        """bootstrap_values: {faction: float_or_None} for every nn-
        controlled faction in the game that just ended. None means a true
        terminal (someone hit the VP win condition - no continuation
        value); a float means the game was cut off by the turn cap and is
        the value head's estimate of the final state, so GAE bootstraps
        from it instead of treating the cutoff as if the game truly
        ended there (see gae.py)."""
        for faction, v in bootstrap_values.items():
            self._bootstrap[(self._current_game_id, faction)] = v

    def trajectories(self):
        """Yields (steps, bootstrap_value_or_None) for every (game,
        faction) that has at least one recorded step. Factions never
        controlled by the recorded nn agent (e.g. a 'greedy'/'random'
        seat mixed into a rollout) never call record_step, so they never
        appear here."""
        for key, steps in self._trajectories.items():
            yield steps, self._bootstrap.get(key, 0.0)

    def steps_in_game(self, game_id):
        """Total recorded steps across every faction in one game - purely
        informational (e.g. rollout.py's progress printouts), not used by
        gae.py."""
        return sum(len(steps) for (gid, _faction), steps in self._trajectories.items() if gid == game_id)
