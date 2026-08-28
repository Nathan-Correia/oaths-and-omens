"""
Generalized Advantage Estimation: turns a RolloutBuffer's per-(game,
faction) trajectories into a flat list of TrainingSample - one per
recorded decision, each carrying everything training/ppo.py needs to
recompute the decision's log-prob/entropy under the current weights and
compute the clipped-surrogate loss against it.

Standard backward-recursion GAE over each trajectory independently (no
mixing across games or factions - a trajectory is exactly one faction's
full sequence of decisions in one game). The one thing worth being
deliberate about: a trajectory's `bootstrap_value` (from
RolloutBuffer.end_game) is None for a true terminal (someone hit the VP
win condition - no value exists beyond "the game is over") vs. a float
for a max_turns truncation (the value head's own estimate of the state
at cutoff, since the game didn't actually end there - see
engine/turn.py's check_game_end docstring, which calls the turn cap
"purely an infra safety net... NOT part of the rules"). Collapsing that
distinction would teach the value function that "turn 100" is
inherently worth 0, which it isn't.
"""


class TrainingSample:
    __slots__ = ("per_hex", "global_feats", "battle_hex_index", "decision_kind",
                 "mask", "action_repr", "old_log_prob", "advantage", "return_")

    def __init__(self, per_hex, global_feats, battle_hex_index, decision_kind,
                 mask, action_repr, old_log_prob, advantage, return_):
        self.per_hex = per_hex
        self.global_feats = global_feats
        self.battle_hex_index = battle_hex_index
        self.decision_kind = decision_kind
        self.mask = mask
        self.action_repr = action_repr
        self.old_log_prob = old_log_prob
        self.advantage = advantage
        self.return_ = return_


def compute_gae(trajectories, gamma, lam):
    """Returns a flat list[TrainingSample] built from every trajectory in
    `trajectories` - an iterable of (steps, bootstrap_value) pairs, e.g. a
    single RolloutBuffer.trajectories() or, when rollout collection was
    split across multiple worker processes (see rollout.collect_parallel),
    itertools.chain.from_iterable(b.trajectories() for b in buffers).
    Buffers themselves are never merged - each worker's internal game-id
    numbering starts at 0 independently, so trajectories() is the correct
    point to combine them, not the buffers' internal dicts."""
    samples = []
    for steps, bootstrap_value in trajectories:
        if not steps:
            continue

        next_value = bootstrap_value if bootstrap_value is not None else 0.0
        advantage = 0.0
        advantages = [0.0] * len(steps)
        for t in reversed(range(len(steps))):
            delta = steps[t].reward + gamma * next_value - steps[t].value
            advantage = delta + gamma * lam * advantage
            advantages[t] = advantage
            next_value = steps[t].value

        for step, adv in zip(steps, advantages):
            samples.append(TrainingSample(
                per_hex=step.per_hex, global_feats=step.global_feats,
                battle_hex_index=step.battle_hex_index, decision_kind=step.decision_kind,
                mask=step.mask, action_repr=step.action_repr, old_log_prob=step.old_log_prob,
                advantage=adv, return_=adv + step.value,
            ))

    return samples
