"""
Agent implementations for engine, kept separate from the engine package
itself - engine/turn.py's run_turn takes plain {faction: callable}
dicts rather than a formal Agent class hierarchy specifically so the
engine never has to know or care what's producing decisions (see that
module's docstring). random_agent.py/greedy_agent.py are scripted
policies; nn_agent/ is a learned one - both are just different callable
sources from the engine's point of view.

compose_agents lets a driver script (run.py, profile_engine.py) assign
a different agent to each faction and merge them into the single set of
dicts run_turn/placement.py's run_city_setup expect.
"""


def compose_agents(assignment, build_fns):
    """assignment: {faction: agent_key}. build_fns: {agent_key: () ->
    (decide_buy, decide_movement, decide_cavalry, decide_target,
    decide_rectification, ...)}, one zero-arg builder per agent kind -
    called at most once each, and only for kinds actually referenced in
    `assignment` (so e.g. picking "nn" for zero factions never imports
    torch). Returns the same tuple shape every build_fns[key]() returns
    (arity inferred from whichever kind gets built first - callers use
    this both for the original 5-tuple of turn.py decisions and for an
    8-tuple that also covers placement.py's setup-phase decisions),
    stitched together faction by faction from whichever kind that
    faction was assigned."""
    combined = None
    built = {}
    for faction, key in assignment.items():
        if key not in built:
            built[key] = build_fns[key]()
        if combined is None:
            combined = tuple({} for _ in built[key])
        for combined_dict, source_dict in zip(combined, built[key]):
            combined_dict[faction] = source_dict[faction]
    return combined
