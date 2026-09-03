# Oaths & Omens

## Overview

Oaths and Omens is a game of territorial conquest, betrayal, and diplomacy played on a hexagonal map. Players command armies of infantry, cavalry, and archers, defend their capital, and race to be the first to reach 50 victory points by building outposts across the map - and destroying everyone else's. Trading, deceit, and shifting alliances are part of the game — anything goes.

## Pieces

Each player controls one faction and begins the game with a bag containing:

- 24 infantry
- 12 cavalry
- 12 archers
- 1 city (your capital)
- 1 D20 die

## Map Creation

The map is built from hexagonal tiles. A standard board uses 217 hexagons; the full set includes up to 300 hexagons, giving players flexibility to build maps of different sizes and shapes.

## Game Setup

Capitals are placed and assigned in two steps: placement, then a draft.

**Placement**: randomize a placement order. In that order, each player places one colourless (unowned) city on any tile of their choosing, subject to the usual restrictions:
- A colourless city must be at least 3 tiles from any other colourless city already placed.
- With 5–7 players, cities cannot be placed on edge tiles.
- With 8–10 players, cities may be placed on edge tiles.

**Draft**: once every city is placed, randomize a separate draft order (independent of the placement order). In that order, each player claims one already-placed city as their capital — never the one they placed themselves.

When only one player and one city are left: if it's the city that player placed, they simply get it — there's no decision to make. Otherwise, that leftover city was placed by some other player who already drafted a different city earlier. The last player chooses: keep the leftover city, or force a swap — take the city the placer already drafted, bumping the placer onto the leftover instead.

After every capital is settled, players may rearrange their starting positions to sit closer to their own capital.

Each player begins the game with 50 silver and 2 kill experience, awarded once at the very first buy phase instead of that turn's regular income.

## Turn Structure

Each turn proceeds through four phases, in order:

1. **Buy** — spend silver and resources accumulated from previous turns.
2. **Movement** — regular movement, then cavalry movement.
3. **Combat** — resolve any battles triggered by movement.
4. **Collect** — gain this turn's silver income, resources, and victory points.

Because income is collected at the *end* of the turn, the buy phase always spends silver left over from prior turns — you can never spend the silver you're about to earn this turn until your next buy phase.

## Economy & the Buy Phase

**Income**: every player collects 3 silver per turn from their capital, plus 1 additional silver for each outpost they control, or 2 additional silver for each outpost they control that has a Barracks upgrade. Income is collected during the Collect phase (see Turn Structure). A player with no cities remaining earns no income and instead loses 1 unit every turn — in practice this can no longer happen, since your capital cannot be captured or destroyed (see Capitals).

**Buying units**: infantry cost 2 silver each. Your capital can always recruit infantry, no matter what's adjacent to it. An outpost can only recruit infantry if no enemy units occupy a tile orthogonally adjacent to it, and is limited to at most 1 new infantry this way per turn, unless it has a Barracks upgrade (see Outpost Upgrades), which removes the 1-per-turn cap but not the adjacent-enemy restriction. Your capital has neither restriction. Cavalry and archers cannot be purchased with silver — they can only be acquired by trading in kill experience (see below).

**Production limits**: each player may have at most 24 infantry, 12 cavalry, and 12 archers in play at once. This is a concurrent limit, not a lifetime total — losing a unit immediately frees up capacity to field another of that type.

**Trading**: players may trade anything with each other — silver, kill experience, units, cities, resources, action cards, and favours. Units and cities may only actually change hands during the buy phase.

## Movement

Units stack together on a tile into a single force called an **army**. An army may hold up to 6 units at a time.

All players move simultaneously: everyone selects and commits their moves for the turn, and all movement is resolved together. If a player does not commit their movement in time, they miss that step.

Movement happens in two parts each turn:

**Regular movement phase**: 3 steps. Each step, every player may move exactly one army 1 hex — even if you control several eligible armies, only one of them may act per step. Any number of units may be split off from that army and moved independently — you don't have to move a whole stack together.

**Cavalry movement phase**: an additional 2 steps, usable only by cavalry units, and subject to the same one-army-per-step rule. Each individual cavalry unit may move up to 2 hexes total during this phase.

If multiple armies end up moving into the same tile, or attack each other, a battle begins. All armies involved in a battle are locked in place — they cannot move again — until the battle phase resolves the fight.

## Battles

There are three ways a battle can begin:

- **Attack/Defense**: an army moves into a tile already occupied by another player's army.
- **Encounter Battle**: two or more armies simultaneously move into the same empty tile.
- **Line Battle**: two armies simultaneously attack each other from adjacent tiles.

Moving into a tile with another player's capital or outpost is always treated as an attack and triggers a battle, even if it has no defending army standing on it — see Capitals and Outposts for what happens next.

Once triggered, all battle types resolve the same way.

### Reinforcements and overstacking

Because armies lock in place as soon as a battle starts, additional armies are free to move into a battle on a later step of the same movement phase, joining the fight as reinforcements. This means a battle can temporarily exceed the normal 6-unit stack limit while it's ongoing. Once the battle ends, if the winning side has more than 6 units remaining, the winning player must reduce their stack back down to 6 by sending the excess units back to any tile their own units came from during that battle (see Capitals for the one case where the whole stack has to go back, not just the excess).

Outside of battle, the 6-unit limit is strict: a peaceful move that would merge into a stack and push it past 6 units simply doesn't happen. The move is illegal and the army stays where it started.

### Resolving a Battle

A battle is fought in rounds. Each round, every player still in the fight rolls their D20 once against a chosen target.

**Choosing targets**: before each round, every player in the battle simultaneously chooses which opposing army they are attacking. No army may be attacked by more than one player in the same round — if two or more players choose the same target, only the player with the larger army follows through with their attack that round; the others' attacks do not happen. Players may also choose to abstain from attacking in a given round.

**Rolling**: each player who has a valid, unconflicted target rolls their D20 against that target:

| Roll | Result |
|------|--------|
| 1–5 | 0 kills |
| 6–15 | 1 kill |
| 16–20 | 2 kills |

If the attacking player has only 1 unit remaining in the battle, a roll of 16–20 results in a single kill instead of two.

All rolls for a round happen simultaneously, and all resulting kills are applied at the same time — including cases where two opposing armies land kills on each other in the same round.

**Casualties**: when an army takes casualties, units are always lost in a fixed order — infantry first, then cavalry, then archers.

A battle continues, round after round, until only one player's army remains standing on the tile. If every remaining player's army is wiped out simultaneously, the tile is left empty with no one victorious.

### Special Unit Abilities

**Archers**: when a player's archers are part of a battle, each archer unit rolls its ability once, before the first round of combat begins. On a roll of 11–20, that archer scores 1 kill against the opposing army with the most units.

**Cavalry**: whenever a cavalry unit dies in battle, its owner immediately rolls its ability. On a roll of 14–20, that cavalry dismounts: a new infantry unit joins the same battle immediately as a live combatant (subject to the normal 12/24 concurrent unit limits) — this can even keep an otherwise-defeated army in the fight. On a roll of 1–13, nothing happens.

## Kill Experience

Whenever a player's unit kills an opposing unit in battle, that player keeps the defeated unit as kill experience.

A player may trade in 1 kill experience and 1 silver to convert one of their own infantry units, anywhere on the board, into a cavalry or archer unit — this is the only way to acquire these unit types.

Kill experience represents accumulated kills and cannot be hoarded indefinitely to deny opponents the ability to spawn units. If a player needs to spawn a unit but all their pieces are currently held by opponents as kill experience, they simply take the piece they need, and a random piece is instead removed from some other player's kill experience holdings to replace it.

## Terrain

| Terrain | Effect |
|---|---|
| **Mountain** | Impassable. |
| **Lake** | Impassable. |
| **Desert** | Any army that ends its full turn on a desert tile loses 1 unit — unless that tile has a city on it, in which case the loss doesn't apply. |
| **Marsh** | Any army that enters a marsh tile is frozen and cannot move again for the rest of that turn. |
| **Plains** | No special effect. |

Mountain and lake tiles can never be placed if doing so would cut off part of the map — if a tile would seal off a pocket of the board with no passable route back to the rest of it, that tile is placed somewhere else instead. Every non-mountain, non-lake tile must stay reachable by land from every other one.

## Capitals

Each player has exactly one capital, placed during setup. A capital cannot be captured or destroyed by any means: entering its tile is always treated as an attack and triggers a battle, even if the capital has no defending army of its own — the capital fights back on its own (see Defense, below) regardless.

If an attacker somehow wins that battle, ownership never changes hands. Instead, the winner is evicted immediately: every one of their surviving units is sent straight back to wherever it came from during that battle, no matter how few units made it through. You simply cannot stand units on another player's capital, win or lose.

**Defense**: a capital gets 2 free defensive shots the instant its tile is attacked, before the first round of combat (11–20 on a d20 = 1 kill each, aimed at the single largest attacking army) — fired even with no defending army present. This stacks with any archers actually stationed there, and is a separate mechanic from the real Archers ability above (kept independent so it can be tuned on its own later).

## Outposts

Outposts are smaller, disposable extensions of a player's territory — unlike a capital, they can be destroyed.

**Building**: costs 3 silver and consumes 1 unit (any type) already standing on the tile you're building on, up to a maximum of 6 outposts per player at once. A player may take at most one outpost action per turn during the buy phase, where an "outpost action" is either building a new outpost or upgrading/converting an existing one (see Outpost Upgrades) — never more than one of either, combined, per turn.

**Placement restrictions**: an outpost cannot be built within 2 tiles of your own capital, within 1 tile of any other player's capital, or within 1 tile of any existing outpost — yours or anyone else's. Outposts may be built on any terrain type that is not impassable, including desert.

**Recruiting**: infantry may be purchased at an outpost for 2 silver, same as at a capital, but only if no enemy units occupy a tile orthogonally adjacent to it, and at most 1 new infantry may be recruited per outpost per turn — unless the outpost has a Barracks upgrade (see Outpost Upgrades).

**Destruction**: entering a tile with an enemy outpost is always treated as an attack and triggers a battle, even if the outpost has no defending army of its own. If, once the battle ends, no other faction's units remain on that tile besides the winner's, the outpost is destroyed — its owner loses it, and the winner does not take ownership of it, but does keep standing there as normal (no forced eviction, unlike a capital).

**Defense**: an outpost gets 1 free defensive shot the instant its tile is attacked, before the first round of combat (11–20 = 1 kill, aimed at the single largest attacking army) — fired even with no defending army present, the same mechanic as a capital's defense but weaker, and tracked separately from it.

### Outpost Upgrades

An outpost may hold at most one upgrade at a time. Upgrades are mutually exclusive — to change an outpost from one upgrade to another, pay the new upgrade's full cost, which converts it directly (no need to strip the old upgrade first). Building an outpost, upgrading an unupgraded outpost, and converting an outpost from one upgrade to another are all "outpost actions" and are each subject to the one-per-turn limit described above.

Upgrade costs:

| Upgrade | Cost |
|---|---|
| **Barracks** | 2 Fish, 4 Wood |
| **Workshop** | 2 Iron, 2 Clay, 4 Wood |
| **Temple** | 2 Fish, 2 Iron, 2 Clay, 4 Wood |

There are three outpost upgrades:

- **Barracks**: removes the 1-infantry-per-turn recruiting cap at that outpost (the no-adjacent-enemy restriction still applies), and increases that outpost's silver income contribution from 1 to 2 per turn.
- **Temple**: adds +1 to the player's total victory-point-per-round output (see Win Condition), on top of what the outpost formula would otherwise produce.
- **Workshop**: doubles that outpost's resource output — 2 of its resource per turn instead of 1 (see Resources).

Note that converting an outpost directly from one upgrade to another still requires paying the new upgrade's full cost.

## Resources

Outposts generate resources every turn, collected during the Collect phase. There are four resource types: **Wood**, **Iron**, **Clay**, and **Fish**. Capitals never generate resources — only outposts do.

An outpost's resource output is determined by its own tile and its adjacent (orthogonally neighboring) tiles, as follows:

- If the outpost is on a **plains** tile, it produces 1 Wood per turn.
- If the outpost is on a **marsh** tile, it produces 1 Clay per turn.
- If the outpost is adjacent to a **mountain**, it produces 1 Iron per turn instead of whatever its own tile would produce.
- If the outpost is adjacent to a **lake**, it produces 1 Fish per turn instead of whatever its own tile would produce.
- If the outpost is adjacent to both a mountain and a lake, its owner chooses each turn whether it produces Iron or Fish.
- If the outpost is on a **desert** tile and is not adjacent to a mountain or lake, it produces no resources.

An outpost only ever produces one resource type per turn, regardless of how many qualifying neighboring tiles it has (e.g. being adjacent to two mountains does not produce 2 Iron). A Workshop upgrade doubles the single resource's output to 2 per turn, but does not grant a second resource type.

Resources may be spent to play action cards, to pay outpost upgrade costs, or traded to other players.

## Action Cards

A player may purchase at most 1 action card per turn, for 1 silver, taken at random from the top of the action card deck. Action cards are kept face-up, visible to all players at all times.

To play a card from hand, the player pays that card's resource cost and meets any other requirements listed on the card, after which the card is discarded. If the action card deck is ever emptied, the discard pile is shuffled to form a new deck.

The specific pool of action cards and their effects are still being finalized — see Open Items.

## Win Condition

Victory points are the win condition. The first player to reach 50 wins.

At the end of every round, each player earns victory points based on how many outposts they currently control — the first one earns nothing, and each additional outpost beyond that earns 1 more point per round: 2 outposts earns 1 VP/round, 3 earns 2 VP/round, 4 earns 3 VP/round, and so on. Each outpost with a Temple upgrade adds +1 to this total. Destroying another player's outpost also immediately awards the winner of that battle 2 victory points, on top of the recurring per-round total.

If, at the end of a round, one or more players are at or above 50 victory points, whoever has the single highest total wins immediately. If the top total is exactly tied between two or more players, whoever of them placed their capital later during setup wins the tiebreak.

## Open Items

The following are intentionally undecided and need values before the game is playable:

- The full action card list and each card's specific effects and resource costs.