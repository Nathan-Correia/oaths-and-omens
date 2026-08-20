# Oaths & Omens

## Overview

Oaths and Omens is a game of territorial conquest, betrayal, and diplomacy played on a hexagonal map. Players command armies of infantry, cavalry, and archers, capture and defend cities, and vie for the highest score by game's end. Trading, deceit, and shifting alliances are part of the game — anything goes.

## Pieces

Each player controls one faction and begins the game with a bag containing:

- 24 infantry
- 12 cavalry
- 12 archers
- 2 cities
- 1 D20 die

## Map Creation

The map is built from hexagonal tiles. A standard board uses 217 hexagons; the full set includes up to 300 hexagons, giving players flexibility to build maps of different sizes and shapes.

## Game Setup

The starting player is determined by a high dice roll. That player places their first city on any tile of their choosing. Play then proceeds clockwise, with each remaining player placing their first city in turn.

Once every player has placed a first city, players place their second city in counter-clockwise order, starting with whichever player placed last in the first round.

City placement restrictions:
- A city must be at least 2 tiles from any opponent's city.
- A city must be at least 1 tile from your own other city.
- With 5–7 players, cities cannot be placed on edge tiles.
- With 8–10 players, cities may be placed on edge tiles.

After both cities are placed, players may rearrange their starting positions to sit closer to their own cities.

Each player begins the game with 50 silver and 2 kill experience, awarded once at the very first buy phase instead of that turn's regular income.

## Economy & the Buy Phase

Each turn begins with a buy phase, right after income is collected.

**Income**: every player collects 3 silver per turn. If a player controls more than 2 cities, they earn an additional 1 silver for every city beyond their second. A player with no cities remaining earns no income and instead loses 1 unit every turn.

**Buying units**: infantry cost 2 silver each and can be purchased at any city you control, as long as no enemy units occupy a tile orthogonally adjacent to that city. Cavalry and archers cannot be purchased with silver — they can only be acquired by trading in kill experience (see below).

**Production limits**: each player may have at most 24 infantry, 12 cavalry, and 12 archers in play at once. This is a concurrent limit, not a lifetime total — losing a unit immediately frees up capacity to field another of that type.

**Trading**: players may trade anything with each other — silver, kill experience, units, cities, and favours. Units and cities may only actually change hands during the buy phase.

## Movement

Units stack together on a tile into a single force called an **army**. An army may hold up to 6 units at a time.

All players move simultaneously: everyone selects and commits their moves for the turn, and all movement is resolved together. If a player does not commit their movement in time, they miss that step.

Movement happens in two parts each turn:

**Regular movement phase**: 3 steps. Each step, every player may move exactly one army 1 hex — even if you control several eligible armies, only one of them may act per step. Any number of units may be split off from that army and moved independently — you don't have to move a whole stack together.

**Cavalry movement phase**: an additional 4 steps, usable only by cavalry units, and subject to the same one-army-per-step rule. Each individual cavalry unit may move up to 2 hexes total during this phase, and a player may not use more than 4 total cavalry steps across their whole cavalry force.

If multiple armies end up moving into the same tile, or attack each other, a battle begins. All armies involved in a battle are locked in place — they cannot move again — until the battle phase resolves the fight.

## Battles

There are three ways a battle can begin:

- **Attack/Defense**: an army moves into a tile already occupied by another player's army.
- **Encounter Battle**: two or more armies simultaneously move into the same empty tile.
- **Line Battle**: two armies simultaneously attack each other from adjacent tiles.

Once triggered, all three battle types resolve the same way.

### Reinforcements and overstacking

Because armies lock in place as soon as a battle starts, additional armies are free to move into a battle on a later step of the same movement phase, joining the fight as reinforcements. This means a battle can temporarily exceed the normal 6-unit stack limit while it's ongoing. Once the battle ends, if the winning side has more than 6 units remaining, the winning player must reduce their stack back down to 6 by sending the excess units back to any tile their own units came from during that battle.

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
| **Forest** | An army attacking out of a forest tile gains +2 to all of its battle rolls. |
| **Plains** | No special effect. |
| **City** | Defending units on a city tile gain +2 to their battle rolls. This bonus stacks with other terrain bonuses. |

## Cities

Cities are captured simply by occupying them: if a player moves an army into a city tile that isn't defended by another army, they immediately take ownership of that city. If the city is defended, ownership only changes hands if the attacker wins the resulting battle and occupies the tile.

## Elimination

A player is eliminated once they control no cities and have no units remaining anywhere on the board. Both conditions must be true — a player with units but no cities, or cities but no units, is not eliminated.

## Win Condition

The game ends after a fixed number of turns. Final scores are tallied from two categories: Cities and Military. In the event of a tie, the tiebreaker order is: most cities, then most units remaining.

**Cities**: each player scores 1 point for every city they control at the end of the game.

**Military**: each player counts their remaining units at game's end. The player with the most units receives 3 points, the second-most receives 2 points, and the third-most receives 1 point.

The player with the highest total score wins the game.