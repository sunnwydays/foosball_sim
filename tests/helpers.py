"""
Helpers for building exact, noise-free scenes.

Rod indices (config.ROD_CONFIGS), with the player at y = 34 (field centre,
where kickoff puts the ball) on each rod:

  idx  x      rod          players  player at y=34
  0    9.0    T0 goalie    3        1
  1    23.6   T0 defense   2        none (gap)
  2    38.1   T1 forward   3        1
  3    52.7   T0 mid       5        2   <- T0 kickoff rod
  4    67.3   T1 mid       5        2   <- T1 kickoff rod
  5    81.9   T0 forward   3        1
  6    96.4   T1 defense   2        none (gap)
  7    111.0  T1 goalie    3        1
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import config
from field import BallState, Field, Rod, SwingCommitment
from simulation import PointResult, simulate_point
from strategy import Strategy

T0_GOA, T0_DEF, T1_FWD, T0_MID, T1_MID, T0_FWD, T1_DEF, T1_GOA = range(8)
MID_CENTRE = 2   # index of the y=34 player on a 5-man rod
GOA_CENTRE = 1   # index of the y=34 player on a 3-man rod


# ---------------------------------------------------------------------------
# Unit-test helpers
# ---------------------------------------------------------------------------

def hold(rod: Rod) -> Rod:
    """Put a hand on the rod with its switch delay already over."""
    rod.controlled = True
    rod.switch_timer = 0.0
    return rod


def lift_others(field: Field, keep=()) -> None:
    """Lift every rod not in `keep` (indices) so only those can touch the ball."""
    for i, rod in enumerate(field.rods):
        if i not in keep:
            rod.up = True


def half_thickness(rod: Rod) -> float:
    """Slab half-depth in x, including the ball radius."""
    return rod.thickness / 2 + config.BALL_RADIUS


def half_width(rod: Rod) -> float:
    """Slab half-width in y, including the ball radius."""
    return rod.width / 2 + config.BALL_RADIUS


def ball_at(rod: Rod, player_idx: int, gap: float, side: int,
            vx: float = 0.0, vy: float = 0.0, dy: float = 0.0) -> BallState:
    """
    Ball `gap` cm off a player's front face on the `side` (+1 = +x) of the rod,
    like debug_snap.py --manual. gap = cm between the ball's surface and the
    face (0 = touching, negative = overlapping). dy offsets it from the
    player's centre line.
    """
    py = rod.player_positions[player_idx]
    return BallState(x=rod.x + side * (half_thickness(rod) + gap), y=py + dy, vx=vx, vy=vy)


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

Pos = tuple[float, float, bool]


def swing(vx: float, vy: float, active_start: float = 0.0,
          window_end: float = config.SWING_DURATION) -> SwingCommitment:
    """A swing with an explicit window; vx/vy are set on the ball as-is."""
    return SwingCommitment(vx=vx, vy=vy, active_start=active_start, window_end=window_end)


class ScriptedStrategy(Strategy):
    """
    A strategy that does exactly what the test says, with no noise.

    hands   : set of rod indices, or callable(ball) -> set.
    pos     : {rod_idx: (target_y, target_x, up) or callable(ball) -> that}
              for held rods; unlisted held rods sit at (0, 0, down). Keep
              target_y at 0: any other target adds movement noise.
    passive : {rod_idx: (target_x, up)} for free rods; unlisted free rods are
              LIFTED, so only the rods a test names can touch the ball.
    swings  : {rod_idx: SwingCommitment}, armed the first time choose_hit is
              asked about that rod (t = 0 for a rod held from kickoff).

    choose_hit's commitment sets the ball's velocity directly, so a swing on
    the kickoff rod launches an exact, noise-free shot through the real
    simulate_point loop.
    """

    def __init__(
        self,
        hands: Union[set[int], Callable[[BallState], set[int]]],
        pos: Optional[dict] = None,
        passive: Optional[dict[int, tuple[float, bool]]] = None,
        swings: Optional[dict[int, SwingCommitment]] = None,
    ):
        self.hands = hands
        self.pos = pos or {}
        self.passive = passive or {}
        self.swings = dict(swings or {})

    def choose_hands(self, team_rods, ball, field, n_hands):
        hands = self.hands(ball) if callable(self.hands) else self.hands
        return set(hands)

    def choose_pos(self, controlled_rods, ball, field):
        out = {}
        for idx, _ in controlled_rods:
            p = self.pos.get(idx, (0.0, 0.0, False))
            out[idx] = p(ball) if callable(p) else p
        return out

    def choose_passive(self, passive_rods, ball, field):
        return {idx: self.passive.get(idx, (0.0, True)) for idx, _ in passive_rods}

    def choose_hit(self, rod, ball, field, t):
        return self.swings.pop(field.rods.index(rod), None)


def up_once_moving(ball: BallState) -> Pos:
    """pos entry for a kickoff rod: down to shoot, lifted once the ball moves."""
    return (0.0, 0.0, ball.speed > 0)


def run_scene(t0: Strategy, t1: Strategy, kickoff: int = 0) -> tuple[Field, PointResult]:
    """Play one recorded, logged point between two scripted teams."""
    field = Field()
    result = simulate_point(field, {0: t0, 1: t1}, kickoff_team=kickoff,
                            record=True, seed=0, collect_action_log=True)
    return field, result


def log_of(result: PointResult, action: str, rod_label: Optional[str] = None) -> list:
    """Action-log entries with this action (and rod label, if given)."""
    return [e for e in result.action_log
            if e.action == action and (rod_label is None or e.rod_label == rod_label)]
