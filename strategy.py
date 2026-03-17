"""
strategy.py — Strategy interface and built-in implementations.

A Strategy is called once per turn and makes two decisions simultaneously:

  choose_shot(rod, ball_y, field)
      → (target_angle, target_speed)
      Called when `rod` has possession. Returns the INTENDED shot direction
      and speed. Actual shot will have noise added by simulation.py based on
      rod.angle_std and rod.speed_std.

  choose_position(rod, ball_x, ball_y, field)
      → y_offset
      Called for EVERY rod of this team each turn (including non-possessing
      rods). Returns the desired sliding offset so rods can defend or set up
      for passes.

Both methods are called before any dice are rolled — that's what makes the
decision simultaneous: attacker and all defenders commit at the same time.

Adding a new strategy
---------------------
Subclass Strategy and implement both methods. Register it in main.py or
pass it directly to run_monte_carlo().
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod

import config
from field import Field, Rod


class Strategy(ABC):

    @abstractmethod
    def choose_shot(
        self,
        possessing_rod: Rod,
        ball_y: float,
        field: Field,
    ) -> tuple[float, float]:
        """
        Decide a shot.

        Parameters
        ----------
        possessing_rod : the rod that currently has the ball.
        ball_y         : current ball y-position (= possessing player's center).
        field          : full field (read-only — do not mutate rod offsets here).

        Returns
        -------
        (target_angle_rad, target_speed_cm_per_s)
        """

    @abstractmethod
    def choose_position(
        self,
        rod: Rod,
        ball_x: float,
        ball_y: float,
        field: Field,
    ) -> tuple[float, float]:
        """
        Decide where to slide `rod`.

        Parameters
        ----------
        rod         : the rod being positioned.
        ball_x/y    : current ball position (use for tracking / anticipation).
        field       : full field (read-only).

        Returns
        -------
        (y_offset, x_offset) — both will be clamped by Rod.set_offset / set_x_offset.
        """


# ---------------------------------------------------------------------------
# AimAtGoalCenter
# ---------------------------------------------------------------------------

class AimAtGoalCenter(Strategy):
    """
    Always aim at the exact center of the opponent's goal.

    Shooting: angle = atan2(goal_center_y - ball_y, goal_x - rod_x)
              speed = MAX_SPEED

    Defending/receiving: slide the rod's middle player to track ball_y.
    This keeps a defender in front of the most likely shot line and also
    positions own-team rods to receive a straight pass.
    """

    def choose_shot(
        self,
        possessing_rod: Rod,
        ball_y: float,
        field: Field,
    ) -> tuple[float, float]:
        goal = field.goal_for_attacker(possessing_rod.team)
        goal_center_y = (goal.y_min + goal.y_max) / 2

        dx = goal.x - possessing_rod.x
        dy = goal_center_y - ball_y
        target_angle = math.atan2(dy, dx)

        return target_angle, config.MAX_SPEED

    def choose_position(
        self,
        rod: Rod,
        ball_x: float,
        ball_y: float,
        field: Field,
    ) -> tuple[float, float]:
        # Align the middle player's center with ball_y.
        # Middle player's base position is always FIELD_HEIGHT / 2 (by construction),
        # so the required offset is simply ball_y - FIELD_HEIGHT/2.
        return ball_y - field.height / 2, 0.0


# ---------------------------------------------------------------------------
# AimAtRandomGoalPoint
# ---------------------------------------------------------------------------

class AimAtRandomGoalPoint(Strategy):
    """
    Shoots at a uniformly random y-position inside the goal opening each turn.

    Models a player who knows the goal location but picks an unpredictable
    target, making it harder for a simple tracking defender to anticipate.

    Defending: same ball-tracking position as AimAtGoalCenter.
    """

    def choose_shot(
        self,
        possessing_rod: Rod,
        ball_y: float,
        field: Field,
    ) -> tuple[float, float]:
        goal = field.goal_for_attacker(possessing_rod.team)
        target_y = random.uniform(goal.y_min, goal.y_max)

        dx = goal.x - possessing_rod.x
        dy = target_y - ball_y
        target_angle = math.atan2(dy, dx)

        return target_angle, config.MAX_SPEED

    def choose_position(
        self,
        rod: Rod,
        ball_x: float,
        ball_y: float,
        field: Field,
    ) -> tuple[float, float]:
        return ball_y - field.height / 2, 0.0


# ---------------------------------------------------------------------------
# RandomStrategy
# ---------------------------------------------------------------------------

class RandomStrategy(Strategy):
    """
    Shoots in a random direction within ±max_spread of the forward direction.
    Positions rods randomly within their valid range.

    Useful as a lower-bound baseline: any real strategy should beat Random.

    Parameters
    ----------
    max_spread : half-width of the angle range (radians). Default = 45°.
    """

    def __init__(self, max_spread: float = math.radians(45)):
        self.max_spread = max_spread

    def choose_shot(
        self,
        possessing_rod: Rod,
        ball_y: float,
        field: Field,
    ) -> tuple[float, float]:
        # Forward direction: 0 for Team 0 (rightward), π for Team 1 (leftward)
        base_angle = 0.0 if possessing_rod.team == 0 else math.pi
        angle = base_angle + random.uniform(-self.max_spread, self.max_spread)
        speed = random.uniform(config.MIN_SPEED, config.MAX_SPEED)
        return angle, speed

    def choose_position(
        self,
        rod: Rod,
        ball_x: float,
        ball_y: float,
        field: Field,
    ) -> tuple[float, float]:
        lo, hi = rod.slide_range
        return random.uniform(lo, hi), 0.0
