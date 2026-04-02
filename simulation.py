"""
simulation.py — Time-stepped foosball simulation engine.

simulate_point() runs one point as a sequence of ticks (dt = 1/FPS).

Tick order
----------
1. Decrement timers (reaction, switch)
2. Strategy: choose_hands → choose_targets (respecting reaction lockout)
3. Move rods toward targets (step_movement)
4. Move ball (step_ball: velocity, friction, wall bounces)
5. Detect player-ball overlaps → strategy: choose_hit
6. Apply hits (add velocity, set reaction timer, update last-hit tracking)
7. Check goal / time limit

Frame recording
---------------
If record=True, every tick's state is appended to a frame list for
visualization or replay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Optional

import numpy as np

import config
from field import Field, BallState, TeamState
from physics import step_ball, find_overlapping_players
from strategy import Strategy


# ---------------------------------------------------------------------------
# Result / Frame dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """Snapshot of the game state at one tick (for replay / visualization)."""
    tick:       int
    time:       float
    ball_x:     float
    ball_y:     float
    ball_vx:    float
    ball_vy:    float
    rod_ys:     list[float]          # y_offset per rod
    rod_xs:     list[float]          # x_offset per rod
    rod_ctrl:   list[bool]           # controlled? per rod
    ups:        list[bool]           # up (flipped) state per rod
    event:      Optional[str] = None # 'hit', 'goal:0', 'goal:1', etc.


@dataclass
class PointResult:
    """
    The outcome of a single simulated point.

    winner : 0 or 1 (team that scored), or None (time limit / dead ball).
    ticks  : number of ticks elapsed.
    time   : game time in seconds.
    frames : list of Frame snapshots (only if record=True).
    """
    winner: Optional[int]
    ticks:  int
    time:   float
    frames: list[Frame] = dc_field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _teams_with_reach(ball: BallState, field: Field) -> set[int]:
    """Return set of team IDs that have at least one player within reach of the ball."""
    teams: set[int] = set()
    for rod in field.rods:
        if rod.up:
            continue
        for py in rod.player_positions:
            if (abs(ball.x - rod._base_x) <= rod.rod_x_reach + rod.thickness / 2
                    and abs(ball.y - py) <= rod.width / 2):
                teams.add(rod.team)
                break
    return teams


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_point(
    field:       Field,
    strategies:  dict[int, Strategy],
    kickoff_team: int = 0,
    n_hands:     int = config.N_HANDS,
    record:      bool = False,
    seed:        Optional[int] = None,
    pos_grid:    Optional[np.ndarray] = None,
    goal_hits:   Optional[list] = None,
) -> PointResult:
    """
    Simulate one foosball point with time-stepped physics.

    Parameters
    ----------
    field        : Field instance (rods will be mutated in-place).
    strategies   : {team_id: Strategy} — one per team.
    kickoff_team : which team kicks off (0 or 1).
    n_hands      : max rods each team can control simultaneously.
    record       : if True, record a Frame per tick for visualization.
    seed         : optional RNG seed for reproducibility.

    Returns
    -------
    PointResult
    """
    if seed is not None:
        np.random.seed(seed)

    dt = config.DT

    # --- Initialize state ---
    field.reset()

    kickoff_rod_idx = config.KICKOFF_ROD[kickoff_team]
    kickoff_rod = field.rods[kickoff_rod_idx]

    ball = BallState(
        x  = kickoff_rod.x,
        y  = field.width / 2,
        vx = 0.0,
        vy = 0.0,
    )

    team_states = {
        0: TeamState(),
        1: TeamState(),
    }

    # Double-hit tracking: (rod_idx, player_idx) of last hit
    last_hit: Optional[tuple[int, int]] = None
    last_hit_pos: Optional[tuple[float, float]] = None  # ball pos of last active hit

    frames: list[Frame] = []

    # --- Pre-game: initial hand assignment (no switch delay) ---
    for team in (0, 1):
        strat = strategies[team]
        ts    = team_states[team]
        team_rods = field.rods_for_team(team)
        initial_hands = strat.choose_hands(team_rods, ball, field, n_hands)
        for rod_idx in initial_hands:
            field.rods[rod_idx].controlled = True
            # No switch_timer — starting positions, not a mid-game switch
        ts.active_rods = initial_hands

        # Set initial targets for controlled rods
        controlled = [(i, field.rods[i]) for i in initial_hands]
        targets = strat.choose_targets(controlled, ball, field)
        for rod_idx, (ty, tx, up) in targets.items():
            rod = field.rods[rod_idx]
            rod.set_target_y(ty)
            rod.set_x_offset(tx)
            rod.up = up

        # Set initial passive rod positions
        all_rod_idxs = {idx for idx, _ in team_rods}
        passive = [(i, field.rods[i]) for i in all_rod_idxs - initial_hands]
        if passive:
            passive_targets = strat.choose_passive(passive, ball, field)
            for rod_idx, (tx, up) in passive_targets.items():
                rod = field.rods[rod_idx]
                rod.set_x_offset(tx)
                rod.up = up

    possession_timer: dict[int, float] = {0: 0.0, 1: 0.0}

    # --- Tick loop ---
    for tick in range(config.MAX_TICKS):
        game_time = tick * dt
        event: Optional[str] = None

        # --------------------------------------------------------------
        # 1. Decrement timers
        # --------------------------------------------------------------
        for ts in team_states.values():
            ts.tick(dt)

        # --------------------------------------------------------------
        # 2. Strategy decisions: hands + targets + up/down
        # --------------------------------------------------------------
        for team in (0, 1):
            strat = strategies[team]
            ts    = team_states[team]
            team_rods = field.rods_for_team(team)

            # Choose which rods to hold
            desired_hands = strat.choose_hands(team_rods, ball, field, n_hands)

            # Determine which rods are newly grabbed vs kept
            prev_hands = ts.active_rods
            released   = prev_hands - desired_hands
            acquired   = desired_hands - prev_hands

            # Released rods stop immediately
            for rod_idx in released:
                rod = field.rods[rod_idx]
                rod.controlled = False
                rod.vy = 0.0

            # Acquired rods get switch delay
            for rod_idx in acquired:
                rod = field.rods[rod_idx]
                rod.controlled = True
                rod.switch_timer = config.SWITCH_DELAY

            # Kept rods stay controlled
            for rod_idx in (desired_hands & prev_hands):
                field.rods[rod_idx].controlled = True

            ts.active_rods = desired_hands

            # Choose targets (only if not in reaction lockout)
            if not ts.reacting:
                controlled = [(i, field.rods[i]) for i in desired_hands]
                targets = strat.choose_targets(controlled, ball, field)
                for rod_idx, (ty, tx, up) in targets.items():
                    rod = field.rods[rod_idx]
                    rod.set_target_y(ty)
                    rod.set_x_offset(tx)
                    rod.up = up

            # If reacting: rods keep moving toward their previous target_y
            # (no new commands issued — this is the reaction time lockout)

            # Passive rod positioning (always available, even during reaction)
            all_rod_idxs = {idx for idx, _ in team_rods}
            passive = [(i, field.rods[i]) for i in all_rod_idxs - desired_hands]
            if passive:
                passive_targets = strat.choose_passive(passive, ball, field)
                for rod_idx, (tx, up) in passive_targets.items():
                    rod = field.rods[rod_idx]
                    rod.set_x_offset(tx)
                    rod.up = up

        # --------------------------------------------------------------
        # 3. Move rods
        # --------------------------------------------------------------
        for rod in field.rods:
            rod.step_movement(dt)

        # --------------------------------------------------------------
        # 4. Move ball
        # --------------------------------------------------------------
        ball_result = step_ball(ball, field, dt,
                                ball_radius=config.BALL_RADIUS,
                                pos_grid=pos_grid)

        # --- Anti-stalling checks ---
        teams_in_reach = _teams_with_reach(ball, field)

        if ball.stopped and not teams_in_reach:
            return PointResult(winner=None, ticks=tick + 1, time=game_time + dt, frames=frames)

        if len(teams_in_reach) == 1:
            possessing = next(iter(teams_in_reach))
            possession_timer[possessing] += dt
            possession_timer[1 - possessing] = 0.0
            if possession_timer[possessing] >= config.POSSESSION_LIMIT:
                return PointResult(winner=1 - possessing, ticks=tick + 1, time=game_time + dt, frames=frames)
        else:
            possession_timer[0] = possession_timer[1] = 0.0

        # Check for goal
        if ball_result.startswith('goal:'):
            winner = int(ball_result.split(':')[1])
            if goal_hits is not None and last_hit_pos is not None:
                goal_team = 1 - winner  # team whose goal the ball entered
                goal_hits.append((*last_hit_pos, goal_team))
            if record:
                frames.append(_make_frame(tick, game_time, ball, field, f'goal:{winner}'))
            return PointResult(
                winner = winner,
                ticks  = tick + 1,
                time   = game_time + dt,
                frames = frames,
            )

        # --------------------------------------------------------------
        # 5. Detect overlaps + strategy decides whether to hit
        # --------------------------------------------------------------
        overlaps = find_overlapping_players(ball, field)

        for rod_idx, player_idx in overlaps:
            rod  = field.rods[rod_idx]
            team = rod.team

            # Double-hit check: same player can't hit twice in a row
            # unless ball has stopped (which resets last_hit)
            if last_hit == (rod_idx, player_idx):
                continue

            # Ask strategy whether to hit
            hit_vel = strategies[team].choose_hit(rod, player_idx, ball, field)

            if hit_vel is not None:
                hvx, hvy = hit_vel

                # Add velocity to ball
                ball.vx += hvx
                ball.vy += hvy

                # Clamp to max speed
                speed = ball.speed
                if speed > config.BALL_MAX_SPEED:
                    factor = config.BALL_MAX_SPEED / speed
                    ball.vx *= factor
                    ball.vy *= factor

                # Set opponent's reaction timer
                opp_team = 1 - team
                team_states[opp_team].reaction_timer = config.REACTION_TIME

                # Update last-hit tracking
                last_hit = (rod_idx, player_idx)
                last_hit_pos = (ball.x, ball.y)
                event = 'hit'

                # Only one hit per tick (first overlap wins)
                break

        # Reset last_hit if ball stopped
        if ball.stopped:
            last_hit = None

        # --------------------------------------------------------------
        # 6. Record frame
        # --------------------------------------------------------------
        if record:
            frames.append(_make_frame(tick, game_time, ball, field, event))

    # --- Time limit reached ---
    return PointResult(
        winner = None,
        ticks  = config.MAX_TICKS,
        time   = config.MAX_GAME_TIME,
        frames = frames,
    )


def _make_frame(
    tick: int,
    time: float,
    ball: BallState,
    field: Field,
    event: Optional[str],
) -> Frame:
    return Frame(
        tick     = tick,
        time     = time,
        ball_x   = ball.x,
        ball_y   = ball.y,
        ball_vx  = ball.vx,
        ball_vy  = ball.vy,
        rod_ys   = [r.y_offset for r in field.rods],
        rod_xs   = [r.x_offset for r in field.rods],
        rod_ctrl = [r.controlled for r in field.rods],
        ups      = [r.up for r in field.rods],
        event    = event,
    )
