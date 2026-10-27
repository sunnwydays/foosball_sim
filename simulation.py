"""
simulation.py — Time-stepped foosball simulation engine.

simulate_point() runs one point as a sequence of ticks (dt = 1/FPS).

Tick order
----------
1. Decrement timers (reaction, switch)
2. Strategy: choose_hands → choose_pos (respecting reaction lockout)
3. Move rods toward targets (step_movement)
4. Move ball (step_ball: velocity, friction, wall bounces), in substeps of at
   most CONTACT_STEP cm so fast balls can't skip a held rod; after each:
5. Check goal, detect held-player overlaps, resolve committed swings
   (hit, set reaction timer, update last-hit tracking) or rigid bounces
6. Dead-ball / stall checks, record frame; time limit after the last tick

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
from field import Field, BallState, TeamState, ActionLogEntry
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
    rod_ctrl:   list[bool]           # controlled? (switch complete) per rod
    rod_switching: list[bool]        # hand assigned but still mid-switch per rod
    ups:        list[bool]           # up (flipped) state per rod
    event:      Optional[str] = None # 'hit', 'goal:0', 'goal:1', etc.
    rod_swings: list                 = dc_field(default_factory=list)  # (active_start, window_end) or None per rod


@dataclass
class PointResult:
    """
    The outcome of a single simulated point.

    winner     : 0 or 1 (team that scored), or None (time limit / dead ball).
    ticks      : number of ticks elapsed.
    time       : game time in seconds.
    frames     : list of Frame snapshots (only if record=True).
    action_log : list of ActionLogEntry (only if collect_action_log=True).
    """
    winner:      Optional[int]
    ticks:       int
    time:        float
    frames:      list[Frame]          = dc_field(default_factory=list)
    action_log:  list[ActionLogEntry] = dc_field(default_factory=list)
    outcome:     str                  = 'goal'   # 'goal' | 'stallout' | 'dead_ball' | 'timeout'
    is_self_goal: bool                = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _teams_with_reach(ball: BallState, field: Field) -> set[int]:
    """Return set of team IDs that have at least one rod within x reach of the ball.
    Y is excluded — a rod can always slide to cover any ball y position.
    Lifted rods count too — the team can lower the rod onto the ball."""
    teams: set[int] = set()
    for rod in field.rods:
        if abs(ball.x - rod._base_x) <= rod.rod_x_reach + rod.thickness / 2 + config.BALL_RADIUS:
            teams.add(rod.team)
    return teams


def _contact_face(
    ball: BallState,
    rod,
    player_idx: int,
    x_pre: float,
    y_pre: float,
) -> Optional[tuple[str, float]]:
    """
    Which face of a rigid (controlled) player the ball touched this substep.

    Returns (axis, exit_sign): axis 'x' for the front face or 'y' for a side
    face, exit_sign the direction back out the side the ball CAME FROM. Returns
    None if the ball is only inside the wide reach box, not the player slab,
    and has not tunnelled through it.

    The face is chosen from where the ball was BEFORE the substep (x_pre,
    y_pre), not from the current overlap: a fast ball entering the front near
    a player's edge can end deeper in x than in y, which a least-penetration
    test on the current position misreads as a side hit.

    Rods move once per tick, before the ball's substeps, so the rod is normally
    taken where it is now: a goalie that slid in front of an incoming ball was
    there first, and blocks it with its front. Only when the ball was already
    inside the slab at the start of the substep did the rod move onto it; then
    the rod's earlier position (rod.x_pre, rod.y_offset_pre) tells which face
    reached the ball, e.g. a rod sliding sideways onto a ball level with it.
    """
    r  = config.BALL_RADIUS
    py = rod.player_positions[player_idx]
    ht = rod.thickness / 2 + r
    hw = rod.width / 2 + r
    dx = ball.x - rod.x
    dy = ball.y - py
    pen_x = ht - abs(dx)
    pen_y = hw - abs(dy)

    # Overlap one substep back in addition to current step to account for ball
    # going through player during a step, causing more overlap on another side.
    rx_pre, py_pre = rod.x, py
    pen_x_pre = ht - abs(x_pre - rx_pre)
    pen_y_pre = hw - abs(y_pre - py_pre)
    if pen_x_pre > 0 and pen_y_pre > 0 and rod.x_pre is not None:
        # Ball started inside the slab: the rod moved onto it this tick
        rx_pre = rod.x_pre
        py_pre = py - rod.y_offset + rod.y_offset_pre
        pen_x_pre = ht - abs(x_pre - rx_pre)
        pen_y_pre = hw - abs(y_pre - py_pre)

    if pen_x_pre >= pen_y_pre:  # which face the ball came through (prev step)
        # Side face (y)
        exit_sign = 1.0 if y_pre - py_pre > 0 else -1.0
        # Tunnelled: started outside the slab and is now past its centre. A ball
        # that was already inside (under a lifted man) is not tunnelling.
        crossed   = pen_y_pre <= 0 and (exit_sign > 0) != (dy > 0)
        # Touching now? Must overlap in x, and in y unless it tunnelled across
        if pen_x <= 0 or (not crossed and pen_y <= 0):
            return None
        return 'y', exit_sign
    # Front face (x)
    exit_sign = 1.0 if x_pre - rx_pre > 0 else -1.0
    crossed   = pen_x_pre <= 0 and (exit_sign > 0) != (dx > 0)
    # Touching now? Must overlap in y, and in x unless it tunnelled through
    if pen_y <= 0 or (not crossed and pen_x <= 0):
        return None
    return 'x', exit_sign


def _resolve_rigid_contact(
    ball: BallState,
    rod,
    player_idx: int,
    x_pre: float,
    y_pre: float,
    reflect: bool,
) -> bool:
    """
    Resolve contact between the ball and a rigid (controlled) player.

    Overlap detection uses the wide rotation-reach box (so swings can connect)
    and the bounce still resolves at the player slab like before, but the ball
    is always ejected through the face on the side it CAME FROM this substep
    (never the far side) and its velocity is set to point away from that face,
    so a bounce can never send the ball back into the player. A ball outside
    the slab that has not tunnelled through it is left untouched. See
    _contact_face for how the face is chosen.

    reflect=True  : rigid bounce at CONTROLLED_SLOWDOWN (held rod, no swing armed)
    reflect=False : absorbed bounce at PASSIVE_SLOWDOWN (currently unused: a
                    rod mid-switch is uncontrolled, so physics.step_ball's
                    passive contact handles it instead)

    Returns True if the ball was modified, False if there was no real contact.
    """
    face = _contact_face(ball, rod, player_idx, x_pre, y_pre)
    if face is None:
        return False
    axis, exit_sign = face
    if axis == 'y':
        r      = config.BALL_RADIUS
        hw     = rod.width / 2 + r
        side_y = rod.player_positions[player_idx] + exit_sign * hw
        if r <= side_y <= rod.field_width - r:
            ball.y  = side_y   # reposition on appropriate face
            slow    = config.CONTROLLED_SLOWDOWN if reflect else 1.0
            ball.vy = exit_sign * abs(ball.vy) * slow
            return True
        # The gap between the man and the rail is narrower than the ball, so it
        # can't leave out the side: squeeze it out the front or back instead
        ball.y    = max(r, min(rod.field_width - r, ball.y))
        exit_sign = 1.0 if x_pre - rod.x > 0 else -1.0
        axis      = 'x'
    if axis == 'x':
        ht = rod.thickness / 2 + config.BALL_RADIUS
        ball.x = rod.x + exit_sign * ht
        if reflect:
            ball.vx = exit_sign * abs(ball.vx) * config.CONTROLLED_SLOWDOWN
        else:
            ball.vx *= config.PASSIVE_SLOWDOWN
    return True


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_point(
    field:              Field,
    strategies:         dict[int, Strategy],
    kickoff_team:       int = 0,
    n_hands:            int = config.N_HANDS,
    record:             bool = False,
    seed:               Optional[int] = None,
    pos_grid:           Optional[np.ndarray] = None,
    goal_hits:          Optional[list] = None,
    stall_hits:         Optional[list] = None,
    dead_hits:          Optional[list] = None,
    collect_action_log: bool = False,
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

    # --- Action log setup ---
    _action_log: list[ActionLogEntry] = []
    # Rod labels: sort each team's rods from own goal outward → goa/def/mid/fwd
    _rod_labels: dict[int, str] = {}
    _role_names = ["goa", "def", "mid", "fwd"]
    for _team in (0, 1):
        _team_rods = sorted(
            [(i, r) for i, r in enumerate(field.rods) if r.team == _team],
            key=lambda ir: ir[1]._base_x if _team == 0 else -ir[1]._base_x,
        )
        for _rank, (_rod_idx, _) in enumerate(_team_rods):
            _role = _role_names[_rank] if _rank < len(_role_names) else str(_rank)
            _rod_labels[_rod_idx] = f"T{_team}-{_role}"

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
    last_hit:      Optional[tuple[int, int]]    = None
    last_hit_pos:  Optional[tuple[float, float]] = None  # ball pos of last active hit
    last_hit_team: Optional[int]                = None   # team of last active swing

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
        targets = strat.choose_pos(controlled, ball, field)
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

    def _expire_swing(rod_idx: int, rod) -> None:
        """Drop a finished swing; log a WHIFF if the ball slipped past it."""
        nonlocal last_hit
        s = rod.pending_swing
        rod.pending_swing = None
        if s.passed is None:
            return
        # Ball still passing under the man as it comes back down: it lands
        # behind the ball, so treat it as already touched instead of letting
        # the next rigid contact shove the ball back out the way it came.
        player_idx = rod.player_in_box(ball.x, ball.y, ball_radius=config.BALL_RADIUS)
        if player_idx is not None:
            last_hit = (rod_idx, player_idx)
        if collect_action_log:
            x, y, vx, vy = s.passed
            _action_log.append(ActionLogEntry(
                game_time    = game_time,
                team         = rod.team,
                rod_label    = _rod_labels[rod_idx],
                action       = 'WHIFF',
                ball_pos     = (x, y),
                intended_vel = (s.vx, s.vy),
                actual_vel   = (vx, vy),   # ball velocity when it passed
            ))

    # --- Tick loop ---
    for tick in range(config.MAX_TICKS):
        game_time = tick * dt
        event: Optional[str] = None

        # --------------------------------------------------------------
        # 1. Decrement timers, expire finished swings
        # --------------------------------------------------------------
        for ts in team_states.values():
            ts.tick(dt)
        for rod_idx, rod in enumerate(field.rods):
            if rod.pending_swing is not None and game_time > rod.pending_swing.over_at:
                _expire_swing(rod_idx, rod)
            rod.mark_pre()   # before strategy and movement move it this tick

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

            # Released rods stop immediately; discard any armed swing to prevent
            # a ghost hit firing after the player has let go. Clear any pending
            # switch so switch_timer cleanly means "a hand is mid-switch here".
            for rod_idx in released:
                rod = field.rods[rod_idx]
                rod.controlled = False
                rod.vy = 0.0
                rod.pending_swing = None
                rod.switch_timer = 0.0

            # Acquired rods get switch delay; uncontrolled until delay expires
            # so step_ball handles them as passive (uncontrolled) rods.
            for rod_idx in acquired:
                rod = field.rods[rod_idx]
                rod.controlled = False
                rod.switch_timer = config.SWITCH_DELAY

            # Kept rods: restore control only once switch delay has expired
            for rod_idx in (desired_hands & prev_hands):
                rod = field.rods[rod_idx]
                if rod.switch_timer <= 0:
                    rod.controlled = True

            ts.active_rods = desired_hands

            # Choose targets (only if not in reaction lockout)
            if not ts.reacting:
                controlled = [(i, field.rods[i]) for i in desired_hands
                             if field.rods[i].switch_timer <= 0]
                targets = strat.choose_pos(controlled, ball, field)
                for rod_idx, (ty, tx, up) in targets.items():
                    rod = field.rods[rod_idx]
                    rod.set_target_y(ty)
                    rod.set_x_offset(tx)
                    rod.up = up

                # Proactive swing commitment: each free controlled rod may arm a
                # swing now. After COMMIT_COOLDOWN has elapsed since the last
                # commit, a stale pending intent can be replaced with a fresh one.
                for rod_idx, rod in controlled:
                    if game_time - rod.last_commit_time >= config.COMMIT_COOLDOWN:
                        commit = strat.choose_hit(rod, ball, field, game_time)
                        if commit is not None:
                            if collect_action_log and rod.pending_swing is not None:
                                _action_log.append(ActionLogEntry(
                                    game_time    = game_time,
                                    team         = team,
                                    rod_label    = _rod_labels[rod_idx],
                                    action       = 'RETRACT',
                                    ball_pos     = (ball.x, ball.y),
                                    intended_vel = (rod.pending_swing.vx, rod.pending_swing.vy),
                                    actual_vel   = None,
                                ))
                            rod.pending_swing = commit
                            rod.last_commit_time = game_time
                            if collect_action_log:
                                _action_log.append(ActionLogEntry(
                                    game_time    = game_time,
                                    team         = team,
                                    rod_label    = _rod_labels[rod_idx],
                                    action       = 'COMMIT',
                                    ball_pos     = (ball.x, ball.y),
                                    intended_vel = (commit.vx, commit.vy),
                                    actual_vel   = None,
                                ))

            # If reacting: rods keep moving toward their previous target_y and
            # cannot arm new swings (a swing armed earlier still resolves).

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
        # 4-5. Move ball + resolve held-rod contact, in substeps
        #
        # Held-rod contact is checked after every CONTACT_STEP cm of ball
        # travel, not once per tick: at 10 FPS a fast shot moves further per
        # tick than a held player's reach box is deep, and would jump over it.
        # --------------------------------------------------------------
        elapsed = 0.0
        while elapsed < dt - 1e-9:
            h = dt - elapsed
            if ball.speed * h > config.CONTACT_STEP:
                h = config.CONTACT_STEP / ball.speed
            elapsed += h
            t_now = game_time + elapsed   # time at the end of this substep

            # --------------------------------------------------------------
            # 4. Move ball
            # --------------------------------------------------------------
            _vx_pre, _vy_pre = ball.vx, ball.vy
            _x_pre,  _y_pre  = ball.x,  ball.y

            ball_result = step_ball(ball, field, h,
                                    ball_radius=config.BALL_RADIUS,
                                    pos_grid=pos_grid)

            # A free man touching the ball (DEFLECT) is the last touch, like a
            # held-rod PASSIVE block, so an earlier HIT stops counting toward
            # an own goal. Checked outside the log block so the result doesn't
            # depend on logging.
            if ball.contacted_player:
                last_hit_team = None

            if collect_action_log:
                # Wall bounces and figurine contact are reported by the physics, so
                # the cause is read from real events rather than inferred from the
                # velocity delta (which is dominated by per-tick friction).
                _walls = [name for flag, name in (
                    (ball.bounce_left,   'L'),
                    (ball.bounce_right,  'R'),
                    (ball.bounce_top,    'T'),
                    (ball.bounce_bottom, 'B'),
                ) if flag]
                if abs(ball.vx) < 0.001 and abs(ball.vy) < 0.001 and (abs(_vx_pre) > 0.01 or abs(_vy_pre) > 0.01):
                    _cause: Optional[str] = 'STOP'
                elif _walls:
                    _cause = 'WALL_' + '_'.join(_walls)
                elif ball.contacted_player:
                    _cause = 'DEFLECT'
                else:
                    _cause = None
                if _cause is not None:
                    _reach = ','.join(
                        _rod_labels[_ri] for _ri, _rr in enumerate(field.rods)
                        if not _rr.up
                        and abs(ball.x - _rr._base_x) <= _rr.rod_x_reach + _rr.thickness / 2 + config.BALL_RADIUS
                    ) or '-'
                    _action_log.append(ActionLogEntry(
                        game_time    = game_time,
                        team         = -1,
                        rod_label    = _reach,
                        action       = _cause,
                        ball_pos     = (ball.x, ball.y),
                        intended_vel = (_vx_pre, _vy_pre),
                        actual_vel   = (ball.vx, ball.vy),
                    ))

            # Check for goal
            if ball_result.startswith('goal:'):
                winner = int(ball_result.split(':')[1])
                is_self_goal = last_hit_team is not None and last_hit_team != winner
                if goal_hits is not None and last_hit_pos is not None:
                    goal_team = 1 - winner  # team whose goal the ball entered
                    goal_hits.append((*last_hit_pos, goal_team, is_self_goal))
                if record:
                    _active = team_states[0].active_rods | team_states[1].active_rods
                    frames.append(_make_frame(tick, game_time, ball, field, f'goal:{winner}', _active))
                if collect_action_log:
                    _scored_on = 1 - winner
                    _suffix = " (own goal)" if is_self_goal else ""
                    _action_log.append(ActionLogEntry(
                        game_time    = game_time + dt,
                        team         = winner,
                        rod_label    = f"GOAL{_suffix}",
                        action       = f"T{winner} scores on T{_scored_on}",
                        ball_pos     = (ball.x, ball.y),
                        intended_vel = None,
                        actual_vel   = None,
                    ))
                return PointResult(
                    winner       = winner,
                    ticks        = tick + 1,
                    time         = game_time + dt,
                    frames       = frames,
                    action_log   = _action_log,
                    outcome      = 'goal',
                    is_self_goal = is_self_goal,
                )

            # --------------------------------------------------------------
            # 5. Detect overlaps + resolve any committed swing
            #
            # A controlled rod connects only if it has a pending swing whose active
            # window brackets this contact. During the backswing and follow-through
            # the man is lifted and the ball passes under it (a whiff unless it is
            # struck in the window); with no swing armed, or after the swing is
            # over, the held rod rigid-bounces the ball. Uncontrolled-rod contact
            # was already handled passively inside step_ball.
            # --------------------------------------------------------------
            overlaps = find_overlapping_players(ball, field)

            for rod_idx, player_idx in overlaps:
                rod  = field.rods[rod_idx]
                team = rod.team

                if rod.pending_swing is not None and t_now > rod.pending_swing.over_at:
                    _expire_swing(rod_idx, rod)   # finished mid-tick (may set last_hit)

                # Double-hit check: same player can't hit twice in a row
                # unless ball has stopped (which resets last_hit)
                if last_hit == (rod_idx, player_idx):
                    continue

                s = rod.pending_swing

                if s is not None and s.active_start <= t_now <= s.window_end:
                    # Swing connects — replace ball velocity with the committed swing vector.
                    ball.vx = s.vx
                    ball.vy = s.vy

                    # Clamp to max speed
                    speed = ball.speed
                    if speed > config.BALL_MAX_SPEED:
                        factor = config.BALL_MAX_SPEED / speed
                        ball.vx *= factor
                        ball.vy *= factor

                    if collect_action_log:
                        _action_log.append(ActionLogEntry(
                            game_time    = game_time,
                            team         = team,
                            rod_label    = _rod_labels[rod_idx],
                            action       = 'HIT',
                            ball_pos     = (ball.x, ball.y),
                            intended_vel = (s.vx, s.vy),
                            actual_vel   = (ball.vx, ball.vy),
                        ))

                    rod.pending_swing = None

                    # Set opponent's reaction timer
                    opp_team = 1 - team
                    team_states[opp_team].reaction_timer = config.REACTION_TIME

                    # Update last-hit tracking
                    last_hit      = (rod_idx, player_idx)
                    last_hit_pos  = (ball.x, ball.y)
                    last_hit_team = team
                    event = 'hit'

                    # Only one hit per substep (first overlap wins)
                    break

                elif rod.controlled and rod.switch_timer <= 0:
                    if s is not None:
                        # Backswing or follow-through: the man is lifted, so the
                        # ball passes under it untouched (not a touch:
                        # last_hit_team is left alone).
                        # The swing stays armed: a slow ball still in reach when
                        # the window opens is struck, one that has left by then
                        # is a whiff, logged when the swing expires.
                        if (s.passed is None and
                                _contact_face(ball, rod, player_idx, _x_pre, _y_pre) is not None):
                            s.passed = (ball.x, ball.y, ball.vx, ball.vy)
                        continue

                    # No swing armed: bounce off the rigid player (no pushback).
                    # That block becomes the last touch, but it isn't a shot, so
                    # an earlier HIT stops counting toward an own goal.
                    if not _resolve_rigid_contact(ball, rod, player_idx,
                                                  _x_pre, _y_pre, reflect=True):
                        continue
                    last_hit_team = None
                    if collect_action_log:
                        _action_log.append(ActionLogEntry(
                            game_time    = game_time,
                            team         = team,
                            rod_label    = _rod_labels[rod_idx],
                            action       = 'PASSIVE',
                            ball_pos     = (ball.x, ball.y),
                            intended_vel = None,
                            actual_vel   = None,
                        ))
                    # A block locks opponent's reaction timer
                    opp_team = 1 - team
                    team_states[opp_team].reaction_timer = config.REACTION_TIME
                    last_hit = (rod_idx, player_idx)
                    break

            # Rods only move once per tick (above), so later substeps start
            # where this one ended
            for rod in field.rods:
                rod.mark_pre()

            if ball_result == 'stopped':
                break

        # --- Anti-stalling checks ---
        teams_in_reach = _teams_with_reach(ball, field)

        if ball.stopped and not teams_in_reach:
            if dead_hits is not None:
                dead_hits.append((ball.x, ball.y))
            return PointResult(winner=None, ticks=tick + 1, time=game_time + dt,
                               frames=frames, action_log=_action_log, outcome='dead_ball')

        if len(teams_in_reach) == 1:
            possessing = next(iter(teams_in_reach))
            possession_timer[possessing] += dt
            possession_timer[1 - possessing] = 0.0
            if possession_timer[possessing] >= config.POSSESSION_LIMIT:
                if stall_hits is not None:
                    stall_hits.append((ball.x, ball.y, possessing))
                return PointResult(winner=1 - possessing, ticks=tick + 1, time=game_time + dt,
                                   frames=frames, action_log=_action_log, outcome='stallout')
        else:
            possession_timer[0] = possession_timer[1] = 0.0

        # Reset last_hit if ball stopped
        if ball.stopped:
            last_hit      = None
            last_hit_team = None

        # --------------------------------------------------------------
        # 6. Record frame
        # --------------------------------------------------------------
        if record:
            _active = team_states[0].active_rods | team_states[1].active_rods
            frames.append(_make_frame(tick, game_time, ball, field, event, _active))

    # --- Time limit reached ---
    return PointResult(
        winner     = None,
        ticks      = config.MAX_TICKS,
        time       = config.MAX_GAME_TIME,
        frames     = frames,
        action_log = _action_log,
        outcome    = 'timeout',
    )


def _make_frame(
    tick: int,
    time: float,
    ball: BallState,
    field: Field,
    event: Optional[str],
    active_rod_idxs: set[int] = frozenset(),
) -> Frame:
    return Frame(
        tick       = tick,
        time       = time,
        ball_x     = ball.x,
        ball_y     = ball.y,
        ball_vx    = ball.vx,
        ball_vy    = ball.vy,
        rod_ys     = [r.y_offset for r in field.rods],
        rod_xs     = [r.x_offset for r in field.rods],
        rod_ctrl   = [r.controlled for r in field.rods],
        rod_switching = [(i in active_rod_idxs) and not r.controlled
                         for i, r in enumerate(field.rods)],
        ups        = [r.up for r in field.rods],
        event      = event,
        rod_swings = [
            (r.pending_swing.active_start, r.pending_swing.window_end, r.pending_swing.vx)
            if r.pending_swing is not None else None
            for r in field.rods
        ],
    )
