"""
simulation.py — Time-stepped foosball simulation engine.

simulate_point() runs one point as a sequence of ticks (dt = 1/FPS).

Tick order
----------
1. Decrement timers (reaction, switch)
2. Strategy: choose_hands → choose_pos (respecting reaction lockout)
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
    Y is excluded — a rod can always slide to cover any ball y position."""
    teams: set[int] = set()
    for rod in field.rods:
        if rod.up:
            continue
        if abs(ball.x - rod._base_x) <= rod.rod_x_reach + rod.thickness / 2 + config.BALL_RADIUS:
            teams.add(rod.team)
    return teams


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
    is always ejected through the face on the side it CAME FROM this tick
    (never the far side) and its velocity is set to point away from that face,
    so a bounce can never send the ball back into the player. A ball that is
    outside the slab and already separating from the rod is left untouched.

    reflect=True  : rigid bounce at CONTROLLED_SLOWDOWN (whiff / no swing armed)
    reflect=False : absorbed bounce at PASSIVE_SLOWDOWN (rod mid-switch)

    Returns True if the ball was modified, False if there was no real contact.
    """
    r  = config.BALL_RADIUS
    py = rod.player_positions[player_idx]
    ht = rod.thickness / 2 + r
    hw = rod.width / 2 + r
    dx = ball.x - rod.x
    dy = ball.y - py
    pen_x = ht - abs(dx)
    pen_y = hw - abs(dy)

    if pen_x >= pen_y:
        # Side face (y)
        exit_sign = 1.0 if y_pre - py > 0 else -1.0
        crossed   = (exit_sign > 0) != (dy > 0)
        if not crossed and pen_y <= 0 and ball.vy * exit_sign > 0:
            return False
        ball.y  = py + exit_sign * hw
        slow    = config.CONTROLLED_SLOWDOWN if reflect else 1.0
        ball.vy = exit_sign * abs(ball.vy) * slow
    else:
        # Front face (x)
        exit_sign = 1.0 if x_pre - rod.x > 0 else -1.0
        crossed   = (exit_sign > 0) != (dx > 0)
        if not crossed and pen_x <= 0 and ball.vx * exit_sign > 0:
            return False
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
        # 4. Move ball
        # --------------------------------------------------------------
        _vx_pre, _vy_pre = ball.vx, ball.vy
        _x_pre,  _y_pre  = ball.x,  ball.y

        ball_result = step_ball(ball, field, dt,
                                ball_radius=config.BALL_RADIUS,
                                pos_grid=pos_grid)

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
                    action       = f"Team {winner} scores on Team {_scored_on}",
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
        # window brackets this contact. Otherwise (whiff, or no swing armed) the
        # held rod rigid-bounces the ball. Uncontrolled-rod contact was already
        # handled passively inside step_ball.
        # --------------------------------------------------------------
        overlaps = find_overlapping_players(ball, field)

        for rod_idx, player_idx in overlaps:
            rod  = field.rods[rod_idx]
            team = rod.team

            # Double-hit check: same player can't hit twice in a row
            # unless ball has stopped (which resets last_hit)
            if last_hit == (rod_idx, player_idx):
                continue

            s = rod.pending_swing

            if s is not None and s.active_start <= game_time <= s.window_end:
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

                # Only one hit per tick (first overlap wins)
                break

            elif rod.controlled and rod.switch_timer <= 0:
                # Whiff (mistimed swing) or no swing armed on a settled rod:
                # bounce the ball off rigid players (no pushback)
                if not _resolve_rigid_contact(ball, rod, player_idx,
                                              _x_pre, _y_pre, reflect=True):
                    continue
                _had_swing = s is not None
                _intended  = (s.vx, s.vy) if _had_swing else None
                rod.pending_swing = None
                if collect_action_log:
                    _action_log.append(ActionLogEntry(
                        game_time    = game_time,
                        team         = team,
                        rod_label    = _rod_labels[rod_idx],
                        action       = 'WHIFF' if _had_swing else 'PASSIVE',
                        ball_pos     = (ball.x, ball.y),
                        intended_vel = _intended,
                        actual_vel   = None,
                    ))
                # NOTE: per approved plan, a whiff bounce also locks the opponent's
                # reaction timer. Revisit if this proves to give the whiffer an
                # unintended tempo advantage.
                opp_team = 1 - team
                team_states[opp_team].reaction_timer = config.REACTION_TIME
                last_hit = (rod_idx, player_idx)
                break


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
