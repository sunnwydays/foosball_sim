"""detectors.py -- detectors for suspicious ball-player contact in a played point.

Shared by find_seeds.py and the test suite's seed sweep. Must not import play:
play.py rewrites config.FPS on import and loads matplotlib, which would make
the tests depend on play.py's (and play_local.py's) settings.

Each detector takes a PointResult played with record=True and
collect_action_log=True.

BACK-BOUNCE  A held rod rigid-bounces a ball that was moving toward the
             opponent's goal, i.e. it hit the player from behind, and sends it
             back toward its own goal (the whiff-from-behind issue).
FACE-SLIDE   The ball stays in contact with a passive player for several
             ticks in a row, then the team owning that rod concedes within
             1 s (the ball sliding along a player's face into the net).
SNAP         A held-rod rigid bounce moves a ball that was NOT touching the
             player's slab (only inside the wide reach box): it is teleported
             onto the face. Detected by wrapping _resolve_rigid_contact.
"""

import contextlib

import config
import simulation

SLIDE_TICKS = 3       # consecutive passive-contact ticks that count as a slide
SLIDE_TO_GOAL = 1.0   # sec - slide must be followed by a goal this soon
SNAP_TOL = 0.1        # cm - a bounce with no real overlap must move the ball more than this


def back_bounces(result) -> list[tuple[float, str]]:
    """(time, rod_label) for each rigid bounce off a player hit from behind."""
    out = []
    for e in result.action_log:
        if e.action != "PASSIVE":   # whiffs pass through, they never bounce
            continue
        tick = round(e.game_time / config.DT)
        if tick < 1 or tick >= len(result.frames):
            continue
        attack_dir = 1 if e.team == 0 else -1
        vx_before = result.frames[tick - 1].ball_vx
        vx_after  = result.frames[tick].ball_vx
        if vx_before * attack_dir > config.STOP_THRESHOLD and vx_after * attack_dir < 0:
            out.append((e.game_time, e.rod_label))
    return out


@contextlib.contextmanager
def track_rigid_contacts():
    """
    Wrap simulation._resolve_rigid_contact and record, for every call that
    moved the ball, how far it moved and whether the ball was actually touching
    the player's slab beforehand. Yields the list (one entry per moved call, in
    the same order as the PASSIVE log entries; whiffs pass through and never
    call it).

    A snap is a move with no real overlap: the wide reach box let the ball in,
    but the ball was still outside the slab (pen <= 0) when it got teleported.
    A ball that crossed the player's centre line this substep, measured from
    the rod (which can jump up to 2 * ROD_X_REACH when a grabbed rod drops its
    tilt), did meet the player even if it is outside the slab now: not a snap.
    """
    calls: list[tuple[float, bool]] = []
    orig = simulation._resolve_rigid_contact

    def wrapped(ball, rod, player_idx, x_pre, y_pre, reflect):
        x0, y0 = ball.x, ball.y
        py = rod.player_positions[player_idx]
        pen_x = rod.thickness / 2 + config.BALL_RADIUS - abs(x0 - rod.x)
        pen_y = rod.width / 2 + config.BALL_RADIUS - abs(y0 - py)
        rx_pre = rod.x if rod.x_pre is None else rod.x_pre
        py_pre = py if rod.y_offset_pre is None else py - rod.y_offset + rod.y_offset_pre
        crossed_x = pen_y > 0 and (x_pre - rx_pre > 0) != (x0 - rod.x > 0)
        crossed_y = pen_x > 0 and (y_pre - py_pre > 0) != (y0 - py > 0)
        touching = (pen_x > 0 and pen_y > 0) or crossed_x or crossed_y
        changed = orig(ball, rod, player_idx, x_pre, y_pre, reflect)
        if changed:
            moved = ((ball.x - x0) ** 2 + (ball.y - y0) ** 2) ** 0.5
            calls.append((moved, not touching))
        return changed

    simulation._resolve_rigid_contact = wrapped
    try:
        yield calls
    finally:
        simulation._resolve_rigid_contact = orig


def snaps(result, calls) -> list[tuple[float, str, float]]:
    """(time, rod_label, cm moved) for each bounce that teleported the ball."""
    bounces = [e for e in result.action_log if e.action == "PASSIVE"]
    return [(e.game_time, e.rod_label, moved)
            for e, (moved, no_overlap) in zip(bounces, calls)
            if no_overlap and moved > SNAP_TOL]


def face_slides(result) -> list[tuple[float, str]]:
    """(slide start time, rod_label) for passive-contact runs that end in a goal."""
    goal = next((e for e in result.action_log if e.rod_label.startswith("GOAL")), None)
    if goal is None:
        return []
    conceding = f"T{1 - goal.team}-"

    out = []
    run_label, run_start, run_len, last_tick = None, 0.0, 0, -2
    for e in result.action_log:
        if e.action != "DEFLECT":
            continue
        tick = round(e.game_time / config.DT)
        labels = [l for l in e.rod_label.split(",") if l.startswith(conceding)]
        label = labels[0] if labels else None
        if label == run_label and tick == last_tick:
            continue  # another substep of the same tick
        if label is not None and label == run_label and tick == last_tick + 1:
            run_len += 1
        else:
            run_label, run_start, run_len = label, e.game_time, 1
        last_tick = tick
        if (run_label is not None and run_len == SLIDE_TICKS
                and goal.game_time - e.game_time <= SLIDE_TO_GOAL):
            out.append((run_start, run_label))
    return out
