"""find_seeds.py -- find replay seeds that show suspicious ball-player contact.

Uses play.py's settings (FPS, teams, skills, kickoff), so every seed printed
here replays identically with:  python play.py <seed>

Usage:  python find_seeds.py [n_seeds]

Detectors
---------
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
import sys

import _path  # noqa: F401  (project root on sys.path)
import play # applies play.py's FPS to config before the sim is imported
import config
import simulation
from play import run_point, TEAM_0, TEAM_1, TEAM_0_SKILL, TEAM_1_SKILL

SLIDE_TICKS = 3       # consecutive passive-contact ticks that count as a slide
SLIDE_TO_GOAL = 1.0   # sec - slide must be followed by a goal this soon
SNAP_TOL = 0.1        # cm - a bounce with no real overlap must move the ball more than this


def back_bounces(result) -> list[tuple[float, str]]:
    """(time, rod_label) for each rigid bounce off a player hit from behind."""
    out = []
    for e in result.action_log:
        if e.action not in ("WHIFF", "PASSIVE"):
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
    the same order as the WHIFF/PASSIVE log entries).

    A snap is a move with no real overlap: the wide reach box let the ball in,
    but the ball was still outside the slab (pen <= 0) when it got teleported.
    """
    calls: list[tuple[float, bool]] = []
    orig = simulation._resolve_rigid_contact

    def wrapped(ball, rod, player_idx, x_pre, y_pre, reflect):
        x0, y0 = ball.x, ball.y
        py = rod.player_positions[player_idx]
        pen_x = rod.thickness / 2 + config.BALL_RADIUS - abs(x0 - rod.x)
        pen_y = rod.width / 2 + config.BALL_RADIUS - abs(y0 - py)
        changed = orig(ball, rod, player_idx, x_pre, y_pre, reflect)
        if changed:
            moved = ((ball.x - x0) ** 2 + (ball.y - y0) ** 2) ** 0.5
            calls.append((moved, pen_x <= 0 or pen_y <= 0))
        return changed

    simulation._resolve_rigid_contact = wrapped
    try:
        yield calls
    finally:
        simulation._resolve_rigid_contact = orig


def snaps(result, calls) -> list[tuple[float, str, float]]:
    """(time, rod_label, cm moved) for each bounce that teleported the ball."""
    bounces = [e for e in result.action_log if e.action in ("WHIFF", "PASSIVE")]
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


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    print(f"{TEAM_0} vs {TEAM_1}  at {config.FPS} FPS, seeds 0..{n - 1}")
    counts = {"BACK-BOUNCE": 0, "FACE-SLIDE": 0, "SNAP": 0}
    hits = 0
    for seed in range(n):
        with track_rigid_contacts() as calls:
            _, result, _ = run_point(seed, TEAM_0, TEAM_1, TEAM_0_SKILL, TEAM_1_SKILL)
        hits += sum(1 for e in result.action_log if e.action == "HIT")
        found = [("BACK-BOUNCE", t, l) for t, l in back_bounces(result)]
        found += [("FACE-SLIDE", t, l) for t, l in face_slides(result)]
        found += [("SNAP", t, f"{l} {m:.1f}cm") for t, l, m in snaps(result, calls)]
        for kind, t, label in found:
            counts[kind] += 1
            print(f"seed={seed:4d}  t={t:5.2f}s  {kind:<11}  {label:<8}  "
                  f"winner={result.winner}  outcome={result.outcome}")
    print("totals: " + "  ".join(f"{k}={v}" for k, v in counts.items())
          + f"  (HITs={hits}, regression check)")


if __name__ == "__main__":
    main()
