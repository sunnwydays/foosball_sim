"""debug_snap.py -- trace held-rod "snaps": the ball teleports onto a player's face.

Usage
-----
  python debug_scripts/debug_snap.py --manual [--gap G] [--xo X] [--dy DY] [--vx VX] [--pre P]
  python debug_scripts/debug_snap.py SEED [--t0 S] [--t1 S] [--all] [--nth N] [--ticks K]

Manual mode (--manual)
----------------------
Builds balls next to the T0 mid rod, on both its +x and -x faces, and runs the
same box check + _resolve_rigid_contact as the tick loop.
By default it sweeps the gap 0..7 cm (0.5 steps); --gap G runs a single one.
Gap is the distance in cm between the ball's surface and the player's face
(0 = touching, > 0 = still in the air). The wide reach box lets the ball in
from about 3 cm further out than the slab itself (more on the far side of a
tilted rod: --xo), so gap=2 is inside the box but NOT touching the player.
--dy puts the ball off the figure's centre (side-face tests).
--pre is how far out the ball started this substep. To test a ball that
tunnelled right through the slab, use a big --pre with a negative --gap
(e.g. --gap -5.5 --pre 8).

Replay mode (SEED)
------------------
Runs the whole point as play.py does and traces every rigid-contact call, one
line each, in the window --t0..--t1 seconds (default 0..3). By default only
snaps are printed; --all prints every call.

It then picks one snap (--nth, default the first in the window) and:
  1. prints the recorded frames around it (--ticks frames after),
  2. replays that single call on a fresh Field (replay_call) to show it needs
     nothing else,
  3. moves the ball away from the rod in 0.5 cm steps and reports how far out
     that same state would still snap (the width of the catch zone).

Columns
-------
  pen_x, pen_y  how deep the ball overlaps the slab (<= 0: not touching)
  x             ball.x before -> after the call
  moved         how far the call moved the ball, in cm
  SNAP          printed when the ball was NOT touching and still got moved

Verdict
-------
Manual mode ends with PASS / FAIL. FAIL = the ball moved with no real overlap.
"""

import argparse

import _path  # noqa: F401  (project root on sys.path)
import play  # applies play.py's FPS to config before the sim is imported
import config
import simulation
from field import Field, BallState
from physics import find_overlapping_players
from debug_face_slide import rod_labels
from detectors import SNAP_TOL
from play import run_point, TEAM_0, TEAM_1, TEAM_0_SKILL, TEAM_1_SKILL


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class Tracer:
    """
    Temporarily wraps simulation._resolve_rigid_contact and records one dict per
    call that moved the ball. simulation.py itself is not modified (the tick loop
    looks the function up by name on every call, so the patch takes effect).
    """

    def __init__(self):
        self.calls: list[dict] = []
        self._orig = simulation._resolve_rigid_contact

    def __enter__(self):
        simulation._resolve_rigid_contact = self._wrapped
        return self

    def __exit__(self, *exc):
        simulation._resolve_rigid_contact = self._orig

    def _wrapped(self, ball, rod, player_idx, x_pre, y_pre, reflect):
        x0, y0, vx0, vy0 = ball.x, ball.y, ball.vx, ball.vy
        py = rod.player_positions[player_idx]
        pen_x = rod.thickness / 2 + config.BALL_RADIUS - abs(x0 - rod.x)
        pen_y = rod.width / 2 + config.BALL_RADIUS - abs(y0 - py)

        changed = self._orig(ball, rod, player_idx, x_pre, y_pre, reflect)

        if changed:
            self.calls.append(dict(
                rod=rod, rod_x=rod.x, pen_x=pen_x, pen_y=pen_y,
                # exact inputs, so the call can be replayed on a fresh Field
                player_idx=player_idx, x_pre=x_pre, y_pre=y_pre,
                x_offset=rod.x_offset, y_offset=rod.y_offset, up=rod.up,
                x0=x0, y0=y0, x1=ball.x, y1=ball.y,
                vx0=vx0, vx1=ball.vx, vy0=vy0, vy1=ball.vy,
                moved=((ball.x - x0) ** 2 + (ball.y - y0) ** 2) ** 0.5,
                snap=(pen_x <= 0 or pen_y <= 0),
            ))
        return changed


HEADER = (f"{'time':>6} {'rod':<8} {'rod.x':>6} {'pen_x':>6} {'pen_y':>6} "
          f"{'x':>15} {'vx':>15} {'moved':>6}")


def print_call(c: dict, label: str, t: float | None = None) -> None:
    ts = f"{t:6.2f}" if t is not None else f"{'-':>6}"
    print(f"{ts} {label:<8} {c['rod_x']:6.2f} {c['pen_x']:6.2f} {c['pen_y']:6.2f} "
          f"{c['x0']:6.2f}->{c['x1']:6.2f} {c['vx0']:6.1f}->{c['vx1']:6.1f} "
          f"{c['moved']:6.2f}{'  SNAP' if c['snap'] else ''}")


# ---------------------------------------------------------------------------
# Manual mode: one hand-built contact
# ---------------------------------------------------------------------------

def contact(field: Field, idx: int, args, gap: float, side: int):
    """
    One hand-built contact on player 0 of rod `idx`.

    side=+1: ball on the +x side of the rod, moving toward it (vx < 0).
    side=-1: the mirror image. gap = cm between the ball's surface and the face
    (negative = the ball centre is already past the face).

    Mirrors the tick loop: the resolver is only called if the ball is inside the
    wide reach box (find_overlapping_players).
    Returns (in_box, call): call is the Tracer dict, or None if the box missed or
    the resolver left the ball alone.
    """
    rod = field.rods[idx]
    ht = rod.thickness / 2 + config.BALL_RADIUS
    py = rod.player_positions[0]
    ball = BallState(x=rod.x + side * (ht + gap), y=py + args.dy,
                     vx=-side * abs(args.vx), vy=0.0)

    # The resolver decides which face the ball "came from" using x_pre, the ball's
    # x at the START of the substep. The ball is moving toward the rod, so it
    # started on the side it is on now, args.pre cm further out.
    x_pre = ball.x + side * args.pre

    if (idx, 0) not in find_overlapping_players(ball, field):
        return False, None
    with Tracer() as tr:
        simulation._resolve_rigid_contact(ball, rod, 0, x_pre, ball.y, reflect=True)
    return True, (tr.calls[0] if tr.calls else None)


def manual(args) -> bool:
    field = Field()
    labels = rod_labels(field)
    idx = next(i for i, l in labels.items() if l == "T0-mid")
    rod = field.rods[idx]
    rod.set_x_offset(args.xo)
    rod.up = False

    ht = rod.thickness / 2 + config.BALL_RADIUS
    # --gap G runs one contact; otherwise sweep 0 .. 7 cm in 0.5 cm steps
    gaps = [args.gap] if args.gap is not None else [i * 0.5 for i in range(15)]
    print(f"manual: T0-mid base_x={rod._base_x:.2f}  x_offset={args.xo:+.1f}  "
          f"rod.x={rod.x:.2f}  half-thickness+r={ht:.2f}  dy={args.dy}  pre={args.pre}")

    ok = True
    for side in (+1, -1):
        print(f"\nball on the {'+x' if side > 0 else '-x'} side, face at x={rod.x + side * ht:.2f}")
        print(f"{'gap':>5} {'in box':>6} {'pen_x':>6} {'pen_y':>6} {'moved':>6}")
        last_snap = None                        # largest gap that still snapped
        for gap in gaps:
            in_box, c = contact(field, idx, args, gap, side)
            if c is None:
                print(f"{gap:5.1f} {str(in_box):>6} {'-':>6} {'-':>6} {'-':>6}")
                continue
            print(f"{gap:5.1f} {str(in_box):>6} {c['pen_x']:6.2f} {c['pen_y']:6.2f} "
                  f"{c['moved']:6.2f}{'  SNAP' if c['snap'] else ''}")
            if c["snap"]:
                last_snap = gap

        # Where the wide box ends, as a gap from the slab face: the box is centred
        # on _base_x with half-width rod_x_reach + thickness/2 + r, the face sits at
        # rod.x = _base_x + x_offset, and the ball centre is at face + side*(ht + gap).
        expected = rod.rod_x_reach - side * args.xo
        print(f"-> last snapping gap: {last_snap}   box edge predicted at gap {expected:.2f}")
        ok = ok and last_snap is None

    print(f"\n-> {'PASS' if ok else 'FAIL'}: "
          + ("ball left alone or really touching" if ok else "ball moved without touching the player"))
    return ok


# ---------------------------------------------------------------------------
# Replay mode: trace a whole recorded point
# ---------------------------------------------------------------------------

def replay_call(c: dict, rod_idx: int, extra_gap: float = 0.0):
    """
    Re-run one recorded _resolve_rigid_contact call on a fresh Field.

    Rebuilds the one rod (offsets, up) and the ball exactly as they were just
    before the call, then calls the function again. extra_gap moves the ball
    that many cm further from the rod, along x, before the call.
    Like the tick loop, it only calls _resolve_rigid_contact if the ball is
    inside the wide reach box (find_overlapping_players); otherwise changed=False.
    Returns (ball, changed).
    """
    field = Field()
    rod = field.rods[rod_idx]
    rod.set_offset(c["y_offset"])
    rod.set_x_offset(c["x_offset"])
    rod.up = c["up"]
    away = 1.0 if c["x0"] - rod.x > 0 else -1.0
    ball = BallState(c["x0"] + away * extra_gap, c["y0"], c["vx0"], c["vy0"])
    if (rod_idx, c["player_idx"]) not in find_overlapping_players(ball, field):
        return ball, False
    changed = simulation._resolve_rigid_contact(
        ball, rod, c["player_idx"], c["x_pre"] + away * extra_gap, c["y_pre"], reflect=True)
    return ball, changed


def from_seed(args) -> None:
    with Tracer() as tr:
        field, result, _ = run_point(args.seed, TEAM_0, TEAM_1, TEAM_0_SKILL, TEAM_1_SKILL)
    labels = rod_labels(field)

    # The k-th call that moved the ball produced the k-th PASSIVE log entry
    # (both happen in the same place in simulation.py; whiffs pass through and
    # never call it), so the log supplies the time.
    bounces = [e for e in result.action_log if e.action == "PASSIVE"]
    n_snaps = sum(c["snap"] for c in tr.calls)
    print(f"seed {args.seed}: {len(tr.calls)} rigid contacts, {n_snaps} snaps in the whole point, "
          f"outcome={result.outcome}")
    print(f"showing the window t={args.t0}..{args.t1} s (change with --t0 / --t1)")
    print(HEADER)
    for e, c in zip(bounces, tr.calls):
        if not (args.t0 <= e.game_time <= args.t1):
            continue
        if c["snap"] or args.all:
            print_call(c, labels[field.rods.index(c["rod"])], e.game_time)

    # --- pick one snap (the --nth snap inside the --t0..--t1 window) ---
    snaps = [(e, c) for e, c in zip(bounces, tr.calls)
             if c["snap"] and args.t0 <= e.game_time <= args.t1]
    if not snaps:
        print(f"\nno snaps in the window t={args.t0}..{args.t1} s ({n_snaps} in the whole point)")
        return
    e, c = snaps[min(args.nth, len(snaps) - 1)]
    rod_idx = field.rods.index(c["rod"])
    label = labels[rod_idx]
    ht = c["rod"].thickness / 2 + config.BALL_RADIUS
    gap = abs(c["x0"] - c["rod_x"]) - ht   # ball-center distance beyond the face
    print(f"\ninspecting snap #{min(args.nth, len(snaps) - 1)} of {len(snaps)} in the window "
          f"({n_snaps} in the whole point): {label} at t={e.game_time:.3f}s  "
          f"x_offset={c['x_offset']:+.2f}  gap to face={gap:.2f} cm")

    # --- 1. the recorded game around the snap ---
    # Frames are saved at the END of each tick, so the first frame with
    # time >= the log time is the tick that contains the snap.
    frames = result.frames
    k = next((i for i, f in enumerate(frames) if f.time >= e.game_time - 1e-9), len(frames) - 1)
    print(f"\nrecorded frames ({label}, rod {rod_idx})   * = tick containing the snap")
    print(f"{'tick':>4} {'t':>6} {'ball x':>7} {'y':>6} {'vx':>7} {'vy':>7} "
          f"{'xo':>6} {'yo':>6} {'up':>5} {'ctrl':>5}  event")
    for i in range(max(0, k - 2), min(len(frames), k + 1 + args.ticks)):
        f = frames[i]
        print(f"{'*' if i == k else ' '}{f.tick:3d} {f.time:6.3f} {f.ball_x:7.2f} {f.ball_y:6.2f} "
              f"{f.ball_vx:7.1f} {f.ball_vy:7.1f} {f.rod_xs[rod_idx]:6.2f} "
              f"{f.rod_ys[rod_idx]:6.2f} {str(f.ups[rod_idx]):>5} "
              f"{str(f.rod_ctrl[rod_idx]):>5}  {f.event or ''}")

    # --- 2. replay JUST this call on a fresh Field ---
    ball, changed = replay_call(c, rod_idx)
    same = abs(ball.x - c["x1"]) < 1e-9 and abs(ball.vx - c["vx1"]) < 1e-9
    print(f"\nisolated replay of the call: changed={changed}  "
          f"x {c['x0']:.2f} -> {ball.x:.2f}  vx {c['vx0']:.1f} -> {ball.vx:.1f}  "
          f"{'(matches the recorded call)' if same else '(DIFFERS from the recorded call!)'}")
    print("-> the snap needs nothing but this one call: no strategy, no other rods, no RNG."
          if same else "-> something outside this call mattered; check replay_call()'s setup.")

    # --- 3. how far out would THIS state still have snapped? ---
    # replay_call shifts the ball away from the rod by `extra_gap`, and only calls
    # the resolver if the shifted ball is still inside the wide reach box.
    print(f"\n{'extra_gap':>9} {'ball gap':>8} {'changed':>7} {'moved':>6}")
    away = 1.0 if c["x0"] - c["rod_x"] > 0 else -1.0
    last_snap_gap = None                     # largest gap that still moved the ball
    for i in range(0, 21):                   # 0 .. 10 cm in 0.5 cm steps
        g = i * 0.5                          # extra_gap for this step, in cm
        ball, changed = replay_call(c, rod_idx, extra_gap=g)

        # How far did the call move the ball? Compare against the pre-call
        # position, which is c["x0"] shifted away from the rod by g.
        moved = abs(ball.x - (c["x0"] + away * g))

        print(f"{g:9.2f} {gap + g:8.2f} {str(changed):>7} {moved:6.2f}")
        if changed and moved > SNAP_TOL:
            last_snap_gap = gap + g          # the ball's gap to the face at this step

    if last_snap_gap is None:
        print("\nthis state no longer snaps at any distance")
    else:
        print(f"\nthis state snaps from as far as {last_snap_gap:.2f} cm from the face")
    # Compare with ROD_X_REACH + |x_offset| on the tilted-away side (the sweep
    # table). After your fix this should be ~0 (or None).



def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seed", type=int, nargs="?", help="seed to replay (as in play.py)")
    p.add_argument("--manual", action="store_true", help="one hand-built contact")
    p.add_argument("--gap", type=float, default=None, help="manual: one gap in cm (default: sweep 0..7)")
    p.add_argument("--pre", type=float, default=0.5, help="manual: how far out the ball started this substep, cm")
    p.add_argument("--xo", type=float, default=0.0, help="manual: rod x_offset")
    p.add_argument("--dy", type=float, default=0.0, help="manual: cm off figure center")
    p.add_argument("--vx", type=float, default=-100.0, help="manual: ball vx (default -100)")
    p.add_argument("--t0", type=float, default=0.0, help="replay: window start (s)")
    p.add_argument("--t1", type=float, default=3.0, help="replay: window end (s)")
    p.add_argument("--all", action="store_true", help="replay: print every contact, not just snaps")
    p.add_argument("--nth", type=int, default=0, help="replay: which snap in the window to inspect (default 0 = first)")
    p.add_argument("--ticks", type=int, default=3, help="replay: recorded frames to show after the snap")
    args = p.parse_args()

    if args.manual:
        manual(args)
    elif args.seed is not None:
        from_seed(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
