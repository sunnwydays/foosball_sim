"""debug_face_slide.py -- replay a face slide in isolation and trace every contact.

Usage
-----
  python debug_face_slide.py SEED [--tick N] [--ticks N]
  python debug_face_slide.py --manual [--xo X] [--dy DY] [--vx VX] [--ticks N]

Replay mode (SEED)
------------------
1. CAPTURE  Run the point, find its first FACE-SLIDE (find_seeds' detector)
            and print the recorded frames around it.
2. ISOLATE  Copy the ball and the sliding rod from the frame just before the
            slide (or --tick N) into a fresh Field. Every other rod is lifted,
            so that one rod is the only thing the ball can touch.
3. REPLAY   Run it twice:
              physics only  -- nothing touches the rod between ticks
              + strategy    -- the rod's strategy re-applies choose_passive()
                               every tick, the way simulate_point() does
4. TRACE    Every substep with player contact prints one line.

Manual mode (--manual)
----------------------
Same replay, but the starting state is hand-built: a ball 10 cm in front of
the T1 goalie, --dy cm off a figure's center, moving toward the goal at --vx.

Verdict
-------
Each run ends with PASS / FAIL. FAIL means the ball hit the figure's front
face on SLIDE_TICKS ticks in a row -- the face-slide fingerprint.
"""

import argparse

import play  # applies play.py's FPS to config before the sim is imported
import config
import physics
from field import Field, BallState
from find_seeds import face_slides, SLIDE_TICKS
from play import run_point, build_strategy, TEAM_0, TEAM_1, TEAM_0_SKILL, TEAM_1_SKILL

ROLE_NAMES = ["goa", "def", "mid", "fwd"]


def rod_labels(field: Field) -> dict[int, str]:
    """rod index -> 'T1-goa' etc., same ordering as simulate_point()."""
    labels = {}
    for team in (0, 1):
        rods = sorted(field.rods_for_team(team),
                      key=lambda ir: ir[1]._base_x if team == 0 else -ir[1]._base_x)
        for rank, (i, _) in enumerate(rods):
            labels[i] = f"T{team}-{ROLE_NAMES[rank]}"
    return labels


# ---------------------------------------------------------------------------
# Substep tracer
# ---------------------------------------------------------------------------

class Tracer:
    """
    Temporarily wraps physics._step_ball_inner so every substep in which the
    ball touches a figure prints one line. physics.py itself is not modified.

    Columns
    -------
    face    front / side face that was hit
    vx, vy  ball velocity before -> after the substep
    xo      rod.x_offset before -> after (physics pushback moves it)
    toward  rod.x_offset * hit_sign after pushback -- the offset_toward_ball
            value that the reflect check compares to GLANCE_THRESHOLD
    dy/hw   where on the face the ball hit (-1 .. 1)
    result  REFLECT (vx flipped), glance (vx kept its sign), caught, side, UP
    """

    def __init__(self, rod):
        self.rod = rod
        self.tick = 0
        self.sub = 0
        self.front_ticks: set[int] = set()
        self.counts = {"REFLECT": 0, "glance": 0, "caught": 0, "side": 0}
        self.glance_vy = 0.0   # total |vy| added by glancing front contacts
        self._inner = physics._step_ball_inner

    def __enter__(self):
        physics._step_ball_inner = self._wrapped
        print(f"{'tick':>4} {'sub':>4} {'face':<5} {'x':>7} {'y':>6} "
              f"{'vx':>16} {'vy':>16} {'xo':>13} {'toward':>6} {'dy/hw':>6}  result")
        return self

    def __exit__(self, *exc):
        physics._step_ball_inner = self._inner

    def _wrapped(self, ball, field, dt, *args):
        rod = self.rod
        vx0, vy0, xo0, up0, rod_x0 = ball.vx, ball.vy, rod.x_offset, rod.up, rod.x

        # contacted_player is reset per step_ball() call, not per substep --
        # isolate it so we see exactly which substep made contact
        flag = ball.contacted_player
        ball.contacted_player = False
        result = self._inner(ball, field, dt, *args)
        hit = ball.contacted_player
        ball.contacted_player = flag or hit

        self.sub += 1
        if hit:
            self._report(ball, vx0, vy0, xo0, up0, rod_x0)
        return result

    def _report(self, ball, vx0, vy0, xo0, up0, rod_x0):
        rod = self.rod
        ht = rod.thickness / 2 + config.BALL_RADIUS
        hw = rod.width / 2 + config.BALL_RADIUS

        # The front-face branch snaps ball.x to rod.x +/- ht (rod.x before pushback)
        front = abs(abs(ball.x - rod_x0) - ht) < 1e-6
        hit_sign = 1.0 if ball.x > rod_x0 else -1.0
        toward = rod.x_offset * hit_sign
        py = min(rod.player_positions, key=lambda p: abs(ball.y - p))
        dy_hw = (ball.y - py) / hw

        if not front:
            kind = "side"
        elif ball.vx == 0.0:
            kind = "caught"
        elif (ball.vx > 0) != (vx0 > 0):
            kind = "REFLECT"
        else:
            kind = "glance"
            self.glance_vy += abs(ball.vy - vy0)
        self.counts[kind] += 1
        if front:
            self.front_ticks.add(self.tick)

        note = kind + ("  -> rod UP" if rod.up and not up0 else "")
        print(f"{self.tick:4d} {self.sub:4d} {'front' if front else 'side':<5} "
              f"{ball.x:7.2f} {ball.y:6.2f} "
              f"{vx0:7.1f}->{ball.vx:7.1f} {vy0:7.1f}->{ball.vy:7.1f} "
              f"{xo0:5.2f}->{rod.x_offset:5.2f} {toward:6.2f} {dy_hw:6.2f}  {note}")

    def longest_front_run(self) -> int:
        best = run = 0
        for t in range(max(self.front_ticks, default=-1) + 1):
            run = run + 1 if t in self.front_ticks else 0
            best = max(best, run)
        return best


# ---------------------------------------------------------------------------
# Running one isolated replay
# ---------------------------------------------------------------------------

def step_tick(ball: BallState, field: Field) -> str:
    """One tick of ball motion, chunked by CONTACT_STEP like simulate_point()."""
    elapsed = 0.0
    while elapsed < config.DT - 1e-9:
        h = config.DT - elapsed
        if ball.speed * h > config.CONTACT_STEP:
            h = config.CONTACT_STEP / ball.speed
        elapsed += h
        result = physics.step_ball(ball, field, h, ball_radius=config.BALL_RADIUS)
        if result != "play":
            return result
    return "play"


def replay(make, n_ticks: int, strat=None) -> bool:
    """
    Run a fresh copy of the scenario for n_ticks; True if it passes.

    make()  -> (field, ball, rod_idx), a new isolated scenario each call
    strat   -> if given, its choose_passive() re-positions the rod every tick
    """
    field, ball, rod_idx = make()
    rod = field.rods[rod_idx]
    result = "play"

    with Tracer(rod) as tr:
        for tick in range(n_ticks):
            tr.tick = tick
            if strat is not None:
                for i, (tx, up) in strat.choose_passive([(rod_idx, rod)], ball, field).items():
                    field.rods[i].set_x_offset(tx)
                    field.rods[i].up = up
            result = step_tick(ball, field)
            if result != "play":
                break

    run = tr.longest_front_run()
    ok = run < SLIDE_TICKS
    c = tr.counts
    print(f"end: {result} at tick {tick}  ball=({ball.x:.2f}, {ball.y:.2f}) "
          f"v=({ball.vx:.1f}, {ball.vy:.1f})")
    print(f"front contacts: REFLECT={c['REFLECT']} glance={c['glance']} caught={c['caught']}  "
          f"side={c['side']}  vy added by glances={tr.glance_vy:.1f} cm/s")
    print(f"longest front-contact run: {run} ticks  ->  "
          f"{'PASS' if ok else 'FAIL'} (fails at {SLIDE_TICKS})")
    return ok


def compare(make, n_ticks: int, strat) -> None:
    print("\n=== physics only ===")
    replay(make, n_ticks)
    print(f"\n=== + strategy ({type(strat).__name__}.choose_passive every tick) ===")
    replay(make, n_ticks, strat)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def isolate_frame(frame, rod_idx):
    """make() for a recorded frame: only rod_idx can touch the ball."""
    def make():
        field = Field()
        for i, rod in enumerate(field.rods):
            rod.set_offset(frame.rod_ys[i])
            rod.set_x_offset(frame.rod_xs[i])
            rod.up = frame.ups[rod_idx] if i == rod_idx else True
        ball = BallState(frame.ball_x, frame.ball_y, frame.ball_vx, frame.ball_vy)
        return field, ball, rod_idx
    return make


def from_seed(args) -> None:
    field, result, _ = run_point(args.seed, TEAM_0, TEAM_1, TEAM_0_SKILL, TEAM_1_SKILL)
    labels = rod_labels(field)
    slides = face_slides(result)
    if not slides:
        print(f"seed {args.seed}: no FACE-SLIDE detected")
        if args.tick is None:
            return
        slide_t, label = None, None
    else:
        slide_t, label = slides[0]
        print(f"seed {args.seed}: FACE-SLIDE on {label} starting t={slide_t:.3f}s")

    if label is None:
        label = "T1-goa"
    rod_idx = next(i for i, l in labels.items() if l == label)
    frames = result.frames

    # Start one frame before the first sliding contact, unless --tick given
    if args.tick is not None:
        start = next(f for f in frames if f.tick == args.tick)
    else:
        before = [f for f in frames if f.time < slide_t - 1e-9]
        start = before[-2] if len(before) >= 2 else frames[0]

    # --- 1. Capture: the recorded game around the slide ---
    deflect_times = [e.game_time for e in result.action_log
                     if e.action == "DEFLECT" and label in e.rod_label.split(",")]
    print(f"\nrecorded frames ({label}, rod {rod_idx})   * = DEFLECT that tick")
    print(f"{'tick':>4} {'t':>6} {'ball x':>7} {'y':>6} {'vx':>7} {'vy':>7} "
          f"{'xo':>6} {'yo':>6} {'up':>5} {'ctrl':>5}  event")
    for f in frames:
        if start.time - 1e-9 <= f.time <= start.time + args.ticks * config.DT:
            hit = any(f.time - config.DT < t <= f.time + 1e-9 for t in deflect_times)
            mark = ">" if f is start else ("*" if hit else " ")
            print(f"{mark}{f.tick:3d} {f.time:6.3f} {f.ball_x:7.2f} {f.ball_y:6.2f} "
                  f"{f.ball_vx:7.1f} {f.ball_vy:7.1f} {f.rod_xs[rod_idx]:6.2f} "
                  f"{f.rod_ys[rod_idx]:6.2f} {str(f.ups[rod_idx]):>5} "
                  f"{str(f.rod_ctrl[rod_idx]):>5}  {f.event or ''}")
    print(f"('>' = replay starts from this frame's state)")

    # --- 2-4. Isolate, replay, trace ---
    spec = TEAM_0 if field.rods[rod_idx].team == 0 else TEAM_1
    compare(isolate_frame(start, rod_idx), args.ticks, build_strategy(spec))


def manual(args) -> None:
    def make():
        field = Field()
        idx, goalie = max(field.rods_for_team(1), key=lambda ir: ir[1].x)
        for i, rod in enumerate(field.rods):
            rod.up = i != idx
        goalie.set_x_offset(args.xo)
        fig_y = goalie.player_positions[0]
        ball = BallState(x=goalie.x - 10, y=fig_y + args.dy, vx=args.vx, vy=0.0)
        return field, ball, idx

    print(f"manual: T1 goalie x_offset={args.xo}  ball {args.dy} cm off center, vx={args.vx}")
    print("(T1 is hit from -x, so a forward tilt toward the ball is a NEGATIVE x_offset)")
    compare(make, args.ticks, build_strategy(TEAM_1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("seed", type=int, nargs="?", help="seed to replay (as in play.py)")
    p.add_argument("--tick", type=int, help="replay from this recorded tick instead")
    p.add_argument("--ticks", type=int, default=30, help="ticks to replay (default 30)")
    p.add_argument("--manual", action="store_true", help="hand-built shot at the T1 goalie")
    p.add_argument("--xo", type=float, default=0.0, help="manual: goalie x_offset")
    p.add_argument("--dy", type=float, default=1.0, help="manual: cm off figure center")
    p.add_argument("--vx", type=float, default=150.0, help="manual: ball speed toward goal")
    args = p.parse_args()

    if args.manual:
        manual(args)
    elif args.seed is not None:
        from_seed(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
