"""
test_physics.py — Ball-physics correctness tests (no framework; run directly).

    python test_physics.py

Covers wall reflection and player-contact bounce direction handling. The key
regression target: a player-contact bounce must NOT reverse a velocity component
that is already carrying the ball AWAY from the player (a "separating" contact),
otherwise the ball briefly travels the wrong way (a spurious trail segment).
"""
from __future__ import annotations

import math

import config
import physics
from field import Field, Rod, BallState


# ---------------------------------------------------------------------------
# Tiny test runner
# ---------------------------------------------------------------------------
_FAILURES: list[str] = []


def _check(name: str, fn) -> None:
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:  # noqa: BLE001 — test harness reports any failure
        _FAILURES.append(name)
        print(f"  FAIL  {name}: {e}")


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# The reflect-toward primitive
# ---------------------------------------------------------------------------

def test_reflect_toward_flips_when_approaching():
    # Ball above obstacle (offset > 0) moving down (v < 0) -> toward -> reflect.
    assert physics._reflect_toward(-5.0, +2.0) == 5.0
    # Ball below obstacle (offset < 0) moving up (v > 0) -> toward -> reflect.
    assert physics._reflect_toward(5.0, -2.0) == -5.0


def test_reflect_toward_keeps_when_separating():
    # Ball above obstacle (offset > 0) already moving up (v > 0) -> separating.
    assert physics._reflect_toward(5.0, +2.0) == 5.0
    # Ball below obstacle (offset < 0) already moving down (v < 0) -> separating.
    assert physics._reflect_toward(-5.0, -2.0) == -5.0


# ---------------------------------------------------------------------------
# Wall reflection (the user's requested assertions)
# ---------------------------------------------------------------------------

def test_side_wall_reflection_is_clean():
    """Fire a ball at a known angle into the top wall; verify a clean bounce."""
    f = Field()
    f.rods = []  # isolate pure wall physics
    r = config.BALL_RADIUS
    dt = 0.1

    vx0, vy0 = 40.0, 30.0  # heading up-and-right toward the top wall
    ball = BallState(x=60.0, y=60.0, vx=vx0, vy=vy0)

    sign_changes = 0
    prev_sign = 1  # vy starts positive
    max_overshoot = 0.0
    bounced_vx = bounced_vy = None

    for _ in range(10):  # enough to complete exactly one (top) bounce
        physics.step_ball(ball, f, dt, friction=0.0, ball_radius=r)
        max_overshoot = max(max_overshoot,
                            r - ball.y, ball.y - (f.width - r), 0.0)
        s = 1 if ball.vy > 0 else -1
        if s != prev_sign:
            sign_changes += 1
            prev_sign = s
            bounced_vx, bounced_vy = ball.vx, ball.vy

    # 1. never exceeds the boundary by more than a floating-point epsilon
    assert max_overshoot <= 1e-9, f"overshoot {max_overshoot}"
    # 2. vy changes sign exactly once during the bounce
    assert sign_changes == 1, f"vy sign changes = {sign_changes}"
    # 3. reflection angle: vx preserved, vy negated (friction off => exact)
    assert _approx(bounced_vx, vx0, 1e-9), f"vx {bounced_vx} != {vx0}"
    assert _approx(bounced_vy, -vy0, 1e-9), f"vy {bounced_vy} != {-vy0}"
    # incoming/outgoing angle magnitudes match (mirror about the wall)
    assert _approx(abs(math.atan2(vy0, vx0)),
                   abs(math.atan2(bounced_vy, bounced_vx)), 1e-9)


# ---------------------------------------------------------------------------
# Player-contact bounce direction handling
# ---------------------------------------------------------------------------

def _single_rod_field(px_y: float = 34.0) -> tuple[Field, Rod]:
    """A field containing exactly one uncontrolled, 1-player rod at mid-field."""
    f = Field()
    rod = Rod(x=60.0, team=0, n_players=1)
    rod.set_offset(px_y - rod._base_positions[0])  # put the player at y=px_y
    f.rods = [rod]
    return f, rod


def test_passive_contact_separating_does_not_reverse():
    """A ball overlapping a player but already moving AWAY must not get flipped
    back toward the player. (This is the spurious 'wrong direction' bug.)"""
    r = config.BALL_RADIUS
    f, rod = _single_rod_field(px_y=34.0)
    py = rod.player_positions[0]

    # Ball just below the player's centre, moving DOWN (away from the player).
    ball = BallState(x=rod.x, y=py - 1.0, vx=0.0, vy=-50.0)
    physics.step_ball(ball, f, dt=0.001, friction=0.0, ball_radius=r)

    assert ball.vy < 0.0, f"separating ball was reversed: vy={ball.vy}"


def test_passive_contact_approaching_still_reflects():
    """Guard must not break legitimate bounces: a ball moving INTO the player
    from below still reflects upward."""
    r = config.BALL_RADIUS
    f, rod = _single_rod_field(px_y=34.0)
    py = rod.player_positions[0]

    # Ball just below the player's centre, moving UP (into the player).
    ball = BallState(x=rod.x, y=py - 1.0, vx=0.0, vy=+50.0)
    physics.step_ball(ball, f, dt=0.001, friction=0.0, ball_radius=r)

    assert ball.vy < 0.0, f"approaching ball did not reflect: vy={ball.vy}"


# ---------------------------------------------------------------------------
# Integration: no separating player-flips across many full points
# (covers the controlled-rod rigid-bounce path in simulation.py too)
# ---------------------------------------------------------------------------

def test_no_separating_player_flips_across_points():
    import simulation
    from strategy import HardOffense, TiltAndGap
    from physics import find_overlapping_players as _real_fop
    from simulation import _make_frame as _real_mf

    saved = (config.FPS, config.DT, config.MAX_TICKS)
    config.FPS = 30
    config.DT = 1.0 / 30
    config.MAX_TICKS = int(config.MAX_GAME_TIME * 30)

    r = config.BALL_RADIUS
    pre: list = []
    post: list = []

    def fop(ball, field):
        ov = _real_fop(ball, field)
        geo = []
        for ri, pi in ov:
            rod = field.rods[ri]
            ppy = rod.player_positions[pi]
            dx = ball.x - rod.x
            dy = ball.y - ppy
            pen_x = (rod.thickness / 2 + r) - abs(dx)
            pen_y = (rod.width / 2 + r) - abs(dy)
            geo.append((dx, dy, pen_x, pen_y))
        pre.append((ball.vx, ball.vy, geo))
        return ov

    def mf(tick, time, ball, field, event):
        post.append((ball.vx, ball.vy))
        return _real_mf(tick, time, ball, field, event)

    simulation.find_overlapping_players = fop
    simulation._make_frame = mf
    try:
        separating = 0
        for seed in range(40):
            pre.clear(); post.clear()
            field = Field()
            simulation.simulate_point(field, {0: HardOffense(), 1: TiltAndGap()},
                                      kickoff_team=1, record=True, seed=seed)
            for (vx0, vy0, geo), (vx1, vy1) in zip(pre, post):
                if not geo:
                    continue
                # A bounce cleanly negates one component and preserves the other
                # (no friction runs between the overlap check and the frame). A
                # committed swing (HIT) instead SETS both components to the swing
                # vector, so the clean-negation test excludes HITs.
                for (dx, dy, pen_x, pen_y) in geo:
                    side = pen_x >= pen_y
                    if side:
                        bounce = abs(vy1 + vy0) < 1e-6 and abs(vx1 - vx0) < 1e-6
                        if bounce and vy0 * dy > 0:   # negated while separating
                            separating += 1
                    else:
                        bounce = abs(vx1 + vx0) < 1e-6 and abs(vy1 - vy0) < 1e-6
                        if bounce and vx0 * dx > 0:
                            separating += 1
                    break
        assert separating == 0, f"{separating} separating player-flips occurred"
    finally:
        simulation.find_overlapping_players = _real_fop
        simulation._make_frame = _real_mf
        config.FPS, config.DT, config.MAX_TICKS = saved


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running physics tests...")
    _check("reflect_toward flips when approaching", test_reflect_toward_flips_when_approaching)
    _check("reflect_toward keeps when separating", test_reflect_toward_keeps_when_separating)
    _check("side wall reflection is clean", test_side_wall_reflection_is_clean)
    _check("passive contact separating does not reverse", test_passive_contact_separating_does_not_reverse)
    _check("passive contact approaching still reflects", test_passive_contact_approaching_still_reflects)
    _check("no separating player-flips across points", test_no_separating_player_flips_across_points)

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED: {', '.join(_FAILURES)}")
        raise SystemExit(1)
    print("All physics tests passed.")
