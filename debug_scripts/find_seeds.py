"""find_seeds.py -- find replay seeds that show suspicious ball-player contact.

Uses play.py's settings (FPS, teams, skills, kickoff), so every seed printed
here replays identically with:  python play.py <seed>

Usage:  python find_seeds.py [n_seeds]

The detectors (BACK-BOUNCE, FACE-SLIDE, SNAP) live in detectors.py, which the
test suite's seed sweep shares; see its docstring for what each one flags.
"""

import sys

import _path  # noqa: F401  (project root on sys.path)
import play # applies play.py's FPS to config before the sim is imported
import config
from detectors import back_bounces, face_slides, snaps, track_rigid_contacts
from play import run_point, TEAM_0, TEAM_1, TEAM_0_SKILL, TEAM_1_SKILL


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
