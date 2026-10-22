"""Time Monte Carlo throughput: single-core and full parallel round-robin generation."""
import os, sys, time

from field import Field
from monte_carlo import run_monte_carlo
from strategy import SmackBall, AimAtGap, HardOffense, DefensiveWall, TiltAndGap, ReactiveBlock
import evolution


def main():
    N = 500
    print(f"CPU cores: {os.cpu_count()}\n")

    print(f"--- Single-core: {N} points per matchup ---")
    matchups = [
        ("SmackBall vs SmackBall", SmackBall(), SmackBall()),
        ("HardOffense vs DefensiveWall", HardOffense(), DefensiveWall()),
        ("TiltAndGap vs ReactiveBlock", TiltAndGap(), ReactiveBlock()),
        ("AimAtGap vs HardOffense", AimAtGap(), HardOffense()),
    ]
    rates = []
    for name, s0, s1 in matchups:
        t0 = time.perf_counter()
        res = run_monte_carlo(Field(), {0: s0, 1: s1}, n_simulations=N)
        dt = time.perf_counter() - t0
        pps = N / dt
        rates.append(pps)
        print(f"  {name:<30} {dt:6.2f}s  {pps:7.0f} pts/s  avg {res.avg_ticks:.0f} ticks/pt")
    print(f"  mean single-core: {sum(rates)/len(rates):.0f} pts/s\n")

    print(f"--- Parallel: one round-robin generation ---")
    pop = evolution.initialize_population()
    n_pairs = len(pop) * (len(pop) - 1) // 2
    n_points = n_pairs * evolution.RR_POINTS
    print(f"  {len(pop)} agents -> {n_pairs} matchups x {evolution.RR_POINTS} pts = {n_points:,} points")
    t0 = time.perf_counter()
    evolution.rr_tourney(pop)
    dt = time.perf_counter() - t0
    print(f"  {dt:.1f}s  ->  {n_points/dt:,.0f} pts/s  ({n_points/dt*60:,.0f} pts/min)")
    print(f"  50 generations ~ {50*dt/60:.0f} min, ~{50*n_points:,} total points")


if __name__ == "__main__":
    main()
