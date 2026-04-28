from dataclasses import dataclass, field as dc_field
from typing import Optional
import heapq
import json
import os

import numpy as np
import random
from itertools import combinations

import config
from field import BallState, Field, Rod
from monte_carlo import run_monte_carlo
from strategy import Strategy, _best_wall_shot, _best_player_deflection, _project_ball_to_x

# config, move this to config.py later perhaps
N_GENERATIONS = 50
# i think we use either this OR N_GENERATIONS? if improvement is less than this
# amount for N_PLATEAU generations, then stop
T_PLATEAU = 0.01
N_PLATEAU = 5

RR_POINTS = 8
MIN_HOF_WR = 0.2
N_PARENTS = 2 # undefined behaviour if != 2
POP_SIZE = 50
N_ELITE = 2
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.2
MUTATION_STRENGTH = 0.1
N_MIGRATE = 2 # currently unused
N_WORKERS = 0   # 0 = use all CPU cores

Genome = np.ndarray

GENE_GROUPS = {
    "skill": ["accuracy", "power_consistency", "movement_control"],
    "shot":  ["aim_at_gap", "aim_off_wall", "aim_off_player"],
    "pass":  ["pass_forward", "pass_back", "pass_side"],
    "indep": ["aggression", "lift_attackers",
              "passive_x_offset_attack", "passive_x_offset_defense",
              "defensive_activity"],
}

_OFFSETS = {}
idx = 0
for name, genes in GENE_GROUPS.items():
    _OFFSETS[name] = (idx, idx + len(genes))
    idx += len(genes)
GENOME_SIZE = idx  # 14

@dataclass
class EvolutionHistory:
    gen:          list[int]   = dc_field(default_factory=list)
    best_wr:      list[float] = dc_field(default_factory=list)
    mean_wr:      list[float] = dc_field(default_factory=list)
    min_wr:       list[float] = dc_field(default_factory=list)
    std_wr:       list[float] = dc_field(default_factory=list)
    best_genomes: list        = dc_field(default_factory=list)  # list of np.ndarray (GENOME_SIZE,)

def _group(genome: Genome, name: str) -> Genome:
    """Slice the genes belonging to a named group out of the flat genome array."""
    lo, hi = _OFFSETS[name]
    return genome[lo:hi]

def _possessing_team(ball: BallState, field: Field) -> Optional[int]:
    """Return team whose rod overlaps the ball, or None if neither/both."""
    teams: set[int] = set()
    for rod in field.rods:
        if rod.up:
            continue
        for py in rod.player_positions:
            if (abs(ball.x - rod._base_x) <= rod.rod_x_reach + rod.thickness / 2
                    and abs(ball.y - py) <= rod.width / 2):
                teams.add(rod.team)
                break
    if len(teams) == 1:
        return next(iter(teams))
    return None

class Agent:
    def __init__(self, genome: np.ndarray) -> None:
        self.genome: Genome = genome
        self.fitness: float = 0.0

    def __lt__(self, other: "Agent") -> bool:
        return self.fitness < other.fitness

    def __gt__(self, other: "Agent") -> bool:
        return self.fitness > other.fitness

class ParameterizedStrategy(Strategy):
    """Strategy whose decisions are driven by a float genome vector."""

    def __init__(self, genome: Genome) -> None:
        self.genome = genome
        self.accuracy, self.power_consistency, self.movement_control = _group(genome, "skill")
        self.aim_gap, self.aim_wall, self.aim_player                = _group(genome, "shot")
        self.pass_fwd, self.pass_back, self.pass_side               = _group(genome, "pass")
        (self.aggression,
         self.lift_atk,
         self.passive_x_off_atk, self.passive_x_off_def,
         self.defensive_activity)                                   = _group(genome, "indep")

    def choose_hands(
        self,
        team_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
        n_hands: int,
    ) -> set[int]:
        # Hold rods closest to the ball; aggression biases toward offensive rods
        # Genes: aggression

        team = team_rods[0][1].team
        attack_dir = 1 if team == 0 else -1

        def score(rod):
            dist_score   = -abs(rod.x - ball.x)
            attack_score = (rod._base_x - field.depth / 2) * attack_dir
            return dist_score + self.aggression * attack_score

        best = heapq.nlargest(n_hands, team_rods, key=lambda pair: score(pair[1]))
        return {rod_idx for rod_idx, _ in best}

    def choose_pos(
        self,
        controlled_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, float, bool]]:
        # Defending rods always track ball/trajectory. Attacking rods lerp between
        # tracking and field center based on defensive_activity.
        # Genes: movement_control, defensive_activity, lift_atk, lift_def

        team = controlled_rods[0][1].team
        attack_dir = 1 if team == 0 else -1
        opp_has_ball = _possessing_team(ball, field) == (1 - team)
        targets = {}

        for rod_idx, rod in controlled_rods:
            rod.movement_control = self.movement_control
            is_attacking = (rod._base_x - field.depth / 2) * attack_dir > 0

            # track projected intercept or current ball.y
            predicted_y = _project_ball_to_x(ball, rod._base_x)
            track_y = predicted_y if predicted_y is not None else ball.y

            # - defending rods always track the ball
            # - attacking rods lerp between tracking (defensive_activity=1) and 
            #   field center (defensive_activity=0)
            target_y_abs = track_y if not is_attacking else (
                track_y * self.defensive_activity + (field.width / 2) * (1 - self.defensive_activity)
            )
            target_y_offset = target_y_abs - field.width / 2

            # x_offset: defensive_activity=1 → max forward lean, 0.5 → neutral, 0 → lean back
            x_offset = (self.defensive_activity - 0.5) * 2 * config.ROD_X_REACH * attack_dir

            if is_attacking and opp_has_ball:
                up = False
            else:
                # lift threshold compared against how far into our defensive half the ball is;
                # (ball.x / field.depth) * attack_dir + 0.5 ranges ~0.5 (ball at own goal)
                # to ~1.5 (ball at opponent goal): higher gene needed to lift when ball is far away
                up = is_attacking and self.lift_atk > (ball.x / field.depth) * attack_dir + 0.5

            targets[rod_idx] = (target_y_offset, x_offset, up)

        return targets

    def choose_passive(
        self,
        passive_rods: list[tuple[int, Rod]],
        ball: BallState,
        field: Field,
    ) -> dict[int, tuple[float, bool]]:
        # Lean and flip uncontrolled rods based on their attacking/defending role.
        # Genes: passive_x_off_atk, passive_x_off_def, lift_atk, lift_def
        if not passive_rods:
            return {}
        
        rod_positions = {}

        team = passive_rods[0][1].team
        attack_dir = 1 if team == 0 else -1

        for rod_idx, rod in passive_rods:
            is_attacking = (rod._base_x - field.depth / 2) * attack_dir > 0
            if is_attacking:
                # center at 0 to include forward and backward lean
                x_offset = (self.passive_x_off_atk - 0.5) * 2 * config.ROD_X_REACH
                up = self.lift_atk > (ball.x / field.depth) * attack_dir + 0.5
            else:
                x_offset = (self.passive_x_off_def - 0.5) * 2 * config.ROD_X_REACH
                up = False

            rod_positions[rod_idx] = x_offset, up
        
        return rod_positions
    
    def choose_hit(
        self,
        rod: Rod,
        player_idx: int,
        ball: BallState,
        field: Field,
    ) -> Optional[tuple[float, float]]:
        # Sample shot vs pass decision, then aim and apply skill noise.
        # Genes: aggression, accuracy, power_consistency, aim_gap/wall/player, pass_fwd/back/side

        rod.accuracy          = self.accuracy
        rod.power_consistency = self.power_consistency

        team = rod.team
        player_y = rod.player_positions[player_idx]
        goal = field.goal_for_attacker(team)
        intended_speed = config.HIT_SPEED * (0.5 + 0.5 * self.aggression)

        if np.random.random() < self.aggression:
            # shoot
            shot_weights = np.array([self.aim_gap, self.aim_wall, self.aim_player])
            shot_weights /= shot_weights.sum()
            shot_choice = np.random.choice(3, p=shot_weights)

            aim_x = goal.x
            aim_y = (goal.y_min + goal.y_max) / 2
            if shot_choice == 0:
                aim_x, aim_y = self._find_gap_target(rod, ball, field)
            elif shot_choice == 1:
                aim_x, aim_y = _best_wall_shot(rod, ball, field, aim_x, aim_y)
            else:
                if team == 0:
                    opp_passive = [r for r in field.rods if r.team != team and r.x > rod.x and not r.controlled]
                else:
                    opp_passive = [r for r in field.rods if r.team != team and r.x < rod.x and not r.controlled]

                if opp_passive:
                    aim_x, aim_y = _best_player_deflection(rod, ball, field, aim_x, aim_y)
                else:
                    aim_x, aim_y = self._find_gap_target(rod, ball, field)
        else:
            # pass
            pass_weights = np.array([self.pass_fwd, self.pass_back, self.pass_side])
            pass_weights /= pass_weights.sum()
            pass_choice = np.random.choice(3, p=pass_weights)

            if pass_choice == 0:
                aim_x, aim_y = self._aim_forward(rod, ball, field)
            elif pass_choice == 1:
                aim_x, aim_y = self._aim_back(rod, ball, field)
            else:
                aim_x, aim_y = self._aim_side(rod, player_idx, ball, field)

        return self._apply_hit(rod, aim_x, aim_y, player_y, intended_speed)


def save_agent(agent: "Agent", path: str, metadata: dict | None = None) -> None:
    obj: dict = {"genome": agent.genome.tolist()}
    if metadata:
        obj["metadata"] = metadata
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_agent(path: str) -> tuple["Agent", dict]:
    with open(path) as f:
        obj = json.load(f)
    return Agent(np.array(obj["genome"])), obj.get("metadata", {})

def initialize_population() -> list[Agent]:
    """
    Create POP_SIZE agents with randomized genomes.
    Normalized groups (skill, shot, pass) are sampled from a Dirichlet
    distribution so they sum to 1. Independent genes are uniform [0, 1].
    """
    population = []
    for _ in range(POP_SIZE):
        parts = []
        for name, genes in GENE_GROUPS.items():
            if name == "indep":
                parts.append(np.random.uniform(0.0, 1.0, len(genes)))
            else:
                parts.append(np.random.dirichlet(np.ones(len(genes))))
        population.append(Agent(np.concatenate(parts)))
    return population

def _matchup_worker(args: tuple[np.ndarray, np.ndarray]) -> tuple[int, int, int]:
    g0, g1 = args
    result = run_monte_carlo(
        Field(),
        {0: ParameterizedStrategy(g0), 1: ParameterizedStrategy(g1)},
        n_simulations=RR_POINTS,
    )
    return result.team0_wins, result.team1_wins, result.n_simulations

def rr_tourney(population: list[Agent]) -> list[float]:
    from multiprocessing import Pool
    n = len(population)
    pairs = list(combinations(range(n), 2))
    args = [(population[i].genome, population[j].genome) for i, j in pairs]

    workers = N_WORKERS if N_WORKERS > 0 else None  # None → cpu_count()
    with Pool(workers) as pool:
        results = pool.map(_matchup_worker, args)

    wins  = [0] * n
    games = [0] * n
    for (i, j), (w0, w1, g) in zip(pairs, results):
        wins[i]  += w0;  wins[j]  += w1
        games[i] += g;   games[j] += g

    return [wins[i] / games[i] for i in range(n)]

def _get_parents(population: list[Agent], scores: list[float]):
    parents = []

    # tournament selection
    for _ in range(N_PARENTS):
        candidates = random.sample(range(POP_SIZE), TOURNAMENT_SIZE)
        best = max(candidates, key=lambda i: scores[i])
        parents.append(population[best])

    return parents

def  _crossover(parents: list[Agent]) -> Agent:
    p0, p1 = parents[0].genome, parents[1].genome
    child_genome = np.empty(GENOME_SIZE)

    for name in GENE_GROUPS:
        lo, hi = _OFFSETS[name]
        if name == "indep":
            # gene-by-gene: each gene independently from either parent
            mask = np.random.rand(hi - lo) < 0.5
            child_genome[lo:hi] = np.where(mask, p0[lo:hi], p1[lo:hi])
        else:
            # block-level: take entire normalized group from one parent
            src = p0 if random.random() < 0.5 else p1
            child_genome[lo:hi] = src[lo:hi]

    return Agent(child_genome)

def _mutate(agent: Agent) -> Agent:
    genome = agent.genome.copy()

    for name in GENE_GROUPS:
        lo, hi = _OFFSETS[name]
        mask = np.random.rand(hi - lo) < MUTATION_RATE
        genome[lo:hi] += mask * np.random.normal(0, MUTATION_STRENGTH, hi - lo)

        if name == "indep":
            genome[lo:hi] = np.clip(genome[lo:hi], 0.0, 1.0)
        else:
            genome[lo:hi] = np.maximum(genome[lo:hi], 0.0)
            total = genome[lo:hi].sum()
            if total > 0:
                genome[lo:hi] /= total
            else:
                # degenerate: all genes zeroed out, reset to uniform
                genome[lo:hi] = np.ones(hi - lo) / (hi - lo)

    return Agent(genome)
        
# genomes -> parents -> children
def make_children(population: list[Agent], scores: list[float]):
    children = []

    for _ in range(POP_SIZE):
        parents = _get_parents(population, scores)
        child = _mutate(_crossover(parents))
        children.append(child)

    return children


def _record(history: EvolutionHistory, gen: int, scores: list[float], population: list["Agent"]) -> None:
    arr = np.array(scores)
    best_idx = int(np.argmax(arr))
    history.gen.append(gen)
    history.best_wr.append(float(arr.max()))
    history.mean_wr.append(float(arr.mean()))
    history.min_wr.append(float(arr.min()))
    history.std_wr.append(float(arr.std()))
    history.best_genomes.append(population[best_idx].genome.copy())

def main() -> None:
    import time
    history = EvolutionHistory()
    population = initialize_population()

    for gen in range(N_GENERATIONS):
        t0 = time.perf_counter()
        scores = rr_tourney(population)
        elapsed = time.perf_counter() - t0
        _record(history, gen, scores, population)
        print(f"gen {gen:02d}  best_wr={history.best_wr[-1]:.3f}  mean_wr={history.mean_wr[-1]:.3f}  ({elapsed:.1f}s)")
        population = make_children(population, scores)

    print("Scoring final population...")
    scores = rr_tourney(population)
    _record(history, N_GENERATIONS, scores, population)

    ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)

    os.makedirs("output/agents", exist_ok=True)
    for rank, (wr, agent) in enumerate(ranked[:5]):
        path = f"output/agents/rank{rank}.json"
        save_agent(agent, path, {"win_rate": round(wr, 4), "rank": rank})
        print(f"  rank {rank}  wr={wr:.3f}  -> {path}")

    from visualization import plot_evolution_stats
    plot_evolution_stats(history, save_path="output/evolution_stats.png")

if __name__ == "__main__":
    import sys
    if "--profile" in sys.argv:
        import cProfile, pstats, io
        # use small params so profiling finishes quickly
        POP_SIZE = 6
        N_GENERATIONS = 1
        pr = cProfile.Profile()
        pr.enable()
        main()
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
        print(s.getvalue())
    else:
        main()



# -------------------------------------------------------
# ---------   to add on after basics done   -------------
# -------------------------------------------------------

# - elites; first learn what problem it solves
# - island model; first learn what problem it solves
# - MAP-Elites; learn what problem it solves

# 50 genomes -> f genomes, only those that pass the hall of fame
# finish implementing this later, after discovering why we need it
# def filter_hof(population):
#     filtered_population = []
#     for individual in population:
#         for hardcoded in hall_of_fame:
#             res = run_monte_carlo(individual, hardcoded)
#             if res.winrate >= min_hof_wr:
#                 filtered_population.append(individual)

#     return filtered_population