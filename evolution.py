from typing import Optional
import heapq

import numpy as np
import random
from itertools import combinations

import config
from field import BallState, Field, Rod
from monte_carlo import run_monte_carlo
from strategy import Strategy

# config, move this to config.py later perhaps, also capitalize
n_generations = 20
# i think we use either this OR n_generations? if improvement is less than this 
# amount for n_plateau generations, then stop
t_plateau = 0.01 
n_plateau = 5

rr_points = 8
min_hof_wr = 0.2
n_parents = 2 # should be 2, undefined behaviour if more or less
pop_size = 50
n_elite = 2
tournament_size = 3
n_migrate = 2 # currently unused
mutation_rate = 0.2      # probability each gene is mutated
mutation_strength = 0.1  # Gaussian std for mutation noise

Genome = list[float]

GENE_GROUPS = {
    "skill": ["accuracy", "speed_consistency", "movement_speed"],
    "shot":  ["aim_at_gap", "aim_off_wall", "aim_off_player"],
    "pass":  ["pass_forward", "pass_back", "pass_side"],
    "indep": ["aggression", "lift_attackers", "lift_defenders",
              "passive_x_offset_attack", "passive_x_offset_defense",
              "defensive_activity"],
}

_OFFSETS = {}
idx = 0
for name, genes in GENE_GROUPS.items():
    _OFFSETS[name] = (idx, idx + len(genes))
    idx += len(genes)
GENOME_SIZE = idx  # 15

def _group(genome: Genome, name: str) -> Genome:
    """Slice the genes belonging to a named group out of the flat genome array."""
    lo, hi = _OFFSETS[name]
    return genome[lo:hi]

class Agent:
    def __init__(self, genome: np.ndarray) -> None:
        self.genome: Genome = genome
        self.fitness: float = 0.0

class ParameterizedStrategy(Strategy):
    """Strategy whose decisions are driven by a float genome vector."""

    def __init__(self, genome: Genome) -> None:
        self.genome = genome
        self.accuracy, self.speed_consistency, self.movement_speed = _group(genome, "skill")
        self.aim_gap, self.aim_wall, self.aim_player                = _group(genome, "shot")
        self.pass_fwd, self.pass_back, self.pass_side               = _group(genome, "pass")
        (self.aggression,
         self.lift_atk, self.lift_def,
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
        # Slide controlled rods toward ball or predicted trajectory
        # Genes: movement_speed, aggression, defensive_activity
        
        team = controlled_rods[0][1].team
        attack_dir = 1 if team == 0 else -1

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
                up = self.lift_def > (ball.x / field.depth) * attack_dir + 0.5

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
        # Genes: aggression, accuracy, speed_consistency, aim_gap/wall/player, pass_fwd/back/side

        team = rod.team
        attack_dir = 1 if team == 0 else -1

def _genome_to_strategy(genome: Genome) -> ParameterizedStrategy:
    return ParameterizedStrategy(genome)

def initialize_population() -> list[Agent]:
    """
    Create pop_size agents with randomized genomes.
    Normalized groups (skill, shot, pass) are sampled from a Dirichlet
    distribution so they sum to 1. Independent genes are uniform [0, 1].
    """
    population = []
    for _ in range(pop_size):
        parts = []
        for name, genes in GENE_GROUPS.items():
            if name == "indep":
                parts.append(np.random.uniform(0.0, 1.0, len(genes)))
            else:
                parts.append(np.random.dirichlet(np.ones(len(genes))))
        population.append(Agent(np.concatenate(parts)))
    return population

def rr_tourney(population: list[Agent]) -> list[float]:
    wins = {agent: 0 for agent in population}
    games = {agent: 0 for agent in population}

    for agent0, agent1 in combinations(population, 2):
        result = run_monte_carlo(
            Field(),
            {0: _genome_to_strategy(agent0.genome), 1: _genome_to_strategy(agent1.genome)},
            n_simulations=rr_points,
        )

        wins[agent0] += result.team0_wins
        wins[agent1] += result.team1_wins
        games[agent0] += result.n_simulations
        games[agent1] += result.n_simulations

    return [wins[a] / games[a] for a in population]

def _get_parents(population: list[Agent], scores: list[float]):
    parents = []

    # tournament selection
    for _ in range(n_parents):
        candidates = random.sample(range(pop_size), tournament_size)
        best = max(candidates, key=lambda i: scores[i])
        parents.append(population[best])

    return parents

def _crossover(parents: list[Agent]) -> Agent:
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
        mask = np.random.rand(hi - lo) < mutation_rate
        genome[lo:hi] += mask * np.random.normal(0, mutation_strength, hi - lo)

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

    for _ in range(pop_size):
        parents = _get_parents(population, scores)
        child = _mutate(_crossover(parents))
        children.append(child)
    
    return children


def main() -> None:
    population = initialize_population()

    for _ in range(n_generations):
        scores = rr_tourney(population)
        population = make_children(population, scores)

    # you get population at the end

if __name__ == "__main__":
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