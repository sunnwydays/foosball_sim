import numpy as np
import random
from itertools import combinations

from field import Field
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

# decoder
def genome_to_strategy(genome: Genome) -> Strategy:
    pass

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
            {0: genome_to_strategy(agent0.genome), 1: genome_to_strategy(agent1.genome)},
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
    # block-level for normalized groups, gene-by-gene for indep
    pass

def _mutate(agent: Agent) -> Agent:
    # mutate some number of random genes by some amount
    pass
        
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