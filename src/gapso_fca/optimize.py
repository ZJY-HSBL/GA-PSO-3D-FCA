"""Bilevel structure search (GA) and independent continuous calibration (PSO)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from .core import BOUNDS, Params, Structure

NEIGHBORHOODS = (6, 18, 26)


@dataclass
class SearchConfig:
    ga_population: int = 30
    ga_generations: int = 30
    crossover_probability: float = 0.80
    mutation_probability: float = 0.05
    pso_particles: int = 30
    pso_iterations: int = 30
    inertia: float = 0.70
    cognitive: float = 1.50
    social: float = 1.50


def pso(objective: Callable[[Params], float], cfg: SearchConfig, seed: int):
    rng = np.random.default_rng(seed)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    x = rng.uniform(lo, hi, (cfg.pso_particles, 6))
    v = np.zeros_like(x)
    scores = np.array([objective(Params.from_array(z)) for z in x])
    best_positions, best_scores = x.copy(), scores.copy()
    j = int(np.argmin(scores))
    global_best, global_score = x[j].copy(), float(scores[j])
    history = [global_score]
    for _ in range(cfg.pso_iterations):
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = (cfg.inertia*v + cfg.cognitive*r1*(best_positions-x) +
             cfg.social*r2*(global_best-x))
        v = np.clip(v, -.2*(hi-lo), .2*(hi-lo))
        x = np.clip(x+v, lo, hi)
        scores = np.array([objective(Params.from_array(z)) for z in x])
        update = scores < best_scores
        best_scores[update] = scores[update]
        best_positions[update] = x[update]
        j = int(np.argmin(best_scores))
        if float(best_scores[j]) < global_score:
            global_best = best_positions[j].copy()
            global_score = float(best_scores[j])
        history.append(global_score)
    return Params.from_array(global_best), global_score, history


def decode(gene):
    return Structure(NEIGHBORHOODS[int(gene[0])], int(gene[1]), int(gene[2])+1)


def ga(search_structure: Callable[[Structure], tuple[Params,float]],
       cfg: SearchConfig, seed: int):
    rng = np.random.default_rng(seed)
    caps = np.array([3,3,2])
    population = rng.integers(0, caps, size=(cfg.ga_population, 3))
    cache = {}
    def scored(gene):
        key = tuple(int(z) for z in gene)
        if key not in cache:
            cache[key] = search_structure(decode(key))
        return cache[key][1]
    history = []
    best_gene = population[0].copy()
    best_score = float('inf')
    for generation in range(cfg.ga_generations):
        scores = np.array([scored(ind) for ind in population])
        i = int(np.argmin(scores))
        if scores[i] < best_score:
            best_score, best_gene = float(scores[i]), population[i].copy()
        history.append(best_score)
        next_pop = [best_gene.copy()]
        def select():
            pair = rng.integers(0, len(population), size=2)
            return population[pair[np.argmin(scores[pair])]].copy()
        while len(next_pop) < cfg.ga_population:
            a, b = select(), select()
            if rng.random() < cfg.crossover_probability:
                mask = rng.random(3) < 0.5
                a, b = np.where(mask,a,b), np.where(mask,b,a)
            for child in (a,b):
                for k in range(3):
                    if rng.random() < cfg.mutation_probability:
                        child[k] = rng.integers(caps[k])
                next_pop.append(child)
                if len(next_pop) == cfg.ga_population:
                    break
        population = np.array(next_pop)
    return decode(best_gene), cache[tuple(best_gene)][0], best_score, history, len(cache)


def optimize(model: str, objective: Callable[[Structure,Params],float],
             cfg: SearchConfig, seed: int, fixed=Structure(), base_params=Params()):
    """GA-FCA fixes parameters, PSO-FCA fixes structure, GA-PSO-FCA nests independent PSOs."""
    if model == 'GA-FCA':
        def structure_cost(s):
            return base_params, objective(s,base_params)
        s,p,cost,history,n_unique = ga(structure_cost,cfg,seed)
        return s,p,cost,history,n_unique
    if model == 'PSO-FCA':
        p,cost,history = pso(lambda x:objective(fixed,x),cfg,seed)
        return fixed,p,cost,history,1
    if model == 'GA-PSO-FCA':
        def structure_cost(s):
            # The random initialization for any structure is stable across GA evaluation order.
            n, rule, power = s.key()
            structure_seed = int((seed * 9973 + n*991 + rule*31 + power) % (2**32))
            p,cost,_ = pso(lambda x:objective(s,x),cfg,structure_seed)
            return p,cost
        return ga(structure_cost,cfg,seed)
    raise ValueError(model)
