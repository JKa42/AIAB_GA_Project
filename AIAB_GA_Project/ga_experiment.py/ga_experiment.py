"""
Genetic Algorithm Experiment: Effect of Mutation Rate on Grid-World Navigation
AIAB Coursework 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
import pickle

# ============================================================
# ENVIRONMENT
# ============================================================

class GridWorld:
    """
    A 10x10 discrete grid-world navigation environment.
    The agent navigates from a fixed start (0,0) to a fixed goal (9,9)
    while avoiding randomly placed obstacles.
    Actions: 0=up, 1=down, 2=left, 3=right
    """

    GRID_SIZE = 10
    OBSTACLE_RATIO = 0.15
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, seed=42):
        self.start = (0, 0)
        self.goal = (self.GRID_SIZE - 1, self.GRID_SIZE - 1)
        self.seed = seed
        self.grid = self._generate_grid(seed)

    def _generate_grid(self, seed):
        rng = np.random.RandomState(seed)
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        n_obstacles = int(self.GRID_SIZE ** 2 * self.OBSTACLE_RATIO)
        placed = 0
        while placed < n_obstacles:
            r = rng.randint(0, self.GRID_SIZE)
            c = rng.randint(0, self.GRID_SIZE)
            if (r, c) not in (self.start, self.goal) and grid[r, c] == 0:
                grid[r, c] = 1
                placed += 1
        return grid

    def simulate(self, genome):
        """
        Execute a genome (sequence of actions) in the environment.
        Returns: final_pos, reached_goal (bool), steps_to_goal (int), obstacles_hit (int)
        """
        pos = self.start
        obstacles_hit = 0
        steps_to_goal = len(genome)

        for step, action in enumerate(genome):
            dr, dc = self.ACTIONS[int(action)]
            new_r = pos[0] + dr
            new_c = pos[1] + dc

            # Out of bounds: agent stays
            if not (0 <= new_r < self.GRID_SIZE and 0 <= new_c < self.GRID_SIZE):
                continue

            # Obstacle: blocked, count hit
            if self.grid[new_r, new_c] == 1:
                obstacles_hit += 1
                continue

            pos = (new_r, new_c)

            if pos == self.goal:
                steps_to_goal = step + 1
                return pos, True, steps_to_goal, obstacles_hit

        return pos, False, steps_to_goal, obstacles_hit


# ============================================================
# FITNESS FUNCTION
# ============================================================

def compute_fitness(genome, env):
    """
    Fitness combines proximity to goal, goal attainment, path efficiency,
    and penalises obstacle collisions.

    F = -d(final_pos, goal) + 100 * reached_goal
        + 0.5 * (L - steps_to_goal) * reached_goal
        - 2 * obstacles_hit

    where d is Manhattan distance and L is genome length.
    """
    final_pos, reached_goal, steps_to_goal, obstacles_hit = env.simulate(genome)

    manhattan = abs(final_pos[0] - env.goal[0]) + abs(final_pos[1] - env.goal[1])
    goal_bonus = 100 if reached_goal else 0
    efficiency = 0.5 * (len(genome) - steps_to_goal) if reached_goal else 0
    obstacle_penalty = 2 * obstacles_hit

    return float(-manhattan + goal_bonus + efficiency - obstacle_penalty)


# ============================================================
# GENETIC ALGORITHM
# ============================================================

class GeneticAlgorithm:
    """
    Standard generational GA with tournament selection,
    single-point crossover, and uniform integer mutation.
    """

    N_ACTIONS = 4

    def __init__(self,
                 population_size=100,
                 genome_length=40,
                 crossover_rate=0.8,
                 mutation_rate=0.05,
                 tournament_size=3,
                 n_elites=2,
                 n_generations=200,
                 env=None,
                 seed=0):

        self.pop_size = population_size
        self.genome_length = genome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.n_elites = n_elites
        self.n_generations = n_generations
        self.env = env
        self.rng = np.random.RandomState(seed)

    # ---- Initialisation ----

    def _init_population(self):
        return [self.rng.randint(0, self.N_ACTIONS, self.genome_length).tolist()
                for _ in range(self.pop_size)]

    # ---- Evaluation ----

    def _evaluate(self, population):
        return [compute_fitness(ind, self.env) for ind in population]

    # ---- Selection (tournament) ----

    def _tournament_select(self, population, fitnesses):
        idx = self.rng.choice(len(population), self.tournament_size, replace=False)
        winner = idx[int(np.argmax([fitnesses[i] for i in idx]))]
        return population[winner][:]

    # ---- Crossover (single-point) ----

    def _crossover(self, p1, p2):
        if self.rng.random() < self.crossover_rate:
            pt = self.rng.randint(1, self.genome_length)
            return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
        return p1[:], p2[:]

    # ---- Mutation (uniform integer) ----

    def _mutate(self, genome):
        for i in range(len(genome)):
            if self.rng.random() < self.mutation_rate:
                genome[i] = self.rng.randint(0, self.N_ACTIONS)
        return genome

    # ---- Main loop ----

    def run(self):
        """
        Execute the GA and return per-generation statistics and final results.
        """
        CONVERGENCE_THRESHOLD = 80.0  # fitness >= threshold implies goal reached

        population = self._init_population()
        fitnesses = self._evaluate(population)

        best_fitness_history = []
        mean_fitness_history = []
        convergence_gen = None

        for gen in range(self.n_generations):
            best_f = max(fitnesses)
            best_fitness_history.append(best_f)
            mean_fitness_history.append(float(np.mean(fitnesses)))

            if convergence_gen is None and best_f >= CONVERGENCE_THRESHOLD:
                convergence_gen = gen

            # Elitism
            elite_idx = np.argsort(fitnesses)[::-1][:self.n_elites]
            new_pop = [population[i][:] for i in elite_idx]

            # Fill remainder
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))

            population = new_pop
            fitnesses = self._evaluate(population)

        # Final best individual
        best_idx = int(np.argmax(fitnesses))
        best_genome = population[best_idx]
        _, reached_goal, steps_to_goal, _ = self.env.simulate(best_genome)

        return {
            'best_fitness_history': best_fitness_history,
            'mean_fitness_history': mean_fitness_history,
            'final_best_fitness': max(fitnesses),
            'reached_goal': reached_goal,
            'steps_to_goal': steps_to_goal if reached_goal else None,
            'convergence_gen': convergence_gen,
            'best_genome': best_genome
        }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(mutation_rates, n_runs=20, n_generations=200,
                   population_size=100, genome_length=40, env_seed=42):
    """
    Run the GA for each mutation rate across multiple independent runs.
    Returns a dict mapping mutation_rate -> list of run result dicts.
    """
    env = GridWorld(seed=env_seed)
    results = {}

    for mr in mutation_rates:
        print(f"\nMutation rate = {mr}")
        runs = []
        for run in range(n_runs):
            # Unique seed per (mutation_rate, run) combination
            ga_seed = int(mr * 1e5) + run
            ga = GeneticAlgorithm(
                population_size=population_size,
                genome_length=genome_length,
                mutation_rate=mr,
                n_generations=n_generations,
                env=env,
                seed=ga_seed
            )
            result = ga.run()
            runs.append(result)
            status = "SUCCESS" if result['reached_goal'] else "FAIL"
            print(f"  Run {run+1:2d}/{n_runs} [{status}]  "
                  f"final_fitness={result['final_best_fitness']:7.2f}  "
                  f"conv_gen={result['convergence_gen']}")
        results[mr] = runs

    return results, env


# ============================================================
# ANALYSIS
# ============================================================

def compute_statistics(results, mutation_rates):
    """
    Aggregate per-condition statistics across runs.
    """
    stats = {}
    for mr in mutation_rates:
        runs = results[mr]
        n = len(runs)
        final_fitnesses = np.array([r['final_best_fitness'] for r in runs])
        success_mask = np.array([r['reached_goal'] for r in runs])
        conv_gens = [r['convergence_gen'] for r in runs if r['convergence_gen'] is not None]
        steps = [r['steps_to_goal'] for r in runs if r['steps_to_goal'] is not None]

        stats[mr] = {
            'success_rate': float(np.mean(success_mask)),
            'n_success': int(np.sum(success_mask)),
            'n_runs': n,
            'mean_final_fitness': float(np.mean(final_fitnesses)),
            'std_final_fitness': float(np.std(final_fitnesses)),
            'mean_convergence_gen': float(np.mean(conv_gens)) if conv_gens else None,
            'std_convergence_gen': float(np.std(conv_gens)) if conv_gens else None,
            'mean_steps_to_goal': float(np.mean(steps)) if steps else None,
        }
    return stats


# ============================================================
# VISUALISATION
# ============================================================

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def plot_all(results, stats, mutation_rates, n_generations, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ---- Figure 1: Best fitness over generations ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, mr in enumerate(mutation_rates):
        hists = np.array([r['best_fitness_history'] for r in results[mr]])
        mean = hists.mean(axis=0)
        std  = hists.std(axis=0)
        gens = np.arange(n_generations)
        ax.plot(gens, mean, label=f'$\\mu$={mr}', color=COLORS[i], linewidth=1.5)
        ax.fill_between(gens, mean - std, mean + std, alpha=0.12, color=COLORS[i])
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Best Fitness', fontsize=11)
    ax.set_title('Best Fitness over Generations (mean ± 1 SD, 20 runs)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig1_best_fitness.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig1_best_fitness.png")

    # ---- Figure 2: Mean population fitness over generations ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, mr in enumerate(mutation_rates):
        hists = np.array([r['mean_fitness_history'] for r in results[mr]])
        mean = hists.mean(axis=0)
        ax.plot(np.arange(n_generations), mean, label=f'$\\mu$={mr}',
                color=COLORS[i], linewidth=1.5)
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Mean Population Fitness', fontsize=11)
    ax.set_title('Mean Population Fitness over Generations (averaged over 20 runs)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig2_mean_fitness.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig2_mean_fitness.png")

    # ---- Figure 3: Success rate bar chart ----
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = [str(mr) for mr in mutation_rates]
    srates = [stats[mr]['success_rate'] * 100 for mr in mutation_rates]
    bars = ax.bar(labels, srates, color=COLORS, edgecolor='grey', linewidth=0.5)
    for bar, val in zip(bars, srates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10)
    ax.set_xlabel('Mutation Rate ($\\mu$)', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Goal-Reaching Success Rate by Mutation Rate (20 runs)', fontsize=11)
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig3_success_rate.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig3_success_rate.png")

    # ---- Figure 4: Final best fitness boxplot ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [[r['final_best_fitness'] for r in results[mr]] for mr in mutation_rates]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=1.5))
    for patch, c in zip(bp['boxes'], COLORS):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    ax.set_xlabel('Mutation Rate ($\\mu$)', fontsize=11)
    ax.set_ylabel('Final Best Fitness', fontsize=11)
    ax.set_title('Distribution of Final Best Fitness by Mutation Rate (20 runs)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_fitness_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig4_fitness_boxplot.png")

    # ---- Figure 5: Grid world visualisation ----
    env = GridWorld(seed=42)
    fig, ax = plt.subplots(figsize=(5, 5))
    grid_display = np.zeros((env.GRID_SIZE, env.GRID_SIZE, 3))
    for r in range(env.GRID_SIZE):
        for c in range(env.GRID_SIZE):
            if env.grid[r, c] == 1:
                grid_display[r, c] = [0.2, 0.2, 0.2]   # obstacle: dark grey
            else:
                grid_display[r, c] = [0.95, 0.95, 0.95] # free: light grey
    # Start and goal
    sr, sc = env.start
    gr, gc = env.goal
    grid_display[sr, sc] = [0.2, 0.6, 0.2]   # start: green
    grid_display[gr, gc] = [0.8, 0.1, 0.1]   # goal: red

    ax.imshow(grid_display, origin='upper')
    ax.set_xticks(np.arange(-0.5, env.GRID_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.GRID_SIZE, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.8)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title('Grid World Environment (10×10)', fontsize=11)

    legend_patches = [
        mpatches.Patch(color=[0.2, 0.6, 0.2], label='Start (0,0)'),
        mpatches.Patch(color=[0.8, 0.1, 0.1], label='Goal (9,9)'),
        mpatches.Patch(color=[0.2, 0.2, 0.2], label='Obstacle'),
        mpatches.Patch(color=[0.95, 0.95, 0.95], label='Free cell'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8,
              framealpha=0.9, edgecolor='grey')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig5_grid_world.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig5_grid_world.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    # Hyperparameters
    MUTATION_RATES  = [0.001, 0.01, 0.05, 0.1, 0.3]
    N_RUNS          = 20
    N_GENERATIONS   = 200
    POPULATION_SIZE = 100
    GENOME_LENGTH   = 40
    ENV_SEED        = 42

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'ga_results')

    # Run experiments
    print("=" * 60)
    print("EXPERIMENT: Effect of Mutation Rate on GA Navigation")
    print("=" * 60)

    results, env = run_experiment(
        mutation_rates=MUTATION_RATES,
        n_runs=N_RUNS,
        n_generations=N_GENERATIONS,
        population_size=POPULATION_SIZE,
        genome_length=GENOME_LENGTH,
        env_seed=ENV_SEED
    )

    # Statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    stats = compute_statistics(results, MUTATION_RATES)

    for mr in MUTATION_RATES:
        s = stats[mr]
        print(f"\nMutation rate = {mr}")
        print(f"  Success rate         : {s['success_rate']*100:.1f}%  "
              f"({s['n_success']}/{s['n_runs']} runs)")
        print(f"  Mean final fitness   : {s['mean_final_fitness']:.2f} "
              f"± {s['std_final_fitness']:.2f}")

        if s['mean_convergence_gen'] is not None:
            print(f"  Mean convergence gen : {s['mean_convergence_gen']:.1f} "
                  f"± {s['std_convergence_gen']:.1f}")
        else:
            print("  Mean convergence gen : N/A (no successful runs)")

        if s['mean_steps_to_goal'] is not None:
            print(f"  Mean steps to goal   : {s['mean_steps_to_goal']:.1f}")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(OUTPUT_DIR, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults saved to {OUTPUT_DIR}")

    # Plots
    print("\nGenerating figures...")
    plot_all(results, stats, MUTATION_RATES, N_GENERATIONS, OUTPUT_DIR)

    print("\nAll done.")