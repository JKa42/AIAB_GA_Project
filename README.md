# AIAB_GA_Project
Genetic Algorithm experiment for grid-world navigation.
# Genetic Algorithm Grid-World Navigation

This project investigates how mutation rate affects the performance of a Genetic Algorithm in a 10x10 grid-world navigation task.

The agent starts at position (0,0) and must reach the goal at (9,9) while avoiding obstacles. Each individual is represented as a fixed-length genome of movement actions.

## Experiment

Five mutation rates are compared:

- 0.001
- 0.01
- 0.05
- 0.1
- 0.3

Each mutation rate is tested across 20 independent runs.

## GA Components

- Tournament selection
- Single-point crossover
- Elitism
- Uniform integer mutation
- Fixed random seed for reproducibility

## Outputs

The experiment generates summary statistics and figures showing:

- Best fitness over generations
- Mean population fitness
- Success rate
- Final best fitness distribution
- Grid-world layout
