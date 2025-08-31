import pandas as pd
import numpy as np
import random

# --- Genetic Algorithm for Portfolio Optimization ---
# This script uses a simple genetic algorithm to find an optimal portfolio
# by maximizing the Sharpe Ratio.

# --- Parameters ---
POPULATION_SIZE = 100
NUM_GENERATIONS = 20
MUTATION_RATE = 0.05
SELECTION_SIZE = 20

# --- Load and Prepare Data ---
try:
    # This simulates loading historical stock data for the NIFTY 100
    df = pd.read_csv('data/stock_data.csv', index_col='Date', parse_dates=True)
    stock_returns = df.pct_change().dropna()
except FileNotFoundError:
    print("Error: 'data/stock_data.csv' not found. Please create the file with stock data.")
    exit()

num_stocks = len(stock_returns.columns)

def calculate_sharpe_ratio(weights):
    """Calculates the Sharpe Ratio for a given set of weights."""
    if np.sum(weights) == 0:
        return 0
    # Annualized portfolio return
    annual_return = np.sum(stock_returns.mean() * weights) * 252
    # Annualized portfolio volatility
    annual_volatility = np.sqrt(np.dot(weights.T, np.dot(stock_returns.cov() * 252, weights)))
    # Assuming a risk-free rate of 0 for simplicity in this example
    sharpe_ratio = annual_return / annual_volatility
    return sharpe_ratio

def create_individual():
    """Generates a random portfolio (an individual in the population)."""
    weights = np.random.random(num_stocks)
    weights /= np.sum(weights)
    return weights

def initialize_population(size):
    """Creates the initial population of random portfolios."""
    return [create_individual() for _ in range(size)]

def select_parents(population):
    """Selects the best portfolios (parents) from the current generation."""
    # Rank portfolios by their Sharpe Ratio
    sorted_population = sorted(population, key=lambda x: calculate_sharpe_ratio(x), reverse=True)
    # Select the top portion of the population
    return sorted_population[:SELECTION_SIZE]

def crossover(parent1, parent2):
    """Combines two parents to create a new child portfolio."""
    child = np.copy(parent1)
    # Randomly select a crossover point
    crossover_point = random.randint(0, num_stocks - 1)
    child[crossover_point:] = parent2[crossover_point:]
    # Re-normalize weights
    child /= np.sum(child)
    return child

def mutate(individual):
    """Randomly mutates a small part of a portfolio's weights."""
    if random.random() < MUTATION_RATE:
        random_index = random.randint(0, num_stocks - 1)
        random_value = random.uniform(-0.1, 0.1)
        individual[random_index] += random_value
        # Ensure weights are positive and sum to 1
        individual = np.clip(individual, 0, 1)
        individual /= np.sum(individual)
    return individual

def run_genetic_algorithm():
    """Executes the genetic algorithm to find the optimal portfolio."""
    population = initialize_population(POPULATION_SIZE)
    best_sharpe = -1

    print("Running Genetic Algorithm...")
    for generation in range(NUM_GENERATIONS):
        parents = select_parents(population)
        next_generation = parents
        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            mutated_child = mutate(child)
            next_generation.append(mutated_child)
        population = next_generation
        
        # Track the best portfolio
        current_best = max(population, key=lambda x: calculate_sharpe_ratio(x))
        current_sharpe = calculate_sharpe_ratio(current_best)
        
        if current_sharpe > best_sharpe:
            best_sharpe = current_sharpe
            print(f"Generation {generation+1}: New best Sharpe Ratio = {best_sharpe:.4f}")

    # Return the best portfolio from the final generation
    final_best_portfolio = max(population, key=lambda x: calculate_sharpe_ratio(x))
    return final_best_portfolio, calculate_sharpe_ratio(final_best_portfolio)

if __name__ == "__main__":
    best_weights, best_sharpe = run_genetic_algorithm()
    print("\n--- Optimization Complete ---")
    print(f"Optimal Sharpe Ratio: {best_sharpe:.4f}")

