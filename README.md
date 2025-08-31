# Equities-Portfolio-Algorithm
This project presents a Python-based algorithmic model designed to construct and optimize an equities portfolio using data from the NIFTY 100 index. The primary objective was to outperform the market benchmark by systematically selecting and weighting stocks for superior risk-adjusted returns.
Methodology
The core of this project is a portfolio optimization algorithm built in Python. It leverages a Genetic Algorithm to find the optimal combination of stock weights. This evolutionary approach efficiently explores a vast solution space to identify portfolios that maximize the Sharpe Ratio—a key metric for risk-adjusted performance. The model was rigorously back-tested on two years of historical data to validate its effectiveness under real market conditions.

Key Outcomes & Performance
Portfolio Over-performance: The optimized portfolio demonstrated a substantial over-performance against the Nifty 100 benchmark.

Return: Generated a 40% total return over the backtesting period.

Risk-Adjusted Return: Achieved a 52% higher Sharpe Ratio compared to the benchmark, indicating superior returns for the level of risk taken.
Repository Structure
run_optimization.py: The main script containing the Genetic Algorithm logic and portfolio construction.

data/: Directory containing the historical stock data used for backtesting.

results/: Directory for outputting performance metrics and portfolio visualizations.

README.md: This file, providing an overview of the project.
