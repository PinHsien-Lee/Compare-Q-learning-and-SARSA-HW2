## Why

Compare Q-Learning and SARSA in the Cliff Walking environment to understand the fundamental differences between off-policy and on-policy reinforcement learning algorithms. This experiment highlights how each algorithm handles exploration and risk, especially in environments with high-cost transitions (like falling off a cliff).

## What Changes

Implement a comprehensive comparison framework for Cliff Walking:
- Implement Q-Learning (Off-policy) and SARSA (On-policy) agents.
- Create an experiment runner to execute multiple runs and episodes.
- Track metrics including episodic rewards, steps per episode, and final learned policies.
- Generate visualizations (Reward curves, Q-value heatmaps, Policy paths).
- Document findings in a comparative analysis report.

## Capabilities

### New Capabilities
- `algorithm-comparison`: Core logic for running Q-Learning vs SARSA experiments and capturing comparative metrics.
- `visualization-engine`: Tools for generating reward curves, heatmaps, and path plots.
- `report-generator`: Automated generation of the experimental analysis report.
- `dev-automation`: Scripts for streamlining the development workflow (startup and teardown).

### Modified Capabilities
- `cliff-walking-env`: Ensure the Cliff Walking environment is properly configured for the experiment (this might already exist in standard libraries, but we'll ensure our implementation is compatible).

## Impact

- `cliff_walking.py`: Main implementation file.
- `results/`: Directory for storing output plots and the final report.
