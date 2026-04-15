## Context

The Cliff Walking problem is a standard reinforcement learning task where an agent must navigate a grid from a start to a goal while avoiding a "cliff" that resets the agent and provides a large negative reward (-100). This setup is ideal for comparing Q-Learning and SARSA because it highlights the difference between learning an optimal policy (Q-Learning) versus a safe policy (SARSA) under exploration.

## Goals / Non-Goals

**Goals:**
- Implement reusable `QLearningAgent` and `SarsaAgent` classes.
- Create an automated experiment pipeline that runs agents for 500 episodes across multiple independent runs.
- Collect metrics: average reward per episode, total steps taken, and final Q-table.
- Produce visualizations:
    - Comparison of learning curves (Reward vs. Episode).
    - Heatmaps of learned Q-values.
    - Visual representation of the optimal path found by each agent using a greedy policy.

**Non-Goals:**
- Tuning hyperparameters for extreme performance (we use standard values: alpha=0.1, gamma=0.9, epsilon=0.1).
- Deep reinforcement learning or neural network implementations.
- Extensive GUI for the environment.

## Decisions

- **Hardware/Language**: Python 3.x with NumPy for numerical operations and Matplotlib for plotting.
- **Environment**: A custom `CliffWalkingEnv` class or existing implementations consistent with Sutton & Barto's definition (4x12 grid, cliff at row 3, columns 1-10).
- **Execution Flow**:
    1. Initialize Agents.
    2. Loop through 50 independent runs.
    3. In each run, train for 500 episodes.
    4. Store reward data in a NumPy array/Pandas DataFrame.
    5. Calculate mean and standard deviation across runs.
    6. Generate plots.

## Risks / Trade-offs

- **Exploration vs. Exploitation**: Using a fixed epsilon (0.1) means Q-Learning will frequently fall into the cliff even after learning the optimal path, resulting in lower episodic rewards compared to SARSA.
- **Stochasticity**: Single runs can be noisy; 50 runs are necessary for statistical significance.
