## 01- Environment Setup

- [ ] 01-01 Implement `CliffWalkingEnv` class matching the 4x12 grid specification and cliff penalty rules.
- [ ] 01-02 Verify environment state transitions and reset behavior.

## 02- Agent Implementation

- [ ] 02-01 Implement `QLearningAgent` with e-greedy exploration and max-Q update logic.
- [ ] 02-02 Implement `SarsaAgent` with e-greedy exploration and actual-next-action update logic.

## 03- Experiment Execution

- [ ] 03-01 Create a training pipeline that supports running multiple independent trials.
- [ ] 03-02 Run the full experiment (50 trials, 500 episodes each) for both agents.
- [ ] 03-03 Save the resulting reward data and final Q-tables to disk.

## 04- Visualization & Analysis

- [ ] 04-01 Generate comparison plot for episodic rewards (mean + std deviation shading).
- [ ] 04-02 Create heatmaps for learned Q-values to visualize action preferences.
- [ ] 04-03 Plot the greedy paths found by each algorithm.
- [ ] 04-04 Compile all findings into a Markdown report (`results/analysis_report.md`).

## 05- Development Automation

- [x] 05-01 Create `startup.sh` for pulling code, reading handover, and suggesting actions.
- [x] 05-02 Create `ending.sh` for updating tasks, archiving changes, writing handover, and pushing code.
- [x] 05-03 Configure Git remote to `https://github.com/PinHsien-Lee/Compare-Q-learning-and-SARSA`.
