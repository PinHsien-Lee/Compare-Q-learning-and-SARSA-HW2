## ADDED Requirements

### Requirement: Q-Learning Update
The Q-Learning algorithm must update the action-value function based on the maximum possible reward from the next state, regardless of the action actually taken.

#### Scenario: Update step
- **WHEN** action $A$ is taken in state $S$ resulting in reward $R$ and next state $S'$
- **THEN** $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma \max_a Q(S', a) - Q(S, A)]$

### Requirement: SARSA Update
The SARSA algorithm must update the action-value function based on the actual next action $A'$ chosen in the next state $S'$ according to the current policy.

#### Scenario: Update step
- **WHEN** action $A$ is taken in state $S$ resulting in reward $R$, next state $S'$, and next action $A'$
- **THEN** $Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma Q(S', A') - Q(S, A)]$

### Requirement: Cliff Handling
Any step into a cliff cell (Row 3, Col 1-10 in 4x12 grid) must result in a reward of -100 and reset the agent to the start state (Row 3, Col 0).

#### Scenario: Stepping into cliff
- **WHEN** an agent moves to a cliff cell
- **THEN** reward is -100 and state is reset to (3, 0)
