"""
Deep Reinforcement Learning HW2
Q-Learning vs SARSA on Cliff Walking Environment

Author: Student
Date: 2025-04-15
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

# ============================================================
# 1. Cliff Walking Environment
# ============================================================

class CliffWalkingEnv:
    """
    4×12 Cliff Walking Gridworld Environment.
    
    Layout:
        Row 0 (top)    : normal cells
        Row 1          : normal cells
        Row 2          : normal cells
        Row 3 (bottom) : [Start] [Cliff...Cliff] [Goal]
    
    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    """
    
    def __init__(self, rows=4, cols=12):
        self.rows = rows
        self.cols = cols
        self.start = (rows - 1, 0)          # Bottom-left
        self.goal = (rows - 1, cols - 1)    # Bottom-right
        # Cliff: bottom row, columns 1 to cols-2
        self.cliff = set((rows - 1, c) for c in range(1, cols - 1))
        self.n_states = rows * cols
        self.n_actions = 4  # Up, Right, Down, Left
        self.action_names = ['↑', '→', '↓', '←']
        self.action_deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.state = self.start
    
    def reset(self):
        """Reset environment to start state."""
        self.state = self.start
        return self.state
    
    def step(self, action):
        """
        Take an action, return (next_state, reward, done).
        """
        dr, dc = self.action_deltas[action]
        r, c = self.state
        nr, nc = r + dr, c + dc
        
        # Boundary check: stay in place if hitting wall
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            next_state = (nr, nc)
        else:
            next_state = self.state
        
        # Check outcomes
        if next_state in self.cliff:
            # Fell off cliff: large penalty, return to start
            return self.start, -100, False
        elif next_state == self.goal:
            # Reached goal: episode ends
            self.state = next_state
            return next_state, -1, True
        else:
            # Normal move
            self.state = next_state
            return next_state, -1, False
    
    def state_to_index(self, state):
        """Convert (row, col) to flat index."""
        return state[0] * self.cols + state[1]
    
    def index_to_state(self, idx):
        """Convert flat index to (row, col)."""
        return (idx // self.cols, idx % self.cols)


# ============================================================
# 2. Agent with ε-greedy Policy
# ============================================================

class RLAgent:
    """Base RL agent with Q-table and ε-greedy policy."""
    
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha        # Learning rate
        self.gamma = gamma        # Discount factor
        self.epsilon = epsilon    # Exploration rate
        # Initialize Q-table to zeros
        self.Q = np.zeros((n_states, n_actions))
    
    def choose_action(self, state_idx):
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            # Break ties randomly
            q_values = self.Q[state_idx]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
    
    def greedy_action(self, state_idx):
        """Greedy action selection (for policy visualization)."""
        q_values = self.Q[state_idx]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return best_actions[0]


# ============================================================
# 3. Q-Learning (Off-Policy)
# ============================================================

def q_learning(env, agent, n_episodes=500):
    """
    Q-Learning algorithm (Off-Policy TD Control).
    
    Update rule:
        Q(S, A) ← Q(S, A) + α [R + γ max_a Q(S', a) - Q(S, A)]
    """
    rewards_per_episode = []
    
    for episode in range(n_episodes):
        state = env.reset()
        s_idx = env.state_to_index(state)
        total_reward = 0
        
        while True:
            # Choose action using ε-greedy
            action = agent.choose_action(s_idx)
            
            # Take action
            next_state, reward, done = env.step(action)
            ns_idx = env.state_to_index(next_state)
            total_reward += reward
            
            # Q-Learning update: use max over next state actions
            best_next_q = np.max(agent.Q[ns_idx])
            td_target = reward + agent.gamma * best_next_q
            td_error = td_target - agent.Q[s_idx, action]
            agent.Q[s_idx, action] += agent.alpha * td_error
            
            # Update state (handle cliff: agent was sent back to start)
            if next_state == env.start and state != env.start and not done:
                # Fell off cliff
                s_idx = env.state_to_index(env.start)
                env.state = env.start
            else:
                s_idx = ns_idx
            
            state = next_state
            
            if done:
                break
        
        rewards_per_episode.append(total_reward)
    
    return rewards_per_episode


# ============================================================
# 4. SARSA (On-Policy)
# ============================================================

def sarsa(env, agent, n_episodes=500):
    """
    SARSA algorithm (On-Policy TD Control).
    
    Update rule:
        Q(S, A) ← Q(S, A) + α [R + γ Q(S', A') - Q(S, A)]
    """
    rewards_per_episode = []
    
    for episode in range(n_episodes):
        state = env.reset()
        s_idx = env.state_to_index(state)
        action = agent.choose_action(s_idx)
        total_reward = 0
        
        while True:
            # Take action
            next_state, reward, done = env.step(action)
            ns_idx = env.state_to_index(next_state)
            total_reward += reward
            
            # Choose next action using ε-greedy (for SARSA update)
            next_action = agent.choose_action(ns_idx)
            
            # SARSA update: use actual next action
            td_target = reward + agent.gamma * agent.Q[ns_idx, next_action]
            td_error = td_target - agent.Q[s_idx, action]
            agent.Q[s_idx, action] += agent.alpha * td_error
            
            # Update state and action
            if next_state == env.start and state != env.start and not done:
                # Fell off cliff
                s_idx = env.state_to_index(env.start)
                env.state = env.start
                action = agent.choose_action(s_idx)
            else:
                s_idx = ns_idx
                action = next_action
            
            state = next_state
            
            if done:
                break
        
        rewards_per_episode.append(total_reward)
    
    return rewards_per_episode


# ============================================================
# 5. Visualization Functions
# ============================================================

def smooth_rewards(rewards, window=10):
    """Apply moving average smoothing to reward curve."""
    smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
    return smoothed


def plot_reward_curves(q_rewards_all, sarsa_rewards_all, save_path='results'):
    """
    Plot average reward curves for Q-Learning and SARSA,
    with individual run shading.
    """
    os.makedirs(save_path, exist_ok=True)
    
    q_mean = np.mean(q_rewards_all, axis=0)
    sarsa_mean = np.mean(sarsa_rewards_all, axis=0)
    q_std = np.std(q_rewards_all, axis=0)
    sarsa_std = np.std(sarsa_rewards_all, axis=0)
    
    episodes = np.arange(1, len(q_mean) + 1)
    
    # Smoothed version
    window = 10
    q_smooth = smooth_rewards(q_mean, window)
    sarsa_smooth = smooth_rewards(sarsa_mean, window)
    episodes_smooth = np.arange(window, len(q_mean) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ---- Raw reward curves ----
    ax1 = axes[0]
    ax1.fill_between(episodes, q_mean - q_std, q_mean + q_std,
                     alpha=0.15, color='#E74C3C')
    ax1.fill_between(episodes, sarsa_mean - sarsa_std, sarsa_mean + sarsa_std,
                     alpha=0.15, color='#3498DB')
    ax1.plot(episodes, q_mean, color='#E74C3C', linewidth=1.5, label='Q-Learning')
    ax1.plot(episodes, sarsa_mean, color='#3498DB', linewidth=1.5, label='SARSA')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward per Episode', fontsize=12)
    ax1.set_title('Raw Reward Curves (Averaged over multiple runs)', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.set_ylim(-200, 0)
    ax1.grid(True, alpha=0.3)
    
    # ---- Smoothed reward curves ----
    ax2 = axes[1]
    ax2.plot(episodes_smooth, q_smooth, color='#E74C3C', linewidth=2, label='Q-Learning')
    ax2.plot(episodes_smooth, sarsa_smooth, color='#3498DB', linewidth=2, label='SARSA')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Total Reward per Episode', fontsize=12)
    ax2.set_title(f'Smoothed Reward Curves (window={window})', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.set_ylim(-200, 0)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Q-Learning vs SARSA \u2014 Cliff Walking (4\u00d712)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'reward_curves.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Reward curves saved to {save_path}/reward_curves.png")


def plot_policy(env, agent, title, filename, save_path='results'):
    """Visualize the learned policy on the gridworld."""
    os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Arrow symbols for each action
    arrow_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    arrow_dx = {0: 0, 1: 0.3, 2: 0, 3: -0.3}
    arrow_dy = {0: 0.3, 1: 0, 2: -0.3, 3: 0}
    
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            
            # Determine cell color
            if state == env.start:
                color = '#27AE60'  # Green for start
            elif state == env.goal:
                color = '#F39C12'  # Gold for goal
            elif state in env.cliff:
                color = '#E74C3C'  # Red for cliff
            else:
                color = '#ECF0F1'  # Light gray for normal
            
            # Draw cell
            rect = plt.Rectangle((c, env.rows - 1 - r), 1, 1,
                                 facecolor=color, edgecolor='#2C3E50',
                                 linewidth=1.5)
            ax.add_patch(rect)
            
            # Draw arrow for non-terminal, non-cliff cells
            if state not in env.cliff and state != env.goal:
                s_idx = env.state_to_index(state)
                best_action = agent.greedy_action(s_idx)
                cx, cy = c + 0.5, env.rows - 1 - r + 0.5
                dx, dy = arrow_dx[best_action], arrow_dy[best_action]
                ax.annotate('', xy=(cx + dx, cy + dy),
                           xytext=(cx, cy),
                           arrowprops=dict(arrowstyle='->', color='#2C3E50',
                                          lw=2.0, mutation_scale=18))
            
            # Labels
            if state == env.start:
                ax.text(c + 0.5, env.rows - 1 - r + 0.15, 'Start',
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color='white')
            elif state == env.goal:
                ax.text(c + 0.5, env.rows - 1 - r + 0.5, 'Goal',
                       ha='center', va='center', fontsize=9, fontweight='bold',
                       color='white')
            elif state in env.cliff:
                ax.text(c + 0.5, env.rows - 1 - r + 0.5, 'Cliff',
                       ha='center', va='center', fontsize=7,
                       color='white', alpha=0.7)
    
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#27AE60', edgecolor='#2C3E50', label='Start'),
        mpatches.Patch(facecolor='#F39C12', edgecolor='#2C3E50', label='Goal'),
        mpatches.Patch(facecolor='#E74C3C', edgecolor='#2C3E50', label='Cliff'),
        mpatches.Patch(facecolor='#ECF0F1', edgecolor='#2C3E50', label='Normal'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Policy visualization saved to {save_path}/{filename}")


def plot_optimal_path(env, agent, title, filename, save_path='results'):
    """
    Trace and visualize the greedy path from Start to Goal.
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Trace the greedy path
    path = []
    state = env.start
    path.append(state)
    max_steps = 100
    
    for _ in range(max_steps):
        if state == env.goal:
            break
        s_idx = env.state_to_index(state)
        action = agent.greedy_action(s_idx)
        dr, dc = env.action_deltas[action]
        r, c = state
        nr, nc = r + dr, c + dc
        if 0 <= nr < env.rows and 0 <= nc < env.cols:
            state = (nr, nc)
        path.append(state)
        if state in env.cliff:
            break
    
    path_set = set(path)
    
    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if state == env.start:
                color = '#27AE60'
            elif state == env.goal:
                color = '#F39C12'
            elif state in env.cliff:
                color = '#E74C3C'
            elif state in path_set:
                color = '#85C1E9'  # Light blue for path
            else:
                color = '#ECF0F1'
            
            rect = plt.Rectangle((c, env.rows - 1 - r), 1, 1,
                                 facecolor=color, edgecolor='#2C3E50',
                                 linewidth=1.5)
            ax.add_patch(rect)
            
            if state == env.start:
                ax.text(c + 0.5, env.rows - 1 - r + 0.5, 'S',
                       ha='center', va='center', fontsize=12, fontweight='bold',
                       color='white')
            elif state == env.goal:
                ax.text(c + 0.5, env.rows - 1 - r + 0.5, 'G',
                       ha='center', va='center', fontsize=12, fontweight='bold',
                       color='white')
            elif state in env.cliff:
                ax.text(c + 0.5, env.rows - 1 - r + 0.5, '✕',
                       ha='center', va='center', fontsize=10,
                       color='white', alpha=0.5)
    
    # Draw path as a connected line
    if len(path) > 1:
        path_x = [c + 0.5 for (r, c) in path]
        path_y = [env.rows - 1 - r + 0.5 for (r, c) in path]
        ax.plot(path_x, path_y, 'o-', color='#2C3E50', linewidth=2.5,
               markersize=6, markerfacecolor='#3498DB', markeredgecolor='#2C3E50',
               zorder=5)
    
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_aspect('equal')
    ax.set_title(f'{title}\n(Path length: {len(path)} steps)', fontsize=13, fontweight='bold', pad=10)
    ax.axis('off')
    
    legend_elements = [
        mpatches.Patch(facecolor='#27AE60', edgecolor='#2C3E50', label='Start'),
        mpatches.Patch(facecolor='#F39C12', edgecolor='#2C3E50', label='Goal'),
        mpatches.Patch(facecolor='#E74C3C', edgecolor='#2C3E50', label='Cliff'),
        mpatches.Patch(facecolor='#85C1E9', edgecolor='#2C3E50', label='Optimal Path'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Optimal path saved to {save_path}/{filename}")


def plot_q_value_heatmap(env, agent, title, filename, save_path='results'):
    """Visualize the max Q-value for each state as a heatmap."""
    os.makedirs(save_path, exist_ok=True)
    
    q_max = np.zeros((env.rows, env.cols))
    for r in range(env.rows):
        for c in range(env.cols):
            s_idx = env.state_to_index((r, c))
            q_max[r, c] = np.max(agent.Q[s_idx])
    
    # Mask cliff cells
    masked_q = np.ma.array(q_max)
    for (r, c) in env.cliff:
        masked_q[r, c] = np.ma.masked
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    cmap = LinearSegmentedColormap.from_list('custom', 
        ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71', '#27AE60'])
    
    im = ax.imshow(masked_q, cmap=cmap, aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Max Q-value', shrink=0.8)
    
    # Mark special cells
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) == env.start:
                ax.text(c, r, 'S', ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')
            elif (r, c) == env.goal:
                ax.text(c, r, 'G', ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')
            elif (r, c) in env.cliff:
                ax.text(c, r, '✕', ha='center', va='center',
                       fontsize=10, color='gray')
            else:
                val = q_max[r, c]
                ax.text(c, r, f'{val:.1f}', ha='center', va='center',
                       fontsize=7, color='black')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Q-value heatmap saved to {save_path}/{filename}")


def plot_stability_analysis(q_rewards_all, sarsa_rewards_all, save_path='results'):
    """
    Analyze and visualize learning stability:
    rolling standard deviation of rewards.
    """
    os.makedirs(save_path, exist_ok=True)
    
    q_mean = np.mean(q_rewards_all, axis=0)
    sarsa_mean = np.mean(sarsa_rewards_all, axis=0)
    
    window = 20
    
    def rolling_std(data, w):
        result = []
        for i in range(len(data) - w + 1):
            result.append(np.std(data[i:i+w]))
        return np.array(result)
    
    q_rolling_std = rolling_std(q_mean, window)
    sarsa_rolling_std = rolling_std(sarsa_mean, window)
    episodes = np.arange(window, len(q_mean) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(episodes, q_rolling_std, color='#E74C3C', linewidth=2, label='Q-Learning')
    ax.plot(episodes, sarsa_rolling_std, color='#3498DB', linewidth=2, label='SARSA')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(f'Rolling Std (window={window})', fontsize=12)
    ax.set_title('Learning Stability Analysis \u2014 Rolling Standard Deviation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'stability_analysis.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Stability analysis saved to {save_path}/stability_analysis.png")


def plot_combined_policies(env, q_agent, sarsa_agent, save_path='results'):
    """Plot both policies side-by-side for direct comparison."""
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    
    arrow_dx = {0: 0, 1: 0.3, 2: 0, 3: -0.3}
    arrow_dy = {0: 0.3, 1: 0, 2: -0.3, 3: 0}
    
    for ax_idx, (agent, label) in enumerate([(q_agent, 'Q-Learning Policy'),
                                               (sarsa_agent, 'SARSA Policy')]):
        ax = axes[ax_idx]
        
        for r in range(env.rows):
            for c in range(env.cols):
                state = (r, c)
                
                if state == env.start:
                    color = '#27AE60'
                elif state == env.goal:
                    color = '#F39C12'
                elif state in env.cliff:
                    color = '#E74C3C'
                else:
                    color = '#ECF0F1'
                
                rect = plt.Rectangle((c, env.rows - 1 - r), 1, 1,
                                     facecolor=color, edgecolor='#2C3E50',
                                     linewidth=1.2)
                ax.add_patch(rect)
                
                if state not in env.cliff and state != env.goal:
                    s_idx = env.state_to_index(state)
                    best_action = agent.greedy_action(s_idx)
                    cx, cy = c + 0.5, env.rows - 1 - r + 0.5
                    dx, dy = arrow_dx[best_action], arrow_dy[best_action]
                    ax.annotate('', xy=(cx + dx, cy + dy),
                               xytext=(cx, cy),
                               arrowprops=dict(arrowstyle='->', color='#2C3E50',
                                              lw=1.8, mutation_scale=16))
                
                if state == env.start:
                    ax.text(c + 0.5, env.rows - 1 - r + 0.15, 'Start',
                           ha='center', va='center', fontsize=7, fontweight='bold',
                           color='white')
                elif state == env.goal:
                    ax.text(c + 0.5, env.rows - 1 - r + 0.5, 'Goal',
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           color='white')
                elif state in env.cliff:
                    ax.text(c + 0.5, env.rows - 1 - r + 0.5, 'Cliff',
                           ha='center', va='center', fontsize=6,
                           color='white', alpha=0.7)
        
        ax.set_xlim(0, env.cols)
        ax.set_ylim(0, env.rows)
        ax.set_aspect('equal')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.axis('off')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#27AE60', edgecolor='#2C3E50', label='Start'),
        mpatches.Patch(facecolor='#F39C12', edgecolor='#2C3E50', label='Goal'),
        mpatches.Patch(facecolor='#E74C3C', edgecolor='#2C3E50', label='Cliff'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
              ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle('Policy Comparison: Q-Learning vs SARSA',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'policy_comparison.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Policy comparison saved to {save_path}/policy_comparison.png")


# ============================================================
# 6. Main Experiment
# ============================================================

def main():
    # ---- Hyperparameters ----
    ALPHA = 0.1       # Learning rate
    GAMMA = 0.9       # Discount factor
    EPSILON = 0.1     # Exploration rate
    N_EPISODES = 500  # Training episodes
    N_RUNS = 50       # Number of independent runs for averaging
    
    SAVE_PATH = 'results'
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    print("=" * 60)
    print("   Q-Learning vs SARSA - Cliff Walking Experiment")
    print("=" * 60)
    print(f"\n  Hyperparameters:")
    print(f"    alpha (learning rate)   = {ALPHA}")
    print(f"    gamma (discount factor) = {GAMMA}")
    print(f"    epsilon (exploration)   = {EPSILON}")
    print(f"    Episodes                = {N_EPISODES}")
    print(f"    Independent runs        = {N_RUNS}")
    print(f"    Grid size               = 4 x 12")
    print()
    
    # Storage for all runs
    q_rewards_all = np.zeros((N_RUNS, N_EPISODES))
    sarsa_rewards_all = np.zeros((N_RUNS, N_EPISODES))
    
    # Keep agents from last run for policy visualization
    last_q_agent = None
    last_sarsa_agent = None
    
    for run in range(N_RUNS):
        # Q-Learning
        env_q = CliffWalkingEnv()
        q_agent = RLAgent(env_q.n_states, env_q.n_actions, ALPHA, GAMMA, EPSILON)
        q_rewards = q_learning(env_q, q_agent, N_EPISODES)
        q_rewards_all[run] = q_rewards
        
        # SARSA
        env_s = CliffWalkingEnv()
        sarsa_agent = RLAgent(env_s.n_states, env_s.n_actions, ALPHA, GAMMA, EPSILON)
        sarsa_rewards = sarsa(env_s, sarsa_agent, N_EPISODES)
        sarsa_rewards_all[run] = sarsa_rewards
        
        last_q_agent = q_agent
        last_sarsa_agent = sarsa_agent
        
        if (run + 1) % 10 == 0:
            print(f"  [Run {run + 1:3d}/{N_RUNS}] "
                  f"Q-Learning avg final reward: {np.mean(q_rewards[-50:]):.1f} | "
                  f"SARSA avg final reward: {np.mean(sarsa_rewards[-50:]):.1f}")
    
    print("\n" + "-" * 60)
    print("  Training Complete! Generating visualizations...")
    print("-" * 60 + "\n")
    
    # ---- Generate all plots ----
    env = CliffWalkingEnv()
    
    # 1. Reward curves
    plot_reward_curves(q_rewards_all, sarsa_rewards_all, SAVE_PATH)
    
    # 2. Policy visualizations (combined)
    plot_combined_policies(env, last_q_agent, last_sarsa_agent, SAVE_PATH)
    
    # 3. Individual policy plots
    plot_policy(env, last_q_agent, 'Q-Learning Learned Policy', 
                'q_learning_policy.png', SAVE_PATH)
    plot_policy(env, last_sarsa_agent, 'SARSA Learned Policy', 
                'sarsa_policy.png', SAVE_PATH)
    
    # 4. Optimal path traces
    plot_optimal_path(env, last_q_agent, 'Q-Learning Greedy Path', 
                      'q_learning_path.png', SAVE_PATH)
    plot_optimal_path(env, last_sarsa_agent, 'SARSA Greedy Path', 
                      'sarsa_path.png', SAVE_PATH)
    
    # 5. Q-value heatmaps
    plot_q_value_heatmap(env, last_q_agent, 'Q-Learning \u2014 Max Q-Values',
                         'q_learning_heatmap.png', SAVE_PATH)
    plot_q_value_heatmap(env, last_sarsa_agent, 'SARSA \u2014 Max Q-Values',
                         'sarsa_heatmap.png', SAVE_PATH)
    
    # 6. Stability analysis
    plot_stability_analysis(q_rewards_all, sarsa_rewards_all, SAVE_PATH)
    
    # ---- Print summary statistics ----
    print("\n" + "=" * 60)
    print("  Summary Statistics (last 50 episodes, averaged over runs)")
    print("=" * 60)
    
    q_final = np.mean(q_rewards_all[:, -50:])
    sarsa_final = np.mean(sarsa_rewards_all[:, -50:])
    q_final_std = np.std(np.mean(q_rewards_all[:, -50:], axis=1))
    sarsa_final_std = np.std(np.mean(sarsa_rewards_all[:, -50:], axis=1))
    
    print(f"\n  Q-Learning:")
    print(f"    Average final reward : {q_final:.2f} ± {q_final_std:.2f}")
    print(f"    Min episode reward   : {np.min(q_rewards_all[:, -50:]):.0f}")
    print(f"    Max episode reward   : {np.max(q_rewards_all[:, -50:]):.0f}")
    
    print(f"\n  SARSA:")
    print(f"    Average final reward : {sarsa_final:.2f} ± {sarsa_final_std:.2f}")
    print(f"    Min episode reward   : {np.min(sarsa_rewards_all[:, -50:]):.0f}")
    print(f"    Max episode reward   : {np.max(sarsa_rewards_all[:, -50:]):.0f}")
    
    # Convergence speed: episode where average reward first exceeds threshold
    threshold = -50
    q_convergence = np.where(np.mean(q_rewards_all, axis=0) > threshold)[0]
    sarsa_convergence = np.where(np.mean(sarsa_rewards_all, axis=0) > threshold)[0]
    
    q_conv_ep = q_convergence[0] + 1 if len(q_convergence) > 0 else 'N/A'
    sarsa_conv_ep = sarsa_convergence[0] + 1 if len(sarsa_convergence) > 0 else 'N/A'
    
    print(f"\n  Convergence (first episode avg reward > {threshold}):")
    print(f"    Q-Learning : Episode {q_conv_ep}")
    print(f"    SARSA      : Episode {sarsa_conv_ep}")
    
    print(f"\n  All results saved to: {os.path.abspath(SAVE_PATH)}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
