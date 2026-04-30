"""
Generate publication-quality plots for Monte Carlo Gridworld RL project.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import copy
import os

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Environment ──────────────────────────────────────────────


class GridWorld:
    def __init__(self):
        self.qTable = None
        self.actionSpace = ('U', 'D', 'L', 'R')
        self.actions = {
            (0, 0): ('D', 'R'), (0, 1): ('L', 'D', 'R'), (0, 2): ('L', 'D', 'R'), (0, 3): ('L', 'D'),
            (1, 0): ('U', 'D', 'R'), (1, 1): ('U', 'L', 'D', 'R'), (1, 2): ('U', 'L', 'D', 'R'), (1, 3): ('U', 'L', 'D'),
            (2, 0): ('U', 'D', 'R'), (2, 1): ('U', 'L', 'D', 'R'), (2, 2): ('U', 'L', 'D', 'R'), (2, 3): ('U', 'L', 'D'),
            (3, 0): ('U', 'R'), (3, 1): ('U', 'L', 'R'), (3, 2): ('U', 'L', 'R')
        }
        self.rewards = {(3, 3): 0.03, (1, 3): -0.01,
                        (2, 1): -0.011, (3, 1): -0.01}
        self.explored = 0
        self.exploited = 0
        self.initialQtable()

    def initialQtable(self):
        self.qTable = {}
        for s in self.actions:
            self.qTable[s] = {a: 0 for a in self.actions[s]}

    def updateQtable(self, newQ, updateRate=0.05):
        for s in self.qTable:
            for a in self.qTable[s]:
                self.qTable[s][a] += updateRate * \
                    (newQ[s][a] - self.qTable[s][a])

    def getRandomPolicy(self):
        return {s: np.random.choice(self.actions[s]) for s in self.actions}

    def reset(self):
        return (0, 0)

    def is_terminal(self, s):
        return s not in self.actions

    def getNewState(self, state, action):
        r, c = state
        if action == 'U':
            r -= 1
        elif action == 'D':
            r += 1
        elif action == 'L':
            c -= 1
        elif action == 'R':
            c += 1
        return (r, c)

    def chooseAction(self, state, policy, exploreRate):
        if exploreRate > np.random.rand():
            self.explored += 1
            return np.random.choice(self.actions[state])
        self.exploited += 1
        return policy[state]

    def move(self, state, policy, exploreRate):
        action = self.chooseAction(state, policy, exploreRate)
        ns = self.getNewState(state, action)
        return action, ns, self.rewards.get(ns, 0)


# ── Training with logging ───────────────────────────────────
np.random.seed(42)
env = GridWorld()
policy = env.getRandomPolicy()

policy_snapshots = {}
q_history = []  # max Q per outer iter
convergence_delta = []

NUM_OUTER = 150
NUM_INNER = 300

for i in range(NUM_OUTER):
    estimatedQ = copy.deepcopy(env.qTable)
    for s in estimatedQ:
        for a in estimatedQ[s]:
            estimatedQ[s][a] = 0
    collected = 0
    for j in range(NUM_INNER):
        traj = []
        state = env.reset()
        steps = 0
        while not env.is_terminal(state) and steps < 30:
            action, ns, reward = env.move(state, policy, exploreRate=0.05)
            traj.append(((state, action), reward))
            state = ns
            steps += 1
        collected += 1
        rewards = 0
        for item in reversed(traj):
            q, reward = zip(item)
            rewards += 0.9 * reward[0]
            estimatedQ[q[0][0]][q[0][1]
                                ] += (1/collected) * (rewards - estimatedQ[q[0][0]][q[0][1]])
        old_q = copy.deepcopy(env.qTable)
        env.updateQtable(estimatedQ)
        for s in policy:
            policy[s] = max(env.qTable[s], key=env.qTable[s].get)

    # metrics
    max_q = max(max(env.qTable[s].values()) for s in env.qTable)
    q_history.append(max_q)
    if i > 0:
        delta = sum(abs(env.qTable[s][a] - old_q[s][a])
                    for s in env.qTable for a in env.qTable[s])
        convergence_delta.append(delta)

    if i in [0, 5, 25, 75, 149]:
        policy_snapshots[i] = copy.deepcopy(policy)

print("Training complete.")
final_policy = copy.deepcopy(policy)
final_q = copy.deepcopy(env.qTable)

# ── PLOT 1: Gridworld Environment Diagram ───────────────────
fig, ax = plt.subplots(figsize=(6, 6))
colors = {
    'start': '#4CAF50', 'goal': '#2196F3', 'hole': '#F44336', 'normal': '#FAFAFA'
}
labels = {}
cell_colors = {}
for r in range(4):
    for c in range(4):
        if (r, c) == (0, 0):
            cell_colors[(r, c)] = colors['start']
            labels[(r, c)] = 'S\n(Start)'
        elif (r, c) == (3, 3):
            cell_colors[(r, c)] = colors['goal']
            labels[(r, c)] = 'T\n(Goal)\n+0.03'
        elif (r, c) in [(1, 3), (2, 1), (3, 1)]:
            cell_colors[(r, c)] = colors['hole']
            labels[(r, c)] = '✕\n(Penalty)'
        else:
            cell_colors[(r, c)] = colors['normal']
            labels[(r, c)] = ''

for r in range(4):
    for c in range(4):
        rect = mpatches.FancyBboxPatch((c, 3-r), 0.92, 0.92, boxstyle="round,pad=0.04",
                                       facecolor=cell_colors[(r, c)], edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)
        txt_color = 'white' if cell_colors[(
            r, c)] != colors['normal'] else '#555'
        ax.text(c+0.46, 3-r+0.46, labels[(r, c)], ha='center', va='center',
                fontsize=11, fontweight='bold', color=txt_color,
                path_effects=[pe.withStroke(linewidth=2, foreground='black')] if txt_color == 'white' else [])

ax.set_xlim(-0.1, 4.1)
ax.set_ylim(-0.1, 4.1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('4×4 GridWorld Environment',
             fontsize=16, fontweight='bold', pad=15)

legend_elements = [mpatches.Patch(fc=colors['start'], ec='#333', label='Start'),
                   mpatches.Patch(fc=colors['goal'],
                                  ec='#333', label='Goal (+reward)'),
                   mpatches.Patch(
                       fc=colors['hole'], ec='#333', label='Penalty (−reward)'),
                   mpatches.Patch(fc=colors['normal'], ec='#333', label='Normal cell')]
ax.legend(handles=legend_elements, loc='upper center',
          bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=9)
fig.savefig(os.path.join(ASSETS_DIR, 'gridworld_environment.png'))
plt.close()
print("Saved: gridworld_environment.png")

# ── PLOT 2: Policy Evolution Grid ────────────────────────────
arrow_map = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→'}
snap_keys = sorted(policy_snapshots.keys())
fig, axes = plt.subplots(1, len(snap_keys), figsize=(3.2*len(snap_keys), 3.5))
for idx, k in enumerate(snap_keys):
    ax = axes[idx]
    p = policy_snapshots[k]
    for r in range(4):
        for c in range(4):
            col = cell_colors[(r, c)]
            rect = mpatches.FancyBboxPatch((c, 3-r), 0.92, 0.92, boxstyle="round,pad=0.02",
                                           facecolor=col, edgecolor='#555', linewidth=1)
            ax.add_patch(rect)
            if (r, c) == (3, 3):
                ax.text(c+0.46, 3-r+0.46, 'T', ha='center', va='center',
                        fontsize=14, fontweight='bold', color='white')
            elif (r, c) in p:
                ax.text(c+0.46, 3-r+0.46, arrow_map[p[(r, c)]], ha='center', va='center', fontsize=18, fontweight='bold',
                        color='#222' if col == colors['normal'] else 'white')
    ax.set_xlim(-0.05, 4.05)
    ax.set_ylim(-0.05, 4.05)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Iter {k}', fontsize=11, fontweight='bold')

fig.suptitle('Policy Evolution During Training',
             fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, 'policy_evolution.png'))
plt.close()
print("Saved: policy_evolution.png")

# ── PLOT 3: Q-value Convergence ──────────────────────────────
fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.plot(q_history, color='#1565C0', linewidth=1.5, alpha=0.85)
ax1.set_xlabel('Training Iteration (outer loop)')
ax1.set_ylabel('Max Q-value', color='#1565C0')
ax1.tick_params(axis='y', labelcolor='#1565C0')
ax1.set_title('Q-Value Convergence & Policy Stability',
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(range(1, len(convergence_delta)+1), convergence_delta,
         color='#E65100', linewidth=1, alpha=0.7)
ax2.set_ylabel('ΔQ (sum of absolute changes)', color='#E65100')
ax2.tick_params(axis='y', labelcolor='#E65100')
fig.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, 'convergence.png'))
plt.close()
print("Saved: convergence.png")

# ── PLOT 4: Final Optimal Policy with Q-value Heatmap ────────
fig, ax = plt.subplots(figsize=(7, 7))
q_vals = np.zeros((4, 4))
for r in range(4):
    for c in range(4):
        if (r, c) in final_q:
            q_vals[r][c] = max(final_q[(r, c)].values())
        elif (r, c) == (3, 3):
            q_vals[r][c] = 0.03

im = ax.imshow(q_vals, cmap='RdYlGn', origin='upper', aspect='equal')
for r in range(4):
    for c in range(4):
        if (r, c) == (3, 3):
            ax.text(c, r, '★\nGOAL', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='#1A237E')
        elif (r, c) in final_policy:
            ax.text(c, r, arrow_map[final_policy[(r, c)]], ha='center',
                    va='center', fontsize=26, fontweight='bold', color='#1A237E')
            ax.text(c, r+0.35, f'{q_vals[r][c]:.2f}',
                    ha='center', va='center', fontsize=8, color='#333')

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(['Col 0', 'Col 1', 'Col 2', 'Col 3'])
ax.set_yticklabels(['Row 0', 'Row 1', 'Row 2', 'Row 3'])
ax.set_title('Learned Optimal Policy & Q-Value Heatmap',
             fontsize=15, fontweight='bold', pad=12)
plt.colorbar(im, ax=ax, label='Max Q-value', shrink=0.8)
for r in range(5):
    ax.axhline(r-0.5, color='white', linewidth=2)
for c in range(5):
    ax.axvline(c-0.5, color='white', linewidth=2)
fig.savefig(os.path.join(ASSETS_DIR, 'optimal_policy_heatmap.png'))
plt.close()
print("Saved: optimal_policy_heatmap.png")

# ── PLOT 5: Exploration vs Exploitation Pie ──────────────────
fig, ax = plt.subplots(figsize=(5, 5))
sizes = [env.exploited, env.explored]
labels_pie = [f'Exploited\n({env.exploited:,})',
              f'Explored\n({env.explored:,})']
colors_pie = ['#1565C0', '#FF8F00']
wedges, texts, autotexts = ax.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
                                  startangle=90, textprops={'fontsize': 11}, pctdistance=0.75,
                                  wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax.set_title('Exploration vs Exploitation\n(ε-greedy, ε = 0.05)',
             fontsize=13, fontweight='bold')
fig.savefig(os.path.join(ASSETS_DIR, 'exploration_exploitation.png'))
plt.close()
print("Saved: exploration_exploitation.png")

# ── PLOT 6: Optimal Path Trace ───────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
for r in range(4):
    for c in range(4):
        rect = mpatches.FancyBboxPatch((c, 3-r), 0.92, 0.92, boxstyle="round,pad=0.04",
                                       facecolor=cell_colors[(r, c)], edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)

# trace optimal path
state = (0, 0)
path = [state]
visited = set()
while state != (3, 3) and state in final_policy and state not in visited:
    visited.add(state)
    a = final_policy[state]
    r, c = state
    if a == 'U':
        r -= 1
    elif a == 'D':
        r += 1
    elif a == 'L':
        c -= 1
    elif a == 'R':
        c += 1
    state = (r, c)
    path.append(state)

for i in range(len(path)-1):
    r1, c1 = path[i]
    r2, c2 = path[i+1]
    ax.annotate('', xy=(c2+0.46, 3-r2+0.46), xytext=(c1+0.46, 3-r1+0.46),
                arrowprops=dict(arrowstyle='->', color='#1A237E', lw=3))
    ax.text(c1+0.46, 3-r1+0.15, str(i+1), ha='center', va='center', fontsize=9,
            fontweight='bold', color='#1A237E',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#1A237E', alpha=0.8))

ax.text(path[-1][1]+0.46, 3-path[-1][0]+0.46, '★',
        ha='center', va='center', fontsize=24, color='gold')
ax.set_xlim(-0.1, 4.1)
ax.set_ylim(-0.1, 4.1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title(f'Optimal Path Trace ({len(path)-1} steps)',
             fontsize=15, fontweight='bold', pad=12)
fig.savefig(os.path.join(ASSETS_DIR, 'optimal_path.png'))
plt.close()
print("Saved: optimal_path.png")

print("\n✓ All plots generated in ./assets/")
