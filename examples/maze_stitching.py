"""
toy task showing the benefit of a learned quasimetric over a regular (unconstrained) distance space

the offline dataset is only short random walks on a branching maze, yet the quasimetric critic stitches them into the global shortest-path distance, while the regular space collapses

run

    python examples/maze_stitching.py
"""

import math
import numpy as np
import torch
from torch import nn
from collections import deque

from x_mlps_pytorch import MLP

from MQE import MQE, MRN

# experiment configuration

SEED = 0
EMBED_DIM = 32
HIDDEN_DIM = 64
CRITIC_STEPS = 6000
BATCH_SIZE = 64
NUM_SEGMENTS = 4000
SEG_LEN = 8
DISCOUNT_FACTOR = 0.99
WAYPOINT_DISCOUNT = 0.9

GOAL_CELL = (1, 1)

# a branching maze - reaching (1, 1) from (1, 7) requires backtracking

MAZE = [
    '#########',
    '#.......#',
    '#.#####.#',
    '#.#...#.#',
    '#.#.#.#.#',
    '#...#...#',
    '#########',
]

HEIGHT, WIDTH = len(MAZE), len(MAZE[0])

CELLS = [(r, c) for r in range(HEIGHT) for c in range(WIDTH) if MAZE[r][c] == '.']
CELL_TO_IDX = {p: i for i, p in enumerate(CELLS)}
IDX_TO_CELL = {i: p for p, i in CELL_TO_IDX.items()}
GOAL_IDX = CELL_TO_IDX[GOAL_CELL]

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def move(r, c, action):
    dr, dc = MOVES[action]
    nr, nc = r + dr, c + dc

    if 0 <= nr < HEIGHT and 0 <= nc < WIDTH and MAZE[nr][nc] == '.':
        return nr, nc

    return r, c

def encode(cell):
    r, c = cell
    return np.array([r / (HEIGHT - 1), c / (WIDTH - 1)], dtype = np.float32)

def bfs_distances(goal_cell):
    dist = {p: -1 for p in CELLS}
    dist[goal_cell] = 0
    queue = deque([goal_cell])

    while queue:
        p = queue.popleft()

        for action in range(4):
            neighbor = move(*p, action)

            if dist[neighbor] < 0:
                dist[neighbor] = dist[p] + 1
                queue.append(neighbor)

    return dist

# offline data - short random walks only

def collect_offline_data(seed):
    rng = np.random.RandomState(seed)
    segments_s, segments_a = [], []

    for _ in range(NUM_SEGMENTS):
        cell = CELLS[rng.randint(len(CELLS))]
        length = rng.randint(2, SEG_LEN + 1)

        seg_s, seg_a = [cell], []

        for _ in range(length):
            valid = [action for action in range(4) if move(*cell, action) != cell]
            action = int(rng.choice(valid))
            seg_a.append(action)
            cell = move(*cell, action)
            seg_s.append(cell)

        segments_s.append(seg_s)
        segments_a.append(seg_a)

    timesteps = SEG_LEN + 2

    states = np.zeros((len(segments_s), timesteps, 2), dtype = np.float32)
    actions = np.zeros((len(segments_s), timesteps, 4), dtype = np.float32)

    for i, (seg_s, seg_a) in enumerate(zip(segments_s, segments_a)):
        seg_s = seg_s + [seg_s[-1]] * (timesteps - len(seg_s))
        seg_a = seg_a + [0] * (timesteps - len(seg_a))

        for t, cell in enumerate(seg_s):
            states[i, t] = encode(cell)

        for t in range(timesteps):
            actions[i, t, seg_a[t]] = 1.

    return torch.tensor(states), torch.tensor(actions)

# distance spaces

class UnconstrainedSpace(nn.Module):
    """a regular scalar distance - no metric structure (repo ablation)"""

    def __init__(self, dim, hidden_dim = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, x, y, reduce_groups = True):
        shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1])
        xy = torch.cat([x.expand(*shape, -1), y.expand(*shape, -1)], dim = -1)
        out = self.net(xy).squeeze(-1)
        return out if reduce_groups else out.unsqueeze(-1)

def build_space(kind):
    if kind == 'mqe':
        return MRN(
            sym_network = MLP(EMBED_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM),
            asym_network = MLP(EMBED_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM),
            distance_groups = 4
        )

    if kind == 'regular':
        return UnconstrainedSpace(EMBED_DIM, HIDDEN_DIM)

    raise ValueError(f'unknown space {kind}')

# training - identical multistep quasimetric objective for both spaces

def train_critic(kind, states, actions, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    mqe = MQE(
        state_encoder = MLP(2, HIDDEN_DIM, HIDDEN_DIM, EMBED_DIM),
        state_action_encoder = MLP(2 + 4, HIDDEN_DIM, HIDDEN_DIM, EMBED_DIM),
        metric_residual_network = build_space(kind),
        discount_factor = DISCOUNT_FACTOR,
        waypoint_discount = WAYPOINT_DISCOUNT,
        action_invariance_loss_weight = 1.0,
        paired_loss_weight = 0.5
    )

    optimizer = torch.optim.Adam(mqe.parameters(), lr = 1e-3)
    num_segments = states.shape[0]

    for _ in range(CRITIC_STEPS):
        indices = torch.randint(0, num_segments, (BATCH_SIZE,))
        loss, _ = mqe(states[indices], actions[indices], states[indices])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mqe.parameters(), 1.0)
        optimizer.step()

    return mqe

# evaluation

@torch.no_grad()
def learned_distances(mqe, goal_cell):
    goal = torch.tensor([encode(goal_cell)])
    encoded_goal = mqe.critic.state_encoder(goal)

    all_states = torch.tensor(np.stack([encode(cell) for cell in CELLS]))
    encoded_states = mqe.critic.state_encoder(all_states)

    return mqe.critic.metric_residual_network(encoded_states, encoded_goal).numpy()

def spearman_correlation(x, y):
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    return np.corrcoef(rank_x, rank_y)[0, 1]

def greedy_control_success(distances):
    """move to a neighboring cell strictly reducing the learned distance"""

    num_cells = len(CELLS)
    successes = 0

    for start in range(num_cells):
        current = start

        for _ in range(4 * num_cells):
            if current == GOAL_IDX:
                break

            best_neighbor, best_dist = None, distances[current]

            for action in range(4):
                neighbor = CELL_TO_IDX[move(*IDX_TO_CELL[current], action)]

                if neighbor != current and distances[neighbor] < best_dist - 1e-9:
                    best_neighbor, best_dist = neighbor, distances[neighbor]

            if best_neighbor is None:
                break

            current = best_neighbor

        successes += int(current == GOAL_IDX)

    return successes

# plotting

def plot_results(true_dist, results, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    _, axes = plt.subplots(1, 4, figsize = (21, 5.2), dpi = 150)

    # rescaled learned distances share the units of the true distances
    steps_per_unit = abs(math.log(DISCOUNT_FACTOR))
    true_steps = true_dist
    mqe_steps = results['mqe']['distances'] / steps_per_unit
    regular_steps = results['regular']['distances'] / steps_per_unit
    shared_vmax = max(true_steps.max(), mqe_steps.max(), 1e-6)

    def draw_maze(ax, title):
        for r in range(HEIGHT):
            for c in range(WIDTH):
                if MAZE[r][c] == '#':
                    ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor = '#334155', edgecolor = '#1e293b'))
        gr, gc = GOAL_CELL
        ax.plot(gc, gr, marker = '*', markersize = 16, color = '#f43f5e', zorder = 5)
        ax.set_xlim(-0.5, WIDTH - 0.5)
        ax.set_ylim(HEIGHT - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize = 12, fontweight = 'bold', pad = 10)

    def draw_heat(ax, values, title):
        grid = np.full((HEIGHT, WIDTH), np.nan)
        for i, (r, c) in enumerate(CELLS):
            grid[r, c] = values[i]
        im = ax.imshow(grid, cmap = 'viridis', vmin = 0, vmax = shared_vmax)
        draw_maze(ax, title)
        cbar = plt.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04)
        cbar.set_label('distance (steps)', fontsize = 9)

    draw_heat(axes[0], true_steps, '1. True shortest-path distance')
    draw_heat(axes[1], mqe_steps, '2. MQE quasimetric MRN')
    draw_heat(axes[2], regular_steps, '3. Regular (unconstrained) space')

    ax = axes[3]
    labels = ['MQE\n(MRN)', 'Regular\n(unconstrained)']
    x = np.arange(2)
    success = np.array([results['mqe']['success'] / len(CELLS) * 100, results['regular']['success'] / len(CELLS) * 100])
    corr = np.array([results['mqe']['spearman'], results['regular']['spearman']]) * 100

    ax.bar(x - 0.2, success, width = 0.38, color = ['#10b981', '#dc2626'], label = 'Greedy control (% goals reached)')
    ax.bar(x + 0.2, corr, width = 0.38, color = ['#6ee7b7', '#fca5a5'], label = 'Rank correlation with true distance')

    for xpos, val in list(zip(x - 0.2, success)) + list(zip(x + 0.2, corr)):
        offset = 3 if val >= 0 else -10
        ax.text(xpos, val + offset, f'{val:.0f}', ha = 'center', fontsize = 10, fontweight = 'bold')

    ax.axhline(0, color = '#334155', lw = 1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(-115, 118)
    ax.set_ylabel('%', fontsize = 10)
    ax.set_title('4. Stitching evaluation', fontsize = 12, fontweight = 'bold', pad = 10)
    ax.legend(fontsize = 8, loc = 'lower left')
    ax.grid(axis = 'y', linestyle = '--', alpha = 0.4)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches = 'tight')
    plt.close()

# main

def main():
    print('collecting short random-walk segments...')
    states, actions = collect_offline_data(SEED)
    print(f'  {len(states)} segments, at most {SEG_LEN} steps each')

    true_dist_map = bfs_distances(GOAL_CELL)
    true_dist = np.array([true_dist_map[cell] for cell in CELLS], dtype = np.float32)

    results = {}

    for kind in ('mqe', 'regular'):
        print(f'training {kind} critic...')
        mqe = train_critic(kind, states, actions, SEED)

        distances = learned_distances(mqe, GOAL_CELL)
        success = greedy_control_success(distances)
        corr = spearman_correlation(true_dist, distances)

        results[kind] = dict(distances = distances, success = success, spearman = corr)
        print(f'  greedy control: {success}/{len(CELLS)}   rank correlation: {corr:.3f}')

    print()
    print('summary (goal cell = (1, 1), trained on local segments only)')
    print(f'  MQE quasimetric MRN : {results["mqe"]["success"]}/{len(CELLS)} goals, rank corr {results["mqe"]["spearman"]:.3f}')
    print(f'  regular space       : {results["regular"]["success"]}/{len(CELLS)} goals, rank corr {results["regular"]["spearman"]:.3f}')

    output_path = 'examples/maze_stitching.png'
    plot_results(true_dist, results, output_path)
    print(f'\nsaved figure to {output_path}')

if __name__ == '__main__':
    main()
