#!/usr/bin/env python3
"""Interactive endpoint selector for Q7 maze search.

Run:
    python3 q7_maze_interactive.py

A matplotlib window will open. Click a reachable cell in the maze; the script
will train a tabular Q-learning agent for that endpoint, compare the route
length with the BFS shortest path length, and draw the learned path.

For non-GUI use:
    python3 q7_maze_interactive.py --goal 8 45
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BLOCK = 8
WALL_THRESHOLD = 80


def load_maze():
    img_path = next((ROOT / "data").rglob("maze.jpg"))
    img = np.asarray(Image.open(img_path).convert("RGB"))
    gray = img.mean(axis=2)

    wall_pixels = gray > WALL_THRESHOLD
    yy, xx = np.where(wall_pixels)
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    crop = gray[y0:y1, x0:x1]
    rows = crop.shape[0] // BLOCK
    cols = crop.shape[1] // BLOCK
    block_mean = crop.reshape(rows, BLOCK, cols, BLOCK).mean(axis=(1, 3))
    occupancy = block_mean < WALL_THRESHOLD

    rgb = img.astype(int)
    yellow = (
        (rgb[:, :, 0] > 150)
        & (rgb[:, :, 1] > 150)
        & ((rgb[:, :, 0] + rgb[:, :, 1] - 2 * rgb[:, :, 2]) > 80)
    )
    sy, sx = np.where(yellow)
    if len(sy) == 0:
        raise RuntimeError("Could not detect the yellow start marker.")
    start = (int((sy.mean() - y0) // BLOCK), int((sx.mean() - x0) // BLOCK))
    occupancy[start] = True
    return occupancy, start


def bfs(occupancy, source):
    distances = np.full(occupancy.shape, -1, dtype=int)
    predecessors = {source: None}
    queue = deque([source])
    distances[source] = 0
    while queue:
        row, col = queue.popleft()
        for dr, dc in ACTIONS:
            nr, nc = row + dr, col + dc
            if (
                0 <= nr < occupancy.shape[0]
                and 0 <= nc < occupancy.shape[1]
                and occupancy[nr, nc]
                and distances[nr, nc] < 0
            ):
                distances[nr, nc] = distances[row, col] + 1
                predecessors[(nr, nc)] = (row, col)
                queue.append((nr, nc))
    return distances, predecessors


def reconstruct_path(predecessors, goal):
    if goal not in predecessors:
        return []
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = predecessors[current]
    return path[::-1]


def q_learning(occupancy, start, goal, episodes=900, alpha=0.35, gamma=0.95, epsilon=0.35, seed=42):
    rng = np.random.default_rng(seed)
    rows, cols = occupancy.shape
    q_values = np.zeros((rows, cols, len(ACTIONS)), dtype=float)
    goal_distances, _ = bfs(occupancy, goal)
    if goal_distances[start] < 0:
        return []
    max_steps = max(400, 4 * (int(goal_distances[start]) + 1))

    for episode in range(episodes):
        state = start
        current_epsilon = max(0.02, epsilon * (1.0 - episode / episodes))
        for _ in range(max_steps):
            row, col = state
            if rng.random() < current_epsilon:
                action = int(rng.integers(len(ACTIONS)))
            else:
                best_value = q_values[row, col].max()
                best_actions = np.flatnonzero(np.isclose(q_values[row, col], best_value))
                action = int(rng.choice(best_actions))

            dr, dc = ACTIONS[action]
            nr, nc = row + dr, col + dc
            if not (0 <= nr < rows and 0 <= nc < cols and occupancy[nr, nc]):
                next_state = state
                reward = -5.0
            else:
                next_state = (nr, nc)
                if next_state == goal:
                    reward = 100.0
                else:
                    phi_state = -float(goal_distances[state])
                    phi_next = -float(goal_distances[next_state])
                    reward = -1.0 + 0.2 * (gamma * phi_next - phi_state)

            next_best = 0.0 if next_state == goal else q_values[next_state].max()
            td_target = reward + gamma * next_best
            q_values[row, col, action] += alpha * (td_target - q_values[row, col, action])
            state = next_state
            if state == goal:
                break

    path = [start]
    state = start
    visited = {start}
    for _ in range(rows * cols):
        if state == goal:
            break
        row, col = state
        candidates = []
        for action, (dr, dc) in enumerate(ACTIONS):
            next_state = (row + dr, col + dc)
            nr, nc = next_state
            if 0 <= nr < rows and 0 <= nc < cols and occupancy[nr, nc]:
                candidates.append((q_values[row, col, action], next_state))
        candidates.sort(key=lambda item: item[0], reverse=True)
        next_state = next((cell for _, cell in candidates if cell not in visited or cell == goal), None)
        if next_state is None:
            break
        path.append(next_state)
        visited.add(next_state)
        state = next_state
    return path


def solve_and_plot(occupancy, start, goal, ax=None):
    if not occupancy[goal]:
        print(f"goal {goal} is a wall. Please choose a dark corridor cell.")
        return None
    distances, predecessors = bfs(occupancy, start)
    bfs_path = reconstruct_path(predecessors, goal)
    if not bfs_path:
        print(f"goal {goal} is not reachable from start {start}.")
        return None
    learned_path = q_learning(occupancy, start, goal)
    reached = bool(learned_path) and learned_path[-1] == goal
    print("start cell:", start)
    print("selected goal cell:", goal)
    print("BFS shortest-path length:", len(bfs_path) - 1)
    print("Q-learning path length:", len(learned_path) - 1 if learned_path else "failed")
    print("Q-learning reached goal:", reached)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    ax.clear()
    ax.imshow(~occupancy, cmap="gray", interpolation="nearest")
    if learned_path:
        arr = np.asarray(learned_path)
        ax.plot(arr[:, 1], arr[:, 0], linewidth=2, label="Q-learning route")
    ax.scatter(start[1], start[0], marker="o", s=70, label="start")
    ax.scatter(goal[1], goal[0], marker="*", s=130, label="goal")
    ax.set_title("Interactive Q-learning maze route")
    ax.set_xlabel("grid column")
    ax.set_ylabel("grid row")
    ax.legend()
    return learned_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", nargs=2, type=int, metavar=("ROW", "COL"), help="solve for a specified goal cell")
    args = parser.parse_args()
    occupancy, start = load_maze()
    print("maze grid shape:", occupancy.shape)
    print("start cell:", start)

    if args.goal is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        solve_and_plot(occupancy, start, tuple(args.goal), ax=ax)
        fig.tight_layout()
        output = ROOT / "q7_interactive_selected_path.png"
        fig.savefig(output, dpi=180)
        print("saved selected route figure to:", output)
        plt.show()
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(~occupancy, cmap="gray", interpolation="nearest")
    ax.scatter(start[1], start[0], marker="o", s=70, label="start")
    ax.set_title("Click a reachable corridor cell as the goal")
    ax.set_xlabel("grid column")
    ax.set_ylabel("grid row")
    ax.legend()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        goal = (int(round(event.ydata)), int(round(event.xdata)))
        print("\nclicked goal:", goal)
        solve_and_plot(occupancy, start, goal, ax=ax)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


if __name__ == "__main__":
    main()
