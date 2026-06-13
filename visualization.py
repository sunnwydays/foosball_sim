"""
visualization.py — Bird's-eye view of the foosball table.

Functions
---------
draw_field(field, ...)
    Render a static snapshot of the table.

replay_point(field, frames, ...)
    Animated playback of a recorded point using matplotlib FuncAnimation.
    Can display live or save to mp4/gif.

Requirements: matplotlib  (pip install matplotlib)
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np

import config
from field import Field, BallState
from simulation import Frame


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
TEAM_COLOR  = {0: "#4C9BE8", 1: "#E8724C"}   # blue / orange
FIELD_GREEN = "#2d6a2d"
LINE_WHITE  = "#ffffff"
WALL_GRAY   = "#888888"
BALL_COLOR  = "#f5f542"

# Per-frame lerp toward the target foot x in the replay (0..1, higher = snappier).
# Purely cosmetic: slides the opaque foot toward the ball when contact is possible.
FOOT_EASE_ALPHA = 0.35


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_base_field(ax: plt.Axes, field: Field) -> None:
    """Draw the green pitch, walls, centre line, and goal openings."""
    ax.set_facecolor(FIELD_GREEN)

    rect = patches.Rectangle(
        (0, 0), field.depth, field.width,
        linewidth=2, edgecolor=LINE_WHITE, facecolor="none",
    )
    ax.add_patch(rect)

    ax.axvline(field.depth / 2, color=LINE_WHITE, linewidth=1, linestyle="--", alpha=0.5)

    goal_vis_depth = 5.0
    for goal in (field.left_goal, field.right_goal):
        gx = -goal_vis_depth if goal.x == 0 else field.depth
        g = patches.Rectangle(
            (gx, goal.y_min), goal_vis_depth, goal.y_max - goal.y_min,
            linewidth=1.5, edgecolor=LINE_WHITE,
            facecolor=TEAM_COLOR[1 - goal.scoring_team], alpha=0.6,
        )
        ax.add_patch(g)


def _draw_rods_from_offsets(
    ax: plt.Axes,
    field: Field,
    rod_ys: list[float],
    rod_xs: list[float],
    rod_ctrl: list[bool],
    rod_ups: list[bool],
    show_reach: bool = False,
    foot_x_override: Optional[list[float]] = None,
) -> None:
    """Draw rods using explicit offset arrays (from a Frame).

    foot_x_override : if given, the opaque foot of each rod is drawn at
        `_base_x + foot_x_override[i]` instead of the rod's x_offset. This is a
        cosmetic embellishment used by the replay to slide the foot toward the
        ball during contact; it does not move the rod line or reach box.
    """
    for i, rod in enumerate(field.rods):
        color = TEAM_COLOR[rod.team]
        alpha = 0.2 if rod_ups[i] else 1.0

        # Apply frame offsets temporarily
        orig_y, orig_x = rod.y_offset, rod.x_offset
        rod.y_offset = rod_ys[i]
        rod.x_offset = rod_xs[i]

        # Rod line (axis of rotation)
        ax.plot(
            [rod._base_x, rod._base_x], [0, field.width],
            color=color, linewidth=1.5, alpha=0.4, zorder=2,
        )

        # Foot x: rod.x by default, or the eased reach override when provided.
        foot_x = rod.x if foot_x_override is None else rod._base_x + foot_x_override[i]

        for py in rod.player_positions:
            # Player foot — fixed size, shifted by x_offset (or reach override)
            player_rect = patches.Rectangle(
                (foot_x - rod.thickness / 2, py - rod.width / 2),
                rod.thickness, rod.width,
                linewidth=0, facecolor=color, alpha=alpha, zorder=3,
            )
            ax.add_patch(player_rect)

            if show_reach:
                reach_rect = patches.Rectangle(
                    (rod._base_x - rod.rod_x_reach - rod.thickness / 2, py - rod.width / 2),
                    2 * rod.rod_x_reach + rod.thickness, rod.width,
                    linewidth=0.5, edgecolor=color, facecolor=color,
                    alpha=0.15, zorder=1,
                )
                ax.add_patch(reach_rect)

        # Controlled indicator
        if rod_ctrl[i]:
            ax.plot(rod._base_x, field.width + 1, 'v', color=color, markersize=5, zorder=5)

        # Restore
        rod.y_offset, rod.x_offset = orig_y, orig_x


def _compute_foot_offsets(field: Field, frames: list[Frame]) -> list[list[float]]:
    """Eased per-frame foot x_offsets for the replay (cosmetic only).

    For each frame and rod, the foot targets the ball's x (clamped to the rod's
    reach) when the rod is controlled and the ball sits inside its reach box;
    otherwise it targets the rod's actual rest offset. A forward exponential lerp
    smooths the motion so the foot slides rather than snaps. This never touches
    sim state — it only derives display positions from recorded frames.

    Returns foot_xs indexed [frame_idx][rod_idx].
    """
    foot_xs: list[list[float]] = []
    prev: Optional[list[float]] = None

    for fr in frames:
        targets: list[float] = []
        for i, rod in enumerate(field.rods):
            orig_y = rod.y_offset
            rod.y_offset = fr.rod_ys[i]
            in_reach = (
                fr.rod_ctrl[i]
                and rod.player_in_box(fr.ball_x, fr.ball_y, config.BALL_RADIUS) is not None
            )
            if in_reach:
                raw = fr.ball_x - rod._base_x
                target = max(-rod.rod_x_reach, min(rod.rod_x_reach, raw))
            else:
                target = fr.rod_xs[i]
            rod.y_offset = orig_y
            targets.append(target)

        if prev is None:
            disp = list(targets)
        else:
            disp = [
                p + FOOT_EASE_ALPHA * (t - p) for p, t in zip(prev, targets)
            ]
        foot_xs.append(disp)
        prev = disp

    return foot_xs


# ---------------------------------------------------------------------------
# Public API — static drawing
# ---------------------------------------------------------------------------

def draw_field(
    field:         Field,
    ball_state:    Optional[BallState]                 = None,
    show_reach:    bool                                = False,
    title:         str                                 = "Foosball Table",
    ax:            Optional[plt.Axes]                  = None,
    fig_facecolor: str                                 = "#1a1a1a",
) -> plt.Axes:
    """Draw a bird's-eye view of the foosball table (static snapshot)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(fig_facecolor)

    _draw_base_field(ax, field)

    # Use current rod offsets
    rod_ys = [r.y_offset for r in field.rods]
    rod_xs = [r.x_offset for r in field.rods]
    rod_ctrl = [r.controlled for r in field.rods]
    rod_ups = [r.up for r in field.rods]
    _draw_rods_from_offsets(ax, field, rod_ys, rod_xs, rod_ctrl, rod_ups, show_reach)

    if ball_state is not None:
        ball_circle = plt.Circle(
            (ball_state.x, ball_state.y), radius=config.BALL_RADIUS,
            color=BALL_COLOR, zorder=6,
        )
        ax.add_patch(ball_circle)

    ax.set_xlim(-5.5, field.depth + 5.5)
    ax.set_ylim(-1, field.width + 2.5)
    ax.set_aspect("equal")
    ax.set_title(title, color="white", pad=6)
    ax.set_facecolor(FIELD_GREEN)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor(WALL_GRAY)

    return ax


# ---------------------------------------------------------------------------
# Public API — animated replay
# ---------------------------------------------------------------------------

def replay_point(
    field:         Field,
    frames:        list[Frame],
    show_reach:    bool  = False,
    interval:      int   = 100,
    title:         str   = "Foosball Replay",
    trail_length:  int   = 20,
    save_path:     Optional[str] = None,
    fig_facecolor: str   = "#1a1a1a",
) -> animation.FuncAnimation:
    """
    Animate a recorded point using matplotlib FuncAnimation.

    Parameters
    ----------
    field        : Field instance (used for geometry/colours).
    frames       : list of Frame objects from simulate_point(record=True).
    show_reach   : if True, draw player hitbox rectangles.
    interval     : milliseconds between frames.
    title        : base title for the plot.
    trail_length : number of past ball positions to show as a trail.
    save_path    : if set, save the animation (e.g. 'replay.mp4' or 'replay.gif').

    Returns
    -------
    The FuncAnimation object (keep a reference to prevent garbage collection).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(fig_facecolor)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.94, bottom=0.07)

    # Pre-extract ball trail coordinates
    ball_xs = [f.ball_x for f in frames]
    ball_ys = [f.ball_y for f in frames]

    # Pre-compute eased foot offsets so feet slide toward the ball on contact.
    foot_xs = _compute_foot_offsets(field, frames)

    def _update(frame_idx: int) -> None:
        ax.cla()
        fr = frames[frame_idx]

        _draw_base_field(ax, field)
        _draw_rods_from_offsets(
            ax, field, fr.rod_ys, fr.rod_xs, fr.rod_ctrl, fr.ups, show_reach,
            foot_x_override=foot_xs[frame_idx],
        )

        # Ball trail
        start = max(0, frame_idx - trail_length)
        trail_x = ball_xs[start:frame_idx + 1]
        trail_y = ball_ys[start:frame_idx + 1]
        if len(trail_x) > 1:
            ax.plot(trail_x, trail_y, color=BALL_COLOR, linewidth=1, alpha=0.3, zorder=4)

        # Ball
        ball_circle = plt.Circle(
            (fr.ball_x, fr.ball_y), radius=config.BALL_RADIUS,
            color=BALL_COLOR, zorder=6,
        )
        ax.add_patch(ball_circle)

        # Velocity arrow
        speed = (fr.ball_vx ** 2 + fr.ball_vy ** 2) ** 0.5
        if speed > 5:
            scale = 0.08
            ax.annotate(
                "", xy=(fr.ball_x + fr.ball_vx * scale, fr.ball_y + fr.ball_vy * scale),
                xytext=(fr.ball_x, fr.ball_y),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.2),
                zorder=7,
            )

        # Hit flash
        if fr.event == 'hit':
            hit_circle = plt.Circle(
                (fr.ball_x, fr.ball_y), radius=5,
                color="white", alpha=0.3, zorder=5,
            )
            ax.add_patch(hit_circle)

        ax.set_xlim(-5.5, field.depth + 5.5)
        ax.set_ylim(-1, field.width + 2.5)
        ax.set_aspect("equal")
        ax.set_facecolor(FIELD_GREEN)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor(WALL_GRAY)

        event_str = f"  [{fr.event}]" if fr.event else ""
        ax.set_title(
            f"{title}  |  t={fr.time:.2f}s  tick {fr.tick}{event_str}",
            color="white", pad=6,
        )

    anim = animation.FuncAnimation(
        fig, _update, frames=len(frames), interval=interval, repeat=False
    )

    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000 // interval)
        else:
            anim.save(save_path, fps=1000 // interval)
        print(f"Saved animation to {save_path}")

    return anim


# ---------------------------------------------------------------------------
# Stats / heatmap visualization
# ---------------------------------------------------------------------------

def _gaussian_blur(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Pure numpy 2D gaussian blur via separable 1D convolutions."""
    if sigma <= 0:
        return grid
    radius = int(3 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    # Apply separably along each axis
    out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, grid)
    out = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 1, out)
    return out

def draw_stats(
    field: Field,
    pos_grid: np.ndarray,
    goal_hits: list,
    stall_hits: Optional[list] = None,
    dead_hits:  Optional[list] = None,
    title: str = "Ball Heatmap & Goal Origins",
    sigma: float = 1.5,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Overlay a ball-position heatmap and outcome markers on the field.

    goal_hits  : list of (x, y, goal_team, is_self_goal) — last active hit before each goal.
    stall_hits : list of (x, y, possessing_team) — ball position at stallout.
    dead_hits  : list of (x, y) — ball position at dead-ball draw.
    """
    import matplotlib.lines as mlines

    smoothed = _gaussian_blur(pos_grid.astype(float), sigma=sigma)

    if smoothed.max() > 0:
        smoothed = smoothed / smoothed.max()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#1a1a1a")
        fig.subplots_adjust(left=0.07, right=0.93, top=0.94, bottom=0.13)

    # Base layer: field
    draw_field(field, ax=ax)

    # Middle layer: heatmap
    ax.imshow(
        smoothed.T,
        origin="lower",
        extent=[0, field.depth, 0, field.width],
        cmap="hot",
        alpha=0.6,
        zorder=2,
        interpolation="bilinear",
        aspect="auto",
    )
    ax.set_aspect("equal")

    # Color palettes
    SCORE_COLOR = {0: "#1a60c0", 1: "#c03010"}   # scorer's team color
    STALL_COLOR = {0: "#6a90c8", 1: "#c07060"}   # stalling team color, desaturated

    legend_handles = []

    # Layer 3: dead-ball draws (grey)
    if dead_hits:
        xs = [p[0] for p in dead_hits]
        ys = [p[1] for p in dead_hits]
        ax.scatter(xs, ys, color="#888888", s=8, alpha=0.6, zorder=3)
        legend_handles.append(mlines.Line2D([], [], color="#888888", marker='o',
                                            linestyle='None', markersize=5, label="Dead ball"))

    # Layer 4: stall positions (desaturated team color)
    if stall_hits:
        for team, color in STALL_COLOR.items():
            pts = [(p[0], p[1]) for p in stall_hits if p[2] == team]
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                           color=color, s=18, alpha=0.85, zorder=4)
        for team, color in STALL_COLOR.items():
            if any(p[2] == team for p in stall_hits):
                label = f"T{team} stall"
                legend_handles.append(mlines.Line2D([], [], color=color, marker='o',
                                                    linestyle='None', markersize=5, label=label))

    # Layer 5: goal hit origins — normal goals (circle) and self-goals (x)
    # p = (x, y, goal_team, is_self_goal); winner = 1 - goal_team
    if goal_hits:
        for goal_team, color in SCORE_COLOR.items():
            scorer = 1 - goal_team
            normal   = [(p[0], p[1]) for p in goal_hits if p[2] == goal_team and not (len(p) > 3 and p[3])]
            selfgoal = [(p[0], p[1]) for p in goal_hits if p[2] == goal_team and (len(p) > 3 and p[3])]
            if normal:
                ax.scatter([p[0] for p in normal], [p[1] for p in normal],
                           color=color, s=12, alpha=0.8, zorder=5, marker='o')
                legend_handles.append(mlines.Line2D([], [], color=color, marker='o',
                                                    linestyle='None', markersize=5,
                                                    label=f"T{scorer} goal"))
            if selfgoal:
                ax.scatter([p[0] for p in selfgoal], [p[1] for p in selfgoal],
                           color=color, s=20, alpha=0.9, zorder=5, marker='x',
                           linewidths=1.5)
                legend_handles.append(mlines.Line2D([], [], color=color, marker='x',
                                                    linestyle='None', markersize=6,
                                                    markeredgewidth=1.5,
                                                    label=f"T{scorer} self-goal"))

    # Legend in the bottom black padding (figure coordinates)
    if legend_handles:
        ax.get_figure().legend(
            handles=legend_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(legend_handles),
            fontsize=7,
            framealpha=0.5,
            facecolor='#1a1a1a',
            edgecolor='#444444',
            labelcolor='white',
        )

    ax.set_title(title, color="white", pad=6)
    return ax


# ---------------------------------------------------------------------------
# Evolution analytics visualization
# ---------------------------------------------------------------------------

def plot_evolution_stats(history, save_path: Optional[str] = None):
    """
    Render a 3x2 multi-panel figure summarizing an evolution run.

    Parameters
    ----------
    history   : EvolutionHistory instance from evolution.py.
    save_path : if set, save PNG to this path (150 dpi).

    Returns
    -------
    matplotlib Figure
    """
    from evolution import GENE_GROUPS, _OFFSETS

    DARK_BG = "#1a1a1a"
    AXES_BG = "#222222"

    _SHOT_COLORS  = ["#4C9BE8", "#E8724C", "#4CE87A"]
    _SKILL_COLORS = ["#f5f542", "#E84C9B", "#9B4CE8"]
    _PASS_COLORS  = ["#4CE8C9", "#E8C94C", "#E84C4C"]
    _INDEP_COLORS = ["#f5f542", "#4CE87A", "#E84C9B", "#4C9BE8", "#E8724C"]

    gens     = history.gen
    genomes  = np.array(history.best_genomes)   # (n_gens, GENOME_SIZE)
    mean_arr = np.array(history.mean_wr)
    std_arr  = np.array(history.std_wr)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Evolution Analytics", color="white", fontsize=16, fontweight="bold")

    def _style(ax, title, ylabel="", ylim=None):
        ax.set_facecolor(AXES_BG)
        ax.set_title(title, color="white", fontsize=11, pad=6)
        ax.set_xlabel("Generation", color="#aaaaaa", fontsize=9)
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.grid(axis="y", color="white", alpha=0.12, linewidth=0.5)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8, framealpha=0.35, labelcolor="white",
                  facecolor="#333333", edgecolor="#555555")

    # (0,0) Fitness Trajectory
    ax = axes[0, 0]
    ax.plot(gens, history.best_wr, color="#4C9BE8", lw=2,   label="best")
    ax.plot(gens, history.mean_wr, color="white",   lw=1.5, label="mean", linestyle="--")
    ax.plot(gens, history.min_wr,  color="#E8724C", lw=1,   label="min",  linestyle=":")
    ax.fill_between(gens, mean_arr - std_arr, mean_arr + std_arr, alpha=0.15, color="white")
    _style(ax, "Fitness Trajectory", ylabel="Win Rate")

    # (0,1) Shot Composition
    ax = axes[0, 1]
    lo, _ = _OFFSETS["shot"]
    for i, (name, color) in enumerate(zip(GENE_GROUPS["shot"], _SHOT_COLORS)):
        ax.plot(gens, genomes[:, lo + i], color=color, lw=1.5, label=name)
    _style(ax, "Shot Composition — Best Agent", ylabel="Weight")

    # (1,0) Skill Allocation
    ax = axes[1, 0]
    lo, _ = _OFFSETS["skill"]
    for i, (name, color) in enumerate(zip(GENE_GROUPS["skill"], _SKILL_COLORS)):
        ax.plot(gens, genomes[:, lo + i], color=color, lw=1.5, label=name)
    _style(ax, "Skill Allocation — Best Agent", ylabel="Weight")

    # (1,1) Pass Composition
    ax = axes[1, 1]
    lo, _ = _OFFSETS["pass"]
    for i, (name, color) in enumerate(zip(GENE_GROUPS["pass"], _PASS_COLORS)):
        ax.plot(gens, genomes[:, lo + i], color=color, lw=1.5, label=name)
    _style(ax, "Pass Composition — Best Agent", ylabel="Weight")

    # (2,0) Independent Genes
    ax = axes[2, 0]
    lo, _ = _OFFSETS["indep"]
    for i, (name, color) in enumerate(zip(GENE_GROUPS["indep"], _INDEP_COLORS)):
        ax.plot(gens, genomes[:, lo + i], color=color, lw=1.5, label=name)
    _style(ax, "Independent Genes — Best Agent", ylabel="Value [0, 1]", ylim=(0, 1))

    # (2,1) Population Diversity
    ax = axes[2, 1]
    ax.plot(gens, history.std_wr, color="#9B4CE8", lw=2, label="std(win_rate)")
    ax.fill_between(gens, 0, history.std_wr, alpha=0.2, color="#9B4CE8")
    _style(ax, "Population Diversity", ylabel="Std Dev of Win Rates")

    fig.tight_layout(rect=[0, 0, 1, 0.97], pad=2.0)

    if save_path:
        import os as _os
        _os.makedirs(_os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=DARK_BG)
        print(f"Saved evolution stats to {save_path}")
        _os.startfile(_os.path.abspath(save_path))

    return fig
