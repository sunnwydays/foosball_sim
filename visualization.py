"""
visualization.py — Bird's-eye view of the foosball table.

Functions
---------
draw_field(field, ball_state=None, trajectory=None, show_reach=False)
    Render a static snapshot of the table.

draw_heatmap(field, trajectories)
    Overlay a 2D heatmap of ball possession positions from Monte Carlo data.

animate_point(field, trajectory_xy)
    Step through a single point's trajectory interactively.

Requirements: matplotlib  (pip install matplotlib)
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np

from field import Field, BallState


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
TEAM_COLOR  = {0: "#4C9BE8", 1: "#E8724C"}   # blue / orange
FIELD_GREEN = "#2d6a2d"
LINE_WHITE  = "#ffffff"
WALL_GRAY   = "#888888"
BALL_COLOR  = "#f5f542"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_base_field(ax: plt.Axes, field: Field) -> None:
    """Draw the green pitch, walls, centre line, and goal openings."""

    # Green background
    ax.set_facecolor(FIELD_GREEN)

    # Outer boundary (white border)
    rect = patches.Rectangle(
        (0, 0), field.width, field.height,
        linewidth=2, edgecolor=LINE_WHITE, facecolor="none",
    )
    ax.add_patch(rect)

    # Centre line
    ax.axvline(field.width / 2, color=LINE_WHITE, linewidth=1, linestyle="--", alpha=0.5)

    # Goals — white rectangles on each end wall
    goal_depth = 5.0   # visual depth (cm) — purely aesthetic
    for goal in (field.left_goal, field.right_goal):
        if goal.x == 0:
            gx = -goal_depth
        else:
            gx = field.width
        g = patches.Rectangle(
            (gx, goal.y_min), goal_depth, goal.y_max - goal.y_min,
            linewidth=1.5, edgecolor=LINE_WHITE,
            facecolor=TEAM_COLOR[goal.scoring_team], alpha=0.6,
        )
        ax.add_patch(g)


def _draw_rods(ax: plt.Axes, field: Field, show_reach: bool) -> None:
    """Draw each rod as a vertical line with player circles."""

    for rod in field.rods:
        color = TEAM_COLOR[rod.team]

        # Rod line
        ax.plot(
            [rod.x, rod.x], [0, field.height],
            color=color, linewidth=1.5, alpha=0.4, zorder=2,
        )

        for py in rod.player_positions:
            # Player body
            circle = plt.Circle(
                (rod.x, py), radius=2.8,
                color=color, zorder=3,
            )
            ax.add_patch(circle)

            # Optional: highlight interception reach (y-reach rectangle)
            if show_reach:
                reach_rect = patches.Rectangle(
                    (rod.x - 4, py - rod.y_reach),
                    8, 2 * rod.y_reach,
                    linewidth=0, facecolor=color, alpha=0.2, zorder=1,
                )
                ax.add_patch(reach_rect)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_field(
    field:         Field,
    ball_state:    Optional[BallState]                 = None,
    trajectory:    Optional[list[tuple[float, float]]] = None,
    show_reach:    bool                                = False,
    title:         str                                 = "Foosball Table",
    ax:            Optional[plt.Axes]                  = None,
    fig_facecolor: str                                 = "#1a1a1a",
) -> plt.Axes:
    """
    Draw a bird's-eye view of the foosball table.

    Parameters
    ----------
    field       : Field instance (rod offsets determine player positions).
    ball_state  : if provided, draw the ball as a yellow circle.
    trajectory  : list of (x, y) — possession transfer positions; drawn as
                  a connected path with arrows.
    show_reach  : if True, shade each player's interception zone (y_reach).
    title       : plot title.
    ax          : existing Axes to draw on (creates new figure if None).

    Returns
    -------
    The Axes object (so callers can further customise or save).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(fig_facecolor)

    _draw_base_field(ax, field)
    _draw_rods(ax, field, show_reach=show_reach)

    # Trajectory path
    if trajectory and len(trajectory) > 1:
        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]
        ax.plot(xs, ys, color="white", linewidth=1, alpha=0.6, zorder=4)
        # Arrow at each step
        for i in range(len(trajectory) - 1):
            ax.annotate(
                "", xy=trajectory[i + 1], xytext=trajectory[i],
                arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
                zorder=5,
            )

    # Ball
    if ball_state is not None:
        ball_circle = plt.Circle(
            (ball_state.x, ball_state.y), radius=2,
            color=BALL_COLOR, zorder=6,
        )
        ax.add_patch(ball_circle)

    # Axes formatting
    ax.set_xlim(-6, field.width + 6)
    ax.set_ylim(-2, field.height + 2)
    ax.set_aspect("equal")
    ax.set_title(title, color="white", pad=8)
    ax.set_facecolor(FIELD_GREEN)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor(WALL_GRAY)

    # Team legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TEAM_COLOR[0],
               markersize=8, label="Team 0 (→)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TEAM_COLOR[1],
               markersize=8, label="Team 1 (←)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              facecolor="#333333", labelcolor="white", framealpha=0.8)

    plt.tight_layout()
    return ax


def draw_heatmap(
    field:        Field,
    trajectories: list[list[tuple[float, float]]],
    bins:         int  = 30,
    title:        str  = "Ball Possession Heatmap",
    fig_facecolor: str = "#1a1a1a",
) -> plt.Axes:
    """
    Overlay a 2D density heatmap of ball positions over the field.

    Collects every (x, y) possession point from all trajectories and
    renders a 2D histogram, showing where the ball tends to end up
    after shots across thousands of simulations.

    Parameters
    ----------
    field        : Field (used for dimensions and rod positions).
    trajectories : list of trajectory lists from MCResult.all_trajectories.
    bins         : number of histogram bins per axis.
    title        : plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(fig_facecolor)
    _draw_base_field(ax, field)
    _draw_rods(ax, field, show_reach=False)

    # Flatten all (x, y) points
    all_x = [x for traj in trajectories for x, y in traj]
    all_y = [y for traj in trajectories for x, y in traj]

    if all_x:
        h, xedges, yedges = np.histogram2d(
            all_x, all_y,
            bins=[bins, bins],
            range=[[0, field.width], [0, field.height]],
        )
        # Normalise
        h = h / h.max() if h.max() > 0 else h

        ax.imshow(
            h.T,
            origin="lower",
            extent=[0, field.width, 0, field.height],
            cmap="hot",
            alpha=0.55,
            aspect="auto",
            zorder=1,
        )

    ax.set_xlim(-6, field.width + 6)
    ax.set_ylim(-2, field.height + 2)
    ax.set_aspect("equal")
    ax.set_title(title, color="white", pad=8)
    ax.tick_params(colors="white")
    plt.tight_layout()
    return ax


def animate_point(
    field:      Field,
    trajectory: list[tuple[float, float]],
    pause:      float = 0.6,
) -> None:
    """
    Step through a single point's trajectory, updating the plot at each
    possession change so you can watch the ball move around the table.

    Parameters
    ----------
    field      : Field (rod offsets should reflect the final state of the point).
    trajectory : list of (x, y) possession positions.
    pause      : seconds between frames.
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (bx, by) in enumerate(trajectory):
        ax.cla()
        _draw_base_field(ax, field)
        _draw_rods(ax, field, show_reach=False)

        # Draw path so far
        if i > 0:
            xs = [p[0] for p in trajectory[:i + 1]]
            ys = [p[1] for p in trajectory[:i + 1]]
            ax.plot(xs, ys, color="white", linewidth=1, alpha=0.5)

        # Current ball
        ball_circle = plt.Circle((bx, by), radius=2, color=BALL_COLOR, zorder=6)
        ax.add_patch(ball_circle)

        ax.set_xlim(-6, field.width + 6)
        ax.set_ylim(-2, field.height + 2)
        ax.set_aspect("equal")
        ax.set_title(f"Turn {i + 1} / {len(trajectory)}", color="white")
        ax.set_facecolor(FIELD_GREEN)

        plt.pause(pause)

    plt.ioff()
    plt.show()
