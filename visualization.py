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
            facecolor=TEAM_COLOR[goal.scoring_team], alpha=0.6,
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
) -> None:
    """Draw rods using explicit offset arrays (from a Frame)."""
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

        for py in rod.player_positions:
            # Player foot — fixed size, shifted by x_offset
            player_rect = patches.Rectangle(
                (rod.x - rod.thickness / 2, py - rod.width / 2),
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
            ax.plot(rod.x, field.width + 1, 'v', color=color, markersize=5, zorder=5)

        # Restore
        rod.y_offset, rod.x_offset = orig_y, orig_x


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

    ax.set_xlim(-6, field.depth + 6)
    ax.set_ylim(-2, field.width + 4)
    ax.set_aspect("equal")
    ax.set_title(title, color="white", pad=8)
    ax.set_facecolor(FIELD_GREEN)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor(WALL_GRAY)

    plt.tight_layout()
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

    # Pre-extract ball trail coordinates
    ball_xs = [f.ball_x for f in frames]
    ball_ys = [f.ball_y for f in frames]

    def _update(frame_idx: int) -> None:
        ax.cla()
        fr = frames[frame_idx]

        _draw_base_field(ax, field)
        _draw_rods_from_offsets(
            ax, field, fr.rod_ys, fr.rod_xs, fr.rod_ctrl, fr.ups, show_reach
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

        ax.set_xlim(-6, field.depth + 6)
        ax.set_ylim(-2, field.width + 4)
        ax.set_aspect("equal")
        ax.set_facecolor(FIELD_GREEN)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor(WALL_GRAY)

        event_str = f"  [{fr.event}]" if fr.event else ""
        ax.set_title(
            f"{title}  |  t={fr.time:.2f}s  tick {fr.tick}{event_str}",
            color="white", pad=8,
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
