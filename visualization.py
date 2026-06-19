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

import bisect
import os
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
BALL_COLOR  = "#ffffff"

# Per-frame lerp toward the target foot x in the replay (0..1, higher = snappier).
# Purely cosmetic: slides the opaque foot toward the ball when contact is possible.
FOOT_EASE_ALPHA = 0.35

# Visual backswing magnitude as a fraction of the forward reach at contact.
# Smaller than 1.0 so the backswing looks like a wind-up, not an equal recoil.
SWING_ANIM_BACKSWING = 0.4


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
    rod_switching: Optional[list[bool]] = None,
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

        # Control indicator: solid triangle = held (switch complete);
        # hollow triangle = a hand is assigned but still mid-switch.
        if rod_ctrl[i]:
            ax.plot(rod._base_x, field.width + 1, 'v', color=color, markersize=5, zorder=5)
        elif rod_switching is not None and rod_switching[i]:
            ax.plot(rod._base_x, field.width + 1, 'v', markersize=5, zorder=5,
                    markerfacecolor='none', markeredgecolor=color, markeredgewidth=1.0)

        # Restore
        rod.y_offset, rod.x_offset = orig_y, orig_x


def _compute_foot_offsets(field: Field, frames: list[Frame]) -> list[list[float]]:
    """Eased per-frame foot x_offsets for the replay (cosmetic only).

    Swing commits drive a two-phase animation timed to the actual sim windows:
      backswing phase  (commit_time → active_start): foot pulls away from the ball
      active phase     (active_start → window_end):  foot slides toward the ball

    Outside a committed swing the foot targets the ball when in reach, otherwise
    the rod's rest x_offset. A forward exponential lerp smooths all transitions.

    Returns foot_xs indexed [frame_idx][rod_idx].
    """
    foot_xs: list[list[float]] = []
    prev: Optional[list[float]] = None

    for fr in frames:
        targets: list[float] = []
        swings = fr.rod_swings if fr.rod_swings else [None] * len(field.rods)

        for i, rod in enumerate(field.rods):
            orig_y = rod.y_offset
            rod.y_offset = fr.rod_ys[i]

            in_reach = (
                fr.rod_ctrl[i]
                and rod.player_in_box(fr.ball_x, fr.ball_y, config.BALL_RADIUS) is not None
            )
            fwd_raw = fr.ball_x - rod._base_x
            fwd     = max(-rod.rod_x_reach, min(rod.rod_x_reach, fwd_raw))

            swing = swings[i]
            if swing is not None:
                active_start, _, vx = swing
                # Use the committed shot direction
                fwd_sign = 1.0 if vx >= 0 else -1.0
                if fr.time < active_start:
                    # Backswing: foot retracts opposite to intended shot direction
                    target = -fwd_sign * rod.rod_x_reach * SWING_ANIM_BACKSWING
                else:
                    # Active window: foot lunges in shot direction, tracks ball when in reach
                    target = fwd if in_reach else fwd_sign * rod.rod_x_reach
            elif in_reach:
                target = fwd
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


def _render_frame(
    ax:           plt.Axes,
    field:        Field,
    frames:       list[Frame],
    ball_xs:      list[float],
    ball_ys:      list[float],
    foot_xs:      list[list[float]],
    frame_idx:    int,
    show_reach:   bool,
    trail_length: int,
    title:        str,
) -> None:
    """Draw a single replay frame into `ax` (clears it first).

    Shared by the GIF animation (`replay_point`) and the interactive viewer
    (`interactive_replay`). `ball_xs`, `ball_ys`, and `foot_xs` are precomputed
    once over the whole point, so any `frame_idx` can be drawn directly without
    re-stepping the simulation.
    """
    ax.cla()
    fr = frames[frame_idx]

    _draw_base_field(ax, field)
    _draw_rods_from_offsets(
        ax, field, fr.rod_ys, fr.rod_xs, fr.rod_ctrl, fr.ups, show_reach,
        foot_x_override=foot_xs[frame_idx],
        rod_switching=getattr(fr, "rod_switching", None),
    )

    # Ball trail
    start = max(0, frame_idx - trail_length)
    trail_x = ball_xs[start:frame_idx + 1]
    trail_y = ball_ys[start:frame_idx + 1]
    if len(trail_x) > 1:
        ax.plot(trail_x, trail_y, color=BALL_COLOR, linewidth=1, alpha=0.3, zorder=4)

    # Ball
    ax.add_patch(plt.Circle(
        (fr.ball_x, fr.ball_y), radius=config.BALL_RADIUS,
        color=BALL_COLOR, zorder=6,
    ))

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
        ax.add_patch(plt.Circle(
            (fr.ball_x, fr.ball_y), radius=5,
            color="white", alpha=0.3, zorder=5,
        ))

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


# ---------------------------------------------------------------------------
# Action-log panel helpers (used by interactive_replay)
# ---------------------------------------------------------------------------

# Action-log panel: fixed font and line spacing (independent of event count).
# When the log outgrows the panel it scrolls, auto-following the current event,
# and the mouse wheel pans it manually. Past/future events are dimmed; the most
# recent event at/before the playhead is bold + bright.
LOG_FONT_SIZE = 8.5    # points
LOG_LINE_PTS = 14.0    # vertical spacing per line, points (~20 lines per panel)
LOG_SCROLL_STEP = 3    # lines moved per wheel notch
LOG_DIM_ALPHA = 0.5
LOG_GOAL_COLOR = "#ffd24c"
LOG_PHYSICS_COLOR = "#999999"


def _format_log_line(e) -> str:
    """One compact monospace line for the viewer's action-log panel.

    Mirrors the terminal log (play.py `_format_event`) but trimmed to fit a
    narrow side panel. `e` is an ActionLogEntry (duck-typed).
    """
    t = f"{e.game_time:5.2f}s"
    bx, by = e.ball_pos
    pos = f"({bx:.0f},{by:.0f})"
    if e.rod_label.startswith("GOAL"):
        return f"{t}  *** {e.action} ***"
    if e.team == -1:  # physics ball-event (wall bounce / deflect / stop)
        reach = e.rod_label if len(e.rod_label) <= 16 else e.rod_label[:15] + "+"
        return f"{t}  {e.action:<9} {pos:<9} [{reach}]"
    av = e.actual_vel if e.actual_vel is not None else e.intended_vel
    vel = f"  v=({av[0]:+.0f},{av[1]:+.0f})" if av is not None else ""
    return f"{t}  {e.rod_label:<7} {e.action:<7} {pos:<9}{vel}"


def _choice_label(opt: str) -> str:
    """Display label for a strategy dropdown entry: bare name, or genome filename."""
    return os.path.basename(opt)[:-5] if opt.endswith(".json") else opt


def _log_window_start(scroll: int, follow: bool, ptr: int, n: int, visible: int) -> int:
    """Top visible line index for the scrolling log panel.

    When `follow`, nudge the window just enough to keep the current event `ptr`
    on screen (no jump while it is already visible); always clamp to a valid
    range. `ptr` is -1 before the first event.
    """
    if follow and ptr >= 0:
        if ptr < scroll:
            scroll = ptr
        elif ptr >= scroll + visible:
            scroll = ptr - visible + 1
    return max(0, min(max(0, n - visible), scroll))


def _events_by_frame(frames: list[Frame], action_log) -> list[int]:
    """Map each action-log entry to the replay frame index where it occurs.

    Events are timestamped with `game_time`; each frame carries the same
    `time`, so the entry lands on the last frame whose time is at or before it
    (goals, logged at game_time+dt, fold onto the final frame). The result is
    parallel to `action_log` and non-decreasing (events are chronological).
    """
    if not action_log:
        return []
    times = [f.time for f in frames]
    last = len(frames) - 1
    return [
        max(0, min(last, bisect.bisect_right(times, e.game_time + 1e-6) - 1))
        for e in action_log
    ]


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
    rod_switching = [r.switch_timer > 0 for r in field.rods]
    rod_ups = [r.up for r in field.rods]
    _draw_rods_from_offsets(ax, field, rod_ys, rod_xs, rod_ctrl, rod_ups, show_reach,
                            rod_switching=rod_switching)

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
    gif_slowdown:  float = 1,
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
        _render_frame(ax, field, frames, ball_xs, ball_ys, foot_xs,
                      frame_idx, show_reach, trail_length, title)

    anim = animation.FuncAnimation(
        fig, _update, frames=len(frames), interval=interval, repeat=False
    )

    if save_path:
        save_fps = max(1, round((1000 // interval) / gif_slowdown))
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=save_fps)
        else:
            anim.save(save_path, fps=save_fps)
        print(f"Saved animation to {save_path}")

    return anim


def interactive_replay(
    field:         Field,
    frames:        list[Frame],
    show_reach:    bool  = False,
    fps:           int   = 30,
    title:         str   = "Foosball Replay",
    trail_length:  int   = 20,
    fig_facecolor: str   = "#1a1a1a",
    rerun_fn                   = None,
    rerun_init: Optional[dict] = None,
    action_log:   Optional[list] = None,
    strategy_choices: Optional[list] = None,
) -> None:
    """Open a scrubbable viewer for a recorded point (like a video player).

    Unlike `replay_point` (which renders to a GIF), this pops up a live
    matplotlib window with a frame slider plus play/pause and step controls.
    All frame data is already cached in `frames`, so scrubbing and stepping are
    instant; nothing is re-simulated.

    Controls
    --------
    space        play / pause
    left / right step back / forward one second (`fps` frames)
    , / .        step back / forward one tick
    home / end   jump to first / last frame
    Plus on-screen buttons and a draggable frame slider.

    When `rerun_fn` is provided a settings panel appears below the slider with
    editable seed and strategy fields, skill / anticipation sliders, a kickoff
    toggle, reset buttons, and a Re-run button. `rerun_fn(settings)` receives a
    dict with keys seed, team0, team1, skill0, skill1, anticipation0,
    anticipation1, kickoff and must return
    `(new_field, new_frames, new_title, seed_used)` or, to keep the action-log
    panel in sync, `(new_field, new_frames, new_title, seed_used, new_action_log)`.

    When `action_log` (a list of ActionLogEntry) is given, a synced log panel is
    drawn to the right of the field: it scrolls with playback/scrubbing and bolds
    the action(s) happening on the current frame.

    `strategy_choices` (a list of strategy names and/or genome .json paths) adds a
    dropdown arrow beside each T0/T1 box; picking an entry fills the box (typing
    still works).

    Requires an interactive matplotlib backend (i.e. do not force "Agg").
    Blocks on `plt.show()` until the window is closed.
    """
    from matplotlib.widgets import Slider, Button, TextBox

    n = len(frames)
    if n == 0:
        return

    ball_xs = [f.ball_x for f in frames]
    ball_ys = [f.ball_y for f in frames]
    foot_xs = _compute_foot_offsets(field, frames)

    has_settings = rerun_fn is not None
    has_log = bool(action_log)
    init = rerun_init or {}

    if has_settings:
        fig = plt.figure(figsize=(11, 9.6))
        if has_log:
            ax     = fig.add_axes([0.035, 0.575, 0.60, 0.40])
            ax_log = fig.add_axes([0.655, 0.575, 0.335, 0.40])
        else:
            ax     = fig.add_axes([0.05, 0.575, 0.90, 0.40])
            ax_log = None
    else:
        if has_log:
            fig    = plt.figure(figsize=(12, 7))
            ax     = fig.add_axes([0.05, 0.26, 0.58, 0.68])
            ax_log = fig.add_axes([0.66, 0.26, 0.31, 0.68])
        else:
            fig    = plt.figure(figsize=(10, 7))
            ax     = fig.add_axes([0.07, 0.26, 0.86, 0.68])
            ax_log = None
    fig.patch.set_facecolor(fig_facecolor)

    state = {
        "idx":    0,
        "playing": False,
        "field":  field,
        "frames": frames,
        "ball_xs": ball_xs,
        "ball_ys": ball_ys,
        "foot_xs": foot_xs,
        "n":      n,
        "title":  title,
        "kickoff": int(init.get("kickoff", 0) or 0),
        "action_log":    list(action_log) if has_log else [],
        "log_frame_idx": _events_by_frame(frames, action_log) if has_log else [],
        "log_scroll":  0,      # index of the top visible log line
        "log_follow":  True,   # auto-scroll to keep the current event in view
        "log_visible": 1,      # lines that fit the panel (set during render)
        "dropdown":    None,   # open strategy-picker popup, if any
    }

    def render_log(idx: int) -> None:
        """Draw the action log at a fixed font; scroll to keep the current event visible."""
        ax_log.cla()
        ax_log.set_facecolor("#111111")
        ax_log.set_xticks([])
        ax_log.set_yticks([])
        for spine in ax_log.spines.values():
            spine.set_edgecolor("#444444")
        log  = state["action_log"]
        fidx = state["log_frame_idx"]
        n = len(log)
        if n == 0:
            ax_log.set_title("action log  (0 events)", color="#cccccc",
                             fontsize=9, pad=4)
            return

        # Fixed font + line spacing; the number of visible lines follows from the
        # (constant) panel geometry, not the event count.
        panel_pts = fig.get_size_inches()[1] * ax_log.get_position().height * 72.0
        visible = max(1, int(panel_pts / LOG_LINE_PTS))
        dy = LOG_LINE_PTS / panel_pts
        state["log_visible"] = visible

        # Most recent event at/before the current frame (stays bright until the next).
        ptr = bisect.bisect_right(fidx, idx) - 1
        cur_frame = fidx[ptr] if ptr >= 0 else None

        # Auto-follow keeps that event on screen; manual wheel scrolling overrides
        # it (re-enabled whenever the playhead moves, see goto()).
        scroll = _log_window_start(state["log_scroll"], state["log_follow"], ptr, n, visible)
        state["log_scroll"] = scroll

        if n > visible:
            shown = f"{scroll + 1}-{min(n, scroll + visible)} / {n}"
            ax_log.set_title(f"action log  ({shown})", color="#cccccc",
                             fontsize=9, pad=4)
        else:
            ax_log.set_title(f"action log  ({n} events)", color="#cccccc",
                             fontsize=9, pad=4)

        y = 0.98
        for j in range(scroll, min(n, scroll + visible)):
            e = log[j]
            current = fidx[j] == cur_frame
            if e.rod_label.startswith("GOAL"):
                color = LOG_GOAL_COLOR
            elif e.team == -1:
                color = LOG_PHYSICS_COLOR
            else:
                color = TEAM_COLOR[e.team]
            ax_log.text(
                0.015, y, ("> " if current else "  ") + _format_log_line(e),
                color=color, fontsize=LOG_FONT_SIZE, family="monospace", va="top",
                ha="left", transform=ax_log.transAxes,
                fontweight="bold" if current else "normal",
                alpha=1.0 if current else LOG_DIM_ALPHA,
            )
            y -= dy

    def render(idx: int) -> None:
        _render_frame(
            ax, state["field"], state["frames"],
            state["ball_xs"], state["ball_ys"], state["foot_xs"],
            idx, show_reach, trail_length, state["title"],
        )
        if ax_log is not None:
            render_log(idx)
        fig.canvas.draw_idle()

    # --- Frame slider --------------------------------------------------------
    if has_settings:
        ax_slider = fig.add_axes([0.12, 0.53, 0.74, 0.018], facecolor="#333333")
    else:
        ax_slider = fig.add_axes([0.12, 0.15, 0.76, 0.03], facecolor="#333333")
    slider = Slider(ax_slider, "frame", 0, n - 1, valinit=0,
                    valstep=1, color="#4C9BE8")
    slider.label.set_color("white")
    slider.valtext.set_color("white")

    def _pause() -> None:
        state["playing"] = False
        btn_play.label.set_text("play")
        fig.canvas.draw_idle()

    def goto(idx: int) -> None:
        idx = max(0, min(state["n"] - 1, idx))
        state["idx"] = idx
        state["log_follow"] = True   # moving the playhead re-engages auto-scroll
        slider.eventson = False
        slider.set_val(idx)
        slider.eventson = True
        render(idx)

    def on_slider(val: float) -> None:
        _pause()
        state["idx"] = int(val)
        state["log_follow"] = True   # scrubbing re-engages auto-scroll
        render(state["idx"])

    slider.on_changed(on_slider)

    # --- Playback timer ------------------------------------------------------
    timer = fig.canvas.new_timer(interval=max(1, int(1000 / fps)))

    def _tick() -> None:
        if not state["playing"]:
            return
        if state["idx"] >= state["n"] - 1:
            _pause()
        else:
            goto(state["idx"] + 1)

    timer.add_callback(_tick)

    # --- Buttons -------------------------------------------------------------
    def _toggle(_event=None) -> None:
        if state["idx"] >= state["n"] - 1:
            goto(0)
        state["playing"] = not state["playing"]
        btn_play.label.set_text("pause" if state["playing"] else "play")
        fig.canvas.draw_idle()

    def _make_btn(rect, label):
        b = Button(fig.add_axes(rect), label, color="#333333", hovercolor="#4C9BE8")
        b.label.set_color("white")
        return b

    if has_settings:
        btn_back_s = _make_btn([0.18, 0.075, 0.10, 0.045], "<< 1s")
        btn_back_t = _make_btn([0.29, 0.075, 0.10, 0.045], "< tick")
        btn_play   = _make_btn([0.40, 0.075, 0.16, 0.045], "play")
        btn_fwd_t  = _make_btn([0.57, 0.075, 0.10, 0.045], "tick >")
        btn_fwd_s  = _make_btn([0.68, 0.075, 0.10, 0.045], "1s >>")
    else:
        btn_back_s = _make_btn([0.20, 0.05, 0.10, 0.05], "<< 1s")
        btn_back_t = _make_btn([0.31, 0.05, 0.10, 0.05], "< tick")
        btn_play   = _make_btn([0.42, 0.05, 0.16, 0.05], "play")
        btn_fwd_t  = _make_btn([0.59, 0.05, 0.10, 0.05], "tick >")
        btn_fwd_s  = _make_btn([0.70, 0.05, 0.10, 0.05], "1s >>")

    btn_back_s.on_clicked(lambda e: (_pause(), goto(state["idx"] - fps)))
    btn_back_t.on_clicked(lambda e: (_pause(), goto(state["idx"] - 1)))
    btn_play.on_clicked(_toggle)
    btn_fwd_t.on_clicked(lambda e: (_pause(), goto(state["idx"] + 1)))
    btn_fwd_s.on_clicked(lambda e: (_pause(), goto(state["idx"] + fps)))

    # --- Settings panel ------------------------------------------------------
    if has_settings:
        skill_def = float(init.get("skill_default", 0.5))
        ant_def   = float(init.get("anticipation_default", config.DEFAULT_ANTICIPATION))

        def _make_tb(rect, label, initial):
            tb_ax = fig.add_axes(rect, facecolor="#333333")
            tb = TextBox(tb_ax, label, initial=initial,
                         color="#333333", hovercolor="#555555")
            tb.label.set_color("white")
            tb.text_disp.set_color("white")
            return tb

        def _make_slider(rect, label, initial, color):
            sl_ax = fig.add_axes(rect, facecolor="#333333")
            sl = Slider(sl_ax, label, 0.0, 1.0, valinit=initial,
                        valstep=0.01, color=color)
            sl.label.set_color("white")
            sl.label.set_fontsize(8)
            sl.valtext.set_color("white")
            sl.valtext.set_fontsize(8)
            return sl

        # Slider inits: fall back to the team default when the value is unset.
        sk0 = init.get("skill0") or (skill_def, skill_def)
        sk1 = init.get("skill1") or (skill_def, skill_def)
        a0  = init.get("anticipation0")
        a1  = init.get("anticipation1")
        a0  = ant_def if a0 is None else a0
        a1  = ant_def if a1 is None else a1
        C0, C1 = TEAM_COLOR[0], TEAM_COLOR[1]

        # Top row: strategy pickers (T0, T1) and the kickoff toggle on the right.
        t0_box = [0.075, 0.475, 0.165, 0.030]
        t1_box = [0.405, 0.475, 0.165, 0.030]
        tb_t0 = _make_tb(t0_box, "T0 ", init.get("team0", ""))
        tb_t1 = _make_tb(t1_box, "T1 ", init.get("team1", ""))
        btn_kickoff = _make_btn([0.70, 0.474, 0.16, 0.032], f"Kickoff: {state['kickoff']}")

        # Strategy dropdowns: an arrow beside each box opens a pick-list of the
        # hardcoded strategies plus any genome .json paths. Picking fills the box
        # (without auto-running); typing still works.
        def _set_tb_silent(tb, value) -> None:
            tb.eventson = False
            tb.set_val(str(value))
            tb.eventson = True

        def _close_dropdown(_event=None) -> None:
            dd = state.get("dropdown")
            if not dd:
                return
            for b in dd["btns"]:
                b.disconnect_events()
            for axx in dd["axes"]:
                axx.remove()
            state["dropdown"] = None
            fig.canvas.draw_idle()

        def _open_dropdown(target_tb, box_rect) -> None:
            _close_dropdown()
            opts = list(strategy_choices or [])
            x, y0, w, _h = box_rect
            pw = max(w + 0.02, 0.16)
            row_h = min(0.028, (y0 - 0.16) / max(1, len(opts)))
            axes, btns = [], []
            for i, opt in enumerate(opts):
                b_ax = fig.add_axes([x, y0 - (i + 1) * row_h, pw, row_h],
                                    facecolor="#262626")
                b_ax.set_zorder(30)
                b = Button(b_ax, _choice_label(opt), color="#262626", hovercolor="#4C9BE8")
                b.label.set_color("white")
                b.label.set_fontsize(7.5)
                b.label.set_horizontalalignment("left")
                b.label.set_x(0.04)
                b.on_clicked(lambda _e, o=opt, tb=target_tb:
                             (_set_tb_silent(tb, o), _close_dropdown()))
                axes.append(b_ax)
                btns.append(b)
            state["dropdown"] = {"axes": axes, "btns": btns, "owner": target_tb}
            fig.canvas.draw_idle()

        arrow_axes: list = []
        if strategy_choices:
            def _toggle_dropdown(target_tb, box_rect) -> None:
                dd = state.get("dropdown")
                if dd and dd.get("owner") is target_tb:
                    _close_dropdown()
                else:
                    _open_dropdown(target_tb, box_rect)

            btn_t0_dd = _make_btn([t0_box[0] + t0_box[2] + 0.004, 0.475, 0.024, 0.030], "v")
            btn_t1_dd = _make_btn([t1_box[0] + t1_box[2] + 0.004, 0.475, 0.024, 0.030], "v")
            btn_t0_dd.on_clicked(lambda _e: _toggle_dropdown(tb_t0, t0_box))
            btn_t1_dd.on_clicked(lambda _e: _toggle_dropdown(tb_t1, t1_box))
            arrow_axes = [btn_t0_dd.ax, btn_t1_dd.ax]

            def _on_press_close(event) -> None:
                # Close an open dropdown when clicking anywhere that is neither an
                # option nor a dropdown arrow (those manage their own open/close).
                dd = state.get("dropdown")
                if not dd or event.inaxes in dd["axes"] or event.inaxes in arrow_axes:
                    return
                _close_dropdown()

            fig.canvas.mpl_connect("button_press_event", _on_press_close)

        # Skill / anticipation sliders (team 0 then team 1).
        sl_acc0 = _make_slider([0.20, 0.430, 0.60, 0.015], "T0 accuracy", sk0[0], C0)
        sl_pow0 = _make_slider([0.20, 0.397, 0.60, 0.015], "T0 power",    sk0[1], C0)
        sl_ant0 = _make_slider([0.20, 0.364, 0.60, 0.015], "T0 anticip",  a0,     C0)
        sl_acc1 = _make_slider([0.20, 0.323, 0.60, 0.015], "T1 accuracy", sk1[0], C1)
        sl_pow1 = _make_slider([0.20, 0.290, 0.60, 0.015], "T1 power",    sk1[1], C1)
        sl_ant1 = _make_slider([0.20, 0.257, 0.60, 0.015], "T1 anticip",  a1,     C1)

        # Bottom row: resets, then seed (beside Re-run for quick tweaks) and Re-run.
        btn_skill_def = _make_btn([0.10, 0.185, 0.17, 0.038], "Skill default")
        btn_ant_def   = _make_btn([0.28, 0.185, 0.17, 0.038], "Anticip default")
        tb_seed       = _make_tb([0.545, 0.185, 0.075, 0.038], "seed ", init.get("seed", ""))
        btn_seed_none = _make_btn([0.625, 0.185, 0.05, 0.038], "none")
        btn_rerun     = _make_btn([0.69, 0.185, 0.13, 0.038], "↻ Re-run")
        btn_rerun_play = _make_btn([0.825, 0.185, 0.13, 0.038], "↻ + ▶")

        fig.text(0.5, 0.15,
                 "strategies: SmackBall  AimAtGap  HardOffense  DefensiveWall  "
                 "TiltAndGap  ReactiveBlock  (or path to a genome .json)",
                 color="#777777", fontsize=7.5, ha="center")

        def _set_seed_text(value) -> None:
            # Update the seed box without firing its on_submit (avoids re-run loops).
            tb_seed.eventson = False
            tb_seed.set_val(str(value))
            tb_seed.eventson = True

        def _reset_skill(_event=None) -> None:
            for sl in (sl_acc0, sl_pow0, sl_acc1, sl_pow1):
                sl.set_val(skill_def)

        def _reset_ant(_event=None) -> None:
            for sl in (sl_ant0, sl_ant1):
                sl.set_val(ant_def)

        def _toggle_kickoff(_event=None) -> None:
            state["kickoff"] = 1 - state["kickoff"]
            btn_kickoff.label.set_text(f"Kickoff: {state['kickoff']}")
            fig.canvas.draw_idle()

        btn_skill_def.on_clicked(_reset_skill)
        btn_ant_def.on_clicked(_reset_ant)
        btn_seed_none.on_clicked(lambda _e: _set_seed_text("none"))
        btn_kickoff.on_clicked(_toggle_kickoff)

        def _do_rerun(_event=None) -> None:
            _pause()
            ax.set_title("Running...", color="white")
            fig.canvas.draw()
            fig.canvas.flush_events()

            seed_raw = tb_seed.text.strip()
            seed_val = None if seed_raw.lower() in ("", "none", "random") else int(seed_raw)

            settings = {
                "seed":          seed_val,
                "team0":         tb_t0.text.strip(),
                "team1":         tb_t1.text.strip(),
                "skill0":        (sl_acc0.val, sl_pow0.val),
                "skill1":        (sl_acc1.val, sl_pow1.val),
                "anticipation0": sl_ant0.val,
                "anticipation1": sl_ant1.val,
                "kickoff":       state["kickoff"],
            }

            try:
                result = rerun_fn(settings)
            except Exception as exc:
                ax.set_title(f"Error: {exc}", color="#ff6666")
                fig.canvas.draw_idle()
                return

            new_field, new_frames, new_title, seed_used = result[:4]
            new_log = result[4] if len(result) > 4 else []

            new_n = len(new_frames)
            if new_n == 0:
                return

            state["field"]   = new_field
            state["frames"]  = new_frames
            state["ball_xs"] = [f.ball_x for f in new_frames]
            state["ball_ys"] = [f.ball_y for f in new_frames]
            state["foot_xs"] = _compute_foot_offsets(new_field, new_frames)
            state["n"]       = new_n
            state["title"]   = new_title
            state["idx"]     = 0
            state["action_log"]    = list(new_log) if (has_log and new_log) else []
            state["log_frame_idx"] = _events_by_frame(new_frames, state["action_log"])
            state["log_scroll"]    = 0
            state["log_follow"]    = True
            btn_play.label.set_text("play")

            slider.valmax = new_n - 1
            slider.ax.set_xlim(0, new_n - 1)
            slider.eventson = False
            slider.set_val(0)
            slider.eventson = True

            _set_seed_text(seed_used)
            render(0)

        def _do_rerun_play(_event=None) -> None:
            _do_rerun()
            if state["n"] > 1:
                state["playing"] = True
                btn_play.label.set_text("pause")
                fig.canvas.draw_idle()

        btn_rerun.on_clicked(_do_rerun)
        btn_rerun_play.on_clicked(_do_rerun_play)
        for _tb in (tb_seed, tb_t0, tb_t1):
            _tb.on_submit(lambda _text: _do_rerun())

    # --- Keyboard ------------------------------------------------------------
    def on_key(event) -> None:
        if event.key == " ":
            _toggle()
        elif event.key == "left":
            _pause(); goto(state["idx"] - fps)
        elif event.key == "right":
            _pause(); goto(state["idx"] + fps)
        elif event.key == ",":
            _pause(); goto(state["idx"] - 1)
        elif event.key == ".":
            _pause(); goto(state["idx"] + 1)
        elif event.key == "home":
            _pause(); goto(0)
        elif event.key == "end":
            _pause(); goto(state["n"] - 1)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # --- Mouse wheel: pan the log panel manually -----------------------------
    def on_scroll(event) -> None:
        if ax_log is None or event.inaxes is not ax_log:
            return
        n = len(state["action_log"])
        max_scroll = max(0, n - state["log_visible"])
        if max_scroll == 0:
            return
        step = -LOG_SCROLL_STEP if event.button == "up" else LOG_SCROLL_STEP
        state["log_scroll"] = max(0, min(max_scroll, state["log_scroll"] + step))
        state["log_follow"] = False   # user took control until the playhead moves
        render(state["idx"])

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    fig.text(
        0.5, 0.005,
        "space: play/pause      left / right: -/+ 1 s      , / . : -/+ 1 tick"
        + ("      wheel over log: scroll" if has_log else ""),
        color="#aaaaaa", fontsize=8, ha="center",
    )

    render(0)
    timer.start()
    plt.show()


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
