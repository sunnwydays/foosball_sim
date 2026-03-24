"""
playground.py — Interactive physics playground for tuning ball constants.

A mini foosball field with one rod (2 players) and a goal.
Sliders let you tweak vx, vy, friction, restitution, and ball radius,
then press Launch to watch the ball bounce around.

Usage:
    python playground.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, CheckButtons

from field import BallState, Field
from physics import step_ball

# ---------------------------------------------------------------------------
# Colours (match visualization.py)
# ---------------------------------------------------------------------------
TEAM_COLOR  = {0: "#4C9BE8", 1: "#E8724C"}
FIELD_GREEN = "#2d6a2d"
LINE_WHITE  = "#ffffff"
WALL_GRAY   = "#888888"
BALL_COLOR  = "#f5f542"

# ---------------------------------------------------------------------------
# Mini field setup
# ---------------------------------------------------------------------------
FIELD_DEPTH  = 60.0
FIELD_WIDTH  = 40.0
GOAL_WIDTH   = 15.0
ROD_X        = 30.0   # rod at the midpoint

def build_field(rod_x: float = ROD_X, rod_y_offset: float = 0.0) -> Field:
    f = Field(
        depth=FIELD_DEPTH,
        width=FIELD_WIDTH,
        goal_width=GOAL_WIDTH,
        rod_x_positions=[rod_x],
        rod_configs=[(1, 2)],
    )
    f.rods[0].set_offset(rod_y_offset)
    return f

mini_field = build_field()

# Ball start position (left side, centered)
BALL_START_X = 10.0
BALL_START_Y = FIELD_WIDTH / 2

# Physics timestep (fine for smooth animation)
PHYSICS_DT = 0.005     # 200 Hz physics
MAX_TIME   = 10.0       # seconds
MAX_STEPS  = int(MAX_TIME / PHYSICS_DT)

# Animation
ANIM_FPS     = 60
FRAME_SKIP   = max(1, int(1 / (ANIM_FPS * PHYSICS_DT)))  # show every Nth step
TRAIL_LENGTH = 80


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_mini_field(ax: plt.Axes, field: Field, ball_radius: float) -> None:
    """Draw the mini field background, walls, rod, and players."""
    ax.set_facecolor(FIELD_GREEN)

    # Field outline
    rect = patches.Rectangle(
        (0, 0), field.depth, field.width,
        linewidth=2, edgecolor=LINE_WHITE, facecolor="none", zorder=2,
    )
    ax.add_patch(rect)

    # Goal (right side — team 0 scores here)
    goal = field.right_goal
    goal_vis_depth = 4.0
    g = patches.Rectangle(
        (field.depth, goal.y_min), goal_vis_depth, goal.y_max - goal.y_min,
        linewidth=1.5, edgecolor=LINE_WHITE,
        facecolor=TEAM_COLOR[goal.scoring_team], alpha=0.6, zorder=2,
    )
    ax.add_patch(g)

    # Left wall goal (team 1 scores here)
    lgoal = field.left_goal
    lg = patches.Rectangle(
        (-goal_vis_depth, lgoal.y_min), goal_vis_depth, lgoal.y_max - lgoal.y_min,
        linewidth=1.5, edgecolor=LINE_WHITE,
        facecolor=TEAM_COLOR[lgoal.scoring_team], alpha=0.6, zorder=2,
    )
    ax.add_patch(lg)

    # Rod and players
    rod = field.rods[0]
    color = TEAM_COLOR[rod.team]
    player_alpha = 0.2 if rod.up else 1.0

    # Rod line
    ax.plot(
        [rod.x, rod.x], [0, field.width],
        color=color, linewidth=1.5, alpha=0.4, zorder=2,
    )

    # Players
    for py in rod.player_positions:
        player_rect = patches.Rectangle(
            (rod.x - rod.thickness / 2, py - rod.width / 2),
            rod.thickness, rod.width,
            linewidth=0, facecolor=color, alpha=player_alpha, zorder=3,
        )
        ax.add_patch(player_rect)

    ax.set_xlim(-6, field.depth + 6)
    ax.set_ylim(-4, field.width + 4)
    ax.set_aspect("equal")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(WALL_GRAY)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    fig = plt.figure(figsize=(12, 6), facecolor="#1a1a1a")
    fig.canvas.manager.set_window_title("Foosball Physics Playground")

    # --- Layout: field on left, sliders on right ---
    ax_field = fig.add_axes([0.02, 0.05, 0.62, 0.90])

    # Slider axes (left, bottom, width, height)
    slider_x      = 0.72
    slider_w      = 0.24
    slider_h      = 0.03
    slider_gap    = 0.065
    top           = 0.90

    row = 0
    def next_slider_ax():
        nonlocal row
        ax = fig.add_axes([slider_x, top - row * slider_gap, slider_w, slider_h])
        row += 1
        return ax

    ax_vx   = next_slider_ax()
    ax_vy   = next_slider_ax()
    ax_fric = next_slider_ax()
    ax_rest = next_slider_ax()
    ax_rad  = next_slider_ax()
    ax_rodx = next_slider_ax()
    ax_rody = next_slider_ax()

    ax_rodup = fig.add_axes([slider_x, top - row * slider_gap, slider_w, slider_h])
    row += 1
    chk_rodup = CheckButtons(ax_rodup, ["Rod Up"], [False])
    ax_rodup.set_facecolor("#1a1a1a")
    for label in chk_rodup.labels:
        label.set_color("white")
        label.set_fontsize(9)

    ax_btn  = fig.add_axes([slider_x, top - (row + 0.5) * slider_gap, slider_w, 0.05])

    # Status text
    ax_status = fig.add_axes([slider_x, top - (row + 2) * slider_gap, slider_w, 0.08])
    ax_status.set_facecolor("#1a1a1a")
    ax_status.axis("off")
    status_text = ax_status.text(
        0.5, 0.5, "Ready — click field, then Launch",
        color="white", fontsize=9, ha="center", va="center",
        transform=ax_status.transAxes,
    )

    # --- Sliders ---
    slider_color = "#555555"
    s_vx   = Slider(ax_vx,   "vx",        -200, 200,  valinit=80,   color=slider_color, valstep=5)
    s_vy   = Slider(ax_vy,   "vy",        -200, 200,  valinit=30,   color=slider_color, valstep=5)
    s_fric = Slider(ax_fric, "friction",    0,   80,   valinit=15,   color=slider_color, valstep=1)
    s_rest = Slider(ax_rest, "restitution", 0.0, 1.0,  valinit=0.85, color=slider_color, valstep=0.05)
    s_rad  = Slider(ax_rad,  "ball radius", 0.0, 3.0,  valinit=1.0,  color=slider_color, valstep=0.1)
    s_rodx = Slider(ax_rodx, "rod x",       1,   FIELD_DEPTH - 1, valinit=ROD_X, color=slider_color, valstep=1)
    s_rody = Slider(ax_rody, "rod y",       -FIELD_WIDTH / 2, FIELD_WIDTH / 2, valinit=0.0, color=slider_color, valstep=0.5)

    # Style slider labels
    for s in (s_vx, s_vy, s_fric, s_rest, s_rad, s_rodx, s_rody):
        s.label.set_color("white")
        s.label.set_fontsize(9)
        s.valtext.set_color("white")
        s.valtext.set_fontsize(8)

    # --- Launch button ---
    btn = Button(ax_btn, "Launch", color="#444444", hovercolor="#666666")
    btn.label.set_color("white")
    btn.label.set_fontsize(11)

    # Keep references to prevent GC
    state = {"anim": None, "ball_x": BALL_START_X, "ball_y": BALL_START_Y, "running": False}

    def _set_button_launch() -> None:
        state["running"] = False
        btn.label.set_text("Launch")
        btn.color = "#444444"
        btn.hovercolor = "#666666"

    def _set_button_stop() -> None:
        state["running"] = True
        btn.label.set_text("Stop")
        btn.color = "#884444"
        btn.hovercolor = "#aa6666"

    def _stop_anim() -> None:
        if state["anim"] is not None:
            try:
                state["anim"].event_source.stop()
            except AttributeError:
                pass
            state["anim"] = None
        _set_button_launch()

    def _current_field() -> Field:
        f = build_field(rod_x=s_rodx.val, rod_y_offset=s_rody.val)
        f.rods[0].up = chk_rodup.get_status()[0]
        return f

    def _redraw_field_with_marker() -> None:
        """Redraw the field with the current ball start marker and velocity arrow."""
        ax_field.cla()
        draw_mini_field(ax_field, _current_field(), s_rad.val)
        bx, by = state["ball_x"], state["ball_y"]
        ax_field.plot(bx, by, 'o', color=BALL_COLOR, markersize=8, alpha=0.6, zorder=5)

        # Velocity arrow (scaled to fit the field visually)
        vx, vy = s_vx.val, s_vy.val
        speed = (vx ** 2 + vy ** 2) ** 0.5
        if speed > 1:
            scale = 0.1
            ax_field.annotate(
                "", xy=(bx + vx * scale, by + vy * scale),
                xytext=(bx, by),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5),
                zorder=7,
            )
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        """Click on the field to set ball start position."""
        if event.inaxes != ax_field:
            return
        # Clamp to field bounds
        bx = max(0.0, min(FIELD_DEPTH, event.xdata))
        by = max(0.0, min(FIELD_WIDTH, event.ydata))
        state["ball_x"] = bx
        state["ball_y"] = by

        # Stop any running animation and redraw
        _stop_anim()
        _redraw_field_with_marker()
        status_text.set_text(f"Start: ({bx:.1f}, {by:.1f})")

    fig.canvas.mpl_connect("button_press_event", on_click)

    # Redraw when relevant sliders change
    def on_slider_change(val) -> None:
        if state["anim"] is None:
            _redraw_field_with_marker()
    s_vx.on_changed(on_slider_change)
    s_vy.on_changed(on_slider_change)
    s_rodx.on_changed(on_slider_change)
    s_rody.on_changed(on_slider_change)
    chk_rodup.on_clicked(lambda _: on_slider_change(None))

    def on_button(event) -> None:
        # Toggle: if running, stop and reset; if stopped, launch
        if state["running"]:
            _stop_anim()
            _redraw_field_with_marker()
            status_text.set_text("Stopped — press Launch")
            fig.canvas.draw_idle()
            return

        # --- Launch ---
        _stop_anim()  # clean up any stale animation
        vx          = s_vx.val
        vy          = s_vy.val
        friction    = s_fric.val
        restitution = s_rest.val
        ball_radius = s_rad.val

        # Build field from current rod sliders
        field = _current_field()

        # Run physics simulation
        ball = BallState(x=state["ball_x"], y=state["ball_y"], vx=vx, vy=vy)
        positions = [(ball.x, ball.y, ball.vx, ball.vy)]
        result = "play"

        for _ in range(MAX_STEPS):
            result = step_ball(ball, field, PHYSICS_DT,
                               friction=friction,
                               restitution=restitution,
                               ball_radius=ball_radius)
            positions.append((ball.x, ball.y, ball.vx, ball.vy))
            if result != "play":
                break

        # Downsample for animation
        frames = positions[::FRAME_SKIP]
        if not frames or frames[-1] != positions[-1]:
            frames.append(positions[-1])

        total_time = len(positions) * PHYSICS_DT
        status_text.set_text(
            f"{result}  |  {total_time:.2f}s  |  {len(frames)} frames"
        )
        _set_button_stop()

        # --- Animate ---
        def update(idx: int) -> None:
            ax_field.cla()
            draw_mini_field(ax_field, field, ball_radius)

            bx, by, bvx, bvy = frames[idx]

            # Trail
            start = max(0, idx - TRAIL_LENGTH)
            trail_x = [f[0] for f in frames[start:idx + 1]]
            trail_y = [f[1] for f in frames[start:idx + 1]]
            if len(trail_x) > 1:
                ax_field.plot(trail_x, trail_y, color=BALL_COLOR,
                              linewidth=1, alpha=0.3, zorder=4)

            # Ball
            ball_circle = plt.Circle(
                (bx, by), radius=max(ball_radius, 0.5),
                color=BALL_COLOR, zorder=6,
            )
            ax_field.add_patch(ball_circle)

            # Velocity arrow
            speed = (bvx ** 2 + bvy ** 2) ** 0.5
            if speed > 5:
                scale = 0.08
                ax_field.annotate(
                    "", xy=(bx + bvx * scale, by + bvy * scale),
                    xytext=(bx, by),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.0),
                    zorder=7,
                )

            # Speed readout
            ax_field.text(
                2, field.width + 2,
                f"speed: {speed:.1f} cm/s",
                color="white", fontsize=8, zorder=10,
            )

            # Auto-reset to Launch when animation finishes
            if idx == len(frames) - 1:
                state["anim"] = None
                _set_button_launch()
                fig.canvas.draw_idle()

        state["anim"] = animation.FuncAnimation(
            fig, update, frames=len(frames),
            interval=max(1, 1000 // ANIM_FPS), repeat=False,
        )
        fig.canvas.draw_idle()

    btn.on_clicked(on_button)

    # Initial field drawing
    _redraw_field_with_marker()

    plt.show()


if __name__ == "__main__":
    main()
