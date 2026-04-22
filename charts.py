"""
charts.py — Plotly visualization helpers for RAGC simulator.

Exports:
    make_heatmap(result_a, result_b)          → side-by-side pressure heat maps
    make_spill_bar(result_a, result_b)        → grouped bar: spill count + cost
    make_score_scatter(result, instructions)  → score scatter for ranked mode
"""

from __future__ import annotations
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from engine import AllocationResult, Instruction, LiveRange


# ─────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────
LIGHT_BG   = "#f8f9fa"
PANEL_BG   = "#ffffff"
GRID_COLOR = "#dee2e6"
TEXT_COLOR = "#212529"
ACCENT_A   = "#339af0"   # Classic — blue
ACCENT_B   = "#40c057"   # Ranked  — green
SPILL_RED  = "#fa5252"
GOLD       = "#fcc419"

_LAYOUT_BASE = dict(
    paper_bgcolor=LIGHT_BG,
    plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_COLOR, family="Helvetica Neue, Arial, sans-serif", size=12),
    margin=dict(l=60, r=30, t=50, b=60),
)


# ─────────────────────────────────────────────────────────────────
# 1.  REGISTER PRESSURE HEAT MAP
# ─────────────────────────────────────────────────────────────────

def make_heatmap(result_a: AllocationResult, result_b: AllocationResult) -> go.Figure:
    """
    Two heat maps side by side.
    X-axis: instruction index
    Y-axis: variable
    Color:  register pressure (number of simultaneously live variables)
            overlaid with spill markers
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Mode A — Classic Chaitin", "Mode B — Pressure-Ranked"],
        horizontal_spacing=0.12,
    )

    for col_idx, result in enumerate([result_a, result_b], start=1):
        df = pd.DataFrame(result.pressure_timeline)
        if df.empty:
            continue

        all_vars   = sorted(df["var"].unique())
        all_instrs = sorted(df["instruction"].unique())
        var_to_y   = {v: i for i, v in enumerate(all_vars)}

        # Build z matrix: rows = variables, cols = instructions
        z = [[0] * len(all_instrs) for _ in all_vars]
        for row in result.pressure_timeline:
            vi = var_to_y[row["var"]]
            ii = all_instrs.index(row["instruction"])
            # Intensity = pressure; loop instructions get a boost for visual clarity
            intensity = row["pressure"] + row["loop_depth"] * 2
            z[vi][ii] = intensity

        x_labels = [f"#{i}" for i in all_instrs]
        y_labels = all_vars

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=x_labels,
                y=y_labels,
                colorscale=[
                    [0.0,  PANEL_BG],
                    [0.15, "#e9ecef"],
                    [0.4,  "#ced4da"],
                    [0.6,  "#adb5bd"],
                    [0.75, "#74c0fc"],
                    [0.88, "#b197fc"],
                    [1.0,  SPILL_RED],
                ],
                showscale=(col_idx == 2),
                colorbar=dict(
                    title=dict(text="Pressure", font=dict(color=TEXT_COLOR)),
                    tickfont=dict(color=TEXT_COLOR),
                ) if col_idx == 2 else {},
                hovertemplate=(
                    "<b>Var:</b> %{y}<br>"
                    "<b>Instr:</b> %{x}<br>"
                    "<b>Pressure:</b> %{z}<extra></extra>"
                ),
            ),
            row=1, col=col_idx,
        )

        # Overlay spill markers
        for spill_var in result.spills:
            if spill_var in var_to_y:
                yi = var_to_y[spill_var]
                fig.add_trace(
                    go.Scatter(
                        x=x_labels,
                        y=[spill_var] * len(x_labels),
                        mode="markers",
                        marker=dict(
                            symbol="x",
                            size=9,
                            color=SPILL_RED,
                            line=dict(width=1.5, color="white"),
                        ),
                        name=f"Spill: {spill_var}",
                        showlegend=(col_idx == 1),
                        hovertemplate=f"<b>SPILL</b>: {spill_var}<extra></extra>",
                    ),
                    row=1, col=col_idx,
                )

    fig.update_layout(
        **_LAYOUT_BASE,
        height=380,
        title=dict(
            text="Register Pressure Heat Map  (color intensity = live vars + loop depth boost)",
            font=dict(size=13, color=TEXT_COLOR),
            x=0.5,
        ),
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        tickfont=dict(size=9),
        title_text="Instruction #",
        title_font=dict(size=10),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        tickfont=dict(size=10),
        title_text="Variable",
        title_font=dict(size=10),
    )
    # Fix subplot titles color
    for ann in fig.layout.annotations:
        ann.font.color = TEXT_COLOR

    return fig


# ─────────────────────────────────────────────────────────────────
# 2.  SPILL COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────────

def make_spill_bar(result_a: AllocationResult, result_b: AllocationResult) -> go.Figure:
    """Grouped bar: spill count and weighted spill cost for both modes."""

    categories  = ["Spill Count", "Spill Cost (weighted)"]
    values_a    = [len(result_a.spills), result_a.spill_cost]
    values_b    = [len(result_b.spills), result_b.spill_cost]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Classic Chaitin",
        x=categories,
        y=values_a,
        marker_color=ACCENT_A,
        text=[str(v) for v in values_a],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=13),
    ))
    fig.add_trace(go.Bar(
        name="Pressure-Ranked",
        x=categories,
        y=values_b,
        marker_color=ACCENT_B,
        text=[str(v) for v in values_b],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=13),
    ))

    # Improvement annotations
    for i, (va, vb) in enumerate(zip(values_a, values_b)):
        if va > 0:
            delta = va - vb
            pct   = (delta / va) * 100
            color = ACCENT_B if delta >= 0 else SPILL_RED
            sign  = "−" if delta >= 0 else "+"
            label = f"{sign}{abs(pct):.0f}%"
            fig.add_annotation(
                x=categories[i],
                y=max(va, vb) * 1.22,
                text=label,
                showarrow=False,
                font=dict(color=color, size=12, family="monospace"),
            )

    fig.update_layout(
        **_LAYOUT_BASE,
        barmode="group",
        height=320,
        title=dict(
            text="Spill Comparison — Classic vs. Pressure-Ranked",
            font=dict(size=13, color=TEXT_COLOR),
            x=0.5,
        ),
        legend=dict(
            font=dict(color=TEXT_COLOR),
            bgcolor="rgba(0,0,0,0)",
            bordercolor=GRID_COLOR,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            title_text="Value",
        ),
        xaxis=dict(showgrid=False),
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# 3.  SCORE SCATTER (RANKED MODE)
# ─────────────────────────────────────────────────────────────────

def make_score_scatter(result: AllocationResult, instructions: list[Instruction]) -> go.Figure:
    """
    Scatter plot of composite scores assigned to each variable during the
    ranked simplification phase.  Spilled variables are highlighted.
    """
    simplify_steps = [
        s for s in result.steps
        if s.action in ("simplify", "potential_spill") and s.score is not None
    ]

    if not simplify_steps:
        fig = go.Figure()
        fig.update_layout(
            **_LAYOUT_BASE,
            height=260,
            title=dict(
                text="Composite Score per Node (Ranked mode only)",
                font=dict(size=13, color=TEXT_COLOR),
                x=0.5,
            ),
        )
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="Score(n) = w₁ / (def_use_dist × (depth+1) × (degree+1) × w₂)  —  run in Ranked mode to see scores",
            showarrow=False,
            font=dict(color=GRID_COLOR, size=14),
        )
        return fig

    nodes  = [s.node  for s in simplify_steps]
    scores = [s.score for s in simplify_steps]
    order  = list(range(len(nodes)))
    colors = [
        SPILL_RED if s.action == "potential_spill" else ACCENT_B
        for s in simplify_steps
    ]
    symbols = [
        "diamond" if s.action == "potential_spill" else "circle"
        for s in simplify_steps
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=order,
        y=scores,
        mode="markers+lines+text",
        text=nodes,
        textposition="top center",
        textfont=dict(size=9, color=TEXT_COLOR),
        marker=dict(
            color=colors,
            symbol=symbols,
            size=11,
            line=dict(width=1, color="white"),
        ),
        line=dict(color=GRID_COLOR, width=1, dash="dot"),
        hovertemplate="<b>%{text}</b><br>Score: %{y:.4f}<br>Order: %{x}<extra></extra>",
    ))

    # Threshold line where potential spills started
    ps_indices = [i for i, s in enumerate(simplify_steps) if s.action == "potential_spill"]
    if ps_indices:
        fig.add_vline(
            x=ps_indices[0] - 0.5,
            line=dict(color=SPILL_RED, width=1, dash="dash"),
            annotation_text="↑ Potential spills begin",
            annotation_font=dict(color=SPILL_RED, size=10),
        )

    fig.update_layout(
        **_LAYOUT_BASE,
        height=280,
        title=dict(
            text="Composite Score per Node — Ranked Simplification Order",
            font=dict(size=13, color=TEXT_COLOR),
            x=0.5,
        ),
        xaxis=dict(
            title_text="Simplification order (left = first removed)",
            showgrid=False,
        ),
        yaxis=dict(
            title_text="Composite score",
            showgrid=True,
            gridcolor=GRID_COLOR,
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# 4.  LIVE RANGE GANTT
# ─────────────────────────────────────────────────────────────────

def make_live_range_gantt(result: AllocationResult) -> go.Figure:
    """
    Horizontal bar chart showing live range of each variable.
    Coloured by register assignment; spills shown in red.
    Loop-depth indicated by bar opacity.
    """
    fig = go.Figure()

    sorted_ranges = sorted(result.live_ranges, key=lambda lr: (lr.start, lr.var))

    _REG_COLORS = [
        "#339af0","#40c057","#faa2c1","#fa5252",
        "#845ef7","#20c997","#fcc419","#ff922b",
    ]

    for lr in sorted_ranges:
        if lr.var in result.spills:
            color = SPILL_RED
            label = f"{lr.var} [SPILL]"
        elif lr.var in result.assignment:
            reg   = result.assignment[lr.var]
            color = _REG_COLORS[(reg - 1) % len(_REG_COLORS)]
            label = f"{lr.var} → R{reg}"
        else:
            color = GRID_COLOR
            label = lr.var

        opacity = 0.55 + min(lr.loop_depth * 0.15, 0.45)

        fig.add_trace(go.Bar(
            x=[lr.end - lr.start + 1],
            y=[lr.var],
            base=[lr.start],
            orientation="h",
            marker=dict(color=color, opacity=opacity, line=dict(width=0)),
            name=label,
            showlegend=False,
            hovertemplate=(
                f"<b>{lr.var}</b><br>"
                f"Live: #{lr.start} – #{lr.end}<br>"
                f"Loop depth: {lr.loop_depth}<br>"
                f"Assignment: {label.split('→')[-1].strip()}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        barmode="overlay",
        height=max(200, len(result.live_ranges) * 28 + 80),
        title=dict(
            text="Live Ranges  (opacity = loop depth, color = register)",
            font=dict(size=13, color=TEXT_COLOR),
            x=0.5,
        ),
        xaxis=dict(
            title_text="Instruction index",
            showgrid=True,
            gridcolor=GRID_COLOR,
            dtick=1,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=11),
        ),
    )
    return fig
