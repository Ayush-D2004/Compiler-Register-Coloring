"""
app.py — RAGC: Pressure-Ranked Register Allocation Simulator
Run:  streamlit run app.py
"""

import streamlit as st
import graphviz
import pandas as pd
from engine import (
    parse_tac, allocate, EXAMPLE_PROGRAMS, AllocationResult
)
from charts import (
    make_heatmap, make_spill_bar,
    make_score_scatter, make_live_range_gantt
)

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAGC — Register Allocation Simulator",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base */
  html, body, [data-testid="stAppViewContainer"] {
    background-color: #f8f9fa;
    color: #212529;
    font-family: "Helvetica Neue", Arial, sans-serif;
  }
  [data-testid="stSidebar"] { background-color: #f1f3f5; }
  [data-testid="stSidebar"] * { color: #212529 !important; }

  /* Headings */
  h1 { color: #5f3dc4 !important; font-size: 1.7rem !important; }
  h2 { color: #339af0 !important; font-size: 1.25rem !important; }
  h3 { color: #40c057 !important; font-size: 1.05rem !important; }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 12px 16px;
  }
  [data-testid="stMetricLabel"] { color: #868e96 !important; font-size: 0.78rem !important; }
  [data-testid="stMetricValue"] { color: #212529 !important; font-size: 1.6rem !important; }
  [data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

  /* Step cards */
  .step-card {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.82rem;
    font-family: "Courier New", monospace;
  }
  .step-simplify  { border-left: 4px solid #339af0; }
  .step-potential_spill { border-left: 4px solid #fcc419; }
  .step-color     { border-left: 4px solid #40c057; }
  .step-spill     { border-left: 4px solid #fa5252; }

  /* Code block */
  .code-block {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 10px 14px;
    font-family: "Courier New", monospace;
    font-size: 0.82rem;
    white-space: pre;
    overflow-x: auto;
  }

  /* Diff badge */
  .badge-better { color: #40c057; font-weight: 600; }
  .badge-worse  { color: #fa5252; font-weight: 600; }
  .badge-same   { color: #868e96; }

  /* Info box */
  .info-box {
    background: #e7f5ff;
    border: 1px solid #a5d8ff;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.83rem;
    color: #212529;
    margin-bottom: 12px;
  }

  /* Divider */
  hr { border-color: #dee2e6; }

  /* Streamlit widgets */
  .stSlider > div > div { background: #dee2e6 !important; }
  .stTextArea textarea {
    background: #ffffff !important;
    color: #212529 !important;
    border: 1px solid #ced4da !important;
    font-family: "Courier New", monospace !important;
    font-size: 0.82rem !important;
  }
  .stSelectbox > div > div {
    background: #ffffff !important;
    color: #212529 !important;
  }
  .stButton > button {
    background: #f1f3f5 !important;
    color: #212529 !important;
    border: 1px solid #ced4da !important;
    border-radius: 8px !important;
  }
  .stButton > button:hover {
    background: #e9ecef !important;
    border-color: #339af0 !important;
  }
  .stTabs [data-baseweb="tab"] {
    color: #868e96 !important;
    background: transparent !important;
  }
  .stTabs [aria-selected="true"] {
    color: #212529 !important;
    border-bottom: 2px solid #339af0 !important;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    example_choice = st.selectbox(
        "Load example program",
        ["— custom —"] + list(EXAMPLE_PROGRAMS.keys()),
    )

    default_code = (
        EXAMPLE_PROGRAMS[example_choice]
        if example_choice != "— custom —"
        else EXAMPLE_PROGRAMS["Simple"]
    )

    st.markdown("**TAC Code Input**")
    st.markdown(
        "<div class='info-box'>"
        "Write simplified Three-Address Code.<br>"
        "<b>Supported:</b> <code>x = y op z</code>, <code>x = y</code>, "
        "<code>x = const</code><br>"
        "<b>Loop markers:</b> <code>for (cond)</code> / bare <code>}</code><br>"
        "<b>Comments:</b> <code>// ...</code>"
        "</div>",
        unsafe_allow_html=True,
    )
    code_input = st.text_area(
        label="tac_code",
        value=default_code,
        height=280,
        label_visibility="collapsed",
    )

    st.markdown("---")
    k = st.number_input("**k — Number of registers**", min_value=2, max_value=12, value=3)

    st.markdown("---")
    st.markdown("**Ranked mode heuristic weights**")
    w1 = st.slider(
        "w₁ — Def-use distance weight",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="Higher → prioritize variables with long live ranges for survival",
    )
    w2 = st.slider(
        "w₂ — Pressure/depth weight",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="Higher → more aggressively protect loop-heavy, high-pressure nodes",
    )

    st.markdown("---")
    run_btn = st.button("▶  Run Simulation", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#45475a;line-height:1.6'>"
        "<b>RAGC</b> — Pressure-Ranked Register Allocation Simulator<br>"
        "CSL 304 Compilers · IIIT Nagpur · 2026<br><br>"
        "<b>Novel contribution:</b> Pressure-ranked simplification order<br>"
        "Inspired by: PresCount (SJTU) + RL4ReAl (IIT-H)"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("# 🖥️ RAGC — Register Allocation with Graph Coloring Simulator")
st.markdown(
    "<div class='info-box'>"
    "<b>Research Project · CSL 304 Compilers · IIIT Nagpur</b> &nbsp;|&nbsp; "
    "Compares <b>Classic Chaitin</b> (arbitrary simplification) vs. "
    "<b>Pressure-Ranked Chaitin</b> (composite score ordering). "
    "Novel contribution: heuristic simplification order that reduces spill cost "
    "by prioritising loop-heavy, high-pressure variables during the Simplify phase."
    "</div>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────
# RUN ENGINE
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_both(code: str, k: int, w1: float, w2: float):
    instructions = parse_tac(code)
    if not instructions:
        return None, None, None
    res_a = allocate(instructions, k, mode="classic", w1=w1, w2=w2)
    res_b = allocate(instructions, k, mode="ranked",  w1=w1, w2=w2)
    return instructions, res_a, res_b


# Initial run on load
instructions, res_a, res_b = run_both(code_input, k, w1, w2)

if run_btn:
    st.cache_data.clear()
    instructions, res_a, res_b = run_both(code_input, k, w1, w2)

if instructions is None:
    st.warning("⚠️ No valid TAC instructions parsed. Check your code input.")
    st.stop()


# ─────────────────────────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Summary Metrics")

col_spacer, c1, c2, c3, c4, c5, c6 = st.columns([0.3, 1, 1, 1, 1, 1, 1])

spill_delta = len(res_a.spills) - len(res_b.spills)
cost_delta  = res_a.spill_cost  - res_b.spill_cost

with c1:
    st.metric("Instructions", len(instructions))
with c2:
    st.metric("Variables", len(res_a.live_ranges))
with c3:
    st.metric("Registers (k)", k)
with c4:
    st.metric(
        "Classic spills",
        len(res_a.spills),
        delta=f"{'−'+str(spill_delta) if spill_delta>0 else ('+'+str(abs(spill_delta)) if spill_delta<0 else '0')} vs ranked",
        delta_color="inverse",
    )
with c5:
    st.metric(
        "Ranked spills",
        len(res_b.spills),
    )
with c6:
    improvement = f"{(cost_delta/res_a.spill_cost*100):.1f}%" if res_a.spill_cost > 0 else "N/A"
    st.metric(
        "Cost reduction",
        improvement,
        delta="Ranked wins" if cost_delta > 0 else ("Tie" if cost_delta == 0 else "Classic wins"),
        delta_color="normal" if cost_delta >= 0 else "inverse",
    )


# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab_heat, tab_compare, tab_graph, tab_steps, tab_data = st.tabs([
    "🌡️ Pressure Heat Map",
    "📉 Comparison Charts",
    "🕸️ Interference Graph",
    "🔬 Step-by-Step",
    "📋 Raw Data",
])


# ── TAB 1: HEAT MAP ───────────────────────────────────────────────
with tab_heat:
    st.markdown("### Register Pressure Heat Map")
    st.markdown(
        "<div class='info-box'>"
        "Each cell shows how many variables are simultaneously live at that instruction. "
        "Darker/warmer = higher pressure. Loop instructions get an intensity boost. "
        "Red ✕ markers indicate spilled variables."
        "</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(make_heatmap(res_a, res_b), use_container_width=True)

    st.markdown("### Live Range Gantt Charts")
    lr_col_a, lr_col_b = st.columns(2)
    with lr_col_a:
        st.markdown("**Mode A — Classic**")
        st.plotly_chart(make_live_range_gantt(res_a), use_container_width=True)
    with lr_col_b:
        st.markdown("**Mode B — Ranked**")
        st.plotly_chart(make_live_range_gantt(res_b), use_container_width=True)


# ── TAB 2: COMPARISON CHARTS ──────────────────────────────────────
with tab_compare:
    st.markdown("### Spill Count & Cost Comparison")
    st.plotly_chart(make_spill_bar(res_a, res_b), use_container_width=True)

    st.markdown("### Composite Score Breakdown (Ranked mode)")
    st.markdown(
        "<div class='info-box'>"
        "Score(n) = w₁ / (def_use_distance × (depth+1) × (degree+1) × w₂)<br>"
        "<b>Higher score → simplified first</b> (short-lived, linear, low-pressure = cheap to spill). "
        "<b>Lower score → kept alive longer</b> (long-lived, loop-heavy, high-pressure = expensive to spill). "
        "Diamond markers = forced potential spills when all nodes have degree ≥ k."
        "</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(make_score_scatter(res_b, instructions), use_container_width=True)

    # Spill detail tables
    cmp_a, cmp_b = st.columns(2)
    with cmp_a:
        st.markdown("**Classic — spilled variables**")
        if res_a.spills:
            spill_data_a = []
            lr_map = {lr.var: lr for lr in res_a.live_ranges}
            for v in res_a.spills:
                lr = lr_map.get(v)
                spill_data_a.append({
                    "Variable": v,
                    "Loop depth": lr.loop_depth if lr else "?",
                    "Live range": f"{lr.start}–{lr.end}" if lr else "?",
                })
            st.dataframe(
                pd.DataFrame(spill_data_a),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.success("No spills in Classic mode!")

    with cmp_b:
        st.markdown("**Ranked — spilled variables**")
        if res_b.spills:
            spill_data_b = []
            lr_map = {lr.var: lr for lr in res_b.live_ranges}
            for v in res_b.spills:
                lr = lr_map.get(v)
                spill_data_b.append({
                    "Variable": v,
                    "Loop depth": lr.loop_depth if lr else "?",
                    "Live range": f"{lr.start}–{lr.end}" if lr else "?",
                })
            st.dataframe(
                pd.DataFrame(spill_data_b),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.success("No spills in Ranked mode!")


# ── TAB 3: INTERFERENCE GRAPH ─────────────────────────────────────
with tab_graph:
    st.markdown("### Interference Graph — Initial State")
    st.markdown(
        "<div class='info-box'>"
        "Nodes = virtual registers (variables). "
        "An edge exists between two variables if they are simultaneously live "
        "at any program point. "
        "In the Select phase, colors represent assigned physical registers; "
        "red diamonds = spills."
        "</div>",
        unsafe_allow_html=True,
    )

    ig_col_a, ig_col_b = st.columns(2)
    with ig_col_a:
        st.markdown("**Initial interference graph (both modes share this)**")
        try:
            st.graphviz_chart(res_a.interference_graph_dot, use_container_width=True)
        except Exception as e:
            st.error(f"Graphviz render error: {e}")

    with ig_col_b:
        st.markdown("**Final assignment — Classic**")
        final_dot_a = [s for s in res_a.steps if s.action in ("color", "spill")]
        if final_dot_a:
            try:
                st.graphviz_chart(final_dot_a[-1].graph_dot, use_container_width=True)
            except Exception as e:
                st.error(str(e))

    ig_col_c, ig_col_d = st.columns(2)
    with ig_col_c:
        st.markdown("**Final assignment — Ranked**")
        final_dot_b = [s for s in res_b.steps if s.action in ("color", "spill")]
        if final_dot_b:
            try:
                st.graphviz_chart(final_dot_b[-1].graph_dot, use_container_width=True)
            except Exception as e:
                st.error(str(e))

    with ig_col_d:
        st.markdown("**Register assignment table**")
        all_vars = sorted(set(list(res_a.assignment) + list(res_b.assignment) +
                              res_a.spills + res_b.spills))
        rows = []
        for v in all_vars:
            rows.append({
                "Variable": v,
                "Classic": f"R{res_a.assignment[v]}" if v in res_a.assignment else "💥 SPILL",
                "Ranked":  f"R{res_b.assignment[v]}" if v in res_b.assignment else "💥 SPILL",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ── TAB 4: STEP-BY-STEP ───────────────────────────────────────────
with tab_steps:
    st.markdown("### Step-by-Step Simulation")

    step_col_a, step_col_b = st.columns(2)

    ACTION_LABELS = {
        "simplify":       ("🔵 Simplify",      "step-simplify"),
        "potential_spill":("🟡 Potential spill","step-potential_spill"),
        "color":          ("🟢 Color",          "step-color"),
        "spill":          ("🔴 Spill",          "step-spill"),
    }

    def render_steps(result: AllocationResult, header: str):
        st.markdown(f"**{header}**")
        simplify_steps = [s for s in result.steps if s.action in ("simplify","potential_spill")]
        select_steps   = [s for s in result.steps if s.action in ("color","spill")]

        with st.expander(f"Simplify phase ({len(simplify_steps)} decisions)", expanded=True):
            for s in simplify_steps:
                label, css = ACTION_LABELS[s.action]
                score_str  = f"  score={s.score:.4f}" if s.score is not None else ""
                st.markdown(
                    f"<div class='step-card {css}'>"
                    f"<b>{label}</b>  <code>{s.node}</code>{score_str}<br>"
                    f"<span style='color:#6c7086'>{s.reason}</span><br>"
                    f"Stack: {' → '.join(s.stack_snapshot) or '(empty)'}"
                    "</div>",
                    unsafe_allow_html=True,
                )

        with st.expander(f"Select phase ({len(select_steps)} decisions)", expanded=True):
            for s in select_steps:
                label, css = ACTION_LABELS[s.action]
                st.markdown(
                    f"<div class='step-card {css}'>"
                    f"<b>{label}</b>  <code>{s.node}</code><br>"
                    f"<span style='color:#6c7086'>{s.reason}</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )

    with step_col_a:
        render_steps(res_a, "Mode A — Classic Chaitin")
    with step_col_b:
        render_steps(res_b, "Mode B — Pressure-Ranked Chaitin")

    # Step-level graph stepper
    st.markdown("---")
    st.markdown("### 🎬 Graph Stepper")
    st.markdown(
        "<div class='info-box'>"
        "Scrub through each algorithm step to see the interference graph evolve."
        "</div>",
        unsafe_allow_html=True,
    )

    stepper_mode = st.radio("Mode to step through", ["Classic", "Ranked"], horizontal=True)
    res_step = res_a if stepper_mode == "Classic" else res_b
    max_step = len(res_step.steps) - 1

    if max_step >= 0:
        step_idx = st.slider(
            "Step",
            min_value=0,
            max_value=max_step,
            value=0,
            key=f"stepper_{stepper_mode}",
        )
        chosen_step = res_step.steps[step_idx]
        label, _ = ACTION_LABELS[chosen_step.action]

        sg_col1, sg_col2 = st.columns([1.4, 1])
        with sg_col1:
            try:
                st.graphviz_chart(chosen_step.graph_dot, use_container_width=True)
            except Exception as e:
                st.error(str(e))
        with sg_col2:
            st.markdown(f"**Step {step_idx}/{max_step}**")
            st.markdown(f"{label}: **`{chosen_step.node}`**")
            st.markdown(f"_{chosen_step.reason}_")
            st.markdown("**Stack state:**")
            if chosen_step.stack_snapshot:
                for i, v in enumerate(reversed(chosen_step.stack_snapshot[-8:])):
                    st.markdown(
                        f"<div class='step-card step-simplify' style='margin-bottom:4px'>"
                        f"<code>{v}</code></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("_(empty)_")


# ── TAB 5: RAW DATA ───────────────────────────────────────────────
with tab_data:
    st.markdown("### Parsed Instructions")
    instr_rows = []
    for instr in instructions:
        instr_rows.append({
            "idx": instr.index,
            "instruction": instr.raw.strip(),
            "loop_depth": instr.loop_depth,
            "defs": ", ".join(sorted(instr.defs)),
            "uses": ", ".join(sorted(instr.uses)),
        })
    st.dataframe(pd.DataFrame(instr_rows), hide_index=True, use_container_width=True)

    st.markdown("### Live Ranges")
    lr_rows = []
    for lr in res_a.live_ranges:
        lr_rows.append({
            "variable": lr.var,
            "start": lr.start,
            "end": lr.end,
            "span": lr.end - lr.start + 1,
            "loop_depth": lr.loop_depth,
            "classic_assign": f"R{res_a.assignment[lr.var]}" if lr.var in res_a.assignment else "SPILL",
            "ranked_assign":  f"R{res_b.assignment[lr.var]}" if lr.var in res_b.assignment else "SPILL",
        })
    st.dataframe(pd.DataFrame(lr_rows), hide_index=True, use_container_width=True)

    st.markdown("### Composite Scores (Ranked mode)")
    from engine import composite_score
    import networkx as nx
    # Rebuild graph briefly for scoring display
    from engine import liveness_analysis, compute_live_ranges, build_interference_graph
    li, lo = liveness_analysis(instructions)
    lrs     = compute_live_ranges(instructions, li, lo)
    G_disp  = build_interference_graph(lrs, li, lo)

    score_rows = []
    for lr in lrs:
        sc = composite_score(lr.var, G_disp, lrs, instructions, w1, w2)
        score_rows.append({
            "variable":       lr.var,
            "def_use_dist":   lr.end - lr.start + 1,
            "loop_depth":     lr.loop_depth,
            "degree (interf)":G_disp.degree(lr.var) if G_disp.has_node(lr.var) else 0,
            "composite_score":round(sc, 5),
            "simplify_priority": "HIGH (early)" if sc > 1.0 else "LOW (late/spill candidate)",
        })
    score_rows.sort(key=lambda r: -r["composite_score"])
    st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)
