"""
engine.py — Core allocation engine for RAGC simulator.

Pipeline:
  TAC text  →  parse_tac()
            →  liveness_analysis()
            →  build_interference_graph()
            →  allocate(mode='classic' | 'ranked')
            →  AllocationResult (steps, assignment, spills, spill_cost)
"""

from __future__ import annotations
import re
import math
from dataclasses import dataclass, field
from typing import Optional
import networkx as nx


# ─────────────────────────────────────────────────────────────────
# 1.  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────

@dataclass
class Instruction:
    index: int          # program-point index (0-based)
    raw: str            # original text
    loop_depth: int     # nesting depth at this instruction
    defs: set[str]      # variables defined here
    uses: set[str]      # variables used here


@dataclass
class LiveRange:
    """Closed interval [start, end] over instruction indices."""
    var: str
    start: int
    end: int
    loop_depth: int     # max loop depth over the live range


@dataclass
class AllocationStep:
    """Snapshot of one Simplify decision."""
    step_num: int
    action: str                        # "simplify" | "potential_spill" | "color" | "spill"
    node: str                          # variable acted upon
    score: Optional[float]            # composite score (None for classic)
    reason: str                        # human-readable explanation
    graph_dot: str                     # DOT source for graphviz at this moment
    remaining_nodes: list[str]
    stack_snapshot: list[str]


@dataclass
class AllocationResult:
    mode: str                          # "classic" or "ranked"
    assignment: dict[str, int]         # var → register number (1-based)
    spills: list[str]                  # variables that were spilled
    spill_cost: int                    # weighted spill cost
    steps: list[AllocationStep]
    live_ranges: list[LiveRange]
    interference_graph_dot: str        # initial full graph DOT
    pressure_timeline: list[dict]      # [{instruction, var, alive, depth}, ...]


# ─────────────────────────────────────────────────────────────────
# 2.  TAC PARSER
# ─────────────────────────────────────────────────────────────────

_LOOP_START = re.compile(r'^\s*(for|while)\s*\(')
_LOOP_END   = re.compile(r'^\s*\}\s*$')

def parse_tac(code: str) -> list[Instruction]:
    """
    Parse simplified TAC.  Supported forms:
        x = y op z        (binary)
        x = y             (copy)
        x = op y          (unary)
        x = const         (constant load)
        LOOP_START        (for / while header — increases depth)
        LOOP_END          (bare } — decreases depth)
        // comment lines  (ignored)

    Variables: any identifier that is NOT a numeric literal.
    """
    instructions: list[Instruction] = []
    depth = 0
    idx = 0

    for raw_line in code.splitlines():
        line = raw_line.strip()

        # Skip blanks and comments
        if not line or line.startswith('//') or line.startswith('#'):
            continue

        # Loop depth tracking (these lines don't produce instructions)
        if _LOOP_START.match(line):
            depth += 1
            continue
        if _LOOP_END.match(line) and depth > 0:
            depth -= 1
            continue

        # Parse assignment  lhs = rhs
        if '=' not in line:
            continue

        lhs, _, rhs = line.partition('=')
        lhs = lhs.strip()
        rhs = rhs.strip()

        defs: set[str] = set()
        uses: set[str] = set()

        # LHS is always a def (must be identifier, not numeric)
        if _is_var(lhs):
            defs.add(lhs)

        # RHS tokens that are identifiers are uses
        tokens = re.split(r'[\s\+\-\*\/\%\(\)\,]+', rhs)
        for tok in tokens:
            tok = tok.strip()
            if _is_var(tok):
                uses.add(tok)

        if defs or uses:
            instructions.append(Instruction(
                index=idx,
                raw=raw_line.rstrip(),
                loop_depth=depth,
                defs=defs,
                uses=uses,
            ))
            idx += 1

    return instructions


def _is_var(tok: str) -> bool:
    """Return True if tok looks like a variable name (not a number, not empty)."""
    if not tok:
        return False
    if re.match(r'^-?\d+(\.\d+)?$', tok):
        return False
    if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tok):
        return True
    return False


# ─────────────────────────────────────────────────────────────────
# 3.  LIVENESS ANALYSIS  (backward data-flow, single-block)
# ─────────────────────────────────────────────────────────────────

def liveness_analysis(instructions: list[Instruction]) -> tuple[
    list[set[str]],   # live_in  per instruction
    list[set[str]],   # live_out per instruction
]:
    """
    Classic backward liveness on a straight-line program.
    For the purpose of this simulator we treat the whole program as
    one basic block.  Loop-back edges are approximated by two passes.
    """
    n = len(instructions)
    live_in  = [set() for _ in range(n)]
    live_out = [set() for _ in range(n)]

    # Two backward passes to approximate loop-back edges
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            instr = instructions[i]
            # live_in[i] = use[i] ∪ (live_out[i] − def[i])
            live_in[i]  = instr.uses | (live_out[i] - instr.defs)
            # live_out[i] = live_in[i+1]
            if i + 1 < n:
                live_out[i] = live_in[i + 1].copy()

    return live_in, live_out


# ─────────────────────────────────────────────────────────────────
# 4.  LIVE RANGES
# ─────────────────────────────────────────────────────────────────

def compute_live_ranges(
    instructions: list[Instruction],
    live_in: list[set[str]],
    live_out: list[set[str]],
) -> list[LiveRange]:
    """
    Compute [first_def_or_use … last_use] for every variable.
    Loop depth of a live range = max depth over its span.
    """
    first: dict[str, int] = {}
    last:  dict[str, int] = {}
    depth_map: dict[str, list[int]] = {}

    def touch(var: str, idx: int, depth: int):
        if var not in first:
            first[var] = idx
        last[var] = max(last.get(var, idx), idx)
        depth_map.setdefault(var, []).append(depth)

    for i, instr in enumerate(instructions):
        for v in instr.defs | instr.uses:
            touch(v, i, instr.loop_depth)
        for v in live_in[i] | live_out[i]:
            touch(v, i, instr.loop_depth)

    ranges = []
    for var in sorted(first):
        max_depth = max(depth_map[var]) if depth_map[var] else 0
        ranges.append(LiveRange(
            var=var,
            start=first[var],
            end=last[var],
            loop_depth=max_depth,
        ))
    return ranges


# ─────────────────────────────────────────────────────────────────
# 5.  INTERFERENCE GRAPH
# ─────────────────────────────────────────────────────────────────

def build_interference_graph(
    live_ranges: list[LiveRange],
    live_in: list[set[str]],
    live_out: list[set[str]],
) -> nx.Graph:
    """
    Add an edge between two variables if they are simultaneously live
    at ANY program point.
    """
    G = nx.Graph()
    for lr in live_ranges:
        G.add_node(lr.var, loop_depth=lr.loop_depth,
                   start=lr.start, end=lr.end)

    n_points = max(
        (max(len(li), len(lo)) for li, lo in zip(live_in, live_out)),
        default=0
    )

    for i in range(len(live_in)):
        alive = live_in[i] | live_out[i]
        alive_list = sorted(alive)
        for a in range(len(alive_list)):
            for b in range(a + 1, len(alive_list)):
                u, v = alive_list[a], alive_list[b]
                if G.has_node(u) and G.has_node(v):
                    G.add_edge(u, v)

    return G


# ─────────────────────────────────────────────────────────────────
# 6.  COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────

def composite_score(
    var: str,
    G: nx.Graph,
    live_ranges: list[LiveRange],
    instructions: list[Instruction],
    w1: float = 1.0,
    w2: float = 1.0,
) -> float:
    """
    Score(n) = w1
               ─────────────────────────────────────────────────────────
               def_use_distance × (nesting_depth + 1) × (local_pressure + 1) × w2

    Higher score → node is cheap to simplify → simplify it first.
    Lower score  → node is expensive (long live range, loop-heavy, high-pressure) → keep it.
    """
    lr_map = {lr.var: lr for lr in live_ranges}
    lr = lr_map.get(var)
    if lr is None:
        return 0.0

    # Def-use distance: how far apart is the first def from the last use?
    def_use_dist = max(1, lr.end - lr.start)

    # Nesting depth of the live range
    nesting = lr.loop_depth

    # Local pressure: number of variables simultaneously live with this one
    # at its definition point — i.e. its degree in the interference graph
    local_pressure = G.degree(var) if G.has_node(var) else 0

    # HIGH score = cheap to simplify = simplify first (short-lived, shallow, low-pressure)
    # LOW score  = expensive = keep alive longer (long-lived, deep loop, high-pressure)
    # def_use_dist in DENOMINATOR: longer live range → lower score → survives longer
    numerator   = w1
    denominator = def_use_dist * (nesting + 1) * (local_pressure + 1) * w2

    return numerator / denominator


# ─────────────────────────────────────────────────────────────────
# 7.  GRAPH → DOT
# ─────────────────────────────────────────────────────────────────

_COLORS = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990",
    "#dcbeff","#9A6324","#fffac8","#800000","#aaffc3",
]

def graph_to_dot(
    G: nx.Graph,
    assignment: dict[str, int] = None,
    spills: set[str] = None,
    highlight: str = None,
    k: int = 8,
) -> str:
    """Render interference graph as DOT source."""
    assignment = assignment or {}
    spills = spills or set()
    lines = [
        'graph G {',
        '  graph [bgcolor="#1e1e2e" pad="0.4" nodesep="0.6" ranksep="0.8"];',
        '  node  [style=filled fontname="Helvetica" fontsize=11 penwidth=1.5];',
        '  edge  [color="#555577" penwidth=1.2];',
    ]
    for node in G.nodes():
        if node in spills:
            color = '#ff4444'
            fcolor = 'white'
            shape = 'diamond'
        elif node in assignment:
            reg = assignment[node]
            color = _COLORS[(reg - 1) % len(_COLORS)]
            fcolor = 'black'
            shape = 'ellipse'
        elif node == highlight:
            color = '#ffdd00'
            fcolor = 'black'
            shape = 'ellipse'
        else:
            color = '#44475a'
            fcolor = 'white'
            shape = 'ellipse'

        label = node
        if node in assignment:
            label = f"{node}\\nR{assignment[node]}"
        elif node in spills:
            label = f"{node}\\nSPILL"

        lines.append(
            f'  "{node}" [label="{label}" fillcolor="{color}" '
            f'fontcolor="{fcolor}" shape={shape}];'
        )
    for u, v in G.edges():
        lines.append(f'  "{u}" -- "{v}";')
    lines.append('}')
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────
# 8.  ALLOCATOR  (Classic + Ranked — shared Select phase)
# ─────────────────────────────────────────────────────────────────

def allocate(
    instructions: list[Instruction],
    k: int,
    mode: str = 'classic',
    w1: float = 1.0,
    w2: float = 1.0,
) -> AllocationResult:
    """
    Run Chaitin-style graph-coloring register allocation.

    mode='classic' — arbitrary (FIFO) simplification order
    mode='ranked'  — pressure-ranked: highest score simplified first
                     (lowest cost to spill → leaves high-cost nodes alive)
    """
    live_in, live_out = liveness_analysis(instructions)
    live_ranges       = compute_live_ranges(instructions, live_in, live_out)
    G_orig            = build_interference_graph(live_ranges, live_in, live_out)

    # Pressure timeline for heat map
    pressure_timeline = _build_pressure_timeline(instructions, live_in, live_out)

    initial_dot = graph_to_dot(G_orig, k=k)

    # Work on a copy
    G = G_orig.copy()

    stack: list[str]           = []
    steps: list[AllocationStep] = []
    step_num = 0

    # ── SIMPLIFY PHASE ──────────────────────────────────────────
    potential_spills: list[str] = []

    while G.number_of_nodes() > 0:
        # Find nodes with degree < k
        low_degree = [n for n in G.nodes() if G.degree(n) < k]

        if low_degree:
            if mode == 'ranked':
                # Sort by score DESCENDING — remove cheapest first
                scored = [
                    (n, composite_score(n, G, live_ranges, instructions, w1, w2))
                    for n in low_degree
                ]
                scored.sort(key=lambda x: -x[1])
                chosen, score = scored[0]
                reason = (
                    f"Ranked: score={score:.3f}  "
                    f"(def-use dist={_du_dist(chosen, live_ranges)}, "
                    f"depth={_depth(chosen, live_ranges)}, "
                    f"degree={G.degree(chosen)})"
                )
            else:
                chosen = low_degree[0]   # FIFO — arbitrary
                score  = None
                reason = f"Classic: first node with degree < {k}"

            stack.append(chosen)
            G_snap = G.copy()
            G.remove_node(chosen)

            steps.append(AllocationStep(
                step_num=step_num,
                action="simplify",
                node=chosen,
                score=score,
                reason=reason,
                graph_dot=graph_to_dot(G_snap, highlight=chosen, k=k),
                remaining_nodes=list(G.nodes()),
                stack_snapshot=stack.copy(),
            ))
            step_num += 1

        else:
            # All remaining nodes have degree ≥ k → potential spill
            if G.number_of_nodes() == 0:
                break

            if mode == 'ranked':
                scored = [
                    (n, composite_score(n, G, live_ranges, instructions, w1, w2))
                    for n in G.nodes()
                ]
                scored.sort(key=lambda x: -x[1])
                chosen, score = scored[0]   # highest score = cheapest to spill
                reason = (
                    f"Ranked spill candidate: score={score:.3f}  "
                    f"(cheapest by composite metric)"
                )
            else:
                chosen = next(iter(G.nodes()))
                score  = None
                reason = "Classic: arbitrary potential spill"

            potential_spills.append(chosen)
            stack.append(chosen)
            G_snap = G.copy()
            G.remove_node(chosen)

            steps.append(AllocationStep(
                step_num=step_num,
                action="potential_spill",
                node=chosen,
                score=score,
                reason=reason,
                graph_dot=graph_to_dot(G_snap, highlight=chosen, k=k),
                remaining_nodes=list(G.nodes()),
                stack_snapshot=stack.copy(),
            ))
            step_num += 1

    # ── SELECT PHASE ────────────────────────────────────────────
    assignment: dict[str, int] = {}
    actual_spills: list[str]   = []
    G_rebuild = nx.Graph()

    while stack:
        node = stack.pop()
        G_rebuild.add_node(node)

        # Re-add edges to already-colored neighbours
        for nb in G_orig.neighbors(node):
            if G_rebuild.has_node(nb):
                G_rebuild.add_edge(node, nb)

        used_colors = {
            assignment[nb]
            for nb in G_orig.neighbors(node)
            if nb in assignment
        }
        # Pick lowest unused register
        color = None
        for c in range(1, k + 1):
            if c not in used_colors:
                color = c
                break

        if color is not None:
            assignment[node] = color
            action = "color"
            reason = f"Assigned R{color}  (neighbours used: {sorted(used_colors)})"
        else:
            actual_spills.append(node)
            action = "spill"
            reason = f"No free register among {k}  (neighbours: {sorted(used_colors)})"

        steps.append(AllocationStep(
            step_num=step_num,
            action=action,
            node=node,
            score=None,
            reason=reason,
            graph_dot=graph_to_dot(
                G_rebuild,
                assignment=assignment,
                spills=set(actual_spills),
                k=k,
            ),
            remaining_nodes=list(G_rebuild.nodes()),
            stack_snapshot=stack.copy(),
        ))
        step_num += 1

    # Spill cost calculation
    spill_cost = _compute_spill_cost(actual_spills, live_ranges, instructions)

    return AllocationResult(
        mode=mode,
        assignment=assignment,
        spills=actual_spills,
        spill_cost=spill_cost,
        steps=steps,
        live_ranges=live_ranges,
        interference_graph_dot=initial_dot,
        pressure_timeline=pressure_timeline,
    )


# ─────────────────────────────────────────────────────────────────
# 9.  HELPERS
# ─────────────────────────────────────────────────────────────────

def _du_dist(var: str, live_ranges: list[LiveRange]) -> int:
    for lr in live_ranges:
        if lr.var == var:
            return max(1, lr.end - lr.start)
    return 1


def _depth(var: str, live_ranges: list[LiveRange]) -> int:
    for lr in live_ranges:
        if lr.var == var:
            return lr.loop_depth
    return 0


def _compute_spill_cost(
    spills: list[str],
    live_ranges: list[LiveRange],
    instructions: list[Instruction],
) -> int:
    """
    Spill inside a loop = 10 units per instruction in the live range.
    Spill in linear code = 1 unit per instruction.
    """
    lr_map = {lr.var: lr for lr in live_ranges}
    total = 0
    for var in spills:
        lr = lr_map.get(var)
        if lr is None:
            total += 1
            continue
        for instr in instructions:
            if lr.start <= instr.index <= lr.end:
                total += 10 if instr.loop_depth > 0 else 1
    return total


def _build_pressure_timeline(
    instructions: list[Instruction],
    live_in: list[set[str]],
    live_out: list[set[str]],
) -> list[dict]:
    """
    For each (instruction_index, variable) pair where the variable is alive,
    emit a record.  Used to build the Plotly heat map.
    """
    timeline = []
    for i, instr in enumerate(instructions):
        alive = live_in[i] | live_out[i] | instr.defs | instr.uses
        pressure = len(alive)
        for var in alive:
            timeline.append({
                "instruction": i,
                "label": f"#{i}: {instr.raw.strip()[:35]}",
                "var": var,
                "alive": 1,
                "pressure": pressure,
                "loop_depth": instr.loop_depth,
            })
    return timeline


# ─────────────────────────────────────────────────────────────────
# 10.  BUILT-IN EXAMPLE PROGRAMS
# ─────────────────────────────────────────────────────────────────

EXAMPLE_PROGRAMS = {
    "Simple": """\
a = 1
b = 2
c = a + b
d = c + a
e = d + b
f = e + c
""",

    "Nested loop (pressure)": """\
i = 0
for (i < 10)
  j = 0
  for (j < 10)
    t = i + j
    sum = sum + t
    j = j + 1
  i = i + 1
""",

    "High spill pressure": """\
i = 0
for (i < 4)
  j = i + 1
  for (j < 4)
    t1 = i + j
    t2 = t1 + 1
    t3 = t2 + i
    j = j + 1
  }
  i = i + 1
}
""",

    "Loop Protection": """\
scale_factor = 2
global_acc = 0
for (i < 40)
  p1 = i + 1
  p2 = p1 * scale_factor
  p3 = p2 + global_acc
  p4 = p3 / 2
  p5 = p4 - 1
  global_acc = p5
  i = i + 1
}
final_result = global_acc * scale_factor
""",

    "Nesting Depth": """\
outer_val = 100
inner_sum = 0
for (i < 10)
  mid_val = i * outer_val
  for (j < 10)
    inner_v1 = i + j
    inner_v2 = inner_v1 * mid_val
    inner_v3 = inner_v2 + outer_val
    inner_sum = inner_sum + inner_v3
    j = j + 1
  }
  i = i + 1
}
output = inner_sum + outer_val
"""
}
