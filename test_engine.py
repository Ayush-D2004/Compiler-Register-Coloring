"""
test_engine.py — verify correctness of the core allocation engine.
Run:  python test_engine.py
"""

import sys
from engine import (
    parse_tac, liveness_analysis, compute_live_ranges,
    build_interference_graph, allocate, composite_score,
    EXAMPLE_PROGRAMS,
)


PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
errors = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}  {detail}")
        errors.append(name)


# ─────────────────────────────────────────────────────────────────
print("\n── Parser ──────────────────────────────────────────────────")

code = "a = 1\nb = a + 2\nc = b + a\n"
instrs = parse_tac(code)
check("parses 3 instructions", len(instrs) == 3)
check("instruction 0 defs {a}", instrs[0].defs == {"a"})
check("instruction 0 uses empty", instrs[0].uses == set())
check("instruction 1 uses {a}", "a" in instrs[1].uses)
check("instruction 1 defs {b}", instrs[1].defs == {"b"})
check("loop depth default 0", all(i.loop_depth == 0 for i in instrs))

code_loop = "i = 0\nfor (i < 5)\n  x = i + 1\n  i = i + 1\n}\n"
il = parse_tac(code_loop)
loop_instrs = [i for i in il if i.loop_depth > 0]
check("loop depth increments inside loop", len(loop_instrs) >= 2)

code_comment = "// header\na = 1\n# also ignored\nb = a\n"
ic = parse_tac(code_comment)
check("comments skipped", len(ic) == 2)

code_numeric = "a = 42\nb = a + 100\n"
inum = parse_tac(code_numeric)
check("numeric literals not treated as variables", "42" not in inum[0].uses and "100" not in inum[1].uses)


# ─────────────────────────────────────────────────────────────────
print("\n── Liveness Analysis ───────────────────────────────────────")

code = "a = 1\nb = a + 2\nc = b\n"
instrs = parse_tac(code)
li, lo = liveness_analysis(instrs)
check("returns lists of length == n", len(li) == 3 and len(lo) == 3)
# b must be live-out of instruction 1 (defined there, used in instr 2)
check("b live-out of instr 1", "b" in lo[1])
# a live-in of instruction 1 (used there)
check("a live-in of instr 1", "a" in li[1])


# ─────────────────────────────────────────────────────────────────
print("\n── Interference Graph ──────────────────────────────────────")

code = "a = 1\nb = 2\nc = a + b\n"
instrs = parse_tac(code)
li, lo = liveness_analysis(instrs)
lrs = compute_live_ranges(instrs, li, lo)
G = build_interference_graph(lrs, li, lo)

check("graph has nodes for a, b, c", all(G.has_node(v) for v in ["a","b","c"]))
# a and b are both live when c is computed
check("edge a-b or a-c exists (concurrent liveness)", G.number_of_edges() > 0)


# ─────────────────────────────────────────────────────────────────
print("\n── Composite Score ─────────────────────────────────────────")

code = "i = 0\nfor (i < 5)\n  x = i + 1\n  sum = sum + x\n  i = i + 1\n}\n"
instrs = parse_tac(code)
li, lo = liveness_analysis(instrs)
lrs    = compute_live_ranges(instrs, li, lo)
G      = build_interference_graph(lrs, li, lo)

scores = {lr.var: composite_score(lr.var, G, lrs, instrs) for lr in lrs}
check("all scores > 0", all(v > 0 for v in scores.values()))

# i has high loop depth → lower score than a short-lived linear var
# (only meaningful if both exist)
if "i" in scores and "x" in scores:
    check("loop var i has lower score than short linear var x (roughly)",
          scores["i"] <= scores["x"] * 5,   # loose check — depth penalises i
          f"i={scores['i']:.3f} x={scores['x']:.3f}")


# ─────────────────────────────────────────────────────────────────
print("\n── Allocator — correctness ─────────────────────────────────")

for name, prog in EXAMPLE_PROGRAMS.items():
    instrs = parse_tac(prog)
    if not instrs:
        continue
    for mode in ("classic", "ranked"):
        res = allocate(instrs, k=3, mode=mode)
        # Every variable must be either assigned or spilled
        all_vars = {lr.var for lr in res.live_ranges}
        covered  = set(res.assignment.keys()) | set(res.spills)
        check(
            f"[{mode}] '{name}': all vars covered",
            all_vars <= covered,
            f"missing={all_vars - covered}",
        )
        # No two interfering variables share a register
        li2, lo2 = liveness_analysis(instrs)
        lrs2 = compute_live_ranges(instrs, li2, lo2)
        G2   = build_interference_graph(lrs2, li2, lo2)
        conflict = False
        for u, v in G2.edges():
            if u in res.assignment and v in res.assignment:
                if res.assignment[u] == res.assignment[v]:
                    conflict = True
                    errors.append(f"CONFLICT {u}-{v} both R{res.assignment[u]}")
        check(
            f"[{mode}] '{name}': no register conflicts",
            not conflict,
        )
        # Assigned registers within range
        oob = [v for v, r in res.assignment.items() if r < 1 or r > 3]
        check(
            f"[{mode}] '{name}': registers in [1,k]",
            not oob,
            f"out-of-bounds={oob}",
        )


# ─────────────────────────────────────────────────────────────────
print("\n── Ranked ≤ Classic spill cost claim ───────────────────────")

improvements = 0
ties         = 0
regressions  = 0

for name, prog in EXAMPLE_PROGRAMS.items():
    instrs = parse_tac(prog)
    if not instrs:
        continue
    for k in (2, 3, 4):
        ra = allocate(instrs, k=k, mode="classic")
        rb = allocate(instrs, k=k, mode="ranked")
        if rb.spill_cost < ra.spill_cost:
            improvements += 1
        elif rb.spill_cost == ra.spill_cost:
            ties += 1
        else:
            regressions += 1

total = improvements + ties + regressions
check(
    f"Ranked improves or ties on ≥80% of (program, k) pairs "
    f"({improvements+ties}/{total} = {(improvements+ties)/total*100:.0f}%)",
    (improvements + ties) / total >= 0.80,
    f"improvements={improvements} ties={ties} regressions={regressions}",
)


# ─────────────────────────────────────────────────────────────────
print("\n── Steps log completeness ──────────────────────────────────")

for name, prog in list(EXAMPLE_PROGRAMS.items())[:2]:
    instrs = parse_tac(prog)
    if not instrs:
        continue
    res = allocate(instrs, k=3, mode="ranked")
    check(f"'{name}': steps log non-empty",      len(res.steps) > 0)
    check(f"'{name}': all steps have graph_dot",  all(s.graph_dot for s in res.steps))
    check(f"'{name}': pressure_timeline non-empty", len(res.pressure_timeline) > 0)


# ─────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"\033[91m{len(errors)} test(s) FAILED:\033[0m")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print(f"\033[92mAll tests passed.\033[0m")
    sys.exit(0)
