# RAGC — Pressure-Ranked Register Allocation Simulator

**CSL 304 Compilers · IIIT Nagpur · 2026**

A full-stack simulation tool comparing **Classic Chaitin** register allocation
against a novel **Pressure-Ranked Chaitin** variant — implemented as the
course research project.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Linux/Mac) also install graphviz binary
sudo apt install graphviz        # Ubuntu/Debian
brew install graphviz             # macOS

# 3. Run tests first (verify correctness)
python test_engine.py

# 4. Launch the simulator
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**

---

## The Novel Contribution

### Problem with Classic Chaitin

In Chaitin's Simplify phase, nodes with degree < k are removed in
**arbitrary (FIFO) order**. This is correct — any node with degree < k
can always be colored in the Select phase — but the _order_ determines
**which nodes remain when k-colorability fails**.

If a loop variable (expensive to spill) happens to be removed early by
chance, it survives to get a color. But if it's left until the graph is
uncolorable, it becomes the spill candidate. Classic Chaitin gives no
guarantee either way.

### Our Fix: Pressure-Ranked Simplification Order

We assign every node a **Composite Score** before each simplification step:

```
Score(n) =                      w₁
           ──────────────────────────────────────────────────────────
           def_use_distance × (nesting_depth + 1) × (local_pressure + 1) × w₂
```

| Factor             | Meaning                                            | Effect on score                                                                 |
| ------------------ | -------------------------------------------------- | ------------------------------------------------------------------------------- |
| `def_use_distance` | Span from first def to last use                    | **Longer = lower score** = kept alive (long-lived vars are costly to spill)     |
| `nesting_depth`    | Max loop depth of the variable's live range        | **Deeper loop = lower score** = kept alive (loop spills cost 10× more)          |
| `local_pressure`   | Degree in the interference graph (live neighbours) | **Higher degree = lower score** = kept alive (many conflicts = hard to recolor) |
| `w₁`, `w₂`         | Tunable weights                                    | Adjustable in sidebar                                                           |

**Nodes with HIGH scores are simplified first** — they are short-lived, linear, low-pressure (cheap to spill).
**Nodes with LOW scores survive longer** — they are long-lived, loop-heavy, or highly connected (expensive to spill).

This is inspired by:

- **PresCount** (SJTU 2024): bank pressure tracking during RCG coloring
- **RL4ReAl** (IIT-H, CC'23): using spill cost as the reward signal

Our contribution: applying a **deterministic composite pressure metric**
to the simplification _order_ — not to the spill _selection_ after failure —
which is an underexplored aspect of Chaitin's algorithm.

---

## Benchmark Results (run test_engine.py to reproduce)

| Program             | k   | Classic spills | Ranked spills | Cost reduction      |
| ------------------- | --- | -------------- | ------------- | ------------------- |
| Simple (no loops)   | 3   | 0              | 0             | —                   |
| Nested loop         | 3   | varies         | ≤ classic     | loop vars protected |
| Live range conflict | 3   | varies         | ≤ classic     | —                   |
| Mixed               | 3   | varies         | ≤ classic     | loop spills avoided |
| High spill pressure | 3   | varies         | ≤ classic     | —                   |

---

## TAC Syntax Reference

```
// This is a comment
a = 1              // constant assignment
b = a + 2          // binary op: +, -, *, /
c = a              // copy
for (x < 10){       // loop start — increases depth by 1
  body...
}                  // loop end — decreases depth
```

Variables: any identifier (`[a-zA-Z_][a-zA-Z0-9_]*`)
Numbers: treated as constants, not variables
