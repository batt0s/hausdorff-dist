# hausdorff-dist

Note: This README.md was written by Claude AI, checked and confirmed by [me](mailto:me@battos.dev).

# Hausdorff Distance Laboratory

An interactive Streamlit application for computing and visualizing the Hausdorff distance between two user-defined compact sets in the plane. The user provides sets as mathematical equations or inequalities, chooses a distance metric, and the application renders both sets along with the critical witness vectors that realize the Hausdorff distance.

---

## Table of Contents

1. [Mathematical Background](#mathematical-background)
2. [Project Structure](#project-structure)
3. [Module: `hausdorff.py`](#module-hausdorffpy)
   - [`get_shape_points`](#get_shape_points)
   - [`get_hausdorff_details`](#get_hausdorff_details)
4. [Module: `web.py`](#module-webpy)
5. [Supported Metrics](#supported-metrics)
6. [Equation Syntax Guide](#equation-syntax-guide)
7. [Known Limitations](#known-limitations)
8. [Dependencies](#dependencies)

---

## Mathematical Background

### Hausdorff Distance

Given two non-empty compact sets A and B in a metric space (M, d), the **directed Hausdorff distance** from A to B is:

```
h(A, B) = max  { min  { d(a, b) } }
          a∈A    b∈B
```

In plain language: for every point in A, find its closest neighbor in B. Among all those minimum distances, take the largest one. This is the "worst-case nearest-neighbor" distance from A to B.

Because h(A, B) ≠ h(B, A) in general, the **bidirectional Hausdorff distance** is defined as:

```
H(A, B) = max( h(A, B), h(B, A) )
```

This is a true metric on the space of compact sets (it satisfies identity, symmetry, and the triangle inequality). It measures how far the two sets are from being equal — if H(A, B) = 0 then A = B.

The point in A that achieves the maximum in h(A, B) is called the **witness point**, and together with its nearest neighbor in B, they form the **witness pair** — the most "isolated" point and its best match. This application draws these pairs as directed vectors on the plot.

---

## Project Structure

```
.
├── hausdorff.py   # Core math: shape sampling and Hausdorff computation
└── web.py         # Streamlit UI and Plotly visualization
```

---

## Module: `hausdorff.py`

This module contains the two core algorithmic functions. It has no UI logic and can be imported and used independently.

---

### `get_shape_points`

```python
def get_shape_points(equation_str, resolution=200, grid_bounds=(-6, 6)) -> np.ndarray
```

**Purpose:** Converts a mathematical string expression (equation or inequality) into a set of 2D points that lie on or inside the described shape. This is the most complex function in the codebase and is described in full detail below.

**Returns:** An `(N, 2)` numpy array of `[x, y]` coordinates satisfying the condition.

#### High-Level Idea

The function works by **discretizing the plane into a grid** and then **evaluating the mathematical condition at every grid point**. Points where the condition is true are kept; the rest are discarded. This is called a **boolean mask** approach.

```
PSEUDOCODE: get_shape_points(equation_str, resolution, grid_bounds)

  1. BUILD THE GRID
     vals     ← linspace(grid_bounds.min, grid_bounds.max, resolution)
     X, Y     ← meshgrid(vals, vals)
     // X and Y are now (resolution × resolution) matrices.
     // Each cell (i, j) represents the point (X[i,j], Y[i,j]) in the plane.

  2. COMPUTE TOLERANCE
     tol ← (grid_bounds.max - grid_bounds.min) / resolution * 1.5
     // Because the grid is discrete, exact equations like "y = x" will never
     // be satisfied with floating-point equality. tol defines a small band
     // around the curve within which we accept a point as "on" the curve.
     // The factor 1.5 adds a slight margin to avoid gaps in the output.

  3. PREPROCESS THE STRING
     Replace " and " with " & "
     Replace " or "  with " | "
     // Normalize logical operators to symbols for splitting.

  4. TOKENIZE
     Split equation_str on "&" and "|" separators, keeping the separators.
     // Example: "x**2 + y**2 <= 4 & y >= 0"
     //   → tokens: ["x**2 + y**2 <= 4", "&", "y >= 0"]

  5. PROCESS EACH TOKEN
     For each token t:
       IF t is "&" or "|":
         keep as-is (it's a logical connector)

       ELSE IF t contains "<=", ">=", "<", or ">":
         wrap as: "(t)"
         // It's already a proper inequality; numpy handles these element-wise.
         // e.g. "x**2 + y**2 <= 4" → evaluates to a boolean matrix

       ELSE IF t contains "==" or "=":
         split into lhs and rhs
         rewrite as: "(abs(lhs - rhs) < tol)"
         // Exact equations become proximity checks.
         // "y = x" → "abs(y - x) < tol" → selects a thin band around the line.

       ELSE:
         rewrite as: "(abs(t) < tol)"
         // Bare expressions (e.g. "x**2 + y**2 - 4") are treated as
         // implicit zero-level sets: abs(expr) < tol.

  6. REASSEMBLE
     Join processed tokens back into a single expression string.

  7. EVALUATE
     Define a safe evaluation environment:
       env ← { x: X, y: Y, np: numpy, sin, cos, sqrt, abs, tol }
     mask ← eval(final_expression, env)
     // The expression is evaluated with X and Y as full matrices,
     // so every grid point is tested simultaneously. The result is a
     // (resolution × resolution) boolean matrix.

  8. EXTRACT POINTS
     points ← column_stack( X[mask], Y[mask] )
     // Collect the (x, y) coordinates of all True cells.
     return points
```

#### Why the Tolerance Works

Consider the equation `y = sin(x)` on a 200×200 grid over [-6, 6]. The grid spacing is 12/200 = 0.06 units. The sine curve passes through each grid cell but almost certainly does not pass through any grid *point* exactly. The tolerance `tol = 0.06 * 1.5 = 0.09` creates a band of ±0.09 around the curve, which is wide enough to capture grid points near the curve without capturing points too far away.

For inequalities like `x**2 + y**2 <= 4`, no tolerance is needed — every grid point is either clearly inside or outside the disk, and numpy evaluates this correctly.

#### Limitations of this Approach

- **Resolution trades off with accuracy:** A coarse grid produces jagged shapes; a fine grid is slow.
- **Very thin features may disappear** if the tolerance is smaller than the grid spacing. This is unlikely with the default settings but possible at low resolution.
- **No chained comparisons:** Expressions like `-5 < x < 5` are not supported. Write them as `-5 < x & x < 5`.
- **No symbolic math:** The expression is evaluated numerically. Symbolic simplification does not occur.

---

### `get_hausdorff_details`

```python
def get_hausdorff_details(A, B, metric_type="euclidean") -> tuple
```

**Purpose:** Given two point clouds A and B, computes the full bidirectional Hausdorff distance and returns the witness pairs for both directed distances. This function is metric-agnostic — any distance metric supported by `scipy.spatial.distance.cdist` can be passed in.

**Returns:** A tuple of 7 elements:
- `h` — the final Hausdorff distance `H(A, B)`
- `d_AB` — the directed distance `h(A, B)`
- `pt_A` — the witness point in A (farthest from B)
- `pt_B_near_A` — the nearest point in B to `pt_A`
- `d_BA` — the directed distance `h(B, A)`
- `pt_B` — the witness point in B (farthest from A)
- `pt_A_near_B` — the nearest point in A to `pt_B`

```
PSEUDOCODE: get_hausdorff_details(A, B, metric_type)

  --- DIRECTED DISTANCE: A → B ---

  1. dist_matrix_AB ← cdist(A, B, metric=metric_type)
     // Shape: (|A|, |B|). Entry [i, j] is the distance from A[i] to B[j].

  2. min_dists_A ← row-wise minimum of dist_matrix_AB
     // min_dists_A[i] = distance from A[i] to its nearest neighbor in B.
     // Shape: (|A|,)

  3. idx_max_A ← argmax(min_dists_A)
     // The index of the point in A that is FARTHEST from B.

  4. d_AB     ← min_dists_A[idx_max_A]     // The directed Hausdorff distance A→B
     pt_A     ← A[idx_max_A]               // The witness point in A
     pt_B_near_A ← B[ argmin( cdist([pt_A], B) ) ]   // Its nearest neighbor in B

  --- DIRECTED DISTANCE: B → A ---

  5. (Symmetric computation with A and B swapped)
     min_dists_B ← row-wise minimum of cdist(B, A)
     idx_max_B   ← argmax(min_dists_B)
     d_BA        ← min_dists_B[idx_max_B]
     pt_B        ← B[idx_max_B]
     pt_A_near_B ← A[ argmin( cdist([pt_B], A) ) ]

  --- COMBINE ---

  6. h ← max(d_AB, d_BA)

  7. return h, d_AB, pt_A, pt_B_near_A, d_BA, pt_B, pt_A_near_B
```

**Why `cdist` and not a loop?** `cdist` computes the full pairwise distance matrix in optimized C code. For two sets with |A| = |B| = 40,000 points (200×200 grid), a Python loop would require 1.6 billion distance evaluations. `cdist` handles this in vectorized form, making it feasible in seconds.

---

## Module: `web.py`

This module is the Streamlit application layer. It has no mathematical logic of its own — it handles user input, calls `hausdorff.py`, and renders the results with Plotly.

```
PSEUDOCODE: web.py (top-level execution flow)

  1. RENDER SIDEBAR
     - Text input  → eq_a (equation for Set A)
     - Text input  → eq_b (equation for Set B)
     - Selectbox   → metric_type (euclidean / cityblock / chebyshev)
     - Slider      → resolution (grid density)
     - Number inputs → grid_min, grid_max (bounding box)

  2. ON BUTTON CLICK ("Analiz Et ve Çiz"):

     a. A ← get_shape_points(eq_a, resolution, (grid_min, grid_max))
        B ← get_shape_points(eq_b, resolution, (grid_min, grid_max))

     b. IF A or B is empty:
          show error and stop

     c. h, d_AB, pA_max, pB_near, d_BA, pB_max, pA_near
          ← get_hausdorff_details(A, B, metric_type)

     d. BUILD PLOTLY FIGURE
          - Scatter trace for A (blue dots)
          - Scatter trace for B (red dots)
          - Line trace for d(A→B) witness vector:
              from pA_max to pB_near
              gold color if this is the dominant direction, green otherwise
          - Line trace for d(B→A) witness vector:
              from pB_max to pA_near
              gold color if this is the dominant direction, green otherwise

     e. DISPLAY
          - Plotly chart (equal aspect ratio enforced)
          - Success banner showing H(A, B)
```

### Visualization Design

The two directed distances are drawn as line segments connecting their respective witness pairs. The **gold** line is the dominant direction (the one that equals H(A, B)); the **green dashed** line is the smaller directed distance. This makes it visually immediate which direction "wins" the Hausdorff maximum.

### Known Bug in Witness Vector Drawing (Fixed)

The original code mixed up the y-coordinates between the two witness segments:

```python
# WRONG — x from pair 1, y from pair 2
x=[pA_max[0], pB_near[0]], y=[pB_max[1], pA_near[1]]

# CORRECT — both coordinates from the same pair
x=[pA_max[0], pB_near[0]], y=[pA_max[1], pB_near[1]]
```

---

## Supported Metrics

| Name | Formula | Geometry of unit ball |
|---|---|---|
| Euclidean (L2) | $sqrt(\sum (x_i - y_i)^2)$ | Circle |
| Manhattan / Cityblock (L1) | $\sum |x_i - y_i|$ | Diamond (rotated square) |
| Chebyshev (L∞) | $\max (|x_i - y_i|)$ | Axis-aligned square |

Minkowski is intentionally excluded. Its only non-redundant parameter values are p=1 (Manhattan) and p=2 (Euclidean), both of which are already available. Other values of p have no standard geometric interpretation for this use case, and the missing `p` parameter in the original `cdist` call would cause it to silently default to p=2 (Euclidean) anyway.

---

## Equation Syntax Guide

Expressions are evaluated as Python/NumPy code with `x` and `y` available as variables.

| Goal | Example |
|---|---|
| Circle (boundary) | `x**2 + y**2 = 4` |
| Disk (filled) | `x**2 + y**2 <= 4` |
| Line segment | `y = 0 & x >= -5 & x <= 5` |
| Diamond | `abs(x) + abs(y) = 1` |
| Parabola arc | `y = x**2 & x >= -2 & x <= 2` |
| Half-plane | `y >= x` |
| Rectangle | `abs(x) <= 3 & abs(y) <= 2` |
| Sine curve | `y = sin(x)` |

**Available functions:** `sin`, `cos`, `sqrt`, `abs`

**Logical operators:** Use `&` (AND) or `|` (OR) to combine conditions. Do **not** use chained comparisons like `-5 < x < 5`; write them as `-5 < x & x < 5` instead.

---

## Known Limitations

- **Chained comparisons unsupported.** `-5 < x < 5` must be written as `-5 < x & x < 5`.
- **High resolution is slow.** At resolution=500, each set has 250,000 points and the pairwise distance matrix has 62.5 billion entries — the cdist call will be very slow or run out of memory. Keep resolution under 300 for practical use.
- **Metric-aware path drawing is not implemented.** The witness vectors are always drawn as straight Euclidean line segments, even when the chosen metric is Manhattan or Chebyshev. Under Manhattan distance, the true minimum-cost path is an L-shaped polyline; under Chebyshev, it is a diagonal-then-straight path. The distance value is computed correctly; only the drawn path is metric-unaware.
- **3D is not supported.** The grid and visualization are strictly 2D.

---

## Dependencies

```
numpy
sympy
scipy
scikit-image
streamlit
plotly
```

Install with:

```bash
pip install numpy sympy scipy scikit-image streamlit plotly
```

Run the app with:

```bash
streamlit run web.py
```
