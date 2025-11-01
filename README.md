# Adaptive-Multi-Resolution-Projection-Graphs (AMRPG)

The goal with this algorithm is to establish fast and the best approximate nearest neighbor search for streaming, high dimensional, and distribution shifting data. Works well when memory/time budget are tight and the data distribution changes gradually. It combines John-Lindenstrauss style projections, multi-resolution indices, online density estimation, and local graph refinement.

---

##  Problem Statement

Given a stream of points **xₜ ∈ ℝᵈ**, maintain a data structure supporting:

- **Insert(xₜ):** add a point under a memory budget **M**  
- **Query(q, k):** return *k* approximate nearest neighbors of **q**

### Guarantees
With probability *(1 − δ)*, returned neighbors have distances within a factor of *(1 + ε)* of the true *k*-nearest neighbors for dense regions.  
The structure adapts online when the data distribution drifts.

---

## Core Ideas

1. **Multi-Resolution Random Projections**  
   Several projection levels (R₀, R₁, …, Rₕ) give coarse-to-fine geometric views of the data.

2. **Sparse Proximity Graphs**  
   Each level keeps its own sparse graph Gᵢ that links locally close nodes.

3. **Adaptive Density Control**  
   Lightweight online density estimation decides where to refine or skip high-resolution detail.

4. **Confidence-Guided Search**  
   Queries escalate from coarse projections to finer ones only when local uncertainty is high.

5. **Time-Decay and Drift Handling**  
   Nodes are aged out as distributions shift, keeping the graph current and balanced.

---

## Data Structures

| Symbol | Description |
|:--|:--|
| Rᵢ | Projection matrix at resolution *i* |
| Gᵢ | Sparse proximity graph at level *i* |
| CellIndexᵢ | Hash map of coarse spatial cells → node IDs |
| Reservoir[node] | Small recent sample of timestamps for density estimation |
| Node | Holds ID, timestamps, projections, and neighbor links |

---

## Algorithm Overview

### Insert(x)

1. Compute projections **pᵢ(x) = Rᵢ × x** for all levels *i*.  
2. Add node to **CellIndex₀** by quantizing **p₀(x)**.  
3. Link the node to nearby points in **G₀**.  
4. For each finer level *i*, if local density ≥ threshold τᵢ, connect in **Gᵢ**.  
5. If total nodes exceed **M**, evict the oldest or least-dense ones.

### Query(q, k)

1. Compute projections **pᵢ(q)**.  
2. Get candidate seeds from coarse cells in **CellIndex₀**.  
3. Expand candidates through **G₀**, guided by confidence scores.  
4. Escalate through finer projections until precision is sufficient.  
5. Return the top-k nearest neighbors.

---

## Complexity

| Operation | Expected Complexity |
|:--|:--|
| Insert | O((Σ mᵢ) · d) |
| Query | O((Σ mᵢ) + α · neighbors inspected) |
| Memory | O(n · Σ mᵢ), limited by budget M |

Typical parameters:
- Projection dims: *mᵢ ≈ (log n / εᵢ²)*  
- Graph degree caps *dᵢ ∈ [8 , 32]*  
- 2–4 projection levels recommended

---

## Theory in Brief

- Random projections approximately preserve pairwise distances (Johnson–Lindenstrauss property).  
- For dense clusters, the hierarchy maintains neighbor fidelity within distortion ≤ (1 + ε).  
- Time-decay ensures adaptation under streaming or shifting data.

---

## Minimal Python Prototype

```python
from amrpg import AMRPG

amrpg = AMRPG(
    d=128,
    levels_dims=[16, 64, 128],
    s=5,
    budget=10000
)

# Insert streaming points
for x in stream:
    amrpg.insert(x)

# Query nearest neighbors
neighbors = amrpg.query(q, k=10)
