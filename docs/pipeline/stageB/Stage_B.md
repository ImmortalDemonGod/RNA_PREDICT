# 🚀 Stage B: Comprehensive RNA Torsion Angle Predictor

---

## 📌 Domain, Inputs & Outputs

### 📥 Inputs

1. **RNA Sequence**
   - Length: **N** residues
   - Residues: `{A, C, G, U}` or modification tokens

2. **2D Adjacency (Base-Pair Matrix)** (from Stage A)
   - Matrix size: **N × N**
   - Values:
     - Binary (`1` paired, `0` unpaired), or
     - Real-valued probabilities `[0,1]`

3. **Optional Node Features**
   - MSA-based evolutionary profiles
   - Secondary-structure metadata (e.g., hairpin loops, non-canonical pairs)

---

## 🎯 Outputs

For each residue **i**, predict backbone torsion angles:
- Angles: **α, β, γ, δ, ε, ζ, χ**
- Optional: **Sugar pucker angle or pseudorotation (Pᵢ)**
  - Range: `[-π, π]`

### 🔧 Constraints & Goals
- **Angle periodicity**: Use sine/cosine representation to manage wraparound
- **Secondary-structure constraints**: base-pairing, backbone continuity, possible pseudoknots
- **Geometric consistency**: influenced by local and distant residues

---

## 🌐 Graph Representation & GDL Principles

Represent RNA as graph **G=(V,E)**:
- Nodes (**V**): Residues `{1, 2, ..., N}`
- Edges (**E**):
  1. Backbone edges: **i ↔ i+1**
  2. Base-pair edges from adjacency: **i ↔ j**
  3. Optional short-range edges: **i ↔ i+2**, **i ↔ i+3** (enhanced local context)

📌 Equivariant under node permutations (adjacency fixed by indexing & base pairing).

---

## 🧩 Node & Edge Feature Construction

### 🔹 Node Features (**nᵢ**)
- Sequence One-Hot (`A/C/G/U`)
- Base-Pair Stats: Sum row from adjacency; indicator "unpaired"
- Optional: MSA evolutionary profiles

Concatenate and embed:
```
𝒉ᵢ⁽⁰⁾ = Linear(nᵢ)
```

### 🔸 Edge Features (**eᵢⱼ**)
- Base-Pair Probability (`adj[i,j]`)
- Type: **Backbone** vs. **Long-range** (canonical/non-canonical)
- Sequence distance: `|i-j|` (binned/clipped)

Embed edges:
```
𝒈ᵢⱼ⁽⁰⁾ = LinearEdge(eᵢⱼ)
```

---

## ⚙️ Graph Transformer Architecture

Employ **Multi-Head Attention + Message Passing**:

### 📋 Detailed Pseudocode

```python
def GraphTransformer(nodes, edges, adjacency, L=6, c_hidden=128):
    h = Linear(nodes)  # [N, c_hidden]
    g = {(i,j): LinearEdge(edges[(i,j)]) for (i,j) in adjacency}

    for layer in range(L):
        # Node→Edge update
        for (i,j) in adjacency:
            x_ij = concat(h[i], h[j], g[(i,j)])
            g[(i,j)] += MLP_edge[layer](x_ij)

        # Edge→Node update (Multi-head Attention)
        new_h = zeros_like(h)
        for i in range(N):
            neighbors = adjacency.neighbors(i)
            attn_scores, vs = [], []
            for j in neighbors:
                score = dot(q_proj(h[i]), k_proj(h[j])) + bias_proj(g[(i,j)])
                attn_scores.append(score)
                vs.append(v_proj(h[j]))
            weights = softmax(attn_scores)
            new_h[i] = sum(weight * vs[j] for j, weight in enumerate(attn_scores))
        h = LayerNorm(h + new_h)

    return h, g
```
- Use edge embedding biases in attention.
- Optional: integrate "pairformer" or AF triangle multiplication.

---

## 🎲 Angle Prediction Head & Loss

### 🎯 Angle Output
Final node embedding (`hᵢ⁽final⁾`) via MLP:
```
anglesᵢ = MLP_final(h[i])  # [7×2] (sin/cos)
```
Then:
```
αᵢ = atan2(sin_αᵢ, cos_αᵢ), etc.
```

### 📏 Loss Function
```
L = (1/(N×7)) ∑ᵢ ∑φ [wrap(θ̂ᵢ - θᵢ)]²
```
- Optional:
  - Angle prior (A-form RNA distributions)
  - 3D coordinate-based regularization (Stage C)

---

## 📚 Training Data & Procedure

### 🗃️ Data Preparation
- Curate RNA structures (PDB)
- Compute torsion angles & adjacencies
- Assemble node & edge features

### 🎓 Training Steps
- Forward pass: GraphTransformer + angle head
- Loss: angle-based MSE or `(sin, cos)` differences
- Optimization: Adam/AdamW, learning rate scheduler
- Validation: angle-level MSE, optional Stage C 3D RMSD check

---

## 🐍 Full Python Implementation
Refer to the complete Python implementation provided above, which includes:
- `TorsionPredictor`, `GraphTransformerBlock`, `NodeEdgeAttention`
- Detailed inline comments for clarity
- Fully modular implementation suitable for PyTorch

---

## 💡 Why "Nearly Foolproof"?
1. 📍 **Graph Representation**: Explicit pairing & backbone adjacency.
2. 🌐 **Local + Global**: Transformer captures interactions at all scales.
3. 🔄 **Angle Periodicity**: Stable sine–cosine encoding.
4. ⚡ **Scalable & Efficient**: Suitable for large datasets.
4. 📐 **Physics-based Priors**: Integrate RNA angle distributions.
5. 🔗 **Stage C Compatible**: Easily feeds into 3D coordinate reconstruction.

---

## 🎖️ Final Thoughts
- Methodically addresses RNA torsion prediction complexities.
- Graph Transformer is flexible, powerful, and scalable.
- Predictive output stable through sine/cosine angle pairs.
- Structured loss function and validation improve robustness.

📌 **Outcome**: Robust, reliable RNA torsion predictor leveraging proven GDL principles.

