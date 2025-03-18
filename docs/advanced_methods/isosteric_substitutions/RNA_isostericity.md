# RNA Isostericity Design  🧬

---

## 1. Introduction 🌟

When an RNA 3D structure is available—experimentally determined (X-ray, cryo-EM) or reliably modeled—local geometry can be directly utilized to propose mutations preserving the overall fold. Traditionally, methods rely on multiple-sequence alignments (MSAs) for identifying co-variation. Here, RNA geometry directly informs isosteric or near-isosteric substitutions, making MSAs secondary or optional.

### 🔑 Key Concepts
- **Leontis–Westhof Classification:**
  - Classifies RNA base pairs into 12 geometric families:
    - `cWW, tWW, cWH, tWH, cWS, tWS, cHH, tHH, cHS, tHS, cSS, tSS`
  - Based on hydrogen-bond edges and glycosidic bond orientations (cis/trans).

- **Isostericity & IsoDiscrepancy Index (IDI):**
  - **Isosteric Pairs (IDI ≤ 2.0):** Overlay well, preserving backbone geometry.
  - **Near-isosteric Pairs (2.0 < IDI ≤ 3.3):** Mildly perturb geometry, potentially affecting stability.

- **Environmental Constraints:**
  - Base triples/quadruples, base–phosphate contacts, stacking interactions, bridging waters, syn/anti configurations, and base–protein interactions.

### Goal 🎯

Develop a robust pipeline to:

- Accept RNA sequence, secondary structure, and 3D coordinates.

- Identify and classify base pairs and tertiary contacts directly from structure.

- Generate geometry-preserving substitution sets.

- Filter substitutions based on detailed environmental constraints.

- Generate, score, and rank candidate RNA sequences.

---

## 2. Data Structures & Inputs 📂

- **RNA Sequence:** String (e.g., `"ACGUGC"`).
- **Secondary Structure:** Base pairs (dot-bracket notation or explicit `(i,j)` pairs).
- **3D Coordinates:** Atomic coordinates (PDB or mmCIF files).
- **Isosteric/IDI Data:** Tables/matrices for each geometric family indicating isosteric and near-isosteric substitutions.
- **Environment Constraints:** Optional but highly recommended (bridging waters, base–phosphate contacts, base–protein interactions, syn/anti conformations, triples, quadruples).

---

## 3. High-Level Workflow 🛠️

1. **Load & Parse 3D Structure:**
    - Identify canonical/noncanonical base pairs and tertiary interactions.

2. **Classify Base Pairs:**
    - Apply Leontis–Westhof classification (e.g., using FR3D).

3. **Extract Isosteric Constraints:**
    - Determine geometry-compatible substitutions using IDI data.

4. **Apply Environment Filters:**
    - Exclude substitutions conflicting with bridging waters, syn/anti configurations, base–protein, base–phosphate contacts.

5. **Constraint Integration:**
    - Merge constraints consistently across nucleotides involved in multiple interactions.

6. **Sequence Generation & Constraint Satisfaction:**
    - Generate candidate sequences systematically using backtracking or constraint-solving algorithms.

7. **Scoring & Ranking:**
    - Evaluate and rank based on geometric accuracy, IDI penalties, substitution frequencies, and thermodynamic considerations.
    - Optional brief 3D refinement for top sequences.

8. **Output:**
    - Clearly ranked feasible RNA sequences.

---

## 3. Detailed Implementation & Pseudo-Code 🧑‍💻

### Detect & Classify Base Pairs 🔍
```python
def detect_and_classify_base_pairs(coords, sequence):
    base_pairs = []
    for (i, j) in candidate_pairs(coords):
        if geometric_criteria_satisfied(i, j, coords):
            family = classify_geometric_family(i, j, coords)
            env_info = gather_environment_info(i, j, coords)
            base_pairs.append((i, j, family, sequence[i], sequence[j], env_info))
    return base_pairs
```

### Build Isosteric Substitution Sets 📐
```python
def get_isosteric_substitutions(family, orig_pair, env_info, isosteric_db):
    return isosteric_db[family].get(orig_pair, set())
```

### Environment Filtering 🌊
```python
def filter_by_env(possible_pairs, env_info):
    filtered = set()
    for (X, Y) in possible_pairs:
        if environment_satisfied(X, Y, env_info):
            filtered.add((X, Y))
    return filtered
```

### Integrate Constraints 📌
```python
def build_pair_options_3D(base_pairs, isosteric_db, sequence):
    n = len(sequence)
    per_position_allowed = {i: set('ACGU') for i in range(n)}
    base_pair_options = {}

    for (i, j, fam, b1, b2, env) in base_pairs:
        candidates = get_isosteric_substitutions(fam, (b1, b2), env, isosteric_db)
        filtered_pairs = filter_by_env(candidates, env)
        base_pair_options[(i, j)] = filtered_pairs
        per_position_allowed[i] &= {x[0] for x in filtered_pairs}
        per_position_allowed[j] &= {x[1] for x in filtered_pairs}

    return per_position_allowed, base_pair_options
```

### Sequence Generation (Backtracking) 🔄
```python
def generate_sequences(sequence, per_pos_allowed, pair_options):
    solutions, partial = [], [None]*len(sequence)

    def backtrack(pos):
        if pos == len(sequence):
            solutions.append(''.join(partial))
            return
        for nt in per_pos_allowed[pos]:
            partial[pos] = nt
            if local_constraints_ok(pos, partial, pair_options):
                backtrack(pos+1)

    backtrack(0)
    return solutions
```

### Scoring & Ranking 📊
```python
def rank_solutions(solutions, base_pairs, scoring_params=None):
    scored_list = []
    for seq_candidate in solutions:
        cost = sum(compute_pair_cost(...) for pair in base_pairs)
        scored_list.append((seq_candidate, cost))
    scored_list.sort(key=lambda x: x[1])
    return scored_list
```

---

## 5. Additional Implementation Notes 📝
- Explicit handling of triple/quadruple interactions, bridging waters.
- Clarify partial vs. full redesign scope to manage computational complexity.
- Address syn/anti constraints explicitly.
- Implement heuristic strategies for computational efficiency.
- MSAs optional for functional validation, not primary design.

---

## 6. Conclusion ✅

This comprehensive RNA redesign pipeline integrates structural geometry, isostericity, environmental constraints, and computational approaches, enabling reliable RNA redesign without primary reliance on MSAs.

---

## 7. References 📚
- **Leontis–Westhof Classification:** Leontis & Westhof (RNA, 2001); Leontis, Stombaugh & Westhof (NAR, 2002).
- **IsoDiscrepancy Index (IDI):** Stombaugh et al. (NAR, 2009).
- **FR3D:** Sarver et al. (J. Math. Biol., 2008).

