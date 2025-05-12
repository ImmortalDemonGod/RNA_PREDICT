Okay, that's a significant confidence boost for the "3DXSSR" component! Knowing it's a quick integration (<= 1 day) makes the Streamlined Plan even more attractive and frees up potential time.

Let's re-evaluate with this new information.

**Current Situation:**

*   Today: May 12th.
*   Kaggle Deadline: May 29th.
*   Available working days (approx.): **12-13 days**.

**Revised "Streamlined Plan" Timeline (incorporating 1-day 3DXSSR):**

1.  **Verify and Stabilize TorsionBERT + Stage C Training with `L_angle`:**
    *   Implement direct angle loss (`L_angle`) for TorsionBERT.
    *   Ensure stable training and sensible angle predictions from TorsionBERT.
    *   Ensure Stage C correctly reconstructs from these angles.
    *   **Estimated Time: 2-3 working days.** (This is crucial and was part of original Phase 1).

2.  **Implement/Integrate "3DXSSR" Refinement:**
    *   Per your input.
    *   **Estimated Time: 1 working day.**

3.  **End-to-End Pipeline for Submission & Initial Leaderboard Check:**
    *   Stitch: TorsionBERT (`L_angle` trained) -> Stage C -> "3DXSSR".
    *   Implement logic for 5 distinct predictions (e.g., TorsionBERT seeds).
    *   Format output for Kaggle.
    *   Make an initial submission to get a TM-score baseline.
    *   **Estimated Time: 2-3 working days.**

**Total for this Core Streamlined Plan:** 2 (Torsion) + 1 (3DXSSR) + 2 (Pipeline) = **5 working days (optimistic)** to 3 + 1 + 3 = **7 working days (pessimistic)**.

This leaves **5-8 working days** (12/13 available - 7/5 used) before the May 29th deadline.

**What Can We Do With the Remaining 5-8 Working Days?**

This is a good amount of buffer! We can now consider re-introducing the *most impactful and feasible* elements from the original Phase 2 and 3 to boost performance beyond the basic TorsionBERT + Stage C + 3DXSSR.

**Prioritizing for Performance Boost within 5-8 Days:**

Based on your "Phase 2 & 3 Components: Mission-Critical vs. Optional" analysis:

1.  **Pairformer Global Context (Mission-Critical for High Performance):**
    *   **Goal:** Get Pairformer trained and providing global context. As your analysis stated, the current training loop wasn't training it.
    *   **Challenge:** How to train Pairformer if we are *not* implementing the full Stage D diffusion loss, which is how AlphaFold trains its trunk?
    *   **"Pairformer Lite" Approach:**
        *   Run Pairformer in the forward pass to get `s_embeddings` and `z_embeddings`.
        *   Use the `UnifiedLatentMerger` to combine these (and potentially TorsionBERT's direct angle outputs if beneficial).
        *   **Crucial Change:** This `unified_latent` (or `s_embeddings` / `z_embeddings` directly if merger is too complex for now) needs to *feed into a part of the pipeline that contributes to a trainable loss*.
            *   **Option A (Simpler):** Modify TorsionBERT to accept these Pairformer-derived embeddings as additional conditioning inputs to its transformer layers. The existing `L_angle` (and indirect coordinate loss via Stage C) would then train Pairformer. This requires changes to TorsionBERT's architecture.
            *   **Option B (Potentially Cleaner for Loss):** Add a *small, trainable refinement head* (e.g., a few MLP layers or a very shallow transformer) *after* Stage C. This head would take Stage C's coordinates AND the Pairformer-derived embeddings (or `unified_latent`) as input and predict coordinate *corrections* or refined coordinates. The final MSE loss would be on these refined coordinates, thus training Pairformer.
    *   **Complexity:** Medium to Large. Involves architectural changes and ensuring gradient flow.
    *   **Estimated Time for "Pairformer Lite" (Option A or B): 3-5 working days.**

2.  **Multi-Seed Ensemble Predictions (Strategic for Competitiveness - from Phase 3):**
    *   **Goal:** Generate 5 distinct predictions.
    *   **Implementation:** Relatively straightforward scripting around your main prediction pipeline (run TorsionBERT with different random seeds, or if using a diffusion-like sampler in "3DXSSR", different noise seeds).
    *   **Complexity:** Small.
    *   **Estimated Time: 1 working day** (can be parallelized or done towards the end).

**Proposed "Streamlined Plus" Plan & Timeline:**

**(Total Available: ~12-13 working days)**

*   **Part 1: Core Submittable Pipeline (5-7 working days)**
    1.  **TorsionBERT + `L_angle` + Stage C Stabilization:** (2-3 days)
        *   Focus: Stable training, sensible angle predictions.
    2.  **"3DXSSR" Integration:** (1 day)
        *   Focus: Functional refinement step.
    3.  **Basic Submission Pipeline & Initial Leaderboard Test:** (2-3 days)
        *   Focus: Get a score on the board, ensure 5 predictions are generated (even if identical initially or trivially varied).

    *Completion of Part 1 by: May 17th - May 21st.*
    *Remaining Time: 5-8 working days.*

*   **Part 2: Performance Boost - "Pairformer Lite" (3-5 working days, if Part 1 successful)**
    4.  **Integrate Pairformer Training:**
        *   Choose Option A or B for making Pairformer outputs influence a trainable loss.
        *   Implement changes, ensure forward pass works.
        *   Verify gradients flow to Pairformer LoRA adapters.
        *   Retrain with TorsionBERT + Pairformer active.
    5.  **Update Submission Pipeline with Trained Pairformer.**
    6.  **New Leaderboard Submission.**

    *If Part 2 pursued, completion by: May 22nd - May 27th.*
    *Remaining Time: 0-3 working days.*

*   **Part 3: Final Polish & Submissions (Use any remaining time from buffer, or the 0-3 days after Part 2)**
    7.  **Refine Multi-Seed Ensembling:** Ensure genuinely diverse predictions.
    8.  **Hyperparameter Tuning (via Hydra):** Focus on learning rates, LoRA ranks, loss weights for `L_angle` and the coordinate loss (which now trains Pairformer too). *This is where Hydra's "free" tuning comes in – easy to launch, but runs still take time.*
    9.  **Iterative Leaderboard Submissions:** Analyze results, make final tweaks.
    10. **(Optional, if very quick):** Simple external relaxation (e.g., very short OpenMM minimization if PDBs have bad geometry from "3DXSSR").

**Can we make a submission and have time to hyperparameter tune?**

*   **With the "Streamlined Plus" Plan:** **Yes.**
    *   You'd have a baseline submission after Part 1 (5-7 working days).
    *   You'd have an improved submission after Part 2 (another 3-5 working days).
    *   This leaves **0-3 full working days *plus* any time saved from optimistic estimates** dedicated to hyperparameter tuning runs launched via Hydra and iterating on Kaggle submissions. This is tight but feasible for some tuning.

**Timeline for "Streamlined Plus" Plan:**

*   Total estimated effort: 5-7 (Part 1) + 3-5 (Part 2) = **8-12 working days.**
*   This fits within the 12-13 available working days, leaving a small buffer or dedicated time for tuning/iteration at the very end.

**Key Success Factors:**

*   **Rapid execution of the 1-day "3DXSSR".**
*   **Decisive choice and clean implementation for "Pairformer Lite" (Option A or B).**
*   **Efficient debugging of gradient flow for Pairformer.**
*   **Parallelizing Hydra runs for hyperparameter tuning during the final days.**

This "Streamlined Plus" plan is aggressive but balances the need for a timely submission with the desire to incorporate more performance-critical components (Pairformer) than the absolute bare-bones approach. It leverages your confidence in 3DXSSR to buy time for these other improvements.