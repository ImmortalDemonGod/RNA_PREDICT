
Below is a helpful overview of the main experimental findings and performance of Liquid-S4 compared to a wide range of baselines on various long-sequence benchmarks. We also include methodological details such as pseudocode for the Liquid-S4 kernel, references to the underlying math, and a hyperparameter table for reproducing the results.

⸻

Main Contributions Recap

Liquid-S4 is a new state-space model that leverages a linear variant of the Liquid Time-Constant network (LTC). It keeps the core ideas of Structured State-Space (S4) layers (i.e., diagonal-plus-low-rank matrix parameterization and HiPPO-based initialization), and introduces a liquid convolution kernel to incorporate higher-order auto-correlations of input signals.
	1.	Liquid-S4 Kernel: We derive an extra convolution kernel that captures covariance terms among input samples, without losing the efficient Cauchy kernel approach introduced by S4.
	2.	Empirical Results: Liquid-S4 consistently achieves better generalization on the Long Range Arena (LRA) benchmark, 1D pixel-level image classification (sCIFAR), raw Speech Commands, and BIDMC medical time series, surpassing S4, S4D, and other strong sequence models.
	3.	Implementation & Efficiency: Liquid-S4 is straightforward to implement, extending S4’s convolution kernel with only minor overhead for the input-correlation kernel.

Below, we present:
	1.	Comparison to Baselines: Tables for LRA and Speech Commands, sCIFAR, and BIDMC datasets.
	2.	Liquid-S4 Kernel Pseudocode
	3.	Hyperparameters

⸻

1. Summary of Experimental Results

A. Long Range Arena (LRA)

Model	ListOps	IMDB	AAN	CIFAR	PathF.	Path-X	Avg.
Transformer	36.37	64.27	57.46	42.44	71.40	–	54.39
Reformer	37.27	56.10	53.40	38.07	68.50	–	50.56
S4-LegS	59.60	86.82	90.90	88.65	94.20	96.35	86.09
S4D-LegS	60.47	86.18	89.46	88.19	93.06	91.95	84.89
S5 (Simplified-S4)	61.00	86.51	88.26	86.14	87.57	85.25	82.46
Liquid-S4 (ours)	62.75	89.02	91.20	89.50	94.80	96.66	87.32

Key Takeaways:
	•	Liquid-S4 achieves new best results on four of the six tasks (ListOps, IMDB, AAN, CIFAR) and ties closely on Pathfinder and Path-X.
	•	Overall average of 87.32% surpasses prior strong methods like S4-LegS (86.09%) and S4D-LegS (84.89%).

⸻

B. BIDMC Vital Signs (Medical Time Series)

Model	HR	RR	SpO2
CKConv (Conv)	2.05	1.214	1.051
S4-LegS	0.332	0.247	0.090
S4-(LegS/FouT)	0.344	0.163	0.080
Liquid-S4 (ours)	0.303	0.158	0.066

(Lower is better: Root-Mean-Squared Error.)

Key Takeaways:
	•	Liquid-S4 obtains lower RMSE than all S4, S4D, and convolution/RNN-based methods on HR, RR, and SpO2 tasks.
	•	Particularly large gains on SpO2 (0.066 vs. 0.080–0.102 for prior S4 variants).

⸻

C. 1-D Pixel-Level CIFAR (sCIFAR)

Model	Accuracy (%)
LSTM	63.01
IndRNN	96.0
S4-LegS	91.80
S4D-Inv	90.69
Liquid-S4 (ours)	92.02

Key Takeaways:
	•	Liquid-S4 sets a new state-of-the-art of 92.0% on sCIFAR (1-D flattening).

⸻

D. Raw Speech Commands (Full 35-Way Task)

Model	# Params	16kHz	8kHz (0-shot)
ResNet-18	216k	77.86	8.74
S4-LegS	307k	96.08	91.32
S4D-Lin	306k	96.25	91.58
Liquid-S4 (ours)	224k	96.78	90.00

Key Takeaways:
	•	On the standard 16kHz test, Liquid-S4 obtains the best accuracy (96.78%), with ~30% fewer parameters vs. S4.
	•	On the zero-shot 8kHz test, S4D-lin does slightly better (91.58% vs. 90.00%), but Liquid-S4 remains competitive.

⸻

2. Liquid-S4 Kernel Computation

Recall that a standard S4 kernel (for discrete input sequence \{u_k\} of length L) is computed via
K(\ell) \;=\; C\,A^\ell\,B,
\quad \ell = 0, 1, …, L-1.
We can turn the linear state-space model into a liquid variant by adding input-dependent transitions. In a linearized LTC system, the continuous-time ODE is:
\frac{d}{dt} x(t) \;=\; (A + B\,u(t))\,x(t) + B\,u(t),
\quad y(t) = C\,x(t).
Upon discretization, the resulting unrolled convolution kernel has two components:
	1.	The standard S4 kernel for \{u_k\}.
	2.	An extra “liquid” kernel that accounts for correlations like u_i \cdot u_j, u_i \cdot u_j \cdot u_k, etc., up to order p.

PB (Power-of-B) Mode

A more efficient variant is to replace A by the identity matrix for the correlation portion. This yields:

\textstyle K_{\mathrm{liquid}}^{(p)} \;=\; C \; \bigl(B^{p}\bigr)
\quad\text{(with flipping/anti-diagonal alignment for time indices).}
In practice, the code merges these kernels and uses the same diagonal-plus-low-rank structure for A. The correlation order p is a small hyperparameter (often 2–4). See the pseudocode.

Liquid-S4 Kernel Pseudocode (in JAX-like style)

# Pseudocode for the Liquid-S4 kernel in PB mode 
# (easier to compute, typical default choice).

def liquid_s4_kernel_PB(A_params, B, C, P, L):
    """
    Args:
       A_params: Parameters for the S4 kernel (e.g. diag + low-rank).
       B:        [N,] input vector
       C:        [N,] output vector
       P:        integer, max correlation order
       L:        integer, sequence length
    Returns:
       kernel_s4: the standard S4 kernel, shape [L,]
       kernel_liquid: the correlation kernel, shape [p <= P, L,]
    """
    # 1) Compute the base S4 kernel via standard approach
    kernel_s4 = s4_convolution_kernel(A_params, B, C, L)  # shape [L,]
    
    # 2) Build correlation kernels for each order p = 2..P
    #    We skip the repeated matrix-power of A, using identity for correlation part.
    kernel_liquid = []
    for p_order in range(2, P+1):
        # "Power of B" approach: simply do C * (B^(p_order)), plus a temporal flip.
        # Implementation detail: you'd broadcast along time dimension L.
        # We denote flipping by an anti-diagonal pass. 
        # Below is conceptual.
        corr_kernel_p = C * (B ** p_order)   # shape [N]
        # replicate or flip for each time-lag combination ...
        # final shape -> [L,], matched with special indexing
        kernel_liquid.append(corr_kernel_p)
        
    return kernel_s4, kernel_liquid



⸻

3. Hyperparameter Settings

Below are typical settings that gave best results. We highlight that Liquid-S4 often requires fewer states or hidden features than S4, thanks to the correlation kernel.

Table: Per-task hyperparameters for Liquid-S4 (with PB kernel).

Task	Depth	H (features)	StateSize	Norm	Dropout	LR	Batch	Epochs	WD	p (order)
LRA-ListOps	9	128	7	BN	0.01	0.002	12	30	0.03	3
LRA-IMDB	4	128	7	BN	0.1	0.003	8	50	0.01	2
LRA-Retrieval	6	256	64	BN	0.2	0.005	16	20	0.05	2
LRA-Image	6	512	512	LN	0.1	0.010	16	200	0.03	2
LRA-Pathfinder	6	256	64	BN	0.0	0.0004	4	200	0.03	2
LRA-PathX	6	320	64	BN	0.0	0.001	8	60	0.05	2
sCIFAR	6	512	512	LN	0.1	0.010	50	200	0.03	3
SpeechCmd (35)	6	128	7	BN	0.0	0.008	10	50	0.05	2
BIDMC (RR/HR)	6	128	256	LN	0.0	0.005–0.01	32	500	0.01	2–4

Notes:
	•	“Depth” = # of Liquid-S4 blocks stacked.
	•	“H” = # of features in the hidden dimension for the feedforward or mixing layers.
	•	“StateSize” = dimension N in the S4 parameterization for the convolution kernel. We often choose smaller N for Liquid-S4.
	•	“p (order)” = max correlation terms. Typically 2 or 3 suffices.

⸻

Conclusions

Liquid-S4 combines the continuous-time insight of Liquid Networks with the diagonal-plus-low-rank S4 approach to produce an additional liquid correlation kernel. With minimal overhead, it achieves top performance across LRA, speech, medical, and image tasks. It surpasses the strong S4 and S4D baselines and remains highly efficient due to the same Cauchy kernel computations that S4 employs.

Code Availability: A reference PyTorch/JAX code will be made available at GitHub: https://github.com/raminmh/liquid-s4.

We hope this encourages broader exploration of combined continuous-time state-space models and polynomial expansions for capturing long-range dependencies.


Below is an addendum highlighting additional points that help complete the bigger picture behind all the core state-space modeling references (on S4, LTC, S5, and now Liquid-S4). These points are often either only mentioned in passing or omitted from the main summary but can be important for completeness.

⸻

Addendum to Liquid-S4 & Related Papers

1. Full Theoretical Details on the S4 Cauchy Kernel & Orthogonal Bases

A crucial aspect of the S4 framework is the detailed derivation of its efficient convolution kernel:
	•	Matrix Powers \{A^k\} are typically expensive for large k. S4 addresses this using a specialized Cauchy Kernel evaluation in the frequency domain.
	•	It involves evaluating \sum_{k=0}^{L-1} C\,A^k\,B quickly via a polynomial transform. In particular, the sum of terms A^k is tackled by viewing \Lambda = \mathrm{diag}(\lambda_i) plus low-rank updates, converting the problem into a Cauchy matrix inversion step.
	•	The \mathcal{O}(N + L\log L) or \mathcal{O}(N + L)-type complexities come from carefully applying FFT plus the “black-box” factorization.

For thoroughness, the user might want references to the specific theorem in Gu et al. (2022) (or Gu et al. (2021)) that details how the Cauchy kernel is set up, and how “Woodbury identity + a set of points on the unit circle” (the roots of unity) factor in.

Additional references:
	•	S4: Efficiently Modeling Long Sequences with Structured State Spaces (Gu et al., ICLR 2022).
	•	Combining Recurrent, Convolutional, and Continuous-Time Models with Linear State Space Layers (Gu et al., NeurIPS 2021).

2. Construction of the LegS Matrix & “Normal + Low-rank” vs. “Diagonal + Low-rank”

The summary mentions the HiPPO-LegS matrix but does not detail how that LegS matrix arises:
	•	LegS stands for “Legendre State,” which is derived from a particular continuous weighting measure on [0,1].
	•	The “scaled Legendre measure” leads to a companion matrix that encodes how polynomials up to degree N track (and compress) the input function’s history in an exponentially decaying manner.
	•	In practice, we rarely keep the entire raw LegS matrix. Instead, we represent it via Normal + Low-Rank or a simpler Diagonal + Low-Rank parameterization.
	•	S4 typically uses:
A_{\mathrm{LegS}} \;=\; V\,\Lambda\,V^* \;-\; P\,Q^\top
\quad\text{(the so-called NPLR form)}
with further constraints to keep eigenvalues stable.

Hence, if the user wants a deeper conceptual view, they might read about how the Legendre polynomials and the continuous-time approach tie to “HiPPO (Highly Productive Polynomial Projections)” to preserve memory of past inputs.

3. Handling Time-Varying or Irregular Sampling

While LTCs and S4 can each handle “time-varying” aspects, the summary only briefly points out LTC’s advantage in continuous domains:
	•	Irregular data: LTC or CfC can handle adaptive time steps easily because one can solve the ODE with variable \Delta t. S5 (or S4 in convolution mode) typically wants uniform steps.
	•	Liquid-S4 might, in principle, handle partial irregularities with a dynamic kernel, but the exact method for that is not spelled out in the main text.
	•	For tasks like “health monitoring with irregular intervals,” the continuous-time representation (like LTC) can be more direct; the user might rely on direct ODE integration or an equivalent parallel-scan approach if the discrete steps vary.

4. Additional Baselines or Sequence Labeling Tasks

Some references compare these methods on tasks that require, e.g., alignment or partial derivatives:
	•	E.g., S4 or LTC in seq2seq tasks with alignment might require more advanced decoding.
	•	Cauchy- or polynomial-based transforms can be adapted to multi-layer RNN stacks with gating.

These are secondary but can matter if the user wants to see how these SSM-based layers handle advanced problems like language modeling with “partial derivatives” or structured prediction tasks (beyond classification).

5. BPTT, Memory Usage, and Parallelization Details
	•	Backprop Through Time (BPTT): The summary mentions Liquid-S4’s efficiency but omits explicit memory cost.
	•	For S4 in convolution mode, offline inference can do all steps in \mathcal{O}(N + L \log L). However, the training memory can differ if done naively.
	•	Some frameworks store partial states or do partial recomputations to keep memory feasible.
	•	Parallelization:
	•	S4 transforms the sequence to the frequency domain and does big parallel ops.
	•	LTC-based or S5-based models often do “scan” type parallel.
	•	It can be important to note that S5 had a parallel scan approach giving \mathcal{O}(L \log L) time with \mathcal{O}(L) processors, whereas S4 uses convolution. Liquid-S4 inherits S4’s approach with an added correlation kernel overhead \sim p \times L.

⸻

Quick “Key Theorem or Structural Insights” Recap
	1.	HiPPO
	•	Theorem: The LegS matrix arises from an orthonormal polynomial basis that can approximate the entire past of a signal on [0,1] under exponential decay.
	•	This ensures near-perfect memory for potentially unbounded input lengths.
	2.	Liquid Time-Constant
	•	Insight: The input modifies the time constant of each neuron, allowing a “bilinear term” (A + B\,u)x. This yields a dynamic system that can more flexibly adapt to input data during inference.
	3.	S4
	•	Core: The diagonal-plus-low-rank (DPLR) or normal-plus-low-rank approach to factor the LegS matrix. Then a black-box Cauchy kernel method for \{CA^kB\}.
	•	Results in \mathcal{O}(N + L) or \mathcal{O}(N + L \log L) for large sequences.
	4.	S5
	•	Core: Switch from many single-input SSMs (S4’s block-diagonal) to a single multi-input SSM, plus diagonalization for parallel-scan.
	•	Gains simpler time-domain approach at similar complexity.
	5.	Liquid-S4
	•	Core: Combine the polynomial expansions of S4 with an additional correlation-based kernel from LTC’s (A + B\,u). Gains better representation with minimal overhead.

Hence, the user can see how each of these builds or modifies something from the previous: LTC adds input gating, S4 speeds up standard linear SSM training, S5 reorganizes the S4 block structure, and Liquid-S4 merges LTC’s gating with the S4 kernel approach.

⸻

Final Note

In practice, each method or extension is addressing a slightly different “gap”:
	•	S4 and S4D: Fast kernel-based linear state-space layers for extremely long sequences.
	•	LTC/CfC: Causal continuous-time RNNs with input-dependent gating, strong out-of-distribution reliability.
	•	S5: A simpler time-domain approach to the S4 concept.
	•	Liquid-S4: Merges LTC’s correlation + gating with S4’s diagonal-plus-low-rank memorization.

If the user wants more advanced tasks (e.g., multi-step forecasting, speech generation, partial observation), or more advanced theorems, the original references are recommended.

	In summary, these notes fill in some of the minor gaps about the LegS matrix derivation, advanced time-varying extensions, memory usage, and how each approach handles big-latency tasks. They also clarify that for truly irregular sampling, LTC-based methods might have an advantage with direct ODE solvers, while S4-based methods (including Liquid-S4) typically assume uniform steps or process data with convolution/FFT.





Below is a paper-focused analysis of AlphaFold 3’s architecture (main text + Supplementary) with an eye toward where one might latch onto “S4-like” (liquid S4 or otherwise) ideas despite the dynamic, 3D adjacency. I’ll highlight specific aspects of the AF3 design that might be overlooked but could help in “bypassing” or re-purposing adjacency to enable large-T diffusion unrolling:

⸻

1. AF3’s Key Architecture Pieces & Where the Bottleneck Arises
	1.	Token + Pair Representation (Sections 2.6, 3.6)
	•	AF3 merges the idea of “residues → tokens” but also lumps each nonstandard residue/ligand atom into per-atom tokens.
	•	The main representation is a 2D pair array \mathbf{z}_{ij} (plus a 1D single representation \mathbf{s}_i).
	•	Even though it’s nominally “N tokens,” each token might be an entire residue or a single ligand atom. Because adjacency is effectively global (the pair array is size N \times N), it’s not trivially 1D.
	2.	Atom Attention (Sections 3.2, 3.7)
	•	AF3 does a “sequence-local atom attention” (Supplementary Fig. 1 and Alg. 5–7) to incorporate fine-grained local geometry before building the token representation.
	•	They chunk the full set of atoms into blocks along the “sequence axis” (the tokens) so each atom can attend to a bigger window of ~128 neighboring atoms. This is already a quasi-1D block approach to adjacency, but it’s still complicated by the fact that the “neighborhood” is purely index-based with a fallback to the true distances for actual geometry (line 3 in Algorithm 5, etc.).
	3.	Diffusion Steps Are Typically Low
	•	AF3’s final diffusion module is run for \sim50 steps at inference by default, or ~200 in certain cases. The paper notes they do only 20 steps in the “mini-rollout” for training.
	•	So, the typical method for geometry-based diffusion is to keep T small because each step involves a big forward pass over the trunk (or at least a partial pass).

In your scenario, you want to:

		•	“Swap the big trunk for an S4-based model that can handle up to 10k steps more cheaply at test time,”
	•	Potentially “lower parameter count” but still do large unrolls in time.

Hence the question: does the AF3 pipeline contain a trick or submodule that we can adapt into an S4-friendly (1D) approach—without incurring the full 2D adjacency penalty at each step?

⸻

2. Where AF3 Might Offer a Bridge for Large-T

2.1 “Sequence-Local Atom Attention” as a 1D Surrogate

In Supplementary Fig. 1 & Algorithm 5/7, they mention:

		•	They only do a banded self-attention of size (N_{\mathrm{atoms}} \times N_{\mathrm{atoms}}) but in rectangular blocks along the diagonal.
	•	Conceptually, each subset of 32 atoms attends to 128 neighbors (Algorithm 7: Nqueries=32, Nkeys=128).

Why This Matters for S4
	•	S4 can handle a single dimension well; if we interpret the “\sim N_{\text{atoms}}” dimension as a 1D axis, we might try to treat it as a 1D sequence.
	•	AF3 does so with partial block-based attention, effectively a local 1D adjacency in the “atom index” space.

However, the catch is that the adjacency in 3D is only approximately captured by that block-limited approach. A standard S4 or Liquid-S4 would rely on a fixed or small-rank operator A over an entire length N. In AF3, that block is a hand-coded mask that slides over the (atom index). So if you want 10k diffusion steps, you could:
	1.	Keep the same block “mask” (i.e. let each atom attend to a 128-neighbor region in the 1D ordering)
	2.	Replace that 2D attention module with a 1D S4-based module (plus some local gating for adjacency).

But you’d still need to figure out how to incorporate actual geometric distances if the atoms move significantly. AF3’s local block is “sequence-based,” not distance-based. So it only implicitly respects geometry once tokens are formed.

2.2 The “Mini-Rollout” Mechanism

Section 4.1 of the Supplement: AF3 does a “short diffusion rollout” from random noise for ~20 steps during training to produce an approximate final structure used for alignment or confidence-head training. This doesn’t train the entire trunk with 20 steps unrolled. They do it partially “off to the side.”
	•	If you want 10k steps for your final inference, maybe you can adopt a similar approach:
	•	Train a smaller S4-based “decoder” that is unrolled for fewer steps during training (like 20–50).
	•	At inference, you ramp up to 10k steps if you want finer increments.
	•	The AF3 approach of “mini-rollout” is basically truncated unrolling plus a final alignment. That’s reminiscent of truncated backprop in RNNs.

Implication: The paper’s training pipeline already acknowledges you don’t want to fully unroll (the trunk is too big). They do partial. That’s conceptually close to a “time-aware S4 approach,” except they aren’t using the trunk repeatedly at each step. Instead, they do it once or a few times and then rely on the cheaper diffusion module.

2.3 Using the “PairformerStack” Fewer Times

AF3 has “Ncycle = 4” recycles in the trunk, and that’s it. Then the diffusion module is a cheap sub-network that sees the trunk embeddings. So the paper does:
	1.	Run Pairformer + MSA stack ~4 times (the big trunk).
	2.	Pass the final embeddings into a smaller DiffusionModule for ~50 steps at inference.

Hence: If you want a large number of diffusion steps T=10k, you only pay for the diffusion module repeated 10k times, not the entire trunk. That’s already in AF3. Possibly you can make that DiffusionModule an S4-based module (like you propose). The trunk is not repeated 10k times.
	•	This is spelled out in Algorithm 1 (MainInferenceLoop) and Algorithm 18 (SampleDiffusion). Notice that the trunk (Pairformer, etc.) is used only a handful of times (Ncycle=4). The big repeated loop (line 2–12 in Algorithm 18) calls DiffusionModule  \(not\ the trunk).
	•	So the foundation for “Test-time scaling on T** is already in the code**: they do the trunk once, the diffusion 50 times. Just push 50 → 10k.

The real adjacency question: Inside the diffusion module, do you see reference to the pair representation \(\mathbf{z}{ij}\)? Yes, they do feed \mathbf{z}{ij} in each time, but that’s presumably frozen after the trunk has finished. The geometry changes at each step, but the trunk features \mathbf{z}_{ij} do not. That’s how they sidestep re-building adjacency for each step. They do a random center and random rotation in each step (Algorithm 19), but that’s a cheap transform, not a re-run of the trunk.

Hence the big question: does that mismatch hamper accuracy if you do 10k steps? Possibly it’s fine, or possibly you want to re-check adjacency after big geometry changes. But the “lack of adjacency re-check” is exactly how AF3 keeps it cheap.

2.4 PDE Head & The “No Re-run” Trick

Another overlooked detail:
	•	They do not re-run the pair embedder or MSA after each partial geometry update. They keep a “static” pair representation from the trunk.
	•	Then the diffusion module is a purely non-geometric transformer with local cross attention.
	•	Even the “atom cross attention” in the diffusion module is partial, not a full adjacency re-check. They are effectively ignoring the fact that adjacency might shift as the structure changes.

Thus, the main “bottleneck” they mention—“3D geometry or a dynamic adjacency matrix is not obviously ‘fixed’ or ‘1D’,”—they solve by ignoring dynamic adjacency at diffusion time. They rely on the trunk to have gleaned enough local geometry constraints, so the diffusion “just polishes” the structure.

⸻

3. “Secret Overlooked Levers” in the Paper

Based on the text, three relevant levers might be under-discussed but helpful:
	1.	Masking or Approx. Adjacency in the Diffusion Decoder
	•	The diffusion module (Alg. 20) uses local “Sequence-local Atom Attention” as a step. But that local shape is chosen somewhat arbitrarily.
	•	You could re-implement that local shape as a 1D S4 if you interpret the atoms in some linear order. That’s effectively a “sliding window.”
	•	The upshot: it’s exactly how they do it in the trunk’s “AtomAttentionEncoder,” so presumably you could unify or share that approach inside the diffusion module.
	2.	Static \mathbf{z}_{ij} Instead of Recomputing
	•	They keep the pair representation \mathbf{z}_{ij} from the trunk. That means they do not attempt to update adjacency for large geometry changes.
	•	If you want 10k steps, you might try the same trick—just keep a static adjacency or a static representation, to feed your S4 “time-slicing.”
	•	This is simpler (and matches the official code). The model may degrade on large domain motions, but that’s the approach AF3 uses.
	3.	“Discrete PDE” Insight
	•	If you re-check the “smooth LDDT loss” or the notion that each step can see local constraints, you might glean that AF3 lumps all the local geometry constraints into a final MSE and LDDT. They do not worry about intermediate adjacency steps.
	•	So you can treat the final or near-final adjacency as “close enough.”

⸻

4. Possible Plan to Incorporate S4 at 10k Steps

Putting it all together:
	1.	Do the trunk once with Ncycle=4, produce the big pair representation \mathbf{z}_{ij}.
	2.	Create an S4-based diffuser that (like AF3’s diffusion module) is repeated T=10k times.
	•	Input: The final single representation \mathbf{s}i and pair representation \mathbf{z}{ij} from the trunk (both remain fixed).
	•	At each step, we have current coordinates \{\mathbf{x}_l\}. Possibly convert them to a 1D format, letting each atom be an index in the “time series.”
	•	Then apply a 1D S4 to that per-atom or per-token sequence. That yields an updated coordinate shift.
	3.	Retain the existing “static adjacency” from \mathbf{z}_{ij}. If you want some local doping (like a gating vector that depends on distance to help the S4 know if two tokens are near each other), you can do what the paper does: partial cross attention with a logistic or gating. But do not re-run the entire Pairformer.
	4.	Because S4 is \mathcal{O}(N \log N) or \mathcal{O}(N) in sequence length, you can manage 10k steps as long as your number of tokens is not monstrous. AF3 tries to keep it below a few thousand tokens.

Yes, adjacency might become stale after many steps of large rearrangements. But that’s how AF3 is done today. If you can live with that approximation, it’s the official route.

⸻

5. Concluding Thoughts
	•	AF3’s Paper Already “Ignores” Dynamic Adjacency at diffusion time to keep it cheap. They effectively treat the trunk’s pair representation as static. This is precisely how they bypass the \mathcal{O}(N^2) adjacency cost.
	•	Large T is feasible if the diffusion sub-network is small enough and does not re-run big adjacency expansions. That’s the principle.
	•	If you specifically want an S4-lens, you can drop in S4 for the “AtomAttentionEncoder/Decoder” in the diffusion module. Overlooked detail: the “sequence-local” chunking can be replaced by a simpler 1D convolution-based SSM (S4).
	•	So the real “bottleneck” is not truly “lack of 1D.” The model overcame that by letting adjacency remain static and by chunking. As a result, 10k steps is plausible if you have enough memory and time to do 10k passes with a small S4-based diffuser.

In short, the two big “overlooked” enablers in the paper are (1) that the trunk is not re-run at each step, (2) the adjacency is effectively not dynamic during the diffusion. They pinned it in place from the final trunk’s pair representation. That’s how you might integrate a 1D S4 approach for large T without re-labelling adjacency at each iteration.

[emoji]: That’s the gist. By focusing on the DiffusionModule in the paper—and how it already does not rely on dynamic adjacency—you can see a path to an “S4-based big unroll” that leverages 10k steps while side-stepping the “3D adjacency explosion.”

Below is a paper-focused analysis of AlphaFold 3’s architecture (main text + Supplementary) with an eye toward where one might latch onto “S4-like” (liquid S4 or otherwise) ideas despite the dynamic, 3D adjacency. I’ll highlight specific aspects of the AF3 design that might be overlooked but could help in “bypassing” or re-purposing adjacency to enable large-T (e.g. 10,000-step) diffusion unrolling at test time:

⸻

1. AF3’s Key Architecture Pieces & Where the Bottleneck Arises
	1.	Tokens & Pair Representations
	•	AF3 merges “residues \to tokens,” plus lumps nonstandard residue/ligand atoms into per-atom tokens.
	•	It then builds a 2D pair array of size (N\times N) (as well as a 1D ‘single representation’), leading to pairwise adjacency.
	•	Because adjacency is effectively global at this stage, it’s not trivially “1D,” so standard 1D S4 can’t just be dropped in for the trunk.
	2.	Atom Attention
	•	Before building the token representation, AF3 does a “sequence-local atom attention” (Supplementary Fig. 1, Algs. 5–7) on blocks of atoms along a “sequence axis” in chunks of size 32 or 128.
	•	Conceptually this is already a pseudo-1D approach to adjacency—the adjacency is purely index-based or block-based, not distance-based.
	3.	Diffusion Steps Are Typically Small
	•	AF3’s final diffusion module is unrolled for ~50–200 steps at inference, rarely more.
	•	So they do not re-run the expensive trunk for each small diffusion step. Instead, they freeze the trunk outputs and feed them once into a cheaper diffusion sub-network that runs multiple times.

Hence, if you want to push T to 10k steps, you need to see how AF3 is already skipping the big adjacency expansions for each step.

⸻

2. Where AF3 Might Provide a Bridge to Large \mathbf{T}

2.1 Static Pair Representation + Cheap Diffusion

Crucially, in AF3:
	•	They run the big trunk (PairFormer, MSA module) a handful of times (Ncycle=4).
	•	They then store the final embeddings \mathbf{z}_{ij} and \mathbf{s}_i.
	•	The diffusion module sees only these final embeddings, does not recast adjacency, and iterates \sim50 times.

That approach already “ignores” dynamic adjacency in the diffusion loop. They rely on a single snapshot of pair representation (and some local ‘atom attention’ step) to apply each incremental geometry update \Delta \mathbf{x}.

If you want to do 10k steps, you can do the same:
	1.	Run the trunk once.
	2.	Keep the final pair representation \mathbf{z}_{ij}.
	3.	Substitute AF3’s small diffusion transformer with a 1D S4-based “diffuser” that you unroll 10k times.

2.2 “AtomAttentionEncoder/Decoder” Substitution

Inside the diffusion module (Algorithm 20 and the “AtomAttentionEncoder/Decoder” calls), there is a local, block-based 1D self-attention over subsets of atoms. You could:
	•	Replace that local self-attention with a 1D S4 approach (sliding window or full).
	•	Maintain the trunk’s final embeddings \mathbf{z}_{ij} as a static gating or skip connection.

Hence you get a 1D S4-lens on your geometry (just as AF3 does with a local attention lens), repeated for each diffusion step, but not re-running the entire trunk adjacency.

2.3 “No Re-run” Trick & Potential Accuracy Gaps

Because AF3 does not re-run adjacency for each step, that’s the “trick” that keeps it feasible. If you want truly correct adjacency after big structural changes, you’d have to re-run the trunk, which is \mathcal{O}(N^2). But AF3’s official pipeline doesn’t do that either—they rely on the trunk’s final embeddings and accept some inaccuracy from ignoring newly formed or broken contacts at large step counts.

⸻

3. Suggested Path to “S4-like” Diffusion in AF3

Putting it all together:
	1.	Trunk Once: Just like AF3, run the trunk 4 times, produce final embeddings \mathbf{z}_{ij}, \mathbf{s}_i.
	2.	S4-based Diffuser: At test time, do 10k steps with a custom S4 block that sees the current coordinates \mathbf{x} and the trunk embeddings. Possibly also keep the partial “atom-attention” if you want a small local fix for adjacency, or skip it.
	3.	Large T: Because the trunk is never re-run, you can scale T up to 10k. The main cost is the repeated S4 pass over the 1D sequence of N atoms. Provided N is a few thousand, that can be feasible.

Yes, you lose truly dynamic adjacency updates. But that’s exactly how AF3 does it (the trunk is not re-called at each step). So your approach is consistent with the official code’s “frozen adjacency” logic, meaning you can unroll many times.

⸻

4. Key Observations to Leverage
	1.	Static adjacency: The pair \mathbf{z}_{ij} is not re-computed, so you don’t pay \mathcal{O}(N^2) each step.
	2.	1D “atom index”: The diffusion sub-network does local “sequence-based” attention. That can be replaced by S4’s 1D kernel, ignoring that the underlying data is 3D geometry.
	3.	Partial adjacency: If needed, you can keep some small gating that references \mathbf{z}_{ij} or distances for local neighbors.

So from a “S4-like” perspective, the overlooked detail is that AF3’s diffusion is basically done in a 1D style anyway, ignoring dynamic adjacency. This is where you can insert Liquid-S4 or standard S4 for a 10k-step unroll, while still depending on the trunk’s final adjacency for global constraints.

⸻

5. Concluding Outline
	•	AF3’s trunk is expensive and used only a handful of times.
	•	The diffusion is a purely local 1D stepping process with a static adjacency embedding.
	•	Hence you can swap out AF3’s small diffusion transformer for a 1D S4 (or Liquid-S4) block—the adjacency is basically not re-queried anyway.
	•	That’s how to get large-T test-time scaling via an S4-lens, and no new big adjacency expansions are needed.

If you accept that geometry-based adjacency is “frozen” after the trunk, you get the best of both worlds: a big diffusion unroll at test time, with a powerful 1D S4-based “decoder,” and no repeated \mathcal{O}(N^2) cost each step.


Below is a conceptual pseudocode draft for how you might create a unified Python-style library that combines the various SSM methods (S4, S4D, LTC gating, correlation kernels, etc.) under one flexible interface. The idea is that you can switch on or off certain features like “diagonal vs. NPLR,” “LTC gating,” “correlation order p,” “offline convolution vs. parallel scan,” etc., all in one core class.

This is only illustrative code—i.e., it won’t run out of the box—and you’ll likely replace placeholders (TODO) with actual implementations for the kernel computations, matrix builds, ODE solvers, etc.

class StateSpaceBlock:
    """
    A unified class to handle different SSM-based sequence layers:
      - S4  or  S4D    (by controlling 'ssm_type': 'NPLR' vs 'diagonal')
      - LTC gating     (by controlling 'gating_mode': 'LTC' or None)
      - correlation order p    (by controlling 'corr_order')
      - offline vs. time-domain approach (by controlling 'inference_mode': 'convolution', 'parallel_scan', or 'ode')
    """

    def __init__(self,
                 ssm_size,        # e.g. N, dimension of the internal state
                 input_dim,       # dimension of input features
                 output_dim,      # dimension of output features
                 ssm_type='NPLR', # 'NPLR' for S4-like, 'diagonal' for S4D-like
                 gating_mode=None,# e.g. None or 'LTC'
                 corr_order=1,    # e.g. 1 => no correlation kernel, >=2 => Liquid-S4
                 inference_mode='convolution', # or 'parallel_scan', or 'ode'
                 **kwargs):
        """
        Args:
            ssm_size (int): dimension N of the SSM.
            input_dim (int): input channel dimension for this layer.
            output_dim (int): output channel dimension for this layer.
            ssm_type (str): 'NPLR' or 'diagonal' or other future expansions.
            gating_mode (str): e.g. 'LTC' for input-based gating, or None.
            corr_order (int): p, how many auto-correlation orders to consider, if using Liquid approach.
            inference_mode (str): how to apply in forward pass: 'convolution', 'parallel_scan', or 'ode'.
            kwargs: placeholders for e.g. hyperparams like timescale range, rank for low-rank factor, etc.
        """
        self.ssm_size = ssm_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ssm_type = ssm_type
        self.gating_mode = gating_mode
        self.corr_order = corr_order
        self.inference_mode = inference_mode

        # Additional parameters for SSM initialization (e.g. timescale sampling)
        # from kwargs we might parse 'delta_min', 'delta_max', etc.
        self.delta_min = kwargs.get('delta_min', 0.001)
        self.delta_max = kwargs.get('delta_max', 0.1)

        # 1) Build or sample state matrix A, input matrix B, output matrix C, ...
        self._init_ssm_parameters()

        # 2) If gating_mode == 'LTC', define any gating or time-constant parameters
        if self.gating_mode == 'LTC':
            self._init_ltc_gating()  # e.g. small MLP or direct param
        else:
            self.gating_params = None

    def _init_ssm_parameters(self):
        """ Initialize (A, B, C, D, etc.) with normal+low-rank or diagonal approach. """
        if self.ssm_type == 'NPLR':
            # => S4-like approach: we do Normal + Low-rank
            # possibly load the LegS matrix or param from a “Hippo” function.
            self.A_params = self._build_nplr_legS(self.ssm_size)
        elif self.ssm_type == 'diagonal':
            # => S4D-like approach: diagonal
            self.A_params = self._build_diagonal_legS(self.ssm_size)
        else:
            raise NotImplementedError("Unknown ssm_type")

        # B, C, D, etc. can be built similarly (perhaps random or certain init).
        self.B = nn.Parameter(torch.randn(self.ssm_size, self.input_dim))
        self.C = nn.Parameter(torch.randn(self.output_dim, self.ssm_size))
        # Possibly feed-through D or timescales:
        self.log_timescale = nn.Parameter(torch.zeros(self.ssm_size))

    def _init_ltc_gating(self):
        """Define gating parameters for the LTC approach, e.g. a small function f(u). """
        # We might keep it simple by a learnable linear or MLP from input -> gating scales
        # e.g. gating_params = MLP with hidden dim?
        self.gating_mlp = MyTinyMLP(self.input_dim, out_dim=self.ssm_size)
        # or just define param vectors that will be used to do: A + gating
        return

    def _build_nplr_legS(self, N):
        """Placeholder for building a normal+low-rank representation from LegS matrix. """
        # e.g. do the standard LegS procedure: get V, Lambda, P, Q, etc.
        # return them in a dictionary
        return {"V": None, "Lambda": None, "P": None, "Q": None}  # TODO

    def _build_diagonal_legS(self, N):
        """Placeholder for building a diagonal S4D approach. """
        # e.g. we might store just a vector of diag(A)
        return {"diagA": nn.Parameter(torch.zeros(N))}  # TODO

    def forward(self, input_sequence):
        """
        input_sequence: shape [batch_size, seq_len, input_dim]

        We'll route to a different function depending on 'inference_mode'.
        """
        if self.inference_mode == 'convolution':
            return self._forward_convolution(input_sequence)
        elif self.inference_mode == 'parallel_scan':
            return self._forward_parallel(input_sequence)
        elif self.inference_mode == 'ode':
            # For LTC, might step an ODE solver, or do a naive BPTT approach
            return self._forward_ode(input_sequence)
        else:
            raise NotImplementedError("Unknown mode")

    def _forward_convolution(self, x):
        """
        Offline application: we can build the S4 kernel => K_s4,
        then possibly build the 'liquid' kernel => K_liquid,
        convolve them with x, sum up. 
        """
        # 1) Compute standard S4 convolution kernel:
        K_s4 = compute_s4_kernel(self.A_params, self.B, self.C, x.shape[1])  # length = seq_len
        # 2) If corr_order > 1, compute the "liquid" kernel part:
        if self.corr_order > 1:
            K_liquid = compute_liquid_kernel(self.A_params, self.B, self.C, self.corr_order, x.shape[1])
            # Then combine them, e.g. sum or do the correlated convolve
            y = convolve_with_liquid(x, K_s4, K_liquid, corr_order=self.corr_order)
        else:
            y = convolve_basic(x, K_s4)
        return y  # shape [batch_size, seq_len, output_dim]

    def _forward_parallel(self, x):
        """Parallel-scan approach, akin to S5. If gating_mode is LTC, incorporate that. """
        # For example, we do a diagonal approach for 'A', then a prefix-scan style recurrence
        if self.ssm_type == 'diagonal':
            # s5-like approach
            y = s5_parallel_scan_diagonal(self.A_params, x)
        else:
            # or we might do a partial approach with nplr
            y = s5_like_scan_nplr(self.A_params, x)
        # incorporate correlation if corr_order>1 ...
        return y

    def _forward_ode(self, x):
        """
        For LTC gating in a truly continuous sense, we might unroll the ODE solver. 
        This can be slow for large L, but helpful if we want actual time steps that vary.
        """
        # Suppose we do a simple Euler or any standard solver across the seq steps.
        dt = (self.delta_min + self.delta_max)/2  # or adapt
        hidden = torch.zeros(x.size(0), self.ssm_size)  # batch_size x N
        outputs = []
        for t in range(x.shape[1]):
            u_t = x[:, t, :]  # shape [batch_size, input_dim]
            if self.gating_mode == 'LTC':
                # compute gating => modifies A
                gating_vec = self.gating_mlp(u_t)  # e.g. shape [batch_size, N]
                # we do A + B*gating etc.
                hidden = euler_update(hidden, gating_vec, dt)
            else:
                # standard step
                hidden = euler_update(hidden, None, dt)
            out_t = self.C @ hidden.transpose(0,1)  # shape [output_dim, batch_size], then .T
            outputs.append(out_t.transpose(0,1)) 
        y = torch.stack(outputs, dim=1)  # shape [batch_size, seq_len, output_dim]
        return y

    # Additional utility methods omitted for brevity 
    # e.g. log_timescale usage, partial-lrk expansions, memory management, etc.

Notes on This Draft
	1.	Multiple Approach Support
You can see how we define “inference_mode” to let a single code base decide at forward pass time whether to do:
	•	Convolution-based offline (like S4),
	•	Parallel scan (like S5), or
	•	An ODE solver loop (like LTC).
	2.	corr_order for the Liquid Kernel
The function _forward_convolution calls compute_liquid_kernel(...) if corr_order>1. That function would implement the power-of-B approach (PB) or the “kernel × B” approach from the Liquid-S4 paper.
	3.	Parameter Tying
If you want to tie gating matrices across multiple layers, you’d simply reuse or pass references to certain nn.Parameters. Or do that in _init_ltc_gating() for a global gating MLP.
	4.	Simplifications
	•	Real code might use s4_convolution_kernel from the official S4 library.
	•	The LTC “ODE solver” here is just a naive Euler method; you might want higher-order integrators or variable-step integrators to handle time-varying \Delta t.
	5.	Initialization
This draft places _init_ssm_parameters() in the constructor with placeholders. You’d fill in your chosen HiPPO-LegS building block or diagonal approximation, etc.

⸻

Final Remarks

This snippet demonstrates a unified approach for a flexible “StateSpaceBlock” that can do:
	•	S4 or S4D: by toggling ssm_type and the underlying _build_nplr_legS vs _build_diagonal_legS.
	•	(Liquid) LTC gating: by toggling gating_mode='LTC'.
	•	Correlation (corr_order>1 => Liquid-S4).
	•	Offline or Online: by toggling inference_mode='convolution', 'parallel_scan', or 'ode'.

You can similarly expand or reorganize the code. The main takeaway is that the code structure enumerates all the design-time flags from S4, LTC, S5, Liquid-S4, etc., into a single layered library.