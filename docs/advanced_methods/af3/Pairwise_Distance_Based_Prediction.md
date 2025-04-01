🧙🏾‍: “Here’s an overview of how AlphaFold 3’s 48-block Pairformer stack works—covering the triangular updates and the single-representation-with-pair-bias piece—plus how it all supports ‘pairwise distance’ prediction.”

[emoji]: Detailed Explanation

Below is a step-by-step look at the “Pairformer” trunk that AF3 uses after the MSA Module. Each of the 48 Pairformer blocks transforms an (N \times N) pair-representation array (\{z_{ij}\}) plus a single-representation vector for each token (\{s_i\}). In total:
	1.	Pair Representation:
	•	An (N \times N)-shaped representation where each entry \mathbf{z}_{ij} (the “pair embedding”) encodes how token i (residue or ligand atom) relates to token j.
	•	These pair embeddings are continuously refined with “triangle updates” (both multiplicative and attention-based).
	2.	Single Representation:
	•	A length-N array \{\mathbf{s}_i\} whose elements store information about each token in isolation (e.g., summarizing the local environment, MSA context, etc.).
	•	AF3 processes that single representation side-by-side with the pair representation, letting them exchange information.

⸻

(A) Triangular Multiplication (Outgoing & Incoming)

Each block begins with two multiplicative updates on the pair array:
	1.	TriangleMultiplicationOutgoing (\text{“outgoing” edges})
	2.	TriangleMultiplicationIncoming (\text{“incoming” edges})

Conceptually, the pair embedding \mathbf{z}_{ij} is viewed as an “edge” connecting node i and node j. The “triangle” part means each edge can be updated by looking at a shared vertex k.
	•	Outgoing means: “Update \mathbf{z}_{ij} by considering edges from i to k and j to k.”
	•	Incoming means: “Update \mathbf{z}_{ij} by considering edges from k to i and k to j.”

Each such update is a learned transformation that ensures geometric consistency among triplets (i,j,k). In simpler terms, it fuses information from the ‘third node’ k, telling \mathbf{z}_{ij} how i\rightarrow k and j\rightarrow k compare or multiply.

⸻

(B) Triangular Self-Attention (Starting & Ending Node)

Next come two specialized attention layers:
	1.	TriangleAttentionStartingNode
	2.	TriangleAttentionEndingNode

They again treat \mathbf{z}{ij} like edges in a complete graph. One triangular attention uses i as the “focal” node, scanning across possible k to update \mathbf{z}{ij}. The other uses j as the focus.
	•	“Starting node” means each edge (i \to j) pays attention to edges (i \to k).
	•	“Ending node” means the edge (i \to j) pays attention to edges (k \to j).

Hence it is a form of pairwise attention that ensures each pair embedding \mathbf{z}_{ij} is consistent with adjacency edges sharing one vertex. This further enforces consistent “triangle geometry.”

⸻

(C) Single Representation Update via Pair-Bias Attention

After the triangular updates, the model updates the single representation \mathbf{s}_i. In AF3, each token’s single vector \mathbf{s}i does a self-attention step in which the attention logits are “biased” by the pair representation \mathbf{z}{ij}.
	•	Concretely, we run an attention over \{\mathbf{s}i\} with something like \text{logit}(i,j) \,+=\, \text{Linear}(\mathbf{z}{ij}).
	•	This means: if pair (i,j) is known (via \mathbf{z}_{ij}) to be highly interacting or geometrically close, the attention between single vectors \mathbf{s}_i and \mathbf{s}_j is boosted.
	•	The end result is that single vectors incorporate pairwise knowledge.
	•	After attention, we pass \{\mathbf{s}_i\} through a small transition (feed-forward) block.

Hence each Pairformer block effectively does:
	1.	Triangular multiplication (outgoing + incoming) on \mathbf{z}_{ij}
	2.	Triangular attention (starting + ending node) on \mathbf{z}_{ij}
	3.	A “transition” on \mathbf{z}_{ij}
	4.	Single attention with pair bias on \mathbf{s}_i
	5.	A “transition” on \mathbf{s}_i

⸻

(D) Ties to ‘Pairwise Distance’–Based Prediction

Because each \mathbf{z}{ij} is a learned representation of how token i and j relate, it is naturally well-suited for predicting distances between them. Indeed, AF3’s final distogram head uses the last \{\mathbf{z}{ij}\} to produce a binned distribution of distances.
	•	The triangular updates effectively ensure that any triplet (i,j,k) sees consistent constraints. This is crucial for capturing geometry.
	•	Unlike a simple binary contact map, the pair representation is a continuous, multi-channel embedding that eventually yields a distribution over distances (the “distogram”)—thus under-the-hood, it is still a “pairwise distance” predictor.

Hence, the 48-block Pairformer architecture is the engine for learning geometry from pairwise tokens, letting the model resolve local/long-range distances. Finally, the Distogram head—reading from \{\mathbf{z}_{ij}\}—produces the discrete distance bins that underlie AF3’s structural accuracy.

Would you like more on how the diffusion module consumes these pair embeddings to generate coordinates?