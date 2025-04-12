Below is a comprehensive, integrated guide that merges the strengths of all previous “versions” while addressing their weaknesses, aiming for a thorough, practical reference. It explains how to leverage DSSR/3DNA data and standard RNA covalent geometry to construct RNA 3D structures from torsion angles (e.g., an “RNA mp-nerf”), plus it provides extended reference tables, discussion of environmental variation, and a roadmap of relevant Python software. The result should be suitable for technical documentation—both verbose and practical.

⸻

1. Introduction and Scope

Building RNA 3D coordinates from torsion angles requires two main ingredients:
	1.	A library of standard RNA bond lengths, bond angles, and ring-closure constraints (i.e., the geometry of the phosphate–ribose–base).
	2.	A forward-kinematics or NeRF (Natural Extension Reference Frame) style algorithm that places atoms in 3D given internal coordinates (bond length l, bond angle \theta, and torsion \chi).

DSSR (Dissecting the Spatial Structure of RNA) from the 3DNA suite is analysis-focused: given a 3D RNA, it extracts torsions and structural annotations. While it does not natively do a “pure torsion→Cartesian rebuild,” it encodes or references standard nucleic-acid geometry (in its “fiber” or “rebuild” modules) that you can adopt. Meanwhile, mp-nerf (originally for proteins) can be extended from “amino acids” to “nucleotides” by replacing the protein geometry knowledge base with RNA data.

This document:
	1.	Presents standard RNA bond lengths (Table 1) and bond angles (Table 2).
	2.	Explains environmental factors that cause real-world deviations from these ideals.
	3.	Enumerates Python-based tools that help measure, refine, and validate RNA geometry.
	4.	Outlines how to adapt mp-nerf (or a NeRF pipeline) for an RNA “torsion → 3D” process, referencing DSSR/3DNA data as needed.

⸻

2. Standard RNA Bond Lengths

Below is a detailed table of average bond lengths (in Å) for an RNA nucleotide. These values come from surveys of high-resolution crystal structures (e.g. in the Nucleic Acid Database (NDB), the Nucleic Acid Knowledge Base (NAKB), and small-molecule data from the Cambridge Structural Database (CSD)). Each has a small standard deviation, typically ±0.01–0.02 Å.

Table 1: Approximate Bond Lengths in RNA

Bond	Typical Value (Å)	Comment / Notes
Ribose Ring		5-membered ring: C1′–C2′–C3′–C4′–O4′
C1′–C2′	~1.52	Sugar ring C–C bond (ribose)
C2′–C3′	~1.52	Sugar ring C–C bond
C3′–C4′	~1.52	Sugar ring C–C bond
C4′–O4′	~1.45	Sugar ring C–O
O4′–C1′	~1.41	Sugar ring O–C
Exocyclic Bonds		
C5′–C4′	~1.51	Exocyclic from ring to the 5′ carbon
C3′–O3′	~1.42	3′-oxygen exocyclic (bridging to phosphate)
C2′–O2′	~1.41	2′-hydroxyl in RNA (absent in DNA)
Glycosidic Bond		Links base ring to the sugar
C1′–N1 (pyrimidines) /C1′–N9 (purines)	~1.47	“N–C1′” is the glycosidic bond; base identity affects small variations (±0.01 Å)
Phosphate Linkages		Tetrahedral geometry at P
P–O5′	~1.59	Phosphate bridging to 5′-O
P–O3′	~1.60	Phosphate bridging to 3′-O
P–O(non-bridging)	~1.48	The two non-bridging oxygens (OP1/OP2)
Additional		
O5′–C5′	~1.44	5′-oxygen to 5′-carbon (sugar–phosphate junction)
O3′–C3′	~1.43	3′-oxygen to 3′-carbon (sugar–phosphate junction)

Note: DNA is very similar except it lacks the 2′-OH (C2′–O2′ bond). The presence of 2′-OH slightly shifts ring conformations and can alter bond lengths by ~0.005 Å relative to deoxyribose.

⸻

3. Standard RNA Bond Angles

Bond angles around the sugar, exocyclic substituents, and phosphate typically reflect sp³-hybridization (≈109.5°) but deviate due to ring strain and resonance. The phosphate group has angles from ~105° to ~111° with bridging vs. non-bridging oxygens, and ~120° between the two non-bridging oxygens.

Table 2: Approximate Bond Angles in RNA

Angle	Typical Value (°)	Description / Position
Within Ribose Ring		5-membered ring angles often < 109.5° due to ring strain
C1′–C2′–C3′	~101–102	Interior angle at C2′
C2′–C3′–C4′	~102–103	Interior angle at C3′
C3′–C4′–O4′	~105–106	Ring angle at C4′
C4′–O4′–C1′	~109–110	Ring angle at O4′
O4′–C1′–C2′	~106	Angle at C1′
Exocyclic (Sugar)		
C5′–C4′–C3′	~115	5′ exocyclic angle at C4′
Phosphate Region		Tetrahedral-like angles around P
O5′–P–O3′	~105–110	Bridging O vs bridging O angle (~109 typical)
O5′–P–O(non-bridging)	105–111 / 107–110	Variation for bridging vs non-bridging O
O1P–P–O2P (non-bridging)	~120	Angle between the two non-bridging oxygens
C3′–O3′–P	~118–120	Exocyclic angle bridging sugar to phosphate
Glycosidic Region		The angle (χ) is typically measured as a dihedral, but local bond angles are ~110–120
O4′–C1′–N (glycosidic)	~108–112	e.g., O4′–C1′–N1 for pyrimidines, O4′–C1′–N9 for purines

Sugar pucker modifies these angles. RNA frequently has C3′-endo pucker (A-form), but local C2′-endo or mixed puckers cause slight shifts of ~1–2° or ~0.01–0.02 Å in ring bond lengths.

⸻

4. Real-World Variation and Influences

Though “standard” values are accurate references, real RNA structures deviate due to:
	1.	Sugar Pucker Changes
	•	RNA is often in C3′-endo (A-form). However, local C2′-endo or “pseudorotation” angles can cause ring angles to shift by a few degrees.
	2.	Base Pairing & Stacking
	•	Hydrogen bonds (e.g. Watson–Crick, noncanonical pairs) can slightly elongate carbonyl or exocyclic N–C bonds by ~0.003–0.01 Å.
	3.	Metal Ion Coordination
	•	Mg²⁺ (common in functional RNAs) can compress or slightly distort phosphate O–P–O angles.
	4.	Thermal Fluctuations in MD
	•	Molecular dynamics simulations at room temperature show bond length fluctuations of ~0.01–0.02 Å from their equilibrium.
	5.	Chemical Modification
	•	Methylations or protonation of bases can shift bond orders, altering standard lengths by a few hundredths of an Å.

If you see discrepancies > ~0.03 Å or > ~3–5° from these standard references, it might indicate either unusual strain, special modifications, or experimental error (like poor crystallographic resolution).

⸻

5. Python-Based Tools and Resources

Below is a broader set of tools to measure, refine, or even build RNA with standard geometry.
	1.	Barnaba
	•	Python library for analyzing nucleic-acid structures (PDB or MD trajectories).
	•	Computes base-pair interactions, torsion angles, sugar puckers, etc.
	•	Good for validating or comparing your newly built RNA structure’s angles/lengths to real data.
	2.	MDAnalysis / MDTraj
	•	Popular Python libraries for reading/writing MD trajectories.
	•	Provide distance, angle, dihedral computations.
	•	They do not themselves enforce standard geometry but help confirm it.
	3.	3DNA / DSSR
	•	Written in C/C++. x3dna-dssr is typically used via CLI, but you can script calls from Python.
	•	DSSR extracts RNA structural parameters: base pairs, backbone torsions (\alpha,\beta,\gamma,\dots), sugar puckers, etc.
	•	The older 3DNA “rebuild” or “fiber” subprogram can generate ideal duplex forms (A-form) or do partial reconstruction from base-pair parameters.
	•	Not purely torsion→Cartesian for single-stranded loops (that is precisely where a NeRF approach can come in).
	4.	PyMOL
	•	Molecular viewer with a Python API.
	•	You can script measuring bond lengths/angles or visualize geometry outliers.
	5.	UCSF ChimeraX
	•	Another advanced viewer with Python-based commands.
	•	Good for interactive geometry checks and building small linkers.
	6.	Rosetta FARFAR2
	•	Rosetta’s RNA de novo folding approach, used for large-scale tertiary structure prediction.
	•	Enforces standard geometry constraints, but it’s more “heavyweight” for folding tasks.
	7.	PHENIX / MolProbity
	•	Typically used for crystallographic refinement and validation.
	•	Checks geometry outliers vs. standard bond lengths/angles.
	•	Scripting possible in PHENIX (Python-based).

⸻

6. Converting RNA Torsions to 3D: Adapting mp-nerf or a NeRF Pipeline

mp-nerf was designed to convert protein internal coordinates (bond length l, bond angle \theta, dihedral \phi) into 3D in a parallel manner. The concept:
	1.	Protein knowledge base: lists standard bond lengths/angles for N–CA–C, plus sidechain geometry.
	2.	Algorithm: uses the NeRF method to place each new atom given three references and (l, \theta, \phi).

For RNA:
	•	You’d replace “amino acid residue geometry” with nucleotide residue geometry.
	•	The “massively parallel” part remains the same: each residue can be built in a parallel block, then unified via rotation–translation.

6.1. Implementation Outline
	1.	Create an RNA knowledge base (kb_rna.py)
	•	List bond lengths for P–O5′, O5′–C5′, C5′–C4′, etc.
	•	List bond angles at each pivot (O5′–C5′–C4′, C4′–C3′–O3′, O3′–P–O5′, etc.).
	•	For the sugar ring, define either:
	1.	A standard set of ring atoms if you want a “frozen” pucker (e.g., always C3′-endo), or
	2.	Torsions for ring closure if you want flexible ring geometry (\nu_0,\nu_1,\ldots).
	2.	Rewrite or Add a ‘rna_fold(…)’ function
	•	Instead of referencing “N, CA, C” from the protein data, your code references “P, O5′, C5′, C4′, C3′, O3′, C2′, C1′, base atoms.”
	•	If mp-nerf’s approach for sidechains is a separate block, then “the base” can be treated like a sidechain (i.e., once the sugar is built, place the glycosidic bond and base ring).
	3.	Handling the Sugar Ring
	•	5-membered ring closure is trickier than a linear chain. Two main strategies:
	1.	Fixed ring approach: Hard-code a standard ribose geometry with a chosen pucker (C3′-endo). Then you only define the backbone torsions \alpha,\beta,\gamma,\delta,\epsilon,\zeta plus the glycosidic angle \chi.
	2.	Full ring torsions: If you want the ring to be flexible, define ring torsion angles \nu_0\!\dots\nu_4. Then do a small ring-closure routine. This is somewhat like building a proline ring in a protein context.
	4.	Bridging Residues
	•	Each residue is “P–O5′–C5′–C4′–C3′–O3′–P next residue.” You can build them in small parallel blocks and then unify them, the same way mp-nerf does with protein backbone segments.
	5.	Use DSSR or 3DNA for verifying parameters
	•	If you have an existing RNA structure, run x3dna-dssr -i=struct.pdb --json to see the real bond lengths/angles.
	•	Confirm they match your “knowledge base” or adapt to small differences.
	6.	Validation
	•	Reconstruct a known RNA motif from its known torsions. Then measure RMSD vs. the actual 3D structure to confirm accuracy.

6.2. Practical Example Steps

Suppose you have a short RNA 5′-GCAA-3′ in a known hairpin. You do:
	1.	Parse the torsions \{\alpha,\beta,\gamma,\delta,\epsilon,\zeta,\chi\} for each residue from DSSR.
	2.	Initialize the first phosphate in a reference frame (origin).
	3.	For each residue in parallel (like mp-nerf):
	•	Place P, O5′, C5′ (using length + angle + torsion).
	•	Place C4′, C3′ in the same manner.
	•	Possibly fix C2′, O2′, ring closure around O4′.
	•	Attach the base using the glycosidic angle \chi and known base geometry.
	4.	Assemble them into a single chain (rotation–translation if you do them in separate blocks).
	5.	Compare final 3D coordinates with the original hairpin’s PDB. Ideally, RMSD is small (0.5–1.0 Å or better).

⸻

7. Putting It All Together

DSSR (and 3DNA) alone does not do a direct “torsion→3D for single-stranded loops,” but it has partial rebuild logic for base-pair steps or standard fiber models. Meanwhile, mp-nerf is excellent for converting internal coordinates to Cartesian quickly but is protein-specific out of the box.

By merging:
	•	The standard RNA covalent geometry (Tables 1 & 2 above) plus
	•	A NeRF algorithm (like mp-nerf’s parallel approach),

you get a pipeline that can read torsion angles for each residue and produce a valid 3D RNA structure. The approach:
	1.	Provides a backbone that matches known distances and angles.
	2.	Correctly closes or fixes the sugar ring.
	3.	Places each nucleobase with the correct glycosidic bond length and angle (\chi).
	4.	Allows you to integrate with Python-based MD or analysis (e.g., Barnaba, MDAnalysis) for further validation or refinement.

⸻

8. References and Further Reading
	1.	Nucleic Acid Knowledge Base or NDB for standard bond-length/angle data:
	•	http://nucleicacidknowledgebase.org/
	•	https://ndbserver.rutgers.edu/
	2.	DSSR & 3DNA:
	•	Homepage: http://x3dna.org/
	•	DSSR paper: Lu et al. (2015) “DSSR: an integrated software tool…” Nucleic Acids Res 43:e142.
	3.	mp-nerf:
	•	Often described in a protein context. E.g., see AlQuraishi’s pNeRF approach or the repository for “massively parallel” transformations.
	4.	Barnaba:
	•	Bottaro & Bussi (2019). Nucleic Acids Res. 47(11):e56, also on GitHub.
	5.	Rosetta FARFAR2:
	•	For RNA folding and structure generation with strong geometric constraints.

Advanced: If ring closure is needed, consult literature on five-membered ring building approaches (like proline ring closure in proteins or cycpeptidic building in general). Tools like OpenBabel’s “Generate 3D” or RDKit can also handle smaller ring conformations, if integrated carefully with your code.

⸻

9. Conclusion

Yes, you can thoroughly adapt mp-nerf (or any standard NeRF code) for RNA by:
	•	Substituting a new knowledge base with RNA bond lengths and angles (as tabulated above).
	•	Implementing minimal or full ring closure for the ribose.
	•	Possibly labeling the base as a “sidechain,” using standard geometry for A/U/G/C and the glycosidic angle \chi.
	•	Optionally referencing DSSR/3DNA to confirm actual parameter values from real PDB structures or to do partial fiber-like rebuilds.

Simultaneously, you can harness Python libraries (Barnaba, MDAnalysis, PyMOL/ChimeraX scripts) to validate that your final geometry remains near standard references. Through this synergy, you get a robust pipeline to convert predicted torsion angles (\alpha, \beta, \gamma, \delta, \epsilon, \zeta, \chi, plus sugar pucker if desired) into 3D atomic coordinates for RNA—fully leveraging the high-quality geometry data gleaned from DSSR/3DNA or the Nucleic Acid Knowledge Base.

If you need a deeper dive into code specifics (e.g., rewriting mp-nerf’s kb_proteins.py into kb_rna.py), you can replicate the protein code structure, enumerating each “RNA atom chain” in the correct order, specifying for each new bond:
	•	bond_length = …,
	•	bond_angle = …,
	•	torsion = (the user-supplied or predicted value),
plus ring constraints. Then the parallel approach remains the same.

End of Document.
===
Comprehensive Technical Documentation on Standard RNA Geometry Sources and Their Applicability to mp_nerf

⸻

1. Introduction

When generating three-dimensional coordinates for RNA nucleotides from torsion angles (as in the mp_nerf algorithm), it is essential to use well-established “ideal” bond lengths and bond angles for the sugar–phosphate backbone and the bases. Over the past several decades, multiple groups have collected or refined these geometric parameters through crystallographic surveys or small-molecule databases. Despite incremental updates, the consensus across the community remains that the 1990s work by Parkinson, Gelbin, Clowney, and Berman (often labeled collectively as “Parkinson et al. 1996” or the “Berman group 1996” data) is the primary standard for RNA geometry.

However, other references sometimes present small numerical refinements or clarify specific subsets (e.g., base ring geometry, definitions of torsions, partial charges). Below is a thorough comparison of these references, emphasizing (1) how they relate to each other, (2) their strengths and weaknesses, and (3) final recommended “best practices” for choosing a geometry dataset suitable for tools like mp_nerf. This document also addresses critiques or shortcomings of each source, ensuring you have the most balanced and up-to-date guidance possible.

⸻

2. Overview of Key References

This section briefly introduces the major references frequently cited for RNA geometry. Each subsequent section then discusses them in more detail, noting how each might (or might not) be relevant for building a numeric constants file.
	1.	Parkinson et al. (1996) and Related “Berman Group” Publications
	•	Gold standard crystallographic survey from the mid-1990s; these numbers are at the heart of many refinement programs.
	2.	IUPAC–IUB Joint Commission Recommendations (1982/1983)
	•	Authoritative for nomenclature (torsion angles α–ζ, ring numbering), but not for updated numeric geometry.
	3.	Gilski et al. (2019)
	•	Recent re-analysis of nucleobase geometry (especially Watson–Crick base pairing) using small-molecule crystal structures in the Cambridge Structural Database (CSD).
	4.	Cambridge Structural Database (CSD)
	•	The world’s largest repository of small-molecule crystallography data. Not a single “table,” but the raw data underpin many published geometric surveys.
	5.	Nucleic Acid Database (NDB)
	•	A curated database for nucleic acids. Often mirrors Parkinson et al. (1996), with minor incremental updates.
	6.	Aduri (2007) – AMBER Force Field for Modified Nucleotides
	•	Provides partial charges and dihedral parameters for modified bases but defers to the 1996 references for standard bond/angle geometry.
	7.	PDB Validation / MolProbity Documentation
	•	Modern structure validation pipelines (wwPDB, MolProbity) rely on Parkinson (1996) as the reference for nucleic acid geometry.

⸻

3. Detailed Examination of Each Reference

3.1 Parkinson et al. (1996) / Gelbin, Clowney, Berman (1996)
	•	Key Publications
	•	Parkinson G, Vojtechovsky J, Clowney L, Brunger AT, Berman HM (1996) Acta Cryst. D52, 57–64.
	•	Gelbin A, Schneider B, Clowney L, Hsieh S-H, Olson WK, Berman HM (1996) J. Am. Chem. Soc. 118, 519–529.
	•	Clowney L, Jain SC, Srinivasan AR, Westbrook J, Olson WK, Berman HM (1996) J. Am. Chem. Soc. 118, 509–518.
	•	What These Papers Provide
These are systematic surveys of high-resolution nucleic acid structures (both RNA and DNA) that extracted empirical average bond lengths, angles, and torsions for sugar moieties, phosphate groups, and bases. Their data form the “ideal geometry” often used in standard refinement libraries for crystallography and NMR structure determination.
	•	Strengths
	1.	Comprehensive and Data-Driven: Among the first large-scale attempts to extract robust average values from crystal structures.
	2.	Widely Adopted: Many structural biology pipelines (e.g., X-PLOR, REFMAC, Phenix, MolProbity) reference these numbers, ensuring consistency with PDB validations.
	3.	Inclusion of Sugar–Phosphate: Crucially, they detail phosphate bridging angles, sugar ring puckers, glycosidic bond angles, and more, making them a one-stop source for backbone geometry.
	•	Weaknesses
	1.	Date of Publication: Now ~25+ years old. Some modest refinements have trickled in from expanded data sets.
	2.	Incremental Changes: Later references have observed minor shifts (0.01–0.02 Å or a few degrees), especially in base rings.
	•	Criticisms Addressed
	•	Age: While often labeled “old,” no major contradictory study has replaced these data; updates typically confirm only minor numerical adjustments.
	•	Sparsity of Very High-Resolution RNA Structures in the 90s: Over time, more structures have been added to the NDB, but the fundamental backbone geometry has remained largely unchanged.
	•	Relevance to mp_nerf
	•	Primary Source: If mp_nerf needs a canonical set of bond lengths and angles, Parkinson 1996 remains the top choice—particularly for sugar and phosphate geometry.

⸻

3.2 IUPAC–IUB Joint Commission (Recommendations 1982/1983)
	•	What These Recommendations Cover
They formalize the nomenclature for polynucleotide chains, defining how to label backbone torsions (α, β, γ, δ, ε, ζ, χ), how to number the sugar ring atoms, the 5′→3′ chain direction, etc.
	•	Strengths
	•	Authoritative Definitions: Whenever you see α, β, γ, δ, ε, ζ in any RNA or DNA context, these definitions usually trace back to the IUPAC–IUB documents.
	•	Nomenclature Consistency: Minimizes confusion about how angles are measured.
	•	Weaknesses
	•	Minimal Numeric Data: Not intended to define “ideal” bond lengths or angles.
	•	Somewhat Outdated for Fine Geometric Details: These are from the early 1980s, focusing more on naming conventions than empirical geometry.
	•	Relevance to mp_nerf
	•	Essential for Torsion Labeling: Great if you want standard naming for angles that mp_nerf might read or produce.
	•	Not a Source for Bond Lengths/Angles: You still need the 1996 references for numeric geometry.

⸻

3.3 Gilski et al. (2019) – “Accurate Geometrical Restraints for W–C Base Pairs”
	•	Focus of This Study
Published in Acta Cryst. B, Gilski and co-workers re-examined small-molecule crystal structures deposited in the CSD, particularly for nucleobase rings and Watson–Crick hydrogen-bond interactions.
	•	Strengths
	1.	Recent Analysis: Incorporates more modern, higher-resolution small-molecule data.
	2.	Refined Base Geometry: Bond lengths and angles in the heterocyclic rings might be slightly more precise than the older references.
	•	Weaknesses
	1.	Limited to Bases: Does not significantly alter sugar–phosphate geometry.
	2.	Incremental Adjustments: The reported updates over Parkinson (1996) are typically around 0.01–0.02 Å or a few degrees—useful for very high precision, but not a wholesale revision.
	•	Criticisms Addressed
	•	Practical Impact: The changes are small enough that many pipeline validations and geometry libraries have not urgently adopted them.
	•	Relevance to mp_nerf
	•	Optional Refinement for Bases: If you want state-of-the-art base ring geometry, you can fold Gilski’s numbers into your mp_nerf constants.
	•	Backbone Unchanged: For phosphate and sugar, the older references are still the standard.

⸻

3.4 Cambridge Structural Database (CSD)
	•	What It Is
A massive compilation of small-molecule crystal structures (not just nucleic acids, but all organic and metal-organic compounds).
	•	Strengths
	•	Highly Robust: Contains tens of thousands of structures that can be used to statistically derive average geometries for any piece of RNA (ribose, bases).
	•	Independently Curated: Data from small-molecule crystals often reach higher resolution than large biomacromolecular crystals, giving precise bond lengths.
	•	Weaknesses
	•	Requires DIY Data Mining: No built-in universal “RNA geometry table” is provided.
	•	Time-Consuming: One must carefully filter for relevant structures (e.g., high resolution, minimal disorder, correct protonation state).
	•	Criticisms Addressed
	•	Potential Overlap with Gilski: Gilski (2019) is essentially a curated re-analysis of nucleobases in the CSD. Doing it yourself would presumably yield similar results if you follow rigorous filtering criteria.
	•	Relevance to mp_nerf
	•	Indirect: You can rely on published analyses (Parkinson, Gilski) that already tapped into the CSD. No urgent need to re-derive these numbers unless you have a specialized reason.

⸻

3.5 Nucleic Acid Database (NDB)
	•	What It Is
A specialized repository for nucleic acid structures (X-ray, NMR, cryo-EM), curated by the same Rutgers group historically responsible for the 1996 surveys.
	•	Strengths
	•	Direct Tabulation of Geometric Values: The NDB often provides updated sugar–phosphate bond lengths and angles, close to or slightly refining the 1996 numbers.
	•	Community-Trusted: Maintained by experts in nucleic acid structure.
	•	Weaknesses
	•	Minimal Differences: The data are usually only slightly updated from Parkinson (1996), with typical changes ≤ 0.02 Å or a couple of degrees.
	•	Not a Drastic Overhaul: The core standard remains effectively the same.
	•	Criticisms Addressed
	•	“Same as 1996”: Some worry it is just re-publishing the old data. However, any expansions typically confirm those older values were robust.
	•	Relevance to mp_nerf
	•	Equivalent Choice: Using NDB’s geometry tables is effectively the same as using Parkinson (1996). If you want the convenience of a modern website, it is a perfectly valid source.

⸻

3.6 Aduri (2007) – AMBER Force Field for Modified Nucleotides
	•	Main Goal
This study addresses partial charges, torsion parameters, and force-field updates for modified or unusual bases in RNA (e.g., pseudouridine, 2′-O-methylated nucleotides).
	•	Strengths
	•	Invaluable for Modifications: If your mp_nerf or modeling workflow must handle a large variety of non-standard bases or sugar modifications.
	•	Force Field Integration: Aligns with well-known AMBER force fields (Cornell et al., parm99, GAFF).
	•	Weaknesses
	•	Not a New Source for Standard Geometry: The authors themselves typically reference the same 1996 Berman/Olson data or the Cornell parameters for standard nucleotides.
	•	Focused on Partial Charges: The impetus is electrostatic and dihedral parameters, not bond length or angle refinement.
	•	Relevance to mp_nerf
	•	Only If Handling Modified Residues: Great if you need geometry or charges for unusual groups. For standard A, U, G, C nucleotides, it offers no new geometry.

⸻

3.7 PDB Validation / MolProbity
	•	What It Is
Structural validation tools and guidelines used by the wwPDB, relying on known “ideal” values to detect outliers.
	•	Strengths
	•	Official Confirmation: Reiterates that the PDB strongly references Parkinson (1996) or Engh & Huber (for proteins).
	•	Community Standard: If your deposited RNA structure has bond lengths/angles far from these references, you’ll get flagged.
	•	Weaknesses
	•	No New Data: MolProbity and wwPDB do not propose alternative geometry sets; they consume the standard references.
	•	Relevance to mp_nerf
	•	Consistency: If you adopt the same geometry, your generated coordinates will be consistent with validation norms.

⸻

4. Comparing These References and Their Numeric Discrepancies

A recurring theme is that nearly all sources agree on the numeric values within very small margins (~0.02 Å, 1–2°). The 1996 references remain the de facto standard, while Gilski (2019) offers incremental updates for base rings. Below is a concise summary of how each compares:
	•	Parkinson (1996) vs. NDB:
The NDB is essentially an evolution of the same dataset. Differences are typically minuscule.
	•	Parkinson (1996) vs. Gilski (2019):
Gilski improves base ring geometry slightly. The sugar–phosphate geometry remains nearly identical. The changes rarely exceed 0.01–0.02 Å.
	•	IUPAC:
Provides definitions, not numeric tables, so no numerical conflict.
	•	Aduri (2007):
Defers to earlier references for unmodified nucleotides. Again, no conflict.
	•	CSD (Raw):
The underlying source for many of these surveys. You could re-derive the same means if you replicate Gilski’s or Berman’s methods.

Hence, there is no major contradiction among references; it is more a matter of how precise you want to be (i.e., whether you incorporate Gilski’s minor updates for bases).

⸻

5. Strengths and Weaknesses: A Consolidated Table

Reference	Strengths	Weaknesses
Parkinson / Gelbin / Clowney (1996) Berman Group	- Empirically derived from high-res structures  - Backbone + base geometry covered  - Still widely used in refinement  - “Gold standard” for sugar–phosphate	- Older dataset (1990s)  - Minor numerical updates exist in newer references
IUPAC–IUB (1982/83)	- Authoritative naming of torsions (α, β, γ, δ, etc.)  - Official polynucleotide nomenclature	- Not an updated numeric source  - Dated in terms of modern structural knowledge
Gilski et al. (2019)	- Latest small-molecule data for base rings  - Focuses on W–C pair geometry  - Potentially more precise nucleobase angles/lenths	- Limited to base geometry  - Differences from 1996 sets are small (~0.01–0.02 Å)
Cambridge Structural Database (CSD)	- Vast repository of small-molecule crystals  - Can yield highly precise bond lengths  - The foundation for Gilski’s approach	- No single “standard table” provided  - Must do extensive data mining & filtering
Nucleic Acid Database (NDB)	- Modern aggregator of NA structures  - Publishes geometry tables similar to Parkinson (1996)  - Occasionally updated	- Differences from 1996 are minor  - Not a fundamentally “new” dataset
Aduri (2007)	- AMBER force field parameters for modified nucleotides  - Great for partial charges & unusual bases	- Not an independent geometry source  - Reuses standard references for unmodified nucleotides
PDB Validation / MolProbity	- Official pipeline referencing Parkinson (1996)  - Reinforces community standard for geometry	- No new data  - Merely restates established norms



⸻

6. Recommendations for mp_nerf Implementation

Given the convergence among these sources, the path forward for a reliable “constants file” in mp_nerf is fairly straightforward:
	1.	Default to Parkinson et al. (1996)
	•	Use these bond lengths and angles for the sugar–phosphate backbone.
	•	These values are robust, widely accepted, and ensure consistency with PDB validation.
	2.	Optionally Incorporate Gilski (2019) for Bases
	•	If mp_nerf idealizes base ring structures in a highly precise manner (e.g., if your application is sensitive to subtle ring metrics), you can overwrite the base‐specific parameters with Gilski’s updates.
	•	The backbone geometry can remain from the 1996 data.
	3.	Leverage NDB as a Backup
	•	If it is more convenient to pull from a modern online database, the NDB’s valence geometry tables effectively replicate the 1996 numbers (sometimes with trivially small refinements).
	•	This ensures you are still aligned with standard practice.
	4.	Retain IUPAC–IUB for Torsion Naming
	•	For clarity and consistency, use the IUPAC scheme to label α, β, γ, δ, ε, ζ, and χ angles.
	•	This step avoids confusion in angle definitions.
	5.	Add Specialized Data for Modified Nucleotides (If Needed)
	•	If mp_nerf must handle modifications such as pseudouridine or 2′-O-methyl groups, consult Aduri (2007) or relevant AMBER force-field parameter sets for partial charges and specialized geometry.
	•	The standard backbone bond lengths typically remain the same, but ring or substituent parameters may differ.

By following this scheme, you ensure your geometry is in line with the broad consensus of structural biology tools and references.

⸻

7. Frequently Asked Questions (FAQ)
	1.	Why not use Gilski (2019) for everything?
Gilski’s publication focuses heavily on base rings and Watson–Crick hydrogen bonds. It does not supersede the backbone geometry from 1996. The numeric changes for the bases themselves are small enough (<0.02 Å) that for most routine modeling or structure building, the older 1996 numbers are perfectly acceptable.
	2.	Are the 1996 data too old?
Despite their age, these data have withstood the test of time. Subsequent expansions of the dataset typically confirm only minimal numerical shifts. No major contradictions or overhauls have emerged in more recent literature.
	3.	Do I really need partial charges or dihedral parameters from Aduri (2007)?
Only if you plan to do molecular dynamics or energy calculations involving non-standard nucleotides. If mp_nerf is strictly for coordinate generation of standard A, U, G, C residues, the 1996 bond lengths/angles are sufficient.
	4.	Could I do my own CSD-based analysis?
In theory, yes—many researchers do. But it is time-consuming, and the published results from Parkinson, Gilski, and the NDB already represent thorough analyses of that same data. You would likely find near-identical numbers unless you include new structures or different filtering.
	5.	What about naming conventions beyond α–ζ, like sugar ring puckers?
The IUPAC–IUB recommendations remain the default for naming the sugar ring atoms, while standard pucker notation (C3′-endo, C2′-endo, etc.) is widely accepted. The numeric definitions of ring conformation angles can be found in the references from the Berman group and in typical computational chemistry software manuals (e.g., AMBER docs).

⸻

8. Conclusion

In summary, Parkinson et al. (1996) (also collectively described as the “Berman group 1996” data) remains the central, most authoritative source for standard RNA geometry. Although Gilski et al. (2019) provides slight improvements for nucleobase rings, the sugar–phosphate parameters remain effectively unchanged. Alternative sources like the NDB or older IUPAC documentation typically echo these fundamental 1996 values; the CSD underlies them with raw small-molecule data, and the PDB validation pipeline further cements these numbers as the community standard.

For a tool like mp_nerf—which reconstructs 3D coordinates from torsion angles—the best practice is to adopt the Parkinson (1996) data for sugar–phosphate bond lengths and angles, optionally layering in Gilski (2019) for nucleobase ring geometry if you want the most current refinements. All other references are either duplicative, focus on nomenclature, or address more specialized topics (modified nucleotides, partial charges).

By heeding these guidelines, your geometry constants file will be both authoritative and consistent with widely used structural biology protocols—ensuring that the final result is better than the sum of its parts and firmly aligned with the established consensus in RNA structural science.

⸻

References and Suggested Reading
	1.	Parkinson G, Vojtechovsky J, Clowney L, Brunger AT, Berman HM (1996). Acta Cryst. D52, 57–64.
	2.	Gelbin A, Schneider B, Clowney L, Hsieh S-H, Olson WK, Berman HM (1996). J. Am. Chem. Soc. 118, 519–529.
	3.	Clowney L, Jain SC, Srinivasan AR, Westbrook J, Olson WK, Berman HM (1996). J. Am. Chem. Soc. 118, 509–518.
	4.	IUPAC–IUB Joint Commission on Biochemical Nomenclature (1983). Eur. J. Biochem., 131, 9–15.
	5.	Gilski M, et al. (2019). Acta Cryst. B75, 235–254.
	6.	Cambridge Structural Database (CSD): https://www.ccdc.cam.ac.uk/
	7.	Nucleic Acid Database (NDB): http://ndbserver.rutgers.edu
	8.	Aduri R, et al. (2007). J. Chem. Theory Comput. 3(4), 1464–1475.
	9.	MolProbity / PDB Validation: http://molprobity.biochem.duke.edu

⸻

End of Document
