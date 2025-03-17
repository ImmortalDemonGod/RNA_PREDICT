FULL PAPER:
Structural bioinformatics
RNA-TorsionBERT: leveraging language models for RNA
3D torsion angles prediction
Cl�ement Bernard 1,2, Guillaume Postic 1, Sahar Ghannay 2, Fariza Tahi 1,�
1Universit�e Paris Saclay, Univ Evry, IBISC, Evry-Courcouronnes 91020, France
2LISN—CNRS/Universit�e Paris-Saclay, Orsay 91400, France
�Corresponding author. Universit�e Paris Saclay, Univ Evry, IBISC, Evry-Courcouronnes 91020, France. E-mail: fariza.tahi@univ-evry.fr
Associate Editor: Jianlin Cheng
Abstract
Motivation: Predicting the 3D structure of RNA is an ongoing challenge that has yet to be completely addressed despite continuous advance-
ments. RNA 3D structures rely on distances between residues and base interactions but also backbone torsional angles. Knowing the torsional
angles for each residue could help reconstruct its global folding, which is what we tackle in this work. This paper presents a novel approach for
directly predicting RNA torsional angles from raw sequence data. Our method draws inspiration from the successful application of language
models in various domains and adapts them to RNA.
Results: We have developed a language-based model, RNA-TorsionBERT, incorporating better sequential interactions for predicting RNA
torsional and pseudo-torsional angles from the sequence only. Through extensive benchmarking, we demonstrate that our method improves
the prediction of torsional angles compared to state-of-the-art methods. In addition, by using our predictive model, we have inferred a torsion
angle-dependent scoring function, called TB-MCQ, that replaces the true reference angles by our model prediction. We show that it accurately
evaluates the quality of near-native predicted structures, in terms of RNA backbone torsion angle values. Our work demonstrates promising
results, suggesting the potential utility of language models in advancing RNA 3D structure prediction.
Availability and implementation: Source code is freely available on the EvryRNA platform: https://evryrna.ibisc.univ-evry.fr/evryrna/RNA-
TorsionBERT.
1 Introduction
RNA is a macromolecule that plays various biological func-
tions in organisms. Similarly to proteins, the biological func-
tion of an RNA may be directly linked to its 3D structure.
Experimental methods such as NMR, X-ray crystallography,
or cryo-EM can determine the 3D structure of RNAs, but
they remain tedious in cost and time. Computational meth-
ods have been developed for predicting the 3D structure from
the sequence, with three different approaches: ab initio,
template-based, and deep learning-based (Bernard et al.
2024c). Currently, no existing method matches the perfor-
mance of AlphaFold 2 for proteins (Jumper et al. 2021), as
shown with the last results on the CASP-RNA challenge (Das
et al. 2023). Reaching AlphaFold’s (Jumper et al. 2021) level
of accuracy is a long shot, notably due to the lack of data
(Schneider et al. 2023). Very recently, the release of
AlphaFold 3 (Abramson et al. 2024) has extended its predic-
tions to a wide range of molecules like DNA, ligand, ion, and
RNA, but the results remain limited for RNA (Abramson
et al. 2024, Bernard et al. 2024a).
RNA can adopt various secondary motifs, along with a
wide range of complex interactions that contribute to its 3D
structure. Research efforts have focused on classifying both
the canonical and noncanonical pairs, further supported by
the description of the backbone conformation (Schneider
et al. 2004). Unlike proteins, RNA backbone structures are
defined by eight torsional angles, the natural manifold of all
these dihedral angles combined being a 8D hypertorus, which
presents a significant challenge both statistically and compu-
tationally. Pyle and colleagues have shown that they can be
approximated by two pseudo-torsional angles (see Fig. 1)
(Wadley et al. 2007). Understanding these torsional angles is
crucial for comprehending the 3D structures of RNA, which
in turn could aid in predicting their folding.
Current predictive methods for RNA 3D structure predic-
tion do not always integrate torsional angles, missing impor-
tant features to comprehend its folding. One work (Zok et al.
2015) has focused on constructing libraries of RNA conform-
ers with torsional angles. It has been used for RNAfitme
(Antczak et al. 2018), which allows editing and refining pre-
dicted RNA 3D structures. Another work has been done to
predict exclusively torsional angles from RNA sequence,
SPOT-RNA-1D (Singh et al. 2021), using a residual convolu-
tional neural network.
In this work, we aim to leverage language models to better
apprehend the prediction of RNA torsional angles from its se-
quence. Indeed, works have been proposed through the years
to work on biological sequences, inspired by the success of
language models like BERT (Devlin et al. 2018). Its adapta-
tion for RNA (Akiyama and Sakakibara 2022) or DNA
(Ji et al. 2021) shows promising results which could be lever-
aged for other RNA structural features prediction.
Received: 8 July 2024; Revised: 11 December 2024; Editorial Decision: 30 December 2024; Accepted: 7 January 2025
© The Author(s) 2025. Published by Oxford University Press.
This is an Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/), which
permits unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited.
Bioinformatics, 2025, 41(1), btaf004
https://doi.org/10.1093/bioinformatics/btaf004
Advance Access Publication Date: 8 January 2025
Original Paper
Downloaded from https://academic.oup.com/bioinformatics/article/41/1/btaf004/7945663 by Ripon College Library user on 17 March 2025
Another interest in torsional angles for RNA 3D structures
is for quality assessment. Without the help of reference
structures, scoring functions have been developed to assess
structural quality. These methods can be knowledge-based
(Capriotti et al. 2011, Tan et al. 2022) using statistical
potentials or deep learning (Townshend et al. 2021). The
knowledge-based scoring functions consider RNA structural
features as inputs like pairwise distances (Capriotti et al.
2011, Bottaro et al. 2014) or with the help of torsional angles
(Tan et al. 2022). We propose here a new scoring function
based on the extension of our model to predict RNA
torsional angles. This scoring function allows us to assess
structural quality in torsional angle space.
This article is organized as follows: we describe our
contributions in two separate points. Each section is divided
into two parts: one for the work on torsional angle prediction
and the other for our proposed scoring function. We detail
our experiments for the torsional angles prediction and the
structural quality assessment tasks before discussing our
approaches’ results and limitations. We then conclude by
discussing the scope of our contributions. The results and
the code of our RNA-TorsionBERT and TB-MCQ are easily
reproducible and freely available on the EvryRNA platform:
https://evryrna.ibisc.univ-evry.fr/evryrna/RNA-TorsionBERT.
2 Materials and methods
This section presents our model for predicting RNA torsional
angles and then the scoring function derived from our model.
2.1 Torsional angles prediction
2.1.1 RNA-TorsionBERT approach
Current methods that use sequence as inputs for RNA-related
approaches only represent sequences as one-hot-encoding
vectors. This representation may be too sparse to consider se-
quential interactions well. This encoding is usually associated
with a convolutional neural network, which is commonly
limited by long-range interactions. A solution could be using
attention mechanisms. Attention-based architecture nonethe-
less requires a huge amount of data to train well, which is not
the case for RNA 3D structure data. To counter this problem,
we can use models pre-trained on a large amount of unla-
beled data. This could bring a better input representation of
the raw sequence, which could then be fine-tuned to specific
tasks. These pre-trained models could input either RNA or
DNA sequences.
Recent advances in language models started with BERT
(Devlin et al. 2018), where the model was pre-trained on
masking and next-sentence prediction tasks before being fine-
tuned on diverse specific language tasks. DNA or RNA can
be seen as a sequence of nucleotides, where their interactions
have a biological meaning. Therefore, methods have been
adapted from BERT to develop a language-based architecture
for either RNA or DNA. The aim is to reproduce the success
of language comprehension for another language. As the size
of the vocabulary is different, modifications should be made
to fit the current language. An example for DNA is
DNABERT (Ji et al. 2021), where the training process was
updated compared to the original BERT by removing the
next sentence prediction and taking K-mers as inputs (contig-
uous sequence of k nucleotides). It was trained on human-
genome data. An example of the adaptation of BERT for
RNA is called RNABERT (Akiyama and Sakakibara 2022).
It is a six-transformer layer pre-trained on two tasks: masking
and structural alignment learning (SAL). RNABERT was
trained on 76 237 human-derived small ncRNAs. Other
methods have been adapted to RNA language but uses MSA
as inputs like RNA-MSM (Zhang et al. 2024). Nonetheless,
they require multiple sequence alignment (MSA) as inputs,
which restricts the use for RNAs. Indeed, there are a numer-
ous amount of unknown structures (Kalvari et al. 2018), and
MSA will restrict the adaptation to future unseen families. In
this article, we decided to only consider sequences as inputs,
and so for the language models.
The aim of our method is, given a pre-trained language
model (DNABERT or RNABERT), to adapt its neuronal
weights to predict RNA torsional and pseudo-torsional
angles from the sequence. We have added layers to adapt the
methods to our multi-token label regression task. Each token
in the input would have 28 labels: two values (sine and co-
sine) for each of the eight torsional angles (the phase P being
represented by its five ribose ring angles) and two pseudo-
torsional angles. The use of pre-trained embedding would
help the model not to start from scratch and update the
learned attention layers for RNA structural features.
2.1.2 RNA-TorsionBERT architecture
The architecture of our method, when based on DNABERT,
is described in Fig. 2 (illustrated with 3-mers). An input se-
quence of size L is tokenized and then fed to the network
with token and positional embeddings. The tokenization pro-
cess usually adds specific tokens (like the CLS and PAD
tokens). As DNABERT could input a maximum of 512
tokens, we set the maximum sequence length to 512 nucleoti-
des. The last hidden state is set to be 768 by the original
DNABERT architecture. We then apply extra layers to map
the hidden state outputs to the desired final output dimension
(Lx28). These extra layers comprise layer normalization, a
linear layer (from 768 to 1024), a GELU activation, another
Figure 1. The eight RNA torsional angles and the two pseudo-torsional
angles. (A) RNA backbone torsional angles. The angles are defined around
O30i − 1=Pi =O50i =C50i (α), Pi =O50i =C50i =C40i (β), O50i =C50i =C40i =C30i (γ),
C50i =C40i =C30i =O30i (δ), C40i =C30i =O30i =Pi þ 1 (ϵ), C30i =O30i =Pi þ 1=O50i þ 1
(f) and the rotation of the base relative to the sugar (χ) O40i =C10i =N1i =C2i
for pyrimidines and O40i =C10i =N9i =C4i for purines. The ribose ring angles
are defined as ν0 (C40i =O40i =C10i =C20i ), ν1 (O40i =C10i =C20i =C30i ), ν2
(C10i =C20i =C30i =C40i ), ν3 (C20i =C30i =C40i =O40i ), and ν4
(C30i =C40i =O40i =C10i ). The ribose ring is usually described by a single
sugar pucker pseudorotation phase P ¼ arctan ν1 þ ν4 − ν0 − ν3
2ν2 ðsin 36� þ sin 72� Þ
� �. (B) RNA
pseudo-torsional angles. η is defined around C40i − 1=Pi =C40i =Pi þ 1 and θ
around Pi =C40i =Pi þ 1=C40i þ 1.
2 Bernard et al.
Downloaded from https://academic.oup.com/bioinformatics/article/41/1/btaf004/7945663 by Ripon College Library user on 17 March 2025
linear layer (1024 to 28), and a Tanh final activation. The fi-
nal output layer is of size 28 because it outputs a sine and a
cosine for the eight torsional (α, β, γ, δ, ϵ, f, χ, and the phase
P being predicted through the five ribose ring angles ν0, ν1,
ν2, ν3, and ν4) and two pseudo-torsional angles (η and θ).
It allows the relief of the periodicity of the different angles.
The Tanh activation maps the outputs to the cosine and
sine range (between −1 and þ1), which is then converted
into angle predictions using the formula α ¼ tan− 1 sinðαÞ
cosðαÞ
� �
(adaptable for the other angles). Details on the training
process are in the Supplementary File.
2.2 Model quality assessment based on
torsional angles
2.2.1 Torsional-based quality assessment metrics
Existing metrics have been developed to assess the quality of
predicted RNA 3D structures with access to a reference. The
most famous one is the RMSD (root-mean-square deviation),
which assesses the general folding of structures. Other met-
rics have been developed and adapted from proteins (Zhang
and Skolnick 2004, Mariani et al. 2013). Some specific met-
rics have also been designed to consider RNA specificities
(Parisien et al. 2009). Only two metrics are torsional-angles-
based: the MCQ (Zok et al. 2014) (mean of circular quanti-
ties), and the Longest Continuous Segment in Torsion Angle
space (LCS-TA) (Wiedemann et al. 2017). The MCQ com-
putes the deviation in angle space without any superposition
of structures and complements other existing metrics. LCS-
TA computes the longest number of continuous residues with
an MCQ below a threshold (usually 10�, 15�, 20�, and 25�).
It is also a superposition-independent metric.
In SPOT-RNA-1D (Singh et al. 2021), the authors intro-
duced the mean-average error (MAE) metric to assess the per-
formance of their method SPOT-RNA-1D in the prediction
of torsional and pseudo-torsional angles. Nonetheless, the
MAE is an arithmetic mean and is not designed for angles.
To compute deviation for circular quantities, we use the
mean of circular quantities (MCQ) (Zok et al. 2014). We do
not consider the LCS-TA as it is more expensive to compute,
and the MCQ is more widely used in RNA-Puzzles (Cruz
et al. 2012; Miao et al. 2015, 2017, 2020). We define the set
of angles for the torsional angles as T ¼ fα; β; γ; δ; ϵ; f; P; χg
and for pseudo-torsional angles PT ¼ fη; θg. Following the
notation in (Zok et al. 2014), for a given structure S of L resi-
dues, let’s note ti;j the torsional angle of type j of the residue
at position i. We denote the difference between two structures
S and S0 as the MCQ(S, S’), defined by:
MCQðS; S0Þ ¼ arctan
Pr
i¼1
PjTj
j¼1 sin Δðti;j; t0i;jÞ
Pr
i¼1
PjTj
j¼1 cos Δðti;j; t0i;jÞ
0
@
1
A
where r is the number of residues in S \ S0 and with:
Δðt; t0Þ ¼
0 if both undefined;
π if either t or t0 is undefined;
minfdiffðt; t0Þ; 2π − diffðt; t0Þg
8
><
>:
and:
diffðt; t0Þ ¼ jmodðtÞ − modðt0Þj
The difference aims to consider periodicity of 2π with
modðtÞ ¼ ðt þ 2πÞ modulo 2π
To have more details on the performances for a specific angle,
we define the MCQ for a specific type of angle j:
MCQðjÞðS; S0Þ ¼ arctan
Pr
i¼1 sin Δðti;j; t0i;jÞ
Pr
i¼1 cos Δðti;j; t0i;jÞ
!
We extend the formulation for pseudo-torsion angles by
just changing the set of angles used, and we name it MCQPT.
2.2.2 Quality assessment scoring functions
Quality assessment of RNA 3D structures requires two struc-
tures, with one being the experimentally solved structure.
Having this reference is a strong asset that is hardly possible
in practice. To rank near-native structures without a refer-
ence, scoring functions have been developed, adapting
free-energy (Capriotti et al. 2011, Tan et al. 2022). Other
methods use deep learning approaches like ARES
(Townshend et al. 2021).
Figure 2. Schema of the proposed language model architecture for torsional and pseudo-torsional angles prediction. Given an RNA sequence, mapping is
applied to each sequence’s nucleotide into a token with embeddings from the language model. CLS and PAD tokens are added to the sequence tokens.
We convert the Uracil (U) with its equivalent in DNA: Thymine (T) for DNABERT model. Then, the language model will output hidden states with a
representation of each token. This is fed into extra layers before entering a last Tanh activation to have the cosine and sine per angle. A postprocessing is
required to convert back to angles [and pseudo-torsional (PT) angles] from the sine and cosine.
RNA-TorsionBERT 3
Downloaded from https://academic.oup.com/bioinformatics/article/41/1/btaf004/7945663 by Ripon College Library user on 17 March 2025
To discriminate near-native structures in the torsional
space, we have derived a scoring function from our RNA-
TorsionBERT model. First, we have replicated a quality as-
sessment metric that uses torsional angles features: the mean
of circular quantities (MCQ) (Zok et al. 2014). Then, we
replaced the true torsional angles with the predicted angles
from our model to compute the MCQ over the near-native
structure. Therefore, the MCQ computation compares the
prediction of our model angles with the angles from the pre-
dicted nonnative structures. This MCQ now becomes a scor-
ing function, as it only takes as input a structure without any
known native structure. We named this scoring function TB-
MCQ for TorsionBERT-MCQ. Figure 3 shows the architec-
ture of TB-MCQ. Given a structure, we extract the torsional
angles and the sequence. The sequence is then pre-processed
by RNA-TorsionBERT, and the inference gives predicted
angles. Then, we compute the MCQ to finally output a struc-
tural quality measure for an RNA 3D structure.
3 Results and discussion
3.1 Results on torsional angles prediction
We present here the different experiments for the torsional
angles prediction task. We used the MCQ presented above as
a criterion to assess our model performances. We mainly fo-
cus on the results for torsional angles, while the results of
MCQPT for pseudo-torsional angles are available in the
Supplementary File.
3.1.1 Datasets
To validate the performances of torsional angle prediction
models, we used different datasets of native structures:
Training: we downloaded each available PDB structure
and removed the structures from the nonredundant
Validation and Test sets presented below. We also ensure the
structures from this dataset have a sequence similarity below
80% compared to the other used datasets. We considered
only the structures of a maximum sequence length of 512
(DNABERT can only input 512 tokens). The final set is com-
posed of 4267 structures with sequences from 11 to 508
nucleotides.
Validation: we used the validation structures from SPOT-
RNA-1D (Singh et al. 2021). It contains 29 structures with
sequences between 33 and 288 nucleotides.
Test: we combined two well-known test sets: RNA-Puzzles
(Cruz et al. 2012) and CASP-RNA (Das et al. 2023). We
combined both of these datasets as a whole Test set to assess
the robustness of our model. It leads to a Test set of 34 struc-
tures (22 from single-stranded RNA of RNA-Puzzles and 12
from CASP-RNA), with sequences from 27 nucleotides to
512 [we cropped the RNA of PDB ID R1138 (720 nt) to 512
nucleotides].
The distribution of the eight torsional angles and the two
pseudo-torsional angles is given in Fig. 4. As the pseudorota-
tion phase P is defined with the five ribose ring angles, their
distributions are shown in Supplementary Fig. S1 of the
Supplementary File. These distributions are similar for
the three datasets, meaning the learned distribution from the
training set could allow good generalization for the model.
3.1.2 Language model selection
Each language model has a different format of inputs (K-
mers for DNABERT and single nucleotides for RNABERT),
we had to select the best tokenization of our RNA sequences.
We also had to decide which pre-trained model was the best
for our task. Therefore, we trained the same DNABERT
model with the different values of K (3, 4, 5, or 6) and
RNABERT on the Training set and observed the performan-
ces on the Validation set.
The results are shown in Table 1 on the Validation set.
In terms of K-mers, DNABERT trained on 3-mers has
better results (MCQ of 19.0) than the other K-mers and
RNABERT, even if it does not outperform them for each tor-
sional angle. The results for the pseudorotation phase P show
that the model does not change the prediction for this angle.
RNABERT only outperforms the other methods for the P
angle, which does not lead to any significant conclusion
for the selection of this model. We observe that for some
angles (β, P, and χ), the choice of models does not have an
impact on the performances. DNABERT with 3-mers outper-
forms RNABERT, which does not input K-mers. This result
remains surprising as we could have thought that
RNABERT, as pre-trained specifically on RNA data, could
Figure 3. TB-MCQ scoring function computation. Given a predicted RNA 3D structure, we extract the sequence and calculates the different torsional
angles. The sequence is fed to our RNA-TorsionBERT model to predict torsional angles. The scoring function is computed by taking the MCQ between
the predicted angles and the real angles.
4 Bernard et al.
Downloaded from https://academic.oup.com/bioinformatics/article/41/1/btaf004/7945663 by Ripon College Library user on 17 March 2025
have done better than the DNABERT model. This difference
might be explained by the K-mers representation that is used
by DNABERT compared to RNABERT, where the size of the
vocabulary is extended, and thus a finer representation of the
inputs is embedded. This could help the model learn a higher
number of interactions and be more adaptable for other
tasks. What could also explain the difference in performances
is the size of the model: DNABERT has a size of around
328MB, whereas RNABERT has around 2MB. From now
on, we name RNA-TorsionBERT (for RNA torsional BERT)
the DNABERT with 3-mers.
3.1.3 Performances
We present here the prediction results obtained by our
method RNA-TorsionBERT on the Test set (presented above)
compared to the state-of-the-art approach SPOT-RNA-1D
(Singh et al. 2021), which is the only method that predicts
RNA torsional angles from the sequence only. We repro-
duced the architecture of SPOT-RNA-1D (because we only
had the code to do inference) and trained it with the exact
same data as RNA-TorsionBERT. We also included the
results for the inferred angles from methods benchmarked in
State-of-the-RNArt (Bernard et al. 2024c). The methods
benchmarked included either ab initio with IsRNA (Zhang
et al. 2021) and RNAJP (Li and Chen 2023), or template-
based with RNAComposer (Popenda et al. 2012), Vfold-
Pipeline (Li et al. 2022), MC-Sym (Parisien and Major 2008),
and 3dRNA (Wang et al. 2019). We also include three deep
learning methods: trRosettaRNA (Wang et al. 2023) and
RhoFold (Shen et al. 2022) and the newly AlphaFold 3
(Abramson et al. 2024). We report the MCQ per angle on the
Test Set in Table 2. MCQPT (pseudo-torsional) results are
available in Supplementary Table S1 of the Supplementary
File. Our RNA-TorsionBERT model has better performances
than SPOT-RNA-1D for every angle. It has an average MCQ
of 17.4 compared to 19.4 for SPOT-RNA-1D. The MCQ im-
provement over SPOT-RNA-1D ranges between 0.2� (for ϵ)
and 4.3� (for δ). It also outperforms the angles inferred from
state-of-the-art methods for RNA 3D structure prediction, in-
cluding the last published method, AlphaFold 3 (Abramson
et al. 2024). Nonetheless, the performances compared to
AlphaFold 3 remains close. RNA-TorsionBERT does not out-
perform it for every angle. trRosettaRNA and RhoFold, two
deep learning methods, have the worst MCQ compared to ab
initio and template-based approaches. It can be explained by
the use of physics in ab initio and template-based
methods that are inferred in the torsional angles. The use of
deep learning approaches might have the counterpart to
not include physics enough, except for AlphaFold 3. Deep
learning methods, while having the best overall results, as
shown in the benchmark done in State-of-the-RNArt
(Bernard et al. 2024c), remain limited in torsional angle
predictions.
3.1.3.1 Results according to sequence length
To study more in details the performances based on the RNA
length, we report in Fig. 5, the MCQ obtained by our
method, SPOT-RNA-1D and AlphaFold 3 depending on the
sequence length for the Test set. We can see our method out-
performs SPOT-RNA-1D for each of the sequence slot.
AlphaFold 3 has lower MCQ for structures with sequences
between 75 and 175 nt. For sequences >200 nucleotides, our
method demonstrates superior performances compared to
both SPOT-RNA-1D and AlphaFold 3, showing the interest
for long range sequences. Results for the MCQPT are shown
in Supplementary Fig. S2 of the Supplementary File.
3.1.3.2 Results according to secondary structure motifs
We report the results of MCQ for three types of secondary
structure motifs (single-stranded, loop and stem) averaged
over the Test set for RNA-TorsionBERT, AlphaFold 3 and
SPOT-RNA-1D in Table 3. We observe that our method
delivers improved performances for each secondary structure
motif [extracted from RNApdbee (Zok et al. 2018)]. It has
an overall MCQ higher for single-stranded than stem motifs.
This behavior is also similar to SPOT-RNA-1D and
AlphaFold 3, which could be explained by the fact that stem
and loop motifs are easier to predict than single-stranded
motifs (and so are the base pairings). Details on the results
for pseudo-torsional angles are available in Supplementary
Table S2 of the Supplementary File.
3.1.3.3 Results according to RNA types
In CASP-RNA, structures can be described as either natural
with or without homologs, or synthetic RNAs. To further
study the different cases where our approach is better than
existing tools, we report the results for the natural (with or
without homologs) and synthetic RNAs in Table 4. Our
method outperforms AlphaFold 3 and SPOT-RNA-1D for
natural RNAs, with the largest gap for RNA without homo-
logs. This could be explained by the reliability of AlphaFold
Figure 4. Polar distribution of the eight torsional angles (α, β, γ, δ, ϵ, f, P, and χ) and the two pseudo-torsional angles (η and θ) for the Training, Validation,
and Test datasets. For each angle, the logarithm of the normalized count is depicted.
Table 1. MCQ for each torsional angle and using all the torsional angles
for the Validation set for DNABERT (3,4, 5, or 6-mers) and RNABERT.
Bold values correspond to the best MCQ values per line.
DNABERT
RNABERT 3-mer 4-mer 5-mer 6-mer
MCQðαÞ 32.3 31.0 31.8 33.9 36.3
MCQðβÞ 17.6 17.6 17.6 17.8 17.8
MCQðγÞ 26.3 22.8 26.5 26.7 28.1
MCQðδÞ 14.4 12.1 15.6 16.7 17.3
MCQðϵÞ 16.1 15.8 16.0 16.0 16.0
MCQðfÞ 22.5 21.7 21.6 21.8 22.2
MCQðPÞ 8.6 8.7 8.8 8.7 8.7
MCQðχÞ 18.2 18.2 18.3 18.5 18.7
MCQ 20.2 19.0 20.2 20.7 21.4
RNA-TorsionBERT 5
Downloaded from https://academic.oup.com/bioinformatics/article/41/1/btaf004/7945663 by Ripon College Library user on 17 March 2025
3 on multiple sequence alignment, and, thus, on the availabil-
ity and quality of homologs for the prediction. AlphaFold 3
has better performances for synthetic RNAs. More details on
the results for different RNA families on RNA-Puzzles are
available in Supplementary Table S3 of the Supplementary
File. Examples of structures are provided in Supplementary
Fig. S3 of the Supplementary File.
3.2 Model quality assessment based on
torsional angles
In this part, we describe the different datasets used for evalu-
ating our scoring function. We used correlation scores to
compare the links of our scoring function to existing metrics.
3.2.1 Datasets
Datasets of near-native structures (or decoys) are necessary to
compare model quality assessment metrics. Indeed, scoring
functions are used to discriminate between near-native struc-
tures, meaning that we need to have nonnative structures to
evaluate the quality of our scoring function.
We used three different datasets with different strategies of
structure generation:
Decoy Test Set I is from RASP (Capriotti et al. 2011), com-
posed of 85 native RNAs with decoys generated with a pre-
dictive model (by applying different sets of Gaussian restraint
parameters). Each RNA has 500 decoys, which are close to
the native structure. We only kept 83 RNAs and removed the
two RNAs that have sequence lengths >512 nucleotides
(PDB ID: 3df3A and 3f1hA).
Decoy Test Set II corresponds to the prediction-models (PM)
subset from rsRNASP (Tan et al. 2022). It has 20 nonredundant
single-stranded RNAs. For each RNA, 40 decoys are generated
with four RNA 3D structure prediction models The decoys are
not as near to native structures as with the Decoy Test Set I.
Decoy Test Set III is the RNA-Puzzles standardized dataset
(Magnus et al. 2020). This dataset comprises 21 RNAs and
dozens of decoy structures for each RNA. The decoys are not
all close to the native structures.
3.2.2 Evaluation measures
Scoring functions aim to discriminate near-native structures.
The Pearson correlation coefficient (PCC) and the enrichment
score (ES) are used to assess the correctness of a given scoring
function. They assess the link between a scoring function and
a given metric.
The Pearson coefficient correlation (PCC) is computed be-
tween the ranked structures based on scoring functions and
structures ranked by metrics. It is defined as:
Table 2. MCQ per torsional angle and for all torsional angles over the Test set for RNA-TorsionBERT compared to SPOT-RNA-1D. We also include
inferred torsional angles from state-of-the-art methods that predict RNA 3D structures from State-of-the-RNArt (Bernard et al. 2024c). Methods are either
deep learning (DL), ab initio (AB), or template-based (TP). Bold values correspond to the best MCQ values per column.
Type MCQðαÞ MCQðβÞ MCQðγÞ MCQðδÞ MCQðϵÞ MCQðfÞ MCQðPÞ MCQðχÞ MCQ
RNA-TorsionBERT DL 29.9 19.0 23.7 9.5 15.3 19.1 8.4 12.2 17.4
SPOT-RNA-1D (Singh et al. 2021) DL 32.5 19.6 26.6 13.7 15.5 20.2 9.8 13.2 19.4
AlphaFold3 (Abramson et al. 2024) DL 29.8 19.9 23.9 8.8 15.2 18.8 8.8 14.1 17.8
IsRNA1 (Zhang et al. 2021) AB 41.9 23.8 33.5 12.5 22.8 31.3 18.9 17.5 24.9
Vfold-Pipeline (Li et al. 2022) TP 41.3 24.1 32.3 14.7 23.4 29.3 17.3 20.6 25.3
RNAComposer (Popenda et al. 2012) TP 43.6 27.5 38.8 13.3 21.4 27.4 16.4 20.6 25.9
RNAJP (Li and Chen 2023) AB 41.3 28.0 33.3 14.0 24.4 32.1 11.2 20.7 26.6
3dRNA (Wang et al. 2019) TP 50.4 31.9 42.3 21.4 31.0 36.1 24.2 23.2 32.5
MC-Sym (Parisien and Major 2008) TP 66.5 26.0 57.9 27.8 22.1 39.6 17.5 23.4 36.0
trRosettaRNA (Wang et al. 2023) DL 59.1 33.8 60.2 21.9 27.9 41.1 28.3 55.4 40.4
RhoFold (Shen et al. 2022) DL 91.4 61.3 67.4 48.1 45.0 53.6 46.7 32.3 54.8
Figure 5. MCQ depending on sequence length (with a window of 25 nt
from 25 nt to 200 nt) for RNA-TorsionBERT, SPOT-RNA-1D, and AlphaFold
3 for the Test set.
Table 3. MCQ for torsional angles averaged over the Test set for RNA-
TorsionBERT, AlphaFold 3, and SPOT-RNA-1D for three secondary
structure motifs: single-stranded, loops, and stems.a Bold values
correspond to the best MCQ values per line.
Motifs RNA-TorsionBERT AlphaFold 3 SPOT-RNA-1D
Single-stranded 24.9 25.3 25.3
Loops 24.1 24.9 24.8
Stems 16.2 16.3 17.7
a Motifs are extracted using RNApdbee (Zok et al. 2018).
Table 4. MCQ per RNA type for the CASP-RNA dataset for RNA-
TorsionBERT, AlphaFold 3, and SPOT-RNA-1D. Molecules are either
natural RNAs with homolog(s) [Natural (H)], natural RNAs without
homolog(s) [Natural (nH)], or synthetic RNAs. The number of times each
model outperforms the others is described in parentheses. Bold values
correspond to the best MCQ values per line.
Type RNA-TorsionBERT AlphaFold 3 SPOT-RNA-1D
Natural (H) 20.6 (3/5) 22.3 (1/5) 22.3 (1/5)
Natural (nH) 11.8 (3/3) 15.5 (0/3) 13.6 (0/3)
Synthetic 16.1 (2/4) 15.9 (2/4) 19.1 (0/4)
All 16.9 (8/12) 18.5 (3/12) 18.8 (1/12)
6 Bernard et al.
Downloaded from https://academic.oup.com/bioinformatics/article/41/1/btaf004/7945663 by Ripon College Library user on 17 March 2025
PCC ¼
PNdecoys
i¼1 ðEn − �EÞðRn − �RÞ
ffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffi
PNdecoys
n¼1 ðEn − �EÞ2
q ffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffi
PNdecoys
n¼1 ðRn − �RÞ2
q
where En and Rn the energy and metric of the nth structure,
respectively. PCC ranges from 0 to 1, where a PCC of 1
means the relationship between metric and energy is
completely linear.
The enrichment score (ES) considers top-ranked structures
from both scoring function and metric. It is defined as:
ES ¼ 100 × jEtop10% \ Rtop10%j
Ndecoys
where jEtop10% \ Rtop10%j is the number of common structures
from the top 10% of structures (measured by the metric) and
the top 10% of structures with the lowest scoring function.
ES ranges between 0 and 10 (perfect scoring). An enrichment
score of 1 means a random prediction, whereas below 1
means a bad score.
3.2.3 TB-MCQ as scoring function
To assess the validity of our scoring function, we computed
with RNAdvisor (Bernard et al. 2024b) the available scoring
functions RASP (Capriotti et al. 2011), ϵSCORE (Bottaro
et al. 2014), DFIRE-RNA (Capriotti et al. 2011), and
rsRNASP (Tan et al. 2022) for the three different Decoys test
sets. We compared TB-MCQ with the state-of-the-art scoring
functions using PCC and ES with the MCQ. The averaged
values are shown in Fig. 6.
TB-MCQ is the scoring function that is the more correlated
to MCQ (PCC of 0.87 and ES of 5.39). rsRNASP still shows
a high correlation to MCQ (PCC of 0.67 and ES of 4.41),
which is surprising as it does not integrate explicit torsional
angles in its computation. What is missing for both scoring
functions to reproduce the MCQ metric perfectly is the accu-
racy of predicted torsional angles. It might be ineffective for
structures that are really close to the native one and where
the inferred angles from these structures are closer to the na-
tive than the predicted ones from RNA-TorsionBERT. PCC
and ES for other distance-based metrics are shown in
Supplementary Fig. S4 of the Supplementary File.
4 Conclusion
In this work, we have developed a language-based model,
RNA-TorsionBERT, to predict RNA torsional and pseudo-
torsional angles from the sequence. With a DNABERT 3-
mers model, the learned embeddings have been used as a
starting point to infer structural features from the sequence.
We have achieved improvement compared to SPOT-RNA-1D
(Singh et al. 2021), the only tool for RNA torsional angle pre-
diction from the raw sequence.
Through an extensive benchmark of state-of-the-art meth-
ods, we have outperformed the angles inferred from the predic-
tive models. We have also included in the benchmark the new
release of AlphaFold, named AlphaFold 3 (Abramson et al.
2024), which gives the best results compared to ab initio,
template-based and deep learning solutions in terms of MCQ
on inferred angles. Our method, RNA-TorsionBERT, remains
better for the prediction of RNA torsional angles with only the
sequence as input, while AlphaFold 3 uses MSA as inputs.
Most protein methods or current deep learning methods
for predicting RNA 3D structures use MSA as inputs, which
is a huge restriction. Indeed, significant families are still un-
known (Kalvari et al. 2018). It also increases the inference
time, where a homology search should be made for each pre-
diction. Our method leverages language model without the
need of homology, which is a benefit for the prediction of
RNA from unknown families.
Through the evaluation of our model for backbone torsional
angles prediction, we have extended this evaluation as a model
quality assessment for RNA 3D structures. Then, we have in-
ferred a scoring function named TB-MCQ. This scoring func-
tion could help the selection of near-native structures in terms
of angle deviation. It is also specific to torsional angles and,
thus, is more related to the angle-based metric MCQ.
Improvements could be made for both RNA-TorsionBERT
and TB-MCQ. The RNA-TorsionBERT performances remain
limited to reconstruct the structures from just the torsional
angles. MCQ remains of high values for the different test
sets, meaning there are still improvements to be made to tor-
sional angle prediction. Indeed, the reconstruction from tor-
sional angles alone is difficult as small angle deviation could
lead to high cumulative divergence. The number of solved
structures remain the main bottleneck to train robust meth-
ods. Different structural tasks could be added to the model,
with the prediction of secondary structure, interatomic dis-
tances, hydrogen bonds, or noncanonical base interactions.
Efforts could be made to improve the language-based model
used, where a model pre-trained more efficiently on RNA
data could help improve the overall performances. The qual-
ity of the scoring function could be enhanced by incorporat-
ing distance atom features, or directly by improving the
prediction of torsional angles itself.
Our RNA-TorsionBERT method can nonetheless be used
as a starting point for the reconstruction of RNA 3D struc-
tures, with ab initio methods, for instance, that include mo-
lecular dynamics to relax the structure. It could also be used
as a feature in a bigger network to predict RNA 3D
conformation.
Supplementary data
Supplementary data are available at Bioinformatics online.
Conflict of interest: None declared.
Funding
This work was supported in part by UDOPIA-ANR-20-
THIA-0013, Labex DigiCosme [project ANR11LABEX004
5DIGICOSME], performed using HPC resources from
Figure 6. PCC and ES between five different scoring functions (RASP,
ϵSCORE, DIFRE-RNA, rsRNASP, and our scoring function TB-MCQ) and
the angle-based metric MCQ. Values are averaged over the three decoy
test sets.
RNA-TorsionBERT 7
Downloaded from https://academic.oup.com/bioinformatics/article/41/1/btaf004/7945663 by Ripon College Library user on 17 March 2025
GENCI/IDRIS [AD011014250], and operated by ANR as
part of the program “Investissement d’Avenir” Idex ParisSaclay
[ANR11IDEX000302].
Data availability
The data underlying this article are freely available online at:
https://evryrna.ibisc.univ-evry.fr/evryrna/RNA-TorsionBERT.
References
Abramson J, Adler J, Dunger J et al. Accurate structure prediction of
biomolecular interactions with AlphaFold 3. Nature 2024;636:E4.
https://doi.org/10.1038/s41586-024-07487-w
Akiyama M, Sakakibara Y. Informative RNA base embedding for RNA
structural alignment and clustering by deep representation learning.
NAR Genom Bioinform 2022;4:lqac012.
Antczak M, Zok T, Osowiecki M et al. RNAfitme: a webserver for
modeling nucleobase and nucleoside residue conformation in fixed-
backbone RNA structures. BMC Bioinformatics 2018;19:304.
https://doi.org/10.1186/s12859-018-2317-9
Bernard C, Postic G, Ghannay S et al. Has AlphaFold 3 reached its suc-
cess for RNAs? bioRxiv, https://doi.org/10.1101/2024.06.13.
598780, 2024a, preprint: not peer reviewed.
Bernard C, Postic G, Ghannay S et al. RNAdvisor: a comprehensive
benchmarking tool for the measure and prediction of RNA struc-
tural model quality. Brief Bioinform 2024b;25:bbae064.
Bernard C, Postic G, Ghannay S et al. State-of-the-RNArt: benchmarking
current methods for RNA 3D structure prediction. NAR Genom
Bioinform 2024c;6:lqae048. https://doi.org/10.1093/nargab/lqae048
Bottaro S, Di Palma F, Bussi G. The role of nucleobase interactions in
RNA structure and dynamics. Nucleic Acids Res 2014;
42:13306–14.
Capriotti E, Norambuena T, Marti-Renom MA et al. All-atom
knowledge-based potential for RNA structure prediction and assess-
ment. Bioinformatics 2011;27:1086–93.
Cruz JA, Blanchet M-F, Boniecki M et al. RNA-Puzzles: a CASP-like
evaluation of RNA three-dimensional structure prediction. RNA
2012;18:610–25.
Das R, Kretsch R, Simpkin A et al. Assessment of three-dimensional
RNA structure prediction in CASP15. Proteins 2023;91:1747–70.
Devlin J, Chang M, Lee K et al. BERT: pre-training of deep bidirectional
transformers for language understanding. CoRR, abs/1810.04805,
2018, preprint: not peer reviewed.
Ji Y, Zhou Z, Liu H et al. DNABERT: pre-trained bidirectional encoder
representations from transformers model for DNA-language in ge-
nome. Bioinformatics 2021;37:2112–20.
Jumper J, Evans R, Pritzel A et al. Highly accurate protein structure pre-
diction with AlphaFold. Nature 2021;596:583–9.
Kalvari I, Argasinska J, Quinones-Olvera N et al. Rfam 13.0: shifting to
a genome-centric resource for non-coding RNA families. Nucleic
Acids Res 2018;46:D335–42.
Li J, Chen S-J. RNAJP: enhanced RNA 3D structure predictions with
non-canonical interactions and global topology sampling. Nucleic
Acids Res 2023;51:3341–56.
Li J, Zhang S, Zhang D et al. Vfold-Pipeline: a web server for RNA 3D
structure prediction from sequences. Bioinformatics 2022;38:4042–3.
Magnus M, Antczak M, Zok T et al. RNA-Puzzles toolkit: a computa-
tional resource of RNA 3D structure benchmark datasets, structure
manipulation, and evaluation tools. Nucleic Acids Res 2020;
48:576–88.
Mariani V, Biasini M, Barbato A et al. lDDT: a local superposition-free
score for comparing protein structures and models using distance
difference tests. Bioinformatics 2013;29:2722–8.
Miao Z, Adamiak RW, Antczak M et al. RNA-Puzzles round III: 3D
RNA structure prediction of five riboswitches and one ribozyme.
RNA 2017;23:655–72. https://doi.org/10.1261/rna.060368.116
Miao Z, Adamiak RW, Antczak M et al. RNA-Puzzles round IV: 3D
structure predictions of four ribozymes and two aptamers. RNA
2020;26:982–95. https://doi.org/10.1261/rna.075341.120
Miao Z, Adamiak RW, Blanchet MF et al. RNA-Puzzles round II: as-
sessment of RNA structure prediction programs applied to three
large RNA structures. RNA 2015;21:1066–84. https://doi.org/10.
1261/rna.049502.114
Parisien M, Major F. The MC-Fold and MC-Sym pipeline infers RNA
structure from sequence data. Nature 2008;452:51–5.
Parisien M, Cruz J, Westhof E et al. New metrics for comparing and
assessing discrepancies between RNA 3D structures and models.
RNA (New York, N.Y.) 2009;15:1875–85.
Popenda M, Szachniuk M, Antczak M et al. Automated 3D structure
composition for large RNAs. Nucleic Acids Res 2012;40:e112.
Schneider B, Moravek Z, Berman HM. RNA conformational classes.
Nucleic Acids Res 2004;32:1666–77. https://doi.org/10.1093/
nar/gkh333
Schneider B, Sweeney BA, Bateman A et al. When will RNA get its
AlphaFold moment? Nucleic Acids Res 2023;51:9522–32. https://
doi.org/10.1093/nar/gkad726
Shen T, Hu Z, Peng Z et al. E2Efold-3D: end-to-end deep learning
method for accurate de novo RNA 3D structure prediction. arXiv,
arXiv:2207.01586, 2022, preprint: not peer reviewed.
Singh J, Paliwal K, Singh J et al. RNA backbone torsion and pseudotor-
sion angle prediction using dilated convolutional neural networks.
J Chem Inf Model 2021;61:2610–22.
Tan Y-L, Wang X, Shi Y-Z et al. rsRNASP: a residue-separation-based
statistical potential for RNA 3D structure evaluation. Biophys J
2022;121:142–56.
Townshend RJL, Eismann S, Watkins AM et al. Geometric deep learn-
ing of RNA structure. Science 2021;373:1047–51.
Wadley LM, Keating KS, Duarte CM et al. Evaluating and learning
from RNA pseudotorsional space: quantitative validation of a re-
duced representation for RNA structure. J Mol Biol 2007;372:
942–57. https://doi.org/10.1016/j.jmb.2007.06.058
Wang J, Wang J, Huang Y et al. 3dRNA v2.0: an updated web server
for RNA 3D structure prediction. Int J Mol Sci 2019;20:4116–8.
Wang W, Feng C, Han R et al. trRosettaRNA: automated prediction of
RNA 3D structure with transformer network. Nat Commun 2023;
14:7266.
Wiedemann J, Zok T, Milostan M et al. LCS-TA to identify similar frag-
ments in RNA 3D structures. BMC Bioinformatics 2017;18:456.
Zhang D, Li J, Chen S-J. IsRNA1: de novo prediction and blind screening
of RNA 3D structures. J Chem Theory Comput 2021;17:1842–57.
Zhang Y, Skolnick J. Scoring function for automated assessment of pro-
tein structure template quality. Proteins 2004;57:702–10.
Zhang Y, Lang M, Jiang J et al. Multiple sequence alignment-based RNA
language model and its application to structural inference. Nucleic
Acids Res 2024;52:e3. https://doi.org/10.1093/nar/gkad1031
Zok T, Popenda M, Szachniuk M. MCQ4Structures to compute similar-
ity of molecule structures. Cent Eur J Oper Res 2014;22:457–73.
Zok T, Antczak M, Riedel M et al. Building the library of rna 3D nucleotide
conformations using the clustering approach. Int J Appl Math Comput
Sci 2015;25:689–700. https://doi.org/10.1515/amcs-2015-0050
Zok T, Antczak M, Zurkowski M et al. RNApdbee 2.0: multifunctional
tool for RNA structure annotation. Nucleic Acids Res 2018;46:
W30–5. https://doi.org/10.1093/nar/gky314
© The Author(s) 2025. Published by Oxford University Press.
This is an Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/), which permits
unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited.
Bioinformatics, 2025, 41, 1–8
https://doi.org/10.1093/bioinformatics/btaf004
Original Paper
8 Bernard et al.
Downloaded from https://academic.oup.com/bioinformatics/article/41/1/btaf004/7945663 by Ripon College Library user on 17 March 2025RNA-TorsionBERT: leveraging language models for RNA 3D
torsion angles prediction
Cl´ement Bernard, Guillaume Postic, Sahar Ghannay, Fariza Tahi
January 3, 2025
Datasets
Figure S1 shows the distribution of the ribose sugar ring angles for the different datasets used.
Their distributions seem quite close, which is also the case for the pseudrotation phase P angle.
Figure S1: Polar distribution of the five ribose sugar ring angles (ν0, ν1, ν2, ν3 and ν4) for the
Training, Validation and Test datasets. For each angle, the logarithm of the normalized count is
depicted
Experimental protocol
We have fine-tuned both DNABERT [1] and RNABERT [2] for the prediction of torsional and
pseudo-torsional angles. For the two models, we used a batch size of 10, the Mean Average loss
1
with a learning rate of 1e-4 and a weight decay of 0.01. We used the AdamW [3] optimizer. All
inputs were padded to have a fixed size of 512 for DNABERT and 440 for RNABERT (limited by
the model), and we trained the models for a maximum of 20 epochs. As there is no RNA of sequence
length between 440 and 512, we used the same datasets for both RNABERT and DNABERT.
Performances
Figure S2: MCQPT per window of 25nt (from 25nt to 200nt) for RNA-TorsionBERT, SPOT-RNA-
1D and AlphaFold 3 inferred angles for the Test set.
2
Figure S3: Structures with the associated MCQ for RNA-Puzzles (A) and CASP-RNA (B). In
blue are reported examples of structures where RNA-TorsionBERT outperforms AlphaFold 3 and
SPOT-RNA-1D. In red are examples of RNA structures where AlphaFold 3 outperforms RNA-
TorsionBERT and SPOT-RNA-1D.
3
Table S1: MCQ per pseudo-torsional angle and MCQPT (MCQ computed for all the pseudo-
torsional angles) over the Test set for RNA-TorsionBERT compared to SPOT-RNA-1D [4]. We
also include inferred torsional angles from state-of-the-art methods that predict RNA 3D structures
from State-of-the-RNArt [5]. Methods are sorted by MCQPT.
Models MCQ(η) MCQ(θ) MCQPT
RNA-TorsionBERT 15.2 20.8 18.0
SPOT-RNA-1D [4] 17.0 21.3 19.1
AlphaFold3 [6] 13.8 17.4 15.6
IsRNA1 [7] 18.9 26.1 22.4
RNAJP [8] 20.3 25.5 22.8
Vfold-Pipeline [9] 21.0 27.6 24.2
RNAComposer [10] 21.0 28.1 24.5
3dRNA [11] 25.5 31.6 28.5
RhoFold [12] 28.1 31.6 29.8
MC-Sym [13] 28.5 32.9 30.6
trRosettaRNA [14] 26.0 36.9 31.3
Table S2: MCQPT for our method RNA-TorsionBERT, AlphaFold 3 [6] and SPOT-RNA-1D [4]
on secondary motifs averaged on the Test set. Secondary motifs are extracted from RNApdbee [15]
Motifs RNA-TorsionBERT AlphaFold 3 SPOT-RNA-1D
Single-stranded 36.4 31.9 48.4
Loops 31.3 30 42.3
Stems 16.3 15.6 24.2
Table S3: MCQ per RNA family for the single-stranded structures from RNA-Puzzles [16–19]
dataset. The number of times each model outperforms the others is described in parentheses. The
models compared are RNA-TorsionBERT, AlphaFold 3 [6] and SPOT-RNA-1D [4].
Family RNA-TorsionBERT AlphaFold 3 SPOT-RNA-1D
Aptamer 18.6 (0/3) 16.4 (2/3) 17.5 (1/3)
Riboregulator 13.0 (1/1) 16.2 (0/1) 13.8 (0/1)
Riboswitch 16 (1/11) 13.9 (9/11) 16.6 (1/11)
Ribozyme 22.6 (3/4) 23.0 (1/4) 24.5 (0/4)
Ricin loop 8.5 (0/1) 6.6 (1/1) 10.9 (0/1)
Virus 13.1 (0/2) 10.3 (2/2) 16.5 (0/2)
All 16.8 (5/22) 15.3 (15/22) 17.7 (1/22)
4
Model quality assessment based on torsional angles
Figure S4: PCC (A) and ES (B) between five different scoring functions (RASP [20], ϵSCORE [21],
DIFRE-RNA [22], rsRNASP [23] and our scoring functions TB-MCQ) and ten metrics (RMSD,
INFall [24], DI [24], GDT-TS [25], CAD-score [26], ϵRMSD [21], TM-score [27, 28], lDDT [29],
MCQ [30], and LCS-TA [31] (with a threshold of 10, 15, 20 and 25)). Values are averaged over the
three decoy test sets.
5
References
1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17. 18. 19. Ji Y, Zhou Z, Liu H, et al. DNABERT: pre-trained Bidirectional Encoder Representations
from Transformers model for DNA-language in genome. Bioinformatics 15 2021;37:2112–20.
Akiyama M and Sakakibara Y. Informative RNA base embedding for RNA structural align-
ment and clustering by deep representation learning. NAR Genomics and Bioinformatics
2022;4:lqac012.
Loshchilov I and Hutter F. Decoupled Weight Decay Regularization. arXiv 2019.
Singh J, Paliwal K, Singh J, et al. RNA Backbone Torsion and Pseudotorsion Angle Prediction
Using Dilated Convolutional Neural Networks. Journal of Chemical Information and Modeling
6 2021;61:2610–22.
Bernard C, Postic G, Ghannay S, and Tahi F. State-of-the-RNArt: benchmarking current
methods for RNA 3D structure prediction. NAR Genomics and Bioinformatics 2024;6:lqae048.
Abramson J, Adler J, Dunger J, et al. Accurate structure prediction of biomolecular interac-
tions with AlphaFold 3. Nature 2024.
Zhang D, Li J, and Chen SJ. IsRNA1: De Novo Prediction and Blind Screening of RNA 3D
Structures. Journal of Chemical Theory and Computation 3 2021;17:1842–57.
Li J and Chen SJ. RNAJP: enhanced RNA 3D structure predictions with non-canonical in-
teractions and global topology sampling. Nucleic Acids Research 7 2023;51:3341–56.
Li J, Zhang S, Zhang D, et al. Vfold-Pipeline: a web server for RNA 3D structure prediction
from sequences. Bioinformatics 2022;38:4042–3.
Popenda M, Szachniuk M, Antczak M, et al. Automated 3D structure composition for large
RNAs. Nucleic Acids Research 14 2012;40:e112–e112.
Wang J, Wang J, Huang Y, et al. 3dRNA v2.0: An Updated Web Server for RNA 3D Structure
Prediction. International Journal of Molecular Sciences 17 2019;20:4116.
Shen T, Hu Z, Peng Z, et al. E2Efold-3D: End-to-End Deep Learning Method for Accurate
de Novo RNA 3D Structure Prediction. arXiv preprint arXiv:2207.01586 2022.
Parisien M and Major F. The MC-Fold and MC-Sym pipeline infers RNA structure from
sequence data. Nature 7183 2008;452:51–5.
Wang W, Feng C, Han R, et al. trRosettaRNA: automated prediction of RNA 3D structure
with transformer network. Nat Commun 2023;14:7266.
Zok T, Antczak M, Zurkowski M, et al. RNApdbee 2.0: multifunctional tool for RNA structure
annotation. Nucleic Acids Research 2018;46:W30–W35.
Cruz JA, Blanchet MF, Boniecki M, et al. RNA-Puzzles : A CASP-like evaluation of RNA
three-dimensional structure prediction. RNA 4 2012;18:610–25.
Miao Z, Adamiak RW, Blanchet MF, et al. RNA-Puzzles Round II: assessment of RNA struc-
ture prediction programs applied to three large RNA structures. RNA 2015;21:1066–84.
Miao Z, Adamiak RW, Antczak M, et al. RNA-Puzzles Round III: 3D RNA structure predic-
tion of five riboswitches and one ribozyme. RNA 5 2017;23:655–72.
Miao Z, Adamiak RW, Antczak M, et al. RNA-Puzzles Round IV: 3D structure predictions
of four ribozymes and two aptamers. RNA 8 2020;26:982–95.
6
20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. Capriotti E, Norambuena T, Marti-Renom MA, et al. All-atom knowledge-based potential for
RNA structure prediction and assessment. Bioinformatics 8 2011;27:1086–93.
Bottaro S, Di Palma F, and Bussi G. The Role of Nucleobase Interactions in RNA Structure
and Dynamics. Nucleic acids research 2014;42.
Capriotti E, Norambuena T, Marti-Renom MA, et al. All-atom knowledge-based potential for
RNA structure prediction and assessment. Bioinformatics 2011;27:1086–93.
Tan YL, Wang X, Shi YZ, et al. rsRNASP: A residue-separation-based statistical potential
for RNA 3D structure evaluation. Biophysical Journal 1 2022;121:142–56.
Parisien M, Cruz J, Westhof E, et al. New metrics for comparing and assessing discrepancies
between RNA 3D structures and models. RNA (New York, N.Y.) 2009;15:1875–85.
Zemla A, Venclovas C, Moult J, et al. Processing and analysis of CASP3 protein structure
predictions. Proteins: Structure, Function, and Bioinformatics 1999;37:22–9.
Olechnovic K, Kulberkyte E, and Venclovas C. CAD-score: A new contact area difference-
based function for evaluation of protein structural models. Proteins 2013;81.
Zhang Y and Skolnick J. Scoring function for automated assessment of protein structure
template quality. Proteins 2004;57:702–10.
Gong S, Zhang C, and Zhang Y. RNA-align: quick and accurate alignment of RNA 3D struc-
tures based on size-independent TM-scoreRNA. Bioinformatics 21 2019;35:4459–61.
Mariani V, Biasini M, Barbato A, et al. lDDT: a local superposition-free score for comparing
protein structures and models using distance difference tests. Bioinformatics (Oxford, Eng-
land) 2013;29:2722–8.
Zok T, Popenda M, and Szachniuk M. MCQ4Structures to compute similarity of molecule
structures. Central European Journal of Operations Research 2013;22.
Wiedemann J, Zok T, Milostan M, et al. LCS-TA to identify similar fragments in RNA 3D
structures. BMC Bioinformatics 2017;18:456.
7