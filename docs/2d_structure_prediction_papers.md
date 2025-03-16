=====
Published online 18 November 2021 Nucleic Acids Research, 2022, Vol. 50, No. 3 e14
https://doi.org/10.1093/nar/gkab1074
UFold: fast and accurate RNA secondary structure
prediction with deep learning
Laiyi Fu 1,2,†, Yingxin Cao2,5,6,†, Jie Wu 3 , Qinke Peng 1 , Qing Nie 4,5,6 and Xiaohui Xie2,*
1 Systems Engineering Institute, School of Electronic and Information Engineering, Xi’an Jiaotong University, Xi’an,
Shaanxi 710049, China, 2 Department of Computer Science, University of California, Irvine, CA 92697, USA,
3 Department of Biological Chemistry, University of California, Irvine, CA 92697, USA, 4 Department of Mathematics,
University of California, Irvine, CA 92697, USA, 5 Center for Complex Biological Systems, University of California,
Irvine, CA 92697, USA and 6 NSF-Simons Center for Multiscale Cell Fate Research, University of California, Irvine,
CA 92697, USA
Received April 30, 2021; Revised September 15, 2021; Editorial Decision October 18, 2021; Accepted October 19, 2021
ABSTRACT
For many RNA molecules, the secondary structure
is essential for the correct function of the RNA. Pre-
dicting RNA secondary structure from nucleotide se-
quences is a long-standing problem in genomics, but
the prediction performance has reached a plateau
over time. Traditional RNA secondary structure pre-
diction algorithms are primarily based on thermo-
dynamic models through free energy minimization,
which imposes strong prior assumptions and is
slow to run. Here, we propose a deep learning-
based method, called UFold, for RNA secondary
structure prediction, trained directly on annotated
data and base-pairing rules. UFold proposes a novel
image-like representation of RNA sequences, which
can be efficiently processed by Fully Convolutional
Networks (FCNs). We benchmark the performance
of UFold on both within- and cross-family RNA
datasets. It significantly outperforms previous meth-
ods on within-family datasets, while achieving a sim-
ilar performance as the traditional methods when
trained and tested on distinct RNA families. UFold
is also able to predict pseudoknots accurately. Its
prediction is fast with an inference time of about
160 ms per sequence up to 1500 bp in length. An
online web server running UFold is available at
https://ufold.ics.uci.edu. Code is available at https:
//github.com/uci-cbcl/UFold.
INTRODUCTION
The biology of RNA is diverse and complex. Aside from
its conventional role as an intermediate between DNA and
protein, cellular RNA consists of many other functional
classes, including ribosomal RNA (rRNA), transfer RNA
(tRNA), small nuclear RNA (snRNA), microRNA and
other noncoding RNAs (1–4). Some RNAs possess cat-
alytic functionality, playing a role similar to protein en-
zymes. The spliceosome, which performs intron splicing,
is assembled from several snRNAs. The microRNAs are
abundant in many mammalian cell types, targeting ∼60%
of genes (5), and are often regarded as biomarkers for di-
verse diseases (6).
Cellular RNA is typically single-stranded. RNA fold-
ing is in large part determined by nucleotide base pair-
ing, including canonical base pairing––A–U, C–G and non-
Watson–Crick pairing G-U, and non-canonical base pair-
ing (7,8). The base-paired structure is often referred to
as the secondary structure of RNA (9). For many RNA
molecules, the secondary structure is essential for the cor-
rect function of the RNA, in many cases, more than the pri-
mary sequence itself. As evidence of this, many homologous
RNA species demonstrate conserved secondary structures,
although the sequences themselves may diverge (10).
RNA secondary structure can be determined from
atomic coordinates obtained from X-ray crystallography,
nuclear magnetic resonance (NMR), or cryogenic elec-
tron microscopy (11–13). However, these methods have low
throughput. Only a tiny fraction of RNAs have experimen-
tally determined structures. To address this limitation, ex-
perimental methods have been proposed to infer base par-
ing by using probes based on enzymes, chemicals, and cross-
linking techniques coupled with high throughput sequenc-
ing (14–17). Although promising, these methods are still at
the early stage of development, unable to provide precise
base-pairing at a single nucleotide solution.
Computationally predicting the secondary structure of
RNA is a long-standing problem in genomics and bioinfor-
matics. Many methods have been proposed over the past
two decades. They can be broadly classified into two cate-
gories: (i) single sequence prediction methods and (ii) com-
* To whom correspondence should be addressed. Tel: +1 949 824 9289; Fax: +1 949 824 4056; Email: xhx@ics.uci.edu
†The authors wish it to be known that, in their opinion, the first two authors should be regarded as Joint First Authors.
C© The Author(s) 2021. Published by Oxford University Press on behalf of Nucleic Acids Research.
This is an Open Access article distributed under the terms of the Creative Commons Attribution-NonCommercial License
(http://creativecommons.org/licenses/by-nc/4.0/), which permits non-commercial re-use, distribution, and reproduction in any medium, provided the original work
is properly cited. For commercial re-use, please contact journals.permissions@oup.com
Downloaded from https://academic.oup.com/nar/article/50/3/e14/6430845 by Ripon College Library user on 16 March 2025
e14 Nucleic Acids Research, 2022, Vol. 50, No. 3 PAGE 2 OF 12
parative methods. In the first category, the most common
method is to search for thermodynamically stable states
through free energy minimization. If the secondary struc-
ture contains only nested base pairing, the energy minimiza-
tion can be efficiently solved through dynamic program-
ming, such as those implemented in Vienna RNAfold (18),
MFold (19), RNAstructure (20) and CONTRAfold (21).
Faster implementations that try to improve the speed of dy-
namic programming include Rfold (22), Vienna RNAplfold
(23), LocalFold (24) and LinearFold (25). Efficient dy-
namic programming algorithms that sample suboptimal
secondary structures from the Boltzmann ensembles of
structures have also been proposed, for example, Centroid-
Fold (26). However, some dynamic programming-based
methods break down when base pairs contain non-nested
patterns, called pseudoknots, which include two stem–loop
structures with half of one stem intercalating between the
two halves of another stem. Predicting secondary struc-
tures with pseudoknots is hard and has shown to be NP-
complete under the energy minimization framework (27).
Methods in the secondary category utilize covariance meth-
ods by aligning related RNA sequences and identifying cor-
related compensatory mutations. The second category of
methods such as (28–30) analyze multiple sequences to de-
termine points of base covariance within the sequences to
help infer base pair positions, and try to predict conserved
structures. Although the list of proposed methods in each of
the two categories is long and diverse (31), the performance
of these methods has not been significantly improved over
time, reaching a performance ceiling of about 80% (32). It
is possible because they fail to account for base pairing re-
sulting from tertiary interactions (33), unstacked base pairs,
pseudoknot, noncanonical base pairing, or other unknown
factors (8).
Recently deep learning techniques have started to emerge
as an alternative approach to functional structure predic-
tion problems including RNA secondary structure predic-
tion problems (34–38). Compared to the thermodynamic
model-based approaches, the learning-based methods ben-
efit from making few assumptions, allowing pseudoknots,
and accounting for tertiary interactions, noncanonical base
pairing, or other previously unrecognized base-pairing con-
straints. Existing deep learning methods differ in model ar-
chitectural design and their choices of model input and out-
put. These methods either treat the input as a sequence,
utilizing LSTM (39) or transformer encoder (40) to cap-
ture long-range interactions between nucleotides (37,41,42).
Other methods aim to integrate deep learning techniques
with dynamic programming or thermodynamic methods
to alleviate prediction biases (34,35,41). However, existing
deep learning approaches still face several challenges: First,
both LSTM and transformer encoder modules involve a
huge number of model parameters, which lead to high com-
putational cost and low efficiency. Second, integrating with
thermodynamic optimization methods will push the models
to assume the assumptions underlying traditional methods,
which can hinder the model performance. Third, because
the performance of deep learning models depends heavily
on the distribution of training data, we need to think about
how to improve the performance of these models on previ-
ously unseen classes of RNA structures (41). Because many
new RNA families have yet to be discovered, it would be im-
portant for the learning-based models to have a good gen-
eralization ability.
Instead of using the nucleotide sequence itself, the in-
put of our model consists of all possible base-pairing maps
within the input sequence. Each map, first represented by a
square matrix of the same dimension as the input sequence
length, denotes the occurrences of one of the 16 possible
base pairs between the input nucleotides. Under this new
representation, the input is treated as a 2D ‘image’ with 16
channels, allowing the model to explicitly consider all long-
range interactions and all possible base pairing, including
non-canonical ones. We include one additional channel to
store the pairing probability between input base pairs cal-
culated based on three paring rules (34) and concatenate it
with the previous 16 channel representation. So, an over-
all 17 channel 2D map is used as our model input. We use
an encoder-decoder framework to extract multi-scale long-
and short-range interaction features of the input sequence,
implemented in a U-Net model (43). For this reason, we will
refer to our method as UFold (stands for U-Net based on
RNA folding). The output of UFold is the predicted contact
score map between the bases of the input sequence. UFold is
fully convolutional, and as such, it can readily handle input
sequences with variable lengths.
We conduct experiments on both known family RNA
sequences and cross family RNA sequences to compare
the performance of UFold against both the traditional
energy minimization-based methods and recent learning-
based methods. We show that UFold yields substantial
performance gain over previous methods on within-family
datasets, highlighting its promising potential in solving
the RNA secondary structure prediction problem. We also
show how to use synthetic data to improve the generaliza-
tion of UFold on the more challenging cases of cross-family
RNA structure prediction.
UFold is fast with an inference time of an average of 160
ms per sequence for RNA sequences with lengths of up to
1500 bp. We have developed an online web server running
UFold RNA secondary structure prediction. The server is
freely available, allowing users to enter sequences and visu-
alize predicted secondary structures.
MATERIALS AND METHODS
Datasets
Several benchmark datasets are used in this study: (a)
RNAStralign (44), which contains 30 451 unique sequences
from 8 RNA families; (b) ArchiveII (45), which contains
3975 sequences from 10 RNA families and is the most
widely used dataset for benchmarking RNA structure pre-
diction performance; (c) bpRNA-1m (46), which contains
102 318 sequences from 2588 families and is one of the
most comprehensive RNA structure datasets available and
(d) bpRNA-new, derived from Rfam 14.2 (41,47), contain-
ing sequences from 1500 new RNA families. RNA families
occurring in bpRNA-1m or any other dataset are excluded
from bpRNA-new. e) PDB dataset from bpRNA and PDB
database (46,48), which contains high-resolution (<3.5 ˚A)
RNA X-ray structures, we also manually downloaded se-
quences that were submitted to PDB from July 2017 to
Downloaded from https://academic.oup.com/nar/article/50/3/e14/6430845 by Ripon College Library user on 16 March 2025
PAGE 3 OF 12 Nucleic Acids Research, 2022, Vol. 50, No. 3 e14
October 2020. In this work, the bpRNA-new dataset is
treated as a cross-family dataset to assess cross-family
model generalization.
The RNAStralign dataset is randomly split into training
and test sets, with 24 895 and 2854 samples, respectively. Re-
dundant sequences between test and training are removed
in the same way as processed in e2efold (36) and MXFold2
(41). For the bpRNA-1m dataset, we followed the same pro-
cessing procedure used in MXfold2 (41) by using the CD-
HIT program (49) to remove redundant sequences and ran-
domly split the dataset into two sub-datasets for training
and testing, named TR0 and TS0, respectively. Redundancy
removed ArchiveII and bpRNA-new are used only for test-
ing. As for the PDB dataset, we used PDB sequences re-
trieved from bpRNA database and PDB database as train-
ing data, and then referred to the name of datasets TS1,
TS2, TS3 from (50) as test set and manually collect their
high-quality RNA secondary structure from the PDB file
using RNApdbee 2.0 (51). Sequences with similarity scores
of greater than 80% to the training data were discarded us-
ing CD-HIT-EST. Details of statistics of the datasets are
listed in Supplementary Tables S1 and S2. In addition,
we also include data augmentation strategy to enlarge the
training set, which is detailed in Results section. All in all,
the training datasets we used in the paper are RNAStralign
training dataset, TR0, augmented training data, and PDB
training data. The test datasets are ArchiveII, TS0, bpRNA-
new and PDB test data (TS1, TS2 and TS3).
Input and output representation
The general problem of the RNA secondary struc-
ture prediction is to predict base pairing patterns given
an input sequence. Let x = (x1, x2, · · · , xL) with xi ∈
{ ′A′, ′U′, ′C′, ′G′ } be an input sequence of length L. The
goal is to predict the secondary structure of x, represented
by a contact matrix A ∈ {0, 1}L×L with Ai j = 1 denoting
a base pairing between bases xi and x j , and 0 otherwise.
UFold utilizes a deep neural network to predict the con-
tact matrix given the input. Next, we describe several design
choices behind UFold (Figure 1).
Most existing learning-based methods treat the input as
a sequence and use recurrent neural nets (RNNs) to model
the interaction between different bases. Gated RNNs, such
as LSTMs and GRUs, are often the method of choice
for dealing with sequential data because of their ability
to model long-range dependencies. However, RNN mod-
els need to be run sequentially, causing issues in both train-
ing and inference. Newer RNA structure prediction mod-
els based on transformers, which do not require the se-
quential data to be processed in order, have also been
proposed (36).
Unlike the previous models, UFold converts the input
sequence directly into an ‘image’. This is done by first en-
coding x with one-hot representation, representing the se-
quence with an L × 4 binary matrix X ∈ {0, 1}L×4 . x is then
transformed into a 16 × L × Ltensor through a Kronecker
product between x and itself, followed by reshaping dimen-
sions (Figure 1a),
K = X ⊗ X (1)
In this representation, input K ∈ {0, 1}16×L×L can be un-
derstood as an image of size L × L with 16 color channels.
Each channel specifies one of the 16 possible base-pairing
rules; K(i, j, k) denotes whether bases x j and xk are paired
according to the i-th base-pairing rule (e.g. i = 2 for A–C
pairing).
To overcome the sparsity bringing by converting sequenc-
ing into 16 channels, we also adopt an extra channel used
in CDPFold (34), which reflects the implicit matching be-
tween bases (more details in Supplementary notes section
1 and Figure S1). We calculate the paring possibilities be-
tween each nucleotide and others from one sequence ac-
cording to three paring rules (34), using these rules we could
calculate the specific values of each nucleotide position with
other nucleotides. These non-binary values may help allevi-
ate the sparsity of the model and provide more information
on paring bases. The calculated matrix W ∈ R1×L×L is then
concatenated with K along the first dimension to get the
final UFold input I of dimension 17 × L × L.
UFold takes I as input and computes Y = f (I; θ) with a
deep convolutional neural net (Figure 1b). The output Y ∈
[0, 1]L×L is a L × Lmatrix, with Yi j denoting the probability
score of nucleotides bases xi and x j being paired.
The new input representation taken by UFold has several
advantages: first, using an image representation allows it to
model all possible long-range interactions explicitly. Base
pairing between distant sequence segments shows up locally
in the image representation. Second, it considers all pos-
sible base pairing patterns, making no distinction between
canonical and non-canonical base pairs. Third, it allows us
to implement a fully convolutional neural model that can
handle variable sequence length, eliminating the need of
padding the input sequence to a fixed length.
Input and scoring network architecture
UFold uses an encoder-decoder architecture for computing
the predicted contact score matrix Y (Figure 1). The model
consists of a sequence of down-sampling layers (encoder)
to derive increasingly complex semantic representations of
the input, followed by a sequence of up-sampling layers (de-
coder), with lateral connections from the encoder to fill in
contextual information. The overall design follows the U-
Net model, widely used in the field of image segmentation.
More detail on the framework is illustrated in Supplemen-
tary file (Section 2).
All operations in UFold are fully convolutional. Thus,
the input sequence can be of variable length, with the out-
put matrix changing correspondingly. This feature is espe-
cially beneficial for RNA secondary structure as the range
of the input sequence length is very large, from tens of nu-
cleotides for small RNAs to thousands of nucleotides for
large RNAs. Padding input sequences to the same length as
done in other methods would have significantly impacted
the efficiency of the algorithm.
UFold is trained by minimizing the cross-entropy be-
tween the predicted probability contact matrix Y and the
true contact matrix A, using stochastic gradient descent.
The predicted matrix of pairs represents the base-pairing
probabilities, which are strictly positive in our model. Our
final layer of activation function takes the form of a sigmoid
Downloaded from https://academic.oup.com/nar/article/50/3/e14/6430845 by Ripon College Library user on 16 March 2025
e14 Nucleic Acids Research, 2022, Vol. 50, No. 3 PAGE 4 OF 12
Figure 1. The overall architecture of UFold. (A) The input sequence is first converted into one-hot representation. A novel representation of the sequence
is then introduced by taking outer product of all combinations of base pair channels, resulting in an image-like representation with 16 channels and with
the same size as the contact map. We calculate a paring possibilities matrix according to three paring rules and concatenate this extra matrix with previous
feature to obtain the final 17 channel input. (B) Detailed architecture of our framework. The input is a 17 × L × Ltensor representation of the original
sequence. The U-Net takes the 17 × L × L tensor as input and outputs an L × L symmetric score matrix Y. After postprocessing, matrix ˆY∗ is the final
prediction of the contact map.
activation σ (x) = 1
1+e−x , where x is an unbounded output
from the previous layer. A positive weight ω of 300 is added
to leverage the imbalanced 0/1 distribution to derive the
loss function as below.
Loss (Y, A; θ) = − ∑
i j
[Ai j log (Yi j
) + (1 − Ai j
) log (1 − Yi j
)] . (2)
where θ is used to represent all parameters in the neural net-
work.
Postprocessing
After the symmetric contact scoring matrix Y is computed
by UFold, we use a postprocessing procedure to derive
the final secondary structure. The postprocessing proce-
dure takes into account four hard constraints in the sec-
ondary structure: (i) the contact matrix should be symmet-
ric; (ii) only canonical plus U–G paring rules are allowed
(this can be relaxed by including other non-canonical base
pairs); (iii) no sharp loops are allowed, for which we set
Downloaded from https://academic.oup.com/nar/article/50/3/e14/6430845 by Ripon College Library user on 16 March 2025
PAGE 5 OF 12 Nucleic Acids Research, 2022, Vol. 50, No. 3 e14
Ai j = 0, ∀i, j with |i − j |lt; 4 and (iv) no overlapping pairs
are allowed, that is, A1 ≤ 1. We follow the steps used in
e2efold by encoding constraints (ii) and (iii) into a matrix
M, defined as M(x):=1 if nucleotides xi and x j can be paired
under constraints (ii) and (iii) and equals to 0 otherwise.
To address the first two constraints, we transform Y ac-
cording to
T (Y) := 1
2
(Y + YT ) ◦ M(x) (3)
where ◦ denotes element-wise multiplication. It ensures that
the transformed Y is symmetric and satisfies constraints (i),
(ii) and (iii).
To address the last constraint, we relax it into a linear
programming problem,
ˆY∗ = argmax
ˆY∈RL×L
〈 ˆY, T (Y)〉 − ρ‖ ˆY‖, subject to ˆY1 ≤ 1 (4)
which tries to find an optimal scoring matrix ˆY that is
most similar to T (Y) while at the same time satisfying the
nonoverlapping pair constraint. The similarity is measured
in terms of the inner product between ˆYand T (Y). ρ is a
hyperparameter controlling the sparsity of the final output.
The final predicted binary contact map is taken to be ˆY∗
after thresholding it with an offset, which is chosen through
a grid search.
Training and evaluation
During training, stratified sampling (36) is applied to the
training set to balance the number of training samples from
each RNA family. The hyperparameters of UFold are tuned
based on the validation set. The number of parameters is
listed in Supplementary Table S3.
To improve model transferability on previously unseen
RNA families, we augment the training set with synthetic
data to train UFold. The synthetic data are generated by
randomly mutating sequences in the bpRNA-new dataset
(previously unseen RNA families). We then use Contrafold
to generate predicted structures on the synthetic data and
treat them as ground truth.
Precision is defined as Pr ec = T P
T P+F P , evaluated on all
predicted base pairs. Recall is defined as Recall = T P
T P+F N .
And F1 score is the harmonic mean of precision and recall,
defined as F1 = 2 · Pr ec·Recall
Pr ec+Recall . We use CPU version of In-
tel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz, and for the
GPU version we are choosing is Nvidia Titan Xp.
RESULTS
To benchmark the performance of different models, we
first conduct three experimental studies: (a) train models
on the RNAStralign training set and evaluate on the RN-
Stralign test set and ArchiveII; and (b) train the exact same
model on the bpRNA-1m training set (TR0) and evaluate
on the bpRNA-1m test set (TS0) as well as on bpRNA-
new(bpnew). (c) fine-tune previous model on PDB train-
ing dataset and evaluate on a standalone test set. Pub-
lished deep learning models usually report results from ei-
ther Study A or Study B. To have a fair and direct com-
parison with previous models, we report results from both,
Figure 2. Violin plot on the ArchiveII dataset. Visualization of F1 value
of UFold versus other 11 RNA secondary structure predictions methods.
following the same data splitting, preprocessing, and evalu-
ation protocols.
In comparing the results from different models, we
treat within- versus cross-family results separately. In both
studies, the test sets, except bpRNA-new(bpnew), contain
mostly within family RNA species, that is, RNA species
from a similar family occurring in the training set. By con-
trast, the bpRNA-new dataset contains only cross-family
RNA species, that is, none of them shares the same RNA
family as those in the training set. Although RNAs that are
from a known family are easier digging into, their folding
patterns can provide more useful information of formation
secondary structure, which it is helpful for the model’s per-
formance on previously unseen families to assess its model
transferability.
Experimental results on within family datasets
In this section, we report the results of our model on
within-family test sets. Figure 2 and Supplementary Ta-
ble S4 summarizes the evaluation results of UFold on the
ArchieveII test set (from Study A), together with the results
of a collection of traditional energy-based, including Con-
textfold (52), Contrafold (21), Linearfold (25), Eternafold
(53), RNAfold (18), RNAStructure (Fold) (54), RNAsoft
(55) and Mfold (19), and recent learning-based methods
MXfold2 (41), SPOT-RNA (37) and e2efold (36). The tra-
ditional methods achieve an F1 score in the range of 0.55–
0.84. A recent state-of-the-art learning-based method im-
proves the F1 score to 0.77 (MXfold2). UFold can further
improve the performance, achieving an F1 score of 0.91.
Compared with MXfold2, UFold achieves an 18% increase
in F1 score, a 22% increase in recall, and a 13% increase in
precision.
Downloaded from https://academic.oup.com/nar/article/50/3/e14/6430845 by Ripon College Library user on 16 March 2025
e14 Nucleic Acids Research, 2022, Vol. 50, No. 3 PAGE 6 OF 12
Figure 3. Violin plot on the TS0 dataset. Visualization of F1 value of
UFold versus other 11 RNA secondary structure predictions methods.
Figure 3 and Supplementary Table S5 summarizes
the evaluation results on the TS0 test set (from Study
B). Since this dataset was also used in two other deep
learning-based methods––SPOT-RNA and MXfold2, we
compare UFold with these two methods along with other
energy-based methods. Again, UFold outperforms both the
deep learning-based and the energy-based methods. UFold
achieves a mean F1 score of 0.654 on this dataset, cor-
responding to a 5.7% improvement over SPOT-RNA, the
state-of-the-art method on this dataset, and 15% improve-
ment over traditional methods. Improvements in recall and
precision also surpass all other methods.
We conduct an experiment to demonstrate whether the
‘image-like’ encoding of sequences helps improve the pre-
diction of long-range interactions. For this experiment, we
use the TS0 dataset as a test dataset since it contains more
versatile sequences of different length and various RNA
families. For each sequence of length L, we define the paired
and unpaired bases with intervals longer than L/2 as long-
range base pairing. We then calculate the precision, recall as
well as F1 score of UFold on these long-range pairing pre-
dictions and compare them to other methods. The results
are reported in Supplementary Figure S2 and Supplemen-
tary Table S6. We find that UFold achieves significantly bet-
ter results than other methods on these long-range pairing
predictions. Moreover, the results also show that the per-
formance of UFold on long-range base pairing prediction
is similar to its performance on short-range base pairings
(Figure 2). By contrast, the performances of all other meth-
ods significantly deteriorate when evaluated on long-range
interactions. These results demonstrate the ‘image-like’ en-
coding facilitates the prediction of long-range interactions.
Table 1. Evaluation results of RNA structures with pseudoknots on the
RNAStralign test dataset
Method Recall Precision Specificity Accuracy
UFold 99% 96.2% 96.8% 87.5%
SPOT-RNA 97.8% 67.7% 61.8% 31.4%
E2Efold 99% 84.4% 84.0% 78.8%
RNAstructure
(ProbKnot)
76.1% 77.8% 81.5% 38.5%
NuPack 93.3% 72.4% 72.2% 51.4%
HotKnotsa 56.5% 50% 83.1% 42.7%
a The sequence number here is 2021, the rest sequence number is 2826.
Predicting secondary structures with pseudoknots is es-
pecially challenging for thermodynamic models. We also
validate the performance of UFold on predicting base pair-
ing in the presence of pseudoknots. For this purpose, we use
all RNA structures in the RNAStralign test set, on which we
then benchmark UFold against other methods that can pre-
dict pseudoknots, including SPOT-RNA, e2efold, RNAs-
tructure(ProbKnot) (56), NuPack (57) and HotKnots (58).
We examined whether ground truth and predictions have
pseudoknot respectively and summarized results in Table
1. As shown in Table 1, all other methods tend to predict
pseudoknot structures for normal sequences. The number
of the pseudoknot pairs of different types is listed in Sup-
plementary Table S7 and accuracy of the pseudoknotted
pairs is also measured. The result is shown in Table 1 as
well. By contrast, UFold still achieves higher recall, pre-
cision and specificity values, while maintaining the high-
est pseudoknotted pairs prediction accuracy compared with
others, highlighting the robustness of UFold predictions in
the presence of pseudoknots.
Experimental results on cross family datasets
In this section, we evaluate the performance of UFold on
previously unseen RNA families. We expect learning-based
methods do poorly on these RNAs since they are not repre-
sented in the training set as shown in Supplementary Table
S8. To address this problem, methods integrating free en-
ergy minimization with deep learning methods have been
proposed, like MXfold2 (41). However, these methods in-
advertently introduce biases into the prediction model and
likely lead to reduced performance on within family RNAs.
Although UFold does not involve any energy minimiza-
tion term in its original design, it uses data augmentation
to improve the performance on cross-family RNAs with
the help of another model Contrafold (21), a probabilis-
tic model which generalizes upon stochastic context-free
grammars (SCFGs) by using discriminative training and
feature-rich scoring found in typical thermodynamic mod-
els. Specifically, for each sequence we randomly choose 20–
30% present of single nucleotides to perform random mu-
tation. For each real sequence, we first generate 3 synthetic
sequences to create a pool of synthetic sequences. We then
use CD-HIT 80 to remove any sequences that have similar-
ity over 80% to real sequences. The resulting synthetic se-
quence pool is then used for generating synthetic data with
size 2000. The synthetic ground truth labels are generated
with Contrafold, which then use to train UFold. Those data
Downloaded from https://academic.oup.com/nar/article/50/3/e14/6430845 by Ripon College Library user on 16 March 2025
PAGE 7 OF 12 Nucleic Acids Research, 2022, Vol. 50, No. 3 e14
Figure 4. Violin plot on the bpRNA-new dataset. Visualization of F1 value
of UFold versus other 11 RNA secondary structure predictions methods.
are then merged with the TR0 training set for model train-
ing.
Figure 4 and Supplementary Table S8 show the eval-
uation results of UFold using the previously pre-trained
model on the bpRNA-new dataset, containing about 1500
previously unseen RNA families. Note that here UFold is
trained only once based on all the training data for the three
testing experiments including ArchiveII, TS0 and bpRNA-
new datasets. UFold can achieve a similar performance on
bpRNA-new dataset as other methods like MXfold2, all
of which involve thermodynamic terms or constraints in
their objectives. By contrast, UFold is a pure learning-based
method. Through data augmentation, it can learn to pre-
dict the structures of RNAs not represented in the training
set and further improved the performance on previously un-
seen family sequences (i.e. bpnew dataset).
Furthermore, UFold is also benchmarked on high-
resolution based RNA secondary structures derived from
the PDB dataset, whose secondary structures have been ex-
perimentally validated. We used pretrained model and fine-
tuned it on PDB sequences retrieved from bpRNA database
and PDB database. Following the partition used in SPOT-
RNA2 (50), we divided the PDB sequences into three sub-
sets: TS1, TS2 and TS3. The overall result is reported in
Figure 5, more detailed results are presented in Supple-
mentary Table S9-S11. Based on the results, UFold is deal-
ing well in recognizing these dense pairing RNA secondary
structures compared with others on this high-quality exper-
imentally validated dataset. We also notice another recent
model SPOT-RNA2 (50) which incorporates evolutionary-
based features besides sequence features, but all the com-
pared models in our results are all only sequence based so
we do not include it in our summarized results. The re-
Figure 5. Violin plot on the PDB dataset. Visualization of F1 value of
UFold versus other 11 RNA secondary structure predictions methods.
sults of splitting these datasets (TS1, TS2 and TS3) are
shown in Supplementary Supplementary Figure S3 and
Supplementary Table S9-S11. In addition, we benchmarked
6 RNAs from PDB dataset, which is measured in SPOT-
RNA paper. We confirmed that none of these sequences
appeared in our training dataset. As shown in Supplemen-
tary Figure S4, UFold produced consistently better results
than SPOT-RNA and other predictors on these 6 RNAs.
Since PDB dataset contains multiple non-canonical pairs,
so we systematically measured the performance of UFold
against SPOT-RNA which is also capable of predicting non-
canonical pairs. The higher mean F1 value in three datasets
indicates the superior ability of predicting non-canonical
pairs of UFold as shown in Supplementary Table S12. These
findings support the effectiveness of UFold in handling non-
canonical pairs. We also explored how the UFold performs
on different Rfam families. We mapped all the sequences
from PDB dataset to Rfam families using Rfam webserver
(https://rfam.xfam.org), during which we found 34 RNA
families matched to Rfam families, covering 47 of the se-
quences in the test set. Among those, we found 26 RNA
families (including 39 sequences) that are overlapped with
training families. We then evaluated the performance of F1
value on two groups: no Rfam family which contains se-
quences that do not match any Rfam or other families in the
training set, and within-family which contains sequences
matching a family in the training set. As reported the re-
sults in Supplementary Figure S5, the sequences that do not
match to any Rfam families even achieve higher mean F1
value as it is shown in Supplementary Figure S6. This fur-
ther demonstrates UFold’s robust performance.
In order to further validate the effectiveness of UFold
prediction, we include the assessment of the statistical sig-
nificance on the performance comparisons between UFold
and other methods. Two types of statistical significance
Downloaded from https://academic.oup.com/nar/article/50/3/e14/6430845 by Ripon College Library user on 16 March 2025
e14 Nucleic Acids Research, 2022, Vol. 50, No. 3 PAGE 8 OF 12
Figure 6. Visualization of two example UFold RNA secondary structure predictions. From top to bottom: ground truth, UFold prediction, and E2Efold
prediction. Two RNA sequences are (A) Aspergillus fumigatus species, the RNA ID is GSP-41122, as recorded in SRPDB database. and (B) Alphaproteobac-
teria subfamily 16S rRNA sequence whose database ID is U13162, as recorded in RNAStralign database(http://rna.urmc.rochester.edu). Non-canonical
base pairs are colored in light green. In both cases, UFold produces predictions more aligned with the ground-truth.
measures are calculated: one based on paired t-tests and the
other based on bootstrapping. The paired t-test P-value re-
sults are shown in Supplementary Table S13, which shows
that UFold performs better than the other methods in a sta-
tistically significant way, with most P-values less than 0.05.
For the PDB dataset, because its three subsets (TS1,
TS2 and TS3) have limited number of sequences, we used
bootstrapping strategy on these datasets to estimate the sta-
tistical significance. The results are summarized in Supple-
mentary Figure S7, which shows that the performance of
UFold is significantly better than nearly all other meth-
ods. For bootstrapping, margins of improvements reside
outside the 95% confidence intervals with steady interval
width (Supplementary Figure S8 and Supplementary Table
S14). Altogether, our results support previous conclusions
and the performance improvements of UFold over previous
methods are statistically significant.
Another point worth noting is that, since UFold chooses
Kronecker product to construct the input, in order to vali-
date whether this is a good choice compared to other con-
catenation such as outer concatenation adopted in SPOT-
RNA (37). We added one additional ablation study, in
which we replace the Kronecker product with outer con-
catenation by first extending the one-hots column wise and
row wise and then concatenating them together to create a
new input matrix. We retrain the whole UFold model with
this input while keeping the rest the same. We use ArchiveII
and bpnew dataset to test the performance in our ablation
study. As it is shown in Supplementary Figure S9, on both
datasets we tested, the Kronecker product design yields bet-
ter results. We think the reason is that the Kronecker prod-
uct design provides a more direct representation of base-
Table 2. Inference time on the RNAStralign test set
Method Time per seq
UFold (Pytorch) 0.16 s (GPU)
MXfold2(Pytorch) 0.31 s (GPU)
E2Efold (Pytorch) 0.40 s (GPU)
SPOT-RNA(Pytorch) 77.80 s (GPU)
CDPfold (tensorflow) 300.107 s
LinearFold (C++) 0.43 s
Eternafold (C++) 6.42 s
RNAsoft (C++) 4.58 s
Mfold (C) 7.65 s
RNAstructure (C) 142.02 s
RNAfold (C) 0.55 s
CONTRAfold (C++) 30.58 s
pairing information. On the other hand, outer concatena-
tion design in theory contains the same information en-
coded in the Kronecker product, but requires more com-
plicated modellings to process this information.
Visualization
After quantitively evaluating the prediction performance,
we visualize the RNA secondary structures predicted by
UFold to check the pairing details of each nucleotide. For
this purpose, the predicted contact maps were first con-
verted to a bpseq format according to base pair positions.
Raw sequences with the corresponding predicted struc-
tures were fed into the VARNA tool (59) to obtain the
visualization result. As a comparison, we also show the
predicted structures from the other three best-performed
methods, MXfold2, SPOT-RNA and e2efold as well as
Downloaded from https://academic.oup.com/nar/article/50/3/e14/6430845 by Ripon College Library user on 16 March 2025
PAGE 9 OF 12 Nucleic Acids Research, 2022, Vol. 50, No. 3 e14
Table 3. Functionality comparison of different RNA structure prediction web servers
Servers
Supported functions UFold SPOT-RNA RNAfold MXfold2 Linearfold Contextfold RNAsoft Contrafold
Sequence type-in Yes Yes Yes Yes Yes Yes Yes Yes
Fasta file Yes No Yes No Yes Yes Yes Yes
Length >600 bp Yes Yes No No Yes Yes No Yes
Online visualization Yes Yes Yes Yes Yes N/A No N/A
Support multi-samples Yes No No No No No No No
the ground-truth structures. Two examples are from the
Aspergillus fumigatus species and Alphaproteobacteria
subfamily 16S rRNA, their RNA IDs are GSP-41122, as
recorded in SRPDB database (60) and U13162 as recorded
in RNAStralign database (http://rna.urmc.rochester.edu),
respectively. They are drawn and shown in Figure 6. In
both cases, UFold generates RNA secondary structures
more similar to the ground-truth when compared with
other state-of-the-art methods like MXfold2, SPOT-RNA
and E2Efold, showing the closest secondary structure to
the ground truth structure. In addition, we also visualized
more examples from PDB database, whose sequences are
retrieved from 2019 to 2021. As the results shown in Sup-
plementary Figures S10 and S11, UFold is capable of pre-
dicting those structures including pseudoknots and non-
canonical pairs more resemble to ground truth structures.
Inference time
The speed of the prediction algorithm is an important factor
in RNA secondary structure prediction, especially for mul-
tiple sequences predicting simultaneously. Traditional en-
ergy minimization-based methods tend to be slow because
of the time complexity of the minimization algorithm. Deep
learning-based methods like MXfold2 and SPOT-RNA uti-
lize LSTM structure, which require significantly more pa-
rameters than UFold, resulting in low efficiency. UFold in-
ference, on the other hand, runs on feedforward neural nets
only. Specifically, it is comprised of a fully connected con-
volutional neural network, which greatly reduces the run-
ning time since all operations are readily parallelizable. It
can also handle multiple sequences at once, leading to sig-
nificantly higher throughput.
The average inference time per sequence of UFold on
the RNAStralign test set (containing sequences longer than
1000 bp) is reported in Table 2, together with the av-
erage running times of other methods. UFold is much
faster than both learning-based and energy-based meth-
ods. UFold is nearly two times faster than MXfold2, and
orders-of-magnitude faster than RNAstruture (Fold), an-
other popular energy-based method. The running times of
UFold and three other recent deep learning-based meth-
ods are also shown in Table 2. All these methods are im-
plemented in PyTorch (61) and thus it allows us to com-
pare their model efficiency directly. Our model is still the
fastest one among all the other deep learning methods, fur-
ther demonstrating the efficiency of UFold. To study the ef-
fect of sequence length on runtime, we demonstrated two
scatter plots of runtime versus length of the sequences. Most
computations of UFold are performed on GPU. We first
plotted the running time cost on GPU calculation which
is shown in Supplementary Figure S12, the runtime is not
significantly affected by sequences length since GPUs have
efficient parallelization supported by modern deep learning
libraries. We then calculated the total runtime (with con-
tact map inference and postprocessing) and compared with
two other fastest methods, RNAfold and Linearfold, which
can deal with variable sequence length of up to 1500 bp.
As shown in Supplementary Figure S13, UFold is almost 5
times faster than the other two methods on the most com-
mon length sequence (∼600 bp) and is at least two times
faster in longer sequences (up to 1500 bp).
Web server
To facilitate the accessibility of UFold, we developed a web
server running UFold on the backend and made it freely
available. Users can type in or upload RNA sequences in
FASTA format. Our server predicts RNA secondary struc-
tures using the pre-trained UFold model (trained on all the
datasets) and stores predicted structures in a dot-bracket
file or bpseq file for end-users to download. Users may
also choose to predict non-canonical pairs or not directly
in the option panel. The server further provides an inter-
face connection to the VARNA tool (59) for visualizing
predicted structures. Most existing RNA prediction servers
only permit predicting one RNA sequence at a time, such
as RNAfold, MXfold2 and SPOT-RNA, and restrict the
length of the input sequence. Our server does not have such
limitations. Its main functionality differences compared to
other servers are highlighted in Table 3. The interface of our
web server is shown in Figure 7.
DISCUSSION
In this study, we present UFold, a new deep learning-
based model for RNA secondary structure prediction.
We benchmark UFold on both within- and cross-family
RNA datasets and demonstrate that UFold significantly
outperforms previous methods on within-family datasets,
achieving 10–30% performance improvement over tradi-
tional thermodynamic methods, and 5–27% improvement
in F1 score over the state-of-the-art learning-based method,
bringing in substantial gains in RNA secondary prediction
accuracy. In the meantime, it achieves a similar performance
as the traditional methods when trained and tested on dis-
tinct RNA families. In addition, UFold is fast, being able to
generate predictions at roughly 160ms per sequence.
A key difference between UFold and previous learning-
based methods is its architectural design. Instead of us-
ing raw sequences as input, UFold converts sequences into
Downloaded from https://academic.oup.com/nar/article/50/3/e14/6430845 by Ripon College Library user on 16 March 2025
e14 Nucleic Acids Research, 2022, Vol. 50, No. 3 PAGE 10 OF 12
Figure 7 UFold web server interface (available at https://ufold.ics.uci.edu). UFold web server allows users to type in or upload their own fasta file with
multiple sequences (no number limits) and the backend pretrained model will predict the corresponding RNA secondary structures and provide users
either ct or bpseq file to download or directly visualize them online.
Downloaded from https://academic.oup.com/nar/article/50/3/e14/6430845 by Ripon College Library user on 16 March 2025
PAGE 11 OF 12 Nucleic Acids Research, 2022, Vol. 50, No. 3 e14
‘images’, explicitly modeling all possible base pairing be-
tween the nucleotides of the input sequence. This choice
of input representation has several important implications:
First, base pairing patterns between distant sequence seg-
ments show up locally in the image representation, mak-
ing the detection and learning of these distant base pair-
ing patterns easier. Second, all base pairing patterns are
explicitly represented in the input, allowing the model to
pick up all potential base pairing rules that might con-
tribute to the formation of the secondary structure. Lastly,
but perhaps most importantly, the image representation al-
lows us to implement a fully convolutional model to pick
up base-pairing features across multiple scales through an
encoder-decoder architecture. This implementation is not
only efficient, with operations highly parallelable and allow-
ing for variable input sequence length, but also highly effec-
tive in combining both local and global features for the final
prediction.
Although UFold demonstrates great potential in solv-
ing the RNA secondary structure prediction problem, as
a learning-based method, its performance is inevitably
closely attached to the quality of training data. Unfortu-
nately, the number of experimentally resolved RNA sec-
ondary structures through X-ray crystallography or NMR
remains small. Many secondary structures in the RNAS-
tralign dataset are computationally generated by align-
ing homologous sequences. Fortunately, high-throughput
methods for determining or constraining the secondary
structures of RNAs are starting to emerge (62,63). We
should also mention that UFold currently predicts RNA
structures only based on sequences. It is well-known that
RNA structures also depend on other factors, such as tem-
perature and salt concentration. How to take these factors
into account in deep learning models remains an open ques-
tion. Because UFold uses a flexible network architecture,
we expect it to be able to incorporate the high-throughput
data and specific factors to improve model training and
inference.
We should note that the method presented here can po-
tentially be applied for protein structure prediction as well.
The number of amino acids is much higher than the num-
ber of bases. It is worth exploring whether all amino acid
pairs, which have 400 pairs, or a subset of them should be
considered in the input representation.
In summary, we show the promising potential of deep
learning in solving the long-standing RNA secondary struc-
ture problem. The new framework presented here brings in
a significant performance gain. We expect the prediction ac-
curacy to be further improved as more and higher quality
training data are becoming available.
DATA AVAILABILITY
An online web server running UFold is available at https:
//ufold.ics.uci.edu. Code is available at https://github.com/
uci-cbcl/UFold.
SUPPLEMENTARY DATA
Supplementary Data are available at NAR Online.
ACKNOWLEDGEMENTS
We acknowledge helpful discussions with MH Celik and
members of the Xie lab.
FUNDING
NSF [IIS-1715017]; NSF [DMS-1763272]; NIH [U54-
CA217378]; Simons Foundation [594598]. Funding for
open access charge: NSF [IIS-1715017]; NSF [DMS-
1763272]; NIH [U54-CA217378].
Conflict of interest statement. None declared.
REFERENCES
1. Noller,H.F. (1984) Structure of ribosomal RNA. Annu. Rev.
Biochem., 53, 119–162.
2. Rich,A. and RajBhandary,U. (1976) Transfer RNA: molecular
structure, sequence, and properties. Annu. Rev. Biochem., 45, 805–860.
3. Allmang,C., Kufel,J., Chanfreau,G., Mitchell,P., Petfalski,E. and
Tollervey,D. (1999) Functions of the exosome in rRNA, snoRNA
and snRNA synthesis. EMBO J., 18, 5399–5410.
4. Geisler,S. and Coller,J. (2013) RNA in unexpected places: long
non-coding RNA functions in diverse cellular contexts. Nat. Rev.
Mol. Cell Biol., 14, 699–712.
5. Gebert,L.F. and MacRae,I.J. (2019) Regulation of microRNA
function in animals. Nat. Rev. Mol. Cell Biol., 20, 21–37.
6. Fu,L. and Peng,Q. (2017) A deep ensemble model to predict
miRNA-disease association. Sci. Rep., 7, 14482.
7. Fallmann,J., Will,S., Engelhardt,J., Gr ¨uning,B., Backofen,R. and
Stadler,P.F. (2017) Recent advances in RNA folding. J. Biotechnol.,
261, 97–104.
8. Westhof,E. and Fritsch,V. (2000) RNA folding: beyond
Watson–Crick pairs. Structure, 8, R55–R65.
9. Fox,G.E. and Woese,C.R. (1975) 5S RNA secondary structure.
Nature, 256, 505–507.
10. Mathews,D.H., Moss,W.N. and Turner,D.H. (2010) Folding and
finding RNA secondary structure. Cold Spring Harb. Perspect. Biol.,
2, a003665.
11. F ¨urtig,B., Richter,C., W ¨ohnert,J. and Schwalbe,H. (2003) NMR
spectroscopy of RNA. ChemBioChem, 4, 936–962.
12. Cheong,H.-K., Hwang,E., Lee,C., Choi,B.-S. and Cheong,C. (2004)
Rapid preparation of RNA samples for NMR spectroscopy and
X-ray crystallography. Nucleic Acids Res., 32, e84.
13. Fica,S.M. and Nagai,K. (2017) Cryo-electron microscopy snapshots
of the spliceosome: structural insights into a dynamic
ribonucleoprotein machine. Nat. Struct. Mol. Biol., 24, 791.
14. Ehresmann,C., Baudin,F., Mougel,M., Romby,P., Ebel,J.-P. and
Ehresmann,B. (1987) Probing the structure of RNAs in solution.
Nucleic Acids Res., 15, 9109–9128.
15. Knapp,G. (1989) [16]Enzymatic approaches to probing of RNA
secondary and tertiary structure. Methods Enzymol., 180, 192–212.
16. Bevilacqua,P.C., Ritchey,L.E., Su,Z. and Assmann,S.M. (2016)
Genome-wide analysis of RNA secondary structure. Annu. Rev.
Genet., 50, 235–266.
17. Underwood,J.G., Uzilov,A.V., Katzman,S., Onodera,C.S.,
Mainzer,J.E., Mathews,D.H., Lowe,T.M., Salama,S.R. and
Haussler,D. (2010) FragSeq: transcriptome-wide RNA structure
probing using high-throughput sequencing. Nat. Methods, 7,
995–1001.
18. Lorenz,R., Bernhart,S.H., Zu Siederdissen,C.H., Tafer,H.,
Flamm,C., Stadler,P.F. and Hofacker,I.L. (2011) ViennaRNA
Package 2.0. Algorith. Mol. Biol., 6, 26.
19. Zuker,M. (2003) Mfold web server for nucleic acid folding and
hybridization prediction. Nucleic Acids Res., 31, 3406–3415.
20. Mathews,D.H. and Turner,D.H. (2006) Prediction of RNA
secondary structure by free energy minimization. Curr. Opin. Struct.
Biol., 16, 270–278.
21. Do,C.B., Woods,D.A. and Batzoglou,S. (2006) CONTRAfold: RNA
secondary structure prediction without physics-based models.
Bioinformatics, 22, e90–e98.
Downloaded from https://academic.oup.com/nar/article/50/3/e14/6430845 by Ripon College Library user on 16 March 2025
e14 Nucleic Acids Research, 2022, Vol. 50, No. 3 PAGE 12 OF 12
22. Kiryu,H., Kin,T. and Asai,K. (2008) Rfold: an exact algorithm for
computing local base pairing probabilities. Bioinformatics, 24,
367–373.
23. Bernhart,S.H., Hofacker,I.L. and Stadler,P.F. (2006) Local RNA base
pairing probabilities in large sequences. Bioinformatics, 22, 614–615.
24. Lange,S.J., Maticzka,D., M ¨ohl,M., Gagnon,J.N., Brown,C.M. and
Backofen,R. (2012) Global or local? Predicting secondary structure
and accessibility in mRNAs. Nucleic Acids Res., 40, 5215–5226.
25. Huang,L., Zhang,H., Deng,D., Zhao,K., Liu,K., Hendrix,D.A. and
Mathews,D.H. (2019) LinearFold: linear-time approximate RNA
folding by 5′-to-3′dynamic programming and beam search.
Bioinformatics, 35, i295–i304.
26. Sato,K., Hamada,M., Asai,K. and Mituyama,T. (2009)
CENTROIDFOLD: a web server for RNA secondary structure
prediction. Nucleic Acids Res., 37, W277–W280.
27. Wang,X. and Tian,J. (2011) Dynamic programming for NP-hard
problems. Procedia Eng., 15, 3396–3400.
28. Gardner,P.P. and Giegerich,R. (2004) A comprehensive comparison
of comparative RNA structure prediction approaches. BMC
Bioinformatics, 5, 140.
29. Havgaard,J.H. and Gorodkin,J. (2014) RNA structural alignments.
Part I. Sankoff-based approaches for structural alignments. In: RNA
Sequence, Structure, and Function: Computational and Bioinformatic
Methods. Springer, pp. 275–290.
30. Washietl,S., Bernhart,S.H. and Kellis,M. (2014) Energy-based RNA
consensus secondary structure prediction in multiple sequence
alignments. In: RNA Sequence, Structure, and Function:
Computational and Bioinformatic Methods.
31. Kings Oluoch,I., Akalin,A., Vural,Y. and Canbay,Y. (2018) A review
on RNA secondary structure prediction algorithms. In: 2018
International Congress on Big Data, Deep Learning and Fighting
Cyber Terrorism (IBIGDELFT). IEEE, ANKARA, Turkey, pp.
18–23.
32. Seetin,M.G. and Mathews,D.H. (2012) RNA structure prediction: an
overview of methods. In: Bacterial Regulatory RNA. Springer, pp.
99–122.
33. Nowakowski,J. and Tinoco,I. Jr (1997) RNA structure and stability.
In: Seminars in Virology. Elsevier, Vol. 8, pp. 153–165.
34. Zhang,H., Zhang,C., Li,Z., Li,C., Wei,X., Zhang,B. and Liu,Y.
(2019) A new method of RNA secondary structure prediction based
on convolutional neural network and dynamic programming. Front.
Genet., 10, 467.
35. Wang,L., Liu,Y., Zhong,X., Liu,H., Lu,C., Li,C. and Zhang,H.
(2019) DMFold: A novel method to predict RNA secondary structure
with pseudoknots based on deep learning and improved base pair
maximization principle. Front. Genet., 10, 143.
36. Chen,X., Li,Y., Umarov,R., Gao,X. and Song,L. (2019) RNA
secondary structure prediction by learning unrolled algorithms. In:
International Conference on Learning Representations.
37. Singh,J., Hanson,J., Paliwal,K. and Zhou,Y. (2019) RNA secondary
structure prediction using an ensemble of two-dimensional deep
neural networks and transfer learning. Nat. Commun., 10, 5407.
38. Wang,S., Peng,J., Ma,J. and Xu,J. (2016) Protein secondary structure
prediction using deep convolutional neural fields. Sci. Rep., 6, 18962.
39. Hochreiter,S. and Schmidhuber,J. (1997) Long short-term memory.
Neural Comput., 9, 1735–1780.
40. Cer,D., Yang,Y., Kong,S., Hua,N., Limtiaco,N., John,R.S.,
Constant,N., Guajardo-C´espedes,M., Yuan,S., Tar,C. et al. (2018)
Universal sentence encoder. arXiv doi:
https://arxiv.org/abs/1803.11175v1, 13 April 2018, preprint: not peer
reviewed.
41. Sato,K., Akiyama,M. and Sakakibara,Y. (2021) RNA secondary
structure prediction using deep learning with thermodynamic
integration. Nat. Commun., 12, 941.
42. Chen,X., Li,Y., Umarov,R., Gao,X. and Song,L. (2019) RNA
secondary structure prediction by learning unrolled algorithms. In:
International Conference on Learning Representations.
43. Ronneberger,O., Fischer,P. and Brox,T. (2015) U-net: convolutional
networks for biomedical image segmentation. In: International
Conference on Medical Image Computing and Computer-Assisted
Intervention. pp. 234–241.
44. Tan,Z., Fu,Y., Sharma,G. and Mathews,D.H. (2017) TurboFold II:
RNA structural alignment and secondary structure prediction
informed by multiple homologs. Nucleic Acids Res., 45, 11570–11581.
45. Sloma,M.F. and Mathews,D.H. (2016) Exact calculation of loop
formation probability identifies folding motifs in RNA secondary
structures. RNA, 22, 1808–1818.
46. Danaee,P., Rouches,M., Wiley,M., Deng,D., Huang,L. and
Hendrix,D. (2018) bpRNA: large-scale automated annotation and
analysis of RNA secondary structure. Nucleic Acids Res., 46,
5381–5394.
47. Kalvari,I., Nawrocki,E.P., Ontiveros-Palacios,N., Argasinska,J.,
Lamkiewicz,K., Marz,M., Griffiths-Jones,S., Toffano-Nioche,C.,
Gautheret,D., Weinberg,Z. et al. (2021) Rfam 14: expanded coverage
of metagenomic, viral and microRNA families. Nucleic Acids Res.,
49, D192–D200.
48. Rose,P.W., Prli´c,A., Altunkaya,A., Bi,C., Bradley,A.R.,
Christie,C.H., Costanzo,L.D., Duarte,J.M., Dutta,S., Feng,Z. et al.
(2016) The RCSB protein data bank: integrative view of protein, gene
and 3D structural information. Nucleic Acids Res., 45, D271–D281.
49. Li,W. and Godzik,A. (2006) Cd-hit: a fast program for clustering and
comparing large sets of protein or nucleotide sequences.
Bioinformatics, 22, 1658–1659.
50. Singh,J., Paliwal,K., Zhang,T., Singh,J., Litfin,T. and Zhou,Y. (2021)
Improved RNA secondary structure and tertiary base-pairing
prediction using evolutionary profile, mutational coupling and
two-dimensional transfer learning. Bioinformatics, 37, 2589–2600.
51. Zok,T., Antczak,M., Zurkowski,M., Popenda,M., Blazewicz,J.,
Adamiak,R.W. and Szachniuk,M. (2018) RNApdbee 2.0:
multifunctional tool for RNA structure annotation. Nucleic Acids
Res., 46, W30–W35.
52. Zakov,S., Goldberg,Y., Elhadad,M. and Ziv-Ukelson,M. (2011) Rich
parameterization improves RNA structure prediction. J. Comput.
Biol., 18, 1525–1542.
53. Wayment-Steele,H.K., Kladwang,W., Participants,E. and Das,R.
(2020) RNA secondary structure packages ranked and improved by
high-throughput experiments. bioRxiv doi:
https://doi.org/10.1101/2020.05.29.124511, 31 May 2020, preprint:
not peer reviewed.
54. Reuter,J.S. and Mathews,D.H. (2010) RNAstructure: software for
RNA secondary structure prediction and analysis. BMC
Bioinformatics, 11, 129.
55. Andronescu,M., Aguirre-Hernandez,R., Condon,A. and Hoos,H.H.
(2003) RNAsoft: a suite of RNA secondary structure prediction and
design software tools. Nucleic Acids Res., 31, 3416–3422.
56. Bellaousov,S. and Mathews,D.H. (2010) ProbKnot: fast prediction of
RNA secondary structure including pseudoknots. RNA, 16,
1870–1880.
57. Zadeh,J.N., Steenberg,C.D., Bois,J.S., Wolfe,B.R., Pierce,M.B.,
Khan,A.R., Dirks,R.M. and Pierce,N.A. (2011) NUPACK: analysis
and design of nucleic acid systems. J. Comput. Chem., 32, 170–173.
58. Ren,J., Rastegari,B., Condon,A. and Hoos,H.H. (2005) HotKnots:
heuristic prediction of RNA secondary structures including
pseudoknots. RNA, 11, 1494–1504.
59. Darty,K., Denise,A. and Ponty,Y. (2009) VARNA: Interactive
drawing and editing of the RNA secondary structure. Bioinformatics,
25, 1974.
60. Andersen,E.S., Rosenblad,M.A., Larsen,N., Westergaard,J.C.,
Burks,J., Wower,I.K., Wower,J., Gorodkin,J., Samuelsson,T. and
Zwieb,C. (2006) The tmRDB and SRPDB resources. Nucleic Acids
Res., 34, D163–D168.
61. Paszke,A., Gross,S., Massa,F., Lerer,A., Bradbury,J., Chanan,G.,
Killeen,T., Lin,Z., Gimelshein,N., Antiga,L. et al. (2019) PyTorch:
An imperative style, high-performance deep learning library. In:
Wallach,H., Larochelle,H., Beygelzimer,A., dAlch´e-Buc,F., Fox,E.
and Garnett,R. (eds). Advances in Neural Information Processing
Systems 32. Curran Associates, Inc., pp. 8024–8035.
62. Strobel,E.J., Yu,A.M. and Lucks,J.B. (2018) High-throughput
determination of RNA structures. Nat. Rev. Genet., 19, 615–634.
63. Lusvarghi,S., Sztuba-Solinska,J., Purzycka,K.J., Rausch,J.W. and Le
Grice,S.F. (2013) RNA secondary structure prediction using
high-throughput SHAPE. JoVE (J. Visual. Exp.), e50243.
Downloaded from https://academic.oup.com/nar/article/50/3/e14/6430845 by Ripon College Library user on 16 March 2025
===
ARTICLE
RNA secondary structure prediction using an
ensemble of two-dimensional deep neural
networks and transfer learning
Jaswinder Singh 1 , Jack Hanson 1, Kuldip Paliwal 1 * & Yaoqi Zhou 2*
The majority of our human genome transcribes into noncoding RNAs with unknown struc-
tures and functions. Obtaining functional clues for noncoding RNAs requires accurate base-
pairing or secondary-structure prediction. However, the performance of such predictions by
current folding-based algorithms has been stagnated for more than a decade. Here, we
propose the use of deep contextual learning for base-pair prediction including those non-
canonical and non-nested (pseudoknot) base pairs stabilized by tertiary interactions. Since
only <250 nonredundant, high-resolution RNA structures are available for model training, we
utilize transfer learning from a model initially trained with a recent high-quality bpRNA
dataset of >10,000 nonredundant RNAs made available through comparative analysis.
The resulting method achieves large, statistically significant improvement in predicting all
base pairs, noncanonical and non-nested base pairs in particular. The proposed method
(SPOT-RNA), with a freely available server and standalone software, should be useful for
improving RNA structure modeling, sequence alignment, and functional annotations.
https://doi.org/10.1038/s41467-019-13395-9 OPEN
1 Signal Processing Laboratory, School of Engineering and Built Environment, Griffith University, Brisbane, QLD 4111, Australia. 2 Institute for Glycomics and
School of Information and Communication Technology, Griffith University, Parklands Dr., Southport, QLD 4222, Australia. *email: k.paliwal@griffith.edu.au;
yaoqi.zhou@griffith.edu.au
NATURE COMMUNICATIONS | (2019) 10:5407 | https://doi.org/10.1038/s41467-019-13395-9 | www.nature.com/naturecommunications 1
1234567890():,;
RNA secondary structure is represented by a list of the
nucleotide bases paired by hydrogen bonding within its
nucleotide sequence. Stacking these base pairs forms the
scaffold driving the folding of RNA three-dimensional struc-
tures1 . As a result, the knowledge of the RNA secondary structure
is essential for modeling RNA structures and understanding their
functional mechanisms. As such, many experimental methods
have been developed to infer paired bases by using one-
dimensional or multiple-dimensional probes, such as enzymes,
chemicals, mutations, and cross-linking techniques coupled with
next-generation sequencing 2,3 . However, precise base-pairing
information at the resolution of single base pairs still requires
high-resolution, three-dimensional RNA structures determined
by X-ray crystallography, nuclear magnetic resonance (NMR), or
cryogenic electron microscopy. With <0:01% of 14 million
noncoding RNAs collected in RNAcentral 4 having experimentally
determined structures 5 , it is highly desirable to develop accurate
and cost-effective computational methods for direct prediction of
RNA secondary structure from sequence.
Current RNA secondary-structure prediction methods can be
classified into comparative sequence analysis and folding algo-
rithms with thermodynamic, statistical, or probabilistic scoring
schemes6 . Comparative sequence analysis determines base pairs
conserved among homologous sequences. These methods are
highly accurate7 if a large number of homologous sequences are
available and those sequences are manually aligned with expert
knowledge. However, only a few thousand RNA families are
known in Rfam8 . As a result, the most commonly used approach
for RNA secondary-structure prediction is to fold a single RNA
sequence according to an appropriate scoring function. In this
approach, RNA structure is divided into substructures such as
loops and stems according to the nearest-neighbor model 9 .
Dynamic programming algorithms are then employed for locat-
ing the global minimum or probabilistic structures from these
substructures. The scoring parameters of each substructure can be
obtained experimentally 10 (e.g., RNAfold11 , RNAstructure 12 , and
RNAshapes 13 ) or by machine learning (e.g., CONTRAfold 14 ,
CentroidFold 15 , and ContextFold 16 ). However, the overall
precision (the fraction of correctly predicted base pairs in all
predicted base pairs) appears to have reached a “performance
ceiling”6 at about 80%17,18 . This is in part because all existing
methods ignore some or all base pairs that result from tertiary
interactions19 . These base pairs include lone (unstacked), pseu-
doknotted (non-nested), and noncanonical (not A–U, G–C, and
G–U) base pairs as well as triplet interactions19,20 . While some
methods can predict RNA secondary structures with pseudoknots
(e.g., pknotsRG 21 , Probknot22 , IPknot 23 , and Knotty24 ) and
others can predict noncanonical base pairs (e.g., MC-Fold 25 , MC-
Fold-DP26 , and CycleFold 27 ), none of them can provide a com-
putational prediction for both, not to mention lone base pairs and
base triplets.
The work presented in this paper is inspired by a recent
advancement in the direct prediction of protein contact maps from
protein sequences by Raptor-X28 and SPOT-Contact29 with deep-
learning neural network algorithms such as Residual Networks
(ResNets)30 and two-dimensional Bidirectional Long Short-Term
Memory cells (2D-BLSTMs)31,32. SPOT-Contact treats the entire
protein “image” as context and used an ensemble of ultra-deep
hybrid networks of ResNets coupled with 2D-BLSTMs for pre-
diction. ResNets can capture contextual information from the
whole sequence “image” at each layer and map the complex rela-
tionship between input and output. Also, 2D-BLSTMs proved very
effective in propagating long-range sequence dependencies in
protein structure prediction29 because of the ability of LSTM cells
to remember the structural relationship between the residues that
are far from each other in their sequence positions during training.
Similar to protein contact map, a RNA secondary structure is a
two-dimensional contact matrix, although its contacts are defined
differently (hydrogen bonds for RNA base pairs and distance cutoff
for protein contacts, respectively). However, unlike proteins, the
small number of nonredundant RNA structures available in the
Protein Data Bank (PDB)5 makes deep-learning methods unsui-
table for direct single-sequence-based prediction of RNA secondary
structure. As a result, machine-learning techniques are rarely
utilized. To our knowledge, the only example is mxfold33 that
employs a small-scale machine-learning algorithm (structured
support vector machines) for RNA secondary-structure prediction.
Its performance after combining with a thermodynamic model
makes some improvement over folding-based techniques. How-
ever, mxfold is limited to canonical base pairs without accounting
for pseudoknots.
Recently, a large database of more than 100,000 RNA
sequences (bpRNA 34 ) with automated annotation of secondary
structure was released. While this database is large enough for us
to employ deep-learning techniques, the annotated secondary
structures from the comparative analysis may not be reliable at
the single base-pair level. To overcome this limitation, we first
employed bpRNA to train an ensemble of ResNets and LSTM
networks, similar to the ensemble used by us for protein contact
map prediction by SPOT-Contact29 . We then further trained the
large model with a small database of precise base pairs derived
from high-resolution RNA structures. This transfer-learning
technique 35 is used successfully by us for identifying molecular
recognition features in intrinsically disordered regions of pro-
teins 36 . The resulting method, called SPOT-RNA, is a deep-
learning technique for predicting all bases paired, regardless if
they are associated with tertiary interactions. The new method
provides more than 53%, 47%, and 10% improvement in F1 score
for non-nested, noncanonical, and all base pairs, respectively,
over the next-best method, compared with an independent test
set of 62 high-resolution RNA structures by X-ray crystal-
lography. The performance of SPOT-RNA is further confirmed
by a separate test set of 39 RNA structures determined by NMR
and 6 recently released nonredundant RNAs in PDB.
Results
Initial training by bpRNA. We trained our models of ResNets
and LSTM networks by building a nonredundant set of RNA
sequences with annotated secondary structure from bpRNA34 at
80% sequence-identity cutoff, which is the lowest sequence-identity
cutoff allowed by the program CD-HIT-EST37 and has been
employed previously by many studies for the same purpose38,39.
This dataset of 13,419 RNAs after excluding those >80% sequence
identities was further randomly divided into 10,814 RNAs for
training (TR0), 1300 for validation (VL0), and 1,305 for an inde-
pendent test (TS0). By using TR0 for training, VL0 for validation,
and the single sequence (a one-hot vector of Lx4) as the only input,
we trained many two-dimensional deep-learning models with
various combinations in the numbers and sizes of ResNets,
BLSTM, and FC layers with a layout shown in Fig. 1. The per-
formance of an ensemble of the best 5 models (validated by VL0
only) on VL0 and TS0 is shown in Table 1. Essentially the same
performance with Matthews correlation coefficient (MCC) at 0.632
for VL0 and 0.629 for TS0 suggests the robustness of the ensemble
trained. The F1 scores, the harmonic mean of precision, and sen-
sitivity are also essentially the same between validation and test
(0.629 vs. 0.626). Supplementary Table 1 further compared the
performance of individual models to the ensemble. The MCC
improves by 2% from 0.617 (the best single model) to 0.629 in TS0,
confirming the usefulness of an ensemble to eliminate random
prediction errors in individual models.
ARTICLE NATURE COMMUNICATIONS | https://doi.org/10.1038/s41467-019-13395-9
2 NATURE COMMUNICATIONS | (2019) 10:5407 | https://doi.org/10.1038/s41467-019-13395-9 | www.nature.com/naturecommunications
Transfer learning with RNA structures. The models obtained
from the bpRNA dataset were transferred to further train on base
pairs derived from high-resolution nonredundant RNA structures
with TR1 (training set), VL1 (validation set), and TS1 (test set)
having 120, 30, and 67 RNAs, respectively. The TS1 set is inde-
pendent of the training data (TR0 and TR1) as it was obtained by
first filtering through CD-HIT-EST at the lowest allowed
sequence-identity cutoff (80%). To further remove potential
homologies, we utilized BLAST-N 40 against the training data
(TR0 and TR1) with an e-value cutoff of 10. To examine
the consistency of the models built, we performed 5-fold cross-
validation by combining TR1 and VL1 datasets. The results of
cross-validation on training data (TR1+VL1) and unseen TS1 for
the ensemble of the same top 5 models are shown in Table 1. The
minor fluctuations on 5-fold with MCC of 0.701 ± 0.02 and F1 of
0.690 ± 0.02 and small difference between 5-fold cross-validation
and test set TS1 (0.701 vs. 0.690 for MCC) indicate the robustness
of the models trained for the unseen data. Table 1 also shows that
the direct application of the model trained by bpRNA leads to a
reasonable but inferior performance on TS1 compared with the
model after transfer learning. The improvement in MCC is 6%
before (0.650) and after (0.690) transfer learning on TS1. Sup-
plementary Tables 2 and 3 compare the result of the ensemble of
models and five individual models for five-fold cross-validation
(TR1+VL1) and independent test set (TS1), respectively. Sig-
nificant improvement of the ensemble over the best single model
is observed with 3% improvement in MCC for cross-validation
and independent tests.
Comparison between transfer learning and direct learning. To
demonstrate the usefulness of transfer learning, we also perform
the direct training of the 5 models with the same ensemble net-
work architecture and hyperparameters (the number of layers, the
depth of layers, the kernel size, the dilation factor, and the
learning rate) on the structured RNA train set (TR1) and vali-
dated by VL1 and tested by TS1. The performance of the
ensemble of five models by direct learning on VL1 and TS1 is
shown in Table 1. Similar performance between validation and
test with MCC = 0.583, 0.571, respectively, confirms the robust-
ness of direct learning. However, this performance is substantially
lower than that of transfer learning (21% reduction of the MCC
value and 30% reduction in F1 score). This confirms the difficulty
of direct learning with a small training dataset of TR1 and the
need for using a large dataset (bpRNA) that can effectively utilize
capabilities of deep-learning networks. Supplementary Table 4
further compared the performance of individual models with the
ensemble by direct learning on TR1. Figure 2a compares the
precision-recall (PR) curves given by initial training (SPOT-
RNA-IT), direct training (SPOT-RNA-DT), and transfer learning
(SPOT-RNA) on the independent test set TS1. The results are
from a reduced TS1 (62 RNAs rather than 67) because some
other methods shown in the same figure do not predict secondary
structure for sequences with missing or invalid bases. Interest-
ingly, direct training starts with 100% precision at very low
sensitivity (recall), whereas both initial training and transfer
learning have high but <100% precision at the lowest achievable
sensitivities for the highest possible threshold that separates
positive from negative prediction. This suggests that the existence
of false positives in bpRNA “contaminated” the initial training.
Nevertheless, the transfer learning achieves a respectable 93.2%
precision at 50% recall. This indicates that the fraction of
potential false positives in bpRNA is small.
Comparison with other secondary-structure predictors.
Figure 2a further compares precision/recall curves given by our
transfer-learning ensemble model with 12 other available RNA
Table 1 Performance of SPOT-RNA on validation and test set after initial training, transfer learning, and direct training.
Method Training set Analysis set MCCa F1b Precision Sensitivity
Initial training TR0 VL0 0.632 0.629 0.712 0.563
TR0 TS0 0.629 0.626 0.709 0.560
TR0 TS1 0.650 0.630 0.897 0.485
Transfer learning TR1+VL1 TR1+VL1 0.701 (0.02c) 0.690 (0.02c) 0.853 (0.02c) 0.580 (0.03c)
TR1+VL1 TS1 0.690 (0.02c) 0.687 (0.01c) 0.888 (0.02c) 0.562 (0.02c)
Direct training TR1 VL1 0.583 0.546 0.854 0.401
TR1 TS1 0.571 0.527 0.870 0.378
aMatthews correlation coefficient
b Harmonic mean of precision and sensitivity
c Standard deviation based on five-fold cross-validation
Initial 3 × 3 convolution
Act./norm./dropout
3 × 3 Convolution
Act./norm./dropout
5× 5 Convolution
Block A x (NA-1)
Act./norm.
2D-BLSTM
Fully-connected
layer
Act./norm./dropout
Block B x (NB-1)
Output layer with
sigmoid activation
Block B Block A
Initial training on
large data set
Transfer learning on
experimental PDB data
RNA one-hot encoding
Lx4
Outer concatenation
LxLX8
bpRNA
data set
Model
0
PreT
Model
1
PreT
Model
2
PreT
Model
3
PreT
Model
4
PreT
RNA one-hot encoding
Lx4
LxLX8
Outer concatenation
PDB
data set
Model
0
Model
1
Model
2
Model
3
Model
4
Ensemble averaging
Hairpin
loop Internal loop
Multiloop
Noncanonical
BulgeStem
Pseudoknot
40
30
50
20
10
70 60
77
1
Fig. 1 Generalized model architecture of SPOT-RNA. The network layout of
the SPOT-RNA, where L is the sequence length of a target RNA, Act.
indicates the activation function, Norm. indicates the normalization
function, and PreT indicates the pretrained (initial trained) models trained
on the bpRNA dataset.
NATURE COMMUNICATIONS | https://doi.org/10.1038/s41467-019-13395-9 ARTICLE
NATURE COMMUNICATIONS | (2019) 10:5407 | https://doi.org/10.1038/s41467-019-13395-9 | www.nature.com/naturecommunications 3
secondary-structure predictors on independent test set TS1. Two
predictors (CONTRAfold and CentroidFold) with probabilistic
outputs are also represented by the PR curves with the remaining
shown as a singular point. The performance of most existing
methods is clustered around the sensitivity of 50% and precision
of 67–83% (Table 2). By comparison, our method SPOT-RNA
improves by 9% in MCC and more than 10% in F1 score over the
next-best mxfold.
The results presented in Fig. 2a are the overall performance at
the base-pair level. Figure 2b shows the distribution of the
F1 score among individual RNAs in terms of median, 25th, and
75th percentiles. SPOT-RNA has the highest median F1 score
along with the highest F1 score (0.348) for the worst-performing
RNA, compared with nearly 0 for all other methods. This
highlights the highly stable performance of SPOT-RNA, relative
to all other folding-based techniques, including mxfold, which
mixes thermodynamic and machine-learning models. The
difference between SPOT-RNA and the next-best mxfold on
TS1 is statistically significant with P value < 0.006 obtained
through a paired t test. Also, we calculated the ensemble defect
(see the “Methods” section) from the predicted base-pair
probabilities for SPOT-RNA, CONTRAfold, and CentroidFold
on TS1. The ensemble defect metric describes the deviation of
probabilistic structural ensembles from their corresponding
native RNA secondary structure, where 0 represents a perfect
prediction. The ensemble defect for SPOT-RNA was 0.19 as
compared with 0.24 and 0.25 for CONTRAfold and Centroid-
Fold, respectively, showing that the structural ensemble
predicted by SPOT-RNA is more similar to target structures
in comparison with the other two predictors.
Our method was trained for RNAs with a maximum length of
500 nucleotides, due to hardware limitations. It is of interest to
determine how our method performs in terms of size depen-
dence. As the maximum sequence length in TS1 was 189,
therefore, we added 32 RNAs of sequence length from 298 to
1500 to TS1 by relaxing the resolution requirement to 4 Å and
including RNA chains complexed with other RNAs (but ignored
inter-RNA base pairs). The reason for relaxing the resolution to
4 Å and including RNA chains complexed with other RNAs
because there were not many high-resolution and single-chain
long RNAs in PDB. Supplementary Fig. 1 compares the F1 score
of each RNA given by SPOT-RNA with that from the next-best
mxfold as a function of the length of RNAs. There is a trend of
lower performance for a longer RNA chain for both methods as
expected. SPOT-RNA consistently outperforms mxfold within
500 nucleotides that our method was trained on. Supplementary
Fig. 1 also shows that mxfold performs better with an average of
F1 score at 0.50, compared with 0.35 by SPOT-RNA on 21 long
RNAs (L > 1000). We found that the poor performance of SPOT-
RNA is mainly because of the failure of SPOT-RNA to capture
ultra long-distance pairs with sequence separation >300. This
failure is caused by the limited long RNA data in training. By
comparison, the thermodynamic algorithm in mxfold can locate
the global minimum regardless of the distance between sequence
positions of the base pairs.
The above comparison may be biased toward our method
because almost all other methods compared can only predict
canonical base pairs, which include Watson–Crick (A–U and
G–C) pairs and Wobble pairs (G–U). To address this potential
bias, Table 2 further compares the performance of SPOT-RNA
with others on canonical pairs, Watson–Crick pairs (A–U and
G–C pairs), and Wobble pairs (G–U), separately on TS1. Indeed,
all methods have a performance boost when noncanonical pairs
are excluded from performance measurement. SPOT-RNA
continues to have the best performance with 6% improvement
in F1 score for canonical pairs and Watson–Crick pairs over the
next-best mxfold and 7% improvement for Wobble pairs over the
next-best ContextFold. mxfold does not perform as well in
predicting Wobble pairs and is only the fourth best.
Base pairs associated with pseudoknots are challenging for both
folding-based and machine-learning-based approaches because
they are often associated with tertiary interactions that are
difficult to predict. To make a direct comparison in the capability
of predicting base pairs in pseudoknots, we define pseudoknot
pairs as the minimum number of base pairs that can be removed
to result in a pseudoknot-free secondary structure. The program
bpRNA 34 (available at https://github.com/hendrixlab/bpRNA)
was used to obtain base pairs in pseudoknots from both native
and predicted secondary structures. Table 3 compares the
performance of SPOT-RNA with all 12 other methods regardless
if they can handle pseudoknots or not for those 40 RNAs with at
least one pseudoknot in the independent test TS1. As none of the
other methods predict multiplets, we ignore the base pairs
associated with the multiplets in the analysis. mxfold remains the
0 0.2 0.4 0.6 0.8 1
Sensitivity/recall
0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
Precision
a
SPOT_RNA
SPOT-RNA-IT
SPOT-RNA-DT
mxfold
ContextFold
CONTRAfold
Knotty
IPknot
RNAfold
Probknot
CentroidFold
RNAstructure
RNAshapes (MFE)
pkiss
CycleFold
SPOT-RNA
mxfold
ContextFold
CONTRAfold
Knotty
IPknot
RNAfold
Probknot
CentroidFold
RNAstructure
RNAshapes (MFE)
pkiss
CycleFold
0
0.2
0.4
0.6
0.8
1
F1-score
b
Fig. 2 Performance comparison of SPOT-RNA with 12 other predictors by using PR curve and boxplot on the test set TS1. a Precision-recall curves on the
independent test set TS1 by initial training (SPOT-RNA-IT, the green dashed line), direct training (SPOT-RNA-DT, the blue dot-dashed line), and transfer
learning (SPOT-RNA, the solid magenta line). Precision and sensitivity results from ten currently used predictors are also shown as labeled with open
symbols for the methods accounting for pseudoknots and filled symbols for the methods not accounting for pseudoknots. CONTRAfold and CentroidFold
were also shown as curves (Gold and Black) because their methods provide predicted probabilities. b Distribution of F1 score for individual RNAs on the
independent test set TS1 given by various methods as labeled. On each box, the central mark indicates the median, and the bottom and top edges of the
box indicate the 25th and 75th percentiles, respectively. The outliers are plotted individually by using the “+” symbol.
ARTICLE NATURE COMMUNICATIONS | https://doi.org/10.1038/s41467-019-13395-9
4 NATURE COMMUNICATIONS | (2019) 10:5407 | https://doi.org/10.1038/s41467-019-13395-9 | www.nature.com/naturecommunications
second best behind SPOT-RNA although it is unable to predict
pseudoknots, due to the number of base pairs in pseudoknots
accounting for only 10% of all base pairs (see Supplementary
Table 7). Table 3 shows that all methods perform poorly with
F1 score < 0.3 for base pairs associated with pseudoknots. Despite
the challenging nature of this problem, SPOT-RNA makes a
substantial improvement over the next-best (pkiss) by 52% in
F1 score.
Noncanonical pairs, triplets, and lone base pairs are also
associated with tertiary interactions other than pseudoknots.
Here, lone base pairs refer to a single base pair without
neighboring base pairs (i.e., [i, j] in the absence of [i − 1, j + 1]
and [i + 1, j − 1]). Triplets refer to the rare occasion of one base
forming base pairs with two other bases. As shown in
Supplementary Table 5, SPOT-RNA makes a 47% improvement
in F1 score for predicting noncanonical base pairs over CycleFold.
Although the sensitivity of prediction given by SPOT-RNA is low
(15.4%), the precision is high at 73.2%. Very low performance for
triplets and lone pairs (F1 score < 0.2) is observed.
Secondary structure of RNAs is characterized by structural
motifs in their layout. For each native or predicted secondary
structure, the secondary-structure motif was classified by
program bpRNA 34 . The performance in predicting bases in
different secondary structural motifs by different methods is
shown in Table 4. According to the F1 score, SPOT-RNA makes
the best prediction in stem base pairs (6% improvement over the
next best), hairpin loop nucleotide (8% improvement), and bulge
nucleotide (11% improvement), although it performs slightly
worse than CONTRAfold in multiloop (by 2%). mxfold is best for
internal loop prediction over the second-best predictor Knotty by
18%. To demonstrate the SPOT-RNA’s ability to predict tertiary
interactions along with canonical base pairs, Supplementary
Figs. 2 and 3 show two examples (riboswitch 41 and t-RNA42 )
from TS1 with one high performance and one average
performance, respectively. For both the examples, SPOT-RNA
is able to predict noncanonical base pairs (in green), pseudoknot
base pairs, and lone pair (in blue), while mxfold and IPknot
remain unsuccessful to predict noncanonical and pseudoknot
base pairs.
To further confirm the performance of SPOT-RNA, we
compiled another test set (TS2) with 39 RNA structures solved
by NMR. As with TS1, TS2 was made nonredundant to our
training data by using CD-HIT-EST and BLAST-N. Figure 3a
compares precision-recall curves given by SPOT-RNA with 12
other RNA secondary-structure predictors on the test set TS2.
SPOT-RNA outperformed all other predictors on this test set
(Supplementary Table 6). Furthermore, Fig. 3b shows the
distribution of the F1 score among individual RNAs in terms of
median, 25th, and 75th percentiles. SPOT-RNA achieved the
highest median F1 score with the least fluctuation although the
difference between SPOT-RNA and the next-best (Knotty this
time) on individual RNAs (shown in Supplementary Fig. 4) is not
significant with P value < 0.16 obtained through a paired t test.
Ensemble defect on TS2 is the smallest by SPOT-RNA (0.14 for
SPOT-RNA as compared with 0.18 and 0.19 by CentroidFold and
CONTRAfold, respectively). Here, we did not compare the
performance in pseudoknots because the number of base pairs in
pseudoknots (a total of 21) in this dataset is too small to make
statistically meaningful comparison.
In addition, we found a total of 6 RNAs with recently solved
structures (after March 9, 2019) that are not redundant according
to CD-HIT-EST and BLAST-N to our training sets (TR0 and
TR1) and test sets (TS1 and TS2). The prediction for a synthetic
construct RNA (released on 26 June 2019, chain H in PDB ID
6dvk) 43 was compared with the native structure in Fig. 4a. For
this synthetic RNA, SPOT-RNA yields a structural topology very
Table 2 Performance of all the predictors according to base-pair types on the test set TS1.
All base pairs Canonical only Watson–Crick only Wobble only
MCCa F1b Precision Sensitivity F1b Precision Sensitivity F1b Precision Sensitivity F1b Precision Sensitivity
SPOT-RNA 0.700 0.690 0.849 0.582 0.773 0.858 0.703 0.790 0.857 0.733 0.592 0.865 0.450
mxfold 0.644 0.628 0.824 0.508 0.728 0.824 0.652 0.749 0.830 0.682 0.519 0.747 0.398
ContextFold 0.636 0.621 0.811 0.503 0.719 0.811 0.646 0.737 0.822 0.668 0.554 0.693 0.462
CONTRAfold 0.621 0.611 0.765 0.508 0.704 0.765 0.652 0.724 0.778 0.677 0.517 0.630 0.439
Knotty 0.611 0.603 0.742 0.508 0.694 0.742 0.652 0.713 0.755 0.676 0.519 0.611 0.450
IPknot 0.596 0.576 0.789 0.454 0.671 0.789 0.583 0.690 0.799 0.608 0.483 0.681 0.374
RNAfold 0.593 0.585 0.724 0.491 0.674 0.724 0.630 0.696 0.742 0.655 0.478 0.554 0.421
ProbKnot 0.582 0.576 0.705 0.486 0.662 0.705 0.624 0.684 0.725 0.648 0.466 0.522 0.421
CentroidFold 0.577 0.569 0.706 0.477 0.656 0.706 0.612 0.675 0.719 0.636 0.476 0.569 0.409
RNAstructure 0.570 0.562 0.702 0.469 0.648 0.702 0.602 0.670 0.719 0.627 0.451 0.532 0.392
RNAshapes 0.564 0.555 0.699 0.460 0.640 0.699 0.591 0.661 0.716 0.614 0.455 0.531 0.398
pkiss 0.543 0.538 0.660 0.454 0.619 0.660 0.582 0.643 0.682 0.608 0.403 0.453 0.363
CycleFold 0.461 0.466 0.476 0.456 0.546 0.551 0.540 0.565 0.566 0.564 0.368 0.403 0.339
aMatthews correlation coefficient
b Harmonic mean of precision and sensitivity
NATURE COMMUNICATIONS | https://doi.org/10.1038/s41467-019-13395-9 ARTICLE
NATURE COMMUNICATIONS | (2019) 10:5407 | https://doi.org/10.1038/s41467-019-13395-9 | www.nature.com/naturecommunications 5
similar to the native secondary structure with F1 score of 0.85,
precision of 97%, and sensitivity of 77%. In particular, SPOT-
RNA captures one noncanonical base pair between G46 and A49
correctly but missed others in pseudoknots. The SPOT-RNA
predictions of Glutamine II Riboswitch (chain A in PDB ID 6qn3,
released on June 12, 2019) 44 and Synthetic Construct Hatchet
Ribozyme (chain U in PDB ID 6jq6, released on June 12, 2019)45
are compared with their respective native secondary structure
in Fig. 4b, c, respectively. For these two RNAs, experimental
evidence suggests strand swapping in dimerization44,45 . Thus,
their monomeric native structures are obtained by replacing the
swapped stand by its original stand. SPOT-RNA is able to predict
both the stems and pseudoknot (in Blue) with an overall F1 score
of 0.90 for Glutamine II Riboswitch. For Hatchet Ribozyme,
SPOT-RNA is able to predict native-like structure with F1 score
of 0.74 although it has missed noncanonical and pseudoknot base
pairs.
Three other RNAs are Pistol Ribozyme (chain A and B in PDB
ID 6r47, released on July 3, 2019)46 , Mango Aptamer (chain B in
PDB ID 6e8u, released on April 17, 2019) 47 , and Adenovirus
Virus-associated RNA (chain C in PDB ID 6ol3, released on July
3, 2019)48 . SPOT-RNA achieves F1 score of 0.57, 0.41, and 0.63
on Pistol Ribozyme, Mango Aptamer, and adenovirus virus-
associated RNA, respectively. For this level of performance, it is
more illustrative to show a one-dimensional representation of
RNA secondary structure (Fig. 5a–c). The figures show that the
relatively poor performance of Pistol Ribozyme and Mango
Aptamer RNAs is in part due to the uncommon existence of
a large number of noncanonical base pairs (in Green).
For adenovirus virus-associated RNA (VA-I), SPOT-RNA’s
prediction is poor. It contains three false-positive stems with
falsely predicted pseudoknots (Fig. 5c).
Performance comparison on these 6 RNAs with 12 other
secondary-structure predictors is shown in Fig. 6. SPOT-RNA
outperforms all other predictors on Synthetic Construct RNA
(Fig. 6a), Glutamine II Riboswitch (Fig. 6b), and Pistol Ribozyme
(Fig. 6c). It is the co-first (same as mxfold) in Mango Aptamer
(Fig. 6e) and the second best (behind mxfold only) in Hatchet
Ribozyme (Fig. 6d). However, it did not do well on adenovirus
virus-associated RNA (Fig. 6f), which was part of RNA puzzle-
2017, when compared with other methods. This poor prediction
compared with other methods is likely because this densely
contacted, base-pairing network without pseudoknots (except
those due to noncanonical base pairs) is most suitable for
folding-based algorithms that maximize the number of stacked
canonical base pairs.
Discussion
This work developed RNA secondary-structure prediction
method purely based on deep neural network learning from a
single RNA sequence. Because only a small number of high-
resolution RNA structures are available, deep-learning models
have to be first trained by using a large database of RNA sec-
ondary structures (bpRNA) annotated according to comparative
analysis, followed by transfer learning to the precise secondary
structures derived from 3D structures. Although the slightly noisy
data in bpRNA lead to an upbound around 96% for the precision
(Fig. 2a), the model generated from transfer learning yields a
substantial improvement (30% in F1 score) over the model based
on direct learning TS1. Without the need for folding-based
optimization, the transfer-learning model yields a method that
can predict not only canonical base pairs but also those base pairs
often associated with tertiary interactions, including pseudoknots,
lone, and noncanonical base pairs. By comparing with 12 current
secondary-structure prediction techniques by using the inde-
pendent test of 62 high-resolution X-ray structures of RNAs, the
method (SPOT-RNA) achieved 93% in precision, which is a 13%
improvement over the second-best method mxfold when the
sensitivity for SPOT-RNA is set to 50.8% as in mxfold.
One advantage of a pure machine-learning approach is that all
base pairs can be trained and predicted, regardless if it is asso-
ciated with local or nonlocal (tertiary) interactions. By compar-
ison, a folding-based method has to have accurate energetic
parameters to capture noncanonical base pairs and sophisticated
algorithms for a global minimum search to account for pseu-
doknots. SPOT-RNA represents a significant advancement in
predicting noncanonical base pairs. Its F1 score improves over
CycleFold by 47% from 17% to 26% although both methods have
a low sensitivity at about 16% (Supplementary Table 5). SPOT-
RNA can also achieve the best prediction of base pairs in pseu-
doknots although the performance of all methods remains low
with an F1 score of 0.239 for SPOT-RNA and 0.157 for the next-
best (pkiss, Table 3). This is mainly because the number of base
pairs in pseudoknots is low in the structural datasets (an average
of 3–4 base pairs per pseudoknot RNA in TS1, see Supplementary
Table 7). Moreover, a long stem of many stacked base pairs is
easier to learn and predict than a few nonlocal base pairs in
pseudoknot. As a reference for future method development, we
Table 3 Performance of all the predictors on 40 pseudoknot RNAs in the test set TS1.
All Base Pairs Base Pairs in Pseudoknots Base Pair not in Pseudoknots
MCCa F1b Precision Sensitivity F1b Precision Sensitivity F1b Precision Sensitivity
SPOT-RNA 0.769 0.764 0.875 0.679 0.239 0.550 0.153 0.797 0.872 0.734
mxfold 0.687 0.682 0.797 0.595 0.000 0.000 0.000 0.714 0.780 0.659
ContextFold 0.686 0.680 0.797 0.594 0.000 0.000 0.000 0.714 0.781 0.658
CONTRAfold 0.659 0.658 0.735 0.595 0.000 0.000 0.000 0.688 0.719 0.659
Knotty 0.678 0.678 0.740 0.625 0.108 0.134 0.090 0.707 0.761 0.660
IPknot 0.638 0.629 0.769 0.533 0.131 0.458 0.076 0.664 0.768 0.585
RNAfold 0.605 0.606 0.666 0.555 0.000 0.000 0.000 0.646 0.666 0.628
ProbKnot 0.610 0.611 0.669 0.562 0.118 0.256 0.076 0.632 0.663 0.603
CentroidFold 0.616 0.616 0.682 0.562 0.000 0.000 0.000 0.644 0.668 0.621
RNAstructure 0.585 0.584 0.650 0.531 0.000 0.000 0.000 0.621 0.647 0.598
RNAshapes 0.569 0.568 0.639 0.512 0.000 0.000 0.000 0.591 0.622 0.563
pkiss 0.564 0.565 0.619 0.520 0.157 0.180 0.139 0.566 0.616 0.523
CycleFold 0.455 0.458 0.423 0.499 0.000 0.000 0.000 0.482 0.422 0.563
aMatthews correlation coefficient
b Harmonic mean of precision and sensitivity
ARTICLE NATURE COMMUNICATIONS | https://doi.org/10.1038/s41467-019-13395-9
6 NATURE COMMUNICATIONS | (2019) 10:5407 | https://doi.org/10.1038/s41467-019-13395-9 | www.nature.com/naturecommunications
also examined the ability of SPOT-RNA to capture triple inter-
actions: one base paired with two other bases. Both precision and
sensitivity are low (12% and 7%, respectively, Supplementary
Table 5). This is mainly because there is a lack of data on base
triples in bpRNA for pretraining and the number of both triplets
and quartets is only 1194 in the structural training set TR1.
To further confirm the performance, SPOT-RNA was applied
to 39 RNA structures determined by NMR (TS2). Unlike X-ray
structures, structures determined by NMRs resulted from mini-
mization of experimental distance-based constraints. These 39
NMR structures, smaller with average length of 51 nucleotides,
have only a total of 21 base pairs in pseudoknots. As a result, they
are much easier to predict for all methods (MCC < 0.7 except
SPOT-RNA for TS1 but >0.74 for most methods in TS2). Despite
of this, SPOT-RNA continues to have the best performance
(Fig. 3, Supplementary Table 6, and Supplementary Fig. 4) as
compared with other 12 predictors. Furthermore, the perfor-
mance of SPOT-RNA was tested on 6 recently released non-
redundant (to TR0 and TR1) RNAs in PDB. SPOT-RNA
performs the best or the same as the best in 4 and the second best
in 1 of the 6 RNAs (Fig. 6).
One limitation of SPOT-RNA is that it was trained by RNAs
shorter than 500 nucleotides due to our hardware limitation.
Within 500 nucleotides, SPOT-RNA provides a consistent
improvement over existing techniques (Supplementary Fig. 1).
However, for really long RNA chains (>1000), a purely machine-
learning-based technique is not as accurate as some of the
folding-algorithm-based methods such as mxfold as shown in
Supplementary Fig. 1. The lack of training for long RNAs is the
main reason. Currently, even if there is no hardware limitation,
the number of high-resolution RNA structures with >500
nucleotides in PDB structures are too few to provide adequate
training. Thus, at this stage, SPOT-RNA is most suitable for RNA
length of <500.
In addition to prediction accuracy, high computational effi-
ciency is necessary for RNA secondary-structure prediction
because genome-scale studies are often needed. We found that the
CPU time for predicting all 62 RNAs in the test set TS1 on a
single thread of 32-core Intel Xenon(R) E5-2630v4 CPU is 540 s,
which is faster than Knotty (2800 s) but slower than IPknot (1.2
s), ProbKnot (13 s), and pkiss (112 s). However, our distributed
version can be easily run on multiple CPU threads or on GPUs.
For example, by running SPOT-RNA on a single Nvidia GTX
TITAN X GPU, the computation time for predicting all 62 RNAs
would be reduced to 39 s. Thus, SPOT-RNA can feasibly be used
for genome-scale studies.
This work has used a single RNA sequence as the only input. It
is quite remarkable that relying on a single sequence alone can
obtain a more accurate method than existing folding methods in
secondary-structure prediction. For protein contact map predic-
tion, evolution profiles generated from PSIBLAST 40 and
HHblits 49 as well as direct coupling analysis among homologous
sequences 50 are the key input vectors responsible for the recent
improvement in highly accurate prediction. Thus, one expects
that a similar evolution-derived sequence profile generated from
BLAST-N and direct/evolution-coupling analysis would further
improve secondary-structure prediction for nonlocal base pairs in
long RNAs, in particular. Indeed, recently, we have shown that
using evolution-derived sequence profiles significantly improves
the accuracy of predicting RNA solvent accessibility and
flexibility 38,39 . For example, the correlation coefficient between
predicted and actual solvent accessibility increases from 0.54 to
0.63 if a single sequence is replaced by a sequence profile from
BLAST-N 38 . However, the generation of sequence profiles and
evolution coupling is computationally time consuming. The
resulting improvement (or lack of improvement) is strongly
Table 4 Performance of all the predictors on secondary-structure motifs on the test set TS1.
Stem
(F1a)
Stem (PR) Stem (SN) Hairpin
loop (F1a)
Hairpin
loop (PR)
Hairpin
loop (SN)
Bulge (F1a) Bulge (PR) Bulge (SN) Internal
loop (F1a)
Internal
loop (PR)
Internal
loop (SN)
Multiloop
(F1a)
Multiloop
(PR)
Multiloop
(SN)
SPOT-RNA 0.762 0.841 0.697 0.686 0.625 0.761 0.369 0.508 0.289 0.266 0.239 0.300 0.562 0.503 0.638
mxfold 0.717 0.769 0.671 0.625 0.525 0.771 0.213 0.360 0.152 0.329 0.270 0.422 0.526 0.465 0.607
ContextFold 0.706 0.755 0.663 0.633 0.513 0.825 0.286 0.539 0.194 0.214 0.170 0.289 0.574 0.544 0.607
CONTRAfold 0.688 0.705 0.671 0.624 0.553 0.715 0.331 0.378 0.294 0.279 0.241 0.331 0.469 0.587 0.391
Knotty 0.670 0.739 0.613 0.600 0.493 0.766 0.295 0.421 0.227 0.279 0.238 0.338 0.549 0.649 0.476
IPknot 0.665 0.754 0.595 0.602 0.510 0.735 0.201 0.474 0.128 0.218 0.202 0.236 0.417 0.339 0.542
RNAfold 0.671 0.686 0.657 0.617 0.539 0.722 0.313 0.500 0.227 0.270 0.218 0.354 0.514 0.555 0.478
ProbKnot 0.625 0.661 0.592 0.571 0.480 0.704 0.276 0.377 0.218 0.209 0.187 0.236 0.481 0.492 0.470
CentroidFold 0.646 0.662 0.632 0.579 0.467 0.761 0.293 0.395 0.232 0.179 0.211 0.156 0.433 0.379 0.506
RNAstructure 0.646 0.665 0.629 0.596 0.508 0.720 0.300 0.440 0.227 0.238 0.204 0.285 0.478 0.546 0.424
RNAshapes 0.627 0.650 0.605 0.574 0.507 0.663 0.310 0.432 0.242 0.238 0.193 0.308 0.433 0.507 0.378
pkiss 0.618 0.684 0.565 0.532 0.449 0.655 0.253 0.457 0.175 0.229 0.183 0.304 0.406 0.494 0.344
CycleFold 0.496 0.431 0.584 0.437 0.564 0.357 0.277 0.333 0.237 0.000 0.000 0.000 0.367 0.374 0.360
a Harmonic mean of precision (PR) and sensitivity (SN)
NATURE COMMUNICATIONS | https://doi.org/10.1038/s41467-019-13395-9 ARTICLE
NATURE COMMUNICATIONS | (2019) 10:5407 | https://doi.org/10.1038/s41467-019-13395-9 | www.nature.com/naturecommunications 7
depending on the number of homologous sequences available in
current RNA sequence databases. If the number of homologous
sequences is too low (which is true for most RNAs), it may
introduce more noise than the signal to prediction as demon-
strated in protein secondary structure and intrinsic disorder
prediction 51,52 . Moreover, synthetic RNAs will not have any
homologous sequences. Thus, we present the method with single-
sequence information as input in this study. Using sequence
profiles and evolutionary coupling as input for RNA secondary-
structure prediction is working in progress.
Another possible method for further improving SPOT-RNA is
to employ the predicted probability as a restraint for folding with
an appropriate scoring function. Such a dual-approach method
will likely improve SPOT-RNA as folding optimization may have
a better capability to capture nonlocal interactions between WC
pairs for long RNAs, in particular as shown in Supplementary
Fig. 1. However, a simple integration may not yield a large
improvement for shorter chains (<500). In mxfold, combining
machine-learning and thermodynamic models leads to 0.6% in
one test set and 5% in another test set33 . Moreover, most ther-
modynamic methods simply ignore noncanonical base pairs and
many do not even account for pseudoknots. mxfold, for example,
employs a pseudoknot-free thermodynamic method to combine
with its machine-learning model. Thus, balancing the perfor-
mance for canonical, noncanonical, and pseudoknots will require
a careful selection of appropriate scoring schemes. A simple
integration may lead to high performance in one type of base pair
at the expense of other types of base pairs. Nevertheless, we found
that if we simply keep only the base pair with the highest pre-
dicted probability in predicted triple interactions, SPOT-RNA
would be improved by another 3% in F1 score (from 0.69 to 0.71
in TS1), confirming that there is some room for improvement.
We will defer this for future studies.
The significantly improved performance in secondary-
structure prediction should allow large improvement in mod-
eling RNA 3D structures. This is because the method predicts
not only canonical base pairs but also provides important ter-
tiary contacts of noncanonical and non-nested base pairs. Thus,
it can serve as a more accurate, quasi-three-dimensional frame
to enable correct folding into the right RNA tertiary structure.
The usefulness of 2D structure prediction for 3D structure
modeling has been demonstrated in RNA Puzzles (blind RNA
structure prediction) 53 . Moreover, improvement in predicting
secondary structural motifs (stems, loops, and bulges, see
Table 4) would allow better functional inference 54,55 , sequence
alignment 56 , and RNA inhibitor design 57 . The method and
datasets are available as a server and stand-alone software
publicly at http://sparks-lab.org/jaswinder/server/SPOT-RNA/
and https://github.com/jaswindersingh2/SPOT-RNA/.
Methods
Datasets. The datasets for initial training were obtained from bpRNA-1m (Ver-
sion 1.0) 34, which consists of 102,348 RNA sequences with annotated secondary
structure. Sequences with sequence similarity of more than 80% were removed by
using CD-HIT-EST 37. About 80% sequence-identity cutoff was the lowest cutoff
allowed by CD-HIT-EST and has been used previously as an RNA nonredundancy
cutoff 38,39 . After removing sequence similarity, 14,565 sequences remained. RNA
sequences with RNA structures from the PDB5 available in this dataset were also
removed as we prepared separate datasets based on RNAs with PDB structure
only 5 . Moreover, due to hardware limitations for training on long sequences, the
maximum sequence length was restricted to 500. After preprocessing, this dataset
contains 13,419 sequences. These sequences were randomly split into 10,814 RNAs
for training (TR0), 1300 for validation (VL0), and 1,305 for independent test (TS0).
Supplementary Table 7 shows the number of RNA sequences and their
Watson–Crick (A–U and G–C), Wobble (G–U), and noncanonical base-pair count
as well as the number of base pairs associated with pseudoknots. The average
sequence lengths in TR0, VL0, and TS0 are all roughly 130. Here, base pairs
associated with pseudoknots are defined as the minimum number of base pairs that
can be removed to result in a pseudoknot-free secondary structure. Pseudoknot
labels were generated by using software bpRNA 34 (available at https://github.com/
hendrixlab/bpRNA).
The datasets for transfer learning were obtained by downloading high-
resolution (<3.5 Å) RNAs from PDB on March 2, 2019 5. Sequences with similarity
of more than 80% among these sequences were removed with CD-HIT-EST 37.
After removing sequence similarity, only 226 sequences remained. These sequences
were randomly split into 120, 30, and 76 RNAs for training (TR1), validation
(VL1), and independent test (TS1), respectively. Furthermore, any sequence in TS1
having sequence similarity of more than 80% with TR0 was also removed, which
reduced TS1 to 69 RNAs. As CD-HIT-EST can only remove sequences with
similarity more than 80%, we employed BLAST-N 40 to further remove potential
sequence homologies with training data with a large e-value cutoff of 10. This
procedure further decreased TS1 from 69 to 67 RNAs.
To further benchmark RNA secondary-structure predictors, we employed 641
RNA structures solved by NMR. Using CD-HIT-EST with 80% identity cutoff
followed by BLAST-N with e-value cutoff of 10 against TR0, TR1, and TS1, we
obtained 39 NMR-solved structures as TS2.
The secondary structure of all the PDB sets was derived from their respective
structures by using DSSR 58 software. For NMR- solved structures, model
1 structure was used as it is considered as the most reliable structure among all. The
numbers of canonical, noncanonical, and pseudoknot base pairs, and base
multiplets (triplets and quartets) for all the sets are listed in Supplementary Table 7.
These datasets along with annotated secondary structure are publicly available at
http://sparks-lab.org/jaswinder/server/SPOT-RNA/ and https://github.com/
jaswindersingh2/SPOT-RNA.
RNA secondary-structure types. For the classification of different RNA
secondary-structure types, we used the same definitions as previously used by
bpRNA 34. A stem is defined as a region of uninterrupted base pairs, with no
intervening loops or bulge. A hairpin loop is a sequence of unpaired nucleotides
with both ends meeting at the two strands of a stem region. An internal loop is
defined as two unpaired strands flanked by closing base pairs on both sides. A
0 0.2 0.4 0.6 0.8 1
Sensitivity/recall
0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1
Precision
a
SPOT_RNA
SPOT-RNA-IT
SPOT-RNA-DT
Knotty
RNAfold
RNAshapes (MFE)
RNAstructure
CONTRAfold
pkiss
Probknot
mxfold
IPknot
ContextFold
CentroidFold
CycleFold
SPOT-RNA
Knotty
RNAfold
RNAshapes (MFE)
RNAstructure
CONTRAfold
pkiss
Probknot
mxfold
IPknot
ContextFold
CentroidFold
CycleFold
0
0.2
0.4
0.6
0.8
1
F1-Measure
b
Fig. 3 Performance comparison of SPOT-RNA with 12 other predictors by using PR curve and boxplot on the test set TS2. a Precision-recall curves on the
independent test set TS2 by various methods as in Fig. 2a labeled. b Distribution of F1 score for individual RNAs on the independent test set TS2 given by
various methods as in Fig. 2b labeled.
ARTICLE NATURE COMMUNICATIONS | https://doi.org/10.1038/s41467-019-13395-9
8 NATURE COMMUNICATIONS | (2019) 10:5407 | https://doi.org/10.1038/s41467-019-13395-9 | www.nature.com/naturecommunications
a b c
Non-canonical
base-pair
50
40
60
30
20
70
80
10
90
95
1
50
40
60
30
20
70
80
10
90
95
1
40
30
1
50
20
10
40
30
50
1
20
10
50
40
60
30
70
80
20
81
1
10
50
40
60
30
70
80
81
120
10
Fig. 4 Comparison of SPOT-RNA prediction with the native structure of a Synthetic Construct, Glutamine II Riboswitch, and Hatchet Ribozyme. The
secondary structure of a synthetic construct RNA (chain H in PDB ID 6dvk), the Glutamine II Riboswitch RNA (chain A in PDB ID 6qn3), and Synthetic
Construct Hatchet Ribozyme (chain U in PDB ID 6jq6) represented by 2D diagram with canonical base pair (BP) in black color, noncanonical BP in green
color, pseduoknot BP and lone pair in blue color, and wrongly predicted BP in magenta color: a predicted structure by SPOT-RNA (at top), with 97%
precision and 77% sensitivity, as compared with the native structure (at bottom) for the Synthetic Construct RNA, b the predicted structure by SPOT-RNA
(at top) with 100% precision and 81% sensitivity, as compared with the native structure (at bottom) for the Riboswitch, c the predicted structure by SPOT-
RNA (at top) with 100% precision and 59% sensitivity, as compared with the native structure (at bottom) for the synthetic construct Hatchet ribozyme.
NATURE COMMUNICATIONS | https://doi.org/10.1038/s41467-019-13395-9 ARTICLE
NATURE COMMUNICATIONS | (2019) 10:5407 | https://doi.org/10.1038/s41467-019-13395-9 | www.nature.com/naturecommunications 9
bulge is a special case of the internal loop where one of the strands is of length zero.
A multiloop consists of a cycle of more than two unpaired strands, connected by
stems. The distribution of different secondary-structure types in TR1, VL1, and
TS1 (excluding multiplet base pairs) is shown in Supplementary Table 8. These
secondary-structure classifications were obtained by using a secondary-structure
analysis program bpRNA 34.
Deep neural networks. We employed an ensemble of deep-learning neural net-
works for pretraining. The ensemble is made of 5 top-ranked models based on their
performance on VL0 with the architecture shown in Fig. 1, similar to what was
used previously for protein contact prediction in SPOT-Contact 29.
The architecture of each model consists of ResNet blocks followed by a 2D-
BLSTM layer and a fully connected (FC) block. An initial convolution layer for pre-
activation was used before our ResNet blocks as proposed in He et al.30. The initial
convolution layer is followed by N A ResNet blocks (Block A in Fig. 1). Each ResNet
block consists of two convolutional layers with a kernel size of 3 ´ 3 and 5 ´ 5,
respectively, and a depth of DRES . The exponential linear units (ELU) 59 activation
function and the layer normalization technique 60 were used. A dropout rate of 25%
was used before each convolution layer to avoid overfitting during training 61. In
some models, we used dilated convolutions that are reported to better learn longer-
range dependencies 62. For the dilated convolutional layers, the dilation factor was
set to 2i%n, where i is the depth of the convolution layer, n is a fixed scalar, and % is
the modulus operator.
The next block in the architecture was a 2D-BLSTM 31,32. The output from the
final ResNet block was activated (with ELU) and normalized (using layer
normalization) before being given as an input to the 2D-BLSTM. The number of
nodes in each LSTM direction cell was D BL. After the 2D-BLSTM, N B FC layers
with D FC nodes were used, as per Block B in Fig. 1. The output of each FC layer was
activated with the ELU function and normalized by using the layer normalization
technique. A dropout rate of 50% was utilized for the hidden FC layers to avoid
overtraining. The final stage of the architecture consisted of an output FC layer
with one node and a sigmoidal activation function. The sigmoid function converts
the output into the probability of each nucleotide being paired with other
nucleotides. The number of outputs was equal to the number of elements in the
upper triangular matrix of size L ´ L, where L is the length of the sequence.
Each model was implemented in Google’s Tensorflow framework (v1.12) 63 and
trained by using the ADAM optimization algorithm 64 with default parameters. All
models were trained on Nvidia GTX TITAN X graphics processing unit (GPU) to
speed up training 65. We trained multiple deep-learning models, based on the
architecture shown in Fig. 1, on TR0 by performing a hyperparameter grid search
over N A, DRES , DBL , N B, and DFC . N A , DRES , DBL , N B, DFC were searched from 16
to 32, 32 to 72, 128 to 256, 0 to 4, and 256 to 512, respectively. These models were
optimized on VL0 and tested on TS0. Transfer learning was then used to further
train these models on TR1. During transfer learning, VL1 was used as the
validation set and TS1 was used as an independent test set.
Transfer learning. Transfer learning 35 involves further training a large model that
was trained on a large dataset for a specific task to some other related task with
limited data. In this project, we used our large dataset bpRNA for initial training,
and then transfer learning was employed by using the small PDB dataset as shown
in Fig. 1. All of the weights/parameters that were learnt on TR0 were retrained for
further training on TR1. During transfer learning, training and validation labels
were formatted in exactly the same way as the initial training as a 2-dimensional
(2D) L ´ L upper triangular matrix where L is the length of the RNA sequence. All
of the labels used during the transfer learning were derived from high-resolution X-
ray structures in the PDB. Some approaches in transfer learning freeze weights for
specific layers and train for other layers. Here, we trained all the weights of the
models without freezing any layer, as this provided better results. Previous work on
protein molecular recognition features (MoRFs) prediction 36 also showed that
using transfer learning by retraining through all of the weights provides a better
result than freezing some of the layers during retraining.
During transfer learning on TS1, we used the same hyperparameters (number
of layers, depth of layers, kernel size, dilation factor, and learning rate) that were
used for the TS0-trained models. All the models were validated for VL1, and based
on the performance of these models on VL1, the 5 best models were selected for the
ensemble. The parameters of these models are shown in Supplementary Table 9.
Input. The input to SPOT-RNA is an RNA sequence represented by a binary one-
hot vector of size L ´ 4, where L is the length of the RNA sequence and 4 cor-
responds to the number of base types (A, U, C, G). In one-hot encoding, a value of
1 was assigned to the corresponding base-type position in the vector and 0 else-
where. A missing or invalid sequence in residue value of −1 was assigned in one-
hot encoded vector.
This one-dimensional (L ´ 4) input feature is converted into two dimensional
(L ´ L ´ 8) by the outer concatenation function as described in RaptorX-
a
b
c
1
10 20 30 40 50 60
65 1
10 20 30 40 50 60
65
1
10 20 30
137
10 20 30
37
1
10 20 30 40 50 60 70 80 90 100 110
122 1
10 20 30 40 50 60 70 80 90 100 110
112
Fig. 5 Comparison of SPOT-RNA prediction with the native structure of a Pistol Ribozyme, Mango aptamer, and Adenovirus Virus-associated RNA. The
secondary structure of a Pistol Ribozyme (chain A and B in PDB ID 6r47), the Mango Aptamer (chain B in PDB ID 6e8u), and the adenovirus virus-
associated RNA (chain C in PDB ID 6ol3) represented by arc diagrams with canonical base pair (BP) in blue color, noncanonical, pseduoknot BP and lone
pair in green color, and wrongly predicted BP in magenta color: a predicted structure by SPOT-RNA (on left), with 93% precision and 41% sensitivity, as
compared with the native structure (on right) for the Pistol Ribozyme, b the predicted structure by SPOT-RNA (on left) with 100% precision and 26%
sensitivity, as compared with the native structure (on right) for the Mango aptamer, c the predicted structure by SPOT-RNA (on left) with 66% precision
and 60% sensitivity, as compared with the native structure (on right) for the adenovirus virus-associated RNA.
ARTICLE NATURE COMMUNICATIONS | https://doi.org/10.1038/s41467-019-13395-9
10 NATURE COMMUNICATIONS | (2019) 10:5407 | https://doi.org/10.1038/s41467-019-13395-9 | www.nature.com/naturecommunications
Contact 28. The input is standardized to have zero mean and unit variance
(according to the training data) before being fed into the model.
Output. The output of our model is a 2-dimensional (2D) L ´ L upper
triangular matrix where L is the length of the RNA sequence. This upper tri-
angular matrix represents the likelihood of each nucleotide to be paired with any
other nucleotide in a sequence. A single threshold value is used to decide
whether a nucleotide is in pair with any other nucleotides. The value of the
threshold was chosen in such a way that it optimizes the performance on the
validation set.
Performance measure. RNA secondary-structure prediction is a binary classifi-
cation problem. We used sensitivity, precision, and F1 score for performance
measure where sensitivity is the fraction of predicted base pairs in all native base
pairs (SN ¼ TP=ðTP þ FNÞ), precision is the fraction of correctly predicted base
pairs (PR ¼ TP=ðTP þ FPÞ), and F1 score is their harmonic mean
(F1 ¼ 2ðPR  SNÞ=ðPR þ SNÞ). Here, TP, FN, and FP denote true positives, false
negatives, and false positives, respectively. In addition to the above metrics that
emphasize on positives, a balanced measure, Matthews correlation coefficient
(MCC)66 was also used. MCC is calculated as
MCC ¼ TP ´ TN  FP ´ FN
ffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffi
ðTP þ FPÞðTP þ FNÞðTN þ FPÞðTN þ FNÞ
p ; ð1Þ
where TN denotes true negatives. MCC measures the correlation between the
expected class and the obtained class. Moreover, a precision-recall (sensitivity)
curve is used to compare our model with currently available RNA secondary-
structure predictors. To show the statistical significance of improvement by
SPOT-RNA over the second-best predictor, a paired t test was used on F1 score
to obtain P value 67 . The smaller the P value is, the more significant the difference
between the two predictors. As the output of the SPOT-RNA is a base-pair
probability, we can use the ensemble defect as an additional performance metric.
The ensemble defect describes the similarity between predicted base-pair
probability and target structure 68 . It can be calculated by appending an extra
column to the predicted probability matrix and target matrix for unpaired bases.
If P and S are predicted and target structures, respectively, and P′ and S′ are
predicted and target structures after appending the extra column, the ensemble
defect (ED) is given by
ED ¼ 1  1
L
X
i ¼ 1 : L
j ¼ 1 : L þ 1
P0
ij S0
ij;
ð2Þ
where L is the length of the sequence. The smaller the value of ED is, the higher
the structural similarity between predicted base-pair probability and target
structure.
Methods comparison. We compared SPOT-RNA with 12 best available pre-
dictors. We downloaded the stand-alone version of mxfold 33 (available at https://
github.com/keio-bioinformatics/mxfold), ContextFold 16 (available at https://www.
cs.bgu.ac.il/negevcb/contextfold/), CONTRAfold 14 (available at http://contra.
stanford.edu/contrafold/), Knotty 24 (available at https://github.com/HosnaJabbari/
Knotty), IPknot 23 (available at http://rtips.dna.bio.keio.ac.jp/ipknot/), RNAfold 11
(available at https://www.tbi.univie.ac.at/RNA/), ProbKnot 22 (available at http://
rna.urmc.rochester.edu/RNAstructure.html), CentroidFold 15 (available at https://
github.com/satoken/centroid-rna-package), RNAstructure 12 (available at http://
rna.urmc.rochester.edu/RNAstructure.html), RNAshapes 13 (available at https://
bibiserv.cebitec.uni-bielefeld.de/rnashapes), pkiss 13 (available at https://bibiserv.
cebitec.uni-bielefeld.de/pkiss), and CycleFold 27 (available at http://rna.urmc.
rochester.edu/RNAstructure.html). In most of the cases, we used default para-
meters for secondary-structure prediction except for pkiss. In pkiss, we used
Strategy C that is slow but thorough in comparison with Strategies A and B that are
fast but less accurate. For CONTRAfold and CentroidFold their performance
metrics are derived from their predicted base-pair probabilities with threshold
values from maximizing MCC.
Reporting summary. Further information on research design is available in
the Nature Research Reporting Summary linked to this article.
0.86 0.82 0.84 0.82 0.82 0.85 0.82 0.81 0.85 0.82 0.82
0.68
0.43
0
0.2
0.4
0.6
0.8
1
F1-score
a 0.90 0.86
0.67
0.18
0.65 0.65 0.61 0.61
0.53
0.63
0.31
0.42
0.63
0
0.2
0.4
0.6
0.8
1
F1-score
b
0.57
0.16
0.46
0.20
0.48 0.55
0.44
0.56 0.55 0.52
0.24
0.41
0.21
0
0.2
0.4
0.6
0.8
1
F1-score
c
0.74
0.59
0.79
0.56
0.68 0.64 0.63 0.63
0.49
0.64 0.64 0.59
0.69
0
0.2
0.4
0.6
0.8
1
F1-score
d
0.41 0.36 0.41
0.28
0.40 0.40 0.40 0.40 0.40 0.40 0.40 0.40 0.34
0
0.2
0.4
0.6
0.8
1
F1-score
e
0.63
0.91
0.73 0.74 0.71
0.88 0.91 0.88 0.82
0.91 0.91
0.69
0.51
0
0.2
0.4
0.6
0.8
1
F1-score
f
SPOT-RNA Knotty mxfold ContextFold CONTRAfold IPknot RNAfold ProbKnot CentroidFold RNAstructure RNAshapes pkiss CycleFold
Fig. 6 Performance comparison of all predictors on 6 recently released (after March 9, 2019) crystal structures. a F1 score of predicted structure on a
synthetic construct RNA (chain H in PDB ID 6dvk), b F1 score of predicted structure on the Glutamine II Riboswitch RNA (chain A in PDB ID 6qn3),
c F1 score of predicted structure on a synthetic construct Hatchet Ribozyme (chain U in PDB ID 6jq6), d F1 score of predicted structure on a Pistol
Ribozyme (chain A & B in PDB ID 6r47), e F1 score of predicted structure on the Mango Aptamer (chain B in PDB ID 6e8u), f F1 score of predicted
structure on the adenovirus virus-associated RNA (chain C in PDB ID 6ol3).
NATURE COMMUNICATIONS | https://doi.org/10.1038/s41467-019-13395-9 ARTICLE
NATURE COMMUNICATIONS | (2019) 10:5407 | https://doi.org/10.1038/s41467-019-13395-9 | www.nature.com/naturecommunications 11
Data availability
The data used by SPOT-RNA for initial training (bpRNA)34 and transfer learning (PDB)5
along with their annotated secondary structure are publicly available at http://sparks-lab.
org/jaswinder/server/SPOT-RNA/ and https://github.com/jaswindersingh2/SPOT-RNA.
Code availability
SPOT-RNA predictor is available as a server at http://sparks-lab.org/jaswinder/server/
SPOT-RNA/ and stand-alone software at https://github.com/jaswindersingh2/SPOT-
RNA to run on a local computer. The web server provides an arc diagram and a 2D
diagram of predicted RNA secondary structure through Visualization Applet for RNA
(VARNA) 69 tool along with a dot plot of SPOT-RNA-predicted base-pair probabilities.
Received: 12 June 2019; Accepted: 1 November 2019;
References
1. Tinoco, I. & Bustamante, C. How RNA folds. J. Mol. Biol. 293, 271–281
(1999).
2. Bevilacqua, P. C., Ritchey, L. E., Su, Z. & Assmann, S. M. Genome-wide
analysis of RNA secondary structure. Annu. Rev. Genet. 50, 235–266 (2016).
3. Tian, S. & Das, R. RNA structure through multidimensional chemical
mapping. Q. Rev. Biophys. 49, e7 (2016).
4. RNAcentral: a comprehensive database of non-coding RNA sequences.
Nucleic Acids Res. 45, D128–D134 (2016).
5. Rose, P. W. et al. The RCSB protein data bank: integrative view of protein,
gene and 3D structural information. Nucleic Acids Res. 45, D271–D281 (2016).
6. Rivas, E. The four ingredients of single-sequence RNA secondary structure
prediction. A unifying perspective. RNA Biol. 10, 1185–1196 (2013).
7. Gutell, R. R., Lee, J. C. & Cannone, J. J. The accuracy of ribosomal RNA
comparative structure models. Curr. Opin. Struct. Biol. 12, 301–310 (2002).
8. Griffiths-Jones, S., Bateman, A., Marshall, M., Khanna, A. & Eddy, S. R. Rfam:
an RNA family database. Nucleic Acids Res. 31, 439–441 (2003).
9. Zuker, M. & Stiegler, P. Optimal computer folding of large RNA sequences
using thermodynamics and auxiliary information. Nucleic Acids Res. 9,
133–148 (1981).
10. Schroeder, S. J. and Turner, D. H. Chapter 17—Optical Melting
Measurements of Nucleic Acid Thermodynamics. In Biophysical, Chemical,
and Functional Probes of RNA Structure, Interactions and Folding: Part A, vol.
468 of Methods in Enzymology, 371–387 (Academic Press, 2009).
11. Lorenz, R. et al. ViennaRNA Package 2.0. Algorithms Mol. Biol. 6, 26 (2011).
12. Reuter, J. S. & Mathews, D. H. RNAstructure: software for RNA secondary
structure prediction and analysis. BMC Bioinforma. 11, 129 (2010).
13. Janssen, S. & Giegerich, R. The RNA shapes studio. Bioinformatics 31,
423–425 (2014).
14. Do, C. B., Woods, D. A. & Batzoglou, S. CONTRAfold: RNA secondary
structure prediction without physics-based models. Bioinformatics 22,
e90–e98 (2006).
15. Sato, K., Hamada, M., Asai, K. & Mituyama, T. CentroidFold: a web server for
RNA secondary structure prediction. Nucleic Acids Res. 37, W277–W280
(2009).
16. Zakov, S., Goldberg, Y., Elhadad, M. & Ziv-ukelson, M. Rich parameterization
improves RNA structure prediction. J. Computational Biol. 18, 1525–1542
(2011).
17. Seetin, M. G. and Mathews, D. H. RNA Structure prediction: an overview of
methods. In (ed Keiler, K. C.) Bacterial Regulatory RNA: Methods and
Protocols, 99–122 (Humana Press, Totowa, NJ, 2012). https://doi.org/10.1007/
978-1-61779-949-5_8.
18. Xu, X. & Chen, S.-J. Physics-based RNA structure prediction. Biophysics Rep.
1, 2–13 (2015).
19. Nowakowski, J. & Tinoco, I. RNA structure and stability. Semin. Virol. 8,
153–165 (1997).
20. Westhof, E. & Fritsch, V. RNA folding: beyond Watson-Crick pairs. Structure
8, R55–R65 (2000).
21. Reeder, J. & Giegerich, R. Design, implementation and evaluation of a
practical pseudoknot folding algorithm based on thermodynamics. BMC
Bioinforma. 5, 104 (2004).
22. Bellaousov, S. & Mathews, D. H. ProbKnot: fast prediction of RNA secondary
structure including pseudoknots. RNA 16, 1870–1880 (2010).
23. Sato, K., Kato, Y., Hamada, M., Akutsu, T. & Asai, K. IPknot: fast and accurate
prediction of RNA secondary structures with pseudoknots using integer
programming. Bioinformatics 27, i85–i93 (2011).
24. Jabbari, H., Wark, I., Montemagno, C. & Will, S. Knotty: efficient and accurate
prediction of complex RNA pseudoknot structures. Bioinformatics 34,
3849–3856 (2018).
25. Parisien, M. & Major, F. The MC-fold and MC-sym pipeline infers RNA
structure from sequence data. Nature 452, 51–55 (2008).
26. zu Siederdissen, C. H., Bernhart, S. H., Stadler, P. F. & Hofacker, I. L. A folding
algorithm for extended RNA secondary structures. Bioinformatics 27,
i129–i136 (2011).
27. Sloma, M. F. & Mathews, D. H. Base pair probability estimates improve the
prediction accuracy of RNA non-canonical base pairs. PLOS Comput. Biol. 13,
1–23 (2017).
28. Wang, S., Sun, S., Li, Z., Zhang, R. & Xu, J. Accurate de novo prediction of
protein contact map by ultra-deep learning model. PLOS Comput. Biol. 13,
1–34 (2017).
29. Hanson, J., Paliwal, K., Litfin, T., Yang, Y. & Zhou, Y. Accurate prediction of
protein contact maps by coupling residual two-dimensional bidirectional long
short-term memory with convolutional neural networks. Bioinformatics 34,
4039–4045 (2018).
30. He, K., Zhang, X., Ren, S. and Sun, J. Identity mappings in deep residual
networks. In (eds Leibe, B., Matas, J., Sebe, N. and Welling, M.) Computer
Vision—ECCV 2016, 630–645 (Springer International Publishing, Cham,
2016).
31. Hochreiter, S. & Schmidhuber, J. Long short-term memory. Neural Comput. 9,
1735–1780 (1997).
32. Schuster, M. & Paliwal, K. K. Bidirectional recurrent neural networks. IEEE
Trans. Signal Process. 45, 2673–2681 (1997).
33. Akiyama, M., Sato, K. & Sakakibara, Y. A max-margin training of RNA
secondary structure prediction integrated with the thermodynamic model. J.
Bioinforma. Comput. Biol. 16, 1840025 (2018).
34. Danaee, P. et al. bpRNA: large-scale automated annotation and analysis of
RNA secondary structure. Nucleic Acids Res. 46, 5381–5394 (2018).
35. Pan, S. J. & Yang, Q. A Survey on Transfer Learning. IEEE Trans. Knowl. Data
Eng. 22, 1345–1359 (2010).
36. Hanson, J., Litfin, T., Paliwal, K. and Zhou, Y. Identifying molecular
recognition features in intrinsically disordered regions of proteins by transfer
learning. Bioinformatics. https://doi.org/10.1093/bioinformatics/btz691
(2019).
37. Fu, L., Niu, B., Zhu, Z., Wu, S. & Li, W. CD-HIT: accelerated for clustering the
next-generation sequencing data. Bioinformatics 28, 3150–3152 (2012).
38. Yang, Y. et al. Genome-scale characterization of RNA tertiary structures and
their functional impact by RNA solvent accessibility prediction. RNA 23,
14–22 (2017).
39. Guruge, I., Taherzadeh, G., Zhan, J., Zhou, Y. & Yang, Y. B-factor profile
prediction for RNA flexibility using support vector machines. J. Comput.
Chem. 39, 407–411 (2018).
40. Altschul, S. F. et al. Gapped BLAST and PSI-BLAST: a new generation
of protein database search programs. Nucleic Acids Res. 25, 3389–3402
(1997).
41. Liberman, J. A., Salim, M., Krucinska, J. & Wedekind, J. E. Structure of a class
II preQ1 riboswitch reveals ligand recognition by a new fold. Nat. Chem. Biol.
9, 353 EP (2013).
42. Goto-Ito, S., Ito, T., Kuratani, M., Bessho, Y. & Yokoyama, S. Tertiary
structure checkpoint at anticodon loop modification in tRNA functional
maturation. Nat. Struct. Amp; Mol. Biol. 16, 1109 EP (2009).
43. Yesselman, J. D. et al. Computational design of three-dimensional RNA
structure and function. Nat. Nanotechnol. 14, 866–873 (2019).
44. Huang, L., Wang, J., Watkins, A. M., Das, R. & Lilley, D. M. J. Structure and
ligand binding of the glutamine-II riboswitch. Nucleic Acids Res. 47,
7666–7675 (2019).
45. Zheng, L. et al. Hatchet ribozyme structure and implications for cleavage
mechanism. Proc. Natl Acad. Sci. 116, 10783–10791 (2019).
46. Wilson, T. J. et al. Comparison of the structures and mechanisms of the Pistol
and Hammerhead ribozymes. J. Am. Chem. Soc. 141, 7865–7875 (2019).
47. Trachman, R. J. et al. Structure and functional reselection of the Mango-III
fluorogenic RNA aptamer. Nat. Chem. Biol. 15, 472–479 (2019).
48. Hood, I. V. et al. Crystal structure of an adenovirus virus-associated RNA.
Nat. Commun. 10, 2871 (2019).
49. Remmert, M., Biegert, A., Hauser, A. & Söding, J. HHblits: lightning-fast
iterative protein sequence searching by HMM-HMM alignment. Nat. Methods
9, 173–175 (2011).
50. De Leonardis, E. et al. Direct-Coupling Analysis of nucleotide coevolution
facilitates RNA secondary and tertiary structure prediction. Nucleic Acids Res.
43, 10444–10455 (2015).
51. Heffernan, R. et al. Single-sequence-based prediction of protein secondary
structures and solvent accessibility by deep whole-sequence learning. J.
Comput. Chem. 39, 2210–2216 (2018).
52. Hanson, J., Paliwal, K. & Zhou, Y. Accurate single-sequence prediction of
protein intrinsic disorder by an ensemble of deep recurrent and convolutional
architectures. J. Chem. Inf. Model. 58, 2369–2376 (2018).
53. Miao, Z. et al. RNA-Puzzles Round III: 3D RNA structure prediction of five
riboswitches and one ribozyme. RNA 23, 655–672 (2017).
ARTICLE NATURE COMMUNICATIONS | https://doi.org/10.1038/s41467-019-13395-9
12 NATURE COMMUNICATIONS | (2019) 10:5407 | https://doi.org/10.1038/s41467-019-13395-9 | www.nature.com/naturecommunications
54. Rabani, M., Kertesz, M. and Segal, E. Computational prediction of RNA
structural motifs involved in post-transcriptional regulatory processes. In (ed
Gerst, J. E.) RNA Detection and Visualization: Methods and Protocols, 467–479
(Humana Press, 2011).
55. Achar, A. & Sætrom, P. RNA motif discovery: a computational overview. Biol.
Direct 10, 61 (2015).
56. Nawrocki, E. P. & Eddy, S. R. Infernal 1.1: 100-fold faster RNA homology
searches. Bioinformatics 29, 2933–2935 (2013).
57. Schlick, T. & Pyle, A. M. Opportunities and challenges in RNA structural
modeling and design. Biophys. J. 113, 225–234 (2017).
58. Lu, X.-J., Bussemaker, H. J. & Olson, W. K. DSSR: an integrated software tool for
dissecting the spatial structure of RNA. Nucleic Acids Res. 43, e142–e142 (2015).
59. Clevert, D.-A., Unterthiner, T. and Hochreiter, S. Fast and Accurate Deep
Network Learning by Exponential Linear Units (ELUs). Preprint at: https://
arxiv.org/abs/1511.07289 (2015).
60. Ba, J. L., Kiros, J. R. and Hinton, G. E. Layer Normalization. Preprint at:
https://arxiv.org/abs/1607.06450 (2016).
61. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. & Salakhutdinov, R.
Dropout: a simple way to prevent neural networks from overfitting. J. Mach.
Learn. Res. 15, 1929–1958 (2014).
62. Bai, S., Kolter, J. Z. and Koltun, V. An empirical evaluation of generic
convolutional and recurrent networks for sequence modeling. CoRR abs/
1803.01271 (2018).
63. Abadi, M. et al. TensorFlow: A System for Large-Scale Machine Learning. In
12th USENIX Symposium on Operating Systems Design and Implementation
(OSDI 16), 265–283 (USENIX Association, Savannah, GA, 2016). https://
www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi.
64. Kingma, D. P. and Ba, J. Adam: A Method for Stochastic Optimization.
Preprint at: https://arxiv.org/abs/1412.6980 (2014).
65. Oh, K.-S. & Jung, K. GPU implementation of neural networks. Pattern
Recognit. 37, 1311–1314 (2004).
66. Matthews, B. Comparison of the predicted and observed secondary structure
of T4 phage lysozyme. Biochim. Biophys. Acta 405, 442–451 (1975).
67. Lovric, M. (ed.) International Encyclopedia of Statistical Science (Springer,
Berlin Heidelberg, 2011). https://doi.org/10.1007/978-3-642-04898-2
68. Martin, J. S. Describing the structural diversity within an RNAas ensemble.
Entropy 16, 1331–1348 (2014).
69. Darty, K., Denise, A. & Ponty, Y. VARNA: Interactive drawing and editing of
the RNA secondary structure. Bioinformatics 25, 1974–1975 (2009).
Acknowledgements
This work was supported by Australia Research Council DP180102060 to Y.Z. and K.P.
and in part by National Health and Medical Research Council (1,121,629) of Australia to
Y.Z. We also gratefully acknowledge the use of the High Performance Computing Cluster
Gowonda to complete this research, and the aid of the research cloud resources provided
by the Queensland CyberInfrastructure Foundation (QCIF). We gratefully acknowledge
the support of NVIDIA Corporation with the donation of the Titan V GPU used for this
research.
Author contributions
J.S., J.H., and K.P. designed the network architectures, J.S. prepared the data, did the data
analysis, and wrote the paper. J.S. and J.H. performed the training and testing of the
algorithms. Y.Z. conceived of the study, participated in the initial design, assisted in data
analysis, and drafted the whole paper. All authors read, contributed to the discussion,
and approved the final paper.
Competing interests
The authors declare no competing interests.
Additional information
Supplementary information is available for this paper at https://doi.org/10.1038/s41467-
019-13395-9.
Correspondence and requests for materials should be addressed to K.P. or Y.Z.
Peer review information Nature Communications thanks the anonymous reviewer(s) for
their contribution to the peer review of this work.
Reprints and permission information is available at http://www.nature.com/reprints
Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims in
published maps and institutional affiliations.
Open Access This article is licensed under a Creative Commons
Attribution 4.0 International License, which permits use, sharing,
adaptation, distribution and reproduction in any medium or format, as long as you give
appropriate credit to the original author(s) and the source, provide a link to the Creative
Commons license, and indicate if changes were made. The images or other third party
material in this article are included in the article’s Creative Commons license, unless
indicated otherwise in a credit line to the material. If material is not included in the
article’s Creative Commons license and your intended use is not permitted by statutory
regulation or exceeds the permitted use, you will need to obtain permission directly from
the copyright holder. To view a copy of this license, visit http://creativecommons.org/
licenses/by/4.0/.
© The Author(s) 2019
NATURE COMMUNICATIONS | https://doi.org/10.1038/s41467-019-13395-9 ARTICLE
NATURE COMMUNICATIONS | (2019) 10:5407 | https://doi.org/10.1038/s41467-019-13395-9 | www.nature.com/naturecommunications 13
===
1
Briefings in Bioinformatics, 23(1), 2022, 1–9
https://doi.org/10.1093/bib/bbab395
Problem Solving Protocol
Prediction of RNA secondary structure including
pseudoknots for long sequences
Kengo Sato and Yuki Kato
Corresponding author: Kengo Sato, 3-14-1 Hiyoshi, Kohoku-ku, Yokohama 223-8522, Japan. Tel.: +81-45-566-1511; E-mail: satoken@bio.keio.ac.jp
Abstract
RNA structural elements called pseudoknots are involved in various biological phenomena including ribosomal frameshifts.
Because it is infeasible to construct an efficiently computable secondary structure model including pseudoknots, secondary
structure prediction methods considering pseudoknots are not yet widely available. We developed IPknot, which uses
heuristics to speed up computations, but it has remained difficult to apply it to long sequences, such as messenger RNA and
viral RNA, because it requires cubic computational time with respect to sequence length and has threshold parameters that
need to be manually adjusted. Here, we propose an improvement of IPknot that enables calculation in linear time by
employing the LinearPartition model and automatically selects the optimal threshold parameters based on the
pseudo-expected accuracy. In addition, IPknot showed favorable prediction accuracy across a wide range of conditions in
our exhaustive benchmarking, not only for single sequences but also for multiple alignments.
Key words: RNA secondary structure prediction; pseudoknots; integer programming
Introduction
Genetic information recorded in DNA is transcribed into RNA,
which is then translated into protein to fulfill its function. In
other words, RNA is merely an intermediate product for the
transmission of genetic information. This type of RNA is called
messenger RNA (mRNA). However, many RNAs that do not fit
into this framework have been discovered more recently. For
example, transfer RNA and ribosomal RNA, which play central
roles in the translation mechanism, nucleolar small RNA, which
guides the modification sites of other RNAs, and microRNA,
which regulates gene expression, have been discovered. Thus,
it has become clear that RNAs other than mRNAs are involved
in various biological phenomena. Because these RNAs do not
encode proteins, they are called non-coding RNAs. In contrast
to DNA, which forms a double-stranded structure in vivo, RNA
is often single-stranded and is thus unstable when intact. In the
case of mRNA, the cap structure at the 5′ end and the poly-A
strand at the 3′ end protect it from degradation. On the other
Kengo Sato is an assistant professor at the Department of Biosciences and Informatics at Keio University, Japan. He received his PhD in Computer Science
from Keio University, Japan, in 2003. His research interests include bioinformatics, computational linguistics and machine learning.
Yuki Kato is an assistant professor at Department of RNA Biology and Neuroscience, Graduate School of Medicine, and at Integrated Frontier Research for
Medical Science Division, Institute for Open and Transdisciplinary Research Initiatives, Osaka University, Japan. His research interests include biological
sequence analysis and single-cell genomics.
Submitted: 16 June 2021; Received (in revised form): 13 August 2021
© The Author(s) 2021. Published by Oxford University Press.
This is an Open Access article distributed under the terms of the Creative Commons Attribution Non-Commercial License (http://creativecommons.org/
licenses/by-nc/4.0/), which permits non-commercial re-use, distribution, and reproduction in any medium, provided the original work is properly cited.
For commercial re-use, please contact journals.permissions@oup.com
hand, for other RNAs that do not have such structures, single-
stranded RNA molecules bind to themselves to form three-
dimensional structures and ensure their stability. Also, as in the
case of proteins, RNAs with similar functions have similar three-
dimensional structures, and it is known that there is a strong
association between function and structure. The determination
of RNA three-dimensional (3D) structure can be performed by X-
ray crystallography, nuclear magnetic resonance, cryo-electron
microscopy, and other techniques. However, it is difficult to apply
these methods on a large scale owing to difficulties associated
with sequence lengths, resolution and cost. Therefore, RNA sec-
ondary structure, which is easier to model, is often computation-
ally predicted instead. RNA secondary structure refers to the set
of base pairs consisting of Watson–Crick base pairs (A–U, G–C)
and wobble base pairs (G–U) that form the backbone of the 3D
structure.
RNA secondary structure prediction is conventionally based
on thermodynamic models, which predict the secondary
2 Sato and Kato
Figure 1. (A) A typical psudoknot structure. The dotted lines represent base pairs.
(B) A linear presentation of the pseudoknot.
structure with the minimum free energy (MFE) among all
possible secondary structures. Popular methods based on
thermodynamic models include mfold [1], RNAfold [2], and
RNAstructure [3]. Recently, RNA secondary structure prediction
methods based on machine learning have also been developed.
These methods train alternative parameters to the thermody-
namic parameters by taking a large number of pairs of RNA
sequences and their reference secondary structures as training
data. The following methods fall under the category of methods
that use machine learning: CONTRAfold [4], ContextFold [5],
SPOT-RNA [6] and MXfold2 [7]. However, from the viewpoint
of computational complexity, most approaches do not support
the prediction of secondary structures that include pseudoknot
substructures.
Pseudoknots are one of the key topologies occurring in RNA
secondary structures. The pseudoknot structure is a structure in
which some bases inside of a loop structure form base pairs with
bases outside of the loop (e.g. Figure 1A). In other words, it is said
to have a pseudoknot structure if there exist base pairs that are
crossing each other by connecting bases of base pairs with arcs,
as shown in Figure 1B. The pseudoknot structure is known to be
involved in the regulation of translation and splicing, and riboso-
mal frameshifts [8–10]. The results of sequence analysis suggest
that the hairpin loops, which are essential building blocks of the
pseudoknots, first appeared in the evolutionary timescale [11],
and then the pseudoknots were configured, resulting in gaining
those functions. We therefore conclude that pseudoknots should
not be excluded from the modeling of RNA secondary structures.
The computational complexity required for MFE predictions
of an arbitrary pseudoknot structure has been proven to be
NP-hard [12, 13]. To address this, dynamic programming-based
methods that require polynomial time (O(n4 )–O(n6 ) for sequence
length n) to exactly compute the restricted complexity of pseu-
doknot structures [12–16] and heuristics-based fast computation
methods [17–20] have been developed.
We previously developed IPknot [21], a fast heuristic-based
method for predicting RNA secondary structures including pseu-
doknots. IPknot decomposes a secondary structure with pseudo-
knots into several pseudoknot-free substructures and predicts
the optimal secondary structure using integer programming (IP)
based on maximization of expected accuracy (MEA) under the
constraints that each substructure must satisfy. The threshold
cut technique, which is naturally derived from MEA, enables
IPknot to perform much faster calculations with nearly com-
parable prediction accuracy relative to other methods. How-
ever, because the MEA-based score uses base pairing probability
without considering pseudoknots, which requires a calculation
time that increases cubically with sequence length, it is difficult
to use for secondary structure prediction of sequences that
exceed 1000 bases, even when applying a threshold cut tech-
nique. Furthermore, as the prediction accuracy can drastically
change depending on the thresholds determined in advance for
each pseudoknot-free substructure, thresholds must be carefully
determined.
To address the limitations of IPknot, we implemented the
following two improvements to the method. The first is the
use of LinearPartition [22] to calculate base pairing probabili-
ties. LinearPartition can calculate the base pairing probability,
with linear computational complexity with respect to sequence
length, using the beam search technique. By employing the
LinearPartition model, IPknot is able to predict secondary struc-
tures while considering pseudoknots for long sequences, includ-
ing mRNA, lncRNA and viral RNA. The other improvement is
the selection of thresholds based on pseudo-expected accuracy,
which was originally developed by Hamada et al. [23]. We show
that the pseudo-expected accuracy is correlated with the ‘true’
accuracy, and by choosing thresholds for each sequence based
on the pseudo-expected accuracy, we can select a nearly optimal
secondary structure prediction.
Materials and Methods
Given an RNA sequence x = x1 · · · x n (x i ∈ {A, C, G, U}), its sec-
ondary structure is represented by a binary matrix y = (yij),
where y ij = 1 if x i and x j form a base pair and otherwise y ij = 0. Let
Y(x) be a set of all possible secondary structures of x including
pseudoknots. We assume that y ∈ Y(x) can be decomposed
into a set of pseudoknot-free substructures y(1) , y(2) , . . . , y(m) , such
that y = ∑m
p=1 y(p) . In order to guarantee the uniqueness of the
decomposition, the following conditions should be satisfied: (i)
y ∈ Y(x) should be decomposed into mutually exclusive sets; that
is, for all 1 ≤ i < j ≤ |x|, ∑m
p=1 y(p)
ij ≤ 1; (ii) every base pair in y(p)
should be pseudoknotted with at least one base pair in y(q) for
∀q < p.
Maximizing expected accuracy
One of the most promising techniques for predicting RNA sec-
ondary structures is the MEA-based approach [4, 24]. First, we
define a gain function of prediction ˆy ∈ Y(x) with regard to the
correct secondary structure y ∈ Y(x) as
Gτ (y, ˆy) = (1 − τ )TP(y, ˆy) + τ TN(y, ˆy), (1)
where TP(y, ˆy) = ∑
i<j I(y ij = 1)I( ˆy ij = 1) is the number of true
positive base pairs, TN(y, ˆy) = ∑
i<j I(y ij = 0)I( ˆy ij = 0) is the number
of true negative base pairs, and τ ∈ [0, 1] is a balancing parameter
between true positives and true negatives. Here, I(condition) is
the indicator function that takes a value of 1 or 0 depending on
whether the condition is true or false.
Our objective is to find a secondary structure that maximizes
the expectation of the gain function (1) under a given probability
distribution over the space Y(x) of pseudoknotted secondary
structures, as follows:
Ey|x [Gτ (y, ˆy)] = ∑
y∈Y(x)
Gτ (y, ˆy)P(y | x). (2)
Here, P(y | x) is a probability distribution of RNA secondary
structures including pseudoknots.
Because the calculation of the expected gain function (2)
is intractable for arbitrary pseudoknots, we approximate Eq.
(2) by the sum of the expected gain function for decomposed
pseudoknot-free substructures ˆy(1) , . . . , ˆy(m) for ˆy ∈ Y(x) such that
IPknot for long sequences 3
ˆy = ∑m
p=1 ˆy(p) , and thus, we find a pseudoknotted structure ˆy and
its decomposition ˆy(1) , . . . , ˆy(m) that maximize
m∑
p=1
∑
y∈Y′ (x)
Gτ (p) (y, ˆy(p) )P′(y | x)
=
m∑
p=1
∑
i<j
[
p ij − τ (p)
]
ˆy(p)
ij + C, (3)
where τ (p) ∈ [0, 1] is a balancing parameter between true positives
and true negatives for a level p, and C is a constant independent
of ˆy. The base pairing probability p ij is the probability that the
bases x i and x j form a base pair, which is defined as
p ij = ∑
y∈Y′ (x)
I(y ij = 1)P′(y | x). (4)
See Section S1 in Supplementary Information for the derivation.
Notably, it is no longer necessary to consider the base pairs
whose probabilities are at most the threshold τ (p) , which we refer
to as the threshold cut.
We can choose P′(y | x), a probability distribution over a set
Y′(x) of secondary structures without pseudoknots, from among
several options. Instead of using a probability distribution with
pseudoknots, we can employ a probability distribution without
pseudoknots, such as the McCaskill model [25] and the CON-
TRAfold model [4], whose computational complexity is O(|x|3 )
for time and O(|x|2 ) for space. Alternatively, the LinearPartition
model [22], which is O(|x|) in both time and space, enables us to
predict the secondary structure of sequences much longer than
1000 bases.
IP formulation
We can formulate our problem described in the previous section
as the following IP problem:
maximize
m∑
p=1
∑
i<j
[
p ij − τ (p)
]
y(p)
ij (5)
subject to y ij ∈ {0, 1} (1 ≤ ∀i < ∀ < j ≤ n), (6)
y(p)
ij ∈ {0, 1} (1 ≤ ∀p ≤ m, 1 ≤ ∀i < ∀j ≤ n), (7)
yij =
m∑
p=1
y(p)
ij (1 ≤ ∀i < ∀j ≤ n), (8)
i−1∑
h=1
y hi +
n∑
h=i+1
y ih ≤ 1 (1 ≤ ∀i ≤ n), (9)
y(p)
ij + y(p)
kl ≤ 1
(1 ≤ p ≤ m, 1 ≤ ∀i < ∀k < ∀j < ∀l ≤ n), (10)
∑
i<k<j<l
y(q)
ij + ∑
k<i′ <l<j′
y(q)
i′ j′ ≥ y(p)
kl
(1 ≤ q < p ≤ m, 1 ≤ ∀k < ∀l ≤ n). (11)
Because Equation (5) is an instantiation of the approximate
estimator (3) and the threshold cut technique is applicable to
Eq. (3), the base pairs y(p)
ij whose base pairing probabilities p ij are
larger than τ (p) need to be considered. The number of variables
y(p)
ij that should be considered is at most |x|/τ (p) because ∑
j<i p ji +
∑
j>i p ij ≤ 1 for 1 ≤ ∀i ≤ |x|. Constraint (9) means that each
base x i is paired with at most one base. Constraint (10) disallows
pseudoknots within the same level p. Constraint (11) ensures
that each base pair at level p is pseudoknotted with at least one
base pair at every lower level q < p to guarantee the uniqueness
of the decomposition y = ∑m
p=1 y(p) .
Pseudo-expected accuracy
To solve the IP problem (5)–(11), we are required to choose the
set of thresholds for each level τ (1) , . . . , τ (m) , each of which is a
balancing parameter between true positives and true negatives.
However, it is not easy to obtain the best set of τ values for
any sequence beforehand. Therefore, we employ an approach
originally proposed by Hamada et al. [23], which chooses a param-
eter set for each sequence among several parameter sets that
predicts the best secondary structure in terms of an approxima-
tion of the expected accuracy (called pseudo-expected accuracy)
and makes the prediction by the best parameter set the final
prediction.
The accuracy of a predicted RNA secondary structure ˆy
against a reference structure y is evaluated using the following
measures:
PPV(y, ˆy) = TP(y, ˆy)
TP(y, ˆy) + FP(y, ˆy) , (12)
SEN(y, ˆy) = TP(y, ˆy)
TP(y, ˆy) + FN(y, ˆy) , (13)
F(y, ˆy) = 2 · PPV(y, ˆy) · SEN(y, ˆy)
PPV(y, ˆy) + SEN(y, ˆy) . (14)
Here, TP(y, ˆy) = ∑
i<j I(y ij = 1)I( ˆy ij = 1), FP(y, ˆy) = ∑
i<j I(y ij = 0)I( ˆy ij =
1) and FN(y, ˆy) = ∑
i<j I(y ij = 1)I( ˆy ij = 0). To estimate the accuracy
of the predicted secondary structure ˆy without knowing the true
secondary structure y, we take an expectation of F(y, ˆy) over the
distribution of y:
F( ˆy) = Ey|x [F(y, ˆy)] = ∑
y∈Y(x)
F(y, ˆy)P(y | x). (15)
However, this calculation is intractable because the number of
y ∈ Y(x) increases exponentially with the length of sequence
x. Alternatively, we first calculate expected TP, FP and FN as
follows:
TP( ˆy) = Ey|x [TP(y, ˆy)] = ∑
i<j
p ij I( ˆy ij = 1), (16)
4 Sato and Kato
FP( ˆy) = Ey|x [FP(y, ˆy)] = ∑
i<j
(1 − p ij)I( ˆy ij = 1), (17)
FN( ˆy) = Ey|x[FN(y, ˆy)] = ∑
i<j
p ij I( ˆy ij = 0). (18)
Then, we approximate F by calculating Equation (14) using TP, FP,
and FN instead of TP, FP and FN, respectively.
In addition to the original pseudo-expected accuracy
described above, we introduce the pseudo-expected accuracy
for crossing base pairs to predict pseudoknotted structures.
Prediction of secondary structures including pseudoknots
depends on both the conventional prediction accuracy of base
pairs described above and the accuracy of crossing base pairs. A
crossing base pair is a base pair xi and x j such that there exists
another base pair x k and x l that is crossing the base pair xi and
x j; that is, k < i < l < j or i < k < j < l. We define the expectations
of true positives, false positives and false negatives for crossing
base pairs as follows:
TP cb( ˆy) = Ey|x [TP(cb(y), cb( ˆy))]
≈ ∑
i<k<j<l
p ij p kl I( ˆy ij = 1 ∧ ˆy kl = 1), (19)
FPcb( ˆy) = Ey|x [FP(cb(y), cb( ˆy))]
≈ ∑
i<k<j<l
(1 − p ij p kl)I( ˆy ij = 1 ∧ ˆy kl = 1), (20)
FNcb( ˆy) = Ey|x [FN(cb(y), cb( ˆy))]
≈ ∑
i<k<j<l
p ij p kl I( ˆy ij = 0 ∨ ˆy kl = 0). (21)
Here, cb(y) is an n×n binary matrix, whose (i, j)-element is yij itself
if there exists k < i < l < j or i < k < j < l such that y kl = 1, and
0 otherwise. Then, we calculate the pseudo-expected F-value for
crossing base pairs F cb using Equation (14) with TP cb, FPcb and FNcb
instead of TP, FP and FN, respectively. Equations (19)–(21) require
O(n4 ) for naive calculations, but can be reduced to acceptable
computational time by utilizing the threshold cut technique.
We predict secondary structures ˆy t (t = 1, . . . , l) for several
threshold parameters {(τ (1)
t , . . . , τ (m)
t ) | t = 1, . . . , l}. Then, we cal-
culate their pseudo-expected accuracy F( ˆy t) + F cb( ˆy t) and choose
the secondary structure ˆy t that maximizes the pseudo-expected
accuracy as the final prediction.
Common secondary structure prediction
The average of the base pairing probability matrices for each
sequence in an alignment has been used to predict the com-
mon secondary structure for the alignment [26, 27]. Let A be
an alignment of RNA sequences that contains k sequences and
|A| denote the number of columns in A. We calculate the base
pairing probabilities of an individual sequence x ∈ A as
p(x)
ij = ∑
y∈Y(x)
I(y ij = 1)P(y | x). (22)
The averaged base pairing probability matrix is defined as
p(A)
ij = 1
k
∑
x∈A
p(x)
ij . (23)
The common secondary structure of the alignment A can be
calculated in the same way by replacing p ij in Equations (5)
with p(A)
ij . While the common secondary structure prediction
based on the average base pairing probability matrix has been
implemented in the previous version of IPknot [21], the present
version employs the LinearPartition model, which enables the
calculation linearly with respect to the alignment length.
Implementation
Our method has been implemented as the newest version of a
program called IPknot. In addition to the McCaskil model [25] and
CONTRAfold model [4], which were already integrated into the
previous version of IPknot, the LinearPartition model [22] is also
supported as a probability distribution for secondary structures.
To solve IP problems, the GNU Linear Programming Kit (GLPK;
http://www.gnu.org/software/glpk/), Gurobi Optimizer (http://gu
robi.com/) or IBM CPLEX Optimizer (https://www.ibm.com/ana
lytics/cplex-optimizer) can be employed.
Datasets
To evaluate our algorithm, we performed computational
experiments on several datasets. We employed RNA sequences
extracted from the bpRNA-1m dataset [28], which is based on
Rfam 12.2 [29], and the comparative RNA web dataset [30] with
2588 families. In addition, we built a dataset that includes
families from the most recent Rfam database, Rfam 14.5 [31].
Since the release of Rfam 12.2, the Rfam project has actively
collected about 1400 RNA families, including families detected
by newly developed techniques. We extracted these newly
discovered families. To limit bias in the training data, sequences
with higher than 80% sequence identity with the sequence
subsets S-Processed-TRA from RNA STRAND [32] and TR0 from
bpRNA-1m [28], which are the training datasets for CONTRAfold
and SPOT-RNA, respectively, were removed using CD-HIT-EST-2D
[33]. We then removed redundant sequences using CD-HIT-EST
[33], with a cutoff threshold of 80% sequence identity.
For the prediction of common secondary structures, the
sequence selected by the above method was used as a seed, and
1–9 sequences of the same Rfam family and with high sequence
identity (≥ 80%) with the seed sequence were randomly
selected to create an alignment. Common secondary structure
prediction was performed on the reference alignments from
Rfam and the alignments calculated by MAFFT [34]. Because
there are sequences from bpRNA-1m that do not have Rfam
reference alignments, only sequences from Rfam 14.5 were
tested for common secondary structure prediction. To capture
the accuracy of the common secondary structure prediction, the
accuracy for the seed sequence is shown.
A summary of the dataset created and utilized is shown in
Table 1.
Results
Effectiveness of pseudo-expected accuracy
First, to show the effectiveness of the automatic selection from
among thresholds τ (1) , . . . , τ (m) based on the pseudo-expected
IPknot for long sequences 5
Table 1. Datasets used in our experiments. Each element of the table shows the number of sequences
Pseudoknot-free Pseudoknotted
Short Medium Long Short Medium Long
Length (nt) (12–150) (151–500) (501–4381) (12–150) (151–500) (501–4381)
(Single)
bpRNA-1m 1971 514 420 125 162 245
Rfam 14.5 6299 723 9 1692 477 151
(Multiple)
Rfam 14.5 5118 554 4 1692 477 151
Figure 2. PPV–SEN plot of IPknot and ThreshKnot for short RNA sequences (≤ 150
nt).
accuracy, Figure 2 and Table S1 in Supplementary Information
show the prediction accuracy on the dataset of short sequences
(≤ 150 nt) using automatic selection and manual selection of the
threshold τ values. For IPknot, we fixed the number of decom-
posed sets of secondary substructures m = 2, and varied thresh-
old parameters τ values for base pairing probability in such a
way that {(τ (1) , τ (2) ) | τ (p) = 2−t, p = 1, 2, t = 1, 2, 3, 4, τ (1) ≥ τ (2)}.
In IPknot with pseudo-expected accuracy, the best secondary
structure in the sense of pseudo-expected F is selected from the
same range of (τ (1) , τ (2) ) for each sequence. For these variants of
IPknot, the LinearPartition model with CONTRAfold parameters
(LinearPartition-C) was used to calculate base pairing probabili-
ties. In addition, we compared the prediction accuracy of IPknot
with that of ThreshKnot [35], which also calculates base pairing
probabilities using LinearPartition-C. We used {2−t | t = 1, 2, 3, 4}∪
{0.3} as the threshold parameter θ for ThreshKnot because the
default threshold parameter of ThreshKnot is θ = 0.3. IPknot
with threshold parameters of τ (1) = 0.125 and τ (2) = 0.125 had
the highest prediction accuracy of F = 0.659. IPknot with pseudo-
expected accuracy has a prediction accuracy of F = 0.658, which
is comparable to the highest accuracy obtained. ThreshKnot
with a threshold of 0.25 has an accuracy of F = 0.656, which is
also comparable to the best accuracy obtained.
The pseudo-expected F-value and “true” F-value are relatively
highly correlated (Spearman correlation coefficient ρ = 0.639),
indicating that the selection of predicted secondary structure
using pseudo-expected accuracy works well.
While the accuracy of the prediction of the entire secondary
structure has already been considered, as shown in Figure 2,
for the prediction of secondary structures with pseudoknots,
it is necessary to evaluate the prediction accuracy focused on
the crossing base pairs. In terms of prediction accuracy limited
to only crossing base pairs, IPknot with pseudo-expected accu-
racy yielded Fcb = 0.258, while the highest accuracy achieved
by IPknot with the threshold parameters and ThreshKnot was
considerably lower at F cb = 0.161 and 0.057, respectively (See
Table S1 in Supplementary Information). We can observe the
similar tendency to the above in Figures S1 and S2, and Tables
S2 and S3 in Supplementary Information for medium (151–
500 nt) and long (> 500 nt) sequences. These results suggest
that prediction of crossing base pairs is improved by selecting
the predicted secondary structure while considering both the
pseudo-expected accuracy of the entire secondary structure and
the pseudo-expected accuracy of the crossing base pairs.
Comparison with previous methods for single RNA
sequences
Using our dataset, we compared our algorithm with several pre-
vious methods that can predict pseudoknots, including Thresh-
Knot utilizing LinearPartition (committed on 17 March 2021)
[22], Knotty (committed on Mar 28, 2018) [22] and SPOT-RNA
(committed on 1 April 2021) [6], and those that can predict
only pseudoknot-free structures, including CONTRAfold (ver-
sion 2.02) [4] and RNAfold in the ViennaRNA package (version
2.4.17) [22]. IPknot has several options for the calculation model
for base pairing probabilities, namely the LinearPartition model
with CONTRAfold parameters (LinearPartition-C), the LinearPar-
tition model with ViennaRNA parameters (LinearPartition-V),
the CONTRAfold model and the ViennaRNA model. In addition,
ThreshKnot has two possible LinearPartition models for calcu-
lating base pairing probabilities. The other existing methods
were tested using the default settings.
We evaluated the prediction accuracy according to the F-
value as defined by Equation (14) for pseudoknot-free sequences
(PKF in Table 2), pseudoknotted sequences (PK in Table 2) and
only crossing base pairs (CB in Table 2) by stratifying sequences
by length: short (12–150 nt), medium (151–500 nt) and long
(500–4381 nt).
For short sequences, SPOT-RNA archived high accuracy, espe-
cially for pseudoknotted sequences. However, a large difference
in accuracy between the bpRNA-1m-derived and Rfam 14.5-
derived sequences can be observed for SPOT-RNA compared with
the other methods (See Tables S4–S9 in Supplementary Informa-
tion). Notably, bpRNA-1m contains many sequences in the same
6 Sato and Kato
Table 2. A comparison of prediction accuracies (F-values) by sequence length for each method
Length Short (12–150 nt) Medium (151–500 nt) Long (501–4381 nt)
PKF PK CB PKF PK CB PKF PK CB
IPknot
(LinearPartition-C) 0.681 0.552 0.258 0.492 0.482 0.128 0.433 0.428 0.061
(LinearPartition-V) 0.669 0.499 0.143 0.478 0.461 0.091 0.380 0.370 0.038
(CONTRAfold) 0.678 0.550 0.259 0.495 0.505 0.154 0.426 0.413 0.066
(ViennaRNA) 0.669 0.500 0.144 0.480 0.461 0.091 0.212 0.317 0.041
ThreshKnot
(LinearPartition-C) 0.681 0.501 0.027 0.493 0.475 0.019 0.439 0.431 0.008
(LinearPartition-V) 0.669 0.484 0.033 0.481 0.456 0.026 0.383 0.372 0.014
Knotty 0.641 0.550 0.315 — — — — — —
SPOT-RNA 0.658 0.621 0.322 0.462 0.479 0.127 — — —
CONTRAfold 0.682 0.519 0.000 0.500 0.497 0.000 0.425 0.415 0.000
RNAfold 0.668 0.472 0.000 0.474 0.442 0.000 0.361 0.347 0.000
PKF, F-value for pseudoknot-free sequences; PK, F-value for pseudoknotted sequences; CB, F-value of crossing base pairs.
Figure 3. Computational time of each method as a function of sequence length. For SPOT-RNA with GPGPU, we used a Linux workstation with Intel Xeon Gold 6136
and NVIDIA Tesla V100. All other computations were performed on Linux workstations with AMD EPYC 7702. For IPknot, we employed IBM CPLEX Optimizer as the IP
solver.
family as the SPOT-RNA training data, and although we per-
formed filtering based on sequence identity, there is still a con-
cern of overfitting. Knotty can predict structures including pseu-
doknots with an accuracy comparable to that of SPOT-RNA, but
as shown in Figure 3, it can perform secondary structure predic-
tion for only short sequences, owing to its huge computational
complexity. Comparing IPknot using the LinearPartition-C and -
V models with its counterparts, the original CONTRAfold model
and ViennaRNA model achieved comparable accuracy. However,
because the computational complexity of the original models
is cubic with respect to sequence length, the computational
time of the original models increases rapidly as the sequence
length exceeds 1500 bases. On the other hand, the computational
complexity of the LinearPartition models is linear with respect
to sequence length, so the base pairing probabilities can be
quickly calculated even when the sequence length exceeds 4000
bases. In addition to calculating the base pairing probabilities, IP
calculations are required, but because the number of variables
and constraints to be considered can be greatly reduced using
the threshold cut technique, the overall execution time is not
significantly affected if the sequence length is several thousand
bases. Because ThreshKnot, like IPknot, uses the LinearPartition
model, it is able to perform fast secondary structure prediction
even for long sequences. However, for the prediction accuracy of
crossing base pairs, ThreshKnot is even less accurate.
Pseudoknots are found not only in cellular RNAs but also in
viral RNAs, performing a variety of functions [8]. Tables S10–S11
in Supplementary Information show the results of the secondary
structure prediction by separating the datasets into cellular
RNAs and viral RNAs, indicating that there is no significant
difference in the prediction accuracy between cellular RNAs and
viral RNAs.
Prediction of common secondary structures
with pseudoknots
Few methods exist that can perform prediction of common
secondary structures including pseudoknots for sequence
alignments longer than 1000 bases. Table 3 and Tables S12–S20 in
Supplementary Information compare the accuracy of IPknot that
employs the LinearPartition model, and RNAalifold in the Vien-
naRNA package. We performed common secondary structure
prediction for the Rfam reference alignment and the alignment
calculated by MAFFT, as well as secondary structure prediction
of single sequences only for the seed sequence included in
the alignment, and evaluated the prediction accuracy for the
seed sequence. In most cases, the prediction accuracy improved
as the quality of the alignment increased (Single < MAFFT
< Reference). IPknot predicts crossing base pairs based on
IPknot for long sequences 7
Table 3. A comparison of prediction accuracies (F-values) of common secondary structure prediction by sequence alignments for each method
Reference MAFFT Single
PKF PK CB PKF PK CB PKF PK CB
IPknot
(LinearPartition-C) 0.765 0.616 0.220 0.732 0.585 0.218 0.718 0.548 0.227
(LinearPartition-V) 0.761 0.565 0.177 0.729 0.529 0.165 0.714 0.494 0.124
RNAalifold 0.804 0.611 0.000 0.745 0.540 0.000 0.716 0.474 0.000
PKF, F-value for pseudoknot-free sequences; PK, F-value for pseudoknotted sequences; CB, F-value of crossing base pairs.
pseudo-expected accuracy, whereas RNAalifold is unable to
predict pseudoknots.
Discussion
Both IPknot and ThreshKnot use the LinearPartition model to
calculate base pairing probabilities, and then perform secondary
structure prediction using different strategies. ThreshKnot pre-
dicts the base pairs x i and x j that are higher than a predeter-
mined threshold θ and have the largest p ij in terms of both i
and j. IPknot predicts the pseudoknot structure with multiple
thresholds τ (1) , . . . , τ (m) in a hierarchical manner based on IP (5)–
(11), and then carefully selects from among these thresholds
based on pseudo-expected accuracy. Because both the pseudo-
expected accuracy of the entire secondary structure as well as
the pseudo-expected accuracy of the crossing base pairs are
taken into account, the prediction accuracy of the pseudoknot
structure is inferred to be enhanced in IPknot.
Because the LinearPartition model uses the same parameters
as the CONTRAfold and ViennaRNA packages, there is no sig-
nificant difference in accuracy between using LinearPartition-C
and -V and their counterparts, the CONTRAfold and ViennaRNA
models. It has been shown that LinearPartition has no significant
effect on accuracy even though it ignores structures whose
probability is extremely low owing to its use of beam search,
which makes the calculation linear with respect to the sequence
length [22]. The LinearPartition model enables IPknot to perform
secondary structure prediction including pseudoknots of very
long sequences, such as mRNA, lncRNA, and viral RNA.
SPOT-RNA [6], which uses deep learning, showed notable
prediction accuracy in our experiments, especially in short
sequences containing pseudoknots, with F-value of 0.621, which
is superior to other methods. However, SPOT-RNA requires
considerable computing resources such as GPGPU and long
computational time. Furthermore, SPOT-RNA showed a large
difference in prediction accuracy between sequences that are
close to the training data and those that are not compared with
the other methods. Therefore, the situations in which SPOT-RNA
can be used are considered to be limited. In contrast, IPknot
uses CONTRAfold parameters, which is also based on machine
learning, but we did not observe as much overfitting with IPknot
as with SPOT-RNA.
Approaches that provide an exact solution for limited-
complexity pseudoknot structures, such as PKNOTS [14], pknot-
sRG [15], and Knotty [16], can predict pseudoknot structures
with high accuracy but demand a huge amount of computation
O(n4 )–O(n6 ) for sequence length n, limiting secondary structure
prediction to sequences only up to about 150 bases. On the
other hand, IPknot predicts the pseudoknot structure using
a fast computational heuristic-based method with the linear
time computation, which does not allow us to find an exact
solution. Instead, IPknot improves the prediction accuracy of the
pseudoknot structure by choosing the best solution from among
several solutions based on the pseudo-expected accuracy.
IPknot uses pseudoknot-free algorithms, such as CON-
TRAfold and ViennaRNA, to calculate base pairing probabilities,
and its prediction accuracy of the resulting secondary structure
strongly depends on the algorithm used to calculate base pairing
probabilities. Therefore, we can expect to improve the prediction
accuracy of IPknot by calculating the base pairing probabilities
based on state-of-the-art pseudoknot-free secondary structure
prediction methods such as MXfold2 [7].
It is well known that common secondary structure prediction
from sequence alignments improves the accuracy of secondary
structure prediction. However, among the algorithms for predict-
ing common secondary structure including pseudoknots, only
IPknot can deal with sequence alignments longer than several
thousand bases. In the RNA virus SARS-CoV-2, programmed -1
ribosomal frameshift (-1 PRF), in which a pseudoknot structure
plays an important role, has been identified and is attracting
attention as a drug target [10]. Because many closely related
strains of SARS-CoV-2 have been sequenced, it is expected that
structural motifs including pseudoknots, such as -1 PRF, can be
found by predicting the common secondary structure from the
alignment.
Conclusions
We have developed an improvement to IPknot that enables cal-
culation in linear time by employing the LinearPartition model
and automatically selects the optimal threshold parameters
based on the pseudo-expected accuracy. LinearPartition can cal-
culate the base pairing probability with linear computational
complexity with respect to the sequence length. By employing
LinearPartition, IPknot is able to predict the secondary structure
considering pseudoknots for long sequences such as mRNA,
lncRNA, and viral RNA. By choosing the thresholds for each
sequence based on the pseudo-expected accuracy, we can select
a nearly optimal secondary structure prediction.
The LinearPartition model realized the predictiction of sec-
ondary structures considering pseudoknots for long sequences.
However, the prediction accuracy is still not sufficiently high,
especially for crossing base pairs. We expect that by learn-
ing parameters from long sequences [36], we can achieve high
accuracy even for long sequences.
Key Points
• We reduced the computational time required by
IPknot from cubic to linear with respect to the
sequence length by employing the LinearPartition
model and enabled the secondary structure prediction
8 Sato and Kato
including pseudoknots for long RNA sequences such
as mRNA, lncRNA, and viral RNA.
• We improved the accuracy of secondary structure pre-
diction including pseudoknots by introducing pseudo-
expected accuracy not only for the entire base pairs
but also for crossing base pairs.
• To the best of our knowledge, IPknot is the only
method that can perform RNA secondary structure
prediction including pseudoknot not only for very
long single sequence, but also for very long sequence
alignments.
Supplementary Data
Supplementary data are available online at Briefings in
Bioinformatics.
Availability
The IPknot source code is freely available at https://githu
b.com/satoken/ipknot. IPknot is also available for use from
a web server at http://rtips.dna.bio.keio.ac.jp/ipknot++/. The
datasets used in our experiments are available at https://
doi.org/10.5281/zenodo.4923158.
Author contributions statement
K.S. conceived the study, implemented the algorithm, col-
lected the datasets, conducted experiments, and drafted
the manuscript. K.S. and Y.K. discussed the algorithm and
designed the experiments. All authors read, contributed to
the discussion of and approved the final manuscript.
Funding
This work was partially supported by a Grant-in-Aid for
Scientific Research (B) (No. 19H04210) and Challenging
Exploratory Research (No. 19K22897) from the Japan Society
for the Promotion of Science (JSPS) to K.S. and a Grant-in-
Aid for Scientific Research (C) (Nos. 18K11526 and 21K12109)
from JSPS to Y.K.
Acknowledgments
The supercomputer system used for this research was made
available by the National Institute of Genetics, Research
Organization of Information and Systems.
References
1. Zuker M. Mfold web server for nucleic acid folding and
hybridization prediction. Nucleic Acids Res 2003;31(13):3406–
15.
2. Lorenz R, Bernhart SH, Höner Zu Siederdissen C, et al. Vien-
naRNA package 2.0. Algorithms Mol Biol 2011;6:26.
3. Reuter JS, Mathews DH. RNAstructure: software for RNA sec-
ondary structure prediction and analysis. BMC Bioinformatics
2010;11:129.
4. Do CB, Woods DA, Batzoglou S. CONTRAfold: RNA secondary
structure prediction without physics-based models. Bioinfor-
matics 2006;22(14):e90–8.
5. Zakov S, Goldberg Y, Elhadad M, et al. Rich parameteri-
zation improves RNA structure prediction. J Comput Biol
2011;18(11):1525–42.
6. Singh J, Hanson J, Paliwal K, et al. RNA secondary struc-
ture prediction using an ensemble of two-dimensional
deep neural networks and transfer learning. Nat Commun
2019;10(1):5407.
7. Sato K, Akiyama M, Sakakibara Y. RNA secondary struc-
ture prediction using deep learning with thermodynamic
integration. Nat Commun 2021;12(1):941.
8. Brierley I, Pennell S, Gilbert RJC. Viral RNA pseudoknots:
versatile motifs in gene expression and replication. Nat Rev
Microbiol 2007;5(8):598–610.
9. Staple DW, Butcher SE. Pseudoknots: RNA structures with
diverse functions. PLoS Biol 2005;3(6):e213.
10. Kelly JA, Olson AN, Neupane K, et al. Structural and
functional conservation of the programmed -1 ribosomal
frameshift signal of SARS coronavirus 2 (SARS-CoV-2). J Biol
Chem 2020;295(31):10741–8.
11. Trifonov EN, Gabdank I, Barash D, et al. Primordia vita.
deconvolution from modern sequences. Orig Life Evol Biosph
December 2006;36(5–6):559–65.
12. Akutsu T. Dynamic programming algorithms for RNA sec-
ondary structure prediction with pseudoknots. Discrete Appl
Math 2000;104(1):45–62.
13. Lyngsø RB, Pedersen CN. RNA pseudoknot prediction in
energy-based models. J Comput Biol 2000;7(3–4):409–27.
14. Rivas E, Eddy SR. A dynamic programming algorithm for
RNA structure prediction including pseudoknots. J Mol Biol
1999;285(5):2053–68.
15. Reeder J, Giegerich R. Design, implementation and evalua-
tion of a practical pseudoknot folding algorithm based on
thermodynamics. BMC Bioinformatics 2004;5:104.
16. Jabbari H, Wark I, Montemagno C, et al. Knotty: efficient and
accurate prediction of complex RNA pseudoknot structures.
Bioinformatics 2018;34(22):3849–56.
17. Ruan J, Stormo GD, Zhang W. An iterated loop matching
approach to the prediction of RNA secondary structures with
pseudoknots. Bioinformatics 2004;20(1):58–66.
18. Ren J, Rastegari B, Condon A, et al. HotKnots: heuristic pre-
diction of RNA secondary structures including pseudoknots.
RNA 2005;11(10):1494–504.
19. Chen X, He S-M, Bu D, et al. FlexStem: improving predictions
of RNA secondary structures with pseudoknots by reducing
the search space. Bioinformatics 2008;24(18):1994–2001.
20. Bellaousov S, Mathews D. H. ProbKnot: fast prediction
of RNA secondary structure including pseudoknots. RNA
2010;16(10):1870–80.
21. Sato K, Kato Y, Hamada M, et al. IPknot: fast and accurate pre-
diction of RNA secondary structures with pseudoknots using
integer programming. Bioinformatics 2011;27(13):i85–93.
22. Zhang H, Zhang L, Mathews DH, et al. LinearPartition:
linear-time approximation of RNA folding partition
function and base-pairing probabilities. Bioinformatics
2020;36(Supplement_1):i258–67.
23. Hamada M, Sato K, Asai K. Prediction of RNA secondary
structure by maximizing pseudo-expected accuracy. BMC
Bioinformatics 2010;11:586.
24. Hamada M, Kiryu H, Sato K, et al. Prediction of RNA secondary
structure using generalized centroid estimators. Bioinformat-
ics 2009;25(4):465–73.
IPknot for long sequences 9
25. McCaskill JS. The equilibrium partition function and base
pair binding probabilities for RNA secondary structure.
Biopolymers 1990;29(6–7):1105–19.
26. Kiryu H, Kin T, Asai K. Robust prediction of consensus sec-
ondary structures using averaged base pairing probability
matrices. Bioinformatics 2007;23(4):434–41.
27. Hamada M, Sato K, Asai K. Improving the accuracy of predict-
ing secondary structure for aligned RNA sequences. Nucleic
Acids Res 2011;39(2):393–402.
28. Danaee P, Rouches M, Wiley M, et al. bpRNA: large-scale auto-
mated annotation and analysis of RNA secondary structure.
Nucleic Acids Res 2018;46(11):5381–94.
29. Nawrocki EP, Burge SW, Bateman A, et al. Rfam 12.0:
updates to the RNA families database. Nucleic Acids Res
2015;43(Database issue):D130–7.
30. Cannone JJ, Subramanian S, Schnare MN, et al. The compara-
tive RNA web (CRW) site: an online database of comparative
sequence and structure information for ribosomal, intron,
and other RNAs. BMC Bioinformatics 2002;3:2.
31. Kalvari I, Nawrocki EP, Ontiveros-Palacios N, et al.
Rfam 14: expanded coverage of metagenomic, viral
and microRNA families. Nucleic Acids Res 2021;49(D1):
D192–200.
32. Andronescu M, Bereg V, Hoos HH, et al. RNA STRAND: the
RNA secondary structure and statistical analysis database.
BMC Bioinformatics 2008;9:340.
33. Fu L, Niu B, Zhu Z, et al. CD-HIT: accelerated for clus-
tering the next-generation sequencing data. Bioinformatics
2012;28(23):3150–2.
34. Katoh K, Standley DM. MAFFT multiple sequence align-
ment software version 7: improvements in performance and
usability. Mol Biol Evol 2013;30(4):772–80.
35. Zhang L, Zhang H, Mathews DH, et al. ThreshKnot: Thresh-
olded ProbKnot for improved RNA secondary structure pre-
diction. arXiv:1912.12796v1 [q-bio.BM] 2019.
36. Rezaur Rahman F, Zhang H, Huang L. Learning
to fold RNAs in linear time. bioRxiv. 2019.
https://doi.org/10.1101/852871.
===
Published as a conference paper at ICLR 2020
RNA SECONDARY STRUCTURE PREDICTION
BY LEARNING UNROLLED ALGORITHMS
Xinshi Chen1∗ , Yu Li2 ∗, Ramzan Umarov2, Xin Gao2,†, Le Song1,3,†
1Georgia Tech 2KAUST 3Ant Financial
xinshi.chen@gatech.edu
{yu.li;ramzan.umarov;xin.gao}@kaust.edu.sa
lsong@cc.gatech.edu
ABSTRACT
In this paper, we propose an end-to-end deep learning model, called E2Efold, for
RNA secondary structure prediction which can effectively take into account the
inherent constraints in the problem. The key idea of E2Efold is to directly pre-
dict the RNA base-pairing matrix, and use an unrolled algorithm for constrained
programming as the template for deep architectures to enforce constraints. With
comprehensive experiments on benchmark datasets, we demonstrate the superior
performance of E2Efold: it predicts significantly better structures compared to
previous SOTA (especially for pseudoknotted structures), while being as efficient
as the fastest algorithms in terms of inference time.
1 INTRODUCTIONG G G A A A C G U U C C G
G
G 1
G 1
A 1
A
A
C
G
U
U 1
C 1
C 1
G
matrix representation
Figure 1: Graph and matrix represen-
tations of RNA secondary structure.
Ribonucleic acid (RNA) is a molecule playing essential roles
in numerous cellular processes and regulating expression of
genes (Crick, 1970). It consists of an ordered sequence of nu-
cleotides, with each nucleotide containing one of four bases:
Adenine (A), Guanine (G), Cytosine (C) and Uracile (U). This
sequence of bases can be represented as
x := (x1, . . . , xL) where xi ∈ {A, G, C, U },
which is known as the primary structure of RNA. The bases
can bond with one another to form a set of base-pairs, which
defines the secondary structure. A secondary structure can be
represented by a binary matrix A∗ where A∗
ij = 1 if the i, j-th
bases are paired (Fig 1). Discovering the secondary structure of RNA is important for understanding
functions of RNA since the structure essentially affects the interaction and reaction between RNA
and other cellular components. Although secondary structure can be determined by experimental
assays (e.g. X-ray diffraction), it is slow, expensive and technically challenging. Therefore, compu-
tational prediction of RNA secondary structure becomes an important task in RNA research and is
useful in many applications such as drug design (Iorns et al., 2007).(ii) Pseudo-knot(i) Nested Structure
Figure 2: Nested and non-nested structures.
Research on computational prediction of RNA secondary
structure from knowledge of primary structure has been
carried out for decades. Most existing methods assume
the secondary structure is a result of energy minimiza-
tion, i.e., A∗ = arg minA Ex(A). The energy function
is either estimated by physics-based thermodynamic ex-
periments (Lorenz et al., 2011; Bellaousov et al., 2013;
Markham & Zuker, 2008) or learned from data (Do et al.,
2006). These approaches are faced with a common problem that the search space of all valid sec-
ondary structures is exponentially-large with respect to the length L of the sequence. To make the
minimization tractable, it is often assumed the base-pairing has a nested structure (Fig 2 left), and
∗Equal contribution. †Co-corresponding.
1
arXiv:2002.05810v1 [cs.LG] 13 Feb 2020
Published as a conference paper at ICLR 2020
the energy function factorizes pairwisely. With this assumption, dynamic programming (DP) based
algorithms can iteratively find the optimal structure for subsequences and thus consider an enormous
number of structures in time O(L3).
Although DP-based algorithms have dominated RNA structure prediction, it is notable that they
restrict the search space to nested structures, which excludes some valid yet biologically important
RNA secondary structures that contain ‘pseudoknots’, i.e., elements with at least two non-nested
base-pairs (Fig 2 right). Pseudoknots make up roughly 1.4% of base-pairs (Mathews & Turner,
2006), and are overrepresented in functionally important regions (Hajdin et al., 2013; Staple &
Butcher, 2005). Furthermore, pseudoknots are present in around 40% of the RNAs. They also assist
folding into 3D structures (Fechter et al., 2001) and thus should not be ignored. To predict RNA
structures with pseudoknots, energy-based methods need to run more computationally intensive
algorithms to decode the structures.
In summary, in the presence of more complex structured output (i.e., pseudoknots), it is challenging
for energy-based approaches to simultaneously take into account the complex constraints while be-
ing efficient. In this paper, we adopt a different viewpoint by assuming that the secondary structure
is the output of a feed-forward function, i.e., A∗ = Fθ (x), and propose to learn θ from data in
an end-to-end fashion. It avoids the second minimization step needed in energy function based ap-
proach, and does not require the output structure to be nested. Furthermore, the feed-forward model
can be fitted by directly optimizing the loss that one is interested in.
Despite the above advantages of using a feed-forward model, the architecture design is challenging.
To be more concrete, in the RNA case, Fθ is difficult to design for the following reasons:
(i) RNA secondary structure needs to obey certain hard constraints (see details in Section 3),
which means certain kinds of pairings cannot occur at all (Steeg, 1993). Ideally, the output of
Fθ needs to satisfy these constraints.
(ii) The number of RNA data points is limited, so we cannot expect that a naive fully connected
network can learn the predictive information and constraints directly from data. Thus, inductive
biases need to be encoded into the network architecture.
(iii) One may take a two-step approach, where a post-processing step can be carried out to enforce
the constraints when Fθ predicts an invalid structure. However, in this design, the deep network
trained in the first stage is unaware of the post-processing stage, making less effective use of
the potential prior knowledge encoded in the constraints.All Binary Structures
Output Space of E2Efold
*All Valid Structures*
Nested Structures
with constraints
(DP applicable)
Figure 3: Output space of E2Efold.
In this paper, we present an end-to-end deep learning solution
which integrates the two stages. The first part of the archi-
tecture is a transformer-based deep model called Deep Score
Network which represents sequence information useful for
structure prediction. The second part is a multilayer network
called Post-Processing Network which gradually enforces the
constraints and restrict the output space. It is designed based
on an unrolled algorithm for solving a constrained optimiza-
tion. These two networks are coupled together and learned
jointly in an end-to-end fashion. Therefore, we call our model E2Efold.
By using an unrolled algorithm as the inductive bias to design Post-Processing Network, the output
space of E2Efold is constrained (illustrated in Fig 3), which makes it easier to learn a good model
in the case of limited data and also reduces the overfitting issue. Yet, the constraints encoded in
E2Efold are flexible enough such that pseudoknots are not excluded. In summary, E2Efold strikes a
nice balance between model biases for learning and expressiveness for valid RNA structures.
We conduct extensive experiments to compare E2Efold with state-of-the-art (SOTA) methods on
several RNA benchmark datasets, showing superior performance of E2Efold including:
• being able to predict valid RNA secondary structures including pseudoknots;
• running as efficient as the fastest algorithm in terms of inference time;
• producing structures that are visually close to the true structure;
• better than previous SOTA in terms of F1 score, precision and recall.
Although in this paper we focus on RNA secondary structure prediction, which presents an impor-
tant and concrete problem where E2Efold leads to significant improvements, our method is generic
2
Published as a conference paper at ICLR 2020
and can be applied to other problems where constraints need to be enforced or prior knowledge is
provided. We imagine that our design idea of learning unrolled algorithm to enforce constraints can
also be transferred to problems such as protein folding and natural language understanding problems
(e.g., building correspondence structure between different parts in a document).
2 RELATED WORK
Classical RNA folding methods identify candidate structures for an RNA sequence energy min-
imization through DP and rely on thousands of experimentally-measured thermodynamic parame-
ters. A few widely used methods such as RNAstructure (Bellaousov et al., 2013), Vienna RNAfold
(Lorenz et al., 2011) and UNAFold (Markham & Zuker, 2008) adpoted this approach. These meth-
ods typically scale as O(L3) in time and O(L2) in storage (Mathews, 2006), making them slow for
long sequences. A recent advance called LinearFold (Huang et al., 2019) achieved linear run time
O(L) by applying beam search, but it can not handle pseudoknots in RNA structures. The prediction
of lowest free energy structures with pseudoknots is NP-complete (Lyngsø & Pedersen, 2000), so
pseudoknots are not considered in most algorithms. Heuristic algorithms such as HotKnots (An-
dronescu et al., 2010) and Probknots (Bellaousov & Mathews, 2010) have been made to predict
structures with pseudoknots, but the predictive accuracy and efficiency still need to be improved.
Learning-based RNA folding methods such as ContraFold (Do et al., 2006) and ContextFold (Za-
kov et al., 2011) have been proposed for energy parameters estimation due to the increasing avail-
ability of known RNA structures, resulting in higher prediction accuracies, but these methods still
rely on the above DP-based algorithms for energy minimization. A recent deep learning model,
CDPfold (Zhang et al., 2019), applied convolutional neural networks to predict base-pairings, but it
adopts the dot-bracket representation for RNA secondary structure, which can not represent pseu-
doknotted structures. Moreover, it requires a DP-based post-processing step whose computational
complexity is prohibitive for sequences longer than a few hundreds.
Learning with differentiable algorithms is a useful idea that inspires a series of works (Hershey
et al., 2014; Belanger et al., 2017; Ingraham et al., 2018; Chen et al., 2018; Shrivastava et al., 2019),
which shared similar idea of using differentiable unrolled algorithms as a building block in neural
architectures. Some models are also applied to structured prediction problems (Hershey et al., 2014;
Pillutla et al., 2018; Ingraham et al., 2018), but they did not consider the challenging RNA sec-
ondary structure problem or discuss how to properly incorporating constraints into the architecture.
OptNet (Amos & Kolter, 2017) integrates constraints by differentiating KKT conditions, but it has
cubic complexity in the number of variables and constraints, which is prohibitive for the RNA case.
Dependency parsing in NLP is a different but related problem to RNA folding. It predicts the de-
pendency between the words in a sentence. Similar to nested/non-nested structures, the correspond-
ing terms in NLP are projective/non-projective parsing, where most works focus on the former and
DP-based inference algorithms are commonly used (McDonald et al., 2005). Deep learning mod-
els (Dozat & Manning, 2016; Kiperwasser & Goldberg, 2016) are proposed to proposed to score the
dependency between words, which has a similar flavor to the Deep Score Network in our work.
3 RNA SECONDARY STRUCTURE PREDICTION PROBLEM
In the RNA secondary structure prediction problem, the input is the ordered sequence of bases
x = (x1, . . . , xL) and the output is the RNA secondary structure represented by a matrix A∗ ∈
{0, 1}L×L. Hard constraints on the forming of an RNA secondary structure dictate that certain
kinds of pairings cannot occur at all (Steeg, 1993). Formally, these constraints are:
(i) Only three types of nucleotides combinations, B := {AU, U A}∪
{GC, CG} ∪ {GU, U G}, can form base-pairs.
∀i, j, if xixj /∈ B,
then Aij = 0.
(ii) No sharp loops are allowed. ∀|i − j| < 4, Aij = 0.
(iii) There is no overlap of pairs, i.e., it is a matching. ∀i, ∑L
j=1 Aij ≤ 1.
(i) and (ii) prevent pairing of certain base-pairs based on their types and relative locations. Incorpo-
rating these two constraints can help the model exclude lots of illegal pairs. (iii) is a global constraint
among the entries of A∗.
3
Published as a conference paper at ICLR 2020
The space of all valid secondary structures contains all symmetric matrices A ∈ {0, 1}L×L that
satisfy the above three constraints. This space is much smaller than the space of all binary matrices
{0, 1}L×L. Therefore, if we could incorporate these constraints in our deep model, the reduced
output space could help us train a better predictive model with less training data. We do this by
using an unrolled algorithm as the inductive bias to design deep architecture.
4 E2EFOLD: DEEP LEARNING MODEL BASED ON UNROLLED ALGORITHM
In the literature on feed-forward networks for structured prediction, most models are designed using
traditional deep learning architectures. However, for RNA secondary structure prediction, directly
using these architectures does not work well due to the limited amount of RNA data points and the
hard constraints on forming an RNA secondary structure. These challenges motivate the design of
our E2Efold deep model, which combines a Deep Score Network with a Post-Processing Network
based on an unrolled algorithm for solving a constrained optimization problem.
4.1 DEEP SCORE NETWORK
The first part of E2Efold is a Deep Score Network Uθ (x) whose output is an L × L symmetric
matrix. Each entry of this matrix, i.e., Uθ (x)ij , indicates the score of nucleotides xi and xj being
paired. The x input to the network here is the L × 4 dimensional one-hot embedding. The specific
architecture of Uθ is shown in Fig 4. It mainly consists of
• a position embedding matrix P which distinguishes {xi}L
i=1 by their exact and relative positions:
Pi = MLP(ψ1(i), . . . , ψ`(i), ψ`+1(i/L), . . . , ψn(i/L)), where {ψj } is a set of n feature maps
such as sin(·), poly(·), sigmoid(·), etc, and MLP(·) denotes multi-layer perceptions. Such posi-
tion embedding idea has been used in natural language modeling such as BERT (Devlin et al.,
2018), but we adapted for RNA sequence representation;
• a stack of Transformer Encoders (Vaswani et al., 2017) which encode the sequence information
and the global dependency between nucleotides;
• a 2D Convolution layers (Wang et al., 2017) for outputting the pairwise scores.input 𝐿×4
position 𝐿×2
𝐿×𝑑
Multiply by W ∈ ℝ*×+
Position Embedding
0100
1000
1000
1000
0010
0100
0001
0001
0010
0010 …
G A A A C G U U C C …
0100
1000
1000
1000
0010
0100
0001
0001
0010
0010 …
G A A A C G U U C C …
1 2 3 4 5 …
1/L 2/L 3/L 4/L 5/L …
feature map 𝜓- , … , 𝜓0
𝐿×𝑛
MLP
𝐿×𝑑
Transformer Encoder
Transformer Encoder
Transformer Encoder
𝐿×2𝑑
𝐿×2𝑑
Sequence Encoder
concat
𝐿×3𝑑
pairwise concat
𝐿×𝐿×6𝑑
2D Convolution
2D Convolution
concat
𝐿×𝐿×1
Output Layers
scores U
symmetrization
Figure 4: Architecture of Deep Score Network.
With the representation power of neural networks, the
hope is that we can learn an informative Uθ such that
higher scoring entries in Uθ (x) correspond well to ac-
tual paired bases in RNA structure. Once the score
matrix Uθ (x) is computed, a naive approach to use
it is to choose an offset term s ∈ R (e.g., s = 0)
and let Aij = 1 if Uθ (x)ij > s. However, such
entry-wise independent predictions of Aij may re-
sult in a matrix A that violates the constraints for a
valid RNA secondary structure. Therefore, a post-
processing step is needed to make sure the predicted
A is valid. This step could be carried out separately
after Uθ is learned. But such decoupling of base-pair
scoring and post-processing for constraints may lead
to sub-optimal results, where the errors in these two
stages can not be considered together and tuned to-
gether. Instead, we will introduce a Post-Processing
Network which can be trained end-to-end together
with Uθ to enforce the constraints.
4.2 POST-PROCESSING NETWORK
The second part of E2Efold is a Post-Processing Net-
work PPφ which is an unrolled and parameterized al-
gorithm for solving a constrained optimization prob-
lem. We first present how we formulate the post-processing step as a constrained optimization prob-
lem and the algorithm for solving it. After that, we show how we use the algorithm as a template to
design deep architecture PPφ.
4
Published as a conference paper at ICLR 2020
4.2.1 POST-PROCESSING WITH CONSTRAINED OPTIMIZATION
Formulation of constrained optimization. Given the scores predicted by Uθ (x), we define the
total score 1
2
∑
i,j (Uθ (x)ij − s)Aij as the objective to maximize, where s is an offset term. Clearly,
without structure constraints, the optimal solution is to take Aij = 1 when Uθ (x)ij > s. Intu-
itively, the objective measures the covariation between the entries in the scoring matrix and the
A matrix. With constraints, the exact maximization becomes intractable. To make it tractable,
we consider a convex relaxation of this discrete optimization to a continuous one by allowing
Aij ∈ [0, 1]. Consequently, the solution space that we consider to optimize over is A(x) :={A ∈ [0, 1]L×L | A is symmetric and satisfies constraints (i)-(iii) in Section 3} .
To further simplify the search space, we define a nonlinear transformation T on RL×L as T ( ˆA) :=
1
2
( ˆA ◦ ˆA + ( ˆA ◦ ˆA)>) ◦ M (x), where ◦ denotes element-wise multiplication. Matrix M is defined as
M (x)ij := 1 if xixj ∈ B and also |i − j| ≥ 4, and M (x)ij := 0 otherwise. From this definition we
can see that M (x) encodes both constraint (i) and (ii). With transformation T , the resulting matrix
is non-negative, symmetric, and satisfies constraint (i) and (ii). Hence, by defining A := T ( ˆA), the
solution space is simplified as A(x) = {A = T ( ˆA) | ˆA ∈ RL×L, A1 ≤ 1}.
Finally, we introduce a `1 penalty term ‖ ˆA‖1 := ∑
i,j | ˆAij | to make A sparse and formulate the
post-processing step as: (〈·, ·〉 denotes matrix inner product, i.e., sum of entry-wise multiplication)
max ˆA∈RL×L
1
2
〈
Uθ (x) − s, A := T ( ˆA)
〉
− ρ‖ ˆA‖1 s.t. A1 ≤ 1 (1)
The advantages of this formulation are that the variables ˆAij are free variables in R and there are
only L inequality constraints A1 ≤ 1. This system of linear inequalities can be replaced by a set
of nonlinear equalities relu(A1 − 1) = 0 so that the constrained problem can be easily transformed
into an unconstrained problem by introducing a Lagrange multiplier λ ∈ RL
+:
min
λ≥0 max
ˆA∈RL×L
1
2 〈Uθ (x) − s, A〉 − 〈λ, relu(A1 − 1)〉
︸ ︷︷ ︸
f
−ρ‖ ˆA‖1. (2)
Algorithm for solving it. We use a primal-dual method for solving Eq. 2 (derived in Appendix B).
In each iteration, ˆA and λ are updated alternatively by:
(primal) gradient step: ˙At+1 ← ˆAt + α · γt
α · ˆAt ◦ M (x) ◦
(
∂f /∂At + (∂f /∂At)>)
, (3)
where
{∂f /∂At = 1
2 (Uθ (x) − s) − (λ ◦ sign(At1 − 1)) 1>,
sign(c) := 1 when c > 0 and 0 otherwise, (4)
(primal) soft threshold: ˆAt+1 ← relu(| ˙At+1| − ρ · α · γt
α), At+1 ← T ( ˆAt+1), (5)
(dual) gradient step: λt+1 ← λt+1 + β · γt
β · relu(At+11 − 1), (6)
where α, β are step sizes and γα, γβ are decaying coefficients. When it converges at T , an approx-
imate solution Round(AT = T ( ˆAT )) is obtained. With this algorithm operated on the learned
Uθ (x), even if this step is disconnected to the training phase of Uθ (x), the final prediction works
much better than many other existing methods (as reported in Section 6). Next, we introduce how
to couple this post-processing step with the training of Uθ (x) to further improve the performance.
4.2.2 POST-PROCESSING NETWORK VIA AN UNROLLED ALGORITHM
We design a Post-Processing Network, denoted by PPφ, based on the above algorithm. After it is
defined, we can connect it with the deep score network Uθ and train them jointly in an end-to-end
fashion, so that the training phase of Uθ (x) is aware of the post-processing step.
5
Published as a conference paper at ICLR 2020
Algorithm 1: Post-Processing Network PPφ(U, M )
Parameters φ := {w, s, α, β, γα, γβ , ρ}
U ← softsign(U − s) ◦ U
ˆA0 ← softsign(U − s) ◦ sigmoid(U )
A0 ← T ( ˆA0); λ0 ← w · relu(A01 − 1)
For t = 0, . . . , T − 1 do
λt+1, At+1, ˆAt+1 = PPcellφ(U, M, λt, At, ˆAt, t)
return {At}T
t=1
Algorithm 2: Neural Cell PPcellφ
Function PPcellφ(U, M, λ, A, ˆA, t):
G ← 1
2 U − (λ ◦ softsign(A1 − 1)) 1>
˙A ← ˆA + α · γαt · ˆA ◦ M ◦ (G + G>)
ˆA ← relu(| ˙A| − ρ · α · γαt)
ˆA ← 1 − relu(1 − ˆA) [i.e.,min( ˆA, 1)]
A ← T ( ˆA); λ ← λ+β·γβ t ·relu(A1−1)
return λ, A, ˆA
The specific computation graph of PPφ is given in Algorithm 1, whose main component is a recurrent
cell which we call PPcellφ. The computation graph is almost the same as the iterative update from
Eq. 3 to Eq. 6, except for several modifications:
• (learnable hyperparameters) The hyperparameters including step sizes α, β, decaying rate γα, γβ ,
sparsity coefficient ρ and the offset term s are treated as learnable parameters in φ, so that there
is no need to tune the hyperparameters by hand but automatically learn them from data instead.
• (fixed # iterations) Instead of running the iterative updates until convergence, PPcellφ is applied
recursively for T iterations where T is a manually fixed number. This is why in Fig 3 the output
space of E2Efold is slightly larger than the true solution space.
• (smoothed sign function) Resulted from the gradient of relu(·), the update step in Eq. 4 contains a
sign(·) function. However, to push gradient through PPφ, we require a differentiable update step.
Therefore, we use a smoothed sign function defined as softsign(c) := 1/(1 + exp(−kc)), where
k is a temperature.
• (clip ˆA) An additional step, ˆA ← min( ˆA, 1), is included to make the output At at each itera-
tion stay in the range [0, 1]L×L. This is useful for computing the loss over intermediate results
{At}T
t=1, for which we will explain more in Section 5.
With these modifications, the Post-Processing Network PPφ is a tuning-free and differentiable un-
rolled algorithm with meaningful intermediate outputs. Combining it with the deep score network,
the final deep model is
E2Efold : {At}T
t=1 =
Post-Process Network
︷ ︸︸ ︷
PPφ( Uθ (x)
︸ ︷︷ ︸
Deep Score Network
, M (x)) . (7)
5 END-TO-END TRAINING ALGORITHM
Given a dataset D containing examples of input-output pairs (x, A∗), the training procedure of
E2Efold is similar to standard gradient-based supervised learning. However, for RNA secondary
structure prediction problems, commonly used metrics for evaluating predictive performances are
F1 score, precision and recall, which are non-differentiable.
Differentiable F1 Loss. To directly optimize these metrics, we mimic true positive (TP), false posi-
tive (FP), true negative (TN) and false negative (FN) by defining continuous functions on [0, 1]L×L:
TP = 〈A, A∗〉, FP = 〈A, 1 − A∗〉, FN = 〈1 − A, A∗〉, TN = 〈1 − A, 1 − A∗〉.
Since F1 = 2TP/(2TP + FP + FN), we define a loss function to mimic the negative of F1 score as:
L−F1(A, A∗) := −2〈A, A∗〉/ (2〈A, A∗〉 + 〈A, 1 − A∗〉 + 〈1 − A, A∗〉) . (8)
Assuming that ∑
ij A∗
ij 6 = 0, this loss is well-defined and differentiable on [0, 1]L×L. Precision and
recall losses can be defined in a similar way, but we optimize F1 score in this paper.
It is notable that this F1 loss takes advantages over other differentiable losses including `2 and
cross-entropy losses, because there are much more negative samples (i.e. Aij = 0) than positive
samples (i.e. Aij = 1). A hand-tuned weight is needed to balance them while using `2 or cross-
entropy losses, but F1 loss handles this issue automatically, which can be useful for a number of
problems (Wang et al., 2016; Li et al., 2017).
6
Published as a conference paper at ICLR 2020
Overall Loss Function. As noted earlier, E2Efold outputs a matrix At ∈ [0, 1]L×L in each itera-
tion. This allows us to add auxiliary losses to regularize the intermediate results, guiding it to learn
parameters which can generate a smooth solution trajectory. More specifically, we use an objective
that depends on the entire trajectory of optimization:
min
θ,φ
1
|D|
∑
(x,A∗)∈D
1
T
T∑
t=1
γT −tL−F1(At, A∗), (9)
where {At}T
t=1 = PPφ(Uθ (x), M (x)) and γ ≤ 1 is a discounting factor. Empirically, we find it
very useful to pre-train Uθ using logistic regression loss. Also, it is helpful to add this additional
loss to Eq. 9 as a regularization.
6 EXPERIMENTS
We compare E2Efold with the SOTA and also the most commonly used methods in the RNA sec-
ondary structure prediction field on two benchmark datasets. It is revealed from the experimental
results that E2Efold achieves 29.7% improvement in terms of F1 score on RNAstralign dataset and
it infers the RNA secondary structure as fast as the most efficient algorithm (LinearFold) among ex-
isting ones. An ablation study is also conducted to show the necessity of pushing gradient through
the post-processing step. The codes for reproducing the experimental results are released.1
Table 1: Dataset Statistics
Type ArchiveII RNAStralign
length #samples length #samples
All 28∼2968 3975 30∼1851 30451
16SrRNA 73∼1995 110 54∼1851 11620
5SrRNA 102∼135 1283 104∼132 9385
tRNA 54∼93 557 59∼95 6443
grp1 210∼736 98 163∼615 1502
SRP 28∼533 928 30∼553 468
tmRNA 102∼437 462 102∼437 572
RNaseP 120∼486 454 189∼486 434
telomerase 382∼559 37 382∼559 37
23SrRNA 242∼2968 35 - -
grp2 619∼780 11 - -
Dataset. We use two benchmark datasets: (i) ArchiveII
(Sloma & Mathews, 2016), containing 3975 RNA struc-
tures from 10 RNA types, is a widely used benchmark
dataset for classical RNA folding methods. (ii) RNAS-
tralign (Tan et al., 2017), composed of 37149 structures
from 8 RNA types, is one of the most comprehensive col-
lections of RNA structures in the market. After removing
redundant sequences and structures, 30451 structures re-
main. See Table 1 for statistics about these two datasets.
Experiments On RNAStralign. We divide RNAStralign
dataset into training, testing and validation sets by strat-
ified sampling (see details in Table 7 and Fig 6), so that
each set contains all RNA types. We compare the performance of E2Efold to six methods includ-
ing CDPfold, LinearFold, Mfold, RNAstructure (ProbKnot), RNAfold and CONTRAfold. Both
E2Efold and CDPfold are learned from the same training/validation sets. For other methods, we
directly use the provided packages or web-servers to generate predicted structures. We evaluate the
F1 score, Precision and Recall for each sequence in the test set. Averaged values are reported in
Table 2. As suggested by Mathews (2019), for a base pair (i, j), the following predictions are also
considered as correct: (i + 1, j), (i − 1, j), (i, j + 1), (i, j − 1), so we also reported the metrics when
one-position shift is allowed.
Table 2: Results on RNAStralign test set. “(S)” indi-
cates the results when one-position shift is allowed.
Method Prec Rec F1 Prec(S) Rec(S) F1(S)
E2Efold 0.866 0.788 0.821 0.880 0.798 0.833
CDPfold 0.633 0.597 0.614 0.720 0.677 0.697
LinearFold 0.620 0.606 0.609 0.635 0.622 0.624
Mfold 0.450 0.398 0.420 0.463 0.409 0.433
RNAstructure 0.537 0.568 0.550 0.559 0.592 0.573
RNAfold 0.516 0.568 0.540 0.533 0.587 0.558
CONTRAfold 0.608 0.663 0.633 0.624 0.681 0.650 Figure 5: Distribution of F1 score.
As shown in Table 2, traditional methods can achieve a F1 score ranging from 0.433 to 0.624,
which is consistent with the performance reported with their original papers. The two learning-based
methods, CONTRAfold and CDPfold, can outperform classical methods with reasonable margin on
1The codes for reproducing the experimental results are released at https://github.com/ml4bio/e2efold.
7
Published as a conference paper at ICLR 2020
some criteria. E2Efold, on the other hand, significantly outperforms all previous methods across all
criteria, with at least 20% improvement. Notice that, for almost all the other methods, the recall is
usually higher than precision, while for E2Efold, the precision is higher than recall. That can be the
result of incorporating constraints during neural network training. Fig 5 shows the distributions of
F1 scores for each method. It suggests that E2Efold has consistently good performance.
To estimate the performance of E2Efold on long sequences, we also compute the F1 scores weighted
by the length of sequences, such that the results are more dominated by longer sequences. Detailed
results are given in Appendix D.3.
Table 3: Performance comparison on ArchiveII
Method Prec Rec F1 Prec(S) Rec(S) F1(S)
E2Efold 0.734 0.66 0.686 0.758 0.676 0.704
CDPfold 0.557 0.535 0.545 0.612 0.585 0.597
LinearFold 0.641 0.617 0.621 0.668 0.644 0.647
Mfold 0.428 0.383 0.401 0.450 0.403 0.421
RNAstructure 0.563 0.615 0.585 0.590 0.645 0.613
RNAfold 0.565 0.627 0.592 0.586 0.652 0.615
CONTRAfold 0.607 0.679 0.638 0.629 0.705 0.662
Table 4: Inference time on RNAStralign
Method total run time time per seq
E2Efold (Pytorch) 19m (GPU) 0.40s
CDPfold (Pytorch) 440m*32 threads 300.107s
LinearFold (C) 20m 0.43s
Mfold (C) 360m 7.65s
RNAstructure (C) 3 days 142.02s
RNAfold (C) 26m 0.55s
CONTRAfold (C) 1 day 30.58s
Test On ArchiveII Without Re-training. To mimic the real world scenario where the users want to
predict newly discovered RNA’s structures which may have a distribution different from the training
dataset, we directly test the model learned from RNAStralign training set on the ArchiveII dataset,
without re-training the model. To make the comparison fair, we exclude sequences that are over-
lapped with the RNAStralign dataset. We then test the model on sequences in ArchiveII that have
overlapping RNA types (5SrRNA, 16SrRNA, etc) with the RNAStralign dataset. Results are shown
in Table 3. It is understandable that the performances of classical methods which are not learning-
based are consistent with that on RNAStralign. The performance of E2Efold, though is not as good
as that on RNAStralign, is still better than all the other methods across different evaluation crite-
ria. In addition, since the original ArchiveII dataset contains domain sequences (subsequences), we
remove the domains and report the results in Appendix D.4, which are similar to results in Table 3.
Inference Time Comparison. We record the running time of all algorithms for predicting RNA
secondary structures on the RNAStralign test set, which is summarized in Table 4. LinearFold is the
most efficient among baselines because it uses beam pruning heuristic to accelerate DP. CDPfold,
which achieves higher F1 score than other baselines, however, is extremely slow due to its DP
post-processing step. Since we use a gradient-based algorithm which is simple to design the Post-
Processing Network, E2Efold is fast. On GPU, E2Efold has similar inference time as LinearFold.
Table 5: Evaluation of pseudoknot prediction
Method Set F1 TP FP TN FN
E2Efold 0.710 1312 242 1271 0
RNAstructure 0.472 1248 307 983 286
Pseudoknot Prediction. Even though E2Efold does
not exclude pseudoknots, it is not sure whether it ac-
tually generates pseudoknotted structures. Therefore,
we pick all sequences containing pseudoknots and com-
pute the averaged F1 score only on this set. Besides, we
count the number of pseudoknotted sequences that are
predicted as pseudoknotted and report this count as true positive (TP). Similarly we report TN, FP
and FN in Table 5 along with the F1 score. Most tools exclude pseudoknots while RNAstructure is
the most famous one that can predict pseudoknots, so we choose it for comparison.E2Efold RNAstructure CONTRAfold true structure RNAstructure CONTRAfoldE2Efoldtrue structuretrue structure E2Efoldtrue structure E2Efold
Visualization. We visualize predicted structures of three
RNA sequences in the main text. More examples are
provided in appendix (Fig 8 to 14). In these figures,
purple lines indicate edges of pseudoknotted elements.
Although CDPfold has higher F1 score than other base-
lines, its predictions are visually far from the ground-
truth. Instead, RNAstructure and CONTRAfold produce
comparatively more reasonable visualizations among all baselines, so we compare with them. These
8
Published as a conference paper at ICLR 2020
two methods can capture a rough sketch of the structure, but not good enough. For most cases,
E2Efold produces structures most similar to the ground-truths. Moreover, it works surprisingly well
for some RNA sequences that are long and very difficult to predict.
Table 6: Ablation study (RNAStralign test set)
Method Prec Rec F1 Prec(S) Rec(S) F1(S)
E2Efold 0.866 0.788 0.821 0.880 0.798 0.833
Uθ +PP 0.755 0.712 0.721 0.782 0.737 0.752
Ablation Study. To exam whether integrating the
two stages by pushing gradient through the post-
process is necessary for performance of E2Efold, we
conduct an ablation study (Table 6). We test the per-
formance when the post-processing step is discon-
nected with the training of Deep Score Network Uθ . We apply the post-processing step (i.e., for
solving augmented Lagrangian) after Uθ is learned (thus the notation “Uθ + PP” in Table 6). Al-
though “Uθ + PP” performs decently well, with constraints incorporated into training, E2Efold still
has significant advantages over it.
Discussion. To better estimate the performance of E2Efold on different RNA types, we include the
per-family F1 scores in Appendix D.5. E2Efold performs significantly better than other methods
in 16S rRNA, tRNA, 5S RNA, tmRNA, and telomerase. These results are from a single model. In
the future, we can view it as multi-task learning and further improve the performance by learning
multiple models for different RNA families and learning an additional classifier to predict which
model to use for the input sequence.
7 CONCLUSION
We propose a novel DL model, E2Efold, for RNA secondary structure prediction, which incorpo-
rates hard constraints in its architecture design. Comprehensive experiments are conducted to show
the superior performance of E2Efold, no matter on quantitative criteria, running time, or visualiza-
tion. Further studies need to be conducted to deal with the RNA types with less samples. Finally, we
believe the idea of unrolling constrained programming and pushing gradient through post-processing
can be generic and useful for other constrained structured prediction problems.
ACKNOWLEDGEMENT
We would like to thank anonymous reviewers for providing constructive feedbacks. This work is
supported in part by NSF grants CDS&E-1900017 D3SC, CCF-1836936 FMitF, IIS-1841351, CA-
REER IIS-1350983 to L.S. and grants from King Abdullah University of Science and Technology,
under award numbers BAS/1/1624-01, FCC/1/1976-18-01, FCC/1/1976-23-01, FCC/1/1976-25-01,
FCC/1/1976-26-01, REI/1/0018-01-01, and URF/1/4098-01-01.
REFERENCES
Brandon Amos and J Zico Kolter. Optnet: Differentiable optimization as a layer in neural networks.
In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 136–
145. JMLR. org, 2017.
Mirela S Andronescu, Cristina Pop, and Anne E Condon. Improved free energy parameters for RNA
pseudoknotted secondary structure prediction. RNA, 16(1):26–42, 2010.
David Belanger, Bishan Yang, and Andrew McCallum. End-to-end learning for structured prediction
energy networks. In Proceedings of the 34th International Conference on Machine Learning-
Volume 70, pp. 429–439. JMLR. org, 2017.
Stanislav Bellaousov and David H Mathews. Probknot: fast prediction of RNA secondary structure
including pseudoknots. RNA, 16(10):1870–1880, 2010.
Stanislav Bellaousov, Jessica S Reuter, Matthew G Seetin, and David H Mathews. RNAstructure:
web servers for RNA secondary structure prediction and analysis. Nucleic acids research, 41
(W1):W471–W474, 2013.
Xiaohan Chen, Jialin Liu, Zhangyang Wang, and Wotao Yin. Theoretical linear convergence of un-
folded ista and its practical weights and thresholds. In Advances in Neural Information Processing
Systems, pp. 9061–9071, 2018.
9
Published as a conference paper at ICLR 2020
Francis Crick. Central dogma of molecular biology. Nature, 227(5258):561, 1970.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
Chuong B Do, Daniel A Woods, and Serafim Batzoglou. Contrafold: RNA secondary structure
prediction without physics-based models. Bioinformatics, 22(14):e90–e98, 2006.
Timothy Dozat and Christopher D Manning. Deep biaffine attention for neural dependency parsing.
arXiv preprint arXiv:1611.01734, 2016.
P Fechter, J Rudinger-Thirion, C Florentz, and R Giege. Novel features in the tRNA-like world of
plant viral RNAs. Cellular and Molecular Life Sciences CMLS, 58(11):1547–1561, 2001.
Christine E Hajdin, Stanislav Bellaousov, Wayne Huggins, Christopher W Leonard, David H Math-
ews, and Kevin M Weeks. Accurate shape-directed RNA secondary structure modeling, including
pseudoknots. Proceedings of the National Academy of Sciences, 110(14):5498–5503, 2013.
John R Hershey, Jonathan Le Roux, and Felix Weninger. Deep unfolding: Model-based inspiration
of novel deep architectures. arXiv preprint arXiv:1409.2574, 2014.
Liang Huang, He Zhang, Dezhong Deng, Kai Zhao, Kaibo Liu, David A Hendrix, and David H
Mathews. Linearfold: linear-time approximate RNA folding by 5’-to-3’dynamic programming
and beam search. Bioinformatics, 35(14):i295–i304, 2019.
John Ingraham, Adam Riesselman, Chris Sander, and Debora Marks. Learning protein structure
with a differentiable simulator. 2018.
Elizabeth Iorns, Christopher J Lord, Nicholas Turner, and Alan Ashworth. Utilizing RNA interfer-
ence to enhance cancer drug discovery. Nature reviews Drug discovery, 6(7):556, 2007.
Eliyahu Kiperwasser and Yoav Goldberg. Simple and accurate dependency parsing using bidirec-
tional lstm feature representations. Transactions of the Association for Computational Linguistics,
4:313–327, 2016.
Yu Li, Sheng Wang, Ramzan Umarov, Bingqing Xie, Ming Fan, Lihua Li, and Xin Gao. Deepre:
sequence-based enzyme ec number prediction by deep learning. Bioinformatics, 34(5):760–769,
2017.
Ronny Lorenz, Stephan H Bernhart, Christian H¨oner Zu Siederdissen, Hakim Tafer, Christoph
Flamm, Peter F Stadler, and Ivo L Hofacker. ViennaRNA package 2.0. Algorithms for molecular
biology, 6(1):26, 2011.
Rune B Lyngsø and Christian NS Pedersen. RNA pseudoknot prediction in energy-based models.
Journal of computational biology, 7(3-4):409–427, 2000.
NR Markham and M Zuker. Unafold: software for nucleic acid folding and hybridization in: Keith
jm, editor.(ed.) bioinformatics methods in molecular biology, vol. 453, 2008.
David H Mathews. Predicting RNA secondary structure by free energy minimization. Theoretical
Chemistry Accounts, 116(1-3):160–168, 2006.
David H Mathews. How to benchmark RNA secondary structure prediction accuracy. Methods,
2019.
David H Mathews and Douglas H Turner. Prediction of RNA secondary structure by free energy
minimization. Current opinion in structural biology, 16(3):270–278, 2006.
Ryan McDonald, Fernando Pereira, Kiril Ribarov, and Jan Hajiˇc. Non-projective dependency pars-
ing using spanning tree algorithms. In Proceedings of the conference on Human Language Tech-
nology and Empirical Methods in Natural Language Processing, pp. 523–530. Association for
Computational Linguistics, 2005.
10
Published as a conference paper at ICLR 2020
Venkata Krishna Pillutla, Vincent Roulet, Sham M Kakade, and Zaid Harchaoui. A smoother way
to train structured prediction models. In Advances in Neural Information Processing Systems, pp.
4766–4778, 2018.
Harsh Shrivastava, Xinshi Chen, Binghong Chen, Guanghui Lan, Srinvas Aluru, and Le Song. Glad:
Learning sparse graph recovery. arXiv preprint arXiv:1906.00271, 2019.
Michael F Sloma and David H Mathews. Exact calculation of loop formation probability identifies
folding motifs in RNA secondary structures. RNA, 22(12):1808–1818, 2016.
David W Staple and Samuel E Butcher. Pseudoknots: RNA structures with diverse functions. PLoS
biology, 3(6):e213, 2005.
Evan W Steeg. Neural networks, adaptive optimization, and RNA secondary structure prediction.
Artificial intelligence and molecular biology, pp. 121–160, 1993.
Zhen Tan, Yinghan Fu, Gaurav Sharma, and David H Mathews. Turbofold ii: RNA structural
alignment and secondary structure prediction informed by multiple homologs. Nucleic acids
research, 45(20):11570–11581, 2017.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information
processing systems, pp. 5998–6008, 2017.
Sheng Wang, Siqi Sun, and Jinbo Xu. Auc-maximized deep convolutional neural fields for protein
sequence labeling. In Joint European Conference on Machine Learning and Knowledge Discovery
in Databases, pp. 1–16. Springer, 2016.
Sheng Wang, Siqi Sun, Zhen Li, Renyu Zhang, and Jinbo Xu. Accurate de novo prediction of protein
contact map by ultra-deep learning model. PLoS computational biology, 13(1):e1005324, 2017.
Shay Zakov, Yoav Goldberg, Michael Elhadad, and Michal Ziv-Ukelson. Rich parameterization
improves RNA structure prediction. Journal of Computational Biology, 18(11):1525–1542, 2011.
Hao Zhang, Chunhe Zhang, Zhi Li, Cong Li, Xu Wei, Borui Zhang, and Yuanning Liu. A new
method of RNA secondary structure prediction based on convolutional neural network and dy-
namic programming. Frontiers in genetics, 10, 2019.
11
Published as a conference paper at ICLR 2020
A MORE DISCUSSION ON RELATED WORKS
Here we explain the difference between our approach and other works on unrolling optimization
problems.
First, our view of incorporating constraints to reduce output space and to reduce sample complexity
is novel. Previous works (Hershey et al., 2014; Belanger et al., 2017; Ingraham et al., 2018) did not
discuss these aspects. The most related work which also integrates constraints is OptNet (Amos &
Kolter, 2017), but its very expensive and can not scale to the RNA problem. Therefore, our proposed
approach is a simple and effective one.
Second, compared to (Chen et al., 2018; Shrivastava et al., 2019), our approach has a different
purpose of using the algorithm. Their goal is to learn a better algorithm, so they commonly make
their architecture more flexible than the original algorithm for the room of improvement. However,
we aim at enforcing constraints. To ensure that constraints are nicely incorporated, we keep the
original structure of the algorithm and only make the hyperparameters learnable.
Finally, although all works consider end-to-end training, none of them can directly optimize the F1
score. We proposed a differentiable loss function to mimic the F1 score/precision/recall, which is
effective and also very useful when negative samples are much fewer than positive samples (or the
inverse).
B DERIVATION OF THE PROXIMAL GRADIENT STEP
The maximization step in Eq. 1 can be written as the following minimization:
min
ˆA∈RL×L
− 1
2 〈Uθ (x) − s, A〉 + 〈λ, relu(A1 − 1)〉
︸ ︷︷ ︸
−f ( ˆA)
+ρ‖ ˆA‖1. (10)
Consider the quadratic approximation of −f ( ˆA) centered at ˆAt:
− ˜fα( ˆA) := − f ( ˆAt) + 〈− ∂f
∂ ˆAt
, ˆA − ˆAt〉 + 1
2α ‖ ˆA − ˆAt‖2
F (11)
= − f ( ˆAt) + 1
2α
∥
∥
∥ ˆA −
(
ˆAt + α ∂f
∂ ˆAt
) ∥
∥
∥2
F , (12)
and rewrite the optimization in Eq. 10 as
min
ˆA∈RL×L
− f ( ˆAt) + 1
2α
∥
∥
∥ ˆA − ˙At+1
∥
∥
∥2
F + ρ‖ ˆA‖1 (13)
≡ min
ˆA∈RL×L
1
2α
∥
∥
∥ ˆA − ˙At+1
∥
∥
∥2
F
+ ρ‖ ˆA‖1, (14)
where
˙At+1 := ˆAt + α ∂f
∂ ˆAt
. (15)
Next, we define proximal mapping as a function depending on α as follows:
proxα( ˙At+1) = arg min
ˆA∈RL×L
1
2α
∥
∥
∥ ˆA − ˙At+1
∥
∥
∥2
F
+ ρ‖ ˆA‖1 (16)
= arg min
ˆA∈RL×L
1
2
∥
∥
∥ ˆA − ˙At+1
∥
∥
∥2
F
+ αρ‖ ˆA‖1 (17)
= sign( ˙At+1) max(| ˙At+1| − αρ, 0) (18)
= sign( ˙At+1)relu(| ˙At+1| − αρ). (19)
Since we always use ˆA ◦ ˆA instead of ˆA in our problem, we can take the absolute value
|proxα( ˙At+1)| = relu(| ˙At+1| − αρ) without loss of generality. Therefore, the proximal gradient
12
Published as a conference paper at ICLR 2020
step is
˙At+1 ← ˆAt + α ∂f
∂ ˆAt
(correspond to Eq. 3) (20)
ˆAt+1 ← relu(| ˙At+1| − αρ) (correspond to Eq. 5). (21)
More specifically, in the main text, we write ∂f
∂ ˆAt
as
∂f
∂ ˆAt
= 1
2
(
∂f
∂At
+ ∂f
∂At
>)
◦ ∂At
∂ ˆAt
(22)
=
( 1
2
∂At
∂ ˆAt
)
◦
(
∂f
∂At
+ ∂f
∂At
>)
(23)
=
( 1
22 ◦ M ◦ (2 ˆAt + 2 ˆA>
t )
)
◦
(
∂f
∂At
+ ∂f
∂At
>)
(24)
=
( 1
22 ◦ M ◦ (2 ˆAt + 2 ˆA>
t )
)
◦
(
∂f
∂At
+ ∂f
∂At
>)
(25)
= M ◦ ˆAt ◦
(
∂f
∂At
+ ∂f
∂At
>)
. (26)
The last equation holds since ˆAt will remain symmetric in our algorithm if the initial ˆA0 is symmet-
ric. Moreover, in the main text, α is replaced by α · γt
α.
C IMPLEMENTATION AND TRAINING DETAILS
We used Pytorch to implement the whole package of E2Efold.
Deep Score Network. In the deep score network, we used a hyper-parameter, d, which was set as
10 in the final model, to control the model capacity. In the transformer encoder layers, we set the
number of heads as 2, the dimension of the feed-forward network as 2048, the dropout rate as 0.1.
As for the position encoding, we used 58 base functions to form the position feature map, which
goes through a 3-layer fully-connected neural network (the number of hidden neurons is 5 ∗ d) to
generate the final position embedding, whose dimension is L by d. In the final output layer, the
pairwise concatenation is carried out in the following way: Let X ∈ RL×3d be the input to the
final output layers in Figure 4 (which is the concatenation of the sequence embedding and position
embedding). The pairwise concatenation results in a tensor Y ∈ RL×L×6d defined as
Y (i, j, :) = [X(i, :), X(j, :)], (27)
where Y (i, j, :) ∈ R6d, X(i, :) ∈ R3d, and X(j, :) ∈ R3d.
In the 2D convolution layers, the the channel of the feature map gradually change from 6∗d to d , and
finally to 1. We set the kernel size as 1 to translate the feature map into the final score matrix. Each
2D convolution layer is followed by a batch normalization layer. We used ReLU as the activation
function within the whole score network.
Post-Processing Network. In the PP network, we initialized w as 1, s as log(9), α as 0.01, β as 0.1,
γα as 0.99, γβ as 0.99, and ρ as 1. We set T as 20.
Training details. During training, we first pre-trained a deep score network and then fine-tuned the
score network and the PP network together. To pre-train the score network, we used binary cross-
entropy loss and Adam optimizer. Since, in the contact map, most entries are 0, we used weighted
loss and set the positive sample weight as 300. The batch size was set to fully use the GPU memory,
which was 20 for the Titan Xp card. We pre-train the score network for 100 epochs. As for the
fine-tuning, we used binary cross-entropy loss for the score network and F1 loss for the PP network
and summed up these two losses as the final loss. The user can also choose to only use the F1 loss or
13
Published as a conference paper at ICLR 2020
use another coefficient to weight the loss estimated on the score network Uθ . Due to the limitation of
the GPU memory, we set the batch size as 8. However, we updated the model’s parameters every 30
steps to stabilize the training process. We fine-tuned the whole model for 20 epochs. Also, since the
data for different RNA families are imbalanced, we up-sampled the data in the small RNA families
based on their size. For the training of the score network Uθ in the ablation study, it is exactly the
same as the training of the above mentioned process. Except that during the fine-tune process, there
is the unrolled number of iterations is set to be 0.
D MORE EXPERIMENTAL DETAILS
D.1 DATASET STATISTICS
Figure 6: The RNAStralign length distribution.
Table 7: RNAStralign dataset splits statistics
RNA type All Training Validation Testing
16SrRNA 11620 9325 1145 1150
5SrRNA 9385 7687 819 879
tRNA 6443 5412 527 504
grp1 1502 1243 123 136
SRP 468 379 36 53
tmRNA 572 461 50 61
RNaseP 434 360 37 37
telomerase 37 28 4 5
RNAStralign 30451 24895 2702 2854
D.2 TWO-SAMPLE HYPOTHESIS TESTING
To better understand the data distribution in different datasets, we provide statistical hypothesis test
results in this section.
We can assume that
(i) Samples in RNAStralign training set are i.i.d. from the distribution P(RNAStrtrain);
(ii) Samples in RNAStralign testing set are i.i.d. from the distribution P(RNAStrtest);
(iii) Samples in ArchiveII dataset are i.i.d. from the distribution P(ArcII).
To compare the differences among these data distributions, we can test the following hypothesis:
(a) P(RNAStrtrain) = P(RNAStrtest)
(b) P(RNAStrtrain) = P(ArchiveII)
14
Published as a conference paper at ICLR 2020
The approach that we adopted is the permutation test on the unbiased empirical Maximum Mean
Discrepancy (MMD) estimator:
MMDu(X, Y ) :=
( N∑
i=1
N∑
j6 =i
k(xi, xj ) +
M∑
i=1
M∑
j6 =i
k(yi, yj ) − 2
mn
N∑
i=1
M∑
j=1
k(xi, yj )
) 1
2
, (28)
where X = {xi}N
i=1 contains N i.i.d. samples from a distribution P1, Y = {yi}M
i=1 contains M
i.i.d. samples from a distribution P2, and k(·, ·) is a string kernel.
Since we conduct stratified sampling to split the training and testing dataset, when we perform
permutation test, we use stratified re-sampling as well (for both Hypothese (a) and (b)). The result
of the permutation test (permuted 1000 times) is reported in Figure 7.
Figure 7: Left: Distribution of MMDu under Hypothesis P(RNAStrtrain) = P(RNAStrtest). Right:
Distribution of MMDu under Hypothesis P(RNAStrtrain) = P(ArchiveII).
The result shows
(a) Hypothesis P(RNAStrtrain) = P(RNAStrtest) can be accepted with significance level 0.1.
(b) Hypothesis P(RNAStrtrain) = P(ArchiveII) is rejected since the p-value is 0.
Therefore, the data distribution in ArchiveII is very different from the RNAStralign training set. A
good performance on ArchiveII shows a significant generalization power of E2Efold.
D.3 PERFORMANCE ON LONG SEQUENCES: WEIGHTED F1 SCORE
For long sequences, E2Efold still performs better than other methods. We compute F1 scores
weighted by the length of sequences (Table 8), such that the results are more dominated by longer
sequences.
Table 8: RNAStralign: F1 after a weighted average by sequence length.
Method E2Efold CDPfold LinearFold Mfold RNAstructure RNAfold CONTRAfold
non-weighted 0.821 0.614 0.609 0.420 0.550 0.540 0.633
weighted 0.720 0.691 0.509 0.366 0.471 0.444 0.542
change -12.3% +12.5% -16.4% -12.8% -14.3% -17.7% -14.3%
The third row reports how much F1 score drops after reweighting.
D.4 ARCHIVEII RESULTS AFTER DOMAIN SEQUENCES ARE REMOVED
Since domain sequence (subsequences) in ArchiveII are explicitly labeled, we filter them out in
ArchiveII and recompute the F1 scores (Table 9).
The results do not change too much before or after filtering out subsequences.
15
Published as a conference paper at ICLR 2020
Table 9: ArchiveII: F1 after subsequences are filtered out.
Method E2Efold CDPfold LinearFold Mfold RNAstructure RNAfold CONTRAfold
original 0.704 0.597 0.647 0.421 0.613 0.615 0.662
filtered 0.723 0.605 0.645 0.419 0.611 0.615 0.659
D.5 PER-FAMILY PERFORMANCES
To balance the performance among different families, during the training phase we conducted
weighted sampling of the data based on their family size. With weighted sampling, the overall
F1 score (S) is 0.83, which is the same as when we did equal-weighted sampling. The per-family
results are shown in Table 10.
Table 10: RNAStralign: per-family performances
16S rRNA tRNA 5S RNA SRP
F1 F1(S) F1 F1(S) F1 F1(S) F1 F1(S)
E2Efold 0.783 0.795 0.917 0.939 0.906 0.936 0.550 0.614
LinearFold 0.493 0.504 0.734 0.739 0.713 0.738 0.618 0.648
Mfold 0.362 0.373 0.662 0.675 0.356 0.367 0.350 0.378
RNAstructure 0.464 0.485 0.709 0.736 0.578 0.597 0.579 0.617
RNAfold 0.430 0.449 0.695 0.706 0.592 0.612 0.617 0.651
CONTRAfold 0.529 0.546 0.758 0.765 0.717 0.740 0.563 0.596
tmRNA Group I intron RNaseP telomerase
F1 F1(S) F1 F1(S) F1 F1(S) F1 F1(S)
E2Efold 0.588 0.653 0.387 0.428 0.565 0.604 0.954 0.961
LinearFold 0.393 0.412 0.565 0.579 0.567 0.578 0.515 0.531
Mfold 0.290 0.308 0.483 0.498 0.562 0.579 0.403 0.531
RNAstructure 0.400 0.423 0.566 0.599 0.589 0.616 0.512 0.545
RNAfold 0.411 0.430 0.589 0.599 0.544 0.563 0.471 0.496
CONTRAfold 0.463 0.482 0.603 0.620 0.645 0.662 0.529 0.548
16
Published as a conference paper at ICLR 2020
D.6 MORE VISUALIZATION RESULTSE2Efold RNAstructure CDPfoldtrue structure
LinearFold MfoldCONTRAfold RNAfold
Figure 8: Visualization of 5S rRNA, B01865.E2Efold RNAstructure CDPfoldtrue structure
LinearFold MfoldCONTRAfold RNAfold
Figure 9: Visualization of 16S rRNA, DQ170870.
17
Published as a conference paper at ICLR 2020E2Efold RNAstructure CDPfoldtrue structure
LinearFold MfoldCONTRAfold RNAfold
Figure 10: Visualization of Group I intron, IC3, Kaf.c.trnL.E2Efold RNAstructure CDPfoldtrue structure
LinearFold MfoldCONTRAfold RNAfold
Figure 11: Visualization of RNaseP, A.salinestris-184.E2Efold RNAstructure CDPfoldtrue structure
LinearFold MfoldCONTRAfold RNAfold
Figure 12: Visualization of SRP, Homo.sapi. BU56690.
18
Published as a conference paper at ICLR 2020E2Efold RNAstructure CDPfoldtrue structure
LinearFold MfoldCONTRAfold RNAfold
Figure 13: Visualization of tmRNA, uncu.bact. AF389956.E2Efold RNAstructure CDPfoldtrue structure
LinearFold MfoldCONTRAfold RNAfold
Figure 14: Visualization of tRNA, tdbD00012019.
19
===

Skip to main content

An official website of the United States government
NCBI home page

Primary site navigation
Search PMC Full-Text Archive

    Advanced Search
    Journal List
    User Guide

As a library, NLM provides access to scientific literature. Inclusion in an NLM database does not imply endorsement of, or agreement with, the contents by NLM or the National Institutes of Health.
Learn more: PMC Disclaimer | PMC Copyright Notice
Genes logo
Genes (Basel)
. 2022 Nov 18;13(11):2155. doi: 10.3390/genes13112155
Direct Inference of Base-Pairing Probabilities with Neural Networks Improves Prediction of RNA Secondary Structures with Pseudoknots
Manato Akiyama 1, Yasubumi Sakakibara 1, Kengo Sato 2,*
Editors: Zihua Hu, Michel Ravelonandro, Lionel Benard

PMCID: PMC9690657  PMID: 36421829
Abstract

Existing approaches to predicting RNA secondary structures depend on how the secondary structure is decomposed into substructures, that is, the architecture, to define their parameter space. However, architecture dependency has not been sufficiently investigated, especially for pseudoknotted secondary structures. In this study, we propose a novel algorithm for directly inferring base-pairing probabilities with neural networks that do not depend on the architecture of RNA secondary structures, and then implement this approach using two maximum expected accuracy (MEA)-based decoding algorithms: Nussinov-style decoding for pseudoknot-free structures and IPknot-style decoding for pseudoknotted structures. To train the neural networks connected to each base pair, we adopt a max-margin framework, called structured support vector machines (SSVM), as the output layer. Our benchmarks for predicting RNA secondary structures with and without pseudoknots show that our algorithm outperforms existing methods in prediction accuracy.

Keywords: RNA secondary structure, deep learning, pseudoknots
1. Introduction

The roles of functional non-coding RNAs (ncRNAs) in regulating transcription and guiding post-transcriptional modification have been recently shown to be critical in various biological processes, ranging from development and cell differentiation in healthy individuals to disease pathogenesis [1]. The well-established relationship between the primary sequence and structure of ncRNAs has motivated research aiming to elucidate the functions of ncRNAs by determining their structures.

Yet, methods for experimentally determining RNA tertiary structures utilizing X-ray crystal structure analysis and nuclear magnetic resonance (NMR) are costly and labor-intensive, thus restricting their application. Accordingly, researchers often carry out computational prediction of RNA secondary structures based on the analysis of base pairs comprising nucleotides joined by hydrogen bonds.

Computational approaches to RNA secondary structure prediction often utilize thermodynamic models (e.g., Turner’s nearest neighbor model [2,3]) that define characteristic substructures, such as base-pair stacking and hairpin loops. In computational approaches, the free energy of each type of substructure is first empirically determined by methods such as optical melting experiments [2]. Then, the free energy of RNA secondary structures can be estimated as the sum of the free energy of their substructures. Dynamic programming can then be used to determine the optimal secondary structure that minimizes free energy for a given RNA sequence. This approach is employed by RNAfold [4], RNAstructure [5] and UNAfold [6], among other tools.

As an alternative to experimental approaches, machine learning can be utilized to train scoring parameters based on the substructures constituting reference structures. This type of approach, as implemented in CONTRAfold [7,8], Simfold [9,10], ContextFold [11] and similar tools, has improved the accuracy of RNA secondary structure prediction. By integrating thermodynamic and machine-learning-based weighting approaches, MXfold avoided overfitting and achieved better performance than models based on either one alone [12]. Furthermore, interest in the use of deep learning for RNA secondary structure prediction is rapidly increasing [13,14,15]. MXfold2 used thermodynamic regularization to train a deep neural network so that the predicted folding score and free energy are as close as possible. This method showed robust prediction results in familywise cross validation, where the test dataset was structurally different from the training dataset.

Another important aspect of RNA secondary structure prediction is the choice of the decoding algorithm used to find the optimal secondary structure from among all possible secondary structures. Two classic decoding algorithms are the minimum free energy (MFE) algorithm, which is used in thermodynamic approaches, and the maximum likelihood estimation (MLE) algorithm, which is used in machine-learning-based approaches. These algorithms find a secondary structure that minimizes the free energy and maximizes the probability or scoring function, respectively. Another option is a posterior decoding algorithm based on the maximum expected accuracy (MEA) principle, which is known to be an effective approach for many high-dimensional combinatorial optimization problems [16]. As researchers usually evaluate the prediction of RNA secondary structures using base-pair-wise accuracy measures, MEA-based decoding algorithms utilize posterior base-pairing probabilities that can be calculated by the McCaskill algorithm [17] or the inside–outside algorithm for stochastic context-free grammars. CONTRAfold [18] and CentroidFold [19] both have MEA-based decoding algorithm implementations that successfully predict RNA secondary structures.

Pseudoknots, an important structural element in RNA secondary structures, occur when at least two hydrogen bonds cross each other, and are typically drawn as two crossing arcs above a primary sequence (Figure 1).
Figure 1.

Figure 1
Open in a new tab

An example of pseudoknots.

Many RNAs, including rRNAs, tmRNAs and viral RNAs, form pseudoknotted secondary structures [20]. Pseudoknots are known to be involved in the regulation of translation and splicing as well as ribosomal frame shifting [21,22]. Furthermore, pseudoknots support folding into 3D structures in many cases [23]. Therefore, the impact of pseudoknots cannot be ignored in the structural and functional analysis of RNAs.

However, all of the aforementioned algorithms cannot consider pseudoknotted secondary structures owing to computational complexity. It has been proven that the problem of finding MFE structures including arbitrary pseudoknots is NP-hard [24,25]. Therefore, practically available algorithms for predicting pseudoknotted RNA secondary structures fall into one of the following two approaches: exact algorithms for a limited class of pseudoknots, such as PKNOTS [26], NUPACK [27,28], pknotsRG [29] and Knotty [30]; and heuristic algorithms that do not guarantee that the optimal structure will be found, such as ILM [31], HotKnots [32,33], FlexStem [34] and ProbKnot [35].

We previously developed IPknot, which enables fast and accurate prediction of RNA secondary structures with pseudoknots using integer programming [36,37]. IPknot adopts an MEA-based decoding algorithm that utilizes base-pairing probabilities combined with an approximate decomposition of a pseudoknotted structure into hierarchical pseudoknot-free structures. The prediction performance of IPknot is sufficient in terms of speed and accuracy compared with heuristic algorithms, and it is much faster than the exact algorithms.

Both thermodynamic approaches and machine-learning-based approaches depend on the method by which a secondary structure is decomposed into substructures, that is, the architecture (as referred to in [38]), to define their parameter space. Turner’s nearest neighbor model is the most well-studied architecture for predicting pseudoknot-free secondary structures, while the energy models for pseudoknotted secondary structures have not been sufficiently investigated, except for the Dirks–Pierce model [27,28] and the Cao–Chen model [39] for limited classes of pseudoknots. To our knowledge, an effective and efficient procedure to find a suitable architecture that can predict RNA secondary structures more accurately is still unknown.

Here, we propose a novel algorithm to directly infer base-pairing probabilities with neural networks instead of the McCaskill algorithm or the inside–outside algorithm, which both depend on the architecture of RNA secondary structures. Then, we employ the inferred base-pairing probabilities as part of a MEA-based scoring function for the two decoding algorithms: Nussinov-style decoding for pseudoknot-free structures, and IPknot-style decoding for pseudoknotted structures. To train the neural networks connected to each base pair, we adopt a max-margin framework, called structured support vector machines (SSVMs), as the output layer. We implement two types of neural networks connected to each base pair: bidirectional recursive neural networks (BiRNN) over tree structures and multilayer feedforward neural networks (FNN) with k-mer contexts around both bases in a pair. Our benchmarks for predicting RNA secondary structures with and without pseudoknots show that the prediction accuracy of our algorithm is superior to that of existing methods.

The major advantages of our work are summarized as follows: (i) our algorithm enables us to accurately predict RNA secondary structures with and without pseudoknots; (ii) our algorithm assumes no prior knowledge of the architecture that defines the decomposition of RNA secondary structures and thus the corresponding parameter space.
2. Methods
2.1. Preliminaries

The RNA sequence structure is modeled following the setup used by Akiyama et al. [12]. First, let , and let represent the set of all finite RNA sequences comprised of bases in . For a sequence , let represent the number of bases in x, referred to as the length of x. Let represent the set of all possible secondary structures formed by x. A secondary structure can be described as a binary-valued triangular matrix , in which if and only if bases and form a base pair linked by hydrogen bonds, including both canonical Watson–Crick base pairs (i.e., G-C and A-U) and non-canonical wobble base pairs (e.g., G-U).
2.2. MEA-Based Scoring Function

We employ the maximum expected accuracy (MEA)-based scoring function originally used for IPknot [36,37].

A secondary structure is assumed to be decomposable into a set of pseudoknot-free substructures satisfying the following two conditions: (i) can be decomposed into a mutually-exclusive set, that is, for , ; and (ii) each base pair in can be pseudoknotted to at least one base pair in for . Each pseudoknot-free substructure is said to belong to level p. For each RNA secondary structure , there exists a positive integer m such that y is decomposable into m substructures without one or more pseudoknots (for more details, see the Supplementary Materials of [36]). Through the above decomposition, arbitrary pseudoknots can be modeled by our method.

First, to construct an MEA-based scoring function, we define a gain function of with respect to the correct secondary structure as follows:
	(1)

Here, represents a base-pair weight parameter, and represent the numbers of true negatives (non-base pairs) and true positives (base pairs), respectively, and is an indicator function returning a value of either 1 or 0 depending on whether the is true or false.

The objective is to identify a secondary structure that maximizes the expected value of the above gain function (1) under a given probability distribution over the space of pseudoknotted secondary structures:
	(2)

Here, is the probability distribution of RNA secondary structures including pseudoknots. The -centroid estimator (2) has been proven to allow us to decode secondary structures accurately based on a given probability distribution [18].

Accordingly, the expected gain function (2) can be approximated as the sum of the expected gain functions for each level of pseudoknot-free substructures in the decomposed set of a pseudoknotted structure . Thus, a pseudoknotted structure and its decomposition can be found that maximize the following expected value:
	(3)

Here, is a weight parameter for level p base pairs and C is a constant that is independent of (for the derivation, see the Supplementary Material of [18]). The base-pairing probability represents the probability of base being paired with . As seen in Section 2.4, we employ one of three algorithms to calculate base-pairing probabilities.

It should be noted that IPknot can be considered an extension of CentroidFold [18]. For the restricted case of a single decomposed level (i.e., ), the approximate expected gain function (3) of IPknot is equivalent to CentroidFold’s -centroid estimator.
2.3. Decoding Algorithms
2.3.1. Nussinov-Style Decoding Algorithm for Pseudoknot-Free Structures

For the prediction of pseudoknot-free secondary structures, we find that maximizes the expected gain (3) with under the following constraints on base pairs:
	(4)
	(5)
	(6)

The constraint defined by Equation (5) means that each base can be paired with at most one base. The constraint defined by Equation (6) disallows pseudoknot.

This integer programming (IP) problem can be solved by dynamic programming as follows, similar to the Nussinov algorithm [40],
	(7)

and then tracing back from .
2.3.2. IPknot-Style Decoding Algorithm for Pseudoknotted Structures

Maximization of the approximate expected gain (3) can be solved as the following IP problem:
	(8)
	(9)
	(10)
	(11)

Note that Equation (3) requires the consideration of only base pairs with base-pairing probabilities being greater than . The constraint defined by Equation (9) means that each base can be paired with, at most, one base. The constraint defined by Equation (10) disallows pseudoknots within the same level p. The constraint defined by Equation (11) ensures that each level-p base pair is pseudoknotted to at least one base pair at each lower level . We set , which is IPknot’s default setting. This suggests that the predicted structure can be decomposed into two pseudoknot-free secondary structures.
2.4. Inferring Base-Paring Probabilities

Our scoring function (3) described in Section 2.2 is calculated by using base-pairing probabilities . In this section, we introduce two approaches for computing base-pairing probabilities. The first approach is a traditional one that is based on the probability distribution of RNA secondary structures, e.g., the McCaskill model [17] for pseudoknot-free structures and its extension to pseudoknotted structures, e.g., the Dirks–Pierce model [27,28]. The second approach proposed in this paper directly calculates base-pairing probabilities using neural networks.
2.4.1. Traditional Models for Base-Pairing Probabilities

The base-pairing probability is defined as
	(12)

from a probability distribution over a set of secondary structures with or without pseudoknots.

For predicting pseudoknot-free structures, the McCaskill model [17] can be mostly used as combined with the Nussinov-style decoding algorithm described in Section 2.3.1. The computational complexity of calculating Equation (12) for the McCaskill model is for time and for space when using dynamic programming. This model was implemented previously as CentroidFold [18,19].

For predicting pseudoknotted structures, we can select from among several models. A naïve model could use the probability distribution with pseudoknots as well as Equation (2) in spite of high computational costs, e.g., the Dirks–Pierce model [27,28] for a limited class of pseudoknots, with a computational complexity of for time and for space. Alternatively, we can employ a probability distribution without pseudoknots for each decomposed pseudoknot-free structure, such as the McCaskill model. Furthermore, to increase the prediction accuracy, we can utilize a heuristic algorithm with iterative refinement that refines the base-pairing probability matrix from the distribution without pseudoknots. See [36] for more details. These three models were implemented in IPknot [36].
2.4.2. Neural Network Models

In this research, we propose two neural network architectures for calculating base-pairing probabilities instead of the probability distribution over all RNA secondary structures.

The first architecture is the bidirectional recursive neural network (BiRNN) over tree structures as shown in Figure 2. Stochastic context-free grammars (SCFG) can model RNA secondary structure without pseudoknots [7,41]. The layers of BiRNN over the tree structure are connected along grammatical trees derived from SCFG that models RNA secondary structures. The BiRNN consists of three matrices—(a) the inside RNN matrix, (b) the outside RNN matrix and (c) the inside–outside matrix—for outputting base-pairing probabilities, each of whose elements contain a network layer (indicated by circles in Figure 2) with 80 hidden nodes. Each layer in the inside or outside matrix is recursively calculated from connected source layers as in the inside or outside algorithm, respectively, for stochastic context-free grammars (SCFG). The ReLU activation function is applied before being input to each recursive node. The base-pairing probability at each position is calculated from the corresponding layers in the inside and outside matrices with the sigmoid activation function. Our implementation of BiRNN assumes a simple RNA grammar

where , a and represent the paired bases, represents the start non-terminal symbol, and represents the empty string.
Figure 2.

Figure 2
Open in a new tab

A bidirectional recursive neural network for calculating base-pairing probabilities. A set of four dots above each base represents the one-hot representation of the base. Each circle indicates a network layer with 80 hidden nodes. Each solid arrow indicate a connection between layers along grammatical trees derived from the RNA grammar. Each dashed arrow represents a connection that aggregates the inside and outside layers to output base-pairing probabilities.

The second architecture employs a simple multilayer feedforward neural network (FNN). To calculate the base-pairing probability , a FNN receives as input two k-mers around the i-th and j-th bases as shown in Figure 3.
Figure 3.

Figure 3
Open in a new tab

A feedforward neural network with -mer contexts around and used to calculate the base-pairing probability . The end-of-loop nodes of the highlighted nucleotides are activated because they are beyond the paired bases.

Each base is encoded by the one-hot encoding of nucleotides and an additional node that indicates the end of the loop, which should be active for s.t. in the left k-mer around or s.t. in the right k-mer around . This encoding can be expected to embed the length of loops and the contexts around the openings and closings of helices. We set for the k-mer context length default (for more details, see Section 3.4). We then construct two hidden layers consisting of 200 and 50 nodes, respectively, with the ReLU activation function and one output node with a sigmoid activation function to output base-pairing probabilities.

Note that the FNN model depends on no assumption of RNA secondary structures, while the BiRNN model assumes an RNA grammar that considers no pseudoknots. Instead, the FNN model can take longer contexts around each base pair into consideration by using longer k-mers.
2.5. Learning Algorithm

We optimize the network parameters by using a max-margin framework called a structured support vector machine (SSVM) [42]. For a training dataset , where represents the k-th RNA sequence and represents the correct secondary structure of the k-th sequence , we identify a that minimizes the objective function
	(13)

where is the scoring function of RNA secondary structure for a given RNA sequence , that is, Equation (4) for Nussinov-style decoding or Equation (8) for IPknot-style decoding. Here, is a loss function of for y defined as
	(14)

where and are tunable hyperparameters that can control the trade-off between sensitivity and specificity in learning the parameters. By default, we used . In this case, the first term of Equation (13) can be calculated using the Nussinov-style decoding algorithm or the IPknot-style decoding algorithm modified by loss-augmented inference [42].

To minimize the objective function (13), stochastic subgradient descent (Algorithm 1) or one of its variants can be applied. We can calculate the gradients with regard to the network parameters for the objective function (13) using the gradients with regard to by the chain rule of differentiation. This means that the prediction errors occurred through the decoding algorithm backpropagating to the neural network that calculates base-pairing probabilities through the connected base pairs.
Algorithm 1 The stochastic subgradient descent algorithm for structured support vector machines (SSVMs); is the predefined learning rate.

    1:

    initialize for all
    2:

    repeat
    3:

      for all  do
    4:

        
    5:

        for all  do
    6:

          
    7:

        end for
    8:

      end for
    9:

    until all the parameters converge

Open in a new tab
3. Results
3.1. Implementation

Our algorithm is implemented as the program Neuralfold, which is short for the neural network-based RNA folding algorithm. We employ Chainer [43] for the neural networks and the Python linear programming solver PuLP [44]. The source code for this implementation is available at https://github.com/keio-bioinformatics/neuralfold/, (accessed on 27 September 2022).
3.2. Datasets

We evaluated our algorithm with the Nussinov-style decoding algorithm for predicting pseudoknot-free RNA secondary structures using four datasets, TrainSetA, TestSetA, TrainSetB and TestSetB, which were established by [45].

TrainSetA and TestSetA are literature-based datasets [7,9,10,41,46] that were constructed to ensure sequence diversity. TrainSetA contains SSU and LSU domains, SRP RNAs, RNase P RNAs and tmRNAs comprising 3166 total sequences spanning 630,279 nt, with 333,466 forming base pairs (47.9%). The sequence lengths range from 10 to 734 nt, with an average length of 199 nt. TestSetA includes sequences from eight RNA families: 5S rRNA, group I and II introns, RNase P RNA, SRP RNA, tmRNA, tRNA, and telomerase RNA. TestSetA contains 697 sequences, with 51.7% of their bases forming base pairs. The sequence length ranges from 10 to 768 nt, with an average length of 195 nt. We excluded a number of sequences that contain pseudoknotted secondary structures in the original data sources from TestSetA. Thus, 593 sequences were selected as TestSetA.

TrainSetB and TestSetB, which contain 22 families with 3D structures [38], were assembled from Rfam [47]. TrainSetB and TestSetB include sequences from Rfam seed alignments with no more than 70% shared identity between sequences. TrainSetB comprises 22 RNA families, and its specific composition is 145.8S rRNAs, 18 U1 spliceosomal RNAs, 45 U4 spliceosomal RNAs, 233 riboswitches (from seven different families), 116 cis-regulatory elements (from nine different families), 3 ribozymes and a single bacteriophage pRNA. TrainSetB was constructed by selecting sequences dissimilar to those in TestSetB. TrainSetB contains 1094 sequences, including 112,398 nt in all, of which 52,065 bases (46.3%) formed base pairs. The sequence length is in the range of 27 to 237 nt with an average length of 103 nt. TrainSetB contains 4.3% noncanonical base pairs. TestSetB also consists of the same 22 RNA families as TrainSetB, TestSetB contains 430 sequences, including 52,097 nt in all, of which 22,728 bases (43.6%) form base pairs. The sequence length is in the range of 27 to 244 nt, with an average length of 121 nt. TestSetB contains 8.3% noncanonical base pairs.

We also evaluated our algorithm with the IPknot-style decoding algorithm for predicting pseudoknotted RNA secondary structures on two datasets. The first dataset is called the pk168 dataset [48], which was compiled from PseudoBase [20]. This dataset includes 16 categories of 168 pseudoknotted sequences with lengths <140 nt.

The second dataset is called RS-pk388, originally established by [36]. This dataset was obtained from the RNA STRAND database and contains 388 non-redundant sequences with lengths between 140 and 500 nt.
3.3. Prediction Performance

We evaluated the accuracy of RNA secondary structure predictions based on sensitivity () and positive predictive value () as follows:

Here, , and represent the numbers of true positives (i.e., the correctly predicted base pairs), false positives (i.e., incorrectly predicted base pairs), and false negatives (i.e., base pairs in the correct structure that were not predicted), respectively. As a balanced measure of and , we utilized their F-value, which is defined as their harmonic mean:

We conducted computational experiments on the datasets described in the previous section using the Nussinov-style decoding algorithm with the McCaskill and neural network models as well as the BiRNN and FNN models. We employed CentroidFold as the Nussinov decoding algorithm with the McCaskill model. We performed experiments on TestSetB using the parameters trained from TrainSetB. As shown in Table 1, the neural network models achieved better accuracy compared with the traditional model. Hereafter, we adopt the FNN model with k-mer contexts as the default Neuralfold model since it yielded better prediction accuracy in this experiment.
Table 1.

Accuracy of inferred base-pairing probabilities for TestSetB.
Implementation 	Model 	SEN 	PPV 	F
Neuralfold 	BiRNN 	0.649 	0.601 	0.624
Neuralfold 	FNN 	0.600 	0.700 	0.646
CentroidFold 	McCaskill 	0.513 	0.544 	0.528
Open in a new tab

The other computational experiments on the pseudoknotted dataset were conducted using the IPknot-style decoding algorithm with the McCaskill model with and without iterative refinement and with the Dirks–Pierce model as well as using Neuralfold with the FNN model. Table 2 shows that the feedforward neural network (FNN) model with 10-fold cross validation is comparable to IPknot with the Dirks–Pierce model for pseudoknots but superior to the McCaskill model both with and without iterative refinement.
Table 2.

Accuracy of inferred base-pairing probabilities for the pk168 dataset.
Implementation 	Model 	SEN 	PPV 	F
Neuralfold 	FNN 			
IPknot 	McCaskill w/o refine. 	0.619 	0.710 	0.661
IPknot 	McCaskill w/refine. 	0.753 	0.684 	0.717
IPknot 	Dirks–Pierce 	0.809 	0.749 	0.778
Open in a new tab

Table 3 shows the computation time for of the following sequences, which vary in length: PKB229 and PKB134 in the pk168 dataset; ASE_00193, CRW_00614 and CRW_00774 in the RNA STRAND database [49].
Table 3.

Computation time for calculating base-pairing probabilities of sequences of various lengths.
ID 	PKB229 	PKB134 	ASE_00193 	CRW_00614 	CRW_00774
Length (nt) 	67 	137 	301 	494 	989
Neuralfold
(FNN)
IPknot 	3.30 s 	27.78 s 	44.73 s 	60.22 s 	3 m 4.2 s
  (w/o refine.) 	0.01 s 	0.05 s 	0.18 s 	0.55 s 	2.64 s
  (w/refine.) 	0.03 s 	0.08 s 	0.31 s 	1.03 s 	5.86 s
  (D&P) 	8.36 s 	9 m 4.7 s 	n/a 	n/a 	n/a
Open in a new tab

Computation time was measured on an Intel Xeon E5-2680 (2.80 GHz) computer with 64 GB of memory and running Linux OS v2.6.32. FNN, feedforward neural network; D&P, Dirks–Pierce. IPknot with D&P failed to compute due to lack of memory for sequence lengths greater than 300.

This shows that the computation time for predicting a pseudoknotted secondary structure using the FNN model is comparably fast to IPknot with the Dirks–Pierce model.
3.4. Effects of Context Length

We evaluated the prediction accuracy obtained with the FNN model on the TestSetB and pk168 datasets for several lengths of k-mers input to neural networks. The accuracy as measured by , , and their F-value for different k-mer lengths is summarized in Figure 4. This analysis indicates that the accuracy is essentially maximized when the k-mer length is 81, and the difference in the accuracy for is negligible.
Figure 4.

Figure 4
Open in a new tab

The accuracy of the FNN model with different lengths of k-mers on the TestSetB dataset (left) and the pk168 dataset (right). , sensitivity; , positive predictive value; F, the F-value based on and .
3.5. Comparison with Previous Methods for Prediction of Pseudoknot-Free Secondary Structures

We compared our algorithm with previous methods for predicting pseudoknot-free RNA secondary structures including CentroidFold [18,19], CONTRAfold [7,8], RNAfold in the Vienna RNA package [4] and ContextFold [29]. For the posterior decoding methods with the trade-off parameter in Equation (4), we used . We performed secondary structure prediction on TestSetA with parameters trained on TrainSetA as well as prediction on TestSetB with the parameters trained on TrainSetB. The PPV–SEN plots for each method shown in Figure 5 indicate that our algorithm accurately predicts pseudoknot-free secondary structures in the datasets including famlilies similar with the training datasets.
Figure 5.

Figure 5
Open in a new tab

Positive predictive value–sensitivity (PPV–SEN) plots comparing our algorithm with competitive methods on TestSetA (Left) and TestSetB (Right).

On the other hand, to investigate the generalization ability of our method, another experiment in which our method was trained on TrainSetB and evaluated for accuracy on TestSetA showed that our method had very low accuracy (, , and ), which suggests that our method is severely overfitted.
3.6. Comparison with Alternative Methods for Predicting Pseudoknotted Secondary Structures

We also compared our algorithm with competing methods for predicting pseudoknotted secondary structures, including IPknot [36], HotKnots [32,33], and pknotsRG [29], as well as methods for predicting pseudoknot-free secondary structures, including CentroidFold [19] and RNAfold [4]. Neuralfold performed 10-fold cross validation on the pk168 and RS-pk388 datasets. Figure 6 shows PPV–SEN plots for each method, indicating that our algorithm works accurately on pseudoknotted datasets.
Figure 6.

Figure 6
Open in a new tab

Positive predictive value–sensitivity (PPV–SEN) plots comparing our algorithm with competitive methods on the pk168 dataset (Left) and the RS-pk388 dataset (Right). For the pk168 dataset, we set , for Neuralfold; , for IPknot with the Dirks–Pierce (D&P) model; , for IPknot with/without refinement; for CentroidFold. For the RS-pk388 dataset, we set , for Neuralfold; , for IPknot without refinement; , for IPknot with refinement; for CentroidFold.
4. Discussion

We propose a novel algorithm for directly inferring base-pairing probabilities with neural networks that enables us to predict RNA secondary structures accurately. Sato et al. [36] previously proposed an iterative algorithm that refines the base-pairing probabilities calculated by the McCaskill algorithm so as to be appropriate for pseudoknotted secondary structure prediction. The direct inference of base-pairing probabilities with neural networks is an approach similar to the iterative refinement algorithm in the sense that both directly update base-pairing probabilities, and the IPknot-style decoding algorithm then uses the base-pairing probabilities. Although the iterative refinement algorithm can improve the prediction accuracy of IPknot to some extent, it should be noted that this is an ad hoc algorithm, as there is no theoretical guarantee of improvement. Meanwhile, the neural networks that infer base-pairing probabilities are trained on given reference secondary structures by the max-margin framework, meaning that we can theoretically expect that the neural network models improve the secondary structure prediction. Indeed, Table 2 shows that our algorithm achieved not only better accuracy than the iterative refinement algorithm, but is also comparable to that of the Dirks–Pierce model, which can calculate exact base-pairing probabilities for a limited class of pseudoknots.

Recently, several methods for predicting RNA secondary structure using deep learning were proposed [13,14,15]. Although most of them use deep learning to compute matrices (N is the sequence length), which can be regarded as base-pairing probability matrices, they do not directly address the constraints that the RNA secondary structure must satisfy (e.g., Equations (5) and (6) for pseudoknot-free structures, and Equations (9)–(11) for pseudoknotted structures). On the other hand, MXfold2 [14] combines the Zuker-style dynamic programming [50] and deep learning to handle the constraints that pseudoknot-free RNA secondary structures must satisfy. UFold [15] predicts RNA secondary structure including pseudoknots using post-processing by linear programming, but does not directly address constraints on RNA secondary structure including pseudoknots when training deep learning models to predict base-pairing probabilities. By combining IPknot-style decoding with the max-margin training, the proposed Neuralfold can directly handle the constraints (9)–(11) that pseudoknotted RNA secondary structure must satisfy, not only when predicting secondary structures, but also when training deep learning models.

It has been pointed out that RNA secondary structure prediction based on machine learning and deep learning is prone to overfitting due to bias in the training data [14,45]. Several methods have been proposed to alleviate overfitting, such as using ensembles of multiple models [13], and integration with thermodynamic models [14]. UFold, on the other hand, employed artificially generated sequences and their predicted secondary structures for data augmentation, which were then used as additional training data to relax overfitting due to bias in the training data [15]. Our proposed method does not provide a strategy to counteract such overfitting and is therefore unsatisfactory at predicting sequences of families that are structurally distant from the training data, as shown in the results. However, by utilizing the ensembles of multiple models, as in SPOT-RNA, and the data augmentation strategy, as in UFold, it is expected to address to some extent the overfitting caused by bias in the training data.

The FNN model takes two k-mers around each base pair as input to infer its base-pairing probability, where k is the context length to model the length of loops and the contexts around the openings and closings of helices. As can be seen in Figure 7, different k-mer context lengths affect the prediction of pseudoknotted secondary structures. For example, consider the input bases when calculating the base-pairing probability of the blue-highlighted base pair (AU) using the FNN model. The FNN model with the context length k = 11 takes as input five bases in both the upstream and downstream directions from bases i and j. As seen in Figure 7 (bottom), the distances from bases A and U are 10 and 13 to Stem 2, respectively. This means that all the bases comprising Stem 2 are not completely located within the context length k = 11 around the base pair AU. On the other hand, for the FNN model with context length k = 41, all the bases of Stem 2 are completely located within the context around the base pair AU. This leads the FNN model to correctly predict the base pair AU, suggesting that a longer context length enables consideration of the dependency between stems in pseudoknotted substructures.
Figure 7.

Figure 7
Open in a new tab

(Top) Comparison between the reference structure of ID PKB189 (top-left) and the predicted structures with context lengths k = 11 (top-middle) and k = 41 (top-right). (Bottom) Distance between two stems (Stem 1 and Stem 2) in the pseudoknotted structure.
5. Conclusions

We propose a novel algorithm for directly inferring base-pairing probabilities with neural networks that enables us to accurately predict RNA secondary structures with pseudoknots. By combining IPknot-style decoding with the max-margin framework, our algorithm trains the model in the end-to-end manner to compute base-pairing probabilities under the constraints that RNA secondary structures, including pseudoknots, must satisfy. HotKnots 2.0 [32], on the other hand, finds a pseudoknotted secondary structure by using an MFE-based heuristic decoding algorithm with energy parameters of the Dirks–Pierce model or the Cao–Chen model trained on pseudoknotted reference structures. One of the advantages of our algorithm over HotKnots 2.0 is that no assumption about the architecture of RNA secondary structures is required. In other words, our model can be trained on arbitrary classes of pseudoknots, while HotKnots cannot be trained on more complicated classes of pseudoknots than the one assumed by the model. Furthermore, our algorithm can compute base-pairing probabilities, which can be used in various applications of RNA informatics, such as family classification [51,52], RNA–RNA interaction prediction [53] and simultaneous aligning and folding [54]. Accurate base-pairing probabilities calculated by our algorithm can improve the quality of such applications.
Acknowledgments

The supercomputer system was provided by the National Institute of Genetics (NIG), Research Organization of Information and Systems (ROIS).
Abbreviations

The following abbreviations are used in this manuscript:
BiRNN 	bi-directional recurrent neural network
FNN 	feedforward neural network
MEA 	maximum expected accuracy
MFE 	minimum free energy
ncRNA 	non-coding RNA
SSVM 	structured support vector machine
Open in a new tab
Author Contributions

Conceptualization, M.A. and K.S.; methodology, M.A. and K.S.; software, M.A.; validation, M.A. and K.S.; writing—original draft preparation, M.A.; writing—review and editing, K.S.; supervision, Y.S. and K.S.; project administration, K.S.; funding acquisition, K.S. All authors have read and agreed to the published version of the manuscript.
Institutional Review Board Statement

Not applicable.
Informed Consent Statement

Not applicable.
Data Availability Statement

Not applicable.
Conflicts of Interest

The authors declare no conflict of interest. The funders had no role in the design of the study; in the collection, analyses, or interpretation of data; in the writing of the manuscript; or in the decision to publish the results.
Funding Statement

This work was supported in part by a Grant-in-Aid for Scientific Research (KAKENHI) (16K00404, 19H04210 and 19K22897) from the Japan Society for the Promotion of Science (JSPS) to K.S.
Footnotes

Publisher’s Note: MDPI stays neutral with regard to jurisdictional claims in published maps and institutional affiliations.
References

    1.Hirose T., Mishima Y., Tomari Y. Elements and machinery of non-coding RNAs: Toward their taxonomy. EMBO Rep. 2014;15:489–507. doi: 10.1002/embr.201338390. [DOI] [PMC free article] [PubMed] [Google Scholar]
    2.Schroeder S.J., Turner D.H. Optical melting measurements of nucleic acid thermodynamics. Meth. Enzymol. 2009;468:371–387. doi: 10.1016/S0076-6879(09)68017-4. [DOI] [PMC free article] [PubMed] [Google Scholar]
    3.Turner D.H., Mathews D.H. NNDB: The nearest neighbor parameter database for predicting stability of nucleic acid secondary structure. Nucleic Acids Res. 2010;38:D280–D282. doi: 10.1093/nar/gkp892. [DOI] [PMC free article] [PubMed] [Google Scholar]
    4.Lorenz R., Bernhart S.H., Honer Zu Siederdissen C., Tafer H., Flamm C., Stadler P.F., Hofacker I.L. ViennaRNA Package 2.0. Algorithms Mol. Biol. 2011;6:26. doi: 10.1186/1748-7188-6-26. [DOI] [PMC free article] [PubMed] [Google Scholar]
    5.Reuter J.S., Mathews D.H. RNAstructure: Software for RNA secondary structure prediction and analysis. BMC BioInform. 2010;11:129. doi: 10.1186/1471-2105-11-129. [DOI] [PMC free article] [PubMed] [Google Scholar]
    6.Zuker M. On finding all suboptimal foldings of an RNA molecule. Science. 1989;244:48–52. doi: 10.1126/science.2468181. [DOI] [PubMed] [Google Scholar]
    7.Do C.B., Woods D.A., Batzoglou S. CONTRAfold: RNA secondary structure prediction without physics-based models. Bioinformatics. 2006;22:e90–e98. doi: 10.1093/bioinformatics/btl246. [DOI] [PubMed] [Google Scholar]
    8.Do C.B., Foo C.S., Ng A. Efficient multiple hyperparameter learning for log-linear models; Proceedings of the 20th International Conference on Neural Information Processing Systems; Vancouver, BC, Canada. 3–6 December 2007; Red Hook, NY, USA: Curran Associates Inc.; 2007. Advances in Neural Information Processing Systems 20. [Google Scholar]
    9.Andronescu M., Condon A., Hoos H.H., Mathews D.H., Murphy K.P. Efficient parameter estimation for RNA secondary structure prediction. Bioinformatics. 2007;23:19–28. doi: 10.1093/bioinformatics/btm223. [DOI] [PubMed] [Google Scholar]
    10.Andronescu M., Condon A., Hoos H.H., Mathews D.H., Murphy K.P. Computational approaches for RNA energy parameter estimation. RNA. 2010;16:2304–2318. doi: 10.1261/rna.1950510. [DOI] [PMC free article] [PubMed] [Google Scholar]
    11.Zakov S., Goldberg Y., Elhadad M., Ziv-Ukelson M. Rich parameterization improves RNA structure prediction. J. Comput. Biol. 2011;18:1525–1542. doi: 10.1089/cmb.2011.0184. [DOI] [PubMed] [Google Scholar]
    12.Akiyama M., Sato K., Sakakibara Y. A max-margin training of RNA secondary structure prediction integrated with the thermodynamic model. J. Bioinform. Comput. Biol. 2018;16:1840025. doi: 10.1142/S0219720018400255. [DOI] [PubMed] [Google Scholar]
    13.Singh J., Hanson J., Paliwal K., Zhou Y. RNA secondary structure prediction using an ensemble of two-dimensional deep neural networks and transfer learning. Nat. Commun. 2019;10:5407. doi: 10.1038/s41467-019-13395-9. [DOI] [PMC free article] [PubMed] [Google Scholar]
    14.Sato K., Akiyama M., Sakakibara Y. RNA secondary structure prediction using deep learning with thermodynamic integration. Nat. Commun. 2021;12:941. doi: 10.1038/s41467-021-21194-4. [DOI] [PMC free article] [PubMed] [Google Scholar]
    15.Fu L., Cao Y., Wu J., Peng Q., Nie Q., Xie X. UFold: Fast and accurate RNA secondary structure prediction with deep learning. Nucleic Acids Res. 2022;50:e14. doi: 10.1093/nar/gkab1074. [DOI] [PMC free article] [PubMed] [Google Scholar]
    16.Carvalho L.E., Lawrence C.E. Centroid estimation in discrete high-dimensional spaces with applications in biology. Proc. Natl. Acad. Sci. USA. 2008;105:3209–3214. doi: 10.1073/pnas.0712329105. [DOI] [PMC free article] [PubMed] [Google Scholar]
    17.McCaskill J.S. The equilibrium partition function and base pair binding probabilities for RNA secondary structure. Biopolymers. 1990;29:1105–1119. doi: 10.1002/bip.360290621. [DOI] [PubMed] [Google Scholar]
    18.Hamada M., Kiryu H., Sato K., Mituyama T., Asai K. Prediction of RNA secondary structure using generalized centroid estimators. Bioinformatics. 2009;25:465–473. doi: 10.1093/bioinformatics/btn601. [DOI] [PubMed] [Google Scholar]
    19.Sato K., Hamada M., Asai K., Mituyama T. CENTROIDFOLD: A web server for RNA secondary structure prediction. Nucleic Acids Res. 2009;37:W277–W280. doi: 10.1093/nar/gkp367. [DOI] [PMC free article] [PubMed] [Google Scholar]
    20.van Batenburg F.H., Gultyaev A.P., Pleij C.W. PseudoBase: Structural information on RNA pseudoknots. Nucleic Acids Res. 2001;29:194–195. doi: 10.1093/nar/29.1.194. [DOI] [PMC free article] [PubMed] [Google Scholar]
    21.Staple D.W., Butcher S.E. Pseudoknots: RNA structures with diverse functions. PLoS Biol. 2005;3:e213. doi: 10.1371/journal.pbio.0030213. [DOI] [PMC free article] [PubMed] [Google Scholar]
    22.Brierley I., Pennell S., Gilbert R.J. Viral RNA pseudoknots: Versatile motifs in gene expression and replication. Nat. Rev. Microbiol. 2007;5:598–610. doi: 10.1038/nrmicro1704. [DOI] [PMC free article] [PubMed] [Google Scholar]
    23.Fechter P., Rudinger-Thirion J., Florentz C., Giege R. Novel features in the tRNA-like world of plant viral RNAs. Cell. Mol. Life Sci. 2001;58:1547–1561. doi: 10.1007/PL00000795. [DOI] [PMC free article] [PubMed] [Google Scholar]
    24.Akutsu T. Dynamic programming algorithms for RNA secondary structure prediction with pseudoknots. Discret. Appl. Math. 2000;104:45–62. doi: 10.1016/S0166-218X(00)00186-4. [DOI] [Google Scholar]
    25.Lyngsø R.B., Pedersen C.N. RNA pseudoknot prediction in energy-based models. J. Comput. Biol. 2000;7:409–427. doi: 10.1089/106652700750050862. [DOI] [PubMed] [Google Scholar]
    26.Rivas E., Eddy S.R. A dynamic programming algorithm for RNA structure prediction including pseudoknots. J. Mol. Biol. 1999;285:2053–2068. doi: 10.1006/jmbi.1998.2436. [DOI] [PubMed] [Google Scholar]
    27.Dirks R.M., Pierce N.A. A partition function algorithm for nucleic acid secondary structure including pseudoknots. J. Comput. Chem. 2003;24:1664–1677. doi: 10.1002/jcc.10296. [DOI] [PubMed] [Google Scholar]
    28.Dirks R.M., Pierce N.A. An algorithm for computing nucleic acid base-pairing probabilities including pseudoknots. J. Comput. Chem. 2004;25:1295–1304. doi: 10.1002/jcc.20057. [DOI] [PubMed] [Google Scholar]
    29.Reeder J., Giegerich R. Design, implementation and evaluation of a practical pseudoknot folding algorithm based on thermodynamics. BMC Bioinform. 2004;5:104. doi: 10.1186/1471-2105-5-104. [DOI] [PMC free article] [PubMed] [Google Scholar]
    30.Jabbari H., Wark I., Montemagno C., Will S. Knotty: Efficient and Accurate Prediction of Complex RNA Pseudoknot Structures. Bioinformatics. 2018;34:3849–3856. doi: 10.1093/bioinformatics/bty420. [DOI] [PubMed] [Google Scholar]
    31.Ruan J., Stormo G.D., Zhang W. An iterated loop matching approach to the prediction of RNA secondary structures with pseudoknots. Bioinformatics. 2004;20:58–66. doi: 10.1093/bioinformatics/btg373. [DOI] [PubMed] [Google Scholar]
    32.Andronescu M.S., Pop C., Condon A.E. Improved free energy parameters for RNA pseudoknotted secondary structure prediction. RNA. 2010;16:26–42. doi: 10.1261/rna.1689910. [DOI] [PMC free article] [PubMed] [Google Scholar]
    33.Ren J., Rastegari B., Condon A., Hoos H.H. HotKnots: Heuristic prediction of RNA secondary structures including pseudoknots. RNA. 2005;11:1494–1504. doi: 10.1261/rna.7284905. [DOI] [PMC free article] [PubMed] [Google Scholar]
    34.Chen X., He S.M., Bu D., Zhang F., Wang Z., Chen R., Gao W. FlexStem: Improving predictions of RNA secondary structures with pseudoknots by reducing the search space. Bioinformatics. 2008;24:1994–2001. doi: 10.1093/bioinformatics/btn327. [DOI] [PubMed] [Google Scholar]
    35.Bellaousov S., Mathews D.H. ProbKnot: Fast prediction of RNA secondary structure including pseudoknots. RNA. 2010;16:1870–1880. doi: 10.1261/rna.2125310. [DOI] [PMC free article] [PubMed] [Google Scholar]
    36.Sato K., Kato Y., Hamada M., Akutsu T., Asai K. IPknot: Fast and accurate prediction of RNA secondary structures with pseudoknots using integer programming. Bioinformatics. 2011;27:85–93. doi: 10.1093/bioinformatics/btr215. [DOI] [PMC free article] [PubMed] [Google Scholar]
    37.Sato K., Kato Y. Prediction of RNA secondary structure including pseudoknots for long sequences. Brief. Bioinform. 2022;23:bbab395. doi: 10.1093/bib/bbab395. [DOI] [PMC free article] [PubMed] [Google Scholar]
    38.Rivas E. The four ingredients of single-sequence RNA secondary structure prediction. A unifying perspective. RNA Biol. 2013;10:1185–1196. doi: 10.4161/rna.24971. [DOI] [PMC free article] [PubMed] [Google Scholar]
    39.Cao S., Chen S.J. Predicting RNA pseudoknot folding thermodynamics. Nucleic Acids Res. 2006;34:2634–2652. doi: 10.1093/nar/gkl346. [DOI] [PMC free article] [PubMed] [Google Scholar]
    40.Nussinov R., Pieczenick G., Griggs J., Kleitman D. Algorithms for loop matching. SIAM J. Appl. Math. 1978;35:68–82. doi: 10.1137/0135006. [DOI] [Google Scholar]
    41.Dowell R.D., Eddy S.R. Evaluation of several lightweight stochastic context-free grammars for RNA secondary structure prediction. BMC Bioinform. 2004;5:71. doi: 10.1186/1471-2105-5-71. [DOI] [PMC free article] [PubMed] [Google Scholar]
    42.Tsochantaridis I., Joachims T., Hofmann T., Altun Y. Large Margin Methods for Structured and Interdependent Output Variables. J. Mach. Learn. Res. 2005;6:1453–1484. [Google Scholar]
    43.Tokui S., Oono K., Hido S., Clayton J. Chainer: A Next-Generation Open Source Framework for Deep Learning; Proceedings of the Workshop on Machine Learning Systems (LearningSys) in The Twenty-Ninth Annual Conference on Neural Information Processing Systems (NIPS); Montréal, QC, Canada. 11–12 December 2015. [Google Scholar]
    44.Mitchell S., Consulting S.M., O’sullivan M., Dunning I. PuLP: A Linear Programming Toolkit for Python. 2011. [(accessed on 27 September 2022)]. Available online: https://optimization-online.org/2011/09/3178/
    45.Rivas E., Lang R., Eddy S.R. A range of complex probabilistic models for RNA secondary structure prediction that includes the nearest-neighbor model and more. RNA. 2012;18:193–212. doi: 10.1261/rna.030049.111. [DOI] [PMC free article] [PubMed] [Google Scholar]
    46.Lu Z.J., Gloor J.W., Mathews D.H. Improved RNA secondary structure prediction by maximizing expected pair accuracy. RNA. 2009;15:1805–1813. doi: 10.1261/rna.1643609. [DOI] [PMC free article] [PubMed] [Google Scholar]
    47.Gardner P.P., Daub J., Tate J., Moore B.L., Osuch I.H., Griffiths-Jones S., Finn R.D., Nawrocki E.P., Kolbe D.L., Eddy S.R., et al. Rfam: Wikipedia, clans and the “decimal” release. Nucleic Acids Res. 2011;39:D141–D145. doi: 10.1093/nar/gkq1129. [DOI] [PMC free article] [PubMed] [Google Scholar]
    48.Huang X., Ali H. High sensitivity RNA pseudoknot prediction. Nucleic Acids Res. 2007;35:656–663. doi: 10.1093/nar/gkl943. [DOI] [PMC free article] [PubMed] [Google Scholar]
    49.Andronescu M., Bereg V., Hoos H.H., Condon A. RNA STRAND: The RNA secondary structure and statistical analysis database. BMC Bioinform. 2008;9:340. doi: 10.1186/1471-2105-9-340. [DOI] [PMC free article] [PubMed] [Google Scholar]
    50.Zuker M., Stiegler P. Optimal computer folding of large RNA sequences using thermodynamics and auxiliary information. Nucleic Acids Res. 1981;9:133–148. doi: 10.1093/nar/9.1.133. [DOI] [PMC free article] [PubMed] [Google Scholar]
    51.Sato K., Mituyama T., Asai K., Sakakibara Y. Directed acyclic graph kernels for structural RNA analysis. BMC Bioinform. 2008;9:318. doi: 10.1186/1471-2105-9-318. [DOI] [PMC free article] [PubMed] [Google Scholar]
    52.Morita K., Saito Y., Sato K., Oka K., Hotta K., Sakakibara Y. Genome-wide searching with base-pairing kernel functions for noncoding RNAs: Computational and expression analysis of snoRNA families in Caenorhabditis elegans. Nucleic Acids Res. 2009;37:999–1009. doi: 10.1093/nar/gkn1054. [DOI] [PMC free article] [PubMed] [Google Scholar]
    53.Kato Y., Sato K., Hamada M., Watanabe Y., Asai K., Akutsu T. RactIP: Fast and accurate prediction of RNA-RNA interaction using integer programming. Bioinformatics. 2010;26:i460–i466. doi: 10.1093/bioinformatics/btq372. [DOI] [PMC free article] [PubMed] [Google Scholar]
    54.Sato K., Kato Y., Akutsu T., Asai K., Sakakibara Y. DAFS: Simultaneous aligning and folding of RNA sequences via dual decomposition. Bioinformatics. 2012;28:3218–3224. doi: 10.1093/bioinformatics/bts612. [DOI] [PubMed] [Google Scholar]

Associated Data

This section collects any data citations, data availability statements, or supplementary materials included in this article.
Data Availability Statement

Not applicable.

Articles from Genes are provided here courtesy of Multidisciplinary Digital Publishing Institute (MDPI)
ACTIONS

    View on publisher site
    PDF (556.0 KB)

RESOURCES
On this page

    Abstract
    1. Introduction
    2. Methods
    3. Results
    4. Discussion
    5. Conclusions
    Acknowledgments
    Abbreviations
    Author Contributions
    Institutional Review Board Statement
    Informed Consent Statement
    Data Availability Statement
    Conflicts of Interest
    Funding Statement
    Footnotes
    References
    Associated Data


Follow NCBI
NCBI on X (formerly known as Twitter)
NCBI on Facebook
NCBI on LinkedIn
NCBI on GitHub
NCBI RSS feed

Connect with NLM
NLM on X (formerly known as Twitter)
NLM on Facebook
NLM on YouTube

National Library of Medicine
8600 Rockville Pike
Bethesda, MD 20894

    Web Policies
    FOIA
    HHS Vulnerability Disclosure

    Help
    Accessibility
    Careers

    NLM
    NIH
    HHS
    USA.gov

====

























Loading web-font Gyre-Pagella/Normal/Regular
[MDPI Open Access Journals]

    Journals Topics Information Author Services Initiatives About 

Sign In / Sign Up Submit
 
Search for Articles:
Advanced
 
Journals
Genes
Volume 13
Issue 11
10.3390/genes13112155
genes-logo
Submit to this Journal Review for this Journal Propose a Special Issue
Article Menu

    Academic Editors
    Zihua Hu
    Michel Ravelonandro
    Lionel Benard
    Subscribe SciFeed
    Recommended Articles
    Related Info Links
    More by Authors Links

Article Views 2376
Citations 4

    Table of Contents
        Abstract
        Introduction
        Methods
        Results
        Discussion
        Conclusions
        Author Contributions
        Funding
        Institutional Review Board Statement
        Informed Consent Statement
        Data Availability Statement
        Acknowledgments
        Conflicts of Interest
        Abbreviations
        References

share Share
announcement Help
format_quote Cite
question_answer Discuss in SciProfiles
first_page
settings
Order Article Reprints
Open AccessArticle
Direct Inference of Base-Pairing Probabilities with Neural Networks Improves Prediction of RNA Secondary Structures with Pseudoknots
by Manato Akiyama
1, Yasubumi Sakakibara
1 [ORCID] and Kengo Sato
2,* [ORCID]
1
Department of Biosciences and Informatics, Keio University, 3-14-1 Hiyoshi, Kohoku-ku, Yokohama 223-8522, Japan
2
School of System Design and Technology, Tokyo Denki University, 5 Senju Asahi-cho, Adachi-ku, Tokyo 120-8551, Japan
*
Author to whom correspondence should be addressed.
Genes 2022, 13(11), 2155; https://doi.org/10.3390/genes13112155
Submission received: 28 September 2022 / Revised: 15 November 2022 / Accepted: 16 November 2022 / Published: 18 November 2022
(This article belongs to the Special Issue Feature Papers in RNA)
Download
keyboard_arrow_down
Browse Figures
Versions Notes

Abstract
Existing approaches to predicting RNA secondary structures depend on how the secondary structure is decomposed into substructures, that is, the architecture, to define their parameter space. However, architecture dependency has not been sufficiently investigated, especially for pseudoknotted secondary structures. In this study, we propose a novel algorithm for directly inferring base-pairing probabilities with neural networks that do not depend on the architecture of RNA secondary structures, and then implement this approach using two maximum expected accuracy (MEA)-based decoding algorithms: Nussinov-style decoding for pseudoknot-free structures and IPknot-style decoding for pseudoknotted structures. To train the neural networks connected to each base pair, we adopt a max-margin framework, called structured support vector machines (SSVM), as the output layer. Our benchmarks for predicting RNA secondary structures with and without pseudoknots show that our algorithm outperforms existing methods in prediction accuracy.
Keywords:
RNA secondary structure; deep learning; pseudoknots

1. Introduction
The roles of functional non-coding RNAs (ncRNAs) in regulating transcription and guiding post-transcriptional modification have been recently shown to be critical in various biological processes, ranging from development and cell differentiation in healthy individuals to disease pathogenesis [1]. The well-established relationship between the primary sequence and structure of ncRNAs has motivated research aiming to elucidate the functions of ncRNAs by determining their structures.
Yet, methods for experimentally determining RNA tertiary structures utilizing X-ray crystal structure analysis and nuclear magnetic resonance (NMR) are costly and labor-intensive, thus restricting their application. Accordingly, researchers often carry out computational prediction of RNA secondary structures based on the analysis of base pairs comprising nucleotides joined by hydrogen bonds.
Computational approaches to RNA secondary structure prediction often utilize thermodynamic models (e.g., Turner’s nearest neighbor model [2,3]) that define characteristic substructures, such as base-pair stacking and hairpin loops. In computational approaches, the free energy of each type of substructure is first empirically determined by methods such as optical melting experiments [2]. Then, the free energy of RNA secondary structures can be estimated as the sum of the free energy of their substructures. Dynamic programming can then be used to determine the optimal secondary structure that minimizes free energy for a given RNA sequence. This approach is employed by RNAfold [4], RNAstructure [5] and UNAfold [6], among other tools.
As an alternative to experimental approaches, machine learning can be utilized to train scoring parameters based on the substructures constituting reference structures. This type of approach, as implemented in CONTRAfold [7,8], Simfold [9,10], ContextFold [11] and similar tools, has improved the accuracy of RNA secondary structure prediction. By integrating thermodynamic and machine-learning-based weighting approaches, MXfold avoided overfitting and achieved better performance than models based on either one alone [12]. Furthermore, interest in the use of deep learning for RNA secondary structure prediction is rapidly increasing [13,14,15]. MXfold2 used thermodynamic regularization to train a deep neural network so that the predicted folding score and free energy are as close as possible. This method showed robust prediction results in familywise cross validation, where the test dataset was structurally different from the training dataset.
Another important aspect of RNA secondary structure prediction is the choice of the decoding algorithm used to find the optimal secondary structure from among all possible secondary structures. Two classic decoding algorithms are the minimum free energy (MFE) algorithm, which is used in thermodynamic approaches, and the maximum likelihood estimation (MLE) algorithm, which is used in machine-learning-based approaches. These algorithms find a secondary structure that minimizes the free energy and maximizes the probability or scoring function, respectively. Another option is a posterior decoding algorithm based on the maximum expected accuracy (MEA) principle, which is known to be an effective approach for many high-dimensional combinatorial optimization problems [16]. As researchers usually evaluate the prediction of RNA secondary structures using base-pair-wise accuracy measures, MEA-based decoding algorithms utilize posterior base-pairing probabilities that can be calculated by the McCaskill algorithm [17] or the inside–outside algorithm for stochastic context-free grammars. CONTRAfold [18] and CentroidFold [19] both have MEA-based decoding algorithm implementations that successfully predict RNA secondary structures.
Pseudoknots, an important structural element in RNA secondary structures, occur when at least two hydrogen bonds cross each other, and are typically drawn as two crossing arcs above a primary sequence (Figure 1).
Genes 13 02155 g001 550
Figure 1. An example of pseudoknots.
Many RNAs, including rRNAs, tmRNAs and viral RNAs, form pseudoknotted secondary structures [20]. Pseudoknots are known to be involved in the regulation of translation and splicing as well as ribosomal frame shifting [21,22]. Furthermore, pseudoknots support folding into 3D structures in many cases [23]. Therefore, the impact of pseudoknots cannot be ignored in the structural and functional analysis of RNAs.
However, all of the aforementioned algorithms cannot consider pseudoknotted secondary structures owing to computational complexity. It has been proven that the problem of finding MFE structures including arbitrary pseudoknots is NP-hard [24,25]. Therefore, practically available algorithms for predicting pseudoknotted RNA secondary structures fall into one of the following two approaches: exact algorithms for a limited class of pseudoknots, such as PKNOTS [26], NUPACK [27,28], pknotsRG [29] and Knotty [30]; and heuristic algorithms that do not guarantee that the optimal structure will be found, such as ILM [31], HotKnots [32,33], FlexStem [34] and ProbKnot [35].
We previously developed IPknot, which enables fast and accurate prediction of RNA secondary structures with pseudoknots using integer programming [36,37]. IPknot adopts an MEA-based decoding algorithm that utilizes base-pairing probabilities combined with an approximate decomposition of a pseudoknotted structure into hierarchical pseudoknot-free structures. The prediction performance of IPknot is sufficient in terms of speed and accuracy compared with heuristic algorithms, and it is much faster than the exact algorithms.
Both thermodynamic approaches and machine-learning-based approaches depend on the method by which a secondary structure is decomposed into substructures, that is, the architecture (as referred to in [38]), to define their parameter space. Turner’s nearest neighbor model is the most well-studied architecture for predicting pseudoknot-free secondary structures, while the energy models for pseudoknotted secondary structures have not been sufficiently investigated, except for the Dirks–Pierce model [27,28] and the Cao–Chen model [39] for limited classes of pseudoknots. To our knowledge, an effective and efficient procedure to find a suitable architecture that can predict RNA secondary structures more accurately is still unknown.
Here, we propose a novel algorithm to directly infer base-pairing probabilities with neural networks instead of the McCaskill algorithm or the inside–outside algorithm, which both depend on the architecture of RNA secondary structures. Then, we employ the inferred base-pairing probabilities as part of a MEA-based scoring function for the two decoding algorithms: Nussinov-style decoding for pseudoknot-free structures, and IPknot-style decoding for pseudoknotted structures. To train the neural networks connected to each base pair, we adopt a max-margin framework, called structured support vector machines (SSVMs), as the output layer. We implement two types of neural networks connected to each base pair: bidirectional recursive neural networks (BiRNN) over tree structures and multilayer feedforward neural networks (FNN) with k-mer contexts around both bases in a pair. Our benchmarks for predicting RNA secondary structures with and without pseudoknots show that the prediction accuracy of our algorithm is superior to that of existing methods.
The major advantages of our work are summarized as follows: (i) our algorithm enables us to accurately predict RNA secondary structures with and without pseudoknots; (ii) our algorithm assumes no prior knowledge of the architecture that defines the decomposition of RNA secondary structures and thus the corresponding parameter space.
2. Methods
2.1. Preliminaries
The RNA sequence structure is modeled following the setup used by Akiyama et al. [12]. First, let Σ = { A , C , G , U } , and let Σ * represent the set of all finite RNA sequences comprised of bases in Σ . For a sequence x = x 1 x 2 ⋯ x n ∈ Σ * , let | x | represent the number of bases in x, referred to as the length of x. Let S ( x ) represent the set of all possible secondary structures formed by x. A secondary structure y ∈ S ( x ) can be described as a | x | × | x | binary-valued triangular matrix y = ( y i j ) i < j , in which y i j = 1 if and only if bases x i and x j form a base pair linked by hydrogen bonds, including both canonical Watson–Crick base pairs (i.e., G-C and A-U) and non-canonical wobble base pairs (e.g., G-U).
2.2. MEA-Based Scoring Function
We employ the maximum expected accuracy (MEA)-based scoring function originally used for IPknot [36,37].
A secondary structure y ∈ S ( x ) is assumed to be decomposable into a set of pseudoknot-free substructures ( y ( 1 ) , y ( 2 ) , … , y ( m ) ) satisfying the following two conditions: (i) y ∈ S ( x ) can be decomposed into a mutually-exclusive set, that is, for 1 ≤ i < j ≤ | x | , ∑ 1 ≤ p ≤ m y i j ( p ) ≤ 1 ; and (ii) each base pair in y ( p ) can be pseudoknotted to at least one base pair in y ( q ) for ∀ q < p . Each pseudoknot-free substructure y ( p ) is said to belong to level p. For each RNA secondary structure y ∈ S ( x ) , there exists a positive integer m such that y is decomposable into m substructures without one or more pseudoknots (for more details, see the Supplementary Materials of [36]). Through the above decomposition, arbitrary pseudoknots can be modeled by our method.
First, to construct an MEA-based scoring function, we define a gain function of y ^ ∈ S ( x ) with respect to the correct secondary structure y ∈ S ( x ) as follows:
G γ ( y , y ^ ) = γ T P ( y , y ^ ) + T N ( y , y ^ ) = ∑ i < j γ I ( y i j = 1 ) I ( y ^ i j = 1 ) + I ( y i j = 0 ) I ( y ^ i j = 0 ) .
(1)
Here, γ > 0 represents a base-pair weight parameter, T N and T P represent the numbers of true negatives (non-base pairs) and true positives (base pairs), respectively, and I ( c o n d i t i o n ) is an indicator function returning a value of either 1 or 0 depending on whether the c o n d i t i o n is true or false.
The objective is to identify a secondary structure y ^ that maximizes the expected value of the above gain function (1) under a given probability distribution over the space S ( x ) of pseudoknotted secondary structures:
E y ∣ x [ G γ ( y , y ^ ) ] = ∑ y ∈ S ( x ) G γ ( y , y ^ ) P ( y ∣ x ) .
(2)
Here, P ( y ∣ x ) is the probability distribution of RNA secondary structures including pseudoknots. The γ -centroid estimator (2) has been proven to allow us to decode secondary structures accurately based on a given probability distribution [18].
Accordingly, the expected gain function (2) can be approximated as the sum of the expected gain functions for each level of pseudoknot-free substructures ( y ^ ( 1 ) , … , y ^ ( m ) ) in the decomposed set of a pseudoknotted structure y ^ ∈ S ( x ) . Thus, a pseudoknotted structure y ^ and its decomposition ( y ^ ( 1 ) , … , y ^ ( m ) ) can be found that maximize the following expected value:
E y ∣ x [ G γ ( y , y ^ ) ] ≃ ∑ 1 ≤ p ≤ m ∑ y ∈ S ( x ) G γ ( p ) ( y , y ^ ( p ) ) P ( y ∣ x ) = ∑ 1 ≤ p ≤ m ∑ i < j ( γ ( p ) + 1 ) p i j − 1 y ^ i j ( p ) + C .
(3)
Here, γ ( p ) > 0 is a weight parameter for level p base pairs and C is a constant that is independent of y ^ (for the derivation, see the Supplementary Material of [18]). The base-pairing probability p i j represents the probability of base x i being paired with x j . As seen in Section 2.4, we employ one of three algorithms to calculate base-pairing probabilities.
It should be noted that IPknot can be considered an extension of CentroidFold [18]. For the restricted case of a single decomposed level (i.e., m = 1 ), the approximate expected gain function (3) of IPknot is equivalent to CentroidFold’s γ -centroid estimator.
2.3. Decoding Algorithms
2.3.1. Nussinov-Style Decoding Algorithm for Pseudoknot-Free Structures
For the prediction of pseudoknot-free secondary structures, we find y ^ that maximizes the expected gain (3) with m = 1 under the following constraints on base pairs:
maximize ∑ i < j ( γ + 1 ) p i j − 1 y ^ i j
(4)
subject to ∑ j = 1 i − 1 y j i + ∑ j = i + 1 | x | y i j ≤ 1 ( 1 ≤ ∀ i ≤ | x | ) ,
(5)
y i j + y k l ≤ 1 ( 1 ≤ ∀ i < ∀ k < ∀ j < ∀ l ≤ | x | ) .
(6)
The constraint defined by Equation (5) means that each base x i can be paired with at most one base. The constraint defined by Equation (6) disallows pseudoknot.
This integer programming (IP) problem can be solved by dynamic programming as follows, similar to the Nussinov algorithm [40],
M i , j = max M i + 1 , j M i , j − 1 M i + 1 , j − 1 + ( γ + 1 ) p i j − 1 max i < k < j M i , k + M k + 1 , j ,
(7)
and then tracing back from M 1 , | x | .
2.3.2. IPknot-Style Decoding Algorithm for Pseudoknotted Structures
Maximization of the approximate expected gain (3) can be solved as the following IP problem:
maximize ∑ 1 ≤ p ≤ m ∑ i < j ( γ ( p ) + 1 ) p i j − 1 y ^ i j ( p )
(8)
subject to ∑ 1 ≤ p ≤ m ∑ j = 1 i − 1 y j i ( p ) + ∑ j = i + 1 | x | y i j ( p ) ≤ 1 ( 1 ≤ ∀ i ≤ | x | ) , y i j ( p ) + y k l ( p ) ≤ 1
(9)
( 1 ≤ ∀ p ≤ m , 1 ≤ ∀ i < ∀ k < ∀ j < ∀ l ≤ | x | ) , ∑ i < k < j < l y i j ( q ) + ∑ k < i ′ < l < j ′ y i ′ j ′ ( q ) ≥ y k l ( p )
(10)
( 1 ≤ ∀ q < ∀ p ≤ m , 1 ≤ ∀ k < ∀ l ≤ | x | ) .
(11)
Note that Equation (3) requires the consideration of only base pairs y i j ( p ) with base-pairing probabilities p i j being greater than θ ( p ) = 1 / ( γ ( p ) + 1 ) . The constraint defined by Equation (9) means that each base x i can be paired with, at most, one base. The constraint defined by Equation (10) disallows pseudoknots within the same level p. The constraint defined by Equation (11) ensures that each level-p base pair is pseudoknotted to at least one base pair at each lower level q < p . We set m = 2 , which is IPknot’s default setting. This suggests that the predicted structure can be decomposed into two pseudoknot-free secondary structures.
2.4. Inferring Base-Paring Probabilities
Our scoring function (3) described in Section 2.2 is calculated by using base-pairing probabilities p i j . In this section, we introduce two approaches for computing base-pairing probabilities. The first approach is a traditional one that is based on the probability distribution of RNA secondary structures, e.g., the McCaskill model [17] for pseudoknot-free structures and its extension to pseudoknotted structures, e.g., the Dirks–Pierce model [27,28]. The second approach proposed in this paper directly calculates base-pairing probabilities using neural networks.
2.4.1. Traditional Models for Base-Pairing Probabilities
The base-pairing probability p i j is defined as
p i j = ∑ y ∈ S ( x ) I ( y i j = 1 ) P ( y ∣ x )
(12)
from a probability distribution P ( y ∣ x ) over a set S ( x ) of secondary structures with or without pseudoknots.
For predicting pseudoknot-free structures, the McCaskill model [17] can be mostly used as P ( y ∣ x ) combined with the Nussinov-style decoding algorithm described in Section 2.3.1. The computational complexity of calculating Equation (12) for the McCaskill model is O ( | x | 3 ) for time and O ( | x | 2 ) for space when using dynamic programming. This model was implemented previously as CentroidFold [18,19].
For predicting pseudoknotted structures, we can select P ( y ∣ x ) from among several models. A naïve model could use the probability distribution with pseudoknots as well as Equation (2) in spite of high computational costs, e.g., the Dirks–Pierce model [27,28] for a limited class of pseudoknots, with a computational complexity of O ( | x | 5 ) for time and O ( | x | 4 ) for space. Alternatively, we can employ a probability distribution without pseudoknots for each decomposed pseudoknot-free structure, such as the McCaskill model. Furthermore, to increase the prediction accuracy, we can utilize a heuristic algorithm with iterative refinement that refines the base-pairing probability matrix from the distribution without pseudoknots. See [36] for more details. These three models were implemented in IPknot [36].
2.4.2. Neural Network Models
In this research, we propose two neural network architectures for calculating base-pairing probabilities instead of the probability distribution over all RNA secondary structures.
The first architecture is the bidirectional recursive neural network (BiRNN) over tree structures as shown in Figure 2. Stochastic context-free grammars (SCFG) can model RNA secondary structure without pseudoknots [7,41]. The layers of BiRNN over the tree structure are connected along grammatical trees derived from SCFG that models RNA secondary structures. The BiRNN consists of three matrices—(a) the inside RNN matrix, (b) the outside RNN matrix and (c) the inside–outside matrix—for outputting base-pairing probabilities, each of whose elements contain a network layer (indicated by circles in Figure 2) with 80 hidden nodes. Each layer in the inside or outside matrix is recursively calculated from connected source layers as in the inside or outside algorithm, respectively, for stochastic context-free grammars (SCFG). The ReLU activation function is applied before being input to each recursive node. The base-pairing probability at each position is calculated from the corresponding layers in the inside and outside matrices with the sigmoid activation function. Our implementation of BiRNN assumes a simple RNA grammar
S → a S a ^ ∣ a S ∣ S a ∣ S S ∣ ϵ ,
where a ∈ Σ , a and a ^ represent the paired bases, S represents the start non-terminal symbol, and ϵ represents the empty string.
Genes 13 02155 g002 550
Figure 2. A bidirectional recursive neural network for calculating base-pairing probabilities. A set of four dots above each base represents the one-hot representation of the base. Each circle indicates a network layer with 80 hidden nodes. Each solid arrow indicate a connection between layers along grammatical trees derived from the RNA grammar. Each dashed arrow represents a connection that aggregates the inside and outside layers to output base-pairing probabilities.
The second architecture employs a simple multilayer feedforward neural network (FNN). To calculate the base-pairing probability p i j , a FNN receives as input two k-mers around the i-th and j-th bases as shown in Figure 3.
Genes 13 02155 g003 550
Figure 3. A feedforward neural network with k ( = 9 ) -mer contexts around x i and x j used to calculate the base-pairing probability p i j . The end-of-loop nodes of the highlighted nucleotides are activated because they are beyond the paired bases.
Each base is encoded by the one-hot encoding of nucleotides and an additional node that indicates the end of the loop, which should be active for x l s.t. l ≥ j in the left k-mer around x i or x l s.t. l ≤ i in the right k-mer around x j . This encoding can be expected to embed the length of loops and the contexts around the openings and closings of helices. We set k = 81 for the k-mer context length default (for more details, see Section 3.4). We then construct two hidden layers consisting of 200 and 50 nodes, respectively, with the ReLU activation function and one output node with a sigmoid activation function to output base-pairing probabilities.
Note that the FNN model depends on no assumption of RNA secondary structures, while the BiRNN model assumes an RNA grammar that considers no pseudoknots. Instead, the FNN model can take longer contexts around each base pair into consideration by using longer k-mers.
2.5. Learning Algorithm
We optimize the network parameters λ by using a max-margin framework called a structured support vector machine (SSVM) [42]. For a training dataset D = { ( x ( k ) , y ( k ) ) } k = 1 K , where x ( k ) represents the k-th RNA sequence and y ( k ) ∈ S ( x ( k ) ) represents the correct secondary structure of the k-th sequence x ( k ) , we identify a λ that minimizes the objective function
L ( λ ) = ∑ ( x , y ) ∈ D max y ^ ∈ S ( x ) f ( x , y ^ ) + Δ ( y , y ^ ) − f ( x , y ) ,
(13)
where f ( x , y ) is the scoring function of RNA secondary structure y ∈ S ( x ) for a given RNA sequence x ∈ Σ * , that is, Equation (4) for Nussinov-style decoding or Equation (8) for IPknot-style decoding. Here, Δ ( y , y ^ ) is a loss function of y ^ for y defined as
Δ ( y , y ^ ) = δ FN × ( # of false negative base pairs ) + δ FP × ( # of false positive base pairs ) = δ FN ∑ i < j I ( y i j = 1 ) I ( y ^ i j = 0 ) + δ FP ∑ i < j I ( y i j = 0 ) I ( y ^ i j = 1 ) ,
(14)
where δ FN and δ FP are tunable hyperparameters that can control the trade-off between sensitivity and specificity in learning the parameters. By default, we used δ FN = δ FP = 0.1 . In this case, the first term of Equation (13) can be calculated using the Nussinov-style decoding algorithm or the IPknot-style decoding algorithm modified by loss-augmented inference [42].
To minimize the objective function (13), stochastic subgradient descent (Algorithm 1) or one of its variants can be applied. We can calculate the gradients with regard to the network parameters λ for the objective function (13) using the gradients with regard to p i j by the chain rule of differentiation. This means that the prediction errors occurred through the decoding algorithm backpropagating to the neural network that calculates base-pairing probabilities through the connected base pairs.
Algorithm 1 The stochastic subgradient descent algorithm for structured support vector machines (SSVMs); η > 0 is the predefined learning rate.

1:
    initialize λ k for all λ k ∈ λ 
2:
    repeat
3:
      for all ( x , y ) ∈ D  do
4:
         y ^ ← arg max y ^ f ( x , y ^ ) + Δ ( y , y ^ ) 
5:
        for all  λ k ∈ λ  do
6:
           λ k ← λ k − η ( γ + 1 ) ∑ i < j ∂ p i j ∂ λ k ( y ^ i j − y i j ) 
7:
        end for
8:
      end for
9:
    until all the parameters converge

3. Results
3.1. Implementation
Our algorithm is implemented as the program Neuralfold, which is short for the neural network-based RNA folding algorithm. We employ Chainer [43] for the neural networks and the Python linear programming solver PuLP [44]. The source code for this implementation is available at https://github.com/keio-bioinformatics/neuralfold/, (accessed on 27 September 2022).
3.2. Datasets
We evaluated our algorithm with the Nussinov-style decoding algorithm for predicting pseudoknot-free RNA secondary structures using four datasets, TrainSetA, TestSetA, TrainSetB and TestSetB, which were established by [45].
TrainSetA and TestSetA are literature-based datasets [7,9,10,41,46] that were constructed to ensure sequence diversity. TrainSetA contains SSU and LSU domains, SRP RNAs, RNase P RNAs and tmRNAs comprising 3166 total sequences spanning 630,279 nt, with 333,466 forming base pairs (47.9%). The sequence lengths range from 10 to 734 nt, with an average length of 199 nt. TestSetA includes sequences from eight RNA families: 5S rRNA, group I and II introns, RNase P RNA, SRP RNA, tmRNA, tRNA, and telomerase RNA. TestSetA contains 697 sequences, with 51.7% of their bases forming base pairs. The sequence length ranges from 10 to 768 nt, with an average length of 195 nt. We excluded a number of sequences that contain pseudoknotted secondary structures in the original data sources from TestSetA. Thus, 593 sequences were selected as TestSetA.
TrainSetB and TestSetB, which contain 22 families with 3D structures [38], were assembled from Rfam [47]. TrainSetB and TestSetB include sequences from Rfam seed alignments with no more than 70% shared identity between sequences. TrainSetB comprises 22 RNA families, and its specific composition is 145.8S rRNAs, 18 U1 spliceosomal RNAs, 45 U4 spliceosomal RNAs, 233 riboswitches (from seven different families), 116 cis-regulatory elements (from nine different families), 3 ribozymes and a single bacteriophage pRNA. TrainSetB was constructed by selecting sequences dissimilar to those in TestSetB. TrainSetB contains 1094 sequences, including 112,398 nt in all, of which 52,065 bases (46.3%) formed base pairs. The sequence length is in the range of 27 to 237 nt with an average length of 103 nt. TrainSetB contains 4.3% noncanonical base pairs. TestSetB also consists of the same 22 RNA families as TrainSetB, TestSetB contains 430 sequences, including 52,097 nt in all, of which 22,728 bases (43.6%) form base pairs. The sequence length is in the range of 27 to 244 nt, with an average length of 121 nt. TestSetB contains 8.3% noncanonical base pairs.
We also evaluated our algorithm with the IPknot-style decoding algorithm for predicting pseudoknotted RNA secondary structures on two datasets. The first dataset is called the pk168 dataset [48], which was compiled from PseudoBase [20]. This dataset includes 16 categories of 168 pseudoknotted sequences with lengths <140 nt.
The second dataset is called RS-pk388, originally established by [36]. This dataset was obtained from the RNA STRAND database and contains 388 non-redundant sequences with lengths between 140 and 500 nt.
3.3. Prediction Performance
We evaluated the accuracy of RNA secondary structure predictions based on sensitivity ( S E N ) and positive predictive value ( P P V ) as follows:
S E N = T P T P + F N , P P V = T P T P + F P .
Here, T P , F P and F N represent the numbers of true positives (i.e., the correctly predicted base pairs), false positives (i.e., incorrectly predicted base pairs), and false negatives (i.e., base pairs in the correct structure that were not predicted), respectively. As a balanced measure of S E N and P P V , we utilized their F-value, which is defined as their harmonic mean:
F = 2 × S E N × P P V S E N + P P V .
We conducted computational experiments on the datasets described in the previous section using the Nussinov-style decoding algorithm with the McCaskill and neural network models as well as the BiRNN and FNN models. We employed CentroidFold as the Nussinov decoding algorithm with the McCaskill model. We performed experiments on TestSetB using the parameters trained from TrainSetB. As shown in Table 1, the neural network models achieved better accuracy compared with the traditional model. Hereafter, we adopt the FNN model with k-mer contexts as the default Neuralfold model since it yielded better prediction accuracy in this experiment.
Table 1. Accuracy of inferred base-pairing probabilities for TestSetB.
Table
The other computational experiments on the pseudoknotted dataset were conducted using the IPknot-style decoding algorithm with the McCaskill model with and without iterative refinement and with the Dirks–Pierce model as well as using Neuralfold with the FNN model. Table 2 shows that the feedforward neural network (FNN) model with 10-fold cross validation is comparable to IPknot with the Dirks–Pierce model for pseudoknots but superior to the McCaskill model both with and without iterative refinement.
Table 2. Accuracy of inferred base-pairing probabilities for the pk168 dataset.
Table
Table 3 shows the computation time for of the following sequences, which vary in length: PKB229 and PKB134 in the pk168 dataset; ASE_00193, CRW_00614 and CRW_00774 in the RNA STRAND database [49].
Table 3. Computation time for calculating base-pairing probabilities of sequences of various lengths.
Table
This shows that the computation time for predicting a pseudoknotted secondary structure using the FNN model is comparably fast to IPknot with the Dirks–Pierce model.
3.4. Effects of Context Length
We evaluated the prediction accuracy obtained with the FNN model on the TestSetB and pk168 datasets for several lengths of k-mers input to neural networks. The accuracy as measured by S E N , P P V , and their F-value for different k-mer lengths k = { 3 , 7 , 11 , 15 , 19 , 21 , 41 , 61 , 81 , 101 , 121 } is summarized in Figure 4. This analysis indicates that the accuracy is essentially maximized when the k-mer length is 81, and the difference in the accuracy for k ≥ 81 is negligible.
Genes 13 02155 g004 550
Figure 4. The accuracy of the FNN model with different lengths of k-mers on the TestSetB dataset (left) and the pk168 dataset (right). S E N , sensitivity; P P V , positive predictive value; F, the F-value based on S E N and P P V .
3.5. Comparison with Previous Methods for Prediction of Pseudoknot-Free Secondary Structures
We compared our algorithm with previous methods for predicting pseudoknot-free RNA secondary structures including CentroidFold [18,19], CONTRAfold [7,8], RNAfold in the Vienna RNA package [4] and ContextFold [29]. For the posterior decoding methods with the trade-off parameter γ in Equation (4), we used γ ∈ { 2 n ∣ n ∈ Z , − 5 ≤ n ≤ 10 } . We performed secondary structure prediction on TestSetA with parameters trained on TrainSetA as well as prediction on TestSetB with the parameters trained on TrainSetB. The PPV–SEN plots for each method shown in Figure 5 indicate that our algorithm accurately predicts pseudoknot-free secondary structures in the datasets including famlilies similar with the training datasets.
Genes 13 02155 g005 550
Figure 5. Positive predictive value–sensitivity (PPV–SEN) plots comparing our algorithm with competitive methods on TestSetA (Left) and TestSetB (Right).
On the other hand, to investigate the generalization ability of our method, another experiment in which our method was trained on TrainSetB and evaluated for accuracy on TestSetA showed that our method had very low accuracy ( S E N = 0.232 , P P V = 0.160 , and F = 0.189 ), which suggests that our method is severely overfitted.
3.6. Comparison with Alternative Methods for Predicting Pseudoknotted Secondary Structures
We also compared our algorithm with competing methods for predicting pseudoknotted secondary structures, including IPknot [36], HotKnots [32,33], and pknotsRG [29], as well as methods for predicting pseudoknot-free secondary structures, including CentroidFold [19] and RNAfold [4]. Neuralfold performed 10-fold cross validation on the pk168 and RS-pk388 datasets. Figure 6 shows PPV–SEN plots for each method, indicating that our algorithm works accurately on pseudoknotted datasets.
Genes 13 02155 g006 550
Figure 6. Positive predictive value–sensitivity (PPV–SEN) plots comparing our algorithm with competitive methods on the pk168 dataset (Left) and the RS-pk388 dataset (Right). For the pk168 dataset, we set γ ( 1 ) = 1 , γ ( 2 ) = 3 for Neuralfold; γ ( 1 ) = 2 , γ ( 2 ) = 4 for IPknot with the Dirks–Pierce (D&P) model; γ ( 1 ) = 2 , γ ( 2 ) = 16 for IPknot with/without refinement; γ = 2 for CentroidFold. For the RS-pk388 dataset, we set γ ( 1 ) = 1 , γ ( 2 ) = 3 for Neuralfold; γ ( 1 ) = 2 , γ ( 2 ) = 2 for IPknot without refinement; γ ( 1 ) = 1 , γ ( 2 ) = 1 for IPknot with refinement; γ = 2 for CentroidFold.
4. Discussion
We propose a novel algorithm for directly inferring base-pairing probabilities with neural networks that enables us to predict RNA secondary structures accurately. Sato et al. [36] previously proposed an iterative algorithm that refines the base-pairing probabilities calculated by the McCaskill algorithm so as to be appropriate for pseudoknotted secondary structure prediction. The direct inference of base-pairing probabilities with neural networks is an approach similar to the iterative refinement algorithm in the sense that both directly update base-pairing probabilities, and the IPknot-style decoding algorithm then uses the base-pairing probabilities. Although the iterative refinement algorithm can improve the prediction accuracy of IPknot to some extent, it should be noted that this is an ad hoc algorithm, as there is no theoretical guarantee of improvement. Meanwhile, the neural networks that infer base-pairing probabilities are trained on given reference secondary structures by the max-margin framework, meaning that we can theoretically expect that the neural network models improve the secondary structure prediction. Indeed, Table 2 shows that our algorithm achieved not only better accuracy than the iterative refinement algorithm, but is also comparable to that of the Dirks–Pierce model, which can calculate exact base-pairing probabilities for a limited class of pseudoknots.
Recently, several methods for predicting RNA secondary structure using deep learning were proposed [13,14,15]. Although most of them use deep learning to compute N × N matrices (N is the sequence length), which can be regarded as base-pairing probability matrices, they do not directly address the constraints that the RNA secondary structure must satisfy (e.g., Equations (5) and (6) for pseudoknot-free structures, and Equations (9)–(11) for pseudoknotted structures). On the other hand, MXfold2 [14] combines the Zuker-style dynamic programming [50] and deep learning to handle the constraints that pseudoknot-free RNA secondary structures must satisfy. UFold [15] predicts RNA secondary structure including pseudoknots using post-processing by linear programming, but does not directly address constraints on RNA secondary structure including pseudoknots when training deep learning models to predict base-pairing probabilities. By combining IPknot-style decoding with the max-margin training, the proposed Neuralfold can directly handle the constraints (9)–(11) that pseudoknotted RNA secondary structure must satisfy, not only when predicting secondary structures, but also when training deep learning models.
It has been pointed out that RNA secondary structure prediction based on machine learning and deep learning is prone to overfitting due to bias in the training data [14,45]. Several methods have been proposed to alleviate overfitting, such as using ensembles of multiple models [13], and integration with thermodynamic models [14]. UFold, on the other hand, employed artificially generated sequences and their predicted secondary structures for data augmentation, which were then used as additional training data to relax overfitting due to bias in the training data [15]. Our proposed method does not provide a strategy to counteract such overfitting and is therefore unsatisfactory at predicting sequences of families that are structurally distant from the training data, as shown in the results. However, by utilizing the ensembles of multiple models, as in SPOT-RNA, and the data augmentation strategy, as in UFold, it is expected to address to some extent the overfitting caused by bias in the training data.
The FNN model takes two k-mers around each base pair as input to infer its base-pairing probability, where k is the context length to model the length of loops and the contexts around the openings and closings of helices. As can be seen in Figure 7, different k-mer context lengths affect the prediction of pseudoknotted secondary structures. For example, consider the input bases when calculating the base-pairing probability of the blue-highlighted base pair (AU) using the FNN model. The FNN model with the context length k = 11 takes as input five bases in both the upstream and downstream directions from bases i and j. As seen in Figure 7 (bottom), the distances from bases A and U are 10 and 13 to Stem 2, respectively. This means that all the bases comprising Stem 2 are not completely located within the context length k = 11 around the base pair AU. On the other hand, for the FNN model with context length k = 41, all the bases of Stem 2 are completely located within the context around the base pair AU. This leads the FNN model to correctly predict the base pair AU, suggesting that a longer context length enables consideration of the dependency between stems in pseudoknotted substructures.
Genes 13 02155 g007 550
Figure 7. (Top) Comparison between the reference structure of ID PKB189 (top-left) and the predicted structures with context lengths k = 11 (top-middle) and k = 41 (top-right). (Bottom) Distance between two stems (Stem 1 and Stem 2) in the pseudoknotted structure.
5. Conclusions
We propose a novel algorithm for directly inferring base-pairing probabilities with neural networks that enables us to accurately predict RNA secondary structures with pseudoknots. By combining IPknot-style decoding with the max-margin framework, our algorithm trains the model in the end-to-end manner to compute base-pairing probabilities under the constraints that RNA secondary structures, including pseudoknots, must satisfy. HotKnots 2.0 [32], on the other hand, finds a pseudoknotted secondary structure by using an MFE-based heuristic decoding algorithm with energy parameters of the Dirks–Pierce model or the Cao–Chen model trained on pseudoknotted reference structures. One of the advantages of our algorithm over HotKnots 2.0 is that no assumption about the architecture of RNA secondary structures is required. In other words, our model can be trained on arbitrary classes of pseudoknots, while HotKnots cannot be trained on more complicated classes of pseudoknots than the one assumed by the model. Furthermore, our algorithm can compute base-pairing probabilities, which can be used in various applications of RNA informatics, such as family classification [51,52], RNA–RNA interaction prediction [53] and simultaneous aligning and folding [54]. Accurate base-pairing probabilities calculated by our algorithm can improve the quality of such applications.
Author Contributions
Conceptualization, M.A. and K.S.; methodology, M.A. and K.S.; software, M.A.; validation, M.A. and K.S.; writing—original draft preparation, M.A.; writing—review and editing, K.S.; supervision, Y.S. and K.S.; project administration, K.S.; funding acquisition, K.S. All authors have read and agreed to the published version of the manuscript.
Funding
This work was supported in part by a Grant-in-Aid for Scientific Research (KAKENHI) (16K00404, 19H04210 and 19K22897) from the Japan Society for the Promotion of Science (JSPS) to K.S.
Institutional Review Board Statement
Not applicable.
Informed Consent Statement
Not applicable.
Data Availability Statement
Not applicable.
Acknowledgments
The supercomputer system was provided by the National Institute of Genetics (NIG), Research Organization of Information and Systems (ROIS).
Conflicts of Interest
The authors declare no conflict of interest. The funders had no role in the design of the study; in the collection, analyses, or interpretation of data; in the writing of the manuscript; or in the decision to publish the results.
Abbreviations
The following abbreviations are used in this manuscript:
BiRNN	bi-directional recurrent neural network
FNN	feedforward neural network
MEA	maximum expected accuracy
MFE	minimum free energy
ncRNA	non-coding RNA
SSVM	structured support vector machine
References

    Hirose, T.; Mishima, Y.; Tomari, Y. Elements and machinery of non-coding RNAs: Toward their taxonomy. EMBO Rep. 2014, 15, 489–507. [Google Scholar] [CrossRef] [PubMed] [Green Version]
    Schroeder, S.J.; Turner, D.H. Optical melting measurements of nucleic acid thermodynamics. Meth. Enzymol. 2009, 468, 371–387. [Google Scholar]
    Turner, D.H.; Mathews, D.H. NNDB: The nearest neighbor parameter database for predicting stability of nucleic acid secondary structure. Nucleic Acids Res. 2010, 38, D280–D282. [Google Scholar] [CrossRef]
    Lorenz, R.; Bernhart, S.H.; Honer Zu Siederdissen, C.; Tafer, H.; Flamm, C.; Stadler, P.F.; Hofacker, I.L. ViennaRNA Package 2.0. Algorithms Mol. Biol. 2011, 6, 26. [Google Scholar] [CrossRef] [PubMed]
    Reuter, J.S.; Mathews, D.H. RNAstructure: Software for RNA secondary structure prediction and analysis. BMC BioInform. 2010, 11, 129. [Google Scholar] [CrossRef] [PubMed] [Green Version]
    Zuker, M. On finding all suboptimal foldings of an RNA molecule. Science 1989, 244, 48–52. [Google Scholar] [CrossRef] [PubMed] [Green Version]
    Do, C.B.; Woods, D.A.; Batzoglou, S. CONTRAfold: RNA secondary structure prediction without physics-based models. Bioinformatics 2006, 22, e90–e98. [Google Scholar] [CrossRef] [Green Version]
    Do, C.B.; Foo, C.S.; Ng, A. Efficient multiple hyperparameter learning for log-linear models. In Proceedings of the 20th International Conference on Neural Information Processing Systems, Vancouver, BC, Canada, 3–6 December 2007; Advances in Neural Information Processing Systems 20. Curran Associates Inc.: Red Hook, NY, USA, 2007. [Google Scholar]
    Andronescu, M.; Condon, A.; Hoos, H.H.; Mathews, D.H.; Murphy, K.P. Efficient parameter estimation for RNA secondary structure prediction. Bioinformatics 2007, 23, 19–28. [Google Scholar] [CrossRef] [Green Version]
    Andronescu, M.; Condon, A.; Hoos, H.H.; Mathews, D.H.; Murphy, K.P. Computational approaches for RNA energy parameter estimation. RNA 2010, 16, 2304–2318. [Google Scholar] [CrossRef] [Green Version]
    Zakov, S.; Goldberg, Y.; Elhadad, M.; Ziv-Ukelson, M. Rich parameterization improves RNA structure prediction. J. Comput. Biol. 2011, 18, 1525–1542. [Google Scholar] [CrossRef] [PubMed] [Green Version]
    Akiyama, M.; Sato, K.; Sakakibara, Y. A max-margin training of RNA secondary structure prediction integrated with the thermodynamic model. J. Bioinform. Comput. Biol. 2018, 16, 1840025. [Google Scholar] [CrossRef] [PubMed]
    Singh, J.; Hanson, J.; Paliwal, K.; Zhou, Y. RNA secondary structure prediction using an ensemble of two-dimensional deep neural networks and transfer learning. Nat. Commun. 2019, 10, 5407. [Google Scholar] [CrossRef] [Green Version]
    Sato, K.; Akiyama, M.; Sakakibara, Y. RNA secondary structure prediction using deep learning with thermodynamic integration. Nat. Commun. 2021, 12, 941. [Google Scholar] [CrossRef]
    Fu, L.; Cao, Y.; Wu, J.; Peng, Q.; Nie, Q.; Xie, X. UFold: Fast and accurate RNA secondary structure prediction with deep learning. Nucleic Acids Res. 2022, 50, e14. [Google Scholar] [CrossRef] [PubMed]
    Carvalho, L.E.; Lawrence, C.E. Centroid estimation in discrete high-dimensional spaces with applications in biology. Proc. Natl. Acad. Sci. USA 2008, 105, 3209–3214. [Google Scholar] [CrossRef] [PubMed] [Green Version]
    McCaskill, J.S. The equilibrium partition function and base pair binding probabilities for RNA secondary structure. Biopolymers 1990, 29, 1105–1119. [Google Scholar] [CrossRef]
    Hamada, M.; Kiryu, H.; Sato, K.; Mituyama, T.; Asai, K. Prediction of RNA secondary structure using generalized centroid estimators. Bioinformatics 2009, 25, 465–473. [Google Scholar] [CrossRef] [Green Version]
    Sato, K.; Hamada, M.; Asai, K.; Mituyama, T. CENTROIDFOLD: A web server for RNA secondary structure prediction. Nucleic Acids Res. 2009, 37, W277–W280. [Google Scholar] [CrossRef] [Green Version]
    van Batenburg, F.H.; Gultyaev, A.P.; Pleij, C.W. PseudoBase: Structural information on RNA pseudoknots. Nucleic Acids Res. 2001, 29, 194–195. [Google Scholar] [CrossRef] [Green Version]
    Staple, D.W.; Butcher, S.E. Pseudoknots: RNA structures with diverse functions. PLoS Biol. 2005, 3, e213. [Google Scholar] [CrossRef] [Green Version]
    Brierley, I.; Pennell, S.; Gilbert, R.J. Viral RNA pseudoknots: Versatile motifs in gene expression and replication. Nat. Rev. Microbiol. 2007, 5, 598–610. [Google Scholar] [CrossRef] [PubMed]
    Fechter, P.; Rudinger-Thirion, J.; Florentz, C.; Giege, R. Novel features in the tRNA-like world of plant viral RNAs. Cell. Mol. Life Sci. 2001, 58, 1547–1561. [Google Scholar] [CrossRef] [PubMed]
    Akutsu, T. Dynamic programming algorithms for RNA secondary structure prediction with pseudoknots. Discret. Appl. Math. 2000, 104, 45–62. [Google Scholar] [CrossRef] [Green Version]
    Lyngsø, R.B.; Pedersen, C.N. RNA pseudoknot prediction in energy-based models. J. Comput. Biol. 2000, 7, 409–427. [Google Scholar] [CrossRef]
    Rivas, E.; Eddy, S.R. A dynamic programming algorithm for RNA structure prediction including pseudoknots. J. Mol. Biol. 1999, 285, 2053–2068. [Google Scholar] [CrossRef]
    Dirks, R.M.; Pierce, N.A. A partition function algorithm for nucleic acid secondary structure including pseudoknots. J. Comput. Chem. 2003, 24, 1664–1677. [Google Scholar] [CrossRef] [Green Version]
    Dirks, R.M.; Pierce, N.A. An algorithm for computing nucleic acid base-pairing probabilities including pseudoknots. J. Comput. Chem. 2004, 25, 1295–1304. [Google Scholar] [CrossRef]
    Reeder, J.; Giegerich, R. Design, implementation and evaluation of a practical pseudoknot folding algorithm based on thermodynamics. BMC Bioinform. 2004, 5, 104. [Google Scholar] [CrossRef] [Green Version]
    Jabbari, H.; Wark, I.; Montemagno, C.; Will, S. Knotty: Efficient and Accurate Prediction of Complex RNA Pseudoknot Structures. Bioinformatics 2018, 34, 3849–3856. [Google Scholar] [CrossRef]
    Ruan, J.; Stormo, G.D.; Zhang, W. An iterated loop matching approach to the prediction of RNA secondary structures with pseudoknots. Bioinformatics 2004, 20, 58–66. [Google Scholar] [CrossRef] [Green Version]
    Andronescu, M.S.; Pop, C.; Condon, A.E. Improved free energy parameters for RNA pseudoknotted secondary structure prediction. RNA 2010, 16, 26–42. [Google Scholar] [CrossRef] [PubMed]
    Ren, J.; Rastegari, B.; Condon, A.; Hoos, H.H. HotKnots: Heuristic prediction of RNA secondary structures including pseudoknots. RNA 2005, 11, 1494–1504. [Google Scholar] [CrossRef] [PubMed] [Green Version]
    Chen, X.; He, S.M.; Bu, D.; Zhang, F.; Wang, Z.; Chen, R.; Gao, W. FlexStem: Improving predictions of RNA secondary structures with pseudoknots by reducing the search space. Bioinformatics 2008, 24, 1994–2001. [Google Scholar] [CrossRef] [Green Version]
    Bellaousov, S.; Mathews, D.H. ProbKnot: Fast prediction of RNA secondary structure including pseudoknots. RNA 2010, 16, 1870–1880. [Google Scholar] [CrossRef] [Green Version]
    Sato, K.; Kato, Y.; Hamada, M.; Akutsu, T.; Asai, K. IPknot: Fast and accurate prediction of RNA secondary structures with pseudoknots using integer programming. Bioinformatics 2011, 27, 85–93. [Google Scholar] [CrossRef] [Green Version]
    Sato, K.; Kato, Y. Prediction of RNA secondary structure including pseudoknots for long sequences. Brief. Bioinform. 2022, 23, bbab395. [Google Scholar] [CrossRef] [PubMed]
    Rivas, E. The four ingredients of single-sequence RNA secondary structure prediction. A unifying perspective. RNA Biol. 2013, 10, 1185–1196. [Google Scholar] [CrossRef] [Green Version]
    Cao, S.; Chen, S.J. Predicting RNA pseudoknot folding thermodynamics. Nucleic Acids Res. 2006, 34, 2634–2652. [Google Scholar] [CrossRef] [PubMed]
    Nussinov, R.; Pieczenick, G.; Griggs, J.; Kleitman, D. Algorithms for loop matching. SIAM J. Appl. Math. 1978, 35, 68–82. [Google Scholar] [CrossRef]
    Dowell, R.D.; Eddy, S.R. Evaluation of several lightweight stochastic context-free grammars for RNA secondary structure prediction. BMC Bioinform. 2004, 5, 71. [Google Scholar] [CrossRef] [Green Version]
    Tsochantaridis, I.; Joachims, T.; Hofmann, T.; Altun, Y. Large Margin Methods for Structured and Interdependent Output Variables. J. Mach. Learn. Res. 2005, 6, 1453–1484. [Google Scholar]
    Tokui, S.; Oono, K.; Hido, S.; Clayton, J. Chainer: A Next-Generation Open Source Framework for Deep Learning. In Proceedings of the Workshop on Machine Learning Systems (LearningSys) in The Twenty-Ninth Annual Conference on Neural Information Processing Systems (NIPS), Montréal, QC, Canada, 11–12 December 2015. [Google Scholar]
    Mitchell, S.; Consulting, S.M.; O’sullivan, M.; Dunning, I. PuLP: A Linear Programming Toolkit for Python. 2011. Available online: https://optimization-online.org/2011/09/3178/ (accessed on 27 September 2022).
    Rivas, E.; Lang, R.; Eddy, S.R. A range of complex probabilistic models for RNA secondary structure prediction that includes the nearest-neighbor model and more. RNA 2012, 18, 193–212. [Google Scholar] [CrossRef] [PubMed]
    Lu, Z.J.; Gloor, J.W.; Mathews, D.H. Improved RNA secondary structure prediction by maximizing expected pair accuracy. RNA 2009, 15, 1805–1813. [Google Scholar] [CrossRef] [PubMed] [Green Version]
    Gardner, P.P.; Daub, J.; Tate, J.; Moore, B.L.; Osuch, I.H.; Griffiths-Jones, S.; Finn, R.D.; Nawrocki, E.P.; Kolbe, D.L.; Eddy, S.R.; et al. Rfam: Wikipedia, clans and the “decimal” release. Nucleic Acids Res. 2011, 39, D141–D145. [Google Scholar] [CrossRef] [Green Version]
    Huang, X.; Ali, H. High sensitivity RNA pseudoknot prediction. Nucleic Acids Res. 2007, 35, 656–663. [Google Scholar] [CrossRef] [Green Version]
    Andronescu, M.; Bereg, V.; Hoos, H.H.; Condon, A. RNA STRAND: The RNA secondary structure and statistical analysis database. BMC Bioinform. 2008, 9, 340. [Google Scholar] [CrossRef] [Green Version]
    Zuker, M.; Stiegler, P. Optimal computer folding of large RNA sequences using thermodynamics and auxiliary information. Nucleic Acids Res. 1981, 9, 133–148. [Google Scholar] [CrossRef]
    Sato, K.; Mituyama, T.; Asai, K.; Sakakibara, Y. Directed acyclic graph kernels for structural RNA analysis. BMC Bioinform. 2008, 9, 318. [Google Scholar] [CrossRef] [Green Version]
    Morita, K.; Saito, Y.; Sato, K.; Oka, K.; Hotta, K.; Sakakibara, Y. Genome-wide searching with base-pairing kernel functions for noncoding RNAs: Computational and expression analysis of snoRNA families in Caenorhabditis elegans. Nucleic Acids Res. 2009, 37, 999–1009. [Google Scholar] [CrossRef] [Green Version]
    Kato, Y.; Sato, K.; Hamada, M.; Watanabe, Y.; Asai, K.; Akutsu, T. RactIP: Fast and accurate prediction of RNA-RNA interaction using integer programming. Bioinformatics 2010, 26, i460–i466. [Google Scholar] [CrossRef] [Green Version]
    Sato, K.; Kato, Y.; Akutsu, T.; Asai, K.; Sakakibara, Y. DAFS: Simultaneous aligning and folding of RNA sequences via dual decomposition. Bioinformatics 2012, 28, 3218–3224. [Google Scholar] [CrossRef] [PubMed]

	
Publisher’s Note: MDPI stays neutral with regard to jurisdictional claims in published maps and institutional affiliations.

© 2022 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/licenses/by/4.0/).
Share and Cite
MDPI and ACS Style

Akiyama, M.; Sakakibara, Y.; Sato, K. Direct Inference of Base-Pairing Probabilities with Neural Networks Improves Prediction of RNA Secondary Structures with Pseudoknots. Genes 2022, 13, 2155. https://doi.org/10.3390/genes13112155
AMA Style

Akiyama M, Sakakibara Y, Sato K. Direct Inference of Base-Pairing Probabilities with Neural Networks Improves Prediction of RNA Secondary Structures with Pseudoknots. Genes. 2022; 13(11):2155. https://doi.org/10.3390/genes13112155
Chicago/Turabian Style

Akiyama, Manato, Yasubumi Sakakibara, and Kengo Sato. 2022. "Direct Inference of Base-Pairing Probabilities with Neural Networks Improves Prediction of RNA Secondary Structures with Pseudoknots" Genes 13, no. 11: 2155. https://doi.org/10.3390/genes13112155
APA Style

Akiyama, M., Sakakibara, Y., & Sato, K. (2022). Direct Inference of Base-Pairing Probabilities with Neural Networks Improves Prediction of RNA Secondary Structures with Pseudoknots. Genes, 13(11), 2155. https://doi.org/10.3390/genes13112155
Note that from the first issue of 2016, this journal uses article numbers instead of page numbers. See further details here.
Article Metrics
Citations
Crossref
 
4
Web of Science
 
3
PMC
 
1
Scopus
 
4
PubMed
 
1
Google Scholar
 
[click to view]
Article Access Statistics
Article access statisticsArticle Views17. Dec18. Dec19. Dec20. Dec21. Dec22. Dec23. Dec24. Dec25. Dec26. Dec27. Dec28. Dec29. Dec30. Dec31. Dec1. Jan2. Jan3. Jan4. Jan5. Jan6. Jan7. Jan8. Jan9. Jan10. Jan11. Jan12. Jan13. Jan14. Jan15. Jan16. Jan17. Jan18. Jan19. Jan20. Jan21. Jan22. Jan23. Jan24. Jan25. Jan26. Jan27. Jan28. Jan29. Jan30. Jan31. Jan1. Feb2. Feb3. Feb4. Feb5. Feb6. Feb7. Feb8. Feb9. Feb10. Feb11. Feb12. Feb13. Feb14. Feb15. Feb16. Feb17. Feb18. Feb19. Feb20. Feb21. Feb22. Feb23. Feb24. Feb25. Feb26. Feb27. Feb28. Feb1. Mar2. Mar3. Mar4. Mar5. Mar6. Mar7. Mar8. Mar9. Mar10. Mar11. Mar12. Mar13. Mar14. Mar15. Mar16. Mar050010001500200025003000
For more information on the journal statistics, click here.
Multiple requests from the same IP address are counted as one view.
Genes, EISSN 2073-4425, Published by MDPI
RSS Content Alert
Further Information
Article Processing Charges
Pay an Invoice
Open Access Policy
Contact MDPI
Jobs at MDPI
Guidelines
For Authors
For Reviewers
For Editors
For Librarians
For Publishers
For Societies
For Conference Organizers
MDPI Initiatives
Sciforum
MDPI Books
Preprints.org
Scilit
SciProfiles
Encyclopedia
JAMS
Proceedings Series
Follow MDPI
LinkedIn
Facebook
Twitter
MDPI

Subscribe to receive issue release notifications and newsletters from MDPI journals
© 1996-2025 MDPI (Basel, Switzerland) unless otherwise stated
Disclaimer Terms and Conditions Privacy Policy
=====

Skip to main content
Skip to article
Elsevier logo

    Journals & Books

Help

    Search

My account
Ripon College

Ripon College does not subscribe to this content on ScienceDirect.
Article preview

    Abstract
    Introduction
    Section snippets
    References (71)

Elsevier
Computers in Biology and Medicine
Volume 182, November 2024, 109207
Computers in Biology and Medicine
Wfold: A new method for predicting RNA secondary structure with deep learning
Author links open overlay panelYongna Yuan
, Enjie Yang, Ruisheng Zhang
Show more
Add to Mendeley
Share
Cite
https://doi.org/10.1016/j.compbiomed.2024.109207
Get rights and content
Highlights

    •
    We develop a deep learning model Wfold for RNA secondary structure prediction.
    •
    The ‘image’ representation makes all possible long-range interactions explicitly.
    •
    The model considers all possible base pairing patterns.
    •
    Wfold combines Unet and transformer, which can effectively mine long-range and local information from input data.

Abstract
Precise estimations of RNA secondary structures have the potential to reveal the various roles that non-coding RNAs play in regulating cellular activity. However, the mainstay of traditional RNA secondary structure prediction methods relies on thermos-dynamic models via free energy minimization, a laborious process that requires a lot of prior knowledge. Here, RNA secondary structure prediction using Wfold, an end-to-end deep learning-based approach, is suggested. Wfold is trained directly on annotated data and base-pairing criteria. It makes use of an image-like representation of RNA sequences, which an enhanced U-net incorporated with a transformer encoder can process effectively. Wfold eventually increases the accuracy of RNA secondary structure prediction by combining the benefits of self-attention mechanism's mining of long-range information with U-net's ability to gather local information. We compare Wfold's performance using RNA datasets that are within and across families. When trained and evaluated on different RNA families, it achieves a similar performance as the traditional methods, but dramatically outperforms the state-of-the-art methods on within-family datasets. Moreover, Wfold can also reliably forecast pseudoknots. The findings imply that Wfold may be useful for improving sequence alignment, functional annotations, and RNA structure modeling.
Graphical abstract
9.27 cm high and 8.58 cm wide.
Image 1

    Download: Download high-res image (259KB)
    Download: Download full-size image

Introduction
The biology of RNA is diverse and complex. The foundations of RNA ranging from catalysis, cell-signalling, to transcriptional regulation [1] are deeply related to their structures rather than their primary sequences. The determination of RNA 3D structures can be performed by X-ray crystallography [2], NMR [3], cryo-electron microscopy [4], and other techniques. To date, there are more than 33 million RNA sequences [5] while less than 0.01 % have experimentally determined structures [6,7]. However, the major difficulties in computing RNA 3D structures have not yet been completely overcome such as high computational complexity, economic cost, and performance limits. It is tough to apply experimental methods on a large scale. RNA secondary structures, which are the backbones of 3D structures and are easier to model, are often computationally predicted. Therefore, accurate and cost-effective computational methods are highly desirable to be developed for direct prediction of RNA secondary structures from sequences, defined as a list of nucleotide bases paired by hydrogen bonding consisting of Watson-Crick base pairs (A-U, G-C) and wobble base pairs (G-U).
Many RNA secondary structure prediction methods have been developed since the 1970s. These methods can be roughly divided into two categories: (i) single sequence prediction methods and (ii) comparative methods. The primary idea behind the first category is to use the minimal free energy (MFE) to look for thermodynamically stable states. Dynamic programming (DP) can effectively address the energy reduction problem if the secondary structure solely includes nested base pairing, such as those implemented in Vienna RNAfold [8], MFold [9], RNAstructure [10] and CONTRAfold [11]. Methods, such as Rfold [12], Vienna RNAplfold [13], LocalFold [14], LinearFold [15] and StemP [16], improved the speed of DP and the time efficiency is reduced from
to
, and then to O(n) [15]. Efficient DP algorithms that sample suboptimal secondary structures from the Boltzmann ensembles of structures have also been proposed, for example, CentroidFold [17]. Some DP-based techniques, however, fail when base pairs contain pseudoknots, which are non-nested patterns that help fold into three-dimensional structures. Pseudoknots are two stem–loop structures in which half of one stem intercalates between the two halves of another stem. Predicting secondary structures with pseudoknots is hard and has shown to be NP-complete under the energy minimization framework [18]. Methods in the second category utilize covariance methods by aligning related RNA sequences and identifying correlated compensatory mutations [[19], [20], [21]]. By analyzing several sequences, these algorithms attempt to anticipate conserved structures and identify base pair placements by identifying points of base covariance within the sequences. Although the list of proposed methods in each of the two categories is long and diverse [22], the performance of these methods has not been significantly improved over time, reaching a performance ceiling of about 80 % [23]. It is probably because they fail to account for base pairing resulting from tertiary interactions [24], unstacked base pairs, pseudoknot, non-canonical base pairing, or other unknown factors [25].
Comprehending the RNA folding mechanism in its whole is really challenging. Machine learning (ML) techniques, on the other hand, are data-driven and do not require any prior knowledge of such mechanisms. Massive training data sets allow machine learning algorithms to discover the fundamental folding patterns. To increase computational performance, machine learning techniques have been used to several facets of RNA secondary structure prediction in the past few decades. Early ML-based methods usually trained basic machine learning models to predict RNA secondary structures [[26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39]], such as support vector machine (SVM) [40], multilayer perceptron (MLP) [[41], [42], [43]]. However, because of their poor accuracy, these methods did not attract much attention. The main causes of that were the tiny training dataset sizes and the shortcomings of the simpler ML models.
Currently, because of the recent explosion of RNA sequence data and the rapid rise of deep learning (DL) techniques, the DL-based methods become the current mainstream methods in terms of accuracy and applicability [[44], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62]]. Convolutional neural networks (CNNs), recurrent neural networks (RNNs), graph neural networks (GNNs), Transformer, and their variations or combinations are the primary deep learning (DL) approaches utilized for RNA structure prediction. For example, SPOT-RNA [50], the first end-to-end DL model which processed and handled tasks from input to output in a seamless, automated manner, without requiring human intervention in intermediary steps, treated the RNA secondary structure as a contact table and employed an ensemble of ultra-deep hybrid networks of ResNets and 2D-BLSTMs for the prediction. SPOT-RNA showed superior performance with two RNA benchmark datasets. Recently, the SPOT-RNA2 model [53] employed evolution-derived sequence profiles and mutational coupling as inputs and outperformed SPOT-RNA for all types of base pairs using the same transfer learning approach. E2Efold [51] was another DL model for RNA secondary structure prediction. It integrated 2 coupled parts, i.e., a transformer-based deep model that encoded sequence information, and a multilayer network based on an unrolled algorithm that gradually enforced the constraints and restricted the output space. UFold [54], a DL-based method proposed for RNA secondary structure prediction, trained directly on annotated data and base-pairing rules. It applied a novel image-like representation of RNA sequences, which can be efficiently processed by Fully Convolutional Networks (FCNs). RNAformer [60] improved prediction accuracy by a two-dimensional latent space, axial attention, and recycling in the latent space. Nevertheless, current deep learning methodologies encounter multiple obstacles: Initially, the LSTM and transformer encoder components encompass an extensive array of model parameters, culminating in significant computational expenses and diminished efficiency. Secondly, amalgamating these with thermodynamic optimization techniques compels the models to adopt the presumptions intrinsic to conventional methods, potentially impairing their effectiveness. Thirdly, given that the efficacy of deep learning models is profoundly reliant on the training data distribution, it becomes crucial to devise strategies to enhance their performance across novel categories of RNA structures previously unencountered.
Inspired by E2Efold and UFold, the former uses a coupled transformer model and a CNN model and the latter applies U-net model based on pure CNN, in this study, we propose an end-to-end deep learning model for predicting RNA secondary structures embedding transformer into U-net. The U-net model is good at capturing local information, while the transformer model is skilled in retaining long-term information. Ultimately, our model obtains an outstanding performance. As the model architecture presents a “W" shape, it is named Wfold. Similar to Ufold, the input sequence is handled as a 17-channel, 2D “image,” which enables the model to explicitly take into account all potential base pairings and long-range interactions rather than relying only on the nucleotide sequence. We test Wfold's performance against benchmark models on RNAStralign, ArchiveII, bpRNA-1m, bpRNA-new, and within-family and cross-family datasets. Wfold performs best in terms of prediction accuracy.
In summary, the advantages of Wfold are as follows:

    i)
    Wfold explicitly models every potential long-range interaction using an “image” representation. Local base pairing appears in the image-like representation between distant sequence segments.
    ii)
    Wfold does not distinguish between canonical and non-canonical base pairs; instead, it takes all 16 conceivable base pairing patterns into account.
    iii)
    Wfold combines CNN (U-net) and transformer (multi-head self-attention) which can handle variable sequence length, eliminating the need of padding the input sequence to a fixed length.

Section snippets
Datasets
To evaluate Wfold, we perform computational experiments on four benchmark datasets: (1) RNAStralign [63], which contains 33,277 unique sequences from 8 RNA families, (2) ArchiveII [64], which contains 3966 sequences from 10 RNA families. These two datasets are the most widely used datasets for benchmarking RNA structure prediction performance recently. (3) bpRNA-1m [65], which is based on Rfam 12.2 [66] with 2588 families. After removing redundant sequences based on sequence identity, the
Overview of wfold
In this study, Wfold, as an end-to-end DL solution, integrates two parts mentioned in the Materials and methods section. The first part of the architecture is a model called Deep Score Network to represent RNA sequence information which is useful for RNA secondary structure prediction. The second part is a multilayer network called Post-Processing Network which gradually enforces the constraints and restricts the output space. Deep Score Network, with the combination of U-net and transformer,
Conclusion
This paper creates a deep neural network-based approach for RNA secondary structure prediction based just on a single RNA sequence. We present a unique DL model for RNA secondary structure prediction called Wfold, which includes strict limitations in its architecture design. Through independent testing and comparison with SOTA RNA secondary-structure prediction algorithms, Wfold obtains satisfactory results in terms of precision, recall, and F1 score. Extensive tests are carried out to
Funding
This study is supported by the National Natural Science Foundation of China (22373043), the Science and Technology Project of Gansu (21YF5GA102, 21YF5GA006, 21ZD8RA008, 22ZD6GA029), the Science and Technology Project of Lanzhou (2023332), the Science and Technology Plan Project of Chengguan District of Lanzhou (2023RCCX0005), and the Super Computing Center of Lanzhou University.
Data and availability
The main codes of methods mentioned in the study are available at https://github.com/EnjieYang/Wfold
.
CRediT authorship contribution statement
Yongna Yuan: Supervision, Conceptualization. Enjie Yang: Writing – original draft, Data curation. Ruisheng Zhang: Writing – original draft, Supervision, Formal analysis.
Declaration of competing interest
None Declared.
References (71)

    X.D. Wang et al.
    Dynamic programming for NP-hard problems
    Process Eng.
    (2011)
    J. Nowakowski et al.
    RNA structure and stability
    Semin. Virol.
    (1997)
    E. Westhof et al.
    RNA folding: beyond Watson–Crick pairs
    Structure
    (2000)
    X. Tang
    Simulating RNA folding kinetics on approximated energy landscapes
    J. Mol. Biol.
    (2008)
    H. Yonemoto et al.
    A semi-supervised learning approach for RNA secondary structure prediction
    Comput. Biol. Chem.
    (2015)
    L. Quan
    Developing parallel ant colonies filtered by deep learned constrain for predicting RNA secondary structure with pseudo-knots
    Neurocomputing
    (2020)
    H. Zhang
    A new method of RNA secondary structure prediction based on convolutional neural network and dynamic programming
    Front. Genet.
    (2019)
    C. Shen
    BAT-Net: an enhanced RNA Secondary Structure prediction via bidirectional GRU-based network with attention mechanism
    Comput. Biol. Chem.
    (2022)
    S. Geisler et al.
    RNA in unexpected places: long non-coding RNA functions in diverse cellular contexts
    Nat. Rev. Mol. Cell Biol.
    (2013)
    E. Westhof
    Twenty years of RNA crystallography
    RNA
    (2015)

View more references
Cited by (0)
View full text
© 2024 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
Recommended articles

    An efficient approach for EMG controlled pattern recognition system based on MUAP identification and segregation
    Computers in Biology and Medicine, Volume 182, 2024, Article 109169
    Anil Sharma, …, Anil Kumar
    Drug-induced torsadogenicity prediction model: An explainable machine learning-driven quantitative structure-toxicity relationship approach
    Computers in Biology and Medicine, Volume 182, 2024, Article 109209
    Feyza Kelleci Çelik, …, Gül Karaduman
    Dynamics of sit-to-stand and stand-to-sit motions based on the trajectory control of the centre of mass of the body: A bond graph approach
    Computers in Biology and Medicine, Volume 182, 2024, Article 109117
    Vivek Soni, Anand Vaz

Show 3 more articles
Article Metrics
Captures

    Mendeley Readers3

PlumX Metrics Logo
View details
Elsevier logo with wordmark

    About ScienceDirect

Remote access
Advertise
Contact and support
Terms and conditions
Privacy policy

Cookies are used by this site. Cookie settings

All content on this site: Copyright © 2025 or its licensors and contributors. All rights are reserved, including those for text and data mining, AI training, and similar technologies. For all open access content, the relevant licensing terms apply.
RELX group home page
====
Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook
Matching Perspective
Cheng Tan 1 2 * Zhangyang Gao 2 1 * Hanqun Cao 3 Xingran Chen 4 Ge Wang 2 Lirong Wu 2 Jun Xia 2
Jiangbin Zheng 2 Stan Z. Li 2
Abstract
The secondary structure of ribonucleic acid
(RNA) is more stable and accessible in the cell
than its tertiary structure, making it essential for
functional prediction. Although deep learning
has shown promising results in this field, current
methods suffer from poor generalization and high
complexity. In this work, we reformulate the RNA
secondary structure prediction as a K-Rook prob-
lem, thereby simplifying the prediction process
into probabilistic matching within a finite solution
space. Building on this innovative perspective, we
introduce RFold, a simple yet effective method
that learns to predict the most matching K-Rook
solution from the given sequence. RFold em-
ploys a bi-dimensional optimization strategy that
decomposes the probabilistic matching problem
into row-wise and column-wise components to
reduce the matching complexity, simplifying the
solving process while guaranteeing the validity of
the output. Extensive experiments demonstrate
that RFold achieves competitive performance and
about eight times faster inference efficiency than
the state-of-the-art approaches. The code is avail-
able at github.com/A4Bio/RFold.
1. Introduction
The functions of RNA molecules are determined by their
structure (Sloma & Mathews, 2016). The secondary struc-
ture, which contains the nucleotide base pairing information,
as shown in Figure 1, is crucial for the correct functions of
RNA molecules (Fallmann et al., 2017). Although experi-
mental assays such as X-ray crystallography (Cheong et al.,
2004), nuclear magnetic resonance (F ¨urtig et al., 2003), and
*Equal contribution 1Zhejiang University 2Westlake University
3The Chinese University of Hong Kong 4University of Michigan.
Correspondence to: Stan Z. Li <Stan.ZQ.Li@westlake.edu.cn>.
Proceedings of the 41 st International Conference on Machine
Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by
the author(s).A A C C U G G U C A G G C C C G G A A G G G A G C A G C C A
A A C C U G G U C A G G C C C G G A A G G G A G C A G C C A
graph representation matrix representation (contact map)
Figure 1. The graph and matrix representation of an RNA sec-
ondary structure example.
cryogenic electron microscopy (Fica & Nagai, 2017) can be
implemented to determine RNA secondary structure, they
suffer from low throughput and expensive cost.
Computational RNA secondary structure prediction meth-
ods have been favored for their high efficiency in recent
years (Iorns et al., 2007). Currently, mainstream meth-
ods can be broadly classified into two categories (Rivas,
2013; Szikszai et al., 2022): (i) comparative sequence anal-
ysis and (ii) single sequence folding algorithm. Compara-
tive sequence analysis determines the secondary structure
conserved among homologous sequences but the limited
known RNA families hinder its development (Gutell et al.,
2002; Griffiths-Jones et al., 2003; Gardner et al., 2009;
Nawrocki et al., 2015). Researchers thus resort to single
RNA sequence folding algorithms that do not need mul-
tiple sequence alignment information. A classical cate-
gory of computational RNA folding algorithms is to use
dynamic programming (DP) that assumes the secondary
structure is a result of energy minimization (Bellaousov
et al., 2013; Nicholas & Zuker, 2008; Lorenz et al., 2011;
Zuker, 2003; Mathews & Turner, 2006; Do et al., 2006).
However, energy-based approaches usually require the base
pairs have a nested structure while ignoring some valid yet
biologically essential structures such as pseudoknots, i.e.,
non-nested base pairs (Chen et al., 2019; Seetin & Math-
ews, 2012; Xu & Chen, 2015), as shown in Figure 2. Since
predicting secondary structures with pseudoknots under the
energy minimization framework has shown to be hard and
1
arXiv:2212.14041v5 [q-bio.BM] 19 Jun 2024
Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching Perspective
NP-complete (Wang & Tian, 2011; Fu et al., 2022), deep
learning techniques are introduced as an alternative.A AC U G U A AC U G U
nested structure non-nested structure
Figure 2. Examples of nested and non-nested secondary structures.
Attempts to overcome the limitations of energy-based meth-
ods have motivated deep learning methods that predict
RNA secondary structures in the absence of DP. SPOT-
RNA (Singh et al., 2019) is a seminal work that ensembles
ResNet (He et al., 2016) and LSTM (Hochreiter & Schmid-
huber, 1997) and applies transfer learning to identify molec-
ular recognition features. SPOT-RNA does not constrain the
output space into valid RNA secondary structures, which de-
grades its generalization ability on new datasets (Jung et al.).
E2Efold (Chen et al., 2019) employs an unrolled algorithm
for constrained programming that post-processes the net-
work output to satisfy the constraints. E2Efold introduces
a convex relaxation to make the constrained optimization
tractable, leading to possible structural constraint violations
and poor generalization ability (Sato et al., 2021; Fu et al.,
2022; Franke et al., 2023; 2022). RTfold (Jung et al.) uti-
lizes the Fenchel-Young loss (Berthet et al., 2020) to en-
able differentiable discrete optimization with perturbations,
but the approximation cannot guarantee the satisfaction of
constraints. Developing an appropriate optimization that
enforces the output to be valid becomes a crucial concern.
Since deep learning-based approaches cannot directly out-
put valid RNA secondary structures, existing approaches
usually formulate the problem into a constrained optimiza-
tion problem and optimize the output of the model to fulfill
specific constraints as closely as possible. However, these
methods typically necessitate iterative optimization, leading
to reduced efficiency. Moreover, the extensive optimization
space involved does not ensure the complete satisfaction
of these constraints. In this study, we introduce a novel
perspective for predicting RNA secondary structures by re-
framing the challenge as a K-Rook problem. Recognizing
the alignment between the solution spaces of the K-Rook
problem and RNA secondary structure prediction, our objec-
tive is to identify the most compatible K-Rook solution for
each RNA sequence. This is achieved by training the deep
learning model on prior data to learn matching patterns.
Considering the high complexity of the solution space in the
symmetric K-Rook problem, we introduced RFold, an inno-
vative approach. This method utilizes a bi-dimensional opti-
mization strategy, effectively decomposing the problem into
separate row-wise and column-wise components. This de-
composition significantly reduces the matching complexity,
thereby simplifying the solving process while guaranteeing
the validity of the output. We conduct extensive experiments
to compare RFold with state-of-the-art methods on several
benchmark datasets and show the superior performance of
our proposed method. Moreover, RFold has faster inference
efficiency than those methods due to its simplicity.
2. Related work
Comparative Sequence Analysis Comparative sequence
analysis determines base pairs conserved among homolo-
gous sequences (Gardner & Giegerich, 2004; Knudsen &
Hein, 2003; Gorodkin et al., 2001). ILM (Ruan et al., 2004)
combines thermodynamic and mutual information content
scores. Sankoff (Hofacker et al., 2004) merges the sequence
alignment and maximal-pairing folding methods (Nussinov
et al., 1978). Dynalign (Mathews & Turner, 2002) and
Carnac (Touzet & Perriquet, 2004) are the subsequent vari-
ants of Sankoff algorithms. RNA forester (Hochsmann et al.,
2003) introduces a tree alignment model for global and local
alignments. However, the limited number of known RNA
families (Nawrocki et al., 2015) impedes the development.
Energy-based Folding Algorithms When the structures
consist only of nested base pairing, dynamic programming
can predict the structure by minimizing energy. Early
works include Vienna RNAfold (Lorenz et al., 2011),
Mfold (Zuker, 2003), RNAstructure (Mathews & Turner,
2006), and CONTRAfold (Do et al., 2006). Faster imple-
mentations that speed up dynamic programming have been
proposed, such as Vienna RNAplfold (Bernhart et al., 2006),
LocalFold (Lange et al., 2012), and LinearFold (Huang et al.,
2019). However, they cannot accurately predict structures
with pseudoknots, as predicting the lowest free energy struc-
tures with pseudoknots is NP-complete (Lyngsø & Pedersen,
2000), making it difficult to improve performance.
Learning-based Folding Algorithms Deep learning
methods have inspired approaches in bioengineering appli-
cations (Wu et al., 2024a;b; Lin et al., 2022; 2023; Tan et al.,
2024; 2023). SPOT-RNA (Singh et al., 2019) is a seminal
work that employs deep learning for RNA secondary struc-
ture prediction. SPOT-RNA2 (Singh et al., 2021) improves
its predecessor by using evolution-derived sequence pro-
files and mutational coupling. Inspired by Raptor-X (Wang
et al., 2017) and SPOT-Contact (Hanson et al., 2018), SPOT-
RNA uses ResNet and bidirectional LSTM with a sigmoid
function. MXfold (Akiyama et al., 2018) combines sup-
port vector machines and thermodynamic models. CDP-
fold (Zhang et al., 2019), DMFold (Wang et al., 2019), and
MXFold2 (Sato et al., 2021) integrate deep learning tech-
niques with energy-based methods. E2Efold (Chen et al.,
2019) constrains the output to be valid by learning unrolled
algorithms. However, its relaxation for making the optimiza-
tion tractable may violate the constraints. UFold (Fu et al.,
2022) introduces U-Net model to improve performance.
2
Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching Perspective
3. Background
3.1. Preliminaries
The primary structure of RNA is a sequence of nucleotide
bases A, U, C, and G, arranged in order and represented as
X “ px1, ..., xLq, where each xi denotes one of these bases.
The secondary structure is the set of base pairings within
the sequence, modeled as a sparse matrix M P t0, 1uLˆL,
where M ij “ 1 indicates a bond between bases i and j. The
key challenges include (i) designing a model, characterized
by parameters Θ, that captures the complex transformations
from the sequence X to the pairing matrix M and (ii) cor-
rectly identifying the sparsity of the secondary structure,
which is determined by the nature of RNA. Thus, the trans-
formation FΘ : X Þ Ñ M is usually decomposed into two
stages for capturing the sequence-to-structure relationship
and optimizing the sparsity of the target matrix:
FΘ :“ Gθg ˝ Hθh , (1)
where Hθh : X Þ Ñ H represents the initial processing
step, transforming the RNA sequence into an intermediate,
unconstrained representation H P RLˆL. Subsequently,
Gθg : H Þ Ñ M parameterizes the optimization stage for the
intermediate distribution into the final sparse matrix M .
3.2. Constrained Optimization-based Approaches
The core problem of secondary structure prediction lies in
sparsity identification. Numerous studies regard this task
as a constrained optimization problem, seeking the optimal
refinement mappings by gradient descent. Besides, keeping
the hard constraints on RNA secondary structures is also
essential, which ensures valid biological functions (Steeg,
1993). These constraints can be formally described as:
• (a) Only three types of nucleotide combinations can form
base pairs: B :“ tAU, UAu Y tGC, CGu Y tGU, UGu.
For any base pair xixj where xixj R B, M ij “ 0.
• (b) No sharp loop within three bases. For any adjacent
bases within a distance of three nucleotides, they cannot
form pairs with each other. For all |i ´ j| ă 4, M ij “ 0.
• (c) There can be at most one pair for each base. For all i
and j, řL
j“1 M ij ď 1, řL
i“1 M ij ď 1.
The search for valid secondary structures is thus a quest
for symmetric sparse matrices P t0, 1uLˆL that adhere to
the constraints above. The first two constraints can be sat-
isfied by defining a constraint matrix ĎM as: ĎM ij :“ 1
if xixj P B and |i ´ j| ě 4, and ĎM ij :“ 0 otherwise.
Addressing the third constraint is critical as it necessitates
employing sparse optimization techniques. Therefore, our
primary objective is to devise an effective sparse optimiza-
tion strategy. This strategy is based on the symmetric in-
herent distribution H and M , which support constraints (a)
and (b), and additionally addresses constraint (c).
SPOT-RNA subtly enforces the principles of sparsity. It
streamlines the pathway from the raw neural network output
H by harnessing the Sigmoid function to distill a sparse
pattern. The transformation applies a threshold to yield a
binary sparse matrix. This process can be represented as:
GpHq “ 1rSigmoidpHqąss. (2)
In this approach, a fixed threshold s of 0.5 is applied, typical
for inducing sparsity. It omits complex constraints or extra
parameters θg , simply using this cutoff to achieve sparse
structure representations.
E2Efold introduces a non-linear transformation to the in-
termediate value xM P RLˆL and an additional regulariza-
tion term } xM }1.
1
2
A
H ´ s, T p xM q
E
´ ρ} xM }1, (3)
where T p xM q “ 1
2 p xM d xM ` p xM d xM qT q d ĎM ensures
symmetry and adherence to RNA base-pairing constraints
(a) and (b), s is the log-ratio bias term set to logp9.0q, and the
ℓ1 penalty ρ} xM }1 promotes sparsity. To fulfill constraint
(c), the objective is combined with conditions T p xM q1 ď 1.
Denote λ P RL
` as the Lagrange multiplier, the formulation
for the sparse optimization is expressed as:
min
λě0 max
xM
1
2
A
H ´ s, T p xM q
E
´ ρ} xM }1
´
A
λ, ReLUpT p xM q1 ´ 1q
E
,
(4)
In the training stage, the optimization objective is the output
of score function S dependent on xM and H. It can be
regarded as an optimization function G parameterized by
θg :
Gθg pHq “ T parg max xM PRLˆL Sp xM , Hqq. (5)
Although the complicated design to the constraints is ex-
plicitly formulated, the iterative updates may fall into sub-
optimal or invalid solutions. Besides, it requires additional
parameters θg , making the model training complicated.
RTfold introduces a differentiable function that incorpo-
rates an additional Gaussian perturbation W . The objective
function is expressed as:
min
xM
1
N
Nÿ
i“1
T
´
H ` ϵW piq¯
´ xM (6)
where T denotes the non-linear transformation to constrain
the initial output H, and N is the number of random sam-
ples. The random perturbation W piq adjusts the distribution
by leveraging the gradient during the optimization process.
3
Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching Perspective
While RTFold designs an efficient differential objective
function, the constraints imposed by the non-linear transfor-
mation on a noisy hidden distribution may lead to biologi-
cally implausible structures.
4. RFold
4.1. Probabilistic K-Rook Matching
The symmetric K-Rook arrangement (Riordan, 2014; Elkies
& Stanley, 2011) is a classic combinatorial problem in-
volving the placement of KpK ď Lq non-attacking Rooks
on an L ˆ L chessboard, where the goal is to arrange the
Rooks such that they form a symmetric pattern. The term
’non-attacking’ means that no two Rooks are positioned in
the same row or column. An interesting parallel can be
drawn between this combinatorial scenario and the domain
of RNA secondary structure prediction, as illustrated in Fig-
ure 3. This analogy stems from the conceptual similarity in
the arrangement patterns required in both cases. The RNA
sequence can be regarded as a chessboard of size L and the
base pairs are the Rooks. The core problem is to determine
an optimal arrangement of these base pairs.a d heb c f g
8
4
1
3
2
7
6
5
A C AUA C f C
U
G
C
A
A
A
C
C
(a) symmetric Rooks arrangement (b) RNA secondary structure
Figure 3. The analogy between the symmetric K-Rook arrange-
ment and the RNA secondary structure prediction.
Given a finite solution space M defined by the symmetric
K-Rook arrangement, we reformulate our objective as a
probabilistic matching problem. The goal is to find the most
matching solution M ˚ P M for the given sequence X.
The optimal solution M ˚ is defined as:
M ˚ “ arg max
M PM PpM |Xq. (7)
According to Bayes’ theorem, the posterior probability can
be represented as PpM |Xq “ PpX|M qPpM q
PpXq . Since the
denominator P pXq is constant for all M , and assuming
that the solution space is finite and each solution within it is
equally likely, we can adopt a uniform prior PpM q in this
context. Therefore, maximizing the posterior probability is
equivalent to maximizing the likelihood PpX|M q. This
leads to the following equation:
M ˚ “ arg max
M PM PpX|M q. (8)
Therefore, our primary task becomes computing the like-
lihood PpX|M q for the given sequence X under each
possible solution M .
4.2. Bi-dimensional Optimization
Computing the likelihood PpX|M q directly poses sig-
nificant challenges. To address this, we propose a bi-
dimensional optimization strategy that simplifies the prob-
lem by decomposing it into row-wise and column-wise com-
ponents. This approach is mathematically represented as:
PpX|M q “ PpX|RqPpX|Cq, (9)
where M is the product of the row-wise component R P
RLˆL and the column-wise component C P RLˆL, i.e.,
M “ R d C. Each component represents the optimal solu-
tion for the row-wise and column-wise matching problems,
respectively. Importantly, the row-wise and column-wise
components are independent, and the comprehensive solu-
tion for the entire problem is derived from the product of
the optimal solutions for these two sub-problems.
Applying Bayes’ theorem, for the row-wise component, we
have PpR|Xq “ PpX|RqPpRq
PpXq . Given that the solution
space of R is both finite and valid, we can regard it as a
uniform distribution. The analysis for the column-wise com-
ponent, PpC|Xq, follows a similar approach. Therefore,
the optimal solution M ˚ can be represented as:
M ˚ “ arg max
R,C PpR|XqPpC|Xq
“ arg max
R PpR|Xq arg max
C PpC|Xq (10)
The next phase involves establishing proxies for PpR|Xq
and PpC|Xq. To this end, we introduce the basic sym-
metric hidden distribution, xH “ pH d HT q d ĎM . The
row-wise and column-wise components are then derived by
applying Softmax functions to xH, resulting in their respec-
tive probability distributions:
RpxHq “ exppxHij q
řL
k“1 exppxHikq , CpxHq “ exppxHij q
řL
k“1 exppxHkj q .
(11)
The final output is the element-wise product of the row-wise
component RpxHq and the column-wise component CpxHq.
This operation integrates the individual insights from both
dimensions, leading to the optimized matrix M ˚:
M ˚ “ arg max RpxHq d arg max CpxHq. (12)
As illustrated in Figure 4, we consider a random symmetric
6 ˆ 6 matrix as an example. For simplicity, we disregard
4
Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching Perspectiveargmax ℛ 𝑯)
⊙ argmax 𝒞(𝑯)̂ )
symmetric matrix 𝑯) argmax ℛ(𝑯) ) argmax 𝒞(𝑯) )
Figure 4. The visualization of arg max Rp xHq d arg max Cp xHq.
the constraints (a-b) from ĎM . This example demonstrates
the outputs of Rp¨q, Cp¨q, and their element-wise product
Rp¨q d Cp¨q. The row-wise and column-wise components
jointly select the value that has the maximum in both its row
and column while keeping the output matrix symmetric.
Given the definition of xH “ pH d HT q d ĎM , it is
evident that xH inherently forms a symmetric and non-
negative matrix. Regarding optimization, the operation
RpxHq d CpxHq can be equivalently simplified to optimizing
1
2 pRpxHq ` CpxHqq. This is because both approaches fun-
damentally aim to maximize the congruence between the
row-wise and column-wise components of xH. The underly-
ing reason for this equivalence is that both optimizing the
Hadamard product and the arithmetic mean of RpxHq and
CpxHq focus on reinforcing the alignment and coherence
across the various dimensions of the matrix.
Moreover, examining the gradients of these operations sheds
light on their computational efficiencies. The gradient of
RpxHqdCpxHq entails a blend of partial derivatives intercon-
nected via element-wise multiplication. It can be formally
expressed as follows:
BpRpxHq d CpxHqqij
B xHij
“CpxHqij ¨ BRpxHqij
B xHij
` RpxHqij ¨ BCpxHqij
B xHij
.
(13)
In contrast, the gradient of 1
2 pRpxHq ` CpxHqq is character-
ized by a straightforward sum of partial derivatives:
BpRpxHq ` CpxHqqij
B xHij
“ BRpxHqij
B xHij
` BCpxHqij
B xHij
. (14)
Element-wise addition, as used in the latter, tends to be
numerically more stable and less susceptible to issues like
floating-point precision errors, which are more common in
element-wise multiplication operations. This stability is
particularly beneficial when dealing with large-scale matri-
ces or when the gradients involve extreme values, where
numerical instability can pose significant challenges.
The proposed simplification not only maintains the math-
ematical integrity of the optimization problem but also
provides computational advantages, making it a desirable
strategy in practical scenarios involving large and intricate
datasets. Consequently, we define the overall loss function
as the mean square error (MSE) between the averaged row-
wise and column-wise components of xH and the ground
truth secondary structure M :
LpM ˚, M q “ 1
L2
›
›
› 1
2 pRpxHq ` CpxHqq ´ M
›
›
›2
. (15)
4.3. Practical Implementation
We identify the problem of predicting H P RLˆL from the
given sequence attention map pZ P RLˆL as an image-to-
image segmentation problem and apply the U-Net model to
extract pair-wise information, as shown in Figure 5.sequence one-hot
𝑿: 𝐿×4
Token embedding
𝐿×𝐷
Seq2map Attention
𝐿×𝐿×1
+
Positional
embedding
𝐿×𝐷
1 32 3264
128
64
256 512
256
128
𝐿×𝐿
(𝐿/2)×(𝐿/2)
(𝐿/4)×(𝐿/4)
(𝐿/8)×(𝐿/8)
(𝐿/16)×(𝐿/16)
1
𝐿×𝐿
𝑯
𝒁+
𝒁
Figure 5. The overview model architecture of RFold.
To automatically learn informative representations from
sequences, we propose a Seq2map attention module. Given
a sequence in one-hot form X P RLˆ4, we first obtain the
sum of the token embedding and positional embedding as
the input of the Seq2map attention. We denote the input as
Z P RLˆD for convenience, where D is the hidden layer
size of the token and positional embeddings.
Motivated by the recent progress in attention mecha-
nisms (Vaswani et al., 2017; Choromanski et al., 2020;
Katharopoulos et al., 2020; Hua et al., 2022), we aim to
develop a highly effective sequence-to-map transforma-
tion based on pair-wise attention. We obtain the query
Q P RLˆD and key K P RLˆD by applying per-dim
scalars and offsets to Z:
Q “ γQZ ` βQ,
K “ γK Z ` βK , (16)
where γQ, γK , βQ, βK P RLˆD are learnable parameters.
Then, the pair-wise attention map is obtained by:
sZ “ ReLU2pQKT {Lq, (17)
5
Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching Perspective
where ReLU2 is an activation function that can be recog-
nized as a simplified Softmax function in vanilla Transform-
ers (So et al., 2021). The output of Seq2map is the gated
representation of sZ:
pZ “ sZ d σp sZq, (18)
where σp¨q is the Sigmoid function that performs as a gate.
5. Experiments
We conduct experiments to compare our proposed RFold
with state-of-the-art and commonly used approaches. Mul-
tiple experimental settings are taken into account, includ-
ing standard structure prediction, generalization evaluation,
large-scale benchmark evaluation, cross-family evaluation,
pseudoknot prediction and inference time comparison. De-
tailed experimental setups can be found in the Appendix B.
5.1. Standard RNA Secondary Structure Prediction
Following (Chen et al., 2019), we split the RNAStralign
dataset into train, validation, and test sets by stratified sam-
pling. We report the results in Table 1. Energy-based meth-
ods achieve relatively weak F1 scores ranging from 0.420
to 0.633. Learning-based folding algorithms like E2Efold
and UFold significantly improve performance by large mar-
gins, while RFold obtains even better performance among
all the metrics. Moreover, RFold obtains about 8% higher
precision than the state-of-the-art method. This suggests
that our optimization strategy is strict to satisfy all the hard
constraints for predicting valid structures.
Table 1. Results on RNAStralign test set. Results in bold and
underlined are the top-1 and top-2 performances, respectively.
Method Precision Recall F1
Mfold 0.450 0.398 0.420
RNAfold 0.516 0.568 0.540
RNAstructure 0.537 0.568 0.550
CONTRAfold 0.608 0.663 0.633
LinearFold 0.620 0.606 0.609
CDPfold 0.633 0.597 0.614
E2Efold 0.866 0.788 0.821
UFold 0.905 0.927 0.915
RFold 0.981 0.973 0.977
5.2. Generalization Evaluation
To verify the generalization ability of our proposed RFold,
we directly evaluate the performance on another benchmark
dataset ArchiveII using the pre-trained model on the RNAS-
tralign training dataset. Following (Chen et al., 2019), we
exclude RNA sequences in ArchiveII that have overlapping
RNA types with the RNAStralign dataset for a fair compari-
son. The results are reported in Table 2.
It can be seen that traditional methods achieve F1 scores
in the range of 0.545 to 0.842. A recent learning-based
method, MXfold2, obtains an F1 score of 0.768, which is
even lower than some energy-based methods. Another state-
of-the-art learning-based method improves the performance
to the F1 score of 0.905. RFold further improves the F1
score to 0.921, even higher than UFold. It is worth noting
that RFold has a relatively lower result in the recall metric
and a significantly higher result in the precision metric. The
reason for this phenomenon may be the strict constraints of
RFold. While none of the current learning-based methods
can satisfy all the constraints we introduced in Sec. 3.2,
the predictions of RFold are guaranteed to be valid. Thus,
RFold may cover fewer pair-wise interactions, leading to
a lower recall metric. However, the highest F1 score still
suggests the great generalization ability of RFold.
Table 2. Results on ArchiveII dataset. Results in bold and under-
lined are the top-1 and top-2 performances, respectively.
Method Precision Recall F1
Mfold 0.668 0.590 0.621
CDPfold 0.557 0.535 0.545
RNAfold 0.663 0.613 0.631
RNAstructure 0.664 0.606 0.628
CONTRAfold 0.696 0.651 0.665
LinearFold 0.724 0.605 0.647
RNAsoft 0.665 0.594 0.622
Eternafold 0.667 0.622 0.636
E2Efold 0.734 0.660 0.686
SPOT-RNA 0.743 0.726 0.711
MXfold2 0.788 0.760 0.768
Contextfold 0.873 0.821 0.842
RTfold 0.891 0.789 0.814
UFold 0.887 0.928 0.905
RFold 0.938 0.910 0.921
5.3. Large-scale Benchmark Evaluation
The bpRNA dataset is a large-scale benchmark, comprises
fixed training (TR0), evaluation (VL0), and testing (TS0)
sets. Following previous works (Singh et al., 2019; Sato
et al., 2021; Fu et al., 2022), we train the model in bpRNA-
TR0 and evaluate the performance on bpRNA-TS0 by using
the best model learned from bpRNA-VL0. The detailed
results can be found in Table 3.
RFold outperforms the prior state-of-the-art method, SPOT-
RNA, by a notable 4.0% in terms of the F1 score. This
improvement in the F1 score can be attributed to the con-
sistently superior performance of RFold in the precision
metric when compared to baseline models. However, it is
important to note that the recall metric remains constrained,
6
Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching Perspective
Table 3. Results on bpRNA-TS0 set.
Method Precision Recall F1
E2Efold 0.140 0.129 0.130
RNAstructure 0.494 0.622 0.533
RNAsoft 0.497 0.626 0.535
RNAfold 0.494 0.631 0.536
Mfold 0.501 0.627 0.538
Contextfold 0.529 0.607 0.546
LinearFold 0.561 0.581 0.550
MXfold2 0.519 0.646 0.558
Externafold 0.516 0.666 0.563
CONTRAfold 0.528 0.655 0.567
SPOT-RNA 0.594 0.693 0.619
UFold 0.521 0.588 0.553
RFold 0.692 0.635 0.644
likely due to stringent constraints applied during prediction.
Table 4. Results on long-range bpRNA-TS0 set. Results in bold
and underlined are the top-1 and top-2 performances, respectively.
Method Precision Recall F1
Mfold 0.315 0.450 0.356
RNAfold 0.304 0.448 0.350
RNAstructure 0.299 0.428 0.339
CONTRAfold 0.306 0.439 0.349
LinearFold 0.281 0.355 0.305
RNAsoft 0.310 0.448 0.353
Externafold 0.308 0.458 0.355
SPOT-RNA 0.361 0.492 0.403
MXfold2 0.318 0.450 0.360
Contextfold 0.332 0.432 0.363
UFold 0.543 0.631 0.584
RFold 0.803 0.765 0.701
Following (Fu et al., 2022), we conduct an experiment on
long-range interactions. Given a sequence of length L, the
long-range base pairing is defined as the paired and unpaired
bases with intervals longer than L{2. As shown in Table 4,
RFold performs unexpectedly well on these long-range base
pairing predictions and improves UFold in all metrics by
large margins, demonstrating its strong predictive ability.
5.4. Cross-family Evaluation
The bpRNA-new dataset is a cross-family benchmark
dataset comprising 1,500 RNA families, presenting a signifi-
cant challenge for pure deep learning approaches. UFold, for
instance, relies on the thermodynamic method Contrafold
for data augmentation to achieve satisfactory results. As
shown in Table 5, the standard UFold achieves an F1 score
of 0.583, while RFold reaches 0.616. When the same data
augmentation technique is applied, UFold’s performance
increases to 0.636, whereas RFold yields a score of 0.651.
This places RFold second only to the thermodynamics-based
method, Contrafold, in terms of F1 score.
Table 5. Results on bpRNA-new. Results in bold and underlined
are the top-1 and top-2 performances, respectively.
Method Precision Recall F1
E2Efold 0.047 0.031 0.036
SPOT-RNA 0.635 0.641 0.620
Contrafold 0.620 0.736 0.661
UFold 0.500 0.736 0.583
UFold + aug 0.570 0.742 0.636
RFold 0.614 0.619 0.616
RFold + aug 0.618 0.687 0.651
5.5. Predict with Pseudoknots
Following E2Efold (Chen et al., 2019), we consider a se-
quence to be a true positive if it is correctly identified as
containing a pseudoknot. For this analysis, we extracted all
sequences featuring pseudoknots from the RNAStralign test
dataset and assessed their predictive accuracy. The results
of this analysis are summarized in the following table:
Table 6. Results on RNA structures with pseudoknots.
Method Precision Recall F1 Score
RNAstructure 0.778 0.761 0.769
SPOT-RNA 0.677 0.978 0.800
E2Efold 0.844 0.990 0.911
UFold 0.962 0.990 0.976
RFold 0.971 0.993 0.982
RFold demonstrates superior performance compared to its
counterparts across all evaluated metrics, i.e., precision,
recall, and F1 score. This consistent outperformance across
multiple dimensions of accuracy underscores the efficacy
and robustness of the RFold approach in predicting RNA
structures with pseudoknots.
5.6. Inference Time Comparison
We compared the running time of various methods for pre-
dicting RNA secondary structures using the RNAStralign
testing set with the same experimental setting and the hard-
ware environment as in (Fu et al., 2022). The results are pre-
sented in Table 7, which shows the average inference time
per sequence. The fastest energy-based method, LinearFold,
takes about 0.43s for each sequence. The learning-based
baseline, UFold, takes about 0.16s. RFold has the highest
inference speed, costing only about 0.02s per sequence. In
particular, RFold is about eight times faster than UFold and
sixteen times faster than MXfold2.
5.7. Ablation Study
Bi-dimensional Optimization To validate the effective-
ness of our proposed bi-dimensional optimization strategy,
we conduct an experiment that replaces them with other op-
7
Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching Perspective
Table 7. Inference time on the RNAStralign test set. Results in bold
and underlined are the top-1 and top-2 performances, respectively.
Method Time
CDPfold (Tensorflow) 300.11 s
RNAstructure (C) 142.02 s
CONTRAfold (C++) 30.58 s
Mfold (C) 7.65 s
Eternafold (C++) 6.42 s
RNAsoft (C++) 4.58 s
RNAfold (C) 0.55 s
LinearFold (C++) 0.43 s
SPOT-RNA(Pytorch) 77.80 s (GPU)
E2Efold (Pytorch) 0.40 s (GPU)
MXfold2 (Pytorch) 0.31 s (GPU)
UFold (Pytorch) 0.16 s (GPU)
RFold (Pytorch) 0.02 s (GPU)
timization methods. The results are summarized in Table 8,
where RFold-E and RFold-S denote our model with the opti-
mization strategies of E2Efold and SPOT-RNA, respectively.
While precision, recall, and F1 score are evaluated at base-
level, we report the validity which is a sample-level metric
evaluating whether the predicted structure satisfies all the
constraints. It can be seen that though RFold-E has compa-
rable performance in the first three metrics with ours, many
of its predicted structures are invalid. The optimization
strategy of SPOT-RNA has incorporated no constraint that
results in its low validity. Moreover, its strategy seems to not
fit our model well, which may be caused by the simplicity
of our proposed RFold model.
Table 8. Ablation study on different optimization strategies
(RNAStralign testing set).
Method Precision Recall F1 Validity
RFold 0.981 0.973 0.977 100.00%
RFold-E 0.888 0.906 0.896 50.31%
RFold-S 0.223 0.988 0.353 0.00%
Seq2map Attention We also conduct an experiment to
evaluate the proposed Seq2map attention. We replace the
Seq2map attention with the hand-crafted features from
UFold and the outer concatenation from SPOT-RNA, which
are denoted as RFold-U and RFold-SS, respectively. In ad-
dition to performance metrics, we also report the average
inference time for each RNA sequence to evaluate the model
complexity. We summarize the result in Table 9. It can be
seen that RFold-U takes much more inference time than
our RFold and RFold-SS due to the heavy computational
cost when loading and learning from hand-crafted features.
Moreover, it is surprising to find that RFold-SS has a little
better performance than RFold-U, with the least inference
time for its simple outer concatenation operation. However,
neither RFold-U nor RFold-SS can provide informative rep-
resentations like our proposed Seq2map attention. With
comparable inference time with the simplest RFold-SS, our
RFold outperforms baselines by large margins.
Table 9. Ablation study on different pre-processing strategies
(RNAStralign testing set).
Method Precision Recall F1 Time
RFold 0.981 0.973 0.977 0.0167
RFold-U 0.875 0.941 0.906 0.0507
RFold-SS 0.886 0.945 0.913 0.0158
Row-wise and Column-wise Componenets We con-
ducted comprehensive ablation studies on the row-wise and
column-wise components of our proposed model, RFold,
by modifying the inference mechanism using pre-trained
checkpoints. These studies were meticulously designed to
isolate and understand the individual contributions of these
components to our model’s performance in RNA secondary
structure prediction. The results, presented across three
datasets—RNAStralign (Table 10), ArchiveII (Table 11),
and bpRNA-TS0 (Table 12)—highlight two key findings:
(i) Removing both the row-wise and column-wise compo-
nents leads to a substantial drop in the model’s performance,
underscoring their pivotal role within our model’s archi-
tecture. This dramatic reduction in effectiveness clearly
demonstrates that both components are integral to achieving
high accuracy. The significant decline in performance when
these components are omitted highlights their essential func-
tion in capturing the complex dependencies within RNA
sequences. (ii) The performance metrics when isolating
either the row-wise or column-wise components are remark-
ably similar across all datasets. This uniformity suggests
that the training process, which incorporates row-wise and
column-wise softmax functions, likely yields symmetric
outputs. Consequently, this symmetry implies that each
component contributes in an almost equal measure to the
model’s overall predictive capacity.
Table 10. Ablation study on row-wise and column-wise compo-
nents (RNAStralign testing set).
Method Precision Recall F1 Validity
RFold 0.981 0.973 0.977 100.00%
RFold w/o C 0.972 0.975 0.973 75.99%
RFold w/o R 0.972 0.975 0.973 75.99%
RFold w/o R,C 0.016 0.031 0.995 0.00%
5.8. Visualization
We visualize two examples predicted by RFold and UFold
in Figure 6. The corresponding F1 scores are denoted at
8
Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching Perspective
Table 11. Ablation study on row-wise and column-wise compo-
nents (ArchiveII).
Method Precision Recall F1 Validity
RFold 0.938 0.910 0.921 100.00%
RFold w/o C 0.919 0.914 0.914 49.14%
RFold w/o R 0.919 0.914 0.914 49.14%
RFold w/o R,C 0.013 0.997 0.025 0.00%
Table 12. Ablation study on row-wise and column-wise compo-
nents (bpRNA-TS0).
Method Precision Recall F1 Validity
RFold 0.693 0.635 0.644 100.00%
RFold w/o C 0.652 0.651 0.637 12.97%
RFold w/o R 0.652 0.651 0.637 12.97%
RFold w/o R,C 0.021 0.995 0.040 0.00%
the bottom right of each plot. The first row of secondary
structures is a simple example of a nested structure. It can
be seen that UFold may fail in such a case. The second row
of secondary structures is much more difficult that contains
over 300 bases of the non-nested structure. While UFold
fails in such a complex case, RFold can predict the structure
accurately. Due to the limited space, we provide more
visualization comparisons in Appendix D.RFoldTrue UFold
A
A
C
C
A
U
U
A
A
G
G
A
A
U
A
G A C C
A
A
G
C
U
C
U
A
G
G
U
G
G
U
U G A G
A
A
A
C
C
C
C U
U
U
G
U
A
U
U A G
U
C
C
U
G
G
A A
A
C
A
G
G
G
C G A C
A
U
U
G
U
C
A
A
A
U
UG
U
U
C
G
G
GG
A
C
C
A
C
C
CG
C
U
A
A
A
U U
A
C
A
U
G
C
U
A
C
CG
C
A
G
C
A
G
U
G
C
U
GA
A
A
G
G
C
C
U
G
U
G
A
G
C
A
C
U
A
G
A
G
G
U
A
A C
G
C
C
U
C
U
A
G G G
A
U
G
G
U
A A
U
A
A
C
G
C
G
U
G
U
A
U
A
G
G
G
U A
U
A
U
C
C
G C A
G
C
GA
A
G
U
U
CU
A
A
G
G
C
C
U
U C
U
G
C
U A
C
G
A
A
U C
G
C
G
U U C A C
A
G
A
C
U
A
G
A
C
G
G
C
A
A
U
G G
G
C
U
C
C
U
U G
C
G
G
G
G
C
U U A A G A U A U A G U C G A
A
C
CC
C
U
C
A
G
A
G
A U
G
A
G
G
A U
G G
A A
U
C
A
A
U G
1
10
20
30
40
50
60
70
80
90
100
110
120
130
140
150 160
170
180
190
200
210
220
230
240
250
260
270 280
290
300
310
A
A
C
C
A
U
U
A
A
G
G
A
A
U
A
G A C C
A
A
G
C
U
C
U
A
G
G
U
G
G
U
U G A G
A
A
A
C
C
C
C U
U
U
G
U
A
U
U A G
U
C
C
U
G
G
A A
A
C
A
G
G
G
C G A C
A
U
U
G
U
C
A
A
A
U
UG
U
U
C
G
G
GG
A
C
C
A
C
C
CG
C
U
A
A
A
U U
A
C
A
U
G
C
U
A
C
CG
C
A
G
C
A
G
U
G
C
U
GA
A
A
G
G
C
C
U
G
U
G
A
G
C
A
C
U
A
G
A
G
G
U
A
A C
G
C
C
U
C
U
A
G G G
A
U
G
G
U
A A
U
A
A
C
G
C
G
U
G
U
A
U
A
G
G
G
U A
U
A
U
C
C
G C A
G
C
GA
A
G
U
U
CU
A
A
G
G
C
C
U
U C
U
G
C
U A
C
G
A
A
U C
G
C
G
U U C A C
A
G
A
C
U
A
G
A
C
G
G
C
A
A
U
G G
G
C
U
C
C
U
U G
C
G
G
G
G
C
U U A A G A U A U A G U C G A
A
C
CC
C
U
C
A
G
A
G
A U
G
A
G
G
A U
G G
A A
U
C
A
A
U G
1
10
20
30
40
50
60
70
80
90
100
110
120
130
140
150 160
170
180
190
200
210
220
230
240
250
260
270 280
290
300
310
A A
C
C
A
U
U
A
A
G
G
A
A
U
A
G A C C
A
A
G
C
U
C
U
A
G
G
U
G
G
U U
G
A G
A
A A C C C C U U U G U A U U A G
U
C
C
U
G
G
A A
A
C
A
G
G
G
C G A C
A
U
U
G
U
C
A
A
A
U
U
G
U
U
C
G
GGG
A
C
C
A
C
C
CG
C
U
A
A
A
U U
A
C
A
U
G
C
U
A
C
C
G
CA
G
C
AGUGCU
G
A
AA
G
G
C
C U G
U G A G C A C U
A
G
A
G
G
U
A
A C
G
C
C
U
C
U A G
G
G
A
U
G
G
U A A
U
A
A
C
G
C
G
U
G
U
A
U
A
G
G
G
U
A U A U
C
C
G
C
A
G
C G
A
A
G
U
U
C
U
A
A
G
G C C U
U
C
U
G
C
U
A
CGA
A
U
C G C G
U
U
C
A
C
A
G
A
C
U
A
G
A
C
G
G
C
A
A
U
G G
G
C
U
C
C
U
U G
C
G
G
G
G
C
U U A A G A U A U
1
10
20
30
40 50
60
70
80
90
100
110
120
130
140
150
160
170
180
190 200
210
220
230
240
250
260
270
RFoldTrue UFold
1.000
0.995 0.558
0.823
Figure 6. Visualization of the true and predicted structures.
6. Conclusion
In this study, we reformulate RNA secondary structure pre-
diction as a K-Rook problem, thus transforming the predic-
tion process into probabilistic matching. Subsequently, we
introduce RFold, an efficient learning-based model, which
utilizes a bidimensional optimization strategy to decom-
pose the probabilistic matching into row-wise and column-
wise components, simplifying the solving process while
guaranteeing the validity of the output. Comprehensive
experiments demonstrate that RFold achieves competitive
performance with faster inference speed.
The limitations of RFold primarily revolve around its strin-
gent constraints. This strictness in constraints implies that
RFold is cautious in predicting interactions, leading to
higher precision but possibly at the cost of missing some
true interactions. Though we have provided a naive so-
lution in Appendix C, it needs further studies to obtain a
better strategy that leads to more balanced precision-recall
trade-offs and more comprehensive structural predictions.
Acknowledgements
This work was supported by National Science and Technol-
ogy Major Project (No. 2022ZD0115101), National Natural
Science Foundation of China Project (No. U21A20427),
Project (No. WU2022A009) from the Center of Synthetic
Biology and Integrated Bioengineering of Westlake Univer-
sity and Integrated Bioengineering of Westlake University
and Project (No. WU2023C019) from the Westlake Univer-
sity Industries of the Future Research Funding.
Impact Statement
RFold is the first learning-based method that guarantees
the validity of predicted RNA secondary structures. Its
capability to ensure accurate predictions. It can be a valuable
tool for biologists to study the structure and function of RNA
molecules. Additionally, RFold stands out for its speed,
significantly surpassing previous methods, marking it as
a promising avenue for future developments in this field.
There are many potential societal consequences of our work,
none of which we feel must be specifically highlighted here.
References
Akiyama, M., Sato, K., and Sakakibara, Y. A max-margin
training of rna secondary structure prediction integrated
with the thermodynamic model. Journal of bioinformatics
and computational biology, 16(06):1840025, 2018.
Andronescu, M., Aguirre-Hernandez, R., Condon, A., and
Hoos, H. H. Rnasoft: a suite of rna secondary struc-
ture prediction and design software tools. Nucleic acids
research, 31(13):3416–3422, 2003.
Bellaousov, S., Reuter, J. S., Seetin, M. G., and Mathews,
D. H. Rnastructure: web servers for rna secondary struc-
ture prediction and analysis. Nucleic acids research, 41
(W1):W471–W474, 2013.
9
Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching Perspective
Bernhart, S. H., Hofacker, I. L., and Stadler, P. F. Local rna
base pairing probabilities in large sequences. Bioinfor-
matics, 22(5):614–615, 2006.
Berthet, Q., Blondel, M., Teboul, O., Cuturi, M., Vert, J.-
P., and Bach, F. Learning with differentiable pertubed
optimizers. Advances in neural information processing
systems, 33:9508–9519, 2020.
Chen, X., Li, Y., Umarov, R., Gao, X., and Song, L. Rna
secondary structure prediction by learning unrolled algo-
rithms. In International Conference on Learning Repre-
sentations, 2019.
Cheong, H.-K., Hwang, E., Lee, C., Choi, B.-S., and
Cheong, C. Rapid preparation of rna samples for nmr
spectroscopy and x-ray crystallography. Nucleic acids
research, 32(10):e84–e84, 2004.
Choromanski, K. M., Likhosherstov, V., Dohan, D., Song,
X., Gane, A., Sarlos, T., Hawkins, P., Davis, J. Q., Mo-
hiuddin, A., Kaiser, L., et al. Rethinking attention with
performers. In International Conference on Learning
Representations, 2020.
Do, C. B., Woods, D. A., and Batzoglou, S. Contrafold:
Rna secondary structure prediction without physics-based
models. Bioinformatics, 22(14):e90–e98, 2006.
Elkies, N. and Stanley, R. P. Chess and mathematics. Recu-
perado el, 11, 2011.
Fallmann, J., Will, S., Engelhardt, J., Gr ¨uning, B., Backofen,
R., and Stadler, P. F. Recent advances in rna folding.
Journal of biotechnology, 261:97–104, 2017.
Fica, S. M. and Nagai, K. Cryo-electron microscopy snap-
shots of the spliceosome: structural insights into a dy-
namic ribonucleoprotein machine. Nature structural &
molecular biology, 24(10):791–799, 2017.
Franke, J., Runge, F., and Hutter, F. Probabilistic trans-
former: Modelling ambiguities and distributions for rna
folding and molecule design. Advances in Neural Infor-
mation Processing Systems, 35:26856–26873, 2022.
Franke, J. K., Runge, F., and Hutter, F. Scalable deep
learning for rna secondary structure prediction. arXiv
preprint arXiv:2307.10073, 2023.
Fu, L., Cao, Y., Wu, J., Peng, Q., Nie, Q., and Xie, X. Ufold:
fast and accurate rna secondary structure prediction with
deep learning. Nucleic acids research, 50(3):e14–e14,
2022.
F ¨urtig, B., Richter, C., W ¨ohnert, J., and Schwalbe, H. Nmr
spectroscopy of rna. ChemBioChem, 4(10):936–962,
2003.
Gardner, P. P. and Giegerich, R. A comprehensive compari-
son of comparative rna structure prediction approaches.
BMC bioinformatics, 5(1):1–18, 2004.
Gardner, P. P., Daub, J., Tate, J. G., Nawrocki, E. P., Kolbe,
D. L., Lindgreen, S., Wilkinson, A. C., Finn, R. D.,
Griffiths-Jones, S., Eddy, S. R., et al. Rfam: updates
to the rna families database. Nucleic acids research, 37
(suppl 1):D136–D140, 2009.
Gorodkin, J., Stricklin, S. L., and Stormo, G. D. Discovering
common stem–loop motifs in unaligned rna sequences.
Nucleic Acids Research, 29(10):2135–2144, 2001.
Griffiths-Jones, S., Bateman, A., Marshall, M., Khanna, A.,
and Eddy, S. R. Rfam: an rna family database. Nucleic
acids research, 31(1):439–441, 2003.
Gutell, R. R., Lee, J. C., and Cannone, J. J. The accuracy
of ribosomal rna comparative structure models. Current
opinion in structural biology, 12(3):301–310, 2002.
Hanson, J., Paliwal, K., Litfin, T., Yang, Y., and Zhou, Y.
Accurate prediction of protein contact maps by coupling
residual two-dimensional bidirectional long short-term
memory with convolutional neural networks. Bioinfor-
matics, 34(23):4039–4045, 2018.
He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learn-
ing for image recognition. In Proceedings of the IEEE
conference on computer vision and pattern recognition,
pp. 770–778, 2016.
Hochreiter, S. and Schmidhuber, J. Long short-term memory.
Neural computation, 9(8):1735–1780, 1997.
Hochsmann, M., Toller, T., Giegerich, R., and Kurtz, S.
Local similarity in rna secondary structures. In Computa-
tional Systems Bioinformatics. CSB2003. Proceedings of
the 2003 IEEE Bioinformatics Conference. CSB2003, pp.
159–168. IEEE, 2003.
Hofacker, I. L., Bernhart, S. H., and Stadler, P. F. Alignment
of rna base pairing probability matrices. Bioinformatics,
20(14):2222–2227, 2004.
Hua, W., Dai, Z., Liu, H., and Le, Q. Transformer quality
in linear time. In International Conference on Machine
Learning, pp. 9099–9117. PMLR, 2022.
Huang, L., Zhang, H., Deng, D., Zhao, K., Liu, K., Hen-
drix, D. A., and Mathews, D. H. Linearfold: linear-time
approximate rna folding by 5’-to-3’dynamic program-
ming and beam search. Bioinformatics, 35(14):i295–i304,
2019.
Iorns, E., Lord, C. J., Turner, N., and Ashworth, A. Utilizing
rna interference to enhance cancer drug discovery. Nature
reviews Drug discovery, 6(7):556–568, 2007.
10
Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching Perspective
Jung, A. J., Lee, L. J., Gao, A. J., and Frey, B. J. Rtfold:
Rna secondary structure prediction using deep learning
with domain inductive bias.
Kalvari, I., Nawrocki, E. P., Ontiveros-Palacios, N., Argasin-
ska, J., Lamkiewicz, K., Marz, M., Griffiths-Jones, S.,
Toffano-Nioche, C., Gautheret, D., Weinberg, Z., et al.
Rfam 14: expanded coverage of metagenomic, viral and
microrna families. Nucleic Acids Research, 49(D1):D192–
D200, 2021.
Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F.
Transformers are rnns: Fast autoregressive transformers
with linear attention. In International Conference on
Machine Learning, pp. 5156–5165. PMLR, 2020.
Knudsen, B. and Hein, J. Pfold: Rna secondary structure
prediction using stochastic context-free grammars. Nu-
cleic acids research, 31(13):3423–3428, 2003.
Lange, S. J., Maticzka, D., M ¨ohl, M., Gagnon, J. N., Brown,
C. M., and Backofen, R. Global or local? predicting
secondary structure and accessibility in mrnas. Nucleic
acids research, 40(12):5215–5226, 2012.
Lin, H., Huang, Y., Liu, M., Li, X. C., Ji, S., and Li,
S. Z. Diffbp: Generative diffusion of 3d molecules
for target protein binding. ArXiv, abs/2211.11214,
2022. URL https://api.semanticscholar.
org/CorpusID:253734621.
Lin, H., Huang, Y., Zhang, H., Wu, L., Li, S.,
Chen, Z., and Li, S. Z. Functional-group-
based diffusion for pocket-specific molecule gen-
eration and elaboration. ArXiv, abs/2306.13769,
2023. URL https://api.semanticscholar.
org/CorpusID:259251644.
Lorenz, R., Bernhart, S. H., H ¨oner zu Siederdissen, C.,
Tafer, H., Flamm, C., Stadler, P. F., and Hofacker, I. L.
Viennarna package 2.0. Algorithms for molecular biology,
6(1):1–14, 2011.
Lyngsø, R. B. and Pedersen, C. N. Rna pseudoknot predic-
tion in energy-based models. Journal of computational
biology, 7(3-4):409–427, 2000.
Mathews, D. H. and Turner, D. H. Dynalign: an algorithm
for finding the secondary structure common to two rna
sequences. Journal of molecular biology, 317(2):191–
203, 2002.
Mathews, D. H. and Turner, D. H. Prediction of rna sec-
ondary structure by free energy minimization. Current
opinion in structural biology, 16(3):270–278, 2006.
Nawrocki, E. P., Burge, S. W., Bateman, A., Daub, J., Eber-
hardt, R. Y., Eddy, S. R., Floden, E. W., Gardner, P. P.,
Jones, T. A., Tate, J., et al. Rfam 12.0: updates to the
rna families database. Nucleic acids research, 43(D1):
D130–D137, 2015.
Nicholas, R. and Zuker, M. Unafold: Software for nucleic
acid folding and hybridization. Bioinformatics, 453:3–31,
2008.
Nussinov, R., Pieczenik, G., Griggs, J. R., and Kleitman,
D. J. Algorithms for loop matchings. SIAM Journal on
Applied mathematics, 35(1):68–82, 1978.
Riordan, J. An introduction to combinatorial analysis. 2014.
Rivas, E. The four ingredients of single-sequence rna sec-
ondary structure prediction. a unifying perspective. RNA
biology, 10(7):1185–1196, 2013.
Ruan, J., Stormo, G. D., and Zhang, W. An iterated loop
matching approach to the prediction of rna secondary
structures with pseudoknots. Bioinformatics, 20(1):58–
66, 2004.
Sato, K., Akiyama, M., and Sakakibara, Y. Rna secondary
structure prediction using deep learning with thermody-
namic integration. Nature communications, 12(1):1–9,
2021.
Seetin, M. G. and Mathews, D. H. Rna structure prediction:
an overview of methods. Bacterial regulatory RNA, pp.
99–122, 2012.
Singh, J., Hanson, J., Paliwal, K., and Zhou, Y. Rna sec-
ondary structure prediction using an ensemble of two-
dimensional deep neural networks and transfer learning.
Nature communications, 10(1):1–13, 2019.
Singh, J., Paliwal, K., Zhang, T., Singh, J., Litfin, T., and
Zhou, Y. Improved rna secondary structure and tertiary
base-pairing prediction using evolutionary profile, mu-
tational coupling and two-dimensional transfer learning.
Bioinformatics, 37(17):2589–2600, 2021.
Sloma, M. F. and Mathews, D. H. Exact calculation of
loop formation probability identifies folding motifs in rna
secondary structures. RNA, 22(12):1808–1818, 2016.
So, D., Ma ´nke, W., Liu, H., Dai, Z., Shazeer, N., and Le,
Q. V. Searching for efficient transformers for language
modeling. Advances in Neural Information Processing
Systems, 34:6010–6022, 2021.
Steeg, E. W. Neural networks, adaptive optimization, and
rna secondary structure prediction. Artificial intelligence
and molecular biology, pp. 121–160, 1993.
Szikszai, M., Wise, M. J., Datta, A., Ward, M., and Mathews,
D. Deep learning models for rna secondary structure
prediction (probably) do not generalise across families.
bioRxiv, 2022.
11
Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching Perspective
Tan, C., Zhang, Y., Gao, Z., Hu, B., Li, S., Liu, Z., and Li,
S. Z. Hierarchical data-efficient representation learning
for tertiary structure-based rna design. In The Twelfth
International Conference on Learning Representations,
2023.
Tan, C., Gao, Z., Wu, L., Xia, J., Zheng, J., Yang, X., Liu,
Y., Hu, B., and Li, S. Z. Cross-gate mlp with protein com-
plex invariant embedding is a one-shot antibody designer.
In Proceedings of the AAAI Conference on Artificial In-
telligence, volume 38, pp. 15222–15230, 2024.
Tan, Z., Fu, Y., Sharma, G., and Mathews, D. H. Turbofold
ii: Rna structural alignment and secondary structure pre-
diction informed by multiple homologs. Nucleic acids
research, 45(20):11570–11581, 2017.
Touzet, H. and Perriquet, O. Carnac: folding families of
related rnas. Nucleic acids research, 32(suppl 2):W142–
W145, 2004.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. At-
tention is all you need. Advances in neural information
processing systems, 30, 2017.
Wang, L., Liu, Y., Zhong, X., Liu, H., Lu, C., Li, C., and
Zhang, H. Dmfold: A novel method to predict rna sec-
ondary structure with pseudoknots based on deep learning
and improved base pair maximization principle. Frontiers
in genetics, 10:143, 2019.
Wang, S., Sun, S., Li, Z., Zhang, R., and Xu, J. Accu-
rate de novo prediction of protein contact map by ultra-
deep learning model. PLoS computational biology, 13(1):
e1005324, 2017.
Wang, X. and Tian, J. Dynamic programming for np-hard
problems. Procedia Engineering, 15:3396–3400, 2011.
Wayment-Steele, H. K., Kladwang, W., Strom, A. I., Lee,
J., Treuille, A., Participants, E., and Das, R. Rna sec-
ondary structure packages evaluated and improved by
high-throughput experiments. BioRxiv, pp. 2020–05,
2021.
Wu, L., Huang, Y., Tan, C., Gao, Z., Hu, B., Lin, H.,
Liu, Z., and Li, S. Z. Psc-cpi: Multi-scale protein
sequence-structure contrasting for efficient and gener-
alizable compound-protein interaction prediction. arXiv
preprint arXiv:2402.08198, 2024a.
Wu, L., Tian, Y., Huang, Y., Li, S., Lin, H., Chawla,
N. V., and Li, S. Z. Mape-ppi: Towards effective
and efficient protein-protein interaction prediction via
microenvironment-aware protein embedding. arXiv
preprint arXiv:2402.14391, 2024b.
Xu, X. and Chen, S.-J. Physics-based rna structure predic-
tion. Biophysics reports, 1(1):2–13, 2015.
Zakov, S., Goldberg, Y., Elhadad, M., and Ziv-Ukelson, M.
Rich parameterization improves rna structure prediction.
Journal of Computational Biology, 18(11):1525–1542,
2011.
Zhang, H., Zhang, C., Li, Z., Li, C., Wei, X., Zhang, B.,
and Liu, Y. A new method of rna secondary structure
prediction based on convolutional neural network and
dynamic programming. Frontiers in genetics, 10:467,
2019.
Zuker, M. Mfold web server for nucleic acid folding and
hybridization prediction. Nucleic acids research, 31(13):
3406–3415, 2003.
12
Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching Perspective
A. Comparison of mainstream RNA secondary
structure prediction methods
We compare our proposed method RFold with several other
leading RNA secondary structure prediction methods and
summarize the results in Table 13. RFold satisfies all
three constraints (a)-(c) for valid RNA secondary struc-
tures, while the other methods do not fully meet some of
the constraints. RFold utilizes a sequence-to-map attention
mechanism to capture long-range dependencies, whereas
SPOT-RNA simply concatenates pairwise sequence infor-
mation and E2Efold/UFold uses hand-crafted features. In
terms of prediction accuracy on the RNAStralign benchmark
test set, RFold achieves the best F1 score of 0.977, outper-
forming SPOT-RNA, E2Efold and UFold by a large margin.
Regarding the average inference time, RFold is much more
efficient and requires only 0.02 seconds to fold the RNAS-
tralign test sequences. In summary, RFold demonstrates
superior performance over previous methods for RNA sec-
ondary structure prediction in both accuracy and speed.
B. Experimental Details
Datasets We use three benchmark datasets: (i) RNAS-
tralign (Tan et al., 2017), one of the most comprehensive
collections of RNA structures, is composed of 37,149 struc-
tures from 8 RNA types; (ii) ArchiveII (Sloma & Mathews,
2016), a widely used benchmark dataset in classical RNA
folding methods, containing 3,975 RNA structures from
10 RNA types; (iii) bpRNA (Singh et al., 2019), is a large
scale benchmark dataset, containing 102,318 structures from
2,588 RNA types. (iv) bpRNA-new (Sato et al., 2021), de-
rived from Rfam 14.2 (Kalvari et al., 2021), containing
sequences from 1500 new RNA families.
Baselines We compare our proposed RFold with base-
lines including energy-based folding methods such as
Mfold (Zuker, 2003), RNAsoft (Andronescu et al., 2003),
RNAfold (Lorenz et al., 2011), RNAstructure (Mathews
& Turner, 2006), CONTRAfold (Do et al., 2006), Con-
textfold (Zakov et al., 2011), and LinearFold (Huang et al.,
2019); learning-based folding methods such as SPOT-
RNA (Singh et al., 2019), Externafold (Wayment-Steele
et al., 2021), E2Efold (Chen et al., 2019), MXfold2 (Sato
et al., 2021), and UFold (Fu et al., 2022).
Metrics We evaluate the performance by precision, recall,
and F1 score, which are defined as:
Precision “ TP
TP ` FP , Recall “ TP
TP ` FN ,
F1 “ 2 Precision ¨ Recall
Precision ` Recall ,
(19)
where TP, FP, and FN denote true positive, false positive
and false negative, respectively.
Implementation details Following the same experimental
setting as (Fu et al., 2022), we train the model for 100 epochs
with the Adam optimizer. The learning rate is 0.001, and
the batch size is 1 for sequences with different lengths.
C. Discussion on Abnormal Samples
Although we have illustrated three hard constraints in 3.2,
there exist some abnormal samples that do not satisfy these
constraints in practice. We have analyzed the datasets used
in this paper and found that there are some abnormal sam-
ples in the testing set that do not meet these constraints. The
ratio of valid samples in each dataset is summarized in the
table below:
As shown in Table 8, RFold forces the validity to be
100.00%, while other methods like E2Efold only achieve
about 50.31%. RFold is more accurate than other methods
in reflecting the real situation.
Nevertheless, we provide a soft version of RFold to relax the
strict constraints. A possible solution to relax the rigid pro-
cedure is to add a checking mechanism before the Argmax
function in the inference. Specifically, if the confidence
given by the Softmax is low, we do not perform Argmax
and assign more base pairs. It can be implemented as the
following pseudo-code:
1 y_pred = row_col_softmax(y)
2 int_one = row_col_argmax(y_pred)
3
4 # get the confidence for each position
5 conf = y_pred * int_one
6 all_pos = conf > 0.0
7
8 # select reliable position
9 conf_pos = conf > thr1
10
11 # select unreliable position with the full
row and column
12 uncf_pos = get_unreliable_pos(all_pos,
conf_pos)
13
14 # assign "1" for the positions with the
confidence higher than thr2
15 # note that thr2 < thr1
16 y_pred[uncf_pos] = (y_pred[uncf_pos] > thr2
).float()
17 int_one[uncf_pos] = y_pred[uncf_pos]
We conduct experiments to compare the soft-RFold and the
original version of RFold in the RNAStralign dataset. The
results are summarized in the Table 15. It can be seen that
soft-RFold improves the recall metric by a small margin.
The minor improvement may be because the number of
abnormal samples is small. We then select those samples
that do not obey the three constraints to further analyze the
performance. The total number of such samples is 179. It
can be seen that soft-RFold can deal with abnormal samples
well. The improvement of the recall metric is more obvious.
13
Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching Perspective
Table 13. Comparison between RNA secondary structure prediction methods and RFold.
Method SPOT-RNA E2Efold UFold RFold
pre-processing pairwise concat pairwise concat hand-crafted seq2map attention
optimization approach ˆ unrolled algorithm unrolled algorithm bi-dimensional optimization
constraint (a) ˆ ✓ ✓ ✓
constraint (b) ˆ ✓ ✓ ✓
constraint (c) ˆ ˆ ˆ ✓
F1 score 0.711 0.821 0.915 0.977
Inference time 77.80 s 0.40 s 0.16 s 0.02 s
Table 14. The ratio of valid samples in the datasets.
Dataset RNAStralign ArchiveII bpRNA
Validity 93.05% 96.03% 96.51%
Table 15. The results of soft-RFold and RFold on the RNAStralign.
Method Precision Recall F1
RFold 0.981 0.973 0.977
soft-RFold 0.978 0.974 0.976
Table 16. The results of soft-RFold and RFold on the abnormal
samples on the RNAStralign.
Method Precision Recall F1
RFold 0.956 0.860 0.905
soft-RFold 0.949 0.889 0.918
D. Visualization
14
Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching PerspectiveRFoldTrue UFold
Figure 7. Visualization of the true and predicted structures.
15

