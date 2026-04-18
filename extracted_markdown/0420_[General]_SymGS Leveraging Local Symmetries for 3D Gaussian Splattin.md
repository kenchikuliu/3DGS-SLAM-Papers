<!-- page 1 -->
SymGS : Leveraging Local Symmetries for 3D Gaussian Splatting Compression
Keshav Gupta*1,3, Akshat Sanghvi*1, Shreyas Reddy Palley1, Astitva Srivastava1, Charu Sharma1,
Avinash Sharma2
1IIIT Hyderabad
2IIT Jodhpur
3University of California, San Diego
{keshav.gupta, akshat.sanghvi, shreyas.palley}@students.iiit.ac.in, astitva.s@research.iiit.ac.in, charu.sharma@iiit.ac.in,
avinashsharma@iitj.ac.in
Figure 1: Our SymGS leverages Reflective Symmetries in a 3DGS scene for compression while preserving rendering quality.
Abstract
3D Gaussian Splatting has emerged as a transformative tech-
nique in novel view synthesis, primarily due to its high ren-
dering speed and photorealistic fidelity. However, its memory
footprint scales rapidly with scene complexity, often reaching
several gigabytes. Existing methods address this issue by in-
troducing compression strategies that exploit primitive-level
redundancy through similarity detection and quantization. We
aim to surpass the compression limits of such methods by
incorporating symmetry-aware techniques, specifically tar-
geting mirror symmetries to eliminate redundant primitives.
We propose a novel compression framework, SymGS, intro-
ducing learnable mirrors into the scene, thereby eliminat-
ing local and global reflective redundancies for compression.
Our framework functions as a plug-and-play enhancement to
state-of-the-art compression methods, (e.g. HAC) to achieve
further compression. Compared to HAC, we achieve 1.66×
compression across benchmark datasets (upto 3× on large-
scale scenes). On an average, SymGS enables 108× com-
pression of a 3DGS scene, while preserving rendering qual-
ity. The project page and supplementary can be found at
symgs.github.io
Introduction
3D Gaussian Splatting (3DGS) (Kerbl et al. 2023a) has
emerged as a revolutionary technology for novel-view syn-
thesis, offering high rendering quality and speed. However,
achieving these high-quality scenes represented using 3DGS
*These authors contributed equally.
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
often require substantial memory, as rendering fidelity typ-
ically increases with the number of Gaussians. Large-scale
scenes frequently exceed several Gigabytes. Addressing this
memory overhead is critical to enabling a broader deploy-
ment of 3DGS, particularly in resource-constrained envi-
ronments such as robotics, web-based 3D applicatitons, and
real-time game engines, where storage and runtime effi-
ciency are essential. Despite recent advances, many ap-
proaches still generate a large number of redundant Gaus-
sians tailored to individual views. Scaffold-GS (Lu et al.
2024) mitigates this by introducing a sparse set of anchor
points, each parameterizing a fixed local Gaussian ensem-
ble via a learned feature vector. This formulation enables
compact scene representation while preserving fidelity, fur-
ther enhanced through anchor growing and pruning strate-
gies. HAC (Chen et al. 2024) extends this framework by
identifying and compressing redundancies in anchor fea-
tures. It leverages a hash grid to learn value distributions
across attributes, followed by an adaptive quantization step
for discretization. Nevertheless, compression performance
saturates as residual redundancies in anchor attributes re-
main insufficiently exploited. Additionally, there is a lack of
interpretability in the aforementioned techniques, as it is dif-
ficult to visually understand the rationale behind the removal
or quantization of specific Gaussian primitives. Therefore, a
new axis needs to be explored for compression of 3DGS.
Many real-world objects like desks and bottles often con-
tain inherent local reflective symmetries regions where one
part is an approximate mirror of another. We observe that
exploiting these symmetries can significantly reduce redun-
dancy, lower storage costs, and produce more structured and
arXiv:2511.13264v2  [cs.CV]  19 Nov 2025

<!-- page 2 -->
interpretable scene representations. However, detecting such
symmetries in a 3D Gaussian Splatting (3DGS) represen-
tation is challenging. 3DGS often suffers from multi-view
noise, view-dependent shading artifacts, and uneven Gaus-
sian densities caused by sparse or biased camera trajecto-
ries. In addition, most symmetries in real-world scenes are
only approximate or partial, and the space of potential mir-
ror planes is large, making reliable detection non-trivial.
To this end, we introduce SymGS, a novel method for
compressing 3DGS by discovering and leveraging reflec-
tive symmetries. We first estimate dominant mirror planes
using a symmetry detection algorithm tailored specifically
for 3D Gaussians. Once a mirror plane is identified, Gaus-
sians on one side are retained, while their counterparts are
reflected across the plane and the originals discarded. The
mirror plane and the retained Gaussians are then jointly op-
timized via differentiable splatting to minimize photomet-
ric loss to maintain visual fidelity. We apply this procedure
recursively, re-running symmetry detection on the remain-
ing Gaussians after each compression step to progressively
uncover additional symmetric structures in a global-to-local
fashion. As illustrated in Fig. 1, our proposed hierarchi-
cal optimization strategy enables 108× compression of the
3DGS scene while maintaining rendering quality, while also
yielding a more interpretable scene decomposition. Addi-
tionally, we demonstrate the plug-and-play capability of our
framework by integrating it into the current SOTA 3DGS
compression method, HAC. We achieve superior compres-
sion rates across standard publicly available datasets, rang-
ing from small single-object scenes to large-scale environ-
ments. In summary, our contributions are as follows:
• We introduce a novel way of compressing 3DGS scenes
by utilizing reflective symmetries in a scene to reduce the
number of Gaussians.
• We design a CUDA-accelerated, grid-based symmetry
detection algorithm tailored for 3DGS, enabling efficient
detection of local symmetries on modern GPUs.
• We improve upon the storage efficiency of HAC by in-
corporating the proposed symmetry-aware framework as
a plugin module, achieving SOTA compression rate.
Figure 2: PSNR vs RCF for 3DGS compression.
Related Works
3DGS Compression
Many works have tried to address the problem of large stor-
age space required by 3DGS scenes. LightGaussian (Fan
et al. 2024) applies vector quantization (VQ) to selec-
tively compress SH coefficients guided by a significance
score, while RDO-Gaussian (Xie et al. 2024) employs rate-
distortion optimized quantization and entropy coding. Com-
pact3D (Zhang et al. 2024) proposes sensitivity-aware VQ
and entropy modeling. Compact3DGS (Smith et al. 2024)
replaces SH colors with hash-grid-MLP decoding, while
Self-Organizing Gaussians (Morgenstern et al. 2024) ar-
range primitives on 2D grids to exploit local smoothness
for image-like compression. EAGLES (Lee et al. 2024b) en-
codes attributes as latent vectors decoded by MLPs, main-
taining compactness while preserving quality. ContextGS
(Zhou et al. 2024) builds on Scaffold-GS (Lu et al. 2024)
by hierarchically predicting anchors across levels. Methods
such as Octree-GS (Ren et al. 2024) utilize depth maps
and level-of-detail octrees for compaction. Among these,
HAC (Chen et al. 2024) stands out by integrating anchor-
based representation, hash-grid assisted context modeling,
and adaptive quantization to achieve superior compression
ratios and rendered quality across diverse datasets. Recently,
HAC++ (Chen et al. 2025) extends HAC by capturing intra-
anchor contextual relationships to further enhance compres-
sion performance.
3D Symmetry Detection
Symmetry detection in 3D point clouds is a fundamental
problem with applications in geometry processing, segmen-
tation, reconstruction, and compression. A seminal work
by Mitra et al. (Mitra, Guibas, and Pauly 2006) introduced
partial symmetry detection in point clouds via a Hough-
transform-like voting scheme, where points with similar lo-
cal structure contribute to a 7D accumulator representing
rotation and translation parameters. While effective, this
method is limited to point sets and suffers from discretiza-
tion artifacts in mirror estimation. Other methods have ap-
proached this task through model-driven algorithms that
detect global or partial symmetries using descriptors like
generalized moments (Martinet et al. 2006), planar reflec-
tive transforms (Podolak et al. 2006), and symmetry fac-
tored embeddings (Lipman et al. 2010). These methods of-
ten assume clean geometry and are sensitive to noise or
partial data. To overcome such limitations, learning-based
approaches have emerged. PRS-Net (Gao et al. 2020) and
E3Sym (Li et al. 2023) employ supervised and unsupervised
deep networks, respectively, to detect global planar symme-
tries. The most recent progress in partial extrinsic symmetry
detection is SymCL and SymML (Kobsik, Lim, and Kobbelt
2023), which leverage geodesic patches and contrastive
learning to produce transformation-invariant features with-
out requiring labels. Despite such progress, no work to date
incorporates symmetry priors into 3DGS. Given the inher-
ent symmetry in most man-made environments (Mitra et al.
2013), symmetry-aware compression holds the potential to
significantly improve efficiency and interpretability. This

<!-- page 3 -->
presents an open and impactful direction for extending sym-
metry detection methods to 3DGS.
Methodology
Our aim is to estimate mirror planes to model reflective sym-
metries at different scales and leverage them for compress-
ing the 3DGS scene. We achieve this through an iterative
process, as illustrated in Fig. 4. First, we detect the most
dominant symmetry in the 3DGS scene. Inspired by (Mi-
tra, Guibas, and Pauly 2006), we define a 3D accumula-
tor grid - a discrete parameter space where each cell cor-
responds to a candidate mirror plane. Every possible pair
of similar-looking Gaussians vote for the plane that lies be-
tween them, and votes from all pairs are accumulated in the
grid. The cell with the largest number of votes corresponds
to the most prominent symmetry present in the scene. We
start with the highest voted symmetry as initial mirror, M0.
Gaussians on one side of M0 are discarded, and the others
are reflected across it. This set of Gaussians (original and re-
flected) is then jointly optimized with the mirror M0 via the
rendering loss. Thus, the process alternates between Symme-
try Detection and Mirror-Aware Optimization. This is ap-
plied recursively to the remaining set of Gaussians, discover-
ing additional mirror planes M1, M2, M3, . . . , progressively
compressing the scene. Fig. 3 shows the illustration of the
SymGS framework.
Symmetry Detection for 3DGS
Given the initial set of Gaussians Ginit, we aim to estimate
the most dominant mirror plane, via a clustering-based
voting strategy explained as follows:
3D Gaussian Clustering: For symmetry detection, we
require clusters of similar-looking Gaussians to avoid
spurious symmetry detection. Hence, it is crucial to
ensure that only visually and geometrically similar Gaus-
sians are paired together. Therefore, all the Gaussians
with attributes in the same range are clustered together.
Specifically, for a Gaussian Gi we define clustering
based on color ci, opacity oi & scale si. We transform
color ci in the HSV space and discretize each channel
(H/S/V) into Nc bins. Opacity oi is divided into No
bins, while scale si is discretized into Ns bins uniformly
across the x, y, and z axes. This results in a total of
NcH ×NcS ×NcV ×Nc ×No ×Nsx ×Nsy ×Nsz = Nclstr
clusters. A Gaussian Gi belonging to a cluster Ck, will be
paired with every other Gaussian Gj ∈Ck.
Mirror Parametrization: Any mirror or plane of re-
flective symmetry can be characterized by the general
equation of a 3D plane, i.e. ax + by + cz = 1. However,
the parameters a, b, and c are individually unbounded
for a given scene, making exhaustive search in this space
impractical. To define a finite and tractable set of candidate
mirrors, we reparameterize the plane using a bounded set
of parameters (α, β, γ). Specifically, we express the plane
normal in spherical coordinates with angular parameters
α ∈[0, π] and β ∈[0, 2π], and define the signed distance
from the origin as γ ∈[0, extent], where extent denotes the
radius of the scene. While this polar representation bounds
the mirror plane parameter space, it remains continuous
and infinitely dense. To make symmetry detection compu-
tationally feasible, we discretize this space into a uniform
3D accumulator voxel-grid A, defined over (α, β, γ) with
dimension dα × dβ × dγ, where:
dα = ⌊π
αres
⌋,
dβ = ⌊2π
βres
⌋,
dγ = ⌊extent
γres
⌋
and, αres, βres, γres are voxel size along the three axes re-
spectively. Each voxel v ∈A represents a possible mirror in
the 3D scene, and its integer value denotes how many Gaus-
sian pairs are associated with that mirror.
For a pair of Gaussians (Gi, Gj) ∈Ck, and their respec-
tive mean 3D positions xi and xj, we initialize a mirror
plane Mij passing between them, with center c and normal
n given as:
c = (xi + xj)
2
,
n = (xi −xj)
|(xi −xj)|
We compute the parameters (α, β, γ) for Mij as:
α = cos−1nx,
β = π + tan−1 ny
nx
,
γ = n · c
The 3D index of the voxel vij ∈A representing mirror Mij
is then computed as:
vα
ij = ⌊α
αres
⌋,
vβ
ij = ⌊β
βres
⌋,
vγ
ij = ⌊γ
γres
⌋.
All the voxels of the accumulator grid are initialized as zero,
i.e. A[vα
ij][vβ
ij][vγ
ij] = 0 for each vij ∈A.
Grid-based Mirror Voting: Initially, all possible pairs of
similar Gaussians (Gi, Gj) ∈Ck define a set of candidate
mirrors, out of which, the one exhibiting the most dominant
reflective symmetry is selected. A vote is cast by a pair of
similar Gaussians for the mirror plane that best explains
their reflective symmetry. In other words, for the pair
of Gaussians (Gi, Gj), the value of the voxel vij, i.e.
A[vα
ij][vβ
ij][vγ
ij] is incremented by one. After accumulating
votes from all possible Gaussian pairs for all clusters into
the common A, the voxel with the highest vote count repre-
sents the most dominant mirror plane, M0. The dominant
mirror plane M0, divides the symmetric subpart of the
scene into two halves, GM0
left (in the direction of normal)
and GM0
right (opposite to the normal direction), consisting
of all the Gaussian pairs that contributed to its votes. The
remaining pairs of Gaussians which do not vote for the
mirror M0 are outside of its influence, denoted as GM0
out.
The symmetry detection algorithm, with O(n2) com-
plexity, is critical for runtime efficiency. To simultaneously
handle millions of Gaussians in large scenes, we design a
CUDA-accelerated variant, reducing the time to identify the
largest mirror from tens of minutes to around 10 seconds for
roughly 1 million Gaussians. All the pairs of 3D Gaussians

<!-- page 4 -->
Figure 3: SymGS Framework: Given a 3DGS scene, we first perform Gaussians clustering and voting to identify the dominant
mirror symmetry. Then, the Gaussians on one side of the mirror are replaced with reflections of their counterparts. Finally, the
modified Gaussian set and mirror parameters are jointly optimized using the standard photometric loss.
Figure 4: Iterative Symmetry Detection & Optimization.
are assigned individual GPU threads, parallely casting their
votes to the shared accumulator grid in an atomic operation.
Please refer to the supplementary for timing analysis for
different implementation choices.
Mirror-Aware Optimization
After the initial estimation of M0, we discard GM0
right and
replace it by reflecting the Gaussians GM0
left. However, this
leads to a significant degradation in rendering quality (see
Fig. 5). This is primarily due to the finite number of thresh-
olds involved in the clustering process, and the discrete ap-
proximation of true mirror parameters owing to the finite
resolution of the accumulator grid.
To account for such inconsistencies, we propose to make
the mirror plane optimizable by allowing its normal and cen-
ter to adjust during training. Given the parameters of M0 and
the attributes of Gaussians GM0
left, during forward pass we
construct the scene by first reflecting the GM0
left across M0 to
generate the symmetrical part of the scene, and then adding
the rest of the Gaussians GM0
out into the scene. The reflection
involves mirroring the position of GM0
left about M0, while
copying the scale and opacity as it is. The final set of Gaus-
sians are then passed to the differential 3DGS rasterizer to
obtain the rendered image as per viewing direction. Finally,
the rendered image is compared with the ground truth (in-
put) using the standard 3DGS rendering loss (Kerbl et al.
2023b). During the backwards pass, the gradient updates are
applied to attributes of GM0
left & GM0
out, and to the parameters
of M0 as well. This joint-optimization of the mirror plane
and Gaussian attributes makes it possible to refine the sym-
metry that best aligns with the corresponding sub-structure
of the scene, while correcting for discretization errors and
inconsistencies involved in initial mirror estimates.
Compression via Reflection
By symmetric duplication of Gaussians GM0
left, discarding
Gaussians GM0
right, we effectively halve the number of prim-
itives that are required to represent the symmetric part of
the scene corresponding to M0. Meanwhile, the remain-
ing Gaussians GM0
out, i.e. those excluded from M0, capture
within it, other non-dominant symmetries, along with asym-
metric or non-redundant details, such as occlusions, clutter,
or scene-specific irregularities. Only the mean positions of
one half of the symmetric Gaussians, i.e. XM0
left and mirror
parameters (α, β, γ) need to be stored, while the remaining
subpart is reconstructed on-the-fly at training time. Please
refer to Fig. 6 for a visual depiction of the reconstruction
process. This reduces the overall storage requirements.
Iterative Compression
After optimizing the scene for the first mirror M0, we fo-
cus on identifying symmetries in the remaining scene, i.e.
GM0
left and GM0
out. More generally, for any subsequent it-

<!-- page 5 -->
Figure 5: Mirror-aware optimization restores the visual details which might get attenuated while reflecting the Gaussians.
Figure 6: At each iteration step, the Gaussians in GMstep
left
replace their symmetrical counterparts in GMstep
right . The full scene is
reconstructed by iteratively reflecting these Gaussians across all previous mirror levels in succession.
eration ‘step’, the subpart of the scene to be considered
for further compression consists of Gaussian set GMstep
= {GMstep−1
left
, GMstep−1
out
} from previous iteration. As illus-
trated in Fig. 4, we follow the same process from Gaus-
sian clustering to mirror-aware optimization, considering
only GMstep, resulting into the current most dominant, op-
timized symmetry plane Mstep. Thus, this iterative pro-
cess from initial step to last step yields a set of mirrors
{M0, M1, M2, . . . , ML}, progressively discovering similar-
ities till level L. At every iteration, we only store mean po-
sitions XMstep
left
& parameters of Mstep, discarding GMstep
right
altogether. For mirror-aware optimization at step L, we re-
construct the scene from the current level. We load attributes
of GML
left and GML
out , reflect the GML
left about ML to obtain
bGML
right (as it is not stored). For step L −1, we obtain
GML−1
left+out = {GML
left, bGML
right, GML
out }. We load already stored
mean positions XML−1
left
and perform nearest neighbor query
search within GML−1
left+out, to obtain GML−1
left
. We then esti-
mate GML−1
out
= GML−1
left+out −GML−1
left
, and further reflect
GML−1
left
to obtain bGML−1
right . We follow the same process re-
cursively for further lower levels, continuing this reconstruc-
tion to obtain the complete scene, which is represented by
GM0
left+out = {GM0
left, bGM0
right, GM0
out}. We then apply mirror-
aware optimization, tuning only the mirror ML. The afore-
mentioned hierarchical compression strategy, where each
mirror Mstep incrementally reduces the number of Gaus-
sians, achieves coarse-to-fine structure-aware compression
across the full scene. Finally, we store only XMl
left and mir-
ror parameters Ml for each level l (denoted collectively as
Xret and M respectively), as well as the last level attributes
GML
left and GML
out . Please find pseudocode in supplementary
Integration with HAC
Our framework for finding and exploiting reflective sym-
metries to reduce the number of primitives can be applied
to other existing compression methods. To showcase this,
we build on top of HAC (Chen et al. 2024), a recent SOTA
in 3DGS compression, integrating symmetry detection and
progressive compression within its pipeline. HAC achieves
good compression rate mainly through its adaptive quanti-
zation module and a hash-grid to estimate the quantization
steps for the anchor attributes. We apply the propsed itera-
tive symmetry detection and optimization process over these
anchors. For a given 3DGS scene, we first run HAC to esti-
mate anchor attributes, and then compute the average color,
opacity, and scale for each anchor by aggregating over its
k associated Gaussians across all views. These aggregated
attributes are then used to cluster the anchors, which is anal-
ogous to the Gaussian clustering process. We then perform
grid-based voting to identify the most dominant mirror plane
at all levels. For each level, we then optimize the mirror
and anchor features, hash-grid, and all the MLPs involved in
HAC. We then iteratively compress the scene by following
the same process described in the previous section. Please
refer to the supplementary for more details.
Experiments & Results
Datasets:
We evaluate our method on the five standard
benchmarking datasets as in HAC, conducting experiments
across a total of 27 scenes: 8 from Synthetic-NeRF (Milden-
hall et al. 2021), all 9 scenes from Mip-NeRF360 (Barron
et al. 2022), 2 from Tanks & Temples (Knapitsch et al.
2017), 2 from DeepBlending (Hedman et al. 2018), and
6 from BungeeNeRF (Xiangli et al. 2023). This selec-
tion allows us to demonstrate compression performance on
both large-scale scenes, such as those in BungeeNeRF, and
smaller, object-centric scenes, such as those in Synthetic-

<!-- page 6 -->
Table 1: SymGS achieves higher Relative Compression Rate (RCF) w.r.t. 3DGS, while maintaining the comparable PSNR Red
and yellow highlight respectively denote the best and second best metrics among the compression methods.
Datasets
Synthetic-NeRF
Mip-NeRF360
Tank & Temples
Deep Blending
BungeeNerf
Methods
PSNR ↑
Size ↓
RCF ↑
PSNR ↑
Size ↓
RCF ↑
PSNR ↑
Size ↓
RCF ↑
PSNR ↑
Size ↓
RCF ↑
PSNR ↑
Size ↓
RCF ↑
3DGS
33.80
68.46
1.00×
27.49
744.7
1.00×
23.69
431.0
1.00×
29.42
663.9
1.00×
24.87
1616
1.00×
Lee et al.
33.33
5.54
12.35×
27.08
48.80
15.26×
23.32
39.43
10.93×
29.79
43.21
15.36×
23.36
82.60
19.56×
Compressed3D
32.94
3.68
18.60×
26.98
28.80
25.85×
23.32
17.28
24.94×
29.38
25.30
26.24×
24.13
55.79
28.96×
EAGLES
32.54
5.74
11.92×
27.15
68.89
10.80×
23.41
34.00
12.67×
29.91
62.00
10.71×
25.24
117.1
13.80×
Light Gaussian
32.73
7.84
8.73×
27.00
44.54
16.71×
22.83
22.43
19.21×
27.01
33.94
19.56×
24.52
87.28
18.51×
SOG
31.05
2.20
31.11×
26.01
23.90
31.15×
22.78
13.05
33.02×
28.92
8.40
79.03×
-
-
-
CompGS
33.09
4.42
15.48×
27.16
50.30
14.80×
23.47
27.97
15.40×
29.75
42.77
15.52×
24.63
104.3
15.49×
Scaffold-GS
33.41
19.36
3.53×
27.50
253.9
2.93×
23.96
86.50
4.98×
30.21
66.00
10.05×
26.62
183.0
8.83×
HAC
33.05
1.38
49.60×
27.29
16.66
44.69×
24.14
8.07
53.40×
29.85
7.80
85.11×
26.44
20.28
79.68×
SymGS (Ours)
33.19
0.95
72.06×
27.29
11.74
63.43×
24.02
7.14
60.36×
29.99
2.59
256.33×
26.24
10.29
157.05×
Figure 7: Qualitative Comparison with HAC, Scaffold-GS and 3DGS.
NeRF. Collectively, these datasets span a diverse range of
indoor and outdoor environments.
Implementation & Training Details:
We build upon the
HAC (Chen et al. 2024) framework written in PyTorch,
trained on an NVIDIA RTX A6000 GPU. We keep the mir-
ror parameters α, β and d trainable with learning rates 1e−3
each. We choose the distance spacing γres according to the
extent of the scene. For Synthetic-NeRF and BungeeNeRF,
γres = 0.01 and for MIP, TandT and DeepBlending, we take
γres = 0.1. αres and βres are fixed to 0.01.
Comparisons
We compare with 3DGS (Kerbl et al. 2023b) and other exist-
ing compression methods (Lee et al. 2024a), Compressed3D
(Niedermayr, Stumpfegger, and Westermann 2024), EA-
GLES (Girish, Gupta, and Shrivastava 2024), Light Gaus-
sian (Fan et al. 2025), Self-Organizing Gaussian Grids
(SOG) (Morgenstern et al. 2025), CompGS (Navaneet et al.
2024), Scaffold-GS (Lu et al. 2024) and HAC (Chen et al.
2024). We use HAC as our primary baseline, and extend it
with our symmetry detection and mirror-aware optimization.
We report the final compressed model size (in MB) along
with PSNR, evaluated after decoding, comparing with exist-
ing SOTA methods in Table 1. Across the five benchmark
datasets, our approach achieves an average Relative Com-
pression Factor (RCF) of 108× compared to 3DGS. In com-
parison to current SOTA HAC, we achieve 1.66× compres-
sion on an average. Specifically, we observe 3.01× com-
pression on Deep Blending, 1.97× on BungeeNeRF, 1.4×
on Mip-NeRF360, 1.4× on Synthetic-NeRF, and 1.1× on
Tanks and Temples, while maintaining visual fidelity.
We show qualitative comparison of SymGS with other meth-
ods in Fig. 7. We highlight the detail-preserving com-
pression capability of SymGS, which are often lost by
quantization-based approaches.
Ablations
Effect of Accumulator Grid Resolution:
As previously
mentioned, the dimensionality of the Voxel Grid A, is de-

<!-- page 7 -->
pendent on the radius or extent of the scene, parametrized
by γres, which controls the level of discretization along this
axis. We analyze the effect of γres by varying it along a
range based on the extent of the scene - 0.01 and 0.1 for
Nerf-Synthetic scenes due to their small extent, and 0.1 to 1
for MIP scenes having a larger extent. A lower value of γres
corresponds to a more fine-grained discretization, and vice
versa. A coarse discretization (large value of γres) leads to
a higher number of incorrect votes between anchors during
symmetry detection, resulting in a lower PSNR. This also
leads to incorrect pair of Gaussians voting for mirror candi-
dates, leading to poor symmetry detection, and thereby de-
creasing the compression rate. This trend can be seen in Fig.
8. Please note that we don’t ablate the effect of αres & βres
as dα they dβ do not model the extent of the scene, but only
the orientation of the mirrors.
Figure 8: Effect of γres on the PSNR (db) storage size (MB).
Effect of Hash-Grid-MLP:
In HAC, the usage of Hash-
grid MLP for adaptive quantization for the anchor attributes
enable anchors to be better aligned to each scene, leading to
a better compression rate. Therefore, we ablate the effect of
fine-tuning the Hash-grid MLP during training. As shown in
Table 2, the rendering quality increases when finetuned.
Table 2: Ablation Study: Effect of Hash Grid MLP and
Training Attributes across multiple mirrors. Values are
shown in PSNR/Size (in MB) format.
Scene
w/o MLP
(L=5) + MLP
(L=1) + MLP
Chair
33.88 / 0.93
34.60 / 0.82
34.34 / 0.80
Drums
26.13 / 1.73
26.40 / 1.62
26.37 / 1.55
Room
31.26 / 3.92
31.92 / 4.15
31.70 / 3.84
Optimizing multiple mirrors at once:
Primarily, the only
anchor parameters affected by introducing a new mirror be-
long to the last level, i.e. GML
left and GML
out . Thus, we only
train these parameters. However, letting the previous level
mirror parameters to update allows lower levels to adjust to
the newly added mirror as well. Thus, instead of training
only the anchors participating in the most recent mirror, we
train the anchors that took part in the last 5 mirror levels.
This effect can be seen in Table 2, showcasing higher PSNR
values, but empirically lower compression rates.
Discussion
SymGS achieves superior compression gains by carefully
exploiting several reflective symmetries present in a 3DGS
scene (as shown in Fig. 9). High compression rates on
BungeeNeRF dataset can be attributed to the prevalence of
large-scale outdoor scenes containing numerous man-made,
symmetrical structures. Similarly, Deep Blending contains
indoor scenes with recurring symmetric patterns across
walls, floors, and household objects, enabling further com-
pression. The relatively low compression gain on Synthetic-
NeRF is due to the fact that these scenes are already small
in scale and exhibit limited geometric and appearance com-
plexity, making them inherently more compact and less re-
dundant. We also observe an increase in PSNR values for
Synthetic-NeRF and Deep Blending. This suggests that en-
forcing structural symmetry not only reduces redundancy
but also enhances rendering fidelity, especially in inherently
symmetric scenes such as Lego (in Synthetic-NeRF) and Dr-
Johnson (in Deep Blending). Notably, the rendering FPS re-
mains the same as HAC, as all the mirror decompositions
are pre-computed, and no additional operations are required
during rendering, making our framework highly practical.
Figure 9: Detected reflective symmetries in 3DGS scenes.
Conclusion
We present a novel framework for compressing 3DGS
scenes by exploiting inherent reflectional symmetries found
in real-world environments. By introducing a CUDA-
accelerated, grid-based symmetry detection algorithm for
3DGS, we enable efficient identification and utilization of
local mirror symmetries to significantly reduce redundancy
in Gaussian representations. Our hierarchical optimization
approach recursively uncovers and compresses symmetric
structures, achieving up to 108× compression while pre-
serving high rendering fidelity. Furthermore, we demon-
strated the versatility of SymGS by integrating it as a
plug-and-play module into the state-of-the-art HAC com-
pression method, resulting in superior compression rates
across diverse datasets. This symmetry-aware compression
not only improves storage efficiency but also yields more
interpretable and structured scene decompositions.

<!-- page 8 -->
References
Barron, J. T.; Mildenhall, B.; Verbin, D.; Srinivasan, P. P.;
and Hedman, P. 2022. Mip-NeRF 360: Unbounded Anti-
Aliased Neural Radiance Fields.
In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 5470–5479.
Chen, Y.; Wu, Q.; Lin, W.; Harandi, M.; and Cai, J. 2024.
HAC: Hash-grid Assisted Context for 3D Gaussian Splatting
Compression. In European Conference on Computer Vision.
Chen, Y.; Wu, Q.; Lin, W.; Harandi, M.; and Cai, J. 2025.
Hac++: Towards 100x compression of 3d gaussian splatting.
arXiv preprint arXiv:2501.12255.
Fan, L.; et al. 2024. LightGaussian: Memory-Efficient 3D
Gaussian Splatting with Pseudo-View Knowledge Distilla-
tion. ArXiv preprint arXiv:2403.XXXX.
Fan, Z.; Wang, K.; Wen, K.; Zhu, Z.; Xu, D.; Wang, Z.; et al.
2025. Lightgaussian: Unbounded 3d gaussian compression
with 15x reduction and 200+ fps. Advances in neural infor-
mation processing systems, 37: 140138–140158.
Gao, L.; Zhang, L.-X.; Meng, H.-Y.; Ren, Y.-H.; Lai, Y.-K.;
and Kobbelt, L. 2020. PRS-Net: Planar Reflective Symme-
try Detection Net for 3D Models.
IEEE Transactions on
Visualization and Computer Graphics, 27(6): 3007–3018.
Girish, S.; Gupta, K.; and Shrivastava, A. 2024.
Eagles:
Efficient accelerated 3d gaussians with lightweight encod-
ings. In European Conference on Computer Vision, 54–71.
Springer.
Hedman, P.; Philip, J.; Price, T.; Frahm, J.-M.; Drettakis, G.;
and Brostow, G. 2018. Deep Blending for Free-viewpoint
Image-based Rendering. 37(6): 257:1–257:15.
Kerbl, B.; Kopanas, G.; Leimk¨uhler, T.; and Drettakis, G.
2023a. 3D Gaussian Splatting for Real-Time Radiance Field
Rendering. ACM Transactions on Graphics (TOG).
Kerbl, B.; Kopanas, G.; Leimk¨uhler, T.; and Drettakis, G.
2023b. 3d gaussian splatting for real-time radiance field ren-
dering. ACM Trans. Graph., 42(4): 139–1.
Knapitsch, A.; Park, J.; Zhou, Q.-Y.; and Koltun, V. 2017.
Tanks and Temples: Benchmarking Large-Scale Scene Re-
construction. ACM Transactions on Graphics, 36(4).
Kobsik, G.; Lim, I.; and Kobbelt, L. 2023.
Partial Sym-
metry Detection for 3D Geometry using Contrastive Learn-
ing with Geodesic Point Cloud Patches.
arXiv preprint
arXiv:2312.08230.
Lee, J. C.; Rho, D.; Sun, X.; Ko, J. H.; and Park, E. 2024a.
Compact 3d gaussian representation for radiance field. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, 21719–21728.
Lee, J. H.; et al. 2024b. EAGLES: Efficient Attribute Gaus-
sian Latent Encoding with Splatting. ArXiv:2404.XXXX.
Li, R.-W.; Zhang, L.-X.; Li, C.; Lai, Y.-K.; and Gao, L.
2023. E3Sym: Leveraging E(3) invariance for unsupervised
3D planar reflective symmetry detection. In Proceedings of
the IEEE/CVF International Conference on Computer Vi-
sion, 14543–14553.
Lipman, Y.; Chen, X.; Daubechies, I.; and Funkhouser, T.
2010. Symmetry factored embedding and distance. ACM
Transactions on Graphics (TOG), 29(4): 1–12.
Lu, T.; Yu, M.; Xu, L.; Xiangli, Y.; Wang, L.; Lin, D.;
and Dai, B. 2024. Scaffold-gs: Structured 3d gaussians for
view-adaptive rendering. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
20654–20664.
Martinet, A.; Soler, C.; Holzschuch, N.; and Sillion, F. X.
2006. Accurate detection of symmetries in 3D shapes. ACM
Transactions on Graphics (TOG), 25(2): 439–464.
Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.;
Ramamoorthi, R.; and Ng, R. 2021.
Nerf: Representing
scenes as neural radiance fields for view synthesis. Com-
munications of the ACM, 65(1): 99–106.
Mitra, N. J.; Guibas, L. J.; and Pauly, M. 2006. Partial and
approximate symmetry detection for 3d geometry.
ACM
Transactions on Graphics (ToG), 25(3): 560–568.
Mitra, N. J.; Pauly, M.; Wand, M.; and Ceylan, D. 2013.
Symmetry in 3D geometry: Extraction and applications.
Computer Graphics Forum, 32(6): 1–23.
Morgenstern, W.; Barthel, F.; Hilsmann, A.; and Eisert,
P. 2025.
Compact 3D Scene Representation via Self-
Organizing Gaussian Grids. In Computer Vision – ECCV
2024, 18–34. Cham: Springer Nature Switzerland.
Morgenstern, W.; et al. 2024. Self-Organizing Gaussians for
3DGS Compression. CVPR.
Navaneet, K.; Pourahmadi Meibodi, K.; Abbasi Kooh-
payegani, S.; and Pirsiavash, H. 2024. Compgs: Smaller and
faster gaussian splatting with vector quantization. In Euro-
pean Conference on Computer Vision, 330–349. Springer.
Niedermayr, S.; Stumpfegger, J.; and Westermann, R. 2024.
Compressed 3d gaussian splatting for accelerated novel view
synthesis. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 10349–10358.
Podolak, J.; Shilane, P.; Golovinskiy, A.; Rusinkiewicz, S.;
and Funkhouser, T. 2006.
A planar-reflective symmetry
transform for 3D shapes. In ACM Transactions on Graphics
(TOG), volume 25, 549–559.
Ren, K.; Jiang, L.; Lu, T.; Yu, M.; Xu, L.; Ni, Z.; and Dai, B.
2024. Octree-GS: Towards Consistent Real-time Rendering
with LOD-Structured 3D Gaussians. arXiv:2403.17898.
Smith, A.; et al. 2024. Compact3DGS: Efficient Color En-
coding for 3DGS. ArXiv:2403.XXXX.
Xiangli, Y.; Xu, L.; Pan, X.; Zhao, N.; Rao, A.; Theobalt, C.;
Dai, B.; and Lin, D. 2023. BungeeNeRF: Progressive Neural
Radiance Field for Extreme Multi-scale Scene Rendering.
arXiv:2112.05504.
Xie, Q.; et al. 2024.
RDO-Gaussian: Rate-Distortion
Optimized 3D Gaussian Splatting.
ArXiv preprint
arXiv:2401.XXXX.
Zhang, X.; et al. 2024. Compact3D: Efficient Compression
of 3D Gaussian Splatting using Sensitivity-Aware VQ. arXiv
preprint arXiv:2402.15423.
Zhou, L.; et al. 2024. ContextGS: Hierarchical Compression
for 3D Gaussian Splatting. CVPR.
