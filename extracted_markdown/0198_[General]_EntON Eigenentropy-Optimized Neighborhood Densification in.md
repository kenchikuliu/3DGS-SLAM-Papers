<!-- page 1 -->
EntON: Eigenentropy-Optimized Neighborhood
Densification in 3D Gaussian Splatting
Miriam Jägera,∗, Boris Jutzia
aInstitute of Photogrammetry and Remote Sensing (IPF), Karlsruhe Institute of
Technology (KIT), Karlsruhe, Germany
Abstract
We present a novel Eigenentropy-optimized neighboorhood densification strat-
egy EntON in 3D Gaussian Splatting (3DGS) for geometrically accurate and
high-quality rendered 3D reconstruction. While standard 3DGS produces
Gaussians whose centers and surfaces are poorly aligned with the underlying
object geometry, surface-focused reconstruction methods frequently sacrifice
photometric accuracy. In contrast to the conventional densification strategy,
which relies on the magnitude of the view-space position gradient, our ap-
proach introduces a geometry-aware strategy to guide adaptive splitting and
pruning. Specifically, we compute the 3D shape feature Eigenentropy from
the eigenvalues of the covariance matrix in the k-nearest neighborhood of
each Gaussian center, which quantifies the local structural order. These Eige-
nentropy values are integrated into an alternating optimization framework:
During the optimization process, the algorithm alternates between (i) stan-
dard gradient-based densification, which refines regions via view-space gradi-
ents, and (ii) Eigenentropy-aware densification, which preferentially densifies
Gaussians in low-Eigenentropy (ordered, flat) neighborhoods to better cap-
ture fine geometric details on the object surface, and prunes those in high-
Eigenentropy (disordered, spherical) regions. We provide quantitative and
qualitative evaluations on two benchmark datasets: small-scale DTU dataset
and large-scale TUM2TWIN dataset, covering man-made objects and urban
scenes. Experiments demonstrate that our Eigenentropy-aware alternating
densification strategy improves geometric accuracy by up to 33% and ren-
dering quality by up to 7%, while reducing the number of Gaussians by up
∗Corresponding author.
Email address: miriam.jaeger@kit.edu (Miriam Jäger)
arXiv:2603.06216v1  [cs.CV]  6 Mar 2026

<!-- page 2 -->
to 50% and training time by up to 23%. Overall, EnTON achieves a favor-
able balance between geometric accuracy, rendering quality and efficiency by
avoiding unnecessary scene expansion.
Keywords:
3D Gaussian Splatting, 3D Reconstruction, Densification,
Features, Eigenvalues, Eigenentropy
1. Introduction
Recent advances in 3D scene reconstruction have been largely driven by
the introduction of Neural Radiance Fields (NeRFs) [32], which demonstrated
impressive photorealistic rendering quality by learning continuous volumet-
ric scene representations. Building on this progress, 3D Gaussian Splatting
(3DGS) has emerged as an explicit scene representation that enables real-time
rendering while maintaining competitive visual fidelity. Instead of implicitly
encoding the scene in a neural network, 3DGS explicitly represents the scene
where geometry is present by using a set of 3D Gaussians, each parameterized
by mean (center), scale, rotation, color, and opacity. During the optimiza-
tion process, the Gaussians are refined and adapted as they grow, shrink,
and adjust in their color and opacity, to minimize the photometric error.
Despite its strong rendering performance, 3DGS can suffer from over-
reconstruction issues [49] or oversized Gaussians [23], particularly in scenes
containing high-frequency geometric details. Such effects often lead to blurred
areas in the rendered image, highlighting the importance of splitting big
Gaussians to allow good reconstruction [21]. Moreover, it is evident that nei-
ther the centers nor the surfaces of Gaussians in 3DGS directly correspond
to object surfaces, which makes their direct use for point cloud and mesh
reconstruction impractical. Although several splatting-based surface recon-
struction approaches have demonstrated high geometric accuracy, they can
involve a trade-off between image quality and geometry [15], underscoring
the need for densification strategies that are guided by the underlying 3D
geometric structure of the scene.
The standard 3DGS densification strategy relies primarily on view-space
position gradients and does not explicitly account for geometric context in-
formation. As a result, the potential to align the splitting and pruning of
Gaussians with their distribution in a local geometric 3D area remains unex-
ploited. This limitation is particularly interesting when it comes to 3D recon-
stuction of man-made objects, as commonly encountered in photogrammetry
2

<!-- page 3 -->
Figure 1: Methodology EntON. Gaussians are adapted based on the Eigenentropy of their
local neighborhood: low Eigenentropy leads to splitting, medium Eigenentropy results in
unchanged Gaussians, and high Eigenentropy triggers pruning. In constrast, 3DGS triggers
densification based on the view-space position gradient: small Gaussians are cloned, large
Gaussians are splitted. EntON uses the level of Eigenentropy to focus densification on
object surfaces, avoiding unnecessary scene expansion and thus efficiently compressing the
information content of the scene representation.
and surveying applications of structures such as buildings. Those scenes are
dominated by locally, piecewise planar surface structured areas that follow
strong geometric regularities. Therefore, we propose a guiding of the densi-
fication toward improved geometric alignment based on the local 3D distri-
bution Gaussians. This can be done using 3D shape features from Gaussian
neighborhoods. In the context of man-made, urban scenes, the shape feature
Eigenentropy has been widely used in semantic segmentation and point cloud
classification [42, 44], making it particularly well suited for this task.
Our approach aims to explicitly exploit the local 3D structural geom-
etry, by introducing a geometry-guided densification and pruning strategy
EntON for 3D Gaussian Splatting (3DGS), and other splatting-based meth-
ods. We integrate the eigenvalue-based 3D shape feature Eigenentropy, com-
puted from the covariance matrix of the k-nearest neighboring Gaussians
around each Gaussian center. During the densification process, the algorithm
alternates between (i) standard gradient-based densification, which refines
under- and over-reconstructed regions via high view-space gradients, and (ii)
Eigenentropy-aware densification, which preferentially densifies Gaussians in
3

<!-- page 4 -->
low-Eigenentropy (ordered, anisotropic, flat) neighborhoods to better cap-
ture fine geometric details on the object surface, and prunes those in high-
Eigenentropy (disordered, isotropic, spherical/scattered) regions.
EntON
guides the optimization toward improved geometric alignment, while com-
pressing the information content of the scene to the object’s surface to avoid
unnecessary scene expansion.
We evaluate our method, EntON, on two different benchmark datasets:
(i) small-scale DTU dataset using 15 diverse scenes, and (ii) large-scale
TUM2TWIN dataset with urban scenes using two representative scenes. Ge-
ometric accuracy is measured using the Chamfer Distance (cloud-to-cloud).
Photometric reconstruction quality is assessed via the Peak Signal-to-Noise
Ratio (PSNR). To assess efficiency, we further report the final number of
Gaussians and the total training time.
We demonstrate that EntON produces Gaussians that are better aligned
with the underlying object surface geometry. By explicitly exploiting local
3D structural geometry, EntON improves geometric accuracy and reduces
unnecessary model growth through targeted densification in planar, low-
Eigenentropy areas, while simultaneously pruning Gaussians that contribute
to locally scattered, high-Eigenentropy areas. Importantly, this geometry-
guided optimization maintains high photometric quality and preserves a com-
petitive model size and training time. Overall, EntON yields the following
key outcomes:
• An improvement in geometric accuracy of up to 32.7% over 3DGS on
average (and up to 39.8% for the best neighborhood configuration) and
8.6% over 2DGS, while remaining competitive with PGSR,
• An improvement in rendering quality of up to 6.8% over 2DGS and
PGSR on average (and up to 7.5% for the best neighborhood configu-
ration), while remaining competitive with 3DGS,
• A reduction in the number of Gaussians of up to 49.6% compared to
3DGS on average (and up to 59.9% for the most compact configura-
tion), as well as reductions of 14.9% compared to PGSR and 5.3%
compared to 2DGS,
• A reduction in training time of 22.7% compared to 3DGS on average
(and up to 29% for the fastest configuration), as well as speedups of up
to 45.0% and 69.8% over 2DGS and PGSR.
4

<!-- page 5 -->
In the following, we first review related work in Section 2, providing an
overview of 3D reconstruction techniques with a focus on Gaussian Splat-
ting, including existing densification and pruning strategies. We also discuss
different types of 3D shape features relevant for EntON. In Section 3, we
then introduce our geometry-guided, Eigenentropy-aware densification strat-
egy for 3DGS. In Section 4 we introduce the experimental setup, including
datasets, evaluation metrics and implementation details. In Section 5, we
first demonstrate the effectiveness of EntON. We then present the results
of our experiments on both small-scale and large-scale benchmark datasets,
comparing EntON with 3DGS, 2DGS, and PGSR in terms of quantitative
metrics and qualitative performance. Section 6 includes an ablation study
for our method. We analyze and discuss the outcomes in Section 7, high-
lighting the advantages of our approach with respect to geometric accuracy,
rendering quality, and memory and time efficiency. Finally, Section 8 sum-
marizes the contributions of our work and outlines potential directions for
future optimizations and practical applications.
2. Related Work
We present an overview of novel view synthesis and 3D reconstruction
techniques (Section 2.1), followed by an introduction to 3D reconstruction
using Gaussian splats. Subsequently, we outline several densification strate-
gies for 3D Gaussian Splatting (Section 2.2), and finally, we briefly introduce
different types of 3D features (Section 2.3), highlighting 3D shape features,
such as Eigenentropy, which we use to steer our geometry-guided densifica-
tion.
2.1. 3D Reconstruction
The foundation for the rapid development of 3D scene reconstruction was
laid with the introduction of Neural Radiance Fields (NeRFs) [32], followed
by numerous publications fostering the research and development of neural
surface reconstructions, point cloud and mesh reconstructions [33, 40, 48,
28, 17] in a variety of fields.
NeRFs are neural networks with multilayer
perceptrons (MLPs), that model the scene implicitly by estimating a color
and volume density for each position and direction.
These estimates are
themselves subject to a certain degree of uncertainty [20].
Following the development of NeRFs, a novel concept for 3D scene re-
construction was introduced, in which scenes are explicitly described. 3D
5

<!-- page 6 -->
Gaussian Splatting (3DGS) [21] represents the scene using a large number of
3D Gaussians. To parameterize the scene, the Gaussians are initialized from
a point cloud generated by Structure from Motion (SfM). This explicit rep-
resentation avoids unnecessary computation in empty space, enables efficient
GPU-based rasterization and allows real-time rendering with state-of-the-
art visual quality [21]. Each Gaussians is defined by its mean, covariance,
opacity, and spherical harmonics for color definition. The covariance is pa-
rameterized using scaling and rotation. These 3D Gaussians are projected
onto the 2D image space, whereby their parameters (mean values for the
Gaussian centers µ, scaling S, rotation R, opacity α, and colors) are then
refined during the optimization process to match the training images. This
optimized process results in scenes with thousands to millions of Gaussians
representing the 3D object geometry. However, a huge amount of storage
space is required, as 3DGS models the scene using a large number of Gaus-
sians. Therefore, various methods focus on training speed or model capacity
[31, 12, 3, 8, 24, 25, 35, 11] In addition, the Gaussians do not take an or-
dered structure in general [10], indicating neither the centers nor the surfaces
of the Gaussians are properly aligned with the actual object surface. And
since the disorder of the Gaussians relies on the image reconstruction loss,
it can result in incomplete geometric information [2]. To address this chal-
lenge, several methods [15, 10, 6, 2, 29, 50, 16, 37, 27, 26, 13] have been
developed to achieve not only photometrically valid but also geometrically
accurate 3D scene representations using 3DGS. The concept of transforming
3D Gaussians into 2D ellipses or planar 3D ellipsoids in order to achieve
higher geometric accuracy is widely used in many approaches. SuGaR [10]
extracts meshes from 3DGS by introducing a regularization term that aligns
Gaussians with the scene surface. Surfels [6] combines 3D Gaussian points’
optimization flexibility with the surface alignment of surfels by flattening 3D
Gaussians into 2D ellipses, setting the z-scale to zero. 2DGS [15] follows a
similar approach and collapses 3D volumes directly into 2D planar Gaussian
disks for view-consistent geometry, using perspective-accurate splatting with
ray-splat intersection and depth and normal consistency terms. PGSR [2]
flattens Gaussians into planes, using unbiased depth rendering to obtain pre-
cise depth information. MVG-Splatting [29] improves 2DGS by optimizing
normal calculation and using an adaptive densification method guided by
depth maps. MIP-Splatting [50] introduces a 3D smoothing filter to con-
strain Gaussian sizes based on the input views’ sampling frequency, elimi-
nating high-frequency artifacts. 3DGS-to-PC [37] converts 3DGS scenes into
6

<!-- page 7 -->
accurate and dense point clouds by sampling points from each Gaussian’s
multivariate distribution proportional to its scale and contribution, with out-
lier rejection to ensure geometric consistency. FeatureGS [16] incorporates
an additional geometric loss term based on 3D shape features. Despite the
high geometric accuracy, is achieves only moderate rendering quality, since
the geometric loss affects all Gaussians equally.
2.2. Densification and Pruning
The optimization in 3DGS follows a successive adjustment of the Gaus-
sians across the iterations based on rendering performance. The goal is to
create based on the sparse SfM point set, a denser and better represen-
tation of the 3D scene.
Thereby empty areas will be densified, focusing
on under-reconstruction (areas with missing geometric features) and over-
reconstruction (areas covered by large Gaussians). Gaussians in these areas
supposedly have high view-space position gradients and are therefore can-
didates for densification. Thereby small Gaussians are cloned, while large
Gaussians are split into two smaller Gaussians [21]. Although this densifi-
cation strategy is effective, it also causes certain issues. In particular, the
3DGS suffers from difficulties with over-reconstruction [49] and oversized
Gaussians [23], which leads to blurred rendered image areas, since splitting
large Gaussians is important to allow good reconstruction [21].
To overcome this issue, AbsGS [49] proposes an homodirectional view-
space positional gradient based on the sum of the absolute values of pixel-
wise sub-gradients to identify large Gaussians as splitting candidates in over-
reconstructed regions. Gaussian Opacity Fields [51] likewise incorporated
a similar approach into their methodology to identify overly blurred areas.
Several works address the frequency of the gradients. Micro-splatting [23]
changes densification in two key aspects: compact splats through covari-
ance regularization by activating adaptive splitting when the trace exceeds
a threshold, and adaptive stronger densification in high-frequency regions
(high gradients).
FreGS [52] uses frequency regularization from high to
low frequency signals following a coarse-to-fine-densification to overcome the
over-reconstruction by analyzing the rendered images in the spectral space.
AD-GS [34] introduces an alternating densification, which combines high
densification for fine details with low densification reduce artifacts. By ap-
plying geometry-constrained training, via edge-aware depth smoothness and
pseudo-view consistency, it balances detail recovery with artifact mitigation.
EFA-GS [39] monitors the frequency behavior of gradients in floater artifacts
7

<!-- page 8 -->
in order to eliminate them. Another approach [22] replaces the view-space
position gradient with a color-cued densification strategy, which leverages the
0th spherical harmonics gradient to identify regions with under-representing
color. To enhance training speedup, FastGS [35] assesses Gaussian impor-
tance via their multi-view consistency and employs corresponding densifica-
tion and pruning strategies.
AbsGS [49] shows that lowering the gradient threshold improves rendering
quality, since fewer high-frequency Gaussians are split. However, when lower
gradient thresholds are used, it leads to an excessive growth in the number of
Gaussians. Moreover, the splitting decision of 3DGS relies on the view-space
position gradients of the Gaussians. This means that densification is driven
without considering the underlying local area of the 3D geometry.
Motivated by this observation, we propose EntON, an alternating densi-
fication and pruning strategy that complements 3DGS densification scheme
by explicitly exploiting geometric contextual information from the Gaussians
of the reconstructed 3D scene. For man-made objects, as commonly encoun-
tered in the surveying of structures such as buildings, surface 3D points
typically lie on structured, piecewise planar surfaces that follow strong ge-
ometric regularities. Instead of relying exclusively on view-space position
gradients, we aim to split and prune Gaussians according to the geometric
structure of their local neighborhoods in 3D space. The densification process
then enhances a specific characterization of local 3D structures of man-made
objects that is consistent with the Manhattan-World assumption [5, 4]. We
achieve this by introducing a geometry-guided denficiation decision process
based on the Eigenentropy of Gaussian neighborhoods, which reinforces the
dominance of structural entropy. To the best of our knowledge, EntON is the
first method that targets geometric accuracy in combination with photomet-
ric quality directly within the 3D Gaussian Splatting densification process.
2.3. 3D Features
Several types of 3D features exist for point cloud-based applications such
as classification, registration, or calibration. Complex features, which cannot
be interpreted directly include descriptors such as Shape Context 3D (SC3D)
[9], Signature of Histogram of OrienTations (SHOT) [38] or Fast Point Fea-
ture Histograms (FPFH) [36].
In contrast, interpretable features [45] are
those that are directly interpretable, such as local 2D and 3D shape features.
To describe the local structure around a 3D point, the spatial arrangement of
other 3D points in the local neighborhood is often considered. Thereby the
8

<!-- page 9 -->
3D covariance matrix, also known as the 3D structure tensor, is well-known
and suitable for characterizing the shape properties of 3D data [42]. It is
derived explicitly for each point from the point itself and its local neighbors.
The three eigenvalues, λ1 ≥λ2 ≥λ3 ≥0, correspond to an orthogonal sys-
tem of eigenvectors (ϵ1, ϵ2, ϵ3), which indicate the direction (rotation) of the
three ellipsoid principal axes and correspond to the extent (scales) of the 3D
ellipsoid along the principal axes. Based on the behavior of the eigenvalues
λ1, λ2, and λ3, linear (λ1 ≫λ2, λ3), planar (λ1 ≈λ2 ≫λ3), and spherical
(λ1 ≈λ2 ≈λ3) structures can be intuitively described. The usage of ge-
ometric 3D shape features has led to thousands of publications in various
fields over the past few decades, especially for the semantic segmentation
and classification [42, 44, 41] of point clouds. But also for calibration [14] or
registration [1] of 3D point clouds.
3. Methodology
In this section, we present our method EntON (Figure 1), which intro-
duces a geometry-guided densification and pruning strategy for 3D Gaussian
Splatting.
The proposed strategy guides densification and pruning based
on local geometric Gaussian 3D neighborhoods, directly improves geometric
accuracy and maintains photometric quality surface-focused through align-
ment of Gaussians. It leverages the 3D shape feature Eigenentropy, derived
from the eigenvalues of the covariance matrix computed over local Gaussian
neighborhoods, which enables targeted control over geometric regularity, pro-
moting planar, low-entropy regions.
We first review the standard 3DGS densification strategy (Section 3.1),
which forms the basis of EntON. Then we introduce the background and
scheme of our geometry-guided densification (Section 3.2), which is based on
the Eigenentropy of local Gaussian neighborhoods. Finally, we present the
resulting alternating densification strategy EntON (Section 3.3).
3.1. 3DGS Densification
The optimization in 3D Gaussian Splatting (3DGS) [21] iteratively ad-
justs the Gaussians based on the rendering performance. This process starts
from a sparse initial set of Gaussians, whose centers correspond to the 3D
points of an SfM point cloud derived from SIFT [30] features. Starting from
the sparse SfM point set, the objective is to adaptively controls the number
9

<!-- page 10 -->
and spatial density of 3D Gaussians to transform the initial sparse represen-
tation into a denser one, while removing nearly transparent Gaussians and
populating previously empty regions [21]. Rendered images, obtained by pro-
jecting the 3D Gaussians onto the 2D image plane, are compared against the
ground-truth training images. The Gaussians are then adapted by splitting/
cloning, or pruning them accordingly.
In the standard adaptive densification and pruning strategy of 3DGS,
Gaussians with opacity α below a predefined threshold are removed (pruned).
Conversely, under-reconstructed regions (areas lacking sufficient geometric
detail) and over-reconstructed regions (areas covered by large Gaussians) are
targeted for densification. Gaussians in these areas typically exhibit high
view-space position gradients and thus serve as candidates for densification.
In 3D Gaussian Splatting [21], the view-space position gradient measures the
sensitivity of the photometric loss to small displacements of the Gaussian
center µ in 3D space, as observed through its 2D projection in each training
view. This gradient is computed via backpropagation through the differen-
tiable rasterizer. The magnitude corresponds to the norm of the gradient
with respect to the projected 2D coordinates in view space, accumulated
and averaged over contributing pixels and views as implemented in the of-
ficial codebase1. A mathematical interpretation (see also [49, 51]) of this
accumulation is:
∇µL =
X
i
∂L
∂pi
· ∂pi
∂µ .
(1)
where the sum runs over all pixels i to which the Gaussian contributes in
a given view, and the full gradient is accumulated over multiple views across
the training iterations. The key densification criterion is then the average
magnitude:
¯g = 1
N
N
X
j=1
∇(j)
µ L

2 ,
(2)
(or an equivalent per-component 2D norm averaging as in the referenced
implementation), where N is the number of accumulated gradient contribu-
tions. In practice, a Gaussian that appears in many different views (due to its
1https://github.com/graphdeco-inria/gaussian-splatting
(last access 07/21/2024)
10

<!-- page 11 -->
visibility) will accumulate gradient contributions across numerous iterations.
Consequently, the accumulated gradients implicitly stem from multiple views.
Densification (cloning or splitting) is triggered when this average gradient ¯g
exceeds the threshold τpos, with τpos = 0.0002 by default [21].
3.2. Geometry-Guided Densification
3D Gaussian Splatting primarily relies on view-space position gradients
to identify Gaussians whose 3D positional changes strongly affect the 2D
rendered image, triggering the densification based on the magnitude of view-
space position gradients. However, this criterion does not explicitly incor-
porate information of the underlying local geometry. Specifically, whether a
Gaussian lies on or near an actual object surface, where refinement for in-
creased geometric detail is most beneficial. This can be done using 3D shape
features from Gaussian neighborhoods. In the task of semantic segmentation
and point cloud classification of urban scenes in previous work [44], the 3D
shape feature Eigenentropy has also proven to be particularly useful. Eige-
nentropy has been shown to be an appropriate 3D feature for characterizing
plane point cloud structures [7] and a powerful tool for scale selection [43],
making it particularly well suited for this task.
To address the limitation, we bias the densification process through an
Eigenentropy-aware densification strategy. Particularly the splitting, toward
Gaussians residing in locally flat, low-Eigenentropy neighborhoods, which are
strong indicators of their alignment with the object surfaces. Specifically, we
assess whether a Gaussian is located in a region exhibiting low to medium
Eigenentropy and near-planar (flat) structure. Gaussians in such neighbor-
hoods are preferentially split to enhance surface reconstruction fidelity and
geometric accuracy. Conversely, Gaussians in high-Eigenentropy regions (dis-
ordered, isotropic, spherical/scattered) are densified more conservatively or
become candidates for pruning. This directs the densification towards ge-
ometrically significant surface areas and is based on the Manhattan World
assumption [5, 4], where artificial environments consist predominantly of flat
structures.
Addressing the local geometry, we extract geometric feature Eigenen-
tropy from a local 3D structure around each Gaussian center: This involves
computing the covariance matrix from a local neighborhood (Section 3.2.1),
normalizing the eigenvalues (Section 3.2.2), and finally extracting the feature
Eigenentropy (Section 3.2.3). We further describe the typical geometric char-
acteristics of the Eigenentropy (Section 3.2.4) that are relevant for EntON,
11

<!-- page 12 -->
followed by the resulting Eigenentropy-aware densification strategy (Section
3.2.5).
3.2.1. Local Neighborhood of Gaussians
To compute the interpretable geometric 3D shape features from the spa-
tial arrangement of points within a local neighborhood around each 3D point
(here: Gaussian center), a neighborhood must first be defined. The neigh-
borhood size serves as a scale parameter and directly influences the resulting
features, as only the selected neighboring points contribute to the eigenvalue
analysis of the local covariance matrix. The 3D covariance matrix, also called
3D structure tensor [19], is well-known to characterize such shape properties
[42] and derived from a point and its local neighborhood. The three eigenval-
ues, λ1 ≥λ2 ≥λ3 ≥0, correspond to an orthogonal system of eigenvectors
(ϵ1, ϵ2, ϵ3), which indicate the direction (rotation) of the three ellipsoid prin-
cipal axes and correspond to the extent (scales) of the 3D ellipsoid along the
principal axes.
Given a point p0 in the 3D space, i.e., the center of a Gaussian, we define
its k-nearest neighbors {p1, p2, . . . , pk}. The centroid ¯p (Equation 3) of this
neighborhood is computed as:
¯p =
1
k + 1
k
X
i=0
pi
(3)
The covariance matrix C (Equation 4) [46] for the neighborhood (Figure
2) is then:
C =
1
k + 1
k
X
i=0
(pi −¯p)(pi −¯p)T
(4)
3.2.2. Eigenvalue Normalization
To ensure consistency, eigenvalues λ1, λ2, λ3 from the Gaussian neigh-
borhood covariance matrix are normalized by dividing by the sum of the
eigenvalues for each case by (Equation 5):
λ′
i =
λi
sum(λ)
for i ∈{1, 2, 3},
(5)
12

<!-- page 13 -->
Z
Y
ε1√λ1
ε2√λ2
ε3√λ3
X
Figure 2: Representation of an ellipsoid from the neighborhood points represented by the
Gaussian centers with the three eigenvectors (ϵ1, ϵ2, ϵ3) and the corresponding eigenvalues
(λ1, λ2, λ3) in the three-dimensional coordinate system.
with
sum(λ) =
3
X
i=1
λi.
(6)
The normalized eigenvalues λ′
1, λ′
2, λ′
3 are then ordered in descending order
for being used for the final geometric 3D feature computation:
λ′
1 ≥λ′
2 ≥λ′
3 ≥0.
3.2.3. Feature Extraction
To enhance these structural properties that 3D point clouds of man-made
objects exhibit, we incorporate the feature-aware densification und pruning
using the k-nearest neighbors (kNN) of each point. This approach allows for
the calculation of spatial features in the local neighborhood of each Gaussian.
The feature Eigenentropy is defined according to [46] as the Shannon
entropy:
EigenentropykNN = −
3
X
i=1
λ′
i log(λ′
i)
(7)
and quantifies the order/disorder of points by measuring the entropy within
their local 3D neighborhood.
3.2.4. Geometric Characteristics of Eigenentropy
The eigenvalues λ1 ≥λ2 ≥λ3 ≥0 of this covariance matrix enable
the characterization of dominant local structures, whereby the Eigenentropy
13

<!-- page 14 -->
exhibits characteristic values for local linear, planar, and spherical distributed
structures:
• For an ideal linear structure (λ′
1 ≈1, λ′
2 ≈λ′
3 ≈0):
E ≈−(1 · log 1 + 0 · log 0 + 0 · log 0) = 0.
This corresponds to highly anisotropic, ordered distributions.
• For an ideal planar structure (λ′
1 = λ′
2 = 1/2, λ′
3 = 0):
E = −
1
2 log 1
2 + 1
2 log 1
2 + 0 · log 0

= log 2 ≈0.693.
In practice, for near-planar distributions where λ′
3 ≈0 but λ′
1 and λ′
2
vary (while summing to 1), the Eigenentropy satisfies E ≤log 2, with
the maximum achieved at the balanced case λ′
1 = λ′
2 = 1/2. This be-
havior is particularly relevant in idealized settings, such as when Gaus-
sians lie perfectly flat on surfaces (e.g., walls or floors in Manhattan-
world scenes).
• For isotropic/spherical structures (λ′
1 ≈λ′
2 ≈λ′
3 ≈1/3):
E ≈−log 1
3 ≈1.099,
reflecting maximal local disorder.
The Eigenentropy provides a scalar measure of the local order versus
disorder (structural entropy) in the 3D neighborhood [42]. Low Eigenentropy
thus indicates highly structured, anisotropic local geometry (favoring flat,
linear or planar areas), while high values signal disordered or volumetric,
spherical distributions.
Figure 3 illustrates this behavior of Eigenentropy
for various normalized eigenvalue configurations. Splitting Gaussians when
their local Eigenentropy falls below a threshold of ln 2 ≈0.693 favors the
presence of flat, anisotropic, low-entropy structures, including both strongly
linear and planar flat geometries. The plotted examples include λ3 = 0.0
(ideal planar), 0.1, 0.15, and 0.2 (increasing deviation from ideal planarity).
The aim is to focus the attention within the densification process to Gaussian
neighborhoods in flat, low-Eigenentropy regions.
14

<!-- page 15 -->
Why not use planarity? Planarity P as a 3D feature (P = λ2−λ3
λ1 ) assigns
high values to ideal planar structures (P ≈1).
However, low planarity
values (P ≈0) are ambiguous, as they occur for both highly linear (λ1 ≫
λ2 ≈λ3) and isotropic, spherical neighborhoods (λ1 ≈λ2 ≈λ3).
As a
result, a single planarity threshold cannot reliably distinguish between flat,
surface-aligned structures and disordered volumetric regions, making robust
splitting and pruning decisions difficult. In contrast, Eigenentropy resolves
this ambiguity by explicitly quantifying the degree of local structural order/
disorder, enabling a more stable geometry-guided densification strategy.
0.40
0.45
0.50
0.60
0.70
0.80
0.90
1.00
1
0.2
0.4
0.6
0.8
1.0
ln 2
ln 3
Eigenentropy(
1,
2,
3)
2 = 1
1,
3 = 0
Emax = 0.693
Emin = 0.000
2 = 0.9
1,
3 = 0.1
Emin = 0.639
Emax = 0.949
2 = 5/6
1,
3 = 1/6
Emin = 0.868
Emax = 1.028
2 = 0.8
1,
3 = 0.2
Emin = 0.950
Emax = 1.055
Figure 3: Eigenentropy E(λ1, λ2, λ3) as a function of the largest normalized eigenvalue
λ1 (with λ1 ≥λ2 ≥λ3 ≥0, P3
i=1 λi = 1). Curves represent different fixed values of λ3
(ideal planar: λ3 = 0; near-planar: λ3 = 0.1; transitional between planar and spherical:
λ3 = 1/6 ≈0.1667; and higher spherical distribution). Markers indicate minimum and
maximum Eigenentropy for each case. The dashed line at ln 2 ≈0.693 corresponds to the
ideal planar case (λ1 = λ2 = 0.5, λ3 = 0).
3.2.5. Eigenentropy-Aware Densification
Our method leverages the eigenvalue-derived 3D shape feature Eigenen-
tropy to assess and enhance local 3D geometric shape properties around
each Gaussian center in a local neighborhood (Figure 1): We integrate a
geometry-guided densification process (illustrated in Figure 4) based on the
3D shape feature Eigenentropy computed from the covariance matrix of the k-
nearest neighbors (kNN) around each Gaussian center. During optimization,
Gaussians are split, or pruned according to their local neighborhood charac-
teristics. This strategy enables a geometry-guided control of the Gaussian
15

<!-- page 16 -->
distribution, promoting structured, low-entropy local neighborhoods. Specif-
ically:
• Gaussians exhibiting low Eigenentropy (indicating more ordered,
anisotropic, flat regions) are preferentially split to increase local density
in these areas and better capture fine geometric details on the objects
surface.
• Gaussians with high Eigenentropy (indicating disordered, isotropic,
spherical/scattered regions) are preferentially pruned.
• Gaussians falling within a transitional Eigenentropy interval are
left unchanged to preserve stability and prevent unnecessary splitting
or premature pruning.
ε1√λ1
ε2√λ2
ε3√λ3
Eigenentropy
low
~0
high
~1
ε1√λ1
ε2√λ2
ε3√λ3
ε1√λ1
ε2√λ2
ε3√λ3
Densification
split
prune
Figure 4: Low Eigenentropy leads to splitting, medium Eigenentropy results in unchanged
Gaussians, and high Eigenentropy triggers pruning. Representation of the ellipsoids based
on neighboring Gaussian centers with the three eigenvectors (ϵ1, ϵ2, ϵ3) and the corre-
sponding eigenvalues (λ1, λ2, λ3) in the three-dimensional coordinate system.
3.3. Alternating Densification
To effectively combine the complementary strengths of solely gradient-
based and geometry-guided densification, we adopt an alternating densifica-
tion strategy EntON (Algorithm 1): it alternates between two complemen-
tary strategies at regular intervals during optimization, (i) gradient-based
densification of 3DGS and (ii) Eigenentropy-aware, geometry-guided densifi-
cation.
16

<!-- page 17 -->
The rationale is as follows: gradient-based densification effectively refines
regions with high view-dependent gradients, while Eigenentropy-aware den-
sification reinforces a specific geometric characteristic of local spatial struc-
tures.
It focuses the attention of the densification on the object surface
and avoids ’unnecessary’ scene expansion in high-Eigenentropy regions, while
compressing the information content of the scene to the object’s surface. Al-
ternating these two strategies prevents both over-reconstruction and under-
reconstruction, leading to a scene representation with photometric fidelity
and geometrically accurate, structured Gaussians. By alternating reverting
to gradient-based densification, we safeguard high-contribution Gaussians
using the default gradient threshold. This maintains an overall reconstruc-
tion quality, while the Eigenentropy-aware steps guide the distribution of the
Gaussians towards a geometrically accurate structures.
Starting, after a pre-training, from iteration 3000, the training alternates
every 100 iterations between two densification modes:
• Gradient-aware densification (3DGS): This strategy ensures that
Gaussians with persistently high view-space position gradients, which
indicate strong photometric contribution, are reliably cloned/split, even
if their local neighborhood does not yet exhibit the Eigenentropy cri-
teria.
• Eigenentropy-aware densification: This strategy relies on the ex-
tracted Eigenentropy of local Gaussian neighborhoods. It selectively re-
inforces geometric accuracy by preferentially splitting Gaussians (even
with lower view-space position gradients) in low-Eigenentropy (ordered,
anisotropic, flat) neighborhoods and pruning those in high-Eigenentropy
(disordered, isotropic, spherical/scattered) regions.
The first 3000 iterations are pure gradient-based densification (3DGS)
to establish a sufficiently dense initial representation. This pre-densification
phase is essential, as the reliability of the eigenvalue-derived features strongly
depend on adequate local point density and neighborhood definition. After
this phase, densification is performed every 100 iterations, following the de-
fault densification schedule used in 3DGS, and we therefore keep the same
step size. The Gaussians are divided further depending on their size into 2, 4
or 8 Gaussians to allow a more uniform spatial distribution, which is inspired
by [23].
17

<!-- page 18 -->
Algorithm 1: Alternating Densification with EntON
for each training iteration t do
if t mod 100 = 0 then
if t < 3000 then
// Pre-training:
Gradient-based 3DGS
densification
clone or split Gaussians with high view-space gradient
magnitude;
else
// Alternating strategy:
switch every 100
iterations
if (t/100) mod 2 = 0 then
// Gradient-based 3DGS densification
clone or split Gaussians with high view-space gradient
magnitude;
else
// Eigenentropy-aware densification
for each Gaussian do
compute local Eigenentropy E from kNN
covariance;
if E ≤τlow then
split;
else if E > τhigh then
prune;
else
// τlow < E ≤τhigh
keep/ continue;
4. Experiments
In this section, we present the experimental setup by introducing the
used datasets (Section 4.1), the evaluation metrics (Section 4.2), and the
implementation details (Section 4.3).
18

<!-- page 19 -->
4.1. Data
EntON is evaluated on two benchmarks, a small-scale and a large-scale
dataset.
Small-Scale dataset. The small-scale dataset DTU [18] consists of scenes fea-
turing real objects, including either 49 or 64 RGB images, corresponding
camera poses, and reference point clouds obtained from a structured-light
scanner (SLS). We specifically focus on the same 12 scenes as as previous
approaches [15, 2, 6, 29].
Large-Scale dataset. The large-scale dataset TUM2TWIN [47] contains large-
scale ourdoor scenes of the Technical University of Munich, including different
building types and sizes. Motivated by GS4BUILDINGS [53], we focus on
two distinct building clusters, each containing approximately 20–30 images.
For geometric 3D accuracy comparison, we use each a subset of the reference
point clouds from UAV laser scanning (ULS).
4.2. Metrics
To evaluate our method quantitatively and qualitatively. For 3D geomet-
ric surface accuracy Chamfer cloud-to-cloud distance by following the DTU
evaluation procedure [18], which masks out points above 10 mm since the
reference point clouds are incomplete. Low Chamfer distance indicates high
geometric accuracy. 2D rendering quality of the images is evaluated with
the Peak Signal-to-Noise Ratio (PSNR) in dB, whereby a high PSNR is tar-
geted. Considering efficiency of our method, we report the training time and
the number of Gaussians needed to represent the scene.
4.3. Implementation Details and Experiments
Implementation. 3D Gaussian Splatting2 for comparison purpose, and as fun-
dament for our method, is processed according to the original implementa-
tion, using default densification strategies and the default parameters with a
view-space position gradient of 0.0002, learning rates of 0.0025 for spherical
harmonics features, 0.05 for opacity adjustments, 0.005 for scaling operations
and 0.001 for rotation transformations, on a NVIDIA RTX3090 GPU. All ex-
periments on the large-scale dataset were performed by using the automatic
2https://github.com/graphdeco-inria/gaussian-splatting
(last access 07/21/2024)
19

<!-- page 20 -->
image resolution downscaling applied by 3DGS in its default configuration,
in order to match the memory constraints of the used GPU. For further
comparison purpose, we use 2D Gaussian Splatting3 and PGSR4, which are
processed according to the original implementation by using default param-
eters.
Neighborhood Definition. The size of the local neighborhood directly influ-
ences the covariance matrix and, consequently, the resulting eigenvalues and
derived shape features, and consequently the Eigenentropy. Different neigh-
borhood sizes capture geometric structure at varying spatial scales, affecting
the sensitivity of Eigenentropy to local surface characteristics. We investi-
gate multiple neighborhood sizes to analyze their impact on the resulting
Eigenentropy-aware densification: We test our approach on different neigh-
borhood sizes of k ∈{25, 50, 75, 100} to analyze the impact of the number
of nearest neighbors on the results. Since the Gaussian spatial density in-
creases during training, we also experiment with adaptive neighborhood siz-
ing, where the kNN is incremented every 2500 iterations (from 25 to 50, to
75, and finally 100). Computing k-nearest neighbors for each Gaussian can
be time-consuming, especially for large-scale scenes. To efficiently handle
neighborhood queries, we leverage the optimized kNN implementations in
PyTorch3D5, which provide GPU-accelerated neighborhood searches.
Densification Thresholds. For 3DGS densifcation we use the default view-
space gradient threshold [21] of 3DGS setted to τpos = 0.0002. The gradient
threshold for EntON is set to τpos = 0.0001, to allow a more sensitive split-
ting of the Gaussians. For the Eigenentropy-aware densification in EntON,
the Eigenentropy thresholds τhigh and τlow are derived from the characteristic
behavior of the Eigenentropy feature for locally structured neighborhoods, in
particular linear and planar configurations with low sphericity, as described
in Section 3.2.4. i) For an ideal planar neighborhood, the Eigenentropy equals
ln(2) is therefore used as the lower threshold τlow = ln(2), enabling splitting
and explicitly favoring Gaussians embedded in low-entropy, flat neighbor-
hood structures. ii) The upper threshold τhigh, which triggers pruning, is
3https://github.com/hbb1/2d-gaussian-splatting
(last access 04/29/2025)
4https://github.com/zju3dv/PGSR
(last access 05/11/2025)
5https://github.com/facebookresearch/pytorch3d (last accessed: 08/02/2024)
20

<!-- page 21 -->
determined empirically based on the observed Eigenentropy distribution of
outlier Gaussians and set to τhigh = 0.95 (Ablation study, Section 6) As illus-
trated in Figure 3 in Section 3.2.4, neighborhoods with Eigenentropy values
approx. E ≥0.868 predominantly correspond to unstructured, spherical dis-
tributions. iii) Gaussians with values within the interval τlow < E ≤τhigh
remain unchanged.
5. Experimental Results
The following sections present both qualitative and quantitative results
of EntON in comparison to 3DGS, 2DGS, and PGSR. First, we demonstrate
that EntON effectively influences the Eigenentropy of local Gaussian neigh-
borhoods in general (Section 5.1), report the results on the small-scale DTU
dataset (Section 5.2), followed by the results on the large-scale TUM2TWIN
dataset (Section 5.3).
5.1. Eigenentropy Distribution
First, we demonstrate that EntON is effective and that the proposed
Eigenentropy-aware densification and pruning strategy actively influences
both the Eigenentropy of local Gaussian neighborhoods and the resulting geo-
metric reconstruction accuracy during training process. The final mean Eige-
nentropy across all Gaussian neighborhoods is significantly lower when apply-
ing our strategy compared to using the 3DGS densification strategy. Table 1
summarizes the per-scene and mean Eigenentropy values after 15 000 train-
ing iterations on the DTU dataset. The training progression further high-
lights this effect (Figure 5). Across all scenes, higher Eigenentropy correlates
with increased geometric error, whereas low to medium Eigenentropy aligns
with superior surface reconstruction accuracy. In 3DGS, the mean Eigenen-
tropy across all DTU scenes remains persistently high (typically above 0.95,
approaching 1.0) and even shows a slight upward trend over time. Corre-
spondingly, geometric accuracy remains limited, with cloud-to-cloud (C2C)
distances only marginally improving from approximately 1.75 mm to around
1.60 mm.
In contrast, EntON causes a rapid decrease in mean Eigenen-
tropy starting from the onset of alternating densification (at iteration 3000).
After approximately 5 000 iterations (i.e., 2 000 iterations of Eigenentropy-
aware densification), the value stabilizes at a consistently lower level be-
tween roughly 0.78 and 0.82. The geometric accuracy follows the same trend:
upon the start of EntON, the mean C2C distance drops sharply from about
21

<!-- page 22 -->
1.75 mm to approximately 1.20 mm and continues to decrease steadily, con-
verging to a stable value of around 1.05 mm after roughly 12 500 iterations.
This inverse relationship between high-Eigenentropy and high C2C distance
is further visualized in Figure 6, by comparing the Eigenentropy point clouds
and cloud-to-cloud (C2C) distance point clouds. Our approach consistently
achieves lower Eigenentropy in structured regions and reduced geometric er-
ror compared to 3DGS, confirming that enforcing of using Gaussians in pla-
nar, low-Eigenentropy regions as splitting candidates to enhance low local
Eigenentropy promotes both geometric regularity and surface accuracy.
Method
24
37
40
55
63
65
69
83
97
105
106
110
114
118
122
mean
3DGS
0.95
0.96
0.99
0.97
0.86
0.90
0.92
0.95
0.91
0.96
0.96
0.95
0.96
0.95
0.95
0.96
EntON
0.82
0.82
0.82
0.82
0.81
0.81
0.82
0.82
0.82
0.81
0.83
0.83
0.82
0.82
0.82
0.82
Table 1: Eigenentropy comparison on the DTU dataset: Mean Eigenentropy of all Gaus-
sian centers within a local neighborhoods after the training process using 15 000 iterations.
2500
5000
7500
10000
12500
15000
Iteration
0.8
0.85
0.9
0.95
1.0
ln(3)
Mean Eigenentropy
Eigenentropy
EntON
Eigenentropy
3DGS
C2C
EntON
C2C
3DGS
0.75
1.00
1.25
1.50
1.75
2.00
Mean C2C Distance in mm
Figure 5: Trend of the mean Eigenentropy ↓and mean cloud to cloud (C2C) distance ↓of
all DTU scenes during the training process. Comparison of 3DGS and EntON. As EntON
starts at iteration 3000, C2C is first reported at iteration 2500.
5.2. Small-Scale Data
The following sections present quantitative (Section 5.2.1) and qualitative
(Section 5.2.2) results of our method EntON in comparison to 3DGS, 2DGS,
22

<!-- page 23 -->
3DGS
EntON (knn100)
EntON (knn75)
EntON (knn50)
EntON (knn25)
Eigenentropy↓
C2C ↓
ln(3)
0
0mm
5mm
10mm
Figure 6: Comparison of Eigenentropy and geometric accuracy for DTU scene scan55.
Point cloud of 3DGS and our method for mean Eigenentropy with different neighborhood
sizes and cloud-to-cloud distance (C2C).
and PGSR. We evaluate the approaches after a fixed number of training iter-
ations and distinguish between four key aspects: geometric surface accuracy,
photometric rendering photometric quality, and efficiency of memory and
training time.
5.2.1. Quantitative Results
The following quantitative results, obtained after a fixed training duration
of 15 000 iterations, the geometric surface accuracy via the Chamfer cloud-
to-cloud distance, rendering quality in terms of PSNR, the final number of
Gaussians, and the total training time.
Overall, the performance of the
evaluated methods is analyzed with respect to different scene characteristics,
including textured, reflective, and rough surfaces or materials.
Geometric Accuracy. We evaluate the geometric accuracy of the reconstructed
surface points using Chamfer cloud-to-cloud (C2C) distance measured against
the reference point cloud, considering only points within 10 mm of the refer-
ence (following the standard DTU evaluation). The reported values represent
the accuracy of the Gaussian centers, which serve as the geometric backbone
for subsequent meshing pipelines.
Table 2 summarizes C2C distances (in mm, ↓) per scene and in average
across all evaluated methods after 15 000 training iterations. EntON achieves
the best overall performance when using small neighborhood sizes. In partic-
23

<!-- page 24 -->
ular, EntON (knn = 25) obtains the lowest mean distance of 0.97 mm, slightly
outperforming PGSR with 1.00 mm. Larger neighborhood sizes lead to pro-
gressively worse accuracy in average: knn = 50 reaches 1.04 mm, knn = 75
yields 1.13 mm, the adaptive variant 1.14 mm, and knn = 100 1.19 mm. Both
2DGS with 1.33 mm and the 3DGS with 1.61 mm in average are clearly out-
performed by our approach and PGSR.
However, the performance remains highly scene dependent.
On well-
textured surfaces (e.g., scans 40, 55, 106, 118, 122), EntON with small neigh-
borhoods (knn = 25 or knn = 50) consistently produces the most accurate
geometry, while its also frequently surpassing PGSR. On reflective or spec-
ular surfaces (e.g., scans 63, 97, 110), accuracy degrades noticeably for very
small neighborhoods. Likely because too aggressive pruning removes relevant
Gaussians in regions with low spatial density of the Gaussians. Larger neigh-
borhoods (knn ≥75 or adaptive) provide greater robustness in such cases.
Although PGSR generally remains more stable under strong reflections and
gloss surfaces.
Both 3DGS and 2DGS suffer substantially from material
properties such as specularity and fine surface structure. On rough-textured
surfaces (e.g., scans 83, 105), EntON and PGSR perform comparably, with
smaller neighborhoods again tending to yield the best local accuracy.
Method
24
37
40
55
63
65
69
83
97
105
106
110
114
118
122
mean
3DGS
1.82
1.84
1.72
1.55
2.17
1.70
1.66
2.37
1.90
1.71
1.50
1.51
1.47
1.30
1.34
1.61
2DGS
1.25
1.51
0.97
0.73
2.03
1.46
1.26
1.97
1.76
1.51
0.78
1.37
0.94
0.69
0.75
1.33
PGSR
0.92
1.15
0.77
0.67
1.43
0.99
0.97
1.28
1.79
1.13
0.60
0.96
0.66
0.68
0.79
1.00
EntON(knnadaptive)
1.13
1.22
0.79
0.57
1.73
1.13
1.06
1.82
1.79
1.27
0.72
1.09
0.73
0.66
0.71
1.14
EntON(knn100)
1.17
1.32
0.88
0.62
1.77
1.23
1.12
1.85
1.80
1.32
0.82
1.11
0.80
0.73
0.77
1.19
EntON(knn75)
1.13
1.29
0.82
0.58
1.68
1.15
1.05
1.71
1.79
1.24
0.76
1.14
0.75
0.68
0.72
1.13
EntON(knn50)
1.08
1.21
0.76
0.55
1.60
1.06
0.98
1.59
1.79
1.21
0.69
1.12
0.68
0.62
0.67
1.04
EntON(knn25)
0.90
1.12
0.70
0.51
1.62
0.95
0.90
1.41
1.81
1.09
0.56
1.21
0.62
0.55
0.61
0.97
Table 2: Surface accuracy. Geometric accuracy comparison on the DTU dataset with
Chamfer cloud-to-cloud distances ↓in mm for points ≤10 mm from the reference, according
to the DTU evaluation script. Best results are highlighted as 1st , 2nd , and 3rd . Mean
scores are listed at the bottom. The training incorporates 15 000 iterations.
Rendering Quality. We assess rendering quality using peak signal-to-noise ra-
tio (PSNR), while higher PSNR values indicate better photometric fidelity.
Table 3 reports the PSNR values (in dB, ↑) per scene and the overall mean.
Our method consistently achieves competitive or superior rendering quality
compared to all baselines. In particular, EntON (knn = 100) and EntON
(knn = 75) deliver the highest average PSNR of 34.71 dB and 34.75 dB, re-
spectively, slightly surpassing 3DGS (34.84 dB) in several configurations. On
24

<!-- page 25 -->
well-textured scenes (e.g., scans 24, 40, 55, 65, 69, 106, 114, 118, 122), EntON
matches or exceeds the rendering quality of 3DGS across most neighborhood
sizes, often producing the highest per-scan PSNR values. Larger neighbor-
hoods (knn = 75 or knn = 100) tend to provide the best overall photometric
fidelity. Smaller neighborhoods (knn = 25 or knn = 50) still yield very strong
results but show slightly lower average PSNR, particularly on scenes with
challenging reflective surfaces or low textured surfaces. In contrast, 2DGS
(32.54 dB), and PGSR (32.32 dB) exhibit noticeably lower rendering quality,
often suffering from detail loss.
Method
24
37
40
55
63
65
69
83
97
105
106
110
114
118
122
mean
3DGS
35.86 30.28 34.59 34.06 37.22 34.02 33.28 34.51 33.53 37.59 37.97 31.55 35.39 36.47 36.33 34.84
2DGS
32.69 28.51 31.74 32.13 35.10 31.92 31.01 31.51 31.27 34.15 35.92 30.17 32.60 35.06 34.38 32.54
PGSR
31.77 27.67 30.92 31.99 34.20 31.82 30.15 30.71 30.83 34.63 36.00 30.90 32.55 35.84 34.80 32.32
EntON(knnadaptive) 35.38 28.70 33.51 35.32 37.01 35.66 32.05 35.05 33.02 37.49 37.74 30.46 34.46 36.24 35.99 34.60
EntON(knn100)
34.36 29.44 34.51 34.50 36.87 35.93 31.47 35.36 32.95 37.24 37.86 30.52 34.06 36.32 36.24 34.71
EntON(knn75)
34.68 28.72 34.90 35.13 36.45 34.94 31.60 34.94 32.93 37.42 37.87 30.25 34.47 35.94 36.47 34.75
EntON(knn50)
34.64 29.14 34.34 34.71 36.23 35.79 31.88 35.39 32.64 37.47 37.40 29.98 34.01 36.19 36.03 34.39
EntON(knn25)
34.36 29.09 34.25 34.96 36.56 35.33 31.64 35.39 31.47 37.39 37.19 29.65 33.63 35.76 36.14 34.19
Table 3: Rendering quality comparison on the DTU dataset. We report the PSNR ↑in
dB. Mean scores are listed. The training incorporates 15 000 iterations. Best results are
highlighted as 1st , 2nd , and 3rd
Efficiency. We evaluate the efficiency of memory and training time, by mea-
suring the final number of Gaussians and training time. A lower number of
Gaussians directly correlates with reduced memory consumption (and faster
rendering). Table 4 reports the number of Gaussians (↓) per scene and in
average. 3DGS produces by far the largest number of Gaussians on average
(392 129), indicating a highly redundant and uncompressed representation.
In comparison, 2DGS (208 572), PGSR (232 004), and EntON achieve sub-
stantially more compact scene representations.
Among our variants, smaller neighborhood sizes lead to significantly fewer
Gaussians while preserving or even improving geometric and rendering qual-
ity (as shown in previous sections). In particular, EntON (knn = 25) achieves
the lowest average number of Gaussians at 157 391, approximately 60% fewer
than 3DGS and clearly outperforming all baselines in compactness. Larger
neighborhoods result in higher Gaussian counts: knn = 50 yields 187 759,
knn = 75 211 925, knn = 100 233 116, and the adaptive variant 228 578 Gaus-
sians. All our neighborhood sizes remain more compact than 3DGS and are
competitive with or better than 2DGS and PGSR, especially at smaller knn
values. Notably, even our most compact variant (knn = 25) starts from the
25

<!-- page 26 -->
same sparse SfM initialization (average 22 771 points) but efficiently densifies
only where necessary, avoiding the excessive densification typical of 3DGS.
Method
24
37
40
55
63
65
69
83
97
105
106
110
114
118
122
Mean
3DGS
611 749
695 233
778 308
622 019
223 472
233 292
249 407
200 975
556 486
238 905
215 615
200 024
276 819
291 762
266 299
392 129
2DGS
317 153
391 404
354 076
341 317
151 758
159 384
169 785
123 622
240 807
127 348
105 397
102 635
132 201
160 033
144 754
208 572
PGSR
338 583
368 363
394 813
376 698
127 514
221 565
253 433
122 908
211 997
205 436
134 614
175 274
166 209
211 044
195 750
232 004
EntON(knnadaptive)
328401
370351
414967
373220
156504
224221
179262
140561
260293
186702
125623
116045
155316
171382
164187
228578
EntON(knn100)
329222
403205
409569
365176
156361
207048
173000
151514
285136
185796
136097
114874
159414
175780
169528
233116
EntON(knn75)
316775
346756
409775
366452
150623
184157
153795
148258
287247
173638
130814
104196
163856
165875
158206
211925
EntON(knn50)
283176
331361
377030
331516
134875
177036
155980
129713
242596
164645
116202
88650
140847
153449
148484
187759
EntON(knn25)
257560
309748
352715
313075
116733
162947
152141
115340
175660
147020
104522
72692
121997
136422
132683
157391
Initial SfM
15 479
24 857
39 158
33 506
10 869
13 203
15 264
10 652
20 467
25 291
33 523
11 382
25 761
27 650
20 975
22 771
Table 4: Number of Gaussians on the DTU dataset. We report the total number of
Gaussians ↓resulting from the EntON, compared to 3DGS, 2DGS, PGSR, and the number
of SfM points used for initialization. The training incorporates 15 000 iterations. Mean
scores are listed. Best results (low total number is better) are highlighted as 1st , 2nd ,
and 3rd .
In addition to memory efficiency, we compare the training time required
by each method to complete 15 000 iterations, serving as a indicator for
computational efficiency. Table 5 presents training time per scene and in av-
erage. Our method consistently achieves the fastest training time across all
neighborhood sizes, with smaller neighborhoods yielding the most significant
speedups. In particular, EntON (knn = 25) requires only 9.14 minutes on
average, which is approximately 29% faster than 3DGS with 12.88 minutes
and substantially faster than 2DGS with 18.11 and PGSR with 32.89minutes
. Larger neighborhoods incur modestly higher training time due to increased
per-iteration costs from processing more neighest neighbors: knn = 50 aver-
ages 9.91 min, knn = 75 reaches 10.21 min, knn = 100 and the adaptive variant
both require 10.52 min. Nevertheless, even the slowest of our configurations
remains markedly faster than the baselines.
Method
24
37
40
55
63
65
69
83
97
105
106
110
114
118
122
Mean
3DGS
18.48
15.61
15.95
13.05
11.86
11.97
11.85
11.22
14.20
11.88
11.03
10.38
11.84
11.02
10.93
12.88
2DGS
21.87
20.24
19.53
18.22
17.22
18.27
17.44
18.00
19.37
18.12
17.32
17.16
18.17
17.19
17.51
18.11
PGSR
33.45
30.04
32.58
28.86
27.55
27.04
31.04
28.08
28.38
40.17
38.07
38.00
40.03
22.20
37.34
32.89
EntON(knnadaptive)
12.35
10.97
11.91
10.76
10.06
9.60
9.57
9.07
9.77
9.46
9.28
9.09
9.36
9.23
9.28
10.52
EntON(knn100)
13.02
11.77
12.74
11.18
10.33
9.75
9.80
9.33
10.26
9.71
9.80
8.80
9.66
9.67
9.95
10.52
EntON(knn75)
12.49
11.27
12.34
10.68
10.17
9.54
9.65
9.39
9.93
9.41
9.78
8.73
9.80
9.18
9.34
10.21
EntON(knn50)
11.78
10.62
11.42
10.37
9.87
9.31
9.01
9.07
9.43
9.25
9.27
8.51
9.14
9.09
8.91
9.91
EntON(knn25)
10.97
10.15
10.95
10.09
9.45
9.02
9.13
9.08
9.10
9.18
8.64
8.09
8.76
8.75
8.83
9.14
Table 5: Training time comparison on the DTU dataset. We report the minutes for
15 000 iterations in min. Mean scores are listed. Best results (low training time is better)
are highlighted as 1st , 2nd , and 3rd . The training incorporates 15 000 iterations.
Our Eigenentropy-aware Gaussian densification approach, guided local
neighborhood of Gaussians, consistently delivers a compelling balance across
26

<!-- page 27 -->
the four key metrics: rendering quality, geometric accuracy, memory effi-
ciency, and training time.
The highest average rendering quality is achieved with larger neighbor-
hood sizes: knn = 75 reaches 34.75 dB and knn = 100 yields 34.71 dB, ap-
proaching or matching 3DGS (34.84 dB) while using 40 to 46 % fewer Gaus-
sians.
Smaller neighborhoods cause a slight PSNR reduction (knn = 25:
34.19 dB; knn = 50: 34.39 dB) but remain substantially superior to 2DGS
(32.54 dB) and PGSR (32.32 dB). In contrast, geometric accuracy strongly
favors smaller neighborhoods.
The best mean C2C distance is obtained
with knn = 25 at 0.97 mm, outperforming PGSR (1.00 mm) and significantly
surpassing 2DGS (1.33 mm) and 3DGS (1.61 mm).
Larger neighborhoods
show progressively higher errors (knn = 50: 1.04 mm; knn = 75: 1.13 mm;
knn = 100: 1.19 mm).
Consumption of memory and training time follow
a similar trend. The most compact representation is achieved at knn = 25
with only 157 k Gaussians on average, a reduction of approximately 60 %
compared to 3DGS (392 k), and the shortest training time of 9.14 min (29 %
faster than 3DGS at 12.88 min).
Larger neighborhoods scale toward 212
to 233 k Gaussians and 10.2 to 10.5 min, remaining markedly more efficient
than both 2DGS and PGSR across all configurations. Figure 7 illustrates the
small trade-off between geometric accuracy and rendering quality. For En-
tON, geometric accuracy improves with smaller neighborhood sizes, whereas
rendering quality benefits from larger neighborhoods. The most balanced
trade-off is observed approximately at the intersection of the curves, around
knn = 50 (or even knn = 75). It is apparent that 2DGS, and particularly
PGSR, achieve good geometric accuracy but exhibit notable weaknesses in
rendering quality. In contrast, 3DGS displays the opposite behavior, pro-
viding high rendering quality at the expense of geometric precision. Using
knn = 50 retains very strong rendering quality (34.39 dB), delivers excellent
geometric accuracy (1.04 mm, corresponding to a 35.4 % improvement over
3DGS), achieves substantial compression (188 k Gaussians, 52 % fewer than
3DGS), and maintains fast training (9.91 min, 23 % faster than 3DGS). It
therefore provides near-state-of-the-art performance in all four aspects si-
multaneously.
5.2.2. Qualitative Results
Similar to the quantitative results, EntON yields promising qualitative
results in terms of geometric accuracy of the 3D point clouds (Section 5.2.2)
and rendering quality (Section 5.2.2).
EntON, consistently accurate and
27

<!-- page 28 -->
25
50
75
100
knn
0.0285
0.0290
0.0295
0.0300
0.0305
0.0310
Mean 1/PSNR
EntON
3DGS
2DGS
PGSR
1.0
1.1
1.2
1.3
1.4
1.5
1.6
Mean C2C Distance in mm
Figure 7: Trend of the mean PSNR (plotted as its inverse to illustrate the trade-off) and
mean cloud to cloud (C2C) distance in EntON of all DTU scenes over different neighbor-
hood sizes knn, in comparison to 3DGS, 2DGS and PGSR.
photometric qualitative results are generated across all 15 scenes, compared
to 3DGS, 2DGS and PGSR.
Geometric Accuracy. The geometric accuracy of Gaussian centers on DTU
dataset, evaluated using the Chamfer cloud-to-cloud distance (Figure 8),
highlights the accurate geometric performance of EntON. Note that the ref-
erence point clouds are incomplete, which leads to high values on the object
edges. The results demonstrate high surface accuracy for EntON and confirm
the quantitative findings. In addition, the Gaussian distributions produced
by EntON exhibit sharp and well-defined structures, indicating efficient split-
ting along object surfaces.
This allows splitting of Gaussians with lower
view-space position gradients when they are located on the surface, which
in turn enhances high-quality, sharp renderings with fine geometric details
and pronounced object boundaries. In contrast, the results obtained with
2DGS and PGSR achieve strong geometric accuracy with a more uniform
point distribution.
However, the resulting point clouds appear smoother,
and object boundaries are partially blurred, leading to less sharply defined
surface edges.
Rendering Quality. The rendering quality (Figure 9) shown by the rendered
test images also underlines the overall strong performance of EntON com-
pared to 3DGS, 2DGS and PGSR. EntON is able to reconstruct fine details
more accurately, avoiding the over-reconstructed regions that are often ob-
served in 2DGS and, to some extent, in 3DGS and PGSR. This advantage
28

<!-- page 29 -->
3DGS
2DGS
PGSR
EntON
scan24
scan37
scan40
scan55
scan114
scan122
0mm
5mm
10mm
Figure 8: Geometric accuracy comparison of 3DGS, 2DGS, PGSR and EntON on the
DTU dataset with Chamfer cloud-to-cloud distances ↓for the same PSNR. Color values
are cropped at 10mm distance.
29

<!-- page 30 -->
is particularly pronounced on textured surfaces, as seen in scenes scan24,
scan40, scan114, scan122.
Some blurring and reduced sharpness are ob-
served for EntON on reflective or homogeneous surfaces, such as in scene
scan37. This is due to fewer Gaussians in these regions, rather no increased
splitting or even pruning due to high Eigenentropy. Notably, EntON achieves
high rendering quality despite using a relatively small number of Gaussians.
While 3DGS also delivers good quality with few over-reconstructed areas,
it relies on a substantially higher number of Gaussians. In contrast, 2DGS
and PGSR use fewer Gaussians but often produce poorer or partially blurred
reconstructions.
5.3. Large-Scale Data
The following sections show qualitative (Section 5.3.1) and quantitative
(Section 5.3.2) results on the TUM2TWIN large-scale dataset of EntON in
comparison to 3DGS, 2DGS and PGSR.
5.3.1. Quantitative Results
The results across the key metrics, geometric accuracy, rendering quality,
number of Gaussians, as well as training time, are presented in Tables 6 - 9.
Geometric Accuracy. We evaluate the geometric accuracy of the reconstructed
surface points using Chamfer cloud-to-cloud (C2C) distance measured against
the reference point cloud, considering only points within 0.5 m of the ref-
erence.
On building1, EntON consistently produces highly accurate sur-
faces across all neighborhood sizes, with the best performance achieved for
knn = 75 (0.179 m) and an average over all neighborhood sizes of 0.183 m.
Compared to the baselines, EntON outperforms 3DGS (0.184 m) and 2DGS
(0.193 m), while slightly underperforming PGSR (0.197 m). Overall, EntON
improves geometric accuracy by up to 2.7% compared to 3DGS (best variant)
and 7.3% compared to 2DGS, while slightly trailing PGSR by 9.1% for the
best variant on building1.
Rendering Quality. We assess rendering quality using peak signal-to-noise ra-
tio (PSNR), where higher values indicate better photometric fidelity. Across
all neighborhood sizes, EntON achieves consistently strong PSNR values, in
average with the best variant (knn = 75) reaching 31.54 dB and the mean
across all neighborhood sizes being 31.44 dB. EntON surpasses the rendering
30

<!-- page 31 -->
3DGS
2DGS
PGSR
EntON
GT
scan24
scan37
scan40
scan114
scan122
Figure 9: Rendering quality comparison of 3DGS, 2DGS, PGSR and EntON on the
small-scale DTU dataset, as well as ground truth (GT) images in original resolution.
31

<!-- page 32 -->
scene
3DGS
2DGS
PGSR
EntON(knn100)
EntON(knn75)
EntON(knn50)
EntON(knn25)
building1
0.184
0.193
0.197
0.184
0.179
0.182
0.187
building2
0.185
0.160
0.177
0.168
0.169
0.167
0.163
mean
0.185
0.177
0.187
0.176
0.174
0.175
0.175
Table 6: Surface accuracy. Geometric accuracy comparison on the TUM2TWIN dataset
with Chamfer cloud-to-cloud distances ↓in m for points ≤0.5 m from the reference, ac-
cording to the DTU evaluation script. Best results are highlighted as 1st , 2nd , and
3rd . Mean scores are listed at the bottom. The training incorporates 15 000 iterations.
quality of 2DGS (28.95 dB) and PGSR (28.87 dB), while remaining competi-
tive with 3DGS (31.68 dB). These results correspond to up to 8.9% improve-
ment compared to PGSR (9.2% for the best variant) and 8.6% compared to
2DGS (8.9% best), while remaining comparable to 3DGS.
scene
3DGS
2DGS
PGSR
EntON(knn100)
EntON(knn75)
EntON(knn50)
EntON(knn25)
building1
32.36
29.90
30.05
31.55
32.57
32.09
31.94
building2
31.00
28.00
27.69
30.84
30.50
30.56
30.78
mean
31.68
28.95
28.87
31.20
31.54
31.26
31.36
Table 7: Rendering quality comparison on the TUM2TWIN dataset. We report the
PSNR ↑in dB. Best results are highlighted as 1st , 2nd , and 3rd . Mean scores are
listed at the bottom. The training incorporates 15 000 iterations.
Efficiency. Efficiency is evaluated in terms of the number of Gaussians and
training time. EntON achieves a mean number of 2 435 371 Gaussians across
neighborhood sizes, with the most compact variant (knn = 50) using only
2 362 827, clearly more compact than all baselines. This corresponds to up to
29.3% fewer Gaussians compared to 3DGS, 15.1% fewer than 2DGS, and 8.1%
fewer than PGSR for the best variant. Training time for 15 000 iterations
show that EntON requires on average 22.27 min, with the fastest variant
completing in 19.30 min. This corresponds to up to 51.0% faster training
compared to PGSR, while remaining competitive with 2DGS and slightly
slower than 3DGS on average.
Overall, EntON demonstrates a compelling balance between rendering
quality, geometric accuracy, memory efficiency, and computational cost on
large-scale urban scenes. Smaller neighborhood sizes prioritize geometric ac-
curacy and efficiency in terms of compactness, whereas larger neighborhoods
slightly enhance photometric quality, by showing that the method scales ef-
fectively while outperforming 2DGS and PGSR, and remaining competitive
with 3DGS.
32

<!-- page 33 -->
scene
3DGS
2DGS
PGSR
EntON(knn100)
EntON(knn75)
EntON(knn50)
EntON(knn25)
building1
3 622 623
3 457 686
3 495 252
2 716 037
2 693 175
2 500 641
2 475 797
building2
3 037 856
2 087 933
1 625 555
2 306 556
2 247 377
2 225 013
2 278 891
mean
3 330 240
2 772 810
2 560 404
2 511 297
2 470 276
2 362 827
2 377 344
Table 8: Number of Gaussians on the TUM2TWIN dataset.
We report the total
number of Gaussians ↓. Best results are highlighted as 1st , 2nd , and 3rd . Mean scores
are listed at the bottom. The training incorporates 15 000 iterations.
scene
3DGS
2DGS
PGSR
EntON(knn100)
EntON(knn75)
EntON(knn50)
EntON(knn25)
building1
18.35
25.32
40.48
27.75
26.24
22.82
20.56
building2
17.42
22.77
37.97
22.21
20.93
19.19
18.04
mean
17.89
24.05
39.23
24.98
23.59
21.00
19.30
Table 9: Training time comparison on the TUM2TWIN dataset. We report the minutes
for 15 000 iterations in min. Best results are highlighted as 1st , 2nd , and 3rd . Mean
scores are listed at the bottom.
5.3.2. Qualitative Results
Geometric Accuracy. The geometric accuracy of Gaussian centers on the
TUM2TWIN large scale dataset, evaluated using the Chamfer cloud-to-cloud
distance (Figure 10), demonstrate high surface accuracy for EntON and con-
firm the quantitative findings.
Rendering Quality. The rendering results on the TUM2TWIN dataset (Fig-
ure 11) largely mirror the observations from the DTU small-scale dataset.
EntON consistently reconstructs fine details more faithfully, avoiding the
over-reconstructed areas that appear in 2DGS and, to a lesser extent, in
3DGS and PGSR. Flat surfaces, such as the roof and façade in building1 (see
upper enlargement), are accurately captured, as are areas like the grass and
roof in building2. On building1 (lower enlargement), EntO, and also 2DGS,
prunes the Gaussiansof the street lamp, but EntON is able to reconstruct the
vehicle and the underlying ground details with high fidelity regions that are
less accurately represented in 2DGS and PGSR. Despite using fewer Gaus-
sians, EntON achieves high rendering fidelity comparable to 3DGS, which
produces similarly clean reconstructions but relies on a substantially higher
number of Gaussians. 2DGS and PGSR, in contrast, use fewer Gaussians
but generate partially blurred areas. Overall, the TUM2TWIN dataset con-
firms that EntON consistently balances rendering quality with compact and
efficient scene representations, especially for man-made environments.
33

<!-- page 34 -->
building1
building2
3DGS
2DGS
PGSR
EntON
GT
0m
1m
2m
Figure 10:
Geometric accuracy comparison of 3DGS, 2DGS, PGSR and EntON
TUM2TWIN dataset with Chamfer cloud-to-cloud distances ↓. Color values are cropped
at 2 m distance.
34

<!-- page 35 -->
building1
building2
3DGS
2DGS
PGSR
EntON
GT
Figure 11: Rendering quality comparison of 3DGS, 2DGS, PGSR and EntON on the
large-scale TUM2TWIN dataset, as well as ground truth (GT) images in original resolu-
tion.
35

<!-- page 36 -->
6. Ablation Study
This section presents an ablation study analyzing the Eigenentropy of
Gaussians in outlier regions. We analyze the Eigenentropy of Gaussians in
3DGS after the training process based on 4 scenes, whose Gaussian centers
lie more than 1 mm from the reference point cloud to guide the choice of a
pruning threshold. We than compute the mean Eigenentropy of these outlier
Gaussians to guide the selection of the upper threshold for pruning in EntON.
Scene
scan40
scan55
scan106
scan122
Mean Eigenentropy (Gaussians, >1 mm distance)
0.9863
0.9617
0.9594
0.9531
Table 10: Mean Eigenentropy of Gaussians in 3DGS corresponding to points with cloud-
to-cloud distance over 1 mm distance to the reference. Values guide the selection of the
upper Eigenentropy threshold for pruning.
Table 10 reports the mean Eigenentropy of outlier Gaussians across se-
lected scans, showing values around Eigenentropy E = 0.95, which moti-
vates the selection of an upper Eigenentropy threshold These observation is
consistent with the theoretical characteristics of Eigenentropy discussed in
Section 3.2.4. Specifically, Gaussians on well-structured surfaces typically
correspond to near-planar distributions, where E ≤log 2 ≈0.693, whereas
outliers exhibit higher Eigenentropy, approaching values near 1, in line with
locally isotropic or unstructured regions. Figure 12 further visualizes these
outlier Gaussians based on the choosed threshold: Gaussians exceeding the
threshold E = 0.95 are marked in red and are considered candidates for
pruning, while points below the threshold are shown in blue.
scan40
scan55
scan106
scan122
Eigenentropy
Figure 12: Comparison of Eigenentropy of points with a cloud-to-cloud distance (C2C)
over 1mm to the reference point cloud. Blue points are below Eigenentropy E = 0.95, red
points above Eigenentropy E = 0.95 and therefore candidates for pruning.
36

<!-- page 37 -->
7. Discussion
The proposed method, EntON, demonstrates clear superiority in terms
of geometric reconstruction accuracy compared to 3D Gaussian Splatting
(3DGS), while maintaining competitive rendering quality. These improve-
ments are achieved through a targeted, Eigenentropy-aware densification and
pruning strategy using local geometric information, that effectively exploits
local geometric structure of Guassians under the Manhattan-World assump-
tion predominant in man-made environments.
Eigenentropy Distribution. A key insight provided by the experiments is
the pronounced and systematic reduction of mean Eigenentropy across lo-
cal Gaussian neighborhoods. EntON consistently achieves significantly lower
final mean Eigenentropy values compared to the baseline. While 3DGS sta-
bilizes at high Eigenentropy levels (∼0.96) and even exhibits a slight in-
crease over the course of training, EntON initiates a sharp decline shortly
after the introduction of Eigenentropy-guided densification (around itera-
tion 3000) and converges to a stable range of approximately 0.78 to 0.82.
This behavior confirms that the proposed adaptive strategy successfully pro-
motes lower-Eigenentropy, more ordered local neighborhoods. Importantly,
this reduction in Eigenentropy shows a strong correlation with improved
geometric accuracy. Scenes and regions characterized by lower mean Eige-
nentropy consistently exhibit reduced Cloud-to-Cloud (C2C) errors, whereas
high-Eigenentropy neighborhoods are associated with markedly larger ge-
ometric deviations.
The qualitative results (Figure 6) further reveal that
EntON selectively achieves low Eigenentropy, particularly in structured, pre-
dominantly planar neighborhoods, precisely the areas where accurate surface
representation is most critical under the Manhattan-World assumption. Con-
sequently, our strategy preferentially increases Gaussian density in these low-
Eigenentropy (ordered, anisotropic, flat) areas by splitting to better capture
fine geometric details on the object surface, while simultaneously pruning
Gaussians in high-Eigenentropy, disordered, or spherically local neighbor-
hoods. In contrast, 3DGS tends to retain, or even accumulate, redundant
Gaussians in less structured areas, resulting in less efficient scene represen-
tations and lower geometric accuracy. In summary, these findings strongly
confirm the suitability of Eigenentropy as an effective guide for the densifi-
cation and pruning criteria. By explicitly guiding towards low-Eigenentropy
(ordered, anisotropic, flat) neighborhoods and pruning high-Eigenentropy
37

<!-- page 38 -->
(disordered, isotropic, spherical/scattered) regions, EntON achieves a sys-
tematically more structured and geometrically accurate Gaussian distribu-
tion while preserving high rendering quality. The results thereby validate the
central hypothesis: Eigenentropy-aware control enables targeted resource al-
location during optimization and delivers clear gains in geometric accuracy.
Particularly in environments dominated by predominant flat structures.
As discussed, the Eigenentropy-aware densification and pruning strategy
systematically promotes Gaussians in low-Eigenentropy (ordered, anisotropic,
flat) local neighborhoods while suppressing those in high-Eigenentropy (dis-
ordered) regions. This targeted adaptation not only reduces the overall mean
Eigenentropy, but directly translates into measurable benefits across three
central performance metrics: geometric accuracy, rendering quality, and con-
sumption of memory and training time.
Geometric Accuracy. In detail, with respect to geometric accuracy, EntON
achieves an average cloud-to-cloud (C2C) distance of 1.03 mm across all
scenes of the DTU dataset.
This performance is very close to the cur-
rent state-of-the-art method PGSR (1.00 mm) and substantially outperforms
both 3DGS (1.609 mm) and 2DGS (1.331 mm). A scene-wise analysis high-
lights the strengths and limitations of the proposed Eigenentropy-aware strat-
egy. In well-textured, predominantly planar regions (e.g., scan40, scan55,
scan106), EntON achieves accuracy nearly identical to that of PGSR. Sim-
ilarly strong performance is observed for scenes containing rough or diffuse
materials (e.g., scan83, scan105). In contrast, for scenes dominated by reflec-
tive surfaces (e.g., scan63, scan97, scan110), EntON exhibits a slight degra-
dation in accuracy compared to PGSR. This scene dependency reveals a fun-
damental limitation of the neighborhood-based and point-density-dependent
pruning and densification strategy.
The computation of Eigenentropy is
based on the covariance matrix derived from a local Gaussian neighborhood.
However, in specular or texture-poor regions, the initial density of Gaussians
is typically significantly lower, as fewer Gaussians are required to represent
homogeneous color regions.
With a fixed neighborhood size k, the same
number of Gaussian neighbors spans a substantially larger spatial extent in
such low-density regions. As a consequence, the resulting covariance ma-
trix tends to become more spherical, leading to an increased Eigenentropy,
even when the underlying surface is inherently planar or low-Eigenentropy.
These areas are thus erroneously interpreted by the method as disordered
or high-Eigenentropy. As a result, splitting is insufficiently triggered, or not
38

<!-- page 39 -->
triggered at all, since the condition is (not) satisfied.
At the same time,
Gaussians with high Eigenentropy are systematically favored for pruning,
further exacerbating the issue. Together, these effects lead to a persistently
reduced Gaussian density precisely in specular or view-dependent regions,
where a higher density would in fact be beneficial to better enforce multi-
view consistency and to reduce geometric errors, particularly for reflective
or glossy surfaces. This behavior explains the slight but measurable degra-
dation in geometric accuracy observed in such scenes compared to methods
such as PGSR. Nevertheless, EntON achieves near state-of-the-art accuracy
with a significantly reduced number of Gaussians. The method deliberately
concentrates Gaussians in regions where they are most critical for accurate
geometric reconstruction, namely in low-Eigenentropy, structured, and pla-
nar areas, where object surface structure is likely to be present.
Rendering Quality. Regarding rendering quality, EntON attains a mean PSNR
of 34.91 dB over the entire small-scale DTU dataset, thereby slightly sur-
passing 3DGS (34.84 dB). Regarding rendering quality on the small-scale
DTU dataset, EntON attains a mean PSNR of 34.39 dB across all neigh-
borhood sizes, slightly below 3DGS (34.84 dB), but outperforming 2DGS
(32.54,dB) and PGSR (32.32,dB). On the large-scale TUM2TWIN dataset,
EntON achieves a mean PSNR of 31.44 dB across all neighborhood sizes, sur-
passing 2DGS (28.95 dB) and PGSR (28.87 dB) while remaining competitive
with 3DGS (31.68 dB).
In well-textured scans, EntON typically matches or slightly exceeds 3DGS
performance, while in low-texture scenes a minor drop in PSNR can be ob-
served. This scene dependent rendering behavior stems directly from the
Eigenentropy-aware strategy: As discussed in the context of geometric ac-
curacy, lower initial point density in low-texture regions leads to a more
isotropic local covariance structure and consequently higher Eigenentropy.
This results in reduced Gaussian density (limited splitting and increased
pruning), forcing both geometry and photometric appearance to be repre-
sented with even fewer primitives. Consequently, such regions may experi-
ence a slight loss of high-frequency details, manifesting in a marginally lower
PSNR.
In contrast, in well-textured and structured regions the method benefits
substantially from the high Gaussian density achieved there. This enables
efficient and precise representation of dominant features such as sharp edges,
texture, and fine surface details. Consequently, such scenes frequently ex-
39

<!-- page 40 -->
hibit rendering quality that is comparable or even superior to 3DGS, despite
a significantly lower total number of Gaussians. Compared to other methods
such as 2DGS or PGSR, EntON overall preserves remarkably good photo-
realistic quality. The efficient resource allocation, achieved by concentrating
Gaussians in structurally and low-entropy regions, thus provides an excellent
trade-off between rendering quality and expansion of the scene.
The ap-
proach sacrifices only minimal perceptual quality in extremely low-texture/
low-density regions, while benefiting strongly from reduced complexity in the
majority of the scenes.
Efficiency. EntON significantly reduces the mean number of Gaussians re-
quired for scene representation to 215 k (compared to 392 k in 3DGS, 232 k in
PGSR, and 209 k in 2DGS). At the same time, training time for 15 000 itera-
tions ranges from 9.87 to 10.66 min, substantially lower than that using 3DGS
(12.88 min), 2DGS (18.11 min), or PGSR (32.89 min) on the small-scale DTU
dataset. On the large-scale TUM2TWIN dataset, EntON reduces the number
of Gaussians compared to baselines, with a mean of 2.36–2.51 million across
all neighborhood sizes (compared to 3.33M in 3DGS, 2.77M in 2DGS, and
2.56M in PGSR). However, the absolute number of Gaussians remains sub-
stantially higher than in DTU, which makes neighborhood calculations more
expensive and explains why training times (19.30 to 24.98 min) are not as
low as on DTU, despite still being considerably faster than 2DGS (24.05 min)
and PGSR (39.23 min). This efficiency gain is a direct consequence of the
Eigenentropy-aware strategy: focused densification in low-Eigenentropy re-
gions, where object surface information is predominantly concentrated (i.e.,
structured, predominantly planar areas), combined with the systematic re-
moval of redundant Gaussians in disordered/ spherical neighborhoods. Over-
all this results in a geometrically meaningful and compact distribution of the
Gaussians in the scene. By eliminating unnecessary primitives, the method
not only reduces memory footprint but also accelerates the training pro-
cess, as fewer parameters overall need to be optimized, in particularly in
non-surface regions. Consequently, EntON achieves high geometric and ren-
dering quality at high efficiency regarding memory and training time, which
is a key advantage for scalable applications and resource-constrained envi-
ronments.
Limitations. Despite the strong performance of EntON across a wide range of
scenes, the proposed method exhibits two principal limitations. First, EntON
40

<!-- page 41 -->
is particularly designed for man-made scenes like in urban environments and
relies on the assumption that meaningful geometric structure can be inferred
from local 3D shape of Gaussian neighborhoods under a Manhattan-World
prior. This assumption is well justified for scenes dominated by nearly-planar,
anisotropic structures such as walls, floors, and architectural elements. How-
ever, it becomes less appropriate for environments characterized by highly
curved, irregular, or scattered geometry, such as vegetation or organic ob-
jects.
In such scenes, local Gaussian neighborhoods are inherently more
isotropic and less aligned with dominant object surface, making the distinc-
tion between low- and high-Eigenentropy neighborhoods less indicative of
true surface relevance. As a result, the EntON may suppress Gaussians in this
scenes that are necessary to faithfully represent complex non-planar geome-
try. Second, the method is sensitive to variations in local Gaussian density
due to its use of a fixed neighborhood size k for Eigenentropy computation.
In regions with low Gaussian density, commonly occurring on reflective or
texture-less surfaces, the same number of neighbors spans a substantially
larger spatial extent. This leads to more spherical covariance estimates and
higher Eigenentropy values, even when the underlying surface is locally pla-
nar. Consequently, such regions may be misdefined as disordered, resulting
in insufficient densification and increased pruning. These limitations high-
light that EntON is best suited for structured, predominantly planar scenes
with sufficiently dense Gaussian distributions, and they point toward adap-
tive neighborhood strategies and geometry–appearance coupling as promising
directions for future work.
In summary, the Eigenentropy-aware densification and pruning strategy
EntON successfully promotes a more structured and surface-aligned distri-
bution of Gaussians. By preferentially densifying in low-Eigenentropy neigh-
borhoods (indicating ordered, anisotropic, flat regions) and pruning Gaus-
sians in high-Eigenentropy, disordered regions, EntON achieves substantially
lower Eigenentropy compared to 3DGS, as evidenced by the rapid and sta-
ble decline starting early in training. This targeted adaptation translates
into clear performance benefits across the key dimensions: high geometric
accuracy, competitive to slightly superior rendering quality (preserving fine
details despite reduced complexity), and markedly improved consumption of
memory and training time efficiency (reduction of number of Gaussians count
to present the scene and shorter training time). The observed scene depen-
dent limitations, particularly the slight degradation on reflective/specular
surfaces, are attributable to the Gaussian spatial density sensitivity of the
41

<!-- page 42 -->
computed Eigenentropy from a fixed number of neigherst neighbors, which
can lead to under-densification in regions with a low spatial density of Gaus-
sians. Nevertheless, the overall results strongly validate Eigenentropy as an
effective guide for local geometric ordered densification, enabling more in-
telligent resource allocation without additional supervision or complex con-
straints.
8. Conclusion
We present EntON, a geometry-guided, Eigenentropy-aware alternating
densification strategy for 3D Gaussian Splatting that leverages local 3D
structural geometry to guide adaptive splitting and pruning.
Gaussians
in low-Eigenentropy (ordered, flat) neighborhoods are preferentially densi-
fied, while those in high-Eigenentropy (disordered, spherical) neighborhoods
are pruned.
By explicitly exploiting the local 3D neighborhood of Gaus-
sians, EntON aligns the densification process with geometric regularities
commonly found in man-made structures and urban scenes. By alternat-
ing between standard gradient-based and Eigenentropy-aware densification,
EntON captures fine geometric details with compact, surface-aligned Gaus-
sian representations, while avoiding unnecessary scene expansion of Gaus-
sians in high-Eigenentropy areas.
As a result, it delivers high geometric
accuracy, while maintaining photometric fidelity. Surface-aligned densifica-
tion and systematic pruning lead to fewer Gaussians, lower memory usage,
and shorter training time compared to conventional 3DGS and other base-
line methods. Experiments on two benchmark datasets demonstrate that
by guiding densification toward low-Eigenentropy regions, EntON enables
3D scene reconstructions that are geometrically and photometrically accu-
rate, memory- and computation-efficient, and suitable for both small- and
large-scale scenes.
References
[1] Bueno, M., Bosché, F., González-Jorge, H., Martínez-Sánchez, J., Arias,
P., 2018. 4-plane congruent sets for automatic registration of as-is 3d
point clouds with 3d bim models. Automation in Construction 89, 120–
134.
[2] Chen, D., Li, H., Ye, W., Wang, Y., Xie, W., Zhai, S., Wang, N.,
Liu, H., Bao, H., Zhang, G., 2024a. Pgsr: Planar-based gaussian splat-
42

<!-- page 43 -->
ting for efficient and high-fidelity surface reconstruction. arXiv preprint
arXiv:2406.06521 .
[3] Chen, Y., Wu, Q., Lin, W., Harandi, M., Cai, J., 2024b. Hac: Hash-
grid assisted context for 3d gaussian splatting compression. European
Conference on Computer Vision , 422–438.
[4] Coughlan, J., Yuille, A.L., 2000.
The manhattan world assumption:
Regularities in scene statistics which enable bayesian inference.
Ad-
vances in Neural Information Processing Systems 13.
[5] Coughlan, J.M., Yuille, A.L., 1999.
Manhattan world: Compass di-
rection from a single image by bayesian inference. Proceedings of the
seventh IEEE international conference on computer vision 2, 941–947.
[6] Dai, P., Xu, J., Xie, W., Liu, X., Wang, H., Xu, W., 2024. High-quality
surface reconstruction using gaussian surfels. ACM SIGGRAPH 2024
Conference Papers , 1–11.
[7] Dittrich, A., Weinmann, M., Hinz, S., 2017. Analytical and numerical
investigations on the accuracy and robustness of geometric features ex-
tracted from 3d point cloud data. ISPRS journal of photogrammetry
and remote sensing 126, 195–208.
[8] Fan, Z., Wang, K., Wen, K., Zhu, Z., Xu, D., Wang, Z., et al., 2024.
Lightgaussian: Unbounded 3d gaussian compression with 15x reduction
and 200+ fps. Advances in neural information processing systems 37,
140138–140158.
[9] Frome, A., Huber, D., Kolluri, R., Bülow, T., Malik, J., 2004. Rec-
ognizing objects in range data using regional point descriptors. Com-
puter Vision-ECCV 2004: 8th European Conference on Computer Vi-
sion, Prague, Czech Republic, May 11-14, 2004. Proceedings, Part III 8
, 224–237.
[10] Guédon, A., Lepetit, V., 2024. Sugar: Surface-aligned gaussian splat-
ting for efficient 3d mesh reconstruction and high-quality mesh render-
ing. Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , 5354–5363.
43

<!-- page 44 -->
[11] Hahlbohm, F., Franke, L., Eisemann, M., Magnor, M., 2026. Faster-
gs: Analyzing and improving gaussian splatting optimization.
arXiv
preprint arXiv:2602.09999 .
[12] Hanson, A., Tu, A., Lin, G., Singla, V., Zwicker, M., Goldstein, T.,
2025. Speedy-splat: Fast 3d gaussian splatting with sparse pixels and
sparse primitives.
Proceedings of the Computer Vision and Pattern
Recognition Conference , 21537–21546.
[13] Held, J., Vandeghen, R., Deliege, A., Hamdi, A., Rebain, D., Giancola,
S., Cioppa, A., Vedaldi, A., Ghanem, B., Tagliasacchi, A., et al., 2025.
Triangle splatting for real-time radiance field rendering. Thirteenth In-
ternational Conference on 3D Vision .
[14] Hillemann, M., Weinmann, M., Mueller, M.S., Jutzi, B., 2019.
Au-
tomatic extrinsic self-calibration of mobile mapping systems based on
geometric 3d features. Remote sensing 11, 1955.
[15] Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S., 2024. 2d gaussian
splatting for geometrically accurate radiance fields. ACM SIGGRAPH
2024 conference papers , 1–11.
[16] Jäger, M., Hillemann, M., Jutzi, B., 2025. Featuregs: Eigenvalue-feature
optimization in 3d gaussian splatting for geometrically accurate and
artifact-reduced reconstruction. ISPRS Open Journal of Photogramme-
try and Remote Sensing , 100100.
[17] Jäger, M., Jutzi, B., 2023. 3d density-gradient based edge detection on
neural radiance fields (nerfs) for geometric reconstruction. International
Archives of the Photogrammetry, Remote Sensing and Spatial Informa-
tion Sciences 48, 1.
[18] Jensen, R., Dahl, A., Vogiatzis, G., Tola, E., Aanæs, H., 2014. Large
scale multi-view stereopsis evaluation. Proceedings of the IEEE confer-
ence on computer vision and pattern recognition , 406–413.
[19] Jutzi, B., Gross, H., 2009.
Nearest neighbour classification on laser
point clouds to gain object structures from buildings. The International
Archives of the Photogrammetry, Remote Sensing and Spatial Informa-
tion Sciences 38, 4–7.
44

<!-- page 45 -->
[20] Jäger, M., Landgraf, S., Jutzi, B., 2025. Density uncertainty quantifi-
cation with nerf-ensembles: Impact of data and scene constraints. In-
ternational Journal of Applied Earth Observation and Geoinformation
137, 104406.
[21] Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G., 2023. 3d gaussian
splatting for real-time radiance field rendering. ACM Trans. Graph. 42,
139–1.
[22] Kim, S., Lee, K., Lee, Y., 2024. Color-cued efficient densification method
for 3d gaussian splatting. Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , 775–783.
[23] Lee, J.W., Lim, H., Yang, S., Choi, J., 2025. Micro-splatting: Maximiz-
ing isotropic constraints for refined optimization in 3d gaussian splat-
ting. arXiv preprint arXiv:2504.05740 .
[24] Lee, S., Kim, Y.G., Sasse, S., Borges, T.M., Sanchez, Y., Ryu, E.S.,
Schierl, T., Hellge, C., 2026.
Gaussianpop: Principled simplification
framework for compact 3d gaussian splatting via error quantification.
arXiv preprint arXiv:2602.06830 .
[25] Li, S., Wu, C., Li, H., Gao, X., Liao, Y., Yu, L., 2026a.
Gscodec
studio: A modular framework for gaussian splat compression. IEEE
Transactions on Circuits and Systems for Video Technology .
[26] Li, Y., Jia, Z., Zhang, Y., Hao, Q., Zhang, S., 2026b.
Objsplat:
Geometry-aware gaussian surfels for active object reconstruction. arXiv
preprint arXiv:2601.06997 .
[27] Li, Z., Huang, J., Chen, R., Che, Y., Guo, Y., Liu, T., Karray, F., Gong,
M., 2024a. Urbangs: Semantic-guided gaussian splatting for urban scene
reconstruction. arXiv preprint arXiv:2412.03473 .
[28] Li, Z., Müller, T., Evans, A., Taylor, R.H., Unberath, M., Liu, M.Y.,
Lin, C.H., 2023. Neuralangelo: High-fidelity neural surface reconstruc-
tion. Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , 8456–8465.
[29] Li, Z., Yao, S., Chu, Y., Garcia-Fernandez, A.F., Yue, Y., Lim, E.G.,
Zhu, X., 2024b. Mvg-splatting: Multi-view guided gaussian splatting
45

<!-- page 46 -->
with adaptive quantile-based geometric consistency densification. arXiv
preprint arXiv:2407.11840 .
[30] Lowe, D.G., 2004. Distinctive image features from scale-invariant key-
points. International journal of computer vision 60, 91–110.
[31] Mallick, S.S., Goel, R., Kerbl, B., Steinberger, M., Carrasco, F.V.,
De La Torre, F., 2024. Taming 3dgs: High-quality radiance fields with
limited resources. SIGGRAPH Asia 2024 Conference Papers , 1–11.
[32] Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoor-
thi, R., Ng, R., 2021. Nerf: Representing scenes as neural radiance fields
for view synthesis. Communications of the ACM 65, 99–106.
[33] Oechsle, M., Peng, S., Geiger, A., 2021. Unisurf: Unifying neural im-
plicit surfaces and radiance fields for multi-view reconstruction. Proceed-
ings of the IEEE/CVF International Conference on Computer Vision ,
5589–5599.
[34] Patle, G., Girgaonkar, N., Somraj, N., Soundararajan, R., 2025. Ad-gs:
Alternating densification for sparse-input 3d gaussian splatting. arXiv
preprint arXiv:2509.11003 .
[35] Ren, S., Wen, T., Fang, Y., Lu, B., 2025. Fastgs: Training 3d gaussian
splatting in 100 seconds. arXiv preprint arXiv:2511.04283 .
[36] Rusu, R.B., Blodow, N., Beetz, M., 2009. Fast point feature histograms
(fpfh) for 3d registration. 2009 IEEE international conference on robotics
and automation , 3212–3217.
[37] Stuart, L.A., Morton, A., Stavness, I., Pound, M.P., 2025.
3dgs-to-
pc: 3d gaussian splatting to dense point clouds, in: Proceedings of the
IEEE/CVF International Conference on Computer Vision, pp. 3730–
3739.
[38] Tombari, F., Salti, S., Di Stefano, L., 2010. Unique signatures of his-
tograms for local surface description. Computer Vision–ECCV 2010:
11th European Conference on Computer Vision, Heraklion, Crete,
Greece, September 5-11, 2010, Proceedings, Part III 11 , 356–369.
46

<!-- page 47 -->
[39] Wang, J., Zhou, P., Li, C., Quan, R., Qin, J., 2025. Low-frequency first:
Eliminating floating artifacts in 3d gaussian splatting. arXiv preprint
arXiv:2508.02493 .
[40] Wang, P., Liu, L., Liu, Y., Theobalt, C., Komura, T., Wang, W., 2021.
Neus: Learning neural implicit surfaces by volume rendering for multi-
view reconstruction. arXiv preprint arXiv:2106.10689 .
[41] Weinmann, M., Jäger, M.A., Wursthorn, S., Jutzi, B., Hübner, P., 2020.
3d indoor mapping with the microsoft hololens: qualitative and quanti-
tative evaluation by means of geometric features. ISPRS Annals of the
Photogrammetry, Remote Sensing and Spatial Information Sciences 1,
165–172.
[42] Weinmann, M., Jutzi, B., Hinz, S., Mallet, C., 2015a. Semantic point
cloud interpretation based on optimal neighborhoods, relevant features
and efficient classifiers. ISPRS Journal of Photogrammetry and Remote
Sensing 105, 286–304.
[43] Weinmann, M., Jutzi, B., Mallet, C., 2014. Semantic 3d scene inter-
pretation: A framework combining optimal neighborhood size selection
with relevant features. ISPRS Annals of the Photogrammetry, Remote
Sensing and Spatial Information Sciences 2, 181–188.
[44] Weinmann, M., Jutzi, B., Mallet, C., 2017.
Geometric features and
their relevance for 3d point cloud classification. ISPRS Annals of the
Photogrammetry, Remote Sensing and Spatial Information Sciences 4,
157–164.
[45] Weinmann, M., Schmidt, A., Mallet, C., Hinz, S., Rottensteiner, F.,
Jutzi, B., 2015b. Contextual classification of point cloud data by ex-
ploiting individual 3d neigbourhoods. ISPRS Annals of the Photogram-
metry, Remote Sensing and Spatial Information Sciences; II-3/W4 2,
271–278.
[46] Weinmann, M., Urban, S., Hinz, S., Jutzi, B., Mallet, C., 2015c. Dis-
tinctive 2d and 3d features for automated large-scale scene analysis in
urban areas. Computers & Graphics 49, 47–57.
[47] Wysocki, O., Schwab, B., Biswanath, M.K., Greza, M., Zhang, Q., Zhu,
J., Froech, T., Heeramaglore, M., Hijazi, I., Kanna, K., et al., 2026.
47

<!-- page 48 -->
Tum2twin: Introducing the large-scale multimodal urban digital twin
benchmark dataset. ISPRS Journal of Photogrammetry and Remote
Sensing 232, 810–830.
[48] Yariv, L., Gu, J., Kasten, Y., Lipman, Y., 2021.
Volume rendering
of neural implicit surfaces. Advances in Neural Information Processing
Systems 34, 4805–4815.
[49] Ye, Z., Li, W., Liu, S., Qiao, P., Dou, Y., 2024. Absgs: Recovering
fine details in 3d gaussian splatting.
Proceedings of the 32nd ACM
International Conference on Multimedia , 1053–1061.
[50] Yu, Z., Chen, A., Huang, B., Sattler, T., Geiger, A., 2024a.
Mip-
splatting:
Alias-free 3d gaussian splatting.
Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition
, 19447–19456.
[51] Yu, Z., Sattler, T., Geiger, A., 2024b. Gaussian opacity fields: Efficient
adaptive surface reconstruction in unbounded scenes. ACM Transactions
on Graphics (ToG) 43, 1–13.
[52] Zhang, J., Zhan, F., Xu, M., Lu, S., Xing, E., 2024. Fregs: 3d gaussian
splatting with progressive frequency regularization. Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition ,
21424–21433.
[53] Zhang, Q., Wysocki, O., Jutzi, B., 2025. Gs4buildings: Prior-guided
gaussian splatting for 3d building reconstruction. ISPRS Annals of the
Photogrammetry, Remote Sensing and Spatial Information Sciences 10,
249–256.
48
