<!-- page 1 -->
Gaussian Set Surface Reconstruction through Per-Gaussian Optimization
Zhentao Huang
School of Computer Science, University of Guelph
zhentao@uoguelph.ca
Di Wu
Faculty of Science and Technology, University of Macau
diwu96063@gmail.com
Zhenbang He
Irving K. Barber Faculty of Science, UBC Okanagan
zbhe96@student.ubc.ca
Minglun Gong
School of Computer Science, University of Guelph
minglun@uoguelph.ca
Figure 1. Comparison of Gaussian primitives distributions from 2DGS, PGSR, and our method across two scenes. Gaussian centers are
visualized as colored points, with hue indicating surface distance error. Areas where the white background is visible through the Gaussians
denote incomplete surface coverage. Red and green boxes highlights regions with improved accuracy and coverage, respectively. Quanti-
tative and visual results confirm that our approach achieves both a lower reconstruction error and a more complete Gaussian representation.
Abstract
3D Gaussian Splatting (3DGS) effectively synthesizes
novel views through its flexible representation, yet fails to
accurately reconstruct scene geometry. While modern vari-
ants like PGSR introduce additional losses to ensure proper
depth and normal maps through Gaussian fusion, they still
neglect individual placement optimization. This results in
unevenly distributed Gaussians that deviate from the la-
tent surface, complicating both reconstruction refinement
and scene editing. Motivated by pioneering work on Point
Set Surfaces, we propose Gaussian Set Surface Reconstruc-
tion (GSSR), a method designed to distribute Gaussians
evenly along the latent surface while aligning their dom-
inant normals with the surface normal.
GSSR enforces
fine-grained geometric alignment through a combination of
pixel-level and Gaussian-level single-view normal consis-
tency and multi-view photometric consistency, optimizing
both local and global perspectives. To further refine the
representation, we introduce an opacity regularization loss
to eliminate redundant Gaussians and apply periodic depth-
and normal-guided Gaussian reinitialization for a cleaner,
more uniform spatial distribution. Our reconstruction re-
sults demonstrate significantly improved geometric preci-
sion in Gaussian placement, enabling intuitive scene editing
and efficient generation of novel Gaussian-based 3D envi-
ronments. Extensive experiments validate GSSR’s effective-
ness, showing enhanced geometric accuracy while preserv-
ing high-quality rendering performance. Our code will be
released upon acceptance of the paper.
1. Introduction
Novel View Synthesis (NVS) remains a fundamental
challenge in computer vision and graphics, enabling pho-
torealistic scene rendering from sparse inputs for applica-
tions ranging from eXtended Reality (XR) to autonomous
1
arXiv:2507.18923v2  [cs.CV]  6 Mar 2026

<!-- page 2 -->
Figure 2. Overview of the proposed GSSR pipeline. Given multi-view posed images and initialized 3D Gaussians, we render the depth
map, depth-based normals and alpha-blended normals. The optimization stage includes four major components: (1) single-view normal
consistency, (2) multi-view photometric consistency (3) RGB rendering loss, and (4) opacity regularization. Additionally, Gaussians are
periodically resampled using our view-based opacity-guided strategy, resulting in a more uniform and accurate distribution.
systems. Modern NVS approaches can be broadly catego-
rized by their treatment of scene geometry: explicit model-
ing methods that reconstruct accurate 3D surfaces [28, 29]
and implicit scene representations that bypass explicit re-
construction [22,24,26].
With advances in deep neural networks, research on
implicit scene representations has gained momentum. In
particular, Neural Radiance Fields (NeRF) [26] leverages
MLPs to model scenes volumetrically, achieving unprece-
dented rendering quality. However, these methods incur sig-
nificant costs: slow training, high computational demands,
and limited editability.
3D Gaussian Splatting [22] overcomes these limita-
tions through its explicit, Gaussian-based representation.
By employing differentiable rasterization of 3D Gaussians,
it achieves real-time rendering while maintaining NeRF-
comparable visual quality. The discrete nature of Gaussians
offers advantages in scene manipulation and surface recon-
struction compared to continuous implicit representations.
However, standard 3DGS prioritizes rendering quality over
geometric precision, often producing misaligned Gaussian
distributions that poorly approximate physical surfaces. Re-
cent extensions like 2DGS [16], PGSR [3], and GauSurf
[31] address this through geometric regularization, optimiz-
ing Gaussians to produce accurate depth and normal maps.
While these methods improve surface estimation, they fo-
cus on aggregate outputs rather than individual Gaussian
placement. Consequently, the Gaussians remain unevenly
distributed and misaligned with underlying surfaces, limit-
ing their effectiveness for geometry-sensitive applications
including scene editing and dynamic object deformation.
Inspired by the foundational work on Point Set Sur-
faces [1], we propose Gaussian Set Surface Reconstruction
(GSSR), a method that optimizes Gaussian distributions by:
(1) anchoring centers to latent scene surfaces, (2) enforcing
spatial uniformity, and (3) aligning dominant normals with
surface geometry. GSSR’s explicit geometric formulation
enables direct compatibility with point-based manipulation
techniques, opening new possibilities for precise scene edit-
ing, animation workflows, and dynamic scene processing –
effectively bridging the gap between neural rendering qual-
ity and practical 3D content creation needs.
GSSR optimize individual Gaussian location and orien-
tation through two key mechanisms: (1) The application of
explicit geometric constraints derived from estimated pla-
nar structures and surface smoothness, encouraging Gaus-
sians to coalesce onto latent scene surfaces; and (2) A joint
optimization of opacity and position coupled with adap-
tive pruning, which aggressively eliminates Gaussians with
low contributions while strategically repositioning others
at underrepresented areas. Our experiments across multi-
ple datasets demonstrate that our approach yields a cleaner
scene representation with more geometrically consistent el-
lipsoids while preserving high rendering quality.
In summary, our key contributions include:
• GSSR, a 3DGS framework optimizing Gaussians for
uniform placement and precise surface alignment
• A geometric regularization technique that advances
beyond
prior
3DGS
methods
in
per-Gaussian
depth/normal accuracy and visual coherence
• An opacity-position optimization strategy that prunes
redundancies while enforcing surface adherence
• State-of-the-art Gaussian accuracy and completeness
across three datasets, without rendering quality loss
2

<!-- page 3 -->
2. Related Work
2.1. Gaussian Splatting for 3D Reconstruction
Due to the explicit Gaussian ellipsoids representation of
3DGS, many effective works [3, 5, 8, 14, 16, 30, 31, 34] that
focus on applying geometric regularization to improve ge-
ometry accuracy are proposed. SuGaR [14] introduces a
signed distance function and density supervision to guide
Gaussians toward object surfaces. 2DGS [16] replaces vol-
umetric ellipsoids with the elliptical disk-shaped Gaussians,
which better align with local geometry and improve sur-
face fidelity during rendering.
GOF [35] integrates im-
plicit neural representation within the Gaussian framework
to enhance geometric detail and multi-view consistency.
PGSR [3] introduces a depth estimation strategy and geo-
metric constraints that yield smoother and more accurate
surface reconstruction. GausSurf [31] incorporates geomet-
ric guidance by combining patch-match MVS for texture-
rich regions and normal priors for texture-less areas. Addi-
tionally, methods like GSDF [34] and NeuSG [4] combine
3DGS with Signed Distance Field (SDF) networks to im-
prove reconstruction quality. While these approaches effec-
tively improve surface smoothness and depth accuracy, they
often result in overly dense and irregular Gaussian distribu-
tions. In contrast, our method not only enhances surface
reconstruction but also produces a cleaner and more spa-
tially consistent Gaussian representation, while maintaining
high-fidelity rendering quality.
2.2. Scene Editing
3D Gaussian Splatting offers photorealistic rendering
and holds great promise for applications in XR, content cre-
ation, and digital twins. However, its ellipsoidal representa-
tion lacks geometric consistency with real-world surfaces,
making intuitive and semantically meaningful scene editing
difficult compared to mesh-based approaches. Recent meth-
ods such as GaussianEditor [6], Gaussian Grouping [33],
Point’n Move [17], and Feng et al. [10] explore 3DGS edit-
ing through semantic prompts, mask-based grouping, or im-
proved Gaussian splitting, but are largely limited to basic
object-level operations such as removal, rotation, and trans-
lation. In parallel, some efforts have explored scene editing
in NeRF-based representations [15, 18, 32, 36], addressing
similar challenges of geometry-awareness and user interac-
tion. While promising, these approaches still face limita-
tions due to implicit nature of NeRF and the lack of con-
trollable structure. In this work, we aim to align the gener-
ated 3D Gaussians as closely as possible with the physical
surfaces, thereby enabling more reliable and flexible editing
using existing point-based manipulation techniques.
3. 3D Gaussian Splatting Preliminaries
3D Gaussian Splatting builds on Elliptical Weighted Av-
erage (EWA) splatting [37], extended with a differentiable
formulation [22] to optimize both the number and parame-
ters of Gaussians for scene representation. Each Gaussian
is defined by its center x ∈R3, opacity α ∈[0, 1], covari-
ance matrix in world space Σi ∈R3×3, and view-dependent
color via 16 SH coefficients. During rendering, the final
pixel color C is computed by compositing N depth-sorted
2D Gaussians:
C(p) =
N
X
i=1
Tiαici,
Ti =
i−1
Y
j=1
(1 −αjG2D
j
(p))
(1)
where ci represents the view-dependent color, and G2D
j
is
the projected 2D Gaussian distribution. To render depth,
previous methods [7,21] replace ci with the Gaussian center
depth zi in Equation 1. However, this yields biased, curved
surfaces. Following PGSR [3], we instead computed unbi-
ased depth using Gaussian surface normals ni, aligned with
the minimum scale axis. The per-Gaussian tangent plane
distance is:
di =
 R⊤
c (µi −tc)
⊤R⊤
c ni,
(2)
where Rc is the world-to-camera rotation and tc is the cam-
era origin. The global depth is then computed via ray-plane
intersection:
D(p) =
d(p)
N(p)⊤K−1 ˜p,
(3)
with K the intrinsic matrix and ˜p the homogeneous pixel
coordinate. This formulation ensures geometry-consistent
depth estimation independent of Gaussian density.
4. Methodology
Given posed input images, we reconstruct a Gaussian
Set Surface (GSS) – a collection of Gaussians that are uni-
formly distributed and precisely aligned with underlying
scene surfaces. This representation combines the geomet-
ric manipulability of Point Set Surfaces [1] with 3DGS’s
strengths in photorealistic novel view synthesis. Our Gaus-
sian Set Surface reconstruction pipeline consists of three
key components: (1) geometric regularization for accurate
depth and normal estimation, (2) per-Gaussian optimization
to refine individual geometric properties, and (3) enhanced
density control enforcing uniform spatial distribution.
4.1. Geometric Regularization
4.1.1
Flattening 3D Gaussian
Accurately capturing real-world scene geometry using 3D
Gaussians is inherently challenging. To better align with the
true surface structure, the Gaussians are instead flattened
3

<!-- page 4 -->
into 2D representations, allowing them to more precisely
conform to the underlying geometry of the scene. Rather
than initializing with 2D Gaussian Splatting (2DGS) [16], a
scale loss Ls is introduced to progressively compress each
Gaussian ellipsoid along its smallest scale axis. This pro-
cess effectively flattens the ellipsoid into a plane that best
approximates the underlying surface geometry. Following
the approach in [3, 4], a penalty is applied directly to the
minimum scale of each Gaussian to enforce flattening:
Ls = ∥min(s1, s2, s3)∥1 ,
(4)
where s1, s2, s3 represents the scale parameters of each
Gaussian ellipsoid.
4.1.2
Single-View Normal Consistency
As illustrated in Figure 3(a), a locally discontinuous
Gaussian-rendered plane may exhibit a smooth normal
field, leading to inconsistency between local surface nor-
mals and depth geometry. To address this issue, encouraged
by prior works [5, 13, 21, 30], a single-view normal loss is
introduced to enforce local geometric consistency for every
pixel p in the image domain Ω:
Lnormal = 1
Ω
X
p∈Ω
∥ndepth(p) −nrendered(p)∥1 ,
(5)
where ndepth(p) denotes the surface normal estimated from
depth gradients of neighboring pixels, and nrendered(p) de-
notes the normal rendered from the Gaussian sets.
4.1.3
Multi-View Photometric Consistency
Photometric consistency is a widely used supervision sig-
nal in Multi-View Stereo (MVS) frameworks, leveraging
color similarity across views to constrain geometry [11,12,
19]. Among various formulations, the Normalized Cross-
Correlation (NCC) metric is particularly robust to illumi-
nation variation and exposure differences.
In this work,
the photometric NCC loss is adpoted to enforce multi-view
consistency by comparing image patches in neighboring
views. For each pixel pr in the reference view, the corre-
sponding pixel pn in a neighboring view is computed using
a plane-induced homography:
pn
=
Hrn · pr,
(6)
Hrn
=
K

Rrn −Trnn⊤
r
dr

K−1
r ,
(7)
where Rrn and Trn denote the relative rotation and trans-
lation from the reference view to the neighboring view, and
nr, dr are the rendered surface normal and plane distance,
respectively. As illustrated in Figure 4(a), after computing
Figure 3. Illustration of (a) Lnormal and (b) Lnormal−G. Gaus-
sian Normal is derived from the direction of the Gaussian’s mini-
mum scale axis. Rendered Normal is computed via alpha blend-
ing of Gaussian normals along each pixel ray. Depth-Based Nor-
mal is estimated from the depth gradients of neighboring pixels.
the corresponding patch in the neighboring view, we fol-
lowed the forward and backward projection error weighted
NCC loss in PGSR [3]:
ϕ(pr) = ∥pr −HnrHrnpr∥
(8)
w(pr) =
(
1
exp(ϕ(pr)),
if ϕ(pr) < 1
0,
if ϕ(pr) ≥1
(9)
Lmv = 1
Ω
X
pr∈Ω
w(pr) (1 −NCC(Ir(pr), In(Hrnpr))) ,
(10)
where ϕ(pr) represents the forward and backward projec-
tion error. If this error exceeds a predefined threshold, the
pixel is considered occluded or associated with significant
geometric inconsistency.
4.2. Instance-Level Gaussian Optimization
Although geometric regularization enforces scene-level
accuracy, it often fails to achieve satisfactory instance-level
accuracy. Specifically, the Gaussians tend to form a thick
layer surrounding the actual object surface, as illustrated
in Figure 5. This is primarily because the pixel-level ge-
ometric loss only constrains the alpha-blended depth and
normal, without explicitly supervising individual Gaussian
ellipsoids.
To address this issue, we further incorporate
geometric constraints to enforce single-view normal con-
sistency and multi-view photometric consistency on each
4

<!-- page 5 -->
Figure 4.
Illustration of pixel-level (a) and Gaussian-level (b)
multi-view photometric loss.
Gaussian instance. To filter out occluded Gaussians, we
select only those whose center depth lies in front of the
rendered depth map within a predefined tolerance for each
training view.
4.2.1
Bilaterally Weighted Normal Loss
To enforce the single-view normal loss, a penalty is applied
to the difference between each filtered Gaussian’s normal
and its depth-inferred normal at the center pixel.
How-
ever, directly adding this loss can introduce instability dur-
ing optimization, especially in the presence of incorrect,
near-transparent, or very small Gaussians. To mitigate this
issue, we adopt a bilaterally weighted normal loss, where
the weights are determined by each Gaussian’s opacity and
splat size:
Lnormal-G
=
1
N
N
X
i=1
ψ(i) ∥ndepth(pi) −ni∥1 , (11)
ψ(i)
=
αi · ri,
(12)
where pi denotes the 2D pixel location obtained by project-
ing the 3D center of the ith Gaussian onto the image plane,
ni is the Gaussian’s normal in camera space, and ψ(i) is a
Figure 5. Comparison of Gaussian density distribution between
PGSR and our method. Our approach produces lower Gaussian
density, especially on planar regions (see red arrows), where the
splats are more compact and narrowly distributed, demonstrating
better geometric compactness and representation efficiency.
visibility-based weight defined as the product of opacity αi
and the projected splat radius ri.
4.2.2
Bilaterally Weighted Photometric Loss
As illustrated in Figure 4(b), the photometric NCC loss is
employed to enforce multi-view consistency by comparing
local image patches sampled around the projected centers
of depth-filtered Gaussians. The same bilateral weight ψ is
applied to each Gaussian instance filtered by depth.
Hrn−G = K

Rrn −Trnn⊤
i
di

K−1
r ,
(13)
Lmv-G = 1
N
X
i∈N
ψ(i) (1 −NCC(Ir(pi), In(Hrn−G · pi))) ,
(14)
This loss encourages the reconstructed geometry to
align with phtometrically consistent regions across views,
thereby improving the geometric accuracy of individual
Gaussian instances.
4.3. Gaussian Density Control
In this framework, the original gradient-based Gaussian
densification scheme from 3DGS is retained, while an ad-
5

<!-- page 6 -->
Table 1. Top: Reconstruction on DTU (Chamfer Distance ↓). Middle: Reconstruction on DTU (Accuracy of Gaussian Centroid ↓). Bottom:
Reconstruction on DTU (Completeness of Gaussian ↓). Red, orange and yellow backgrounds denote the best, second-best, and third-best
results respectively.
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
2DGS* [16]
0.49
0.79
0.34
0.42
0.95
0.95
0.83
1.25
1.24
0.64
0.62
1.34
0.44
0.69
0.48
0.76
GOF* [35]
0.49
0.83
0.36
0.38
1.33
0.87
0.73
1.24
1.32
0.66
0.73
1.26
0.52
0.82
0.51
0.80
PGSR* [3]
0.34
0.55
0.39
0.35
0.78
0.58
0.49
1.09
0.63
0.59
0.47
0.50
0.30
0.37
0.34
0.52
GausSurf† [31]
0.35
0.55
0.34
0.34
0.77
0.58
0.51
1.10
0.69
0.60
0.43
0.49
0.32
0.40
0.37
0.52
Ours
0.33
0.57
0.37
0.33
0.78
0.62
0.51
1.11
0.68
0.59
0.48
0.55
0.30
0.38
0.35
0.53
2DGS*
0.66
1.02
0.55
0.60
0.88
1.03
0.92
0.55
0.95
0.47
0.52
0.99
0.63
0.42
0.41
0.71
GOF*
0.98
1.28
0.99
0.85
1.11
1.29
1.08
0.77
1.10
0.62
0.65
1.05
0.79
0.48
0.61
0.91
PGSR*
0.58
0.68
0.85
0.71
0.74
0.64
0.68
0.51
0.74
0.45
0.46
0.60
0.46
0.37
0.37
0.59
Ours
0.32
0.47
0.46
0.29
0.56
0.42
0.39
0.43
0.52
0.33
0.26
0.33
0.21
0.26
0.25
0.37
2DGS*
0.92
0.87
1.02
0.72
1.06
1.47
1.25
1.94
1.70
1.11
1.56
1.51
0.81
1.33
1.09
1.22
GOF*
0.71
0.72
0.81
0.59
0.96
1.33
1.12
1.78
1.70
0.99
1.38
1.14
0.69
1.19
0.92
1.07
PGSR*
0.75
0.72
0.88
0.64
0.82
1.16
0.97
1.69
1.44
0.96
1.30
0.90
0.68
1.05
0.86
0.99
Ours
0.62
0.61
0.58
0.51
0.56
0.88
0.67
1.52
1.17
0.82
0.85
0.54
0.46
0.64
0.55
0.73
* Reproduced results using the authors’ official implementation.
† The source code is not available; only mesh-based evaluation results are reported.
Figure 6. Comparison of Gaussian centroid precision on TnT. Less red points in reconstructed scenes indicate higher accuracy.
ditional opacity loss and Gaussian resampling strategy are
introduced to perform density control.
4.3.1
Depth Filtered Opacity Loss
One major reason that Gaussians form a thick layer sur-
rounding the actual object surface is that many of them re-
main semi-transparent. To address this issue, we introduce
an opacity regularization loss that encourages each Gaus-
sian’s opacity to converge toward either 0 (fully transparent
and removable) or 1 (fully opaque):
Lopacity = 1
N
N
X
i=1
(log αi + log(1 −αi)) ,
(15)
4.3.2
Depth & Normal Reinitialization
To adaptively control the spatial distribution of Gaussians,
we draw inspiration from the periodic resampling strat-
egy in Mini-Splatting [9], extending it with a view-based,
opacity-guided approach.
Unlike Mini-Splatting, which
6

<!-- page 7 -->
Table 2. Top: Reconstruction on TnT (F1 Score ↑). Middle: Re-
construction on TnT (Precision of Gaussian Centroid ↑). Bottom:
Reconstruction on TnT (Completeness of Gaussian ↑).
Barn
Caterpillar
Courthouse
Ignatius
Meetingroom
Truck
Mean
2DGS* [16]
0.46
0.24
0.15
0.49
0.19
0.45
0.33
GOF* [35]
0.55
0.39
0.28
0.71
0.26
0.57
0.46
PGSR* [3]
0.65
0.45
0.21
0.81
0.33
0.62
0.51
GausSurf† [31]
0.50
0.42
0.30
0.73
0.39
0.65
0.50
Ours
0.64
0.42
0.22
0.80
0.32
0.59
0.50
2DGS*
0.65
0.59
0.62
0.71
0.47
0.71
0.62
GOF*
0.65
0.60
0.62
0.77
0.49
0.70
0.64
PGSR*
0.63
0.58
0.56
0.70
0.49
0.69
0.61
Ours
0.66
0.66
0.52
0.83
0.47
0.78
0.65
2DGS*
0.08
0.07
0.04
0.09
0.04
0.11
0.07
GOF*
0.12
0.09
0.05
0.15
0.05
0.16
0.10
PGSR*
0.16
0.13
0.08
0.16
0.09
0.20
0.14
Ours
0.20
0.19
0.08
0.28
0.07
0.32
0.19
does not explicitly consider visibility or density imbalance,
our method leverages per-view opacity statistics to prevent
oversampling in view-crowded regions and ensures that new
Gaussians are allocated more evenly across the scene. For
each training view, we compute the accumulated opacity
αacc by rendering all visible Gaussians, i.e., those posi-
tioned in front of the rendered depth map. Regions with
αacc ≫1 indicate areas that are unnecessarily oversampled.
The sampling weight at each pixel p is therefore defined as
the inverse transmittance: 1−αacc(p). The number of newly
sampled Gaussians for the view is determined by:
Nnew = Nper-view · 1
|Ω|
X
p∈Ω
(1 −αacc(p)) ,
(16)
where Nper-view is a predefined constant that controls the
base number of Gaussians sampled per view. To prioritize
under-represented regions, 3D points are drawn from the
rendered point cloud via multinomial sampling, with prob-
abilities proportional to the sampling weights. For each se-
lected point, its 3D position and normal are extracted from
the rendered depth and normal maps, and then transformed
to world coordinates. To enhance robustness, a bilateral fil-
ter is applied to the depth and normal maps, incorporating
both spatial proximity and value similarity. New Gaussians
are initialized at these sampled points are flattened accord-
ing to the corresponding normals.
5. Experiments
This section presents an evaluation of the proposed
method, including implementation details and ablation
studies. The source code will be released upon the ac-
ceptance of the paper.
5.1. Datasets and Implementation
Datasets: We evaluate our method on three datasets:
DTU [20], Tanks and Temples (TnT) [23] and Mip-
NeRF360 [2], covering both indoor and outdoor environ-
ments.
The DTU dataset consists of 124 sets of high-
quality images of various objects with complex geometries
and textures, captured under controlled lighting with cam-
era poses. The Tanks and Temples dataset serves as a bench-
mark for complex large-scale scene reconstruction using
high-resolution video input. Following PGSR [3], we eval-
uate 3D reconstruction performance on 15 DTU scenes and
6 Tanks and Temples scenes. To assess novel view syn-
thesis, we use the Mip-NeRF360 dataset, which contains
high-resolution images of unbounded outdoor scenes.
Implementation: The implementation of our method
is based on PyTorch. All experiments are conducted on a
desktop equipped with an NVIDIA RTX 4090 GPU. Our
training strategy and hyperparameters are generally consis-
tent with previous works 3DGS [22] and PGSR [3]. All
scenes are trained for 30,000 iterations. The final training
loss is defined as:
L = LRGB + λ1Lnormal + λ2Lnormal-G
+ λ3Lopacity + λ4Lmv + λ5Lmv-G.
(17)
We set the weights as follows: λ1 = 0.015, λ2 =
0.0075, λ3 = 0.0001, λ4 = 0.15, and λ5 = 0.15. Depth
and normal reinitialization are performed at iterations 5,000
and 10,000. We set the number of Gaussian sampled per-
view Nper-view = 10, 000.
5.2. Results
Reconstruction: To assess overall reconstruction accu-
racy, we use the F1 score on the Tanks and Temples dataset
and the Chamfer Distance on the DTU dataset. We compare
the reconstruction performance of GSSR against recent GS-
based methods, including 2DGS [16], GOF [35], PGSR [3],
GausSurf [31]. For evaluation, we first render a depth map
from each training view and then apply Truncated Signed
Distance Function (TSDF) Fusion [27] to integrate them
into a unified TSDF field. A mesh is subsequently extracted
using the Marching Cube algorithm [25], which is used to
compute the reconstruction metrics. As shown in the top
rows of Tables 1 and 2, our method achieves performance
comparable to state-of-the-art results on both datasets.
Gaussian Instance Accuracy: To assess the geometric
fidelity of individual Gaussians, we report accuracy (aver-
age point-to-ground-truth distance) and completeness (aver-
age ground-truth-to-point distance) as the two components
of Chamfer Distance on the DTU dataset. Similarly, we
report precision and recall as components of the F1 score
for the Tanks and Temples dataset. As shown in Table 1
and Table 2, GSSR consistently outperforms all other meth-
ods in Gaussian-level geometric quality. Figure 1 provides
a visual comparison on the DTU dataset between GSSR,
2DGS [16], and PGSR [3], where GSSR yields both lower
position error and more complete coverage. Figure 6 shows
Gaussian center precision across three Tanks and Temples
7

<!-- page 8 -->
Table 3. Quantitative results on the Mip-NeRF360 [2] dataset.
Method
Indoor scenes
Outdoor scenes
Average on all scenes
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
2DGS [16]
30.40
0.916
0.195
24.34
0.717
0.246
27.37
0.817
0.221
GOF [35]
30.79
0.924
0.184
24.82
0.750
0.202
27.81
0.837
0.193
PGSR* [3]
30.20
0.930
0.158
24.77
0.752
0.204
27.49
0.841
0.181
GausSurf [31]
30.05
0.920
0.183
25.09
0.753
0.212
27.57
0.837
0.198
Ours
29.79
0.924
0.168
24.47
0.724
0.250
27.13
0.824
0.209
Figure 7. Scene editing comparison: placing an object on a flat surface. GSSR yields the cleanest composition.
Table 4. Ablation study on the DTU dataset.
Method
Chamfer Distance ↓
Precision ↓
Completeness ↓
w/o Resampling
0.57
0.48
0.92
w/o Lnormal
0.53
0.40
0.68
w/o Lmv
0.63
0.43
0.85
w/o Ls
0.54
0.38
0.76
w/o Lmv−G, Lnormal−G
0.51
0.56
0.75
Full model
0.53
0.37
0.73
scenes, demonstrating GSSR’s higher accuracy. Figure 5
visualizes the spatial density of Gaussians, where GSSR
produces a more uniform distribution and thinner layers
aligned with object surfaces.
Novel View Synthesis: To evaluate novel view synthesis
quality, we follow the experimental setup of 3DGS [22] and
conduct validation on the Mip-NeRF360 dataset [2]. Our
method is compared against several GS-based approaches.
As shown in Table 3, the proposed framework not only
achieves a strong surface reconstruction performance, but
also delivers competitive results in novel view rendering.
Scene Editing By design, GSSR produces Gaussians
that are both aligned with and evenly distributed along the
latent surfaces. As showsn in Figure 7, when placing an ad-
ditional object on a surface, our method yields cleaner and
more coherent compositions, where PGSR [3] and our abla-
tion variants introduce noticeable artifacts. More examples
are provided in supplementary material.
5.3. Ablation Study
To further evaluate the effectiveness of key components
in our proposed method, we perform ablation studies on the
DTU dataset. We report the quality of the reconstructed
mesh, as well as the accuracy and completeness of the Gaus-
sian centers. Detailed quantitative results for each ablated
variant are presented in Table 4. The resampling strategy
plays a critical role in improving completeness by reduc-
ing under-coverage. Loss terms Lnormal-G, and Lmv-G signifi-
cantly enhance Gaussian instance accuracy, while the multi-
view consistency loss Lmv is crucial for improving over-
all geometric quality. Overall, the full model consistently
achieves the best trade-off across all metrics, demonstrating
the importance of each component in enhancing geometric
fidelity and scene coverage.
6. Conclusion
We
present
Gaussian
Set
Surface
Reconstruction
(GSSR), a novel approach inspired by Point Set Surfaces [1]
that represents scenes using dense, geometrically-precise
3D Gaussians with uniform spatial distribution. GSSR en-
forces geometric accuracy through multi-scale constraints
(pixel-level and Gaussian-level) while preserving photore-
alistic rendering capabilities. Our framework introduces:
(1) an opacity regularization loss to prune redundant Gaus-
sians, and (2) a view-adaptive resampling strategy for opti-
mal spatial distribution. Comprehensive evaluation across
three real-world datasets demonstrates GSSR’s superior ge-
ometric consistency and rendering fidelity compared to
state-of-the-art 3DGS methods, producing cleaner distribu-
tions that better adhere to scene geometry.
Limitation:
While GSSR significantly improves per-
Gaussian accuracy, two limitations remain: First, these
gains do not directly translate to improved mesh recon-
struction or alpha-blended depth quality. Second, the view-
based Gaussian sampling requires empirical parameter set-
8

<!-- page 9 -->
ting, though this provides coarse control over Gaussian den-
sity. Future work will investigate adaptive sampling strate-
gies to automate this process.
References
[1] M. Alexa, J. Behr, D. Cohen-Or, S. Fleishman, D. Levin, and
C.T. Silva. Point set surfaces. In Proceedings Visualization,
2001. VIS ’01., pages 21–29, 537, 2001. 2, 3, 8
[2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 5470–5479, 2022. 7, 8
[3] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie,
Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and
Guofeng Zhang. Pgsr: Planar-based gaussian splatting for ef-
ficient and high-fidelity surface reconstruction. IEEE Trans-
actions on Visualization and Computer Graphics, 2024. 2,
3, 4, 6, 7, 8
[4] Hanlin Chen, Chen Li, and Gim Hee Lee. Neusg: Neural im-
plicit surface reconstruction with 3d gaussian splatting guid-
ance. arXiv preprint arXiv:2312.00846, 2023. 3, 4
[5] Hanlin Chen, Fangyin Wei, Chen Li, Tianxin Huang, Yun-
song Wang, and Gim Hee Lee. Vcr-gaus: View consistent
depth-normal regularizer for gaussian surface reconstruc-
tion. Advances in Neural Information Processing Systems,
37:139725–139750, 2024. 3, 4
[6] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xi-
aofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping
Liu, and Guosheng Lin. Gaussianeditor: Swift and control-
lable 3d editing with gaussian splatting. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 21476–21485, 2024. 3
[7] Kai Cheng, Xiaoxiao Long, Kaizhi Yang, Yao Yao, Wei Yin,
Yuexin Ma, Wenping Wang, and Xuejin Chen. Gaussianpro:
3d gaussian splatting with progressive propagation. In Forty-
first International Conference on Machine Learning, 2024. 3
[8] Lue Fan, Yuxue Yang, Minxing Li, Hongsheng Li, and
Zhaoxiang Zhang. Trim 3d gaussian splatting for accurate
geometry representation. arXiv preprint arXiv:2406.07499,
2024. 3
[9] Guangchi Fang and Bing Wang.
Mini-splatting: Repre-
senting scenes with a constrained number of gaussians. In
European Conference on Computer Vision, pages 165–181.
Springer, 2024. 6
[10] Qiyuan Feng, Gengchen Cao, Haoxiang Chen, Tai-Jiang Mu,
Ralph R. Martin, and Shi-Min Hu. A new split algorithm for
3d gaussian splatting, 2024. 3
[11] Qiancheng Fu, Qingshan Xu, Yew Soon Ong, and Wenbing
Tao. Geo-neus: Geometry-consistent neural implicit surfaces
learning for multi-view reconstruction. Advances in Neural
Information Processing Systems, 35:3403–3416, 2022. 4
[12] Silvano Galliani, Katrin Lasinger, and Konrad Schindler.
Gipuma: Massively parallel multi-view stereo reconstruc-
tion.
Publikationen der Deutschen Gesellschaft f¨ur Pho-
togrammetrie, Fernerkundung und Geoinformation e. V,
25(361-369):2, 2016. 4
[13] Jian Gao, Chun Gu, Youtian Lin, Zhihao Li, Hao Zhu, Xun
Cao, Li Zhang, and Yao Yao. Relightable 3d gaussians: Re-
alistic point cloud relighting with brdf decomposition and
ray tracing. In European Conference on Computer Vision,
pages 73–89. Springer, 2024. 4
[14] Antoine Gu´edon and Vincent Lepetit.
Sugar:
Surface-
aligned gaussian splatting for efficient 3d mesh reconstruc-
tion and high-quality mesh rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5354–5363, 2024. 3
[15] Ayaan Haque, Matthew Tancik, Alexei A Efros, Aleksander
Holynski, and Angjoo Kanazawa. Instruct-nerf2nerf: Edit-
ing 3d scenes with instructions.
In Proceedings of the
IEEE/CVF international conference on computer vision,
pages 19740–19750, 2023. 3
[16] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically ac-
curate radiance fields. In ACM SIGGRAPH 2024 conference
papers, pages 1–11, 2024. 2, 3, 4, 6, 7, 8
[17] Jiajun Huang, Hongchuan Yu, Jianjun Zhang, and Hammadi
Nait-Charif. Point’n move: Interactive scene object manipu-
lation on gaussian splatting radiance fields. IET Image Pro-
cessing, 18:3507–3517, 07 2024. 3
[18] Zhentao Huang, Yukun Shi, Neil Bruce, and Minglun
Gong.
Seald-nerf: Interactive pixel-level editing for dy-
namic scenes by neural radiance fields.
arXiv preprint
arXiv:2402.13510, 2024. 3
[19] Zhentao Huang, Yukun Shi, and Minglun Gong. Visibility-
aware pixelwise view selection for multi-view stereo match-
ing.
In International Conference on Pattern Recognition,
pages 130–144. Springer, 2025. 4
[20] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engil Tola,
and Henrik Aanæs. Large scale multi-view stereopsis eval-
uation. In 2014 IEEE Conference on Computer Vision and
Pattern Recognition, pages 406–413. IEEE, 2014. 7
[21] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xi-
aoxiao Long, Wenping Wang, and Yuexin Ma. Gaussian-
shader: 3d gaussian splatting with shading functions for re-
flective surfaces. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, pages
5322–5332, 2024. 3, 4
[22] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics,
42(4):1–14, 2023. 2, 3, 7, 8
[23] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen
Koltun.
Tanks and temples:
Benchmarking large-scale
scene reconstruction. ACM Transactions on Graphics (ToG),
36(4):1–13, 2017. 7
[24] Marc Levoy and Pat Hanrahan. Light field rendering. In
Proceedings of the 23rd Annual Conference on Computer
Graphics and Interactive Techniques, SIGGRAPH ’96, page
31–42, New York, NY, USA, 1996. Association for Comput-
ing Machinery. 2
[25] William E Lorensen and Harvey E Cline. Marching cubes:
A high resolution 3d surface construction algorithm. In Sem-
inal graphics: pioneering efforts that shaped the field, pages
347–353. 1998. 7
9

<!-- page 10 -->
[26] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
2
[27] Richard A Newcombe, Shahram Izadi, Otmar Hilliges,
David Molyneaux, David Kim, Andrew J Davison, Pushmeet
Kohi, Jamie Shotton, Steve Hodges, and Andrew Fitzgibbon.
Kinectfusion: Real-time dense surface mapping and track-
ing. In 2011 10th IEEE international symposium on mixed
and augmented reality, pages 127–136. Ieee, 2011. 7
[28] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited.
In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
4104–4113, 2016. 2
[29] Johannes L Sch¨onberger,
Enliang Zheng,
Jan-Michael
Frahm, and Marc Pollefeys.
Pixelwise view selection for
unstructured multi-view stereo. In Computer Vision–ECCV
2016: 14th European Conference, Amsterdam, The Nether-
lands, October 11-14, 2016, Proceedings, Part III 14, pages
501–518. Springer, 2016. 2
[30] Matias Turkulainen, Xuqian Ren, Iaroslav Melekhov, Otto
Seiskari, Esa Rahtu, and Juho Kannala. Dn-splatter: Depth
and normal priors for gaussian splatting and meshing.
In
2025 IEEE/CVF Winter Conference on Applications of Com-
puter Vision (WACV), pages 2421–2431. IEEE, 2025. 3, 4
[31] Jiepeng Wang, Yuan Liu, Peng Wang, Cheng Lin, Junhui
Hou, Xin Li, Taku Komura, and Wenping Wang.
Gaus-
surf: Geometry-guided 3d gaussian splatting for surface re-
construction. arXiv preprint arXiv:2411.19454, 2024. 2, 3,
6, 7, 8
[32] Xiangyu Wang, Jingsen Zhu, Qi Ye, Yuchi Huo, Yunlong
Ran, Zhihua Zhong, and Jiming Chen. Seal-3d: Interactive
pixel-level editing for neural radiance fields. In Proceedings
of the IEEE/CVF International Conference on Computer Vi-
sion, pages 17683–17693, 2023. 3
[33] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke.
Gaussian grouping: Segment and edit anything in 3d scenes.
In Aleˇs Leonardis, Elisa Ricci, Stefan Roth, Olga Rus-
sakovsky, Torsten Sattler, and G¨ul Varol, editors, Computer
Vision – ECCV 2024, pages 162–179, Cham, 2025. Springer
Nature Switzerland. 3
[34] Mulin Yu, Tao Lu, Linning Xu, Lihan Jiang, Yuanbo Xiangli,
and Bo Dai. Gsdf: 3dgs meets sdf for improved rendering
and reconstruction. arXiv preprint arXiv:2403.16964, 2024.
3
[35] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian
opacity fields: Efficient adaptive surface reconstruction in
unbounded scenes. ACM Transactions on Graphics (TOG),
43(6):1–13, 2024. 3, 6, 7, 8
[36] Yu-Jie Yuan, Yang-Tian Sun, Yu-Kun Lai, Yuewen Ma,
Rongfei Jia, and Lin Gao. Nerf-editing: Geometry editing of
neural radiance fields. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
18353–18364, 2022. 3
[37] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and
Markus Gross. Ewa volume splatting. In Proceedings Visu-
alization, 2001. VIS’01., pages 29–538. IEEE, 2001. 3
10
