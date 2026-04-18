<!-- page 1 -->
Moving Light Adaptive Colonoscopy
Reconstruction via Illumination-Attenuation-Aware
3D Gaussian Splatting
Hao Wanga,†, Ying Zhoua,†, Haoyu Zhaob, Rui Wanga, Qiang Hua, Xing Zhangc, Qiang Lia,∗, Zhiwei Wanga,∗
aWuhan National Laboratory for Optoelectronics, Huazhong University of Science and Technology
bSchool of Computer Science, Wuhan University
cWuhan United Imaging Healthcare Surgical Technology Co., Ltd
{liqiang8, zwwang}@hust.edu.cn
Abstract—3D Gaussian Splatting (3DGS) has emerged as a
pivotal technique for real-time view synthesis in colonoscopy, en-
abling critical applications such as virtual colonoscopy and lesion
tracking. However, the vanilla 3DGS assumes static illumination
and that observed appearance depends solely on viewing angle,
which causes incompatibility with the photometric variations in
colonoscopic scenes induced by dynamic light source/camera.
This mismatch forces most 3DGS methods to introduce structure-
violating vaporous Gaussian blobs between the camera and
tissues to compensate for illumination attenuation, ultimately
degrading the quality of 3D reconstructions. Previous works only
consider the illumination attenuation caused by light distance,
ignoring the physical characters of light source and camera. In
this paper, we propose ColIAGS, an improved 3DGS framework
tailored for colonoscopy. To mimic realistic appearance under
varying illumination, we introduce an Improved Appearance
Modeling with two types of illumination attenuation factors,
which enables Gaussians to adapt to photometric variations
while preserving geometry accuracy. To ensure the geometry
approximation condition of appearance modeling, we propose
an Improved Geometry Modeling using high-dimensional view
embedding to enhance Gaussian geometry attribute prediction.
Furthermore, another cosine embedding input is leveraged to
generate illumination attenuation solutions in an implicit manner.
Comprehensive experimental results on standard benchmarks
demonstrate that our proposed ColIAGS achieves the dual capa-
bilities of novel view synthesis and accurate geometric reconstruc-
tion. It notably outperforms other state-of-the-art methods by
achieving superior rendering fidelity while significantly reducing
Depth MSE. Code will be available.
Index Terms—3D Reconstruction, Gaussian Splatting, Surgical
AI.
I. INTRODUCTION
Colorectal cancer (CRC) remains one of the leading causes
of cancer-related deaths globally [1], and early detection via
colonoscopy is crucial for improving survival rates. However,
the complex anatomical structure of the colon and rectum,
including the plicae circulares, often obstructs the view during
colonoscopy, leading to missed detections [2]. To address this
challenge, 3D reconstruction of the colon through advanced
imaging techniques is essential. A reconstructed 3D model
not only enhances visibility through novel view synthesis but
also supports critical applications such as surgical planning [3],
training [4], and follow-up screenings [5].
Recent advancements have been made in the field of
endoscopic scene recovery. Traditional algorithms, such as
simultaneous localization and mapping (SLAM) [5] and struc-
ture from motion (SfM) [6], often generate 3D point cloud
models from feature point extraction. However, these models
lack realistic appearance. Neural Radiance Fields [7] (NeRF)-
based approaches [4], [8]–[13] achieve photorealistic render-
ings through ray tracing but are computationally intensive,
requiring lengthy training and inference times. 3D Gaussian
Splatting (3DGS) [14] has emerged as a promising alternative
for various application [15], [16]. For endoscopy, it demon-
strates significant improvements in both rendering realism and
computational efficiency for endoscopy reconstruction [17]–
[19] and surgical video reconstruction [20]–[23].
However, the vanilla 3DGS assumes static illumination and
that observed appearance depends solely on viewing angle,
consequently failing to handle the illumination attenuation
inherent in practical colonoscopy scenarios. To compensate,
most existing 3DGS methods generate fog-like, structure-
violating Gaussian blobs, which obscure their behind structure-
preserving Gaussians, leading to significant 3D reconstruction
errors [24].
Existing approaches [8], [25] have attempted to incorpo-
rate light source distance into the appearance modeling of
NeRF/GS frameworks, which partially alleviates the afore-
mentioned issues. However, in endoscopic scenes, illumination
attenuation is influenced not only by the distance to the light
source, but also by the orientation of both the light source and
the camera — factors that these methods fail to account for.
In this paper, we propose ColIAGS, an improved 3DGS
variant that addresses the aforementioned incompatibility is-
sue, enhancing both rendering fidelity and geometric recon-
struction accuracy in colonoscopy scenes. Specifically, we
first perform Illumination Factor Extraction on colonoscopy
scenes and derive a novel lighting modeling approximation
within the 3DGS framework. This model establishes that
the observed appearance depends not only on the viewing
arXiv:2510.18739v1  [cs.CV]  21 Oct 2025

<!-- page 2 -->
angle, but also on two more types of illumination attenuation.
However, this modeling approximation holds only under the
condition that Gaussians sufficiently adhere to the tissue
surface and capture structural details. To this end, we introduce
an Improved Geometry Modeling with View Embedding for
restricted viewing angles to enhance the Gaussians’ geome-
try attribute prediction, thereby satisfying the approximation
criteria. To achieve more efficient appearance modeling, we
further proposed an Improved Appearance Modeling with
Illumination Attenuation, employing an MLP with cosine
embedding inputs to implicitly solve the attenuation behavior
from high-dimensional feature representations. Consequently,
establishes the observed appearance as a function of both
camera-to-surface distance and orientation.
Our contributions can be summarized as follows:
1) We propose ColIAGS, an improved 3DGS variant that
improves both rendering fidelity and 3D reconstruction
accuracy significantly.
2) We introduce a novel light modeling framework tailored
for the colonoscopic scenario and two unique cosine
embedding schemes to guarantee the feasibility and
effectiveness of the framework.
3) Experiments on two public benchmarks demonstrate the
superiority of ColIAGS over state-of-the-art methods:
(1) it maintains comparable rendering fidelity while
significantly improving geometric accuracy relative to
realistic rendering techniques, and (2) it achieves a 2.04
dB improvement in PSNR for novel view synthesis and
reduces Depth MSE by 78% compared to geometry-
preserving approaches.
II. RELATED WORK
A. 3D Reconstruction on Endoscopy
Reconstruction is a comprehensive concept, encompassing
image-level reconstruction [26] and 3D assets reconstruction.
3D reconstruction in endoscopic scenes has been a long-
standing research problem. Some works focus on upstream
tasks such as depth estimation [27] using self-supervised
paradigm or based on foundation model [28]. Besides others
adopt SLAM-based approaches enhanced by deep learning.
For instance, RNNSLAM combines Direct Sparse Odometry
(DSO) with a recurrent neural network (RNN)-based monoc-
ular depth estimator to jointly recover scene geometry and
camera poses.
Recently, emerging techniques such as Neural Radiance
Fields (NeRF) and 3D Gaussian Splatting (3DGS) have gained
increasing attention, with growing interest in their application
to endoscopic reconstruction. In laparoscopic surgery, methods
like [4], [10]–[12] introduce dynamic modeling into NeRF to
reconstruct deformable tissues to support operation navigation
and surgical training. While [20]–[23], [29] maker incorpo-
rating dynamic modeling into 3DGS, reducing training and
inference cost.
In the context of colonoscopy, ColonNeRF [9] employs
visibility-based supervision to capture the global structure of
the colon, while REIM-NeRF [8] integrates light source posi-
tions to address brightness variations caused by a moving light
source. However, the inherently slow training and inference
processes of NeRF significantly restrict its use in real-time
applications.
To overcome these limitations, several works have adopted
3DGS. GaussianPancake [18] integrates SLAM frameworks
with camera and depth priors, but its limited exploitation
of the representational power of Gaussians results in subpar
performance. EndoGSLAM [30] and Endo2DTAM [31] incor-
porate Gaussians into a SLAM system for 3D representation,
but fails to account for illumination attenuation, leading to
severe artifacts during mapping. PR-Endo [25] attempts to
resolve this by leveraging physics-based inverse rendering, but
the high computational cost required for optimization greatly
impedes its practical deployment. Both REIM-NeRF [8] and
PR-Endo [25] model illumination attenuation solely based
on the distance of the moving light source, ignoring the
effects of light and camera orientation, which in turn degrades
reconstruction performance.
In contrast, our method explicitly models multiple illumi-
nation attenuation factors and enhances perceptual capability
under limited viewpoints. This ensures photometric fidelity
and geometric consistency without requiring extensive opti-
mization time. Furthermore, our framework can be seamlessly
integrated into existing SLAM systems, offering flexibility in
initialization and deployment.
B. Lighting modeling
Previous studies have established various approaches to
illumination modeling. Visentini-Scarzanella et al. [32] assume
Lambertian surfaces and calibrate camera with light simulta-
neously. Modrzejewski et al. [33] conducted a comprehensive
analysis of different light source models, demonstrating that
their Spot Light Source (SLS) model achieves an optimal bal-
ance between computational complexity and accuracy. More-
over, Batlle et al. [34] adapt a similar modeling philosophy, but
differs in consolidating multiple endoscopic light sources into
a single virtual light source positioned at the camera’s optical
center. This simplified lighting model enables the unification
of the light spread function and camera vignetting effects
into a single formulation, which inspires us to jointly model
illumination attenuation with respect to various contributing
factors.
III. METHOD
A. Preliminaries of 3DGS and Neural Gaussians
1) 3DGS: Images are 2D observations of a 3D real-world
scene. As illustrated in the gray box of Fig.1, 3D Gaussian
Splatting (3DGS) represents the underlying scene using a set
of anisotropic 3D Gaussians Θ = {µ, s, q, α, c}, where µ is
the mean position, s, q are the scaling matrix and rotation
quaternion components of 3D covariance matrix, α is the
opacity, and c is the Gaussian’s color defined by the spherical
harmonics to model view-variant color.

<!-- page 3 -->
Fig. 1. Overview of our proposed ColIAGS framework. The pipeline of ColIAGS contains two proposed components, i.e., the Improved Geometry Modeling
with View Embedding(the yellow box) to enhance the geometry precision, ensuring the approximation conditions to incorporate illumination factors, and the
Improved Appearance Modeling with Illumination Attenuation(the blue box) to model the two types of illumination attenuation.
3DGS efficiently renders the scene using tile-based rasteri-
zation. First, the 3D Gaussians are projected onto the 2D image
plane [35]. Then, the 2D Gaussians are sorted, and α-blending
is applied:
C(x′) =
X
i∈N
ciσi
i−1
Y
j=1
(1 −σj)
(1)
where x′ is the queried pixel position, N denotes the number
of sorted 2D Gaussians associated with the queried pixel, and
σ is the 2D Gaussian opacity defined by its 3D counterpart α.
By leveraging a differentiable rasterizer, Θ is learnable and
optimized end-to-end via view reconstruction training.
In addition, the gaussian-surface distance can be approx-
imated by rendering their depths similarly to Eq. (1), as
follows:
D(x′) =
X
i∈N
ziσi
i−1
Y
j=1
(1 −σj),
zi = R−1(µi −xc)z
(2)
where zi is the i-th Gaussian’s depth [36] in the camera
coordinate system. R and xc are the camera rotation and
position, and (·)z means fetching the coordinate’s z-value.
2) Neural Gaussians: The anchor-based neural Gaussian
technique was originally proposed by [37] for efficient on-the-
fly rendering and redundancy reduction via feature-enriched
anchors. The scene is then hierarchically voxelized, with
each voxel center assigned an anchor containing a context
feature fa, a scaling factor la ∈R3, and k learnable offsets
Ov ∈Rk×3. Anchors within the viewing frustum dynamically
generate k Gaussians, whose positions are calculated by:
µ = xa + Ov · la
(3)
while other attributes in Θ are predicted by individual MLPs
with fa, viewing distance d = ∥xa −xc∥, and direction
v = (xa −xc)/d pointing from the camera position xc to
the anchor’s xa as input.
Both GS and Neural Gaussians are designed for the natural
scene scenario. However, natural scenes typically have wide
fields of view and static lighting conditions, which simplify
the reconstruction process. In contrast, the colonoscopy is
characterized by light sources that are physically attached to
the endoscope itself. The vanilla application of these methods
on the colonoscopic scene leads to geometrically implausible
reconstructions and inaccurate modeling of lighting. To this
end, we establish a specialized model tailored to the complex
illumination of colonoscopy and implement it in an efficient
manner.
B. Lighting Modeling and Illumination Attenuation Factor
Extraction
In contrast to natural scenarios, where viewing angle is
the dominant factor for appearance, endoscopic scenes exhibit
appearance that is further determined by two types of illumi-
nation attenuation. The first arises from the movement of the
light source, where tissues closer to the light source appear
brighter than those farther away. The second results from the
physical characteristics of both the light source and the camera

<!-- page 4 -->
Fig. 2.
(a) illustrates the lighting attenuation in a conventional point light
model, where the illumination depends on both the light source’s orientation
and its distance to the object surface, while camera vignetting effects are
influenced by the viewing direction. (b) presents a simplified lighting model,
where the light source is approximated to be co-located with the camera, and
its direction is assumed to align with the camera’s orientation.
imaging system, including light spread function and vignetting
effects.
As shown in the Fig. 2(a), the image color is not only
negatively correlated with the light source distance dl, but also
related to the light spread function and the camera vignetting
effects, which can be specified by µ(x) and V (x):
µ(x) = cosk1(ψ), ψ =< vl, zl >
V (x) = cosk2(θ), θ =< vc, zc >
C ∝d−1
l
, µ(x), V (x)
(4)
where ψ/θ are the angle between light/camera view direction
vl/vc and light/camera orientation zl/zc, k1, k2 are the ex-
ponents related to the physical characteristics, and dl is the
distance between the light source and the surface of the object.
Inspired by [34], the lighting model in colonoscopy can be
approximated by assuming the light source is attached to the
camera, as illustrated in Fig. 2(b).
To extract illumination attenuation factors in our Neural
Gaussians framework, we approximate object’s surface with
Gaussians, using v/d as the direction/distance from cam-
era/light to object surface. Furthermore, attenuation effects
caused by both the light spread function and the camera
vignetting effects can be modeled as a unified relationship
µ′(x) with Gaussians’ color c, i.e., ψ = θ:
c ∝µ′(x) = µ(x)V (x) = cosk θ
(5)
where θ is the angle between view direction v between
view direction and light/camera’s orientation and k is an
undetermined exponent.
However, the imperfect geometry attributes of Gaussians
leave gap between the assumption and reality, i.e., Gaussians
cannot properly adhere to object surfaces or is not fine enough
to present structure details. Additionally, since the exponen-
tial parameter k varies between different physical characters,
directly estimate it will be hindered by camera’s auto gain as
shown in [34].
C. Improved Modeling with Cosine Embedding
To accommodate the aforementioned lighting model, we
primarily improve the geometry modeling in Neural Gaussians
to enable the desired approximation. Firstly, we leverage depth
loss to constrain the position of Gaussians close to object
surface:
Ld =
X
x′
|D(x′) −ˆD(x′)|
(6)
However, optimizing fine structures within a limited viewing
range (which is common in colonoscopy) often leads to losses
of structure details [7], i.e., using large-scale Gaussians to
represent low-frequency information, such as blurred or over-
smoothed regions.
To enhance the high-frequency representational capability of
Neural Gaussians’ geometry attributes, we introduce a cosine-
based embedding function γ into geometry modeling, which
projects the original input v into a high-dimensional space
using a set of cosusoidal basis functions. This Improved
Geometry Modeling with View Embedding can be expressed
as follows:
α = Fα(fa, γ(v)),
s = Fs(fa, γ(v)),
r = Fr(fa, γ(v)).
(7)
Given the improved geometry modeling, we can directly
model lighting based on the approximation derived from Equa-
tion (5). Instead of explicitly estimating the unknown exponent
k, we employ the ability of the MLP that implicitly learn to
mimic the attenuation behavior from the high-dimensional rep-
resentation. While similarly based on the cosine embedding,
this Improved Appearance Modeling with Illumination
Attenuation distinctively incorporates θ into the input of Fc:
c = Fc(fa, v, d, γ(cos θ))
(8)
The color c of each Gaussian is now influenced by both camera
(light) distance and orientation, accurately modeling the two
types of illumination attenuation in colonoscopy.
D. Loss Function
To train our ColIAGS, we use the image reconstruction
losses [14], i.e., L1 and LD−SSIM, as well as the above depth
constraint Ld. An extra regularization Lscale [39] is applied
to prevent the overlapping and large volume of Gaussians.
The total constraints are calculated as follows:
L = (1 −λ1)L1 + λ1LD−SSIM + λ2Ldepth + λ3Lscale (9)
IV. EXPERIMENT SETTINGS
A. Implementation Details
We train ColIAGS using Pytorch [40] framework on a
single NVIDIA GeForce RTX 4090. Following the previous
settings [14], [37], [41], we set λ1, λ2, λ3 as 0.2, (0.2-
0.01 with exponential decay), 0.01, respectively. The cosine
embedding strategies in IMIA and IRVE are configured with
dimensions of 10 and 5, respectively(Dv = 10 Dθ = 5).
B. Dataset and Evaluation Metrics
We follow the protocol of existing study [8], [18], [25],
conduct evaluation on C3VD [42], C3VD with EndoGSLAM
Initialization [30], and RotateColon [25]

<!-- page 5 -->
TABLE I
QUANTITATIVE COMPARISON RESULTS ON C3VD AND ROTATECOLON. THE BEST RESULTS ARE MARKED IN BOLD AND THE SECOND-BEST
UNDERLINED.
Methods
C3VD
RotateColon
Depth MSE ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
REIM NeRF [8]
1.480
33.96
0.86
0.32
10.96
0.62
0.57
2DGS [38]
4.839
32.93
0.88
0.24
19.58
0.87
0.29
Gaussian Pancakes [18]
1.222
32.93
0.88
0.24
20.10
0.88
0.27
3DGS [14]
0.355
34.10
0.89
0.22
20.53
0.87
0.26
Scaffold-GS [37]
0.195
34.26
0.89
0.21
22.58
0.90
0.22
PR-Endo [25]
0.674
34.15
0.89
0.23
21.90
0.87
0.28
Ours
0.042
36.30
0.91
0.15
23.29
0.89
0.21
TABLE II
QUANTITATIVE COMPARISON RESULTS ON C3VD WITH ENDOGSLAM
INITIALIZATION. THE BEST RESULTS ARE MARKED IN BOLD AND THE
SECOND-BEST UNDERLINED.
Method
Depth MSE ↓
PSNR ↑
SSIM ↑
LPIPS ↓
REIM NeRF [8]
0.495
34.34
0.88
0.34
2DGS [38]
0.682
32.05
0.88
0.34
Gaussian Pancakes [18]
1.630
32.28
0.88
0.32
3DGS [14]
0.333
33.56
0.89
0.29
Scaffold-GS [37]
0.215
33.61
0.89
0.29
PR-Endo [25]
0.351
34.00
0.89
0.29
Ours
0.122
35.58
0.91
0.23
1) C3VD: consists of 22 colonoscopy videos with the
resolution of 1350 × 1080 captured from various colon phan-
tom models, equipped with registered depth maps and corre-
sponding camera poses. These models exhibit the mentioned
illumination variations, thus ensuring a comprehensive setup
for evaluating the effectiveness of our method. In practice, we
undistorted and resized the image to 338×270 [18]. For each
scene’s video, we split the frame data into training and testing
sets using a 7: 1 ratio.
2) RotateColon: is developed specifically to evaluate novel
view synthesis under extended rotations proposed by previous
work [25], including intense rotations not observed during
the training sequence. The per-frame’s depth and camera are
recorded with their in-house simulator. We adhere to the
established evaluation protocol by partitioning the dataset into
381 frames for training and 223 frames for testing, ensuring
direct comparability with prior work [25].
Note that, the original settings on the two datasets men-
tioned above utilize ground-truth camera poses for initializa-
tion. To further evaluate the robustness and practical applica-
bility of our method, we follow previous work [18], [25] by
using SLAM outputs as the initialization for GS model. To
avoid the potential impact of minor discrepancies in SLAM
results, we utilize the output released by previous work [25]
for consistency. It contains 10 sequences selected from C3VD
as [30], and the camera poses and depth maps used are
generated by the EndoGSLAM [30] framework. This setting
validates algorithm feasibility under suboptimal initialization
by deliberately adopting SLAM-based initialization, replicat-
ing the challenging conditions in Gaussian Pancakes [18].
To be consistent with existing works [8], [18], [25], we thor-
oughly evaluate the model performance using comprehensive
metrics, including peak signal-to-noise ratio (PSNR), struc-
tural similarity index (SSIM) and learned perceptual image
patch similarity (LPIPS) to measure the rendering quality, and
depth mean square error (Depth MSE) for geometry accuracy.
It’s notable that as RotateColon did not provide depth ground-
truth for test views, we only evaluate Depth MSE on C3VD
and C3VD with EndoGSLAM Initialization.
V. RESULTS AND DISCUSSION
We compare ColIAGS with 6 state-of-the-art (SOTA) meth-
ods, i.e., 3DGS [14], 2DGS [38], Scaffold-GS [37], REIM
NeRF [8], Gaussian Pancakes [18] and PR-Endo [25]. We
evaluate their performance and make comparisons using their
released codes. Note that, to ensure fairness in comparisons,
we also incorporate depth loss Ld in Eq. 6 to 3DGS and
Scaffold-GS, which is a necessary condition for preserving
fundamental geometry.
A. Comparison with State-of-the-arts
Table I presents the comparison results between ColIAGS
and other state-of-the-art methods. We can observe that our
method consistently outperforms other methods on both C3VD
and RotateColon benchmarks, demonstrating superior render-
ing quality and geometric accuracy.
Among these SOTA methods, Scaffold-GS and PR-Endo
achieve the second and third performance. While both are
specifically designed for complex scenes, with PR-Endo addi-
tionally incorporating endoscopic-specific considerations, e.g.,
constrained camera trajectories and view-dependent lighting,
their suboptimal performance ultimately derives from insuffi-
cient geometric optimization. In contrast, ColIAGS surpasses
them with a reduction of 0.155 mm2 in Depth MSE and an
improvement of 2.04 dB in PSNR for rendering quality.
To validate robustness under realistic clinical conditions,
i.e., the cases without the ground truth, we specifically evaluate
the scenario where depth maps and camera poses generated by
EndoGSLAM are used for initialization. As shown in Table II,
our method maintains superior performance in all metrics,
which indicates strong potential for clinical deployment.
Fig 3 visualizes the performance comparison in novel view
synthesis and geometric reconstruction. As illustrated in the

<!-- page 6 -->
Fig. 3. Qualitative comparison on C3VD, against Scaffold-GS [37], Gaussians Pancakes [18] and PR-Endo [25].The top three rows display the rendered RGB
images, while the bottom three rows show the depth mse error map.
top three rows, ColIAGS achieves realistic rendering quality
by accurately modeling illumination attenuation and preserv-
ing sharper high-frequency details, e.g., irregular highlight
patterns and fold discontinuities. The visualization comparison
in the bottom three rows reveals that our method generates
more smoothed depth maps containing reduced noise artifacts
and higher fidelity to the ground truth depth, confirming the
necessity of modeling illumination attenuation and incorporat-
ing cosine embeddings.
B. Ablation Study
To further verify the effectiveness of the two proposed tech-
niques tailored for colonoscopic 3DGS, i.e., Improved Geom-
etry Modeling with View Embedding (IMVE) and Improved
Appearance Modeling with Illumination Attenuation (IMIA),
we conduct ablation study on C3VD. Also, we investigate
the impact of the cosine embedding dimensions (dim) in both
modules.
1) Effectiveness of Two Components: We develop four vari-
ants by enabling/disabling each component. Specifically, when
the improved modeling is disabled, we use the original input
mode in the vanilla Scaffold-GS. In addition, we develop a
variant without the depth constraint Ldepth, aiming to indicate
the necessity of applying it to guarantee basic geometry.
Table III shows the quantitative comparison between the
complete ColIAGS and other variants. Based on these results,
three key observations can be made as follows:
(1) When comparing the 1th to 2th rows, the absence of
depth supervision leads to a substantial decrease in geometric
accuracy. This justifies our experimental setting to incorporate
the depth loss into other variants for comparison, ensuring a
fair evaluation of our proposed improvements.
(2) As can be seen from the 2th to the 4th rows of
Table III, using either IMVE or IMIA can bring significant
improvements in both rendering quality (in terms of PSNR,
p < 0.016) while IMVE also improve geometry precision (in
terms of Depth MSE, p < 0.03.
(3) A comparison between the 3th and 4th rows reveals that
although incorporating IMIA alone (without IMVE) leads to a
certain improvement in rendering quality, the suboptimal ge-
ometry accuracy induces notable approximation errors, which
ultimately undermine both the rendering results and the overall
geometric fidelity.
(4) The bottom three rows of Table III indicate that IMVE
and IMIA are not mutually excluded, bringing a further
reduction in Depth MSE by at least 78%. Therefore, ColIAGS
is able to acquire a model that effectively combines novel view
synthesis and geometric reconstruction.

<!-- page 7 -->
TABLE III
ABLATION STUDY ON C3VD [42]. Ld, IMVE AND IMIA REFER TO DEPTH LOSS, IMPROVED GEOMETRY MODELING WITH VIEW EMBEDDING AND
IMPROVED APPEARANCE MODELING WITH ILLUMINATION ATTENUATION, RESPECTIVELY.
Ld
IMVE
IMIA
Depth MSE ↓
PSNR ↑
SSIM ↑
LPIPS ↓
%
%
%
147.644
34.92
0.90
0.20
!
%
%
0.195
34.26
0.89
0.21
!
%
!
0.135
35.18
0.90
0.18
!
!
%
0.064
36.01
0.90
0.17
!
!
!
0.042
36.30
0.91
0.15
TABLE IV
ABLATION STUDY ON C3VD [42] ABOUT EMBEDDING DIMENSIONS ON v
AND θ, RESPECTIVELY.THE BEST RESULTS ARE MARKED IN BOLD.
Dv
Dθ
Depth MSE↓
PSNR ↑
SSIM ↑
LPIPS ↓
10
0
0.045
36.08
0.90
0.16
10
5
0.042
36.30
0.91
0.15
10
10
0.051
36.29
0.91
0.16
5
5
0.057
36.07
0.91
0.17
10
5
0.042
36.30
0.91
0.15
15
5
0.087
36.06
0.90
0.16
2) Impact of the cosine embedding dimension: Although
both IMVE and IMIA adopt cosine embedding functions, the
distinct roles they play in our framework necessitate separate
design considerations regarding their embedding dimensional-
ities. Specifically, IMVE focuses on enhancing geometry mod-
eling on high-frequency details, while IMIA aims to capture
illumination variations. These differing functional objectives
imply that a shared embedding configuration may lead to
suboptimal performance for one or both modules.
To identify the optimal dimensional settings for each
embedding, we perform a two-stage greedy hyperparameter
search over the dimensions of the embedding functions, de-
noted as Dv for the view direction embedding in IMVE and
Dθ for the angular embedding in IMIA. As illustrated in
Table IV, we first fix Dv = 10, a reasonable baseline inspired
by prior work, and evaluate three candidate values for Dθ. This
step yields an optimal setting of Dθ = 5 for IMIA under the
fixed Dv. Subsequently, we fix Dθ = 5 and conduct a similar
search over Dv to fine-tune the configuration for IMVE.
Through this sequential tuning process, we determine that
Dv = 10 and Dθ = 5 constitute the optimal configura-
tion, achieving a favorable balance between model capacity
and generalization performance. This tailored configuration
ensures that each module can effectively fulfill its respective
purpose, leading to superior overall performance in our ren-
dering pipeline.
VI. CONCLUSION
In conclusion, we present ColIAGS, an enhanced 3D Gaus-
sian Splatting (3DGS) framework that overcomes the limita-
tions of vanilla 3DGS in colonoscopic reconstruction by ef-
fectively modeling dynamic illumination variations while pre-
serving geometric accuracy. Unlike existing methods, which
only consider the light source distance, our method introduces
illumination-aware appearance modeling with two attenua-
tion factors and enhances geometry precision through high-
dimensional view embedding. Specifically, ColIAGS consists
of two key modules, i.e., Improved Geometry Modeling with
View Embedding (IMVE) and Improved Appearance Mod-
eling with Illumination Attenuation (IMIA). IMVE enhances
geometric representation by introducing high-frequency de-
tails, thereby improving the accuracy of appearance modeling.
IMIA incorporates both the camera (or light source) distance
and orientation to accurately model the two types of illumi-
nation attenuation observed in colonoscopy while implicitly
optimizing the illumination attenuation solutions through an
MLP. The comprehensive comparisons with six state-of-the-
art methods on two public benchmarks, namely C3VD and
RotateColon, demonstrate that ColIAGS achieves superior per-
formance in both novel view synthesis and geometry precision,
significantly improving rendering fidelity with a PSNR gain
of 2.04 dB while reducing Depth MSE by 78%, making
ColIAGS a promising technique for high-fidelity colonoscopy
applications. In the future, we will explore the field of pose-
free paradigm while incorporating the improved modeling in
this paper.
REFERENCES
[1] R. L. Siegel, A. N. Giaquinto, and A. Jemal, “Cancer statistics, 2024,”
CA: a cancer journal for clinicians, vol. 74, no. 1, pp. 12–49, 2024.
[2] D. K. Rex, P. S. Schoenfeld, J. Cohen, I. M. Pike, D. G. Adler, B. M.
Fennerty, J. G. Lieb, W. G. Park, M. K. Rizk, M. S. Sawhney et al.,
“Quality indicators for colonoscopy,” Official journal of the American
College of Gastroenterology— ACG, vol. 110, no. 1, pp. 72–90, 2015.
[3] M. Adamkiewicz, T. Chen, A. Caccavale, R. Gardner, P. Culbertson,
J. Bohg, and M. Schwager, “Vision-only robot navigation in a neural
radiance world,” IEEE Robotics and Automation Letters, vol. 7, no. 2,
pp. 4606–4613, 2022.
[4] C. Yang, K. Wang, Y. Wang, X. Yang, and W. Shen, “Neural lerplane
representations for fast 4d reconstruction of deformable tissues,” in
International Conference on Medical Image Computing and Computer-
Assisted Intervention.
Springer, 2023, pp. 46–56.
[5] R. Ma, R. Wang, Y. Zhang, S. Pizer, S. K. McGill, J. Rosenman,
and J.-M. Frahm, “Rnnslam: Reconstructing the 3d colon to visualize
missing regions during a colonoscopy,” Medical image analysis, vol. 72,
p. 102100, 2021.
[6] A. R. Widya, Y. Monno, K. Imahori, M. Okutomi, S. Suzuki, T. Gotoda,
and K. Miki, “3d reconstruction of whole stomach from endoscope
video using structure-from-motion,” in 2019 41st annual international

<!-- page 8 -->
conference of the IEEE engineering in medicine and biology society
(EMBC).
IEEE, 2019, pp. 3900–3904.
[7] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[8] D. Psychogyios, F. Vasconcelos, and D. Stoyanov, “Realistic endoscopic
illumination modeling for nerf-based data generation,” in International
Conference on Medical Image Computing and Computer-Assisted Inter-
vention.
Springer, 2023, pp. 535–544.
[9] Y. Shi, B. Lu, J.-W. Liu, M. Li, and M. Z. Shou, “Colonnerf: Neural
radiance fields for high-fidelity long-sequence colonoscopy reconstruc-
tion,” arXiv preprint arXiv:2312.02015, 2023.
[10] Y. Wang, Y. Long, S. H. Fan, and Q. Dou, “Neural rendering for
stereo 3d reconstruction of deformable tissues in robotic surgery,” in
International conference on medical image computing and computer-
assisted intervention.
Springer, 2022, pp. 431–441.
[11] W. Li, Y. Hayashi, M. Oda, T. Kitasaka, K. Misawa, and K. Mori,
“Endoself: Self-supervised monocular 3d scene reconstruction of de-
formable tissues with neural radiance fields on endoscopic videos,” in
International Conference on Medical Image Computing and Computer-
Assisted Intervention.
Springer, 2024, pp. 241–251.
[12] R. Bu, C. Xu, J. Shan, H. Li, G. Wang, Y. Miao, and H. Wang, “Dnfplane
for efficient and high-quality 4d reconstruction of deformable tissues,” in
International Conference on Medical Image Computing and Computer-
Assisted Intervention.
Springer, 2024, pp. 176–186.
[13] J. Guo, J. Wang, R. Wei, D. Kang, Q. Dou, and Y.-h. Liu, “Uc-nerf:
Uncertainty-aware conditional neural radiance fields from endoscopic
sparse views,” IEEE Transactions on Medical Imaging, 2024.
[14] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering.” ACM Trans. Graph.,
vol. 42, no. 4, pp. 139–1, 2023.
[15] H. Zhao, C. Zeng, L. Zhuang, Y. Zhao, S. Xue, H. Wang, X. Zhao, Z. Li,
K. Li, S. Huang et al., “High-fidelity simulated data generation for real-
world zero-shot robotic manipulation learning with gaussian splatting,”
arXiv preprint arXiv:2510.10637, 2025.
[16] H. Zhao, H. Wang, X. Zhao, H. Fei, H. Wang, C. Long, and H. Zou,
“Efficient physics simulation for 3d scenes via mllm-guided gaussian
splatting,” arXiv preprint arXiv:2411.12789, 2024.
[17] J. Guo, J. Wang, D. Kang, W. Dong, W. Wang, and Y.-h. Liu, “Free-
surgs: Sfm-free 3d gaussian splatting for surgical scene reconstruc-
tion,” in International Conference on Medical Image Computing and
Computer-Assisted Intervention.
Springer, 2024, pp. 350–360.
[18] S. Bonilla, S. Zhang, D. Psychogyios, D. Stoyanov, F. Vasconcelos,
and S. Bano, “Gaussian pancakes: geometrically-regularized 3d gaus-
sian splatting for realistic endoscopic reconstruction,” in International
Conference on Medical Image Computing and Computer-Assisted Inter-
vention.
Springer, 2024, pp. 274–283.
[19] C. Li, B. Y. Feng, Y. Liu, H. Liu, C. Wang, W. Yu, and Y. Yuan,
“Endosparse: Real-time sparse view synthesis of endoscopic scenes
using gaussian splatting,” in International Conference on Medical Image
Computing and Computer-Assisted Intervention.
Springer, 2024, pp.
252–262.
[20] H. Zhao, X. Zhao, L. Zhu, W. Zheng, and Y. Xu, “Hfgs: 4d gaus-
sian splatting with emphasis on spatial and temporal high-frequency
components for endoscopic scene reconstruction,” arXiv preprint
arXiv:2405.17872, 2024.
[21] S. Yang, Q. Li, D. Shen, B. Gong, Q. Dou, and Y. Jin, “Deform3dgs:
Flexible deformation for fast surgical scene reconstruction with gaussian
splatting,” in International Conference on Medical Image Computing
and Computer-Assisted Intervention.
Springer, 2024, pp. 132–142.
[22] H. Liu, Y. Liu, C. Li, W. Li, and Y. Yuan, “Lgs: A light-weight
4d gaussian splatting for efficient surgical scene reconstruction,” in
International Conference on Medical Image Computing and Computer-
Assisted Intervention.
Springer, 2024, pp. 660–670.
[23] Y. Huang, B. Cui, L. Bai, Z. Guo, M. Xu, M. Islam, and H. Ren, “Endo-
4dgs: Endoscopic monocular scene reconstruction with 4d gaussian
splatting,” in International Conference on Medical Image Computing
and Computer-Assisted Intervention.
Springer, 2024, pp. 197–207.
[24] J. Lin, Z. Li, X. Tang, J. Liu, S. Liu, J. Liu, Y. Lu, X. Wu, S. Xu, Y. Yan
et al., “Vastgaussian: Vast 3d gaussians for large scene reconstruction,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 5166–5175.
[25] J. Kaleta, W. Smolak-Dy˙zewska, D. Malarz, D. Dall’Alba, P. Ko-
rzeniowski, and P. Spurek, “Pr-endo: Physically based relightable gaus-
sian splatting for endoscopy,” arXiv preprint arXiv:2411.12510, 2024.
[26] H. Zhao, W. Dong, R. Yu, Z. Zhao, B. Du, and Y. Xu, “Morestyle:
relax low-frequency constraint of fourier-based image reconstruction in
generalizable medical image segmentation,” in International Conference
on Medical Image Computing and Computer-Assisted Intervention.
Springer, 2024, pp. 434–444.
[27] Z. Wang, Y. Zhou, S. He, T. Li, F. Huang, Q. Ding, X. Feng, M. Liu, and
Q. Li, “Monopcc: Photometric-invariant cycle constraint for monocular
depth estimation of endoscopic images,” Medical Image Analysis, vol.
102, p. 103534, 2025.
[28] B. Cui, M. Islam, L. Bai, A. Wang, and H. Ren, “Endodac: Efficient
adapting foundation model for self-supervised depth estimation from
any endoscopic camera,” in International Conference on Medical Image
Computing and Computer-Assisted Intervention.
Springer, 2024, pp.
208–218.
[29] Y. Liu, C. Li, C. Yang, and Y. Yuan, “Endogaussian: Gaussian
splatting for deformable surgical scene reconstruction,” arXiv preprint
arXiv:2401.12561, 2024.
[30] K. Wang, C. Yang, Y. Wang, S. Li, Y. Wang, Q. Dou, X. Yang, and
W. Shen, “Endogslam: Real-time dense reconstruction and tracking in
endoscopic surgeries using gaussian splatting,” in International Confer-
ence on Medical Image Computing and Computer-Assisted Intervention.
Springer, 2024, pp. 219–229.
[31] Y. Huang, B. Cui, L. Bai, Z. Chen, J. Wu, Z. Li, H. Liu, and H. Ren,
“Advancing dense endoscopic reconstruction with gaussian splatting-
driven surface normal-aware tracking and mapping,” arXiv preprint
arXiv:2501.19319, 2025.
[32] M. Visentini-Scarzanella and H. Kawasaki, “Simultaneous camera, light
position and radiant intensity distribution calibration,” in Image and
Video Technology.
Springer, 2015, pp. 557–571.
[33] R. Modrzejewski, T. Collins, A. Hostettler, J. Marescaux, and A. Bartoli,
“Light modelling and calibration in laparoscopy,” International journal
of computer assisted radiology and surgery, vol. 15, no. 5, pp. 859–866,
2020.
[34] V. M. Batlle, J. M. Montiel, and J. D. Tard´os, “Photometric single-view
dense 3d reconstruction in endoscopy,” in 2022 IEEE/RSJ International
Conference on Intelligent Robots and Systems (IROS).
IEEE, 2022, pp.
4904–4910.
[35] M. Zwicker, H. Pfister, J. Van Baar, and M. Gross, “Ewa volume
splatting,” in Proceedings Visualization, 2001. VIS’01.
IEEE, 2001,
pp. 29–538.
[36] J. Chung, J. Oh, and K. M. Lee, “Depth-regularized optimization for 3d
gaussian splatting in few-shot images,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2024, pp. 811–
820.
[37] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai,
“Scaffold-gs: Structured 3d gaussians for view-adaptive rendering,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024, pp. 20 654–20 664.
[38] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian splatting
for geometrically accurate radiance fields,” in ACM SIGGRAPH 2024
conference papers, 2024, pp. 1–11.
[39] S. Lombardi, T. Simon, G. Schwartz, M. Zollhoefer, Y. Sheikh, and
J. Saragih, “Mixture of volumetric primitives for efficient neural render-
ing,” ACM Transactions on Graphics (ToG), vol. 40, no. 4, pp. 1–13,
2021.
[40] S. Imambi, K. B. Prakash, and G. Kanagachidambaresan, “Pytorch,” in
Programming with TensorFlow: solution for edge computing applica-
tions.
Springer, 2021, pp. 87–104.
[41] B. Kerbl, A. Meuleman, G. Kopanas, M. Wimmer, A. Lanvin, and
G. Drettakis, “A hierarchical 3d gaussian representation for real-time
rendering of very large datasets,” ACM Transactions on Graphics (TOG),
vol. 43, no. 4, pp. 1–15, 2024.
[42] T. L. Bobrow, M. Golhar, R. Vijayan, V. S. Akshintala, J. R. Garcia,
and N. J. Durr, “Colonoscopy 3d video dataset with paired depth from
2d-3d registration,” Medical image analysis, vol. 90, p. 102956, 2023.
