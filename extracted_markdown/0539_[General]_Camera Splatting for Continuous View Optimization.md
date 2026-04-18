<!-- page 1 -->
Camera Splatting for Continuous View Optimization
Gahye Lee
Hyomin Kim
Gwangjin Ju
Jooeun Son
Hyejeong Yoon
Seungyong Lee
POSTECH
{gahye0509, min00001, gwangjin, jeson, hjyoon02, leesy}@postech.ac.kr
(a) Rendered Camera Splats
(b) Camera Splatting
(c) View Synthesis Results
Point Camera
Point Camera
GT
FVS
Ours
View Dependency Score
Figure 1. We propose Camera Splatting, a novel view optimization framework for novel view synthesis. Each camera is modeled as a 3D
Gaussian, referred to as a camera splat, and virtual cameras, termed point cameras, are placed at 3D points sampled near the surface to
observe the distribution of camera splats. View optimization is achieved by continuously and differentiably refining the camera splats so that
desirable target distributions are observed from the point cameras, in a manner similar to the original 3D Gaussian splatting. Compared to
the Farthest View Sampling (FVS) approach, our optimized views demonstrate superior performance in capturing complex view-dependent
phenomena, including intense metallic reflections and intricate textures such as text.
Abstract
Given limited camera budgets, identifying informative
views is essential for effectively reconstructing a radiance
field to achieve high-quality novel view synthesis. These
views should adaptively sample the radiance field based on
the geometry and view-dependent appearance of the scene.
Existing methods typically optimize views by selecting from
a predefined camera set or adopting a greedy next-best-
view strategy, limiting the flexibility and accuracy of the
optimization. In this paper, we propose a novel continu-
ous view optimization framework, called Camera Splatting,
which enables joint optimization of multiple views. Our key
idea is to represent each camera as a specialized 3D Gaus-
sian (camera splat), allowing for gradient-based optimiza-
tion within a continuous camera parameter space. To eval-
uate view optimality, we introduce virtual cameras (point
cameras) placed at sampled points in the scene, which ren-
der camera splats and provide feedback on their spatial
positions and orientations. The resulting optimized views
effectively capture the view-dependent appearance of the
scene at the sampled points. Integrated into a Gaussian
splatting pipeline, our approach can efficiently and simul-
taneously optimize a large number of views. Quantitative
and qualitative evaluations demonstrate the effectiveness of
the views optimized by Camera Splatting for novel view syn-
thesis under a limited camera budget.
1. Introduction
Novel view synthesis aims to generate unseen views of a
3D scene from a limited set of captured images. Central to
this task is the radiance field, a continuous representation
arXiv:2509.15677v1  [cs.CV]  19 Sep 2025

<!-- page 2 -->
encoding both scene geometry and view-dependent appear-
ance [1, 8, 13, 21]. Formally, a radiance field is defined as
a continuous mapping from a 5D input space, consisting of
3D spatial coordinates x and 2D directions ω, to color c and
density σ.
Reconstructing a radiance field requires sampling across
both x and ω. In practice, this sampling is achieved by
capturing images from various camera views.
A dense
and uniform capture would yield the ideal and best recon-
struction. However, constraints such as a limited camera
budget and computational resource make this approach im-
practical. Selecting the most informative views is essential
for efficiently capturing the radiance field, enabling high-
quality novel view synthesis under a limited camera bud-
get [9, 14, 17, 19, 23].
Early researches on view optimization focused on geom-
etry reconstruction with constraints from multi-view geom-
etry [28, 33]. With recent advancements in novel view syn-
thesis, view optimization methods prioritize improving the
quality of novel view synthesis. They measure the quality
of the reconstructed radiance field to identify regions that
require additional sampling [9, 23]. However, the optimiza-
tion strategies of previous methods often yield suboptimal
results. Some methods use discrete optimization that se-
lects views from a predefined set of candidate camera posi-
tions [9, 14, 23, 30], which limits flexibility in the solution
space. Others employ a greedy next-best-view optimization
strategy [5, 16, 19, 32], iteratively selecting views one by
one. It improves the efficiency of view selection but could
result in local optima.
Ideally, a view optimization strategy should provide
adaptive and balanced sampling of the radiance field for
the target scene.
The strategy should also support con-
tinuous and flexible exploration of the camera parameter
space, eliminating the need for a predefined set of cam-
eras. In addition, all camera views should be optimized si-
multaneously, avoiding local minima that may arise from a
greedy next-best-view approach. Computational efficiency
and scalability are also essential to handle a large number
of optimized views.
In this paper, we present Camera Splatting, a novel view
optimization framework inspired by 3D Gaussian Splatting
(3DGS) [13], which supports continuous and joint opti-
mization of multiple cameras. Our core innovation is to
represent each camera as a specialized 3D Gaussian, called
a camera splat, that encodes optimizable camera parame-
ters for adaptively sampling the radiance field. We also in-
troduce virtual cameras, termed point cameras, which are
strategically placed at sampled points in the scene to evalu-
ate the current radiance field sampling achieved by the cam-
era splats. Leveraging the efficient and fully differentiable
3DGS optimization pipeline, our framework directly opti-
mizes camera parameters by rendering camera splats from
point cameras, enabling continuous, efficient, and scalable
optimization that can handle a large number of cameras si-
multaneously.
We validate our Camera Splatting framework by employ-
ing it to select optimal views for 3D Gaussian Splatting of
both object and scene scales. We demonstrate the effec-
tiveness of our framework through quantitative and qual-
itative evaluations on various camera configurations. We
also show that view-dependent effects, even in a scene with
complex materials and detailed textures, can be accurately
captured by our optimized views.
In summary, our main contribution is proposing a novel
continuous view optimization framework, Camera Splat-
ting, designed for a high-quality novel view synthesis,
which consists of:
• A novel view representation using a 3D Gaussian, cam-
era splat, that allows for continuous and differential opti-
mization of camera parameters.
• A novel virtual camera model, point camera, that enables
the evaluation of the radiance field sampling achieved by
camera splats
• A continuous, efficient, and scalable optimization pro-
cess, capable of simultaneously handling a large number
of cameras, inspired by 3DGS.
2. Related Work
View optimization methods have been extensively studied.
Early methods focused on geometry reconstruction, guided
by techniques such as multi-view geometry or stereo match-
ing. With advances in novel view synthesis, recent methods
focus on optimizing view selection for high-quality radi-
ance field reconstruction.
View optimization for geometry reconstruction
The
main goal of view optimization methods for geometry re-
construction is to maximize coverage, ensuring sufficient
geometric constraints between views [7, 15, 20, 27, 33].
For example, Aerial Path Planning (APP) [28] introduces
reconstructability heuristics for stereo matching by leverag-
ing pairwise relationships among views. While APP lever-
ages the downhill simplex method to optimize the view set,
Offsite Aerial Path Planning (OAPP) [33] employs bun-
dle adjustment for the optimization. Submodular [25] pro-
poses local surface coverage through angular diversity and
selects the best trajectories to maximize global scene cov-
erage. Although these geometry-driven approaches effec-
tively capture the complete surface, they do not consider
view-dependent appearances needed for novel view synthe-
sis.
View optimization for novel view synthesis
To identify
optimal views for novel view synthesis, several studies have

<!-- page 3 -->
proposed uncertainty metrics on radiance fields [10, 12, 16,
24, 26, 29]. FisherRF [9] leverages Fisher Information de-
rived from the Hessian matrix of the loss function to select
views that maximize information gain. ActiveNeRF [23]
defines uncertainty as color variance estimated from each
spatial position and trains the variance using NeRF [21]
framework. Manifold sampling [19] models the radiance
field using stochastic variables, considering the sampling
variance as an uncertainty measure to optimize the view.
Recently, Kopanas and Drettakas [14] proposed an alter-
native by explicitly considering observation frequency and
angular uniformity to achieve balanced view frequency and
directional diversity.
These approaches address view-dependent appearance
implicitly through radiance field uncertainty measures. In
contrast, our framework explicitly considers view depen-
dency in the optimization process, directly improving sam-
pling of viewing directions. This ensures high-quality ra-
diance field reconstructions with accurate view-dependent
details.
Optimization strategies
Different strategies have been
adopted to identify optimal views based on various defini-
tions of view optimality. For computational efficiency and
straightforward implementation, most existing methods rely
on discrete optimization such as view selection from candi-
date views [2, 6, 9, 14, 23, 31, 33]. This approach can re-
strict solution optimality due to the fixed candidate camera
set. Furthermore, the combinatorial complexity of this op-
timization grows exponentially with the number of views
being optimized.
To manage the complexity, Roberts et
al. [25] separately optimized rotation and translation pa-
rameters, which increases the risk of suboptimal solutions.
Alternatively, continuous optimization has also been
adopted for flexible exploration of the solution space.
APP [28] incrementally optimizes the camera configura-
tion by randomly selecting and optimizing one view at a
time in the continuous parameter space. Manifold Sam-
pling [19] proposes a differentiable uncertainty metric for
radiance fields, enabling gradient-based optimization of in-
dividual views. However, these approaches optimize views
with a greedy strategy that can fall into local optima.
Our framework addresses these limitations by simultane-
ously optimizing multiple camera views within a continu-
ous parameter space. Leveraging 3DGS-inspired optimiza-
tion, our framework achieves efficient and scalable joint op-
timization, effectively avoiding locally optimal solutions.
3. Radiance Field Sampling
3.1. Radiance Field
A Radiance Field is a core concept for representing the ap-
pearance of a 3D scene in a volumetric manner, which can
Figure 2. Illustration of sampling density and view optimality. In
both images, red circles indicate regions with weak view depen-
dency, while blue circles represent regions with strong view depen-
dency. As shown in the left image, sampling should concentrate
near surfaces, with density adapting to local view dependency. The
right image visualizes an ideal view configuration, where the view
distribution effectively captures the target surface with adaptive
sampling density based on view-dependency.
be defined by:
Fθ : (x, ω)
7→
(σ, c),
(1)
where x ∈R3 is a position vector, ω ∈S2 is a viewing di-
rection vector, and the output consists of a volumetric den-
sity σ ∈R and a view-dependent color c ∈R3.
Recently, 3D Gaussian Splatting (3DGS) [13] is pro-
posed to represent the radiance field effectively. In 3DGS,
each 3D Gaussian is characterized by parameters such as a
position µi, a covariance matrix Σi, opacity αi, and color ci
represented with spherical harmonics. For rendering, these
primitives are projected onto the 2D image plane, and their
colors are combined using alpha blending. They approx-
imate the volume rendering equation [11], reducing to a
simpler expression:
Lo(r) =
N
X
i=1
Ti αi ci,
Ti =
i−1
Y
j=1
(1 −αj),
(2)
where the Lo(r) represents the rendered radiance along ray
r, and Ti denotes the accumulated transmittance. This ap-
proach enables fully differentiable rendering for efficient
parameter update.
3.2. View Optimality for Radiance Field Sampling
Accurate radiance field reconstruction requires appropriate
sampling in both the spatial domain x and the viewing di-
rectional domain ω. However, dense sampling in both do-
mains is often impractical due to limited resources. View
selection needs to focus on important position and direction
with an effective sampling strategy.
In the spatial domain x, sampling should focus on re-
gions of interest in the scene, which would primarily be the
scene surfaces. The sampling in the directional domain ω
for a given position x should cover all viewing directions,

<!-- page 4 -->
(a) Proxy Geometry Initialization
(b) Camera Splat Initialization
(e) Optimized Views
Initial Proxy Geometry
Camera Splat
Point Camera
Initial Captured Image
(d) Camera Splat Rendering
(c) Self-Occlusion Mask
Figure 3. Overall process of view optimization using our Camera Splatting framework, highlighting the initialization, core components of
the framework, and the resulting optimized view configuration.
and the sampling density should adapt to the local view-
dependent appearance. For example, reflective surfaces de-
mand denser directional sampling compared to matte sur-
faces. As illustrated in Fig. 2, optimal sampling positions
are placed near scene surfaces, with varying sampling den-
sities depending on the view dependency.
With these desired sampling properties, we define opti-
mal views as cameras whose emitted view rays best approx-
imate the target sampling of both positions and viewing di-
rections in the radiance field. Optimal views should cover
all viewing directions at surface points, while the sampling
density in viewing directions should increase with the view-
dependency of surface points.
4. Camera Splatting
We introduce Camera Splatting, a novel continuous and
efficient view optimization framework inspired by 3DGS.
Given a proxy geometry and view dependency information
on the geometry, our framework simultaneously optimizes
all camera views to adaptively sample the radiance field.
4.1. Camera Splat
In Camera Spaltting, we model the physical cameras as spe-
cialized 3D Gaussians called camera splats C. Each cam-
era splat encodes both extrinsic and intrinsic properties of a
camera. These include its center position µ, rotation vector
r, uniform scale s, field-of-view (FoV) θ, and opacity α:
Ci = {ui, ri, s, θ, α}.
(3)
Here, FoV θ and opacity α are fixed constants, and the
scale parameter s is a shared scalar across all camera
splats. These shared parameters help identify overly clus-
tered views and maintain directional uniformity during op-
timization. The detailed rationale on this design choice is
discussed in Section 5.
4.2. Point Camera
We evaluate the camera splats by rendering them from vir-
tual cameras called point cameras P. Each point camera is
omnidirectional, allowing it to measure the directional sam-
pling density from all visible views at its position.
Point cameras are positioned on the proxy geometry to
represent regions of interest that likely contain scene sur-
faces. They also utilize the local view dependency informa-
tion of the proxy geometry. Details on the initialization of
point cameras are provided in the supplementary material.
4.3. Camera Splat Rendering at Point Camera
We render the opacity values of camera splats from point
cameras, based on the 3DGS rendering pipeline. During the
rendering process, we apply a binary visibility mask based
on FoV θ and orientation r of each camera splat. A camera
splat is considered visible from a point camera if the an-
gle between the camera splat’s orientation vector ri and the
vector vi pointing from the camera splat to the point cam-
era is within θ
2. Based on Eq. (2), our modified rendering
equation for a ray ˆr from a point camera Pi is defined as:
Irender(ˆr) =
N
X
i=1
Tiαimi
Ti =
i−1
Y
j=1
(1 −αimi)
(4)
where mi(Pi) =
(
1
if ∠(vi, ri) ≤θ
2
0
otherwise
.
Since camera splats share a fixed opacity value, the ren-
dered image reflects both the coverage and density of the
directional sampling.
Higher image intensities indicate
more overlapping camera splats, implying dense sampling
of view directions, while zero intensity implies no sampling
around that viewing direction.
Rendering camera splats from a point camera Pi may
mistakenly include those occluded by the proxy scene ge-
ometry. To address this, we introduce a self-occlusion mask
Iocc(Pi), computed by rendering the proxy geometry from
each point camera Pi (Fig. 4). This mask is applied to the

<!-- page 5 -->
Figure 4. Visualization of the self-occlusion mask (Section 4.3)
and the estimated View-Dependency Score (VDS) (Section 5.2).
(left) For a given point camera, the self-occlusion mask captures
visibility considering proxy geometry occlusions. (right) The VDS
is computed for each point camera to determine the optimal view-
ing direction sampling density.
rendered camera splats during optimization, effectively re-
ducing the influence of self-occluded camera splats.
5. Continuous View Optimization
Our optimization goal is to identify camera views that ef-
fectively sample the radiance field with adaptive density
of view directions using the proxy geometry and its view-
dependency information. To achieve this, we ensure that
each point on the proxy geometry is observed uniformly
from all viewing directions, with the sampling density
adapted to its local view-dependency characteristics.
5.1. Ground Truth Image
For each point camera, we define the ideal camera splat con-
figuration as being distributed uniformly over all viewing
directions emitting from the point camera. As noted in Sec-
tion 4.3, the pixel intensity rendered at a point camera re-
flects the density of camera splats in each viewing direction.
Consequently, a rendered image with uniform pixel inten-
sity suggests that the camera splats are evenly distributed
across viewing directions. Based on this observation, we
define a base ground truth image Ibase to be a monochro-
matic image with constant intensity equal to the fixed opac-
ity of camera splats.
5.2. View Dependency Score
Different materials exhibit unique reflectance properties, re-
quiring varying sampling densities across viewing direc-
tions. To achieve this, we introduce a View-Dependency
Score Function (VDSF), a predefined function that esti-
mates the view dependency of each point on the proxy
geometry (Fig. 4). The function indicates the relative di-
rectional sampling density needed at each point, enabling
adaptive allocation of views within a fixed camera budget.
We control the desired sampling density at each point
camera Pi by scaling the base ground truth image Ibase with
the estimated view-dependency score:
Igt(Pi) = VDSF(Pi) × Ibase.
(5)
This VDSF-based scaling strategy directly controls the an-
gular spacing between camera splats at each point camera
Pi. Higher Igt values lead to increased overlaps among pro-
jected camera splats, resulting in higher sampling density of
viewing directions for Pi.
In this paper, we define the VDSF as a data-driven cubic
polynomial function. Details of the VDSF are provided in
the supplementary material.
5.3. Gradient-based View Optimization
Image loss
Given N point cameras and camera splats C,
we define the image loss Limage as the Mean Squared Error
(MSE) between the rendered camera splat images Irender and
the ground truth images Igt. To account for self-occlusions,
we multiply the pixel-wise difference by the occlusion mask
Iocc:
Limage = 1
N
N
X
i=1
 Irender(Pi) −Igt(Pi)

⊙Iocc(Pi)
2
2 .
(6)
However, directly optimizing with this image loss raises
two issues: (1) If the scale of each camera splat is not glob-
ally coherent, the optimization can trivially adjust individ-
ual scales to match intensity targets (e.g., one big camera
splat and others with zero scale) without achieving direc-
tional uniformity. (2) The projected image-space scales of
camera splats can be manipulated with perspective fore-
shortening by adjusting the depths from the point camera.
Then, optimization can meet intensity requirements through
adjusting individual scales or depths, rather than improving
angular uniformity.
To prevent these undesirable solutions, we set the global
scale parameter to be shared among all camera splats, ensur-
ing each camera splat to have a similar projected area. Addi-
tionally, we normalize the scale of each camera splat based
on its depth from the point camera Pi. This normalization
mitigates perspective foreshortening, encouraging that all
camera splats have identical scales in image space. As a re-
sult, optimization genuinely targets directional uniformity.
Directional regularizer
To ensure that camera splats ori-
ent toward scene surfaces without causing under-covered
regions, we introduce a regularization term that encourages
orientations of the camera splats to align with the surface
parts they observe.
For the i-th point camera Pi and the j-th camera splat
Cj, we calculate the cosine similarity between the vector vi
which directs from Pi to Cj, and the camera splat’s rotation
vector rj. The directional regularizer is defined as:
Lreg =
N
X
i=1
M
X
j=1
m(i)
occw(i) cosSim
 rj, vi

,
(7)

<!-- page 6 -->
where m(i)
occ is a weight derived from the self-occlusion
mask Iocc of the i-th point camera for the ray vi, and w(i)
is a coverage weight for Pi. We assign a high value to the
weight w for a point camera with lower coverage, encourag-
ing camera splats to orient towards the point cameras with
low sampling density. Detailed formulation of the coverage
weight is provided in the supplementary material.
Boundary regularizer
We apply a boundary regularizer
Lbound that penalizes camera splats deviating from a pre-
defined boundary range. This regularization helps prevent
camera splats from drifting into invalid regions where they
are not visible to any point cameras and become unused dur-
ing optimization. See the supplementary for detailed formu-
lation.
Final loss
We define the final loss function as the sum of
the image loss and the regularizers:
Ltotal = Limage + Lreg + Lbound.
(8)
Since our framework is fully differentiable, camera splats
C are optimized by directly minimizing Ltotal.
6. View Optimization Using 3DGS
Our framework can optimize views for high-quality radi-
ance field reconstruction, given proxy geometry and view
dependency information defined over the proxy geome-
try.
The proxy geometry can be obtained through vari-
ous methods, including NeRF [21], 3D Gaussian Splatting
(3DGS) [13], and 3D reconstruction methods (e.g., Kinect-
Fusion [22]). Once optimized, the selected views can be
used for a range of radiance field reconstruction frame-
works, including NeRF and 3DGS.
In this paper, for our experiments, we adopt 3DGS to
reconstruct high-quality radiance fields, due to its recent
widespread adoption. We also use 3DGS to build the proxy
geometry for view optimization.
We reconstruct a proxy geometry by optimizing a 3DGS
scene using sparse input images and their associated camera
poses. Point cameras are then placed on the Gaussian prim-
itives of the optimized 3DGS, and the view dependency in-
formation is obtained for each point camera by computing
VDSF using the sparse input images. Camera splats are
randomly initialized around the proxy geometry obtained
by 3DGS.
During each optimization iteration, a subset of point
cameras is sampled, and camera splats are rendered from
the sampled point cameras to evaluate the directional sam-
pling densities. Before optimization, camera splats are cre-
ated from the views used for capturing the sparse input im-
ages, and they remain fixed during the optimization process
50 cameras
150 cameras
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
Manifold
25.99
0.850
0.146
26.47
0.866
0.136
FVS
26.20
0.866
0.140
26.72
0.880
0.130
Ours (w/o VDS)
26.50
0.874
0.227
27.26
0.891
0.120
Ours
27.05
0.883
0.126
28.46
0.910
0.106
Table 1. Quantitative comparison on close-view renderings for the
test set of the NSVF dataset.
50 cameras
150 cameras
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
Manifold
32.82
0.967
0.031
33.23
0.970
0.030
FVS
33.10
0.972
0.027
33.77
0.976
0.026
Ours (w/o VDS)
32.99
0.972
0.027
33.67
0.975
0.026
Ours
32.74
0.970
0.028
33.19
0.973
0.025
Table 2. Quantitative comparison on far-view renderings for the
test set of the NSVF dataset.
Indoor Christmas scene
Minimal room scene
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
Random
32.00
0.896
0.152
34.29
0.942
0.288
Ours (w/o VDS)
32.43
0.899
0.146
34.33
0.950
0.274
Ours
33.92
0.918
0.123
34.06
0.948
0.274
Table 3. Quantitative comparison on the test set of the BlenderKit
dataset.
to prevent the optimized views from overlapping with the
initial views.
Once the optimized camera splats are obtained, the scene
is re-scanned for a high-quality 3DGS reconstruction. De-
tailed pseudocode and additional training strategies are pro-
vided in the supplementary material.
This view optimization scenario can be applied to diverse
radiance field reconstruction tasks. For example, in large-
scale environments where manual assessment is challeng-
ing, an autonomous system can first capture a small set of
rough images. Our algorithm then suggests additional opti-
mal camera placements to the robot. This approach enables
an efficient, scalable, and fully automated scene acquisition
workflow.
7. Experiments
7.1. Experimental Setting
We evaluate our method against two recent view optimiza-
tion methods: Farthest View Sampling (FVS) and Manifold
Sampling [19]. FVS demonstrated the state-of-the-art per-
formance in NeRF Director [30], while Manifold Sampling
represents the state-of-the-art in continuous view optimiza-
tion. Additionally, we compare our method with and with-
out VDSF to demonstrate its effectiveness.
We conduct comparisons using the Synthetic-NSVF
dataset [18], which consists of eight diverse scenes. We
define the test set by uniformly sampling 200 views on both
a close and a far sphere around the scene center, totaling
400 images. This setup is the same as the camera space

<!-- page 7 -->
GT
FVS
Manifold
Ours
GT
FVS
Manifold
Ours
FVS
Manifold
Ours
Low VDS
High VDS
Figure 5. Qualitative comparison of optimized views. The fourth
column in the first row visualizes the View-Dependency Scores
(VDS) from two different point cameras, with blue indicating high
VDS (high view dependency) and red indicating low VDS. The
first three columns in the first row illustrate the final optimized
views. The second and third rows present rendered images ob-
tained from radiance fields reconstructed using these optimized
views.
used in FVS and the initial views of Manifold Sampling.
For each scene, our method begins with 20 initial views and
optimizes additional 50 and 150 views. To ensure fair eval-
uation, we provide the same 20 initial views as additional
views for FVS and Manifold Sampling.
Furthermore, we evaluate our method on large scale in-
door scenes, minimal room and indoor Christmas, high-
quality environments featuring distinct scene properties
from BlenderKit [3]. The test set is constructed by ran-
domly sampling 200 points on the surface of an ellipsoid
that fits within the empty space of each indoor scene, with
viewing directions aligned to the surface normals. For this
BlenderKit dataset, we use 100 initial views and optimize
300 additional views. The results are compared with a base-
line of randomly sampling 300 views from the volume.
7.2. Evaluation on View Optimization
We first compare our view optimization results with FVS
and Manifold Sampling on NSVF dataset. The quantitative
and qualitative comparisons are shown in Tables 1 and 2,
and Figs. 5 and 8. Our method demonstrates superior per-
formance in close-view reconstruction, as shown in Table 1.
Remarkably, our method with 50 additional cameras outper-
forms both FVS and Manifold Sampling with both 50 and
150 additional views. Fig. 8 illustrates that our method cap-
tures intricate view-dependent appearances, such as metal-
lic reflections in Wine Holder and detailed textures in Still
Life. These results describe the advantage of continuous and
unconstrained view optimization, especially under limited
camera budgets. The slightly lower PSNR compared to the
Iteration
Voronoi cell area
Coverage
Iteration
Figure 6. The left graph illustrates the Voronoi cell areas defined
by optimized camera splats over training iterations, reflecting their
directional sampling distribution. The right graph evaluates over-
all scene coverage across training iterations, comparing the results
obtained with and without the directional regularizer.
baselines in far-view evaluations (Table 2) is attributed to
pixel-level artifacts such as aliasing, whereas our method re-
mains competitive in structural similarity metrics like SSIM
and LPIPS.
As shown in Fig. 8, the optimized views from our frame-
work are positioned closer to the scene compared to other
methods.
When 3DGS is trained using these close-up
views, it may produce pixel-level artifacts, such as aliasing,
when rendered from far-views. This results in slightly lower
PSNR, a pixelwise metric, in far-view evaluations (Table 2).
However, our method remains competitive in structural sim-
ilarity metrics such as SSIM and LPIPS, indicating that the
overall scene structure and appearance are well preserved.
The experiments on BlenderKit,
the large indoor
scenes, are shown in Table 3 and Fig. 9.
Our method
achieves higher reconstruction quality on the Indoor Christ-
mas scene, which includes complex lighting and view-
dependent materials, such as semi-transparent curtains. In
Minimal Room scene, VDS makes minimal difference as
the scene consists of low view-dependency regions, reduc-
ing the impact of view-dependent sampling.
Regarding optimization latency, FVS completes view se-
lection in about 1 second, while Manifold Sampling takes
approximately 1 hour on the NSVF dataset. Our method
completes optimization in approximately 1 minute across
all datasets, including the BlenderKit dataset with 300 op-
timized views. This demonstrates that our method is both
scalable and efficient, making it suitable for a wide range of
scene types and sizes.
7.3. Analysis of Camera Splatting
We validate our core rationale of directional adjustment
guided by the View Dependency Score (VDS). We set up
a proxy geometry of a sphere divided into two hemispheres,
each assigned high and low VDS values, respectively. We
measure the density and directional uniformity for each
point camera with spherical Voronoi diagrams [4] of camera
splats as shown in the left graph of Fig. 6. The results show
higher density (smaller Voronoi cells) in the high-VDS re-
gion and lower density (larger Voronoi cells) in the low-

<!-- page 8 -->
VDS region. The reduced standard deviation of cell areas
confirms improved directional uniformity.
We evaluate the effectiveness of our directional regular-
izer in achieving global scene coverage. In this experiment,
the camera splats are initialized in two ways, random po-
sitions oriented toward the scene center and identical posi-
tions and orientations. We measure the coverage ratio of
point cameras for both initialization schemes, as shown in
the right graph of Fig. 6. Our regularizer encourages camera
splats to orient toward undersampled point camera, effec-
tively maximizing coverage across the entire set while the
optimization without directional regularizer fails to cover
the scene.
We also provide a detailed qualitative analysis of opti-
mized view placements and reconstruction results on the
wine holder scene from the NSVF dataset (Fig. 5). Our
method adaptively distributes camera views according to
the local View Dependency Score (VDS). Views are densely
placed around regions with high VDS, such as the metal-
lic wheel, while fewer views are positioned around regions
with low VDS, such as the diffuse box. This strategic allo-
cation effectively enhances the quality of view-dependent
appearance without compromising overall reconstruction
accuracy.
7.4. Discussions
Robustness to the proxy geometry
We evaluate the ro-
bustness of our method with respect to the quality of the
proxy geometry. We optimize 3DGS scenes using varying
numbers of initial input views, resulting in proxy geome-
tries of different quality levels.
As shown in Table 4, view optimization fails when the
proxy geometry is reconstructed with only 10 input views,
as the the 3DGS contains severe artifacts such as floating
blobs. These artifacts lead to poor point camera placement,
degrading the view optimization process. In contrast, when
using 20 or more input views, the proxy geometry contains
far fewer artifacts, enabling successful view optimization.
This experiment shows that our method consistently main-
tains high performance as long as the proxy geometry is not
severely degraded.
Densification for camera splats
While the 3DGS frame-
work employs densification to progressively increase the
number of Gaussians, our framework does not adopt den-
sification for camera splats. This is because our target cam-
era budget is predefined, and the coarse region of interest
is already known from the proxy geometry. This allows us
to initialize the camera splats in reasonable positions from
the beginning. We then jointly optimize all camera splats
simultaneously, eliminating the need for a separate densifi-
cation step.
10
20
50
100
PSNR ↑
25.55
29.63
29.21
28.95
SSIM ↑
0.863
0.929
0.929
0.925
LPIPS ↓
0.123
0.058
0.061
0.063
Table 4. Evaluation of the robustness of our method against the
quality of proxy geometry, which is reconstructed using varying
numbers of initial input views.
GT
Ours
Figure 7. Illustration of a failure case caused by sparse initial ge-
ometry in a scene dominated by textureless regions.
Failure cases and future directions
As we discussed
earlier, significantly flawed proxy geometries can degrade
optimization results.
In some indoor scenes dominated
by textureless regions such as uniformly colored walls,
3DGS tends to produce imbalanced distributions of Gaus-
sian primitives, as illustrated in Fig. 7. Such failures lead
to sparse 3D Gaussian distributions and imbalanced point
camera placements, resulting in suboptimal view optimiza-
tion quality. Future work could explore more robust proxy
geometries, such as coarse geometric primitives, to mitigate
these issues.
8. Conclusion
We introduce Camera Splatting, a novel gradient-based
view optimization framework designed for radiance field re-
construction. Departing from discrete view sampling meth-
ods, Camera Splatting leverages continuous exploration of
the joint view parameter space, effectively capturing com-
plex view-dependent effects and covering entire scene ge-
ometry.
Thanks to the computational efficiency of the
Gaussian Splat rendering technique, our method achieves
rapid evaluation of viewpoint quality, resulting in conver-
gence within one minute, even during simultaneous batch
optimization of a large number of views. Our framework
provides highly informative views, facilitating more effi-
cient and accurate radiance field reconstructions, beneficial
across diverse scenarios and scene scales.
References
[1] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P.
Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded
anti-aliased neural radiance fields. CVPR, 2022. 2
[2] Christian Beder and Richard Steffen. Determining an initial
image pair for fixing the scale of a 3d reconstruction from an

<!-- page 9 -->
image sequence. In Joint Pattern Recognition Symposium,
pages 657–666. Springer, 2006. 3
[3] BlenderKit, 2021. 7
[4] Manuel Caroli, Pedro MM de Castro, S´ebastien Loriot,
Olivier Rouiller, Monique Teillaud, and Camille Wormser.
Robust and efficient delaunay triangulations of points on or
close to a sphere. In Experimental Algorithms: 9th Inter-
national Symposium, SEA 2010, Ischia Island, Naples, Italy,
May 20-22, 2010. Proceedings 9, pages 462–473. Springer,
2010. 7
[5] Xiao Chen, Quanyi Li, Tai Wang, Tianfan Xue, and Jiang-
miao Pang. Gennbv: Generalizable next-best-view policy for
active 3d reconstruction. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 16436–16445, 2024. 2
[6] Enrique Dunn and Jan-Michael Frahm. Next best view plan-
ning for active model improvement. In BMVC, pages 1–11,
2009. 3
[7] Xinyi Fan, Linguang Zhang, Benedict Brown, and Szymon
Rusinkiewicz. Automated view and path planning for scal-
able multi-object 3d scanning. ACM Transactions on Graph-
ics (TOG), 35(6):1–13, 2016. 2
[8] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong
Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels:
Radiance fields without neural networks. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 5501–5510, 2022. 2
[9] Wen Jiang, Boshu Lei, and Kostas Daniilidis.
Fisherrf:
Active view selection and uncertainty quantification for
radiance fields using fisher information.
arXiv preprint
arXiv:2311.17874, 2023. 2, 3
[10] Liren Jin, Xieyuanli Chen, Julius R¨uckin, and Marija
Popovi´c. Neu-nbv: Next best view planning using uncer-
tainty estimation in image-based neural rendering. In 2023
IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS), pages 11305–11312. IEEE, 2023. 3
[11] James T Kajiya. The rendering equation. In Proceedings of
the 13th annual conference on Computer graphics and inter-
active techniques, pages 143–150, 1986. 3
[12] Alex Kendall and Yarin Gal. What uncertainties do we need
in bayesian deep learning for computer vision? Advances in
neural information processing systems, 30, 2017. 3
[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 2, 3, 6
[14] Georgios Kopanas and George Drettakis.
Improving nerf
quality by progressive camera placement for free-viewpoint
navigation. 2023. 2, 3
[15] Michael Krainin, Brian Curless, and Dieter Fox.
Au-
tonomous generation of complete 3d object models using
next best view manipulation planning. In 2011 IEEE interna-
tional conference on robotics and automation, pages 5031–
5037. IEEE, 2011. 2
[16] Soomin Lee, Le Chen, Jiahao Wang, Alexander Liniger,
Suryansh Kumar, and Fisher Yu. Uncertainty guided pol-
icy for active robotic 3d reconstruction using neural radiance
fields. IEEE Robotics and Automation Letters, 7(4):12070–
12077, 2022. 2, 3
[17] Monica MQ Li, Pierre-Yves Lajoie, and Giovanni Beltrame.
Frequency-based view selection in gaussian splatting recon-
struction. arXiv preprint arXiv:2409.16470, 2024. 2
[18] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and
Christian Theobalt. Neural sparse voxel fields. Advances
in Neural Information Processing Systems, 33:15651–15663,
2020. 6, 11
[19] Linjie Lyu, Ayush Tewari, Marc Habermann, Shunsuke
Saito, Michael Zollh¨ofer, Thomas Leimk¨uhler, and Christian
Theobalt. Manifold sampling for differentiable uncertainty
in radiance fields. In SIGGRAPH Asia 2024 Conference Pa-
pers, pages 1–11, 2024. 2, 3, 6
[20] Miguel Mendoza, J Irving Vasquez-Gomez, Hind Taud,
L Enrique Sucar, and Carolina Reta.
Supervised learning
of the next-best-view for 3d object reconstruction. Pattern
Recognition Letters, 133:224–231, 2020. 2
[21] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In ECCV, 2020. 2, 3, 6
[22] Richard A. Newcombe, Shahram Izadi, Otmar Hilliges,
David Molyneaux, David Kim, Andrew J. Davison, Push-
meet Kohi, Jamie Shotton, Steve Hodges, and Andrew
Fitzgibbon. Kinectfusion: Real-time dense surface mapping
and tracking. In 2011 10th IEEE International Symposium
on Mixed and Augmented Reality, pages 127–136, 2011. 6
[23] Xuran Pan, Zihang Lai, Shiji Song, and Gao Huang. Ac-
tivenerf: Learning where to see with uncertainty estimation.
In European Conference on Computer Vision, pages 230–
246. Springer, 2022. 2, 3
[24] Yunlong Ran, Jing Zeng, Shibo He, Jiming Chen, Lincheng
Li, Yingfeng Chen, Gimhee Lee, and Qi Ye. Neurar: Neural
uncertainty for autonomous 3d reconstruction with implicit
neural representations. IEEE Robotics and Automation Let-
ters, 8(2):1125–1132, 2023. 3
[25] Mike Roberts, Debadeepta Dey, Anh Truong, Sudipta Sinha,
Shital Shah, Ashish Kapoor, Pat Hanrahan, and Neel Joshi.
Submodular trajectory optimization for aerial 3d scanning.
In International Conference on Computer Vision (ICCV)
2017, 2017. 2, 3
[26] Luca Savant, Diego Valsesia, and Enrico Magli.
Mod-
eling uncertainty for gaussian splatting.
arXiv preprint
arXiv:2403.18476, 2024. 3
[27] William R Scott, Gerhard Roth, and Jean-Franc¸ois Rivest.
View planning for automated three-dimensional object re-
construction and inspection.
ACM Computing Surveys
(CSUR), 35(1):64–96, 2003. 2
[28] Neil Smith, Nils Moehrle, Michael Goesele, and Wolfgang
Heidrich. Aerial path planning for urban scene reconstruc-
tion: A continuous optimization method and benchmark.
2018. 2, 3
[29] Niko S¨underhauf, Jad Abou-Chakra, and Dimity Miller.
Density-aware nerf ensembles: Quantifying predictive un-
certainty in neural radiance fields.
In 2023 IEEE Inter-
national Conference on Robotics and Automation (ICRA),
pages 9370–9376. IEEE, 2023. 3

<!-- page 10 -->
[30] Wenhui
Xiao,
Rodrigo
Santa
Cruz,
David
Ahmedt-
Aristizabal, Olivier Salvado, Clinton Fookes, and Leo Le-
brat. Nerf director: Revisiting view selection in neural vol-
ume rendering. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 20742–
20751, 2024. 2, 6
[31] Zimu Yi, Ke Xie, Jiahui Lyu, Minglun Gong, and Hui Huang.
Where to render: Studying renderability for ibr of large-scale
scenes. In 2023 IEEE Conference Virtual Reality and 3D
User Interfaces (VR), pages 356–366. IEEE, 2023. 3
[32] Huangying Zhan, Jiyang Zheng, Yi Xu, Ian Reid, and Hamid
Rezatofighi. Activermap: Radiance field for active mapping
and planning. arXiv preprint arXiv:2211.12656, 2022. 2
[33] Xiaohui Zhou, Ke Xie, Kai Huang, Yilin Liu, Yang Zhou,
Minglun Gong, and Hui Huang. Offsite aerial path planning
for efficient urban scene reconstruction. ACM Transactions
on Graphics (TOG), 39(6):1–16, 2020. 2, 3

<!-- page 11 -->
Farthest View 
Sampling
Manifold
Sampling 
Ours 
(w/o VDS)
Ours
GT
Still Life
Steam Train
Toad
Figure 8. Qualitative evaluation on the Synthetic-NSVF dataset [18]. The first row shows renderings from optimized views that show the
overall scenes from a distant perspective. The second row provides close-up renderings of the regions highlighted by pink squares, our
view optimization successfully captures detailed radiance fields and intricate view-dependent appearances.

<!-- page 12 -->
Random Sampling
Ours (w/o VDS)
Ours 
GT
Minimal Room Scene
Indoor Christmas Scene
Figure 9. Qualitative evaluation on the BlenderKit dataset. Each row shows close-up renderings corresponding to regions highlighted by
pink squares in the first column images. Radiance fields reconstructed using our optimized viewpoints effectively capture detailed geometry
and complex view-dependent appearances.
