<!-- page 1 -->
StreamME: Simplify 3D Gaussian Avatar within Live Stream
LUCHUAN SONG, University of Rochester, USA
YANG ZHOU, Adobe Research, USA
ZHAN XU, Adobe Research, USA
YI ZHOU, Adobe Research, USA
DEEPALI ANEJA, Adobe Research, USA
CHENLIANG XU, University of Rochester, USA
17 dB
21 dB
26 dB
“Joker”
(d) Environment Relighting
(a) Training in Live Stream
(b) Real-Time Animation
(c) Text Toonify in 5min
“Caricature”
1s         …
10s         …
5min
shadow
Fig. 1. The StreamME takes live stream (or monocular) video as input to enable rapid 3D head avatar reconstruction. It achieves impressive speed, capturing
the basic facial appearance within 10 seconds (PSNR = 21 dB) and reaching high-quality fidelity (PSNR = 26 dB) within 5 minutes, as shown in (a). Notably,
StreamME reconstructs facial features through on-the-fly training, allowing simultaneous recording and modeling without the need for pre-cached data
(e.g. pre-training model). Additionally, StreamME facilitates real-time animation in (b), toonify in (c) and relighting in (d) (the background light image is shown
in bottom right and we move the light position to create shadows on face) from the 5-minute reconstructed appearance, supporting the applications in VR and
online conference. Natural face©Xuan Gao et al. (CC BY), and ©Wojciech Zielonka et al. (CC BY).
We propose StreamME, a method focuses on fast 3D avatar reconstruction.
The StreamME synchronously records and reconstructs a head avatar from
live video streams without any pre-cached data, enabling seamless integra-
tion of the reconstructed appearance into downstream applications. This
exceptionally fast training strategy, which we refer to as on-the-fly training,
is central to our approach. Our method is built upon 3D Gaussian Splat-
ting (3DGS), eliminating the reliance on MLPs in deformable 3DGS and
relying solely on geometry, which significantly improves the adaptation
speed to facial expression. To further ensure high efficiency in on-the-fly
training, we introduced a simplification strategy based on primary points,
which distributes the point clouds more sparsely across the facial surface,
optimizing points number while maintaining rendering quality. Leveraging
the on-the-fly training capabilities, our method protects the facial privacy
and reduces communication bandwidth in VR system or online conference.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than ACM
must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,
to post on servers or to redistribute to lists, requires prior specific permission and/or a
fee. Request permissions from permissions@acm.org.
SIGGRAPH Conference Papers ’25, August 10–14, 2025, Vancouver, BC, Canada
© 2025 Association for Computing Machinery.
ACM ISBN 979-8-4007-1540-2/2025/08...$15.00
https://doi.org/10.1145/3721238.3730635
Additionally, it can be directly applied to downstream application such as
animation, toonify, and relighting. Please refer to our project page for more
details: https://songluchuan.github.io/StreamME/.
CCS Concepts: • Computing methodologies →Animation; Motion pro-
cessing.
Additional Key Words and Phrases: On-the-fly Training, 3D Gaussian Splat-
ting, Point-Clouds Simplification
ACM Reference Format:
Luchuan Song, Yang Zhou, Zhan Xu, Yi Zhou, Deepali Aneja, and Chen-
liang Xu. 2025. StreamME: Simplify 3D Gaussian Avatar within Live Stream.
In Special Interest Group on Computer Graphics and Interactive Techniques
Conference Conference Papers (SIGGRAPH Conference Papers ’25), August
10–14, 2025, Vancouver, BC, Canada. ACM, New York, NY, USA, 12 pages.
https://doi.org/10.1145/3721238.3730635
1
INTRODUCTION
The rapid reconstruction of head avatars reconstruction and reen-
actment of facial expression dynamics from a single video have
become a rapidly advancing research tpoic, with vast potential for
applications in VR/AR, digital human development, holographic
communication, live streaming, and more. Recently, the volumet-
ric models (e.g. Instance-NeRF [Liu et al. 2023] and 3DGS [Kerbl
1
arXiv:2507.17029v1  [cs.GR]  22 Jul 2025

<!-- page 2 -->
SIGGRAPH Conference Papers ’25, August 10–14, 2025, Vancouver, BC, Canada
Luchuan Song, Yang Zhou, Zhan Xu, Yi Zhou, Deepali Aneja, and Chenliang Xu
et al. 2023]) have endeavored to achieve both high-quality and effi-
cient rendering. For instance, INSTA [Zielonka et al. 2022] employs
Instant-NGP [Müller et al. 2022] to accelerate rendering through
engineering optimizations. AvatarMAV [Xu et al. 2023] leverages
the learnable blendshape as motion representation to achieve fast
recovery of head avatar. FlashAvatar [Xiang et al. 2023] simulates
the head avatar with a large number of Gaussian points in UV space.
However, they continue to face challenges in balancing rendering
quality and storage overhead, which constrains their applicability
in consumer applications.
In this paper, we advance rapid facial reconstruction techniques
to address the existing limitations. Additionally, we introduce a
novel head avatar reconstruction task, termed on-the-fly training
for reconstruction, which pushes the efficiency boundaries of fast
reconstruction even further. Based on these observations, prior
methods have uniformly separated training and inference processes
due to efficiency constraints (refer as offline training). Such as, while
AvatarMAV [Xu et al. 2023] achieves efficient training speeds offline,
it cannot support frame-by-frame training for reconstruction within
the live streaming. Our on-the-fly training approach offers multiple
advantages, including (i) protect facial privacy by eliminating the
need to pre-cached personal facial models on the external machines,
(ii) only 3DGS parameters are transmitted in stream video, rather
than the full images (about 70% compression) and (iii) the synchro-
nous training and recording with real-time visualization, allowing
for immediate re-recording of under-trained facial areas.
We propose a novel on-the-fly head avatar reconstruction method
named StreamME. Different from the all previous head avatar
reconstruction methods [Qian et al. 2023; Song et al. 2021a; Wang
et al. 2023; Xiang et al. 2023; Xu et al. 2024, 2023; Zheng et al. 2022,
2023], the StreamME avoids dependence on multiple MLP layers to
capture deformable facial dynamics (e.g., facial expression motion),
significantly reducing expression recovery time and enabling true
on-the-fly training. Specifically, we attach the 3D Gaussian point
clouds to the tracked head mesh surface, allowing the points to
move in tandem with mesh deformations. However, the point clouds
associated with the deformed mesh on the 3D head template do not
fully preserve the geometric properties of the face, which results
in noise cloud artifacts around the rendered face and reducing the
realism. From our method, we dynamically adjust the initial 3D
Gaussian points through anchor-based pruning-and-clone strategy.
Instead of selecting all points from the tracked head mesh as 3D
Gaussian points, we identify specific anchor points that accurately
capture facial motion. The 3D Gaussian points are then updated
based on these selected anchors, optimizing for head representation.
This strategy improves efficiency from eliminating points that do
not contribute to facial motion, while preserving the motion anchor
points critical for controlling facial deformation.
Meanwhile, we find in practice that more replicated 3D Gaussian
points will lead to better quality but reduce speed, especially the
training speed involving backpropagation. Therefore, we explored
a method to gradually simplify the point clouds, which balance the
number of point clouds and rendering quality. Here, we introduce
two assumptions for simplifying point clouds: (i) the points should
be distributed around the facial surface, rather than within it, as
internal points remain unobservable due to occlusion; (ii) the small-
size 3D Gaussian points with minimal volume, contribute negligibly
to image quality and the impact is imperceptible. In optimization,
these two assumptions serve as foundational principles. We ensure
that 3D Gaussians are progressively distributed around and outside
the surface, while occluded and small-sized points are removed, en-
hancing execution speed. This strategy yields a sparser 3D Gaussian
representation of the head avatar, substantially improving efficiency
without compromising rendering quality.
With the help of Motion-Aware Anchor Points selection and Gauss-
ian Points Simplification strategy, we achieve on-the-fly photo-realistic
head avatar representation within approximately 5 minutes of live
streaming, as shown in Figure 1 (a). Moreover, the 3D Gaussian
properties learned within 5 minutes can be applied to cross-identity
head animation, facial toonification, environment relighting, and
other applications with minimal fine-tuning, as illustrated in Fig-
ure 1 (b, c, d). This flexibility significantly broadens the application
scope of our method. Furthermore, we demonstrate the superiority
of our method through extensive experiments and comparisons
with both instant and long-term training approaches. In summary,
our contributions include the following aspects:
(1) We present the on-the-fly head avatar reconstruction method,
which is able to reconstruct facial appearance from the live streams
within about 5 minutes by pure Pytorch code. To the best of our
knowledge, we are the first to reconstruct and visualize the head
avatar within the on-the-fly training.
(2) We emphasize the efficiency in training, and introduce motion
saliency anchor selection and point cloud simplification strategy.
The anchor selection minimizes reliance on MLPs within the de-
formation field, while point cloud simplification strategy reduces
computational redundancy from 3D Gaussian points.
(3) A series of downstream applications are attached, which have
demonstrated the advances of our approach and provided novel
insight for the on-the-fly training method.
2
RELATED WORKS
2.1
Instant Head Avatar
The instant head avatar is an evolving field rooted in traditional
photorealistic head reconstruction, and improved by novel tech-
niques that reduce dependency on long-term training (e.g., StyleA-
vatar [Wang et al. 2023], IM Avatar [Zheng et al. 2022], PointA-
vatar [Zheng et al. 2023], Deep-Video-Portrait [Kim et al. 2018], NeR-
Face [Gafni et al. 2021], KeypointNeRF [Mihajlovic et al. 2022] e.t.c)
to achieve high-quality results. Recently, with the introduction of
3DGS [Kerbl et al. 2023] and NeRF acceleration [Müller et al. 2022]
has driven rapid advancements in this field. The most notable works
in this area include INSTA [Zielonka et al. 2023], AvatarMAV [Xu
et al. 2023], and FlashAvatar [Xiang et al. 2023]. And the INSTA
and FlashAvatar utilize mesh geometry sampling, with Instant-NGP
and 3DGS employed for accelerated rendering, respectively. The
AvatarMAV [Xu et al. 2023] employs blendshapes and uses the learn-
able MLPs to blend multiple implicit representations. Beyond these
approaches, other head avatar reconstruction methods [Gafni et al.
2021; Kim et al. 2018; Thies et al. 2016; Wang et al. 2023; Zheng et al.
2023] usually require several hours to several days to complete.
2

<!-- page 3 -->
StreamME: Simplify 3D Gaussian Avatar within Live Stream
SIGGRAPH Conference Papers ’25, August 10–14, 2025, Vancouver, BC, Canada
a) Monocular Track with Mask
i) 3DGS Properties Warming-Up (Optional)
Initialized Texture/Normal
Optimized Texture/Normal
{𝛼, s, r, o*, 𝜇, Tex, SH}
c) Displaced Coarse Mesh
{𝛼+∆𝛼, s, r, o*, 𝜇, Tex, SH}
+ MLP
b) Warm-Up with Auxiliary 3DGS Properties
+ ∆𝛼
{𝛼,
s,
r,
o,
𝜇,
Idx,
Tex*,
SH*}
GT
Mesh
Sampled
Points
iii) On-The-Fly Reconstruction
Participant
Camera Mount
On-The-Fly Reconstruction
ii) StreamME Pipeline via Anchor Duplication and Simplification
System Setup
{𝛼+∆𝛼,
s +∆𝑠,
r +∆𝑟,
o +∆𝑜,
𝜇+∆𝜇,
Idx,
Tex*,
SH*}
d) On-The-Fly Initialization
e) Anchor Points Duplication
f) Gaussian Points Simplification
g) Rendered Head
Differentiable Tile 
Rasterizer
N = 16053
N = 7741
N = 10149
Global 
Rot and Translation
: Webcam (on-the-fly) or monocular input (offline)
o*
: The binary mask which is initialized by points
: The opacity property  is not included in optimization
Tex : Learnable texture property to assist deformation
SH
: Learnable lighting property assist deformation
Tex* : Frozen texture property in on-the-fly setup 
SH* : Frozen lighting property in on-the-fly setup 
Idx from 
Sampled Points
Idx
Fig. 2. The overview the pipeline of StreamME. We list three components at here. i) The 3DGS Properties Warming-Up (Optional): we introduce two auxiliary
learnable 3D Gaussian attribute texture and illumination, refining the UV vertex positions to improve facial geometry detail (e.g. here, we show the coarse
displacement for the vertices around the hair). This step is optional, and the users may also opt to use the tracked head without displacement. ii) Anchor
Duplication and Simplification: we freeze the Tex and SH attributes and introduce a binary learnable mask, initialized with all values set to 1, from the UV
vertices sampled on the mesh. Natural face©Xuan Gao et al. (CC BY).
Additionally, the above methods utilize linearly decomposed fa-
cial expression parameters as coarse conditions, by neural convo-
lution or MLP layers to refine the details such as hair and mouth
details. In contrast to these approaches, we maintain high fidelity
with more efficient in training and inference. With only the 3D
Gaussian primitives attached to explicit geometry, we significantly
reduce the learning burden from learnable neural networks and
accelerate the training process. To our knowledge, this is the first
work to achieve on-the-fly training for head avatar reconstruction.
2.2
Real-Time Face Estimation
The on-the-fly head avatar reconstruction depends on both efficient
3DGS training and real-time 3D head geometry estimation. As in
previous works [Cao et al. 2022; Chan et al. 2022; Liu et al. 2022;
Shao et al. 2024; Song et al. 2023, 2021b; Wang et al. 2022; Xiang
et al. 2023; Xu et al. 2023], the preprocessed data is derived from
estimating the pose and deformation of the facial template with
tools such as MICA [MIC 2022]. It is highly time-consuming, for
instance, processing a 10-minute video can require one day, which
severely limiting its real-time applicability. In our case, we integrate
a real-time face estimation module to supply pose and head mesh
data on-the-fly within the pipeline. Notably, we have meticulously
designed the system to balance resource allocation across facial
parsing, head tracking, and 3DGS reconstruction within the real-
time pipeline.
2.3
Deformable-3DGS Representations
The 3D Gaussian Splatting (3DGS) [Kerbl et al. 2023] is designed
to represent and render static 3D scenes. Building on static 3DGS,
deformable-3DGS [Bae et al. 2024; Chen et al. 2024; Li et al. 2023;
Luiten et al. 2024; Qian et al. 2024; Song et al. 2024b; Tu et al.
2024; Zhu et al. 2024] incorporates MLPs or CNNs to predict and
render geometric deformations, with 3D Gaussian properties pre-
dicted frame-by-frame over time. Specifically, these methods retain
a canonical 3D Gaussian space, optimizing the MLP-based deforma-
tion field [Zhang et al. 2023, 2024a,b] conditioned on timestamps.
Our method also employs the canonical-to-world space strategy.
First, the tracked deformed meshes are positioned within canonical
space. Then, 3D Gaussian points are placed around the surface
and transformed into world space, incorporating pose information
for rendering. Additionally, our method eliminates the need for
learnable layers to predict deformations, instead, the deformations
are derived from 3D Gaussian points around the meshes which are
deformed from canonical space.
3
METHOD
The StreamME generates the 3D head avatar within few minutes
from streaming. And progressively simplifying the 3D Gaussian
point representation of the face while retaining anchor points essen-
tial to capturing facial motion. In this section, we will introduce the
simplification 3D Gaussians representation and the system pipeline,
which is also shown in Figure 2.
3.1
Preliminaries
In our approach, we build upon the 3DGS [Kerbl et al. 2023], a point-
based representation from 3D Gaussian properties for dynamic 3D
head reconstruction. In 3DGS, a covariance matrix (Σ) and the mean
of point (𝜇) are defined for each Gaussian 𝑔:
𝑔= 𝑒−1
2 (x−𝜇)𝑇Σ−1(x−𝜇).
(1)
For differentiable optimization, the covariance matrix Σ is decom-
posed into a rotation matrix R and a scaling matrix S, parameterized
by a learnable quaternion 𝑟and scaling vector 𝑠. For spatial at-
tributes, the 3D Gaussians possess the color of the projected pixels
3

<!-- page 4 -->
SIGGRAPH Conference Papers ’25, August 10–14, 2025, Vancouver, BC, Canada
Luchuan Song, Yang Zhou, Zhan Xu, Yi Zhou, Deepali Aneja, and Chenliang Xu
Anchor Points
Anchor with Bind Points
(a) The initialize anchor points and binding points
Before Motion-Aware by Idx
After Motion-Aware by Idx
(b) Motion-Aware Idx for Simplification
Before Simplification by Idx
After Simplification by Idx
(c) Simplification the over-small size/opacity points 
Anchor Duplication
Anchor Points
Fig. 3. We employ the proposed Idx for anchor binding, elimination the motion irrelevant points and simplification the over-small size/opacity points to
reduce computational overhead. Specifically, (a) The anchor points are sampled from the mesh and multiple duplicated for detail representation. (b) The
motion-aware Idx tends to remove points, as there are none motion gradients around the forehead in canonical space. (c) The learnable masks from Idx for
deleting the small size and opacity points within training. Please zoom-in for details. Natural face©Xuan Gao et al. (CC BY).
are from the splatting and overlapping of the 3D Gaussians points:
C =
∑︁
𝑖∈𝑁
c𝑖𝛼𝑖
𝑁−1
Ö
𝑗=1
(1 −𝛼𝑗).
(2)
Here, 𝛼𝑖represents the density derived from the projection of 3D
Gaussians with Σ and opacity𝑜, and 𝑁denotes the number of points.
Consequently, we define the 3D Gaussian attributes as follows:
𝑔= {𝛼,𝑠,𝑟,𝑜, 𝜇}.
(3)
The attributes are optimized in backpropagation rendering pipeline.
Generally, Gaussian properties derived from precise geometry result
in higher rendering quality.
3.2
Gaussian Properties Warm-up
Inspired by previous works [Qian et al. 2023; Xu et al. 2024; Zielonka
et al. 2024], we adjust vertices in the tracked coarse head geometry
to account for deformations caused by features like hair, the stan-
dard facial templates fall short in representing the full head shape.
However, the preprocessing step in those method (e.g. DMTet [Shen
et al. 2021] or VHAP [Qian 2024]) require substantial time to re-
orient the geometry and frequently suffer from collapse due to the
limitations of single or sparse views.
We propose a warm-up step that introduces additional learnable
texture (Tex) and illumination (SH) parameters to assist vertex
deformation via 3DGS, achieving reorientation within 20 seconds.
Specifically, we follow FlashAvatar [Xiang et al. 2023] to initialize
the 3D Gaussian points from UV coordinates. Then, the vertex
positions are learned in 3D Gaussian properties 𝛼. We apply the
learnable texture with illumination to the diffuse color features
of each 3D Gaussian point and the normal consistency between
adjacent faces on a mesh surface to keep the smoothness, as shown in
Figure 2. It is worth noting that this explicit geometric deformation
step is optional and could be skipped to improve the operation
efficiency. Furthermore, the auxiliary learnable parameters Tex and
SH are optimized only during the warm-up step and remain frozen
in subsequent stages. In practice, we set the static Direct-Current
component (the self.feature_dc) in 3DGS as the SH.
3.3
Motion-Aware Anchor Points
Our primary optimization strategy for 3D Gaussians involves grad-
ually removing points irrelevant to facial motion and duplicating
those that contribute significantly (termed as anchor points), as
shown in Figure 3 (a). For static scene or object reconstruction, the
gradient accumulation of each 3D Gaussians position in the opti-
mizer serves as the basis for duplication and pruning. However, this
point cloud optimization method is not suitable for our task, as the
regions around the eyes and teeth is dynamic.
Meanwhile, we initialize a learnable binary auxiliary Gaussian
attribute from the UV vertices, named as Idx ∈{0, 1}𝑁(𝑁is the
number of points), as shown in Figure 2. The accumulation of density
gradients does not only come from the optimized gradients but also
from positional differences relative to the normalized canonical
point cloud. In practice, we apply the canonical point cloud (¯P0)
of the face in the first frame as the normalized reference. Then for
each Gaussian point, we calculate the gradient as the following:
Idx =
1,
< 𝛼𝑔𝑟𝑎𝑑, ∇P >max≥𝜖,
0,
< 𝛼𝑔𝑟𝑎𝑑, ∇P >max< 𝜖.
(4)
Here,𝜖represents the threshold for the maximum between optimizer
and motion gradient, the ∇P is the accumulated gradient of P relative
to ¯P0 (first-order difference). Points with a value of 1 in Idx will be
cloned, while those with a value of 0 will be pruned. The 𝛼𝑔𝑟𝑎𝑑is
the gradient of density.
This approach retains points that contribute to facial motion
while eliminating those do not, as in Figure 3 (b). These motion
anchors effectively control the proliferation of non-contributory
points and adapt swiftly to facial expression changes without requir-
ing additional neural networks (e.g., MLPs or CNNs) for fitting facial
motion via expression. However, due to duplication, the excessive
density around the anchor points leads to additional computational
overhead as the anchor points and their duplicates proliferate expo-
nentially.
4

<!-- page 5 -->
StreamME: Simplify 3D Gaussian Avatar within Live Stream
SIGGRAPH Conference Papers ’25, August 10–14, 2025, Vancouver, BC, Canada
3.4
Gaussian Points Simplification
To regulate the over-proliferation of Gaussian anchor points, the
basic strategy periodically reduces opacity values, removing points
that consistently remain at low opacity levels. However, this opacity-
based control method is effective in static scene reconstruction,
but it conflicts with our method. Generally, points that are fully
transparent (with opacity 0) typically do not contribute to motion,
such as those located on the forehead around the head. Then, we
propose two methods for optimizing the number of these points,
removing excessively small points and aligning remaining points
closer to geometric surface. Points positioned closer to surface better
represent details, while those farther away tend to introduce noise.
We apply the learnable binary index Idx as the mask on the
opacity property 𝑜and scale property 𝑠(updated to 𝑜′,𝑠′), as:
[𝑠′,𝑜′] = Idx · [𝑠,𝑜],
(5)
which means that those Gaussian points with motion gradient con-
tributions (Idx value with 1) are additionally applied on the scale
and opacity. Then, we adopt the straight-through estimator [Bengio
et al. 2013; Lee et al. 2024; Van Den Oord et al. 2017] for gradient
calculation with binary parameters, as the following:
Idx′ = sigm(Idx) + sg[U[sigm(Idx) > 𝜖′] −sigm(Idx)],
(6)
where the sigm, sg, U are the sigmoid function, stop gradients and
indicator function (mapping with 0 and 1). The 𝜖′ are the pre-defined
threshold, which is set to 0.01. During forward-propagation, the
mask applied on 𝑜and 𝑠via the value of Idx. In back-propagation,
gradients are obtained from the derivative of sigm(Idx), which
solves the non-differentiability of binarization. It is shown in Figure 3
(c), the redundant points are eliminated.
Meanwhile, we propose a point-to-surface measurement to re-
duce the proliferation of points located far from the face, as these
points contribute little yet significantly increase computational cost.
Specifically, the projected distance of each point to the initialization
surface is calculated, and the positions of discrete points and their
corresponding surface points are indexed by the the anchor points
and point clusters bound by the anchor points in:
𝑑=
∑︁
𝐴∈Idx
Idx ·
∑︁
P∈𝐴
||(P −P𝐴
0 ) · 𝑛𝐴
0 ||,
(7)
where 𝑛𝐴
0 is unit normal of anchor points P𝐴
0 , and P is the bound
points with anchor P𝐴
0 , then we find the anchor points through
Idx. The regularization on 𝑑will make the bound points around the
anchor point closer to the corresponding position on the surface,
and minimizing the offset.
4
EXPERIMENTS
4.1
Implementation Details
4.1.1
Datasets. We perform experiments with 6 subjects monocular
videos from the public datasets GaussianBlendshape [Ma et al. 2024],
NeRFBlendshape [Gao et al. 2022], StyleAvatar [Wang et al. 2023],
and InstantAvatar [Zielonka et al. 2023] in the offline training setup.
Moreover, we apply self-captured video via webcam within on-the-
fly training. The online/offline experiments are presented with the
resolution of 512 × 512. In offline setup, we take an average length
of 1000-5000 frames for training (about 80%) while the test dataset
includes frames with novel expressions and poses (about 20%), which
is aligned with baseline methods. The sequential/random inputs are
applied for online/offline set-up, respectively. For each frame, the
RobustVideoMatting [Lin et al. 2022] is used to remove background.
4.1.2
Hyperparameters. The 3D Gaussian points are initialized
with learning rates for Gaussian properties {𝛼,𝑠,𝑟,𝑜, 𝜇, Idx} set
to {1𝑒−5, 5𝑒−3, 1𝑒−3, 5𝑒−2, 2.5𝑒−3, 1𝑒−4}. During pre-training, the
learning rates for auxiliary properties {Tex, SH} are {2𝑒−4, 1𝑒−4},
while the others remain the same. The threshold of 𝛼𝑔𝑟𝑎𝑑and ∇P is
set to 0.01. The 𝛼𝑔𝑟𝑎𝑑is computed by the accumulated gradient of 𝛼
(the self.xyz_gradient_accum in 3DGS). For the online optimization,
we construct each mini-batch with multiple images, comprising one
newly captured frame and several previously captured frames, to
mitigate forgetting of historical information.
4.1.3
Losses. We apply 𝐿1 and SSIM in training. Meanwhile, the
distance between the points and surface in Eq. 7 as regularization.
For warm-up phrase, we introduce dark channel loss to separate
the learnable Tex and SH coefficients. During training, the weights
for the 𝐿1 and SSIM losses are set to 1 and 0.1, respectively, with
a regularization term weight of 0.01. Additionally, a dark channel
loss with the weight of 10 is incorporated for the warm-up phrase.
4.1.4
Pipeline Details. We set the head geometry corresponding to
the initial frame as canonical points clouds, and the position differ-
ence of the subsequent point clouds relative to the canonical point
cloud is used as the motion gradient. The motion gradients are also
cloned with the anchor points and deleted with non-contributing
points. We achieve that by introducing learnable parameters Idx.
The points with Idx value corresponds to 0 are deleted, and the
points with value as 1 are cloned. We allocate approximately 30
seconds for the warm-up phase within on-the-fly pipeline (a longer
warm-up duration is not recommended, as it may cause collapse).
We execute the cloning/pruning every 1500 optimization iterations.
4.2
Baseline
Given the challenges associated with high-efficient avatar recon-
struction, especially completing reconstruction within minutes while
achieving re-animation, few methods exactly handle this task. There-
fore, we select several related works for comparison, as,
• AvatarMAV [Xu et al. 2023]: This method applies a NeRF-
based implicit neural blend representation. By training a
lightweight-MLP, it integrates multiple learnable implicit
neural shapes for appearance. It claims to accomplish head
reconstruction in 5 minutes (256 × 256 resolution). Here, the
resolution is reset to 512 × 512 to align with baselines.
• FlashAvatar [Xiang et al. 2023]: The FlashAvatar builds on
3DGS, utilizing UV sampling for Gaussian initialization and
an offset network with MLPs to dynamically model variations
in facial expressions. It reports inference speed but omits the
time required for reconstruction.
• GaussianBlendshape [Ma et al. 2024]: It is the state-of-
the-art in monocular reconstruction based on 3DGS. This
5

<!-- page 6 -->
SIGGRAPH Conference Papers ’25, August 10–14, 2025, Vancouver, BC, Canada
Luchuan Song, Yang Zhou, Zhan Xu, Yi Zhou, Deepali Aneja, and Chenliang Xu
Ground-Truth StreamMe
FlashAvatar
GaussianBS
AvatarMAV
~3min
~40min
~25min
~15min
Fig. 4. The perceptual evaluation of our method and baselines for self-
reenactment. Baseline methods are evaluated with their default settings,
and time consumption is recorded under the official provided iteration.
Blue arrows on the ground-truth highlight regions of interest for closer
inspection. Please zoom in for detailed comparison. Natural face©Lizhen
Wang et al. (CC BY), and ©Wojciech Zielonka et al. (CC BY).
approach models facial motion through the mixture of ex-
plicitly learnable blendshapes aligned with the pre-tracked
FLAME expression coefficients.
The FlashAvatar and GaussianBlendshape are from the preprocessed
FLAME via MICA [MIC 2022], which is time-intensive and unsuit-
able for on-the-fly reconstitution. We also acknowledge other related
methods, such as NeRFBlendshape [Gao et al. 2022], HeadGaS [Dhamo
et al. 2024] and MonoGaussianAvatar [Chen et al. 2024] etc. How-
ever, we exclude these methods from comparison due to their on-
the-fly reconstruction setups and comparable performance (e.g.,
INSTA [Zielonka et al. 2023] and AvatarMAV employ fast training
by sampling rays from NeRF [Mildenhall et al. 2020], yet exhibit
slow inference when releasing all rays).
4.3
Numerical Results and Comparisons
We take two criteria for numerical evaluations, one is from quanti-
tative measurement, the other is from human assessment.
4.3.1
Quantitative Metrics. It is based on three aspects. (1) Image
Quality: We use the PSNR, LPIPS [Zhang et al. 2018] and MSE for the
evaluation of self-reenactment image quality. (2) Inference Frame
Rate: The inference frame rate (FPS) the measurement of the number
of frames generated within per second without introducing the head
tracking, which is not the speed of the pipeline. It is measured on a
single NVIDIA RTX4090 GPU. (3) Memory Storage: The memory
Table 1. (1) Left part: Quantitative evaluation results compared with baseline
methods. The best scores are highlighted in bold, with the second-best
underlined. Symbols ↓and ↑denote whether lower or higher values indicate
superior performance, respectively. All experiments were conducted on a
single NVIDIA RTX4090 machine. (2) Right part: A 5-point Likert scale is
used for the user study, where scores closer to 5 signify better performance.
Methods
PSNR↑
MSE↓
LPIPS↓
FPS↑
Mem.↓
MS↑
VQ↑
dB
→0
→0
MB
→5
Quantitative Results
User Study
AvatarMAV
24.1
0.047
0.137
2.58
14.1
3.1
2.3
FlashAvatar
27.8
0.021
0.109
94.5
12.6
3.8
3.1
GaussianBlendshape
26.4
0.017
0.112
22.9
872
3.7
3.6
StreamME
29.7
0.012
0.095
139
2.52
3.9
4.1
storage (Mem.) is the storage capacity occupied by the models, the
compact models offer advantages in both computational efficiency
and storage requirements. We use the Megabyte (MB) as units. The
quantitative experiments are performed on the self-reenactment.
4.3.2
User Study. We sample 6 distinct identities, each represented
by 20 video clips (5 for self-reenactment and 15 for cross-reenactment),
and invite 30 participants for the human evaluation. The Mean Opin-
ion Scores (MOS) rating protocol is employed, with participants
asked to assess the generated videos across four criteria: (1) MS
(Motion Synchronization): To what extent do you agree that the
head motion in the animated videos is synchronized with the driv-
ing source? and (2) VQ (Video Quality): To what extent do you agree
that the overall video quality is high, considering factors such as
frame quality, temporal consistency, and so forth? A 5-point Likert
scale is used for each criterion, with scores ranging from 1 to 5,
where 1 represents "strongly disagree" and 5 represents "strongly
agree" (higher scores indicate better performance).
4.4
Quality Comparison with Baseline Methods
The quality comparison results of self-reenactment are shown in
Table 1 and Figure 4 respectively. From Table 1, our method achieves
the best results in user study and quantitative evaluations. The
FlashAvatar is a powerful baseline, but still requires a lot of time to
train the learnable MLP layers. The AvatarMAV is fast but exhibits
limited detail preservation at 512 × 512 resolution.
The perceptual comparisons for cross-reenactment are illustrated
in Figure 5, where our method consistently delivers superior results.
The AvatarMAV and FlashAvatar exhibit noticeable artifacts on out-
of-distribution expressions, as their learnable MLP layers struggle
to adapt to expression changes within few iterations (for efficiency).
GaussianBlendshape, on the other hand, underperforms due to the
lack of conditions aligned with blendshapes during training. Our
approach circumvents the need for MLPs and blendshapes as train-
ing conditions by directly binding appearance representations to
the point cloud, enabling efficient training while preserving high
quality across diverse facial expressions.
4.5
Efficiency Comparison with Baseline Methods
In addition to quality comparisons, we further validate the efficiency
improvements achieved by our method. Specifically, we evaluate
the models at various training stages (iterations and time) on test-
set. The model capacity at each training slot will be examined for
comparison. The results are shown in Figure 6 and Table 2, it can
6

<!-- page 7 -->
StreamME: Simplify 3D Gaussian Avatar within Live Stream
SIGGRAPH Conference Papers ’25, August 10–14, 2025, Vancouver, BC, Canada
Ground-Truth
StreamMe
FlashAvatar
GaussianBS
AvatarMAV
Fig. 5. The perceptual evaluation of our method and baselines for cross-
reenactment. Our method demonstrates superior preservation of high-
frequency facial details, including features such as teeth and hair. Please
zoom in for a closer view. Natural face©Lizhen Wang et al. (CC BY), ©Xuan
Gao et al. (CC BY) and ©Wojciech Zielonka et al. (CC BY).
Table 2. The quantitative results of image quality and number of iterations
(Iters.) at each time point. The Iters. represents the training iteration speed
in the same unit time, a higher value indicates better training efficiency. Our
method achieves comparable quality in just two minutes, whereas other
methods require 30 minutes to reach the same standard.
Methods
PSNR
Iters.
PSNR
Iters.
PSNR
Iters.
PSNR
Iters.
Time = 1s
Time = 10s
Time = 2min
Time = 30in
AvatarMAV
3.12
7.6𝑒1
21.7
7.9𝑒2
24.1
9.6𝑒3
24.4
1.4𝑒5
FlashAvatar
4.90
9.7𝑒1
12.5
9.4𝑒2
16.9
1.2𝑒4
26.8
1.7𝑒5
GaussianBlendshape
7.29
2.9𝑒1
15.6
3.7𝑒2
25.2
4.8𝑒3
25.8
6.1𝑒4
StreamME
10.8
1.4𝑒2
23.1
1.5𝑒3
27.2
1.6𝑒4
29.8
3.4𝑒5
be found that compared with the baseline method, our approach
achieves convergence within 2 minutes, and simulates dynamic
expression from the outset. As shown in Figure 6, our method syn-
thesizes detailed tooth within just 10 seconds and simulates dynamic
expressions without any warm-up phase. The efficiency is due to it
Ground-Truth StreamMe
FlashAvatar GaussianBS
AvatarMAV
5s
10s
2min
30min
1s
Fig. 6. The visualization of efficiency comparison with baseline methods,
with results recorded from 1 second until the point at which all methods
achieve convergence (about 30 minutes). Please zoom in to compare the fine
details of teeth, eyes, hair, and the subtle differences in facial expressions.
It is noteworthy that our method is able to achieve the representation of
expressions and details with few iterations and time consumption. Natural
face ©Wojciech Zielonka et al. (CC BY).
is geometric foundation, avoiding reliance on learnable MLPs. Addi-
tionally, as shown in the Table 2, different from the nearly constant
training speed of baseline methods, our method improves training
efficiency over time due to the progressively sparse point clouds,
which reduce computational redundancy.
4.6
Ablation Study
In this section, we present ablation studies on Gaussian properties
warm-up, Gaussian motion anchors, and Gaussian simplification to
validate the importance of these modules.
4.6.1
Ablation study on Gaussian properties warm-up. We present
the improvement provided by the 3D Gaussian properties warm-up.
As illustrated in Figure 2, we label it as optional since it functions
equivalently to the point cloud deformation MLP. Specifically, it pre-
simulates the point cloud offset of the 3D head template, providing
prior positional information for the 3D Gaussians. To conduct the
ablation study, we introduce the position offset MLP (delta MLP)
7

<!-- page 8 -->
SIGGRAPH Conference Papers ’25, August 10–14, 2025, Vancouver, BC, Canada
Luchuan Song, Yang Zhou, Zhan Xu, Yi Zhou, Deepali Aneja, and Chenliang Xu
Ground-Truth
Corresponding Mesh
w/o deform MLP
w/ deform MLP
w/ Warm-Up
w/o Warm-Up
Fig. 7. The visualize of ablation study on Gaussian properties warm-up
phrase. Here, the explicit geometric deformation serves the same function
as the deform MLP during the 3DGS training phase. For results without the
deform MLP, those initialized with accurate geometric deformation from
the warm-up phase show improved quality over those initialized from the
template head, particularly in outer facial regions like the hair via red arrow,
as seen in the second column (left to right). However, introducing the MLP
in 3DGS training compensates effectively for this difference, as illustrated
in the third column (left to right), since both approaches focus on adjusting
point positions. Natural face©Luchuan Song et al. (CC BY).
Rotation = -90°
Rotation = -30°
Rotation = +30°
Rotation = +90°
Appearance
Fig. 8. The warm-up phase helps geometry adjustment for 3D Gaussian
initialization. We showcase its head geometry simulation from multiple
viewpoints, highlighting the consistent of hair geometry. The vertex offset
obtained could subsequently be applied to improve the head tracking. Nat-
ural face©Wojciech Zielonka et al. (CC BY).
with (w/) and without (w/o) warm-up intervention, as shown in
Figure 7. In the results without the delta MLP, warm-up contributes
to clearer details in out-of-face areas, such as hair. Incorporating the
MLP mitigates this issue, improving the detail clarity, while slightly
reduce execution efficiency.
At the same time, the properties warm-up will provide head ge-
ometry, which only takes about 10 seconds and several appearances
from different perspectives for fitting, as shown in Figure 8, which is
much faster than head fitting algorithms such as VHAP [Qian 2024]
and MonoNPHM [Giebenhain et al. 2024]. Although its geometric
Ground-Truth
w/o 3D Gaussian Anchor
w/ 3D Gaussian Anchor
1min
5min
10min
10s
Fig. 9. The visualize of image quality via 3D Gaussian anchor. We demon-
strate this through cross-reenactment perceptual evaluation. With the 3D
Gaussian anchor, the model instantly adapts to dynamic expressions. We
highlight the shape of the mouth for comparison. Natural face©Yufeng
Zheng et al. (CC BY) and ©Lizhen Wang et al. (CC BY).
Ground-Truth
w/o Idx
w/o Simplification
w/ Simplification
Fig. 10. The visualize of points cloud visualization via 3D Gaussian anchor
and simplification. By utilizing the gradient with the anchor points, the
point cloud is filtered to eliminate points with low contributions to motion.
This simplification further optimizes the number of point clouds. As shown
by the detailed representation of hair, the optimized point clouds preserve
high-frequency features. Natural face©Yufeng Zheng et al. (CC BY).
accuracy could be further enhanced, it strikes a balance between
geometry and 3D Gaussian feature representation.
8

<!-- page 9 -->
StreamME: Simplify 3D Gaussian Avatar within Live Stream
SIGGRAPH Conference Papers ’25, August 10–14, 2025, Vancouver, BC, Canada
0.0
0.2
0.4
0.6
0.8
1.0
0.024
0.037
0.041
0.015
w/o Simplification
Ground-Truth
1min
5min
10min
30s
0.035
0.045
w/ Simplification
0.022
0.017
Fig. 11. The visualize of the 3D Gaussian simplification ablation study. We
present the rendered appearance and MSE pixel error maps corresponding
to with (w/) and without(w/o) simplification at different time slots. The
mean error values are in the upper right corner of each error map. As the
first pruning does not occur at 0 sec, we begin recording from the 30th
second. The introduction of simplification will not result in any reduction in
qualitative and quantitative results, but lead to a significant improvement
in efficiency. Natural face©Luchuan Song et al. (CC BY).
The Ablation Study on 3D Gaussian Simplification
0
5e3
10e3
15e3
20e3
25e3
30e3
5
10
15
20
25
30
35
40
Iteration Steps
Training Speed [millisec.]
Pruning by Simplification 
FPS: 64
FPS: 161
N Points: 1.7e4
N Points: 2.1e5
N Points: 1.9e4
N Points: 1.6e5
[Converged]
1e3
Fig. 12. Comparisons in the ablation study on w/ and w/o 3D Gaussian
simplification. For w/o simplification, as the number of iterations grows,
the time per iteration also increases due to the overhead from redundant
points. In contrast, simplified pruning (w/ simplification) enables the system
to maintain high efficiency throughout. The FPS represents the training
iterations per second. The corresponding number of 3D Gaussian points
are annotated for reference. And the dark gray region indicates that w/
simplification has converged.
4.6.2
Ablation study on motion-aware anchor. The motion-aware
anchor selection leverages the motion gradient prior, progressively
duplicating points associated with motion. This approach enables
dynamic facial expressions to be directly mapped onto geometric
structures, bypassing the need for extensive iterative training steps
for adaptation. As shown in Figure 9, the motion-aware anchor
efficiently adapts to dynamic facial expressions at the beginning of
training, whereas without it, a warp-up period of approximately 10
minutes would be required. Additionally, it optimizes the number
of point clouds by incorporating motion gradients, and eliminates
the points unrelated to facial movement, as illustrated in Figure 10.
4.6.3
Ablation study on 3D Gaussian simplification. The simplifi-
cation of 3D Gaussian points is designed to reduce computational
load without sacrificing rendering quality, as shown in Figure 10
(w/ Simplification). Although denser point clouds are generally as-
sociated with higher detail, in our case, redundant points contribute
minimally to 3DGS. Figure 11 presents the comparison of results
with (w/) and without (w/o) simplification. As shown, point cloud
simplification does not result in a decrease in quality, as confirmed
by the error maps and mean error values. Specially, after more
than 10 minutes of training (about 1𝑒5 iterations), w/ simplification
achieves approximately 8 −10 times the reduction of points num-
ber compared to w/o simplification (the numbers are from 200𝑘to
24𝑘), which has a significant improvement in efficiency. Addition-
ally, the quantitative comparison results are shown in the Figure 12,
the method with simplification (w/ simplification) maintains high
efficiency throughout the training phase. Moreover, the average
number of point clouds for training subjects after convergence is
9,807, compared to 13,453 for FlashAvatar and 62,530 for Gaussian-
Blendshape, which demonstrates the cloning and pruning effectively
reduce the number of points.
5
APPLICATIONS
We extend our method to support a series of downstream applica-
tions in addition to cross-reenactment (facial relighting and tooni-
fication). We introduce these applications with streamME as the
foundation and provide baselines methods for comparison.
5.1
Facial Relighting Application
Since the baseline methods do not support facial relighting, we
exclude them from this discussion. We propose incorporating addi-
tional 3D Gaussian auxiliary properties during the warm-up phase,
specifically with the spherical harmonics (𝑆𝐻) containing the light-
ing features. Furthermore, the specular reflections are adaptively
decomposed during 3D Gaussian Splatting training. The accurate
geometry and lighting decomposition allow for relighting by adjust-
ing the pre-set light and light direction. Different from the previous
methods [Cai et al. 2024; Hou et al. 2024; Lin et al. 2024; Qiu et al.
2024], our relighting training can be completed in just a few minutes,
without the need for long-term training. However, constrained by
the limited illumination information available in monocular video,
our method cannot match the performance of approaches utilizing
large datasets. Nonetheless, this represents an exploration of down-
stream applications and marks the first introduction of the concept
of relighting in the monocular 3D Gaussian Splatting (3DGS) head
avatar as far as we know. As shown in Figure 13, we present the relit
appearance and geometry, the rendered face and diffuse shading
are reflected in corresponding color of background images. The
spherical harmonics are estimated from the provided background
9

<!-- page 10 -->
SIGGRAPH Conference Papers ’25, August 10–14, 2025, Vancouver, BC, Canada
Luchuan Song, Yang Zhou, Zhan Xu, Yi Zhou, Deepali Aneja, and Chenliang Xu
Relight Portrait
Source Env Light
Env1 Light
Env2 Light
Env3 Light
Env1
Env2
Env3
Diffuse Shading
Env4 Light
Env4
Fig. 13. The visualize of relightable head reconstruction. From left to right,
we set the source light source and the ambient light from different back-
grounds to relight the face. Please pay attention to the colors reflected on
the facial and geometry surface, which from the SH parameters re-rendering
via the 3D Gaussian field. The artifacts around geometric and face due to the
lack of 3D representation from single view setting. Natural face©Wojciech
Zielonka et al. (CC BY), and background images©Adobe (CC BY).
image, and the global illumination is derived by averaging the es-
timated results. We acknowledge that compare to some advanced
relighting methods [He et al. 2024; Li et al. 2024; Yoon et al. 2024],
our approach still has limitations in this application. However, it
should be evaluated in the efficiency and monocular input setting.
5.2
Facial Toonification Application
We follow TextToon [Song et al. 2024a] and PortraitGen [Gao et al.
2024] to implement toonification (or stylization) application. These
methods are based on stable diffusion [Rombach et al. 2022] for
adaptive editing of the rendered images. It is worth noting that
baseline methods are also capable of achieving adaptive editing.
To provide a more comprehensive evaluation, we include editing
with these methods for comparison. Specifically, we provide the
Text2Image [Brooks et al. 2023] module with same setting (denoise
steps, editing strength, guidance scale and boundary ratio e.t.c.)
for baseline methods. The comparison results are shown in the
Figure 14. As highlighted by the blue arrows, our method achieves
richer details in texture editing compared to the baseline methods.
6
LIMITATIONS
The primary limitation of our method lies in its reliance on the
single-view setting and the absence of a complete 3D facial struc-
ture, which hinders accurate reconstruction or learning of the full
facial shape within the 3D Gaussian explicit space. This limitation
often leads to rendering artifacts during head motion under extreme
postures, as shown in first row in Figure 15. Second, the diversity of
rendered appearances and facial expressions depends heavily on the
training data, as in second row in Figure 15. Attempting to recon-
struct appearances outside the distribution of the training data often
yields disappointing results. It is worth emphasizing that this is not
a limitation of our method specifically, but rather an inherent limi-
tation of the single view setting, almost all related methods [Chen
Joker as in DC
Pixar Style
Ground-Truth
StreamMe
FlashAvatar
GaussianBS
AvatarMAV
Fig. 14. The visualize of toonification head reconstruction. We apply the
TextToon [Song et al. 2024a] with two different prompts as "Joker in DC" and
"Pixar Style" for the reconstructed appearance. We perform the perceptual
ccomparison of toonification capabilities against the baseline methods. Blue
arrows are used to highlight regions for attention. It can be found that our
method achieves better toonification ability on details than the baselines
with different styles. Natural face©Wojciech Zielonka et al. (CC BY).
Extreme Pose
Expression
Ground-Truth
In-Distribution Samples
Out-of-Distribution Samples
Fig. 15. The visualize of limitation of monocular head reconstruction via
our method. We present the results from cross-reenactment, as well as
reconstruction outcomes for extreme poses and facial expressions. The in-
distribution indicates that the training set includes similar poses and facial
expressions, while out-of-distribution denotes the opposite. We use the blue
arrows to highlight the artifacts. Natural face and ©Xuan Gao et al. (CC BY).
et al. 2024; Kim et al. 2018; Song et al. 2024a; Wang et al. 2023; Xu
et al. 2023; Zheng et al. 2023] declare that.
7
DISCUSSION AND CONCLUSION
We present StreamME, a on-the-fly head avatar training (or re-
construction) method from monocular live stream. It analyzes the
number of point clouds in the 3D Gaussian field, evaluates the points
distribution to achieve a more optimal arrangement over explicit
face geometry, and removes points with lower contributions on scale
and opacity to reduce computational complexity. The designs help
our method achieves a diversity of facial expressions solely through
geometry without dependence on the learnable MLPs, which signifi-
cantly improves training speed. Furthermore, a series of applications
(e.g. toonification and relighting) have been developed based on our
method, which bringing more exploration directions in the future.
10

<!-- page 11 -->
StreamME: Simplify 3D Gaussian Avatar within Live Stream
SIGGRAPH Conference Papers ’25, August 10–14, 2025, Vancouver, BC, Canada
REFERENCES
2022. Towards Metrical Reconstruction of Human Faces.
Jeongmin Bae, Seoha Kim, Youngsik Yun, Hahyun Lee, Gun Bang, and Youngjung Uh.
2024. Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian
Splatting. arXiv preprint arXiv:2404.03613 (2024).
Yoshua Bengio, Nicholas Léonard, and Aaron Courville. 2013. Estimating or propagating
gradients through stochastic neurons for conditional computation. arXiv preprint
arXiv:1308.3432 (2013).
Tim Brooks, Aleksander Holynski, and Alexei A Efros. 2023. Instructpix2pix: Learning
to follow image editing instructions. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition. 18392–18402.
Ziqi Cai, Kaiwen Jiang, Shu-Yu Chen, Yu-Kun Lai, Hongbo Fu, Boxin Shi, and Lin Gao.
2024. Real-time 3D-aware portrait video relighting. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. 6221–6231.
Chen Cao, Tomas Simon, Jin Kyu Kim, Gabe Schwartz, Michael Zollhoefer, Shun-
Suke Saito, Stephen Lombardi, Shih-En Wei, Danielle Belko, Shoou-I Yu, et al. 2022.
Authentic volumetric avatars from a phone scan. ACM Transactions on Graphics
(TOG) 41, 4 (2022), 1–19.
Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello,
Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. 2022.
Efficient geometry-aware 3d generative adversarial networks. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition. 16123–16133.
Yufan Chen, Lizhen Wang, Qijing Li, Hongjiang Xiao, Shengping Zhang, Hongxun Yao,
and Yebin Liu. 2024. Monogaussianavatar: Monocular gaussian point-based head
avatar. In ACM SIGGRAPH 2024 Conference Papers. 1–9.
Helisa Dhamo, Yinyu Nie, Arthur Moreau, Jifei Song, Richard Shaw, Yiren Zhou, and
Eduardo Pérez-Pellitero. 2024. Headgas: Real-time animatable head avatars via
3d gaussian splatting. In European Conference on Computer Vision. Springer, 459–
476.
Guy Gafni, Justus Thies, Michael Zollhöfer, and Matthias Nießner. 2021. Dynamic Neural
Radiance Fields for Monocular 4D Facial Avatar Reconstruction. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
8649–8658.
Xuan Gao, Haiyao Xiao, Chenglai Zhong, Shimin Hu, Yudong Guo, and Juyong Zhang.
2024. Portrait Video Editing Empowered by Multimodal Generative Priors. arXiv
preprint arXiv:2409.13591 (2024).
Xuan Gao, Chenglai Zhong, Jun Xiang, Yang Hong, Yudong Guo, and Juyong Zhang.
2022. Reconstructing Personalized Semantic Facial NeRF Models From Monocular
Video.
ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia) 41, 6
(2022). https://doi.org/10.1145/3550454.3555501
Simon Giebenhain, Tobias Kirschstein, Markos Georgopoulos, Martin Rünz, Lourdes
Agapito, and Matthias Nießner. 2024. MonoNPHM: Dynamic Head Reconstruc-
tion from Monocular Videos. In Proc. IEEE Conf. on Computer Vision and Pattern
Recognition (CVPR).
Mingming He, Pascal Clausen, Ahmet Levent Taşel, Li Ma, Oliver Pilarski, Wenqi
Xian, Laszlo Rikker, Xueming Yu, Ryan Burgert, Ning Yu, et al. 2024. DifFRelight:
Diffusion-Based Facial Performance Relighting. arXiv preprint arXiv:2410.08188
(2024).
Andrew Hou, Zhixin Shu, Xuaner Zhang, He Zhang, Yannick Hold-Geoffroy, Jae Shin
Yoon, and Xiaoming Liu. 2024. COMPOSE: Comprehensive Portrait Shadow Editing.
arXiv preprint arXiv:2408.13922 (2024).
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023.
3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions
on Graphics 42, 4 (2023).
Hyeongwoo Kim, Pablo Garrido, Ayush Tewari, Weipeng Xu, Justus Thies, Matthias
Niessner, Patrick Pérez, Christian Richardt, Michael Zollhöfer, and Christian
Theobalt. 2018. Deep video portraits. ACM transactions on graphics (TOG) 37,
4 (2018), 1–14.
Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. 2024. Com-
pact 3d gaussian representation for radiance field. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. 21719–21728.
Junxuan Li, Chen Cao, Gabriel Schwartz, Rawal Khirodkar, Christian Richardt, Tomas
Simon, Yaser Sheikh, and Shunsuke Saito. 2024. URAvatar: Universal Relightable
Gaussian Codec Avatars. arXiv preprint arXiv:2410.24223 (2024).
Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. 2023. Spacetime Gaussian Feature Splatting
for Real-Time Dynamic View Synthesis. arXiv preprint arXiv:2312.16812 (2023).
Min-Hui Lin, Mahesh Reddy, Guillaume Berger, Michel Sarkis, Fatih Porikli, and Ning
Bi. 2024. EdgeRelight360: Text-Conditioned 360-Degree HDR Image Generation for
Real-Time On-Device Video Portrait Relighting. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. 831–840.
Shanchuan Lin, Linjie Yang, Imran Saleemi, and Soumyadip Sengupta. 2022. Robust high-
resolution video matting with temporal guidance. In Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision. 238–247.
Feng-Lin Liu, Shu-Yu Chen, Yu-Kun Lai, Chunpeng Li, Yue-Ren Jiang, Hongbo Fu, and
Lin Gao. 2022. Deepfacevideoediting: Sketch-based deep editing of face videos.
ACM Transactions on Graphics (TOG) 41, 4 (2022), 1–16.
Yichen Liu, Benran Hu, Junkai Huang, Yu-Wing Tai, and Chi-Keung Tang. 2023. Instance
neural radiance field. In Proceedings of the IEEE/CVF International Conference on
Computer Vision. 787–796.
Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. 2024. Dynamic
3D Gaussians: Tracking by Persistent Dynamic View Synthesis. In 3DV.
Shengjie Ma, Yanlin Weng, Tianjia Shao, and Kun Zhou. 2024. 3D Gaussian Blendshapes
for Head Avatar Animation. arXiv preprint arXiv:2404.19398 (2024).
Marko Mihajlovic, Aayush Bansal, Michael Zollhoefer, Siyu Tang, and Shunsuke Saito.
2022. KeypointNeRF: Generalizing image-based volumetric avatars using relative
spatial encoding of keypoints. In European conference on computer vision. Springer,
179–197.
Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ra-
mamoorthi, and Ren Ng. 2020. NeRF: Representing Scenes as Neural Radiance Fields
for View Synthesis. In Computer Vision–ECCV 2020: 16th European Conference,
Glasgow, UK, August 23–28, 2020, Proceedings, Part I. 405–421.
Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. 2022. Instant
neural graphics primitives with a multiresolution hash encoding. arXiv preprint
arXiv:2201.05989 (2022).
Shenhan Qian. 2024. Versatile Head Alignment with Adaptive Appearance Priors.
(September 2024). https://github.com/ShenhanQian/VHAP
Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide Davoli, Simon Giebenhain,
and Matthias Nießner. 2023. GaussianAvatars: Photorealistic Head Avatars with
Rigged 3D Gaussians. arXiv preprint arXiv:2312.02069 (2023).
Zhiyin Qian, Shaofei Wang, Marko Mihajlovic, Andreas Geiger, and Siyu Tang.
2024.
3dgs-avatar: Animatable avatars via deformable 3d gaussian splatting.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. 5020–5030.
Haonan Qiu, Zhaoxi Chen, Yuming Jiang, Hang Zhou, Xiangyu Fan, Lei Yang, Wayne
Wu, and Ziwei Liu. 2024. Relitalk: Relightable talking portrait generation from a
single video. International Journal of Computer Vision (2024), 1–16.
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.
2022. High-resolution image synthesis with latent diffusion models. In Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition. 10684–
10695.
Zhijing Shao, Zhaolong Wang, Zhuang Li, Duotun Wang, Xiangru Lin, Yu Zhang,
Mingming Fan, and Zeyu Wang. 2024. Splattingavatar: Realistic real-time human
avatars with mesh-embedded gaussian splatting. arXiv preprint arXiv:2403.05087
(2024).
Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. 2021. Deep
marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis.
Advances in Neural Information Processing Systems 34 (2021), 6087–6101.
Luchuan Song, Lele Chen, Celong Liu, Pinxin Liu, and Chenliang Xu. 2024a. Text-
Toon: Real-Time Text Toonify Head Avatar from Single Video. arXiv preprint
arXiv:2410.07160 (2024).
Luchuan Song, Bin Liu, Guojun Yin, Xiaoyi Dong, Yufei Zhang, and Jia-Xuan Bai. 2021a.
Tacr-net: editing on deep video and voice portraits. In Proceedings of the 29th ACM
International Conference on Multimedia. 478–486.
Luchuan Song, Pinxin Liu, Lele Chen, Guojun Yin, and Chenliang Xu. 2024b. Tri 2-plane:
Thinking Head Avatar via Feature Pyramid. In European Conference on Computer
Vision. Springer, 1–20.
Luchuan Song, Guojun Yin, Zhenchao Jin, Xiaoyi Dong, and Chenliang Xu. 2023. Emo-
tional listener portrait: Neural listener head generation with emotion. In Proceedings
of the IEEE/CVF International Conference on Computer Vision. 20839–20849.
Luchuan Song, Guojun Yin, Bin Liu, Yuhui Zhang, and Nenghai Yu. 2021b. Fsft-net:
face transfer video generation with few-shot views. In 2021 IEEE international
conference on image processing (ICIP). IEEE, 3582–3586.
Justus Thies, Michael Zollhofer, Marc Stamminger, Christian Theobalt, and Matthias
Nießner. 2016. Face2face: Real-time face capture and reenactment of rgb videos.
In Proceedings of the IEEE conference on computer vision and pattern recognition.
2387–2395.
Hanzhang Tu, Ruizhi Shao, Xue Dong, Shunyuan Zheng, Hao Zhang, Lili Chen, Meili
Wang, Wenyu Li, Siyan Ma, Shengping Zhang, et al. 2024. Tele-Aloha: A Low-
budget and High-authenticity Telepresence System Using Sparse RGB Cameras.
arXiv preprint arXiv:2405.14866 (2024).
Aaron Van Den Oord, Oriol Vinyals, et al. 2017. Neural discrete representation learning.
Advances in neural information processing systems 30 (2017).
Lizhen Wang, Xiaochen Zhao, Jingxiang Sun, Yuxiang Zhang, Hongwen Zhang, Tao
Yu, and Yebin Liu. 2023. StyleAvatar: Real-time Photo-realistic Portrait Avatar from
a Single Video. arXiv preprint arXiv:2305.00942 (2023).
Ziyan Wang, Giljoo Nam, Tuur Stuyck, Stephen Lombardi, Michael Zollhöfer, Jessica
Hodgins, and Christoph Lassner. 2022. Hvh: Learning a hybrid neural volumet-
ric representation for dynamic hair performance capture. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6143–6154.
Jun Xiang, Xuan Gao, Yudong Guo, and Juyong Zhang. 2023. FlashAvatar: High-Fidelity
Digital Avatar Rendering at 300FPS. arXiv preprint arXiv:2312.02214 (2023).
11

<!-- page 12 -->
SIGGRAPH Conference Papers ’25, August 10–14, 2025, Vancouver, BC, Canada
Luchuan Song, Yang Zhou, Zhan Xu, Yi Zhou, Deepali Aneja, and Chenliang Xu
Yuelang Xu, Benwang Chen, Zhe Li, Hongwen Zhang, Lizhen Wang, Zerong Zheng, and
Yebin Liu. 2024. Gaussian Head Avatar: Ultra High-fidelity Head Avatar via Dynamic
Gaussians. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR).
Yuelang Xu, Lizhen Wang, Xiaochen Zhao, Hongwen Zhang, and Yebin Liu. 2023.
AvatarMAV: Fast 3D Head Avatar Reconstruction Using Motion-Aware Neural
Voxels. In ACM SIGGRAPH 2023 Conference Proceedings.
Jae Shin Yoon, Zhixin Shu, Mengwei Ren, Xuaner Zhang, Yannick Hold-Geoffroy,
Krishna Kumar Singh, and He Zhang. 2024. Generative Portrait Shadow Removal.
arXiv preprint arXiv:2410.05525 (2024).
Junzhe Zhang, Yushi Lan, Shuai Yang, Fangzhou Hong, Quan Wang, Chai Kiat Yeo,
Ziwei Liu, and Chen Change Loy. 2023. Deformtoon3d: Deformable neural radiance
fields for 3d toonification. In Proceedings of the IEEE/CVF International Conference
on Computer Vision. 9144–9154.
Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018.
The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In CVPR.
Zeliang Zhang, Mingqian Feng, Zhiheng Li, and Chenliang Xu. 2024a. Discover and mit-
igate multiple biased subgroups in image classifiers. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. 10906–10915.
Zeliang Zhang, Wei Yao, and Xiaosen Wang. 2024b. Bag of tricks to boost adversarial
transferability. arXiv preprint arXiv:2401.08734 (2024).
Yufeng Zheng, Victoria Fernández Abrevaya, Marcel C Bühler, Xu Chen, Michael J
Black, and Otmar Hilliges. 2022.
Im avatar: Implicit morphable head avatars
from videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. 13545–13555.
Yufeng Zheng, Wang Yifan, Gordon Wetzstein, Michael J Black, and Otmar Hilliges.
2023. Pointavatar: Deformable point-based head avatars from videos. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21057–
21067.
Ruijie Zhu, Yanzhe Liang, Hanzhi Chang, Jiacheng Deng, Jiahao Lu, Wenfei Yang,
Tianzhu Zhang, and Yongdong Zhang. 2024. MotionGS: Exploring Explicit Motion
Guidance for Deformable 3D Gaussian Splatting. arXiv preprint arXiv:2410.07707
(2024).
Wojciech Zielonka, Timo Bolkart, Thabo Beeler, and Justus Thies. 2024. Gaussian Eigen
Models for Human Heads. arXiv preprint arXiv:2407.04545 (2024).
Wojciech Zielonka, Timo Bolkart, and Justus Thies. 2022. Towards metrical recon-
struction of human faces. In European Conference on Computer Vision. Springer,
250–269.
Wojciech Zielonka, Timo Bolkart, and Justus Thies. 2023. Instant volumetric head
avatars. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. 4574–4584.
12
