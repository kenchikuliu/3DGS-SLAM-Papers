<!-- page 1 -->
NGM-SLAM: Gaussian Splatting SLAM with
Radiance Field Submap
Jingwei Huang1, Mingrui Li2 #, Lei Sun Aaron3, Xuxiang Tian 4, Tianchen Deng 5, Hongyu Wang1 ∗
1Dalian University of Technology
2University of Electronic Science and Technology of China
3University of Pennsylvania, University of Pennsylvania
4Carnegie Mellon University
5Shanghai Jiaotong University
2905450254@mail.dlut.edu.cn
Abstract
Gaussian Splatting has garnered widespread attention due to its exceptional perfor-
mance. Consequently, SLAM systems based on Gaussian Splatting have emerged,
leveraging its capabilities for rapid real-time rendering and high-fidelity mapping.
However, current Gaussian Splatting SLAM systems usually struggle with large
scene representation and lack effective loop closure adjustments and scene general-
ization capabilities. To address these issues, we introduce NGM-SLAM, the first
GS-SLAM system that utilizes neural radiance field submaps for progressive scene
expression, effectively integrating the strengths of neural radiance fields and 3D
Gaussian Splatting. We have developed neural implicit submaps as supervision
and achieve high-quality scene expression and online loop closure adjustments
through Gaussian rendering of fused submaps. Our results on multiple real-world
scenes and large-scale scene datasets demonstrate that our method can achieve
accurate gap filling and high-quality scene expression, supporting both monocular,
stereo, and RGB-D inputs, and achieving state-of-the-art scene reconstruction and
tracking performance.
1
Introduction
SLAM systems[1, 15–17, 36, 40] have long been a fundamental concern in the domains of robotics
and AR/VR. Dense SLAM[6, 12, 18–20, 22], in particular, commands a broader range of applications
and demand compared to its sparse counterparts. Traditional dense SLAM systems[32, 33] utilize
explicit representations like voxels, point clouds, and TSDF[21], and have achieved commendable
results in tracking. However, the limitations of traditional dense SLAM[37, 39] in high-fidelity
modeling, gap filling, and texture details restrict their wider application.
Neural implicit SLAM, leveraging NERF-based[23–25, 27] implicit representations, has crafted
a complete pipeline that includes tracking and mapping, significantly enhancing the perceptual
capabilities of SLAM systems[33]. Despite these advancements, current neural implicit SLAM
systems face limitations in operational speed, real-time capabilities, and memory requirements.
Recent developments in SLAM methods based on 3D Gaussian Splatting (3DGS)[28] not only retain
the high-fidelity mapping benefits of NERF-SLAM systems but also demonstrate advantages in
rendering speed and precision, challenging the dominant position of neural implicit SLAM in dense
reconstruction. 3DGS employs continuous Gaussian volumes characterized by color opacity and
∗Use footnote for providing further information about author (webpage, alternative address)—not for
acknowledging funding agencies.
arXiv:2405.05702v8  [cs.RO]  24 Apr 2025

<!-- page 2 -->
orientation for scene representation, combining the intuitive flexibility of explicit representations
with the continuous differentiability of implicit expressions.
However, it’s notable that 3DGS, compared to NERF[48–51], shows limitations in gap-filling capabil-
ities mainly due to its reliance on point cloud models as the primary input and lacks the generalizing
inference capabilities from neural networks. Additionally, current 3DGS-SLAM systems lack robust
loop detection, which is crucial for correcting global drift errors and preventing map loss. This
disadvantage becomes particularly apparent in large scenes due to cumulative drift.
To address these challenges, we propose a SLAM system based on neural submaps and 3DGS
representation. We first establish submaps represented by neural radiance fields and construct a global
keyframe list. When triggering map thresholds, we create new neural submaps and use previous
submaps as priors to guide Gaussian rendering. Subsequently, we perform local bundle adjustment
(BA). Between submaps, we implement a submap fusion strategy and trim the generated submaps’
Gaussians. Upon loop closure detection, we perform real-time coarse-to-fine global loop adjustments,
adjusting the map poses corresponding to anchor frames and applying global BA and global BA
Gaussian rendering loss to correct poses and associated local map keyframes. We significantly
mitigate errors caused by accumulated drift and map drift, while correcting global map loops with
minimal computational overhead. Experimental results demonstrate that our method achieves state-
of-the-art performance in tracking and mapping and is scalable to large-scale scenes. In summary,
our contributions are as follows:
• We introduce the first progressive dense Gaussian splatting SLAM system based on neural
submaps, achieving high-fidelity mapping through a local-to-global reconstruction strategy.
We enable large-scale scene inference and effectively leverage the advantages of both
representation methods.
• We propose a global loop correction strategy, including coarse-to-fine submap correction
and global bundle adjustment loss, enabling real-time adjustment of submaps and map
correction.
• We propose effective Gaussian pruning and multiscale Gaussian rendering strategies, ensur-
ing the system can remove redundant Gaussians while enhancing anti-aliasing capabilities,
improving rendering speed, and accuracy.
• Our system supports monocular, stereo and RGB-D inputs, demonstrating competitive
tracking and mapping performance on 5 datasets, and supports real-time inference at 5 FPS
in large-scale scenes.
2
Related Work
Dense Visual SLAM Dense real-time scene mapping is considered a key method for solving problems
related to scene understanding, autonomous driving, and AR/VR applications[35]. Early methods like
KinectFusion utilized an explicit scene representation approach to achieve comprehensive tracking
and mapping, offering more accurate scene expression and geometric shapes compared to sparse point-
based methods. Traditional dense SLAM systems extensively employ various explicit representations
such as voxels and point clouds. The advantages of traditional dense SLAM methods lie in their
accurate and mature tracking systems, but they have limitations in providing high-fidelity models and
lack generalized reasoning capabilities.
Recent years have seen significant attention given to methods based on neural radiance fields[2–
5]. iMAP[47] established the first complete neural implicit SLAM framework, achieving scalable
and efficient scene representation. However, limitations due to a single MLP architecture can lead
to tracking loss and mapping errors in larger-scale scenes. NICE-SLAM[9] utilizes a pretrained
multi-MLP system with frozen parameters to achieve accurate tracking and scene expression, yet it
faces drift issues in filling gaps. ESLAM[19] uses tri-plane features for scene representation, while
Go-SLAM[53] supports multimodal inputs from monocular, stereo, and RGB-D cameras. However,
these neural implicit methods[7] generally lack online loop closure correction, often resulting in the
loss of high-frequency details due to local over-smoothing.
3D Gaussian Based SLAM Recently, 3D Gaussian based scene representation methods have gar-
nered broad interest. Compared to NERF-based neural implicit methods[43–45], 3D Gaussian
methods[42] combine the advantages of explicit and implicit expressions. They capture high-fidelity
2

<!-- page 3 -->
three-dimensional scenes through a differentiable rasterization process, avoiding the per-pixel ray
casting required by neural fields, thus achieving high-speed rendering. Photo-SLAM[59] achieves
high-quality real-time representation using a Gaussian pyramid-based approach. SplaTAM[54]
employs anisotropic Gaussian representation, achieving real-time tracking and rendering. MonoGS-
Matsuki:Murai:etal:CVPR2024 achieves faster scene representation through Gaussian shape regular-
ization and geometric verification, but lacks loop closure correction to eliminate accumulated errors.
However, these methods face many challenges. Because 3D Gaussian (3DGS) representations cannot
learn scene features for inference, they cannot fill gaps like NERF-based methods. Furthermore,
current 3DGS SLAM methods still lack stable and complete tracking systems, such as loop detection
and online bundle adjustment (BA). We propose an incremental scene representation method that
combines the strengths of NeRF and 3DGS for complementary scene expression—a method that
can learn features while retaining complete high-frequency details. By establishing a neural implicit
submap based on NERF as a prior to guide Gaussian rendering, our method can effectively fill gaps
and achieve more comprehensive scene reconstruction. The use of multi-scale Gaussian rendering
ensures optimized rendering speed and enhances local details. The online adjustment of our neural
submap allows global BA without the need for re-rendering. Although our method adopts a hybrid
rendering approach, our parallel submap tracking and optimization avoid the costs associated with
global scene updates, thereby improving overall system speed.
3
Method
We present the system pipeline of NGM-SLAM in Figure 1. Our system first tracks and constructs
submaps based on RGB/RGB-D image streams using neural radiation fields (Sec 3.1). Then, the
neural submaps are utilized to construct Gaussian priors supervised submaps (Section 3.2). Following
alignment of submap poses, we execute a submap fusion strategy (Sec 3.3). Finally, we design a
local-to-global loop closure detection and bundle optimization process (Sec 3.4). We will elaborate
on the entire process of our system in the methodology.
Figure 1: The system includes two modules: tracking and mapping. After the initial submap starts to
be established, the tracking module continuously estimates the camera pose and detects loops, while
passing keyframes of the submap to the mapping module. The mapping module first constructs a
neural submap that also serves as a prior for the multi-scale GS(Gaussian Splatting) submap, and
performs parallel rendering between submaps. Local Bundle Adjustment (BA) is conducted within
submaps to correct pose and mapping errors, and Global BA is executed on all anchor frames when a
loop closure is detected. Finally, the resulting GS maps are stitched together.
3

<!-- page 4 -->
3.1 Neural Submap Construction
Neural Submap Construction In current 3D Gaussian-based SLAM systems, due to the lack of
generalized inferencing capabilities of neural networks, we have introduced data-driven, incremental
neural submaps as a foundational supervisory mechanism to fill voids and enhance map representation.
Our approach is based on ORB feature point tracking[29], as feature point extraction is performed
only on the current input frame, which avoids the frequent drift issues associated with submap
creation and hole filling.
Initially, we establish a local submap list of keyframes and a global list of keyframes. This facilitates
the effective implementation of local/global Bundle Adjustment (BA) processes. When tracking
begins, the first frame of the local submap is simultaneously added to the local keyframes and global
keyframes list. We set a keyframe threshold for submaps; when a submap accumulates enough
keyframes to reach this threshold, we establish a new local submap.
To better achieve map fusion and avoid excessive fusion errors between submaps, we utilize the
DBOW model to assess the co-visibility relationships between keyframes. If the map connection
frames lack co-visibility, the current submap’s connecting frame is added as the first frame to a newly
created submap to ensure stable map integration.
Neural Implicit Rendering Due to the significantly higher speed of Gaussian rendering compared to
neural implicit rendering, we adopt a progressive rendering strategy to achieve real-time rendering
between submaps and the global map. We initially perform neural implicit rendering using only a
sparse set of keyframes for each submap to obtain foundational supervision. Once the submap is
preliminarily adjusted, full-frame poses are utilized for Gaussian rendering to refine the mapping
process. To reduce the rendering cost, we employ a multi-resolution hash-encoded radiance field as a
prior.
The radiance field f is a continuous function that maps a three-dimensional point position p ∈R3
and a viewing direction d ∈S2 to a volume density σ ∈R+ and a NeRF RGB color value C ∈R3.
Inspired by NeRF volume rendering, the final color prediction for a pixel is approximated through
ray marching and integration using sample points:
CN =
N
X
j=1
TNjαjCj,
where
TNj =
j−1
Y
k=1
(1 −αk),
αj = 1 −e−σjδj
(1)
where TNj is the NeRF transmittance, αj is the alpha value for xj, and δj is the distance between
adjacent sample points. In neural radiance fields, f is parameterized as an MLP with ReLU activation
fθ, and the network parameters θ are optimized via gradient descent on the reconstruction loss:
L(θ) =
X
r∈Rb
Cθ
N(r) −CGT(r)
2
(2)
where r ∈Rb is a batch of rays sampled from the set of all rays.
3.2 Gaussian Submap
Multi-scale Gaussian Rendering Utilizing neural submap priors, we represent the scene using a set
of anisotropic 3D Gaussian distributions.The scene is depicted as points associated with a position
p ∈R3, opacity o ∈[0, 1], third-order spherical harmonics (SH) coefficients k ∈R16, a 3D scale
s ∈R3, and a 3D rotation R ∈SO(3) represented by a 4D quaternion q ∈R3. c is the color value
when rendering in 3D Gaussian. Inspired by equation (1), this representation can be rendered to a
camera’s image plane, with a correctly ordered list of points:
cGS =
Np
X
j=1
cjαjTGi
where
TGi =
j−1
Y
i=1
(1 −αi)
(3)
Where TGj is the 3D Gaussian transmittance. Each Gaussian Gi contains optical properties: color
ci and opacity αi. For a continuous 3D representation, the mean xi and covariance Ci defined in
world coordinates describe the Gaussian’s location and its ellipsoidal shape. The 3D Gaussians
N(xW , ZW ) in world coordinates are associated with 2D Gaussians N(xI, ZI) on the image plane
through the projection transformation:
xI = π (TCW · xW ) ,
ZI = JWZW WT JT
(4)
4

<!-- page 5 -->
W is the viewing transformation, J denotes the Jacobian of the affine approximation of the projective
transformation [57], and ZW denotes the 3D covariance matrix. The radiance fields provided by
neural submaps serve as the foundational supervision for rendering, but they are prone to aliasing
effects that degrade the rendering quality during the sampling process. This is particularly evident
when constructing submaps with many small Gaussians, leading to severe artifacts. We employ a
multi-scale Gaussian rendering approach inspired by [52], aggregating smaller Gaussians into larger
ones to improve the rendering quality. We represent from Gaussian functions at four detail levels
corresponding to 1×, 4×, 16×, and 64× down-sampling resolutions. During the training process,
smaller fine-level Gaussian functions are aggregated to create coarser-level larger Gaussians. The
selection of Gaussian bodies is based on pixel coverage, including or excluding according to the
coverage range defined by the inverse of the highest frequency component in that region fmax =
1
Sk .
Specifically, Gaussians at the edges of submaps are not aggregated to facilitate submap fusion
operations and submap alignment.
Ray-Guided Gaussian Pruning To reduce the number of ineffective Gaussian volumes during
Gaussian rendering, thereby improving rendering speed, we adopted a pruning method in the Gaussian
rendering process guided by ray sampling. We employed an importance assessment strategy to
remove invalid points from all the Gaussians. The importance score is defined by the contribution
of the aggregated Gaussian particles to the rays in all input images. To improve filtering efficiency,
we introduced a sparse point cloud composed of ORB map points as guidance, which typically
corresponds to regions near textured surfaces. Inspired by [56], we used a point cloud counter to
gather statistics and counted rays with a number of nearby sparse points exceeding the threshold t1 as
the ray set Kr. The importance score can be expressed as:
E(pi) =
max
If ∈If ,r∈If ,r∈Kr(αr
i T r
i )
(5)
where If is the rendered image and If is the target image. αr
i τ r
i represents the contribution of
Gaussian i to the final color prediction of a pixel, as described in equation (4). The mask values are
computed as follows:
mi = m(pi) = 1(E(pi) < tprune)
(6)
where tprune ∈[0, 1] is a threshold used to control the number of points representing the scene. All
Gaussian distributions with a mask value of 1 are removed from the scene. Finally, we execute default
rasterization, ensuring that our rendering speed does not decrease and improving accuracy through
pruning.
3.3 Submap Fusion
We represent the scene as the sum of multiple local scenes:
{Ii, Di}M
i=1 7→

SF1
σ1, SF2
σ2, . . . , SFn
σn
	
(7)
The series {Ii, Di}M
i=1 represents a sequence of RGB-D inputs, where SF n
σn denotes the submap
representation. When generating a new submap, all submaps are anchored based on the spatial
positions of local keyframe poses. After each local Bundle Adjustment (BA), the central pose of the
map is adjusted for re-anchoring. To avoid overlapping artifacts at the edges of rendered submaps, we
remove Gaussian bodies outside all submap boundaries, effectively reducing boundary artifacts. Then,
we proceed with submap stitching. To ensure seamless maps, we apply the Gaussian aggregation
method described in Sec 3.2, aggregating smaller Gaussians at map boundaries into larger ones.
We observe that the merged boundaries are seamless. After loop closure adjustment and global BA
execution, we repeat the map fusion process. Our submap fusion strategy avoiding excessive memory
consumption due to continuous map expansion.
3.4 Loop Closure and BA
Local-Global Loop Closure To correct accumulative drift, we perform local Bundle Adjustment
(BA) within each submap, involving only local submap keyframe corrections. Inspired by [29], we
utilize the Bag of Words (BoW) model for relevance detection among global keyframes. When loop
closure conditions are met, a global optimization process is initiated. To align global submaps and
fuse submaps, we adopt a coarse-to-fine global adjustment strategy.
5

<!-- page 6 -->
In contrast to traditional methods, we first optimize the pose of anchor frames using BA and perform
a submap fusion process. This prevents drift at the boundaries of Gaussian submaps. Afterwards, we
fix the position of anchor frames, execute a global BA process based on the global keyframe list, and
then perform a second Gaussian submap fusion process to complete loop closure. This enables our
system to correct significant drift while avoiding missing and overlapping artifacts caused by map
misalignment. Additionally, we randomly sample rays from all keyframes involved in global BA to
guide the generation and fusion of Gaussian bodies, implementing the process described in Section
3.2 to further correct rendering errors.
Local-Global Bundle Adjustment Unlike methods that adjust using neural point clouds, our ap-
proach does not correct all mapping errors at once. Instead, we perform coarse-to-fine rendering
adjustments. Thanks to the speed advantage of 3D Gaussian rendering, we can achieve real-time
re-rendering and construct bundle adjustment (BA) losses.
To ensure geometric shape and appearance consistency, we distort the rendered RGB and depth to the
co-visible keyframes, constructing the loss according to the following equations:
LBA−Igb =
N−1
X
i=1
N
X
j=i+1
T j
i · R(Ti, c) −F c
j


(8)
LBA−depth =
N−1
X
i=1
N
X
j=i+1
T j
i · R(Ti, d) −F d
j


(9)
where F c
j and F d
j are the color and depth of keyframe j, respectively. R(Ti, c) and R(Ti, d) represent
the rendered RGB and depth. Thus, the overall loss function LBA used for joint optimization of the
corresponding keyframe poses and 3D Gaussian scene representation is a weighted sum of the above
losses. In our experiments, we found that the scale explosion of aggregated Gaussian bodies during
the BA process could affect rendering. Therefore, we apply an L2 loss Lrgs to 3D Gaussians with
scales exceeding the threshold t2. The total rendering loss L we obtain is:
LBA = λ1Lcolor + λ2Ldepth + λ3Lrgs
(10)
Where λ1, λ2, and λ3 are weighting coefficients. To address potential local errors caused by
forgetting Gaussian submaps, we conduct extra optimization iterations for all co-visible keyframes.
Unlike methods using neural point clouds, our approach benefits from faster rendering and lower
computational costs of Gaussian rasterization. Unlike the high computational cost in [53], we avoid
globally re-rendering neural radiation fields.
4
Experiments
Implementation Details. We implemented NGM-SLAM on a desktop computer equipped with
an Intel i7-12700K and an NVIDIA RTX 3090 Ti with 24 GB. Our implementation utilized mixed
programming in C++ and Python. To ensure fair comparisons, we provided experimental data for
both monocular and RGB-D setups. Detailed parameter settings are available in the supplementary
materials. Our baselines include traditional methods, neural implicit approaches, and state-of-the-art
(SOTA) systems based on 3D Gaussian SLAM, including ORBSLAM3[29], BAD-SLAM[6], Vox-
Fusion[38], DROID-SLAM[22], Co-SLAM[10], ESLAM[19], Go-SLAM[53], Point-SLAM[31],
SplaTAM[54] and MonoGS[41]. For some data, we used the results reported in the respective papers
of these methodologies.
Datasets and Metrics. We utilized four RGB-D datasets, which include 8 small room sequences and
4 large-scale multi-room sequences from the Replica[13] dataset, featuring complex corridors and
stairs. Additionally, we used 3 indoor scene sequences captured by three real sensors from the TUM
RGB-D[14] dataset, 6 sequences from the ScanNet[11] dataset, comprising large-scale real indoor
scenes, 4 indoor scene sequences from the ScanNet++[55] dataset, and 3 indoor sequences with
challenging large perspective changes from the EuRoC[8] dataset. For tracking, we employed ATE
RMSE (cm) as the benchmark. For reconstruction, we compared rendering accuracy using PSNR,
SSIM, and LPIPS. Our computational results represent the average of five experiments rendered
along the camera’s direction for all frames. We assessed running speed and computational resource
requirements using FPS and GPU Memory Usage.
6

<!-- page 7 -->
4.1
Evaluation on Replica
Metrics
PSNR(dB)↑
SSIM↑
LPIPS↓
ATE(cm)↓
Tracking FPS↑
System FPS↑
GPU Usage(G)↓
NICE-SLAM [9]
24.42
0.81
0.23
2.35
2.33
1.91
6.27
Co-SLAM [10]
30.24
0.86
0.18
1.16
14.58
12.64
5.83
Go-SLAM [10]
24.15
0.77
0.35
1.12
10.74
8.26
14.44
Point-SLAM [31]
33.49
0.97
0.14
0.73
1.10
0.42
7.31
SplaTAM [54]
31.81
0.96
0.16
0.55
1.07
0.42
18.87
MonoGS [41]
34.05
0.96
0.12
0.58
4.58
2.26
27.99
NGM-SLAM(Mono)
35.02
0.96
0.13
8.51
16.11
3.82
7.62
NGM-SLAM
37.43
0.98
0.08
0.51
20.54
5.71
5.98
Table 1: The average results of five measurements for eight scenes of a sequence of smaller rooms in
the Replica[13] dataset are reported for PSNR (dB), SSIM, LPIPS, ATE (cm), Tracking FPS, System
FPS, and GPU usage. The best results are bolded, and the second best results are indicated with an
underline.
As shown in Table 1, we provide experimental data for monocular and RGB-D small-scale room
scenes, comparing them with other methods. We demonstrate the quality evaluation of scene
reconstruction, including PSNR(dB), SSIM, LPIPS, and ATE RMS (cm), as well as the running
speed. Our method achieves state-of-the-art results, ensuring real-time performance while running
on lower memory, which gives it an advantage over other 3DGS-SLAM methods. It serves as a
foundation for extension to mobile platforms such as robots. Our monocular mode can also achieve
reconstruction quality close to RGB-D without depth supervision. As shown in Figure 1, we compare
the reconstruction results of monocular and RGB-D modes with other methods on room sequences
from the Replica[13] dataset. Compared to the baseline method, our reconstruction shows better
detail, clearer texture, fills holes, and avoids ghosting and local reconstruction errors.
Figure 2: We present scene and local detail results on four sequences in the Replica[13] dataset,
including monocular and RGB-D reconstruction. Our method exhibits superior detail expression and
overall reconstruction, while preserving the finest texture details.
As shown in Table 2, we present the tracking results on five larger-scale scenes from the Replica[13]
dataset. Our method’s ATE results show a 23.9% improvement in accuracy compared to Go-
SLAM[53]. As illustrated in Figure 2, compared to the baseline method, we avoid the accumulation
of errors caused by large-scale tracking. In complex large-scale scenes such as Apartment-1, which
includes multiple rooms and corridors between rooms, accumulated errors can lead to catastrophic
7

<!-- page 8 -->
Part
Apartment-0
Apartment-1
Apartmen-2
Frl-apartment-0
Frl-apartment-4
Average
NICE-SLAM [9]
16.99
14.52
8.19
2.01
2.37
8.46
Co-SLAM [10]
8.60
9.78
6.31
1.94
0.81
5.89
Go-SLAM [53]
14.10
3.54
1.00
0.40
0.12
3.23
Go-SLAM(Mono) [53]
29.81
17.43
5.39
1.50
2.02
11.23
Point-SLAM [31]
13.49
10.97
8.22
2.21
1.48
7.27
MonoGS [41]
19.02
9.37
2.52
0.98
0.97
6.57
NGM-SLAM(Mono)
17.86
21.18
13.89
8.61
5.66
13.24
NGM-SLAM
5.23
4.91
1.01
0.81
0.33
2.46
Table 2: The performance of ATE RMSE (cm) on 5 large-scale scene sequences from the Replica[13]
dataset. The average of 5 measurements is taken for each sequence. The best result is indicated in
bold, and the second-best result is underlined. Our method outperformed the baseline method.
forgetting of the scene, highlighting the importance of loop closure. Meanwhile, we ensure advantages
in detail representation and reasonable hole filling.
Figure 3: The reconstruction results on four large-scale apartment sequences, each consisting of
multiple rooms, in the Replica[13] dataset demonstrate that our method achieves more accurate
reconstruction compared to Nerf-based approaches and state-of-the-art MonoGS[41]. It avoids
catastrophic forgetting. Moreover, as demonstrated in the final sequence showcasing window details,
we can achieve reasonable background completion and scene generalization.
4.2
Evaluation on ScanNet
Part
scene0000
scene0059
scene0106
scene0169
scene0181
scene0207
Average
NICE-SLAM [9]
8.64
12.25
8.09
10.28
12.93
5.59
9.63
Co-SLAM [10]
7.13
11.14
9.36
5.90
11.81
7.14
8.75
Vox-Fusion [38]
16.62
24.23
8.41
27.33
23.31
9.49
18.23
ESLAM [19]
7.54
8.52
7.39
8.17
9.13
5.61
7.73
Point-SLAM [31]
10.24
7.81
8.65
22.16
14.77
9.54
12.20
SplaTAM [54]
12.83
10.10
17.72
12.08
11.10
7.47
11.88
NGM-SLAM(w/o loop)
7.42
6.43
7.31
6.81
12.33
7.92
8.05
NGM-SLAM(w/ loop)
6.71
6.26
7.24
5.83
10.12
7.44
7.27
Table 3: The performance of ATE RMSE (cm) on 5 large-scale scene sequences from the Replica[13]
dataset. The average of 5 measurements is taken for each sequence. The best result is indicated in
bold, and the second-best result is underlined. Our method outperformed the baseline method.
Our tracking results on the ScanNet[11] dataset are shown in Table 3. We demonstrate more robust
tracking when loop closure detection and bundle optimization are enabled. In Figure 2, we illustrate
the local reconstruction results, where our method can reasonably fill in gaps, such as chairs and
walls. Additionally, by eliminating the accumulation errors in mapping and correcting mapping errors
8

<!-- page 9 -->
caused by incorrect scene updates, we can accurately recover the geometric shapes of objects, such
as bicycles, shoes, and cellos in the scene.
Figure 4: On large-scale multi-room sequences in the ScanNet dataset, our method demonstrates
superior error accumulation correction capability compared to current 3DGS-based approaches. We
can accurately ensure consistency across multiple views, avoiding erroneous scene reconstructions
such as blurry shoes and bicycles, while also preventing local detail collapse caused by aliasing
artifacts.
4.3
Evaluation on TUM RGB-D and EuRoC
Our ATE RMSE (cm) and construction results in indoor scenes for TUM RGB-D[14] datasets as
shown in Tables 4 and EuRoC[8] dataset in Tables 5. Our method achieves competitive results
compared to traditional approaches in tracking performance.
fr1_desk
fr2_xyz
fr3_office
AVG
DI-Fusion [12]
4.4
2.0
5.8
4.1
BAD-SLAM [6]
1.7
1.1
1.7
1.5
ESLAM [19]
2.3
1.1
2.4
2.0
MonoGS [41]
1.52
1.58
1.65
1.58
NGM-SLAM(Mono)
1.57
0.72
1.42
1.24
NGM-SLAM
1.72
0.40
1.00
1.04
Table 4: The performance of ATE RMSE (cm) on TUM RGB-D[14]
dataset. The best result is indicated in bold, and the second-best
result is underlined.
V101
V102
V103
AVG
ORB-SLAM2 [16]
0.035
0.020
0.048
0.034
ORB-SLAM3 [29]
0.035
0.025
0.052
0.037
SVO [58]
0.045
0.040
0.070
0.051
Go-SLAM [53]
0.041
0.040
0.024
0.035
DROID-SLAM [22]
0.037
0.026
0.023
0.029
NGM-SLAM
0.033
0.027
0.020
0.027
Table 5: The performance of ATE RMSE (cm) for stereo
input on the EuRoC[8] dataset, with the best result indi-
cated in bold and the second-best result underlined.
5
Conclusion
We propose NGM-SLAM, the first Gaussian SLAM system based on neural submaps. Through the
priors provided by neural submaps and loop closure adjustments, we achieve high-quality tracking and
reconstruction of large-scale scenes, enabling local-to-global loop closures and correcting cumulative
errors. We strike a balance between high-quality texture detail representation and real-time operation.
Experimental results demonstrate that our approach outperforms state-of-the-art NERF/GS-based
SLAM methods in terms of rendering and tracking accuracy.
9

<!-- page 10 -->
References
[1] T. Pire, T. Fischer, G. Castro, P. De Crist’oforis, J. Civera, and J. J. Berlles. S-PTAM: Stereo parallel
tracking and mapping. Robotics and Autonomous Systems, 93:27–42, 2017.
[2] S. Zakharov, W. Kehl, A. Bhargava, and A. Gaidon. Autolabeling 3d objects with differentiable rendering
of sdf shape priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 12224–12233, 2020.
[3] Chen Wang, Xian Wu, Yuan-Chen Guo, Song-Hai Zhang, Yu-Wing Tai, and Shi-Min Hu. NeRF-SR: High
Quality Neural Radiance Fields using Supersampling. In Proceedings of the 30th ACM International
Conference on Multimedia, pages=6445–6454, 2022.
[4] Jeffrey Ichnowski, Yahav Avigal, Justin Kerr, and Ken Goldberg. Dex-nerf: Using a neural radiance field
to grasp transparent objects. arXiv preprint arXiv:2110.14217, 2021.
[5] Benran Hu, Junkai Huang, Yichen Liu, Yu-Wing Tai, and Chi-Keung Tang. NeRF-RPN: A general
framework for object detection in NeRFs. arXiv preprint arXiv:2211.11646, 2022.
[6] Thomas Schops, Torsten Sattler, and Marc Pollefeys. Bad slam: Bundle adjusted direct RGB-D slam. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages=134–144,
2019.
[7] Z. Zhu, S. Peng, V. Larsson, et al., “Nicer-slam: Neural implicit scene encoding for RGB SLAM,” arXiv
preprint arXiv:2302.03594, 2023.
[8] M. Burri, J. Nikolic, P. Gohl, et al. "The EuRoC micro aerial vehicle datasets," The International Journal
of Robotics Research, vol. 35, no. 10, pp. 1157-1163, 2016.
[9] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, "Nice-slam: Neural
implicit scalable encoding for slam," in Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 2022, pp. 12786–12796.
[10] Hengyi Wang, Jingwen Wang, Lourdes Agapito. "Co-SLAM: Joint Coordinate and Sparse Parametric
Encodings for Neural Real-Time SLAM,"in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2023, pp. 13293-13302.
[11] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, "Scannet: Richly-annotated
3d reconstructions of indoor scenes," in Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, 2017, pp. 5828–5839.
[12] J. Huang, S.-S. Huang, H. Song, and S.-M. Hu. "Di-fusion: Online implicit 3d reconstruction with deep
priors," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021,
pp. 8932–8941.
[13] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J.J. Engel, R. Mur-Artal, C. Ren, S. Verma,
et al. "The Replica Dataset: A digital replica of indoor spaces," arXiv preprint arXiv:1906.05797, 2019.
[14] J. Sturm, W. Burgard, and D. Cremers. "Evaluating egomotion and structure-from-motion approaches
using the TUM RGB-D benchmark," in Proc. of the Workshop on Color-Depth Camera Fusion in
Robotics at the IEEE/RJS International Conference on Intelligent Robot Systems (IROS), 2012, pp. 1–7.
[15] R.A. Newcombe, S.J. Lovegrove, and A.J. Davison. "DTAM: Dense tracking and mapping in real-time,"
in 2011 International Conference on Computer Vision, IEEE, 2011, pp. 2320–2327.
[16] R. Mur-Artal, J.M.M. Montiel, and J.D. Tardos. "ORB-SLAM: a versatile and accurate monocular SLAM
system," IEEE Transactions on Robotics, vol. 31, no. 5, pp. 1147–1163, 2015.
[17] Z. Liao, Y. Hu, J. Zhang, X. Qi, X. Zhang, and W. Wang. "So-slam: Semantic object slam with scale
proportional and symmetrical texture constraints," IEEE Robotics and Automation Letters, vol. 7, no. 2,
pp. 4008–4015, 2022.
[18] M. R"unz and L. Agapito, “Co-fusion: Real-time segmentation, tracking and fusion of multiple objects,”
in 2017 IEEE International Conference on Robotics and Automation (ICRA), 2017, pp. 4471–4478.
[19] Michael Bloesch, Jan Czarnowski, Ronald Clark, Stefan Leutenegger, and Andrew J Davison.
"CodeSLAM—learning a compact, optimisable representation for dense visual SLAM." In Proceedings
of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2560-2568. 2018.
10

<!-- page 11 -->
[20] R. Craig and R. C. Beavis, “TANDEM: matching proteins with tandem mass spectra,” Bioinformatics,
vol. 20, no. 9, pp. 1466–1467, 2004.
[21] Y. Yao, Z. Luo, S. Li, T. Shen, T. Fang, and L. Quan, “Recurrent MVSNet for high-resolution multi-view
stereo depth inference,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2019, pp. 5525–5534.
[22] Z. Teed and J. Deng, "Droid-slam: Deep visual slam for monocular, stereo, and RGB-D cameras,"
Advances in Neural Information Processing Systems, vol. 34, pp. 16558-16569, 2021.
[23] A. Yu, V. Ye, M. Tancik, and A. Kanazawa, "Pixelnerf: Neural radiance fields from one or few images,"
in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp.
4578-4587.
[24] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan, "Depth-supervised NeRF: Fewer views
and faster training for free," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2022, pp. 12882-12891.
[25] Y.-C. Guo, D. Kang, L. Bao, Y. He, and S.-H. Zhang, "Nerfren: Neural radiance fields with reflections,"
in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp.
18409-18418.
[26] Q. Xu, Z. Xu, J. Philip, S. Bi, Z. Shu, K. Sunkavalli, and U. Neumann. "Point-nerf: Point-based
neural radiance fields," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2022, pp. 5438-5448.
[27] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, “Mip-nerf:
A multiscale representation for anti-aliasing neural radiance fields,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2021, pp. 5855–5864.
[28] Bernhard Kerbl, et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering," ACM Transac-
tions on Graphics (ToG), 42.4 (2023): 1-14.
[29] C. Campos, R. Elvira, J. J. G. Rodríguez, J. M. M. Montiel, and J. D. Tardós, "ORB-SLAM3: An
Accurate Open-Source Library for Visual, Visual-Inertial, and Multimap SLAM," IEEE Transactions on
Robotics, vol. 37, no. 6, pp. 1874–1890, 2021.
[30] Shi, X., Li, D., Zhao, P., et al., “Are we ready for service robots? The OpenLORIS-Scene datasets for
lifelong SLAM,” In Proceedings of the 2020 IEEE International Conference on Robotics and Automation
(ICRA), IEEE, 2020, pp. 3139-3145.
[31] Erik Sandström, Yanchao Li, Luc Van Gool, et al., “Point-SLAM: Dense Neural Point Cloud-Based
SLAM,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp.
18433-18444.
[32] T. Whelan, M. Kaess, M. Fallon, H. Johannsson, J. Leonard, and J. McDonald, "Kintinuous: Spatially
extended kinectfusion," in Proc. Workshop RGB-D, Adv. Reason. Depth Cameras, 2012, article 4.
[33] T. Deng, H. Xie, J. Wang, W. Chen. "Long-Term Visual Simultaneous Localization and Mapping: Using
a Bayesian Persistence Filter-Based Global Map Prediction," IEEE Robotics & Automation Magazine,
vol. 30, no. 1, pp. 36-49, 2023.
[34] T. Deng, G. Shen, T. Qin, J. Wang, W. Zhao, J. Wang, D. Wang, W. Chen. "PLGSLAM: Progressive
Neural Scene Representation with Local to Global Bundle Adjustment," arXiv preprint arXiv:2312.09866,
2023.
[35] T. Deng, S. Liu, X. Wang, Y. Liu, D. Wang, W. Chen. "ProSGNeRF: Progressive Dynamic Neural Scene
Graph with Frequency Modulated Auto-Encoder in Urban Scenes," arXiv preprint arXiv:2312.09076,
2023.
[36] T. Qin, P. Li, S. Shen. "Vins-mono: A robust and versatile monocular visual-inertial state estimator,"
IEEE Transactions on Robotics, vol. 34, no. 4, pp. 1004-1020, 2018.
[37] T. Whelan, M. Kaess, H. Johannsson, et al. "Real-time large-scale dense RGB-D SLAM with volumetric
fusion," The International Journal of Robotics Research, vol. 34, no. 4-5, pp. 598-626, 2015.
[38] X. Yang, H. Li, H. Zhai, et al., “Vox-Fusion: Dense tracking and map with voxel-based neural implicit
representation,” in 2022 IEEE International Symposium on Mixed and Augmented Reality (ISMAR),
IEEE, 2022, pp. 499-507.
11

<!-- page 12 -->
[39] M. Hosseinzadeh, K. Li, Y. Latif, et al. "Real-time monocular object-model aware sparse SLAM," in
2019 International Conference on Robotics and Automation (ICRA), IEEE, 2019, pp. 7123-7129.
[40] T. Qin, P. Li, S. Shen, “Vins-mono: A robust and versatile monocular visual-inertial state estimator,”
IEEE Transactions on Robotics, 34(4): 1004-1020, 2018.
[41] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, Andrew J. Davison, “Gaussian Splatting SLAM,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
[42] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong Wang, Xuelong Li, “GS-SLAM: Dense
Visual SLAM with 3D Gaussian Splatting,” in CVPR, 2024.
[43] X. Kong, S. Liu, M. Taher, et al., “vmap: Vectorised object map for neural field slam,” in Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 952-961.
[44] M. Li, J. He, Y. Wang, et al. "End-to-End RGB-D SLAM With Multi-MLPs Dense Neural Implicit
Representations," IEEE Robotics and Automation Letters, 2023.
[45] C. M. Chung, Y. C. Tseng, Y. C. Hsu, et al., “Orbeez-SLAM: A Real-Time Monocular Visual SLAM
with ORB Features and NeRF-Realized Map,” in 2023 IEEE International Conference on Robotics and
Automation (ICRA), IEEE, 2023, pp. 9400-9406.
[46] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohi, J. Shotton, S.
Hodges, and A. Fitzgibbon, "Kinectfusion: Real-time dense surface mapping and tracking," in 2011 10th
IEEE international symposium on mixed and augmented reality, 2011, pp. 127-136.
[47] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, "iMAP: Implicit mapping and positioning in real-time," in
Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 6229–6238.
[48] A. Rosinol, J. J. Leonard, and L. Carlone, "NeRF-SLAM: Real-Time Dense Monocular SLAM with
Neural Radiance Fields," arXiv preprint arXiv:2210.13641, 2022.
[49] T. Müller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics primitives with a multiresolution
hash encoding,” ACM Transactions on Graphics (ToG), vol. 41, no. 4, pp. 1–15, 2022.
[50] P. Wang, L. Liu, Y. Liu, C. Theobalt, T. Komura, and W. Wang, “Neus: Learning neural implicit surfaces
by volume rendering for multi-view reconstruction,” arXiv preprint arXiv:2106.10689, 2021.
[51] H. Turki, D. Ramanan, and M. Satyanarayanan, “Mega-nerf: Scalable construction of large-scale nerfs
for virtual fly-throughs,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2022, pp. 12922–12931.
[52] Zehao Yu, Torsten Sattler, Andreas Geiger, “Gaussian Opacity Fields: Efficient High-quality Compact
Surface Reconstruction in Unbounded Scenes,” arXiv:2404.10772, 2024.
[53] Y. Zhang, F. Tosi, S. Mattoccia, et al. "Go-slam: Global optimization for consistent 3D instant recon-
struction," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp.
3727-3737.
[54] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva
Ramanan, Jonathon Luiten, “SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
[55] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, Angela Dai, “ScanNet++: A High-Fidelity
Dataset of 3D Indoor Scenes,” in Proceedings of the IEEE/CVF International Conference on Computer
Vision (ICCV), October 2023, pp. 12-22.
[56] M. Niemeyer, F. Manhardt, M. J. Rakotosaona, et al. RadSplat: Radiance Field-Informed Gaussian
Splatting for Robust Real-Time Rendering with 900+ FPS. arXiv preprint arXiv:2403.13806, 2024.
[57] M. Zwicker, H. Pfister, J. Van Baar, et al. EWA volume splatting. In Proceedings Visualization, 2001.
VIS’01. IEEE, 2001, pp. 29-538.
[58] C. Forster, Z. Zhang, M. Gassner, M. Werlberger, and D. Scaramuzza. SVO: Semidirect visual odometry
for monocular and multicamera systems. IEEE Transactions on Robotics, 33(2):249–265, 2016.
[59] Huang, Huajian, Li, Longwei, Cheng Hui, and Yeung, Sai-Kit. "Photo-SLAM: Real-time Simultaneous
Localization and Photorealistic Mapping for Monocular, Stereo, and RGB-D Cameras." In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
12
