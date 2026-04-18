<!-- page 1 -->
5720
IEEE TRANSACTIONS ON ROBOTICS, VOL. 41, 2025
Toward Accurate, Efﬁcient, and Robust RGB-D
Simultaneous Localization and Mapping in
Challenging Environments
Hui Zhao
, Fuqiang Gu
, Member, IEEE, Jianga Shang
, Member, IEEE, Xianlei Long
, Jiarui Dou
,
Chao Chen
, Senior Member, IEEE, Huayan Pu
, and Jun Luo
Abstract—Visual
simultaneous
localization
and
mapping
(SLAM) is crucial to many applications such as self-driving vehicles
and robot tasks. However, it is still challenging for existing visual
SLAM approaches to achieve good performance in low-texture or
illumination-changing scenes. In recent years, some researchers
have turned to edge-based SLAM approaches to deal with the chal-
lenging scenes, which are more robust than feature-based and di-
rectSLAMmethods.Nevertheless,existingedge-basedmethodsare
computationally expensive and inferior than other visual SLAM
systems in terms of accuracy. In this study, we propose EdgeSLAM,
a novel RGB-D edge-based SLAM approach to deal with challeng-
ing scenarios that is efﬁcient, accurate, and robust. EdgeSLAM is
built on two innovative modules: efﬁcient edge selection and adap-
tive robust motion estimation. The edge selection module can efﬁ-
ciently select a small set of edge pixels, which signiﬁcantly improves
the computational efﬁciency without sacriﬁcing the accuracy. The
motion estimation module improves the system’s accuracy and
robustness by adaptively handling outliers in motion estimation.
Extensive experiments were conducted on technical university of
munich (TUM) RGBD, imperial college london (ICL)-National
University of Ireland Maynooth (NUIM), and ETH zurich 3D re-
construction (ETH3D) datasets, and experimental results show that
EdgeSLAM signiﬁcantly outperforms ﬁve state-of-the-art methods
in terms of efﬁciency, accuracy, and robustness, which achieves
29.17% accuracy improvements with a high processing speed of
up to 120 frames/s and a high positioning success rate of 97.06%.
Received 27 January 2025; revised 28 July 2025; accepted 21 August 2025.
Date of publication 16 September 2025; date of current version 8 October 2025.
This work was supported in part by the National Natural Science Foundation
of China under Grant 42174050, Grant 42474027, Grant 62322601, and Grant
T2421001, in part by the Fundamental Research Funds for the Central Uni-
versities under Grant 2024IAIS-QN017, and in part by the Excellent Youth
Foundation of Chongqing under Grant CSTB2023NSCQJQX0025. This article
was recommended for publication by Guest Associate Editor S. Leutenegger and
Editor J. Civera upon evaluation of the reviewers’ comments. (Corresponding
authors: Fuqiang Gu; Jianga Shang.)
Hui Zhao is with the School of Geography and Information Engineer-
ing, China University of Geosciences, Wuhan 430078, China (e-mail: zhao-
hui@cug.edu.cn).
Fuqiang Gu, Xianlei Long, and Jiarui Dou are with the College of Com-
puter Science, Chongqing University, Chongqing 401331, China (e-mail:
gufq@cqu.edu.cn; xianlei.long@cqu.edu.cn; 202114131181@cqu.edu.cn).
Jianga Shang is with the School of Computer Science, China University
of Geosciences, Wuhan 430078, China, and also with the Engineering Re-
search Center of Natural Resource Information Management and Digital Twin
Engineering Software, Ministry of Education, Wuhan 430078, China (e-mail:
jgshang@cug.edu.cn).
Chao Chen, Huayan Pu, and Jun Luo are with the State Key Lab-
oratory of Mechanical Transmissions, Chongqing University, Chongqing
400044, China (e-mail: cschaochen@shu.edu.cn; phygood_2001@shu.edu.cn;
luojun@cqu.edu.cn).
Digital Object Identiﬁer 10.1109/TRO.2025.3610173
Index Terms—Edge selection, edge-based simultaneous localiza-
tion and mapping (SLAM), robust pose estimation, visual SLAM,
visual odometry.
I. INTRODUCTION
V
ISUAL simultaneous localization and mapping (SLAM)
estimates camera pose and 3-D structure from image
streams in unknown environments, with broad applications in
autonomous vehicles and robotics [1], [2]. Existing SLAM
methods fall into two categories: feature-based and direct.
Feature-based methods [3] rely on keypoint extraction and
matching, which demands high visual texture and incurs con-
siderable computational cost. Direct methods [4] align pixel
intensities or depth directly, offering better efﬁciency but re-
quire photometric consistency and smooth motion due to their
nonconvex objective.
Despite extensive research, achieving a strong balance among
accuracy, efﬁciency, and robustness remains difﬁcult, particu-
larly in low-texture scenes or under dynamic illumination [5],
[6].Feature-basedmethodsstrugglewithtexturelessareas,while
direct methods fail under sudden lighting changes. These lim-
itations are exacerbated on resource-constrained platforms like
mobile devices or drones.
Recently, edge-based SLAM methods [7], [8], [9] have
emerged as promising alternatives. Image edges remain stable
across illumination variations and can be reliably observed even
in textureless scenes. Edge-based methods use 3-D/2-D edge
registration with geometric alignment, offering robustness to
lighting changes. Nevertheless, there are still several challenges
associated with edge-based approaches, which are outlined as
follows.
1) To enforce the estimation accuracy, most edge-based sys-
tems have to exploit a large amount of measurements
to solve a nonlinear least squares problem [10], [11],
which drastically increases the computational complexity
of edge-based methods.
2) The edge registration can be easily plagued with outliers
in practice due to measurement errors or erroneous data
association [9], which leads to either longer convergence
time or poor estimates.
To address these challenges, we propose EdgeSLAM, a novel
edge-based RGB-D SLAM framework that achieves high accu-
racy, efﬁciency, and robustness in challenging scenarios. Unlike
1941-0468 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artiﬁcial intelligence and similar technologies.
Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 2 -->
ZHAO et al.: TOWARD ACCURATE, EFFICIENT, AND ROBUST RGB-D SLAM IN CHALLENGING ENVIRONMENTS
5721
Fig.1.
Performance of the proposed EdgeSLAMascompared to SOTASLAM
methods on TUM RGBD, ICL-NUIM, and ETH3D datasets. Metrics include
RMSE of absolute trajectory error (RMSE ATE), average processing time per
frame (Proc. Time), and success rate across all datasets. EdgeSLAM achieves a
promising 0.017-m error, 7.9-ms processing time, and 97.06% success rate.
previous methods that use all extracted edge pixels, EdgeS-
LAM selectively tracks a small subset of informative edges,
signiﬁcantly reducing computation while preserving accuracy.
Speciﬁcally, we ﬁrst formulate edge selection as a submatrix
selection problem and solve it via a greedy algorithm that con-
siders spatial correlation and uncertainty, achieving linear com-
plexity with a proven approximation bound. Then, we develop
an adaptive robust motion estimation scheme that dynamically
chooses appropriate robust kernels based on residual distribu-
tions, improving resilience to outliers. This is cast as a uniﬁed
optimization problem with a provable upper bound and a relaxed
solver with convergence guarantees. As illustrated in Fig. 1,
EdgeSLAM consistently outperforms state-of-the-art (SOTA)
SLAM systems across three public benchmarks, achieving su-
perior accuracy, speed (120 frames/s), and robustness (97.06%
success rate).
In summary, our main contributions are as follows.
1) We introduce EdgeSLAM, a complete edge-based SLAM
system with tracking, mapping, and loop closure. It em-
ploys informative edge selection and adaptive robust esti-
mation to ensure efﬁciency and stability.
2) We propose a submodular edge selection algorithm to
reduce computation and preserve the spectral properties
of motion estimation.
3) We design an adaptive robust estimation method that
handles outliers effectively via automatic kernel selection,
enhancing robustness and reducing trajectory drift.
4) We conduct extensive experiments on TUM RGBD, ICL-
NUIM, and ETH3D, demonstrating substantial perfor-
mance gains over ﬁve SOTA methods in both accuracy
and runtime. It achieves 29.17% accuracy improvements,
120-frames/s processing speed, and 97.06% success rate,
which signiﬁcantly outperforms other methods.
Note that EdgeSLAM extends our prior work, EdgeVO [12],
byincorporatingloopclosureandadaptiveestimation.Theseim-
provements lead to signiﬁcant performance gains in challenging
environments, with only a marginal increase in computational
cost.
II. RELATED WORKS
This work focuses on visual SLAM, categorizing it into
feature-based methods, direct methods, and edge-based meth-
ods. We provide a brief overview of representative works in
each category and recommend readers wanting more details to
more comprehensive survey articles.
A. Feature-Based SLAM
Parallel tracking and mapping [13] is the ﬁrst keyframe-based
SLAM system, which separates tracking and mapping and run
in two parallel threads. It relies on feature correspondence for
data association and bundle adjustment (BA) for pose and map
optimization. The studies in [14] further integrated the iterative
closest point (ICP) algorithm [15] to develop an RGB-D camera-
based SLAM system. However, these methods are prone to
tracking loss and are suitable only for small-scale environments.
Recently, the SOTA ORB-SLAM3 [16] has greatly improved the
robustness of feature-based methods, which performs data asso-
ciation based on ORB features. The camera pose is estimated by
tracking with reference keyframe or tracking with motion model
and reﬁned by tracking with local maps. Another popular SLAM
system is RTAB-Map [17], which is still actively maintained and
updated. It offers greater ﬂexibility and integration with different
sensors and has now become a standard for robotic localization.
As known, these methods require sufﬁcient feature points,
uniform spatial distribution, and reliable matching to achieve
accurate and robust pose estimation, which are difﬁcult to ob-
tain in low-texture environments. Thus, some methods have
been proposed to increase the system robustness. For example,
StructSLAM [18] and PL-SLAM [19] integrate feature points
and the building lines detected by the LSD [20] for localization
and mapping. The works in [21] and [22] utilize plane features
to estimate pose through point-plane error [23] or photometric
alignment. Furthermore, the study in [24] combines point, line,
and plane features to achieve more robust pose tracking.
The integration of multiple features essentially enhances
the utilization of image information, mitigating the limitations
posed by low-texture scenes to some extent. However, the data
association incurs higher computational costs, affecting the sys-
tem’s operational efﬁciency. In this study, we try to select a small
subset of features by using the proposed submodular partition
optimization module to avoid massive computational cost.
B. Direct SLAM
Another popular type of visual SLAM is direct methods,
which couple pose estimation and data association together
through direct alignment. An early direct method is DTAM [25],
which generates dense depth maps at selected keyframes and
performs real-time camera tracking through a frame-to-model
photometric alignment. DTAM assumes the photometric con-
sistency, but when the illumination of the environment changes,
the estimated camera trajectory can drift. Thus, the work in [26]
and ElasticFusion [27] combines photometric error and ICP
geometric error, achieving the robust tracking and mapping
of an RGB-D camera by minimizing both errors. Building on
ElasticFusion, BundleFusion [28] additionally incorporates a
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 3 -->
5722
IEEE TRANSACTIONS ON ROBOTICS, VOL. 41, 2025
feature matching module, employing a coarse-to-ﬁne strategy
to achieve globally consistent estimation.
These methods require high computational costs for BA due
mainly to the extensive use of pixels to enhance accuracy and
robustness. They often need the assistance of GPUs for real-time
performance, making them unsuitable for platforms with limited
resources. Instead, DVO-SLAM [29] reduces the resolution of
the image to estimate camera poses, and RGBD-TAM [30]
and LSD-SLAM [31] only utilize pixels with large intensity
gradients. These methods reduce the computational burden by
processing fewer pixels, achieving real-time performance only
on a CPU.
Although direct methods are more robust than feature-based
methods in low-texture scenes, they are more sensitive to illu-
mination changes due to the optimization performed on pho-
tometric values. There are some improvements to address this
issue. For instance, the work in [32] introduces a robust method
to compensate for afﬁne lighting changes, while DSO [4] further
exploits photometric camera calibration, including lens attenua-
tion, gamma correction, and known exposure times. These meth-
ods alleviate the strong dependence on photometric consistency
to some extent, but there is still a gap in practical applications.
C. Edge-Based SLAM
Compared with key points and lines, edges are more general
features and easily observed in man-made scenes, which can im-
prove the robustness of visual SLAM algorithms in low-texture
environments. However, edges are difﬁcult to associate, track,
and optimize over time, since there are no proper descriptors for
matching. To tackle these issues, two types of approaches have
been proposed in the literature. The ﬁrst one utilizes the image
intensity of edge pixels to achieve data association and camera
motion tracking. For instance, the work in [33] minimizes photo-
metric error over sparse edge pixels to compute camera relative
pose. The system in [34] estimates the edge correspondences
using a bidirectional sparse iterative and pyramidal version
of the Lucas–Kanade optical ﬂow and further reﬁnes these
correspondences with geometrical relationships among three
views. However, these approaches are intensity-based, which
are vulnerable to illumination changes as the direct methods.
In contrast, the other types of edge methods track camera
motion with ICP paradigm methods, which often witness more
robust results. Tarrio and Pedre [10] proposed a monocular VO
approach that performs edge alignment by searching for the
closest edge pixel along the normal direction. They subsequently
enhanced the accuracy of edge association by exploiting the
continuous nature of edges and the epipolar constraints [35].
However, this nearest neighbor (NN) search method is com-
putationally expensive and error-prone. Thus, the work in [36]
speeds up the NN search by prestructuring the data in a K-D tree
and evaluating edge correspondences according to the pixel 2-D
Euclidean distance and image gradient directions. Subsequently,
another method was proposed by Kim et al. [37], which utilizes
multiple oriented quadtrees and node caching schemes to im-
prove edge-matching speed and success rates. However, these
methods require substantial time for edge pixel preprocessing,
making it challenging to meet real-time requirements.
More recently, Euclidean distance ﬁelds (EDFs) have been in-
troduced to improve the efﬁciency of edge correspondence. The
work in [38] utilizes 2-D distance transformation to precompute
the nearest edge pixel of any pixel in the image, which converts
theclosestedgepixelsearchanddistancecalculationduringeach
iteration into query operations. The study in [8] successfully
applied 2-D distance ﬁelds for 3-D/2-D edge registration, which
show promising results in terms of accuracy and robustness.
Based on these approaches, Schenk and Fraundorfer [8] ﬁrst
proposed a complete edge-based SLAM system that runs on a
CPU in real time, and it comprises tracking, loop closure, and
relocalization. Then, Zhou et al. [9] analyzed that these methods
require bilinear interpolation or subgradient computation to
deal with the discontinuity of the EDF, which increases the
computational burden. Thus, they proposed approximate nearest
neighbor ﬁelds (ANNFs) that are computed in the same way as
EDFs and further expanded them to oriented nearest neighbor
ﬁelds (ONNFs) by classifying edge pixels into multiple maps
according to the image gradient directions.
However, the performance of these approaches is still inferior
as compared to feature-based or direct methods. Speciﬁcally,
the alignment in these approaches involves in a large number of
edge pixels, which can be an excessive computational burden.
Besides, these edge-based methods are still sensitive to outliers
and occlusions, leading to edge registration bias, as described
in [39]. In this study, we propose a robust kernel function
to discard massive outliers, making motion estimation more
accurately and robustly.
III. PRELIMINARIES
For each incoming frame Ic, we use the ICP-based algorithm
to estimate the camera motion ξ ∈se(3) relative to its reference
frame Ir by aligning their respective edge pixels. The edge align-
ment is performed by reprojecting valid edge pixel set from Ir
to Ic and minimizing the distance to the closest edge pixels in Ic.
To speed up edge alignment, we ﬁrst compute the distance ﬁeld
D : Ω ⊂R2 −→R+ of Ic using distance transform [38]. For an
edge pixel pi ∈R2 with inverse depth μi in the reference frame,
the reprojected distance residual ri under the transformation ξ
is deﬁned as
ri(ξ) = D(ω(μi, pi, ξ))
(1)
where ω(·) is a warping function for reprojecting the edge pixel
from reference frame to current frame. For any pixel q in the
image, its nearest edge pixel is denoted by q′; thus, the distance
function can be expressed as
D(q) = ||q −q′||.
(2)
Then, we compute all residuals for the valid edge pixel set
{pi|i = 1, . . ., N} in the reference frame. The edge alignment
problem is converted to a nonlinear least squares optimization
problem, which can be solved by estimating the optimal relative
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 4 -->
ZHAO et al.: TOWARD ACCURATE, EFFICIENT, AND ROBUST RGB-D SLAM IN CHALLENGING ENVIRONMENTS
5723
camera motion ξ∗, namely
ξ∗= arg min
ξ
N

i=1
||ri(ξ)||2.
(3)
From a statistical perspective, if the measurement error fol-
lows an identically distributed Gaussian, the estimated motion
parameter ξ∗is optimal. Instead, the estimation may become
exceedingly inaccurate in the case of non-Gaussian. In order to
mitigate the impact of outlier measurements, robust kernel func-
tions are commonly applied. These functions can downweight
large residuals attributed to outliers and hence reduce their
inﬂuence on the optimization. The estimated motion parameter
ξ∗can be written as
ξ∗= arg min
ξ
N

i=1
ρ(ri(ξ))
(4)
where ρ(·) is a robust kernel function, with options such as
Huber, Cauchy, and others.
To solve the nonlinear least squares problem, one can use iter-
ative methods such as Gauss–Newton or Levenberg–Marquardt.
In each iteration, such methods perform a ﬁrst-order lineariza-
tion of each residual about the current value of ξ by computing
the motion Jacobian Ji ∈R1×6 and then solve the linear least
squares problem to update the motion. Thus, the reprojected
distance residual ri(ξ) under the linear transformation can be
described as
ri(ξ) ≈ri(ξk) + JiΔξ.
(5)
Stacking all pixelwise Jacobians Ji as J = [JT
1, . . . , JT
n]T, the
least squares covariance of the motion estimation is calculated
as
Σ = (JTJ)−1
(6)
where the covariance matrix Σ quantiﬁes the uncertainty of mo-
tion estimation, which depends heavily on the spectral properties
of the Jacobian matrix J. If we use more valid edge pixels to
track, the singular values of J would increase in magnitude, and
the accuracy of motion estimation is more likely to be improved.
IV. OVERVIEW OF EDGESLAM
The architecture of EdgeSLAM, as illustrated in Fig. 2, is built
upon three dedicated parallel threads: tracking, local mapping,
and loop closure. The tracking thread estimates the relative cam-
era motion between the current frame and the latest keyframe
and then decides if a new keyframe should be created. If a new
keyframe is necessary, we propagate it to the local mapping
thread and reﬁne the relevant state variables via using a sliding
window optimization. After the reﬁnement, we try to ﬁnd reli-
able loops only on keyframes and close loops through the global
pose graph optimization (PGO). In particular, we perform edge
selection for each keyframe to enhance the efﬁciency of EdgeS-
LAM, and integrate the adaptive robust estimation into relative
motion estimation, local optimization, and PGO. Next, we elab-
orate on each key component. For edge selection and robust
motion estimation, we will detail them in two separate sections.
Fig. 2.
Overview of the EdgeSLAM. It mainly comprises three main modules:
tracking, local mapping, and loop closure. Edge selection is applied for each
keyframe, while adaptive robust estimation is integrated into all least squares
optimizations within these three modules.
Fig. 3.
Processing ﬂow for each incoming frame. (a) Gray image extracted
from the RGBD frame. (b) Corresponding edge image, where its color encodes
gradient magnitude information: red-high and blue-low. (c) Distance ﬁeld pyra-
mid where the color code is blue-near and red-far. (a) Gray image. (b) Edge
image. (c) Distance ﬁelds.
A. Tracking
In order to estimate the camera’s relative motion, we ﬁrst
preprocesseachframetoobtaintheedgeimageanddistanceﬁeld
pyramid. Then, a coarse-to-ﬁne optimization is performed for
edge alignment based on the distance ﬁeld pyramid. Finally, we
decide whether a keyframe is created according to the alignment
status. More details of these steps are provided as follows.
1) Frame Preprocessing: As shown in Fig. 3, we ﬁrst detect
edge pixels by using the Canny algorithm [40] when the current
frame comes. It works well in low-texture or illumination-
changing scenes because it locally ﬁnds the strongest edge pixels
by nonmaximum suppression of high-gradient regions. Then,
we compute the distance ﬁeld of the edge image and create a
three-level distance ﬁeld pyramid for achieving robust optimiza-
tion of edge alignment. Instead of performing edge detection
and distance transform, we directly generate a low-resolution
distance ﬁeld by downscaling from the highest resolution level
by linear interpolation [8].
2) Relative Motion Estimation: The relative camera motion
ξc,k between the current frame Ic and the latest keyframe Ik is
estimated within the ICP framework. Speciﬁcally, we use only
the selected edge pixels in Ik and adopt a coarse-to-ﬁne opti-
mization scheme with the distance ﬁeld pyramid to handle the
large displacement. The edge point reprojection and alignment
are layer-by-layer performed from the lowest resolution level to
the highest level. After the ﬁnal iteration, the optimal camera
relative motion ξ∗
c,k is obtained.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 5 -->
5724
IEEE TRANSACTIONS ON ROBOTICS, VOL. 41, 2025
3) Outlier Rejection: In order to improve the robustness and
accuracy of camera tracking, we try to identify and remove
potential outliers in each iterative optimization. At ﬁrst, the
residual terms that exceed a certain threshold are removed from
the objective function described in (4). We set the threshold
separately for each level in the distance ﬁeld pyramid. Second,
if a pair of putative edge correspondences is reasonable, their
gradient directions should be as consistent as possible under the
assumptionthatthereisnolargerotationbetween Ik andIc.Note
that the closest edge pixel coordinate is additionally retained in
extracting the distance ﬁeld. We reproject an edge pixel p from
Ik to Ic and query its closest edge pixel p′. Then, the matched
pair is regarded as outliers if the inner product of their gradient
directions is under a certain margin
g(p) · g(p′) < η
(7)
where g(·) is the normalized image gradient on the edge pixel. A
larger threshold η means stricter requirements for consistency,
which is empirically set to 0.6 in this work.
4) Keyframe Decision: In general, well-distributed key-
frames are very important for the performance of the whole
system. Similar to [4], we use the following criteria to decide
whether the current frame will be selected as a keyframe.
1) We use the mean squared error of optical ﬂow t from the
last frame to the latest frame to measure the changes in
the ﬁeld of view and the mean ﬂow without rotation t′ to
measure the occlusions. The current frame is selected as
a keyframe if w1t + w2t′ > 1, where w1 and w2 are the
weights of the two indicators.
2) If the number of edge correspondences between Ic and Ik
is below the 30% of the average edge correspondences,
we treat the current frame as a new keyframe.
B. Local Mapping
To improve the consistency of the camera trajectory, we
maintain a small local window W of keyframes in the local
mapping component, where W is set between 5 and 7. For each
new keyframe, the edge pixel that satisfy certain criterions in
the window are activated to create new geometric constraints,
and then, we perform a sliding window optimization to jointly
reﬁne the inverse depths of all active edge pixel, global camera
poses, and the intrinsic c within the window. The speciﬁc steps
of local mappings are as follows.
1) Edge Activation: When a new keyframe is added, we use
it to activate the edge pixels of the previous keyframes in the
window. In order to obtain the evenly distributed edge pixels
and reliable geometric constraints, we divide the image in grids
into ﬁxed size (e.g., 20 × 20 pixels). We then reproject all edge
pixels into these grids, while the active edge pixels are selected
from each grid. All the following criteria need to be satisﬁed at
the same time to activate an edge pixel.
1) The candidate edge pixel’s geometric residual cannot ex-
ceed a certain threshold, which is set as the median of all
residuals.
2) The reprojected gradient direction of the candidate edge
pixel should be as consistent as possible with the image
gradient direction of its closest edge pixel in the new
frame. We set the angle between the two directions to a
value in the range [0◦, 30◦].
3) The edge pixel that are tracked for longer period of time
are considered to be more reliable. Therefore, we count the
number of times each edge pixel was successfully tracked
and select the older one as the active edge pixel.
2) SlidingWindowOptimization: ForthekeyframesIi andIj
in the window, the geometric residual is computed by reproject-
ing the active edge pixel p in the reference frame Ii, associated
with the inverse depth μ, into the target frame Ij, namely
r = D(π(μ, p, ξi, ξj, c))
(8)
where π(·) denotes the warping function that reprojects the point
p from frame Ii to frame Ij with the reﬁned world poses ξi
and ξj. All state variables in the window are denoted by χ.
The optimal state vector is estimated by minimizing the overall
residuals over the window, which is described as
χ∗= arg min
χ

i∈W

p∈S

j∈W\i
w(r)r2
(9)
where w(·) is determined by the adaptive robust kernel function,
which will be detailed in Section VI-B.
To retain the efﬁciency of our method, we ﬁx the bound size
of the local window and marginalize one of previous keyframes
before adding a new one. Following [4], we keep the latest two
keyframes in the window and marginalize a keyframe if it is
further away from the newest one or it has less edge pixels
visible in others. Before marginalizing one keyframe, we ﬁrst
adapt the marginalization strategy with the Schur complement
to marginalize its active edge pixels and the edge pixels that are
not observed in the last two keyframes.
C. Loop Closure
Loop closure detection is a crucial component in SLAM
systems, which can signiﬁcantly reduce accumulated drift er-
rors and improve the overall consistency and accuracy of the
estimated trajectory. As suggested in [8], we ﬁrst perform loop
detection to obtain the reliable candidate keyframe for loop
closure and then correct the loop by PGO. The main procedures
are given as follows.
1) Loop Detection: Our loop detection method identiﬁes
loops between nonneighboring keyframes that capture similar
scenes from different viewpoints. An erroneous loop can cause
a signiﬁcant drift in the overall trajectory, and therefore, we need
to ensure the reliability of loop detection. To address this, we
adopt a two-stage approach that considers both appearance and
geometric consistency.
First, we ﬁnd loop candidates based on the visual similarity
between nonneighboring keyframes. Similar to the work in [8],
we also adopt the random ferns to encode the image information
of each keyframe, primarily because it directly utilizes RGB
valueswithoutrequiringtime-consumingfeatureextraction.The
similarity scores between keyframes can be computed using
the Hamming distance of their fern descriptor, where a lower
score indicates stronger visual similarity. Initially, we compute
the similarity score between the current keyframe It and all
keyframes within the sliding window, and take the average score
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 6 -->
ZHAO et al.: TOWARD ACCURATE, EFFICIENT, AND ROBUST RGB-D SLAM IN CHALLENGING ENVIRONMENTS
5725
Mmin as the threshold to ﬁnd matching with nonneighboring
keyframes. Subsequently, we compute the similarity scores of
all nonneighboring keyframes and the current keyframe It and
retain only the keyframes whose scores are less than Mmin as
loop candidates.
2) Geometric Veriﬁcation: Then, we ﬁnd a reliable loop from
candidates through a bidirectional geometric veriﬁcation pro-
cess. For each keyframes Ik in loop candidates, we reproject its
edge pixels to the current keyframe It to construct the residuals
{ri} and then backproject the edge pixels from It to Ik to
construct the geometric residuals {rj}. The optimal relative
transformation ξ∗is determined by minimizing the bidirectional
geometric residuals, namely
ξ∗= arg min
ξ

i
wir2
i +

j
wjr2
j.
(10)
After optimization, we count the number of inliers for each
candidate keyframe and select the keyframe with the highest
number of inliers as the ﬁnal loop candidate.
3) Pose Graph Optimization: When we ﬁnd out a reliable
loop detection, we use it to correct any existing drift in the
estimation trajectory. Speciﬁcally, we model the loop correction
as a global PGO problem, with nodes representing the mea-
sured world poses of the keyframes and edges representing the
estimated relative constraints. In the case, the error of PGO is
deﬁned as
rij = ξij ⊕(ξi ⊖ξj)
(11)
where ξij is the relative transformation between keyframes i and
j, ξi and ξj are their corrected world poses, and ⊕and ⊖are
operations in the Lie algebra space.
V. EDGE SELECTION VIA SUBMODULAR PARTITION
OPTIMIZATION
According to (6), one should use all available edge pixels
to minimize the uncertainty of motion estimation, which can
potentially improve the accuracy and robustness. However, the
number of edge pixels detected in each frame is very large (e.g.,
tens of thousands), and using all of them will greatly reduce
the computational efﬁciency. In this study, we try to use only
a small subset of edge pixels to speed up motion estimation
without sacriﬁcing the accuracy and robustness.
Inspired by the work in [41], we introduce a subset selection
scheme to achieve the aforementioned objective. The selection
scheme considers three criteria: 1) the selected edge pixels are
moreinformativetoreducetheuncertainty;2)edgepixelsshould
be evenly distributed across the image to ensure stable tracking;
and 3) their corresponding 3-D edge points are highly likely to
be observed in subsequent frames, forming effective constraints
for motion estimation. In the following, we formulate the subset
selection problem as a submodular partition optimization prob-
lem to capture all these aspects, and propose an efﬁcient method
called as stochastic partition greedy to solve it.
A. Submodularity in Subset Selection
As suggested in [41], the informative edge selection is equiv-
alent to ﬁnd a submatrix (i.e., a subset of row blocks) in J that
Fig. 4.
Comparison of edge selection. (a) All the edge pixels available in the
image. (b) Solving the submatrix selection problem leads to selected edge pixels
clusters in the image. (c) If optimized over the partition matroid, the selected
edge pixels are well distributed. (a) Overall. (b) Clustered. (c) Uniform.
preserves the overall spectral properties of J as much as possible.
Let U = {0, 1, . . . , n −1} be the indices of row blocks in full
matrix J. S denotes the index subsets of selected row blocks,
[J(S)] is the corresponding concatenated submatrix, and k is the
number of selected row blocks. Then, the selection problem of
informative edge pixels can be formulated as
arg maxS⊆U logdet([J(S)]T [J(S)])
s.t.
|S| = k
(12)
where logdet(·) is a submodular function to compute the log
determinant of a matrix, which quantiﬁes the spectral properties
of the matrix.
It is known that the submodular optimization problem is
NP-hard; the stochastic greedy method [42] is commonly used
to provide a near-optimal solution with a 1 −1/e −ϵ approxi-
mation guarantee. It starts with an empty set, and in each round i,
it randomly samples a subset R ∈U\S. Then, it picks up an ele-
ment e ∈R with the objective of maximizing the marginal gain
Δe(S) = logdet(S ∪e) −logdet(S).
(13)
B. Efﬁcient Partition Selection
As shown in Fig. 4(b), solving the aforementioned problem
often leads to the phenomenon that selected edge pixels tend
to cluster in the image, as adjacent edge pixels have similar
Jacobianmatrix,resultinginsimilargainsforspectralproperties.
This spatial correlation can result in suboptimal and potentially
unstable tracking outcomes. To address this issue, we introduce
partition matroids for the submatrix selection problem, which
ensures that the selected edge pixels are evenly distributed
throughout the image.
Speciﬁcally, the index set U of row blocks in full matrix is
divided into k disjoint partitions, where U = k
i=1 Pi, and the
edge pixels with the constraint |S ∩Pi| = 1 are selected. We
denote F as a set of all feasible solution S ∈F, and M =
(U, F) as the partition matroid. The edge selection problem is
reformulated as
arg max
S∈F
logdet(H(S)) subject to |S ∩Pi| = 1
(14)
where H(S) = JT (S)J(S) is the Hessian matrix of the edge
pixels. We impose an imaginary grid on the image and count
the edge pixels in each grid to build partitions. This forces the
selected edge pixels to spread evenly over the whole image, as
shown in Fig. 4(c).
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 7 -->
5726
IEEE TRANSACTIONS ON ROBOTICS, VOL. 41, 2025
In order to reduce the computational complexity of submatrix
selection, we design a special approximation algorithm, namely
stochastic partition greedy. At the ith iteration, we randomly
pick up a partition Pi from U. Then, an edge pixel is selected
from the partition when the following condition holds:
e∗:= arg max
e∈Pi
logdet(H(Si−1) + H(e) + λI)
(15)
where Si−1 is the subset selected at previous iteration. The
introduction of the diagonal matrix λI is to improve its numerical
stability. In this way, each edge pixel is computed only once, and
the complexity is thus reduced to O(n). Then, we provide the
approximation guarantee for the proposed method.
Proposition 1: Let T denote the optimal solution for the prob-
lem (14) and S represent the approximation solution provided
by the stochastic partition greedy. The approximation guarantee
is, thus, f(S) ≥1
2f(T).
Proof: Since the function f(S) = logdet(H(S)) is submod-
ular and monotonously increasing, we have that
f(T) ≤f(S) +

e∈T \S Δe(S).
(16)
Suppose that T\S = k
i=1 Ti, where Ti ⊆Pi. As Si−1 is a
subset of S, it now follows that
f(T) ≤f(S) +
k

i=1

e∈Ti Δe(S)
≤f(S) +
k

i=1

e∈Ti Δe(Si−1).
(17)
Thus, the stochastic partition greedy algorithm chooses a par-
tition Pi and picks an element e ∈Pi to maximize the marginal
value Δe(Si−1). Let Δei(Si−1) = maxe∈Pi Δe(Si−1). Then, it
is clear that
Δe(Si−1) ≤Δei(Si−1).
(18)
Since |Ti| ≤1 and f(S) = k
i=1 Δei(Si−1), (17) further
yields
f(T) ≤f(S) +
k

i=1
Δei(Si−1)
≤2f(S).
(19)
C. Reobservation Probability Modeling
In order to ensure the effectiveness of the selected edge pixels,
we model the probability of corresponding 3-D edge points
being reobserved in subsequent frames. The problem is modeled
by introducing a Bernoulli distribution B = {b0, . . . , bn−1},
where bi is a binary variable. bi = 1 means that the ith edge
point is reobserved in the subsequent frame, while bi = 0 means
that such edge point is absent in the subsequent frame. Then, we
rewrite the submodular function as
f(S) =
k

i=1
biρei(Si−1).
(20)
Algorithm 1: Efﬁcient Partition Selection.
It is clear that if the edge point of ei is reobserved, then
bi = 1 and it is able to provide valid marginal gain. Otherwise,
this marginal gain simply disappears, and we try to ﬁnd other
valid edge points to maximize the function. Since bi is a random
variable, the value of the function logdet is a stochastic quantity.
Hence, we should maximize the expectation of this function. Let
E(bi) = pi denote the probability of the edge point of ei being
reobserved, and we have
E(f(S)) =
k

i=1
piρei(Si−1).
(21)
We can judge whether or not an edge point is reobserved from
two aspects. First, as the camera moves, some edge points extend
beyond the camera’s ﬁeld of view, and it would not be reobserved
for the subsequent frame. In this study, we perform the visibility
check and ﬁlter out these invisible edge points based on the prior
camera trajectories. Second, the more distinctive the appearance
of the edge point, the more likely it is to be reobserved. Thus, we
model the probability of the edge point of ei being reobserved
as
pi =
1
1 + exp(a −mi)
(22)
where a is the high threshold in the Canny algorithm, and mi is
the gradient magnitude of the edge pixel ei.
In this way, the gradient magnitude of an edge pixel is larger
than the high threshold. It is thus more likely to be detected
by the Canny algorithm in the next frames, which is a desired
behavior. Finally, we provide the pseudocode of the complete
edge selection approach in Algorithm 1.
To enhance depth reliability in edge alignment, we integrate
depth quality into the edge selection process. Edge pixels with
depth values falling outside the sensor’s valid range are ﬁrst
excluded to eliminate measurements inﬂuenced by sensor limi-
tations. For the remaining pixels, directional depth averaging is
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 8 -->
ZHAO et al.: TOWARD ACCURATE, EFFICIENT, AND ROBUST RGB-D SLAM IN CHALLENGING ENVIRONMENTS
5727
applied along the estimated edge orientation within a 5-pixel
neighborhood. This approach leverages the observation that
depth values tend to be consistent along the tangent direction
of edges, whereas discontinuities are more likely to occur in the
normal direction.
VI. ADAPTIVE AND ROBUST MOTION ESTIMATION
Edge alignment in SLAM is vulnerable to outliers arising
from spurious correspondences and measurement noise [9],
necessitating robust optimization for reliable motion estimation.
Two mainstream approaches are commonly employed: random
sample consensus (RANSAC)-based methods [43] that itera-
tively sample minimal sets to identify inliers, and M-estimators
that incorporate robust kernel functions into continuous op-
timization [44]. For edge alignment tasks, M-estimators are
generally more effective as they directly accommodate uncertain
correspondences without requiring explicit edge matching.
Widely used robust kernels, including Huber, Cauchy,
Geman–McClure, and Welsch [45], differ in how they model
residual distributions. However, selecting an optimal kernel
remains a challenge, as kernel performance varies with environ-
mental structure, measurement noise, and optimization conﬁgu-
ration. To address this limitation, recent adaptive approaches
aim to dynamically suppress outliers. For instance, dynamic
covariance scaling [46] adapts weights in a manner similar
to the Geman–McClure kernel, Max-Mixture models [47] se-
lect among predeﬁned Gaussian components, and graduated
nonconvexity [48] gradually transitions from convex to non-
convex formulations via parameter scheduling. Despite their
improvements, these methods remain constrained to ﬁxed kernel
families, with parameter adaptation decoupled from the state
estimation process. As a result, they lack the ﬂexibility to adapt
across kernels in the face of diverse and evolving residual
distributions.
In this section, we explore the automatic adaptation of kernels
to the residual distribution online, enhancing the robustness of
motion estimation to outliers without relying on prior knowl-
edge. To achieve this, a generalized robust kernel function is in-
troducedasin[49],whichdynamicallytunessimilarlytopopular
kernels like Huber, Cauchy, Geman–McClure, and Welsch based
on the current residual distribution. We integrate the choice of
robust kernels with motion estimation into a uniﬁed optimization
problem and solve it through a relaxation of the problem.
A. General Robust Kernel Function
Barron [49] presents a single robust kernel function that
integrates many common kernels. Given the current residual r,
scale parameter c, and shape parameter α, the simplest form of
the general robust function is described as
ρ(r, α, c) = |α −2|
α
 r2/c2
|α −2| + 1
α/2
−1

.
(23)
As shown in Fig. 5, the parameter α controls the robustness
of the loss function. By adjusting the value of α, one can
generate various kernel functions such as L2 loss (α = 2),
Fig. 5.
General robust kernel ρ(r, α, c) (left) and the corresponding kernel
weights (right) for different values of its shape parameter α. Smaller α values
results in greater robustness to outliers.
pseudo-Huber loss (α = 1), Cauchy loss (α = 0), Geman–
McClure loss (α = −2), and Welsch (α = −∞). This ﬂexibility
greatly enhances the adaptability of motion estimation to differ-
ent residual distributions.
The choice of an appropriate value of α enables the op-
timization process to effectively handle outliers. In order to
automatically determine the best kernel shape, we treat α as
an additional unknown parameter to be optimized along with
motion parameter ξ while minimizing the generalized loss
ξ∗, α∗= arg min
ξ,α
N

i=1
ρ(ri(ξ), α, c).
(24)
However, solving the optimization problem naively remains
problematic. The process tends to weighting down all residu-
als to small values without affecting the motion parameter ξ.
This signiﬁcantly exacerbates the singularity of the Hessian
matrix, resulting in ill-conditioned optimization problems and
thus preventing convergence. Moreover, since it is not possible
to derive the loss function derivative with respect to α, the grid
search [50] is often used to ﬁnd a suboptimal α. However,
the grid search requires multiple iterations over all residuals,
resulting in high computational complexity that is unsuitable
for real-time constraints.
B. Family of Upper Bounds
Instead of directly minimizing the raw kernel function (24),
we try to construct a sequence of tractable surrogate functions,
which is the upper bound of the objective. Then, we iteratively
optimize these functions to approximate the solution to the
original problem. In order to ﬁnd the family of upper bounds,
we introduce the Black–Rangarajan duality theory [51] that
transforms the general robust kernel to the outlier process.
Proposition 2: Given the general robust kernel ρ(r, α, c),
deﬁne a function φ(m) = ρ(r), where m = r2/c2. When the
shape parameter satisﬁes α < 2 and the residual weight is
w := c2ρ′(r)/r, then there exists an analytical outlier process
function that is equivalent to the general robust kernel, namely
ρ(r, α, c) = 1
2wr2
c2 + Φ(w, α),
w ∈[0, 1].
(25)
Proof: In order to prove the equivalence, we need to derive
the appropriate function Φ(w, α). Now, let the parameter c and
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 9 -->
5728
IEEE TRANSACTIONS ON ROBOTICS, VOL. 41, 2025
α be constants; the outlier process function is rewritten as
f(m) = 1
2wm + Φ(w).
(26)
Differentiating the function with respect to w, and setting it to
0, we have
Φ′(w) = −1
2m.
(27)
To ﬁnd Φ(w), we need to integrate (27) with respect to w. To
do so, we redeﬁne w to simplify the derivation of the functional
form of Φ(w). Given the equation φ(m) = ρ(r), the weight w
is redeﬁned as
w = 2φ′(m) = c2
r ρ′(r).
(28)
After the redeﬁnition, (27) can be rewritten as
Φ′(2φ′(m)) = −1
2m.
(29)
Multiplying 2φ′′(m) on both sides of (29) and integrating with
respect to m, we have
	
2Φ′(2φ′(m))φ′′(m)dm = −
	
mφ′′(m)dm
(30)
Φ(2φ′(m)) = −mφ′(m) + φ(m).
(31)
As the function Φ(w) is expressed as a function of w, we
substitute m = φ′−1( 1
2w) as derived from (28). Consequently,
the outlier process function can be formulated as
Φ(w) = φ

φ′−1
1
2w

−1
2wφ′−1
1
2w

.
(32)
From the aforementioned derivation, we can establish an equiv-
alence between ρ(r) and f(m). Substituting (28) and (31) into
(25) yields
f(m) = mφ′(m) −mφ′(m) + φ(m)
= ρ(r).
(33)
Notice that w is deﬁned in terms of φ′(m), we derive its form as
φ′(m) = 1
2

m
|α −2| + 1
 α−2
2
.
(34)
Thus, when α < 2, the weight w varies between 0 and 1, namely
lim
m→0 2φ′(m) = 1
lim
m→∞2φ′(m) = 0.
(35)
From Proposition 2, the general robust kernel is equivalently
transformed into the outlier process function, which enables
outlier suppression through weight adjustments. Note that the
residual weights w are only affected by the parameter α when
given the residuals. This means that we can adaptively tune
weights based on the distribution of residuals. Moreover, the
function Φ(·) acts as a regularization term, penalizing weights
that are too small, thereby preventing optimization from trivial
solutions. Next, we scale the residual weight w to derive the
upper bound of the general robust kernel.
Proposition 3: Suppose that there exists an outlier process
function f(r, α, c) that is equivalent to the general robust kernel
ρ(r, α, c). When the scale parameter satisﬁes c ≥1 and the
residual weight is w := ρ′(r)/r, the outlier process function is
the upper bound of the general robust function, namely
ρ(r, α, c) ≤f(r, α, c)
(36)
where the equality holds true when c = 1.
Proof: Since the outlier process function is equivalent to the
general robust kernel function, as being proved in Proposition 2,
we treat the parameter α and c as constants. Following the previ-
ous speciﬁcations outlined, the function Φ(w) can be computed
as
Φ(w) = |α −2|
α


1 −α
2

w
α
α−2 + α
2 w −1

.
(37)
Thus, the second derivative of the function f(m) with respect
to w is given as
∂2f(m)/∂w2 = Φ′′(w)
= w
4−α
α−2 .
(38)
Since that the residual weight w is ∈[0, 1], it now follows that
∂2f(m)/∂w2 > 0.
(39)
As in the proof of Proposition 2, when w := c2ρ′(r)/r, we have
∂f(m)/∂w = 0.
(40)
From (40) and (39), we can determine that w := c2ρ′(r)/r is the
minimum value of the function f(m) with respect to w. Notice
that the residual weight here is deﬁned as w := ρ′(r)/r, which
differs from that in Proposition 2. When the scale parameter
satisﬁes c ≥1, it follows that
w ≤c2
r ρ′(r).
(41)
In the case, the value of the function f(m) is an upper bound
for the general robust function ρ(r), namely
ρ(r) ≤f(m).
(42)
It is obvious that when c = 1, the weights are equal, thus
preserving the equivalence between the general robust function
and outlier process function.
■
The upper bound f(r, α, c) offers a more straightforward
computational form compared to the general robust kernel
ρ(r, α, c). As α decreases, it gradually exhibits the same out-
lier suppression behavior as the original function. As seen in
Fig. 6, this bounding property arises from lifting the general
robust kernel through weight scaling. In particular, we argue
that this weight scaling enhances robust estimation performance,
manifesting itself in the following aspects.
1) Thescaledweightsapproachzeroinlargeresidualregions,
ensuring that the upper bound maintains the robustness to
outliers. Especially, this robustness increasingly resem-
bles that of general robust kernel as α decreases.
2) In small residual regions, the scaled weights are signiﬁ-
cantly lower than those of the original function, reducing
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 10 -->
ZHAO et al.: TOWARD ACCURATE, EFFICIENT, AND ROBUST RGB-D SLAM IN CHALLENGING ENVIRONMENTS
5729
Fig. 6.
Upper bounds f(r) (top) of general robust kernel ρ(r) and their associated weight functions (bottom) for different α values with c = 1.5. The function
f(r) is an lifted version of ρ(r) when α < 2. Except in regions with large residuals, f(r) exhibits the same suppression behavior as ρ(r), especially when α < 0.
This lifting characteristic mainly arises from the scaling of weights.
oversensitivity to minor perturbations and enhancing the
stability of the estimation process.
3) This scaled strategy avoids excessive focus on the inlier
region during estimation process, aiding in accurately
capturing the overall distribution of residuals.
4) We require c ≥1 primarily to prevent excessively sharp
function responses, which would exacerbate nonlinearity
and increase the likelihood of optimization falling into
local minima.
Overall, the family of upper bounds exhibits well-behaved
performances in terms of robustness, efﬁciency, and conver-
gence. These desirable attributes render them a favorable choice
for approximating general robust optimization (24), enabling
reliable estimation in the presence of outliers while mitigating
adverse effects.
C. Relaxed Optimization and Convergence
In order to solve the adaptive robust optimization prob-
lem (24), we approximate its optimal solution by minimiz-
ing the family of upper bounds, which is commonly referred
as the majorization–minimization (MM) method [52]. As its
suggests, majorization–minimization involves constructing an
upper bound surrogate function that majorizes the objective
function and then minimizing this surrogate to approximate the
optimal solution. In the case, we leverage the MM framework to
derive a new algorithm for adaptive robust estimation, namely
ξ∗, α∗= arg min
ξ,α
N

i=1
f(ri(ξ), α, c).
(43)
We treat the parameter α as a latent variable and dynamically
select its appropriate value during each iteration to facilitate
robust motion estimation. Instead of selecting the optimal value
of α by minimizing the upper bound function, we propose an
relaxed approach inspired by the MM framework to facilitate
α selection more efﬁciently. At the tth iteration, computing the
Algorithm 2: Relaxed Adaptive Robust Estimation.
current residual rt−1 under the motion parameter ξt−1, αt is
updated such that the criterion holds, namely
f(rt−1, αt) ≤ηρ(rt−1, αt−1) + (1 −η)f(rt−1, αt−1)
= f(rt−1, αt−1) −ηdt−1
(44)
where η ∈(0, 1) is the progress coefﬁcient. A smaller value of
the coefﬁcient allows for gradual exploratory progress, while a
larger value leads to greedy progress. dt−1 is the gap between
the upper bound and the general robust function in the previous
iteration, and dt−1 = f(rt−1, αt−1) −ρ(rt−1, αt−1).
After updating the parameter α, a new upper bound is actually
established, enabling us to shift the focus toward optimizing the
motion parameters based on (43). As described in Section VI-B,
the upper bound minimization can be formulated as a weighted
nonlinear least squares problem and solve it by Gauss–Newton
or Levenberg–Marquardt methods. The algorithm of adaptive
relaxed robust estimation is summarized in Algorithm 2.
Moreover, with the family of the bounds f and the update
rule for α, we assert that Algorithm 2 possesses convergence
guarantees. Speciﬁcally, the sequence of iterations generated by
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 11 -->
5730
IEEE TRANSACTIONS ON ROBOTICS, VOL. 41, 2025
our algorithm is guaranteed to converge to a stationary point of
the original general robust loss objective, provided that suitable
regularity conditions are satisﬁed. Next, we provide guarantees
on the algorithm’s approximation to the original objective func-
tion.
Proposition 4: Suppose that f is the family of upper bounds
for the general robust function ρ, and the update of α satisﬁes
the rule (44); then, the gap dt between the function f and ρ
converges to 0.
Proof: To prove the converges of dt, We ﬁrst deﬁne that
vt = f(rt, αt) −ηdt.
(45)
Summing over t = 1, . . . , T, we have
T

t=1
dt = 1
η
T

t=1
(f(rt, αt) −vt).
(46)
From (44), we have vt ≥f(rt, αt+1). Substituting it into (46),
we can obtain
T

t=1
dt ≤1
η
T

t=1
(f(rt, αt) −f(rt, αt+1))
≤1
η
T

t=1
(f(rt−1, αt) −f(rt, αt+1)).
(47)
After the telescopic sum, it yields
T

t=1
dt ≤1
η (f(r0, α1) −f(rT , αT +1)).
(48)
■
As described in Section VI-B, the function f has a lower
bound; thereby, we have T
t=1 dt ≤∞. In addition, dt has been
deﬁned as
dt = f(rt, αt) −ρ(rt, αt) ≥0.
(49)
From
dt ≥0
and
T
t=1 dt ≤∞,
we
can
deduce
that
limt→∞dt = 0.
Proposition 4 implies that Algorithm 2 stops after ﬁnite steps,
and the stationary points of the upper bound function precisely
approximate those of the original function. Then, we should en-
sure that the upper bound minimization converges to stationary
points.
Proposition 5: Suppose that the upper bound function f
satisﬁes the construction rule (36), and the update of α adheres
to the inequality constraint (44); then, the sequence of iterations
{ξt}t generated by the Algorithm 2 converges to a stationary
point of f.
Proof: As described in Section VI-B, the second derivative of
the upper bound function is bounded; thereby, it has a Lipschitz
gradient with constant L, and we have
f(rt, αt) ≤f(rt−1, αt) + ∇f(rt−1, αt)(rt −rt−1)
+ L
2 ||rt −rt−1||2.
(50)
We assume that the reduction of residuals follows that rt =
rt−1 −1
L∇f(rt−1, αt). Then, substituting it into (50) yields
f(rt, αt) ≤f(rt−1, αt) −1
2L||∇f(rt−1, αt)||2.
(51)
From the inequality (44), we have
dt−1 = f(rt−1, αt−1) −ρ(rt−1, αt−1)
≥f(rt−1, αt) −ρ(rt−1, αt−1)
≥f(rt, αt) −ρ(rt−1, αt−1) + 1
2L||∇f(rt−1, αt)||2.
(52)
■
As described in Proposition 4, we know that dt →0, which
implies that the function f precisely approximates the original
function ρ, and the function f is a lower bound. Thereby, we
have
||∇f(rt, αt+1)||2 ≤dt →0.
(53)
Hence, upper bound minimization would converge to a station-
ary point.
VII. EXPERIMENTS AND RESULTS
In this section, we conduct a series of experiments on the
standard RGB-D benchmark datasets. We ﬁrst compare our pro-
posed EdgeSLAM with SOTA RGB-D SLAM systems in terms
of accuracy, robustness, and efﬁciency. Then, we validate the
effectiveness of edge selection and adaptive robust estimation
in EdgeSLAM.
A. Experimental Setup
We evaluate the proposed method on three public datasets:
TUM RGBD [53], ICL-NUIM [54], and ETH3D [55], which
include both real-world and simulated environments. These
datasets provide synchronized RGB images and depth images,
along with ground truth camera trajectories. Also, they capture
the challenges like illumination changes, low-texture, and fast
motion, which are critical for assessing the stability of SLAM
tracking. All experiments are implemented on a desktop com-
puter with an Intel Core i7-12700 CPU, an NVIDIA GeForce
RTX 3060 GPU, and 16-GB RAM. Notably, all computations
for the proposed method are carried out exclusively on the CPU.
B. Accuracy Analysis of EdgeSLAM
We compare EdgeSLAM with several baseline methods in
terms of accuracy, robustness, and efﬁciency. These baseline
methods range from direct methods, feature-based methods, to
edge-based methods. All aforementioned SLAM systems are
evaluated on the same computing platform, except Canny-VO,
whose results are directly taken from the original paper. The
speciﬁc baseline methods selected are as follows.
1) ORB-SLAM3 [16] is a feature-based SLAM approach
including tracking, as well as local and global mapping.
It achieves high accuracy and robustness by using feature
matching to establish data associations.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 12 -->
ZHAO et al.: TOWARD ACCURATE, EFFICIENT, AND ROBUST RGB-D SLAM IN CHALLENGING ENVIRONMENTS
5731
TABLE I
COMPARISON OF THE RMSE ATE (M) ON THE TUM-RGBD DATASET
2) RTAB-Map [17] is an actively maintained appearance-
based SLAM system. It primarily relies on feature match-
ing for motion estimation and loop closure.
3) PLP-SLAM [24] is a visual SLAM system that integrates
point, line, and plane features on top of ORB-SLAM2. By
leveraging multifeature fusion, it enhances the robustness
of camera pose estimation, improving stability in complex
environments.
4) BADSLAM[55]isadirectRGB-DSLAMsystemthatruns
on GPU device, which presents a direct BA formulation
that integrates both geometric and photometric error for
tracking and mapping.
5) ElasticFusion [27] is a map-centric SLAM approach that
alsorequiresGPUdevices,whichcombinesgeometricand
photometric error for frame-to-model pose estimation.
6) RGBD-DSO [56] is a direct visual odometry method that
extends DSO from monocular to RGB-D input. By incor-
porating photometric calibration, it enhances the camera’s
adaptability to illumination change conditions.
7) Canny-VO [9] is an edge-based visual odometry method. It
adapts the ANNFs and ONNFs to estimate camera relative
motion.
8) RESLAM [8] is a popular open-source edge-based SLAM
system, where the camera motion is estimated based on
EDFs.
9) EdgeVO [12] is an edge-based visual odometry approach
from our previous work, offering efﬁcient and robust
camera tracking.
The overall accuracy of the mentioned algorithms is evaluated
on three datasets: TUM RGBD, ICL-NUIM, and ETH3D. The
root-mean-squarederror(RMSE)ofthetranslationalcomponent
of the absolute trajectory error (ATE) is taken as the performance
metric. A lower value indicates a higher positioning accuracy.
We run each algorithm ﬁve times and take the average RMSE
to prevent ﬂuctuation.
Table I provides a detailed comparison of the accuracy of
various SLAM algorithms on the TUM-RGBD dataset. The
proposed EdgeSLAM demonstrates high accuracy, frequently
achieving RMSE values close to those of the leading algorithm,
ORB-SLAM3, and PLP-SLAM. While ORB-SLAM3 generally
records the lowest RMSE due to its robust data association
capabilities,EdgeSLAM’sperformanceisoftennearlyasstrong,
showcasing its competitive accuracy. In most cases, EdgeSLAM
exhibitsasigniﬁcantadvantageintheaccuracymetricoverdirect
methods like RGBD-DSO, BAD SLAM, and ElasticFusion,
as well as edge-based methods such as EdgeVO, RESLAM,
and Canny-VO. Remarkably, our method exhibits more reli-
able accuracy compared to all baseline methods, even in low-
texture environments such as fr3_str_notex_far and fr3_cabinet
sequences, and in fast motion scenarios, e.g., fr1_desk and
fr1_desk2.
As shown in Table II, EdgeSLAM also demonstrates ex-
ceptional accuracy on the ICL-NUIM dataset and achieves a
lower error than all the baseline methods in half of cases. It
witnesses that our method achieves an accuracy comparable
to or surpassing that of direct methods, while exhibiting su-
perior consistency in illumination-changing scenarios, such as
sequence Living Room kt3. Notably, EdgeSLAM signiﬁcantly
outperforms feature-based methods, i.e., ORB-SLAM3 and
RTAB-Map, in most cases. This performance gap underscores
the efﬁcacy of our edge-based approach in capturing essential
scene geometry, even in challenging indoor settings with limited
distinctive points. Meanwhile, PLP-SLAM attains an accuracy
comparable to EdgeSLAM by fusing multiple feature types,
therebymitigatingtheconstraintinsufﬁciencyoftenencountered
by purely feature-based methods. Furthermore, EdgeSLAM’s
improved accuracy over RESLAM, another edge-based method,
highlights the effectiveness of our speciﬁc edge selection and
motion estimation method.
We compare the accuracy of various baseline methods on
the ETH3D dataset in Table III, from which BAD SLAM
performs the best overall. EdgeSLAM achieves comparable
accuracy with ORB-SLAM3 in most cases, slightly below
BAD SLAM but superior to the other methods. Speciﬁcally,
EdgeSLAM demonstrates remarkable resilience in illumination-
changing environments, as shown in sequence cables_2 and
cables_3. Despite employing photometric calibration, RGBD-
DSO struggles in these scenarios, often exhibiting reduced ac-
curacy or complete tracking failure. And in low-texture sce-
narios, EdgeSLAM maintains good accuracy, whereas other
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 13 -->
5732
IEEE TRANSACTIONS ON ROBOTICS, VOL. 41, 2025
TABLE II
COMPARISON OF THE RMSE ATE (M) ON THE ICL-NUIM DATASET
TABLE III
COMPARISON OF THE RMSE ATE ON THE ETH3D DATASET
approaches often fail or lose accuracy, as in sequence desk_3 and
repetitive.
In summary, the proposed EdgeSLAM exhibits consistently
high accuracy across different datasets. As shown in Fig. 1,
the average ATE RMSE of EdgeSLAM on these datasets is
0.017 m, which is at least 29.17% lower than other SOTA
SLAM methods, including the ORB-SLAM3 and BAD SLAM
that perform the best among eight baseline methods. It can
maintain a good accuracy even in challenging environments
that typically impede feature-based and direct approaches. In
addition, compared to our previous work, EdgeVO, EdgeSLAM
improves accuracy by 34.62%, demonstrating the effectiveness
of its robust estimation and loop closure strategies. Furthermore,
we show some example trajectories and sparse reconstruction of
sequences fr3_long_ofﬁce, fr3_cabinet, Living Room kt0, and
desk_3, as shown in the ﬁrst row of Fig. 7. It is clear that
the estimated trajectories are very close to the ground truth
trajectories, which implies that EdgeSLAM achieves a very high
positioning accuracy.
C. Efﬁciency Comparison of EdgeSLAM
We compare the computational efﬁciency of EdgeSLAM with
baseline methods on three datasets, as shown in Fig. 8. The
average per-frame processing time of eight methods is calculated
for each dataset, and the box plots depict the comparative results.
It is clear that EdgeSLAM exhibits the best computational
efﬁciency among all baseline methods and could achieve about
120 Hz reporting the result on all datasets. Notably, EdgeSLAM
not only maintains the lowest average processing times but
also shows remarkable stability, with minimal variance across
different scenarios. This indicates that EdgeSLAM is robust and
less affected by varying scenes.
By contrast, feature-based approaches show higher process-
ing times and greater variances, particularly RTAB-Map and
PLP-SLAM. This inconsistency stems from the ﬂuctuating com-
putational demands of feature extraction and matching in diverse
environments. EdgeSLAM avoids relying on these computation-
ally demanding and variable processes by leveraging edge pixels
as the primary visual features, resulting in more consistent and
efﬁcient results across different environments.
Direct methods demonstrate competitive results, particularly
BAD SLAM, which achieves lower processing times close to
EdgeSLAM in some scenarios. ElasticFusion, while less efﬁ-
cient, still shows reasonable performance in certain datasets.
However, both BAD SLAM and ElasticFusion rely heavily on
GPU acceleration to achieve these results. Despite RGBD-DSO
operates solely on the CPUs, it generally incurs higher process-
ing times than other direct methods. On the contrary, our method
surpasses their efﬁciency by only using CPUs, primarily due to
the utilization of sparse edge pixels. Furthermore, EdgeSLAM’s
superior efﬁciency over RESLAM, another edge-based method,
highlights the effectiveness of our edge selection processing.
As summarized in Fig. 1, EdgeSLAM achieves the lowest
average processing time of 7.9 ms among SLAM methods across
all datasets, with only a marginal 7.5% increase over our previ-
ous EdgeVO approach. BAD SLAM takes the second position
in terms of efﬁciency, with an average per-frame processing
time of 8.6 ms. In comparison, other SOTA methods exhibit
signiﬁcantly higher processing times, exceeding EdgeSLAM’s
consumed time by at least 50%. More speciﬁcally, Table IV pro-
vides a detailed analysis of the average processing time for key
operations within EdgeSLAM’s three main threads: tracking,
local mapping, and loop closure. These results demonstrate the
computational efﬁciency of each module within EdgeSLAM,
enabling real-time performance on resource-limited platforms
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 14 -->
ZHAO et al.: TOWARD ACCURATE, EFFICIENT, AND ROBUST RGB-D SLAM IN CHALLENGING ENVIRONMENTS
5733
Fig. 7.
First row from left to right shows the trajectories and maps for sequences: fr3_long_ofﬁce, fr3_cabinet, Living Room kt0, and desk_3. The second row
illustrates selected edge pixels by our efﬁcient partition selection module for sample frames, where selected edge pixels are evenly distributed across the image.
The third row shows the estimated values of α at the frame. We observe that the values of α gradually decrease over successive iterations, indicating an increasingly
robust estimation process where the inﬂuence of potential outliers is progressively diminished.
Fig. 8.
Comparison of processing time (ms) per frame on TUM RGBD, ICL-NUIM, and ETH3D datasets. EdgeSLAM achieves the shortest and most consistent
processing time among SLAM methods across all datasets exclusively on the CPU.
such as smartphones. This implies that EdgeSLAM can work in
real time on resource-limited platforms such as smartphones.
D. Robustness Exploration Against SOTA Methods
We evaluate the robustness of the baseline methods based on
their tracking failures. From Tables I–III, EdgeSLAM demon-
strates stable performance across all sequences, even in chal-
lenging scenarios such as low-texture and illumination changes.
EdgeSLAM consistently achieves stable results, a capability that
other methods often struggle to maintain. These results demon-
strate that EdgeSLAM achieves superior robustness against
other SOTA methods.
In detail, the tracking of feature-based methods often fails
or drifts signiﬁcantly in the sequence with poor texture such
as fr3_str _ notex_far, fr3_str _ notex_near, fr3_cabinet, Of-
ﬁce Room kt1, plant _ scene_1, repetitive, and sofa_1. This
is due to their challenges in extracting sufﬁcient and reliable
features for tracking. RTAB-Map tends to perform slightly better
than ORB-SLAM3 in such scenarios as ORB-SLAM3’s strict
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 15 -->
5734
IEEE TRANSACTIONS ON ROBOTICS, VOL. 41, 2025
TABLE IV
AVERAGE PROCESSING TIME (MS) OF EACH THREAD IN EDGESLAM
matching strategy limits the number of feature associations,
while PLP-SLAM achieves comparable performance through
integrating points and lines to enhance robustness in low-texture
and geometrically ambiguous environments.
For direct methods, including BAD SLAM and ElasticFusion,
illumination changes, such as sequence fr2_desk, Living Room
kt3, cables_3, and desk_3, negatively affect photometric-based
optimization. Thus, they sometimes result in a low accuracy in
these challenging scenarios. Moreover, ElasticFusion performs
poorly in low-texture environments due to the lack of signiﬁcant
intensity variations, making photometric error less effective
as a constraint. BAD SLAM mitigates these issues by using
photometric gradient measurements. However, its alternating
optimization of pose and structure can lead to convergence prob-
lems, as observed in sequence Living Room kt2 and Ofﬁce Room
kt0. In contrast, RGBD-DSO integrates photometric calibration
to mitigate illumination variations and selects evenly distributed
high-gradient pixels, enhancing pose constraints in low-texture
environments. These factors contribute to its improved robust-
ness.
Unlike feature-based methods, edge-based SLAM remains
effectiveinlow-textureenvironmentssuchasfr3_str_notex_far,
fr3_str _ notex_near, fr3_cabinet, Ofﬁce Room kt1, plant _
scene_1, repetitive, and sofa _1, where salient visual features are
insufﬁcient for robust tracking. In these scenarios, high-gradient
edge pixels persist along geometric boundaries, providing sufﬁ-
cient structural cues for tracking. This capability is exempliﬁed
by the fr3_ cabinet sequence, as shown in the second column of
Fig. 7, where edge information remains reliably exploitable.
Moreover, the optimization based on the distance ﬁeld
paradigm provides a larger convergence radius and is insensi-
tive to illumination changes. Consequently, edge-based SLAM
demonstrates superior robustness in these scenarios. Notably,
the proposed EdgeSLAM exhibits superior robustness compared
to RESLAM, especially in low-texture environments. This is
primarilyattributedtothatouredgeselectionandadaptiverobust
estimation techniques are more effective, which enhance track-
ing stability and mitigate the impact of unreliable measurements
or outliers on motion estimation.
As shown in Fig. 9, we calculated the success rate for each
method, deﬁned as the ratio of successfully tracked sequences
to the total number of sequences. EdgeSLAM achieves the
highest success rate of 97.06%, signiﬁcantly outperforming
Fig. 9.
Statistics of the success rates (left) and ATE RMSE (right) for var-
ious SLAM methods on TUM RGBD, ICL-NUIM, and ETH3D. EdgeSLAM
achieved the highest success rate at 97.06%, the lowest average localization error
at 0.017 m, and the smallest error variability.
other methods, including our previous work, EdgeVO. In detail,
the proposed method only failed to track in sequence man-
nequin_1, primarily due to the similarity between the cam-
era movement direction and the edge direction, which led to
estimation degradation. Furthermore, we calculated the mean
and standard deviation of the ATE RMSE for each method
across all sequences. EdgeSLAM achieves the lowest average
error of 0.017 m and the lowest standard deviation of 0.016 m,
demonstrating its ability to maintain consistent high accuracy in
challenging environments.
Furthermore, we evaluate the robustness of EdgeSLAM in dy-
namicenvironments,andtheATERMSEontheTUMRGBDse-
quence with moving persons is calculated. The results are com-
pared with DSLAM [57], DynaSLAM [58], DS-SLAM [59],
NID-SLAM [60], and 3DS-SLAM [61], which are visual SLAM
approaches speciﬁcally designed for dynamic scenes, as well as
static-scene methods such as BAD SLAM. We directly take the
reported ATE RMSE values from their respective publications
and presented in Table V. It is clear that EdgeSLAM successfully
operates on nearly all sequences, with only one failure case.
This failure is primarily due to signiﬁcant human movement
in the scene, which disrupts system initialization. Moreover,
EdgeSLAM achieves the lowest localization error in more than
half of the sequences, largely attributed to the adaptive robust
estimation effectively mitigating outlier inﬂuence caused by dy-
namic objects. To further illustrate this, we visualize the weight
distribution during the adaptive robust estimation process, using
sequences fr3_walking_xyz and fr3_walking_halfsphere as ex-
amples. As shown in Fig. 10, low-weighted edge pixels predom-
inantly appear in dynamic regions, reducing their impact on pose
estimation, while higher weighted edge pixels are concentrated
in static areas, playing a crucial role in accurate estimation.
E. Ablation Study
In this section, we perform ablation studies on the proposed
edge selection method and adaptive robust motion estimation.
As shown in Fig. 11, we ﬁrst evaluate their impact on the system
byindividuallydisablingthesetwomodules,resultinginfourex-
perimental conﬁgurations, i.e., EdgeSLAM-SR with both edge
selection and adaptive robust estimation, EdgeSLAM-S with
edge selection, EdgeSLAM-R with adaptive robust estimation,
and EdgeSLAM-N without these two modules. The experiments
are conducted on TUM RGBD datasets, where we report the
trajectory drift rate computed as drift = RMSE × 100%/length,
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 16 -->
ZHAO et al.: TOWARD ACCURATE, EFFICIENT, AND ROBUST RGB-D SLAM IN CHALLENGING ENVIRONMENTS
5735
TABLE V
COMPARISON OF RMSE ATE (M) OF DYNAMIC SCENES ON THE TUM RGBD DATASET
Fig. 10.
Adaptive robust weight in dynamic scenes. Each row shows four evenly spaced frames from fr3_walking_xyz (top) and fr3_walking_halfsphere (bottom)
sequence. Red highlights low-weighted outliers, predominantly on moving objects (e.g., humans), while blue indicates high-weighted inliers. By adaptively
downweighting these outliers, our method enhances pose estimation robustness in dynamic scenes.
Fig. 11.
Ablation studies of edge selection and adaptive robust estimation on
theTUMRGB-Ddataset.Weevaluatetheimpactofourproposedcomponentsby
comparing the drift rates and average per-frame processing time of EdgeSLAM
under four conﬁgurations: 1) without any enhancements (EdgeSLAM-N); 2)
withedgeselectiononly(EdgeSLAM-S);3)withadaptiverobustestimationonly
(EdgeSLAM-R); and 4) with both edge selection and adaptive robust estimation
enabled (EdgeSLAM-SR).
as well as the average per-frame processing times for each
conﬁguration.
From the perspective of drift, EdgeSLAM-SR exhibits the
lowest drift with a concentrated distribution, indicating its
superior accuracy and stability. EdgeSLAM-S and EdgeSLAM-
R also signiﬁcantly outperform the baseline EdgeSLAM-N,
demonstrating the effectiveness of each individual module in
enhancing accuracy. Notably, EdgeSLAM-R exhibits a slightly
more uniform and concentrated error distribution compared to
EdgeSLAM-S, suggesting that the adaptive robust estimation
module contributes to improved robustness and stability, espe-
cially in the presence of outliers and noisy measurements.
As for processing time, EdgeSLAM-R exhibits the longest
processing time, even surpassing EdgeSLAM-N, suggesting that
the robust estimation process introduces additional computa-
tional overhead. In contrast, EdgeSLAM-S has the shortest and
most consistent processing times, illustrating that reducing the
number of edge pixels substantially decreases computational
cost. Although EdgeSLAM-SR has a slightly higher processing
time than EdgeSLAM-S, it achieves a favorable balance between
accuracy and efﬁciency by integrating two modules.
In the following, we conduct a more detailed ablation study
of these two modules.
1) Ablation on Efﬁcient Partition Selection: To further val-
idate the effectiveness of the edge selection module in EdgeS-
LAM, we compare the proposed efﬁcient partition selec-
tion method with the baseline method, namely lazier greedy
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 17 -->
5736
IEEE TRANSACTIONS ON ROBOTICS, VOL. 41, 2025
Fig. 12.
Approximation and processing time of efﬁcient partition selection.
Using lazier greedy selection as a baseline, our method exhibits a tradeoff, sac-
riﬁcing a small degree of approximation accuracy in exchange for substantially
improved computational efﬁciency.
selection [41]. The latter models feature selection as a sub-
modular function optimization problem with cardinality con-
straints,whichcanbesolvedbyusingthelazier-than-lazygreedy
method [42]. As shown in Fig. 12, experiments are conducted
on the TUM RGBD datasets, where we computed the logdet
value and processing time for each frame across individual
sequences. A higher logdet value indicates that the selected edge
pixel subset better preserves the covariance matrix’s spectral
properties, closely approximating the original set’s constraints
on motion estimation.
As shown in the second row of Fig. 7, our method enables
the selected edge pixels to be evenly distributed throughout
the image, where these edge pixels are more distinctive and
informative, resulting in more stable tracking. Although the
logdet value of efﬁcient partition selection displayed on the
left part of Fig. 12 is relatively lower than the lazier greedy
selection, our method slightly sacriﬁces approximation quality,
but it signiﬁcantly improves computational efﬁciency by a large
margin compared to the lazier greedy selection (see from the
right part of Fig. 12). From the theoretical analysis in Sec-
tion V-B, we know that the computational complexity of our
method is only O(n). In contrast, the baseline method requires
atleasttwicethenumberoffunctionevaluationscomparedtoour
approach. Therefore, we can conclude that the proposed method
is more suitable for real-time SLAM, especially with abundant
observation data and strong spatial correlations.
We further assess the impact of the proposed reobservation
probability model by comparing trajectory drift across dif-
ferent conﬁgurations. Speciﬁcally, we evaluate the full edge
selection variant (EdgeSLAM-S) against a reduced version
(EdgeSLAM-S-), in which the reobservation module is disabled,
usingEdgeSLAM-Nasthebaseline.AssummarizedinTableVI,
our complete approach achieves a 23.1% reduction in mean drift
relative to the baseline and outperforms EdgeSLAM-S-, which
achieves a 14.3% improvement. These results highlight the
effectiveness of the reobservation module in enhancing tracking
robustness.
2) Ablation on Relaxed Adaptive Robust Estimation: Next,
we compare the proposed adaptive robust estimation method
against ﬁxed kernel including Huber, Cauchy, and Student
TABLE VI
EFFECT OF REOBSERVATION MODELING ON TRAJECTORY DRIFT (%)
Fig. 13.
Cumulative error curves of EdgeSLAM with different robust kernels
on the TUM RGB-D dataset. Each conﬁguration was conducted a total of 150
times on the dataset. The proposed relaxed adaptive robust estimator appears in
the upper left corner, indicating that it achieves more accurate motion estimation
by effectively suppressing outliers.
t-distribution [9], as well as the truncated general kernel [50]
and truncated least squares based on graduated nonconvexity
(GNC-TLS) [48]. We integrated these baseline methods into
EdgeSLAM and conducted experiments on the TUM RGBD
datasets. Each method was conducted ten times on each se-
quence, and the drift rates were computed. As demonstrated
in Fig. 13, we report the cumulative error plots to compare the
performance of the different methods. On the curve, each point
represents the number of runs (y-axis) that achieved a drift rate
lower than a given threshold (x-axis); the higher, the better.
In the plot, our proposed relaxed adaptive robust estimation
method consistently outperforms other methods across all drift
rates,asevidencedbyitscumulativeerrorcurvebeingpositioned
at the top-left corner. This indicates that for the same number
of runs, our method is capable of maintaining lower drift rates
compared to other approaches. In other words, given a speciﬁc
drift rate threshold, our method yields a higher number of
runs that satisfy the accuracy requirement, demonstrating its
superior robustness and adaptability in handling various levels
of noise and outliers. This superior performance is attributed
to the dynamic adjustment of the α parameter, which gradually
decreases over iterations. As shown in the third row of Fig. 7, the
decreasing values of α lead to more effective outlier rejection,
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 18 -->
ZHAO et al.: TOWARD ACCURATE, EFFICIENT, AND ROBUST RGB-D SLAM IN CHALLENGING ENVIRONMENTS
5737
Fig. 14.
Robustness comparison for BA on the BAL dataset. We evaluate the Trafalgar sequence under outlier rates from 20% to 80% using various robust
estimators. Our method consistently achieves the lowest translation errors (top) and rotation errors (bottom) across all outlier rates.
concentrating the estimation on the most consistent measure-
ments. This adaptive behavior enables the system to transition
smoothly from an initial and more inclusive estimation to a
highly robust ﬁnal estimate.
In the drift rate range of below 1%, the cumulative error
curves of the other methods exhibit high overlap, indicating
that when noise interference is minimal and the scenes are
relatively simple, they perform similarly well in controlling
trajectory drift. However, as the drift rate threshold increases,
thetruncatedgeneralkerneldemonstratesbetterrobustnesscom-
pared to GNC-TLS and ﬁxed kernels. Notably, when the drift
rate exceeds 2%, the robustness of ﬁxed kernels deteriorates
signiﬁcantly compared to GNC-TLS. This suggests that ﬁxed
kernel functions will easily struggle to adapt to complex noisy
environments.
To further evaluate the robustness of different robust ker-
nels, we integrate them into BA, a key optimization process
in SLAM and 3-D reconstruction. Experiments are conducted
on the Trafalgar sequence from the bundle adjustment (BAL)
dataset [62], comprising 257 camera views, 65 132 landmarks,
and 225 911 observations. Outliers are introduced by adding
noise in the range of [−3, 3] pixels to a speciﬁed proportion
{20%, 40%, 60%, 80%} of the observations. BA is optimized
using the Gauss–Newton method, initialized by perturbing the
dataset-provided rotations by [−10◦, 10◦] and translations by
[−5, 5]. The optimized camera poses obtained without pertur-
bations serve as the reference ground truth. Rotational and
translational errors are then computed for each experimental
setting to assess the impact of outliers. The results are presented
in Fig. 14; all baseline methods effectively suppress outliers,
achieving relatively stable pose estimation errors as the outlier
ratio increases. However, our adaptive robust estimation consis-
tentlyyields thelowest errors across all scenarios, demonstrating
its superior resilience to varying levels of outlier contamination.
This highlights its effectiveness in enhancing BA robustness and
underscores its potential as a generalizable approach for SLAM
and 3-D reconstruction.
VIII. CONCLUSION
In this article, we propose EdgeSLAM, a novel RGBD SLAM
method for accurate, efﬁcient, and robust pose estimation. This
is done by using the two proposed methods: efﬁcient edge
selection and adaptive robust estimation. First, the proposed
edge selection method signiﬁcantly reduces the number of edge
pixels required for motion estimation and results in great com-
putational efﬁciency improvement over existing other SLAM
methods without sacriﬁcing accuracy and robustness. Second,
the proposed adaptive robust motion estimation algorithm can
automatically select appropriate robust kernel functions based
on the residual distribution, achieving precise motion estimation
even with numerous outliers. These advancements greatly mit-
igate the inherent limitations of edge-based SLAM. Extensive
experimental evaluations on three public datasets demonstrate
that EdgeSLAM is an excellent edge-based SLAM system to
achieve a balanced performance in terms of accuracy, efﬁciency,
and robustness.
We believe that edge-based SLAM is a highly promising re-
searchdirection,suchashowtofurtherimprovetheaccuracyand
efﬁciency of edge alignment in highly dynamic environments.
In the future, we will further investigate these issues and explore
the integration of inertial measurement units to further enhance
the system’s performance.
REFERENCES
[1] T. Taketomi, H. Uchiyama, and S. Ikeda, “Visual SLAM algorithms: A
survey from 2010 to 2016,” IPSJ Trans. Comput. Vis. Appl., vol. 9, no. 1,
2017, Art. no. 16.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 19 -->
5738
IEEE TRANSACTIONS ON ROBOTICS, VOL. 41, 2025
[2] G. Younes, D. Asmar, E. Shammas, and J. Zelek, “Keyframe-based monoc-
ular SLAM: Design, survey, and future directions,” Robot. Auton. Syst.,
vol. 98, pp. 67–88, 2017.
[3] R. Mur-Artal and J. D. Tardós, “Orb-SLAM2: An open-source SLAM
system for monocular, stereo, and RGB-D cameras,” IEEE Trans. Robot.,
vol. 33, no. 5, pp. 1255–1262, Oct. 2017.
[4] J. Engel, V. Koltun, and D. Cremers, “Direct sparse odometry,” IEEE
Trans. Pattern Anal. Mach. Intell., vol. 40, no. 3, pp. 611–625, Mar. 2018.
[5] Z. Chen, W. Sheng, G. Yang, Z. Su, and B. Liang, “Comparison and
analysis of feature method and direct method in visual SLAM technology
for social robots,” in Proc. 13th World Congr. Intell. Control Autom., 2018,
pp. 413–417.
[6] Z. Zunjie and F. Xu, “Real-time indoor scene reconstruction with RGBD
and inertia input,” in Proc. IEEE Int. Conf. Multi. Expo (ICME), 2019,
pp. 7–12.
[7] M. Kuse and S. Shen, “Robust camera motion estimation using direct
edge alignment and sub-gradient method,” in Proc. IEEE Int. Conf. Robot.
Autom., 2016, pp. 573–579.
[8] F. Schenk and F. Fraundorfer, “ReSLAM: A real-time robust edge-based
SLAM system,” in Proc. Int. Conf. Robot. Autom., 2019, pp. 154–160.
[9] Y. Zhou, H. Li, and L. Kneip, “Canny-VO: Visual odometry with RGB-
D cameras based on geometric 3-D–2-D edge alignment,” IEEE Trans.
Robot., vol. 35, no. 1, pp. 184–199, Feb. 2019.
[10] J. J. Tarrio and S. Pedre, “Realtime edge-based visual odometry for a
monocular camera,” in Proc. IEEE Int. Conf. Comput. Vis., 2015, pp. 702–
710.
[11] F. Schenk and F. Fraundorfer, “Robust edge-based visual odometry using
machine-learned edges,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst.,
2017, pp. 1297–1304.
[12] H. Zhao, J. Shang, K. Liu, C. Chen, and F. Gu, “EdgeVO: An efﬁcient and
accurate edge-based visual odometry,” in Proc. IEEE Int. Conf. Robot.
Autom., 2023, pp. 10630–10636.
[13] G. Klein and D. Murray, “Parallel tracking and mapping for small ar
workspaces,” in Proc. 6th IEEE ACM Int. Symp. Mixed Augmented Reality,
2007, pp. 225–234.
[14] F. Endres, J. Hess, J. Sturm, D. Cremers, and W. Burgard, “3-D mapping
with an RGB-D camera,” IEEE Trans. Robot., vol. 30, no. 1, pp. 177–187,
Feb. 2014.
[15] S. Rusinkiewicz and M. Levoy, “Efﬁcient variants of the ICP algorithm,”
in Proc. 3rd Int. Conf. 3-D Digit. Imag. Model., 2001, pp. 145–152.
[16] C. Campos, R. Elvira, J. J. G. Rodrı´guez, J. M. Montiel, and J. D. Tardós,
“Orb-SLAM3: An accurate open-source library for visual, visual–inertial,
and multimap SLAM,” IEEE Trans. Robot., vol. 37, no. 6, pp. 1874–1890,
Dec. 2021.
[17] M. Labbé and F. Michaud, “RTAB-Map as an open-source LiDAR and
visual simultaneous localization and mapping library for large-scale and
long-term online operation,” J. Field Robot., vol. 36, no. 2, pp. 416–446,
2019.
[18] H. Zhou, D. Zou, L. Pei, R. Ying, P. Liu, and W. Yu, “StructSLAM: Visual
SLAM with building structure lines,” IEEE Trans. Veh. Technol., vol. 64,
no. 4, pp. 1364–1375, Apr. 2015.
[19] A. Pumarola, A. Vakhitov, A. Agudo, A. Sanfeliu, and F. Moreno-Noguer,
“Pl-SLAM: Real-time monocular visual SLAM with points and lines,” in
Proc. IEEE Int. Conf. Robot. Autom., 2017, pp. 4503–4508.
[20] R. G. Von Gioi, J. Jakubowicz, J.-M. Morel, and G. Randall, “LSD: A fast
line segment detector with a false detection control,” IEEE Trans. Pattern
Anal. Mach. Intell., vol. 32, no. 4, pp. 722–732, Apr. 2010.
[21] M. Kaess, “Simultaneous localization and mapping with inﬁnite planes,”
in Proc. IEEE Int. Conf. Robot. Autom., 2015, pp. 4605–4611.
[22] L. Ma, C. Kerl, J. Stückler, and D. Cremers, “CPA-SLAM: Consistent
plane-model alignment for direct RGB-D SLAM,” in Proc. IEEE Int. Conf.
Robot. Autom., 2016, pp. 1285–1291.
[23] A. Segal, D. Haehnel, and S. Thrun, “Generalized-ICP,” in Proc. Robot.:
Sci. Syst. Conf., 2009, vol. 2, pp. 435–442.
[24] F. Shu, J. Wang, A. Pagani, and D. Stricker, “Structure PLP-SLAM:
Efﬁcient sparse mapping and localization using point, line and plane for
monocular, RGB-D and stereo cameras,” in Proc. IEEE Int. Conf. Robot.
Autom., 2023, pp. 2105–2112.
[25] R. A. Newcombe, S. J. Lovegrove, and A. J. Davison, “DTAM: Dense
tracking and mapping in real-time,” in Proc. Int. Conf. Comput. Vis., 2011,
pp. 2320–2327.
[26] T. Whelan, M. Kaess, M. Fallon, H. Johannsson, J. Leonard, and J.
McDonald, “Kintinuous: Spatially extended kinectfusion,” MIT Comput.
Sci. Artif. Intell. Lab., Cambridge, MA, USA, Tech. Rep. MIT-CSAIL-
TR-2012-020, 2012.
[27] T. Whelan, S. Leutenegger, R. Salas-Moreno, B. Glocker, and A. Davison,
“ElasticFusion: Real-time dense SLAM and light source estimation,” Int.
J. Robot. Res. (IJRR), vol. 35, no. 14, pp. 1697–1716, 2016.
[28] A. Dai, M. Nießner, M. Zollhöfer, S. Izadi, and C. Theobalt, “Bundle-
Fusion: Real-time globally consistent 3D reconstruction using on-the-ﬂy
surface reintegration,” ACM Trans. Graph., vol. 36, no. 4, 2017, Art. no. 24.
[29] C. Kerl, J. Sturm, and D. Cremers, “Dense visual SLAM for RGB-D
cameras,”inProc.IEEE/RSJInt.Conf.Intell.RobotsSyst.,2013,pp.2100–
2106.
[30] A. Concha and J. Civera, “Rgbdtam: A cost-effective and accurate RGB-D
tracking and mapping system,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots
Syst., 2017, pp. 6756–6763.
[31] J. Engel, T. Schöps, and D. Cremers, “LSD-SLAM: Large-scale direct
monocular SLAM,” in Proc. Eur. Conf. Comput. Vis., 2014, pp. 834–849.
[32] J. Engel, J. Stückler, and D. Cremers, “Large-scale direct SLAM with
stereo cameras,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst., 2015,
pp. 1935–1942.
[33] J.-L. Hsu and H.-Y. Lin, “Sparse edge visual odometry using an RGB-D
camera,” in Proc. 11th Asian Control Conf., 2017, pp. 964–969.
[34] S. Maity, A. Saha, and B. Bhowmick, “Edge SLAM: Edge points based
monocular visual SLAM,” in Proc. IEEE Int. Conf. Comput. Vis. Work-
shops, 2017, pp. 2408–2417.
[35] J. J. Tarrio, C. Smitt, and S. Pedre, “Se-SLAM: Semi-dense structured
edge-based monocular SLAM,” 2019, arXiv:1909.03917.
[36] C. Kim, P. Kim, S. Lee, and H. J. Kim, “Edge-based robust RGB-D visual
odometry using 2-D edge divergence minimization,” in Proc. IEEE/RSJ
Int. Conf. Intell. Robots Syst., 2018, pp. 1–9.
[37] C. Kim, J. Kim, and H. J. Kim, “Edge-based visual odometry with stereo
cameras using multiple oriented quadtrees,” in Proc. IEEE/RSJ Int. Conf.
Intell. Robots Syst., 2020, pp. 5917–5924.
[38] P.F.FelzenszwalbandD.P.Huttenlocher,“Distancetransformsofsampled
functions,” Theory Comput., vol. 8, no. 1, pp. 415–428, 2012.
[39] I. Nurutdinova and A. Fitzgibbon, “Towards pointless structure from mo-
tion: 3D reconstruction and camera parameters from general 3D curves,”
in Proc. IEEE Int. Conf. Comput. Vis., 2015, pp. 2363–2371.
[40] J. Canny, “A computational approach to edge detection,” IEEE Trans.
Pattern Anal. Mach. Intell., vol. PAMI-8, no. 6, pp. 679–698, Nov. 1986.
[41] Y. Zhao and P. A. Vela, “Good feature matching: Toward accurate, ro-
bust vo/vSLAM with low latency,” IEEE Trans. Robot., vol. 36, no. 3,
pp. 657–675, Jun. 2020.
[42] B. Mirzasoleiman, A. Badanidiyuru, A. Karbasi, J. Vondrák, and A.
Krause, “Lazier than lazy greedy,” in Proc. AAAI Conf. Artif. Intell., 2015,
pp. 1812–1818.
[43] M. A. Fischler and R. C. Bolles, “Random sample consensus: A paradigm
for model ﬁtting with applications to image analysis and automated car-
tography,” Commun. ACM, vol. 24, no. 6, pp. 381–395, 1981.
[44] Z. Zhang, “Parameter estimation techniques: A tutorial with application
to conic ﬁtting,” Image Vis. Comput., vol. 15, no. 1, pp. 59–76, 1997.
[45] K. MacTavish and T. D. Barfoot, “At all costs: A comparison of robust
cost functions for camera correspondence outliers,” in Proc. 12th Conf.
Comput. Robot Vis., 2015, pp. 62–69.
[46] P. Agarwal, G. D. Tipaldi, L. Spinello, C. Stachniss, and W. Burgard,
“Robust map optimization using dynamic covariance scaling,” in Proc.
IEEE Int. Conf. Robot. Autom., 2013, pp. 62–69.
[47] E. Olson and P. Agarwal, “Inference on networks of mixtures for robust
robot mapping,” Int. J. Robot. Res., vol. 32, no. 7, pp. 826–840, 2013.
[48] H. Yang, P. Antonante, V. Tzoumas, and L. Carlone, “Graduated non-
convexity for robust spatial perception: From non-minimal solvers to
global outlier rejection,” IEEE Robot. Autom. Lett., vol. 5, no. 2,
pp. 1127–1134, Apr. 2020.
[49] J. T. Barron, “A general and adaptive robust loss function,” in Proc.
IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2019, pp. 4331–4339.
[50] N. Chebrolu, T. Läbe, O. Vysotska, J. Behley, and C. Stachniss, “Adaptive
robust kernels for non-linear least squares problems,” IEEE Robot. Autom.
Lett., vol. 6, no. 2, pp. 2240–2247, Apr. 2021.
[51] M. J. Black and A. Rangarajan, “On the uniﬁcation of line processes,
outlier rejection, and robust statistics with applications in early vision,”
Int. J. Comput. Vis., vol. 19, no. 1, pp. 57–91, 1996.
[52] S. N. Parizi, K. He, R. Aghajani, S. Sclaroff, and P. Felzenszwalb, “Gener-
alized majorization-minimization,” in Proc. Int. Conf. Mach. Learn., 2019,
pp. 5022–5031.
[53] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A bench-
mark for the evaluation of RGB-D SLAM systems,” in Proc. IEEE/RSJ
Int. Conf. Intell. Robots Syst., 2012, pp. 573–580.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 20 -->
ZHAO et al.: TOWARD ACCURATE, EFFICIENT, AND ROBUST RGB-D SLAM IN CHALLENGING ENVIRONMENTS
5739
[54] A. Handa, T. Whelan, J. McDonald, and A. J. Davison, “A benchmark for
RGB-D visual odometry, 3D reconstruction and SLAM,” in Proc. IEEE
Int. Conf. Robot. Autom., 2014, pp. 1524–1531.
[55] T. Schops, T. Sattler, and M. Pollefeys, “Bad SLAM: Bundle adjusted
direct RGB-D SLAM,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit., 2019, pp. 134–144.
[56] Z. Yuan, K. Cheng, J. Tang, and X. Yang, “RGB-D DSO: Direct sparse
odometry with RGB-D cameras for indoor scenes,” IEEE Trans. Multime-
dia, vol. 24, pp. 4092–4101, 2021.
[57] W. Dai, Y. Zhang, P. Li, Z. Fang, and S. Scherer, “RGB-D SLAM in
dynamic environments using point correlations,” IEEE Trans. Pattern
Anal. Mach. Intell., vol. 44, no. 1, pp. 373–389, Jan. 2022.
[58] B. Bescos, J. M. Fácil, J. Civera, and J. Neira, “DynaSLAM: Tracking,
mapping, and inpainting in dynamic scenes,” IEEE Robot. Autom. Lett.,
vol. 3, no. 4, pp. 4076–4083, Oct. 2018.
[59] C. Yu et al., “DS-SLAM: A semantic visual SLAM towards dynamic
environments,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst., 2018,
pp. 1168–1174.
[60] Z. Xu, J. Niu, Q. Li, T. Ren, and C. Chen, “NID-SLAM: Neural implicit
representation-based RGB-D SLAM in dynamic environments,” in Proc.
IEEE Int. Conf. Mult. Expo (ICME), 2024, pp. 1–6.
[61] G. S. Krishna, K. Supriya, and S. Baidya, “3DS-SLAM: A 3D object
detection based semantic SLAM towards dynamic indoor environments,”
2023, arXiv:2310.06385.
[62] S. Agarwal, N. Snavely, S. M. Seitz, and R. Szeliski, “Bundle adjustment
in the large,” in Proc. 11th Eur. Conf. Comput. Vis., 2010, pp. 29–42.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 02:55:45 UTC from IEEE Xplore.  Restrictions apply.
