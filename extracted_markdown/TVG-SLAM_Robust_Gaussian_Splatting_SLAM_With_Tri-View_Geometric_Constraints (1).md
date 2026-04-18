<!-- page 1 -->
1314
IEEE ROBOTICS AND AUTOMATION LETTERS, VOL. 11, NO. 2, FEBRUARY 2026
TVG-SLAM: Robust Gaussian Splatting SLAM With
Tri-View Geometric Constraints
Zhen Tan
, Xieyuanli Chen
, Member, IEEE, Lei Feng, Yangbing Ge
, Shuaifeng Zhi
, Member, IEEE,
Jiaxiong Liu, Graduate Student Member, IEEE, and Dewen Hu
, Senior Member, IEEE
Abstract—Recent advances in 3D Gaussian Splatting (3DGS)
have enabled RGB-only SLAM systems to achieve high-ﬁdelity
scene representation. However, the heavy reliance of existing sys-
tems on photometric rendering loss for camera tracking under-
mines their robustness, especially in unbounded outdoor environ-
ments with severe viewpoint and illumination changes. To address
these challenges, we propose TVG-SLAM, a robust RGB-only
3DGS SLAM system that leverages a novel tri-view geometry
paradigm to ensure consistent tracking and high-quality mapping.
We establish temporally consistent tri-view correspondences to
enable three core contributions. First, Hybrid Geometric Con-
straints combine geometric cues (trifocal and 3D alignment) with
photometric supervision, ensuring stable pose estimation when
illumination shifts cause rendering inconsistencies. Second, TUGI
estimates uncertainty from multi-view 3D geometric consistency—
capturing cross-view stability rather than pairwise 2D matching
conﬁdence—to guide principled Gaussian initialization. Third,
DART adaptively attenuates photometric trust when rendering
degrades, allowing geometric priors to govern optimization. Ex-
periments on multiple public outdoor datasets show that TVG-
SLAM outperforms prior RGB-only 3DGS-based SLAM systems.
Notably, in the most challenging dataset, our method reduces the
average ATE by 69.0% while achieving state-of-the-art rendering
quality.
Index Terms—3D Gaussiang splatting, multi-view geometry,
SLAM.
I. INTRODUCTION
S
IMULTANEOUS Localization and Mapping (SLAM) is a
cornerstone technology for robotics, autonomous driving,
and augmented reality [1]. The evolution of SLAM has seen a
paradigm shift from traditional methods [2], [3], [4], [5], [6], [7],
[8], [9], [10] to neural rendering-based approaches [11], [12],
[13], [14], [15], [16], [17], [18], [19]. Recently, 3D Gaussian
Splatting (3DGS) [20] has signiﬁcantly advanced the ﬁeld. By
representing scenes with explicit Gaussian primitives, 3DGS en-
ables photorealistic real-time rendering alongside high-ﬁdelity
Received 29 June 2025; accepted 17 November 2025. Date of publication
5 December 2025; date of current version 16 December 2025. This article
was recommended for publication by Associate Editor K. Skinner and Editor
J. Civera upon evaluation of the reviewers’ comments. This work was sup-
ported in part by the National Natural Science Foundation of China under
Grant U25B2069 and Grant 62403478, in part by the Young Elite Scientists
Sponsorship Program by CAST under Grant 2023QNRC001, and in part by
The science and technology innovation Program of Hunan Province under Grant
2025RC3110. (Corresponding author: Xieyuanli Chen.)
The authors are with the College of Intelligence Science and Technology,
National University of Defense Technology, Changsha 410073, China (e-mail:
xieyuanli.chen@nudt.edu.cn).
The code of our method is released at https://github.com/MagicTZ/TVG-
SLAM.
Digital Object Identiﬁer 10.1109/LRA.2025.3641103
Fig. 1.
Our system integrates tri-view geometric constraints to achieve both
high-ﬁdelity representation and a highly accurate camera trajectory that closely
aligns with the ground truth.
mapping, inspiring novel 3DGS-based SLAM frameworks [21],
[22], [23], [24], [25], [26].
Despite promising results, current 3DGS-based SLAM sys-
tems face a fundamental challenge: their camera tracking de-
pends heavily on photometric consistency between the cur-
rent image and rendered views [22], [24], [27]. This as-
sumption, inherited from early NeRF-based [28] SLAM meth-
ods like iMAP [11] and NICE-SLAM [12], is prone to
failure under real-world conditions involving rapid motion,
lighting changes, or texture-less areas. This fragility is es-
pecially pronounced for RGB-only systems in unbounded
outdoor environments, where dynamic illumination (e.g.,
shadows, clouds, varying sun angles) and large viewpoint
shifts critically hinder robustness. Additionally, existing map-
ping methods rely on heuristic Gaussian initialization, of-
ten leading to geometric inaccuracies and suboptimal scene
representations.
To address the abovementioned challenges, we propose TVG-
SLAM (as shown in Fig. 1), a novel system centered around
a tri-view geometric paradigm for reliable tracking and high-
ﬁdelity mapping. We establish consistent tri-view correspon-
dences across consecutive frames by bridging pairwise dense
matches, which serve as the foundation for our core geometric
reasoning modules.
Building on dense tri-view matching, our tracking module
introduces Hybrid Geometric Constraints that combine photo-
metric loss with trifocal-based 2D reprojection and 3D align-
ment. For mapping, we propose TUGI, a probabilistic initial-
ization strategy that uses multi-view geometric uncertainty to
guide Gaussian initialization. To further enhance stability, we
introduce DART, which dynamically downweighs photometric
cues when mapping falls behind, mitigating drift. This design
addresses a limitation of photometric-only tracking: under il-
lumination changes, even a correct pose produces rendering
2377-3766 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artiﬁcial intelligence and similar technologies.
Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:02 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 2 -->
TAN et al.: TVG-SLAM: ROBUST GAUSSIAN SPLATTING SLAM WITH TRI-VIEW GEOMETRIC CONSTRAINTS
1315
Fig. 2.
The TVG-SLAM pipeline. Our system processes incremental RGB images by ﬁrst building robust tri-view matches. In tracking, we utilize Hybrid
Geometric Constraints—jointly optimizing photometric, trifocal, and 3D alignment losses—to estimate the camera pose robustly. In mapping, keyframes are
integrated into the map using our Tri-view Uncertainty-guided Gaussian Initialization (TUGI) strategy, which leverages multi-view geometric consistency to
initialize new Gaussians for a high-ﬁdelity scene representation.
inconsistencies, causing the optimizer to incorrectly adjust the
pose to compensate. Our photometric-independent geometric
constraints provide supervision based solely on spatial corre-
spondence geometry. When rendering degrades due to mapping
lag or illumination shifts, DART down-weights photometric
terms, allowing geometric priors to dominate the optimization.
This preserves tracking stability precisely when the bright-
ness constancy assumption fails. Taken together, these com-
ponents form a cohesive SLAM system that tightly integrates
photometric and geometric constraints. TVG-SLAM outper-
forms prior RGB-only 3DGS-based methods in both track-
ing accuracy and rendering ﬁdelity across challenging outdoor
benchmarks.
The main contributions are listed as follows:
r We ﬁrst introduce the tri-view geometry paradigm into the
GS-based SLAM framework, enabling robust data associ-
ation and geometric reasoning across frames.
r We design Hybrid Geometric Constraints that jointly lever-
age photometric consistency, trifocal 2D reprojection, and
3D alignment losses, improving pose robustness under
large viewpoint and illumination changes.
r We propose TUGI, a probabilistic Gaussian initialization
strategy that encodes multi-view geometric uncertainty
into the Gaussian parameters, improving map quality and
rendering ﬁdelity.
r We develop DART, a dynamic weighting mechanism that
attenuates photometric loss supervision when the map
becomes stale, mitigating tracking drift in asynchronous
SLAM scenarios.
II. RELATED WORK
A. Pose Optimization in NeRF and 3DGS
Neural rendering methods such as NeRF [28] and 3DGS [20]
achieve high-quality scene reconstruction but typically rely on
known or externally estimated camera poses. To remove this
dependency, recent works have proposed jointly optimizing
poses and scene parameters. Early NeRF-based works like
NeRF– [29] and BARF [30] pioneered this joint optimization
for static scenes. Subsequent methods [31], [32], [33], [34], [35],
[36] further improved robustness for complex camera trajecto-
ries, but these approaches are often limited to ofﬂine settings
and may require sparse priors or global optimization. Other
methods, including CF-NeRF [37] and CF-3DGS [38], aim to
eliminate explicit pose estimation entirely by leveraging dense
correspondences or learned feature alignment. While promising,
these systems are still designed for static, ofﬂine reconstruction
and lack support for online tracking, incremental mapping, or
spatiotemporal consistency.
B. Neural SLAM and Neural Rendering SLAM
Recent works have explored neural SLAM systems that ex-
plicitly target robust long-range tracking and data associa-
tion.MASt3R-SLAM[39]integratestransformer-basedfeatures
into the SLAM pipeline, improving correspondence reliability
across wide baselines. VGGT-SLAM [40] and VGGT-Long [41]
further demonstrate strong performance in long-range odom-
etry and sequence-level trajectory estimation, leveraging Vi-
sion Transformer backbones to enhance robustness under large
viewpoint changes. These systems mainly emphasize accurate
pose estimation but lack explicit scene representations for high-
ﬁdelity rendering, limiting their applicability to downstream
tasks such as AR/VR or simulation.
In parallel, another line of research has integrated neural scene
representations into SLAM pipelines, aiming to jointly achieve
tracking and photorealistic mapping. Early systems [11], [12],
[15], [16], [19], [42], [43] incorporated neural radiance ﬁelds
(NeRF) [28], but the implicit nature of NeRF leads to slow
rendering and expensive optimization, limiting online applica-
tions. More recent works [21], [22], [23], [24], [25], [26], [44]
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:02 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 3 -->
1316
IEEE ROBOTICS AND AUTOMATION LETTERS, VOL. 11, NO. 2, FEBRUARY 2026
adopt 3D Gaussian Splatting (3DGS) [20], an explicit scene
representation enabling real-time, differentiable rendering and
fast updates. Although efﬁcient, 3DGS-based SLAM methods
often rely on photometric consistency for tracking, making them
fragile under illumination changes, large viewpoint shifts, and
low-texture regions in unbounded outdoor environments.
To overcome these challenges, we propose TVG-SLAM,
an RGB-only SLAM system built upon a tri-view geometry
paradigm with dense geometric constraints and uncertainty-
guided Gaussian initialization. This design enhances tracking
robustness while preserving the high-ﬁdelity, real-time render-
ing capabilities inherent to Gaussian Splatting, thereby bridging
the gap between Neural SLAM and Neural Rendering SLAM
paradigms.
III. METHOD
A. System Overview
As illustrated in Fig. 2, TVG-SLAM is an RGB-only SLAM
system designed for challenging outdoor environments. Its
pipeline consists of three tightly integrated components that
couple geometric reasoning with photometric rendering. First,
a dense matching module builds reliable tri-view correspon-
dences across frames. These are fed into a hybrid geomet-
ric tracking module that jointly optimizes photometric, 2D
reprojection, and 3D alignment losses. This module features
our DART mechanism to mitigate drift from mapping delays
by adaptively reducing reliance on photometric cues. Finally,
an uncertainty-guided mapping module incrementally recon-
structs the 3D Gaussian map, initializing new Gaussians with
uncertainty-aware priors derived from multi-view consistency.
B. Preliminaries: 3D Gaussian Representation
Our system uses 3DGS [20] for scene representation. Unlike
implicit representations like NeRF, 3DGS models a scene us-
ing a set of explicit, interpretable primitives: anisotropic 3D
Gaussians. Each Gaussian is deﬁned by a mean (position)
µ ∈R3, a covariance matrix Σ ∈R3×3, a color c ∈R3 (stored
as spherical harmonic coefﬁcients), and an opacity α ∈R. The
covariance matrix Σ, which describes the shape and orientation,
is decomposed into a scaling matrix S and a rotation matrix R
(Σ = RT ST SR) for efﬁcient optimization.
To render a 2D image, 3DGS employs an efﬁcient, differ-
entiable pipeline. For a given camera pose T, the 3D mean
µ is projected into the camera’s coordinate system. The 3D
covariance Σ is then projected into a 2D covariance Σ′ via an
afﬁne approximation. Finally, the color C(p) at any pixel p is
computed by alpha-blending all N sorted Gaussians along the
camera ray:
C(p) =
N

i=1
ciαi
i−1

j=1
(1 −αj),
(1)
where αi is the i-th Gaussian’s opacity. The entire process is
differentiable, including Gaussian parameters and the camera
pose T, enabling joint optimization via gradient descent.
C. Dense Tri-View Matching
Reliable data association is a prerequisite for the geometric
supervision used by our tracking and mapping modules. We
adopt DUST3R as a modular pairwise matcher to obtain dense
correspondences between Ik–It and Ik−1–Ik, denoted Mk,t and
Mk−1,k. We then bridge them at the intermediate keyframe Ik by
keeping only pairs that share the same pixel pk, yielding a tem-
porally consistent tri-view set Mk−1,k,t = {(pk−1, pk, pt)}.
This pixel-consistency naturally ﬁlters transient or inconsistent
matches. Each retained triplet provides redundant 3D estimates
that drive our tracking (trifocal L2D, 3D alignment L3D) and
mapping (uncertainty σ2
g) modules.
Each retained triplet provides redundant 3D estimates, which
are then used by the tracking module (Hybrid Geometric Con-
straints with trifocal and 3D alignment losses) and the mapping
module (uncertainty-guided Gaussian initialization via variance
σ2
g). Thus, while DUST3R supplies initial pairwise matches, our
contributions lie in enforcing tri-view consistency and leverag-
ing it for downstream geometric reasoning.
D. Hybrid Geometric Tracking
Given a set of reliable tri-view correspondences Mk−1,k,t,
we formulate a hybrid objective to estimate the 6-DoF pose
Tt ∈SE(3) of the current tracking frame It. Our core tri-view
paradigm enables the formulation of multiple geometric con-
straints. Speciﬁcally, we design a loss function that integrates
photometric consistency with two complementary geometric
terms derived from tri-view matches: a 2D constraint based on
the classical trifocal tensor and a direct 3D alignment constraint.
The total loss is:
Ltrack = λpLphoto + λ2DL2D + λ3DL3D,
(2)
where λp, λ2D, λ3D are the weights for the photometric, 2D
trifocal, and 3D alignment losses, respectively.
Photometric Loss: Lphoto measures the similarity between the
observed image It and the rendered view ˆIt from the 3D Gaus-
sian map using the estimated pose Tt. We use a combination of
L1 and SSIM losses:
Lphoto = (1 −γ)LL1(It, ˆIt) + γLSSIM(It, ˆIt),
(3)
where γ is a ﬁxed hyperparameter.
Trifocal Constraint Loss: L2D enforces a pure multi-view
geometric constraint based on the trifocal tensor [45], [46],
without requiring explicit 3D reconstruction. For each tri-view
correspondence (pk−1, pk, pt), where pk−1 and pk are matched
points in two keyframes and pt is the corresponding point in
the current frame, we ﬁrst compute the trifocal tensor T from
the relative camera poses. The trifocal tensor consists of three
3 × 3 matrices T = {T1, T2, T3}, each corresponding to one
component of the point pk−1 = (p1
k−1, p2
k−1, p3
k−1)⊤.
Using the epipolar line transfer formula, the corresponding
epipolar line lt ∈R3 in the third view is computed as:
lt =
3

i=1
pi
k−1Tipk,
(4)
We then deﬁne the loss as the sum of squared algebraic residuals,
which approximate the geometric distance between the point
pt = (px
t , py
t , pz
t ) and the epipolar line lt = (lx
t , ly
t , lz
t ):
L2D=

t
ρ

(px
t lz
t −pz
t lx
t )2+(py
t lz
t −pz
t ly
t )2+(px
t ly
t −py
t lx
t )2
,
(5)
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:02 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 4 -->
TAN et al.: TVG-SLAM: ROBUST GAUSSIAN SPLATTING SLAM WITH TRI-VIEW GEOMETRIC CONSTRAINTS
1317
where ρ(·) is a Huber loss function to reduce the inﬂuence of
outliers. In essence, this loss minimizes the geometric distance
between an observed point in the current frame and its corre-
sponding epipolar line, which is determined by the other two
views and their relative poses.
Trifocal constraints offer stronger geometric observability
than pairwise epipolar geometry. As analyzed in [47], they
remain effective under degenerate motions like collinear tra-
jectories, making them especially suitable for outdoor SLAM
with low parallax and straight-line motion.
This strategy offers a strong map-independent constraint for
robust tracking, especially in challenging regions where photo-
metric consistency is unreliable. To ensure robustness, we only
use matches that satisfy favorable geometric conditions (e.g.,
sufﬁcient parallax) to avoid degenerate conﬁgurations.
3D Alignment Loss: L3D provides a non-projective geometric
constraint directly in 3D space. Instead of relying on traditional
triangulation, we leverage pointmaps generated by [48]. For a
tri-view match (pk−1, pk, pt), we can obtain two correspond-
ing 3D points from the two pairwise matches: Pc from the
(Ik, It) pointmap (in the current frame’s coordinates) and Pp
from the (Ik−1, Ik) pointmap (in the previous frame’s coor-
dinates). Ideally, these points should align in 3D space after
the correct transformation. We ﬁrst estimate a relative scale
factor s and then compute the transformation Tcur→prev from
the current to the previous frame. The 3D alignment loss is
deﬁned as:
L3D =

(Pc,Pp)∈C3D
ρ

∥Tcur→prev(sPc) −Pp∥2
2

,
(6)
where C3D is the set of all valid 3D-3D correspondences. The
scale factor s is estimated using Procrustes alignment [49]
between the two point sets.
Dynamic Attenuation of Rendering Trust (DART): The rela-
tive weights in Eq. 2 determine the inﬂuence of different loss
terms on pose optimization. However, ﬁxed weights fail to adapt
to the dynamic nature of SLAM, especially under asynchronous
tracking and mapping. When the mapping thread lags—e.g.,
during aggressive motion—the rendered view used for Lphoto
may be based on an outdated map, degrading its reliability.
Over-reliance on this inaccurate photometric supervision can
lead to erroneous pose updates.
To address this, we propose DART, a dynamic re-weighting
strategy that modulates the inﬂuence of Lphoto based on the
freshness of the underlying 3D map. Speciﬁcally, we use the
number of frames processed since the last keyframe, denoted as
ΔNf, as a proxy for map staleness. A larger ΔNf implies the
rendered view is less reliable.
We model the photometric loss weight λp as a decreasing
sigmoid-like function of ΔNf, ensuring a smooth and continu-
ous attenuation:
λp = wmin + (wmax −wmin)σ(k(ΔNf −Nm)),
(7)
where σ(x) = 1/(1 + ex) is the sigmoid function, wmin and
wmax denote the minimum and maximum photometric weights,
Nm is the midpoint, and k controls the sharpness of the transi-
tion.
With DART, when the map is recently updated (small ΔNf),
λp remains high, leveraging accurate photometric supervision.
As the map becomes stale, λp smoothly decays, allowing the
system to rely more on our robust, map-independent geometric
constraints (L2D and L3D). This adaptive, trust-aware reweight-
ing mechanism enhances tracking robustness in rapidly chang-
ing environments.
E. Uncertainty-Guided Mapping
During mapping, our system incrementally builds a globally
consistent and geometrically accurate 3D Gaussian map using
newly selected keyframes. Existing methods often use heuristics
(e.g., based on photometric gradients or depth residuals) to
decide how to initialize new Gaussians. This approach underuti-
lizes the rich information available from tri-view geometry, lead-
ing to inaccurate or redundant Gaussians in poorly constrained
regions. We therefore propose TUGI, a principled Gaussian
initialization strategy guided by tri-view geometric variance.
1) Uncertainty Estimation From Tri-View Consistency: To
assess the geometric reliability of new candidate points, we
estimate their 3D uncertainty directly from tri-view matches.
Given a triplet of frames (Ik−1, Ik, Ik+1), dense pairwise cor-
respondences provide two or more independent 3D estimates of
the same scene point via pointmaps. For each tri-view correspon-
dence (pk−1, pk, pk+1), we retrieve its associated 3D positions
from the matched pointmaps: Pk←k−1 from the (Ik−1, Ik) pair,
Pk←k+1 from (Ik, Ik+1), and Pk←k+1→k−1 from (Ik+1, Ik)
reprojected into the same coordinate frame. These 3D points
are then transformed into a common reference frame (typically
Ik or Ik−1) using known relative poses.
We deﬁne the uncertainty score σ2
g as the isotropic variance
of these estimates:
σ2
g = 1
N
N

i=1
Pi −P
2
2 ,
(8)
where ¯Pisthecentroidofthevalid3DpositionsandN ≥2isthe
number of valid estimates. This score captures the multi-view
consistency of the point: smaller σ2
g indicates high agreement
and low geometric uncertainty.
It is worth noting that conﬁdence maps from DUST3R, while
useful for assessing the local reliability of 2D correspondences,
are fundamentally different from the uncertainty required here.
DUST3R’s scores are pairwise and view-speciﬁc, hence they
are not calibrated across multiple views and do not directly
characterize the 3D consistency of a point. In contrast, our σ2
g
formulation explicitly measures variance among independent
3D estimates from tri-view matches, providing a principled and
geometry-awareuncertaintycuethatcandirectlyguideGaussian
initialization.
2) TUGI.
Uncertainty-Guided
Gaussian
Initialization:
TUGI encodes the geometric reliability of each 3D point into
the initial state of its corresponding Gaussian primitive. For
a candidate point P with estimated tri-view variance σ2
g, we
initialize the Gaussian center µnew at P and compute its color
cnew as the mean of pixel intensities from all three views,
ensuring appearance robustness across viewpoints.
To reﬂect geometric conﬁdence, we scale the Gaussian covari-
ance proportionally to the estimated standard deviation (Snew ∝
	
σ2g), which allows uncertain points to begin with broader
support, providing greater ﬂexibility to the optimizer.
We further modulate the initial opacity as a decreasing func-
tion of variance to mitigate rendering artifacts from unreli-
able regions. Speciﬁcally, the opacity is deﬁned as αnew =
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:02 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 5 -->
1318
IEEE ROBOTICS AND AUTOMATION LETTERS, VOL. 11, NO. 2, FEBRUARY 2026
TABLE I
EVALUATION ON THE WAYMO DATASET, COMPARING TRACKING ACCURACY (ATE RMSE [M] ↓) AND MAPPING QUALITY (PSNR, SSIM, LPIPS)
TABLE II
EVALUATION ON THE SMALL CITY DATASET
sigmoid−1(a(1 −k
	
σ2g)), where a is a base opacity and k
controls the degree of attenuation. In this way, TUGI adaptively
downweights uncertain primitives during early optimization,
facilitating convergence toward a compact and accurate map.
IV. EXPERIMENT
Datasets: To comprehensively evaluate our system, we con-
duct experiments on three challenging outdoor datasets. The
Waymo Open Dataset [50] features long-range driving se-
quences with prolonged low-parallax motion and large texture-
less regions. The Small City [51] sequences are characterized
by a multitude of dynamic objects and signiﬁcant illumination
changes. Finally, the Cambridge Landmarks Dataset [52]
involves hand-held camera captures with extremely aggres-
sive motions and frequent lighting variations. Together, these
datasets systematically test the tracking robustness and mapping
quality under various real-world conditions.
Metrics: We evaluate our system in terms of both tracking
accuracy and mapping quality. For tracking performance, we use
the root-mean-square error (RMSE) of the absolute trajectory
error (ATE). For mapping quality, we employ three widely-used
metrics: peak signal-to-noise ratio (PSNR), structural similarity
index (SSIM), and learned perceptual image patch similarity
(LPIPS).
Baselines: We benchmark our method against state-of-the-
art neural rendering SLAM methods (RGB-only), includ-
ing the NeRF-based NICER-SLAM [19] and three leading
3DGS-based methods: Photo-SLAM [25], MonoGS [24], and
OpenGS-SLAM [26].
TABLE III
EVALUATION ON THE CAMBRIDGE LANDMARKS DATASET
Implementation Details: We use DUST3R [48] as our dense
matcher and initialize each frame’s pose using matches to the
previous keyframe. For keyframe selection, we follow common
criteria in [24]. We adopt the relative scale accumulation strategy
used in [26], allowing us to maintain metric coherence across tri-
view correspondences. The tracking optimization is performed
for 40 iterations per frame and 80 iterations for mapping. Adam
optimizer is used with learning rates of 0.001 for rotation and
0.002 for translation. The weights for the geometric losses in
Eq. (2) are empirically set to λ2D = 0.01 and λ3D = 0.01. For
the DART mechanism described in Eq. (7), we set the weight
bounds wmax = 1.0 and wmin = 0.1, with midpoint Nm = 5
and steepness factor k = 0.8 . All experiments are conducted on
a desktop computer equipped with an NVIDIA RTX 4090 GPU
and an Intel Core i9-13900K CPU.
A. Result & Analysis
We conducted a comprehensive evaluation of TVG-SLAM
on three challenging outdoor datasets, benchmarking it against
state-of-the-art GS SLAM methods. Quantitative results are
detailed in Tables I to III, with qualitative comparisons illus-
trated in Figs. 3 and 4. The results consistently demonstrate
that our method achieves signiﬁcant superiority in both tracking
accuracy and mapping quality.
Performance on the Waymo Dataset: This dataset presents a
severe test of SLAM robustness, characterized by long-range
driving scenarios with low parallax and unbounded environ-
ments featuring large sky areas. As shown in Table I, TVG-
SLAM achieves a substantial improvement in tracking accuracy.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:02 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 6 -->
TAN et al.: TVG-SLAM: ROBUST GAUSSIAN SPLATTING SLAM WITH TRI-VIEW GEOMETRIC CONSTRAINTS
1319
Fig. 3.
Rendering quality comparison on the Waymo, Small City, and Cambridge Landmarks datasets in unbounded outdoor scenes. Compared to prior methods,
our approach preserves ﬁner scene details and sharper structures, especially under large viewpoint changes, whereas previous methods often fail to preserve ﬁne
structures in rendered images.
TABLE IV
ABLATION STUDY OF OUR KEY COMPONENTS ON THE WAYMO DATASET
(SCENE: 153495). WE EVALUATE THE IMPACT ON TRACKING ACCURACY
(ATE) AND MAPPING QUALITY (PSNR).
The average ATE is reduced by approximately 28.2% compared
to OpenGS-SLAM, and shows an order-of-magnitude advantage
over MonoGS and Photo-SLAM. This accuracy boost is a direct
result of our robust hybrid geometric constraints in the tracking.
In low-parallax, long-straight-road scenarios, purely photomet-
ric methods are prone to signiﬁcant drift, whereas our L2D and
L3D losses provide stable geometric references that effectively
suppress this drift. As illustrated by the trajectory comparison
in Fig. 4, OpenGS-SLAM exhibits signiﬁcant drift, whereas our
trajectoryremainscloselyalignedwiththegroundtruth. Further-
more,thispreciseposeestimationprovidesasolidfoundationfor
high-quality mapping, enabling us to achieve an average PSNR
of 25.38, signiﬁcantly outperforming all competing methods.
TABLE V
EFFICIENCY ANALYSIS OF TVG-SLAM
Performance on Small City & Cambridge Landmarks: These
two datasets demand higher levels of system stability and re-
sponsiveness, characterized by numerous dynamic elements
and aggressive camera motion, respectively. Our advantages
become even more pronounced in these scenarios. As reported
in Tables II and III, TVG-SLAM reduces the average ATE by
65.7% on Small City and 69.0% on Cambridge Landmarks
compared to OpenGS-SLAM. This highlights the robustness
of our tracking framework, where the DART mechanism plays
a critical role. During aggressive motions, when mapping la-
tency increases rendering uncertainty, DART adaptively down-
weights the unreliable photometric loss, forcing the system
to rely more on stable geometric constraints and preventing
tracking failure. Our method also excels in mapping quality.
Compared to OpenGS-SLAM, our method improves the average
PSNR by 6.5% on Small City and 4.6% on Cambridge Land-
marks, respectively. The qualitative results in Fig. 3 visually
conﬁrm the superiority of our strategy. Compared to the blurry,
artifact-riddenrenderingsfrombaselinemethods,ourrenderings
preserve ﬁner details and clearer geometric structures. This
is attributable to our uncertainty-guided initialization, which
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:02 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 7 -->
1320
IEEE ROBOTICS AND AUTOMATION LETTERS, VOL. 11, NO. 2, FEBRUARY 2026
Fig. 4.
Trajectory comparison on challenging outdoor sequences. Black
dashed lines: ground truth; red: our method; blue: OpenGS-SLAM. Our method
maintains superior tracking accuracy while OpenGS-SLAM exhibits signiﬁcant
drift during rapid motion.
generates more physically plausible Gaussian primitives,
thereby enabling higher-ﬁdelity mapping on top of accurate
tracking.
TVG-SLAM outperforms existing methods in both low-
parallax driving and dynamic hand-held scenarios, demonstrat-
ing the effectiveness of our tri-view geometry and uncertainty-
guided mapping framework.
B. Ablation Study
This section conducts ablation experiments to validate the
effectiveness of our proposed key components. Results for a
representative Waymo scene are presented in Table IV.
Impact of Geometric Constraints: Removing the 2D tri-view
trifocal loss (w/o L2D) or the 3D alignment loss (w/o L3D) leads
to a notable increase in ATE, rising from 0.870 m to 1.193 m and
1.038 m, respectively. Furthermore, removing both geometric
constraints together (w/o TGC) results in the worst performance
(ATE = 1.269 m), highlighting their complementary roles.
These results conﬁrm that tri-view geometric supervision is
essential for accurate and stable pose tracking, especially under
challenging photometric conditions.
Impact of DART: As shown in Table IV, disabling DART (w/o
DART) degrades tracking performance, increasing ATE from
0.870 m to 1.053 m (↑21.0%). Fig. 5 further shows that DART
reduces both ATE and RPE by 29.3% and 55.3%, respectively,
with lower variances. These results validate that dynamically
attenuating photometric loss during mapping staleness enables
more reliance on robust geometric constraints, enhancing both
the accuracy and stability of pose estimation under fast motion
and scene changes. This validates the beneﬁt of DART in asyn-
chronous SLAM systems, where rendering lag can otherwise
compromise pose estimation.
Impact of TUGI: Among all single-component ablations,
removing TUGI leads to the largest degradation, with ATE
Fig. 5.
Ablation in DART on the Waymo dataset (scene: 158686).
increasing by 38.3% and PSNR decreasing by 0.72 dB. This
highlights that a well-initialized map is essential not only for
stable pose estimation but also for preserving rendering ﬁdelity,
as TUGI encodes geometric uncertainty into Gaussian shape and
opacity to enhance both accuracy and plausibility of the 3D map.
Efﬁciency Analysis: As shown in Table V, dense matching
with [48] accounts for most of the runtime. On our hard-
ware (RTX 4090 + i9-13900 K), the current system runs at
0.85 FPS. Our modular design allows the matcher to be re-
placed/accelerated without changing the geometric reasoning.
V. CONCLUSION
We presented TVG-SLAM, an RGB-only SLAM system
that addresses the limitations of photometric-based tracking in
3DGS. By leveraging a tri-view geometry paradigm, our system
introduces dense tri-view matching, Hybrid Geometric Track-
ing, uncertainty-guided Gaussian initialization (TUGI), and an
adaptive photometric weighting mechanism (DART). Experi-
ments on challenging outdoor datasets demonstrate that TVG-
SLAM achieves state-of-the-art performance in both tracking
and mapping, especially under large viewpoint changes and dy-
namic lighting. Future work will explore lightweight alternatives
to dense matching for real-time performance, incorporate loop
closure for large-scale consistency, and extend the system to
dynamic environments.
REFERENCES
[1] C. Cadena et al., “Past, present, and future of simultaneous localization and
mapping: Toward the robust-perceptual age,” IEEE Trans. Robot., vol. 32,
no. 6, pp. 1309–1332, Dec. 2016.
[2] R. A. Newcombe, S. J. Lovegrove, and A. J. Davison, “DTAM: Dense
trackingandmappinginreal-time,”inProc.IEEE/CVFIntl.Conf.Comput.
Vis., 2011, pp. 2320–2327.
[3] R. F. Salas-Moreno, R. A. Newcombe, H. Strasdat, P. H. Kelly, and A. J.
Davison, “SLAM : Simultaneous localisation and mapping at the level of
objects,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2013,
pp. 1352–1359.
[4] R. Mur-Artal and J. D. Tardós, “ORB-SLAM2: An open-source SLAM
system for monocular, stereo, and RGB-D cameras,” IEEE Trans. Robot.,
vol. 33, no. 5, pp. 1255–1262, Oct. 2017.
[5] C. Campos, R. Elvira, J. J. G. Rodrı´guez, J. M. Montiel, and J. D.
Tardós, “ORB-SLAM3: An accurate open-source library for visual,
visual–inertial, and multimap SLAM,” IEEE Trans. Robot., vol. 37, no. 6,
pp. 1874–1890, Dec. 2021.
[6] T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker, and A.
J. Davison, “ElasticFusion: Real-time dense SLAM and light source
estimation,” Int. J. Robot. Res., vol. 35, no. 14, pp. 1697–1716, 2016,
doi: 10.1177/0278364916669237.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:02 UTC from IEEE Xplore.  Restrictions apply.

<!-- page 8 -->
TAN et al.: TVG-SLAM: ROBUST GAUSSIAN SPLATTING SLAM WITH TRI-VIEW GEOMETRIC CONSTRAINTS
1321
[7] R. A. Newcombe et al., “KinectFusion: Real-time dense surface mapping
and tracking,” in Proc. 10th IEEE Int. Symp. Mixed Augmented Reality,
2011, pp. 127–136.
[8] M. Bloesch, J. Czarnowski, R. Clark, S. Leutenegger, and A. J. Davison,
“CodeSLAM—Learning a compact, optimisable representation for dense
visual SLAM,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.,
2018, pp. 2560–2568.
[9] T. Schops, T. Sattler, and M. Pollefeys, “BAD SLAM: Bundle adjusted
direct RGB-D SLAM,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit., 2019, pp. 134–144.
[10] Y. Ge, L. Zhang, Y. Wu, and D. Hu, “PIPO-SLAM: Lightweight visual-
inertial SLAM with preintegration merging theory and pose-only de-
scriptions of multiple view geometry,” IEEE Trans. Robot., vol. 40,
pp. 2046–2059, 2024.
[11] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, “iMAP: Implicit mapping
and positioning in real-time,” in Proc. IEEE/CVF Intl. Conf. Comput. Vis.,
2021, pp. 6209–6218.
[12] Z. Zhu et al., “NICE-SLAM: Neural implicit scalable encoding for
SLAM,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022,
pp. 12786–12796.
[13] P. Wang, L. Liu, Y. Liu, C. Theobalt, T. Komura, and W. Wang, “Neus:
Learning neural implicit surfaces by volume rendering for multi-view
reconstruction,” in Proc. Conf. Neural Inf. Process. Syst., 2021, pp. 27171–
27183.
[14] C.-M. Chung et al., “Orbeez-SLAM: A real-time monocular visual SLAM
with ORB features and NeRF-realized mapping,” in Proc. IEEE Int. Conf.
Robot. Automat., 2023, pp. 9400–9406.
[15] H.Wang,J.Wang,andL.Agapito,“Co-SLAM:Jointcoordinateandsparse
parametric encodings for neural real-time SLAM,” in Proc. IEEE/CVF
Conf. Comput. Vis. Pattern Recognit., 2023, pp. 13293–13302.
[16] Y. Zhang, F. Tosi, S. Mattoccia, and M. Poggi, “Go-SLAM: Global
optimization for consistent 3D instant reconstruction,” in Proc. IEEE/CVF
Int. Conf. Comput. Vis., 2023, pp. 3704–3714.
[17] M. Li, J. He, Y. Wang, and H. Wang, “End-to-end RGB-D SLAM with
multi-MLPs dense neural implicit representations,” IEEE Robot. Automat.
Lett., vol. 8, no. 11, pp. 7138–7145, Nov. 2023.
[18] W. Guo, B. Wang, and L. Chen, “Neuv-SLAM: Fast neural multiresolution
voxel optimization for RGBD dense SLAM,” IEEE Trans. Multimedia,
pp. 7546–7556, 2025.
[19] Z. Zhu et al., “Nicer-SLAM: Neural implicit scene encoding for RGB
SLAM,” in Proc. Int. Conf. 3D Vis., 2024, pp. 42–52.
[20] B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis, “3D Gaussian
splatting for real-time radiance ﬁeld rendering,” ACM Trans. Graph.,
vol. 42, no. 4, pp. 1–14, 2023.
[21] C. Yan et al., “GS-SLAM: Dense visual SLAM with 3D Gaussian splat-
ting,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024,
pp. 19595–19604.
[22] V. Yugay, Y. Li, T. Gevers, and M. R. Oswald, “Gaussian-SLAM: Photo-
realistic dense slam with Gaussian splatting,” 2023, arXiv:2312.10070.
[23] N. Keetha et al., “SplaTAM: Splat, track & map 3D Gaussians for dense
RGB-D SLAM,” in Proc. IEEE/CVF Conf. Comput. Visi. Pattern Recog-
nit., 2024, pp. 21 357–21 366.
[24] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, “Gaussian splatting
SLAM,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024,
pp. 18039–18048.
[25] H. Huang, L. Li, H. Cheng, and S.-K. Yeung, “Photo-SLAM: Real-time
simultaneouslocalizationandphotorealisticmappingformonocularstereo
and RGB-D cameras,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit., 2024, pp. 21 584–21 593.
[26] S. Yu, C. Cheng, Y. Zhou, X. Yang, and H. Wang, “RGB-only Gaussian
splatting slam for unbounded outdoor scenes,” 2025, arXiv:2502.15633.
[27] J. Engel, V. Koltun, and D. Cremers, “Direct sparse odometry,” IEEE
Trans. Pattern Anal. Mach. Intell., vol. 40, no. 3, pp. 611–625, Mar. 2018.
[28] B. Mildenhall, P.P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “NeRF: Representing scenes as neural radiance ﬁelds for view
synthesis,” Commun. ACM, vol. 65, no. 1, pp. 99–106, 2021.
[29] Z. Wang, S. Wu, W. Xie, M. Chen, and V. A. Prisacariu, “NeRF–:
Neural radiance ﬁelds without known camera parameters,” 2021,
arXiv:2102.07064.
[30] C.-H. Lin, W.-C. Ma, A. Torralba, and S. Lucey, “BaRF: Bundle-adjusting
neural radiance ﬁelds,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2021,
pp. 5742–5751.
[31] Y. Jeong, S. Ahn, C. Choy, A. Anandkumar, M. Cho, and J. Park, “Self-
calibrating neural radiance ﬁelds,” in Proc. IEEE/CVF Intl. Conf. Comput.
Vis., 2021, pp. 5846–5854.
[32] Y. Chen et al., “Local-to-global registration for bundle-adjusting neural
radiance ﬁelds,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.,
2023, pp. 8264–8273.
[33] W. Bian, Z. Wang, K. Li, J.-W. Bian, and V. A. Prisacariu, “Nope-NeRF:
Optimising neural radiance ﬁeld with no pose prior,” in Proc. IEEE/CVF
Conf. Comput. Vis. Pattern Recognit., 2023, pp. 4160–4169.
[34] Z. Tan, Z. Zhou, Y. Ge, Z. Wang, X. Chen, and D. Hu, “Td-NeRF: Novel
truncated depth prior for joint camera pose and neural radiance ﬁeld
optimization,” in Proc. IEEE/RSJ Intl. Conf. Intell. Robots Syst., 2024,
pp. 372–379.
[35] Z. Yuan, X. Chen, J. Wang, S. Li, and T. Zhang, “RA-NeRF:
Towards high-ﬁdelity and robust NeRF-based SLAM with monoc-
ular camera,” in Proc. IEEE Int. Conf. Robot. & Autom., 2024,
pp. 9172–9179.
[36] Y. Ran et al., “Ct-NeRF: Incremental optimizing neural radiance ﬁeld and
poses with complex trajectory,” IEEE Trans. Circuits Syst. Video Technol.,
vol. 35, pp. 10 110–10 121, 2025.
[37] Q. Yan et al., “Cf-NeRF: Camera parameter free neural radiance ﬁelds
with incremental learning,” in Proc. Conf. Advancements Artif. Intell.,
2024, vol. 38, no. 6, pp. 6440–6448.
[38] Y. Fu, S. Liu, A. Kulkarni, J. Kautz, A. A. Efros, and X. Wang, “Colmap-
free3DGaussiansplatting,”inProc.IEEE/CVFConf.Comput.Vis.Pattern
Recognit., 2024, pp. 20796–20805.
[39] R. Murai, E. Dexheimer, and A. J. Davison, “Mast3r-SLAM: Real-time
dense SLAM with 3D reconstruction priors,” in Proc. IEEE/CVF Conf.
Comput. Vis. Pattern Recognit., 2025, pp. 16695–16705.
[40] D. Maggio, H. Lim, and L. Carlone, “VGGT-SLAM: Dense RGB SLAM
optimized on the SL (4) manifold,” Adv. Neural Inf. Process. Syst., vol. 39,
2025.
[41] K. Deng, Z. Ti, J. Xu, J. Yang, and J. Xie, “VGGT-LONG: Chunk it, loop it,
align it–pushing VGGT’s limits on kilometer-scale long RGB sequences,”
2025, arXiv:2507.16443.
[42] X. Yang, H. Li, H. Zhai, Y. Ming, Y. Liu, and G. Zhang, “Vox-
fusion: Dense tracking and mapping with voxel-based neural implicit
representation,” in Proc. Int. Symp. Mixed Augmented Reality, 2022,
pp. 499–507.
[43] L. Liso, E. Sandström, V. Yugay, L. V. Gool, and M. R. Oswald, “Loopy-
SLAM:DenseneuralSLAMwithloopclosures,”inProc.IEEE/CVFConf.
Comput. Vis. Pattern Recognit., 2024, pp. 20363–20373.
[44] E. Sandström et al., “Splat-SLAM: Globally optimized RGB-only SLAM
with 3D Gaussians,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit., 2025, pp. 1680–1691.
[45] R. I. Hartley and A. Zisserman, Multiple View Geometry in Computer
Vision, 2nd ed. Cambridge, U.K.: Cambridge Univ. Press, 2004, ISBN:
0521540518.
[46] C.Sweeney,T.Sattler,T.Hollerer,M.Turk,andM.Pollefeys,“Optimizing
the viewing graph for structure-from-motion,” in Proc. IEEE/CVF Intl.
Conf. Comput. Vis., 2015, pp. 801–809.
[47] V. Indelman, S. Williams, J. Gallier, and F. Dellaert, “Incremental light
bundle adjustment,” in Proc. IEEE Int. Conf. Robot. & Automat., 2012,
pp. 5364–5371.
[48] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “Dust3R:
Geometric 3D vision made easy,” in Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit., 2024, pp. 20697–20709.
[49] S. Umeyama, “Least-squares estimation of transformation parameters
between two point patterns,” IEEE Trans. Pattern Analalysis Mach. Intell.,
vol. 13, no. 04, pp. 376–380, Apr. 1991.
[50] P. Sun et al., “Scalability in perception for autonomous driving: Waymo
open dataset,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.,
2020, pp. 2446–2454.
[51] B. Kerbl, A. Meuleman, G. Kopanas, M. Wimmer, A. Lanvin, and G.
Drettakis, “A hierarchical 3D Gaussian representation for real-time ren-
dering of very large datasets,” ACM Trans. Graph., vol. 43, no. 4, pp. 1–15,
2024.
[52] R. Ferens and Y. Keller, “Hyperpose: Camera pose localization using
attention hypernetworks,” 2023, arXiv:2303.02610.
Authorized licensed use limited to: Peng Cheng Laboratory. Downloaded on March 31,2026 at 12:22:02 UTC from IEEE Xplore.  Restrictions apply.
