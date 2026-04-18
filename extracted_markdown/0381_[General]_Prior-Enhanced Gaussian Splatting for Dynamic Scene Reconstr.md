<!-- page 1 -->
Prior-Enhanced Gaussian Splatting for Dynamic Scene Reconstruction
from Casual Video
MENG-LI SHIH, University of Washington,
YING-HUAN CHEN, National Yang Ming Chiao Tung University,
YU-LUN LIU, National Yang Ming Chiao Tung University,
BRIAN CURLESS, University of Washington,
Fig. 1. Dynamic free-view synthesis of reconstruction from monocular video.
We introduce a fully automatic pipeline for dynamic scene reconstruction
from casually captured monocular RGB videos. Rather than designing a new
scene representation, we enhance the priors that drive Dynamic Gaussian
Splatting. Video segmentation combined with epipolar-error maps yields
object-level masks that closely follow thin structures; these masks (i) guide
an object-depth loss that sharpens the consistent video depth, and (ii) support
skeleton-based sampling plus mask-guided re-identification to produce reli-
able, comprehensive 2-D tracks. Two additional objectives embed the refined
priors in the reconstruction stage: a virtual-view depth loss removes floaters,
and a scaffold-projection loss ties motion nodes to the tracks, preserving
Authors’ Contact Information: Meng-Li Shih, University of Washington, Seattle, Wash-
ington, , mlshih@cs.washington.edu; Ying-Huan Chen, National Yang Ming Chiao
Tung University, Hsinchu City, , yinghuan0419@gmail.com; Yu-Lun Liu, National Yang
Ming Chiao Tung University, Hsinchu City, , yulunliu@cs.nycu.edu.tw; Brian Curless,
University of Washington, Seattle, Washington, , curless@cs.washington.edu.
This work is licensed under a Creative Commons Attribution 4.0 International License.
SA Conference Papers ’25, Hong Kong, Hong Kong
© 2025 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2137-3/2025/12
https://doi.org/10.1145/3757377.3763910
fine geometry and coherent motion. The resulting system surpasses previ-
ous monocular dynamic scene reconstruction methods and delivers visibly
superior renderings. Project page: https://priorenhancedgaussian.github.io/
CCS Concepts: • Computing methodologies →Rendering; Point-based
models.
ACM Reference Format:
Meng-Li Shih, Ying-Huan Chen, Yu-Lun Liu, and Brian Curless. 2025. Prior-
Enhanced Gaussian Splatting for Dynamic Scene Reconstruction from Casual
Video. In SIGGRAPH Asia 2025 Conference Papers (SA Conference Papers ’25),
December 15–18, 2025, Hong Kong, Hong Kong. ACM, New York, NY, USA,
13 pages. https://doi.org/10.1145/3757377.3763910
1
Introduction
Reconstructing a faithful, time-varying 3-D scene from a casual
hand-held monocular RGB video is a long-standing goal in com-
puter graphics and vision. With the advent of 3-D Gaussian Splatting
(3D-GS) [Kerbl et al. 2023], a static scene can be distilled into tens
of thousands of anisotropic Gaussians and rasterized in real time.
Extending this idea to moving footage leads to dynamic Gaussian
SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong.
arXiv:2512.11356v1  [cs.CV]  12 Dec 2025

<!-- page 2 -->
2
•
Meng-Li Shih, Ying-Huan Chen, Yu-Lun Liu, and Brian Curless
Splatting (DGS): Gaussians translate and rotate through time, en-
abling free-view playback and immersive AR/VR experiences (see
Sec. 2 for detail).
Solving this ill-posed problem requires both a carefully designed
representation for motion and geometry, and a diverse set of 2-D
priors from foundation models such as consistent depth, dynamic
masks, and 2-D point trajectories to initialize and supervise the
reconstruction. Recent systems such as MoSca [Lei et al. 2024] and
Shape-of-Motion [Wang et al. 2024b] combine these aspects and
achieve striking realism.
Yet dynamic objects still reveal the limits of current pipelines:
thin structures blur or vanish, and complex motions lose coherence
(Fig. 2). Most prior work tackles these artifacts by introducing richer
scene representations or elaborate optimization schedules, while
paying little attention to the details of 2-D priors from foundation
models. We find that the quality of these priors has become one
of the major bottlenecks. Upgrading the depth, masks, and tracks
fed into a DGS system markedly improves reconstruction quality.
We therefore focus on enhancing prior quality. Our pipeline (1)
extracts salient dynamic-object masks that tightly follow thin
parts such as limbs, (2) refines the consistent video depth, recov-
ering detailed dynamic structures while preserving global geometry,
and (3) builds robust, comprehensive 2-D trajectories that sur-
vive occlusions and provide thorough coverage of moving surfaces
These higher-fidelity priors supply far stronger supervision. To let
the model exploit them fully, we further introduce two additional
loss terms, scaffold-projection loss and virtual-view depth loss),
that propagate the improved priors directly into the Gaussian cloud
and its underlying motion model.
We integrate these components and develop a fully automatic
pipeline for dynamic-scene reconstruction from monocular RGB
video. Experiments on the DyCheck dataset show quantitative gains,
while qualitative comparisons on DAVIS reveal significant improve-
ments in visual quality over previous monocular DGS systems.
(a) Shape of Motion
(b) MoSca
Fig. 2. Limitations of existing methods. Shape of Motion [Wang et al.
2024b] shows incoherent motion near boundaries and dis-occlusions of
the object (red and blue boxes in (a)), indicating insufficient regularization.
MoSca [Lei et al. 2024] improves coherence by constraining motion with a
set of scaffolds, yet it still struggles to represent thin objects with complex
motion, as highlighted by the red box in (b).
2
Related Work
Dynamic Novel View Synthesis. Recent years have seen rapid
progress in dynamic scene reconstruction for novel-view synthe-
sis. Many methods assume multi-view inputs with known calibra-
tion [Attal et al. 2023; Bae et al. 2024; Bansal et al. 2020; Cao and
Johnson 2023; Duan et al. 2024; Fridovich-Keil et al. 2023; Huang
et al. 2024; Kratimenos et al. 2023, 2024; Lee et al. 2024; Li et al.
2024a, 2020, 2021; Liang et al. 2025; Lin et al. 2023, 2024; Lombardi
et al. 2019; Luiten et al. 2023; Shaw et al. 2023; Sun et al. 2024; Wang
et al. 2022; Wu et al. 2023, 2024b; Xu et al. 2024; Yan et al. 2024; Zhu
et al. 2024], while others focus on the more practical but challenging
monocular setting [Bui et al. 2023; Das et al. 2023; Jeong et al. 2024;
Kwak et al. 2025; Lei et al. 2024; Li et al. 2020, 2022, 2023a; Liang et al.
2023; Liu et al. 2024; Miao et al. 2024; Park et al. 2024; Stearns et al.
2024; Wang et al. 2022, 2024b; Yoon et al. 2020; You et al. 2023; Zhang
et al. 2025a, 2023; Zhao et al. 2023, 2024]. Monocular reconstruction
is particularly challenging due to limited parallax, occlusion, and
motion blur, often leading to unstable or incomplete geometry.
Alongside this divide in input assumptions, approaches also differ
in their scene representations. NeRF-based models [Athar et al. 2022;
Du et al. 2021; Fang et al. 2022; Gao et al. 2021; Jiang et al. 2022; Li
et al. 2020, 2023b; Mildenhall et al. 2021; Shih et al. 2024; Song et al.
2023; Xian et al. 2021] rely on implicit volumetric fields, but are
slow to train and difficult to edit. Recent dynamic extensions of 3D
Gaussian Splatting [Kerbl et al. 2023; Lei et al. 2024; Liu et al. 2024;
Park et al. 2024; Stearns et al. 2024; Wang et al. 2024b; Yang et al.
2024a, 2023b] offer real-time rendering and more explicit control via
point-based structures, making them especially suitable for dynamic
content. Some recent methods also adopt a feedforward strategy, di-
rectly predicting novel views from monocular input without explicit
3D reconstruction [Liang et al. 2024; Wu et al. 2024a].
Similar to MoSca [Lei et al. 2024], SplineGS [Park et al. 2024], and
Shape of Motion [Wang et al. 2024b], our method adopts an explicit
deformation representation to model dynamic scenes. In contrast to
these approaches, we further enhance the quality of priors obtained
from foundation models and boost the quality of reconstruction.
Camera Pose Estimation from Monocular Video. Most prior dy-
namic reconstruction pipelines rely on COLMAP for camera pose
estimation. While effective in static environments, COLMAP often
fails in dynamic or narrow-baseline videos due to foreground mo-
tion and small baseline. Recent methods such as Robust CVD [Kopf
et al. 2020], CasualSAM [Zhang et al. 2022], and MegaSAM [Li
et al. 2024b] address this limitation by jointly optimizing depth
and pose from monocular videos with the prior from mono-depth.
Other approaches adopt feedforward networks that directly regress
poses from video frames [Feng et al. 2025; Wang et al. 2025; Zhang
et al. 2024], enabling faster inference but often requiring large-scale
training data and suffering in low-texture regions.
For dynamic novel view synthesis, approaches like RoDynRF [Liu
et al. 2023], SplineGS [Park et al. 2024], and MoSca [Lei et al. 2024]
propose COLMAP-free pipelines by using motion masks to isolate
static background regions, enabling geometry learning and camera
estimation without external SfM tools.
SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong.

<!-- page 3 -->
Prior-Enhanced Gaussian Splatting for Dynamic Scene Reconstruction from Casual Video
•
3
Fig. 3. Method Overview. Our three-stage approach is: (I) Initialization, (II) Lifting to 3-D, and (III) Dynamic scene reconstruction. (I) Initialization(Sec. 3.4).
(a) Video segmentation 𝑆(𝑙)
𝑡
[Kirillov et al. 2023; Ravi et al. 2024a; Zrporz 2024] is combined with epipolar-error masks 𝐸𝑇
𝑡=1 derived from optical flow 𝐹𝑡→𝑡′ [Teed
and Deng 2020], yielding dynamic-object masks 𝑀(𝑜)
𝑡
that cover each dynamic object in its entirety. (b) Using mask 𝑀(𝑜)
𝑡
and mono-depths ˜𝐷𝑇
𝑡=1 [Wang et al.
2024a], we apply an object-depth loss that sharpens Mega-SAM’s consistent depths 𝐷𝑇
𝑡=1 [Li et al. 2024b]. (c) From images 𝐼𝑇
𝑡=1 and masks 𝑀(𝑜)
𝑡
we identify
and sample tracks 𝑢(𝑜)
𝑡
along thin structures; combined with uniformly sampled tracks from 𝐸𝑇
𝑡=1, they fully covering each moving object. Mask-guided
re-identification, using the dynamic masks 𝑀(𝑜)
𝑡
, restores tracks lost to occlusion and further improves reconstruction quality. (II) Lifting to 3-D(Sec. 3.5).
Following [Lei et al. 2024], 2-D pixels and tracks are promoted to 3-D Gaussians and motion-scaffold nodes 𝑣(𝑚). A space–time regularizer then produces a
motion-coherent 3-D initialization. (III) Dynamic scene reconstruction(Sec. 3.6). For a training view Π𝑡we render ˆ𝐼𝑡, ˆ𝐷𝑡, and ˆ𝐹𝑡→𝑡1 and supervise them
with 𝐿rgb, 𝐿depth, and 𝐿gaussian
track
. Geometry regularizer 𝐿arap, 𝐿vel, and 𝐿acc act on the scaffold nodes 𝑣(𝑚). Two extra terms, 𝐿scaffold
track
and 𝐿virtual
depth , tether 𝑣(𝑚) to
accurately preserve object structure in motion and remove floaters, respectively. The final reconstruction result renders photorealistic views at any time and
viewpoint.
Geometry and motion prior. Recent advances in vision foundation
models have led to high-quality predictions across a variety of image-
based tasks. This progress has made them increasingly popular in
dynamic reconstruction pipelines, where monocular input suffers
from ambiguity in geometry and motion. Depth models [Hu et al.
2024; Piccinelli et al. 2024; Wang et al. 2024a; Yang et al. 2024b]
provide geometric cues. MoGe [Wang et al. 2024a] predicts affine-
invariant 3D point maps, and MegaSAM [Li et al. 2024b] produces
consistent video depth and pose from DepthAnything [Yang et al.
2024b]; Point tracking methods [Doersch et al. 2022, 2024, 2023;
Harley et al. 2022; Karaev et al. 2023; Xiao et al. 2024; Zhang et al.
2025b] estimate dense point trajectories across time. [Doersch et al.
2024; Karaev et al. 2023; Xiao et al. 2024] can track long-range pixels’
trajectory; and segmentation models [Huang et al. 2025; Kirillov et al.
2023; Ravi et al. 2024b; Yang et al. 2023a] produce masks that help
isolate foreground motion and preserve fine structural detail. Recent
work train learnable modules to extract dynamic objects [Goli et al.
2024; Huang et al. 2025; Karazija et al. 2024]. In contrast, we directly
combine video segmentation with geometric heuristics for a simpler
yet effective solution.
3
Method
We first review Motion Scaffold (MoSca) [Lei et al. 2024] in Sec. 3.1.
Sec. 3.3 gives a high-level outline of our pipeline, and Secs. 3.4–3.6
describe the three stages in detail.
3.1
Preliminary: Motion Scaffold
Motion Scaffold (MoSca) [Lei et al. 2024] represents a dynamic scene
with (i) a collection of dynamic 3-D Gaussians that translate and
rotate over time to represent geometry and appearance, and (ii)
a sparse set of 3-D motion scaffold nodes 𝑣(𝑚) ∈V whose time-
varying rigid transforms 𝑄(𝑚)
𝑡
= [R(𝑚)
𝑡
, t(𝑚)
𝑡
] encode the motion.
SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong.

<!-- page 4 -->
4
•
Meng-Li Shih, Ying-Huan Chen, Yu-Lun Liu, and Brian Curless
These nodes𝑣(𝑚) are connected as a graph G = (V, E) by 𝐾-nearest-
neighbor links; dual-quaternion blending over G yields a smooth
deformation field that drives every dynamic Gaussian.
3.2
Problem Statement
We tackle dynamic view synthesis from a single handheld RGB
video of everyday scenes. While prior work concentrates on ex-
pressive scene representations, we show that higher-quality priors
obviously boosts final reconstruction quality.
3.3
Method Overview
Figure 3 sketches our three-stage pipeline. Given𝑇input frames 𝐼𝑇
𝑡=1,
we reconstruct a scene of dynamic Gaussians whose rotation and
translation are governed by a motion-scaffold graph G = (V, E) in
three stages, (1) Initialization (Sec. 3.4), (2) Lifting to 3-D (Sec. 3.5),
and (3) Dynamic-scene reconstruction (Sec. 3.6).
3.4
Initialization
In this stage (see Fig. 3(a)), we prepare the following ingredients that
enables the subsequent 3-D/4-D reconstruction: (1) video segmen-
tation {𝑆(𝑙)
𝑡
}𝑇,𝐿
𝑡=1,𝑙=1 [Chan et al. 2022; Ravi et al. 2024b; Zrporz 2024],
where 𝐿is number of segments, (2) optical flow 𝐹𝑡→𝑡′ [Teed and
Deng 2020], (3) epipolar-error masks (EPI error masks) 𝐸𝑇
𝑡=1 [Liu et al.
2023], (4) dynamic-object masks {𝑀(𝑜)
𝑡
}𝑇,𝑂
𝑡=1,𝑜=1, where 𝑂is number
of dynamic objects, (5) single-image (mono) depth maps ˜𝐷𝑇
𝑡=1 [Wang
et al. 2024a], (6) camera poses Π𝑇
𝑡=1, (7) consistent video depth maps
𝐷𝑇
𝑡=1 [Li et al. 2024b], and (8) 2-D point tracks {𝑢(𝑛)
𝑡
}𝑇,𝑁
𝑡=1,𝑛=1, where
𝑁is number of tracks.
EPI error masks 𝐸𝑇
𝑡=1 reveal dynamic surfaces but cover only the
moving parts. We enlarge these regions with video-segmentation
cues 𝑆(𝑙)
𝑡
to obtain robust object-level masks 𝑀(𝑜)
𝑡
that cover each
dynamic object in its entirety. The following subsections detail how
these masks are constructed and how they improve both the video
depths 𝐷𝑇
𝑡=1 and the 2-D point tracks 𝑢(𝑛)
𝑡
on the dynamic objects.
Dynamic Object Mask Selection. To identify the salient dynamic
objects we compute the intersection between 𝑆(𝑙)
𝑡
and 𝐸𝑇
𝑡=1 (Fig. 4). A
segment is kept if it covers at least 𝜏salient = 0.05 of the total moving
surface:
| 𝑆(𝑙) ∩𝐸𝑇
𝑡=1|
|𝐸𝑇
𝑡=1|
≥𝜏salient.
Static segments occasionally pass this test because of appearance
change (e.g. moving shadows); we discard those whose own motion
area is small (𝜏appearance = 0.2):
| 𝑆(𝑙) ∩𝐸𝑇
𝑡=1|
|𝑆(𝑙) |
≥𝜏appearance.
The filtered results become the per-frame dynamic object masks
{𝑀(𝑘)
𝑡
}.
Depth Refinement. Although Mega-SAM [Li et al. 2024b] provides
temporally consistent video depths 𝐷𝑇
𝑡=1, its estimates are over-
smoothed on thin, fast-moving parts of dynamic objects (Fig. 5).
(a) Image
(b) EPI error mask
(c) Video Segmentation
(d) Car Segment
(e) Car Segment & EPI
Error Mask
(f) Dynamic mask pass
1st test
(g) Road Segment
(h) Road Segment & EPI
Error Mask
(i) Dynamic mask pass
1st & 2nd test
Fig. 4. Dynamic Object Mask Selection. We intersect the EPI error mask
𝐸𝑇
𝑡=1 (b) with each video segment 𝑆(𝑙) (c). The intersections for the car and
road segments appear in (e) & (h). Applying the two-pass test from Sec. 3.4
removes road shadows as outliers, leaving only the car as the dynamic-
object mask 𝑀(𝑜)
𝑡
(i).
To restore the fine detail, we (1) replace Mega-SAM’s initial mono-
depths ˜𝐷𝑇
𝑡=1 [Yang et al. 2024b] with [Wang et al. 2024a] and (2) add
an object-depth loss 𝐿object
depth to its consistency optimization.
Specifically, for each dynamic-object mask 𝑀(𝑜)
𝑡
we (i) crop the
mono-depth to obtain an object-only map ˜𝐷(𝑜)
𝑡
= 𝑀(𝑜)
𝑡
⊙˜𝐷𝑡; (ii)
align this map to the current video depth 𝐷𝑡via a mask-restricted
scale–shift fit [Wang et al. 2024a] 𝛼(𝑜)
𝑡
˜𝐷(𝑜)
𝑡
+𝛽(𝑜)
𝑡
; and (iii) add 𝐿object
depth
to Mega-SAM’s consistency optimization stage
𝐿object
depth =
1
𝑇|Ω|
𝑂
∑︁
𝑜=1
𝑇∑︁
𝑡=0
∑︁
𝑝∈Ω
𝑀(𝑜)
𝑡
(𝑝)
𝐷𝑡(𝑝) − 𝛼(𝑜)
𝑡
˜𝐷𝑡(𝑝) + 𝛽(𝑜)
𝑡
,
where Ω denotes domain of image 𝐼.
Because the loss is confined to object pixels, it sharpens thin struc-
tures while leaving the global scale and overall temporal consistency
of 𝐷𝑇
𝑡=1 intact. As a result, fine details are enriched and thin, moving
objects are reconstructed more faithfully (see Fig.5 and Fig.17).
Mask-guided point tracker. Uniformly sampling tracks from 𝐸𝑇
𝑡=1 [Lei
et al. 2024] biases toward large, rigid parts (e.g. torsos) and under-
samples thin, high-motion regions (e.g. limbs). We therefore adopt
skeleton-sample: for each object, we extract its 2-D medial-axis skele-
ton from the object mask 𝑀(𝑜)
𝑡
, dilate it by 5 pixels, weight pixels
by inverse distance to the mask boundary, and draw an additional
1
6 of tracks from this distribution. During trajectory estimation,
frames are resized so that the longer side is 512 pixels, ensuring
all trajectory-related heuristics operate at a consistent resolution.
Compared with uniform sampling, this strategy yields noticeably
better quality in thin regions (see Fig.6 and Fig. 13).
2-D trackers often lose a point when they re-emerge from occlu-
sion, leaving the track labelled occluded and creating holes in the
SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong.

<!-- page 5 -->
Prior-Enhanced Gaussian Splatting for Dynamic Scene Reconstruction from Casual Video
•
5
(a) Image & Dynamic
mask
(b) MoGe Mono-Depth
˜𝐷
(c) Consistent Video
Depth 𝐷w/o 𝐿object
depth
(d) Aligned object depth
(e) 𝐿object
depth
(f) Consistent Video
Depth 𝐷w/ 𝐿object
depth
Fig. 5. Depth Refinement. Mega-SAM depths lack fine detail: in (c) the
hockey stick is swallowed by the background. We restore this detail using
the high-quality MoGe depth in (b). We extract the object-only depth ˜𝐷(𝑜)
𝑡
for dynamic mask from (b) and align it to the consistent depth 𝐷𝑡, producing
(d). Then, the object-depth loss 𝐿object
depth between (c) and (d) (visualized in
(e)) is added it to Mega-SAM’s optimization. The refinement produces
depth maps that cleanly preserve the hockey stick, leading to higher-quality
reconstruction (see (f) and Fig. 17; details in Sec. 3.4).
reconstruction (Fig. 7). Points that remain on the same object usu-
ally move coherently with its surface, so we treat the object masks
𝑀(𝑜)
𝑡
as a re-identification oracle: if a track originates in 𝑀(𝑜) and
its current position is still inside that mask at frame 𝑡, we relabel it
visible. The recovered tracks 𝑢(𝑛)
𝑡
re-enter optimization, supply the
missing supervision, and restore the surrounding geometry (Fig. 7).
To suppress self-occlusion cases where a track is hidden by an-
other part of the same object, yet still falls inside the same mask
𝑀(𝑜)
𝑡
(Fig. 8), we leverage motion cues. For each recovered track,
we resample its position over a short window [𝑡−2, 𝑡+2]. If the
resampled trajectory { ˆ𝑢𝑡′} ever diverges from the original {𝑢𝑡′} by
more than 𝜏self-occ = 10 pixels:
max
𝑡′∈[𝑡−2,𝑡+2] Dist( ˆ𝑢𝑡′,𝑢𝑡′) > 𝜏self-occ,
we keep the track marked as invisible.
Leveraging the masks 𝑀(𝑜)
𝑡
, we (i) generate high-quality, tempo-
rally consistent depth maps 𝐷𝑇
𝑡=1 and (ii) obtain numerous robust
tracks 𝑢(𝑛)
𝑡
, especially along thin, complexly moving structures,
thereby enhancing the quality of dynamic-scene reconstruction.
3.5
Lifting to 3-D
We keep the lifting stage identical to the procedure in MoSca [Lei
et al. 2024]. In brief, dynamic pixels are back-projected to initialize
3-D Gaussians, and long-term 2-D tracks are promoted to motion-
scaffold nodes 𝑣(𝑚). The ensuing joint space–time optimization
enforces coherent motion and fills in trajectories through occlusion.
(See.Fig. 3 (II))
(a) Image & Dynamic
Mask
(b) Uniform Sampling
from EPI Error Mask
(c) Novel View Synthesis
w/o Skeleton Sampling
(d) Skeleton of Dynamic
Mask
(e) Skeleton Sampling
from Dynamic Mask
(f) Novel View Synthesis
w/ Skeleton Sampling
Fig. 6. Skeleton Sampling. 2-D tracks seed object motion. MoSca draws
them uniformly inside EPI error masks (b), so thin parts are undersampled
and disappear in the result (c). We instead extract a skeleton from each
dynamic mask 𝑀(𝑜)
𝑡
(d) and add extra samples in a narrow band around it
(e). These skeleton points, only 1
6 of the uniform set, provide dense coverage
of limbs, sharply resolving them in the reconstruction (f). Detail is in Sec. 3.4.
(a) Sample Point at 𝑡𝑎
(b) Track 𝑡𝑎→𝑡𝑏
(c) Novel View Synthesis
at 𝑡𝑏w/o
Re-identification
(d) Find Tracks in
Dynamic Mask at 𝑡𝑏
(e) Re-identify Tracks at
𝑡𝑏
(f) Novel View Synthesis
at 𝑡𝑏w/
Re-identification
Fig. 7. Track Re-identification. Points sampled at 𝑡𝑎are tracked toward
𝑡𝑏(blue dots, (a)). After an occlusion, a re-emerged point is not re-identified
(red dashed path and cross, (b)), leaving a gap in the reconstruction (c). We
mark a track as visible whenever its position lies inside the same dynamic
mask 𝑀(𝑜)
𝑡
where it originated. This restores the missing trajectory (e) and
completes the reconstruction (f). Detail is in Sec. 3.4.
3.6
Dynamic Scene Reconstruction
We render RGB ˆ𝐼𝑡, depth ˆ𝐷𝑡, and optical flow ˆ𝐹𝑡→𝑡′ with a Gaussian-
splatting renderer [Ye et al. 2025] and supervise them with the 2-D
priors from Sec. 3.4:
𝐿rgb = ∥ˆ𝐼𝑡−𝐼𝑡∥1 + 0.1 · 𝑆𝑆𝐼𝑀(ˆ𝐼𝑡, 𝐼𝑡),
𝐿depth = ∥ˆ𝐷𝑡−𝐷𝑡∥1,
𝐿gaussian
track
= ∥𝑢(𝑛)
𝑡
+ ˆ𝐹𝑡→𝑡′ [𝑢(𝑛)
𝑡
] −𝑢(𝑛)
𝑡′ ∥2.
Appearance 𝐿rgb and geometry 𝐿depth losses supervise Gaussians;
track loss propagates motion from Gaussians to the scaffold nodes,
which are regularized with as-rigid-as-possible (i.e. ARAP) (𝐿arap),
SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong.

<!-- page 6 -->
6
•
Meng-Li Shih, Ying-Huan Chen, Yu-Lun Liu, and Brian Curless
(a) Non self-occlusion
(b) Self-occlusion
Fig. 8. Self-occlusion filtering. Self-occlusion occurs when one part of
an object hides another. During re-identification (Sec. 3.4) such points can
be mistakenly marked visible. We resample each suspect point and retrack
it over a short window (cyan curve). If the point is truly self-occluded, the
new track adheres to the occluding surface and diverges from the original
trajectory (red dashed curve). We measure the 2-D distance 𝐷𝑖𝑠𝑡(·, ·) be-
tween the two paths and relabel the track occluded whenever this distance
exceeds 𝜏self-occ.
velocity (𝐿vel), and acceleration (𝐿acc) terms to ensure smooth, co-
herent motion. Although these losses follow [Lei et al. 2024], we
introduce two additional terms that better constrain structure of
scene and dynamic-object.
Virtual-view depth loss. Small camera baseline videos let Gaus-
sians overfit training views, producing floaters visible only from
novel viewpoints (Fig. 9). Inspired by few-shot reconstruction reg-
ularizers [Chen et al. 2022; Jain et al. 2021; Niemeyer et al. 2022;
Truong et al. 2023; Yin et al. 2024], we generate a virtual camera
Πvirtual
𝑡
by randomly translating Π𝑡in the image plane, warp 𝐷𝑡to
that view to obtain 𝐷virtual
𝑡
, and compute the following regulariza-
tion for reconstruction.
𝐿virtual
depth = ∥ˆ𝐷virtual
𝑡
−𝐷virtual
𝑡
∥1.
Before Regularization
After Regularization
(a) Training View
Synthesis
(b) Virtual View
Synthesis
(c) Virtual View
Synthesis
(d) Training View Depth
Loss
(e) Virtual View Depth
Loss
(f) Virtual View Depth
Loss
Fig. 9. Virtual-view depth loss. Gaussian splatting fits the training views
well (a, d) but shows floaters from virtual viewpoints (b). A depth loss
computed in these views, 𝐿virtual
depth , localizes the errors of floaters (e); adding
it as a regularizer removes the floaters and produces cleaner renderings (c,
f), enabling broader novel-view synthesis.
Table 1. Quantitative comparison on iPhone dataset [Gao et al. 2022].
Method
(a) Pose-free RGB
(b) Depth & pose
mPSNR
mSSIM
mLPIPS
mPSNR
mSSIM
mLPIPS
S.o.M.
17.15
0.625
0.278
17.32
0.598
0.296
MoSca
17.24
0.603
0.304
19.32
0.706
0.264
DpDy
–
–
–
–
0.559
0.516
Cat4D
–
–
–
18.24
0.666
0.227
Ours
17.63
0.648
0.268
19.43
0.711
0.260
Table 2. Quantitative comparison on NVIDIA dataset without ground-
truth camera poses.
RoDynRF
MoSca
Ours
PSNR/LPIPS
25.38/0.079
26.54/0.073
26.58/0.067
Table 3. Ablation study.
Ours w/o Lscaffold
track
w/o Mask-Guided Track w/o Lvirtual
depth
PSNR
17.63
17.61
17.55
17.64
SSIM
0.648
0.637
0.632
0.630
LPIPS 0.268
0.270
0.274
0.277
Scaffold-projection loss. ARAP regularization (𝐿arap) occasionally
lets scaffold nodes drift off the object, especially on thin, fast-moving
parts (Fig. 10). Because each node 𝑣(𝑚) originates from a track 𝑢(𝑛)
𝑡
,
we project the node into the camera, get 𝑣2𝐷and penalize its 2-D
distance to the corresponding track:
𝐿scaffold
track
= ∥𝑣2𝐷−𝑢(𝑛)
𝑡
∥2.
(a) w/o 𝐿scaffold
track
(b) w/ 𝐿scaffold
track
Fig. 10. Scaffold tracking regularization. ARAP regularization can let
scaffold nodes drift off the object, especially on thin or fast-moving parts.
The initial scaffolds 𝑣(𝑚) (blue dots) are pushed away from the dynamic
object during training (red dots). We add the loss 𝐿scaffold
track
to regularize the
scaffold’s projected 2-D position toward the tracked points 𝑢(𝑛)
𝑡
(see (b)).
4
Experimental Results
4.1
Implementation details.
We set 𝜏salient = 0.05 and 𝜏appearance = 0.2 to pick dynamic masks.
Skeletons (OpenCV) guide sampling: 3000 of 19384 tracks are drawn
from the skeleton, the remaining 16,384 are uniformly sampled as
in [Lei et al. 2024]. Tracks are propagated with SpatialTracker [Xiao
et al. 2024], and self-occluded ones are pruned using𝜏self−occ = 10 px.
Virtual views are generated by translating Π𝑡up to 0.18median(𝐷𝑡)
in the image plane. All other settings follow [Lei et al. 2024]. The
pipeline proceeds through (a) dynamic mask selection (17 min), (b)
pose estimation and depth refinement (11 min), (c) mask-guided
SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong.

<!-- page 7 -->
Prior-Enhanced Gaussian Splatting for Dynamic Scene Reconstruction from Casual Video
•
7
point tracking (12 min), (d) lifting and initialization to 3-D (5 min),
and (e) reconstruction (25 min).
4.2
Datasets
iPhone DyCheck [Gao et al. 2022]. This benchmark contains 14
casually captured handheld videos (200–500 frames at 360×480) of
dynamic scenes, seven of which include two static “held-out” cam-
eras for novel-view evaluation. Besides RGB, ARKit provides noisy
LiDAR depth and approximate poses. We adopt two complementary
protocols: (i) depth & pose: both LiDAR depth and ARKit poses are
available during training, following setting in MoSca [Lei et al. 2024].
(ii) pose-free RGB: neither LiDAR depth nor ARKit poses is available
during training, making the task relies on pure monocular RGB. For
fair comparison under pose-free RGB, all methods perform identical
test-time camera poses optimization while freezing their learned
scene representations. Evaluation reports novel view PSNR, SSIM,
and LPIPS.
NVIDIA multi-view videos [Yoon et al. 2020]. This dataset offers
eight dynamic scenes filmed by a 16-camera rig with very small
baselines. Following RoDynRF [Liu et al. 2023], we use a single
forward-facing view as training input and treat the remaining 15
as unseen target views. No depth and camera poses are provided in
our evaluation. We also adopt the same test-time poses optimization
for all methods. Quantitative assessment uses PSNR and LPIPS.
DAVIS [Perazzi et al. 2016]. To examine real-world robustness,
we employ the 50 validation clips of DAVIS, which contain high-
motion Internet videos (960x540, 25-40 fps) that lack calibration,
depth, or multi-view supervision. Each sequence is treated as an
unconstrained monocular video. All methods train using only RGB
frames and synthesize novel viewpoints on a circular trajectory
for qualitative evaluations. Because ground truth geometry is un-
available, we provide side-by-side renderings and highlight typical
success and failure modes rather than reporting quantitative scores.
4.3
Baselines
We benchmark our pipeline against two state-of-the-art monocular
4D reconstruction systems chosen for their relevance to our pose-
free setting and their publicly available code.
Shape of Motion [Wang et al. 2024b]. represents dynamic scenes
with per-point linear combinations of 𝔰𝔢(3) motion bases, but re-
quires accurate camera extrinsics as input. Following its public
repository, we first estimate the training-view poses and depths
with MegaSAM [Li et al. 2024b] and then feed these data to Shape
of Motion for full-sequence optimization. At test time, we keep
the learned scene parameters fixed and perform the same pose re-
finement that we apply to all methods in our pose-free protocol,
ensuring a fair comparison.
MoSca [Lei et al. 2024]. is a complete system that jointly solves
for camera parameters, and dynamic scene reconstruction. The
original evaluation exploits either LiDAR depth or ground truth
poses provided by the DyCheck dataset. To follow the setting of
dynamic reconstruction from monocular RGB video, we run the
official code without providing LiDAR depth maps, or ground truth
camera parameters thereby limiting supervision to the same RGB
frames available to our method. Following the information from
their paper and code repository, we substitute LiDAR with [Piccinelli
et al. 2024] on DyCheck and [Hu et al. 2024] on DAVIS. All other
hyperparameters are kept at their default values.
For completeness, we also compare our method with DpDy [Wang
et al. 2024c] and Cat4D [Wu et al. 2024a]. Note that, unlike our
approach, these baselines require additional information (e.g. camera
poses).
4.4
Qualitative Comparison
iPhone DyCheck. Fig. 12 compares rendering results produced
by Shape of Motion and MoSca with those of ours on DyCheck
sequences. Shape of Motion suffers from severe tearing artifacts
along depth discontinuities (e.g., box edges and driver arms), and its
color bleeding reveals motion basis over-fitting. MoSca mitigates
some tearing through joint pose optimization, yet its reliance on
dense per-frame depth warping leaves large holes and “melting”
geometry on rapidly moving limbs. In comparison, our method
reconstructs complete geometry with crisp textures. Thin structures
such as the cardboard slots and steering wheel spokes are preserved,
and hand motion appears smooth.
DAVIS in-the-wild videos. Fig. 11 showcases challenging DAVIS
clips containing fast-moving cars, dancers, fountains, and roller
skaters. All sequences exhibit large inter-frame motion and complex
occlusions with no auxiliary sensors. Shape of Motion frequently
collapses background geometry outside the narrow view frustum
estimated from its noisy monocular poses (see blurred facades and
missing road), while MoSca’s per-pixel depth fusion produces ar-
tifacts on thin, rapidly moving objects (highlighted by red boxes).
Our method retains fine details such as the dancer’s raised arm and
the fountain’s statues by globally aggregating track-conditioned ob-
servations. It maintains consistent background parallax even where
foreground motion is extreme.
4.5
Quantitative Comparison
iPhone DyCheck. Table 1 summarizes results for the (a) pose-free
RGB and the (b) depth & pose settings, respectively. In the strictly
monocular regime (Table 1 (a)), our method performs on par with
Shape of Motion and MoSca. When LiDAR depth and ground truth
poses are provided (Table 1 (b)), all methods improve, and our scores
again slightly outperform the baselines.
NVIDIA multi-view videos. On the forward-facing NVIDIA bench-
mark (Table 2), our method is marginally ahead of MoSca and clearly
above RoDynRF. Although the absolute gains are modest due to the
small-baseline setup, the results confirm that our tracker-guided
4D splatting generalizes beyond handheld monocular footage to
multi-camera data.
4.6
Ablation Study
We ablate the three main ingredients of our pipeline on the pose-free
iPhone DyCheck setting (see Tab. 3) and also show representative
visual effects on DAVIS (see Fig. 13, 14, 15, 16, 17, and 18).
SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong.

<!-- page 8 -->
8
•
Meng-Li Shih, Ying-Huan Chen, Yu-Lun Liu, and Brian Curless
Mask-Guided Point Tracker. Omitting this module’s skeleton
sampling leaves thin regions badly under-sampled. In Fig. 13, the
rhino’s fore-leg is almost lost, whereas the full model places ample
tracks along the limb and recovers both geometry and texture. Long-
range tracks also break at occlusions. This module’s track re-ID
step, guided by the dynamic masks, stitches them back together.
Without it, the walker’s left leg (Fig. 14) vanishes while briefly
hidden behind the right, and the reconstruction fails. Removing this
module overall causes a perceptible drop in fidelity (see Table 3).
Scaffold-track loss 𝐿scaffold
track
. This loss anchors scaffold nodes to the
2-D tracks. Without it, the ARAP term 𝐿arap pulls nodes on thin
moving parts together, producing artifacts around leg in Fig. 15
(a). While the numerical gap in Table 3 is small, visual continuity
improves markedly with this constraint.
Virtual-view depth loss 𝐿virtual
depth . Without this loss, floaters can pro-
liferate: in Fig. 18 the floor and building facade fragment into streaks.
Introducing 𝐿virtual
depth removes these artifacts, yielding a clean result.
Consistently, LPIPS rises without this loss as shown in Table 3.
Furthermore, our depth refinement step raises depth quality, re-
covers the true geometry of thin structures, and markedly improves
the final reconstruction (see Fig. 16 and Fig. 17).
5
Conclusions
This work demonstrates that stronger priors can markedly boost
the performance of Dynamic Gaussian Splatting. By extracting
salient object masks, refining video depth, and building reliable
2-D tracks, we supply the Gaussian cloud and its motion scaffold
with much richer supervision. Two complementary losses intro-
duced in Sec. 3.6 further suppress floaters and preserve coherence
on thin, fast-moving parts. The resulting pipeline is fully automatic
and surpasses previous monocular DGS methods.
Limitations. Our system copies motion- or focus-blur from the in-
put video (see Fig. 19) and leaves unseen regions empty. Integrating
a video generative model could deblur the frames and hallucinate
plausible content for those gaps.
Acknowledgments
This work was supported by Lenovo and the UW Reality Lab.
SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong.

<!-- page 9 -->
Prior-Enhanced Gaussian Splatting for Dynamic Scene Reconstruction from Casual Video
•
9
Fig. 11. Qualitative Comparison on DAVIS Dataset.
SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong.

<!-- page 10 -->
10
•
Meng-Li Shih, Ying-Huan Chen, Yu-Lun Liu, and Brian Curless
(a) Shape of Motion
(b) MoSca
(c) Ours
(d) Shape of Motion
(e) MoSca
(f) Ours
Fig. 12. Qualitative Comparison on iPhone Dataset. Without ground-
truth camera poses and without LiDAR depth, we use only RGB frames as
input to the method.
(a) w/o skeleton sampling
(b) w/ skeleton sampling
Fig. 13. Qualitative comparison of skeleton sampling.
(a) w/o track re-identification
(b) w/ track re-identification
Fig. 14. Qualitative comparison of track re-identification.
(a) w/o 𝐿scaffold
track
(b) w/ 𝐿scaffold
track
Fig. 15. Effect of scaffold-track loss 𝐿scaffold
track
.
(a) w/o 𝐿scaffold
track
(b) w/ 𝐿scaffold
track
Fig. 16. Effect of depth refinement on video depth 𝐷.
(a) Monocular video
(b) Novel view synthesis w/o Depth Refinement
(c) Novel view synthesis w/ Depth Refinement
Fig. 17. Effect of depth refinement on novel view synthesis.
(a) w/o 𝐿virtual
depth
(b) w/ 𝐿virtual
depth
Fig. 18. Qualitative comparison of virtual-view depth loss 𝐿virtual
depth .
(a) Monocular video
(b) Novel view synthesis
Fig. 19. Failure case. Our system copies motion- or focus-blur from the
input video, so blur visible in the monocular frame (a) persists in the syn-
thesized novel view (b).
SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong.

<!-- page 11 -->
Prior-Enhanced Gaussian Splatting for Dynamic Scene Reconstruction from Casual Video
•
11
References
ShahRukh Athar, Zexiang Xu, Kalyan Sunkavalli, Eli Shechtman, and Zhixin Shu.
2022. Rignerf: Fully controllable neural 3d portraits. In Proceedings of the IEEE/CVF
conference on Computer Vision and Pattern Recognition. 20364–20373.
Benjamin Attal, Jia-Bin Huang, Christian Richardt, Michael Zollhoefer, Johannes Kopf,
Matthew O’Toole, and Changil Kim. 2023. HyperReel: High-Fidelity 6-DoF Video
with Ray-Conditioned Sampling. arXiv preprint arXiv:2301.02238 (2023).
Jeongmin Bae, Seoha Kim, Youngsik Yun, Hahyun Lee, Gun Bang, and Youngjung
Uh. 2024. Per-gaussian embedding-based deformation for deformable 3d gaussian
splatting. In European Conference on Computer Vision. Springer, 321–335.
Aayush Bansal, Minh Vo, Yaser Sheikh, Deva Ramanan, and Srinivasa Narasimhan.
2020. 4D Visualization of Dynamic Events from Unconstrained Multi-View Videos.
arXiv preprint arXiv:2005.13532 (2020).
Minh-Quan Viet Bui, Jongmin Park, Jihyong Oh, and Munchurl Kim. 2023. DyBluRF:
Dynamic Deblurring Neural Radiance Fields for Blurry Monocular Video. arXiv
preprint arXiv:2312.13528 (2023).
Ang Cao and Justin Johnson. 2023. HexPlane: A Fast Representation for Dynamic
Scenes. arXiv preprint arXiv:2301.09632 (2023).
Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini
De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al.
2022. Efficient geometry-aware 3D generative adversarial networks. In CVPR.
Di Chen, Yu Liu, Lianghua Huang, Bin Wang, and Pan Pan. 2022. Geoaug: Data
augmentation for few-shot nerf with geometry constraints. In European Conference
on Computer Vision. Springer, 322–337.
Devikalyan Das, Christopher Wewer, Raza Yunus, Eddy Ilg, and Jan Eric Lenssen. 2023.
Neural Parametric Gaussians for Monocular Non-Rigid Object Reconstruction. arXiv
preprint arXiv:2312.01196 (2023).
Carl Doersch, Ankush Gupta, Larisa Markeeva, Adrià Recasens, Lucas Smaira, Yusuf
Aytar, João Carreira, Andrew Zisserman, and Yi Yang. 2022. TAP-Vid: A Benchmark
for Tracking Any Point in a Video. arXiv preprint arXiv:2211.03726 (2022).
Carl Doersch, Pauline Luc, Yi Yang, Dilara Gokay, Skanda Koppula, Ankush Gupta,
Joseph Heyward, Ignacio Rocco, Ross Goroshin, João Carreira, et al. 2024. Bootstap:
Bootstrapped training for tracking-any-point. In Proceedings of the Asian Conference
on Computer Vision. 3257–3274.
Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao
Carreira, and Andrew Zisserman. 2023. TAPIR: Tracking Any Point with per-frame
Initialization and temporal Refinement. arXiv preprint arXiv:2306.08637 (2023).
Yilun Du, Yinan Zhang, Hong-Xing Yu, Joshua B Tenenbaum, and Jiajun Wu. 2021.
Neural radiance flow for 4d view synthesis and video processing. In 2021 IEEE/CVF
International Conference on Computer Vision (ICCV). IEEE Computer Society, 14304–
14314.
Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wenzheng Chen, and Baoquan
Chen. 2024. 4d-rotor gaussian splatting: towards efficient novel view synthesis for
dynamic scenes. In ACM SIGGRAPH 2024 Conference Papers. 1–11.
Jiemin Fang, Taoran Yi, Xinggang Wang, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu,
Matthias Nießner, and Qi Tian. 2022. Fast dynamic radiance fields with time-aware
neural voxels. In SIGGRAPH Asia 2022 Conference Papers. 1–9.
Haiwen Feng, Junyi Zhang, Qianqian Wang, Yufei Ye, Pengcheng Yu, Michael J. Black,
Trevor Darrell, and Angjoo Kanazawa. 2025. St4RTrack: Simultaneous 4D Recon-
struction and Tracking in the World. arXiv preprint arXiv:2504.13152 (2025).
Sara Fridovich-Keil, Giacomo Meanti, Frederik Warburg, Benjamin Recht, and Angjoo
Kanazawa. 2023. K-Planes: Explicit Radiance Fields in Space, Time, and Appearance.
arXiv preprint arXiv:2301.10241 (2023).
Chen Gao, Ayush Saraf, Johannes Kopf, and Jia-Bin Huang. 2021. Dynamic view syn-
thesis from dynamic monocular video. In Proceedings of the IEEE/CVF International
Conference on Computer Vision. 5712–5721.
Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. 2022.
Monocular dynamic view synthesis: A reality check. Advances in Neural Information
Processing Systems 35 (2022), 33768–33780.
Lily Goli, Sara Sabour, Mark Matthews, Marcus Brubaker, Dmitry Lagun, Alec Jacobson,
David J Fleet, Saurabh Saxena, and Andrea Tagliasacchi. 2024. RoMo: Robust Motion
Segmentation Improves Structure from Motion. arXiv preprint arXiv:2411.18650
(2024).
Adam W. Harley, Zhaoyuan Fang, and Katerina Fragkiadaki. 2022. Particle Video
Revisited: Tracking Through Occlusions Using Point Trajectories. arXiv preprint
arXiv:2204.04153 (2022).
Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan,
and Ying Shan. 2024. Depthcrafter: Generating consistent long depth sequences for
open-world videos. arXiv preprint arXiv:2409.02095 (2024).
Nan Huang, Wenzhao Zheng, Chenfeng Xu, Kurt Keutzer, Shanghang Zhang, Angjoo
Kanazawa, and Qianqian Wang. 2025. Segment Any Motion in Videos. arXiv preprint
arXiv:2503.22268 (2025).
Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan
Qi. 2024. Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.
4220–4230.
Ajay Jain, Matthew Tancik, and Pieter Abbeel. 2021. Putting nerf on a diet: Semantically
consistent few-shot view synthesis. In Proceedings of the IEEE/CVF International
Conference on Computer Vision. 5885–5894.
Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho. 2024. RoDyGS: Robust
Dynamic Gaussian Splatting for Casual Videos. arXiv preprint arXiv:2412.03077
(2024).
Wei Jiang, Kwang Moo Yi, Golnoosh Samei, Oncel Tuzel, and Anurag Ranjan. 2022.
Neuman: Neural human radiance field from a single video. In European Conference
on Computer Vision. Springer, 402–418.
Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi,
and Christian Rupprecht. 2023. CoTracker: It is Better to Track Together. arXiv
preprint arXiv:2307.07635 (2023).
Laurynas Karazija, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. 2024. Learning
segmentation from point trajectories. Advances in Neural Information Processing
Systems 37 (2024), 112573–112597.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023.
3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions
on Graphics 42, 4 (2023).
Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura
Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, Piotr
Dollár, and Ross Girshick. 2023. Segment Anything. arXiv preprint arXiv:2304.02643
(2023).
Johannes Kopf, Xuejian Rong, and Jia-Bin Huang. 2020. Robust Consistent Video Depth
Estimation. arXiv preprint arXiv:2012.05901 (2020).
Agelos Kratimenos, Jiahui Lei, and Kostas Daniilidis. 2023. DynMF: Neural Motion
Factorization for Real-time Dynamic View Synthesis with 3D Gaussian Splatting.
arXiv preprint arXiv:2312.00112 (2023).
Agelos Kratimenos, Jiahui Lei, and Kostas Daniilidis. 2024. Dynmf: Neural motion
factorization for real-time dynamic view synthesis with 3d gaussian splatting. In
European Conference on Computer Vision. Springer, 252–269.
Sangwoon Kwak, Joonsoo Kim, Jun Young Jeong, Won-Sik Cheong, Jihyong Oh, and
Munchurl Kim. 2025. MoDec-GS: Global-to-Local Motion Decomposition and Tem-
poral Interval Adjustment for Compact Dynamic 3D Gaussian Splatting. arXiv
preprint arXiv:2501.03714 (2025).
Junoh Lee, ChangYeon Won, Hyunjun Jung, Inhwan Bae, and Hae-Gon Jeon. 2024.
Fully explicit dynamic gaussian splatting. Advances in Neural Information Processing
Systems 37 (2024), 5384–5409.
Jiahui Lei, Yijia Weng, Adam Harley, Leonidas Guibas, and Kostas Daniilidis. 2024.
MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds.
arXiv preprint arXiv:2405.17421 (2024).
Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. 2024a. Spacetime gaussian feature splatting
for real-time dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition. 8508–8520.
Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. 2020. Neural Scene
Flow Fields for Space-Time View Synthesis of Dynamic Scenes. arXiv preprint
arXiv:2011.12950 (2020).
Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. 2021. Neural scene flow
fields for space-time view synthesis of dynamic scenes. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. 6498–6508.
Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. 2022. LayeredGS: Effi-
cient Dynamic Scene Rendering and Point Tracking with Multi-Layer Deformable
Gaussian Splatting. arXiv preprint arXiv:2211.11082 (2022).
Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. 2023a. Fast View Synthesis
of Casual Videos with Soup-of-Planes. arXiv preprint arXiv:2304.01716 (2023).
Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo
Kanazawa, Aleksander Holynski, and Noah Snavely. 2024b. MegaSaM: Accurate,
Fast, and Robust Structure and Motion from Casual Dynamic Videos. arXiv preprint
arXiv:2412.04463 (2024).
Zhengqi Li, Qianqian Wang, Forrester Cole, Richard Tucker, and Noah Snavely. 2023b.
Dynibar: Neural dynamic image-based rendering. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. 4273–4284.
Hanxue Liang, Jiawei Ren, Ashkan Mirzaei, Antonio Torralba, Ziwei Liu, Igor Gilitschen-
ski, Sanja Fidler, Cengiz Oztireli, Huan Ling, Zan Gojcic, et al. 2024. Feed-Forward
Bullet-Time Reconstruction of Dynamic Scenes from Monocular Videos. arXiv
preprint arXiv:2412.03526 (2024).
Yiqing Liang, Numair Khan, Zhengqin Li, Thu Nguyen-Phuoc, Douglas Lanman, James
Tompkin, and Lei Xiao. 2023. GauFRe: Gaussian Deformation Fields for Real-time
Dynamic Novel View Synthesis. arXiv preprint arXiv:2312.11458 (2023).
Yiqing Liang, Numair Khan, Zhengqin Li, Thu Nguyen-Phuoc, Douglas Lanman, James
Tompkin, and Lei Xiao. 2025. Gaufre: Gaussian deformation fields for real-time
dynamic novel view synthesis. In 2025 IEEE/CVF Winter Conference on Applications
of Computer Vision (WACV). IEEE, 2642–2652.
Haotong Lin, Sida Peng, Zhen Xu, Tao Xie, Xingyi He, Hujun Bao, and Xiaowei Zhou.
2023. Im4D: High-Fidelity and Real-Time Novel View Synthesis for Dynamic Scenes.
arXiv preprint arXiv:2310.08585 (2023).
SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong.

<!-- page 12 -->
12
•
Meng-Li Shih, Ying-Huan Chen, Yu-Lun Liu, and Brian Curless
Youtian Lin, Zuozhuo Dai, Siyu Zhu, and Yao Yao. 2024. Gaussian-flow: 4d reconstruc-
tion with dynamic 3d gaussian particle. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition. 21136–21145.
Qingming Liu, Yuan Liu, Jiepeng Wang, Xianqiang Lyv, Peng Wang, Wenping Wang,
and Junhui Hou. 2024. MoDGS: Dynamic Gaussian Splatting from Casually-captured
Monocular Videos. arXiv preprint arXiv:2406.00434 (2024).
Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu Tseng, Ayush Saraf, Changil
Kim, Yung-Yu Chuang, Johannes Kopf, and Jia-Bin Huang. 2023. Robust Dynamic
Radiance Fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition.
Stephen Lombardi, Tomas Simon, Jason Saragih, Gabriel Schwartz, Andreas Lehrmann,
and Yaser Sheikh. 2019. Neural Volumes: Learning Dynamic Renderable Volumes
from Images. arXiv preprint arXiv:2011.13961 (2019).
Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. 2023. Dynamic
3D Gaussians: Tracking by Persistent Dynamic View Synthesis. arXiv preprint
arXiv:2308.09713 (2023).
Xingyu Miao, Yang Bai, Haoran Duan, Yawen Huang, Fan Wan, Yang Long, and Yefeng
Zheng. 2024. CTNeRF: Cross-Time Transformer for Dynamic Neural Radiance Field
from Monocular Video. arXiv preprint arXiv:2401.04861 (2024).
Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ra-
mamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields
for view synthesis. Commun. ACM 65, 1 (2021), 99–106.
Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas
Geiger, and Noha Radwan. 2022. Regnerf: Regularizing neural radiance fields for
view synthesis from sparse inputs. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition. 5480–5490.
Jongmin Park, Minh-Quan Viet Bui, Juan Luis Gonzalez Bello, Jaeho Moon, Jihyong Oh,
and Munchurl Kim. 2024. SplineGS: Robust Motion-Adaptive Spline for Real-Time
Dynamic 3D Gaussians from Monocular Video. arXiv preprint arXiv:2412.09982
(2024).
Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and
Alexander Sorkine-Hornung. 2016. A benchmark dataset and evaluation methodol-
ogy for video object segmentation. In Proceedings of the IEEE conference on computer
vision and pattern recognition. 724–732.
Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc
Van Gool, and Fisher Yu. 2024. UniDepth: Universal monocular metric depth esti-
mation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. 10106–10116.
Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu
Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. 2024a.
Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714
(2024).
Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu
Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun,
Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick,
Piotr Dollár, and Christoph Feichtenhofer. 2024b. SAM 2: Segment Anything in
Images and Videos. arXiv preprint arXiv:2408.00714 (2024).
Richard Shaw, Michal Nazarczuk, Jifei Song, Arthur Moreau, Sibi Catley-Chandar,
Helisa Dhamo, and Eduardo Perez-Pellitero. 2023. SWinGS: Sliding Windows for
Dynamic 3D Gaussian Splatting. arXiv preprint arXiv:2312.13308 (2023).
Meng-Li Shih, Jia-Bin Huang, Changil Kim, Rajvi Shah, Johannes Kopf, and Chen Gao.
2024. Modeling ambient scene dynamics for free-view synthesis. In ACM SIGGRAPH
2024 Conference Papers. 1–11.
Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele Chen, Junsong Yuan, Yi Xu,
and Andreas Geiger. 2023. Nerfplayer: A streamable dynamic scene representation
with decomposed neural radiance fields. IEEE Transactions on Visualization and
Computer Graphics 29, 5 (2023), 2732–2742.
Colton Stearns, Adam Harley, Mikaela Uy, Florian Dubost, Federico Tombari, Gordon
Wetzstein, and Leonidas Guibas. 2024. Dynamic Gaussian Marbles for Novel View
Synthesis of Casual Monocular Videos. arXiv preprint arXiv:2406.18717 (2024).
Jiakai Sun, Han Jiao, Guangyuan Li, Zhanjie Zhang, Lei Zhao, and Wei Xing. 2024.
3dgstream: On-the-fly training of 3d gaussians for efficient streaming of photo-
realistic free-viewpoint videos. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition. 20675–20685.
Zachary Teed and Jia Deng. 2020. Raft: Recurrent all-pairs field transforms for optical
flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August
23–28, 2020, Proceedings, Part II 16. Springer, 402–419.
Prune Truong, Marie-Julie Rakotosaona, Fabian Manhardt, and Federico Tombari. 2023.
Sparf: Neural radiance fields from sparse and noisy poses. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4190–4200.
Chaoyang Wang, Peiye Zhuang, Aliaksandr Siarohin, Junli Cao, Guocheng Qian, Hsin-
Ying Lee, and Sergey Tulyakov. 2024c. Diffusion priors for dynamic view synthesis
from monocular videos. arXiv preprint arXiv:2401.05583 (2024).
Liao Wang, Jiakai Zhang, Xinhang Liu, Fuqiang Zhao, Yanshun Zhang, Yingliang Zhang,
Minye Wu, Jingyi Yu, and Lan Xu. 2022. Fourier plenoctrees for dynamic radiance
field rendering in real-time. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition. 13524–13534.
Qianqian Wang, Vickie Ye, Hang Gao, Jake Austin, Zhengqi Li, and Angjoo Kanazawa.
2024b. Shape of Motion: 4D Reconstruction from a Single Video. arXiv preprint
arXiv:2407.13764 (2024).
Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A. Efros, and Angjoo
Kanazawa. 2025. Continuous 3D Perception Model with Persistent State. arXiv
preprint arXiv:2501.12387 (2025).
Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and
Jiaolong Yang. 2024a. Moge: Unlocking accurate monocular geometry estima-
tion for open-domain images with optimal training supervision. arXiv preprint
arXiv:2410.19115 (2024).
Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu,
Qi Tian, and Xinggang Wang. 2023. 4D Gaussian Splatting for Real-Time Dynamic
Scene Rendering. arXiv preprint arXiv:2310.08528 (2023).
Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu,
Qi Tian, and Xinggang Wang. 2024b. 4d gaussian splatting for real-time dynamic
scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition. 20310–20320.
Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron,
and Aleksander Holynski. 2024a. Cat4d: Create anything in 4d with multi-view
video diffusion models. arXiv preprint arXiv:2411.18613 (2024).
Wenqi Xian, Jia-Bin Huang, Johannes Kopf, and Changil Kim. 2021. Space-time neural
irradiance fields for free-viewpoint video. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition. 9421–9431.
Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue, Sida Peng, Yujun Shen, and
Xiaowei Zhou. 2024. SpatialTracker: Tracking Any 2D Pixels in 3D Space. arXiv
preprint arXiv:2404.04319 (2024).
Zhen Xu, Yinghao Xu, Zhiyuan Yu, Sida Peng, Jiaming Sun, Hujun Bao, and Xiaowei
Zhou. 2024. Representing long volumetric video with temporal gaussian hierarchy.
ACM Transactions on Graphics (TOG) 43, 6 (2024), 1–18.
Jinbo Yan, Rui Peng, Luyang Tang, and Ronggang Wang. 2024. 4D Gaussian Splatting
with Scale-aware Residual Field and Adaptive Optimization for Real-time rendering
of temporally complex dynamic scenes. In Proceedings of the 32nd ACM International
Conference on Multimedia. 7871–7880.
Jinyu Yang, Mingqi Gao, Zhe Li, Shang Gao, Fangjing Wang, and Feng Zheng. 2023a.
Track Anything: Segment Anything Meets Videos. arXiv preprint arXiv:2304.11968
(2023).
Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang
Zhao. 2024b. Depth anything: Unleashing the power of large-scale unlabeled data. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
10371–10381.
Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. 2024a.
Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction.
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.
20331–20341.
Zeyu Yang, Hongye Yang, Zijie Pan, Xiatian Zhu, and Li Zhang. 2023b. Real-time pho-
torealistic dynamic scene representation and rendering with 4d gaussian splatting.
arXiv preprint arXiv:2310.10642 (2023).
Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto
Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, and Angjoo Kanazawa. 2025. gsplat:
An open-source library for Gaussian splatting. Journal of Machine Learning Research
26, 34 (2025), 1–17.
Ruihong Yin, Vladimir Yugay, Yue Li, Sezer Karaoglu, and Theo Gevers. 2024.
FewViewGS: Gaussian Splatting with Few View Matching and Multi-stage Training.
arXiv preprint arXiv:2411.02229 (2024).
Jae Shin Yoon, Kihwan Kim, Orazio Gallo, Hyun Soo Park, and Jan Kautz. 2020. Novel
view synthesis of dynamic scenes with globally coherent depths from a monocular
camera. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. 5336–5345.
Meng You, Mantang Guo, Xianqiang Lyu, Hui Liu, and Junhui Hou. 2023. Decou-
pling Dynamic Monocular Videos for Dynamic View Synthesis. arXiv preprint
arXiv:2304.01716 (2023).
Bowei Zhang, Lei Ke, Adam W Harley, and Katerina Fragkiadaki. 2025b. TAPIP3D:
Tracking Any Point in Persistent 3D Geometry. arXiv preprint arXiv:2504.14717
(2025).
Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester
Cole, Deqing Sun, and Ming-Hsuan Yang. 2024. MonST3R: A Simple Approach for
Estimating Geometry in the Presence of Motion. arXiv preprint arXiv:2410.03825
(2024).
Xinyu Zhang, Haonan Chang, Yuhan Liu, and Abdeslam Boularias. 2025a.
Mo-
tion Blender Gaussian Splatting for Dynamic Reconstruction.
arXiv preprint
arXiv:2503.09040 (2025).
Yifan Zhang, Yifan Wang, Xiaoming Zhao, and Yebin Liu. 2023. DynPoint: Dynamic
Neural Point For View Synthesis. arXiv preprint arXiv:2310.18999 (2023).
Zhoutong Zhang, Forrester Cole, Zhengqi Li, Michael Rubinstein, Noah Snavely, and
William T. Freeman. 2022. Structure and Motion from Casual Videos. In Computer
SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong.

<!-- page 13 -->
Prior-Enhanced Gaussian Splatting for Dynamic Scene Reconstruction from Casual Video
•
13
Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022,
Proceedings, Part XXXIII (Tel Aviv, Israel). Springer-Verlag, Berlin, Heidelberg, 20–37.
https://doi.org/10.1007/978-3-031-19827-4_2
Xiaoming Zhao, Yifan Wang, Yifan Zhang, and Yebin Liu. 2023. Pseudo-Generalized
Dynamic View Synthesis from a Video. arXiv preprint arXiv:2310.08587 (2023).
Xiaoming Zhao, Yifan Wang, Yifan Zhang, and Yebin Liu. 2024. DynOMo: Online Point
Tracking by Dynamic Monocular Reconstruction. arXiv preprint arXiv:2409.02104
(2024).
Ruijie Zhu, Yanzhe Liang, Hanzhi Chang, Jiacheng Deng, Jiahao Lu, Wenfei Yang,
Tianzhu Zhang, and Yongdong Zhang. 2024. Motiongs: Exploring explicit motion
guidance for deformable 3d gaussian splatting. Advances in Neural Information
Processing Systems 37 (2024), 101790–101817.
Zrporz. 2024. AutoSeg-SAM2. https://github.com/zrporz/AutoSeg-SAM2 Automated
image segmentation tool based on Segment Anything Model (SAM).
SA Conference Papers ’25, December 15–18, 2025, Hong Kong, Hong Kong.
