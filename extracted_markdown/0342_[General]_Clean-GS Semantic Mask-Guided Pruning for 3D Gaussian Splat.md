<!-- page 1 -->
Clean-GS: Semantic Mask-Guided Pruning for 3D Gaussian
Splatting
Subhankar Mishra
smishra@niser.ac.in
School of Computer Sciences
National Institute of Science Education and Research
Bhubaneswar, India
Abstract
3D Gaussian Splatting produces high-quality scene reconstructions but generates hun-
dreds of thousands of spurious Gaussians (floaters) scattered throughout the environment.
These artifacts obscure objects of interest and inflate model sizes, hindering deployment in
bandwidth-constrained applications. We present Clean-GS, a method for removing back-
ground clutter and floaters from 3DGS reconstructions using sparse semantic masks. Our
approach combines whitelist-based spatial filtering with color-guided validation and out-
lier removal to achieve 60-80% model compression while preserving object quality.
Un-
like existing 3DGS pruning methods that rely on global importance metrics, Clean-GS
uses semantic information from as few as 3 segmentation masks (1% of views) to iden-
tify and remove Gaussians not belonging to the target object. Our multi-stage approach
consisting of (1) whitelist filtering via projection to masked regions, (2) depth-buffered
color validation, and (3) neighbor-based outlier removal isolates monuments and objects
from complex outdoor scenes. Experiments on Tanks and Temples show that Clean-GS
reduces file sizes from 125MB to 47MB while maintaining rendering quality, making 3DGS
models practical for web deployment and AR/VR applications. Our code is available at
https://github.com/smlab-niser/clean-gs
1
Introduction
3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) represents scenes as collections of anisotropic Gaussians
with position, covariance, opacity, and spherical harmonic (SH) coefficients for view-dependent appearance.
This explicit representation allows real-time rendering while achieving quality comparable to neural radiance
fields (Mildenhall et al., 2020). However, 3DGS reconstructions of outdoor scenes generate hundreds of thou-
sands of floaters (spurious Gaussians scattered throughout the environment to fill gaps in reconstruction).
These floaters, combined with background elements, obscure objects of interest and inflate model sizes.
Isolating clean objects from cluttered 3DGS reconstructions addresses critical needs:
cultural heritage
archival requires monuments without surrounding floaters and modern infrastructure; AR/VR applications
need clean objects without environmental artifacts; web deployment demands smaller models.
Existing
3DGS compression methods (Lee et al., 2024; Fan et al., 2023) focus on global pruning through importance
scoring or gradient-based selection. These approaches cannot distinguish object Gaussians from floaters and
background (both may exhibit high opacity and reconstruction gradients).
We present Clean-GS, a semantic mask-guided pruning method that removes floaters and isolates objects
from 3DGS reconstructions. Three stages operate sequentially: whitelist filtering removes entire environment
and distant floaters, depth-buffered color validation eliminates visible artifacts near the object, and neighbor-
based outlier removal cleans isolated floaters. Spatial consistency across sparse semantic masks (as few as 3
views out of 302) provides sufficient signal for effective floater removal.
Our method achieves 60-80% compression on monument isolation tasks while preserving object detail. Pro-
cessing requires 2-5 minutes on commodity multi-core CPUs. The approach is particularly effective with
1
arXiv:2601.00913v1  [cs.CV]  1 Jan 2026

<!-- page 2 -->
Original GS
Clean GS
Segmentation  Masks
Figure 1: 3D Gaussian Splatting reconstructions contain massive numbers of floaters: artifacts scattered
throughout the scene that obscure objects of interest. Clean-GS removes environmental floaters and back-
ground elements, producing clean isolated reconstructions. Temple: 525K→198K Gaussians (62% reduction,
125MB→47MB).
sparse supervision: on a 302-view temple dataset, using only 3 manually-created masks yields 62% compres-
sion, with diminishing returns from additional masks.
Contributions:
• Three-stage floater removal system for 3DGS object isolation using sparse semantic masks
• Depth-buffered color validation to eliminate visible floaters near objects
• Neighbor-based outlier removal for cleaning isolated artifact Gaussians
• 60-80% compression on monument datasets by removing environmental floaters
2
Related Work
2.1
3D Gaussian Splatting
3D Gaussian Splatting (Kerbl et al., 2023) represents scenes as collections of 3D Gaussians, each with position,
covariance, opacity, and spherical harmonic (SH) coefficients for view-dependent color. Unlike neural radiance
fields (NeRF) (Mildenhall et al., 2020), 3DGS allows real-time rendering through rasterization-based splatting
while achieving comparable or superior quality. Recent work has extended 3DGS to dynamic scenes (Luiten
et al., 2024), improved anti-aliasing (Yu et al., 2023), and enhanced geometry (Huang et al., 2024).
2.2
Neural Scene Compression
Compression of neural 3D representations has been extensively studied for NeRF. Methods include neu-
ral pruning (Wu et al., 2023), vector quantization (Chen et al., 2023), and knowledge distillation (Kurz
et al., 2022).
TensoRF (Chen et al., 2022) achieves compression through tensor decomposition, while
VQAD (Takikawa et al., 2022) uses learned codebooks. However, these methods focus on global compression
rather than semantic object isolation.
2.3
3DGS Pruning and Compression
Several recent works address 3DGS compression. LightGaussian (Fan et al., 2023) prunes Gaussians based
on importance scores derived from opacity and contribution to reconstruction loss, then applies knowl-
edge distillation for refinement. Compact3D (Lee et al., 2024) uses sensitivity-based pruning to identify
removable Gaussians.
Mini-Splatting (Fan et al., 2024) introduces pupil-based pruning that considers
view-dependent importance.
2

<!-- page 3 -->
These methods share a common limitation: they use global importance metrics (opacity, gradient magnitude,
reconstruction loss) that cannot distinguish between desired and undesired scene elements. A tree with high
opacity and strong gradients will be preserved even if the goal is to isolate a temple statue.
2.4
Semantic 3D Reconstruction
Semantic segmentation has been integrated into 3D reconstruction pipelines for scene understanding.
Semantic-NeRF (Zhi et al., 2021) jointly learns geometry and semantics. Panoptic 3D Scene Reconstruc-
tion (Dahnert et al., 2021) combines instance segmentation with 3D reconstruction. However, these methods
focus on semantic understanding rather than semantic-guided compression.
2.5
Object Isolation and Background Removal
Object isolation from complex scenes has been studied in multi-view stereo (MVS) and photogrammetry.
Traditional approaches use binary masks or trimap-based matting (Sun et al., 2004) to guide reconstruction.
More recent learning-based methods (Sengupta et al., 2020) leverage semantic segmentation for background
removal. Our work brings semantic mask guidance to 3DGS pruning, enabling object isolation from already-
trained models.
2.6
Point Cloud Outlier Removal
Point cloud processing literature provides outlier removal techniques.
Statistical Outlier Removal
(SOR) (Rusu and Cousins, 2011) uses k-NN distances with statistical thresholds. DBSCAN (Ester et al.,
1996) clusters points by density. Radius Outlier Removal filters points with few neighbors in a radius. While
these techniques are well-established for geometric point clouds, their application to 3D Gaussians (which
have opacity, color, and covariance) requires adaptation, which we provide in our neighbor-based outlier
removal stage.
Our approach differs from prior work by: (1) using semantic masks to whitelist desired Gaussians be-
fore any importance-based pruning, (2) validating colors with depth buffering to remove floaters, and (3)
demonstrating effectiveness with extremely sparse masks (3 views out of 302).
3
Method
Given a trained 3DGS model with N Gaussians {Gi}N
i=1 and a set of M views with binary semantic masks
{Mj}M
j=1 indicating object regions, our goal is to produce a pruned model {G′
k}N′
k=1 where N ′ ≪N, containing
only Gaussians representing the target object. Our method operates in three stages: whitelist filtering retains
Gaussians projecting to object regions, color validation removes visible artifacts using depth buffering, and
outlier removal eliminates isolated floaters via k-NN analysis.
3.1
Stage 1: Whitelist Filtering
The first stage eliminates all Gaussians that never project to object regions in masked views.
Projection.
For each Gaussian Gi with 3D position xi ∈R3 and camera j with intrinsics Kj, rotation
R(j)
c2w, and translation t(j)
c2w, we compute the 2D projection:
R(j)
w2c = (R(j)
c2w)T ,
t(j)
w2c = −R(j)
w2ct(j)
c2w
(1)
xcam = R(j)
w2cxi + t(j)
w2c
(2)
d = xcam[2],
u = Kjxcam
d
(3)
3

<!-- page 4 -->
where u = [u, v]T are pixel coordinates and d is depth. We reject projections with d ≤0 or u outside image
bounds.
Whitelist Construction.
For each Gaussian, we check if it projects to an object region (white pixels) in
any masked view:
wi =
M
_
j=1
⊮[Mj(u(j)
i ) > 0]
(4)
where ⊮[·] is the indicator function and u(j)
i
is the projection of Gi to view j. The whitelist contains all
Gaussians with wi = 1.
This stage removes background and environment based on spatial evidence. If a Gaussian never projects to
the object in any masked view, it cannot be part of the object.
3.2
Stage 2: Color Validation
Whitelisted Gaussians may still include floaters (artifacts that project near the object but have incorrect
colors). We validate colors using depth buffering.
Depth Buffering.
For each masked view j, we render only whitelisted Gaussians with depth buffering.
At each pixel p, we keep only the Gaussian with minimum depth:
g∗(p) = arg
min
i∈W,u(j)
i
=p
d(j)
i
(5)
where W is the whitelist set and d(j)
i
is the depth of Gaussian i at view j.
Color Matching.
For each front-layer Gaussian i at pixel p, we compute its RGB color ci from SH
coefficients:
ci = C0 · f (i)
dc + 0.5
(6)
where C0 = 0.28209479 is the spherical harmonic constant and f(i)
dc are the DC components. We compare
with the expected color cmask(p) from the masked image:
δi = ∥ci −cmask(p)∥2
(7)
Filtering.
We keep Gaussian i if either:
• It was never rendered as front-layer in any masked view (occluded), OR
• It has color match δi < τ in at least one view where it was rendered
This approach is conservative: we only remove Gaussians with clear evidence of color mismatch.
3.3
Stage 3: Outlier Removal
After whitelist filtering and color validation, remaining artifacts are typically isolated in 3D space.
We
provide three outlier removal strategies:
4

<!-- page 5 -->
Spatial Outlier Removal.
Remove Gaussians far from the scene center:
c = 1
|S|
X
i∈S
xi,
di = ∥xi −c∥2
(8)
where S is the set of kept Gaussians. We remove Gaussians with di > percentile(d, pspatial) where pspatial =
99 by default.
Neighbor-Based Outlier Removal.
Remove Gaussians far from their k nearest neighbors:
¯di = 1
k
X
j∈kNN(i,k)
∥xi −xj∥2
(9)
We remove Gaussians with ¯di > percentile( ¯d, pneighbor) where k = 10 and pneighbor = 95 by default.
Multi-View Consistency.
In Stage 1, instead of requiring appearance in ≥1 masked view, require
appearance in ≥m views:
wi =


M
X
j=1
⊮[Mj(u(j)
i ) > 0] ≥m


(10)
This removes Gaussians that only appear in one view by chance (likely artifacts).
3.4
Implementation Details
Parallelization.
We parallelize across views using multiprocessing with 96 CPU cores. Each worker pro-
cesses one view independently, computing projections and color matches. Results are aggregated with boolean
operations (whitelist) or counting (multi-view).
Resolution Handling.
COLMAP undistortion may change image resolutions. We resize masks to match
camera parameters before projection.
Parameters.
We use color threshold τ = 0.40 (Euclidean distance in RGB space), k = 10 neighbors,
spatial percentile pspatial = 99, neighbor percentile pneighbor = 95, and minimum views m = 2 for multi-view
mode. These were determined empirically and work well across datasets.
4
Experiments
4.1
Experimental Setup
Datasets.
We evaluate on two monument isolation scenarios:
• Temple: An outdoor temple structure from Tanks and Temples (Knapitsch et al., 2017) captured
with 302 views. We create 3 semantic masks (1% of views) using SAM (Kirillov et al., 2023) to
indicate the temple region. Original model: 525,717 Gaussians, 125 MB.
• Isha Statue: A large outdoor statue with 103 views and corresponding semantic masks created
using SAM. Original model: 1,112,566 Gaussians, 263.6 MB.
5

<!-- page 6 -->
Dataset / Method
Gaussians
Size
Compression
Time
Temple (302 views, 3 masks)
Original
525K
125 MB
0%
-
Basic
208K
49.2 MB
60.4%
20.7s
Clean-GS
198K
46.7 MB
62.4%
36.8s
Isha (103 views, 103 masks)
Original
1.1M
263.6 MB
0%
-
Basic
233K
55.1 MB
79.1%
-
Clean-GS
221K
52.4 MB
80.1%
40.4s
Table 1: Compression and timing results.
Basic method applies whitelist filtering and color validation.
Clean-GS adds neighbor-based outlier removal for additional 2% compression.
Stage
Removed
Remaining
Compression
Original
-
526K
0%
+ Whitelist
163K
363K
31%
+ Color Validation
155K
208K
60%
+ Neighbor Removal
10K
198K
62%
Table 2: Pipeline stage ablation on Temple dataset. Whitelist filtering provides most compression (31%),
color validation adds 29%, and neighbor removal contributes an additional 2%.
Variants.
We compare:
• Original: Unmodified 3DGS model
• Basic: Whitelist + Color validation only (stages 1-2)
• Clean-GS: Full method with neighbor-based outlier removal (stages 1-3, recommended)
• Clean-GS (combined): All outlier removal strategies (most aggressive)
Implementation.
All experiments run on 96-core Intel Xeon CPU. We use color threshold τ = 0.40, k=10
neighbors, and 95th percentile for neighbor removal.
Note.
We do not report PSNR/SSIM as we aim to change the scene (remove background), not preserve
it. Visual quality assessment is more appropriate for object isolation tasks.
4.2
Quantitative Results
Table 1 shows compression and timing results on both datasets. Clean-GS achieves 60-80% compression
with modest processing time.
4.3
Ablation Study
Stage-by-Stage Analysis.
Table 2 shows the contribution of each pipeline stage on the Temple dataset
(525K original Gaussians).
Whitelist filtering eliminates background and environment. Color validation removes floaters and artifacts.
Neighbor removal cleans isolated outliers.
The Temple dataset uses 3 semantic masks out of 302 total views (1%), achieving 62% compression with
sparse supervision.
6

<!-- page 7 -->
4.4
Processing Time Analysis
Clean-GS processes on commodity CPUs.
On the Temple dataset, the basic method (whitelist + color
validation) requires 20.7 seconds for 60.4% compression. Adding neighbor-based outlier removal increases
processing time to 36.8 seconds (+16.1s, +78%) while achieving 62.4% compression (+2% additional reduc-
tion). On the larger Isha dataset, Clean-GS processes in 40.4 seconds for 80.1% compression. For most
applications, the neighbor mode provides the optimal balance of quality and speed.
4.5
Qualitative Results
Figure 2 shows rendered comparisons between original and cleaned models. Clean-GS removes background
elements while preserving structure and details. Figure 3 shows the 3 semantic masks used (1% of 302 views).
4.6
Outlier Removal Strategies
We compare three outlier removal strategies. Neighbor-based removal identifies floaters far from their k-
nearest neighbors, removing 10,400 Gaussians (5% of whitelisted). Spatial removal conservatively eliminates
extreme outliers beyond the 99th percentile distance from scene center, removing 2,080 Gaussians (1%).
Multi-view consistency requires Gaussians to appear in ≥2 masked views, removing 66,921 Gaussians (18%,
effective with many masks but too strict with sparse supervision). Combining all strategies removes 75,315
Gaussians (21%) but risks over-pruning. We recommend neighbor-based removal for best quality-compression
balance.
4.7
Limitations
If the object is partially occluded in all masked views, those regions will be incorrectly removed; mask
diversity with different viewpoints mitigates this. When background elements have similar colors to the
object, color validation may preserve some background Gaussians, though neighbor removal helps. Processing
requires multi-core CPU (2-5 minutes on 96 cores, 1.4-3.2 Gaussians/ms); GPU acceleration could improve
speed 10-100×.
4.8
Applications
Cultural heritage archival can isolate monuments from tourist-filled plazas. AR/VR applications can place
isolated objects in virtual environments. Web deployment benefits from 60-80% size reduction (47MB vs.
125MB loads 2.7× faster). Game engines and modeling tools can extract clean 3D assets.
5
Conclusion
We presented Clean-GS, a method for semantically-guided pruning of 3D Gaussian Splatting models. Spatial
consistency across semantic masks, combined with color validation and outlier removal, provides sufficient
signal for object isolation with sparse supervision.
The three-stage approach (whitelist filtering, depth-buffered color validation, and neighbor-based outlier
removal) achieves 60-80% model compression. Experiments on monument isolation tasks demonstrate effec-
tiveness with 3 segmentation masks out of 302 views.
The method addresses requirements in cultural heritage preservation, AR/VR content creation, and web-
based 3D visualization. Processing reduces model sizes from 125MB to 47MB (temple) and 264MB to 52MB
(statue), improving deployment feasibility for bandwidth-constrained applications.
5.1
Future Work
Integrating foundation models (SAM (Kirillov et al., 2023), GroundingDINO (Liu et al., 2023)) for automatic
mask generation from text prompts would eliminate manual annotation. Fine-tuning remaining Gaussians
7

<!-- page 8 -->
after pruning could recover quality loss. Extending 3DGS to jointly learn semantics during training could
enable pruning without post-processing masks.
GPU-accelerated projection and rendering could reduce
processing time from 20-40 seconds to seconds. Formal analysis of whitelist filtering guarantees (minimum
mask count, viewpoint diversity requirements) would provide stronger foundations. Integrating into heritage
platform Tirtha (Shivottam and Mishra, 2023) could streamline monument digitization workflows.
References
Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In
ECCV, 2022.
Jiangfan Chen, Dejia Xu, Yichen Liao, and Zhangyang Wang. Compressnerf: Compressing neural radiance
field with vector quantization. In CVPR Workshops, 2023.
Manuel Dahnert, Ji Hou, Angela Dai, and Matthias Nießner. Panoptic 3d scene reconstruction from a single
rgb image. In NeurIPS, 2021.
Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. A density-based algorithm for discovering
clusters in large spatial databases with noise. In KDD, volume 96, pages 226–231, 1996.
Guangchi Fan, Bing Wang, Yuqi Xie, Dejia Chen, Zehao Xie, Zhixin Tan, Wenzheng Wang, and Bao-
quan Song. Mini-splatting: Representing scenes with a constrained number of gaussians. arXiv preprint
arXiv:2403.14166, 2024.
Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, and Zhangyang Wang. Lightgaussian: Un-
bounded 3d gaussian compression with 15x reduction and 200+ fps. arXiv preprint arXiv:2311.17245,
2023.
Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao.
2d gaussian splatting for
geometrically accurate radiance fields. In SIGGRAPH, 2024.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for
real-time radiance field rendering. In ACM Transactions on Graphics (SIGGRAPH), volume 42, 2023.
Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao,
Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. ICCV, 2023.
Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-
scale scene reconstruction. ACM Transactions on Graphics, 36(4), 2017.
Andreas Kurz, Thomas Neff, Zhaoyang Lv, Michael Zollhöfer, and Markus Steinberger. Adanerf: Adaptive
sampling for real-time rendering of neural radiance fields. In ECCV, 2022.
KL Lee, Dmitrii Larionov, Jie Yang, Alexander Cherepkov, Vladimir Glinskiy, Dhawal Wankhede, and Rus-
lan Babushkin. Compact3d: Compressing gaussian splat radiance field models with vector quantization.
arXiv preprint arXiv:2311.18159, 2024.
Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang,
Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object
detection. arXiv preprint arXiv:2303.05499, 2023.
Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking
by persistent dynamic view synthesis. In 3DV, 2024.
Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren
Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020.
Radu Bogdan Rusu and Steve Cousins. 3d is here: Point cloud library (pcl). In ICRA, 2011.
8

<!-- page 9 -->
Soumyadip Sengupta, Vivek Jayaram, Brian Curless, Steven M Seitz, and Ira Kemelmacher-Shlizerman.
Background matting: The world is your green screen. In CVPR, 2020.
Jyotirmaya Shivottam and Subhankar Mishra. Tirtha-an automated platform to crowdsource images and
create 3d models of heritage sites. In Proceedings of the 28th International ACM Conference on 3D Web
Technology, pages 1–15, 2023.
Jian Sun, Jiaya Jia, Chi-Keung Tang, and Heung-Yeung Shum. Poisson matting. ACM Transactions on
Graphics (SIGGRAPH), 23(3):315–321, 2004.
Towaki Takikawa, Joey Litalien, Kangxue Yin, Karsten Kreis, Charles Loop, Derek Nowrouzezahrai, Alec
Jacobson, Morgan McGuire, and Sanja Fidler. Variable bitrate neural fields. In SIGGRAPH, 2022.
Lifeng Wu, Zhiqin Lin, Song-Hai Zhang, Xiang Xiao, and Yasutaka Furukawa. Neural pruning: Learning to
prune neural radiance fields. arXiv preprint arXiv:2305.10108, 2023.
Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d
gaussian splatting. In arXiv preprint arXiv:2311.16493, 2023.
Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and Andrew J Davison. In-place scene labelling and
understanding with implicit scene representation. In ICCV, 2021.
9

<!-- page 10 -->
Original (526K Gaussians)
Clean-GS (198K Gaussians, 62% reduction)
Figure 2: Rendered comparison across five viewpoints: front, front alternate, side, back, back alternate.
Clean-GS removes environmental floaters and background while preserving temple structure.
10

<!-- page 11 -->
Training Image
View 00002
Semantic Mask Overlay
View 00156
View 00262
Figure 3: All 3 semantic masks used for Temple isolation (1% of 302 views). Left: Training images. Right:
Mask overlays.
11
