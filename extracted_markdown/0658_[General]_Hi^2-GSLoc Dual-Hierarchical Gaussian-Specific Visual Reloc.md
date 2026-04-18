<!-- page 1 -->
Hi2-GSLoc: Dual-Hierarchical Gaussian-Specific Visual Relocalization
for Remote Sensing⋆
Boni Hua,b, Zhenyu Xiaa,b, Lin Chena,b, Pengcheng Hana,b and Shuhui Bua,b,∗
aSchool of Aeronautics, Northwestern Polytechnical University, Xi’an, 710072, China
bNational Key Laboratory of Aircraft Configuration Design, Xi’an, 710072, China
A R T I C L E I N F O
Keywords:
Visual Localization
Gaussian Splatting
UAV Relocalization
Dense Matching
A B S T R A C T
Visual relocalization, which estimates the 6-degree-of-freedom (6-DoF) camera pose from query
images, is fundamental to remote sensing and UAV applications. Existing methods face inherent
trade-offs: image-based retrieval and pose regression approaches lack precision, while structure-based
methods that register queries to Structure-from-Motion (SfM) models suffer from computational
complexity and limited scalability. These challenges are particularly pronounced in remote sensing
scenarios due to large-scale scenes, high altitude variations, and domain gaps of existing visual
priors. To overcome these limitations, we leverage 3D Gaussian Splatting (3DGS) as a novel
scene representation that compactly encodes both 3D geometry and appearance. We introduce Hi2-
GSLoc, a dual-hierarchical relocalization framework that follows a sparse-to-dense and coarse-to-
fine paradigm, fully exploiting the rich semantic information and geometric constraints inherent
in Gaussian primitives. To handle large-scale remote sensing scenarios, we incorporate partitioned
Gaussian training, GPU-accelerated parallel matching, and dynamic memory management strategies.
Our approach consists of two stages: (1) a sparse stage featuring a Gaussian-specific consistent render-
aware sampling strategy and landmark-guided detector for robust and accurate initial pose estimation,
and (2) a dense stage that iteratively refines poses through coarse-to-fine dense rasterization matching
while incorporating reliability verification. Through comprehensive evaluation on simulation data,
public datasets, and real flight experiments, we demonstrate that our method delivers competitive
localization accuracy, recall rate, and computational efficiency while effectively filtering unreliable
pose estimates. The results confirm the effectiveness of our approach for practical remote sensing
applications.
1. Introduction
In our increasingly automated world, unmanned aerial
vehicles have become indispensable for diverse remote sens-
ing applications—from agricultural monitoring to disaster
response and urban planning Wang et al. (2025b); Ye et al.
(2024); Yin et al. (2025). At the core of autonomous naviga-
tion lies visual relocalization: the ability to estimate precise
6-DoF camera poses from single images against pre-built
scene representations. While this capability has been ex-
tensively studied for ground-level scenarios, remote sensing
environments present unique and formidable challenges that
render existing approaches inadequate.
Remote sensing relocalization faces several critical chal-
lenges that distinguish it from conventional scenarios. First,
the scale disparity is enormous—scenes span kilometers
with highly repetitive patterns and sparse distinctive land-
marks, making traditional feature matching computationally
prohibitive and prone to ambiguous correspondences Ye
et al. (2024). Second, altitude-induced geometric ambiguity
creates significant localization uncertainty, as small angular
errors propagate to large positional deviations at operational
altitudes. Third, existing visual features suffer from severe
domain gaps, as most are trained on ground-level imagery
⋆This work was supported by the National Natural Science Foundation
of China (NSFC) under Grant No. 42130112 and the Postdoctoral Fellow-
ship Program of CPSF under Grant No. GZB20240986.
∗Corresponding author E-mail address: bushuhui@nwpu.edu.cn
ORCID(s):
and fail to capture the distinctive geometric and photometric
characteristics of aerial perspectives. Finally, dramatic view-
point variations between mapping and query phases, coupled
with illumination changes and seasonal variations, can cause
catastrophic failures in structure-based methods.
Current relocalization approaches struggle to address
these challenges effectively, as shown in Figure 1. Image
retrieval methods Berton and Masone (2025); Keetha et al.
(2023); Arandjelovic et al. (2016); Hu et al. (2024), while
demonstrating robustness through contrastive learning, are
fundamentally constrained by database density and suffer
from perceptual aliasing in repetitive aerial scenes. Direct
pose regression approaches Kendall et al. (2015); Walch
et al. (2016); Chen et al. (2021) lack the geometric ground-
ing necessary for high-precision localization and struggle
with generalization across varying scales. Structure-based
methods Sarlin et al. (2019); Brachmann and Rother (2021)
achieve superior accuracy through explicit 2D-3D corre-
spondences but face computational bottlenecks in large-
scale scenarios and correspondence failures under sub-
stantial viewpoint changes. Recent NeRF-based approaches
Zhao et al. (2024); Yen-Chen et al. (2021) offer promising
analysis-by-synthesis capabilities but suffer from prohibitive
computational costs and limited real-time applicability.
3DGS presents a compelling solution to these chal-
lenges. Unlike implicit neural representations, 3DGS pro-
vides explicit, interpretable 3D geometry while maintaining
efficient rendering capabilities Kerbl et al. (2023). Crucially,
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 1 of 17
arXiv:2507.15683v1  [cs.CV]  21 Jul 2025

<!-- page 2 -->
Leveraging social media news
(c) Gaussian-specific (ours)
Exhaustive
Noisy 
Matches
(b) Structure-based
Limited
Precision
(a) Image-based
(d) Query Image
Dense
rasterization
Perceptual 
Aliasing
Depth
RGB
Landmarks
Figure 1: Different approaches for visual relocalization. (a)
Image-based: returns location tags from database through
image retrieval, or directly regresses 6-DoF pose from images.
(b) Structure-based: establishes correspondences between 2D
pixels in query images and 3D sparse SfM models, then
solves perspective projection optimization equations. (c) Our
proposed Gaussian-Specific approach from Sparse Landmarks
Sampling to Dense Rasterization Matching.
it encodes both geometric constraints and scene-specific
appearance features without relying on external visual priors
that may suffer from domain gaps. This makes it particularly
well-suited for remote sensing scenarios where traditional
visual features often fail. However, existing 3DGS-based
relocalization methods Sidorov et al. (2025); Zhai et al.
(2025) are designed for small-scale indoor scenes and lack
the specialized components needed for large-scale remote
sensing applications.
To bridge this gap, we introduce Hi2-GSLoc, a dual-
hierarchical relocalization framework specifically tailored
for remote sensing scenarios. Specifically, we first estimate
an initial pose by registering the input query image to
a 3D Gaussian sparse model based on consistent render-
aware Gaussian landmark sampling and landmark-guided
keypoint detection. Subsequently, we render dense Gaussian
rasterization outputs (including feature maps, RGB, and
depth) based on the initial pose, then employ coarse-to-fine
windowed probabilistic mutual matching with effective iter-
ative refinement to optimize the pose. Finally, consistency-
based pose validation filters outliers, achieving accurate and
robust relocalization in large-scale remote sensing scenes.
Our main contributions are:
(1) We introduce the first 3DGS-based relocalization frame-
work specifically designed for remote sensing scenarios.
Our Hi2-GSLoc employs a dual-hierarchical sparse-to-
dense and coarse-to-fine pipeline that integrates par-
titioned Gaussian training, GPU-accelerated parallel
matching, and dynamic memory management strategies
to efficiently handle large-scale remote sensing scenes.
(2) To address feature domain adaptation and depth ambi-
guity inherent in remote sensing scenarios, we propose
a consistent render-aware landmark sampling strategy
(C.R-A.S) coupled with a landmark-guided keypoint de-
tector (L-G.D) that fully exploits geometric constraints
and scene-specific representations embedded in Gaus-
sian primitives, enabling robust and accurate initial pose
estimation.
(3) We design an iterative dense refinement stage that
matches rendered Gaussian features with query features
through coarse-to-fine windowed probabilistic mutual
matching (PMM), coupled with a consistency-based
pose validation mechanism to filter unreliable estimates.
(4) Extensive experiments validate competitive localiza-
tion accuracy and recall rates with enhanced robustness
through reliable pose filtering, while maintaining com-
putational efficiency.
2. Related works
Visual relocalization research encompasses three pri-
mary paradigms: image-based relocalization that operates
solely on image information through retrieval or direct
pose regression, structure-based relocalization that lever-
ages explicit 3D scene geometry from SfM reconstruction,
and analysis-by-synthesis approaches that optimize camera
poses through rendering and comparison with query images.
2.1. Image-based relocalization
Image-based approaches operate exclusively on visual
information without relying on explicit 3D scene structure,
broadly categorized into retrieval-based localization and
regression-based pose estimation methods.
Retrieval-based methods achieve localization through
learned global descriptors. NetVLAD Arandjelovic et al.
(2016) pioneered this direction by extracting robust global
descriptors via contrastive learning for location retrieval.
Recent advances have leveraged foundation vision mod-
els Keetha et al. (2023); Lu et al. (2024, 2025); Wang
et al. (2025a), developed viewpoint-invariant representa-
tions Berton et al. (2023), and introduced comprehensive
frameworks that integrate diverse methods, training strate-
gies, and datasets Berton and Masone (2025), achieving sub-
stantial improvements. However, these approaches funda-
mentally depend on database image density and distribution,
potentially yielding significant localization errors in sparse
coverage scenarios. Moreover, most existing models are
predominantly trained on ground-level datasets with limited
aerial imagery, leading to substantial domain gaps when
applied to remote sensing scenarios.
Regression-based methods directly predict 6-DoF cam-
era poses from single images. PoseNet Kendall et al. (2015)
introduced the first CNN-based framework for end-to-end
pose regression. Subsequent improvements have incorpo-
rated temporal information Walch et al. (2016); Clark et al.
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 2 of 17

<!-- page 3 -->
Leveraging social media news
(2017), geometric losses and priors Kendall and Cipolla
(2017); Brahmbhatt et al. (2017), and photometric consis-
tency constraints Chen et al. (2021) to enhance pose accu-
racy. Despite outputting complete 6-DoF poses, these meth-
ods typically achieve performance comparable only to image
retrieval baselines Arandjelovic et al. (2016) and fall short
of structure-based approaches Zhao et al. (2024) in terms
of precision. Moreover, being inherently data-driven, they
exhibit significant performance degradation when applied to
domains outside their training distribution.
2.2. Structure-based relocalization
Structure-based relocalization methods Camposeco et al.
(2018); Taira et al. (2021); Li et al. (2020) leverage 3D
scene information reconstructed from SfM to establish 2D-
3D correspondences between query images and scenes, sub-
sequently employing Perspective-n-Point (PnP) Gao et al.
(2003); Ke and Roumeliotis (2017) solvers for camera pose
estimation. While these approaches fully exploit scene ge-
ometry to achieve high pose accuracy, they are susceptible
to noisy feature matches and computationally expensive for
large-scale scenes. Consequently, image retrieval methods
Arandjelovic et al. (2016); Berton et al. (2023) are typi-
cally applied as a preprocessing step to coarsely localize
the visible scene structure relative to query images Sarlin
et al. (2019); Taira et al. (2021), significantly reducing
localization time. Meanwhile, global image features contain-
ing semantic information enhance scene understanding and
improve system robustness. Subsequently, 3D point features
extracted from the scene image database are matched with
2D keypoint features from query images using identical al-
gorithms to establish 2D-3D correspondences DeTone et al.
(2018); Revaud et al. (2019); Dusmanu et al. (2019); Sarlin
et al. (2020); Lindenberger et al.; Sun et al. (2021); Jiang
et al. (2024). HLoc Sarlin et al. (2019) integrates diverse
global retrieval methods Arandjelovic et al. (2016); Berton
et al. (2023); Berton and Masone (2025), feature detectors
DeTone et al. (2018); Tyszkiewicz et al. (2020), and feature
matchers Sarlin et al. (2020); Lindenberger et al.; Jiang et al.
(2024) to enhance localization performance.
To mitigate outlier effects, recent advances such as
DSAC Brachmann et al. (2017); Brachmann and Rother
(2021) employ CNNs to predict scene coordinates and score
hypotheses while introducing differentiable RANSAC algo-
rithms. LoFTR Sun et al. (2021) and OmniGlue Jiang et al.
(2024) adopt detector-free matching and DINOv2 Oquab
et al. (2023) vision foundation models, respectively dedi-
cated to improving robustness under weak texture and large
viewpoint variations. Despite these advances, structure-
based methods are vulnerable to localization failures un-
der substantial viewpoint changes and suffer from compu-
tational bottlenecks during feature matching, particularly
challenging for large-scale scenarios.
2.3. Analysis-by-Synthesis
Analysis-by-synthesis methods optimize camera poses
by analyzing relationships between synthesized and query
images. These approaches function either as pose correspon-
dence modules or standalone relocalization frameworks, ef-
fectively addressing matching failures caused by large view-
point variations Chen et al. (2022, 2021). iNeRF Yen-Chen
et al. (2021) directly inverts NeRF models by iteratively
optimizing photometric differences between rendered and
query images to refine camera pose initialization. DirectPN
Chen et al. (2021) integrates NeRF to provide photometric
consistency supervision for pose regression by minimiz-
ing color discrepancies between query images and those
rendered from predicted poses. Dfnet Chen et al. (2022)
extends this concept by measuring consistency in feature
space, demonstrating enhanced localization performance.
PNeRFLoc Zhao et al. (2024) employs explicit point-based
neural representations to leverage geometric constraints and
perform 2D-3D feature matching for 6-DoF pose estima-
tion. However, practical applications remain limited due to
NeRF’s computationally expensive scene training and view
synthesis processes, as well as substantial memory require-
ments for storing descriptors and correspondence graphs
from sparse SfM models.
Compared to NeRF, 3DGS Kerbl et al. (2023) employs
explicit representations enabling fast, high-quality view
synthesis. Recent advances Feng et al. (2025); Wang et al.
(2024); Mallick et al. (2024) have demonstrated real-time,
high-fidelity rendering of large-scale scenes using 3DGS.
The latest analysis-by-synthesis methods Sidorov et al.
(2025); Zhai et al. (2025); Cheng et al. (2024) integrate
3DGS into relocalization pipelines, combining structure-
based coarse pose estimation with photometric rendering
optimization in unified end-to-end frameworks. However,
existing 3DGS-based relocalization methods exhibit sig-
nificant limitations for remote sensing applications. They
either neglect the rich 3D geometric information embed-
ded in Gaussian primitives Liu et al. (2025) or directly
adapt existing 2D image detectors without Gaussian-specific
optimization Sidorov et al. (2025), resulting in subopti-
mal performance in geometry-sensitive aerial scenarios.
While Huang et al. (2025) introduced a Gaussian scene-
specific detector that improved accuracy, it lacks the memory
optimization and scalability strategies essential for large-
scale remote sensing environments, limiting its practical
applicability.
3. Methodlogy
This section details our dual-hierarchical Gaussian-
based relocalization framework, which comprises four core
components: (1) 3D Gaussian Splatting foundations and
adaptations for remote sensing scenarios, (2) Consistent
render-aware Gaussian landmark sampling, (3) Landmark-
guided keypoint detection, and (4) Coarse-to-fine dense
pose refinement and validation. The complete pipeline is
illustrated in Figure 2.
3.1. 3D Gaussian splatting for remote sensing
3DGS Kerbl et al. (2023) represents scenes using mil-
lions of 3D Gaussians—colored ellipsoids with transparency
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 3 of 17

<!-- page 4 -->
Leveraging social media news
Gaussian 
Splatting
Gaussian 
Landmarks
Feature 
Extracter
① 
Initial
Pose
② 
Refine
Pose 
2d-3d
Pose 
2d-2d
Refine
Pose 
2d-2d
③  
Final 
Pose
Database
Query
Feature 
Depth 
Landmark
-Guided
Detector
√
√
√√√
Consistent 
Render-Aware 
Sampling
Rendering
PnP
PMM
PMM
1/઀
Coarse to Fine 
Feature 
Image
Sparse to Dense
Probabilistic 
Mutual Matching
Keypoints 
ℳ
੟ℳ
ℳե
ℳե
ե
ℳե
ձ
1/઀
઀×઀
઀×઀
Consistency 
verification
Figure 2: Overview of our Hi𝟐-GSLoc pipeline. The method consists of three stages: (1) initial pose estimation through consistent
render-aware landmark sampling and landmark-guided keypoint detection, (2) pose optimization via dense rasterization and
coarse-to-fine iterative matching, and (3) consistency-based verification to filter unreliable results.
that decays according to a Gaussian distribution from their
centers. The method initially employs SfM to estimate cam-
era poses and generate sparse point clouds, which are subse-
quently transformed into initial 3D Gaussians. These Gaus-
sians undergo optimization via Stochastic Gradient Descent
(SGD) with adaptive density control, dynamically adding
and removing ellipsoids based on gradient magnitudes and
predefined criteria to achieve compact, unstructured scene
representations. The framework employs tile-based raster-
ization for efficient real-time rendering of photorealistic
scenes.
Our approach integrates consistent render-aware sam-
pling strategy and landmark-guided keypoint detector with
3DGS, embedding Gaussian features into 3D representa-
tions to enhance relocalization accuracy. Specifically, our
scene representation comprises original Gaussian primitives
augmented with feature fields. The trainable attributes of the
i-th Gaussian primitive include center (𝑥𝑖, 𝑦𝑖, 𝑧𝑖), rotation
𝑞𝑖, scale 𝑠𝑖, opacity 𝛼𝑖, color 𝑐𝑖, and feature 𝑓𝑖, denote as
Θ𝑖= {(𝑥𝑖, 𝑦𝑖, 𝑧𝑖), 𝑞𝑖, 𝑠𝑖, 𝛼𝑖, 𝑐𝑖, 𝑓𝑖}. To address challenges in
remote sensing large-scale applications including memory
constraints, extensive optimization time, and appearance
variations, we adopt a partitioning strategy from VastGaus-
sian Lin et al. (2024). As shown in the left of Figure 3,
large scenes are divided into multiple cells using progres-
sive partitioning, where point clouds and training views
are allocated to these cells for parallel optimization before
seamless merging. Each cell contains fewer 3D Gaussians,
enabling optimization within limited memory constraints
and reducing training time through parallelization.
The training process follows Feature-3DGS Zhou et al.
(2023), jointly optimizing radiance and feature fields, as
illustrated in the right of Figure 3. Color attributes 𝑐are
rasterized into rendered RGB images 𝐼𝑟using alpha blend-
ing, while feature attributes 𝑓are rendered into feature maps
̄𝐹𝑟through identical rasterization. The ground truth dense
feature map extract from the training image 𝐼∈ℝ3×𝐻×𝑊
is denoted as 𝐹𝑡(𝐼) ∈ℝ𝐷×𝐻′×𝑊′, where 𝐷represents the
dense feature dimensionality. 𝐹𝑡(𝐼) and query feature maps
are both obtained using standard local feature extractors
DeTone et al. (2018); Revaud et al. (2019). The overall
training loss combines radiance field loss 𝑟𝑔𝑏and feature
field loss 𝑓:
= 𝛼𝑓+ 𝛽𝑟𝑔𝑏.
(1)
The feature field loss 𝑓computes the L1 norm loss between
ground truth feature maps 𝐹𝑡(𝐼) and rendered feature maps
̄𝐹𝑟:
𝑓= 1
𝑁
𝑁
∑
𝑖=1
| ̄𝐹𝑟
𝑖−𝐹𝑡
𝑖(𝐼)|.
(2)
The radiance field loss 𝑟𝑔𝑏comprises L1 loss between
ground truth images 𝐼and appearance-varied rendered im-
ages 𝐼𝑎, and 𝐷−𝑆𝑆𝐼𝑀loss between directly rendered im-
ages 𝐼𝑟:
𝑟𝑔𝑏= (1 −𝜆) 1
𝑁
𝑁
∑
𝑖=1
|𝐼𝑖, 𝐼𝑎
𝑖| + 𝜆𝐷−𝑆𝑆𝐼𝑀(𝐼, 𝐼𝑟), (3)
where 𝐷−𝑆𝑆𝐼𝑀denotes the D-SSIM loss Kerbl et al.
(2023), which penalizes structural differences to align struc-
tural information in 𝐼𝑟with 𝐼while L1 loss between
appearance-varied rendering 𝐼𝑎and 𝐼fits ground truth
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 4 of 17

<!-- page 5 -->
Leveraging social media news
ℒ1
ℒէ−֣֣ջ֋
Ƴ՝
Ƴ
Decoupled appearance modeling
Dense Feature 
Extractor
ư֩(Ƴ)
ℒձ
Progressive partitioning
training strategy 
Ƴ֡
ư֡
Rasterization
Figure 3: 3D feature Gaussian splatting of remote sensing and
the training process jointly optimize 𝑟𝑔𝑏and 𝑓.
images that may exhibit appearance variations relative to
other images. After training, 𝐼𝑟achieves consistent appear-
ance across views, enabling 3D Gaussians to learn averaged
appearance and correct geometry from all input views. The
complete Feature Gaussian scene obtained from this training
process is denoted as .
3.2. Consistent render-aware sampling
Exhaustive matching against all Gaussians in a 3DGS
model is computationally intensive, and this challenge be-
comes even more severe in large-scale remote sensing sce-
narios. Additionally, irrelevant points and Gaussians can
easily produce noisy correspondences, degrading localiza-
tion accuracy. To address this issue, traditional structure-
based methods select keypoint-like landmarks through 2D
features (e.g., corners, edges, and semantic descriptors).
SceneSqueezer Yang et al. (2022) and DetectLandmarks Do
and Sinha (2024) employ differentiable optimization or learn
point importance to reduce map points. SplatLoc Zhai et al.
(2025) obtains Gaussian landmarks by learning saliency
probability scores of primitives.
In contrast to these landmark selection methods, our
approach incorporates visibility, semantic, and geometric
constraints throughout the Gaussian rendering process to
ensure robust feature matching across different viewpoints.
We design batch processing with dynamic memory man-
agement to address computational bottlenecks in large-scale
scenarios. As illustrated in Figure 4, we first assign saliency
scores to each Gaussian primitive by perceiving visibility
and semantic features during rendering, then batch-process
spatial nearest neighbor groups to select the highest-scoring
primitives as landmarks.
Significance scoring. Traditional methods are “feature-
driven”: they first detect feature points and then search for
matches. Our method is “geometry-driven”: it establishes
correspondences based on 3D geometry and then evaluates
feature quality scores. This provides strong geometry pri-
ors that ensure correspondences are spatially coherent and
physically plausible, reducing the likelihood of outliers and
improving overall matching reliability.
The scoring process is illustrated in the left panel of
Figure 4. Each camera viewpoint 𝑖corresponds to a training
image 𝐼and a set of visible Gaussian primitives 𝐺𝑖. Follow-
ing a rigorous stereo geometric coordinate transformation
pipeline, we compute the transformation of visible Gaussian
primitives 𝐺𝑖from world coordinates (𝑋, 𝑌, 𝑍) to camera
coordinates and then to pixel coordinates (𝑈′, 𝑉′) ∈𝐼,
thereby obtaining image features 𝐹𝑡(𝑈′, 𝑉′) corresponding
to the visible Gaussian primitives features 𝐹𝐺𝑖. The matching
score 𝑆(𝐺𝑖) of one camera viewpoint 𝑖is computed as the
cosine similarity between the extracted 2D image features
𝐹𝑡(𝑈′, 𝑉′) at corresponding positions and the Gaussian
features 𝐹𝐺𝑖:
𝑆(𝐺𝑖) =
𝐹𝐺𝑖⋅𝐹𝑡(𝑈′, 𝑉′)
||𝐹𝐺𝑖||2 × ||𝐹𝑡(𝑈′, 𝑉′)||2
.
(4)
By performing the above operations for each viewpoint, we
obtain the visibility count of each Gaussian across different
viewpoints, along with the corresponding feature similarity
scores. The final similarity score is obtained by averaging
across all viewpoints. The total score 𝑆() of all Gaussian
primitives across all viewpoints is:
𝑆() =
𝑛
∑
𝑖=1
𝑆(𝐺𝑖).
(5)
For the 𝑗𝑡ℎGaussian 𝑔𝑗, its final significance score 𝑆(𝑔𝑗) is
computed as the average of the total score 𝑆() and visibility
count 𝑀:
𝑆(𝑔𝑗) = 1
𝑀𝑆(𝑗).
(6)
By selecting Gaussian landmarks based on these scores, we
can ensure they are easily identifiable and matchable across
different viewpoints.
Render gradient visibility check.
Existing methods
for matching and landmark selection rely solely on 2D
features, ignoring 3D information Leroy et al. (2024). Unlike
approaches that determine visibility based only on explicit
depth information, we analyze the complete rendering pro-
cess of Gaussian explicit neural fields. We consider depth oc-
clusion (points occluded by other Gaussian points), opacity
(visibility of semi-transparent Gaussian points), rendering
weights (actual contribution to the final rendering), and
other factors to determine point visibility. This enables us to
obtain Gaussian landmarks that are easily identifiable across
different viewpoints. Specifically, this is based on gradient
determination during the backpropagation process—only
Gaussian primitives that contribute to the final rendered im-
age receive gradients during backpropagation. The detailed
procedure is shown in Algorithm 1, where we obtain the
final image coordinates (𝑈′, 𝑉′) corresponding to visible
Gaussians (𝑋, 𝑌, 𝑍) based on rendering visibility and image
projection bounds.
Memory-efficient sampling. Higher feature similarity
scores indicate that the corresponding Gaussian features are
more suitable for matching. However, texture-rich regions
tend to have higher Gaussian density, so selecting features
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 5 of 17

<!-- page 6 -->
Leveraging social media news
Dense Feature 
Extractor
Gaussian
 Feature
GT
 Feature
ǜ, ǝ, Ǟ
Ǚ, ǚ
Ǚ’, ǚ’
similarity
score
Render gradient 
visibility check
visibility-aware
semantic-aware
k-NN distance
initial sample
final sample
unsampled
low score
high score
spatial-aware
Significance Scoring 
1֩ℎ batch
ǒ֩ℎ batch
......
Memory-efficient 
sampling
Figure 4: Consistent render-aware sampling. From left to right: significance scoring and memory-efficient sampling strategy
based on scores. The entire process incorporates feature and visibility constraints during Gaussian rendering, and spatial distance
constraints between Gaussian points.
Algorithm 1 Render gradient visibility check with projec-
tion filtering
Input: Gaussian model , Camera pose 𝑇𝑤𝑐, Intrinsic 𝐾,
Image 𝐼∈ℝ3×𝐻×𝑊
Output: Visible (𝑈′, 𝑉′)
1: Extract Gaussian parameters: (𝑋, 𝑌, 𝑍), 𝛼, 𝑞, 𝑐, 𝑠
2: 𝑅𝐺𝐵←𝑟𝑎𝑠𝑡𝑒𝑟𝑖𝑧𝑎𝑡𝑖𝑜𝑛((𝑋, 𝑌, 𝑍), 𝛼, 𝑞, 𝑐, 𝑠, 𝐾, 𝑇𝑤𝑐)
3: 𝑅𝐺𝐵.𝑠𝑢𝑚().𝑏𝑎𝑐𝑘𝑤𝑎𝑟𝑑()
4: for 𝑗∈[1, 𝑁] do
5:
𝑀𝑟[𝑗] ←(‖∇(𝑋, 𝑌, 𝑍)[𝑗]‖ > 0)
6: end for
7: (𝑋, 𝑌, 𝑍).𝑔𝑟𝑎𝑑.𝑧𝑒𝑟𝑜_()
8: // Project Gaussians to image space
9: (𝑋, 𝑌, 𝑍)ℎ𝑜𝑚𝑜←[(𝑋, 𝑌, 𝑍), 𝟏]
10: (𝑋, 𝑌, 𝑍)𝑐𝑎𝑚←(𝑇𝑊𝐶× (𝑋, 𝑌, 𝑍)𝑇
ℎ𝑜𝑚𝑜)[∶3]
11: 𝑑𝑒𝑝𝑡ℎ𝑠∶𝑑←(𝑋, 𝑌, 𝑍)𝑐𝑎𝑚[2]
12: // Perspective division
13: (𝑋, 𝑌, 𝑍)𝑐𝑎𝑚_ℎ𝑜𝑚𝑜←(𝑋, 𝑌, 𝑍)𝑐𝑎𝑚∕𝑑
14: // Project to pixel coordinates
15: (𝑈, 𝑉) ←(𝐾× (𝑋, 𝑌, 𝑍)𝑐𝑎𝑚_ℎ𝑜𝑚𝑜)[∶2]
16: // Boundary and visibility filtering
17: 𝑀𝑖
←
((𝑈, 𝑉)[0]
≥
0) ∧((𝑈, 𝑉)[0]
<
𝑊) ∧
((𝑈, 𝑉)[1] ≥0) ∧((𝑈, 𝑉)[1] < 𝐻)
18: 𝑀←𝑀𝑖∧𝑀𝑟
19: (𝑈′, 𝑉′) ←(𝑈, 𝑉)[∶, 𝑀]
20: return (𝑈′, 𝑉′)
based solely on scores may lead to insufficient coverage in
other regions, particularly problematic in remote sensing
images containing large areas of ground and vegetation.
To ensure uniform landmark distribution across the entire
scene, we employ a two stage selection strategy. We first
obtain initial samples 𝕃𝑜= {𝑙1, 𝑙2, ..., 𝑙𝑄} through random
sampling, where 𝑄is the number of samples. Then, we con-
duct score-based competition within the spatial K-nearest
neighbors (kNN) of each initial sample to derive the final
landmarks 𝕃. The final selected landmarks are determined
by:
𝐿= {𝑙∗
𝑖∣𝑙∗
𝑖= arg max
𝑔∈𝑁𝑘(𝑙𝑖) S(𝑔), ∀𝑙𝑖∈𝐿𝑜},
(7)
where 𝑁𝑘(𝑙𝑖)
=
{𝑔
∈

∶
‖𝑔−𝑙𝑖‖
≤
𝑟𝑖} is the
kNN neighborhood of a initial gaussian sample 𝑙𝑖, 𝑟𝑖is the
neighborhood search radius, and 𝑆(𝑔) is the significance
score of each Gaussian 𝑔from Eq.6.
To address memory constraints in large-scale remote
sensing scenarios, we partition the Gaussian processing
into manageable batches and implement dynamic memory
management that actively releases intermediate computation
results after each batch, ensuring efficient memory utiliza-
tion.
3.3. Landmark-guided detector
Directly matching dense feature maps with sampled
landmarks is infeasible, as dense feature maps contain
numerous position-irrelevant and unsuitable redundant fea-
tures for matching. GSplatLoc Sidorov et al. (2025) di-
rectly uses cosine similarity between image features ex-
tracted by existing 2D image detectors (XFeat Potje et al.
(2024)) and Gaussian features based on XFeat distillation
to obtain matching relationships, without considering 3D
information in the Gaussian model or retraining for Gaus-
sian scenes, resulting in significantly reduced accuracy in
remote sensing scenarios. Moreover, off-the-shelf detectors
Sun et al. (2021); DeTone et al. (2018); Dusmanu et al.
(2019); Revaud et al. (2019) typically detect scene-agnostic
predefined keypoints, making them unsuitable for matching
with sampled landmarks in feature Gaussian scenes. To
address this problem, we train a Gaussian-specific landmark-
guided detector that can process feature maps 𝐹𝑡(𝐼) and
generate a probability map 𝐸(𝐼) ∈ℝ1×𝐻×𝑊, representing
the probability of 2D features being landmarks. The network
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 6 of 17

<!-- page 7 -->
Leveraging social media news
architecture and training pipeline are illustrated in the Figure
5. Specifically, our detector 𝐷𝜃(𝐹𝑡(𝐼)) is a shallow CNN
appended after existing feature extractors DeTone et al.
(2018); Potje et al. (2024), where 𝜃represents the network
parameters.
The training process is conducted in a self-supervised
manner, leveraging 3D geometric constraints to enhance the
quality and consistency of 2D feature point detection. Simi-
lar to determining robust matching points visible across dif-
ferent viewpoints through render gradient visibility checks,
our detector aims to detect Gaussian points that are render-
visible in the current image viewpoint. Specifically, we
project the center of each Gaussian from the selected Gaus-
sian landmarks onto the current image plane and obtain
ground truth Gaussian matching points for the current view-
point based on rendering visibility. We then use binary cross-
entropy loss to optimize the detector 𝐷𝜃:
𝑑𝑒𝑡(𝐸(𝐼), 𝐸𝐺𝑇) = −
1
𝐻× 𝑊
𝐻
∑
ℎ=1
𝑊
∑
𝑤=1
[𝐸𝐺𝑇
ℎ,𝑤log(𝐸(𝐼)ℎ,𝑤) + (1 −𝐸𝐺𝑇
ℎ,𝑤) log(1 −𝐸(𝐼)ℎ,𝑤)]
(8)
During inference, non-maximum suppression (NMS) is
applied to the output probability map of the trained detector
to ensure uniform distribution of detected keypoints. The
final detected keypoints can be represented as follows:
= {(𝑢𝑖, 𝑣𝑖) ∣𝐸(𝑢𝑖, 𝑣𝑖) > 𝜏& 𝐸(𝑢𝑖, 𝑣𝑖) =
max
𝑟(𝑢𝑖,𝑣𝑖) 𝐸(𝑢, 𝑣)},
(9)
where represents the final set of detected keypoints, (𝑢𝑖, 𝑣𝑖)
are the pixel coordinates of keypoints, 𝜏is the confidence
threshold, 𝑟is the NMS suppression radius, and 𝑟(𝑢𝑖, 𝑣𝑖)
denotes the neighborhood of radius 𝑟centered at (𝑢𝑖, 𝑣𝑖).
The entire pipeline follows rigorous stereo geometric
coordinate transformations to generate supervision informa-
tion, embodying the “3D-guides-2D” philosophy. Therefore,
the detected keypoints and subsequently established matches
possess more accurate geometric relationships, ensuring pre-
cise localization even in scale-sensitive scenarios such as
remote sensing.
3.4. Dense rasterization matching
Using the 3D landmarks obtained from C.R-A.S and the
2D keypoints detected by the L-G.D, we establish the top-𝑘
2D-3D correspondences based on cosine similarity between
features and solve PnP to obtain the initial pose:
{𝑅∗, 𝑡∗}, ∗= arg min
𝑅,𝑡
∑
𝑖∈
𝜌(‖𝑝2𝐷
𝑖
−𝜋(𝐾[𝑅|𝑡] ̃
𝑔3𝐷
𝑖
)‖2, 𝜏),
(10)
where 𝜌(⋅, 𝜏) is the robust loss function with threshold 𝜏(re-
projection error), 𝜋(⋅) is the projection function: 𝜋([𝑥, 𝑦, 𝑧]𝑇) =
[𝑥∕𝑧, 𝑦∕𝑧]𝑇, ∗is the optimal inlier set, 𝐾is the camera
ℒթխ֩
256×H×W
128×H×W
64×H×W
32×H×W
Conv+SiLu
Conv
Sigmoid
Feat. Extractor
Landmark-Guided Detector
Render gradient 
visibility check
Sampled 
Landmarks
Projection
H×W
H×W
H×W
H×W
Training
Ư(Ƴ)
Ưճ֧
Figure 5: Lnadmark-guided detector. Training 2D image
keypoints detector guided by sampled 3D Gaussian landmarks.
intrinsic matrix, 𝑝2𝐷
𝑖
∈ℝ2 is the 2D point on the query
image plane 𝑔3𝐷
𝑖
∈ℝ3 is the 3D Gaussian point in world
coordinates. As shown in Figure 2, based on the initial pose,
we can render dense feature and depth maps from the com-
plete Feature Gaussian scene , and then iteratively optimize
the pose through coarse-to-fine dense feature matching.
Coarse to fine pose refinement. To improve computa-
tional efficiency, we first perform matching on coarse query
and rendered feature maps, reducing the search space from
𝑂(𝐻𝑓× 𝑊𝑓× 𝐻𝑓× 𝑊𝑓) to 𝑂(𝐻𝑐× 𝑊𝑐× 𝐻𝑐× 𝑊𝑐). In
this work, we set 𝐻𝑓∕𝐻𝑐= 8, thereby achieving a 4096-
fold reduction in search space while avoiding the storage
of massive matrices generated by high-resolution image
processing in remote sensing.
Subsequently, we perform sliding window-based sub-
pixel level matching on fine query and rendered feature
maps. This approach ensures accurate pose estimation pre-
cision while employing parallel GPU computation across
windows, significantly improving efficiency. The window
size is 𝑤
=
𝐻𝑓∕𝐻𝑐, which adaptively accommodates
different resolutions. Finally, we fully leverage the depth
information from Gaussian rendering for 3D constraints. The
specific solution is based on 2D-3D PnP algorithm with
RANSAC, as described in Eq. 10.
Consistency verification. To prevent large errors in
initial pose estimation that could cause significant view-
point differences between dense rasterization and query
views, we iteratively execute 𝑛rounds of rendering and
coarse-to-fine pose optimization with the optimized dense
pose, and perform consistency pose filtering by checking
angular differences across multiple results. In practice,
𝑛is set to 3, as computed in Algorithm. 2, “trace
←
min(3.0, max(trace(𝑅𝑟𝑒𝑙), −1.0))” is used to prevent compu-
tational collapse due to rounding errors. During a single pose
computation, if we detect pose inconsistency between any
two coarse-to-fine iterative dense pose calculations—i.e.,
angular difference exceeding threshold 𝜏= 20◦—we con-
sider the result unreliable and directly skip to compute the
next query, ensuring the localization system is not affected
by erroneous results under extreme conditions.
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 7 of 17

<!-- page 8 -->
Leveraging social media news
Algorithm 2 Pose Consistency Verification
Input: Dense pose results {𝑇𝑖}𝑛
𝑖=1, Threshold 𝜏= 20◦
Output: Final pose or Status
1: // Compute Pose Difference
2: Function Ψ(𝑇1, 𝑇2):
3:
𝑅1 ←𝑇1[∶3, ∶3], 𝑅2 ←𝑇2[∶3, ∶3]
4:
𝑡1 ←𝑇1[∶3, 3], 𝑡2 ←𝑇2[∶3, 3]
5:
𝑅𝑟𝑒𝑙←𝑅1 × 𝑅𝑇
2
6:
// Numerical Stabilization
7:
𝑡𝑟𝑎𝑐𝑒←𝑚𝑖𝑛(3.0, 𝑚𝑎𝑥(trace(𝑅𝑟𝑒𝑙), −1.0))
8:
𝜃𝑑𝑖𝑓𝑓←180
𝜋arccos
(
𝑡𝑟𝑎𝑐𝑒−1
2
)
9:
𝑑𝑡𝑟𝑎𝑛𝑠←‖𝑡1 −𝑡2‖2
10:
return 𝜃𝑑𝑖𝑓𝑓, 𝑑𝑡𝑟𝑎𝑛𝑠
11: EndFunction
12: // Main consistency verification loop
13: for 𝑖= 1 to 𝑛−1 do
14:
𝜃𝑑𝑖𝑓𝑓, 𝑑𝑡𝑟𝑎𝑛𝑠←Ψ(𝑇1, 𝑇2)
15:
if 𝜃𝑑𝑖𝑓𝑓> 𝜏then
16:
return “unreliable”
17:
end if
18: end for
19: return final_pose: 𝑇𝑛
Probabilistic mutual matching. During the dense ras-
terization matching process, relying solely on cosine simi-
larity can easily produce many-to-one and one-to-many mis-
matches. Therefore, we design probabilistic mutual match-
ing. As shown in Figure 2, we compute the cosine similarity
between the query feature map and rendered feature map to
obtain matrix , then calculate bidirectional softmax to get
the mutually constrained probability matrix :
=
exp(∕𝜏)
∑
𝑗exp(𝑖𝑗∕𝜏) ⊙
(
exp(𝑇∕𝜏)
∑
𝑖exp(𝑗𝑖∕𝜏)
)𝑇
(11)
where 𝜏is the temperature parameter that can adjust the
retention of matching relationships with different confidence
levels. Finally, we apply mutual nearest neighbor (MNN)
search on to establish correspondences 𝑐. The entire
coarse-to-fine matching process executes the above opera-
tions, where fine feature map matching 𝑓
𝑐is generated
from 𝑤×𝑤windows extracted at each position of the coarse
matching. This operation significantly enhances the quality
of dense matching 𝑐
𝑐and 𝑓
𝑐, leading to more accurate
pose estimation.
4. Experiments and analysis
In this section, we first introduce the benchmark datasets
and a remote sensing dataset we collected, then outline the
experimental settings and evaluation metrics. Finally, we
provide detailed analysis of performance comparisons be-
tween our proposed method and other existing approaches,
along with ablation studies.
4.1. Datasets
To comprehensively evaluate the effectiveness and ro-
bustness of our proposed localization methods, we conduct
extensive experiments across three categories of datasets
representing different scales and deployment scenarios.
Standard outdoor localization dataset. We first eval-
uate our method against state-of-the-art approaches using
the widely adopted Cambridge Landmarks dataset Kendall
et al. (2015), which comprises five outdoor scenes captured
with mobile phones. This dataset presents typical visual
localization challenges including dynamic object occlusion,
illumination variations, and motion blur.
Large-scale aerial dataset. For large-scale scenarios,
we utilize the Mill 19-Rubble dataset Turki et al. (2022),
which provides extensive aerial imagery suitable for eval-
uating UAV localization algorithms under challenging real-
world conditions.
Xi-MSTS. To evaluate algorithm performance across
varied deployment conditions and validate the effectiveness
across different data modalities, we construct the Xi-MSTS
(Xi’an Multi-Scene Temporal Sensing) dataset, which com-
prises both real-world and synthetic scenarios. Specifically,
the dataset includes three real-world scenes captured within
Xi’an, China: Village, Construction and Campus, spanning
multiple years (2016-2020) with significant heterogene-
ity in spatial scales, terrain characteristics, and imaging
conditions. Additionally, we include one synthetic scene
(Hills-UE4) generated using Unreal Engine 4 to assess
algorithm generalization in different applications. The real-
world scenes are captured using different UAV platforms
and camera configurations, with high-precision ground truth
poses obtained through RTK-GPS measurements, ensuring
centimeter-level positioning accuracy. The synthetic scene
provides controlled experimental conditions with known
ground truth. Figure 6 presents representative samples from
Xi-MSTS, and Table 1 provides detailed statistics and tech-
nical specifications. This diverse dataset, encompassing both
real and synthetic environments, enables comprehensive
evaluation of algorithm robustness and cross-domain gen-
eralization capability across various deployment scenarios.
4.2. Implementation details
Our training configuration follows VastGaussian Lin
et al. (2024) with modifications for feature learning. The
feature field is trained with a learning rate of 0.001, following
Feature 3DGS Zhou et al. (2023). All scenes undergo
training for 30,000 iterations. The densification process is
scheduled from iteration 500 to 20,000 with an interval of
500 iterations. This progressive densification allows adap-
tive scene representation refinement while maintaining train-
ing stability. To manage computational complexity while
preserving essential details, we employ different resolution
settings: Xi-MSTS-Village is trained at 1/4 resolution, while
other Xi-MSTS scenes, Cambrideg Landmarks and Mill 19-
Rubble are trained at 1/2 resolution. For sparse matching,
we extract 16,384 landmarks per scene to ensure sufficient
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 8 of 17

<!-- page 9 -->
Leveraging social media news
Table 1
Dataset summary.
Datasets
Numbers Resolution
View
Height/𝑚Sensor
Area/𝑘𝑚2 Acquisition time
Cambridge Landmarks
4991
1920 × 1080 Ground-view
None
Phone
0.013
2015
Mill 19-Rubble
1678
4608 × 3456 Oblique photography Low
None
None
2022
Xi-MSTS: Hills-UE4
538
1920 × 1080 Terrain following
None
AirSim
0.2540
2025
Xi-MSTS: Village
709
6475 × 3906 Vertical photography 830-845
DJI
1.0056
2020-0512-6pm.
Xi-MSTS: Campus
457
1920 × 1080 Vertical photography 485
DJI
1.0735
2016-01-26
Xi-MSTS: Construction
533
5472 × 3648 Vertical photography 626-647
Hasselblad 0.8332
2019-09-12-11am.
Table 2
Quantitative comparison of various advanced methods on the Cambridge Landmarks dataset. The results are shown below, with
red indicating the best performance and blue indicating the second best.
Methods
Kings
Hospital
Shop
Church
Avg.↓[cm/◦]
Imaged-
based
PoseNet
166/4.86
262/4.90
141/7.18
245/7.95
204/6.23
MS-Transformer
83/1.47
181/2.39
86/3.07
162/3.99
128/2.73
Learn-𝜃2 PN
99/1.06
217/2.94
105/3.97
149/3.43
143/2.85
LSTM PN
99/3.65
151/4.29
118/7.44
152/6.68
130/5.51
Geo. PN
88/1.04
320/3.29
88/3.78
157/3.32
163/2.86
Structure-
based
SIFT
13/0.22
20/0.36
4.0/0.21
8.0/0.25
11.25/0.26
HSCNet
18/0.30
19/0.30
6/0.30
9.0/0.30
13.0/0.30
HLoc (SP +SG)
11/0.20
15.1/0.31
4.2/0.20
7.0/0.22
9.3/0.23
DSAC*
17.9/0.31
21.1/0.40
5.2/0.24
15.4/0.51
14.9/0.37
Analysis
-by-
synthesis
Dfnet
43/0.87
46/0.87
16/0.59
50/1.49
39/0.96
NeRFMatch
12.5/0.23
20.9/0.38
8.4/0.40
10.9/0.35
13.2/0.34
PNeRFLoc
24/0.29
28/0.37
6.0/0.27
40/0.55
24.5/0.37
CROSSFIRE
47/0.7
43/0.7
20.0/1.2
39/1.4
37.3/1.00
GSplatLoc
31/0.49
16/0.68
4.0/0.34
14/0.42
16.25/0.49
Hi2-GSLoc
14.6/0.15
11.5/0.21
2.9/0.12
4.6/0.13
8.4/0.15
(a) Xi-MSTS: Hills-UE4
(c) Xi-MSTS: Campus
(b) Xi-MSTS: Village
(d) Xi-MSTS: Construction
(e) Mill 19-Rubble
Figure 6: Representative image samples from five diverse
scenes in the Xi-MSTS dataset and Mill 19-Rubble, showcasing
significant heterogeneity in spatial scales, terrain characteris-
tics, and imaging conditions.
spatial coverage for relocalization. And the Landmark-
Guided keypoint detector is trained for 30,000 iterations
using a learning rate of 0.001 with cosine decay scheduling.
This learning rate schedule ensures stable convergence while
preventing overfitting to specific scenes. All experiments are
conducted on a single RTX 3090 GPU. Training times are
approximately 150 minutes or less for Feature Gaussian op-
timization and under 50 minutes for scene-specific detector
training per scene, demonstrating the practical efficiency of
our approach.
4.3. Evaluation metric
We employ two complementary metrics to comprehen-
sively evaluate localization performance. The median lo-
calization error quantifies both translational and rotational
accuracy: translational error (TE) measures the Euclidean
distance between ground truth and estimated camera posi-
tions, while angular error (AE) captures the angular devi-
ation between ground truth and predicted camera orienta-
tions. The localization recall rate represents the percentage
of test images successfully localized within predefined error
thresholds. Specifically, an image is considered success-
fully localized when both translational and rotational errors
fall below specified tolerance levels simultaneously. These
metrics collectively provide a comprehensive assessment
of typical accuracy (via median error) and overall system
reliability (via recall rate) for each evaluated method.
4.4. Relocalization Benchmark
To validate the effectiveness of our proposed method
Hi2-GSLoc, we compare it with state-of-the-art methods
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 9 of 17

<!-- page 10 -->
Leveraging social media news
Table 3
Quantitative comparison of state-of-the-art methods on various kind of remote sensing dataset with SfM ground truth. The results
are shown below, with red and blue indicating the best and second-best performance across our unfiltered estimates and other
methods.
method
AE ↓
TE ↓
500/10◦↑
200/5◦↑
5/5◦↑
2/2◦↑
Inference/s ↓
SP+SG
4.4167
66.9698
50.3
50.25
1.01
0.00
11.9648
disk+LG
2.1353
49.1413
50.3
50.25
0.00
0.00
33.2160
NetVLAD+disk+LG
1.9572
30.237
90.00
90.00
0.00
0.00
4.8623
MegaLoc+disk+SG
1.9385
28.4464
100.0
100.0
0.00
0.00
4.9951
Eigenplaces+disk+LG
2.1647
49.1749
50.3
50.25
0.00
0.00
4.8997
GSplatLoc
108.34
959.34
5.23
0.65
0.00
0.00
8.232
ours
0.0128
0.1021
93.87
93.87
93.87
93.87
2.4689
Mill 19-Rubble
ours(final)
0.0119
0.0997
100.0
100.0
100.0
100.0
0.00024
SP+SG
1.2779
30.7426
58.4
57.83
0.00
0.00
10.1121
disk+LG
8.8689
89.0069
55.1
0.00
0.00
0.00
22.5649
NetVLAD+disk+LG
7.4086
69.4838
56.2
0.00
0.00
0.00
3.3210
MegaLoc+disk+SG
4.1653
33.5057
90.7
89.71
0.00
0.00
3.4698
Eigenplaces+disk+LG
8.2131
95.8683
47.6
0.00
0.00
0.00
3.0021
GSplatLoc
95.434
453.334
8.5
6.5
3.3
2.6
8.213
ours
0.1062
0.4552
98.19
98.19
98.19
98.19
1.1209
Hills-UE4
ours(final)
0.1050
0.4574
100.0
100.0
100.0
100.0
0.00025
SP+SG
3.1137
8.9276
100.0
100.0
19.66
0.00
13.7490
disk+LG
1.2339
3.8313
100.0
100.0
60.11
26.96
38.5623
NetVLAD+disk+LG
1.1983
3.5315
100.0
100.0
59.55
29.21
5.4126
MegaLoc+disk+SG
1.2304
3.6837
100.0
100.0
58.43
33.14
5.6213
Eigenplaces+disk+LG
1.2581
3.7340
100.0
100.0
58.98
30.89
5.6379
GSplatLoc
3.2028
313.9821
75.42
20.00
2.85
2.85
8.562
ours
0.0291
0.1456
100.0
100.0
100.0
100.0
5.0126
Construction
ours(final)
0.0291
0.1456
100.0
100.0
100.0
100.0
0.00028
SP+SG
3.2101
19.8647
100.0
95.63
0.00
0.00
10.0601
disk+LG
10.0266
41.5737
49.8
0.00
0.00
0.00
21.3221
NetVLAD+disk+LG
10.0365
41.3977
49.3
0.00
0.00
0.00
3.0146
MegaLoc+disk+SG
10.0564
41.0556
47.6
0.00
0.00
0.00
3.2234
Eigenplaces+disk+LG
10.0737
41.5415
48.0
0.00
0.00
0.00
3.0126
GSplatLoc
73.8292
447.6325
31.37
26.14
9.81
1.96
8.2341
ours
0.0377
0.1578
98.69
98.69
98.69
98.69
1.5514
Campus
ours(final)
0.0362
0.1552
100.0
100.0
100.0
100.0
0.00026
on the widely-used Cambridge Landmarks Dataset Kendall
et al. (2015) for outdoor localization. As shown in Ta-
ble 2, we select fourteen representative methods across
three categories for comparison: five image-based methods
(PoseNet Kendall et al. (2015), MS-Transformer Shavit et al.
(2021), Learn-𝜃2 PN Kendall and Cipolla (2017), LSTM
PN Walch et al. (2016), and Geo. PN Kendall and Cipolla
(2017)), four structure-based methods (SIFT, HSCNet Li
et al. (2020), HLoc Sarlin et al. (2019) (SuperPoint DeTone
et al. (2018) + SuperGlue Sarlin et al. (2020)), and DSAC*
Brachmann et al. (2017)), and five analysis-by-synthesis
methods (Dfnet Chen et al. (2022), NeRFMatch Zhou et al.
(2024), PNeRFLoc Zhao et al. (2024), CROSSFIRE Moreau
et al. (2023), and GSplatLoc Sidorov et al. (2025)). We
report the median translation (cm) and rotation errors (◦)
in Table 2. Previous analysis-by-synthesis methods Zhai
et al. (2025); Zhao et al. (2024); Sidorov et al. (2025);
Zhou et al. (2024) have demonstrated superior performance
on indoor datasets compared to outdoor scenarios, where
structure-based methods typically achieve higher accuracy.
Unlike these approaches, our Hi2-GSLoc maintains com-
petitive performance on outdoor datasets, consistently out-
performing structure-based methods in terms of localiza-
tion precision. Specifically, our method achieves superior
rotation accuracy across all evaluated scenes compared to
existing approaches. For translation accuracy, Hi2-GSLoc
demonstrates competitive performance in Hospital, Shop,
and Church scenes. When averaged across all scenes, Hi2-
GSLoc surpasses all current state-of-the-art methods in both
translation and rotation metrics.
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 10 of 17

<!-- page 11 -->
Leveraging social media news
(a) Tarjectories of Xi-MSTS-Construction
(b) Tarjectories of Xi-MSTS-Village
Figure 7: 3D trajectory comparisons between our computed poses (red solid lines) and RTK-GPS ground truth (blue dashed lines)
for Xi-MSTS-Construction (top) and Xi-MSTS-Village (bottom) scenes, displayed in both top-view and 3D perspectives.
(a) Xi-MSTS-Construction
(b) Xi-MSTS-Village
Figure 8: Positioning error analysis for Construction and Village. Comprehensive error analysis showing horizontal position
errors (green), altitude errors (blue), 3D position errors (magenta), and error distributions for Xi-MSTS-Construction (left) and
Xi-MSTS-Village (right).
4.5. Relocalization in Remote sensing
Building upon these promising results in standard out-
door localization (Table 2), we further investigate the per-
formance of Hi2-GSLoc against competitive analysis-by-
synthesis and structure-based methods in the more challeng-
ing remote sensing domain. We conduct extensive experi-
ments across diverse UAV scenarios with varying flight al-
titudes, illumination conditions, viewing angles, and terrain
types. The evaluation encompasses real flight data, public
datasets, and synthetic environments. Table 3 presents the
comprehensive evaluation results. “Ours” denotes the pose
estimation results after sparse matching and iterative dense
matching optimization, while “Ours (final)” represents the
results after our reliability filtering mechanism. Notably, as
shown in the gray-shaded regions of Table 3, our filtering
mechanism (Consistency Verification) successfully elimi-
nates 100% of unreliable pose estimates, ensuring robust per-
formance in challenging remote sensing scenarios. The red
and blue numbers indicate the best and second-best results
among our unfiltered estimates and competing methods,
respectively. The results demonstrate that beyond filtering
unreliable pose estimates, our Hi2-GSLoc achieves superior
recall rates and the lowest translation and rotation errors
across all evaluated datasets, while maintaining efficient
inference time.
To further validate the accuracy of our relocalization
results, we conduct comprehensive trajectory analysis on
the Construction and Village scenes. Figure 7 presents 2D
top-view and 3D trajectory comparisons between our esti-
mated poses and RTK-GPS ground truth. The visualizations
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 11 of 17

<!-- page 12 -->
Leveraging social media news
Table 4
Ablation study on Hi𝟐-GSLoc pipeline. Average median errors, recall rates, train and inference time are reported on Mill 19-Rubble
and Xi-MSTS with SfM ground truth.
method
AE↓
TE↓
500/10◦↑
200/5◦↑
5/5◦↑
2/2◦↑
Train/s ↓
Inference/s ↓
10000 (initial)
0.1289
0.6425
86.73
85.71
82.65
82.14
39m02s
1.1603
10000 (refine)
0.0194
0.1343
87.24
87.24
87.24
87.24
44m02s
1.2034
10000 (final)
0.0174
0.1206
100.0
100.0
100.0
100.0
39m02s
0.00025
20000 (initial)
0.1037
0.5183
91.32
90.30
88.77
87.75
77m53s
1.1508
20000 (refine)
0.0136
0.1060
91.32
91.32
91.32
91.32
83m53s
1.2612
20000 (final)
0.0130
0.1013
100.0
100.0
100.0
100.0
83m53s
0.00026
30000 (initial)
0.0987
0.5242
93.36
91.83
90.36
88.26
116m19s
1.1533
30000 (refine)
0.0128
0.1021
93.87
93.87
93.87
93.87
132m39s
1.2996
Mill 19-Rubble
30000 (final)
0.0119
0.0997
100.0
100.0
100.0
100.0
132m39s
0.00024
10000 (initial)
0.2032
0.6933
96.38
96.38
96.38
96.38
25m35s
0.3053
10000 (refine)
0.1334
0.5967
96.99
96.99
96.99
96.99
31m35s
0.7135
10000 (final)
0.1319
0.5621
100.0
100.0
100.0
100.0
31m35s
0.00028
20000 (initial)
0.1973
0.6631
98.19
98.19
98.19
98.19
53m21s
0.2882
20000 (refine)
0.1111
0.5199
98.79
98.19
98.19
98.19
59m31s
0.8149
20000 (final)
0.1107
0.5082
100.0
100.0
100.0
100.0
59m31s
0.00024
30000 (initial)
0.1957
0.6612
98.19
98.19
98.19
98.19
80m40s
0.2502
30000 (refine)
0.1062
0.4552
98.19
98.19
98.19
98.19
86m53s
0.8407
Hills-UE4
30000 (final)
0.1050
0.4574
100.0
100.0
100.0
100.0
86m53s
0.00025
10000 (initial)
0.1148
0.5139
100.0
100.0
100.0
98.31
41m51s
0.7673
10000 (refine)
0.0259
0.1381
100.0
100.0
100.0
100.0
48m58s
3.8986
10000 (final)
0.0259
0.1381
100.0
100.0
100.0
100.0
48m58s
0.00033
20000 (initial)
0.0957
0.4434
100.0
100.0
100.0
99.43
82m58s
0.5872
20000 (refine)
0.0276
0.1389
100.0
100.0
100.0
100.0
90m01s
4.6024
20000 (final)
0.0276
0.1389
100.0
100.0
100.0
100.0
90m01s
0.00032
30000 (initial)
0.0821
0.4288
100.0
100.0
100.0
100.0
123m28s
0.5506
30000 (refine)
0.0291
0.1456
100.0
100.0
100.0
100.0
131m28s
4.6657
Construction
30000 (final)
0.0291
0.1456
100.0
100.0
100.0
100.0
131m28s
0.00028
10000 (initial)
0.1832
0.7072
96.73
96.73
96.73
96.07
24m23s
0.0976
10000 (refine)
0.0633
0.2860
97.38
97.38
97.38
97.38
28m33s
1.3850
10000 (final)
0.0605
0.2836
100.0
100.0
100.0
100.0
28m33s
0.00024
20000 (initial)
0.1500
0.6396
98.03
98.03
98.03
97.38
54m38s
0.1043
20000 (refine)
0.0519
0.2229
98.69
98.69
98.69
98.69
50m26s
1.4446
20000 (final)
0.0512
0.2219
100.0
100.0
100.0
100.0
54m38s
0.00026
30000 (initial)
0.1386
0.6359
98.69
98.69
98.69
97.38
76m09s
0.1265
30000 (refine)
0.0377
0.1578
98.69
98.69
98.69
98.69
80m39s
1.4045
Campus
30000 (final)
0.0362
0.1552
100.0
100.0
100.0
100.0
80m39s
0.00026
10000 (initial)
0.1703
0.6013
86.51
86.51
86.51
86.51
40m36S
1.0731
10000 (refine)
0.0557
0.1868
88.76
88.76
88.76
88.76
48m57s
0.5045
10000 (final)
0.0501
0.1709
100.0
100.0
100.0
100.0
48m57s
0.00024
20000 (initial)
0.1402
0.4532
92.13
92.13
92.13
92.18
76m54s
1.0006
20000 (refine)
0.0459
0.1691
94.38
94.38
94.38
94.38
85m22s
0.7126
20000 (final)
0.0443
0.1611
100.0
100.0
100.0
100.0
85m22s
0.00025
30000 (initial)
0.1382
0.4665
92.13
92.13
92.13
92.13
112m34s
0.9844
30000 (refine)
0.0432
0.1550
93.82
93.82
93.82
93.82
120m59s
0.8321
Village
30000 (final)
0.0410
0.1508
100.0
100.0
100.0
100.0
120m59s
0.00026
demonstrate excellent alignment between our computed tra-
jectories (red solid lines) and RTK-GPS references (blue
dashed lines) across both scenes. Quantitative analysis re-
veals exceptional precision with mean absolute errors of
0.00000092◦latitude and 0.00000104◦longitude for Con-
struction, and 0.00000119◦latitude and 0.00000092◦longi-
tude for Village.
Figure 8 provides detailed error characterization for both
scenarios. The Construction scene exhibits consistent po-
sitioning performance throughout the flight sequence, with
mean errors of 15.6 cm horizontally and 2.4 cm in altitude.
In contrast, the Village scene, captured at higher flight
altitude, demonstrates the impact of increased elevation on
measurement precision. While occasional altitude variations
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 12 of 17

<!-- page 13 -->
Leveraging social media news
Table 5
Ablation study on consistent render-aware sampling strategy and landmark-guided keypoint detector in initial pose estimation.
C.R-A.S
SuperPoint
L-G.D
AE ↓
TE ↓
500/10◦↑
200/5◦↑
5/5◦↑
2/2◦↑
✓
105.267
1050.596
27.17
25.12
18.97
14.35
✓
✓
0.1676
0.9184
82.56
80.00
74.35
71.28
✓
19.328
404.723
48.97
45.91
37.75
30.10
Rubble
✓
✓
0.0987
0.5242
93.36
91.83
90.36
88.26
✓
0.3211
1.0398
84.33
83.73
82.53
78.91
✓
✓
0.2320
0.8183
93.97
93.97
93.97
89.75
✓
0.3119
0.9557
92.77
92.16
92.16
86.14
Hills-UE4
✓
✓
0.1957
0.6612
98.19
98.19
98.19
98.19
✓
0.2814
1.3515
85.95
84.26
83.14
70.78
✓
✓
0.1836
0.8829
99.43
99.43
99.43
97.75
✓
0.3248
1.5097
84.26
83.70
82.02
64.60
Construction
✓
✓
0.0821
0.4288
100.0
100.0
100.0
100.0
✓
0.2189
0.8284
96.73
96.73
96.73
91.50
✓
✓
0.2147
0.8218
97.38
97.38
97.38
93.46
✓
0.2256
0.8529
96.73
96.73
96.73
93.46
Campus
✓
✓
0.1386
0.6359
98.69
98.69
98.69
97.38
✓
101.7418
1156.8260
26.40
24.15
23.03
15.16
✓
✓
0.4753
1.6064
62.71
62.14
61.01
49.47
✓
58.2024
225.4948
45.45
43.18
41.47
32.38
Village
✓
✓
0.1382
0.4665
92.13
92.13
92.13
92.13
(up to 70.8 cm) occur due to the elevated flight condi-
tions, horizontal positioning maintains robustness with a
mean error of 17.2 cm. The error distribution histograms
indicate that most positioning errors fall within acceptable
ranges, yielding median 3D errors of 14.9 cm and 16.2 cm
respectively. This evaluation demonstrates the effectiveness
of our Hi2-GSLoc method across varying flight conditions
and terrain characteristics, establishing its reliability for real-
world remote sensing applications.
4.6. Ablation study
In this section, we present comprehensive ablation stud-
ies to analyze the contribution of each component in our Hi2-
GSLoc framework to relocalization performance.
Dual-hierarchical localization pipeline. In Table 4, we
report the median errors and recall rates of our algorithm at
different stages across five scenes from Xi-MSTS and Mill
19-Rubble datasets. The “initial” stage refers to the pose
estimation results from sparse matching between retrieved
images and Gaussian landmarks. The “refine” stage repre-
sents the iteratively optimized poses through dense matching
with rasterized packages (features, depth, and RGB) ren-
dered from the initial pose. The “final” stage denotes the
results after consistency checking and filtering of unreliable
estimates. The numbers 10,000, 20,000, and 30,000 indi-
cate different training iterations for the scene-specific Gaus-
sian models. The results demonstrate that dense matching
consistently improves localization accuracy over the sparse
matching stage across all Gaussian model configurations.
Our consistency verification mechanism successfully fil-
ters out 100% of unreliable results from any stage of the
(a) Initial Pose by Random Sampling + L-G.D 
(b) Initial Pose by C.R-A.S + L-G.D  
Figure 9: Comparison of dense matching results under differ-
ent initial pose estimation strategies. (a) Random sampling
and (b) our C.R-A.S with L-G.D. Orange boxes highlight
incorrect matches from poor rendering, blue boxes show
accurate matches. Samples below the blue dashed line have
recoverable errors, while those above have excessive errors that
cannot be corrected by dense matching.
pipeline while requiring minimal computational overhead
(about 0.24ms per inference). The majority of scenes achieve
optimal performance when using Gaussian models trained
for 30,000 iterations, indicating the importance of sufficient
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 13 of 17

<!-- page 14 -->
Leveraging social media news
Table 6
Ablation study on dense rasterization matching for pose optimization.
SuperPoint
Gaussian
PMM
AE ↓
TE ↓
500/10◦↑
200/5◦↑
5/5◦↑
2/2◦↑
✓
116.4793
595.9725
3.0769
1.5384
1.0256
0.5128
✓
✓
113.3258
544.8418
4.6153
3.0769
1.0256
0.5128
✓
0.0264
0.1242
93.36
92.82
91.76
90.77
Rubble
✓
✓
0.0128
0.1021
93.87
93.87
93.87
93.87
✓
0.1144
0.7321
95.78
95.78
95.78
95.78
✓
✓
0.1076
0.7441
98.19
98.19
98.19
98.19
✓
0.1046
0.4473
98.19
98.19
98.19
98.19
Hills-UE4
✓
✓
0.1062
0.4552
98.19
98.19
98.19
98.19
✓
122.1261
793.8126
0.00
0.00
0.00
0.00
✓
✓
130.8981
1119.1679
0.00
0.00
0.00
0.00
✓
0.0936
0.4278
100.0
100.0
100.0
100.0
Construction
✓
✓
0.0291
0.1456
100.0
100.0
100.0
100.0
✓
0.0899
1.0922
97.03
97.03
96.34
95.03
✓
✓
0.2256
0.8529
96.73
96.73
96.73
93.46
✓
0.0682
0.2535
98.69
98.69
98.69
98.69
Campus
✓
✓
0.0377
0.1578
98.69
98.69
98.69
98.69
✓
17.2569
75.0487
45.4545
43.75
42.04
29.54
✓
✓
0.7102
3.0643
56.25
53.41
51.70
42.04
✓
0.0526
0.1663
92.61
92.61
92.61
92.61
Village
✓
✓
0.0432
0.1550
93.82
93.82
93.82
93.82
(a) Superpoint feature coarse to fine matching
(b) Gaussian feature coarse to fine matching
Figure 10: Comparison of coarse to fine dense matching with different dense feature extractor. (a) SuperPoint-based features
and (b) our Gaussian features. From left to right in both (a) and (b): initial coarse matching and iterative dense matching.
training for high-quality scene representation. This ablation
study validates the effectiveness and robustness of each
component in our hierarchical relocalization pipeline.
Initial pose estimation. The accuracy of initial pose es-
timation directly determines the quality of rendered images,
which subsequently affects dense matching precision. We
conduct ablation studies on five scenes from the Mill 19-
Rubble and Xi-MSTS datasets to evaluate the effectiveness
of our C.R-A.S and L-G.D components, as shown in Table
5. For each scene, we compare four configurations: random
sampling strategy, SuperPoint detector, and our proposed
C.R-A.S for landmark selection combined with L-G.D for
query image keypoint detection. The results demonstrate
that using both C.R-A.S and L-G.D consistently achieves
the lowest pose errors and highest recall rates across all
scenes. Notably, in Mill 19-Rubble and Xi-MSTS-Village,
the combination of C.R-A.S and L-G.D achieves 73.91%
and 76.97% higher recall rates respectively compared to the
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 14 of 17

<!-- page 15 -->
Leveraging social media news
random sampling and SuperPoint baseline, demonstrating
that our modules significantly improve initial pose accuracy
while providing greater stability.
Figure 9 visualizes the render and dense matching results
under different initial pose estimation strategies. Subfigure
(a) shows results obtained using random sampling for initial
pose estimation, while subfigure (b) presents results from
our proposed Consistent Render-Aware Sampling strategy.
Orange boxes highlight dense matching results under incor-
rectly rendered views, whereas blue boxes indicate accurate
matching results under precise initial pose rendering. Addi-
tionally, samples below the blue dashed line represent dense
matching results with relatively small initial pose errors,
which can be further refined through subsequent iterative
dense matching optimization. In contrast, samples above the
dashed line suffer from excessive initial pose errors, and
even dense matching cannot recover accurate poses from
such poor initialization. This confirms that accurate initial
pose estimation is a critical prerequisite for reliable system
localization.
Dense rasterization matching. We compare feature ex-
traction strategies (rendered Gaussian features vs. extracting
features from rendered RGB images using existing feature
extraction networks) and matching strategies (with/without
probabilistic mutual matching (PMM)). Results are pre-
sented in Table 6. The experimental results reveal significant
performance degradation when employing SuperPoint for
feature extraction across multiple scenes. Notably, in the
Rubble, Construction, and Village scenes, all evaluation
metrics demonstrate substantial decline, with the Construc-
tion scene experiencing complete localization failure (all
recall metrics drop to 0%). This performance collapse can be
attributed to SuperPoint’s limited generalization capability
on high-resolution aerial imagery, which is characterized by
repetitive patterns and extreme viewpoint variations typical
of remote sensing scenarios.
Figure 10 provides a qualitative comparison between
initial coarse matching and iterative dense matching re-
sults, contrasting SuperPoint-based features (left) with our
trained Gaussian features (right). The visualization clearly
demonstrates that inappropriate feature extraction methods
lead to erroneous dense pose estimation, even when accu-
rate initial views are rendered from correct initial poses.
This error propagation results in progressively deteriorating
views and poses through iterations, creating an unrecover-
able optimization failure. These results demonstrate that our
Gaussian-based scene-specific feature extraction approach
achieves superior robustness and reliability compared to
generic feature descriptors. The performance degradation of
pre-trained models like SuperPoint can be attributed to the
significant domain gap between existing training datasets
and remote sensing imagery.
Additionally, to determine the optimal number of it-
erations for our method, we conducted an ablation study
across five diverse datasets. As illustrated in Figure 11,
both the median AE and median TE demonstrate rapid
convergence within the first few iterations. The results show
Figure 11: the Median AE (top) and Median TE (bottom) for
five datasets over iterations 0-10.
that most datasets achieve significant error reduction by
iteration 3, with marginal improvements observed in sub-
sequent iterations. Specifically, the angular error stabilizes
around iteration 3 across all datasets, while the translation
error exhibits similar convergence behavior. Considering the
trade-off between accuracy and computational efficiency,
we selected iteration 3 as the optimal configuration for our
method. This choice ensures that our approach maintains
high localization accuracy while keeping the inference time
reasonable for practical applications.
5. Conclusion
This paper introduces 3D Gaussian Splatting (3DGS) as
a novel map representation for visual relocalization, signifi-
cantly expanding the capabilities of UAV navigation in large-
scale remote sensing scenarios. Building upon this scene
representation, we present Hi2-GSLoc, a dual-hierarchical
relocalization framework that systematically addresses criti-
cal challenges in remote sensing through three key technical
innovations: (1) Scalable scene processing through parti-
tioned Gaussian training coupled with dynamic memory
management, enabling efficient handling of large-scale envi-
ronments; (2) Scene-specific feature learning via consistent
render-aware landmark sampling that effectively exploits
Gaussian geometric constraints to enhance feature repre-
sentation quality; (3) Robust and accurate pose estimation
through a coarse-to-fine refinement strategy with consis-
tency validation, ensuring reliable localization under chal-
lenging conditions. Comprehensive experimental evaluation
on the Mill 19-Rubble and Xi-MSTS datasets demonstrates
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 15 of 17

<!-- page 16 -->
Leveraging social media news
the effectiveness and practical utility of our approach. Hi2-
GSLoc achieves superior recall rates, maintains computa-
tional efficiency, and delivers centimeter-level accuracy at
high altitudes using only visual information. Furthermore,
the method exhibits exceptional robustness by effectively
filtering unreliable localizations through our consistency
validation mechanism, making it particularly well-suited for
practical UAV applications in demanding remote sensing
environments.
CRediT authorship contribution statement
Boni Hu: Writing - original draft, Writing - review &
editing, Methodology, Investigation, Visualization. Zhenyu
Xia: Methodology, Dataset collection. Lin Chen: Results
visualization. Pengcheng Han: Dataset collection, Writing–
review & editing. Shuhui Bu: Conceptualization of this
study, Writing– review & editing.
Declaration of competing interest
The authors declare that they have no known competing
financial interests or personal relationships that could have
appeared to influence the work reported in this paper.
Acknowledgments
This work was supported by the National Natural Sci-
ence Foundation of China (NSFC) under Grant No. 42130112
and the Postdoctoral Fellowship Program of CPSF under
Grant No. GZB20240986.
References
Arandjelovic, R., Gronat, P., Torii, A., Pajdla, T., Sivic, J., 2016. Netvlad:
Cnn architecture for weakly supervised place recognition, in: Proceed-
ings of the IEEE conference on computer vision and pattern recognition,
pp. 5297–5307.
Berton, G., Masone, C., 2025. Megaloc: One retrieval to place them all.
arXiv preprint arXiv:2502.17237 .
Berton, G., Trivigno, G., Caputo, B., Masone, C., 2023. Eigenplaces: Train-
ing viewpoint robust models for visual place recognition, in: Proceedings
of the IEEE/CVF International Conference on Computer Vision, pp.
11080–11090.
Brachmann, E., Krull, A., Nowozin, S., Shotton, J., Michel, F., Gumhold,
S., Rother, C., 2017. Dsac-differentiable ransac for camera localization,
in: Proceedings of the IEEE conference on computer vision and pattern
recognition, pp. 6684–6692.
Brachmann, E., Rother, C., 2021. Visual camera re-localization from rgb
and rgb-d images using dsac. IEEE transactions on pattern analysis and
machine intelligence 44, 5847–5865.
Brahmbhatt, S., Gu, J., Kim, K., Hays, J., Kautz, J., 2017. Geometry-aware
learning of maps for camera localization. arXiv: Computer Vision and
Pattern Recognition,arXiv: Computer Vision and Pattern Recognition .
Camposeco, F., Cohen, A., Pollefeys, M., Sattler, T., 2018. Hybrid scene
compression for visual localization. arXiv: Computer Vision and Pattern
Recognition,arXiv: Computer Vision and Pattern Recognition .
Chen, S., Li, X., Wang, Z., Prisacariu, V.A., 2022. Dfnet: Enhance absolute
pose regression with direct feature matching, in: European Conference
on Computer Vision, Springer. pp. 1–17.
Chen, S., Wang, Z., Prisacariu, V., 2021. Direct-posenet: Absolute pose
regression with photometric consistency, in: 2021 International Confer-
ence on 3D Vision (3DV). URL: http://dx.doi.org/10.1109/3dv53792.
2021.00125, doi:10.1109/3dv53792.2021.00125.
Cheng, Y., Jiao, J., Wang, Y., Kanoulas, D., 2024. Logs: Visual localization
via gaussian splatting with fewer training images.
arXiv preprint
arXiv:2410.11505 .
Clark, R., Wang, S., Markham, A., Trigoni, N., Wen, H., 2017. Vidloc: A
deep spatio-temporal model for 6-dof video-clip relocalization. Cornell
University - arXiv,Cornell University - arXiv .
DeTone, D., Malisiewicz, T., Rabinovich, A., 2018.
Superpoint: Self-
supervised interest point detection and description, in: 2018 IEEE/CVF
Conference on Computer Vision and Pattern Recognition Workshops
(CVPRW). URL: http://dx.doi.org/10.1109/cvprw.2018.00060, doi:10.
1109/cvprw.2018.00060.
Do, T., Sinha, S.N., 2024. Improved scene landmark detection for camera
localization, in: 2024 International Conference on 3D Vision (3DV),
IEEE. pp. 975–984.
Dusmanu, M., Rocco, I., Pajdla, T., Pollefeys, M., Sivic, J., Torii, A., Sattler,
T., 2019. D2-net: A trainable cnn for joint description and detection of
local features, in: 2019 IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR). URL: http://dx.doi.org/10.1109/cvpr.
2019.00828, doi:10.1109/cvpr.2019.00828.
Feng, G., Chen, S., Fu, R., Liao, Z., Wang, Y., Liu, T., Hu, B., Xu, L., Pei, Z.,
Li, H., et al., 2025. Flashgs: Efficient 3d gaussian splatting for large-scale
and high-resolution rendering, in: Proceedings of the Computer Vision
and Pattern Recognition Conference, pp. 26652–26662.
Gao, X.S., Hou, X.R., Tang, J., Cheng, H.F., 2003. Complete solution clas-
sification for the perspective-three-point problem. IEEE Transactions on
Pattern Analysis and Machine Intelligence , 930–943URL: http://dx.
doi.org/10.1109/tpami.2003.1217599, doi:10.1109/tpami.2003.1217599.
Hu, B., Chen, L., Chen, R., Bu, S., Han, P., Li, H., 2024. Curriculumloc: En-
hancing cross-domain geolocalization through multi-stage refinement.
IEEE Transactions on Geoscience and Remote Sensing .
Huang, Z., Yu, H., Shentu, Y., Yuan, J., Zhang, G., 2025. From sparse to
dense: Camera relocalization with scene-specific detector from feature
gaussian splatting, in: Proceedings of the Computer Vision and Pattern
Recognition Conference, pp. 27059–27069.
Jiang, H., Karpur, A., Cao, B., Huang, Q., Araujo, A., 2024. Omniglue:
Generalizable feature matching with foundation model guidance, in:
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 19865–19875.
Ke, T., Roumeliotis, S., 2017.
An efficient algebraic solution to the
perspective-three-point problem.
Cornell University - arXiv,Cornell
University - arXiv .
Keetha, N., Mishra, A., Karhade, J., Jatavallabhula, K.M., Scherer, S.,
Krishna, M., Garg, S., 2023. Anyloc: Towards universal visual place
recognition. IEEE Robotics and Automation Letters 9, 1286–1293.
Kendall, A., Cipolla, R., 2017. Geometric loss functions for camera pose
regression with deep learning, in: 2017 IEEE Conference on Computer
Vision and Pattern Recognition (CVPR). URL: http://dx.doi.org/10.
1109/cvpr.2017.694, doi:10.1109/cvpr.2017.694.
Kendall, A., Grimes, M., Cipolla, R., 2015.
Posenet: A convolutional
network for real-time 6-dof camera relocalization, in: Proceedings of the
IEEE international conference on computer vision, pp. 2938–2946.
Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G., 2023. 3d gaussian
splatting for real-time radiance field rendering .
Leroy, V., Cabon, Y., Revaud, J., 2024. Grounding image matching in 3d
with mast3r, in: European Conference on Computer Vision, Springer.
pp. 71–91.
Li, X., Wang, S., Zhao, Y., Verbeek, J., Kannala, J., 2020. Hierarchical
scene coordinate classification and regression for visual localization, in:
2020 IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion (CVPR). URL: http://dx.doi.org/10.1109/cvpr42600.2020.01200,
doi:10.1109/cvpr42600.2020.01200.
Lin, J., Li, Z., Tang, X., Liu, J., Liu, S., Liu, J., Lu, Y., Wu, X., Xu,
S., Yan, Y., Yang, W., 2024.
Vastgaussian: Vast 3d gaussians for
large scene reconstruction.
URL: https://arxiv.org/abs/2402.17427,
arXiv:2402.17427.
Lindenberger, P., Sarlin, P.E., Pollefeys, M., Zurich, E., Mixed, M., .
Lightglue: Local feature matching at light speed .
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 16 of 17

<!-- page 17 -->
Leveraging social media news
Liu, C., Chen, S., Bhalgat, Y.S., Hu, S., Cheng, M., Wang, Z., Prisacariu,
V.A., Braud, T., 2025.
Gs-cpr: Efficient camera pose refinement via
3d gaussian splatting, in: The Thirteenth International Conference on
Learning Representations.
Lu, F., Jin, T., Lan, X., Zhang, L., Liu, Y., Wang, Y., Yuan, C., 2025.
Selavpr++: Towards seamless adaptation of foundation models for
efficient place recognition. arXiv preprint arXiv:2502.16601 .
Lu, F., Zhang, L., Lan, X., Dong, S., Wang, Y., Yuan, C., 2024. Towards
seamless adaptation of pre-trained models for visual place recognition.
arXiv preprint arXiv:2402.14505 .
Mallick, S.S., Goel, R., Kerbl, B., Steinberger, M., Carrasco, F.V.,
De La Torre, F., 2024. Taming 3dgs: High-quality radiance fields with
limited resources, in: SIGGRAPH Asia 2024 Conference Papers, pp. 1–
11.
Moreau, A., Piasco, N., Bennehar, M., Tsishkou, D., Stanciulescu, B.,
Fortelle, A., 2023. Crossfire: Camera relocalization on self-supervised
features from an implicit representation .
Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V.,
Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al., 2023. Dinov2:
Learning robust visual features without supervision.
arXiv preprint
arXiv:2304.07193 .
Potje, G., Cadar, F., Araujo, A., Martins, R., Nascimento, E.R., 2024.
Xfeat: Accelerated features for lightweight image matching, in: 2024
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), pp. 2682–2691. doi:10.1109/CVPR52733.2024.00259.
Revaud, J., Weinzaepfel, P., Souza, C., Pion, N., Csurka, G., Cabon, Y.,
Humenberger, M., 2019. R2d2: Repeatable and reliable detector and
descriptor.
arXiv: Computer Vision and Pattern Recognition,arXiv:
Computer Vision and Pattern Recognition .
Sarlin, P.E., Cadena, C., Siegwart, R., Dymczyk, M., 2019.
From
coarse to fine: Robust hierarchical localization at large scale, in: 2019
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR).
URL: http://dx.doi.org/10.1109/cvpr.2019.01300, doi:10.
1109/cvpr.2019.01300.
Sarlin, P.E., DeTone, D., Malisiewicz, T., Rabinovich, A., 2020.
Su-
perglue: Learning feature matching with graph neural networks, in:
2020 IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion (CVPR). URL: http://dx.doi.org/10.1109/cvpr42600.2020.00499,
doi:10.1109/cvpr42600.2020.00499.
Shavit, Y., Ferens, R., Keller, Y., 2021.
Learning multi-scene absolute
pose regression with transformers, in: 2021 IEEE/CVF International
Conference on Computer Vision (ICCV), pp. 2713–2722. doi:10.1109/
ICCV48922.2021.00273.
Sidorov, G., Mohrat, M., Gridusov, D., Rakhimov, R., Kolyubin, S., 2025.
Gsplatloc: Grounding keypoint descriptors into 3d gaussian splatting for
improved visual localization .
Sun, J., Shen, Z., Wang, Y., Bao, H., Zhou, X., 2021. Loftr: Detector-free lo-
cal feature matching with transformers, in: 2021 IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR). URL: http://
dx.doi.org/10.1109/cvpr46437.2021.00881, doi:10.1109/cvpr46437.2021.
00881.
Taira, H., Okutomi, M., Sattler, T., Cimpoi, M., Pollefeys, M., Sivic,
J., Pajdla, T., Torii, A., 2021.
Inloc: Indoor visual localization with
dense matching and view synthesis.
IEEE Transactions on Pattern
Analysis and Machine Intelligence , 1293–1307URL: http://dx.doi.
org/10.1109/tpami.2019.2952114, doi:10.1109/tpami.2019.2952114.
Turki, H., Ramanan, D., Satyanarayanan, M., 2022. Mega-nerf: Scalable
construction of large-scale nerfs for virtual fly-throughs, in: Proceedings
of the IEEE/CVF conference on computer vision and pattern recogni-
tion, pp. 12922–12931.
Tyszkiewicz, M., Fua, P., Trulls, E., 2020. Disk: Learning local features
with policy gradient.
Advances in Neural Information Processing
Systems 33, 14254–14265.
Walch, F., Hazirbas, C., Leal-Taixé, L., Sattler, T., Hilsenbeck, S., Cremers,
D., 2016. Image-based localization using lstms for structured feature
correlation. Cornell University - arXiv,Cornell University - arXiv .
Wang, C., Chen, S., Song, Y., Xu, R., Zhang, Z., Zhang, J., Yang, H.,
Zhang, Y., Fu, K., Du, S., et al., 2025a. Focus on local: Finding reliable
discriminative regions for visual place recognition, in: Proceedings of
the AAAI Conference on Artificial Intelligence, pp. 7536–7544.
Wang, X., Yi, R., Ma, L., 2024.
Adr-gaussian: Accelerating gaussian
splatting with adaptive radius, in: SIGGRAPH Asia 2024 Conference
Papers, pp. 1–10.
Wang, Z., Shi, D., Qiu, C., Jin, S., Li, T., Qiao, Z., Chen, Y., 2025b.
Vecmaplocnet: Vision-based uav localization using vector maps in gnss-
denied environments. ISPRS Journal of Photogrammetry and Remote
Sensing 225, 362–381.
Yang, L., Shrestha, R., Li, W., Liu, S., Zhang, G., Cui, Z., Tan, P., 2022.
Scenesqueezer: Learning to compress scene for camera relocalization,
in: Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pp. 8259–8268.
Ye, Q., Luo, J., Lin, Y., 2024.
A coarse-to-fine visual geo-localization
method for gnss-denied uav with oblique-view imagery. ISPRS Journal
of Photogrammetry and Remote Sensing 212, 306–322.
Yen-Chen, L., Florence, P., Barron, J.T., Rodriguez, A., Isola, P., Lin, T.Y.,
2021. inerf: Inverting neural radiance fields for pose estimation, in: 2021
IEEE/RSJ International Conference on Intelligent Robots and Systems
(IROS), IEEE. pp. 1323–1330.
Yin, P., Jiao, J., Zhao, S., Xu, L., Huang, G., Choset, H., Scherer, S., Han, J.,
2025. General place recognition survey: Towards real-world autonomy.
IEEE Transactions on Robotics .
Zhai, H., Zhang, X., Zhao, B., Li, H., He, Y., Cui, Z., Bao, H., Zhang,
G., 2025. Splatloc: 3d gaussian splatting-based visual localization for
augmented reality. IEEE Transactions on Visualization and Computer
Graphics .
Zhao, B., Yang, L., Mao, M., Bao, H., Cui, Z., 2024.
Pnerfloc: Visual
localization with point-based neural radiance fields, in: Proceedings of
the AAAI Conference on Artificial Intelligence, pp. 7450–7459.
Zhou, Q., Maximov, M., Litany, O., Leal-Taixé, L., 2024.
The nerfect
match: Exploring nerf features for visual localization.
URL: https:
//arxiv.org/abs/2403.09577, arXiv:2403.09577.
Zhou, S., Chang, H., Jiang, S., Fan, Z., Zhu, Z., Xu, D., Chari, P., You, S.,
Wang, Z., Kadambi, A., 2023. Feature 3dgs: Supercharging 3d gaussian
splatting to enable distilled feature fields .
J.K. Krishnan et al.: Preprint submitted to Elsevier
Page 17 of 17
