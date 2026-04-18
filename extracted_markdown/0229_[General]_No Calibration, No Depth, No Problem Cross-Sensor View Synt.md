<!-- page 1 -->
No Calibration, No Depth, No Problem: Cross-Sensor View Synthesis with 3D
Consistency
Cho-Ying Wu, Zixun Huang, Xinyu Huang, Liu Ren
Bosch Research North America & Bosch Center for Artificial Intelligence (BCAI)
{Cho-Ying.Wu, Zixun.Huang, Xinyu.Huang, Liu.Ren}@us.bosch.edu
Figure 1. Problem Setup. Given unpaired RGB-X images from sensors, the task is to synthesize X-images that are pixel-wise aligned with
the RGB views for multi-modal applications. Traditional 3D approaches rely on complete 3D priors—including depth and the poses/intrinsics
of both modalities—to align and render cross-sensor images. In contrast, our scalable framework removes these dependencies, enabling
RGB-guided X-image synthesis without the 3D priors for X to replace calibration for different types of sensors and metric depth acquisition.
Abstract
We present the first study of cross-sensor view synthesis
across different modalities. We examine a practical, fun-
damental, yet widely overlooked problem: getting aligned
RGB-X data, where most RGB-X prior work assumes such
pairs exist and focuses on modality fusion, but it empirically
requires huge engineering effort in calibration. We propose
a match-densify-consolidate method. First, we perform RGB-
X image matching followed by guided point densification.
Using the proposed confidence-aware densification and self-
matching filtering, we attain better view synthesis and later
consolidate them in 3D Gaussian Splatting (3DGS). Our
method uses no 3D priors for X-sensor and only assumes
nearly no-cost COLMAP for RGB. We aim to remove the
cumbersome calibration for various RGB-X sensors and ad-
vance the popularity of cross-sensor learning by a scalable
solution that breaks through the bottleneck in large-scale
real-world RGB-X data collection. See project page.
1. Introduction
Sensors other than the RGB spectrum are useful for various
applications, such as NIR (Near-Infrared) sensors for night
vision in autonomous driving [9, 12, 47, 57], or thermal
sensors for leak detection or robust autonomous systems [22,
30, 39, 43, 56, 58]. Although with diverse applications,
they remain relatively underexplored in research. First, they
are much less in data quantity compared with RGB, and
the research on other sensors needs to rely on their paired
and view-aligned RGB to provide rich information from
foundation models for scene understanding [19, 62, 102].
For sensors such as thermal cameras, images are natively
low-texture and rely on RGB to provide features.
However, real data captured by different sensors typically
suffer from multiple issues, making it challenging to build
an RGB-X paired dataset, where X denotes other types of
sensors in the work. The traditional industrial settings lever-
age 3D reprojection, which requires cumbersome sensor
calibration including measuring intrinsics, sensor synchro-
nization, relative pose estimation, and metric depth, where
errors within each stage will be propagated and affect the end
results, and it still cannot solve occlusion from displacement.
Structure-from-motion methods like COLMAP [70] are
widely used in scene reconstruction and considered much
less effort in sensor calibration. Yet, it only works on RGB
scenes and usually fails to work on X or RGB-X settings,
particularly on less-texture sensors such as thermal cameras.
The RGB-X systems usually require additional high-
1
arXiv:2602.23559v1  [cs.CV]  27 Feb 2026

<!-- page 2 -->
Figure 2. Homography warping assumes 3D planar structures and
causes visible misalignment (statue areas) when the scene contains
distinct fore-/background layers.
precision dense depth sensors, such as RGB-Thermal-Depth
system [41], but it further complicates sensor registration
efforts and is not scalable. Some datasets simply consider
fixed depth [54] for re-projection without solving the rooted
issue. Note that RGB-D systems already encode 3D informa-
tion that can be re-projected to RGB’s view (still requiring
intrinsics and limited by depth sensor precision and density),
while other sensors without such 3D priors cannot.
Some recent work performs cross-modality matching of
keypoints between RGB and X views [66, 79, 95] based on
RGB image matchers. However, they either produce sparse
or incomplete correspondences or have highly noisy matches
in low-confidence regions. Their goal is to estimate the cam-
era pose or homography matrix H ∈R3×3 for warping,
where a target view pixel p′ and source view pixel p (both in
homogeneous coordinates) are related by p′ = H−1p. How-
ever, homography warping assumes 3D planar structures,
indicating it can only distort or shear the view as a plane
but cannot create 3D effects with disparity. The warping
works when the scene lies around a depth layer, but will
show distinct misalignment in counterexamples (See Fig. 2).
This work is the first scalable framework for cross-
sensor view synthesis to acquire paired and aligned RGB-X
data without cumbersome calibration or metric depth, to
facilitate the research on sensors without 3D priors and
beyond RGB cameras (Fig. 1), such as Thermal Cameras,
NIR Cameras, and Synthetic-Aperture Radar (SAR).
First, we adopt cross-modal image matchers to match
keypoints between RGB and X views and accumulate X’s
keypoints across frames onto the current RGB view to form
an aligned sparse or semi-dense X-map. Then, both the RGB
and X-map are input to our trained RGB-guided X-densifier,
which reconstructs the dense X-map. With a handful of key-
points aggregated in X-map and RGB images as guidance,
the model generates a dense X-image that satisfies the input
X-map and infills the void areas using RGB and surrounding
points as cues. However, we find that densification may lead
to irregular or noisy structures affected by noisy matches in
low-confidence areas. A higher confidence threshold yields
a sparser X-map that must rely more on cross-domain guid-
ance from RGB, while a lower threshold introduces noisy
X-points and distorts structures. We propose Confidence-
Aware Densification and Fusion (CADF) that densifies and
fuses multi-level threshold X-maps into one X-image.
Then, we perform patch rejection by our self-matching
mechanism—We look into the patch features from the image
matcher with the prior that each patch should be matched to
the same patch for RGB-X pairs. We calculate a similarity
matrix between each patch and reject low-similarity patches.
After that, we perform a finer-stage densification. To further
enhance multi-view consistency, we adopt RGB-X 3D Gaus-
sian Splatting (3DGS) on RGB’s views to consolidate in the
3D space for multi-view consistency. The work only consid-
ers running COLMAP on RGB for 3DGS training, which
is standard and considered no-cost. In the experiments, we
further show that even without using 3DGS, we still attain
better performance than others.
Contributions. Our contributions are as follows.
• The first scalable framework for cross-sensor view synthe-
sis on RGB-X to get view-aligned pairs without calibration
or depth—studying a fundamental but widely overlooked
problem in prior cross-modal/sensor learning.
• A match-densify-consolidate framework. It begins with
matched sparse keypoints and presents a densification ap-
proach with our CADF module that integrates matching
confidence into the densifier for better synthesis, followed
by subsequent filtering, re-densification, and 3DGS to
build up 3D consistency.
• Extensive evaluation demonstrating our method attains the
state of the art on the methods without 3D priors, even still
outperforming them without leveraging 3DGS.
2. Related Work
2.1. RGB-X Task and Data Curation
Imaging sensors beyond RGB like NIR or thermal cam-
eras are useful in industrial applications, especially in
autonomous driving for adverse weather or night vision
to ensure safety [7, 31, 40, 61, 71, 73].
Some works
study sensor fusion strategy for segmentation, detection,
or tracking [13, 14, 23, 26, 27, 29, 34, 42, 52, 53, 75,
76, 81, 87, 93, 97, 99, 100, 104, 105] and applications in
robotics [2, 20, 33, 49, 55, 98]
Most works focus on sensor fusion and design fusion mod-
ules. However, they all assume pixel-wise aligned RGB-X
inputs already, which require substantial engineering effort
in sensor calibration, synchronization, and precise metric
depth from sensors like Lidar or ToF to help alignment.
Some works adopt image translation from RGB to generate
X-images, such as RGB-to-thermal, to get pseudo-paired
data [1, 6, 45, 92]. They do not use real information in X,
and the generation needs to deal with inherent ambiguity.
E.g., a cup of water can be flexible in its exact temperature
(cool or cold) and hard to judge by its appearance.
In contrast to prior works, we focus on this fundamental
problem in data and propose a scalable framework to reduce
2

<!-- page 3 -->
the engineering effort to obtain paired RGB-X real data.
2.2. Cross-Modal Image Matching
Image matching is a fundamental problem. Early methods
adopt handcrafted descriptors [8, 77] and find a transfor-
mation with RANSAC [21]. During the deep learning era
some works adopt CNN [16, 17, 60, 80, 82], graph neu-
ral network [69, 72, 101], transformers [11, 59, 74, 84],
lightweight attention [51, 64], or benefit from foundation
models [18, 32]. They focus on matching in the RGB domain
from different views, and performance severely degrades for
cross-modal matching. Further, image matching aims for
relative pose estimation and still needs 3D priors such as
intrinsics and precise metric depth for re-projection. Without
3D priors, they still use an estimated homography matrix to
warp to the target view, which is confined by its underlying
assumption (Fig. 2).
Recent feed-forward reconstruction methods,
like
DUSt3R [85], VGGT [83], or MapAnything [37], also im-
plicitly predicts intrinsics, depth, and 3D pointmap and re-
constructs the scene. However, they cannot perform cross-
modal matching, and reconstruction would fail due to feature
dissimilarity.
Few works have explored cross-modal matching. ReD-
Feat [15] uses an adapter and detector to get features
and compute detection scores for RGB-thermal pairs.
XoFTR [79] uses transformers for coarse-to-fine RGB-
thermal matching and achieves stronger performance. MIN-
IMA [66] presents a cross-modal image generation engine
(e.g., RGB-to-thermal) to finetune image matchers, such as
LightGLUE [51] as its featured model, thereby enhancing
feature robustness across modalities. However, they still
estimate poses or homography matrices and remain confined
by the assumptions behind.
3. Methods
We enable RGB-X cross-sensor view synthesis without re-
quiring 3D priors for X, such as scene metric depth and cali-
bration for cross-sensor relative pose and intrinsics (Fig. 1).
Detailed method is shown in Fig. 3. We begin with the
smallest components— pixel-level keypoints from a cross-
modal image matcher— and perform RGB-X densification
to generate X-images aligned with RGB. Our Confidence-
Aware Densification and Fusion (CADF) module integrates
image-matching confidence into the densification stage to
achieve better synthesis. Then, we propose self-matching
for filtering, where we use the matcher as an evaluator
to filter out erroneous patches, followed by finer-stage re-
densification. Lastly, we employ RGB-X 3DGS using RGB
camera poses along with densified and aligned X-view, and
consolidate both sensors into an unified 3D space. Notably,
the pipeline only requires cheap and widely-used COLMAP
on RGB. We show that even without 3DGS and COLMAP,
our densified maps still outperform baselines in experiments.
3.1. RGB-X Matching
Matching. Given an RGB image I and X-modality image X,
image matching finds a matching set that maps coordinates
pI in I to pX in X with a confidence score c. Most matchers
can afford sparse or semi-dense matching in practice, where
memory consumption directly scales with the number of
candidates. The matching results are filtered by a threshold
δ, where a match is kept when c >= δ.
Cross-modal matchers typically yield far fewer corre-
spondences than RGB-based ones. This is partly due to
the inherent challenge of aligning features across modali-
ties. Further, modalities like thermal or SAR often contain
large homogeneous regions (e.g., uniform-temperature areas
lacking surface texture), making it difficult to extract feature
points for reliable matching.
We begin from an image matcher and stack N frames’
X-keypoints into corresponding RGB’s coordinates to get
Xm.
Xm[p] =
P
n 1[p = pI
n] X[pX
n ]
P
n 1[p = pIn]
,
(1)
where n ∈N, and 1[·] is the indicator function that equals 1
if the condition holds and 0 otherwise. If p ̸= pI
n, ∀n ∈N,
Xm[p] = −1, meaning void area.
Area Sampling. To ameliorate matching in texture-less or
complex areas, such as sky, ground, walls, or grass, we adopt
GroundedSAM [67] to segment these areas on RGB images.
Then, we uniformly sample points from the corresponding
warped X-images (via homography) within the mask and
still void areas. To prevent warping errors from heavily
influencing subsequent densification, and to allow erroneous
matches to be filtered out later, we only sample 5% points of
such areas.
Xm[p] = XW [p], p ∼U
 {p | M(p) = 1 ∧Xm[p] = -1}

, (2)
where U is uniform sampling, XW denotes the warped X-
image to RGB view, using a homography matrix computed
from RGB-X correspondences, and M is the area mask.
3.2. Confidence-Aware Densification and Fusion
Densification. The obtained Xm is sparse/ semi-dense, and
then we densify it by a network D to get a dense map. D is
pretrained on diverse RGB–X modalities using paired data,
where the model takes an RGB image along with a down-
sampled, sparsified X-map as inputs. The model is trained
to densify X into a dense map. The architecture contains
a backbone of recurrent units [106, 107] and dynamic spa-
tial propagation (DySPN) layers [50] that refine the output
with spatial attention from known points in a recurrent fash-
ion. For inference, I and Xm are inputs. Using a handful
of points as cues in Xm, D generates a better dense map
3

<!-- page 4 -->
Figure 3. Method Overview. Our approach consists of three stages. In the first stage, we perform cross-modality feature matching to
establish correspondences between RGB and X-images. The matched points are sampled and accumulated onto RGB views to produce
semi-dense X-images Xm along with multi-level confidence maps Cm. In the second stage, we conduct RGB-guided densification to get
dense X-images from RGB with semi-dense X as cues. Our Confidence-Aware Densification and Fusion module integrates confidence maps
from the image matching stage to guide the densification to concentrate on higher-confidence points and get robust Xd In the final stage, our
proposed self-matching mechanism further filters inconsistent patches, and the results are fed back into the densification stage for refinement.
To further improve multi-view consistency, we train RGB-X 3DGS using COLMAP-calibrated RGB views to consolidate both modalities
into a unified 3D RGB-X radiance field, which improves multi-view consistency and further enables cross-sensor view synthesis.
that conforms to known conditions Xm and fills in unknown
areas from I and neighboring points.
Confidence-Aware Fusion. Naively densifying Xm would
lead to highly irregular structures due to uncertainty in image
matching. To overcome the issue, we design Confidence-
Aware Densification and Fusion (CADF) module that inte-
grates the matching and densification stages by using the
image-matching confidence c to improve the RGB-X densi-
fication.
First, we integrate c into DySPN refinement. The original
DySPN formulates the recurrent iteration as follows: at
timestep t, the refined result Lt+1 combines current update
Lt, weighted by affinity score w, and known points Xm,
weighted by backbone-predicted certainty map Cs
Lt+1 = (1 −Cs)
X
r
X
(a,b)
wr,a,b ∗Lt
a,b + CsXm,
(3)
, where r denotes filters and (a, b) denotes the coordinates.
We propose to interleave the recurrent steps with the con-
fidence map Cm from image matching that aggregates point
confidence c into a map. Cm weighs down the contribution
for probably incorrect original keypoints and concentrates
the iterative refinement at the higher-confidence points.
Lt+1 = (1 −CsCm)
X
r
X
(a,b)
wr,a,b ∗Lt
a,b + CsCmXm.
(4)
Next, we find that a higher δ keeps more robust points
but leads to a sparser map that purely relies on RGB as cues
for densification. A lower δ accepts more but noisy points as
known conditions, which leads to irregular structures (See
Supplementary for visuals). We perform K-level thresh-
olding and fusion. Suppose we have K thresholds δk for
confidence c, thresholded keypoints Xm,k, and their densi-
fied X-images ˆ
Xd,k, ∀k ∈[1, K], we propose a fusion block
F at the end of densification to fuse them. F is first pretrained
on single-image enhancement to suppress noise, deblur, and
sharpen edges. ˆ
Xd,k are input to F, and we mean pool over
k to output Xd, as the result for the CADF stage. We train
F with self-supervised losses, including cosine similarity
loss (Eq. 5) using SigLIP2 image encoder [78] along with
RGB-X self-matching loss (Eq. 7)—both using RGB images
to guide the enhancement and fusion for X-images.
Lcos(I, Xd) = 1 −
fSigLIP(I)⊤fSigLIP(Xd)
∥fSigLIP(I)∥2 ∥fSigLIP(Xd)∥2
,
(5)
where both RGB and X are capturing the same scene that
could be matched with the same descriptions, and thus the
cosine similarity loss helps maximize feature map similarity.
3.3. Self-Matching Filtering and 3D Consolidation
Self-Matching. We propose a self-matching mechanism
used both in fusion and the following filtering stage. Dif-
ferent from image matching— given two images, find a
matching set that maps coordinates from one to another—
now we have roughly aligned RGB-X pairs with the prior
that coordinates in one image should map to the same loca-
tions in another. We can use the image matcher as a judge to
evaluate whether the densified X-image is valid for matching
to the same coordinates.
For training F, we look into the patch features from the
transformer-based matcher and compute patch-level similar-
ity matrix A, which is computed by scaled dot product of
the RGB-X features.
A = FIF ⊤
X
τ
,
(6)
4

<!-- page 5 -->
Figure 4. Visual Results on METU-VisTIR-Cloudy. Our results attain much clearer, sharper, and smoother surface for rendering.
where τ is a scale factor, and FI and FX are features for
RGB and X in the coarse matching layer of the matcher [79].
The ideal similarity matrix should be diagonal. We com-
pute the loss that maximizes the diagonal sum and minimizes
the off-diagonal parts by
Lsim(A) = −Tr(A)
∥A∥F
+ λ∥A ⊙(ˆ1 −I)∥1
∥A∥F
,
(7)
where ∥.∥F is the Frobenius norm, Tr(·) is trace for a matrix,
I is the identity matrix, λ is a weight, and ˆ1 is the one-matrix.
Filtering. We further use A to reject wrongly densified
patches in Xd. Looking into the diagonal of A, we com-
pute a measure of concentration q = Q50(A)
Q99(A), with Q(·) as
the quantile function. A higher q indicates stronger self-
matching results and thus fewer patches need to be rejected,
while a lower q means the opposite. We take (1-q)-quantile
of A’s diagonal as the threshold and filter out patches of
smaller scores.
Fine-Stage Densification. We then perform the fine-stage
densification based on the filtered X-images. We follow the
previous stage to perform densification without K-levels,
and in Eq. 4 we take the normalized similarity score from A
as Cm.
RGB-X 3DGS. To further make per-frame densification
3D-consistent, we consolidate the densified X-images by
3DGS [28, 38, 96]. We train RGB-X 3DGS, where we
further add X-channels for each Gaussian. Unlike [36, 63],
where they render multi-modality (e.g., color and language
embedding) from separate channels and disentangle the GS
parameters, here we just keep one set of parameters. Because
Table 1. Results on METU-VisTIR-Cloudy. Results are the mean
of all six sequences. We compare with warping by different image
matchers, all trained and rendered by 3DGS.
Method
Icos ↑p30↑p50↑p70↑p90↑ITM↑ITcos↑
XoFTR [79]
0.62
25.13 27.49 29.31 31.48
0.69
0.39
LightGlue [51]
0.61
25.89 28.28 30.14 32.35
0.91
0.40
LoFTR [74]
0.66
29.38 32.07 33.95 36.04
0.89
0.45
MINIMAlg [66]
0.67
29.93 32.78 34.72 36.99
0.88
0.44
Ours
0.69
31.18 34.39 36.43 38.72
0.92
0.45
we find raw sensor imaging is often noisy or low-resolution,
which cannot match the higher quality of RGB, and the latter
can better position each 3D Gaussian precisely.
4. Experiments
X-Modalities and Datasets.
1. RGB-Thermal. The popular METU-VisTIR-Cloudy
Dataset [79], which includes six sequences captured by raw
and unpaired RGB and thermal sensors, is adopted for test-
ing. We also use four scenes from RGBT-Scenes [54] con-
taining paired RGB-thermal data for evaluation. We apply
random rotation, shift, perspective warping, and radial dis-
tortion to create unpaired RGB-thermal data. To train D for
this modality, we use synthetic RGB-thermal pairs created
by MINIMA [66], which uses the MegaDepth Dataset [48]
to create a large volume of paired data.
2. RGB-NIR. We adopt five sequences from RGB-NIR-
Stereo [39], including one shorter and four extracted from
two much longer sequences. We follow the same style as
RGBT-Scenes to create unpaired data. To train D, we adopt
5

<!-- page 6 -->
Figure 5. Comparison on Temporal Consistency for Image Generation. StyleBooth [25] generation for thermal images cannot guarantee
temporal consistency due to inherent ambiguity, while ours densification creates more consistent multi-views. NIR is closer to the visual
spectrum and thus easier to ensure consistency, but the specialized method PixNext [35] still cannot ensure the correct intensity. Compared
with our strategy, image translation from the original domain still suffers from inaccurate transformations, whereas our match-densify-
consolidate approach uses information from the target domain as anchors for densification and achieves better results.
Table 2. Comparison on Temporal Consistency. Compared
with pure image generation, our match-densify-consolidate strategy
obtains better multiview consistency from lower MEt3R score.
MEt3R↓
Sc1
Sc2
Sc3
Sc4
Sc5
Sc6
Mean
StyleBooth [25] 0.202 0.331 0.357 0.285 0.289 0.319 0.297
Ours
0.141 0.123 0.143 0.258 0.158 0.202 0.171
Deep-NIR [68] with synthetic paired RGB-NIR data.
3. RGB-SAR. We adopt three sub-Gigapixel satellite RGB-
SAR pairs from DDHR-HK [65] and cut them into 512x512
patches for evaluation. We do not perform RGB-X 3DGS
only on this modality since satellite images inherently cannot
capture multi-view 3D. Several RGB-SAR datasets [10, 46,
91, 103] are used to train D.
Implementation Details. We adopt the transformer-based
XoFTR [79] as the image matcher, which is trained on RGB-
thermal but also presents strong ability for across different
spectrum [66]. We choose N = 7 (-3 to +3 frame1), K = 3
levels, δ = 0.15, 0.3, 0.5, λ = 0.1, τ = 0.1, and c = 0.3 for
those area-sampled points. Fusion module F is an image-
enhancement network [86] and pretrained on DIV2K [3]
for image denoising, deblurring, and super-resolution. See
Supplementary for details of training D and F.
Metrics. For datasets with only unpaired sequences with-
out groundtruth, like METU-VisTIR-Cloudy, we calculate
(1) image cosine similarity (Icos) for RGB-X image fea-
tures from SigLIP-2 image encoder [78], (2) 30/50/70/90-
percentile (p30-p90) on the diagonal of the similarity ma-
trix of XoFTR for RGB-thermal, (3) image captioning and
matching score— We use popular MLLM BLIP-2 [44] to
first caption the RGB image and append ”, in a thermal im-
1The work reasonably assumes rough alignment. A standard RGB-X
data capture can trigger RGB-X shutters around the same time with similar
scene coverage, but hard to be exact or aligned, as this is affected by on-the-
fly system loading, physical displacement of sensors, or different intrinsics.
Figure 6. With the aid of sharper and clearer thermal images, the
RGB view synthesis quality is also slightly enhanced.
age.” to the text. The caption and thermal images are used to
calculate image-text matching score (ITM) and image-text
cosine similarity (ITcos). Specifically, the match/mismatch
score comes from a classification head that predicts two
logits zmatch and zmismatch from its linear alignment. Then
probmismatch =
ezmismatch
ezmatch+ezmismatch .
For datasets with groundtruth, we calculate standard
PSNR, SSIM, LPIPS as the fidelity scores, or RMSE/ MAE
if the maps carry physical units such as ◦C. To evaluate tem-
poral consistency, we use MEt3R [4], which lower scores
indicate better multi-view consistency.
Baselines. We include homography warping from different
image matchers, including XoFTR [79], LightGLUE [51]
LoFTR [74], and MINIMAlg [66] as the featured matcher,
followed by 3DGS training using their warped views. We
also include cross-modal image generation, where models
6

<!-- page 7 -->
Figure 7. Visual Results on RGB-NIR-Stereo. Our view synthesis showcases better structures closer to the groundtruth (GT).
are trained with specific modalities, such as StyleBooth [25]
used in [66] for RGB-thermal and [35] for RGB-NIR.
The work compares against the above methods without
3D priors for X, which aligns with our scope. We also
consider 3D reprojection with estimated metric depth (from
DepthAnythingV2 [94]), estimated camera pose (from essen-
tial matrix using matched keypoints), and groundtruth intrin-
sics for X. However, the pipeline cannot establish a reason-
able baseline because metric depth and cross-modal camera
pose estimation are far from robust in the wild [24, 88–90].
4.1. RGB-Thermal
METU-VisTIR-Cloudy. The dataset contains unpaired
RGB and thermal sequences without groundtruth, and we
adopt several metrics calculated against color images instead.
See Metrics. From Tab. 1, our method consistently performs
the best across all metrics, showing superior view-synthesis
quality and faithful alignment with associated images. Espe-
cially for p30-p90, the higher similarity scores validate that
our X-images make it easier for a pretrained image matcher
to build correspondences. Fig. 4 shows a visual comparison,
where our results show the most clear object structures of the
thermal images. Further, we find that we also attain sharper
RGB view synthesis (Fig. 6) than others due to better thermal
images used in 3D consolidation.
We then compare with image generation from RGB views.
We follow MINIMA [66], which uses StyleBooth [25] to
craft paired RGB-thermal. Inherent ambiguity between ap-
pearance and true temperature makes it hard to estimate the
exact temperature only from the appearance. This makes it
unstable to generate consistent thermal images, while ours
acquires true values by image matching and anchors those
points in densification to synthesize views conforming to the
true values. Tab. 2 shows that StyleBooth presents worse
multiview consistency, where higher MEt3R [4] indicates
Table 3. Comparison on RGBT-Scenes. We show results for train
and novel view synthesis. Each scene reports RMSE↓and MAE↓
in ◦C. Novel views from the official split contain simpler views
with lower errors than train views.
Train view
Building Landscape Parterre RoadBlock
Mean
RMSE MAE RMSE
MAE
RMSE MAE RMSE
MAE
RMSE MAE
XoFTR [79]
2.50 1.81 1.08 0.79 2.25 1.57 1.21 0.93 1.76 1.28
LightGLUE [51] 2.48 1.79 1.10 0.81 2.28 1.57 1.26 0.97 1.78 1.29
LoFTR [74]
2.44 1.75 1.06 0.77 2.20 1.52 1.36 0.99 1.77 1.26
MINIMA [66]
2.51 1.83 1.08 0.80 2.35 1.66 1.36 1.04 1.83 1.33
Ours
2.43 1.72 1.02 0.71 2.18 1.49 1.18 0.91 1.70 1.21
Novel View
Building Landscape Parterre RoadBlock
Mean
RMSE MAE RMSE
MAE
RMSE MAE RMSE
MAE
RMSE MAE
XoFTR [79]
2.70 1.92 1.02 0.70 1.34 0.84 0.47 0.40 1.38 0.97
LightGLUE [51] 2.78 2.00 1.01 0.73 1.21 0.71 0.85 0.68 1.46 1.03
LoFTR [74]
2.89 2.11 1.02 0.71 1.47 1.01 0.69 0.49 1.52 1.08
MINIMA [66]
2.55 1.78 0.98 0.65 0.91 0.58 0.33 0.26 1.19 0.82
Ours
2.33 1.70 0.97 0.66 0.81 0.55 0.35 0.27 1.12 0.80
worse consistency. Fig. 5 shows the differences in the upper.
RGBT-Scenes. We follow the official split and report train
and novel view synthesis results. Since the dataset contains
paired RGB-thermal views with real temperatures, we com-
pute RMSE and MAE in ◦C against groundtruth. Tab. 3
displays the results, where our method shows lower errors in
nearly all sequences.
4.2. RGB-NIR
RGB-NIR-Stereo NIR captures the near-infrared spectrum,
which is close to the visible range, and hence we compute
regular image quality metrics, including PSNR, SSIM, and
LPIPS, against groundtruth. Tab. 4 shows the comparison,
where our method consistently achieves better scores across
all the metrics, validating our pipeline. We show a visual
comparison in Fig. 7, where our method produces better
scene structure. In Fig. 5, the lower panel, we show a vi-
7

<!-- page 8 -->
Table 4. Results on RGB-NIR-Stereo. Image quality metrics are
shown. All methods are run with 3DGS.
Method
PSNR↑
SSIM↑
LPIPS↓
PixNext [35]
11.283
0.441
0.452
XoFTR [79]
14.846
0.321
0.486
LightGlue [51]
16.253
0.377
0.455
LoFTR [74]
20.179
0.551
0.356
MINIMA [66]
20.392
0.568
0.360
Ours
21.152
0.581
0.344
Table 5. Ablation Study. We ablate each component in turn from
the pipeline and report the average scores on RGB-NIR-Stereo.
Method
PSNR↑
SSIM↑
LPIPS↓
Ours (final)
21.152
0.581
0.344
-3DGS
21.042
0.597
0.378
-Self-matching & filtering
20.235
0.522
0.386
-DySPN confidence
19.621
0.508
0.396
-Multi-level thresholds
19.215
0.495
0.420
-Area sampling
16.454
0.408
0.467
sual comparison to a RGB→NIR image generation method,
PixNext [35]. The method shows better generation consis-
tency due to affinity to the visual spectrum, which makes
image translation easier. But it still fails to produce the
correct intensity compared to the groundtruth. This is also
reflected in Tab. 4 with lower image quality scores.
Ablation Studies. We show ablation studies in Tab. 5, where
we gradually remove each component and show the results,
including 3DGS, self-matching and filtering, DySPN con-
fidence, multi-level thresholds, and area sampling. From
the table, each proposed component contributes to the final
results. By integrating the image matching confidence into
densification, the PSNR increases by 1dB. The self-matching
and filtering help filter out poorly generated patches and
condition on more robust points for re-densification, which
accounts for 0.8dB improvement. 3DGS improves the per-
formance and gets better view synthesis.
Pre-GS Comparison.We compare results without the final
Gaussian Splatting, where we further discard the require-
ments for RGB camera poses and intrinsics from COLMAP.
Tab. 6 shows the comparison, where all the methods in com-
parison are run without 3DGS. Our method still performs
best. Notably, even without the GS-stage, we attain a mean
PSNR of 21.042 (Tab. 5, also averaged from 5 sequences in
Tab. 6), which is still better than all the methods run with
3DGS in Tab. 4. The results show that our sampling and
CADF strategy is successful.
4.3. RGB-SAR
We conduct experiments on SAR view synthesis. We seg-
ment three DDHR-HK SAR and satellite images into over-
lapping 512x512 patches, and perform matching and den-
Table 6. Comparison without 3DGS for all methods. PSNR↑on
each sequence is shown on RGB-NIR-Stereo.
Method
09-28-16-
48-17/1
09-28-16-
48-17/2
09-28-17-
06-04/1
09-28-17-
06-04/2
10-26-15-
53-03
PixNext [35]
11.19
11.99
12.81
12.22
8.65
XoFTR [79]
14.08
14.01
14.72
16.57
16.02
LightGlue [51] 14.03
14.10
14.61
16.67
16.07
LoFTR [74]
19.58
17.15
18.54
19.19
16.02
MINIMA [66] 21.16
18.80
19.67
21.76
19.70
Ours
21.47
20.27
20.24
22.67
20.56
Table 7. Comparison DDHR-HK SAR. We show image quality
metrics against groundtruth.
Method
PSNR↑
SSIM↑
LPIPS↓
XoFTR [79]
13.552
0.168
0.575
LoFTR [74]
13.462
0.164
0.615
LightGlue [51]
13.084
0.161
0.650
MINIMA [66]
14.849
0.229
0.377
Ours
17.102
0.302
0.339
sification on the patches. We do not perform 3DGS for all
the methods in comparison and show image quality metrics
against groundtruth. From Tab. 7, ours also scores the best in
all metrics, demonstrating our superiority in this challenging
task, where SAR signals are challenging for cross-modal
matching. See Supplementary for more results, visualization
including SAR’s, and other experiments, including feed-
forward 3D reconstruction models and discussion on depth-
reprojection.
5. Conclusion
We present the first study of cross-sensor view synthesis
across different types of modalities— studying a practical,
fundamental, yet widely overlooked problem in cross-modal
learning on data alignment. We propose a novel and scalable
framework in a match-densify-consolidate fashion to fulfill
the goal, including integrating image matching and cross-
modal densification together, using a carefully designed con-
fidence guidance mechanism, a self-matching approach to
further enhance quality, and consolidating cross-modalities
in a unified 3D field. Extensive experiments on diverse
sensor datasets validate our strategy. We expect the work
to relieve the burden of sensor calibration and advance the
popularity of cross-sensor learning in computer vision.
Limitations. First, the work focuses on static scenes and
does not address dynamic objects, which is an issue in 3D
consolidation and is still open for 3DGS research. Further,
sensors such as thermal cameras are usually noisy and low-
resolution [5]. Poor data quality can affect results and require
in-domain denoising or enhancement on the signal level.
Still, we rely on image matching and cannot solve extremely
homogeneous cases without effective descriptors, a general
bottleneck for matching-based solutions.
8

<!-- page 9 -->
References
[1] Rachael Abbott, Neil M Robertson, Jesus Martinez del Rin-
con, and Barry Connor. Unsupervised object detection via
lwir/rgb translation. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition Work-
shops, pages 90–91, 2020. 2
[2] NG Aditya, PB Dhruval, et al. Thermal voyager: A com-
parative study of rgb and thermal cameras for night-time
autonomous navigation. In 2024 IEEE International Con-
ference on Robotics and Automation (ICRA), pages 14116–
14122. IEEE, 2024. 2
[3] Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge
on single image super-resolution: Dataset and study. In The
IEEE Conference on Computer Vision and Pattern Recogni-
tion (CVPR) Workshops, 2017. 6
[4] Mohammad Asim, Christopher Wewer, Thomas Wimmer,
Bernt Schiele, and Jan Eric Lenssen. Met3r: Measuring
multi-view consistency in generated images. In Proceedings
of the Computer Vision and Pattern Recognition Conference,
pages 6034–6044, 2025. 6, 7
[5] Arnaud Barral, Pablo Arias, and Axel Davy. Fixed pattern
noise removal for multi-view single-sensor infrared cam-
era. In Proceedings of the IEEE/CVF Winter Conference on
Applications of Computer Vision, pages 1669–1678, 2024. 8
[6] Amanda Berg, Jorgen Ahlberg, and Michael Felsberg. Gen-
erating visible spectrum images from thermal infrared. In
Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition Workshops, pages 1143–1152,
2018. 2
[7] Mario Bijelic, Tobias Gruber, Fahim Mannan, Florian Kraus,
Werner Ritter, Klaus Dietmayer, and Felix Heide. Seeing
through fog without seeing fog: Deep multimodal sensor
fusion in unseen adverse weather. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 11682–11692, 2020. 2
[8] Matthew Brown, Richard Szeliski, and Simon Winder. Multi-
image matching using multi-scale oriented patches. In 2005
IEEE Computer Society Conference on Computer Vision
and Pattern Recognition (CVPR’05), pages 510–517. IEEE,
2005. 3
[9] Nicolas Bustos, Mehrsa Mashhadi, Susana K Lai-Yuen,
Sudeep Sarkar, and Tapas K Das. A systematic literature
review on object detection using near infrared and thermal
images. Neurocomputing, 560:126804, 2023. 1
[10] Ahmed Alejandro Cardona-Mesa, Rub´en Dar´ıo V´asquez-
Salazar,
Luis G´omez,
Carlos M Travieso-Gonz´alez,
Andr´es F Garavito-Gonz´alez, Esteban V´asquez-Cano, and
Jean Pierre D´ıaz-Paz. Dataset of sentinel-1 sar and sentinel-2
rgb-ndvi imagery. Data in Brief, 57:111160, 2024. 6
[11] Hongkai Chen, Zixin Luo, Lei Zhou, Yurun Tian, Ming-
min Zhen, Tian Fang, David Mckinnon, Yanghai Tsin, and
Long Quan. Aspanformer: Detector-free image matching
with adaptive span transformer. In European conference on
computer vision, pages 20–36. Springer, 2022. 3
[12] Yukyung Choi, Namil Kim, Soonmin Hwang, Kibaek Park,
Jae Shin Yoon, Kyounghwan An, and In So Kweon. Kaist
multi-spectral day/night data set for autonomous and assisted
driving. IEEE Transactions on Intelligent Transportation
Systems, 19(3):934–948, 2018. 1
[13] Sri Aditya Deevi, Connor Lee, Lu Gan, Sushruth Nagesh,
Gaurav Pandey, and Soon-Jo Chung. Rgb-x object detection
via scene-specific fusion modules. In Proceedings of the
IEEE/CVF Winter Conference on Applications of Computer
Vision, pages 7366–7375, 2024. 2
[14] Fuqin Deng, Hua Feng, Mingjian Liang, Hongmin Wang,
Yong Yang, Yuan Gao, Junfeng Chen, Junjie Hu, Xiyue
Guo, and Tin Lun Lam. Feanet: Feature-enhanced attention
network for rgb-thermal real-time semantic segmentation.
In 2021 IEEE/RSJ international conference on intelligent
robots and systems (IROS), pages 4467–4473. IEEE, 2021.
2
[15] Yuxin Deng and Jiayi Ma. Redfeat: Recoupling detection
and description for multimodal feature learning. IEEE Trans-
actions on Image Processing, 32:591–602, 2022. 3
[16] Mihai Dusmanu, Ignacio Rocco, Tomas Pajdla, Marc Polle-
feys, Josef Sivic, Akihiko Torii, and Torsten Sattler. D2-net:
A trainable cnn for joint description and detection of lo-
cal features. In Proceedings of the ieee/cvf conference on
computer vision and pattern recognition, pages 8092–8101,
2019. 3
[17] Johan Edstedt, Ioannis Athanasiadis, M˚arten Wadenb¨ack,
and Michael Felsberg.
Dkm: Dense kernelized feature
matching for geometry estimation. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 17765–17775, 2023. 3
[18] Johan Edstedt,
Qiyu Sun,
Georg B¨okman,
M˚arten
Wadenb¨ack, and Michael Felsberg. Roma: Robust dense
feature matching. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition (CVPR),
pages 19790–19800, 2024. 3
[19] Ruoyu Fan, Wang Zhao, Matthieu Lin, Qi Wang, Yong-Jin
Liu, and Wenping Wang. Generalizable thermal-based depth
estimation via pre-trained visual foundation model. In 2024
IEEE International Conference on Robotics and Automation
(ICRA), pages 14614–14621. IEEE, 2024. 1
[20] Zhen Feng, Yanning Guo, and Yuxiang Sun. Cekd: Cross-
modal edge-privileged knowledge distillation for seman-
tic scene understanding using only thermal images. IEEE
Robotics and Automation Letters, 8(4):2205–2212, 2023. 2
[21] Martin A Fischler and Robert C Bolles. Random sample
consensus: a paradigm for model fitting with applications to
image analysis and automated cartography. Communications
of the ACM, 24(6):381–395, 1981. 3
[22] Gianni Franchi, Marwane Hariat, Xuanlong Yu, Nacim
Belkhir, Antoine Manzanera, and David Filliat. Infraparis:
A multi-modal and multi-task autonomous driving dataset.
In Proceedings of the IEEE/CVF Winter Conference on Ap-
plications of Computer Vision, pages 2973–2983, 2024. 1
[23] Oriel Frigo, Lucien Martin-Gaffe, and Catherine Wacongne.
Doodlenet: Double deeplab enhanced feature fusion for
thermal-color semantic segmentation. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 3021–3029, 2022. 2
[24] Yuliang Guo, Sparsh Garg, S Mahdi H Miangoleh, Xinyu
Huang, and Liu Ren. Depth any camera: Zero-shot metric
9

<!-- page 10 -->
depth estimation from any camera. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
26996–27006, 2025. 7
[25] Zhen Han, Chaojie Mao, Zeyinzi Jiang, Yulin Pan, and
Jingfeng Zhang. Stylebooth: Image style editing with multi-
modal instruction. In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision (ICCV) Workshops,
pages 1947–1957, 2025. 6, 7
[26] Lingyi Hong, Shilin Yan, Renrui Zhang, Wanyun Li, Xinyu
Zhou, Pinxue Guo, Kaixun Jiang, Yiting Chen, Jinglun Li,
Zhaoyu Chen, et al. Onetracker: Unifying visual object
tracking with foundation models and efficient tuning. In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 19079–19091, 2024. 2
[27] Xiaojun Hou, Jiazheng Xing, Yijie Qian, Yaowei Guo, Shuo
Xin, Junhao Chen, Kai Tang, Mengmeng Wang, Zhengkai
Jiang, Liang Liu, et al. Sdstrack: Self-distillation symmetric
adapter learning for multi-modal visual object tracking. In
Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 26551–26561, 2024.
2
[28] Zixun Huang, Cho-Ying Wu, Yuliang Guo, Xinyu Huang,
and Liu Ren. 3dgeer: 3d gaussian rendering made exact
and efficient for generic cameras. In The Fourteenth In-
ternational Conference on Learning Representations, 2026.
5
[29] Tianrui Hui, Zizheng Xun, Fengguang Peng, Junshi Huang,
Xiaoming Wei, Xiaolin Wei, Jiao Dai, Jizhong Han, and Si
Liu. Bridging search region interaction with template for
rgb-t tracking. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 13630–
13639, 2023. 2
[30] Mohd Shawal Jadin and Kamarul Hawari Ghazali. Gas
leakage detection using thermal imaging technique. In 2014
UKSim-AMSS 16th International Conference on Computer
Modelling and Simulation, pages 302–306. IEEE, 2014. 1
[31] Wei Ji, Jingjing Li, Cheng Bian, Zongwei Zhou, Jiaying
Zhao, Alan L Yuille, and Li Cheng. Multispectral video
semantic segmentation: A benchmark dataset and baseline.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 1094–1104, 2023. 2
[32] Hanwen Jiang, Arjun Karpur, Bingyi Cao, Qixing Huang,
and Andr´e Araujo. Omniglue: Generalizable feature match-
ing with foundation model guidance. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 19865–19875, 2024. 3
[33] Jiajun Jiang, Xingxin Chen, Weichen Dai, Zelin Gao, and
Yu Zhang. Thermal-inertial slam for the environments with
challenging illumination. IEEE Robotics and Automation
Letters, 7(4):8767–8774, 2022. 2
[34] Shuangping Jin, Bingbing Yu, Minhao Jing, Yi Zhou, Jiajun
Liang, and Renhe Ji. Darkvisionnet: Low-light imaging via
rgb-nir fusion with deep inconsistency prior. In Proceedings
of the AAAI Conference on Artificial Intelligence, pages
1104–1112, 2022. 2
[35] Youngwan Jin, Incheol Park, Hanbin Song, Hyeongjin Ju,
Yagiz Nalcakan, and Shiho Kim. Pix2next: Leveraging
vision foundation models for rgb to nir image translation.
Technologies, 2025. 6, 7, 8
[36] Saimouli Katragadda, Cho-Ying Wu, Yuliang Guo, Xinyu
Huang, Guoquan Huang, and Liu Ren. Online language
splatting. In Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV), 2025. 5
[37] Nikhil Keetha, Norman M¨uller, Johannes Sch¨onberger,
Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno
Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes,
et al. Mapanything: Universal feed-forward metric 3d recon-
struction. In 3DV, 2026. 3
[38] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
George Drettakis, et al. 3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 5
[39] Jinnyeong Kim and Seung-Hwan Baek. Pixel-aligned rgb-nir
stereo imaging and dataset for robot vision. In Proceedings
of the Computer Vision and Pattern Recognition Conference,
pages 11482–11492, 2025. 1, 5
[40] Janghyun Kim, Ukcheol Shin, Seokyong Heo, and Jinsun
Park. Exploiting cross-modal cost volume for multi-sensor
depth estimation. In Proceedings of the Asian Conference
on Computer Vision, pages 1420–1436, 2024. 2
[41] Da Kong, Yu Zhang, and Weichen Dai. Direct near-infrared-
depth visual slam with active lighting. IEEE robotics and
Automation Letters, 6(4):7057–7064, 2021. 2
[42] Chenglong Li, Lei Liu, Andong Lu, Qing Ji, and Jin Tang.
Challenge-aware rgbt tracking. In European conference on
computer vision, pages 222–237. Springer, 2020. 2
[43] Haotian Li, Henry K Chu, and Yuxiang Sun. Improving
rgb-thermal semantic scene understanding with synthetic
data augmentation for autonomous driving. IEEE Robotics
and Automation Letters, 2025. 1
[44] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-
2: Bootstrapping language-image pre-training with frozen
image encoders and large language models. In Interna-
tional conference on machine learning, pages 19730–19742.
PMLR, 2023. 6
[45] Ting Li, Mao Ye, Tianwen Wu, Nianxin Li, Shuaifeng Li,
Song Tang, and Luping Ji.
Pseudo visible feature fine-
grained fusion for thermal object detection. In Proceedings
of the Computer Vision and Pattern Recognition Conference,
pages 6710–6719, 2025. 2
[46] Xue Li, Guo Zhang, Hao Cui, Shasha Hou, Shunyao Wang,
Xin Li, Yujia Chen, Zhijiang Li, and Li Zhang. Mcanet: A
joint semantic segmentation framework of optical and sar
images for land use classification. International Journal
of Applied Earth Observation and Geoinformation, 106:
102638, 2022. 6
[47] You Li, Julien Moreau, and Javier Ibanez-Guzman. Emer-
gent visual sensors for autonomous vehicles. IEEE Trans-
actions on Intelligent Transportation Systems, 24(5):4716–
4737, 2023. 1
[48] Zhengqi Li and Noah Snavely. Megadepth: Learning single-
view depth prediction from internet photos. In Proceedings
of the IEEE conference on computer vision and pattern
recognition, pages 2041–2050, 2018. 5
10

<!-- page 11 -->
[49] Mingjian Liang, Junjie Hu, Chenyu Bao, Hua Feng, Fuqin
Deng, and Tin Lun Lam. Explicit attention-enhanced fu-
sion for rgb-thermal perception tasks. IEEE Robotics and
Automation Letters, 8(7):4060–4067, 2023. 2
[50] Yuankai Lin, Tao Cheng, Qi Zhong, Wending Zhou, and
Hua Yang. Dynamic spatial propagation network for depth
completion.
In Proceedings of the AAAI conference on
artificial intelligence, pages 1638–1646, 2022. 3
[51] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Polle-
feys. Lightglue: Local feature matching at light speed. In
Proceedings of the IEEE/CVF international conference on
computer vision, pages 17627–17638, 2023. 3, 5, 6, 7, 8
[52] Jinyuan Liu, Zhu Liu, Guanyao Wu, Long Ma, Risheng
Liu, Wei Zhong, Zhongxuan Luo, and Xin Fan.
Multi-
interactive feature learning and a full-time multi-modality
benchmark for image fusion and segmentation. In Proceed-
ings of the IEEE/CVF international conference on computer
vision, pages 8115–8124, 2023. 2
[53] Jinyuan Liu, Guanyao Wu, Zhu Liu, Di Wang, Zhiying Jiang,
Long Ma, Wei Zhong, and Xin Fan. Infrared and visible im-
age fusion: From data compatibility to task adaption. IEEE
Transactions on Pattern Analysis and Machine Intelligence,
2024. 2
[54] Rongfeng Lu, Hangyu Chen, Zunjie Zhu, Yuhang Qin, Ming
Lu, Chenggang Yan, et al. Thermalgaussian: Thermal 3d
gaussian splatting. In The Thirteenth International Confer-
ence on Learning Representations, 2025. 2, 5
[55] Rishabh Madan, Skyler Valdez, David Kim, Sujie Fang,
Luoyan Zhong, Diego T Virtue, and Tapomayukh Bhat-
tacharjee. Rabbit: A robot-assisted bed bathing system with
multimodal perception and integrated compliance. In Pro-
ceedings of the 2024 ACM/IEEE international conference
on human-robot interaction, pages 472–481, 2024. 2
[56] Jonas Mirlach, Lei Wan, Andreas Wiedholz, Hannan Ejaz
Keen, and Andreas Eich. R-livit: A lidar-visual-thermal
dataset enabling vulnerable road user focused roadside per-
ception. In ICCV, 2025. 1
[57] Farzeen Munir, Shoaib Azam, and Moongu Jeon. Sstn:
Self-supervised domain adaptation thermal object detection
for autonomous driving. In 2021 IEEE/RSJ international
conference on intelligent robots and systems (IROS), pages
206–213. IEEE, 2021. 1
[58] Farzeen Munir, Shoaib Azam, Muhammd Aasim Rafique,
Ahmad Muqeem Sheri, Moongu Jeon, and Witold Pedrycz.
Exploring thermal images for object detection in underexpo-
sure regions for autonomous driving. Applied soft computing,
121:108793, 2022. 1
[59] Junjie Ni, Guofeng Zhang, Guanglin Li, Yijin Li, Xinyang
Liu, Zhaoyang Huang, and Hujun Bao.
Eto: Efficient
transformer-based local feature matching by organizing mul-
tiple homography hypotheses. Advances in Neural Informa-
tion Processing Systems, 37:60260–60274, 2024. 3
[60] Yuki Ono, Eduard Trulls, Pascal Fua, and Kwang Moo Yi.
Lf-net: Learning local features from images. Advances in
neural information processing systems, 31, 2018. 3
[61] Edoardo Palladin, Roland Dietze, Praveen Narayanan, Mario
Bijelic, and Felix Heide. Samfusion: Sensor-adaptive multi-
modal fusion for 3d object detection in adverse weather. In
European Conference on Computer Vision, pages 484–503.
Springer, 2024. 2
[62] Jay N Paranjape, Celso de Melo, and Vishal M Patel. F-
vita: Foundation model guided visible to thermal translation.
arXiv preprint arXiv:2504.02801, 2025. 1
[63] Qucheng Peng, Benjamin Planche, Zhongpai Gao, Meng
Zheng, Anwesa Choudhuri, Terrence Chen, Chen Chen, and
Ziyan Wu. 3d vision-language gaussian splatting. In The
Thirteenth International Conference on Learning Represen-
tations, 2025. 5
[64] Guilherme Potje, Felipe Cadar, Andr´e Araujo, Renato Mar-
tins, and Erickson R Nascimento. Xfeat: Accelerated fea-
tures for lightweight image matching. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 2682–2691, 2024. 3
[65] Bo Ren, Shibin Ma, Biao Hou, Danfeng Hong, Jocelyn
Chanussot, Jianlong Wang, and Licheng Jiao. A dual-stream
high resolution network: Deep fusion of gf-2 and gf-3 data
for land cover classification. International Journal of Ap-
plied Earth Observation and Geoinformation, 112:102896,
2022. 6
[66] Jiangwei Ren, Xingyu Jiang, Zizhuo Li, Dingkang Liang,
Xin Zhou, and Xiang Bai. Minima: Modality invariant
image matching. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pages 23059–23068, 2025.
2, 3, 5, 6, 7, 8
[67] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang
Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng
Yan, et al. Grounded sam: Assembling open-world models
for diverse visual tasks. arXiv preprint arXiv:2401.14159,
2024. 3
[68] Inkyu Sa, Jong Yoon Lim, Ho Seok Ahn, and Bruce Mac-
Donald. deepnir: Datasets for generating synthetic nir im-
ages and improved fruit detection system using deep learning
techniques. Sensors, 22(13):4721, 2022. 6
[69] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz,
and Andrew Rabinovich. Superglue: Learning feature match-
ing with graph neural networks.
In Proceedings of the
IEEE/CVF conference on computer vision and pattern recog-
nition, pages 4938–4947, 2020. 3
[70] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited. In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
4104–4113, 2016. 1
[71] Furqan Ahmed Shaik, Abhishek Reddy, Nikhil Reddy Billa,
Kunal Chaudhary, Sunny Manchanda, and Girish Varma.
Idd-aw: A benchmark for safe and robust segmentation of
drive scenes in unstructured traffic and adverse weather. In
Proceedings of the IEEE/CVF Winter Conference on Appli-
cations of Computer Vision, pages 4614–4623, 2024. 2
[72] Yan Shi, Jun-Xiong Cai, Yoli Shavit, Tai-Jiang Mu, Wensen
Feng, and Kai Zhang. Clustergnn: Cluster-based coarse-
to-fine graph neural network for efficient feature matching.
In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pages 12517–12526, 2022. 3
[73] Ukcheol Shin, Jinsun Park, and In So Kweon. Deep depth
estimation from thermal image.
In Proceedings of the
11

<!-- page 12 -->
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 1043–1053, 2023. 2
[74] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and
Xiaowei Zhou. Loftr: Detector-free local feature matching
with transformers. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
8922–8931, 2021. 3, 5, 6, 7, 8
[75] Yuxiang Sun, Weixun Zuo, and Ming Liu. Rtfnet: Rgb-
thermal fusion network for semantic segmentation of urban
scenes. IEEE Robotics and Automation Letters, 4(3):2576–
2583, 2019. 2
[76] Hao Tang, Zechao Li, Dong Zhang, Shengfeng He, and
Jinhui Tang. Divide-and-conquer: Confluent triple-flow
network for rgb-t salient object detection. IEEE Transactions
on Pattern Analysis and Machine Intelligence, 2024. 2
[77] Engin Tola, Vincent Lepetit, and Pascal Fua. A fast local
descriptor for dense matching. In 2008 IEEE conference on
computer vision and pattern recognition, pages 1–8. IEEE,
2008. 3
[78] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muham-
mad Ferjad Naeem,
Ibrahim Alabdulmohsin,
Nikhil
Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil
Mustafa, et al. Siglip 2: Multilingual vision-language en-
coders with improved semantic understanding, localization,
and dense features. arXiv preprint arXiv:2502.14786, 2025.
4, 6
[79] ¨Onder Tuzcuo˘glu, Aybora K¨oksal, Bu˘gra Sofu, Sinan
Kalkan, and A Aydin Alatan. Xoftr: Cross-modal feature
matching transformer. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition Work-
shops, pages 4275–4286, 2024. 2, 3, 5, 6, 7, 8
[80] Michał Tyszkiewicz, Pascal Fua, and Eduard Trulls. Disk:
Learning local features with policy gradient. Advances in
neural information processing systems, 33:14254–14265,
2020. 3
[81] Zifu Wan, Pingping Zhang, Yuhao Wang, Silong Yong, Si-
mon Stepputtis, Katia Sycara, and Yaqi Xie. Sigma: Siamese
mamba network for multi-modal semantic segmentation. In
2025 IEEE/CVF Winter Conference on Applications of Com-
puter Vision (WACV), pages 1734–1744. IEEE, 2025. 2
[82] Bing Wang, Changhao Chen, Zhaopeng Cui, Jie Qin,
Chris Xiaoxuan Lu, Zhengdi Yu, Peijun Zhao, Zhen Dong,
Fan Zhu, Niki Trigoni, et al. P2-net: Joint description and
detection of local features for pixel and point matching. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 16004–16013, 2021. 3
[83] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. Vggt:
Visual geometry grounded transformer. In Proceedings of
the Computer Vision and Pattern Recognition Conference,
pages 5294–5306, 2025. 3
[84] Qing Wang, Jiaming Zhang, Kailun Yang, Kunyu Peng, and
Rainer Stiefelhagen. Matchformer: Interleaving attention
in transformers for feature matching. In Proceedings of
the Asian conference on computer vision, pages 2746–2762,
2022. 3
[85] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris
Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d
vision made easy. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, pages
20697–20709, 2024. 3
[86] Yan Wang. Edge-enhanced feature distillation network for
efficient super-resolution. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR) Workshops, pages 777–785, 2022. 6
[87] Cho-Ying Wu and Ulrich Neumann. Scene completeness-
aware lidar depth completion for driving scenario.
In
ICASSP 2021-2021 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP), pages
2490–2494. IEEE, 2021. 2
[88] Cho-Ying Wu, Jialiang Wang, Michael Hall, Ulrich Neu-
mann, and Shuochen Su. Toward practical monocular indoor
depth estimation. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition (CVPR),
pages 3814–3824, 2022. 7
[89] Cho-Ying Wu, Quankai Gao, Chin-Cheng Hsu, Te-Lin Wu,
Jing-Wen Chen, and Ulrich Neumann. Inspacetype: Recon-
sider space type in indoor monocular depth estimation. In
The British Machine Vision Conference (BMVC), 2024.
[90] Cho-Ying Wu, Yiqi Zhong, Junying Wang, and Ulrich Neu-
mann. Boosting generalizability towards zero-shot cross-
dataset single-image indoor depth by meta-initialization.
In 2024 IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS), pages 10051–10058. IEEE,
2024. 7
[91] Yuming Xiang, Rongshu Tao, Feng Wang, Hongjian You,
and Bing Han. Automatic registration of optical and sar
images via improved phase congruency model. IEEE Jour-
nal of Selected Topics in Applied Earth Observations and
Remote Sensing, 13:5847–5861, 2020. 6
[92] Jiuhong Xiao, Roshan Nayak, Ning Zhang, Daniel Tortei,
and Giuseppe Loianno. Thermalgen: Style-disentangled
flow-based generative models for rgb-to-thermal image trans-
lation. arXiv preprint arXiv:2509.24878, 2025. 2
[93] Qin Xu, Yiming Mei, Jinpei Liu, and Chenglong Li. Multi-
modal cross-layer bilinear pooling for rgbt tracking. IEEE
Transactions on Multimedia, 24:567–580, 2021. 2
[94] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xi-
aogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth
anything v2. Advances in Neural Information Processing
Systems, 37:21875–21911, 2024. 7
[95] Meng Yang, Fan Fan, Zizhuo Li, Songchu Deng, Yong Ma,
and Jiayi Ma. Distillmatch: Leveraging knowledge distil-
lation from vision foundation model for multimodal image
matching. arXiv preprint arXiv:2509.16017, 2025. 2
[96] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and
Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splat-
ting. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 19447–19456,
2024. 5
[97] Jiaming Zhang, Huayao Liu, Kailun Yang, Xinxin Hu, Ruip-
ing Liu, and Rainer Stiefelhagen. Cmx: Cross-modal fusion
for rgb-x semantic segmentation with transformers. IEEE
Transactions on intelligent transportation systems, 24(12):
14679–14694, 2023. 2
12

<!-- page 13 -->
[98] Pengyu Zhang, Jie Zhao, Dong Wang, Huchuan Lu, and Xi-
ang Ruan. Visible-thermal uav tracking: A large-scale bench-
mark and new baseline. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 8886–8895, 2022. 2
[99] Qiang Zhang, Shenlu Zhao, Yongjiang Luo, Dingwen Zhang,
Nianchang Huang, and Jungong Han. Abmdrnet: Adaptive-
weighted bi-directional modality difference reduction net-
work for rgb-t semantic segmentation. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 2633–2642, 2021. 2
[100] Tianlu Zhang, Hongyuan Guo, Qiang Jiao, Qiang Zhang,
and Jungong Han. Efficient rgb-t tracking via cross-modality
distillation. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 5404–5413,
2023. 2
[101] Zhen Zhang and Wee Sun Lee. Deep graphical feature
learning for the feature matching problem. In Proceedings
of the IEEE/CVF International Conference on Computer
Vision, pages 5087–5096, 2019. 3
[102] Jiayi Zhao, Fei Teng, Kai Luo, Guoqiang Zhao, Zhiyong
Li, Xu Zheng, and Kailun Yang. Unveiling the potential
of segment anything model 2 for rgb-thermal semantic seg-
mentation with language guidance. IEEE/RSJ international
conference on intelligent robots and systems (IROS), 2025.
1
[103] Xuyang Zhao, Lijun Zhao, Hongyi Li, Zheng Zhang, and
Ping Tang. A multimodal visible-sar dataset for airport
detection in remote sensing imagery. Science Data Bank,
Dataset, Version 1.0, 2025. Available at https://doi.
org/10.57760/sciencedb.j00001.01276. 6
[104] Xu Zheng, Yuanhuiyi Lyu, and Lin Wang.
Learning
modality-agnostic representation for semantic segmentation
from any modalities. In European Conference on Computer
Vision, pages 146–165. Springer, 2024. 2
[105] Wujie Zhou, Jinfu Liu, Jingsheng Lei, Lu Yu, and Jenq-Neng
Hwang. Gmnet: Graded-feature multilabel-learning network
for rgb-thermal urban scene semantic segmentation. IEEE
Transactions on Image Processing, 30:7790–7802, 2021. 2
[106] Yiming Zuo and Jia Deng. Ogni-dc: Robust depth comple-
tion with optimization-guided neural iterations. In European
Conference on Computer Vision, pages 78–95. Springer,
2024. 3
[107] Yiming Zuo, Willow Yang, Zeyu Ma, and Jia Deng. Omni-
dc: Highly robust depth completion with multiresolution
depth integration. In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision, pages 9287–9297,
2025. 3
13
