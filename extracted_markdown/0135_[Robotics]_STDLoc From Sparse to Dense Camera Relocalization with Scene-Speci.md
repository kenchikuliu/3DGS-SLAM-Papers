<!-- page 1 -->
From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from
Feature Gaussian Splatting
Zhiwei Huang1,2
Hailin Yu2†
Yichun Shentu2
Jin Yuan2
Guofeng Zhang1†
1State Key Lab of CAD & CG, Zhejiang University
2SenseTime Research
Abstract
This paper presents a novel camera relocalization method,
STDLoc, which leverages Feature Gaussian as scene repre-
sentation. STDLoc is a full relocalization pipeline that can
achieve accurate relocalization without relying on any pose
prior. Unlike previous coarse-to-fine localization methods
that require image retrieval first and then feature matching,
we propose a novel sparse-to-dense localization paradigm.
Based on this scene representation, we introduce a novel
matching-oriented Gaussian sampling strategy and a scene-
specific detector to achieve efficient and robust initial pose
estimation. Furthermore, based on the initial localization
results, we align the query feature map to the Gaussian
feature field by dense feature matching to enable accu-
rate localization. The experiments on indoor and outdoor
datasets show that STDLoc outperforms current state-of-
the-art localization methods in terms of localization accu-
racy and recall. Our code is available on the project web-
site: https://zju3dv.github.io/STDLoc.
1. Introduction
Camera relocalization is a fundamental task in computer vi-
sion. It involves estimating the 6DoF camera pose from a
query image relative to a pre-built scene map. This capa-
bility has broad applications in fields such as AR/VR, au-
tonomous driving, and robotics.
An appropriate scene representation is critical for cam-
era relocalization. Traditional methods [25, 30, 43, 44, 46,
54, 55] leverage sparse 3D point clouds pre-reconstructed
by Structure-from-Motion (SfM) [48] as the scene repre-
sentation, with each point associated with one or more 2D
descriptors [31].
In the localization stage, reliable local
features [14, 31, 40, 41] are first extracted from the query
image, and these features are matched either with refer-
ence images [44, 52] or directly with the 3D point cloud
†Corresponding authors.
This work was partially supported by NSF of China (No.62425209).
teaser
Landmarks
Feature Field
Query Image
Stitched Localization Result
Sparse
Dense
<>
Figure 1. STDLoc: Sparse-to-Dense Localization. We lever-
age Feature Gaussian as the scene representation, which supports
direct 2D-3D sparse matching on landmarks and enables the align-
ment of the query feature map to the feature field through dense
matching.
[43, 46, 62, 65]. Then, the 6DoF pose is estimated using
the Perspective-n-Point (PnP) [24] algorithm based on the
established 2D-3D correspondences. This type of method,
also called feature matching-based localization, can achieve
high localization accuracy in rich-texture scenarios but de-
grade in weak-texture environments [50] due to insufficient
feature correspondences.
Semi-dense [18, 52] or dense
matching [16] can effectively alleviate this problem, but
makes SfM computation unacceptable at the mapping stage.
Mesh can provide dense depth, but the localization accuracy
is seriously affected by artifacts [38, 39].
Many works have explored encoding scene information
into a deep neural network. They train a neural network to
regress absolute poses directly (APR) [9, 20] or scene co-
ordinates (SCR) [8, 50]. The APR method generally has
1
arXiv:2503.19358v1  [cs.CV]  25 Mar 2025

<!-- page 2 -->
limited accuracy and poor generalization to unseen views
[7]. Due to the regression of dense scene coordinates, the
SCR method is more accurate than the feature matching-
based localization method in indoor weak-texture scenes
[6].
However, since it directly encodes the scene infor-
mation into the network weights, it cannot dynamically ad-
just the network capacity according to the size of the target
scene. Therefore, its accuracy in large outdoor scenes is
relatively limited [45].
Recently, implicit scene representations represented by
NeRF [34] and explicit scene representations represented by
3D Gaussian Splatting (GS) [21] have demonstrated photo-
graphic novel view synthesis ability. Therefore, some works
explore leveraging NeRF [64, 69] and Gaussian [2, 53]
as scene representation to design localization algorithms,
showing promising results. However, most current meth-
ods do not have a full pipeline. Some methods [10, 36]
utilize the novel view synthesis capability of NeRF or GS
for data augmentation. Another direction of research fo-
cuses on pose refinement through iterative optimization. Al-
though some of the current methods [3, 53, 64] achieve high
accuracy by minimizing the photometric error, which is un-
suitable for environments with illumination changes. More-
over, these approaches rely on external methods to provide
initial pose estimates.
To address the abovementioned limitations, we propose
STDLoc, a novel camera relocalization method based on
Gaussian Splatting, as shown in Fig. 1. Inspired by Feature
3DGS [70], we leverage feature field distillation to extend
the functionality of the original GS so that it can represent
not only the radiance field but also the feature field, which
we call Feature Gaussian and use as the scene representa-
tion for localization. Based on this representation, we make
the following contributions. (1) We propose a novel sparse-
to-dense camera relocalization pipeline that leverages Fea-
ture Gaussian as the scene representation. The proposed
localization method uses sparse features to obtain the ini-
tial pose and dense features for pose refinement, which en-
ables accurate camera relocalization. Instead of perform-
ing image retrieval and then feature matching, this pipeline
achieves a new coarse-to-fine localization paradigm. (2) We
introduce a novel matching-oriented sampling strategy to
address the challenge of selecting landmarks from millions
of Gaussians. This strategy significantly reduces the num-
ber of Gaussians, selecting only a small subset while ensur-
ing they are multi-view consistent and evenly distributed.
(3) Directly matching the dense feature map with the sam-
pled landmarks still results in an unacceptable computa-
tional load. Therefore, we introduce a scene-specific detec-
tor that can effectively detect landmarks from the extracted
dense feature map. The proposed scene-specific detector
can be trained in a self-supervised manner. (4) Based on
these landmarks, the camera pose can be easily estimated
by feature matching and the PnP algorithm, then refined by
aligning the dense feature map with the feature field. We
conducted extensive experiments to validate the proposed
pipeline’s effectiveness. The results indicate that our ap-
proach surpasses state-of-the-art methods in terms of local-
ization accuracy and recall.
2. Related Work
We categorize the related works into three sections based
on scene representation: structure-based (feature matching-
based) methods, neural-based methods, and radiance field-
based methods.
2.1. Structure-Based Methods
Structure-based methods [15, 22, 38, 43, 46] have demon-
strated remarkable success in recent years.
These ap-
proaches typically involve several key steps: feature extrac-
tion [14, 31], feature matching [18, 27, 44], and pose es-
timation [24] inside a RANSAC [17] loop. Active Search
[46] directly matches 2D features to the 3D point cloud to
obtain 2D-3D correspondences. HLoc [43] employs image
retrieval method [1] to achieve coarse-to-fine localization.
This pipeline can integrate different feature detection meth-
ods [14, 40, 41] and feature matching approaches [44, 60]
to improve localization performance. Since sparse feature
matching struggles with weak-texture challenges, LoFTR
[52] adopts a detector-free manner and produces semi-dense
matches to improve accuracy and robustness.
Structure-
based methods make good use of scene geometry informa-
tion, providing high accuracy.
2.2. Neural-Based Methods
Neural-based methods can be divided into absolute pose
regression (APR) and scene coordinate regression (SCR),
which use neural networks to represent the target scene im-
plicitly. APR methods [10, 20, 35, 49] predict the absolute
pose from a single query image. Despite their simplicity
and efficiency, APR methods are often limited in accuracy
and generalization in large-scale environments. SCR meth-
ods do not directly predict camera pose but instead infer the
dense scene coordinates [4, 15, 42, 47, 56, 58, 61]. Ben-
efiting from dense correspondences, SCR methods gener-
ally outperform structure-based methods in indoor scenes.
DSAC* [6] and ACE [8] have greatly reduced the mapping
time and storage cost of SCR methods. GLACE [59] im-
proves the scalability of SCR methods to large-scale scenes
by introducing co-visibility and feature diffusion, enhanc-
ing accuracy, and avoiding overfitting without relying on
3D models or depth maps.
Although SCR methods can
achieve high accuracy in small indoor scenes, their perfor-
mance is still limited in large-scale outdoor scenes. Unlike
SCR methods, our method achieves high accuracy in both
indoor and outdoor scenes.
2

<!-- page 3 -->
2.3. Radiance Field-Based Methods
Radiance field-based localization methods have recently be-
come an active research area, driven by the impressive scene
representation capabilities demonstrated by NeRF [34] and
Gaussian Splatting [19, 21].
Some methods leverage their excellent novel view syn-
thesis capabilities for pose refinement [26, 29, 64, 68] or
data augmentation [32, 36].
Among these, inverse ren-
dering has emerged as a key approach for camera local-
ization.
iNeRF [64] is the first method to directly opti-
mize camera pose via inverse rendering, inspiring the de-
velopment of subsequent methods. For example, PNeR-
FLoc [68] combines 2D-3D feature matching for initial
pose estimation with novel view synthesis for pose refine-
ment, and NeRFMatch [69] utilizes NeRF’s internal fea-
tures for precise 2D-3D matching and then optimizes cam-
era poses by minimizing photometric error. Instead of pho-
tometric alignment, CROSSFIRE [37] leverages dense lo-
cal features from volumetric rendering to improve robust-
ness. NeuraLoc [67] learns complementary features to es-
tablish accurate 2D-3D correspondences. In contrast to the
above methods, Lens [36] uses NeRF to synthesize addi-
tional views to expand the training dataset. Compared with
NeRF, Gaussian-based localization methods offer a more
efficient alternative by leveraging 3DGS’s competitive ren-
dering quality and speed. 6DGS [2] has attracted attention
by directly estimating 6DoF camera poses from a 3DGS
model without requiring a prior pose. Several concurrent
methods [3, 13, 28, 51, 66] to ours have also made signif-
icant advances. GSplatLoc [51] integrates dense keypoint
descriptors into 3DGS to get a coarse pose and then opti-
mizes it using a photometric warping loss. LoGS [13] first
estimates the initial pose via image retrieval and local fea-
ture matching with a PnP solver, followed by refinement
through analysis-by-synthesis. GS-CPR [28] leverages the
3DGS model to render high-quality images for matching to
enhance the localization accuracy of neural-based methods.
In contrast to previous radiance field-based methods, our
approach is a full localization pipeline that introduces a
novel sparse-to-dense paradigm. The initial pose is effi-
ciently estimated using the sampled scene landmarks and
the scene-specific detector. Then, the feature field provided
by Feature Gaussian is used to further refine the camera
pose. Our method produces accurate pose estimation and
is robust to illumination changes and weak-texture regions.
3. Method
Our method is a full localization pipeline that utilizes Fea-
ture Gaussian as the scene representation. In this section,
we introduce our method in detail.
In Sec. 3.1, we de-
scribe how to train Feature Gaussian, then in Sec. 3.2 and
Sec. 3.3, we detail our matching-oriented sampling strat-
Rasterization
Dense Feature 
Extractor
Figure 2. Feature Gaussian is trained by optimizing the radiance
field loss Lrgb and feature field loss Lf jointly.
egy and scene-specific detector, respectively.
Finally, in
Sec. 3.4, we present our novel sparse-to-dense camera re-
localization pipeline.
3.1. Feature Gaussian Training
Our scene representation consists of original Gaussian
primitives augmented with a feature field. The trainable at-
tributes of i-th Gaussian primitive include center xi, rota-
tion qi, scale si, opacity αi, color ci and feature fi, denoted
as Θi = {xi, qi, si, αi, ci, fi}. Our training process refer-
ences Feature 3DGS [70], jointly optimizing the radiance
field and the feature field, as illustrated in Fig. 2. Theoreti-
cally, our pipeline can be applied to any 3DGS variants with
explicit primitives.
We initialize the Gaussian primitives using SfM point
clouds. For clarity and conciseness, we denote  F_t( I
) \in \ mathbb {R}^{D\times H'\times W'} as the dense feature map extracted from the
training image  I  \in \mathbb {R}^{3\times H\times W} , where D is the dimension of
the local feature. The ground truth feature map Ft(I) can
be obtained using a general local feature extractor, such as
SuperPoint [14]. The Gaussian radiance field employs the
alpha-blending technique to rasterize color attribute  c into
the rendered RGB image  \hat {I} . The same rasterize method is
applied to feature attribute f to render the feature map ˆFs.
The overall training loss L is combined with radiance
field loss Lrgb and feature field loss Lf.
The radiance
field loss Lrgb consists of L1 loss L1 and D-SSIM loss
LD−SSIM between the training image I and the rendered
image ˆI, while the feature field loss Lf calculate L1 dis-
tance between the ground truth feature map Ft(I) and the
rendered feature map ˆFs:
  \m a th c al {L}_ {rg b } = (1-\lam bda )\mathcal {L}_1(I, \hat {I}) + \lambda \mathcal {L}_{D-SSIM}(I, \hat {I}), 
(1)
  \ mathcal { L}_{f} = \mathcal {L}_1(F_t(I), \hat {F_s}), 
(2)
  \math c al {L} = \alpha \mathcal {L}_{rgb} + \beta \mathcal {L}_{f}. 
(3)
3

<!-- page 4 -->
Anchor Sampling
Low
High
Score assignment
Anchor-Guided Selection
Spatial distance
k-NN
Selection with 
the highest score
Figure 3. Matching-Oriented Sampling. Each Gaussian is as-
signed a matching score, followed by anchor sampling. For each
anchor, the k nearest Gaussians are identified based on spatial dis-
tance, from which the highest-scoring Gaussian is selected.
In practice, we set the weight hyperparameters λ = 0.2,
α = 1.0, and β = 1.0 for training. We denote the complete
Feature Gaussian scene obtained from training as G.
3.2. Matching-Oriented Sampling
Exhaustive matching with all Gaussians in the scene is time-
consuming. Besides, numerous ambiguous Gaussians may
adversely affect feature matching. To address these chal-
lenges, we propose a novel matching-oriented sampling
strategy, which can select Gaussians that are well-suited for
matching from the millions of primitives in G.
Our ob-
jective is to ensure that the selected Gaussians are evenly
distributed throughout the scene and are recognizable from
various perspectives.
To quantify the quality of Gaussian, we develop a scor-
ing strategy. Specifically, as shown in Fig. 3, for each Gaus-
sian  g_i and training image  I , we project the center of gi
onto  I , yielding a 2D coordinate denoted as (ui, vi), then
we extract the corresponding 2D feature from the feature
map  F_t(I) .
The cosine similarity between the Gaussian
feature fi and the extracted 2D feature is used to compute
the matching score. Let  \mathcal {V}_{i} denote the set of images where
Gaussian gi is visible, and the final score s(gi) is obtained
by averaging the matching scores across all images in  \mathcal {V}_{i} :
  s(g _
i
) = 
\
frac
 {1} {\lvert \mathcal {V}_i\rvert } \sum _{I \in \mathcal {V}_i} \langle f_i, F_t(I)[u_i, v_i] \rangle . u
(4)
Here Ft(I)[ui, vi] denotes the extraction of 2D feature from
Ft(I) at position (ui, vi) using bilinear interpolation. A
higher score indicates that the corresponding Gaussian fea-
ture is more suitable for matching.
However, selecting Gaussians solely based on their
scores may lead to an uneven spatial distribution. Specif-
ically, regions with rich textures tend to exhibit a high den-
sity of Gaussians, whereas other areas may suffer from
Rasterization
Dense Feature 
Extractor
Heatmap
GT Heatmap
Projection
Scene-specific
Detector
Figure 4. Scene-Specific Detector Training. The centers of sam-
pled landmarks are projected onto 2D images to guide the training
of the scene-specific detector.
insufficient Gaussian coverage, adversely affecting perfor-
mance in those regions. To address this issue, we first em-
ploy standard downsampling techniques, such as random or
farthest point sampling, to ensure a spatially uniform distri-
bution. Specifically, we sample a fixed number of Gaussians
as anchors to guide the selection process. For each anchor,
we identify the k nearest neighbors in terms of spatial dis-
tance and then select the Gaussian with the highest score
within this neighborhood as the final result.
The number of Gaussians sampled through this method
is significantly reduced compared to the original, result-
ing in a set of Gaussians that are uniformly distributed and
highly recognizable from various perspectives. The exper-
imental results demonstrate that sampling just a few thou-
sand Gaussians is sufficient to achieve effective localiza-
tion. We denote this set of sampled Gaussians as scene
landmarks ˜G.
3.3. Scene-Specific Detector
Directly matching the dense feature map with sampled land-
marks remains infeasible, as the dense feature map contains
a large number of features, many of which are unsuitable
for matching, particularly in invalid regions such as the
sky. Moreover, off-the-shelf detectors such as SuperPoint
[14] typically detect pre-defined keypoints that are scene-
agnostic and therefore not well-suited for matching with the
sampled landmarks in Feature Gaussian scenes.
To address this issue, we propose a scene-specific feature
detector  \mathcal {D_{\theta }} that processes the feature map  F_t(I) and gener-
ates a heatmap  \ h at {K} \in \mathbb {R}^{1\times H' \times W'} , indicating the probability
of the 2D feature as a landmark:
  \ hat {K} = \mathcal {D_{\theta }}(F_t(I)). 
(5)
Specifically, our detector  \mathcal {D_{\theta }} is a shallow convolution neural
network (CNN), and  \theta represents the network parameters.
We train  \mathcal {D_{\theta }} in a self-supervised manner, as shown in
Fig. 4. The center of each Gaussian in  \Tilde {\mathcal {G}} is projected onto
the image plane, where the corresponding pixel position is
4

<!-- page 5 -->
Query Feat. Map
Query Image
Rasterization
Sparse Features
Final
Sparse Stage
Dense Feature 
Extractor
Scene-specific
Detector
Dense Stage
Crop
Patches
Coarse to Fine Match
MNN Match
…
MNN Match
Figure 5. Overview of the sparse-to-dense localization pipeline based on Feature Gaussian.
set to 1 to obtain the ground truth heatmap K. We use bi-
nary cross-entropy loss to optimize detector  \mathcal {D_{\theta }} :
  \m a th cal {L} _ {\ t ex t {de t }} = -K\log (\hat {K}) - (1-K)\log (1-\hat {K}). 
(6)
During inference, we apply non-maximum suppression
(NMS) to the heatmap ˆK to ensure that the detected key-
points are evenly distributed.
3.4. Sparse-to-Dense Localization
Thanks to the matching-oriented landmark sampling strat-
egy and the scene-specific feature detector, we can effi-
ciently estimate the initial pose. Furthermore, the feature
field provided by Feature Gaussian supports the establish-
ment of the sparse-to-dense localization pipeline.
The sparse-to-dense localization pipeline is illustrated in
Fig. 5. Firstly, sparse feature matching is conducted among
the sampled landmarks  \tilde {\mathcal {G}} and sparse local features detected
by  \mathcal {D_{\theta }} . Based on the 2D-3D correspondences obtained from
the sparse matching, the initial camera pose can be solved
through the PnP algorithm. Then, a dense feature map can
be rendered from the full Feature Gaussian  \mathcal {G} , followed by
coarse-to-fine dense feature matching to refine the pose.
Sparse Stage: Given a query image Iq, we first extract a
dense query feature map Ft(Iq) and utilize the  \mathcal {D_{\theta }} to detect
sparse local features. Then, we compute the cosine simi-
larity between the sparse local features with all landmark
features from ˜G. For each local feature, the landmark with
the highest similarity is selected as a match. The 2D coordi-
nates of the local features and the center coordinates of the
corresponding landmarks are treated as 2D-3D correspon-
dences, forming the sparse match set Msparse. Finally, the
PnP algorithm with RANSAC is applied to estimate the ini-
tial pose  \xi _{sparse} .
Dense Stage: With the initial pose  \xi _{sparse} , we render
a dense feature map ˆFs and a depth map ˆD from the Fea-
ture Gaussian scene  \mathcal {G} . Then, we perform dense matching
between ˆFs and Ft(Iq). Inspired by LoFTR [52], we first
conduct matching at the low resolution of D×Hf/8×Wf/8
and then refine it at the full resolution of D×Hf ×Wf. No-
tably, to fully leverage the dense information from the fea-
ture field, we directly render high-resolution feature maps
and subsequently downsample them to low resolution via
bilinear interpolation.
During matching, we first compute a correlation matrix
Sc using cosine similarity between coarse feature maps, fol-
lowed by a dual-softmax operation to obtain a probability
matrix Pc:
  \ mathcal {
P }_c = \ text {sof
t max}(\frac {1}{\tau } \mathcal {S}_c)_{row} \cdot \text {softmax}(\frac {1}{\tau } \mathcal {S}_c)_{col}, 
(7)
where τ is a temperature parameter. Then, we apply mutual
nearest neighbor (MNN) search on Pc to establish coarse
correspondences  \mathcal {M}_c .
Based on coarse matches  \mathcal {M}_c , we extract an 8 × 8 patch
for each coarse correspondence from the corresponding lo-
cations in the high-resolution feature map. Following the
same procedure, we compute the correlation matrix Sf and
probability matrix Pf on these patches and apply MNN
search to obtain the refined matches  \mathcal {M}_f .
Finally, we lift the 2D correspondences to 3D using the
rendered depth map ˆD and solve for the pose ξdense through
the PnP algorithm with RANSAC. The dense stage allows
for iterative pose refinement to achieve higher accuracy.
4. Experiments
In this section, we first present the localization results of
our pipeline on indoor and outdoor datasets in Sec. 4.1.
Next, we conduct a detailed ablation study in Sec. 4.2 to
evaluate the effectiveness of our matching-oriented sam-
pling strategy, scene-specific detector, and overall localiza-
tion pipeline. Finally, we provide a qualitative analysis of
the localization results in Sec. 4.3.
Datasets: We choose the 7-Scenes dataset [50] and the
Cambridge Landmarks dataset [20] to evaluate our pipeline,
5

<!-- page 6 -->
Method
Chess
Fire
Heads
Office
Pumpkin
RedKitchen
Stairs
Avg.↓[cm/°]
FM
AS (SIFT)
3/0.87
2/1.01
1/0.82
4/1.15
7/1.69
5/1.72
4/1.01
3.71/1.18
HLoc (SP+SG)
2.39/0.84
2.29/0.91
1.13/0.77
3.14/0.92
4.92/1.30
4.22/1.39
5.05/1.41
3.31/1.08
DVLAD+R2D2
2.56/0.88
2.21/0.86
0.98/0.75
3.48/1.00
4.79/1.28
4.21/1.44
4.60/1.27
3.26/1.07
SCR
DSAC*
0.50/0.17
0.78/0.29
0.50/0.34
1.19/0.35
1.19/0.29
0.72/0.21
2.65/0.78
1.07/0.35
ACE
0.55/0.18
0.83/0.33
0.53/0.33
1.05/0.29
1.06/0.22
0.77/0.21
2.89/0.81
1.10/0.34
NBE+SLD
0.6/0.18
0.7/0.26
0.6/0.35
1.3/0.33
1.5/0.33
0.8/0.19
2.6/0.72
1.16/0.34
NeuMap
2/0.81
3/1.11
2/1.17
3/0.98
4/1.11
4/1.33
4/1.12
3.14/0.95
NeRF/GS
DFNet+NeFeS50
2/0.57
2/0.74
2/1.28
2/0.56
2/0.55
2/0.57
5/1.28
2.43/0.79
CROSSFIRE
1/0.4
5/1.9
3/2.3
5/1.6
3/0.8
2/0.8
12/1.9
4.43/1.38
PNeRFLoc
2/0.80
2/0.88
1/0.83
3/1.05
6/1.51
5/1.54
32/5.73
7.29/1.76
NeRFMatch
0.95/0.30
1.11/0.41
1.34/0.92
3.09/0.87
2.21/0.60
1.03/0.28
9.26/1.74
2.71/0.73
STDLoc (Ours)
0.46/0.15
0.57/0.24
0.45/0.26
0.86/0.24
0.93/0.21
0.63/0.19
1.42/0.41
0.76/0.24
Table 1. Localization Results on 7-Scenes. We report the median translation errors and rotation errors for each scene. The best and
second-best results are bolded and underlined, respectively.
Method
Court
King’s
Hospital
Shop
St. Mary’s
Avg.↓[cm/°]
FM
AS (SIFT)
24/0.13
13/0.22
20/0.36
4/0.21
8/0.25
13.8/0.23
HLoc (SP+SG)
17.7/0.11
11.0/0.20
15.1/0.31
4.2/0.20
7.0/0.22
11.0/0.21
SCR
DSAC*
33.0/0.21
17.9/0.31
21.1/0.40
5.2/0.24
15.4/0.51
18.5/0.33
ACE (Poker)
27.9/0.14
18.6/0.33
25.7/0.51
5.1/0.26
9.5/0.33
17.4/0.31
GLACE
19.0/0.12
18.9/0.32
18.0/0.42
4.4/0.23
8.4/0.29
13.7/0.28
NeuMap
6/0.10
14/0.19
19/0.36
6/0.25
17/0.53
12.4/0.29
NeRF/GS
DFNet+NeFeS50
-
37/0.54
52/0.88
15/0.53
37/1.14
35.3/0.77
CROSSFIRE
-
47/0.7
43/0.7
20/1.2
39/1.4
37.3/1.00
PNeRFLoc
81/0.25
24/0.29
28/0.37
6/0.27
40/0.55
35.8/0.35
NeRFMatch
19.6/0.09
12.5/0.23
20.9/0.38
8.4/0.40
10.9/0.35
14.5/0.29
STDLoc (Ours)
15.7/0.06
15.0/0.17
11.9/0.21
3.0/0.13
4.7/0.14
10.1/0.14
Table 2. Localization Results on Cambridge Landmarks. Median translation errors and rotation errors are reported, with the best results
in bold and the second best underlined.
which is widely used in visual localization benchmarks.
The 7-Scenes dataset consists of 7 indoor scene sequences
captured by a handheld camera, covering rich-texture and
weak-texture situations. The Cambridge Landmarks dataset
consists of 5 outdoor scenes captured by mobile phones, in-
cluding challenges such as dynamic object occlusion, illu-
mination changes, and motion blur.
Training Details: Our training parameters follow the
original 3DGS [21], attached with a learning rate of 0.001
for the feature field according to Feature 3DGS [70]. We
adopt SuperPoint [14] as the default feature extractor. Each
scene is trained for 30,000 steps. When training on Cam-
bridge Landmarks, we use an off-the-shelf model [12] to
mask out dynamic objects and the sky to reduce distrac-
tions.
CLAHE histogram equalization algorithm [71] is
used to cope with illumination changes. We set the num-
ber of sampled anchors to 16,384 and the number of nearest
neighbors to 32. The scene-specific detector is a 4-layer
CNN with SiLU activation, trained for 30,000 steps with a
learning rate of 0.001 and an additional cosine decay. All
scenes are trained on one RTX4090 GPU, and each scene
consumes about 90 minutes for Feature Gaussian training
and 30 minutes for scene-specific detector training.
Localization Details: The radius of non-maximum sup-
pression (NMS) is set to 4 pixels, and we sample 2,048 key-
points for sparse matching. In the dense stage, the longer
side of the feature maps is set to 640 for the fine level and
80 for the coarse level. We use PoseLib [23] as our default
pose solver. We perform 4 iterations on 7-Scenes and 1 it-
eration on Cambridge Landmarks.
4.1. Relocalization Benchmark
We compare STDLoc with SOTA localization methods to
demonstrate our competitive performance. The methods in-
6

<!-- page 7 -->
Method
5cm 5°↑
2cm 2°↑
HLoc (SP+SG)
95.7%
84.5%
DSAC*
97.8%
80.7%
ACE
97.1%
83.3%
NeRFMatch
78.2%
-
STDLoc (Ours)
99.1%
90.9%
Table 3. Average Recall on 7-Scenes. We report the recall rate at
thresholds of 5cm 5° and 2cm 2°.
clude structure-based methods: AS [46], HLoc (SP+SG)
[14, 43, 44], DVLAD+R2D2 [41, 57]; regression-based
methods: DSAC* [5], ACE [8], NBE+SLD [15], NeuMap
[56], GLACE [59]; and radiance field-based methods:
NeFeS50 [11], CROSSFIRE [37], PNeRFLoc [68], NeRF-
Match [69]. In the HLoc and DVLAD+R2D2 experiments,
we used the default setting to retrieve the top 10 images.
Indoor Localization: We report the median translation
and rotation errors on the 7-Scenes dataset in Tab. 1. Our
proposed method, STDLoc, improves localization accuracy
and achieves the best performance on the 7-Scenes dataset.
STDLoc reduces both translation and rotation errors by ap-
proximately 30% compared to the best-performing SCR-
based method ACE [8] and DSAC* [5]. It is worth mention-
ing that STDLoc also achieves the highest recall rate among
existing approaches, with 99.1% at 5cm,5° and 90.9% at
2cm,2°, as shown in Tab. 3.
Outdoor Localization: We report the median trans-
lation and rotation errors on the Cambridge Landmarks
dataset in Tab. 2. Unlike indoor scenes, where the SCR-
based method is significantly better than the FM-based
method, the results in outdoor scenes show the opposite
trend. The existing SOTA radiance field-based methods per-
form worse than the other two methods. However, STDLoc
consistently outperforms all methods mentioned in the table
regarding rotation accuracy. It also demonstrates competi-
tive translation accuracy in scenes of Hospital, Shop, and
St. Mary’s. Due to insufficient 3D Gaussian reconstruction,
the accuracy is slightly lower than HLoc and NeuMap in the
other two scenes. In terms of the average metrics across all
scenes, STDLoc outperforms all current SOTA methods.
4.2. Ablation Study
Matching-Oriented Sampling Strategy: The sparse stage
emphasizes achieving high recall rates, so we report the 5m,
10° recall metrics on the Cambridge Landmarks dataset,
demonstrating the effectiveness of our matching-oriented
sampling strategy (M.O.). We evaluate our sampling strat-
egy by integrating it with farthest point sampling (FPS) and
random sampling (RS). In Tab. 4, the comparison between
①and ③shows that FPS performs worse than RS due to
its failure to consider the distribution density of Gaussian,
128
256
512
1024
2048
4096
8192 16384 32768
Number of Landmarks
0.5
0.6
0.7
0.8
0.9
1.0
Recall
Figure 6. Recall vs. Number of Landmarks. We report the 5m
10° recall on Court with varying numbers of landmarks.
a) SuperPoint Detector.
b) Scene-specific Detector.
Figure 7. Qualitative Comparison of Detectors. We present de-
tection results from the SuperPoint and our scene-specific detector,
processed with NMS and limited to 2048 points.
whereas areas with high Gaussian density typically contain
more information. However, the comparisons between ①
and ②as well as ③and ④demonstrate that regardless of
whether FPS or RS is used, our matching-oriented sampling
strategy significantly improves the quality of the sampled
landmarks.
Fig. 6 shows the recall trend versus the number of sam-
pled landmarks. Combined with the detector, our sampling
strategy maintains a high recall rate even at low sampling
rates. With only 1,024 sampled landmarks, the recall rage
still exceeds 90%. In the experiment, we set the default
sampling number to 16,384 to achieve saturated recall.
Scene-Specific Detector: We qualitatively compare our
scene-specific detector (S.S.D.) with the SuperPoint detec-
tor in Tab. 4. The comparisons between ③and ⑤as well
as ④and ⑥demonstrate that our S.S.D. effectively extracts
more suitable features from the feature map for matching
with sampled landmarks. We visualize the detection results
in Fig. 7. Our detector captures more points on buildings
while fewer points in the sky, aligning more closely with the
actual Gaussian distribution and yielding a higher recall.
7

<!-- page 8 -->
No.
Sampling
Detector
5m 10°↑
2m 5°↑
①
FPS
SuperPoint
57.9%
47.1%
②
FPS+M.O.
SuperPoint
90.5%
86.5%
③
RS
SuperPoint
95.0%
89.7%
④
RS+M.O.
SuperPoint
98.2%
96.5%
⑤
RS
S.S.D.
99.0%
97.6%
⑥
RS+M.O.
S.S.D.
99.6%
99.1%
Table 4. Ablation Study on Sampling Strategy and Detector.
We report 5m 10° and 2m 5° recall on Cambridge Landmarks.
Stage
Err.↓[cm/°]
5m 10°↑
Image Retrieval
586/7.9
48.2%
Sparse
13.8/0.21
99.6%
Sparse + Dense (RGB)
15.3/0.3
99.4%
Sparse + Dense (Feat.)
10.1/0.14
99.4%
Table 5. Ablation Study on Localization Pipeline. Average me-
dian error and recall are reported on Cambridge Landmarks.
Gaussian
Feature
Err.↓[cm/°]
3DGS
SuperPoint
10.1/0.14
2DGS
SuperPoint
10.4/0.14
3DGS
R2D2
10.1/0.15
2DGS
R2D2
10.8/0.18
Table 6. Comparison of Different Gaussians and Features. The
average median error on Cambridge Landmarks is reported.
Pipeline Effectiveness: In Tab. 5, we report the average
median error and recall for different stages of our pipeline
and image retrieval method [1] on Cambridge Landmarks.
The term “Sparse” refers to the sparse stage, while “Dense
(Feat.)” indicates rendering the feature map directly, and
“Dense (RGB)” means rendering RGB images first, then
extracting the feature map from the rendered image. Our
sparse stage achieves significantly higher accuracy and re-
call than the image retrieval method. The Dense (Feat.)
stage further enhances localization accuracy over the sparse
stage. However, the Dense (RGB) stage performs worse,
which can be attributed to the rendered RGB images be-
ing of low quality due to noise factors such as illumination
changes in the training phase. The dense stage exhibits a
slight loss of recall, which can be attributed to the occlusion
of floaters that prevent rendering a complete feature map
and RGB image.
Flexiblility: Our pipeline is capable of adapting to var-
ious explicit Gaussian representations and features.
In
Tab. 6, we conduct experiments on the Cambridge Land-
marks dataset with 2DGS [19] and R2D2 [41] feature. The
results demonstrate that our pipeline performs effectively
Figure 8. Visualization Results. We show the query feature map
(left), rendered feature map (middle), and stitched image (right) of
the query and rendered images.
across various combinations of Gaussians and features.
Running Time: We evaluate the running time of various
modules of STDLoc on the Cambridge Landmarks dataset
using a single NVIDIA RTX 4090 GPU. With one iteration
for the dense stage, STDLoc achieves an inference speed
of approximately 7 FPS. The detailed time-consuming for
each module can be found in the appendix Tab. A.
4.3. Qualitative Analysis
We present some visualization results in Fig. 8 to provide
a clearer understanding of our localization process. Our
method effectively renders high-quality feature maps for
dense matching and high-quality RGB images based on lo-
calization results. Additional visualizations, including fail-
ure case studies, can be found in the appendix Sec. C.
5. Conclusion
This paper proposes a novel sparse-to-dense camera relo-
calization method that leverages Feature Gaussian as the
scene representation. Based on this scene representation,
we propose a new matching-oriented sampling strategy and
a scene-specific detector to facilitate efficient and robust
sparse matching to obtain the initial pose. Then, the lo-
calization accuracy is improved by aligning the dense fea-
ture map to the trained feature field through dense matching.
Our proposed method can produce accurate pose estimation
and is robust to illumination-change and weak-texture sit-
uations. Experimental results demonstrate that our method
outperforms the current SOTA methods.
Limitations and Future Work: Due to the training
limitations of Feature Gaussian, our method cannot be ex-
tended to very large scenes.
However, this may be ad-
dressed through techniques such as Level of Detail (LoD)
on 3DGS or chunked training. Additionally, the floating ar-
tifacts in 3DGS can adversely affect localization. Reducing
these artifacts is an important research direction for 3DGS.
We believe that advancements in 3DGS can be easily inte-
grated into our framework, further improving localization
performance.
8

<!-- page 9 -->
References
[1] Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas Pa-
jdla, and Josef Sivic.
NetVLAD: CNN architecture for
weakly supervised place recognition. In CVPR, pages 5297–
5307, 2016. 2, 8
[2] Matteo Bortolon, Theodore Tsesmelis, Stuart James, Fabio
Poiesi, and Alessio Del Bue.
6DGS: 6D pose estimation
from a single image and a 3D gaussian splatting model. In
ECCV, 2024. 2, 3
[3] Kazii Botashev, Vladislav Pyatov, Gonzalo Ferrer, and Sta-
matios Lefkimmiatis. GSLoc: Visual localization with 3D
gaussian splatting. arXiv preprint arXiv:2410.06165, 2024.
2, 3
[4] Eric Brachmann and Carsten Rother. Learning less is more -
6D camera localization via 3D surface regression. In CVPR,
pages 4654–4662, 2018. 2
[5] Eric Brachmann and Carsten Rother.
Visual camera re-
localization from RGB and RGB-D images using DSAC.
IEEE TPAMI, 2021. 7
[6] Eric Brachmann and Carsten Rother.
Visual camera re-
localization from RGB and RGB-D images using DSAC.
IEEE TPAMI, 44(9):5847–5865, 2021. 2
[7] Eric Brachmann, Martin Humenberger, Carsten Rother, and
Torsten Sattler. On the limits of pseudo ground truth in visual
camera re-localisation. In ICCV, pages 6218–6228, 2021. 2
[8] Eric Brachmann, Tommaso Cavallari, and Victor Adrian
Prisacariu. Accelerated Coordinate Encoding: Learning to
relocalize in minutes using RGB and poses. In CVPR, 2023.
1, 2, 7
[9] Shuai Chen, Zirui Wang, and Victor Prisacariu.
Direct-
posenet: Absolute pose regression with photometric consis-
tency. In 3DV, pages 1175–1185. IEEE, 2021. 1
[10] Shuai Chen,
Xinghui Li,
Zirui Wang,
and Victor A
Prisacariu. DFNet: Enhance absolute pose regression with
direct feature matching.
In ECCV, pages 1–17. Springer,
2022. 2
[11] Shuai Chen, Yash Bhalgat, Xinghui Li, Jia-Wang Bian, Kejie
Li, Zirui Wang, and Victor Adrian Prisacariu. Neural refine-
ment for absolute pose regression with feature synthesis. In
CVPR, pages 20987–20996, 2024. 7
[12] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexan-
der Kirillov, and Rohit Girdhar.
Masked-attention mask
transformer for universal image segmentation.
In CVPR,
pages 1290–1299, 2022. 6
[13] Yuzhou Cheng,
Jianhao Jiao,
Yue Wang,
and Dim-
itrios Kanoulas.
LoGS: Visual localization via gaus-
sian splatting with fewer training images.
arXiv preprint
arXiv:2410.11505, 2024. 3
[14] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabi-
novich. Superpoint: Self-supervised interest point detection
and description. In CVPRW, pages 224–236, 2018. 1, 2, 3,
4, 6, 7
[15] Tien Do, Ondrej Miksik, Joseph DeGol, Hyun Soo Park, and
Sudipta N Sinha. Learning to detect scene landmarks for
camera localization. In CVPR, pages 11132–11142, 2022. 2,
7
[16] Johan
Edstedt,
Qiyu
Sun,
Georg
B¨okman,
M˚arten
Wadenb¨ack, and Michael Felsberg.
RoMa: Robust dense
feature matching. In CVPR, pages 19790–19800, 2024. 1
[17] Martin A Fischler and Robert C Bolles.
Random sample
consensus: a paradigm for model fitting with applications to
image analysis and automated cartography. Communications
of the ACM, 24(6):381–395, 1981. 2
[18] Khang Truong Giang, Soohwan Song, and Sungho Jo.
Learning to produce semi-dense correspondences for visual
localization. In CVPR, pages 19468–19478, 2024. 1, 2
[19] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2D gaussian splatting for geometrically ac-
curate radiance fields.
In ACM SIGGRAPH, pages 1–11,
2024. 3, 8
[20] Alex Kendall, Matthew Grimes, and Roberto Cipolla.
PoseNet: A convolutional network for real-time 6-DOF cam-
era relocalization. In ICCV, pages 2938–2946, 2015. 1, 2,
5
[21] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler,
and George Drettakis. 3D gaussian splatting for real-time
radiance field rendering. ACM TOG, 42(4):1–14, 2023. 2, 3,
6, 12
[22] Minjung Kim, Junseo Koo, and Gunhee Kim. EP2P-Loc:
End-to-end 3D point to 2D pixel localization for large-scale
visual localization. In ICCV, pages 21527–21537, 2023. 2
[23] Viktor Larsson and contributors. PoseLib - Minimal Solvers
for Camera Pose Estimation, 2020. 6
[24] Vincent Lepetit, Francesc Moreno-Noguer, and Pascal Fua.
EPnP: An accurate O(n) solution to the PnP problem. IJCV,
81(2):155–166, 2009. 1, 2
[25] Yunpeng Li, Noah Snavely, and Daniel P Huttenlocher. Lo-
cation recognition using prioritized feature matching.
In
ECCV, pages 791–804. Springer, 2010. 1
[26] Yunzhi Lin, Thomas M¨uller, Jonathan Tremblay, Bowen
Wen, Stephen Tyree, Alex Evans, Patricio A Vela, and Stan
Birchfield. Parallel inversion of neural radiance fields for
robust pose estimation. In ICRA, pages 9377–9384. IEEE,
2023. 3
[27] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Polle-
feys. LightGlue: Local feature matching at light speed. In
ICCV, pages 17627–17638, 2023. 2
[28] Changkun Liu, Shuai Chen, Yash Sanjay Bhalgat, Siyan HU,
Ming Cheng, Zirui Wang, Victor Adrian Prisacariu, and Tris-
tan Braud. GS-CPR: Efficient camera pose refinement via 3D
gaussian splatting. In ICLR, 2025. 3
[29] Jianlin Liu, Qiang Nie, Yong Liu, and Chengjie Wang.
NeRF-Loc: Visual localization with conditional neural ra-
diance field. In ICRA, pages 9385–9392. IEEE, 2023. 3
[30] Liu Liu, Hongdong Li, and Yuchao Dai.
Efficient global
2D-3D matching for camera localization in a large-scale 3D
map. In ICCV, pages 2372–2381, 2017. 1
[31] David G Lowe. Object recognition from local scale-invariant
features. In ICCV, pages 1150–1157. IEEE, 1999. 1, 2
[32] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi,
Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duck-
worth.
NeRF in the wild: Neural radiance fields for un-
constrained photo collections. In CVPR, pages 7210–7219,
2021. 3
9

<!-- page 10 -->
[33] N. Max. Optical models for direct volume rendering. IEEE
TVCG, 1(2):99–108, 1995. 12
[34] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF:
Representing scenes as neural radiance fields for view syn-
thesis. In ECCV, 2020. 2, 3
[35] Arthur Moreau, Nathan Piasco, Dzmitry Tsishkou, Bog-
dan Stanciulescu, and Arnaud de La Fortelle.
Coordinet:
uncertainty-aware pose regressor for reliable vehicle local-
ization. In WACV, pages 2229–2238, 2022. 2
[36] Arthur Moreau, Nathan Piasco, Dzmitry Tsishkou, Bogdan
Stanciulescu, and Arnaud de La Fortelle. LENS: Localiza-
tion enhanced by NeRF synthesis. In CoRL, pages 1347–
1356. PMLR, 2022. 2, 3
[37] Arthur Moreau, Nathan Piasco, Moussab Bennehar, Dzmitry
Tsishkou, Bogdan Stanciulescu, and Arnaud de La Fortelle.
CROSSFIRE: Camera relocalization on self-supervised fea-
tures from an implicit representation. In ICCV, pages 252–
262, 2023. 3, 7
[38] Vojtech Panek, Zuzana Kukelova, and Torsten Sattler.
Meshloc: Mesh-based visual localization. In ECCV, pages
589–609. Springer, 2022. 1, 2
[39] Vojtech Panek, Zuzana Kukelova, and Torsten Sattler. Visual
localization using imperfect 3D models from the internet. In
CVPR, pages 13175–13186, 2023. 1
[40] Guilherme Potje, Felipe Cadar, Andr´e Araujo, Renato Mar-
tins, and Erickson R Nascimento. XFeat: Accelerated fea-
tures for lightweight image matching. In CVPR, pages 2682–
2691, 2024. 1, 2
[41] Jerome Revaud, Philippe Weinzaepfel, C´esar Roberto de
Souza, and Martin Humenberger. R2D2: repeatable and re-
liable detector and descriptor. In NeurIPS, 2019. 1, 2, 7,
8
[42] Jerome Revaud, Yohann Cabon, Romain Br´egier, JongMin
Lee, and Philippe Weinzaepfel. SACReg: Scene-agnostic
coordinate regression for visual localization. In CVPR, pages
688–698, 2024. 2
[43] Paul-Edouard Sarlin, Cesar Cadena, Roland Siegwart, and
Marcin Dymczyk. From coarse to fine: Robust hierarchical
localization at large scale. In CVPR, pages 12716–12725,
2019. 1, 2, 7
[44] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz,
and Andrew Rabinovich.
SuperGlue:
Learning feature
matching with graph neural networks. In CVPR, 2020. 1,
2, 7
[45] Paul-Edouard Sarlin, Ajaykumar Unagar, Mans Larsson,
Hugo Germain, Carl Toft, Viktor Larsson, Marc Pollefeys,
Vincent Lepetit, Lars Hammarstrand, Fredrik Kahl, et al.
Back to the feature: Learning robust camera localization
from pixels to pose. In CVPR, pages 3247–3257, 2021. 2
[46] Torsten Sattler, Bastian Leibe, and Leif Kobbelt. Improving
image-based localization by active correspondence search.
In ECCV, pages 752–765. Springer, 2012. 1, 2, 7
[47] Torsten Sattler, Will Maddern, Carl Toft, Akihiko Torii,
Lars Hammarstrand, Erik Stenborg, Daniel Safari, Masatoshi
Okutomi, Marc Pollefeys, Josef Sivic, et al. Benchmarking
6DOF outdoor visual localization in changing conditions. In
CVPR, pages 8601–8610, 2018. 2
[48] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited. In CVPR, pages 4104–4113, 2016.
1
[49] Yoli Shavit and Yosi Keller.
Camera pose auto-encoders
for improving pose regression. In ECCV, pages 140–157.
Springer, 2022. 2
[50] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram
Izadi, Antonio Criminisi, and Andrew Fitzgibbon. Scene co-
ordinate regression forests for camera relocalization in RGB-
D images. In CVPR, pages 2930–2937, 2013. 1, 5
[51] Gennady Sidorov, Malik Mohrat, Ksenia Lebedeva, Ruslan
Rakhimov, and Sergey Kolyubin.
GSplatLoc: Grounding
keypoint descriptors into 3D gaussian splatting for improved
visual localization. arXiv preprint arXiv:2409.16502, 2024.
3
[52] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and
Xiaowei Zhou. LoFTR: Detector-free local feature matching
with transformers. In CVPR, pages 8922–8931, 2021. 1, 2,
5
[53] Yuan Sun, Xuan Wang, Yunfan Zhang, Jie Zhang, Caigui
Jiang, Yu Guo, and Fei Wang. iComMa: Inverting 3D gaus-
sians splatting for camera pose estimation via comparing and
matching. arXiv preprint arXiv:2312.09031, 2023. 2
[54] Linus Sv¨arm, Olof Enqvist, Fredrik Kahl, and Magnus Os-
karsson. City-scale localization for cameras with known ver-
tical direction. IEEE TPAMI, 39(7):1455–1461, 2016. 1
[55] Hajime Taira, Masatoshi Okutomi, Torsten Sattler, Mircea
Cimpoi, Marc Pollefeys, Josef Sivic, Tomas Pajdla, and Ak-
ihiko Torii.
InLoc: Indoor visual localization with dense
matching and view synthesis. In CVPR, pages 7199–7209,
2018. 1
[56] Shitao Tang, Sicong Tang, Andrea Tagliasacchi, Ping Tan,
and Yasutaka Furukawa. Neumap: Neural coordinate map-
ping by auto-transdecoder for camera localization. In CVPR,
pages 929–939, 2023. 2, 7
[57] Akihiko Torii, Relja Arandjelovic, Josef Sivic, Masatoshi
Okutomi, and Tomas Pajdla. 24/7 place recognition by view
synthesis. In CVPR, pages 1808–1817, 2015. 7
[58] Julien Valentin, Matthias Nießner, Jamie Shotton, Andrew
Fitzgibbon, Shahram Izadi, and Philip HS Torr. Exploiting
uncertainty in regression forests for accurate camera relocal-
ization. In CVPR, pages 4400–4408, 2015. 2
[59] Fangjinhua
Wang,
Xudong
Jiang,
Silvano
Galliani,
Christoph Vogel, and Marc Pollefeys.
GLACE: Global
local accelerated coordinate encoding.
In CVPR, pages
21562–21571, 2024. 2, 7
[60] Qiang Wang. MAD-DR: Map compression for visual local-
ization with matchness aware descriptor dimension reduc-
tion. In ECCV, pages 261–278. Springer, 2025. 2
[61] Tao Xie, Kun Dai, Siyi Lu, Ke Wang, Zhiqiang Jiang, Jing-
han Gao, Dedong Liu, Jie Xu, Lijun Zhao, and Ruifeng Li.
OFVL-MS: Once for visual localization across multiple in-
door scenes. In ICCV, pages 5516–5526, 2023. 2
[62] Shen Yan, Yu Liu, Long Wang, Zehong Shen, Zhen Peng,
Haomin Liu, Maojun Zhang, Guofeng Zhang, and Xiaowei
Zhou. Long-term visual localization with mobile sensors. In
CVPR, pages 17245–17255, 2023. 1
10

<!-- page 11 -->
[63] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen,
Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey
Hu, Matthew Tancik, and Angjoo Kanazawa.
gsplat: An
open-source library for Gaussian splatting. arXiv preprint
arXiv:2409.06765, 2024. 12
[64] Lin Yen-Chen, Pete Florence, Jonathan T Barron, Alberto
Rodriguez, Phillip Isola, and Tsung-Yi Lin. iNeRF: Inverting
neural radiance fields for pose estimation. In IROS, pages
1323–1330. IEEE, 2021. 2, 3
[65] Hailin Yu, Youji Feng, Weicai Ye, Mingxuan Jiang, Hujun
Bao, and Guofeng Zhang. Improving feature-based visual
localization by geometry-aided matching.
arXiv preprint
arXiv:2211.08712, 2022. 1
[66] Hongjia Zhai, Xiyu Zhang, Boming Zhao, Hai Li, Yijia He,
Zhaopeng Cui, Hujun Bao, and Guofeng Zhang.
Splat-
loc: 3D gaussian splatting-based visual localization for aug-
mented reality. TVCG, pages 1–11, 2025. 3
[67] Hongjia Zhai, boming Zhao, Hai Li, Xiaokun Pan, Yijia He,
Zhaopeng Cui, Hujun Bao, and Guofeng Zhang. NeuraLoc:
Visual localization in neural implicit map with dual comple-
mentary features. In ICRA, 2025. 3
[68] Boming Zhao, Luwei Yang, Mao Mao, Hujun Bao, and
Zhaopeng Cui. PNeRFLoc: Visual localization with point-
based neural radiance fields.
In AAAI, pages 7450–7459,
2024. 3, 7
[69] Qunjie Zhou, Maxim Maximov, Or Litany, and Laura Leal-
Taix´e. The NeRFect match: Exploring NeRF features for
visual localization. ECCV, 2024. 2, 3, 7
[70] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Ze-
hao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang
Wang, and Achuta Kadambi. Feature 3DGS: Supercharging
3D gaussian splatting to enable distilled feature fields. In
CVPR, pages 21676–21685, 2024. 2, 3, 6, 12
[71] Karel Zuiderveld. Contrast limited adaptive histogram equal-
ization.
In Graphics Gems IV, page 474–485. Academic
Press Professional, Inc., USA, 1994. 6
11

<!-- page 12 -->
From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from
Feature Gaussian Splatting
Supplementary Material
A. Feature Gaussian Training
Our training strategy references Feature 3DGS [70].
To
adapt the feature field for the localization task and improve
robustness, we make the following modifications:
1. Thanks to the development of the CUDA accelerated ras-
terization tool gsplat [63], we can efficiently render fea-
ture maps while preserving the original feature dimen-
sions, eliminating the need for the speed-up module pro-
posed in Feature 3DGS for upsampling feature channels
after rasterization. Specifically, the feature  f stored in
the Gaussian primitive  g has the same dimension  D as
the feature map  F_t(I) extracted using the general local
feature extractor. This also enables direct matching be-
tween the 2D query features and the 3D features of the
Gaussian primitives.
2. The rendering process for the radiance field of Feature
Gaussian is based on the alpha blending rasterization
method [33]. Let  \mathcal {N} denote the set of Gaussians asso-
ciated with a pixel, sorted in front-to-back order. The
pixel color  C is computed by blending the color  c of
Gaussians as follows:
  
C
 = 
\sum _{i \in \mathcal {N}} c_i \alpha _i T_i, 
(8)
where  T_i is the transmittance factor accounting for the
accumulated opacity α of all preceding Gaussians, de-
fined as:
  T
_i 
=
 \p
ro d  _{j=1}^{i-1} (1 - \alpha _j). 
(9)
This alpha blending approach is also applied to render
the feature field. However, due to the vector triangle
inequality, directly accumulating features is unsuitable.
To address this, we introduce L2 normalization into the
alpha blending formula. Specifically, we normalize the
Gaussian feature  f before rasterization to mitigate the
influence of feature magnitude, and we further normal-
ize the accumulated features after rasterization. The final
rendered feature  F_s is therefore expressed as:
  F _s =
 \
t
ext
 {norm}\left
 
(\sum _{i=1}^{n} \text {norm}(f_i) \alpha _i T_i\right ), 
(10)
where  \text {norm}(\cdot ) denotes the L2 normalization operation.
This two-step normalization process ensures stability
and robustness in the feature field training and render-
ing.
Module
Time (ms)
Feature Extraction
3.7
S.S.D.
6.4
Sparse Matching
17.4
Pose Estimation (Sparse)
15.8
Rasterization
23
Dense Matching
13.2
Pose Estimation (Dense)
72.8
Total
152.3
Table A. Detailed Time Consumption Analysis.
B. Matching-Oriented Sampling Algorithm
The algorithm of the matching-oriented sampling strategy
is illustrated in Algorithm 1.
C. Qualitative Analysis
Challenging Cases. In Fig. B Fig. C, we present the lo-
calization results of STDLoc in challenging scenarios in-
volving weak texture and varying illumination conditions.
For illumination changes, we demonstrate sample cases
from both the Cambridge Landmarks dataset and real scene,
where STDLoc achieves precise localization results. In the
weak texture scenario, we provide a comparative analysis
with HLoc in the Stairs scenario from the 7-Scenes dataset.
STDLoc extracts denser matches, enabling more accurate
pose estimation.
Failure Cases. As shown in Tab. 5, the recall of (5m,
10°) in the dense stage is slightly lower than that in the
sparse stage. The localization results for these failure cases
are illustrated in Fig. A. The first column is the query image,
the second column is the sparse matching result between
the query image and landmarks, and the third column is the
dense feature map of the query image and the feature map
rendered based on the sparse stage localization pose.
The sparse stage successfully provides an accurate pose
estimation, but the feature map rendered in the dense stage
is indistinct, leading to the failure of dense matching. This
lack of distinguishability in the dense feature map is caused
by floaters in the scene, which is a common issue in 3DGS
[21] scenes. Therefore, reducing floaters in the scene can
be effective in minimizing these failure cases.
In addi-
tion, this situation reflects the robustness of our sparse
stage, which can effectively eliminate the influence of these
12

<!-- page 13 -->
Floater Occlusion
4.8/1.7
4.6/84
1.8/0.27
830/110
1.7/0.37
187/37
10.9/0.04
7510/91
Floater Occlusion
Floater Occlusion
Floater Occlusion
Figure A. Failure Cases Visualization. Visualization results of some examples where localization is successful in the sparse stage but
failed in the dense stage. The translation error (cm) and rotation error (°) are indicated below the corresponding stage.
Render
Render
Query
Query
HLoc
STDLoc
Render
Query
Render
Query
Render
Query
Render
Query
l
Landmarks
l
S.S.D. Points
l
Landmarks
l
SP Points
Figure B. Comparison with HLoc in Weak Texture Scenario.
Render
Render
Query
Query
HLoc
STDLoc
Render
Query
Render
Query
Render
Query
Render
Query
l
Landmarks
l
S.S.D. Points
Figure C. Localization Results in Illumination Changes Scenar-
ios.
floaters through the matching-oriented sampling strategy.
More Visualization Results. Fig. D presents the local-
ization visualization results across all scenes for both the
Cambridge Landmarks and 7-Scenes datasets. From left to
right, each column shows the query image, its correspond-
ing dense feature map, the sparse matching result between
the query image and landmarks, the rendered feature map
from the final dense stage, and the stitched result of the
query image with the rendered image using the final pose.
The visualization of the sparse matching results is achieved
by rendering Gaussian landmarks based on the pose esti-
mated in the sparse stage, followed by drawing the matches.
The third column of the figure demonstrates that our
sparse stage achieves robust 2D-3D matching results. This
is attributed to our proposed matching-oriented sampling
strategy and scene-specific detector.
Furthermore, the
second-to-last column demonstrates the capability to learn
the feature field using Feature Gaussian. As shown in the
last column, the rendered image aligns precisely with the
query image, highlighting the high accuracy of our local-
ization method. By leveraging the learned feature field, our
approach exhibits remarkable robustness against illumina-
tion changes and weak texture.
13

<!-- page 14 -->
Algorithm 1 Matching-Oriented Sampling Algorithm
Require: Gaussians G, training images {I}, feature maps {Ft(I)}, anchor number n, nearest neighbors k
Ensure: Sampled landmarks ˜G
1: for each Gaussian g ∈G do
\triangleright Assign scores for each Gaussian
2:
f ←norm(g.feature)
\triangleright Normalize Gaussian feature
3:
V ←{The set of images where g is visible}
4:
s ←0
5:
for each Image I ∈V do
6:
(u, v) ←Project(g.center, I)
\triangleright The pixel coordinates of the projected Gaussian center
7:
fimg ←norm(GridSample(Ft(I), (u, v)))
\triangleright Extract 2D feature using bilinear interpolation
8:
s ←s + ⟨f, fimg⟩
9:
end for
10:
g.score ←s/|V|
\triangleright Average score across images
11: end for
12: A ←RandomSampling(G, n)
\triangleright Randomly sample anchors
13: ˜G ←∅
14: for each anchor a ∈A do
\triangleright Anchor-guided selection
15:
Na ←FindkNearestNeighbors(a, G, k)
16:
g∗←arg max
g∈Na
g.score
\triangleright Select Gaussian with the highest score among neighbors
17:
˜G ←˜G ∪{g∗}
18: end for
14

<!-- page 15 -->
Figure D. More Visualizations. We show all scenes on both the Cambridge Landmarks and 7-Scenes datasets.
15
