<!-- page 1 -->
SEGSPLAT: FEED-FORWARD GAUSSIAN SPLATTING
AND OPEN-SET SEMANTIC SEGMENTATION
Peter Siegel
ETH Z¨urich
psiegel@ethz.ch
Federico Tombari
Google
dbarath@inf.ethz.ch
Marc Pollefeys
ETH Z¨urich, Microsoft
dbarath@inf.ethz.ch
Daniel Barath
ETH Z¨urich
dbarath@inf.ethz.ch
Figure 1: Visualization of learned 3D language features of the previous state-of-the-art method,
LangSplat Qin et al. (2024), and our SegSplat. While LangSplat requires per-scene training and
generates imprecise features, our SegSplat captures smooth regions more consistently and needs no
training. While being effective, our SegSplat is also 59× faster than LangSplat.
ABSTRACT
Efficient 3D scene reconstruction with open-set semantic understanding is vital for
robotics and augmented reality. However, existing methods either require costly per-
scene optimization to incorporate semantics into expressive 3D representations like
3D Gaussian Splats (3DGS), or, if feed-forward, focus predominantly on geometry
and appearance. This paper introduces SegSplat, the first framework to predict 3D
Gaussian Splats with associated open-set semantic features in a purely feed-forward
manner. Building upon efficient sparse-view 3DGS techniques, SegSplat uniquely
integrates semantic knowledge derived from 2D open-set segmentation models (e.g.,
SAM and CLIP) without any training. We achieve this by constructing a compact
semantic memory bank from features in the input images and assigning each
Gaussian a discrete index to this bank. This approach enables rich semantic feature
association with minimal additional storage and computational overhead, thus
preserving the rapid inference capabilities of feed-forward 3DGS. We demonstrate
that SegSplat delivers geometric fidelity comparable to state-of-the-art feed-forward
reconstruction methods while simultaneously enabling versatile open-set semantic
querying, all without necessitating scene-specific optimization. Our work bridges a
critical gap, paving the way for practical, on-the-fly generation of semantically rich
3D environments.
1
arXiv:2511.18386v1  [cs.CV]  23 Nov 2025

<!-- page 2 -->
1
INTRODUCTION
The ability to reconstruct 3D environments with rich, open-set semantic understanding is increasingly
critical for intelligent systems, enabling applications ranging from robotic navigation to immersive
augmented reality. These scenarios demand not only accurate geometry but also the capacity to
identify, query, and manipulate arbitrary high-level concepts – such as ”a specific type of chair”,
”any vehicle”, or ”all metallic objects” – directly within the 3D representation Kerr et al. (2023).
While progress has been made in both 3D reconstruction Mildenhall et al. (2020); Yu et al. (2021);
Kerbl et al. (2023) and 2D open-set semantic segmentation Xu et al. (2023); Li et al. (2022); Liang
et al. (2023), integrating these capabilities into a unified and efficient 3D representation that supports
open-vocabulary queries without per-scene fine-tuning remains a fundamental challenge.
Recent advances in neural rendering, particularly Neural Radiance Fields (NeRFs) Mildenhall
et al. (2020), have demonstrated compelling view synthesis through implicit scene representations.
However, their reliance on per-scene optimization limits scalability and real-time deployment. In
contrast, 3D Gaussian Splatting (3DGS) Kerbl et al. (2023) represents scenes as collections of
3D Gaussians, enabling high-quality, real-time rendering. Despite these rendering advantages,
training times remain substantial Kerbl et al. (2023), motivating a shift towards faster feed-forward
approaches Xu et al. (2024); Chen et al. (2024); Ye et al. (2024a).
To bridge the semantic gap, recent works extend 3DGS with language-aligned features Qin et al.
(2024); Shi et al. (2023); Ye et al. (2024b), typically associating each Gaussian with a feature vector
(e.g., from CLIP Radford et al. (2021)). These enable open-vocabulary querying, but most methods
still depend on per-scene optimization (first constructing the 3DGS, then fitting semantic features),
making them ill-suited for dynamic or large-scale environments.
This paper introduces SegSplat, a novel framework that predicts 3D Gaussian Splats enriched with
open-set semantic features in a single feed-forward pass. Building on recent progress in sparse-view
3DGS reconstruction, we leverage the DepthSplat architecture Xu et al. (2024) for fast and robust
geometric prediction. Our key contribution is the seamless integration of semantic information,
extracted via 2D open-set segmentation models (e.g., SAM Kirillov et al. (2023) with CLIP), directly
into the predicted Gaussians.
To achieve this, we construct a compact semantic memory bank from features observed in the input
views and assign a discrete index (analogous to a one-hot code) to each 3D Gaussian primitive. This
allows each Gaussian to reference a powerful semantic descriptor while introducing minimal storage
and computational overhead, preserving fast inference.
In summary, SegSplat provides a unified, efficient pipeline that jointly reconstructs 3D scene ge-
ometry and embeds open-vocabulary semantic features – eliminating the need for per-scene opti-
mization. We validate our method on challenging datasets including 3D-OVS Liu et al. (2023a) and
RealEstate10k Zhou et al. (2018), demonstrating that SegSplat achieves accurate feed-forward 3DGS
quality while enabling robust open-set semantic querying.
2
RELATED WORK
Our work integrates progress in three key areas: 3D scene representations, their accelerated sparse
feed-forward generation, and the incorporation of open-vocabulary semantic understanding. This
review situates SegSplat’s contribution: a method for feed-forward generation of 3D Gaussian
Splatting (3DGS) models with associated open-set semantic features.
The pursuit of high-fidelity 3D scene models have evolved from traditional representations like
meshes, which often involve complex creation pipelines, towards learnable techniques. Neural
Radiance Fields (NeRF) Mildenhall et al. (2020) were a significant advance, using neural networks
to synthesize photorealistic novel views. However, NeRFs typically require lengthy per-scene
optimization, limiting their practical use. 3D Gaussian Splatting (3DGS) Kerbl et al. (2023) offered a
compelling alternative, explicitly representing scenes with 3D Gaussian primitives. This approach
achieves state-of-the-art visual quality and real-time rendering via a differentiable rasterizer, making
3DGS a strong foundation for complex scene understanding tasks. However, such approaches still
require expensive per-scene optimizations.
2

<!-- page 3 -->
To address the computational demands of this per-scene optimization, research has shifted towards
feed-forward methods that generate 3D representations in a single feed-forward pass from a sparse
input images. These models are trained on large datasets to predict scene parameters directly. For
3DGS, methods like MVSplat Chen et al. (2024), DepthSplat Xu et al. (2024), and NoPoSplat Ye et al.
(2024a) demonstrate rapid reconstruction by employing pipelines that typically involve 2D feature
extraction, geometric inference (e.g., depth estimation), and regression of Gaussian attributes. While
these approaches efficiently reconstruct geometry and appearance, they do not inherently produce
semantic information. Integrating semantics requires separate, post-hoc processing.
Incorporating semantic understanding into 3D models is crucial for higher-level reasoning. A common
strategy is to ”lift” features or predictions from powerful pre-trained 2D foundation models, such
as CLIP Radford et al. (2021) for image-text understanding and SAM Kirillov et al. (2023) for
class-agnostic segmentation. Projecting or distilling information from these 2D models into 3D
representations is effective but presents challenges, including maintaining multi-view consistency
and handling occlusions.
Several works have focused on creating 3D representations that store and query semantic information.
For NeRFs, LERF Kerr et al. (2023) enabled open-vocabulary querying by distilling CLIP features
into a 3D language field, but this process requires per-scene optimization for the semantic component.
Similar efforts for 3DGS, such as LangSplat Qin et al. (2024) and LEGaussians Shi et al. (2023),
associate language features with Gaussian primitives, also relying on per-scene optimization to embed
these semantics.
The high dimensionality of language features stored within each Gaussian makes it computationally
expensive for language-embedded 3DGS methods to rasterize them directly Qin et al. (2024). To
address this issue various compression and quantization strategies have been proposed. LangSplat
adopts a pretrained scene-specific autoencoder Qin et al. (2024) to compress the language features
into a lower-dimensional space used for rasterization. After novel view rendering, the autoencoder is
used to decode the language features to their original size. LEGaussians uses quantized language
feature index maps to avoid storing and rasterizing a full language feature for each Gaussian Shi et al.
(2023). A low-dimensional semantic feature vector is added to each gaussian which is then decoded
into a discrete index in the feature index map using a small scene-specific MLP after the rasterization
step.
Gaussian Grouping Ye et al. (2024b) reparametrizes the open-set segmentation problem as a generic
object segmentation task. It learns an identity encoding for each Gaussian representing object
associations from video-tracked object masks found in the input images. Because each gaussian is not
directly imbued with any open-set semantic features this approach requires a secondary step where
a 2D object detection model such as GroundingDINO Liu et al. (2023b) is used to associate a text
prompt with a 3D object in the scene. While methods like LBG Chacko et al. (2025) offer training-free
mechanisms to add semantics to pre-existing 3DGS models, they depend on an already reconstructed
geometric model and do not integrate semantic generation with geometric reconstruction.
Many existing methods for semantic 3D scene understanding, thus, face a bottleneck: even if geometry
is reconstructed quickly, embedding rich semantic features often requires a separate, computationally
intensive per-scene optimization step. This limits their applicability in scenarios demanding rapid,
on-the-fly generation of semantic 3D models. SegSplat directly addresses this limitation. It proposes
a framework to predict both 3D Gaussian Splat geometry and associated open-set semantic features
in a single feed-forward pass from sparse image inputs tailored to the 3DGS representation produced
by feed-forward gaussian splatting. This concurrent generation of geometry and semantics, without
per-scene semantic optimization, is the key distinction of our approach.
3
PROPOSED METHOD
Our proposed method, SegSplat, enables feed-forward generation of 3D Gaussian Splatting (3DGS)
representations augmented with open-set semantic features from sparse multi-view images. The core
idea is to: (1) construct a semantic feature bank by leveraging Segment Anything Model (SAM)
and CLIP extracted from input views, followed by clustering to identify representative semantic
concepts. (2) Utilize a feed-forward model (DepthSplat) to generate 3D Gaussian primitives, where
3

<!-- page 4 -->
Gaussian
Parameters
Prediction
Semantic 3D Gaussians
{(μj , αj , Σj , cj, ej)}H×W ×K
Semantic
Parameters
Prediction
Multi-View Input Images
Semantic Index Maps
CLIP Memory Bank
Novel View RGB
& Language
Feature Maps
Render & Decode
Text Encoder
"Dressing doll"
Relevancy
Computation
Relevancy Map
Figure 2: SegSplat predicts 3D Gaussian splats embedded with language features from sparse
multi-view images, without any training. Our pipeline leverages pretrained DepthSplat to estimate
3D Gaussian parameters per pixel, and uses SAM+CLIP to extract segmentation masks and CLIP
embeddings. To ensure memory efficiency, we construct a CLIP feature memory bank and represent
per-object semantics using one-hot index maps aligned with this bank. These semantic indices are
appended to the Gaussians predicted by DepthSplat. After splatting, we reconstruct full-length
language features via an element-wise product between the rendered index maps and the memory
bank. Novel-view querying is then performed on the decoded CLIP feature image.
each primitive is associated with a semantic index corresponding to an entry in our feature bank. (3)
Render both appearance and semantic features for novel views, enabling open-vocabulary querying.
3.1
SEMANTIC PARAMETER PREDICTION
The aim of this component of the pipeline is to represent semantic information found in the input
images with a memory bank of CLIP features and corresponding per-pixel semantic indices. This
memory bank will allow to have an efficient Gaussian-to-semantic assignment by simply assigning
the index of the semantics from the bank to a given Gaussian primitive. Given K input images
{Ik ∈RH×W ×3}K
k=1 and their corresponding camera projection matrices {Pk ∈R3×4}K
k=1, we
perform the following steps:
1. Mask Extraction: For each input image Ik, we employ SAM Kirillov et al. (2023) to
generate a set of Nk object masks {mki ⊂R2}Nk
i=1. We prompt SAM at a “whole” object
granularity and apply non-maximum suppression to ensure minimal overlap between masks.
Each pixel (u, v) in image Ik is thus associated with at most one mask.
2. CLIP Feature Extraction: For each mask mki, we create a cropped image patch. Pixels
outside the mask mki within this crop are zeroed out. We then compute a DCLIP-dimensional
CLIP image embedding fki ∈RDCLIP using a pre-trained CLIP image encoder Radford et al.
(2021). This yields a collection of Ntotal = PK
k=1 Nk mask-specific CLIP features.
3. Clustering and Bank Formation: To create a concise semantic representation, we pool
all Ntotal CLIP features {fki}. We then apply K-Means clustering to group these features
into M semantic clusters. The centroids of these M clusters form our semantic feature bank
B = [b1, . . . , bM]T ∈RM×DCLIP used in later processes. Each original CLIP feature fki is
assigned to its closest cluster centroid bm.
4. Per-Pixel Semantic Index Maps: Using the bank, we generate a semantic index map
Sk ∈{1, . . . , M}H×W for each input image Ik. For a pixel (u, v) in Ik belonging to mask
mki (assigned to cluster m), we set Sk(u, v) = m. Pixels not belonging to any mask are be
assigned a background index 0. This results in K semantic index maps {Sk}K
k=1.
4

<!-- page 5 -->
3.2
FEED-FORWARD GENERATION OF SEMANTIC GAUSSIANS
We adopt a feed-forward Gaussian Splatting architecture, specifically a frozen DepthSplat model Xu
et al. (2024), to generate the 3D scene representation. Beginning with K sparse input images
{Ii}K
i=1, (Ii ∈RH×W ×3), and their corresponding camera projection matrices {Pi}K
i=1, (P i ∈
R3×4), DepthSplat predicts the per-pixel Gaussian parameters {(µj, αj, Σj, cj)}H×W ×K
j=1
for each
image. Gaussian parameters µj, αj, Σj, and cj denote the 3D Gaussians position, opacity, covariance,
and color (represented as spherical harmonics), respectively.
Because DepthSplat predicts per-pixel Gaussian parameters for each pixel in each input view, we can
directly append a computed one-hot semantic encoding ej derived from the semantic index maps
calculated in Section 3.1 to the other Gaussian parameters. Our full Gaussian splatting representation
becomes gj = {(µj, αj, Σj, cj, ej)} ∈RH×W ×K
j=1
. Such a representation allows for the light-weight
assignment of semantics to Gaussians without significantly inflating the number of parameters that
need to be stored for each primitive. By the end of this step, each Gaussian has a corresponding
semantic class assigned.
3.3
DIFFERENTIABLE RENDERING OF COLOR AND SEMANTICS
To synthesize a novel view given a camera pose Pnovel, we use the standard tile-based 3D Gaussian
splatting rasterization process Kerbl et al. (2023). For each pixel v in the novel view it is the following.
Color Rendering: For each pixel v, the color C(v) ∈R3 is rendered as follows:
C(v) =
X
i∈N
ciαi
i−1
Y
j=1
(1 −αj),
(1)
where ci denotes the color of the i-th Gaussian, N is the set of Gaussians within a tile, and αi =
oiG2D
i
(v), with oi being the opacity of the i-th Gaussian and G2D
i
(·) denoting the projection of the
i-th Gaussian onto 2D.
Semantic Index Rendering: Each 3D Gaussian Gj in our scene representation is associated with a
one-hot semantic vector ei ∈{0, 1}M, indicating its assignment to one of the M semantic concepts
derived from our feature bank. To render the semantic information at a pixel in a novel view, we
adapt the splatting mechanism used for color as follows:
E(v) =
X
i∈N
eiαi
i−1
Y
j=1
(1 −αj),
(2)
where E(v) represents the semantic embedding rendered at pixel v. This vector E(v) represents a
blended distribution of semantic concepts at the pixel, where each element (E(v))m indicates the
aggregated presence of the m-th semantic concept.
CLIP Feature Map Recovery: After rendering, each pixel possesses a blended semantic vector
E(v) ∈RM. Our semantic feature bank B ∈RM×DCLIP stores the DCLIP-dimensional CLIP
embeddings for the M representative semantic concepts (i.e., the m-th row of B, denoted bT
m, is the
CLIP feature vector for the m-th semantic concept). To recover a continuous CLIP feature vector
F (v) ∈RDCLIP for the pixel, we use E(v) to linearly combine the feature vectors stored in the bank
B. Specifically, the m-th component of E(v), (E(v))m, acts as the weight for the m-th semantic
feature vector bm as follows:
F (v) =
M
X
m=1
(E(v))m · bm = E(v)T B.
(3)
This operation effectively translates the rendered (potentially mixed) semantic indices back into the
rich, continuous CLIP feature space. The resulting F (v) for all pixels forms a dense CLIP feature
map for the novel view, which can then be used for open-vocabulary tasks.
As a result, the contributions of each CLIP feature in the memory bank is weighted by the ren-
dered mask encoding. Finally, before performing open-vocabulary querying, the CLIP features are
normalized to unit length.
5

<!-- page 6 -->
3.4
OPEN-VOCABULARY QUERYING
With the rendered CLIP feature map F (v), we can perform open-vocabulary queries. Given a text
query qtext, we compute its CLIP text embedding ϕqry ∈RDCLIP. For each pixel with rendered CLIP
image feature ϕimg = F (v), we calculate a relevancy score following LERF Kerr et al. (2023) as:
relevancy(ϕimg, ϕqry) =
min
i∈{obj, thg, stf}

exp(τ · ϕimg · ϕqry)
exp(τ · ϕimg · ϕqry) + exp(τ · ϕimg · ϕicanon)

(4)
where ϕicanon are CLIP text embeddings of predefined canonical phrases like “object”, “things”, “stuff”,
and τ is a temperature parameter. Relevancy scores below a predefined threshold (e.g., 0.5) are set to
0, and the remaining scores can be further thresholded to obtain fine-grained segmentation masks
corresponding to the input query.
4
EXPERIMENTS
In this section, we evaluate our method on standard benchmarks for 3D open-set semantic seg-
mentation, and we also demonstrate that SegSplat achieves the same photometric scores as the
state-of-the-art DepthSplat.
Datasets. We evaluate our method on two datasets: RealEstate10K (RE10k) (Zhou et al., 2018)
and 3D-OVS (Liu et al., 2023a). RE10k comprises a large collection of real estate videos with
estimated camera parameters. For RE10k, we use an official subset and follow the train/test split of
DepthSplat (Xu et al., 2024). As RE10k lacks mask-level annotations, we present qualitative results
for semantic segmentation.
3D-OVS (Liu et al., 2023a) features scenes with long-tailed object distributions in diverse back-
grounds. For sparse multi-view reconstruction on 3D-OVS, we adopt the train/test split generation
methodology from MVSplat (Chen et al., 2024) and DepthSplat (Xu et al., 2024). Our test split
uses input views with at least 60% projected overlap; target views are those containing ground-truth
semantic labels. We report Intersection over Union (IoU) for 3D semantic segmentation on 3D-OVS.
Baselines. We compare SegSplat against several state-of-the-art methods. It is crucial to distinguish
between methods that perform per-scene optimization or require access to the entire scene context for
semantic understanding, and feed-forward approaches like ours that operate on a limited set of input
views (typically two in our experiments).
First, we consider baselines that leverage extensive information from the target scene, often through
per-scene training or optimization. Consequently, they are expected to achieve higher accuracy and
primarily serve as a reference for the semantic segmentation task, rather than direct competitors to
our feed-forward approach. This group includes open-vocabulary 2D segmentation models such
as LSeg (Li et al., 2022), ODISE (Xu et al., 2023), and OV-Seg (Liang et al., 2023), for which
metrics are typically derived from individual views without enforcing 3D multi-view consistency.
We also include NeRF-based methods like FFD (Kobayashi et al., 2022), LERF (Kerr et al., 2023),
and 3D-OVS (Liu et al., 2023a), requiring per-scene optimization. Furthermore, non-feed-forward
Gaussian Splatting semantic methods such as GS-Grouping (Ye et al., 2024b), and LangSplat (Qin
et al., 2024) fall into this category, as they typically operate on or optimize semantic features for a
3DGS scene representation, benefiting from full scene context.
Second, to ensure a fair comparison for our feed-forward SegSplat, we create a feed-forward compar-
ison group. For this, we adapt the state-of-the-art LangSplat (Qin et al., 2024) to operate within the
same feed-forward, sparse-view framework as SegSplat. Instead of using the pre-trained or optimized
3DGS representations from their original implementations, LangSplat is initialized using the exact
same Gaussians predicted by our common frozen DepthSplat backbone from only two input views.
All methods in this group utilize identical semantic features extracted from SAM (masks) and CLIP
(embeddings) as input. LangSplat requires a scene-specific autoencoder. We train it for each scene
individually (2k iterations, batch size 2, learning rate 2.0 × 10−2, AdamW optimizer). The time
taken for this per-scene autoencoder training is factored into LangSplat’s reported run-time to ensure
a fair comparison against methods that do not require scene-specific training.
Implementation details. Our pipeline utilizes the 37M parameter DepthSplat model, pre-trained
on RE10k with 256×256 resolution images, to predict initial Gaussian parameters. The DepthSplat
6

<!-- page 7 -->
Table 1: 3D semantic segmentation performance comparison (mIoU %) on the 3D-OVS dataset (Liu
et al., 2023a). Methods are grouped by their operational principle. Best results within the feed-forward
category (run only on two input views) and leading results among contextual (non-feed-forward;
trained on all images from a scene) methods are shown in bold.
Principle
Method
Bed
Bench
Room
Sofa
Lawn
Overall
2D Image Segmentation
LSeg (Li et al., 2022)
56.0
6.0
19.2
4.5
17.5
20.6
ODISE (Xu et al., 2023)
52.6
24.1
52.5
48.3
39.8
43.5
OV-Seg (Liang et al., 2023)
79.8
88.9
71.4
66.1
81.2
77.5
NeRF
FFD (Kobayashi et al., 2022)
56.6
6.1
25.1
3.7
42.9
26.9
LERF (Kerr et al., 2023)
73.5
53.2
46.6
27.0
73.7
54.8
3D-OVS (Liu et al., 2023a)
89.5
89.3
92.8
74.0
88.2
86.8
Gaussian Splatting
GS-Grouping (Ye et al., 2024b)
83.0
91.5
85.9
87.3
90.6
87.7
LangSplat (Qin et al., 2024)
92.5
94.2
94.1
90.0
96.1
93.4
Feed-Forward GS
LangSplat (Qin et al., 2024)
59.4
75.1
7.2
50.9
46.0
47.7
SegSplat
66.2
75.9
19.8
64.5
80.8
61.4
Table 2: Novel view synthesis quality for SegSplat and its base geometric model, DepthSplat, on
the RE10k and 3D-OVS datasets. Metrics reported are PSNR↑, SSIM↑, and LPIPS↓. The identical
performance demonstrates that SegSplat’s method of integrating semantic features does not degrade
the geometric and appearance reconstruction fidelity of the underlying DepthSplat model, as SegSplat
utilizes the same Gaussian parameters for rendering.
Dataset
Method
PSNR ↑
SSIM ↑
LPIPS ↓
3D-OVS (Liu et al., 2023a)
DepthSplat (Xu et al., 2024)
16.99
0.392
0.428
SegSplat
16.99
0.392
0.428
RE10K (Zhou et al., 2018)
DepthSplat (Xu et al., 2024)
25.89
0.881
0.122
SegSplat
25.89
0.881
0.122
model is not fine-tuned on 3D-OVS, allowing us to evaluate the zero-shot generalization of our
semantic assignment approach. To ensure view-invariant semantic rendering, the spherical harmonics
degree for the semantic index encoding is set to zero during rasterization. We replace DepthSplat’s
original rasterizer (Kerbl et al., 2023) with gsplat (Ye et al., 2025) for rendering both color and
our semantic index encodings.
For mask extraction from input images (Section 3.1), we employ the SAM ViT-H model (Kirillov
et al., 2023). We use the CLIP ViT-B/16 model (Radford et al., 2021), trained on the LAION-2B
English subset of LAION-5B (Schuhmann et al., 2022), to extract features from SAM-identified
objects and to encode text prompts. The number of clusters M for K-Means during semantic feature
bank construction (Section 3.1) is M = λNtotal/K where Ntotal is the total number of masks from all
K input views, and λ = 1.2. This heuristic aims to accommodate varying object visibility across
views. For open-vocabulary querying (Section 3.4), relevancy scores are computed as in LERF (Kerr
et al., 2023). Scores below 0.5 are set to 0. The resulting relevancy map is then thresholded at 0.5 to
produce binary segmentation masks.
All experiments are conducted on an NVIDIA RTX-5090 GPU, and we report inference/processing
times. The primary evaluation of SegSplat’s performance and efficiency should be made against other
methods within the ”Feed-Forward GS”, as they operate under similar constraints.
4.1
RESULTS
3D Semantic Segmentation Performance. We evaluated open-vocabulary 3D semantic segmentation
capabilities of the proposed SegSplat on the 3D-OVS dataset, with results presented in Table 1. The
methods are benchmarked using mean Intersection over Union (mIoU) and are categorized based on
their underlying methodology. Examples are shown in Figs. 3, 4 and 5.
The table includes several baselines that perform per-scene optimization or leverage extensive infor-
mation from the target scene. These encompass 2D image segmentation techniques (LSeg, ODISE,
OV-Seg), NeRF-based semantic methods (FFD, LERF, 3D-OVS), and Gaussian Splatting approaches
7

<!-- page 8 -->
that optimize semantics for the entire scene (GS-Grouping, LangSplat). These methods, such as the
original LangSplat which achieves an overall mIoU of 93.4%, provide a strong performance reference
due to their comprehensive access to scene data.
Our primary evaluation focuses on the feed-forward Gaussian Splatting (Feed-Forward GS) category,
which operates under the constraint of using only a few input images (two in our setup) without any
per-scene optimization for semantics. Within this challenging setting, SegSplat achieves an overall
mIoU of 61.4%. This significantly surpasses the adapted feed-forward version of LangSplat, which
scores 47.7% mIoU when constrained to the same feed-forward pipeline using an identical DepthSplat
backbone and semantic inputs. SegSplat demonstrates superior performance across the majority of
reported object categories in this direct comparison, underscoring its efficacy in generating robust
semantic segmentations in a single pass.
While a performance difference exists between feed-forward approaches like SegSplat and methods
that utilize full scene context and optimization, SegSplat establishes a strong baseline for efficient,
open-set semantic understanding with 3D Gaussian Splatting in a purely feed-forward manner.
Moreover, it does not require additional per-scene training as LangSplat does.
8adebbb68f2c3f84
ed477bdf8582adff
Ground Truth Color
e4f4574df7938f37
RGB Rendering
Ground Truth Features
Semantic Rendering
Ground Truth Color
RGB Rendering
Ground Truth Features
Semantic Rendering
Figure 3: Comparison of predicted and ground truth (GT) color and semantic maps for novel views
rendered by SegSplat on the RealEstate10K dataset Zhou et al. (2018). Semantic maps are visualized
using PCA. Ground truth semantic features are obtained by applying SAM+CLIP to the corresponding
GT novel view images. Each group of four columns shows: GT RGB image, SegSplat-rendered RGB,
GT semantics, and SegSplat-rendered semantics. This sequence is repeated for a second novel view.
Table 3 reports open-set 3D semantic segmentation results on the ScanNet++ dataset (Yeshwanth
et al., 2023). SegSplat achieves the highest mean IoU among feed-forward approaches. We compare
against the recent SAB3R variants (Chen et al., 2025) and LSM (Fan et al., 2024). SegSplat with
SAM attains 26.0 percent mIoU and surpasses prior feed-forward baselines by a clear margin.
EfficientSAM reduces preprocessing time from 9.70 seconds to 0.98 seconds with a moderate
decrease in accuracy, which shows that SegSplat remains effective when using faster mask extraction.
Replacing the DepthSplat backbone with MVSplat yields 22.2 percent mIoU, demonstrating that
SegSplat generalizes across feed-forward Gaussian Splatting backbones. Inference remains 0.03
seconds for all SegSplat variants.
Geometric Reconstruction Quality. We evaluated whether the integration of our semantic com-
ponent affects the underlying novel view synthesis quality. SegSplat utilizes the geometric and
appearance parameters for its Gaussians directly from the frozen DepthSplat model. The primary
goal of this comparison is to demonstrate that our method for adding semantic features preserves the
rendering fidelity of the base model.
Table 2 presents standard image quality metrics (PSNR, SSIM and LPIPS) for SegSplat and baseline
DepthSplat in the 3D-OVS and RE10k datasets. As shown, SegSplat achieves identical performance
to DepthSplat across all metrics on both datasets. This confirms that our approach for incorporating
open-set semantic understanding does not introduce any degradation to the high-fidelity geometric
and appearance reconstruction capabilities provided by the underlying feed-forward 3DGS model.
8

<!-- page 9 -->
Bed
Bench
Room
Sofa
Ground Truth Color
Lawn
RGB Rendering
Ground Truth Features
Semantic Rendering
Ground Truth Color
RGB Rendering
Ground Truth Features
Semantic Rendering
Figure 4: Comparison of predicted and ground truth (GT) color and semantic maps for novel views
rendered by SegSplat on the 3D-OVS dataset Liu et al. (2023a). Semantic maps are visualized using
PCA. Ground truth semantic features are obtained by applying SAM+CLIP to the corresponding GT
novel view images. Each group of four columns shows: GT RGB image, SegSplat-rendered RGB,
GT semantics, and SegSplat-rendered semantics. This sequence is repeated for a second novel view.
Bench
RGB
LangSplat
SegSplat
“Portuguese egg tart”
“dressing doll”
“green grape”
“mini offroad car”
“orange cat”
“pebbled concrete wall”
“wood”
(a) “Bench” Scene open-vocabulary segmentation.
Lawn
RGB
LangSplat
SegSplat
“New York Yankees cap”
“black headphone”
“green lawn”
“hand soap”
“red apple”
“stapler”
(b) “Lawn” Scene open-vocabulary segmentation.
Figure 5: A qualitative comparison of the masks produced by SegSplat and LangSplat on the 3D-OVS
dataset Liu et al. (2023a). We show results for two scenes and two different novel views. We observe
that our method produces more accurate segmentation masks.
Processing Time. Table 4 presents a detailed runtime comparison of SegSplat and LangSplat,
measured on a single RTX 5090 GPU. While both methods require initial 2D semantic feature
extraction using SAM+CLIP – which accounts for the majority of the offline cost – SegSplat
eliminates the need for scene-specific optimization by performing semantic-to-Gaussian assignment
in a single feed-forward pass. In contrast, LangSplat relies on an expensive 565-second scene-specific
training stage for semantic association. Excluding shared preprocessing, SegSplat achieves a total
inference-time runtime of under 0.2 seconds, over three orders of magnitude faster than LangSplat.
9

<!-- page 10 -->
Table 3: Open-set 3D semantic segmentation results on ScanNet++ (Yeshwanth et al., 2023). SegSplat
achieves the highest mean IoU among feed-forward methods while retaining low inference time. We
report results for SegSplat using EfficientSAM (Xiong et al., 2024) and also using an MVSplat (Chen
et al., 2024) backbone. Preprocessing time corresponds to SAM-based mask extraction.
Method
mean IoU (%)
Training
Preprocessing
Inference
SAB3R (B) (Chen et al., 2025)
4.6
not reported
–
≤0.108 s
SAB3R (C) (Chen et al., 2025)
17.3
not reported
–
≤0.108 s
SAB3R (CD) (Chen et al., 2025)
17.5
not reported
–
≤0.108 s
LSM (Fan et al., 2024)
21.4
3 days on 8×A100
–
0.108 s
SegSplat (SAM)
26.0
–
9.70 s
0.030 s
SegSplat (E-SAM (Xiong et al., 2024))
21.2
–
0.98 s
0.030 s
SegSplat w/ MVSplat
22.2
–
9.70 s
0.030 s
Table 4: Runtime breakdown of SegSplat and LangSplat (in seconds). Per-component runtime on
the 3D-OVS dataset measured on a single RTX 5090 GPU and averaged after two warm-up batches.
LangSplat needs scene-specific training (over 9 mins per scene) before inference, while SegSplat
performs all steps in a single feed-forward pass. After SAM+CLIP features are extracted, SegSplat
maintains a total inference time under 0.2 seconds, making it suitable for real-time tasks.
Component
SegSplat (s)
LangSplat (s)
2D Semantic Feature Extraction
9.7000
9.7000
Semantic Gaussians Prediction
0.0284
565.0236
Gaussian Rasterization
0.0013
0.0016
Language Feature Decoder
0.0004
0.0001
Text Querying
0.0003
0.0002
Total
9.7304
575.1255
This efficiency makes SegSplat highly suitable for real-time or on-the-fly 3D scene segmentation
applications, particularly in dynamic or large-scale environments.
Limitations. Despite its advancements, SegSplat has limitations defining future research directions.
Its semantic understanding quality is tied to the performance of the underlying 2D foundation models
(SAM and CLIP) and the K-Means clustering used for its semantic bank. A key challenge is our
current reliance on aggregating per-image 2D semantic predictions without a learned 3D fusion
mechanism to optimally handle view inconsistencies or occlusions during the 2D-to-3D lifting
process. Additionally, geometric accuracy is inherited from the frozen DepthSplat backbone, and the
system currently focuses on static scenes. Consequently, while SegSplat offers significant efficiency
with sparse inputs, its semantic segmentation accuracy lags behind methods that perform per-scene
optimization using more extensive view information, a trade-off inherent to its feed-forward design.
5
CONCLUSION
We have introduced SegSplat, a novel framework designed to bridge the gap between rapid, feed-
forward 3D reconstruction and rich, open-vocabulary semantic understanding. By constructing a
compact semantic memory bank from multi-view 2D foundation model features and predicting
discrete semantic indices alongside geometric and appearance attributes for each 3D Gaussian
in a single pass, SegSplat efficiently imbues scenes with queryable semantics. Our experiments
demonstrate that SegSplat achieves geometric fidelity comparable to state-of-the-art feed-forward 3D
Gaussian Splatting methods while simultaneously enabling robust open-set semantic segmentation,
crucially without requiring any per-scene optimization for semantic feature integration. This work
represents a significant step towards practical, on-the-fly generation of semantically aware 3D
environments, vital for advancing robotic interaction, augmented reality, and other intelligent systems.
REFERENCES
Rohan Chacko, Nicolai Haeni, Eldar Khaliullin, Lin Sun, and Douglas Lee. Lifting by gaussians: A
simple, fast and flexible method for 3d instance segmentation. In Winter Conference on Applications
10

<!-- page 11 -->
of Computer Vision (WACV), 2025.
Xuweiyi Chen, Tian Xia, Sihan Xu, Jianing Yang, Joyce Chai, and Zezhou Cheng. Sab3r: Semantic-
augmented backbone in 3d reconstruction. arXiv preprint arXiv:2506.02112, 2025.
Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen
Cham, and Jianfei Cai. MVSplat: Efficient 3d gaussian splatting from sparse multi-view images.
arXiv preprint arXiv:2403.14627, 2024.
Zhiwen Fan, Jian Zhang, Wenyan Cong, Peihao Wang, Renjie Li, Kairun Wen, Shijie Zhou, Achuta
Kadambi, Zhangyang Wang, Danfei Xu, et al. Large spatial model: End-to-end unposed images to
semantic 3d. Advances in neural information processing systems, 37:40212–40229, 2024.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting
for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.
Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language
embedded radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer
Vision, pp. 19729–19739, 2023.
Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete
Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings
of the IEEE/CVF international conference on computer vision, pp. 4015–4026, 2023.
Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitzmann. Decomposing nerf for editing via
feature field distillation. Advances in neural information processing systems, 35:23311–23330,
2022.
Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Ren´e Ranftl. Language-driven
semantic segmentation. arXiv preprint arXiv:2201.03546, 2022.
Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter
Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted clip.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.
7061–7070, 2023.
Kunhao Liu, Fangneng Chen, Hengshuang Zhao, and Qi Ji. Weakly supervised 3d open-vocabulary
segmentation with masked image modeling and text-image pairs. In Advances in Neural Informa-
tion Processing Systems (NeurIPS), 2023a.
Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei
Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for
open-set object detection. arXiv preprint arXiv:2303.05499, 2023b.
Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and
Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In European
Conference on Computer Vision (ECCV), pp. 405–421. Springer, 2020.
Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d
language gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 20051–20060, 2024.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In International conference on machine learning, pp.
8748–8763. PmLR, 2021.
Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi
Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski,
Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia
Jitsev. LAION-5b: An open large-scale dataset for training next generation image-text models.
In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks
Track, 2022. URL https://openreview.net/forum?id=M3Y74vmsMcY.
11

<!-- page 12 -->
Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua Guan. Language embedded 3d gaussians
for open-vocabulary scene understanding. arXiv preprint arXiv:2311.18482, 2023.
Yunyang Xiong, Bala Varadarajan, Lemeng Wu, Xiaoyu Xiang, Fanyi Xiao, Chenchen Zhu, Xiaoliang
Dai, Dilin Wang, Fei Sun, Forrest Iandola, et al. Efficientsam: Leveraged masked image pretraining
for efficient segment anything. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pp. 16111–16121, 2024.
Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger,
and Marc Pollefeys.
Depthsplat: Connecting gaussian splatting and depth.
arXiv preprint
arXiv:2410.13862, 2024.
Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Open-
vocabulary panoptic segmentation with text-to-image diffusion models. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2955–2966, 2023.
Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys, Ming-Hsuan Yang, and Songyou Peng.
No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. ICLR
2025, 2024a.
Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit
anything in 3d scenes. In European Conference on Computer Vision, pp. 162–179. Springer,
2024b.
Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari,
Jianbo Ye, Jeffrey Hu, Matthew Tancik, and Angjoo Kanazawa. gsplat: An open-source library for
gaussian splatting. Journal of Machine Learning Research, 26(34):1–17, 2025.
Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-
fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on
Computer Vision, pp. 12–22, 2023.
Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from
one or few images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pp. 4578–4587, 2021.
Qian Zhou, Torsten Sattler, Laura Leal-Taix´e, Thomas Funkhouser, and Steven M. Seitz.
Realestate10k dataset. https://google.github.io/realestate10k/, 2018.
A
APPENDIX
You may include other additional sections here.
12
