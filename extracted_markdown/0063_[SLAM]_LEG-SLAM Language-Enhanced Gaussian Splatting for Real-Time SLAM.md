<!-- page 1 -->
arXiv:2506.03073v1  [cs.CV]  3 Jun 2025
LEG-SLAM: Language-Enhanced Gaussian Splatting for Real-Time SLAM
Roman Titkov1,†, Egor Zubkov1, Dmitry Yudin1,2, Jaafar Mahmoud3,
Malik Mohrat3, Gennady Sidorov3
Abstract
Modern Gaussian Splatting methods have proven highly ef-
fective for real-time photorealistic rendering of 3D scenes.
However, integrating semantic information into this rep-
resentation remains a significant challenge, especially in
maintaining real-time performance for SLAM (Simultane-
ous Localization and Mapping) applications. In this work,
we introduce LEG-SLAM — a novel approach that fuses an
optimized Gaussian Splatting implementation with visual-
language feature extraction using DINOv2 followed by
learnable feature compressor based on Principal compo-
nent analysis, while enabling an online dense SLAM. Our
method simultaneously generates high-quality photorealis-
tic images and semantically labeled scene maps, achiev-
ing real-time scene reconstruction with more than 10 fps
on the Replica dataset and 18 fps on ScanNet.
Experi-
mental results show that our approach significantly out-
performs state-of-the-art methods in reconstruction speed
while achieving competitive rendering quality.
The pro-
posed system eliminates the need for prior data prepara-
tion such as camera’s ego motion or pre-computed static se-
mantic maps. With its potential applications in autonomous
robotics, augmented reality, and other interactive domains,
LEG-SLAM represents a significant step forward in real-
time semantic 3D Gaussian-based SLAM. Project page:
https://titrom025.github.io/LEG-SLAM/
1. Introduction
In recent years, 3D Gaussian Splatting has become a pop-
ular method for representing and rendering novel views in
3D scenes, especially in applications that demand high per-
formance and real-time processing. However, traditional
approaches, which are primarily focused on photorealistic
rendering, often fail short to simultaneously deliver high-
quality semantic information—a critical requirement for ap-
1Center for Cognitive Modeling, Moscow Institute of Physics and Tech-
nology, Russia
2AIRI, Moscow, Russia
3Sberbank of Russia, Robotics Center, Moscow, Russia
†Corresponding author: titkov.re@phystech.edu
2D Input 
Image 
and Depth 
Sequence
Visual-Language 
Feature Encoder
SLAM
Gaussian 
Splatting
Mapper
Text 
Query
3D 
Output 
Scene
e.g. 
“vase”
Pose
Inference
Rendering and Segmentation
Output Image, Heatmap, Mask
Training
Text 
Encoder
Figure 1. Simplified diagram of the proposed LEG-SLAM ap-
proach implementing real-time SLAM with language-enhanced
Gaussian Splatting
plications in autonomous driving, augmented reality, and
robotics.
By real-time, we mean a delay in the operation of the
algorithm that does not exceed the period of receipt of the
input sensory data. For example, for indoor robots, the fre-
quency of image reception is typically 10 frames per sec-
ond.
Following these advantages of the 3D representation, re-
cent works such as LangSplat [23], demonstrate the possi-
bility of integrating language embeddings to support open-
vocabulary queries.
Despite its high semantic quality,
LangSplat suffers from low performance in scene construc-
tion, limiting its real-time applicability. A similar approach,
presented in [16], integrates semantic features into 3D
Gaussian Splatting for accurate mapping and tracking; how-
ever, it operates in a closed-vocabulary mode and requires a
pre-prepared semantic map, which reduces its adaptability
for mapping unseen environments.
In this work, we present the first real-time SLAM
(Simultaneous Localization and Mapping) method with
visual-language features named LEG-SLAM (Language-
Enhanced Gaussian Splatting SLAM). Our approach simul-

<!-- page 2 -->
taneously renders photorealistic RGB images and creates
accurate semantic masks, while executing a dense SLAM
system in real-time. To achieve this, we have developed an
efficient implementation of Gaussian Splatting, integrated
with a semantic extraction module based on DINOv2 [22]
features from RGB images and Principal component analy-
sis for feature compression. This approach significantly re-
duces processing time, saves memory and provides accept-
able segmentation accuracy, making the system suitable for
real-time operation.
One of the main advantages of our proposed method is
the ability to construct a semantic representation of a 3D
scene in real-time, without the need for prior data prepara-
tion. Unlike existing solutions that require static semantic
maps or extensive pre-processing, our approach processes
incoming RGB-D images on the fly, automatically extract-
ing semantic features and building the 3D scene map.
Our main contributions:
• We developed an efficient Gaussian Splatting algorithm
with original visual-language feature encoding based on
DINOv2 and Principal Component Analysis (PCA).
• We implemented the first real-time SLAM with language
features in an unified manner, enabling open-vocabulary
segmentation without the need for a pre-prepared static
semantic map. It allowed us to achieve up to 30 times
faster performance than that of the analogues.
• We achieved real-time map construction for input images
frame rate of 10 fps and competitive results in seman-
tic segmentation quality on popular ScanNet and Replica
datasets.
2. Related work
2.1. 3D Scene Representation Techniques
Historically, 3D scenes have been represented using point
clouds, meshes, and voxels. Point clouds are collections
of points with XYZ coordinates and optional attributes like
color. Meshes use polygons, usually triangles, to model ob-
ject surfaces. Voxels divide 3D space into cubic elements
analogous to 2D pixels.
Recently, neural implicit representations such as Neu-
ral Radiance Fields (NeRF) [21] have gained popularity.
NeRF models a continuous 3D radiance field using neural
networks, enabling highly photorealistic scene reconstruc-
tions. However, it is computationally intensive. Extensions
like Semantic-NeRF [32] and EditNeRF [17] add seman-
tic labels and editing capabilities but remain impractical for
real-time use.
On the other hand, 3D Gaussian Splatting (3DGS) [12]
is a promising new representation that uses 3D Gaussians.
It achieves photorealistic 1080p rendering at 60 FPS by
leveraging point-based alpha blending and a differentiable
tiled rasterizer.
The explicit Gaussian parameterization
also facilitates efficient scene editing. Dynamic 3D Gaus-
sians [18] extends 3DGS to handle dynamic scenes. Other
works [9, 30] generate 3DGS scenes from text and images.
2.2. Semantic 3D Scene Understanding
Large annotated datasets and multimodal models like CLIP
[24] have advanced open-vocabulary semantic segmenta-
tion methods such as OpenSeg [4] and LSeg [13]. However,
most operate on single images without ensuring multi-view
consistency, limiting their utility for 3D scene analysis.
NeRF-based approaches like Semantic-NeRF [32] and
Panoptic Lifting [25] incorporate semantic embeddings for
enhanced 3D scene understanding but suffer from high
computational complexity.
Recent works [5, 16] extend
3DGS with semantic embeddings for open-vocabulary seg-
mentation and reconstruction, but remain computationally
demanding, which prevents the real-time use.
2.3. Neural Representations in SLAM
Neural representations have improved the reconstruction
quality, speed, and semantic understanding of SLAM sys-
tems. SplaTAM [11] uses silhouette-guided optimization
for better reconstruction. GS-SLAM [28] and GS-ICP [6]
integrate adaptive Gaussian expansion and G-ICP pose
tracking for accuracy and scalability.
Hybrid neural-coordinate representations address effi-
ciency issues. Co-SLAM [27] combines multi-resolution
hash grids and one-blob encoding for real-time global bun-
dle adjustment without keyframes.
E-SLAM [10] de-
codes multi-scale axis-aligned feature planes into TSDF
and RGB, achieving state-of-the-art accuracy and speed.
Photo-SLAM [7] connects explicit geometry and implicit
photometric features via hyper primitives and Gaussian-
Pyramid training, surpassing existing methods in quality
and efficiency.
For semantic mapping, DNS SLAM [14] leverages
2D semantic priors and multi-view constraints.
SNI-
SLAM [34] refines scene representations through feature
collaboration and one-way correlation decoding. 3D Gaus-
sian frameworks like SGS-SLAM [16] incorporate seman-
tic colored labels but may miss higher-level semantics.
SemGauss-SLAM [33] and NEDS-SLAM [8] embed low-
dimensional semantic features into 3D Gaussians but rely
on 2D segmentation, limiting open-set robustness.
In summary, neural representations have revolutionized
3D scene understanding and SLAM. Gaussian-based tech-
niques show great promise for efficient, high-quality, and
semantically-aware scene representation. However, further
research is needed to develop real-time open-vocabulary un-
derstanding while preserving multi-view consistency and
scalability.

<!-- page 3 -->
DINOv2
Embedding 
compressor
Interpolation
LEG
Rasterizer
Gaussian 
Pyramid
37x37x768
37x37xK
HxWxK
Poses
ORB-SLAM3
Sparse
view
Gaussians
Input Images
Input Depths
HxWx3
HxWx1
HxWx1
Loss
HgxWgxN
HgxWgxN
HxWx3
 Image P0
 V-L. Emb. P0
 Depth P0
 Image P1
 V-L. Emb. P1
 Depth P1
 Image Pn
 V-L. Emb. Pn
 Depth Pn
Pred. Depth
Pred. Image
Pred. V-L. Emb.
P0
P1
Pn
Visual-
Language  
Embeddings
Figure 2. A detailed scheme of the developed LEG-SLAM method. Its distinctive features include a fast approach to obtaining visual-
linguistic features, a computationally efficient SLAM implementation, and a learning-based approach to a language-enhanced rasterizer
for Gaussian Splatting.
3. Methodology
3.1. Proposed approach
Unlike traditional approaches, where RGB image rendering
and semantic segmentation are performed separately, our
method integrates these processes into a unified pipeline,
ensuring consistency between the visual and semantic rep-
resentation of the 3D scene. Figure 2 provides an overview
of the proposed method. At the core of our approach lies
the Gaussian Splatting method, which is used for efficient
3D scene representation and rendering. To extract visual
features, we employ the DINOv2 model [22], which gener-
ates a compact embedding map characterizing local scene
features. These embeddings are then compressed using an
autoencoder, significantly reducing data size and accelerat-
ing subsequent processing.
The integration of text queries into the semantic seg-
mentation process is performed using the Talk2DINO mod-
ule [1], which converts text embeddings from CLIP [24]
space into the DINOv2 [22] embedding space. This process
is applied to an already reconstructed 3D scene, enabling
open-vocabulary queries, where the semantic features ex-
tracted from the image are matched with textual descrip-
tions in real time.
By
combining
all
these
modules
into
a
single
rasterization-based pipeline, our method enables the simul-
taneous output of high-quality RGB images and semantic
masks, a crucial feature for real-time SLAM systems where
minimizing latency is essential.
The computational effi-
ciency of our approach is detailed in Table 1, which presents
the average processing time per frame across different reso-
lutions. The results highlight the scalability of our method,
demonstrating that even at higher resolutions, the total pro-
cessing time remains within real-time constraints.
Processing Time Per Frame (ms)
Processing Stage
640×480
986×728
1200×680
Image preprocessing
8.72
12.36
14.57
Feature extraction
31.91
40.38
48.33
Feature compression
1.57
2.14
2.36
Tracking + Mapping
14.82
26.85
29.95
Average processing time
57.02
78.65
95.21
Table 1. Average processing time for each pipeline stage per frame
at different resolutions.
The implementation details of each component are pro-
vided in the following sections. Section 3.2 describes the
process of obtaining visual features using DINOv2 [22],
while Section 3.3 explains the embedding compression step
using the autoencoder. The distillation process of semantic
information into the scene representation is detailed in Sec-
tion 3.4, and the procedure for generating semantic queries
using Talk2DINO [1] is covered in Section 3.5.
3.2. Visual features generation
To extract visual features, our system utilizes a pretrained
DINOv2 [22] model, which takes an input image of size
518 × 518 pixels and produces an embedding map of
size 37 × 37 × 768.
In this map, each of the 37 × 37
cells corresponds to a local scene fragment, while the 768-
dimensional vector represents a high-level description of its
visual characteristics.
For further processing, the embedding map is passed to
the next system module without resizing. This approach

<!-- page 4 -->
allows working with a compact scene representation, re-
ducing computational costs. To decrease the dimensional-
ity of embeddings, an encoder compresses them from 768
to K dimensions. This significantly reduces computational
complexity during subsequent processing while maintain-
ing most of the essential information. The details of this
compression are discussed in the next section.
Using DINOv2 enables the formation of an informa-
tive semantic representation, which is then distilled into the
Gaussian Splatting pipeline. This ensures high segmenta-
tion quality while maintaining real-time processing capa-
bilities.
3.3. Embedding Compressor
To optimize the processing of embeddings generated by
the DINOv2 [22] model, our system employs a Principal
Component Analysis (PCA)-based compressor. This mod-
ule reduces the dimensionality of the embedding map from
37 × 37 × 768 to 37 × 37 × K, significantly lowering com-
putational costs while preserving essential semantic infor-
mation.
The PCA model was trained using 1,000 ImageNet text
classes. For each class, a text embedding was obtained via
CLIP [24], projected into the DINOv2 [22] latent space us-
ing Talk2DINO [1], and augmented with noise while main-
taining a cosine similarity of at least 0.95 with the original.
This strategy allows PCA to efficiently compress embed-
dings beyond ImageNet classes by learning a general fea-
ture projection rather than specific class representations.
After compression, the embedding map is upsampled to
the original image resolution using bilinear interpolation,
ensuring spatial consistency with the input image. The re-
sulting H ×W ×K representation is then used in the Gaus-
sian Splatting pipeline for joint rendering of RGB images
and semantic maps. This approach enables real-time per-
formance while preserving segmentation accuracy.
3.4. Integration of Semantic Features and Gaussian
Splatting
In our approach, semantic features are incorporated to the
Gaussian splatting pipeline for simultaneous rendering of
RGB images and semantic masks. This allows to effectively
combine color and semantic information in a scene, ensur-
ing consistency of representations and high performance.
Semantic vector representing local features from the se-
mantic embedding map is added to the initial parameters of
each Gaussian. This semantic vector is optimized simulta-
neously with gaussian parameters (such as position, scale,
orientation and color), which allows to obtain an accurate
match between visual and semantic data.
The optimization of semantic features vector includes
the following steps:
1. Initialization of semantic vectors:
During the ini-
tial initialization of the gaussian, the semantic vector
H × W × K is assigned zero values. If the gaussian
is obtained by splitting of an existing gaussian, then the
semantic features are copied.
2. Parameter optimization: During the training process,
the gaussian parameters are optimized by minimizing the
error between the rendered and the reference 2d repre-
sentations. Simultaneous training on a single gaussian
cloud ensures consistency of visual and semantic data.
3. Simultaneous
rendering
of
RGB
and
semantic
masks: The resulting gaussian parameters are used to
simultaneously form a photorealistic image of a scene
and corresponding semantic masks, where each gaussian
contributes to both representations.
Using a single pipeline for visual and semantic data
can significantly reduce computational costs, while ensur-
ing high accuracy and speed of reconstruction.
The loss function includes three terms. The first of them
Lcolor is the standard loss function for color features from
the original 3DGS method. The second term Lcos sim is re-
sponsible for calculating the cosine similarity between the
rendered semantic feature map and the one obtained from
DINOv2. The third term L1 optimizes the depth value. The
inputs of the function are the rendered image Ipred, the lan-
guage feature map Lpred and the depth Dpred, as well as the
gt image Igt, the depth Dgt and the compressed DINOv2
features Lgt.
L(Ipred, Lpred, Dpred; Igt, Lgt, Dgt) =
= (1 −λ)L1(Ipred, Igt) + λLssim(Ipred, Igt)
|
{z
}
Lcolor
+
+ Lcos sim(Lpred, Lgt)+
+ L1(Dpred, Dgt).
(1)
To speed up and improve the quality of reconstruction,
optimization of gaussians is performed on input images
which resolution is consistently increasing (Hg × Wg). The
resolution of the input images is reduced by using a Gaus-
sian pyramid. This approach allows to quickly set fairly
accurate parameters for gaussians, which ensures fast con-
vergence on large-dimensional images.
3.5. Segmentation by Text Query Using DINOv2
Features
To perform segmentation based on a text query, our sys-
tem employs the Talk2DINO model, which converts text
embeddings from CLIP space into the embedding space of
DINOv2. This transformation is necessary because CLIP
and DINOv2 embeddings are trained on different tasks and
exist in distinct vector spaces. Talk2DINO bridges this gap,
allowing textual descriptions to be linked with visual fea-

<!-- page 5 -->
tures of the scene, enabling accurate open-vocabulary seg-
mentation.
In our approach, an embedding map of size H × W ×
K is created by projecting the reconstructed 3D scene onto
the current camera viewpoint. This projection is computed
based on the contributions of all Gaussians in the scene to
each pixel of the image.
The segmentation process consists of several steps:
1. Text Embedding Generation: The text query is con-
verted into an embedding of size 1×768 using the CLIP
model.
2. Feature Space Alignment: Talk2DINO transforms this
embedding into the DINOv2 feature space, producing a
vector of size 1 × 768.
3. Dimensionality Reduction: PCA is applied to com-
press the text embedding to 1 × K, aligning it with the
visual embeddings from the H × W × K map. This
reduces computational overhead during similarity com-
parison.
4. Semantic Similarity Computation: The embedding
map H ×W ×K, generated via scene projection, is used
to compute a similarity map with the text query. This
is achieved by performing a dot product between each
scene embedding and the compressed text embedding.
The result is a semantic similarity map of size H×W×1.
By projecting the reconstructed scene through Gaussians
onto a 2D plane, the system generates an up-to-date seman-
tic feature map for each frame, which is then used to seg-
ment objects based on the query.
4. Experiments
In this section, we conduct a series of experiments to eval-
uate the effectiveness of LEG-SLAM in the task of open-
vocabulary 3D scene reconstruction with semantic under-
standing. We analyze the method’s performance on various
datasets and compare it with existing approaches.
First, we evaluate the quality of 3D reconstruction on the
Replica dataset. Then, we assess semantic segmentation ac-
curacy on ScanNet dataset, demonstrating open-vocabulary
scene understanding. Next, we analyze SLAM performance
on Replica, examining the accuracy of camera trajectory es-
timation and tracking speed. Additionally, we investigate
the impact of different factors, including embedding com-
pression methods, visual feature extraction architectures,
and rendering strategies.
Finally, we compare LEG-SLAM with existing methods
in terms of 3D reconstruction quality, semantic segmenta-
tion accuracy, and computational efficiency, highlighting its
advantages for real-time language-enhanced SLAM.
4.1. Experimental Setup
4.1.1. Datasets
To evaluate the quality of scene reconstruction and semantic
segmentation in an open-vocabulary setting, we use several
widely adopted datasets in 3D Gaussian Splatting research.
Replica [26] is a high-quality synthetic dataset contain-
ing realistic 3D-rendered indoor environments. It is widely
used for evaluating 3D reconstruction methods, as it pro-
vides scenes with detailed geometry and textures. In our
experiments, Replica serves as a benchmark for assessing
the accuracy of scene reconstruction.
ScanNet [3] is an RGB-D dataset consisting of real-
world scanned indoor scenes with annotated 3D camera
poses and 2D object class labels. Due to the presence of se-
mantic annotations, it is well-suited for evaluating semantic
segmentation performance. Our experiments are conducted
on 12 validation scenes from ScanNet.
Each of these datasets plays a distinct role in our eval-
uation. While Replica provides a controlled environment
for assessing reconstruction quality, ScanNet allows us to
analyze both reconstruction and semantic segmentation in
real-world indoor scenes. This diverse selection ensures a
comprehensive assessment of LEG-SLAM across multiple
challenging scenarios, covering both 3D reconstruction and
semantic scene understanding.
4.1.2. Implementation Details
All experiments were conducted on an NVIDIA RTX 4090
GPU with 24 GB of video memory. We use DINOv2 ViT-
B/14 as the backbone for extracting visual features and
CLIP ViT-B/16 for processing text queries. To project tex-
tual embeddings into the visual feature space, we employ
the Talk2DINO module.
4.2. Quantitative Analysis
To further illustrate the effectiveness of our method in open-
vocabulary semantic understanding, Figure 3 presents RGB
renderings and corresponding semantic heatmaps for vari-
ous text queries across different scenes. Each image show-
cases how LEG-SLAM successfully localizes and high-
lights queried objects, even for ambiguous or spatially com-
plex environments.
For instance, Figure 3a demonstrates accurate identifi-
cation of a “basket”, while figure 3b highlights the correct
localization of a “chair”. Figure 3c showcases the method’s
ability to recognize a “plant”, where fine-grained textures
and occlusions make segmentation challenging.
Finally,
Figure 3d presents a successful retrieval of a “vase”, cap-
turing both its geometric structure and material properties.
These qualitative results confirm that LEG-SLAM not
only provides high-fidelity 3D reconstructions but also ac-
curately aligns semantic information with the rendered en-

<!-- page 6 -->
Method
ATE RMSE ↓
Depth L1 ↓
FPS ↑
PSNR (dB) ↑
SSIM ↑
LPIPS ↓
Semantics
NICE-SLAM [35]
2.51
1.903
0.198
24.42
0.809
0.233
/
Vox-Fusion [29]
1.47
2.913
1.07
24.41
0.801
0.236
/
Co-SLAM [27]
1.06
1.513
6.41
30.24
0.939
0.252
/
ESLAM [10]
0.62
0.945
3.02
29.08
0.929
0.239
/
SplaTAM [11]
0.41
0.49
0.21
34.11
0.968
0.102
/
Gaussian-SLAM [31]
0.31
0.68
0.63
42.08
0.996
0.018
/
MonoGS [20]
0.79
/
0.445
37.50
0.96
0.07
/
NEDS-SLAM [8]
0.354
0.47
-
34.76
0.962
0.088
Closed-Vocabulary
RGBDS-SLAM [2]
0.589
0.342
-
38.85
0.967
0.035
Closed-Vocabulary
SNI-SLAM [34]
0.456
0.766
2.15
-
-
-
Closed-Vocabulary
GS3SLAM [15]
0.37
-
-
36.26
0.989
0.052
Closed-Vocabulary
SGS-SLAM [16]
0.412
0.356
2.11
34.15
0.97
0.096
Closed-Vocabulary
OVO-Gaussian-SLAM [19]
0.31
0.68
0.28
42.08
0.996
0.018
Open-Vocabulary
Photo-SLAM [7] (baseline)
1.24
/
27.35
30.91
0.883
0.143
/
LEG-SLAM (Ours)
0.94
2.15
10.56
32.12
0.9151
0.1456
Open-Vocabulary
Table 2. Quantitative comparison of SLAM accuracy, speed, rendering quality, and semantic capabilities for LEG-SLAM and other SLAM
methods. The results are averaged over 8 scenes from the Replica dataset.
(a) Text query: ”basket”
(b) Text query: ”chair”
(c) Text query: ”plant”
(d) Text query: ”vase”
Figure 3. Visualization of RGB and semantic heatmap outputs for different text queries. Each image contains both an RGB rendering (top)
and a semantic heatmap (bottom) highlighting regions matching the given query.
vironment, making it well-suited for open-vocabulary ob-
ject localization in real-time applications.
4.3. Scene Reconstruction
To evaluate the quality of 3D reconstruction, we conducted
experiments on the Replica dataset. Table 2 presents the
results of our method compared to existing SLAM ap-
proaches.
Our method achieves competitive reconstruction quality,
slightly trailing behind SplaTAM in terms of PSNR and
SSIM. This difference arises from the additional semantic
rendering in our approach, which introduces constraints on
the optimization process. Despite this, our method main-
tains strong perceptual quality, as reflected in LPIPS scores,
demonstrating its effectiveness in jointly reconstructing ge-
ometry and semantics in real-time applications.
4.4. Open-Vocabulary Semantic Segmentation
We compare LEG-SLAM against state-of-the-art methods
on the ScanNet dataset. The segmentation quality is evalu-
ated using mIoU and mAcc metrics.
Table 3 presents a comparative analysis of NeRF and
Gaussian Splatting-based methods, reporting the average
segmentation metrics and training time across 12 ScanNet
scenes. Figure 4 visually illustrates the segmentation results
for different methods.
Results show that while LEG-SLAM does not achieve
the highest mIoU, it significantly outperforms competing
methods in runtime efficiency, running at 18 FPS on Scan-
Net.
On average, it processes a scene in 1.5 minutes,
whereas other methods require tens of minutes or even
hours per scene.

<!-- page 7 -->
LSeg
Feature3DGS
LangSplat
Semantic
Gaussians
LEG-SLAM
(Ours)
GT
Figure 4. Comparison of semantic segmentation performance across different pipelines on the Replica dataset, including ground truth for
reference.
Method
Backbone
mIoU,
%
mAcc,
%
Training
Time
LERF
NeRF+CLIP
31.2
61.7
45 min
PVLFF
NeRF+LSeg
52.9
67.0
65 min
LangSplat
3DGS+CLIP
24.7
42.0
180 min
Feature3DGS
3DGS+LSeg
59.2
75.1
150 min
OVO-Gaussian-SLAM
3DGS+CLIP
29.3
41.1
138 min
OVO-mapping
-
38.1
50.5
2.6 min
Semantic Gaussians
3DGS+LSeg
62.0
77.0
90 min
LEG-SLAM
3DGS+DINOv2
41.4
74.3
1.5 min
Table 3. Comparison of Open-Vocabulary segmentation methods
on 12 Scenes of ScanNet (Average 1800 Frames per Scene)
4.5. SLAM Comparison
We evaluate the performance of LEG-SLAM in the SLAM
task by comparing it with existing state-of-the-art methods
on the Replica dataset. The primary metrics include ATE
RMSE for trajectory accuracy and rendering quality. Addi-
tionally, we emphasize the system’s speed, which is critical
for real-time applications.
As shown in Table 2, LEG-SLAM achieves competi-
tive reconstruction accuracy while significantly outperform-
ing other methods in terms of speed, processing frames
at 10 FPS. While SplaTAM yields slightly better recon-
struction and rendering quality, it does not support real-
time performance. The added complexity of optimizing 3D
Gaussians for both geometric and semantic consistency in-
troduces minor reconstruction trade-offs but enables open-
vocabulary semantic segmentation, making LEG-SLAM
the only method in this comparison capable of integrating
real-time SLAM with open-vocabulary scene understand-
ing.
4.6. Ablation studies
4.6.1. Comparison of Different Feature Extraction Back-
bones
Selecting an efficient feature extraction method is crucial
for real-time performance. We compare two architectures:
DINOv2 ViT-B/14 and OpenSeg. Both provide high-quality
visual features but differ in computational efficiency and
adaptability to open-vocabulary tasks.
DINOv2 is a self-supervised model trained to produce
generalizable visual embeddings without relying on prede-
fined categories. This makes it well-suited for diverse envi-
ronments, where it maintains high feature informativeness.
In contrast, OpenSeg is trained on COCO categories and
performs well in segmentation but struggles with general-
ization in open-vocabulary settings.
To evaluate computational efficiency, we measured pro-
cessing time, including embedding compression using
PCA. Results show that OpenSeg + PCA takes 100 ms,
while DINOv2 + PCA completes the same operation in 33
ms — a 3× speedup.
Given the constraints of real-time
operation, this significant reduction in latency makes DI-
NOv2 the preferred choice, offering a balance between fea-
ture quality and computational cost.
Method
CosSim ↑
mIoU ↑
Time (ms) ↓
MLP-AE (2-layer)
0.9379
37.3
0.85
MLP-AE (4-layer)
0.9551
34.8
2.96
PCA
0.9570
41.2
0.60
Table 4.
Comparison of embedding compression methods (all
methods compress to 64 dimensions)

<!-- page 8 -->
Dim
Cos. Sim.↑
mIoU↑
PSNR↑
Train FPS↑
3
0.7423
6.6
24.34
18.38
6
0.8348
20.1
25.07
18.81
9
0.8814
27.2
24.83
18.72
12
0.9070
32.0
24.53
18.63
16
0.9298
37.1
24.7
18.4
24
0.9532
39.1
24.75
18.22
32
0.9662
40.1
24.48
18.09
48
0.9808
40.4
24.30
17.52
64
0.9880
41.4
24.26
17.55
80
0.9920
42.2
23.96
17.63
96
0.9946
42.3
23.65
17.58
128
0.9973
42.7
23.57
17.44
Table 5. Results of PCA-based embedding compression experi-
ments
4.6.2. Influence of Embedding Compression Methods
Efficient embedding compression is crucial for real-time ap-
plications. Table 4 compares PCA and two MLP-based Au-
toencoders (MLP-AE), all compressing embeddings from
768 to 64 dimensions.
PCA achieves the best trade-off between accuracy and
efficiency, yielding the highest cosine similarity and seg-
mentation quality while maintaining the fastest inference
time.
The 2-layer MLP-AE shows a noticeable drop in
segmentation accuracy, while the deeper 4-layer variant im-
proves reconstruction quality but suffers from significantly
higher computational cost. Despite its deeper architecture,
the 4-layer model does not outperform PCA, making it less
suitable for real-time deployment.
Given these results, PCA emerges as the most effective
choice, offering a superior balance between accuracy, com-
putational speed, and implementation simplicity.
4.6.3. Impact of embedding dimensionality
To
determine
the
optimal
dimensionality
for
PCA-
compressed embeddings, we conducted two separate eval-
uations: compression quality analysis using Cosine Sim-
ilarity and full pipeline performance measured by mIoU,
PSNR, and training speed in FPS.
For compression quality assessment, we randomly se-
lected 10 frames from each of the 12 ScanNet validation
scenes and extracted DINOv2 embeddings at the original
resolution of 648 × 484 × 768. To prevent redundancy from
neighboring pixels, embeddings were sampled with a stride
of 4, ensuring a diverse set of semantic features. PCA was
then applied with embedding dimensionalities ranging from
3 to 128, and reconstruction quality was assessed using Co-
sine Similarity.
To evaluate the overall impact on pipeline performance,
the full LEG-SLAM pipeline was executed on complete
ScanNet scenes, where segmentation quality was measured
using mIoU, PSNR, and training FPS.
Figure 5. VRAM consumption across different feature compressor
embedding dimensions
The results indicate that increasing the embedding di-
mensionality improves Cosine Similarity. However, after
reaching 64 dimensions, these improvements become negli-
gible, while processing time continues to rise. Additionally,
larger embedding sizes introduce increased VRAM con-
sumption, complexity in optimizing Gaussian parameters,
leading to a decline in PSNR.
Interestingly, at an embedding dimensionality of 3, com-
pression is highly lossy, resulting in poor reconstruction of
feature embeddings. This leads to ineffective optimization
of Gaussian parameters, causing an overgeneration of Gaus-
sians as the pipeline attempts to compensate for lost infor-
mation. This behavior significantly increases VRAM con-
sumption (as shown in Figure 5) without yielding meaning-
ful improvements in scene reconstruction or segmentation
quality.
The selected dimensionality of 64 provides a favorable
balance, preserving essential semantic information while
maintaining efficient optimization and reconstruction qual-
ity. It ensures that feature embeddings retain sufficient ex-
pressiveness without unnecessary computational overhead.
5. Conclusion
We introduced LEG-SLAM, a novel approach that com-
bines 3D Gaussian Splatting with open-vocabulary seman-
tic understanding, enabling real-time 3D reconstruction and
interactive scene interpretation based on textual queries.
Unlike existing methods, our approach does not rely on pre-
defined object categories and provides efficient integration
of semantic and geometric information.
Experiments demonstrate that LEG-SLAM achieves
competitive reconstruction quality on the Replica dataset
while maintaining high-speed performance for semantic
segmentation on ScanNet.
The use of DINOv2 for fea-
ture extraction and PCA for embedding compression en-
sures computational efficiency without significant loss of
accuracy. However, integrating semantic information dur-
ing optimization slightly affects reconstruction quality due
to additional constraints in Gaussian optimization.

<!-- page 9 -->
Furthermore,
LEG-SLAM
significantly
accelerates
open-vocabulary Gaussian Splatting-based segmentation.
Compared to LERF, our method achieves a 30× speedup,
making real-time interactive scene understanding feasible
for practical applications.
Despite minor trade-offs in reconstruction quality due to
semantic constraints, LEG-SLAM offers a unique balance
between speed, adaptability, and real-time capability, mak-
ing it a promising solution for future 3D scene understand-
ing tasks.
References
[1] Luca Barsellotti, Lorenzo Bianchi, Nicola Messina, Fabio
Carrara, Marcella Cornia, Lorenzo Baraldi, Fabrizio Falchi,
and Rita Cucchiara.
Talking to DINO: Bridging Self-
Supervised Vision Backbones with Language for Open-
Vocabulary Segmentation, 2024. arXiv:2411.19331 [cs]. 3,
4
[2] Zhenzhong Cao, Chenyang Zhao, Qianyi Zhang, Jinzheng
Guang, and Yinuo Song Jingtai Liu.
RGBDS-SLAM: A
RGB-D Semantic Dense SLAM Based on 3D Multi Level
Pyramid Gaussian Splatting, 2024. arXiv:2412.01217 [cs].
6
[3] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Hal-
ber, Thomas Funkhouser, and Matthias Niebner. ScanNet:
Richly-Annotated 3D Reconstructions of Indoor Scenes.
2017. Accepted: 2021-10-08T19:50:29Z. 5
[4] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Scal-
ing Open-Vocabulary Image Segmentation with Image-Level
Labels. In Computer Vision – ECCV 2022: 17th European
Conference, Tel Aviv, Israel, October 23–27, 2022, Proceed-
ings, Part XXXVI, pages 540–557, Berlin, Heidelberg, 2022.
Springer-Verlag. 2
[5] Jun Guo, Xiaojian Ma, Yue Fan, Huaping Liu, and Qing Li.
Semantic Gaussians: Open-Vocabulary Scene Understand-
ing with 3D Gaussian Splatting, 2024.
arXiv:2403.15624
[cs]. 2
[6] Seongbo Ha, Jiung Yeon, and Hyeonwoo Yu. RGBD GS-
ICP SLAM, 2024. eprint: 2403.12550. 2
[7] Huajian Huang, Longwei Li, Hui Cheng, and Sai-Kit Ye-
ung.
Photo-SLAM: Real-time Simultaneous Localization
and Photorealistic Mapping for Monocular Stereo and RGB-
D Cameras. pages 21584–21593, 2024. 2, 6
[8] Yiming Ji, Yang Liu, Guanghu Xie, Boyu Ma, and Zongwu
Xie.
NEDS-SLAM: A Neural Explicit Dense Semantic
SLAM Framework using 3D Gaussian Splatting.
IEEE
Robotics and Automation Letters, 9(10):8778–8785, 2024.
arXiv:2403.11679 [cs]. 2, 6
[9] Lutao Jiang, Xu Zheng, Yuanhuiyi Lyu, Jiazhou Zhou, and
Lin Wang. BrightDreamer: Generic 3D Gaussian Genera-
tive Framework for Fast Text-to-3D Synthesis, 2024. eprint:
2403.11273. 2
[10] Mohammad Mahdi Johari, Camilla Carta, and Francois
Fleuret.
ESLAM: Efficient Dense SLAM System Based
on Hybrid Representation of Signed Distance Fields.
In
2023 Ieee/Cvf Conference On Computer Vision And Pat-
tern Recognition (Cvpr), pages 17408–17419. Los Alamitos,
2023. ISSN: 1063-6919. 2, 6
[11] Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula,
Gengshan Yang, Sebastian Scherer, Deva Ramanan, and
Jonathon Luiten. SplaTAM: Splat, Track & Map 3D Gaus-
sians for Dense RGB-D SLAM spla-tam . github . io. 2, 6
[12] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler,
and George Drettakis. 3D Gaussian Splatting for Real-Time
Radiance Field Rendering. ACM Transactions on Graphics,
42(4):1–14, 2023. 2
[13] Boyi Li, Kilian Q. Weinberger, Serge Belongie, Vladlen
Koltun, and Ren´e Ranftl. Language-driven Semantic Seg-
mentation, 2022. arXiv:2201.03546 [cs]. 2
[14] Kunyi Li, Michael Niemeyer, Nassir Navab, and Federico
Tombari.
DNS SLAM: Dense Neural Semantic-Informed
SLAM, 2023. eprint: 2312.00204. 2
[15] Linfei Li, Lin Zhang, Zhong Wang, and Ying Shen. GS3
LAM: Gaussian Semantic Splatting SLAM. In Proceedings
of the 32nd ACM International Conference on Multimedia,
pages 3019–3027, Melbourne VIC Australia, 2024. ACM. 6
[16] Mingrui Li, Shuhong Liu, Heng Zhou, Guohao Zhu, Na
Cheng, Tianchen Deng, and Hongyu Wang. SGS-SLAM:
Semantic Gaussian Splatting For Neural Dense SLAM.
pages 163–179. 2025. arXiv:2402.03246 [cs]. 1, 2, 6
[17] Steven Liu, Xiuming Zhang, Zhoutong Zhang, Richard
Zhang, Jun-Yan Zhu, and Bryan Russell. Editing Conditional
Radiance Fields. In 2021 IEEE/CVF International Confer-
ence on Computer Vision (ICCV), pages 5753–5763, Mon-
treal, QC, Canada, 2021. IEEE. 2
[18] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and
Deva Ramanan. Dynamic 3D Gaussians: Tracking by Per-
sistent Dynamic View Synthesis, 2023. arXiv:2308.09713
[cs]. 2
[19] Tomas Berriel Martins, Martin R. Oswald, and Javier Civera.
OVO-SLAM: Open-Vocabulary Online Simultaneous Local-
ization and Mapping, 2024. arXiv:2411.15043 [cs] version:
1. 6
[20] Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and
Andrew J. Davison.
Gaussian Splatting SLAM, 2024.
arXiv:2312.06741 [cs]. 6
[21] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik,
Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF:
Representing Scenes as Neural Radiance Fields for View
Synthesis, 2020. arXiv:2003.08934 [cs]. 2
[22] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, and
others. DINOv2: Learning Robust Visual Features without
Supervision. Transactions on Machine Learning Research
Journal, pages 1–31, 2024. 2, 3, 4
[23] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and
Hanspeter Pfister. Langsplat: 3d language gaussian splatting.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 20051–20060, 2024.
1
[24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,

<!-- page 10 -->
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever.
Learning Transferable Vi-
sual Models From Natural Language Supervision, 2021.
arXiv:2103.00020 [cs]. 2, 3, 4
[25] Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bul´o, Nor-
man M¨uller, Matthias Nießner, Angela Dai, and Peter
Kontschieder. Panoptic Lifting for 3D Scene Understanding
with Neural Fields, 2022. arXiv:2212.09802 [cs]. 2
[26] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik
Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal,
Carl Ren, Shobhit Verma, Anton Clarkson, Mingfei Yan,
Brian Budge, Yajie Yan, Xiaqing Pan, June Yon, Yuyang
Zou, Kimberly Leon, Nigel Carter, Jesus Briales, Tyler
Gillingham, Elias Mueggler, Luis Pesqueira, Manolis Savva,
Dhruv Batra, Hauke M. Strasdat, Renzo De Nardi, Michael
Goesele, Steven Lovegrove, and Richard Newcombe. The
Replica Dataset: A Digital Replica of Indoor Spaces, 2019.
arXiv:1906.05797 [cs]. 5
[27] Hengyi Wang, Jingwen Wang, and Lourdes Agapito. Co-
SLAM: Joint Coordinate and Sparse Parametric Encodings
for Neural Real-Time SLAM, 2023. arXiv:2304.14377 [cs].
2, 6
[28] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong
Wang, and Xuelong Li. GS-SLAM: Dense Visual SLAM
with 3D Gaussian Splatting, 2024. arXiv:2311.11700 [cs]. 2
[29] Xingrui Yang, Hai Li, Hongjia Zhai, Yuhang Ming, Yuqian
Liu, and Guofeng Zhang.
Vox-Fusion: Dense Tracking
and Mapping with Voxel-based Neural Implicit Represen-
tation.
In 2022 IEEE International Symposium on Mixed
and Augmented Reality (ISMAR), pages 499–507, 2022.
arXiv:2210.15858 [cs]. 6
[30] Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi
Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang
Wang. GaussianDreamer: Fast Generation from Text to 3D
Gaussians by Bridging 2D and 3D Diffusion Models, 2024.
eprint: 2310.08529. 2
[31] Vladimir Yugay, Yue Li, Theo Gevers, and Martin R. Os-
wald. Gaussian-SLAM: Photo-realistic Dense SLAM with
Gaussian Splatting, 2024. arXiv:2312.10070 [cs]. 6
[32] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and
Andrew J. Davison.
In-Place Scene Labelling and Un-
derstanding with Implicit Scene Representation,
2021.
arXiv:2103.15875 [cs]. 2
[33] Siting Zhu, Renjie Qin, Guangming Wang, Jiuming Liu, and
Hesheng Wang. SemGauss-SLAM: Dense Semantic Gaus-
sian Splatting SLAM, 2024. eprint: 2403.07494. 2
[34] Siting Zhu, Guangming Wang, Hermann Blum, Jium-
ing Liu,
Liang Song,
Marc Pollefeys,
and Hesheng
Wang. SNI-SLAM: Semantic Neural Implicit SLAM, 2024.
arXiv:2311.11016 [cs]. 2, 6
[35] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hu-
jun Bao, Zhaopeng Cui, Martin R. Oswald, and Marc Polle-
feys. NICE-SLAM: Neural Implicit Scalable Encoding for
SLAM, 2022. arXiv:2112.12130 [cs]. 6

<!-- page 11 -->
LEG-SLAM: Language-Enhanced Gaussian Splatting for Real-Time SLAM
Supplementary Material
1. ScanNet dataset
To further analyze the performance of LEG-SLAM, we
evaluate the reconstruction and semantic segmentation met-
rics on individual ScanNet scenes. Table 1 presents per-
scene results, where PSNR, SSIM, and LPIPS measure re-
construction quality, while mIoU and mAcc assess semantic
segmentation accuracy.
For semantic segmentation, LEG-SLAM operates in an
open-vocabulary setting, but to quantitatively evaluate per-
formance, we align with the standard ScanNet benchmark.
The evaluation uses 20 semantic categories, which are de-
tailed in Table 2, alongside their per-class IoU scores.
To ensure effective feature compression, the same 20
categories were used to train the PCA encoder for Scan-
Net dataset evaluation. This alignment allows PCA to learn
the optimal projection for retaining key semantic infor-
mation while reducing computational complexity.
How-
ever, since PCA learns a generalized compression strat-
egy rather than memorizing specific class distributions, our
method remains effective even on previously unseen cat-
egories.
While segmentation quality is more consistent
on trained categories, LEG-SLAM successfully extends to
open-vocabulary scenarios, making it applicable across di-
verse 3D environments.
Scene
PSNR (dB)
SSIM
LPIPS
mIoU
mAcc
0050 02
21.51
0.6963
0.3399
0.3157
0.5474
0144 01
25.23
0.8167
0.1914
0.5388
0.6758
0221 01
23.69
0.7314
0.3868
0.4939
0.6852
0300 01
26.66
0.8410
0.2041
0.5340
0.6204
0354 00
20.98
0.8307
0.2048
0.5554
0.7378
0389 00
26.51
0.8302
0.2183
0.5382
0.8238
0423 02
23.14
0.7491
0.2386
0.4499
0.4633
0427 00
25.41
0.8080
0.2163
0.4985
0.6413
0494 00
24.96
0.8264
0.1902
0.5872
0.7972
0616 00
19.03
0.7386
0.3031
0.5872
0.8007
0645 02
22.03
0.7797
0.2892
0.4229
0.7224
0693 00
27.90
0.8394
0.1870
0.4115
0.7032
Table 1. Quantitative evaluation on ScanNet scenes
2. Replica dataset
To contextualize LEG-SLAM’s performance, we compare
it against existing SLAM methods on different Replica
scenes. Table 3 summarizes the results, providing a de-
tailed breakdown of reconstruction quality across multiple
environments. The table reports PSNR, SSIM, and LPIPS
metrics, averaged over all scenes and shown individually for
each tested environment.
Class
IoU
Wall
0.364
Floor
0.729
Cabinet
0.198
Bed
0.759
Chair
0.497
Sofa
0.275
Table
0.577
Door
0.450
Window
0.441
Shelves
0.521
Counter
not presented
Curtain
0.626
Ceiling
0.338
Refrigerator
0.385
Television
0.394
Person
not presented
Toilet
0.471
Sink
0.034
Lamp
0.219
Bag
0.181
Mean IoU
0.414
Mean Accuracy
0.743
Table 2. Per-class Intersection over Union (IoU) scores for Scan-
Net dataset. Some classes (counter, person) were not evaluated
due to annotation errors.
3. Impact of Embedding Dimensionality on Se-
mantic Quality
To analyze the effect of embedding compression, we com-
pare the semantic reconstruction quality across different
PCA embedding dimensions, as shown in Figure 1. The vi-
sualizations illustrate how varying the dimensionality from
3 to 64 affects the ability to preserve fine-grained semantic
details while maintaining computational efficiency.
Lower-dimensional embeddings (3–9 dimensions) result
in a significant loss of semantic information, leading to
blurry segmentations and imprecise object boundaries. The
compression at this level fails to retain fine details, causing
objects to merge and reducing the distinctiveness of differ-
ent semantic regions. As the dimensionality increases to
12–24, the quality of segmentation improves considerably.
The reconstructed semantic maps become more structured,
with clearer object outlines and more accurate spatial dis-
tributions. However, some artifacts remain, particularly in
complex scene regions where fine-grained features are nec-
essary for precise segmentation.
At higher embedding dimensions (32–64), the seman-
tic reconstructions closely resemble the original feature

<!-- page 12 -->
Methods
Metrics
Avg.
Room0
Room1
Room2
Office0
Office1
Office2
Office3
Office4
NICE-SLAM
PSNR↑
24.42
22.12
22.47
24.52
29.07
30.34
19.66
22.23
24.94
SSIM↑
0.809
0.689
0.757
0.814
0.874
0.886
0.797
0.801
0.856
LPIPS↓
0.233
0.330
0.271
0.208
0.229
0.181
0.235
0.209
0.198
Co-SLAM
PSNR↑
30.24
27.27
28.45
29.06
34.14
34.87
28.43
28.76
30.91
SSIM↑
0.939
0.910
0.909
0.932
0.961
0.969
0.938
0.941
0.955
LPIPS↓
0.252
0.324
0.294
0.266
0.209
0.196
0.258
0.229
0.236
ESLAM
PSNR↑
29.08
25.32
27.77
29.08
33.71
30.20
28.09
28.77
29.71
SSIM↑
0.929
0.875
0.902
0.932
0.960
0.923
0.943
0.948
0.945
LPIPS↓
0.336
0.313
0.298
0.248
0.184
0.228
0.241
0.196
0.204
SplaTAM
PSNR↑
33.98
32.48
33.72
34.96
38.34
39.04
31.90
29.70
31.68
SSIM↑
0.969
0.975
0.970
0.982
0.982
0.982
0.965
0.950
0.946
LPIPS↓
0.099
0.072
0.096
0.074
0.083
0.093
0.100
0.118
0.155
SGS-SLAM
PSNR↑
34.66
32.50
34.25
35.10
38.54
39.20
32.90
32.05
32.75
SSIM↑
0.973
0.976
0.978
0.981
0.984
0.980
0.967
0.966
0.949
LPIPS↓
0.096
0.070
0.094
0.070
0.086
0.087
0.101
0.115
0.148
Ours
PSNR↑
32.94
27.92
30.89
32.77
36.19
36.99
29.23
31.47
33.57
SSIM↑
0.914
0.8246
0.9028
0.9312
0.9509
0.9409
0.9229
0.9208
0.9367
LPIPS↓
0.148
0.2081
0.1409
0.1190
0.1310
0.1439
0.1685
0.1460
0.1269
Table 3. Comparison of reconstruction quality of SLAM methods on different Replica scenes.
(a) Dim = 3
(b) Dim = 6
(c) Dim = 9
(d) Dim = 12
(e) Dim = 16
(f) Dim = 24
(g) Dim = 32
(h) Dim = 48
(i) Dim = 64
Figure 1. Comparison of PCA embedding compression on the backpack query across different dimensions. Each row represents different
compression levels (3–64), demonstrating how embedding dimensionality affects reconstruction quality and semantic consistency.
space.
Objects maintain well-defined contours, and the
color-coded segmentations demonstrate a high level of se-
mantic accuracy.
The 64-dimensional case provides the
most faithful reconstruction, preserving both large-scale
structure and fine object details.
However, beyond this
point, the benefits of increasing dimensionality further be-
come negligible while computational overhead continues to
grow.
These results confirm that PCA effectively retains essen-
tial semantic information while ensuring computational ef-
ficiency. The selection of 64 as the embedding dimension
achieves a balance between compression quality, segmen-
tation accuracy, and real-time performance, making it the
optimal choice for LEG-SLAM.
