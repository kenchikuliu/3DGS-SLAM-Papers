<!-- page 1 -->
Taking Language Embedded 3D Gaussian Splatting into the Wild
Yuze Wang*
Yue Qi†‡
State Key Laboratory of Virtual Reality Technology and Systems,
School of Computer Science and Engineering, Beihang University
(a) Unconstrained Photo 
Collection Input
(b) Open-Vocabulary Segmentation Results
(c) Applications
Winged Horse
Window
Oceanus
Relief
Corinthian Order
…
Architectural Style Pattern Recognition
“Gothic”
“Baroque”
“Renaissance”
“Neoclassical”
8
2
0
0
Vote
Interactive Roaming with Open Vocabulary Queries
Interactive 
Roaming
Text Query:
“ Oceanus”
Appearance
 Interpolation 
3D Scene Editing
Duplicate
Duplicate
Scale Up
Figure 1: (a) Our method takes an unconstrained photo collection as input. (b) After optimization, our method achieve high-quality
open-vocabulary segmentation. (c) The proposed method supports various applications, including interactive roaming with open-
vocabulary queries, architectural style pattern recognition, and 3D scene editing.
ABSTRACT
Recent advances in leveraging large-scale Internet photo collections
for 3D reconstruction have enabled immersive virtual exploration
of landmarks and historic sites worldwide. However, little atten-
tion has been given to the immersive understanding of architectural
styles and structural knowledge, which remains largely confined to
browsing static text-image pairs. Therefore, can we draw inspira-
tion from 3D in-the-wild reconstruction techniques and use uncon-
strained photo collections to create an immersive approach for un-
derstanding the 3D structure of architectural components? To this
end, we extend language embedded 3D Gaussian splatting (3DGS)
and propose a novel framework for open-vocabulary scene under-
standing from unconstrained photo collections. Specifically, we
first render multiple appearance images from the same viewpoint as
the unconstrained image with the reconstructed radiance field, then
extract multi-appearance CLIP features and two types of language
feature uncertainty maps-transient and appearance uncertainty-
derived from the multi-appearance features to guide the subsequent
optimization process. Next, we propose a transient uncertainty-
aware autoencoder, a multi-appearance language field 3DGS rep-
resentation, and a post-ensemble strategy to effectively compress,
learn, and fuse language features from multiple appearances. Fi-
nally, to quantitatively evaluate our method, we introduce PT-OVS,
a new benchmark dataset for assessing open-vocabulary segmenta-
tion performance on unconstrained photo collections. Experimental
*e-mail: wangyuze1998@buaa.edu.cn
†e-mail: qy@buaa.edu.cn
‡Corresponding Author
results show that our method outperforms existing methods, deliv-
ering accurate open-vocabulary segmentation and enabling applica-
tions such as interactive roaming with open-vocabulary queries, ar-
chitectural style pattern recognition, and 3D scene editing. Visit our
project page at https://yuzewang1998.github.io/takinglangsplatw/.
Index Terms:
3D Gaussian Splatting, Unconstrained Photo Col-
lection, Open-Vocabulary Understanding, Multi-Appearance Lan-
guage Features.
1
INTRODUCTION
Over the past few decades, researchers have used images collected
from the Internet to reconstruct landmarks and historic sites, en-
abling their digital exploration and navigation [59, 60]. However,
when it comes to understanding architectural styles and structural
knowledge, people still primarily rely on static text-image pairs,
lacking immersive visual experiences.
Can we take inspiration
from successful 3D in-the-wild reconstruction methods to achieve
low-cost fine-grained 3D understanding of architectural compo-
nents from Internet-sourced unconstrained photo collections? This
could help users intuitively and immersively understand a build-
ing’s structure, style, and historical context.
Internet-sourced unconstrained photo collections often contain
images captured under diverse conditions—across different years,
lighting environments, camera devices, and viewpoints—and may
include transient occluders such as pedestrians and vehicles moving
through the scene. While the understanding of architectural compo-
nents through unconstrained photo collections is still in its infancy,
recent advances in 3D in-the-wild reconstruction techniques offer
valuable insights.
Some studies have extended neural radiance fields (NeRFs) [40]
and 3D Gaussian splatting (3DGS) [25] to unconstrained photo
collection by predicting transient mask and latent appearance em-
arXiv:2507.19830v2  [cs.GR]  5 Aug 2025

<!-- page 2 -->
bedding for each unconstrained image.
NeRF-based methods
[10, 16, 38, 75] typically condition on appearance embeddings to
generate image-specific radiance fields via a neural network, while
most 3DGS-based approaches [13,30,64,65,72,80] achieve this by
applying learnable affine transformations to the spherical harmonic
(SH) coefficients for each 3D Gaussian. These methods enable real-
time rendering with smooth appearance and illumination changes of
the landmarks and historic sites. On the other hand, some 3DGS-
based methods [49, 51, 57, 81] achieve open-vocabulary scene un-
derstanding using high-quality, well-captured photo collections. By
learning 3D language embedded Gaussians with multi-view CLIP
[52] features, they enable open-vocabulary segmentation. Similar
to in-the-wild 3D scene reconstruction, open-vocabulary scene un-
derstanding from unconstrained photo collections also faces chal-
lenges, such as appearance variations and transient occluders in
each unconstrained image.
However, adapting these 3D open-
vocabulary scene understanding or in-the-wild radiance field recon-
struction methods to 3D understanding from unconstrained photo
collections is not trivial: First, due to the limited primary light
sources in outdoor scenes—typically the sky and the sun—and the
fact that appearance changes follow physical laws, modeling vari-
ous appearances in a latent space is feasible. However, modeling
latent language features (such as CLIP) in a similar manner is ex-
tremely challenging. Changes in language features can be highly
noticeable and irregular due to variations in appearance, scale, oc-
clusions, photo filters, and viewpoints. The left part of Fig. 2 vi-
sualizes how different appearances from the same viewpoint lead
to varying language features, while the right part shows how tran-
sient occluders introduce unexpected language features. Addition-
ally, applying a black-and-white filter, for example, may implic-
itly introduce the semantics of an ”old photograph,” which is no
value to the understanding of architectural components. Secondly,
prior in-the-wild 3D scene reconstruction methods assume that the
radiance of a spatial point under varying appearances is additive,
represented as an affine transformation of base radiance. However,
this assumption does not hold for CLIP features due to their non-
interpretability. Furthermore, the semantics of architectural com-
ponents should remain independent of appearance. For instance,
whether it is day or night, rain or shine, Oceanus remains at the
center of the Trevi Fountain.
In this work, we take language embedded 3D Gaussian splat-
ting from well-captured photo collections to unconstrained ones.
Specifically, we first render multiple appearance images from the
same viewpoint as the unconstrained images, with the reconstructed
in-the-wild radiance field, and extract multi-appearance CLIP fea-
tures for post-ensemble. Surprisingly, we find that fusing these
pseudo-CLIP features with the original CLIP features can enhance
open-vocabulary understanding performance. Additionally, we ex-
tract two types of language feature uncertainty maps—appearance
uncertainty map and transient uncertainty map—from the original
and multi-appearance CLIP features. These maps quantify seman-
tic uncertainty caused by appearance variations and transient oc-
cluders, and are incorporated into language feature compression
and language field reconstruction process. To address storage con-
straints, we propose a transient uncertainty-awared autoencoder to
compress the CLIP feature into a scene-specific lower-dimensional
representation. We further introduce a multi-appearance language
3DGS (MALE-GS) representation to learn the multi-appearance
CLIP features.
Then we propose a post-feature ensemble strat-
egy to fuse the rendered multi-appearance CLIP features, enabling
open-vocabulary queries. Finally, while benchmarks and evalua-
tion schemes exist for well-captured photo collections, none have
been established for outdoor unconstrained scenes. Therefore, we
introduce Photo Tourism open-vocabulary segmentation (PT-OVS)
dataset, a new benchmark dataset derived from the public Photo
Tourism [60] dataset, providing dense annotations for seven scenes,
Visualization of
CLIP Feature
Renderings
or
Unconstrained 
Photo
Figure 2: Left – Different appearances under the same viewpoint lead
to significant variations in extracted CLIP features. Right – Transient
occlusions present in the unconstrained image collection introduce
unexpected CLIP features.
including architectural components and their historical contexts,
such as the ”Rose Window” and ”Last Judgment” in Notre-Dame de
Paris. Experimental results show that our method significantly out-
performs state-of-the-art methods in open-vocabulary scene under-
standing, while enabling applications such as interactive roaming
with open-vocabulary queries, architectural style pattern recogni-
tion, and 3D scene editing.
2
RELATED WORK
2.1
Scene Representation and Radiance Fields
Traditional methods have explored various 3D representations, such
as volumes [47], point clouds [46, 48], meshes [17], and im-
plicit functions [42, 45], across a wide range of computer vision
and graphics applications. In recent years, neural radiance fields
(NeRFs) [40] leveraged multi-layer perceptrons (MLPs) to model
radiance fields, achieving high-fidelity novel view synthesis results.
A growing number of NeRF extensions emerged, targeting various
goals such as enhancing visual effects [5,6,21,67], improving ren-
dering speed [11, 53], and enabling scene editing [66, 68, 74, 78],
among others. 3D Gaussian splatting (3DGS) [25] builded upon
the NeRF concept by representing scenes as a set of 3D anisotropic
Gaussians. This approach enabled real-time rendering and high-
quality novel view synthesis while ensuring fast convergence. Nu-
merous studies further enhanced 3DGS, improving rendering qual-
ity [37, 56, 77], reducing storage requirements [15, 31], optimiz-
ing training processes [20, 27], and exploring applications such as
simultaneous localization and mapping (SLAM) [39, 73], artifical
intelligence generated content (AIGC) [34, 50, 63], and scene un-
derstanding [4,18,49,51,57], among other applications.
2.2
Novel View Synthesis for Unconstrained Photo Col-
lections
Although NeRF and 3DGS performed well with well-captured
photo collections, they faced challenges when applied to uncon-
strained photo collections.
In real-world scenes, such as those
in Internet photo collections [60], challenges arise not only from
transient occluders, like moving pedestrians and vehicles, but also
from varying illumination conditions. NeRF-W [38] was the first
approach to apply NeRF for reconstructing scenes from uncon-
strained photo collections. It employed generative latent optimiza-
tion (GLO) to learn appearance embeddings for each unconstrained
image. To render novel appearances from arbitrary images, Ha-
NeRF [10] and CR-NeRF [75] employed convolutional neural net-
works (CNNs) to extract latent appearance embeddings for each
input image. To address the slow convergence of NeRF, K-Planes
[16] introduced a volumetric NeRF representation that combined
planar factorization with an MLP decoder. Recently, several stud-
ies explored replacing NeRF representations with 3DGS for this
task.
SWAG [13] proposed a learnable hash-grid latent appear-
ance feature representation for appearance modeling. GS-W [80]
introduced intrinsic and dynamic appearance features for each 3D
Gaussian to handle variant appearances. WE-GS [65] introduced a

<!-- page 3 -->
lightweight spatial attention module that simultaneously predicted
appearance embeddings and transient masks for each unconstrained
image. SLS [55] and WildGaussians [30] leveraged DINO [41] fea-
tures extracted from each unconstrained image, which were then
fed into a trainable neural network to predict transient occlud-
ers. Wild-GS [72] introduced hierarchical appearance decompo-
sition and an explicit local appearance modeling strategy. Finally,
Splatfacto-W [71] implemented 3DGS for unconstrained photo col-
lections within the NeRFstudio [62] framework.
While these works have demonstrated promising results in in-
the-wild scene radiance field reconstruction, few have explored the
use of unconstrained photo collections for scene understanding.
2.3
Open-Vocabulary 3D Scene Understanding
Open-vocabulary scene understanding is a fundamental task in
computer graphics and computer vision. Bridging 3D represen-
tations with natural language descriptions is crucial for enabling
various applications, such as visual question answering, semantic
segmentation, and object localization. Leveraging vision-language
models (VLMs) like CLIP [52], methods such as CLIP2Scene
[9] and OpenScene [44] developed point cloud-based language-
embedded representations. With the advancement of NeRF, several
work [26, 29, 36] incorporated dense CLIP features extracted from
multi-view images into NeRF-based scene representations. Very
recently, several work aimed to bridge 3DGS representations with
natural language descriptions to improve convergence efficiency,
including LangSplat [49], LEGaussians [57], GOI [51], GS Group-
ing [76], and Feature 3DGS [81]. Following these work, CLIP-
GS [35] and FastLGS [23] further improved the rendering and con-
vergence efficiency for language-embedded 3DGS by combining
semantic feature grid or 3D coherent self-training.
However, the aforementioned methods mainly focused on indoor
scenes with well-captured photo collections. In contrast, we aim to
extend 3D open-vocabulary scene understanding to outdoor envi-
ronments with unconstrained photo collections, such as Internet-
sourced photo collections of landmarks and historic buildings.
3
PRELIMINARIES AND CHALLENGES
3.1
3DGS in the wild
3DGS [25] represents scenes with millions of 3D anisotropic Gaus-
sians. Each Gaussian Gk(x) is parameterized by its covariance ma-
trix Σk, center position µk, opacity αk, and features fk. In vanilla
3DGS, fk is the spherical harmonic (SH) coefficients ck to model
view-dependent appearance. Given the Jacobian matrix J of the
affine projective transformation and the viewing transformation W,
the 2D covariance matrix Σ′
k is computed as:
Σ′
k = JWΣkW T JT .
(1)
These 3D Gaussians are projected into 2D splats and the color of a
pixel r, denoted as ˜I(r), is computed as follows:
˜I(r) = ∑
i∈N
Tiα′
i fi,
Ti =
i−1
∏
j=1
(1−α′
j).
(2)
The term α′
k represents the final multiplied opacity of αk. The at-
tributes of each 3D Gaussian are optimized using the L1 loss L1
and the structural similarity index (SSIM) [69] loss Lssim between
the rendered image ˜I and the ground truth (GT) image I.
To adapt 3DGS to unconstrained photo collections, some work
[30,64,65,80] predict the appearance embedding li ∈Rda and tran-
sient mask Mi ∈RWt×Ht for each image Ii:
li,Mi = fα(Ii), fβ (Ii).
(3)
Typically, fα(.) and fβ (.) are learnable CNNs. With injecting the
appearance embedding li into each 3D Gaussian, the appearance-
specific radiance of each 3D Gaussian can be obtained with another
learnable neural network fγ:
cik = ck + fγ(ck,µk,li).
(4)
By blending each 3D Gaussian with cik using Eq. (2), an
appearance-specific rendered image ˜Ii is obtained. The additional
supervision is provided by the predicted transient mask Mi:
Lc = L1((1−Mi)⊙˜Ii,(1−Mi)⊙Ii)
+λLSSIM((1−Mi)⊙˜Ii,(1−Mi)⊙Ii),
(5)
where ⊙represents the Hadamard product and λ is hyperparameter.
3.2
Language Embedded 3DGS
Some work [51,58,76,81] achieve open-vocabulary understanding
with 3DGS. They first extract pixel-level CLIP features from each
image:
Fi = fclip(Ii).
(6)
By associating each 3D Gaussian Gk(x) with additional language
features l fk, rendering the feature map ˜Fi using the tile-based ras-
terizer with Eq. (2), and supervising the rendered language feature
map with 2D language feature Fi, a language embedded 3DGS can
be reconstructed:
Llang = ∥˜Fi −Fi∥.
(7)
However, these language embedded 3DGS methods struggle with
unconstrained photo collections due to inconsistencies in multi-
view CLIP features caused by varying appearances and transient
occlusions. Moreover, the non-additive nature of CLIP features
prevents the direct adaptation from in-the-wild radiance field re-
construction methods to in-the-wild language field reconstruction.
4
METHOD
Given a set of posed unconstrained images I = {I1,I2,...IT } and a
3D Gaussian radiance field GRF of the scene constructed by in-the-
wild 3DGS method, such as WE-GS [65]. Our method expands
GRF with open-vocabulary semantics, enabling text-based queries.
Fig. 3 depicts the framework of our proposed method.
4.1
Multi-Appearance Pixel-Level Language Feature Ex-
traction
Prior methods typically extract pixel-level CLIP features from im-
ages using either hierarchical SAM-based methods [49] or hierar-
chical cropped-image patch-based methods [26, 57]. Due to the
scale variations in unconstrained images, we adopt a similar SAM-
based approach, first applying SAM [28] to segment the image into
distinct regions. Each segmented region is then cropped and pro-
cessed through the CLIP image encoder to extract its correspond-
ing CLIP features. Following LangSplat [49], we extract features at
three semantic levels: subpart, part, and whole. To mitigate the in-
consistency in illumination across unconstrained images, we gener-
ate multi-appearance CLIP features by first pre-rendering with mul-
tiple appearances and then encoding them with CLIP, as follows:
˜I j
i = frender(Ii, fα(Ij),GRF),
(8)
and
F j
i = fclip(˜I j
i ),
(9)
where ˜I j
i represents the rendered image at the camera pose of Ii,
with its appearance conditioned on Ij. In total, we render N −1
novel appearance images, along with a self-appearance rendering
(i = j). We denote the set of multi-appearance pixel-level CLIP

<!-- page 4 -->
(a)  Multi-appearance Pixel-level Language Feature Extraction
Reconstructed
Radiance Field
Multi-Appearance
Renderings
Multi-Appearance
CLIP features
Pixel-Level 
CLIP Enc.
Pixel-Level 
CLIP Enc.
Transient 
Uncertainty Map
Appearance 
Uncertainty Map
Unconstrained Image
CLIP features
(b)  Transient Uncertainty-Aware Autoencoder
Sampling 
Probability
(c) Multi-appearance Language Embedded 3DGS
Sampling 
Probability
Transient 
Uncertainly Map
Transient and 
Appearance 
Uncertainly Map
Multi-
Appearance
CLIP features
MALE-GS
Reconstructed
MALE-GS
“ Triangular
Pediment”
Pixel-Level 
CLIP Enc.
Multi-Appearance
Score Maps
Background
Filter
(d) Post Ensemble and Open-Vocabulary Querying
“ sky”
“ background”
Open-Vocabulary 
Query Text
Open-Vocabulary 
Segmentation Result
Unconstrained 
Photo Collection
𝓛𝑎𝑒
Encoder
Decoder
Fixed-Vocabulary
Query Texts
Operation 
Flow
Condition
Flow
Optimizable
Frozen
3D Radiance
Field
3D Language
Field
…
Feature
Ensemble
Figure 3: Overview of our framework. (a) Given an image from an unconstrained photo collection, multi-appearance renderings are generated
using the reconstructed radiance field. CLIP features are then extracted from both the unconstrained image and renderings using a pixel-
level CLIP encoder. The language feature appearance and transient uncertainty map are derived from the extracted features. (b) A transient
uncertainty-aware autoencoder is learned to compresses and restores the dense CLIP features. (c) With the MALE-GS representation, multi-
appearance CLIP features are propagated into the 3D field, with uncertainty maps serving as optimization constraints. (d) Post-ensemble and
background filter modules ensure high-quality open-vocabulary segmentation results.
features for Ii as: {F1
i ,...,FN−1
i
,Fi
i }. We propose two constraints
for selecting N −1 novel appearances for each scene. The first is
the rendering quality constraint, which requires that the L1 distance
between the rendered image and the source image of the selected
appearance be lower than εq:
∥frender(Ij, fα(Ij),GRF)−Ij∥< εq.
(10)
This ensures the quality of the rendered image, further maintain-
ing the reliability of the CLIP features.
Additionally, to avoid
overly similar appearances, we introduce an appearance embed-
ding distance constraint, which requires the Manhattan distance
Manhat(.,.) between the appearance embeddings of any two se-
lected novel appearance images, Ij and Ik, to exceed εd:
Manhat( fα(Ij), fα(Ik)) > εd.
(11)
Additionally, we introduce two types of language feature uncer-
tainty: appearance uncertainty and transient uncertainty. The ap-
pearance uncertainty map quantifies the variation in the CLIP fea-
tures across multiple appearances:
UA
i = 1
N (
N−1
∑
j=1
(F j
i −¯Fi)2 +(Fi
i −¯Fi)2),
(12)
where ¯Fi represents the mean of the multi-appearance CLIP fea-
tures. The transient uncertainty map is designed to measure the
likelihood of transient occluders at language feature level:
UT
i = (Fi
i −Fi)2,
(13)
where Fi
i represents the language feature extracted from the self-
appearance rendering, and Fi is the language feature extracted from
the original unconstrained image. Note that Fi
i is only used for
uncertainty calculation and will not be used in the later process.
4.2
Transient Uncertainty-Aware Autoencoder
Due to the significant demands of storing high-dimensional lan-
guage features for each 3D Gaussian, we pre-train an autoencoder
[19] to compress the CLIP features into a lower-dimensional rep-
resentation. These compressed features are then learned by the 3D
language field, and decoded back to high-dimensional space dur-
ing the evaluation stage. However, the proposed multi-appearance
CLIP features introduce N −1 times feature maps to compress.
Training the autoencoders on all the extracted CLIP features across
different appearances significantly increases training time. More-
over, we observe that fusing the features during the preprocessing
(e.g., by averaging CLIP features across appearances) before train-
ing the autoencoder leads to a notable performance drop (as shown
in Tab. 5). We attribute this degradation to the expanded domain of
definition of scene-specific CLIP features.
From our experiments, we identified a simple yet effective
method: using only the CLIP features extracted from the original
unconstrained images for autoencoder training. This strategy re-
sults in better segmentation performance. We believe this approach
helps narrow the domain gap between the decompressed language
feature maps of the rendered images and the original images. After
compressing and decompressing the language features of the ren-
dered image, they can be more closely aligned with the features
extracted from the original image.
Besides, we introduce a transient uncertainty-aware optimization
strategy when training of the autoencoder:
Lae = ΣT
t=1∥fD( fE(Ft ⊙(1−UT
t )))−Ft ⊙(1−UT
t )∥,
(14)

<!-- page 5 -->
Table 1: Quantitative experimental results on the proposed PT-OVS dataset, with the first, second, and third values highlighted in red, orange,
and yellow, respectively. Our method demonstrates superior overall performance compared to state-of-the-art approaches.
Brandenburg Gate
Trevi Fountain
Todaiji Temple
Pantheon
Taj Mahal
Buckingham Palace
Notre-Dame de Paris
mIoU↑
mPA↑
mP↑
mIoU↑
mPA↑
mP↑
mIoU↑
mPA↑
mP↑
mIoU↑
mPA↑
mP↑
mIoU↑
mPA↑
mP↑
mIoU↑
mPA↑
mP↑
mIoU↑
mPA↑
mP↑
LEGaussians [57]
0.158
0.713
0.161
OOM
OOM
OOM
0.043
0.402
0.043
0.356
0.644
0.356
0.112
0.839
0.114
0.166
0.476
0.169
OOM
OOM
OOM
Feature 3DGS [81]
0.028
0.748
0.029
OOM
OOM
OOM
0.042
0.648
0.046
0.078
0.704
0.105
0.069
0.442
0.073
0.071
0.386
0.071
OOM
OOM
OOM
GS Grouping [76]
0.456
0.961
0.539
0.010
0.583
0.010
0.052
0.906
0.053
0.009
0.611
0.009
0.107
0.818
0.107
0.105
0.107
0.105
0.116
0.624
0.116
LangSplat [49]
0.275
0.787
0.331
0.539
0.944
0.649
0.203
0.704
0.217
0.801
0.971
0.898
0.445
0.911
0.491
0.297
0.798
0.459
0.449
0.813
0.481
Ours
0.619
0.985
0.846
0.593
0.969
0.757
0.321
0.940
0.381
0.928
0.990
0.977
0.613
0.967
0.693
0.540
0.925
0.668
0.736
0.926
0.892
where T is the number of unconstrained images, an encoder fE
maps the D-dimensional CLIP features Ft to a C-dimensional rep-
resentation. (D is 512 in vanilla CLIP, and we set C to 3 in our
experiments.) Additionally, to further accelerate autoencoder train-
ing, we set a condition where pixels with transient uncertainty val-
ues larger than τu(set to 0.9 in our implementation) are assumed to
be transient occluders and excluded from optimization.
4.3
Multi-Appearance Language Embedded 3DGS
After training the autoencoder, we transform all multi-appearance
CLIP features {F1t ,...,FN−1
t
,Ft} into scene-specific compressed
multi-appearance language features {H1t ,...,HN−1
t
,Ht}. To learn
the multi-appearance language feature field, we propose the multi-
appearance language 3DGS (MALE-GS) representation. Each 3D
Gaussian is assigned N language features {l f1,...,l fN−1,l f} to
learn N −1 novel appearances and a self-appearance language fea-
tures. In our experiments, we set N = 4. To maintain rendering
efficiency, we also adopt a tile-based rasterizer with Eq. (2) to ren-
der multi-appearance language features.
We optimize the 3D language field using the following transient-
aware and appearance-aware language feature distance loss:
Llang =
T
∑
t=1
(∥Ht ⊙(1−UA
t )⊙(1−UT
t )−˜Ht ⊙(1−UA
t )⊙(1−UT
t )∥+
N−1
∑
n=1
∥Hn
t ⊙(1−UA
t )⊙(1−UT
t )−˜Hn
t ⊙(1−UA
t )⊙(1−UT
t )∥).
(15)
Here, the transient uncertainty map prevents transient occluders
from contaminating the 3D Gaussian language field, while the ap-
pearance uncertainty map suppresses gradient updates for 3D Gaus-
sians with significant language feature ambiguity across different
appearances. With other parameters of each MALE-GS fixed (e.g.,
position µ, opacity α), we only optimize the multi-appearance lan-
guage features {l f1,...,l fN−1,l f}. For more details, please refer to
the supplementary material.
4.4
Post Ensemble and Open-Vocabulary Querying
After optimizing MALE-GS, open-vocabulary 3D queries can be
performed with post-feature ensemble. Given a specific viewpoint,
the compressed multi-appearance language features are rendered.
Then, using the trained decoder, we decode these compressed lan-
guage features back into high-dimensional multi-appearance CLIP
features { ˜F1,..., ˜FN−1, ˜F}. For readability, we denote ˜F as ˜FN.
Given a text query Qq
text, we first compute N relevancy score map
for each feature map with the following equation:
scorei
q = min j
exp(Qq
text · ˜Fi)
exp(Qq
text · ˜Fi)+exp(Q j
canon · ˜Fi)
.
(16)
These score maps represent how much closer the rendered em-
bedding is towards the query embedding compared to the canon-
ical embeddings.
All renderings use the same canonical texts
{Q j
canon| j = 1,2,...,J}, such as ”things” and ”stuff”. We find with
such strategy, the background, especially the sky, will get a high
score. So, we propose a backgroud filter to additional exclude the
background, with the query texts ”sky” and ”background”, and set
the canonical texts as the actually query text Qq
text. We use 1 minus
its output as an additional score for subsequent fusion, referring to
this as scoreBq.
We find that the maximum value in the score map is positively
correlated with the segmentation accuracy. Therefore, we use the
maximum value of each score map as the weight to perform a
weighted fusion of the multi-appearance score maps:
scoreq =
N
∑
i=1
max(scoreiq ·scoreBq)
∑N
j=1 max(scorej
q ·scoreBq)
·scorei
q ·scoreB
q.
(17)
For open-vocabulary segmentation, we filter out pixels with fused
relevancy scores lower than a predefined threshold τ (τ is 0.4 in our
experiments).
4.5
The PT-OVS Benchmark
To evaluate our method, we require unconstrained photo col-
lections paired with ground-truth open-vocabulary segmentation
maps. However, to the best of our knowledge, no such dataset cur-
rently exists. Therefore, we introduce a new benchmark, PT-OVS,
assembled from the Photo Tourism [60] dataset. We selected seven
scenes covering landmarks from various countries, architectural
styles, and historical contexts. NeRF-W [38] introduced an uncon-
strained photo dataset for the Brandenburg Gate and Trevi Fountain
to support in-the-wild radiance field reconstruction. They filtered
out low-quality and heavily occluded images using NIMA [61]
and DeepLab V3 [8]. Following a similar approach, we curated a
dataset for five additional scenes: Buckingham Palace, Notre-Dame
de Paris, Pantheon, Taj Mahal, and Todaiji Temple. We associate
these landmarks with open-vocabulary descriptions, such as ”Iron
Cross” and ”Bronze Sculpture” for Brandenburg Gate, and ”Rose
Window” and ”Last Judgment” for Notre-Dame de Paris. We la-
beled 10–20 ground-truth segmentation maps for each scene based
on selected high-quality unoccluded photos. More details about the
proposed benchmark can be found in the supplementary material.
5
EXPERIMENTS
5.1
Implementation Details
We use the SAM ViT-H model [14] for segmentation and the Open-
CLIP ViT-B/16 model [12] for CLIP feature extraction from each
unconstrained image. We implement our method in Python using
the PyTorch framework [43], integrating custom CUDA accelera-
tion kernels based on the differentiable Gaussian rasterization pro-
posed by 3DGS [25]. We train the uncertainty-aware autoencoder
for 100 epochs with a learning rate of 0.0001 and train the MALE-
GS for 30,000 iterations with a learning rate 0.0025. All experi-
ments are run on a single NVIDIA RTX-4090 GPU. For additional
implementation details, please refer to the supplementary material.
5.2
Baselines and Metrics
We compare our approach to state-of-the-art 3DGS-based open-
vocabulary scene understanding methods, including LangSplat
[49], Feature 3DGS [81], GS Grouping [76], and LEGaussians [57].
These methods focus on reconstructing language embedded 3DGS
for static scenes from well-captured photo collections. We make

<!-- page 6 -->
RGB
Ours
LangSplat
LEGaussians
GS Grouping
GT
Ours
LangSplat
LEGaussians
GS Grouping
GT
3D Scene: Todaiji Temple
“Eval”
“Door”
“Column”
Empty
Empty
3D Scene: Brandenburg Gate
“Column”
“Iron Cross”
“Bronze Sculpture”
Empty
Empty
Empty
3D Scene: Trevi Fountain
“Oceanus”
“Winged Horses”
“Reliefs”
OOM
OOM
OOM
Empty
Empty
3D Scene: Notre-Dame de Pairs
“Rose Window”
“Last Judgment”
Empty
Empty
OOM
OOM
3D Scene: Pantheon
“Obelisk of the Pantheon” “Triangular Pediment”
Empty
3D Scene: Taj Mahal
“Onion Dome”
“Arch-Shaped Doorways”
Empty
RGB
3D Scene: Buckingham Palace
“Ionic Column”
Empty
Empty
“Queen Victoria Memorial”
Figure 4: Qualitative experimental results on the proposed PT-OVS dataset, comparing our method with state-of-the-art approaches.
minimal adjustments to adapt them to unconstrained images with
varying resolutions. We quantitatively evaluate segmentation qual-
ity using mean intersection over union (mIoU), mean pixel accuracy
(mPA), and mean precision (mP), comparing the predicted segmen-
tation masks with the ground truth.
5.3
Comparisons
5.3.1
Quantitative Comparison
The quantitative results are shown in Tab. 1. Note that since all
these methods use CLIP to extract language features from multi-
view images, their level of ”understanding of architectural knowl-
edge” remains consistent.
The differences in segmentation ac-
curacy arise from their ability to robustly reconstruct the lan-
guage embedded 3DGS. Specifically, compared to the second-best
method, LangSplat, our approach achieves an average improvement
of 19.1% in mIoU across seven scenes.
LEGaussians and Fea-
ture 3DGS encountered out-of-memory (OOM) issues in the Trevi
Fountain and Notre-Dame de Paris scenes, preventing them from
reconstructing the language embedded 3DGS. These two scenes
each contain approximately 1,700 and 3,000 unconstrained images,
respectively. Due to the varying scales of these images, the recon-
structed scenes are rich in detail and contain a larger number of 3D
Gaussians. In Feature 3DGS, each 3D Gaussian explicitly stores an
uncompressed 512-dimensional CLIP features, making it impossi-
ble to load all 3D Gaussians into memory for optimization in these
scenes. For LEGaussians, the dense quantization of language fea-
tures for a large number of unconstrained images, along with addi-
tional learnable attributes per 3D Gaussian, also leads to OOM.
As shown in Tab. 2, we compare the efficiency of our pro-
posed method with state-of-the-art approaches. The training time

<!-- page 7 -->
encompasses the total duration of radiance field training, autoen-
coder training, and language field training. In terms of storage, our
method introduces only an additional 40.6MB overhead compared
to LangSplat. Furthermore, our method achieves state-of-the-art
performance in both inference speed and accuracy. Overall, our ap-
proach achieves a better balance between efficiency and accuracy.
Table 2: Comparison about the efficiency on the proposed PT-OVS
dataset [24]. Metrices are averaged over 5 scenes (except for Notre-
Dame de Paris and Trevi Fountain). ”TT”, ”IT”, and ”ST” are ”training
time ”, ”inference time”, and ”storage”, respectively.
TT (h)↓
IT (s) ↓
ST (MB) ↓
mIoU (%) ↑
LEGaussians [57]
3.15
0.18
362.8
16.7
Feature 3DGS [81]
3.92
2.10
709.7
5.8
GS Grouping [76]
0.81
2.42
134.6
14.6
LangSplat [49]
2.23
0.17
74.4
40.4
WE-GS (Ours)
3.08
0.17
115.0
60.4
5.3.2
Qualitative Comparison
In Fig. 4, we visualize the segmentation results of both the base-
line methods and our approach. Taking Brandenburg Gate as an
example, our method accurately identifies architectural components
(e.g., ”Column”), components materials (e.g., ”Bronze Sculpture”),
and even the background knowledge associated with the architec-
ture (e.g., ”Iron Cross”). Other baseline methods directly extract
pixel-level CLIP features from images captured under varying ap-
pearances and do not account for the exclusion of occluders. As a
result, the extracted multi-view CLIP features lack 3D consistency,
making accurate 3D language field reconstruction difficult.
To further demonstrate the generalization ability of our method,
we conduct experiments on two scenes from the NeRF-on-the-go
dataset [54]. As illustrated in Fig. 5, our method successfully re-
constructs language-embedded 3DGS in both indoor and outdoor
natural scenes with unconstrained photo collections.
Unconstrained Photo 
Collection Input
Segmentation 
Results View 1
Segmentation 
Results View 2
Segmentation 
Results View 3
basketball
flowers
yellow mug
green watering can
bicycle
tree
cobblestones
Figure 5: Qualitative experimental results on the NeRF-on-the-go
dataset [54], demonstrating the generalization ability of our method.
5.4
Ablation Study
We validate the design choices of the proposed method on all scenes
of the PT-OVS dataset. Tab. 3, Tab. 4, and Tab. 5 present the quanti-
tative results, while Fig. 6 and Fig. 7 present the qualitative results.
5.4.1
The Influence of Multi-Appearance Enhancement
Row 1 of Tab. 3 shows that introducing multiple appearances en-
hancement improves segmentation accuracy. Fig. 6 visualizes the
multiple appearance renderings across different scenes. Addition-
ally, we conducted an ablation study to investigate whether the
choice of novel appearances affects segmentation accuracy.
As
shown in Tab. 4, we first perform a baseline experiment (row 1),
Table 3: Ablation studies of our method. Metrics are averaged over 7
scenes in the proposed PT-OVS dataset.
mIoU↑
mPA↑
mP↑
Preprocess
(1) w/o Multi-Appearance Enh.
0.481
0.877
0.570
(2) w/o Post Ensemble
0.502
0.890
0.613
(3) w/o Uncertainly-aware Lae
0.529
0.906
0.630
(4) τu = 0.85
0.613
0.953
0.742
(5) τu = 0.95
0.620
0.956
0.745
Language Field
Learning
(6) w/o TUM & AUM
0.612
0.955
0.733
(7) w/o TUM
0.620
0.957
0.743
(8) w/o AUM
0.612
0.954
0.735
Evaluation
(9) w/o Bkg. filter
0.608
0.952
0.731
(10) ImgLvlMax. Ens.
0.590
0.952
0.698
(11) PixMax. Ens.
0.564
0.949
0.650
(12) PixAvg. Ens.
0.621
0.956
0.651
(13) PixWeightedAvg. Ens.
0.563
0.949
0.648
Ours
(14) Completed Model
0.621
0.957
0.745
Table 4: Ablation studies on the choice of novel appearance uncon-
strained images selection. Metrics are computed 5 times with dif-
ferent random seeds, reporting the average and fluctuation over 7
scenes in the proposed PT-OVS dataset.
mIoU↑
mPA↑
mP↑
Novel App.
Selection
(1) w/ same App.
0.485±0.03
0.877±0.02
0.574±0.02
(2) w/o εd&εq
0.561±0.13
0.947±0.07
0.652±0.11
(3) w/o εd
0.620±0.09
0.958±0.06
0.745±0.13
(4) w/o εq
0.563±0.17
0.947±0.12
0.652±0.17
Number of
Novel App.
(5) N=2
0.607±0.08
0.951±0.08
0.731±0.04
(6) N=3
0.611±0.05
0.953±0.05
0.732±0.04
(7) N=5
0.628±0.04
0.961±0.03
0.748±0.03
(8) N=6
0.628±0.04
0.960±0.03
0.749±0.02
Ours
(9) N=4
0.626±0.05
0.960±0.03
0.749±0.04
where the three novel appearance unconstrained images are all
set to the original unconstrained image. The segmentation accu-
racy in this setting is similar to that of the method without multi-
appearance enhancement (row 1 of Tab. 3). In rows 2-4 of Tab. 4,
we relaxed the constraints on the selection of novel appearance un-
constrained images. It can be observed that the rendering quality
constraint εq for novel appearance rendering selection is particu-
larly important. This is because in-the-wild scene radiance field
reconstruction methods may fail to render photo-realistic images
under certain appearances. When such images are used to extract
CLIP features, they exhibit a significant domain gap compared to
the features extracted from the original image. Additionally, the
appearance embedding distance constraint εd is also crucial, as it
prevents the appearance from becoming too similar, which would
otherwise reduce performance back to the baseline (row 1). Row
2 of Tab. 3 shows that performing multi-appearance fusion before
training the autoencoder significantly reduces segmentation accu-
racy. Finally, we investigated the optimal number of novel appear-
ances (Row 5-8 of Tab. 4). We found that setting N to 4 is the
most reasonable choice. Although setting N to 5 or 6 yields slightly
better segmentation accuracy, it introduces additional storage over-
head.
5.4.2
The Influence of Using Feature Uncertainty Map
Rows 3 and 6-8 of Tab. 3 quantitatively demonstrate the important
role of the feature uncertainty map in both the preprocessing and
language field learning stages. Transient uncertainty map (TUM)
effectively prevents the autoencoder from learning unnecessary fea-
tures, such as those of transient occluders. In the language field
learning stage, the introduction of the appearance uncertainty map
(AUM) plays a more significant role in improving segmentation ac-
curacy, as shown in rows 3 and 6-8 of Tab. 3. As shown in Rows
4–5 of Tab. 3, a lower τu results in degraded performance. While
a larger τu yields comparable performance, it increases the training
time of the uncertainty-aware autoencoder. Rows 6-7 of Fig. 6 visu-
alize the appearance uncertainty map and the transient uncertainty

<!-- page 8 -->
Transient 
Uncertainty Map
Appearance
Uncertainty Map
Novel App.
 Rendering 1
Novel App.
 Rendering 2
Novel App.
 Rendering 3
Self App.
 Rendering
Unconstrained
 Image
Trevi Fountain
Notre-Dame de Pairs
Todaiji Temple
Pantheon
Figure 6: Visualization of self and novel appearance renderings,
along with the appearance and transient uncertainty maps across
different scenes. In the uncertainty maps, brighter colors represent
higher uncertainty for each pixel.
map across different scenes. It can be observed that the transient
uncertainty maps accurately represent occluders in unconstrained
images, preventing the corresponding language features from be-
ing optimized during the autoencoder and language field learning
processes. Meanwhile, the appearance uncertainty map identifies
regions where language feature vary significantly across different
appearances, enabling smoother optimization during training.
5.4.3
Analyze of the Post-Ensemble Strategy
We designed several ablations to ensemble the multiple appear-
ance language score maps. Rows 9–13 of Tab. 3 demonstrate the
impact of different ensemble functions on segmentation accuracy.
”ImgLvlMax. Ens.” (row 8) indicates selecting the score map with
the highest maximum value from the score maps as the final en-
semble result. ”PixMax. Ens.” (row 9) and ”PixAvg. Ens.” (row
10) refer to ensemble functions that take the maximum and average
values, respectively, at the pixel level from different score maps.
”PixWeightedAvg. Ens.” (row 11) is the method most similar to
ours. It also employ a weighted average fusion approach, but it op-
erates at the pixel level. We find that this approach can introduce
more high-frequency noise into the final results, which decrease
segmentation accuracy.
5.4.4
The Impact of the Autoencoder’s Training Set
As shown in Tab. 5, incorporating CLIP features from rendered
novel appearance images into the autoencoder’s training set not
only significantly increases training time (by approximately 3.5
hours) but also reduces segmentation accuracy (by about 1%
Table 5: Ablation studies on the training autoencoder with novel ap-
pearance language features. ”ATT” is ”autoencoder training time”.
Metrics are averaged over 7 scenes in the proposed PT-OVS dataset.
ATT (h) ↓
mIoU↑
mPA↑
mP↑
w/
Novel App. Features
4.62
0.612
0.953
0.733
w/
Avg. Novel App. Features
9.83
0.502
0.890
0.613
w/o Novel App. Features (Ours)
1.17
0.621
0.957
0.745
mIoU). We attribute this decline to the expanded domain of defini-
tion of scene-specific CLIP features, as the training set size roughly
quadruples. The experimental setting in row 2 of Tab. 5 mirrors
that in row 2 of Tab. 3. Integrating multi-appearance features prior
to autoencoder training notably increases computation, as the fea-
ture granularity shifts from superpixel- to pixel-level. To alleviate
this, we apply similarity-based clustering to the fused pixel-level
features and average within clusters, reducing the training load in
this experiment.
5.4.5
The Influence of the Quality of the Reconstructed Ra-
diance Field
Does the quality of reconstructed radiance field affect scene under-
standing accuracy? To investigate this, we designed several base-
line experiments: (a) WE-GS + Ours: This is the method uses in
our method. We reconstruct the scene using WE-GS [65], render
novel appearances, and perform scene understanding based on the
radiance fields reconstructed by WE-GS. (b) 3DGS† + Ours: To
investigate the impact of 3D geometric priors on scene understand-
ing accuracy, we use vanilla 3DGS [25] for scene reconstruction
and WE-GS for novel appearance rendering. This baseline allows
us to assess how the initialization of the spatial point of 3D Gaus-
sians affects the scene understanding accuracy. (c) GS-W + Ours:
This method uses GS-W [80] for scene reconstruction and novel
appearance rendering, followed by scene understanding based on
the radiance field reconstructed by GS-W. This baseline helps us to
evaluate the impact of different in-the-wild reconstruction methods
on scene understanding accuracy.
As shown in Fig. 7, vanilla 3DGS, which is used for static scene
reconstruction with well-captured photo collections, produces a
noisy point cloud with a large number of points (433.1K). This
negatively impacts segmentation accuracy, resulting in a 9.8% de-
crease in the mIoU metric. On the other hand, although the GS-W
+ Ours method yields slightly lower-quality radiance field in-the-
wild reconstruction compared to WE-GS + Ours (with a PSNR dif-
ference of 2.8dB on the test set), the scene understanding perfor-
mance remains comparable. This demonstrates the robustness of
our method.
5.5
Application
We present three applications:
interactive roaming with open-
vocabulary queries, architectural style pattern recognition, and 3D
segmentation and scene editing.
5.5.1
Interactive Roaming with Open Vocabulary Queries
The proposed method can seamlessly integrate with in-the-wild
radiance field reconstruction approaches (e.g., WE-GS [65], GS-
W [80]). We develop an interactive system for users to remotely ex-
plore landmarks like the Trevi Fountain, with free-viewpoint roam-
ing, lighting variations, and open-vocabulary queries. We encour-
age readers to review the video results provided in the supplemen-
tary materials for additional details.
5.5.2
Architectural Style Pattern Recognition
As shown in Fig. 8, our method is also effective for architectural
style pattern recognition. Given a reconstructed language embed-
ded 3DGS with our method, we query architectural style keywords

<!-- page 9 -->
GT
Rendering
Result
Scene Info.
Segmentation
Result
Heatmap
WE-GS + Ours
3DGS†+ Ours
GS-W+ Ours
193.6K
  61.4%
  24.3dB
Nptr:
mIoU:
PSNR:
122.3K
  52.1%
  21.0dB
Nptr:
mIoU:
PSNR:
433.1K
 61.9%
  27.1dB
Nptr:
mIoU:
PSNR:
Figure 7: The ablation studies on the quality of reconstructed radi-
ance field. WE-GS + Ours refers to our method. 3DGS† + Ours uses
WE-GS [65] for novel appearance rendering while initializing the lan-
guage field with vanilla 3DGS. GS-W + Ours employs GS-W [80] for
both novel appearance rendering and language field initialization. In
the Scene Info., Nptr denotes the number of initialized 3D Gaussians
in Brandenburg Gate scene, while mIoU and PSNR indicate the av-
erage mIoU on the PT-OVS test set and the average PSNR on the
PT [60] test set, respectively—both measured on the same scene.
Unconstrained 
Photo Collection
Vote Result
“Neoclassical”
“Georgian”
“Victorian Gothic”
“Palladian”
 “Edwardian Baroque”
9
2
1
0
0
…
…
Heatmaps for Textual 
Query
Figure 8: Application of architectural style pattern recognition.
(e.g., ”Neoclassical”, ”Georgian”, ”Palladian”) across multiple un-
constrained images, generating multiple fused score maps. We take
a ”winner takes all” approach, selecting the highest value from dif-
ferent style score maps to vote for the corresponding style. Then,
the predicted architectural style is determined by the most fre-
quent vote. Compared to vanilla CLIP, our 3D language field-based
method achieves more accurate recognition. For additional results,
please refer to the supplementary materials.
Additionally, we make an interesting discovery: the intermedi-
ate results of architectural style pattern recognition, specifically the
heatmap of the fused score map, can be viewed as a visualization
of ”what makes Buckingham Palace a neoclassical building.”
5.5.3
3D Segmentation and Scene Editing
Our method can further be applied to 3D segmentation and scene
editing.
After each 3D Gaussian has encoded the compressed language
features, we decode them back into vanilla CLIP features at the 3D
level. Using the same fusion approach as described in the method
section, we perform open-vocabulary segmentation at the 3D Gaus-
sian level instead of the 2D pixel level. 3D open-vocabulary seg-
mentation has numerous applications in AR/VR, and we demon-
strate the application of 3D scene editing.
As shown in Fig. 9, we query and edit ”Oceanus” and ”Winged
Horse” in the scene of Trevi Fountain. Since our method can inte-
grate seamlessly into the 3DGS editing and rendering pipeline, we
use the open-source SuperSplat project [2] to scale, duplicate, and
manipulate the selected 3D Gaussians, creating novel 3D scenes.
Please refer to the supplemental materials for more details.
View 1
View 2
Original 
Scene
Renderings
Query&
Segment
“Oceanus”
Query&
Duplicate
“Winged Horse”
Scale Up
“Oceanus”
Figure 9: Application of 3D segmentation and scene editing.
6
LIMITATIONS
While the proposed method outperforms previous approaches, it
still has limitations. Firstly, our method builds on CLIP features
to link text and images. However, as illustrated in Fig. 10, CLIP
fails to handle certain long-tail vocabulary effectively. This issue
could be mitigated by fine-tuning CLIP on more diverse, long-tail-
oriented datasets or replacing it with stronger vision-language mod-
els such as BLIP [33] or SigLIP [79]. Secondly, although the multi-
ple appearance enhancement strategy effectively handles 3D open-
vocabulary understanding from unconstrained photo collections, it
introduces additional parameters that need to be learned.
Original Image
Predicted Fail Case
Ground Truth
Heatmap
Query：“ Papal coat of arms”
Figure 10: Failure Case. CLIP performs poorly on certain long-tail
vocabulary, leading to failure in our method.
7
CONCLUSION
In this work, we take language embedded 3D Gaussian splatting
into the wild.
We present a framework for reconstructing lan-
guage fields from unconstrained photo collections using 3DGS. Our
framework incorporates several key components. First, we pro-
pose a multi-appearance CLIP feature enhancement strategy for
denoising and augmenting unconstrained language features. Us-
ing the reconstructed radiance field, we render and extract mul-

<!-- page 10 -->
tiple appearance language features from the same viewpoint and
extract both the language feature transient uncertainty and appear-
ance uncertainty map from them. Next, we introduce a transient
uncertainty-aware autoencoder, a multi-appearance language field
3DGS representation, and a post-ensemble strategy to compress,
learn, and fuse multiple appearance language features. Addition-
ally, we introduce PT-OVS, a new benchmark dataset to evaluate
open-vocabulary segmentation accuracy based on text queries for
building components and historical contexts. This paper enables
the creation of semantically rich 3D content for AR/VR/MR from
unconstrained photo collections, providing new insights into inter-
active querying, annotation, and the generation and editing of 3D
content in AR/VR/MR environments.
SUPPLEMENTAL MATERIALS
In this supplementary material, we provide an in-depth explanation
of our method, detailing additional the PT-OVS dataset in Sec. A,
detailing implementation details in Sec. B, and demonstrating more
experiment details and results in Sec. C.
A
MORE DETAILS OF THE PT-OVS DATASET
Figure A: Some filtered unconstrained images, which are typically of
low quality or severely occluded.
Table A: Number of training images, evaluation images, and query
texts per PT-OVS scene.
Train Img.
Eval. Img.
textual queries
Brandenburg Gate
763
16
Column, Iron Cross, Bronze Sculpture
Trevi Fountain
1689
27
Column, Oceanus, Winged Horses, Window, Reliefs
Buckingham Palace
1379
11
Queen Victoria Memorial, Ionic Column, Pediment
Notre-Dame de Paris
2934
15
Rose Window, Last Judgment
Pantheon
1308
11
Triangular Pediment, Obelisk of the Pantheon, Corinthian Column
Taj Mahal
1238
14
Onion Dome, Arch-shaped doorways, Jali Windows
Todaiji Temple
859
8
Eave, Door, Column
We select and download scenes from the train and validation set
of the Image Matching Challenge Photo Tourism (IMC-PT) 2020
dataset [1]. These scenes contain unconstrained images with pro-
vided relatively accurate camera intrinsic and extrinsic parameters,
as well as initial point clouds. We aim to select landmarks from dif-
ferent locations, styles, and types. In total, we choose seven scenes:
Buckingham Palace, Notre-Dame de Paris, Pantheon, Taj Mahal,
Brandenburg Gate, Todaiji Temple, and Trevi Fountain. First, we
need to filter the unconstrained photo collections by removing low-
quality images, such as those with severe occlusions or excessively
low resolution. NeRF-W [38] has already provided filtered results
for Brandenburg Gate and Trevi Fountain. For these two scenes,
we follow the same selection as NeRF-W. For the other five scenes,
we apply the following filtering criteria. First, we apply the Neu-
ral Image Assessment (NIMA) [61] to perform an initial filtering
of the unconstrained photo collections. Specifically, we remove un-
constrained images with a NIMA score lower than 4. We further
remove images in which transient objects occupy more than 80%
of the image area, as determined by a LSeg [32] model trained on
the ADE20K dataset. Fig. A shows examples of unconstrained im-
ages that were filtered out, from the scenes of Pantheon and Taj
Mahal.
“Column” “Bronze 
Sculpture”
“Column”
“Iron Cross”
“Ironic 
Column”
“Queen Victoria 
Memorial”
“Queen Victoria 
Memorial”
“Iron Cross”
“Lust 
Judgment” “Rose Window”
“Rose Window”
“Lust Judgment”
“Triangular 
Pediment”
“Obelisk of 
the Pantheon”
“Triangular 
Pediment”
“Corinthian 
Column”
“Onion
 Dome”
“Arch-shaped 
doorways”
“Arch-shaped 
doorways”
“Onion Dome”
“Eave”
“Column”
“Door”
“Eave”
“Column” “Oceanus”
“Column”
“Window”
Figure B: Several evaluation images and their corresponding ground
truth segmentation across seven scenes.
For evaluation, we carefully selected 10-20 high-quality unoc-
cluded images from each scene. Annotations were made using the
open-source image annotation tool ISAT [22]. For selecting open-
vocabulary query texts, we mainly chose content that appeared in
the scene’s description on Wikipedia.
This includes commonly
used architectural components (e.g., ”Window”), architectural style
terms (e.g., ”Ionic Column”), proper nouns (e.g., ”Oceanus”), and
material nouns (e.g., ”Bronze Sculpture”). Tab. A provides a de-
tailed breakdown of the number of unconstrained images used for
training, the number of evaluation images, and the number of text
queries for each scene. It is worth noting that, to provide a com-
prehensive evaluation of open-vocabulary segmentation from un-
constrained photo collections, we designed query texts that con-
sider not only long-tail classes but also more general classes. For
instance, we created long-tail queries such as ”Ironic Column” as
well as more common queries like ”column”. Additionally, we in-
cluded distractor queries. For example, in the Pantheon scene, we
used ”Obelisk of the Pantheon” as the open-vocabulary query text
instead of ”Obelisk”, to evaluate whether existing methods would
segment the entire building instead of only extracting the obelisk.
Fig. B presents several evaluation images along with their corre-

<!-- page 11 -->
sponding ground truth segmentation.
B
MORE IMPLEMENTATION DETAILS
B.1
Details of Multi-Appeaarance Pixel-Level Language
Feature Extraction
B.1.1
Details of Radiance Field Reconstruction
Table B: Selected multi-appearance rendering IDs for all scenes.
Novel App. 1
Novel App. 2
Novel App. 3
Brandenburg Gate
15080601 1551250100.jpg
01738801 5114523193.jpg
59826471 8014732885.jpg
Trevi Fountain
15457887 10227170235.jpg
45182190 511249303.jpg
80288369 2336500045.jpg
Buckingham Palace
04781012 3416228976.jpg
11220321 6429817645.jpg
42080522 204096736.jpg
Notre-Dame de Paris
73272860 3039447416.jpg
03158689 7322662838.jpg
72005271 4157221941.jpg
Pantheon
00318896 2265892479.jpg
02882184 6968792622.jpg
04938646 2803242734.jpg
Taj Mahal
75255818 297567547.jpg
76552970 5828212829.jpg
86954812 2533844894.jpg
Todaiji Temple
08855480 12135166146.jpg
10449189 312833816.jpg
36907783 8321442187.jpg
For in-the-wild radiance field reconstruction, we re-implement a
simplified version of WE-GS [65]. In WE-GS, a CBAM [70] mod-
ule is introduced to simultaneously predict the transient mask and
appearance embeddings for each unconstrained image. The only
difference in our implementation is that we do not use the chan-
nel and spatial attention blocks mentioned in WE-GS. Instead, we
directly concatenate the predicted transient mask with the feature
maps obtained through a CNN to achieve the attention mechanism,
and then predict the appearance embeddings. All other network
structures and hyperparameter settings follow the vanilla WE-GS
across all 7 scenes.
B.1.2
Details of Multi-Appearance Image Selection
As shown in Tab. B, we provide a detailed list of the multi-
appearance rendering IDs selected for all scenes to ensure the re-
producibility of the proposed method.
B.1.3
Details of Uncertainty Map Cauculation
For computing the appearance uncertainty map and transient un-
certainty map, we apply a normalization step to address the issue
of differing value ranges across different uncertainty maps. Specif-
ically, for each scene, we record the maximum and minimum pixel
values across all transient and appearance uncertainty maps. We
then normalize each uncertainty map accordingly:
UA
i =
UA
i −min
i UA
i
max
i
UA
i −min
i UA
i
,
∀Ii ∈I,
(18)
and
UA
i =
UT
i −min
i UT
i
max
i
UT
i −min
i UT
i
,
∀Ii ∈I.
(19)
B.2
Details of Transient Uncertainty-Aware Autoen-
coder
Our autoencoder is implemented using MLPs, which compress the
512-dimensional CLIP features into 3-dimensional latent features.
Fig. C illustrates the network architecture of our model. We imple-
ment the model using PyTorch [43] and train it for 100 epochs with
a learning rate of 0.0001 across all scenes.
B.3
More Details of MALE-GS Representation and Opti-
mization
We implement the MALE-GS in Python using the PyTorch frame-
work [43], integrating custom CUDA acceleration kernels based on
3DGS’s differentiable Gaussian rasterization [25]. We train the lan-
guage features for 30,000 iterations with a learning rate of 0.0025.
As described in the main paper, we also use a hierarchical SAM-
based method to generate pixel-level language features. We adopt
512
16
Original
CLIP Feature
256
64
32
3
32
64
128
256
512
512
3
Encoder
Decoder
Compressed
CLIP Feature
Restored 
CLIP Feature
Figure C: The model architecture of autoencoder. Green rectangles
represent input or output data, while blue and red rounded rectangles
denote the encoder and decoder components, respectively. Each
rounded rectangle represents an MLP, with numbers indicating the
output dimensions.
Original 
Image
Feature Vis. of
Novel App. 1
Feature Vis. of
Novel App. 2
Feature Vis. of
Novel App. 3
Feature Vis. of
Self App.
Figure D: Visualization of learned compressed language features
from multi-appearance renderings.
Since the scene-specific com-
pressed language feature maps learned by MALE-GS are three-
dimensional, we directly map these three dimensions to RGB for vi-
sualization.
the same strategy as LangSplat [49], using SAM to define three se-
mantic level: subpart, part, and whole. This results in three SAM
segmentation maps and corresponding CLIP feature maps at these
semantic level. To address the multi-scale nature of language fea-
tures, we train three hierarchical MALE-GS. Given a query text,
we evaluate the three MALE-GS models and select the one with
the highest score, similarly with LangSplat.

<!-- page 12 -->
Transient 
Uncertainty 
Map
Appearance
Uncertainty 
Map
Unconstrained
Image
Transient 
Uncertainty 
Map
Appearance
Uncertainty 
Map
Unconstrained
Image
Figure E: More visualization results of the appearance uncertainty
map and transient uncertainty map.
B.4
More
Details
of
Post
Ensemble
and
Open-
vocabulary Querying
The complete set of canonical texts consists of ”object”, ”things”,
”scene”, ”sky”, and ”building”. To mitigate the influence of out-
liers, we apply a mean filter with a kernel size of 20 to smooth the
relevancy maps, similar to LangSplat [49].
C
MORE EXPERIMENT DETAILS AND RESULTS
C.0.1
The Influence of Using Background Filter
As shown in row 9 of Tab. 3 of the main paper, the introduction
of the background filter slightly improves segmentation accuracy.
This is because we find that in outdoor scenes, existing methods
using canonical phrases such as ”object” and ”stuff” as negative
prompts cause the scoring mechanism to focus on the sky and other
background areas. As illustrated in Fig. G, the background filter
significantly suppresses these regions, leading to more accurate seg-
mentation results.
C.1
Visualization of the Learned Multi-Appearance Lan-
guage Features
We demonstrate that CLIP features extracted from renderings with
different appearances exhibit notable differences.
As shown in
Fig. D, the learned language feature maps after MALE-GS training
vary significantly across appearances from the same viewpoint. Our
proposed post-ensemble strategy effectively integrates these varia-
tions, leading to improved segmentation performance.
C.2
More Qualitative Results
Fig. E shows additional visualizations of the language feature ap-
pearance and transient uncertainty maps. Fig. F presents additional
open-vocabulary segmentation results of our method. Due to the
varying resolutions of the unconstrained images, some result im-
ages have been resized for clearer presentation.
C.3
More Application Details and Results
C.3.1
More Details and Results of Interactive Roaming with
Open-vocabulary Queries
We implement the interactive roaming with open-vocabulary
queries application by incorporating the proposed method into WE-
GS. Since our open-vocabulary text query method is a downstream
task of radiance field reconstruction, it can easily adapt to any in-
the-wild radiance field reconstruction method, such as WildGaus-
sians [30], Look at the Sky [64], and GS-W [80]. In the specific
implementation, we use in-the-wild radiance field reconstruction
methods for appearance interpolation. Once the user inputs a query
text, the system highlights the query results and allows interactive
viewing from any viewpoint. We encourage readers to watch the
supplementary video.
In the application case of the video supplemental material, we
show the Trevi Fountain scene, involving appearance interpolation,
text queries, and free-viewpoint roaming simultaneously. This sys-
tem enhances the user’s immersion in the process of understanding
architectural components. It involves smooth interpolation between
5 different appearances and 3 different text queries, with the view-
point zoomed in after the query results are retrieved. For archi-
tectural component highlighting, we first obtain the segmentation
results for the corresponding viewpoint using the proposed method.
Then, we apply the Canny algorithm [7] to extract the edges of the
segmentation mask. For text annotation, since we have the segmen-
tation results, we can ensure that the annotation does not overlap
with the queried architectural component, positioning it adjacent to
the segmented region instead. As shown in Fig. H, when querying
”Relief”, the system accurately highlights and text-annotates the
region, enabling visualization from any viewpoint and appearance.
We encourage readers to review the video results provided in the
supplementary materials for additional details.
C.3.2
More Details and Results of Architectural Style Pat-
tern Recognition
To enable architectural style pattern recognition with our method,
similar to CLIP [52], we predefine a set of candidate vocabular-
ies and sequentially query them, selecting the most likely vocabu-
lary as the pattern recognition result. we leverage large language
model GPT-4 [3] to generate predefined candidate patterns. Given
the scene name as input, GPT-4 outputs five candidate patterns.
(Notably, we find that if the large language model directly predicts
the architectural style of a scene, it may occasionally misidentify
certain styles.) These five candidate patterns are then used in our
method, where we randomly sample some unconstrained images
and apply the ”winner takes all” strategy mentioned in the main
paper to vote for the most probable architectural style.
Fig.
I presents additional results, where we also visualize
heatmaps to illustrate ”what makes a certain building a certain ar-
chitectural style.” Additionally, we report results obtained by di-
rectly using raw CLIP for pattern recognition. The results indicate
that the proposed method achieves more accurate architectural style
pattern recognition.
C.3.3
More Details of 3D Segmentation and Scene Editing
As described in the main paper, we decode the compressed lan-
guage features at the per-Gaussian level and compare them with the
text CLIP features to obtain a 3D mask, indicating which 3D Gaus-
sians are selected. Once the 3D Gaussians are selected, we dis-
card the multi-appearance language features within each Gaussian,
converting the proposed MALE-GS representation into the vanilla
3DGS representation. This transformation allows seamless integra-
tion into any downstream applications.
We import the converted 3DGS into SuperSplat and utilize its
built-in transformation and scaling tools to manipulate the selected
3D Gaussians and merge them with the original scene. This enables
the creation of an edited 3D scene that supports free-viewpoint
roaming.
C.4
Video Supplemental Material
We provide a detailed introduction to our method and experimental
results in the video supplemental material. We encourage the read-

<!-- page 13 -->
“Winged Horse”
“Oceanus”
“Column”
“Reliefs”
“Oceanus”
“Onion Dome”
“Onion Dome”
“Onion Dome”
“Arch-shaped Doorways”
“Arch-shaped Doorways”
“Column”
“Window”
“Eave”
“Door”
“Eave”
“Queen Victoria Memorial”
“Queen Victoria Memorial”
“Triangular Pediment”
“Triangular Pediment”
“Corinthian Column”
“Rose Window”
“Last Judgment”
“Rose Window”
“Rose Window”
“Last Judgment”
“Queen Victoria Memorial”
“Queen Victoria Memorial”
“Ironic Column”
“Ironic Column”
“Ironic Column”
“Icon Cross”
“Bronze Sculpture”
“Column”
“Column”
“Bronze Sculpture”
Figure F: More open-vocabulary segmentation results in the proposed PT-OVS dataset.
w/o 
Bkg. Filter
Completed 
Model
“Queen Victoria Memorial”
“Triangular Pediment”
“Bronze sculpture”
Figure G: Ablation studies on the background filter module.
The
heatmap depicts correlation: warmer colors represent stronger cor-
relations, cooler colors indicate weaker correlations, and the absence
of color signifies no correlation.
ers to watch the supplementary video for a clearer understanding of
our approach.
REFERENCES
[1] Imc-pt 2020 dataset. https://www.cs.ubc.ca/ kmyi/imw2020/data.html,
2020. 10
[2] Supersplat. https://superspl.at/editor, 2025. 9
[3] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman,
D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. Gpt-4
technical report. arXiv preprint arXiv:2303.08774, 2023. 12
[4] Y. Bao, C. Tang, Y. Wang, and H. Li. Seg-wild: Interactive segmen-
tation based on 3d gaussian splatting for unconstrained image collec-
tions. arXiv preprint arXiv:2507.07395, 2025. 2
[5] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla,
and P. P. Srinivasan. Mip-nerf: A multiscale representation for anti-
aliasing neural radiance fields. In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision, pp. 5855–5864, 2021. 2
[6] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hed-
man. Mip-nerf 360: Unbounded anti-aliased neural radiance fields.
In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 5470–5479, 2022. 2
[7] J. Canny. A computational approach to edge detection. IEEE Trans-
actions on Pattern Analysis and Machine Intelligence, (6):679–698,
1986. 12
[8] L.-C. Chen. Rethinking atrous convolution for semantic image seg-
mentation. arXiv preprint arXiv:1706.05587, 2017. 5
[9] R. Chen, Y. Liu, L. Kong, X. Zhu, Y. Ma, Y. Li, Y. Hou, Y. Qiao, and
W. Wang. Clip2scene: Towards label-efficient 3d scene understanding
by clip. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 7020–7030, 2023. 3
[10] X. Chen, Q. Zhang, X. Li, Y. Chen, Y. Feng, X. Wang, and J. Wang.
Hallucinated neural radiance fields in the wild. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pp. 12943–12952, 2022. 2
[11] Z. Chen, T. Funkhouser, P. Hedman, and A. Tagliasacchi. Mobilenerf:
Exploiting the polygon rasterization pipeline for efficient neural field
rendering on mobile architectures. In Proceedings of the IEEE/CVF

<!-- page 14 -->
Figure H: Application of interactive roaming with open-vocabulary queries. The images, shown from left to right and top to bottom, are keyframes
from a video generated by our method. Users can freely roam from any viewpoint, adjust lighting, and highlight architectural components with
open-vocabulary queries. We encourage readers to watch our supplementary video.
Unconstrained 
Photo Collection
Heatmaps for Textual 
Query
“Neoclassical”
“Georgian”
“Victorian Gothic”
“Palladian”
 “Edwardian Baroque”
9
2
1
0
0
…
…
“French Gothic”
“Romanesque”
“Renaissance”
“Baroque”
 “Neoclassical”
14
1
…
…
0
0
0
Ours
Vote Result
Vanilla CLIP
 Vote Result
“Mughal”
“Persian”
“Baroque”
“Gothic Revival”
 “Neoclassical”
14
0
…
…
0
0
0
2
…
…
0
1
0
“Neoclassical”
“Georgian”
“Victorian Gothic”
“Palladian”
 “Edwardian Baroque”
6
2
1
3
0
“French Gothic”
“Romanesque”
“Renaissance”
“Baroque”
 “Neoclassical”
12
3
0
0
0
“Mughal”
“Persian”
“Baroque”
“Gothic Revival”
 “Neoclassical”
12
2
0
0
0
“Japanese Buddhist”
“Edo Period Vernacular”
“Chinese Tang Dynasty”
“Zen Minimalist”
 “Neoclassical”
6
6
0
2
0
“Japanese Buddhist”
“Edo Period Vernacular”
“Chinese Tang Dynasty”
“Zen Minimalist”
 “Neoclassical”
1
Figure I: More examples of architectural style pattern recognition ap-
plications.
Conference on Computer Vision and Pattern Recognition, pp. 16569–
16578, 2023. 2
[12] M. Cherti, R. Beaumont, R. Wightman, M. Wortsman, G. Ilharco,
C. Gordon, C. Schuhmann, L. Schmidt, and J. Jitsev. Reproducible
scaling laws for contrastive language-image learning. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recog-
nition, pp. 2818–2829, 2023. 5
[13] H. Dahmani, M. Bennehar, N. Piasco, L. Roldao, and D. Tsishkou.
Swag: Splatting in the wild images with appearance-conditioned gaus-
sians. In European Conference on Computer Vision, pp. 325–340.
Springer, 2024. 2
[14] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly,
J. Uszkoreit, and N. Houlsby. An image is worth 16x16 words: Trans-
formers for image recognition at scale. In International Conference
on Learning Representations, 2021. 5
[15] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, Z. Wang, et al. Light-
gaussian: Unbounded 3d gaussian compression with 15x reduction
and 200+ fps. Advances in Neural Information Processing Systems,
37:140138–140158, 2024. 2
[16] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and
A. Kanazawa. K-planes: Explicit radiance fields in space, time, and
appearance. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pp. 12479–12488, 2023. 2
[17] T. Groueix, M. Fisher, V. G. Kim, B. C. Russell, and M. Aubry. A
papier-mˆach´e approach to learning 3d surface generation.
In Pro-
ceedings of the IEEE/CVF conference on computer vision and pattern
recognition, pp. 216–224, 2018. 2
[18] Y. Guo, J. Hu, Y. Qu, and L. Cao. Wildseg3d: Segment any 3d objects
in the wild from 2d images. arXiv preprint arXiv:2503.08407, 2025.
2
[19] G. E. Hinton and R. R. Salakhutdinov. Reducing the dimensionality
of data with neural networks. science, 313(5786):504–507, 2006. 4
[20] L. H¨ollein, A. Boˇziˇc, M. Zollh¨ofer, and M. Nießner. 3dgs-lm: Faster
gaussian-splatting optimization with levenberg-marquardt.
arXiv
preprint arXiv:2409.12892, 2024. 2
[21] W. Hu, Y. Wang, L. Ma, B. Yang, L. Gao, X. Liu, and Y. Ma. Tri-
miprf: Tri-mip representation for efficient anti-aliasing neural radi-
ance fields. In Proceedings of the IEEE/CVF International Conference
on Computer Vision, pp. 19774–19783, 2023. 2
[22] S. Ji and H. Zhang. ISAT with Segment Anything: An Interactive
Semi-Automatic Annotation Tool, 2024. Updated on 2025-02-07. 10
[23] Y. Ji, H. Zhu, J. Tang, W. Liu, Z. Zhang, X. Tan, and Y. Xie. Fastlgs:
Speeding up language embedded gaussians with feature grid map-
ping. In Proceedings of the AAAI Conference on Artificial Intelligence,
vol. 39, pp. 3922–3930, 2025. 3
[24] Y. Jin, D. Mishkin, A. Mishchuk, J. Matas, P. Fua, K. M. Yi, and
E. Trulls. Image matching across wide baselines: From paper to prac-
tice. International Journal of Computer Vision, 129(2):517–547, Oct.
2020. doi: 10.1007/s11263-020-01385-0 7
[25] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis. 3d gaussian
splatting for real-time radiance field rendering. ACM Transactions on
Graphics, 42(4), 2023. 1, 2, 3, 5, 8, 11
[26] J. Kerr, C. M. Kim, K. Goldberg, A. Kanazawa, and M. Tancik.
Lerf: Language embedded radiance fields.
In Proceedings of the
IEEE/CVF International Conference on Computer Vision, pp. 19729–
19739, 2023. 3
[27] S. Kheradmand, D. Rebain, G. Sharma, W. Sun, Y.-C. Tseng, H. Isack,
A. Kar, A. Tagliasacchi, and K. M. Yi. 3d gaussian splatting as markov
chain monte carlo. Advances in Neural Information Processing Sys-
tems, 37:80965–80986, 2024. 2
[28] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson,
T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al. Segment any-
thing. In Proceedings of the IEEE/CVF International Conference on
Computer Vision, pp. 4015–4026, 2023. 3
[29] S. Kobayashi, E. Matsumoto, and V. Sitzmann. Decomposing nerf for
editing via feature field distillation. Advances in Neural Information
Processing Systems, 35:23311–23330, 2022. 3
[30] J. Kulhanek, S. Peng, Z. Kukelova, M. Pollefeys, and T. Sattler.
Wildgaussians: 3d gaussian splatting in the wild. In The Thirty-eighth
Annual Conference on Neural Information Processing Systems, 2024.
2, 3, 12
[31] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park. Compact 3d gaussian
representation for radiance field. arXiv preprint arXiv:2311.13681,
2023. 2
[32] B. Li, K. Q. Weinberger, S. Belongie, V. Koltun, and R. Ranftl.
Language-driven semantic segmentation. In International Conference
on Learning Representations. 10
[33] J. Li, D. Li, C. Xiong, and S. C. Hoi. Blip: Bootstrapping language-
image pre-training for unified vision-language understanding and gen-
eration. In International Conference on Machine Learning (ICML),
2022. 9
[34] X. Li, Z. Lai, L. Xu, Y. Qu, L. Cao, S. Zhang, B. Dai, and R. Ji. Di-
rector3d: Real-world camera trajectory and 3d scene generation from
text. arXiv preprint arXiv:2406.17601, 2024. 2
[35] G. Liao, J. Li, Z. Bao, X. Ye, J. Wang, Q. Li, and K. Liu. Clip-gs:
Clip-informed gaussian splatting for real-time and view-consistent 3d
semantic understanding. arXiv preprint arXiv:2404.14249, 2024. 3

<!-- page 15 -->
[36] K. Liu, F. Zhan, J. Zhang, M. Xu, Y. Yu, A. El Saddik, C. Theobalt,
E. Xing, and S. Lu. Weakly supervised 3d open-vocabulary segmenta-
tion. Advances in Neural Information Processing Systems, 36:53433–
53456, 2023. 3
[37] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai. Scaffold-
gs: Structured 3d gaussians for view-adaptive rendering. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 20654–20664, 2024. 2
[38] R. Martin-Brualla, N. Radwan, M. S. Sajjadi, J. T. Barron, A. Doso-
vitskiy, and D. Duckworth. Nerf in the wild: Neural radiance fields
for unconstrained photo collections. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 7210–
7219, 2021. 2, 5, 10
[39] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison. Gaussian splat-
ting slam. In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 18039–18048, 2024. 2
[40] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoor-
thi, and R. Ng. Nerf: Representing scenes as neural radiance fields for
view synthesis. Communications of the ACM, 65(1):99–106, 2021. 1,
2
[41] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khali-
dov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. Dinov2:
Learning robust visual features without supervision. arXiv preprint
arXiv:2304.07193, 2023. 3
[42] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove.
Deepsdf: Learning continuous signed distance functions for shape
representation. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pp. 165–174, 2019. 2
[43] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan,
T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf,
E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner,
L. Fang, J. Bai, and S. Chintala. Pytorch: An imperative style, high-
performance deep learning library.
In H. Wallach, H. Larochelle,
A. Beygelzimer, F. d'Alch´e-Buc, E. Fox, and R. Garnett, eds., Ad-
vances in Neural Information Processing Systems 32, pp. 8024–8035.
Curran Associates, Inc., 2019. 5, 11
[44] S. Peng, K. Genova, C. Jiang, A. Tagliasacchi, M. Pollefeys,
T. Funkhouser, et al. Openscene: 3d scene understanding with open
vocabularies. In Proceedings of the IEEE/CVF conference on Com-
puter Vision and Pattern Recognition, pp. 815–824, 2023. 3
[45] S. Peng, M. Niemeyer, L. Mescheder, M. Pollefeys, and A. Geiger.
Convolutional occupancy networks. In European Conference on Com-
puter Vision, pp. 523–540. Springer, 2020. 2
[46] C. R. Qi, H. Su, K. Mo, and L. J. Guibas. Pointnet: Deep learning
on point sets for 3d classification and segmentation. In Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition,
pp. 652–660, 2017. 2
[47] C. R. Qi, H. Su, M. Nießner, A. Dai, M. Yan, and L. J. Guibas. Vol-
umetric and multi-view cnns for object classification on 3d data. In
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 5648–5656, 2016. 2
[48] C. R. Qi, L. Yi, H. Su, and L. J. Guibas. Pointnet++: Deep hierarchical
feature learning on point sets in a metric space. Advances in Neural
Information Processing Systems, 30, 2017. 2
[49] M. Qin, W. Li, J. Zhou, H. Wang, and H. Pfister. Langsplat: 3d lan-
guage gaussian splatting. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pp. 20051–20060, 2024.
2, 3, 5, 7, 11, 12
[50] Y. Qu, D. Chen, X. Li, X. Li, S. Zhang, L. Cao, and R. Ji. Drag your
gaussian: Effective drag-based editing with score distillation for 3d
gaussian splatting. arXiv preprint arXiv:2501.18672, 2025. 2
[51] Y. Qu, S. Dai, X. Li, J. Lin, L. Cao, S. Zhang, and R. Ji. Goi: Find 3d
gaussians of interest with an optimizable open-vocabulary semantic-
space hyperplane. In Proceedings of the 32nd ACM International Con-
ference on Multimedia, pp. 5328–5337, 2024. 2, 3
[52] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transfer-
able visual models from natural language supervision. In International
Conference on Machine Learning, pp. 8748–8763. PMLR, 2021. 2, 3,
12
[53] C. Reiser, R. Szeliski, D. Verbin, P. Srinivasan, B. Mildenhall,
A. Geiger, J. Barron, and P. Hedman. Merf: Memory-efficient ra-
diance fields for real-time view synthesis in unbounded scenes. ACM
Transactions on Graphics (TOG), 42(4):1–12, 2023. 2
[54] W. Ren, Z. Zhu, B. Sun, J. Chen, M. Pollefeys, and S. Peng. Nerf
on-the-go: Exploiting uncertainty for distractor-free nerfs in the wild.
In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 8931–8940, 2024. 7
[55] S. Sabour, L. Goli, G. Kopanas, M. Matthews, D. Lagun, L. Guibas,
A. Jacobson, D. J. Fleet, and A. Tagliasacchi. Spotlesssplats: Ignoring
distractors in 3d gaussian splatting. arXiv preprint arXiv:2406.20055,
2024. 3
[56] Y. Shen, Z. Zhang, X. Li, Y. Qu, Y. Lin, S. Zhang, and L. Cao.
Evolving high-quality rendering and reconstruction in a unified
framework with contribution-adaptive regularization. arXiv preprint
arXiv:2503.00881, 2025. 2
[57] J.-C. Shi, M. Wang, H.-B. Duan, and S.-H. Guan. Language embedded
3d gaussians for open-vocabulary scene understanding. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 5333–5343, 2024. 2, 3, 5, 7
[58] J.-C. Shi, M. Wang, H.-B. Duan, and S.-H. Guan. Language embedded
3d gaussians for open-vocabulary scene understanding. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 5333–5343, 2024. 3
[59] N. Snavely.
Scene reconstruction and visualization from internet
photo collections: A survey. IPSJ Transactions on Computer Vision
and Applications, 3:44–66, 2011. 1
[60] N. Snavely, S. M. Seitz, and R. Szeliski. Photo tourism: exploring
photo collections in 3d. In ACM SIGGRAPH 2006 Papers, pp. 835–
846. ACM New York, NY, USA, 2006. 1, 2, 5, 9
[61] H. Talebi and P. Milanfar. Nima: Neural image assessment. IEEE
Transactions on Image Processing, 27(8):3998–4011, 2018. 5, 10
[62] M. Tancik, E. Weber, E. Ng, R. Li, B. Yi, T. Wang, A. Kristoffersen,
J. Austin, K. Salahi, A. Ahuja, et al. Nerfstudio: A modular frame-
work for neural radiance field development. In ACM SIGGRAPH 2023
Conference Proceedings, pp. 1–12, 2023. 3
[63] J. Tang, J. Ren, H. Zhou, Z. Liu, and G. Zeng. Dreamgaussian: Gen-
erative gaussian splatting for efficient 3d content creation. In Interna-
tional Conference on Learning Representations, 2024. 2
[64] Y. Wang, J. Wang, R. Gao, Y. Qu, W. Duan, S. Yang, and Y. Qi.
Look at the sky: Sky-aware efficient 3d gaussian splatting in the wild.
IEEE Transactions on Visualization and Computer Graphics, pp. 1–
11, 2025. doi: 10.1109/TVCG.2025.3549187 2, 3, 12
[65] Y. Wang, J. Wang, and Y. Qi. We-gs: An in-the-wild efficient 3d gaus-
sian representation for unconstrained photo collections. arXiv preprint
arXiv:2406.02407, 2024. 2, 3, 8, 9, 11
[66] Y. Wang, J. Wang, Y. Qu, and Y. Qi. Rip-nerf: Learning rotation-
invariant point-based neural radiance field for fine-grained editing and
compositing. In Proceedings of the 2023 ACM International Confer-
ence on Multimedia Retrieval, pp. 125–134, 2023. 2
[67] Y. Wang, J. Wang, C. Wang, W. Duan, Y. Bao, and Y. Qi. Scarf: Scal-
able continual learning framework for memory-efficient multiple neu-
ral radiance fields. Computer Graphics Forum, 43(7):e15255, 2024.
2
[68] Y. Wang, J. Wang, C. Wang, and Y. Qi.
Rise-editing: Rotation-
invariant neural point fields with interactive segmentation for fine-
grained and efficient editing. Neural Networks, 187:107304, 2025.
doi: 10.1016/j.neunet.2025.107304 2
[69] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. Image
quality assessment: from error visibility to structural similarity. IEEE
Transactions on Image Processing, 13(4):600–612, 2004. 3
[70] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon. Cbam: Convolutional
block attention module. In European Conference on Computer Vision,
pp. 3–19. Springer, 2018. 11
[71] C. Xu, J. Kerr, and A. Kanazawa. Splatfacto-w: A nerfstudio imple-
mentation of gaussian splatting for unconstrained photo collections.
arXiv preprint arXiv:2407.12306, 2024. 3
[72] J. Xu, Y. Mei, and V. Patel. Wild-gs: Real-time novel view synthesis
from unconstrained photo collections. Advances in Neural Informa-
tion Processing Systems, 37:103334–103355, 2024. 2, 3

<!-- page 16 -->
[73] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li. Gs-
slam: Dense visual slam with 3d gaussian splatting. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion, pp. 19595–19604, 2024. 2
[74] B. Yang, C. Bao, J. Zeng, H. Bao, Y. Zhang, Z. Cui, and G. Zhang.
Neumesh: Learning disentangled neural mesh-based implicit field for
geometry and texture editing. In European Conference on Computer
Vision, pp. 597–614. Springer, 2022. 2
[75] Y. Yang, S. Zhang, Z. Huang, Y. Zhang, and M. Tan. Cross-ray neu-
ral radiance fields for novel-view synthesis from unconstrained image
collections. In Proceedings of the IEEE/CVF International Confer-
ence on Computer Vision, pp. 15901–15911, 2023. 2
[76] M. Ye, M. Danelljan, F. Yu, and L. Ke. Gaussian grouping: Segment
and edit anything in 3d scenes. In European Conference on Computer
Vision, pp. 162–179. Springer, 2024. 3, 5, 7
[77] Z. Yu, A. Chen, B. Huang, T. Sattler, and A. Geiger. Mip-splatting:
Alias-free 3d gaussian splatting.
In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 19447–
19456, 2024. 2
[78] Y.-J. Yuan, Y.-T. Sun, Y.-K. Lai, Y. Ma, R. Jia, and L. Gao. Nerf-
editing: geometry editing of neural radiance fields. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern Recogni-
tion, pp. 18353–18364, 2022. 2
[79] X. Zhai, A. Steiner, A. Kolesnikov, and L. Beyer.
Siglip: Scal-
ing up contrastive vision-language pretraining.
arXiv preprint
arXiv:2303.15343, 2023. 9
[80] D. Zhang, C. Wang, W. Wang, P. Li, M. Qin, and H. Wang. Gaussian
in the wild: 3d gaussian splatting for unconstrained image collections.
In European Conference on Computer Vision, pp. 341–359. Springer,
2024. 2, 3, 8, 9, 12
[81] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari, S. You,
Z. Wang, and A. Kadambi. Feature 3dgs: Supercharging 3d gaus-
sian splatting to enable distilled feature fields. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pp. 21676–21685, 2024. 2, 3, 5, 7
