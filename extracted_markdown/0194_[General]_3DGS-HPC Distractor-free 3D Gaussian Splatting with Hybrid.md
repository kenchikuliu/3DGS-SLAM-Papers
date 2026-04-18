<!-- page 1 -->
3DGS-HPC: Distractor-free 3D Gaussian Splatting
with Hybrid Patch-wise Classification
Jiahao Chen1,2
Yipeng Qin3
Ganlong Zhao4
Xin Li2
Wenping Wang2
Guanbin Li1∗
1Sun Yat-sen University
2Texas A&M University
3Cardiff University
4Centre for Perceptual and Interactive Intelligence
HPC (Ours)
SLS-mlp [43]
WildGaussians [22]
3DGS [19]
Training #1
Training #2
Training #3
Training #4
Rendered #1
Rendered #2
Figure 1. Comparison between previous methods and our proposed Hybrid Patch-wise Classification (HPC). When training 3DGS
with static scenes disturbed by transient distractors, previous methods utilize image semantic priors but fail to separate transient distractors
(e.g., pedestrians and shadows) from static backgrounds (e.g., mountains). By addressing their inherent semantic limitations, our method
can generate binary masks that better represent transient distractors, thereby removing related artifacts in rendered results (yellow boxes).
Abstract
3D Gaussian Splatting (3DGS) has demonstrated remark-
able performance in novel view synthesis and 3D scene re-
construction, yet its quality often degrades in real-world
environments due to transient distractors, such as mov-
ing objects and varying shadows. Existing methods com-
monly rely on semantic cues extracted from pre-trained vi-
sion models to identify and suppress these distractors, but
such semantics are misaligned with the binary distinction
between static and transient regions and remain fragile un-
der the appearance perturbations introduced during 3DGS
optimization. We propose 3DGS-HPC, a framework that
*The corresponding author is Guanbin Li.
circumvents these limitations by combining two comple-
mentary principles: a patch-wise classification strategy that
leverages local spatial consistency for robust region-level
decisions, and a hybrid classification metric that adaptively
integrates photometric and perceptual cues for more reli-
able separation. Extensive experiments demonstrate the su-
periority and robustness of our method in mitigating dis-
tractors to improve 3DGS-based novel view synthesis.
1. Introduction
Recent advances in 3D Gaussian Splatting (3DGS) [15, 17,
19, 20, 29, 63] have enabled high-quality novel view synthe-
sis with real-time performance, establishing it as a strong
1
arXiv:2603.07587v1  [cs.CV]  8 Mar 2026

<!-- page 2 -->
alternative to Neural Radiance Fields (NeRF) [34]. How-
ever, similar to NeRF, 3DGS is built upon the assumption
that all training images capture a completely static scene.
In real-world settings, this assumption is frequently violated
by transient distractors (i.e., dynamic or temporally incon-
sistent elements such as pedestrians, vehicles, or shadows)
that introduce artifacts and significantly degrade reconstruc-
tion quality. The diversity and unpredictability of these dis-
tractors in appearance, scale, and motion make them dif-
ficult to detect and remove reliably. As a result, transient
distractors remain a key obstacle to deploying 3DGS for ro-
bust real-world scene reconstruction.
To mitigate the effects of distractors, most existing meth-
ods follow a simple yet effective paradigm: generating
pixel-wise binary masks to label and filter out transient pix-
els in the training images, thereby preventing the model
from being corrupted by distractors during optimization.
This formulation reduces the task to estimating accurate
binary masks, essentially a binary classification problem.
Current research addresses this problem from two perspec-
tives: classification granularity and classification metric.
Early methods [10, 33, 42, 59] perform pixel-wise classi-
fication using photometric errors, but the resulting masks
are often noisy and spatially inconsistent, as pixel-level pre-
dictions lack sufficient local context.
Recent studies in-
troduce semantic priors to improve both aspects.
Some
works [6, 22, 23, 37, 40, 43] enforce semantic coherence
by unifying pixel-wise classification results into semanti-
cally consistent regions, thus enhancing spatial stability,
while others [22, 32] adopt perceptual metrics that measure
semantic similarity between training and rendered images.
However, as shown in Fig. 1, semantic-coherence methods
rely on external semantics derived from vision foundation
models that are not explicitly designed to distinguish static
from transient regions, resulting in a semantic mismatch be-
tween predicted and ground-truth transient areas. Percep-
tual metrics also suffer from semantic fragility, as their fea-
ture representations are highly sensitive to minor appear-
ance perturbations, leading to unstable responses and the
misclassification of static regions as transient.
In this work, we address the challenges of semantic mis-
match and fragility in existing methods by introducing a
novel framework for distractor-free 3DGS, termed Hybrid
Patch-wise Classification (HPC). From the perspective of
classification granularity, HPC avoids reliance on seman-
tics derived from unrelated tasks and instead leverages the
assumption of local spatial consistency. By directly per-
forming classification at the patch level, it captures richer
local context than pixel-wise classification while maintain-
ing higher robustness and efficiency than semantic region
grouping. In terms of the classification metric, HPC inte-
grates photometric and perceptual cues into a hybrid metric
that adaptively balances color fidelity and semantic simi-
larity, yielding more reliable separation between static and
transient regions. Extensive experiments demonstrate that
HPC significantly improves the robustness and fidelity of
3DGS reconstructions in the presence of diverse distractors.
Our contributions are summarized as follows:
• We propose Hybrid Patch-wise Classification (HPC), a
novel framework for distractor-free 3D Gaussian Splat-
ting (3DGS) that achieves robust reconstruction in real-
world scenes containing transient distractors.
• To overcome the limitations of previous semantic-based
methods, HPC introduces two complementary designs:
a patch-wise classification approach that leverages local
spatial consistency for efficient and context-aware clas-
sification, and a hybrid classification metric that adap-
tively fuses photometric and perceptual cues to enhance
static/transient separation.
• We validate the effectiveness of HPC through exten-
sive experiments, showing consistent improvements over
state-of-the-art methods in both reconstruction quality
and robustness to distractors.
2. Related Work
NeRF & 3DGS in Non-static Scenes.
NeRF [34] has
emerged as a promising solution to achieve high-quality
3D scene reconstruction and novel view synthesis, while
3DGS [19] has been proposed as a compelling alternative
due to its real-time rendering speed and comparable render-
ing quality. However, both approaches and many of their
variants assume that the scene to be reconstructed is static
and are therefore not applicable to real-world scenes con-
taining non-static elements. In contrast to methods that ex-
plicitly reconstruct dynamic components [14, 18, 26, 30,
57, 60, 61], our work follows another research line that
treats these non-static elements as transient distractors and
removes them to enhance the reconstruction of static con-
tent. Early attempts [28, 39, 44, 47, 48, 50] use pre-trained
semantic segmentation models [7–9, 11] to detect transient
distractors of known classes, but this prior information is
intractable to obtain in practice. Subsequent research has
focused on more generalizable approaches, which can be
roughly grouped into two paradigms: uncertainty model-
ing [10, 13, 22, 23, 27, 32, 33, 37, 40, 53, 55, 59, 64],
which relies on neural networks to predict pixel uncertain-
ties of training images; and pixel classification [3, 6, 42,
43, 51, 54, 58], which directly classifies pixels as static or
transient based on their reconstruction errors. We view the
former as a soft form of the latter, as both rely on the Mem-
orization Effect [1, 2, 46, 54] that neural networks tend to
learn contextually consistent samples faster than inconsis-
tent samples. Consequently, pixels with larger reconstruc-
tion errors or higher uncertainties are regarded as transient
and excluded from optimization during training.
2

<!-- page 3 -->
Rendered Image 
Training Image
Photometric Error
Perceptual Error
Perceptual
Error Metric
Photometric
Error Metric
(b) Hybrid Classification Metric
GMM-based
Classification
Percentile-based
Classification
(c) Patch-wise Classification Approach
3D Gaussian Splatting
Loss Function
(a) 3DGS Training
Static Map ȩ(ե)
Percentile
Threshold
Static Map ȩ(֙)
Static Map ȩ(ձ)
Patch-wise Partition
Figure 2. Overview of Hybrid Patch-wise Classification (HPC). Given training images disturbed by transient distractors, our method
(a) optimizes 3D Gaussian Splatting following standard practice, (b) classifies static and transient regions using a novel hybrid metric
that combines the strengths of photometric and perceptual cues, and (c) performs classification at the patch level to exploit local spatial
consistency, thereby enhancing robustness and efficiency without relying on external semantics such as segmentation or detection models.
Semantics in Distractor-free NeRF & 3DGS. Recent ad-
vances in vision foundation models [5, 21, 36, 38, 41, 49]
have inspired a new line of distractor-free NeRF and 3DGS
methods that exploit semantic priors to better separate tran-
sient distractors from static backgrounds. These semantic
priors are typically utilized in two ways:
• Enhancing semantically coherent classification. To en-
courage semantically similar pixels to produce coherent
classification output, some methods [22, 32, 40, 43] incor-
porate image semantics as an auxiliary input to the pre-
diction network, while others [6, 32, 37, 43] segment the
image into semantically consistent regions and use this
structure to refine classification results.
• Serving as perceptual error metrics. To address the limi-
tations of photometric error metrics (e.g., RGB L1 error,
SSIM error) [6, 33, 40, 42] in distinguishing transient dis-
tractors from static content with similar colors or textures,
recent studies [22, 32] adopt perceptual metrics derived
from feature embeddings like DINOv2 [36] to capture
semantic-level image differences.
Although these approaches achieve notable improvements,
their reliance on external semantics introduces two chal-
lenges: semantic mismatch between general-purpose vi-
sion semantics and the specific static/transient distinction,
and semantic fragility that small appearance perturbations
may lead to unstable feature responses and misclassifica-
tion.
In our work, we address this gap by proposing a
novel distractor-free 3DGS framework, which consists of
i) a patch-wise classification approach that leverages lo-
cal spatial consistency to improve robustness and efficiency
without relying on external semantics, and ii) a hybrid clas-
sification metric that combines perceptual and photometric
error cues to correct semantic perturbability.
3. Preliminaries
Given a set of multi-view images I =

Ii ∈RH×W ×3	NI
i=1
captured from a scene, we have:
3D Gaussian Splatting (3DGS) [19]. 3DGS represents the
scene as a set of learnable 3D Gaussians G = {gj}NG
j=1,
each characterized by its position, covariance, color coef-
ficients, and opacity. Given the specific camera parameters
Pi, 3DGS render the corresponding image ˆIi in one forward
pass instead of pixel batches like NeRF [34]:
ˆIi = Frender (G, Pi) ,
(1)
where Frender is the rendering function that performs 2D pro-
jection, tile splitting, depth-based sorting, and α-blending.
During training, G is optimized by minimizing the photo-
metric error between the rendered image ˆIi and the training
image Ii using the loss L from the original paper [19]:
L(ˆIi, Ii) = (1 −λ)L1(ˆIi, Ii) + λLSSIM(ˆIi, Ii),
(2)
where λ = 0.2, and L1 and LSSIM are L1 loss and SSIM
loss, respectively.
3DGS in Non-static Scenes. An effective strategy to avoid
artifacts caused by transient distractors is to mask out pixels
related to distractors (i.e., transient pixels) during training.
Following existing work [6, 42, 43], we introduce a binary
map Mi ∈{0, 1}H×W for each image Ii, where 1 indicates
static pixels and 0 denotes transient ones. We then apply Mi
as masks to Eq. (2) to suppress distractor regions:
L(ˆI′
i, I′
i) = (1 −λ) L1(ˆI′
i, I′
i) + λLSSIM(ˆI′
i, I′
i),
(3)
where ˆI′
i = Mi ⊙ˆIi and I′
i = Mi ⊙Ii. Thus, the quality
of reconstruction critically depends on the accuracy of the
binary maps, which our method aims to improve.
3

<!-- page 4 -->
(a) RGB Image
(b) Superpixel
(c) SAM2
(d) Feature Clustering
(e) Patch-wise
(f) Reference Map
Figure 3.
Visualization of Different Partitioning Strategies.
When grouping pixels into multiple semantically consistent re-
gions with blue borders, existing methods (b-d) fail to faithfully
represent transient regions (e.g., pedestrians and shadows) due to
semantic mismatch between different tasks, while our patch-wise
partition strategy (e) can accurately extract these regions.
4. Method
As Eq. (3) implies, the performance of the reconstruction
model depends on the quality of binary maps, which are
generated by classifying each pixel as static or transient. To
this end, we propose a novel Hybrid Patch-wise Classifica-
tion (HPC) framework (Fig. 2) that addresses the pixel clas-
sification challenge from two complementary perspectives:
the classification approach (Sec. 4.1) and the classification
metric (Sec. 4.2) as follows.
4.1. Patch-wise Classification Approach
Recent methods have increasingly focused on incorporating
semantic consistency to refine noisy pixel-wise classifica-
tion results by grouping pixels into regions with coherent in-
ternal semantics. This is typically achieved using superpixel
algorithms [52], promptable object segmentation [21, 38],
or semantic feature clustering [35, 49]. The motivation be-
hind this trend stems from the observation that although
static and transient pixels exhibit bimodal error distributions
during training [42, 54], accurate separation remains diffi-
cult in practice due to the limited context at the pixel level,
leading to sensitivity to local perturbations and inconsistent
predictions. Despite their success, these methods overlook
the subtle differences in semantics between distractor-free
3DGS and other tasks (e.g., semantic segmentation), thus
producing suboptimal results.
Semantic Mismatch between Different Tasks. As illus-
trated by the binary map in Eq. (3), distractor-free 3DGS
requires only two semantic categories: static and transient.
However, vision foundation models provide semantics op-
timized for general visual understanding rather than this bi-
nary separation, and no existing method can reliably extract
these specific semantics. As a result, state-of-the-art ap-
proaches adopt workaround solutions by leveraging seman-
Rendered Image
Error Map (L1)
Error Map (DINOv2)
Trained Image
Static Map (L1)
Static Map (DINOv2)
Figure 4. Visualization of Different Error Metrics. Given train-
ing images with distractors and corresponding rendered images,
traditional photometric metrics (L1) produce noisy static maps due
to low-level appearance ambiguities (e.g., similar colors). Instead,
perceptual metrics based on vision foundation models (DINOv2)
produce clean static maps with clear boundaries.
tics defined for other tasks, under the assumption that static
and transient regions can be approximated as a union of se-
mantic regions for other tasks. For instance, as shown in
Fig. 3, superpixel methods cluster pixels based on low-level
appearance cues and are easily confused by color ambigui-
ties (e.g., black hairs vs. dark forests); SAM2 [38] segments
objects via prompt-based priors and tends to overlook subtle
transient cues (e.g., merging human shadows into the same
ground region); and feature clustering groups pixels by sim-
ilarity in the embedding space of vision foundation models,
which emphasizes semantic similarity rather than temporal
or photometric consistency, resulting in misclassifications
similar to those of SAM2. Although each provides a form
of semantic partitioning, none of them faithfully captures
the binary semantic distinction required for distractor-free
3DGS, leading to mismatches and reduced performance.
Patch-wise Partition Strategy.
To bridge this gap, we
propose a simple yet effective alternative that completely
avoids external semantics.
Instead of relying on task-
specific semantic grouping, we leverage the assumption of
local spatial consistency (i.e., neighboring pixels within
a small region tend to share the same property) to over-
partition each training image into regular, non-overlapping
patches and directly perform classification at the patch
level. Patch-wise classification provides richer contextual
cues than pixel-wise classification and greater robustness
to local perturbations, while significantly reducing compu-
tational cost by aggregating pixels into fewer units to be
classified. Although this mechanical partition may produce
boundaries that do not align precisely with the edges of the
object, existing work [32] has demonstrated that reconstruc-
tion quality remains largely unaffected, provided that tran-
sient distractors are effectively identified.
Concretely, given a pixel-wise error map εi ∈RH×W
4

<!-- page 5 -->
Perceptual (DINOv2)
Photometric (L1)
Rendered Image / Feature
Training Image / Feature
Error Map
Static Map (GMM)
Static Map (Percentile)
Figure 5. Comparison of different classification approaches with different error metrics. Bottom row: Although perceptual error
metrics are effective in capturing semantic differences, they can produce anomalous results in textureless regions (e.g., green wall and
white tablecloth) due to visual perturbability (e.g., blurring) between training and rendered images. Top row: In contrast, while inherently
noisier, photometric error metrics demonstrate greater reliability in estimating the overall proportion of transient pixels within an image,
thus guiding perception error metrics for more accurate classification.
for image Ii, we reshape it into a sequence of non-
overlapping patches ˆεi ∈RNE×P 2, where (P, P) is the
patch resolution and NE = HW/P 2 is the number of
patches. The mean error of each patch forms a compact
patch-wise error sequence ¯εi ∈RNE for classification.
Classification Approaches (Percentile & GMM). After
aggregation, we can classify patches based on their total er-
rors E ∈RNI·NE using two approaches:
• Percentile-based Classification. If the static proportion T
of the scene is known, transient patches can be filtered
using a percentile threshold:
Mi (j) = I [¯εi (j) ≤percentile (E, T )] ,
(4)
where Mi (j) is the static mask of the j-th patch of the
i-th image, and I [·] is the indicator function that returns 1
if the predicate is true and 0 otherwise.
• GMM-based Classification. In practice, the exact static
proportion varies with different scenes and is hard to ob-
tain in advance. To achieve accurate distinction in various
settings, we follow prior works [24, 54] to fit the error dis-
tribution via a two-component one-dimensional Gaussian
Mixture Model (GMM). The probability density function
P (¯εi (j)) can be defined as:
P (¯εi (j)) =
2
X
k=1
βk · N
 ¯εi (j)
 µk, σ2
k

,
(5)
where βk represents the mixing component of the k-th
gaussian distribution N
 ·
 µk, σ2
k

with the mean µk and
the variance σ2
k. Let µ1 ≤µ2, we can view the first dis-
tribution as static, and then determine whether the j-th
patch of the i-th image is static by the probability that its
error value ¯εi (j) belongs to the static distribution:
Mi (j) = I
"
β1 · N
 ¯εi (j)
 µ1, σ2
1

P2
k=1 βk · N (¯εi (j) | µk, σ2
k)
≥0.5
#
. (6)
4.2. Hybrid Classification Metrics
While patch-wise classification improves spatial granular-
ity and robustness, the effectiveness of distractor removal
also depends on the reliability of the underlying classifica-
tion metric. Recent state-of-the-art methods replace tradi-
tional photometric error metrics with perceptual error met-
rics (e.g., DINOv2 [36]) to capture object-level inconsisten-
cies between reference and rendered images. As shown in
Fig. 4, these perceptual metrics offer clear advantages: they
enable image comparison at the semantic level while ex-
hibiting robustness to ambiguities at the appearance level
(e.g., objects with similar colors). Rather than contribut-
ing to this line of work by designing a better perceptual
metric, our study takes a step back to critically examine
the paradigm itself. Specifically, we identify an inherent
limitation shared by existing perceptual metrics: semantic
fragility in specific image regions. To address this chal-
lenge, we propose a novel hybrid classification metric that
synergistically combines perceptual and photometric cues,
leading to more robust and accurate static map construction.
Semantic Fragility of Perceptual Metrics. Despite their
effectiveness, perceptual metrics can still yield anoma-
lous results due to inconsistent semantic predictions of vi-
sion foundation models for minor appearance perturbations
(e.g., blurring or color jitter). Since feature extractors in
these models are optimized for object-centric understand-
ing rather than appearance consistency, their outputs are
tuned to salient object-related features, leading to unstable
activations in textureless regions. This issue is especially
pronounced in low-frequency regions during 3DGS opti-
mization, where appearance perturbations between train-
5

<!-- page 6 -->
Method
Statue
Android
Yoda
Crab (2)
Avg.
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
RobustNeRF [42]
21.56
.841
.129
24.38
.828
.088
33.72
.940
.079
32.77
.934
.077
28.11
.886
.093
NeRF-HuGS [6]
22.14
.850
.120
24.40
.834
.088
33.64
.945
.057
33.83
.939
.061
28.50
.892
.081
NeRF On-the-go [40]
22.34
.849
.109
24.42
.823
.084
30.64
.933
.068
29.65
.917
.079
26.76
.881
.085
3DGS [19]
21.54
.848
.135
23.62
.814
.097
30.34
.945
.086
30.35
.944
.084
26.46
.888
.101
MemE-3DGS [54]
22.65
.851
.096
24.14
.810
.079
31.41
.928
.115
32.05
.940
.093
27.56
.882
.096
WildGaussians [22]
23.02
.861
.127
25.15
.837
.092
31.91
.944
.111
31.42
.936
.114
27.88
.894
.111
SLS-mlp [43]
22.81
.841
.119
25.04
.834
.079
35.52
.957
.071
34.66
.953
.076
29.51
.896
.086
HybridGS [27]
22.87
.862
.104
25.14
.848
.068
35.19
.958
.073
34.94
.955
.074
29.53
.906
.080
T-3DGS [32]
22.68
.855
.107
24.67
.822
.073
33.02
.942
.095
33.11
.943
.092
28.37
.890
.092
Ours (VGG)
23.04
.878
.096
25.12
.853
.064
35.78
.962
.068
35.43
.960
.066
29.84
.913
.074
Ours (ResNet)
23.09
.879
.095
25.01
.851
.065
35.88
.962
.068
35.49
.960
.066
29.87
.913
.073
Ours (DINOv2)
23.38
.880
.094
25.02
.851
.065
35.79
.962
.068
35.07
.957
.071
29.81
.913
.074
Yoda
Statue
Ground Truth
NeRF-HuGS [6]
3DGS [19]
WildGaussians [22]
SLS-mlp [43]
HybridGS [27]
Ours (DINOv2)
Figure 6. Quantitative and Qualitative Comparisons on the RobustNeRF Dataset [42]. The first , second , and third best values are
highlighted. Quantitatively, our method with different vision models not only improves the performance of native 3DGS, but also outper-
forms previous methods on most quality metrics. Qualitatively, our method can better preserve static details while ignoring distractors.
ing and predicted images can dominate and result in ab-
normally high perceptual errors, causing static areas to be
incorrectly classified as transient (Fig. 5). Given that low-
texture regions constitute a significant portion of natural
images [4, 12, 25, 31], perceptual metrics are therefore in-
herently prone to erroneously filter out a greater number of
static regions compared to their photometric counterparts.
Hybrid Metric Formulation. We therefore propose a hy-
brid classification metric that combines perceptual and pho-
tometric cues.
In this framework, the photometric error
(processed via GMM-based classification) provides an es-
timated static pixel proportion, which guides the percentile
thresholding used in perceptual error-based classification.
By jointly measuring image differences in both color and
semantic spaces, this hybrid approach not only improves
classification accuracy but also enables adaptive control
over the static proportion within a reasonable range.
Specifically, we first obtain the photometric error map
ε(c)
i
and the perceptual error map ε(p)
i
between the rendered
image ˆIi and the training image Ii by using the RGB L1
loss and the feature cosine similarity loss, respectively:
ε(c)
i
= ∥ˆIi −Ii∥1,
(7)
ε(p)
i
= 1 −
ˆfi · fi
∥ˆfi∥2 · ∥fi∥2
,
(8)
where ˆfi and fi are feature maps extracted from vision foun-
dation models. We follow the process in Sec. 4.1 to reorga-
nize photometric error maps into the patch-wise error se-
quence, and then classify it through Eq. (6) to obtain the
static map M(c)
i
and the estimated static proportion T (c):
T (c) =
PNI
i=1
PNE
j=1 M(c)
i
(j)
NI · NE
.
(9)
Similarly, we reorganize the perceptual error maps and ob-
tain their static map M(p)
i
through Eq. (4) with T (c). The
final static map M(f)
i
can be defined as:
M(f)
i
= M(c)
i
∩M(p)
i
.
(10)
In the end, the static map M(f)
i
is applied to Eq. (3) to filter
out transient distractors during training.
6

<!-- page 7 -->
Method
Low Occlusion
Medium Occlusion
High Occlusion
Avg.
Mountain
Fountain
Corner
Patio
Spot
Patio-High
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
RobustNeRF [42]
21.37
.732
.151
20.97
.717
.191
24.30
.863
.099
21.33
.781
.127
21.83
.686
.372
19.78
.632
.317
21.60
.735
.210
NeRF-HuGS [6]
21.65
.741
.148
21.44
.732
.174
26.52
.879
.087
22.06
.812
.108
23.06
.823
.157
22.45
.776
.173
22.86
.794
.141
NeRF On-the-go [40]
21.27
.719
.148
16.11
.482
.429
25.96
.865
.086
21.47
.793
.117
24.06
.865
.088
22.25
.790
.158
21.85
.752
.171
3DGS [19]
20.79
.749
.153
21.27
.773
.157
22.77
.832
.154
17.66
.715
.204
18.60
.737
.389
18.09
.701
.296
19.86
.751
.225
MemE-3DGS [54]
20.45
.699
.173
21.70
.744
.152
25.26
.867
.081
16.81
.728
.183
23.68
.880
.121
20.68
.764
.187
21.43
.780
.149
WildGaussians [22]
20.57
.706
.224
22.33
.757
.180
26.13
.881
.102
21.82
.823
.126
24.96
.867
.121
23.29
.814
.157
23.18
.808
.152
SLS-mlp [43]
21.99
.726
.189
22.48
.760
.168
26.33
.889
.090
22.50
.828
.108
24.34
.855
.132
23.15
.805
.152
23.47
.810
.140
HybridGS [27]
21.26
.618
.348
20.68
.631
.251
24.84
.817
.119
21.90
.784
.149
23.81
.712
.191
21.59
.689
.216
22.35
.708
.212
T-3DGS [32]
20.50
.689
.195
22.45
.770
.137
25.50
.870
.081
21.99
.842
.094
24.83
.887
.116
22.43
.806
.147
22.95
.811
.128
Ours (VGG)
21.64
.778
.134
23.31
.812
.121
25.18
.898
.082
22.69
.846
.096
24.99
.902
.091
23.54
.851
.120
23.56
.848
.107
Ours (ResNet)
22.40
.786
.128
23.58
.815
.117
25.44
.899
.079
22.64
.846
.094
25.75
.904
.090
23.17
.851
.120
23.83
.850
.105
Ours (DINOv2)
22.20
.784
.129
23.40
.813
.117
25.72
.900
.077
22.67
.845
.095
25.83
.899
.096
23.36
.850
.122
23.86
.849
.106
Patio-High
Corner
Fountain
Ground Truth
NeRF-HuGS [6]
3DGS [19]
WildGaussians [22]
SLS-mlp [43]
HybridGS [27]
Ours (DINOv2)
Figure 7. Quantitative and Qualitative Comparisons on the On-the-go Dataset [40]. The first , second , and third best values are
highlighted. Our method consistently achieves top-three quantitative results in scenes with varying ratios of transient distractors. In terms
of visual quality, our method can better reconstruct static content, while previous methods suffer from missing or distorted details.
5. Experiments
5.1. Experimental Setups
Datasets. We use two public datasets in our experiments:
• RobustNeRF Dataset [42]. This real-world dataset in-
cludes 4 controlled indoor scenes, each with 1-150 com-
mon objects placed as distractors. Following the baseline
settings, we report results on the latest version [43] and
downsample images by 8x for training and evaluation.
• On-the-go Dataset [40]. This real-world dataset contains
multiple casually captured indoor and outdoor scenes,
each with varying ratios of distractors (from 5% to over
30%). Following the baseline settings, we report results
on the 6 selected scenes representing different occlusion
rates, and downsample images by 8x for training and eval-
uation (except images of Patio downsampled by 4x).
Implementation Details.
Please see the supplementary
materials for more details.
In short, we develop our
method based on open-source gsplat [62], and follow the
same model architecture and training strategies as native
3DGS [19], except that a modified loss Eq. (3) is applied
to filter out distractors. We warm up the model without ap-
plying static maps in the first 500 iterations and then up-
date static maps every 100 iterations. For patch-wise clas-
sification approaches, we set the patch size to 16 by de-
fault.
For perceptual error metrics, we separately apply
three vision foundation models with different architectures:
VGG16 [45], ResNet18 [16], and DINOv2 ViT-S/14 [36].
5.2. Evaluation on Novel View Synthesis
Baselines.
In addition to native 3DGS, we compare
our method to three state-of-the-art NeRF-based methods
and five recent 3DGS-based methods: RobustNeRF [42],
7

<!-- page 8 -->
Partition
RobustNeRF (Avg.)
On-the-go (Avg.)
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
(a) N/A
29.67
.911
.077
22.95
.834
.127
(b) Superpixel [52]
29.74
.912
.075
23.29
.846
.109
(c) SAM2 [38]
29.45
.913
.074
23.19
.844
.112
(d) Feat. Clustering [49]
29.70
.912
.075
23.71
.848
.107
(e) Patch-wise (Ours)
29.81
.913
.074
23.86
.849
.106
Statue
Image
(a)
(b)
(c)
(d)
(e)
Figure 8. Comparisons of Partitioning Strategies. Pixel-wise
method (a) produces noisy static maps, while other methods (b-d)
confuse the desktop with transient shadows. Instead, our method
(e) accurately distinguishes between transient and static elements.
NeRF-HuGS [6], NeRF On-the-go [40], MemE-3DGS [54],
WildGaussians [22], SLS-mlp [43], HybridGS [27], and T-
3DGS [32]. Among them, NeRF-HuGS, NeRF On-the-go,
WildGaussians, SLS-mlp and T-3DGS utilize the semantic
priors [21, 36, 41] to enhance the semantic consistency of
the results, while WildGaussians and T-3DGS additionally
use DINOv2 cosine error as a perceptual metric. The above
models and ours are trained on images disturbed by distrac-
tors and evaluated on clean images. We report image syn-
thesis quality based on PSNR, SSIM [56] and LPIPS [65].
Comparisons on the RobustNeRF dataset (Fig. 6). Com-
pared to native 3DGS, applying our method leads to sub-
stantial PSNR improvements of 1.39 to 5.54 dB, indicat-
ing that our method can generate high-quality static maps to
prevent native 3DGS from being affected by transient dis-
tractors. Compared to other baselines, our method achieves
almost all the best quantitative results and maintains a
good balance between ignoring transient distractors (e.g.,
pink and dark artifacts) and preserving static details (e.g.,
the mirror and wall structure). Furthermore, our method
does not show significant performance differences when us-
ing different vision foundation models, demonstrating the
adaptability and generalizability of our method.
Comparisons on the On-the-go dataset (Fig. 7).
The
results and conclusions are similar to those from the Ro-
bustNeRF dataset. Specifically, even with semantic priors,
WildGaussians and SLS-mlp still fail to eliminate distrac-
tors (e.g., temporal shadows) while incorrectly losing static
details (e.g., ground textures and the bicycle frame). In-
stead, our method achieves better visual quality by address-
ing inherent mismatches and biases in external semantics to
better separate transient distractors from static background.
5.3. Ablation Study
Based on our method with DINOv2, we test two key com-
ponents to study their effects on different datasets. More
Photometric Perceptual
Avg.
RobustNeRF (Avg.)
On-the-go (Avg.)
Static% PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
(a)
GMM
-
85.31
29.82
.912
.074
22.83
.835
.123
(b)
-
GMM
73.22
29.13
.905
.085
23.40
.834
.129
(c)
-
Percentile
86.26
29.55
.912
.074
23.31
.845
.116
(d)
GMM
GMM
72.62
29.48
.909
.083
23.06
.825
.135
(e)
GMM
Percentile
82.75
29.81
.913
.074
23.86
.849
.106
Fountain
Image
(a)
(b)
(c)
(d)
(e)
Figure 9. Comparisons of Classification Metrics. Method (a)
fails to distinguish distractors (e.g., clothing) that closely resem-
ble static backgrounds in color space, while Method (b) tends to
incorrectly remove certain static regions (e.g., wall surfaces). A
naive combination (d) does not yield meaningful improvements.
In contrast, our method (e) effectively separates transient distrac-
tors from the static background.
experiments and related analyzes can be found in the sup-
plementary material.
Partitioning Strategies. As shown in Fig. 8, method (a)
without any partitioning strategy performs the worst. Meth-
ods (b-d) rely on low-level appearance cues or deep seman-
tic priors to organize pixels, achieving suboptimal results.
In contrast, our method (e) performs the best, indicating that
the simple patch-wise strategy is sufficient for this task.
Classification Metrics.
As shown in Fig. 9, using per-
ceptual metrics alone (b) performs worse than using color
metrics alone (a) in some cases, and their difference in es-
timated static proportions is aligned with the analysis of
Sec. 4.2. Method (d), which simply combines them, cannot
achieve better performance. By using the static proportion
estimated by color metrics to guide perceptual metrics for
classification, our method (e) achieves the best results.
6. Conclusion
In conclusion, we presented Hybrid Patch-wise Classifica-
tion (HPC), a novel framework for enabling distractor-free
3D Gaussian Splatting. By combining a patch-wise clas-
sification strategy with a hybrid error metric that fuses per-
ceptual and color cues, our method effectively addresses the
limitations of existing approaches that rely on mismatched
external semantics. Extensive experiments demonstrate that
HPC significantly improves reconstruction quality and ro-
bustness in the presence of transient distractors, offering
a practical and generalizable solution for real-world 3DGS
applications. Please see the supplementary details for limi-
tations and future work.
References
[1] Eric Arazo, Diego Ortego, Paul Albert, Noel O’Connor, and
Kevin McGuinness. Unsupervised label noise modeling and
8

<!-- page 9 -->
loss correction.
In International conference on machine
learning, pages 312–321. PMLR, 2019. 2
[2] Devansh Arpit, Stanisław Jastrz˛ebski, Nicolas Ballas, David
Krueger, Emmanuel Bengio, Maxinder S Kanwal, Tegan
Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio,
et al. A closer look at memorization in deep networks. In
International conference on machine learning, pages 233–
242. PMLR, 2017. 2
[3] Yanqi Bao, Jing Liao, Jing Huo, and Yang Gao. Distractor-
free generalizable 3d gaussian splatting.
arXiv preprint
arXiv:2411.17605, 2024. 2
[4] Robert W Buccigrossi and Eero P Simoncelli. Image com-
pression via joint statistical characterization in the wavelet
domain.
IEEE transactions on Image processing, 8(12):
1688–1701, 1999. 6
[5] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou,
Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerg-
ing properties in self-supervised vision transformers. In Pro-
ceedings of the IEEE/CVF international conference on com-
puter vision, pages 9650–9660, 2021. 3
[6] Jiahao Chen, Yipeng Qin, Lingjie Liu, Jiangbo Lu, and
Guanbin Li. Nerf-hugs: Improved neural radiance fields in
non-static scenes using heuristics-guided segmentation. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 19436–19446, 2024. 2,
3, 6, 7, 8, 1
[7] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos,
Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image
segmentation with deep convolutional nets, atrous convolu-
tion, and fully connected crfs. IEEE transactions on pattern
analysis and machine intelligence, 40(4):834–848, 2017. 2
[8] Liang-Chieh Chen, George Papandreou, Florian Schroff, and
Hartwig Adam. Rethinking atrous convolution for seman-
tic image segmentation. arXiv preprint arXiv:1706.05587,
2017.
[9] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian
Schroff, and Hartwig Adam. Encoder-decoder with atrous
separable convolution for semantic image segmentation. In
Proceedings of the European conference on computer vision
(ECCV), pages 801–818, 2018. 2
[10] Xingyu Chen, Qi Zhang, Xiaoyu Li, Yue Chen, Ying Feng,
Xuan Wang, and Jue Wang. Hallucinated neural radiance
fields in the wild. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
12943–12952, 2022. 2
[11] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexan-
der Kirillov, and Rohit Girdhar.
Masked-attention mask
transformer for universal image segmentation. In Proceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 1290–1299, 2022. 2
[12] Taeg Sang Cho, Neel Joshi, C Lawrence Zitnick, Sing Bing
Kang, Richard Szeliski, and William T Freeman. A content-
aware image prior. In 2010 IEEE Computer Society Con-
ference on Computer Vision and Pattern Recognition, pages
169–176. IEEE, 2010. 6
[13] Hiba Dahmani, Moussab Bennehar, Nathan Piasco, Luis
Roldao, and Dzmitry Tsishkou. Swag: Splatting in the wild
images with appearance-conditioned gaussians. In European
Conference on Computer Vision, pages 325–340. Springer,
2024. 2
[14] Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wen-
zheng Chen, and Baoquan Chen. 4d-rotor gaussian splatting:
towards efficient novel view synthesis for dynamic scenes.
In ACM SIGGRAPH 2024 Conference Papers, pages 1–11,
2024. 2
[15] Antoine Guédon and Vincent Lepetit.
Sugar:
Surface-
aligned gaussian splatting for efficient 3d mesh reconstruc-
tion and high-quality mesh rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 5354–5363, 2024. 1
[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In Proceed-
ings of the IEEE conference on computer vision and pattern
recognition, pages 770–778, 2016. 7, 1
[17] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and
Shenghua Gao. 2d gaussian splatting for geometrically ac-
curate radiance fields. In ACM SIGGRAPH 2024 conference
papers, pages 1–11, 2024. 1
[18] Nan Huang, Xiaobao Wei, Wenzhao Zheng, Pengju An,
Ming Lu, Wei Zhan, Masayoshi Tomizuka, Kurt Keutzer,
and Shanghang Zhang.
S3gaussian:
Self-supervised
street gaussians for autonomous driving.
arXiv preprint
arXiv:2405.20323, 2024. 2
[19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics, 42
(4):1–14, 2023. 1, 2, 3, 6, 7, 5
[20] Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Wei-
wei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar,
Andrea Tagliasacchi, and Kwang Moo Yi. 3d gaussian splat-
ting as markov chain monte carlo. Advances in Neural Infor-
mation Processing Systems, 37:80965–80986, 2024. 1
[21] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,
Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-
head, Alexander C Berg, Wan-Yen Lo, et al. Segment any-
thing. In Proceedings of the IEEE/CVF international con-
ference on computer vision, pages 4015–4026, 2023. 3, 4, 8,
1
[22] Jonas Kulhanek, Songyou Peng, Zuzana Kukelova, Marc
Pollefeys, and Torsten Sattler. Wildgaussians: 3d gaussian
splatting in the wild. arXiv preprint arXiv:2407.08447, 2024.
1, 2, 3, 6, 7, 8, 5
[23] Jaewon Lee, Injae Kim, Hwan Heo, and Hyunwoo J Kim.
Semantic-aware occlusion filtering neural radiance fields in
the wild. arXiv preprint arXiv:2303.03966, 2023. 2
[24] Junnan Li, Richard Socher, and Steven CH Hoi. Dividemix:
Learning with noisy labels as semi-supervised learning.
arXiv preprint arXiv:2002.07394, 2020. 5
[25] Xin Li and Michael T Orchard. New edge-directed inter-
polation. IEEE transactions on image processing, 10(10):
1521–1527, 2001. 6
[26] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaus-
sian feature splatting for real-time dynamic view synthesis.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 8508–8520, 2024. 2
9

<!-- page 10 -->
[27] Jingyu Lin, Jiaqi Gu, Lubin Fan, Bojian Wu, Yujing Lou,
Renjie Chen, Ligang Liu, and Jieping Ye. Hybridgs: Decou-
pling transients and statics with 2d and 3d gaussian splatting.
arXiv preprint arXiv:2412.03844, 2024. 2, 6, 7, 8, 5
[28] Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu
Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Jo-
hannes Kopf, and Jia-Bin Huang. Robust dynamic radiance
fields. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 13–23, 2023. 2
[29] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin
Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d
gaussians for view-adaptive rendering.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20654–20664, 2024. 1
[30] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and
Deva Ramanan. Dynamic 3d gaussians: Tracking by per-
sistent dynamic view synthesis. In 2024 International Con-
ference on 3D Vision (3DV), pages 800–809. IEEE, 2024. 2
[31] Siwei Lyu. Automatic image orientation determination with
natural image statistics. In Proceedings of the 13th annual
ACM international conference on Multimedia, pages 491–
494, 2005. 6
[32] Alexander Markin, Vadim Pryadilshchikov, Artem Ko-
marichev, Ruslan Rakhimov, Peter Wonka, and Evgeny Bur-
naev. T-3dgs: Removing transient objects for 3d scene re-
construction. arXiv preprint arXiv:2412.00155, 2024. 2, 3,
4, 6, 7, 8, 1, 5
[33] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi,
Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duck-
worth. Nerf in the wild: Neural radiance fields for uncon-
strained photo collections. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition,
pages 7210–7219, 2021. 2, 3
[34] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. In European Conference on Computer Vision, pages
405–421. Springer, 2020. 2, 3
[35] Daniel Müllner. Modern hierarchical, agglomerative cluster-
ing algorithms. arXiv preprint arXiv:1109.2378, 2011. 4,
1
[36] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al.
Dinov2: Learning robust visual features without supervision.
arXiv preprint arXiv:2304.07193, 2023. 3, 5, 7, 8, 1
[37] Wongi Park, Myeongseok Nam, Siwon Kim, Sangwoo
Jo, and Soomok Lee.
Forestsplats: Deformable transient
field for gaussian splatting in the wild.
arXiv preprint
arXiv:2503.06179, 2025. 2, 3, 1
[38] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang
Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman
Rädle, Chloe Rolland, Laura Gustafson, et al.
Sam 2:
Segment anything in images and videos.
arXiv preprint
arXiv:2408.00714, 2024. 3, 4, 8, 1
[39] Konstantinos
Rematas,
Andrew
Liu,
Pratul
P
Srini-
vasan, Jonathan T Barron, Andrea Tagliasacchi, Thomas
Funkhouser, and Vittorio Ferrari. Urban radiance fields. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 12932–12942, 2022. 2
[40] Weining Ren, Zihan Zhu, Boyang Sun, Jiaqi Chen, Marc
Pollefeys, and Songyou Peng. Nerf on-the-go: Exploiting
uncertainty for distractor-free nerfs in the wild. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 8931–8940, 2024. 2, 3, 6, 7, 8
[41] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Björn Ommer.
High-resolution image
synthesis with latent diffusion models.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 10684–10695, 2022. 3, 8, 1
[42] Sara Sabour, Suhani Vora, Daniel Duckworth, Ivan Krasin,
David J Fleet, and Andrea Tagliasacchi.
Robustnerf: Ig-
noring distractors with robust losses.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 20626–20636, 2023. 2, 3, 4, 6, 7
[43] Sara Sabour, Lily Goli, George Kopanas, Mark Matthews,
Dmitry Lagun, Leonidas Guibas, Alec Jacobson, David J
Fleet, and Andrea Tagliasacchi.
Spotlesssplats:
Ignor-
ing distractors in 3d gaussian splatting.
arXiv preprint
arXiv:2406.20055, 2024. 1, 2, 3, 6, 7, 8, 5
[44] Nicolas Schischka, Hannah Schieber, Mert Asim Karaoglu,
Melih Gorgulu, Florian Grötzner, Alexander Ladikos, Nas-
sir Navab, Daniel Roth, and Benjamin Busam. Dynamon:
Motion-aware fast and robust camera localization for dy-
namic neural radiance fields. IEEE Robotics and Automation
Letters, 2024. 2
[45] Karen Simonyan and Andrew Zisserman. Very deep convo-
lutional networks for large-scale image recognition. arXiv
preprint arXiv:1409.1556, 2014. 7, 1
[46] Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin,
and Jae-Gil Lee. Learning from noisy labels with deep neural
networks: A survey. IEEE transactions on neural networks
and learning systems, 34(11):8135–8153, 2022. 2
[47] Jiaming Sun, Xi Chen, Qianqian Wang, Zhengqi Li, Hadar
Averbuch-Elor, Xiaowei Zhou, and Noah Snavely. Neural
3d reconstruction in the wild.
In ACM SIGGRAPH 2022
conference proceedings, pages 1–9, 2022. 2
[48] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Prad-
han, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron,
and Henrik Kretzschmar. Block-nerf: Scalable large scene
neural view synthesis. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
8248–8258, 2022. 2
[49] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng
Phoo, and Bharath Hariharan.
Emergent correspondence
from image diffusion. Advances in Neural Information Pro-
cessing Systems, 36:1363–1389, 2023. 3, 4, 8, 1
[50] Haithem Turki,
Deva Ramanan,
and Mahadev Satya-
narayanan.
Mega-nerf:
Scalable construction of large-
scale nerfs for virtual fly-throughs.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 12922–12931, 2022. 2
[51] Paul Ungermann, Armin Ettenhofer, Matthias Nießner, and
Barbara Roessle.
Robust 3d gaussian splatting for novel
10

<!-- page 11 -->
view synthesis in presence of distractors.
arXiv preprint
arXiv:2408.11697, 2024. 2
[52] Michael Van den Bergh, Xavier Boix, Gemma Roig, and Luc
Van Gool. Seeds: Superpixels extracted via energy-driven
sampling. International Journal of Computer Vision, 111:
298–314, 2015. 4, 8, 1
[53] Yihao Wang, Marcus Klasson, Matias Turkulainen, Shuzhe
Wang, Juho Kannala, and Arno Solin.
Desplat: Decom-
posed gaussian splatting for distractor-free rendering. arXiv
preprint arXiv:2411.19756, 2024. 2
[54] Yukun Wang, Kunhong Li, Minglin Chen, Longguang Wang,
Shunbo Zhou, Kaiwen Xue, and Yulan Guo. Distractor-free
novel view synthesis via exploiting memorization effect in
optimization. In European Conference on Computer Vision,
pages 477–493. Springer, 2024. 2, 4, 5, 6, 7, 8
[55] Yuze Wang, Junyi Wang, and Yue Qi. We-gs: An in-the-wild
efficient 3d gaussian representation for unconstrained photo
collections. arXiv preprint arXiv:2406.02407, 2024. 2
[56] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
moncelli. Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing,
13(4):600–612, 2004. 8
[57] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng
Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang.
4d gaussian splatting for real-time dynamic scene rendering.
In Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 20310–20320, 2024. 2
[58] Congrong
Xu,
Justin
Kerr,
and
Angjoo
Kanazawa.
Splatfacto-w:
A nerfstudio implementation of gaussian
splatting for unconstrained photo collections. arXiv preprint
arXiv:2407.12306, 2024. 2
[59] Yifan Yang, Shuhai Zhang, Zixiong Huang, Yubing Zhang,
and Mingkui Tan. Cross-ray neural radiance fields for novel-
view synthesis from unconstrained image collections.
In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 15901–15911, 2023. 2
[60] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang.
Real-time photorealistic dynamic scene representation and
rendering with 4d gaussian splatting.
arXiv preprint
arXiv:2310.10642, 2023. 2
[61] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing
Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-
fidelity monocular dynamic scene reconstruction.
In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 20331–20341, 2024. 2
[62] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen,
Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey
Hu, Matthew Tancik, and Angjoo Kanazawa.
gsplat: An
open-source library for Gaussian splatting. arXiv preprint
arXiv:2409.06765, 2024. 7, 1
[63] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and
Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splat-
ting. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 19447–19456,
2024. 1
[64] Dongbin Zhang, Chuming Wang, Weitao Wang, Peihao Li,
Minghan Qin, and Haoqian Wang. Gaussian in the wild: 3d
gaussian splatting for unconstrained image collections. In
European Conference on Computer Vision, pages 341–359.
Springer, 2024. 2
[65] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 586–595, 2018. 8, 1
11

<!-- page 12 -->
3DGS-HPC: Distractor-free 3D Gaussian Splatting
with Hybrid Patch-wise Classification
Supplementary Material
Table 1. Settings for Our Perceptual Error Metrics.
Vision Model
Extracted Feature Layers
VGG-16 [45]
{conv1_2, conv2_2, conv3_3, conv4_3, conv5_3}
ResNet-18 [16]
{layer1, layer2, layer3, layer4}
DINOv2 ViT-S/14 [36]
{layer2, layer5, layer8, layer11}
Table 2. Performance of Our Patch-wise Classification vs. Its
Refinement Variant. Our method can significantly reduce train-
ing time and have sightly better performance in visual metrics.
RobustNeRF (Avg.)
On-the-go (Avg.)
PSNR↑SSIM↑LPIPS↓Train hrs.↓PSNR↑SSIM↑LPIPS↓Train hrs.↓
(a) Refinement Variant
29.74
.912
.074
2.53
23.69
.848
.111
2.09
(b) Ours
29.81
.913
.074
0.22
23.86
.849
.106
0.25
7. Implementation & Experimental Details
3DGS. As mentioned in Sec. 5.1 of the main paper, we
develop our model based on the open-source codebase
gsplat [62], which reproduces native 3DGS [19] and aligns
its performance. We follow the same training settings and
adaptive density control strategy as the native model, except
that we do not remove large-scale Gaussians during peri-
odic pruning, which would otherwise result in some scene
backgrounds represented by large-scale Gaussians having
degraded geometry and appearance. All experiments of our
method are conducted on an NVIDIA RTX 4090 GPU.
Update of Static Maps. Similar to prior works [22, 32], we
do not apply or update static maps in the first 500 steps to
warm up the 3DGS optimization and ensure the initializa-
tion of the scene reconstruction. We then apply static maps
in the remaining steps and update them every 100 steps by
rendering all training images with the current 3DGS. Since
3DGS resets the opacity every 3,000 steps during training,
we only apply but do not update static maps within 200
steps after each opacity reset.
Perceptual Error Metrics. Inspired by LPIPS [65], we
implement our perceptual metrics by extracting multi-layer
features from vision foundation models. Since feature maps
from different layers have varying spatial resolutions, we
first compute cosine error maps at each layer and then up-
sample them to the input resolution using bilinear interpo-
lation. The final error map is obtained by averaging these
aligned maps. The vision models and their specific layers
used for feature extraction are summarized in Tab. 1.
Partitioning Strategies. We implement different partition-
ing strategies for comparison:
• Superpixel. Following ForestSplats [37], we leverage the
superpixel algotithm SEEDS [52] from the OpenCV li-
Table 3. Quantitative Efficiency Comparisons between Differ-
ent Methods. Our method outperforms most other methods in
training time and memory usage.
Avg. Train Time (min)↓Avg. GPU Mem (GB)↓
SLS-mlp [43]
11.91
3.60
WildGaussians [22]
24.10
16.02
T-3DGS [32]
26.75
4.36
Ours (VGG)
16.35
2.09
Ours (ResNet)
13.29
1.98
Ours (DINOv2)
17.52
2.05
brary and set the superpixel number of each image to 100
by default.
• SAM2. The Segment Anything Model 2 (SAM2) [38] is a
foundataion model extended from SAM [21] for prompt-
able visual segmentation in images and videos. Following
NeRF-HuGS [6], we take a regular point grid with a res-
olution of 64 × 64 as input prompt to generate instance
masks for each training image.
• Feature Clustering. Following SpotLessSplats [43], we
execute agglomerative clustering [35] on the feature
map from the 2nd upsampling layer of Stable Diffusion
v2.1 [41, 49], and set the cluster number of each image to
100 by default.
8. Additional Experiments
8.1. Path-wise Classification Implementation
In Sec. 4.1 of the main paper, we propose to classify patches
directly based on their average errors rather than using them
to refine results after per-pixel classification. As shown in
Tab. 2, when the patch size is set to 16, our proposed method
significantly reduces training time due to the reduction in
the number of classification samples and performs slightly
better in visual metrics. We therefore uniformly use our
proposed patch-wise classification implementation in other
experiments for efficiency.
8.2. Efficiency Comparisons of Different Methods
As shown in Tab. 3, we additionally provide quantitative
efficiency comparisons of different methods across all the
scenes used in the main paper. Specifically, SLS-mlp [43]
uses semantic priors as model input, while WildGaus-
sians [22] and T-3DGS [32] utilize perceptual error met-
rics. Compared to these methods, Ours (ResNet) achieves
the smallest GPU memory usage and is only slower than
SLS-mlp in average training time.
1

<!-- page 13 -->
Corner
Android
Ground Truth
WildGaussians [22]
SLS-mlp [43]
Ours (DINOv2)
Ours (DINOv2)†
Figure 10. Comparisons on Challenging Cases. Our method achieves better visual quality in most regions (e.g., walls and floors) with
default settings, while it can further capture frequently occluded details (e.g., shoes and plaid patterns) by increasing warm-up steps (†).
Input Image
Patch = 1
Patch = 4
Patch = 16
Patch = 64
Figure 11. Performance of Different Patch Sizes. Patch sizes
that are extremely small (< 8) or large (> 16) will lead to subop-
timal quantitative results, and the resulting mask will fail to accu-
rately label and filter out transient distractors (e.g., people).
8.3. More Results on Android & Corner Scenes
As shown in Figs. 6 and 7 of the main paper, our method
outperforms previous methods in most visual metrics across
various scenes.
However, it brings less improvement in
PSNR on Android and Corner scenes compared to others.
We ascribe this to a longstanding challenge in distractor-
free NeRF/3DGS methods: the inherent ambiguity between
infrequently observed static objects and genuinely transient
ones. Since transient objects are typically identified based
on reconstruction errors arising from their limited presence
in the training images, rarely visible static content may be
Ground Truth
Warm = 0
Warm = 500 Warm = 1,500 Warm = 2,500
Figure 12. Performance of Different Warm-up Steps. When
the number of warm-up steps exceeds 1, 000, the model is at risk
of “memorizing” transient distractors, leading to artifacts in the
rendered results.
mistakenly classified as transient. Nevertheless, our method
can mitigate this issue and improve PSNR by straightfor-
ward hyperparameter tuning (i.e., the warm-up steps) to pre-
serve more static details (Fig. 10). Please see Fig. 15 for
additional qualitative results.
8.4. Choice of Hyperparameters
Our method has two key hyperparameters: i) the patch size
for the patch-wise partitioning strategy, and ii) the warm-up
step in the 3DGS optimization process. Following the main
paper, we conduct experiments to investigate their choices.
2

<!-- page 14 -->
Method
Brandenburg Gate
Sacre Coeur
Trevi Fountain
Avg.
PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓PSNR↑SSIM↑LPIPS↓
3DGS [19]
21.20
.889
.131
16.73
.833
.166
17.66
.729
.210
18.53
.817
.169
NeRF-HuGS [6]
24.69
.890
.132
21.09
.819
.169
22.63
.748
.195
22.80
.819
.165
WildGaussians [22]
27.77
.927
.133
22.56
.859
.177
23.63
.766
.228
24.65
.851
.179
Ours
20.76
.879
.139
16.56
.822
.177
17.31
.691
.261
18.21
.797
.193
Ours w/ Embed.
27.90
.932
.094
22.68
.873
.126
22.35
.785
.176
24.31
.863
.132
Brandenburg Gate
Training Images
Ground Truth
WildGaussians [22]
Ours
Ours w/ Embed.
Figure 13. Comparisons on the Phototourism Dataset [33]. The first , second , and third best values are highlighted. Our extended
method can easily adapt to scenes with appearance variations by assigning learnable embeddings to each image and Gaussian.
Ground Truth
3DGS [19]
WildGaussians [22]
SLS-mlp [43]
Ours (DINOv2)
Figure 14. A Failure Case. Without prior knowledge from the generative model, 3DGS and its variants cannot correctly predict the scene
content that has never been observed before.
Patch Size. As shown in Fig. 11, the average PSNRs and
SSIMs of our method using different perceptual error met-
rics first increase and then decrease as the patch size grad-
ually increases on both the RobustNeRF and On-the-go
datasets. This indicates that the best batch size is achieved
by striking a balance between maintaining local spatial con-
sistency and flexibly representing transient distractors. Nev-
ertheless, our method consistently achieves better perfor-
mance than pixel-wise classification (i.e., Patch Size = 1)
even without carefully choosing the patch size, consistent
with the results of ablation studies (Sec. 5.3).
Warm-up Step. The warm-up phase aims to prevent static
details from being excessively removed during early train-
ing, as the model will underfit the scene and have large er-
rors when initialized. As shown in Fig. 12, the use of too
few warm-up steps often degrades performance as it unnec-
essarily removes static details, while the use of too many
steps causes 3DGS to overfit to transient distractors. We
therefore set the default warm-up step to 500.
8.5. Extension to Scenes w/ Appearance Variations
Our method is designed for scenes with stable appear-
ances, but can extend naturally to non-static appearances
caused by diverse shooting environments and shooting
equipment (e.g., different weather changes or inconsistent
white balances). Specifically, we apply a widely adopted
paradigm [6, 22, 33] to our method, which assigns learn-
able appearance embeddings to each image and Gaussian.
When rendering a specific image, each Gaussian embed-
ding is combined with the image embedding and processed
by a lightweight MLP, thereby scaling and biasing the cor-
responding Gaussian colors to achieve color changes.
We
conduct
an
experiment
on
the
Phototourism
Dataset [33], which includes three scenes of cultural land-
marks with varying appearances and various distractors. As
shown in Fig. 13, our extended method can capture ap-
pearance variations compared to the original method, and
achieves quantitative performance comparable to previous
baseline methods, leading in SSIM and LPIPS metrics. In
terms of visual quality, our extended method effectively re-
duces artifacts caused by appearance variations compared
to the original method, resulting in cleaner rendered results.
9. Limitations and Future Work
As mentioned above, despite the superiority of our method,
it still faces a longstanding challenge in distractor-free
NeRF/3DGS: reliably distinguishing between transient dis-
tractors and infrequently visible static elements.
These
static elements can be highly reflective objects whose ap-
pearance changes drastically under limited views (e.g., the
3

<!-- page 15 -->
ground reflection in Fig. 10), or objects that are rarely ob-
served (e.g., black shoes in Fig. 10). In such cases, these
static regions may be mistakenly treated as transient ele-
ments due to insufficient visibility. Therefore, as long as the
reconstruction error remains the primary criterion for distin-
guishing transients, this challenge is likely to persist. Ad-
dressing it may require a paradigm shift that could offer sub-
stantial benefits to the community. In addition, our method
inherits the limitations of native 3DGS and its variants, par-
ticularly the inability to reconstruct unobserved scene con-
tent (Fig. 14). Future work may address this limitation by
incorporating pre-trained generative models.
10. Additional Qualitative Results
10.1. Evaluation on Novel View Synthesis
We provide more qualitative results of view synthesis on
different datasets. As shown in Fig. 15, existing methods
exhibit limitations in managing transient distractors, while
our method effectively balances the exclusion of transient
distractors and retention of static details, thereby enhancing
the rendering quality. Please zoom in for better details.
10.2. Evaluation on Static Map Generation
We also provide more qualitative results of static maps. As
shown in Figs. 16 and 17, existing methods that leverage
semantic priors still exhibit limited efficacy in separating
transient distractors from static backgrounds. Conversely,
our proposed framework can generate more accurate static
maps by addressing the inherent limitations of semantics.
4

<!-- page 16 -->
Spot
Patio
Mountain
Crab (2)
Android
Ground Truth
3DGS [19]
WildGaussians [22]
SLS-mlp [43]
HybridGS [27]
T-3DGS [32]
Ours (DINOv2)
Figure 15. Qualitative Results of View Synthesis on Different Datasets.
5

<!-- page 17 -->
<———————– Crab (2) ———————-> <————————- Yoda ————————> <———————– Android ———————-> <———————— Statue ————————>
Training Image
NeRF-HuGS [6]
WildGaussians [22]
SLS-mlp [43]
Ours (DINOv2)
Figure 16. Qualitative Results of Static Map Generation on the RobustNeRF Dataset.
6

<!-- page 18 -->
<————————- Spot ————————> <—————– Patio —————–> <———————— Corner ———————–> <———————- Fountain ———————->
Training Image
NeRF-HuGS [6]
WildGaussians [22]
SLS-mlp [43]
Ours (DINOv2)
Figure 17. Qualitative Results of Static Map Generation on the On-the-go Dataset.
7
