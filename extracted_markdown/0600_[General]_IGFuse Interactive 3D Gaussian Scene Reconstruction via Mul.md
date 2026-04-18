<!-- page 1 -->
IGFuse: Interactive 3D Gaussian Scene Reconstruction via Multi-Scans Fusion
Wenhao Hu1,4∗, Zesheng Li3, Haonan Zhou2, Liu Liu4, Xuexiang Wen2, Zhizhong Su4,
Xi Li1, Gaoang Wang1,2†
1College of Computer Science and Technology, Zhejiang University
2ZJU-UIUC Institute, Zhejiang University
3Nanyang Technological University
4Horizon Robotics
Multi-Scan
Fusion
Scene Decomposition
Optimized Objects
Novel State
Novel State 1
Novel State M
Optimized Background
Multi-Scans
Object Rearrangement
Scan 1
Scan N
Figure 1: Given multiple observed scene scans, we perform multi-state optimization to jointly reconstruct consistent Gaussian
fields. The scene is then decomposed into objects and background, which are jointly represented and constrained across scans.
This enables the interactive generation of new scene states with coherent object compositions and realistic rendering.
Abstract
Reconstructing complete and interactive 3D scenes remains a
fundamental challenge in computer vision and robotics, par-
ticularly due to persistent object occlusions and limited sen-
sor coverage. Multi-view observations from a single scene
scan often fail to capture the full structural details. Exist-
ing approaches typically rely on multi-stage pipelines—such
as segmentation, background completion, and inpainting—or
require per-object dense scanning, both of which are error-
prone, and not easily scalable. We propose IGFuse, a novel
framework that reconstructs interactive Gaussian scene by
fusing observations from multiple scans, where natural object
rearrangement between captures reveal previously occluded
regions. Our method constructs segmentation-aware Gaus-
sian fields and enforces bi-directional photometric and se-
mantic consistency across scans. To handle spatial misalign-
ments, we introduce a pseudo-intermediate scene state for
unified alignment, alongside collaborative co-pruning strate-
gies to refine geometry. IGFuse enables high-fidelity render-
ing and object-level scene manipulation without dense obser-
vations or complex pipelines. Extensive experiments validate
the framework’s strong generalization to novel scene configu-
rations, demonstrating its effectiveness for real-world 3D re-
construction and real-to-simulation transfer. Our project page
*This work was done during an internship at Horizon Robotics.
†Corresponding author.
is available at https://whhu7.github.io/IGFuse
1
Introduction
Reconstructing interactive 3D scenes from partially ob-
served environments remains a core challenge in vision and
robotics (Zhu et al. 2024; Wang et al. 2024; Pang et al.
2025; Mendonca, Bahl, and Pathak 2023). Recent advances
in 3D Gaussian Splatting (Kerbl et al. 2023) have enabled
explicit scene representations by modeling geometry and
appearance using compact Gaussian primitives. Some ap-
proaches, such as Gaussian Grouping (Ye et al. 2023) and
DecoupledGaussian (Wang et al. 2025), aim to support in-
teractive scene reconstruction by combining instance-level
segmentation with inpainting-based refinement. While par-
tially effective, these multi-stage pipelines face several chal-
lenges. Feature-based segmentation often produces inaccu-
racies—especially near object boundaries and occluded re-
gions—resulting in misclassified Gaussians, and visual arti-
facts. These issues require additional post-processing, which
increases system complexity. Furthermore, inpainting meth-
ods frequently fail to recover fine background details, lead-
ing to unrealistic or blurry reconstructions. These limita-
tions compromise the overall fidelity and consistency of the
reconstructed scene and reduce the system’s reliability in
arXiv:2508.13153v1  [cs.CV]  18 Aug 2025

<!-- page 2 -->
Composition
Target Gaussian
Object 
Gaussian
Object 
Gaussian
（b）multiple object with  
dense observations
Background 
Gaussian
Segmented
Gaussian Field
Post-processing
Inpainting
Target Gaussian
（a）single scan with  
multi-stage processing
Target Gaussian
Optimized  
Gaussian
Segmented 
Gaussian Field
（c）Our end to end multi-scans 
gaussian fusion
Optimization
Segmented 
Gaussian Field
Figure 2: Comparison of different paradigms for constructing interactive 3D Gaussian. (a) Traditional single-scan pipelines
rely on multi-stage post-processing and inpainting, which may introduce accumulated artifacts. (b) Object-centric approaches
require dense multi-view observations of all components, followed by explicit composition. (c) Our proposed end-to-end multi-
scans fusion model jointly optimizes multi-state Gaussian fields via cross-state supervision, effectively compensating for oc-
clusions across different observations and enabling interactive Gaussian reconstruction.
downstream applications involving object-level understand-
ing or manipulation.
In parallel, recent research has explored integrating 3D
Gaussian Splatting into interactive and physically grounded
simulation frameworks (Barcellona et al. 2024; Yu et al.
2025; Yang et al. 2025; Lou et al. 2024; Han et al. 2025;
Zhu et al. 2025b). Methods such as RoboGSim (Li et al.
2024b) and SplatSim (Qureshi et al. 2024) leverage Gaus-
sian representations to construct photorealistic virtual envi-
ronments from real-world observations. However, these ap-
proaches typically depend on dense multi-view object cap-
tures to achieve high-fidelity reconstructions, which limits
scalability in practical scenarios.
To address these limitations, we propose leveraging mul-
tiple observations of the same scene captured under natural
object rearrangements caused by human interactions. These
interaction-driven scene states expose previously occluded
areas and implicitly provide geometric cues for refining seg-
mentation and structure. Motivated by these insights, we in-
troduce IGFuse, a novel framework for reconstructing in-
teractive 3D scenes by fusing observations across multiple
scans. Our method constructs segmentation-aware Gaussian
fields for each scan and jointly optimizes them by enforc-
ing bi-directional photometric and semantic consistency. To
align scans captured under different scene layouts, we in-
troduce a pseudo scene state that serves as a intermedi-
ate reference frame. Additionally, we design collaborative
co-pruning strategies to suppress misaligned or inconsistent
Gaussians and enhance geometric completeness.
IGFuse enables high-fidelity rendering and object-level
scene manipulation—without requiring dense view cap-
tures, or multi-stage pipelines. Our framework generalizes
well to novel rearranged scene states, offering a scalable and
robust solution for 3D scene reconstruction in interactive en-
vironments. In summary, our main contributions are:
• We propose IGFuse, a framework for interactive 3D
scene reconstruction from multi-scan observations driven
by real-world object rearrangements.
• We construct segmentation-aware Gaussian fields and
enforce bi-directional photometric and semantic consis-
tency across scans to jointly complete the scene.
• We introduce a pseudo-intermediate Gaussian state for
unified alignment across perturbed scene configurations,
improving fusion quality and geometric coherence.
2
Related Works
2.1
3D Gaussian Segmentation
Recent methods have extended Gaussian Splatting to per-
form scene segmentation (Zhu et al. 2025a; Hu et al. 2025,
2024). GaussianEditor (Chen et al. 2024) projects 2D seg-
mentation masks onto 3D Gaussians via inverse rendering.
Gaussian Grouping (Ye et al. 2023) attaches segmentation
features to each Gaussian and aligns multi-view IDs using
video segmentation (Cheng et al. 2023), while Gaga (Lyu
et al. 2024) resolves cross-view inconsistencies via a 3D-
aware memory bank. FlashSplat (Shen, Yang, and Wang
2024) proposes a fast, globally optimal LP-based segmen-
tation method. OpenGaussian (Wu et al. 2024b) and In-
stanceGaussian (Li et al. 2024a) use contrastive learning for
point-level segmentation. GaussianCut (Jain, Mirzaei, and
Gilitschenski 2024) formulates a graph-cut optimization to
separate foreground and background. COB-GS (Zhang et al.

<!-- page 3 -->
2025) improves boundary precision via adaptive splitting
and visual refinement.
However, 3D segmentation alone is insufficient for inter-
active reconstruction, as 2D biases often result in flawed 3D
masks. This necessitates post-processing and inpainting (Liu
et al. 2024; Cao et al. 2024a; Huang, Chou, and Wang
2025) to fill gaps caused by object movement—leading to
a complex and error-prone pipeline. In contrast, our method
fuses multi-scan observations under varied configurations
to achieve mutual visibility and end-to-end reconstruction
without explicit inpainting. Object transitions help calibrate
segmentation errors, producing clean and consistent 3D
Gaussians suited for interaction tasks.
2.2
Interactive Scene Reconstruction
Some approaches simulate real-world interactions by con-
structing implicit generative models from video data.
UniSim (Yang et al. 2023) predicts visual outcomes condi-
tioned on diverse actions using an autoregressive framework
over heterogeneous datasets. iVideoGPT (Wu et al. 2024a)
encodes observations, actions, and rewards into token se-
quences for scalable next-token prediction via compressive
tokenization. However, these methods often lack 3D and
physical consistency and are generally difficult to train. Re-
cent work focuses on enabling interactive simulators by
integrating reconstructed real scenes into physics engines.
RoboGSim (Li et al. 2024b) embeds 3D Gaussians into
Isaac Sim. SplatSim (Qureshi et al. 2024) replaces meshes
with Gaussian splats for photorealistic rendering. PhysGaus-
sian (Xie et al. 2024) and Spring-Gaus (Zhong et al. 2024)
enable mesh-free physical simulation using Newtonian or
elastic models. NeuMA (Cao et al. 2024b) refines simulation
using image-space gradients. However, these methods typi-
cally rely on dense, per-object 3D capture. In contrast, our
method is more lightweight and scalable—requiring only a
few multi-scan observations under varying scene configura-
tions.
3
Method
3.1
Preliminary
Segmented Gaussian Splatting (Ye et al. 2023) models a
scene as a set of 3D Gaussians, each parameterized as G =
{x, Σ, α, c, s}, where x denotes the 3D center position, Σ
represents the spatial covariance matrix, α is the opacity co-
efficient, c is the RGB color vector, and s is a learnable fea-
ture vector used for segmentation.
During rendering, each Gaussian is projected onto the 2D
image plane using a differentiable α-blending mechanism.
Both the final pixel color C and segmentation feature S are
computed by accumulating Gaussian contributions weighted
by their projected opacities α′
i:
C =
X
i∈N
ciα′
i
i−1
Y
j=1
(1 −α′
j),
S =
X
i∈N
siα′
i
i−1
Y
j=1
(1 −α′
j)
(1)
3.2
Modeling from Multi-Scan Observations
Given a set of scans X1, X2, . . . , XN, where each scan Xi =
(Ii, Si) contains image observations Ii and segmentation
masks Si captured under different object configurations, our
goal is to fuse multi-scan observations and construct an in-
teractive 3D scene representation. This representation sup-
ports realistic rendering under interaction signals, where ar-
bitrary object movements produce plausible and consistent
results.
To achieve this, we treat each scan as a discrete scene state
and construct a corresponding segmentation-aware Gaussian
field Gi, where i ∈1, 2, . . . , N. These Gaussian fields en-
code geometry, appearance, and segmentation under differ-
ent object layouts. The differences across fields {G1, ..., GN}
reflect object-level interactions and structural changes in the
scene.
To integrate information across scans, we adopt a training
strategy that randomly samples a pair (Gi, Gj) in each epoch.
Using known rigid object transformations, we align the pair
and fuse their information by enforcing bi-directional photo-
metric and semantic consistency. This enables mutual super-
vision, helping to refine occlusion-prone regions and correct
segmentation errors. The fusion process is formulated as a
joint optimization:
(G∗
i , G∗
j ) = arg min
Gi,Gj Ljoint
(2)
Given the optimized fields and transformation T, we syn-
thesize a new interactive scene configuration Gt through ex-
plicit Gaussian transformation:
{G∗
i , G∗
j }
T−→Gt
(3)
By jointly optimizing over scan pairs and explicitly mod-
eling object-level transformation, our framework constructs
a coherent and manipulable 3D Gaussian scene representa-
tion without relying on dense captures or multi-stage post-
processing.
3.3
Gaussian State Transfer
To model scene-level transformations, the operator T is de-
fined as an object-aware function that applies per-Gaussian
rigid transformations based on semantic identity. Let the
Gaussian field be decomposed into foreground and back-
ground subsets:
G = Gfg ∪Gbg,
Gfg =
O
[
o=1
G(o)
fg
(4)
where each foreground object o is associated with a rigid
transformation T(o). For any Gaussian gi ∈G, let oi denote
the object to which it belongs. Then, T is applied as:
T(gi) =
T(oi) · gi,
if gi ∈Gfg
gi,
if gi ∈Gbg
(5)
This formulation ensures spatially consistent transformation
and geometric fidelity of object-level transformation while
preserving the static background.

<!-- page 4 -->
𝐺𝐺i = {𝑥𝑥i, Σ𝑖𝑖, 𝛼𝛼𝑖𝑖, 𝑐𝑐𝑖𝑖, 𝑠𝑠𝑖𝑖}
𝐺𝐺i→𝑝𝑝= {𝑥𝑥𝑖𝑖
′, Σ𝑖𝑖
′, 𝛼𝛼𝑖𝑖, 𝑐𝑐𝑖𝑖, 𝑠𝑠𝑖𝑖}
𝐺𝐺𝑖𝑖→𝑗𝑗= {𝑥𝑥𝑖𝑖
′′, Σ𝑖𝑖
′′, 𝛼𝛼𝑖𝑖, 𝑐𝑐𝑖𝑖, 𝑠𝑠𝑖𝑖}
𝐺𝐺𝑗𝑗= {𝑥𝑥𝑗𝑗, Σ𝑗𝑗, 𝛼𝛼𝑗𝑗, 𝑐𝑐𝑗𝑗, 𝑠𝑠𝑗𝑗}
𝐺𝐺𝑗𝑗→𝑝𝑝= {𝑥𝑥𝑗𝑗
′, Σ𝑗𝑗
′, 𝛼𝛼𝑗𝑗, 𝑐𝑐𝑗𝑗, 𝑠𝑠𝑗𝑗}
𝐺𝐺j→𝑖𝑖= {𝑥𝑥𝑗𝑗
′′, Σ𝑗𝑗
′′, 𝛼𝛼𝑗𝑗, 𝑐𝑐𝑗𝑗, 𝑠𝑠𝑗𝑗}
Scan j
𝐼𝐼𝐼𝐼𝐼𝐼𝐼𝐼𝐼𝐼𝑆𝑆𝑆𝑆𝑆𝑆𝐼𝐼𝑗𝑗
𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑗𝑗
Scan i
𝐼𝐼𝐼𝐼𝐼𝐼𝐼𝐼𝐼𝐼𝑆𝑆𝑆𝑆𝑆𝑆𝐼𝐼𝑖𝑖
𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑆𝑖𝑖
Gaussian State i
Pseudo-state
Gaussian State Transfer
Gaussian State j
Gaussian State Transfer
Collaborative Co-Pruning
Collaborative Co-Pruning
ℒ𝑝𝑝seudo
ℒalign
ℒalign
ℒr
ℒr
Figure 3: Overview of our dual-state Gaussian alignment pipeline. Given two input scans (scan i and scan j), the Gaussians in
state i are initially constrained by corresponding image observations. After transferring to state j (i.e., Gi →Gi→j), the Gaus-
sians are further supervised by state j’s image via an alignment loss Lalign, and regularized through a co-pruning strategy that
enforces 3D consistency by removing mismatched or redundant components. The reverse transfer (Gj →Gj→i) is performed
symmetrically. Additionally, both states are transferred into a shared pseudo-state space (Gi→p, Gj→p), where a pseudo-state
loss Lpseudo encourages tighter cross-state alignment.
3.4
Bidirectional Alignment
To ensure geometric and semantic consistency across differ-
ent scene states, we enforce that the rendered outputs from
transformed Gaussian fields align with the ground-truth ob-
servations in the corresponding target states. As mentioned
before, we apply transformation Ti→j to Gi and transfor-
mation Tj→i to Gj. For any viewpoint v, the transformed
Gaussian fields are rendered into RGB images and segmen-
tation masks, which are then compared with the correspond-
ing ground-truth observations (Iv
i , Sv
i ) and (Iv
j , Sv
j ) from
the original states. The total alignment loss combines photo-
metric and segmentation consistency, defined as:
Lalign(Gi, Gj, θ) =
R (Ti→j(Gi), v) −Iv
j

1
+ ∥R (Tj→i(Gj), v) −Iv
i ∥1
+ CE
 fθ (M (Ti→j(Gi), v)) , Sv
j

+ CE (fθ (M (Tj→i(Gj), v)) , Sv
i ) (6)
where R(·, v) and M(·, v) denote the rendering functions
that generate the RGB image and segmentation feature from
viewpoint v, as defined in Equation 1. The segmentation out-
put is obtained via a shared classifier fθ, which is jointly ap-
plied to both Gi and Gj. Specifically, fθ consists of a linear
layer that projects each identity embedding to a (K + 1)-
dimensional space, where K is the number of instance
masks in the 3D scene (Ye et al. 2023). The cross-entropy
loss CE(·, ·) measures the semantic alignment between pre-
dicted and ground-truth masks.
This bidirectional consistency encourages each trans-
formed Gaussian field to accurately reconstruct the scene
content of the opposite state, thereby reinforcing object-level
correspondence and enhancing alignment across different
scene configurations.
3.5
Pseudo-state Guided Alignment
To enhance the generalizability of the interactive Gaussian
across diverse scene configurations, we introduce a pseudo-
state Gp that serves as an intermediate reference for supervi-
sion. This pseudo-state is constructed by applying geometric
constraints, such as collision and boundary regularization, to
synthesize a virtual configuration between the two observed
states. Unlike the original states, the pseudo-state is not tied
to any specific observation but provides a common state that
facilitates consistent alignment between Gi and Gj.
We compute transformation matrices Ti→p and Tj→p to
transfer the original fields Gi and Gj into Gp. By transform-
ing both fields into this shared pseudo-state, we enable direct
comparison and alignment of their rendered outputs. Specif-
ically, we render the transformed fields from the same view-
point v and enforce photometric and semantic consistency
between them. The corresponding loss is defined as:
Lpseudo(Gi, Gj, θ) = ∥R (Ti→p(Gi), v) −R (Tj→p(Gj), v)∥1
+ CE

fθ (M (Ti→p(Gi), v)) ,
fθ (M (Tj→p(Gj), v))

(7)
By leveraging a dynamically constructed pseudo-state as
an adaptive supervision signal, the model can better recon-
cile differences between the two input states and generalize
more effectively to unseen or intermediate scene configura-
tions.
3.6
Collaborative Co-Pruning
Inspired by geometric consistency-based filtering strate-
gies (Zhang et al. 2024), we introduce a co-pruning mech-
anism to suppress residual artifacts arising from imper-
fect segmentation during cross-state Gaussian transfer. The

<!-- page 5 -->
mechanism removes spatially inconsistent Gaussians by
evaluating geometric agreement between the two states.
When a Gaussian field is transferred from one state to an-
other, unmatched or misaligned points may remain due to
occlusion, noise, or over-segmentation. Our strategy prunes
these outliers by checking whether transferred Gaussians
can be reliably explained by the geometry of the target field.
For each transformed Gaussian gk ∈Ti→j(Gi), we iden-
tify its nearest neighbor gl ∈Gj using Euclidean distance.
A Gaussian is marked for pruning if the spatial deviation
between gk and gl exceeds a predefined threshold τ. The
binary pruning indicator mi is computed as:
mi = 1 (∥xk −xl∥2 > τ)
(8)
where xk and xl are the 3D centers of gk and gl, and 1(·)
denotes the indicator function. Gaussians with mi = 1 are
discarded as unreliable or redundant. A symmetric process
is applied in the opposite direction, using Gj transformed to
the frame of Gi to prune outliers in Gj, resulting in a collab-
orative co-pruning scheme.
3.7
Training Objective
The overall training objective combines three loss terms:
Ljoint(Gi, Gj, θ) = Lr(Gi, θ) + Lr(Gj, θ) + λaLalign(Gi, Gj, θ)
+ λpLpseudo(Gi, Gj, θ)
(9)
where Lr denotes the same reconstruction loss adopted from
Gaussian Grouping (Ye et al. 2023) (detailed in the ap-
pendix), Lalign enforces bidirectional rendering consistency,
and Lpseudo introduces regularization through pseudo-state
supervision. The weights λa and λp are used to balance the
contributions of each term.
4
Experiment
4.1
Dataset
To support multi-scan scene modeling, we construct both
synthetic and real-world datasets. The synthetic dataset is
generated in Blender (Blender Online Community 2023),
where N textured objects from BlenderKit (BlenderKit
2023) are placed within a static background. Additional
scans are created by randomly altering object poses to re-
flect different interaction states. Real-world data is captured
in a similar manner using handheld RGB cameras, resulting
in 7 synthetic and 5 real scenes. For evaluation, we generate
a test configuration for each scene by randomly reposition-
ing objects. We then render images from predefined camera
views and compute PSNR and SSIM against ground truth
images to assess interaction fidelity under novel object ar-
rangements. Further implementation and dataset details are
provided in the appendix.
4.2
Experimental Setup
Implementation details
During training, we first opti-
mize the segmented Gaussians using only Lr for 10,000
epochs, then jointly train with Lalign and Lpseudo to refine
the dual Gaussian field for another n×5000 epochs, where
n is the total number of scans in the scene. The output clas-
sification linear layer has 16 input channels and 256 output
channels. The pruning threshold parameter τ is set to 0.5. In
training, we set λa = 1.0 and λp = 1.0. We use the Adam
optimizer for both gaussians and linear layer, with a learning
rate of 0.0025 for segmentation feature and 0.0005 for lin-
ear layer. All datasets are trained on a single NVIDIA 4090
GPU.
Baselines
We compare our method with representative
Gaussian Splatting-based scene modeling frameworks. Ex-
isting pipelines often involve multi-stage processing, includ-
ing segmentation, background completion, inpainting, and
fine-tuning. We include several representative segmentation
methods in our comparison. GaussianEditor (Chen et al.
2024) performs segmentation through inverse rendering op-
timization. Gaussian Grouping (Ye et al. 2023) clusters
Gaussians based on feature similarity. GaussianCut (Jain,
Mirzaei, and Gilitschenski 2024) formulates segmentation
as a graph-cut optimization problem over Gaussian prim-
itives. We also include Decoupled Gaussian (Wang et al.
2025), which segments objects using Gaussian segmenta-
tion feature, then performs remeshing and LaMa-based re-
finement to complete the scene.
4.3
Novel State Synthesis
The qualitative results on both synthetic and real-world
datasets are shown in Figure 4. GaussianEditor (based on
inverse rendering) struggles to precisely segment object
boundaries, resulting in edge artifacts that necessitate heavy
post-processing. Gaussian Grouping (segmentation-feature
based) improves performance but still leaves many residual
Gaussians, especially for objects with large contact areas
between their bottom surface and the background. Graus-
siancut (graph-based) achieves the best results among the
baselines, although slight boundary artifacts remain. De-
coupledGaussian incorporates background Gaussian com-
pletion, 2D inpainting, and Gaussian fine-tuning. Its 2D in-
painting module LaMa produces the most visually coherent
results. However, it still struggles to faithfully restore im-
ages with complex backgrounds in real-world data. In con-
trast, our method achieves the highest PSNR and SSIM for
novel-state synthesis across both datasets, while maintain-
ing an end-to-end pipeline and avoiding complex multi-stage
post-processing.
As shown in Table 1 and 2, segmentation-only methods
yield lower PSNR and SSIM, while adding inpainting im-
proves performance—particularly on synthetic scenes where
backgrounds are typically simpler and more structured. In
contrast, real-world scenes often involve cluttered or tex-
tured backgrounds, making accurate hole filling more chal-
lenging and less reliable. Instead of relying on inpainting,
we leverage the complementary information across multiple
scene states to supervise the optimization of Gaussians, en-
abling more accurate and consistent scene representation.
4.4
Ablation Study
Table 3 presents the ablation study evaluating the con-
tribution of each component in our framework: Bidirec-

<!-- page 6 -->
GaussianEditor
Gaussiancut
Gaussian Grouping
DecoupledGaussian
IGFuse (ours)
Ground Truth
Figure 4: Qualitative comparison of novel state synthesis under different pipelines. We evaluate on both real-world scenes (top
three) and a synthetic scene. While existing methods struggle with object mixing, boundary artifacts, or background corruption,
our method achieves significantly more accurate and complete novel state results, closely matching the ground-truth.
Model
PSNR
SSIM
GaussianEditor (CVPR 2024)
28.25
0.946
Gaussian Grouping (ECCV 2024)
28.93
0.950
Gaussiancut (NIPS 2024)
29.01
0.956
DecoupledGaussian (CVPR 2025)
30.27
0.959
IGFuse (ours)
36.93
0.978
Table 1: Quantitative comparison of novel state synthesis
quality on the synthetic dataset.
Model
PSNR
SSIM
GaussianEditor (CVPR 2024)
21.02
0.849
Gaussian Grouping (ECCV 2024)
21.68
0.853
Gaussiancut (NIPS 2024)
21.81
0.864
DecoupledGaussian (CVPR 2025)
22.28
0.855
IGFuse (ours)
27.18
0.907
Table 2: Quantitative comparison of novel state synthesis
quality on the real-world dataset.
tional alignment (B), Collaborative co-pruning (C), and
Pseudo-state guided alignment (P). Using only Bidirectional
alignment already provides a strong baseline, achieving a
PSNR of 35.10. Introducing co-pruning yields a slight im-
provement in structural quality. This is because Bidirec-
tional alignment tends to reassign residual Gaussians to
B
C
P
PSNR ↑
SSIM ↑
"
-
-
35.10
0.971
"
"
-
35.55
0.974
"
"
"
36.93
0.978
Table 3: Ablation study of B (Bidirectional alignment),
C (Collaborative co-pruning), and P (Pseudo-state guided
alignment).
have background-like colors or reduced opacity. While co-
pruning helps eliminate these floaters, its overall impact
on PSNR is limited. In contrast, incorporating Pseudo-state
guided alignment results in a substantial increase in PSNR.
This improvement arises from the fact that occlusion am-
biguities cannot be fully resolved with only two configu-
rations, additional pseudo-states provide richer supervision
across multiple viewpoints, enhancing alignment between
the two Gaussian fields and leading to more consistent and
photorealistic reconstructions.
4.5
Dual Guassian Convergence
We investigate the convergence behavior of two Gaussian
fields trained from different synthetic scenes. In the absence
of Pseudo-state Guided Alignment, PSNR and SSIM dif-
fer significantly when evaluated in the target state. These
discrepancies stem from occlusions and viewpoint differ-
ences that lead to misalignments between the two fields.

<!-- page 7 -->
Gaussian
Pseudo
Synthetic 1
Synthetic 2
PSNR
SSIM
PSNR
SSIM
G1
w/o
37.04
0.979
37.51
0.977
G2
w/o
37.16
0.977
36.14
0.974
G1
w/
39.26
0.984
37.59
0.977
G2
w/
39.26
0.984
37.50
0.977
Table 4: PSNR and SSIM of G1 and G2 in Syntheti 1 and
Synthetic 2 scenes, with and without pseudo-state supervi-
sion.
Even with Bidirectional Alignment, such inconsistencies
persist, indicating incomplete convergence. By incorporat-
ing Pseudo-state guided Alignment, we enforce consistency
across object compositions in both fields, allowing them
to observe complementary content and provide mutual su-
pervision. This promotes convergence toward a shared and
coherent optimized representation. Empirically, Gaussian
fields trained from either state yield nearly identical PSNR
and SSIM when evaluated under same test configuration,
demonstrating effective alignment and mutual consistency.
4.6
Background Separation
As shown in Figure 5, when separating only the background,
both Gaussian Grouping and Gaussiancut leave residual
Gaussians from objects, with larger objects causing notice-
able holes. Although DecoupledGaussian employs LaMa to
inpaint object mask regions, the inpainting often produces
blurry results, especially in complex backgrounds. In con-
trast, our multi-scan fusion approach effectively generates
complete and seamless background reconstructions.
Gaussiancut
Gaussian Grouping
DecoupledGaussian
IGFuse (ours)
V1：294
Figure 5: Comparison of different techniques for back-
ground separation. IGFuse (ours) vs. Decoupled Gaussian,
Gaussiancut, and Gaussian Grouping.
4.7
Training Iteration
To determine a suitable number of iterations for optimiz-
ing the Gaussian fields, we evaluated multiple real-world
Figure 6: PSNR vs. Training Iterations with Variance Range
scenes by measuring the mean and variance of PSNR on the
test state across different iteration counts. For each scene
with n scans, we normalize the total number of iterations by
n—that is, the iteration count refers to how many optimiza-
tion steps each individual scan undergoes. We observe that
the PSNR stabilizes around 5000 iterations per scan, indi-
cating convergence. Since we align scene pairs analogously
to constructing an undirected graph, we set the final number
of training iterations to n × 5000, where n is the total num-
ber of scans in the scene. Additionally, we observe a grad-
ual increase in variance. This is because, in early iterations,
motion-related artifacts result in uniformly low-quality re-
constructions. As optimization progresses and overall qual-
ity improves, differences in native PSNR across scenes be-
come more pronounced, leading to increased variance.
5
Limitations
Despite its effectiveness, IGFuse has several limitations.
Existing optimization methods are designed for the entire
scene. However, since backgrounds across different scans
often share similar structures, focusing optimization specifi-
cally on object–background boundaries in future work could
lead to a more lightweight model. Additionally, our model
does not handle lighting variations, causing static shadows
even when objects move, which affects realism. Incorporat-
ing relighting into the framework could further enhance sim-
ulation fidelity in future work.
6
Conclusion
We present IGFuse, an end-to-end framework for interac-
tive 3D scene reconstruction via multi-scan fusion. By lever-
aging object-level transformations across multiple observed
scene states, our method overcomes challenges caused by
occlusions and segmentation ambiguity. Through bidirec-
tional consistency and pseudo-state alignment, IGFuse re-
fines geometry and semantics to produce high-quality Gaus-
sian fields that support accurate rendering and object-aware
manipulation. Extensive experiments validate the effective-
ness of our approach for interactive scene reconstruction in
vision tasks.

<!-- page 8 -->
References
Barcellona, L.; Zadaianchuk, A.; Allegro, D.; Papa, S.;
Ghidoni, S.; and Gavves, E. 2024.
Dream to Manip-
ulate: Compositional World Models Empowering Robot
Imitation Learning with Imagination.
arXiv preprint
arXiv:2412.14957.
Blender Online Community. 2023. Blender - a 3D modelling
and rendering package. https://www.blender.org.
BlenderKit. 2023.
BlenderKit: 3D Asset Library for
Blender. https://www.blenderkit.com. Accessed in 2023.
Cao, C.; Yu, C.; Wang, F.; Xue, X.; and Fu, Y. 2024a. Mvin-
painter: Learning multi-view consistent inpainting to bridge
2d and 3d editing. arXiv preprint arXiv:2408.08000.
Cao, J.; Guan, S.; Ge, Y.; Li, W.; Yang, X.; and Ma, C.
2024b. NeuMA: Neural material adaptor for visual ground-
ing of intrinsic dynamics. Advances in Neural Information
Processing Systems, 37: 65643–65669.
Chen, Y.; Chen, Z.; Zhang, C.; Wang, F.; Yang, X.; Wang, Y.;
Cai, Z.; Yang, L.; Liu, H.; and Lin, G. 2024. Gaussianeditor:
Swift and controllable 3d editing with gaussian splatting. In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, 21476–21485.
Cheng, H. K.; Oh, S. W.; Price, B.; Schwing, A.; and Lee,
J.-Y. 2023. Tracking anything with decoupled video seg-
mentation. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, 1316–1326.
Han, X.; Liu, M.; Chen, Y.; Yu, J.; Lyu, X.; Tian, Y.; Wang,
B.; Zhang, W.; and Pang, J. 2025.
Re ˆ 3 Sim: Gener-
ating High-Fidelity Simulation Data via 3D-Photorealistic
Real-to-Sim for Robotic Manipulation.
arXiv preprint
arXiv:2502.08645.
Hu, W.; Chai, W.; Hao, S.; Cui, X.; Wen, X.; Hwang, J.-N.;
and Wang, G. 2025. Pointmap Association and Piecewise-
Plane Constraint for Consistent and Compact 3D Gaussian
Segmentation Field. arXiv preprint arXiv:2502.16303.
Hu, X.; Wang, Y.; Fan, L.; Fan, J.; Peng, J.; Lei, Z.; Li, Q.;
and Zhang, Z. 2024.
Semantic anything in 3d gaussians.
arXiv preprint arXiv:2401.17857.
Huang, S.-Y.; Chou, Z.-T.; and Wang, Y.-C. F. 2025. 3D
Gaussian Inpainting with Depth-Guided Cross-View Con-
sistency. arXiv preprint arXiv:2502.11801.
Jain, U.; Mirzaei, A.; and Gilitschenski, I. 2024. Gaussian-
Cut: Interactive segmentation via graph cut for 3D Gaussian
Splatting. In The Thirty-eighth Annual Conference on Neu-
ral Information Processing Systems.
Kerbl, B.; Kopanas, G.; Leimk¨uhler, T.; and Drettakis, G.
2023. 3d gaussian splatting for real-time radiance field ren-
dering. ACM Transactions on Graphics, 42(4): 1–14.
Li, H.; Wu, Y.; Meng, J.; Gao, Q.; Zhang, Z.; Wang, R.; and
Zhang, J. 2024a. InstanceGaussian: Appearance-Semantic
Joint Gaussian Representation for 3D Instance-Level Per-
ception. arXiv preprint arXiv:2411.19235.
Li, X.; Li, J.; Zhang, Z.; Zhang, R.; Jia, F.; Wang, T.; Fan,
H.; Tseng, K.-K.; and Wang, R. 2024b.
Robogsim: A
real2sim2real robotic gaussian splatting simulator.
arXiv
preprint arXiv:2411.11839.
Liu, Z.; Ouyang, H.; Wang, Q.; Cheng, K. L.; Xiao, J.; Zhu,
K.; Xue, N.; Liu, Y.; Shen, Y.; and Cao, Y. 2024. Infusion:
Inpainting 3d gaussians via learning depth completion from
diffusion prior. arXiv preprint arXiv:2404.11613.
Lou, H.; Liu, Y.; Pan, Y.; Geng, Y.; Chen, J.; Ma, W.; Li, C.;
Wang, L.; Feng, H.; Shi, L.; et al. 2024. Robo-gs: A physics
consistent spatial-temporal model for robotic arm with hy-
brid representation. arXiv preprint arXiv:2408.14873.
Lyu, W.; Li, X.; Kundu, A.; Tsai, Y.-H.; and Yang, M.-H.
2024. Gaga: Group Any Gaussians via 3D-aware Memory
Bank. arXiv preprint arXiv:2404.07977.
Mendonca, R.; Bahl, S.; and Pathak, D. 2023.
Struc-
tured world models from human videos.
arXiv preprint
arXiv:2308.10901.
Oquab, M.; Darcet, T.; Moutakanni, T.; Vo, H.; Szafraniec,
M.; Khalidov, V.; Fernandez, P.; Haziza, D.; Massa, F.; El-
Nouby, A.; et al. 2023. Dinov2: Learning robust visual fea-
tures without supervision. arXiv preprint arXiv:2304.07193.
Pang, J.-C.; Tang, N.; Li, K.; Tang, Y.; Cai, X.-Q.; Zhang,
Z.-Y.; Niu, G.; Sugiyama, M.; and Yu, Y. 2025. Learning
View-invariant World Models for Visual Robotic Manipula-
tion. In The Thirteenth International Conference on Learn-
ing Representations.
Qureshi, M. N.; Garg, S.; Yandun, F.; Held, D.; Kantor, G.;
and Silwal, A. 2024. Splatsim: Zero-shot sim2real transfer
of rgb manipulation policies using gaussian splatting. arXiv
preprint arXiv:2409.10161.
Ravi, N.; Gabeur, V.; Hu, Y.-T.; Hu, R.; Ryali, C.; Ma, T.;
Khedr, H.; R¨adle, R.; Rolland, C.; Gustafson, L.; et al. 2024.
Sam 2: Segment anything in images and videos.
arXiv
preprint arXiv:2408.00714.
Shen, Q.; Yang, X.; and Wang, X. 2024. Flashsplat: 2d to 3d
gaussian splatting segmentation solved optimally. In Euro-
pean Conference on Computer Vision, 456–472. Springer.
Wang, G.; Pan, L.; Peng, S.; Liu, S.; Xu, C.; Miao, Y.; Zhan,
W.; Tomizuka, M.; Pollefeys, M.; and Wang, H. 2024. Nerf
in robotics: A survey. arXiv preprint arXiv:2405.01333.
Wang, M.; Zhang, Y.; Ma, R.; Xu, W.; Zou, C.; and
Morris, D. 2025.
DecoupledGaussian: Object-Scene De-
coupling for Physics-Based Interaction.
arXiv preprint
arXiv:2503.05484.
Wu, J.; Yin, S.; Feng, N.; He, X.; Li, D.; Hao, J.; and Long,
M. 2024a.
ivideogpt: Interactive videogpts are scalable
world models. Advances in Neural Information Processing
Systems, 37: 68082–68119.
Wu, Y.; Meng, J.; Li, H.; Wu, C.; Shi, Y.; Cheng, X.; Zhao,
C.; Feng, H.; Ding, E.; Wang, J.; et al. 2024b. OpenGaus-
sian: Towards Point-Level 3D Gaussian-based Open Vocab-
ulary Understanding. arXiv preprint arXiv:2406.02058.
Xie, T.; Zong, Z.; Qiu, Y.; Li, X.; Feng, Y.; Yang, Y.;
and Jiang, C. 2024.
Physgaussian: Physics-integrated 3d
gaussians for generative dynamics. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 4389–4398.

<!-- page 9 -->
Yang, M.; Du, Y.; Ghasemipour, K.; Tompson, J.; Schuur-
mans, D.; and Abbeel, P. 2023. Learning interactive real-
world simulators. arXiv preprint arXiv:2310.06114, 1(2):
6.
Yang, S.; Yu, W.; Zeng, J.; Lv, J.; Ren, K.; Lu, C.; Lin, D.;
and Pang, J. 2025. Novel Demonstration Generation with
Gaussian Splatting Enables Robust One-Shot Manipulation.
arXiv preprint arXiv:2504.13175.
Ye, M.; Danelljan, M.; Yu, F.; and Ke, L. 2023. Gaussian
grouping: Segment and edit anything in 3d scenes. arXiv
preprint arXiv:2312.00732.
Yu, J.; Fu, L.; Huang, H.; El-Refai, K.; Ambrus, R. A.;
Cheng, R.; Irshad, M. Z.; and Goldberg, K. 2025.
Real2Render2Real: Scaling Robot Data Without Dynam-
ics Simulation or Robot Hardware.
arXiv preprint
arXiv:2505.09601.
Zhang, J.; Jiang, J.; Chen, Y.; Jiang, K.; and Liu, X. 2025.
COB-GS: Clear Object Boundaries in 3DGS Segmentation
Based on Boundary-Adaptive Gaussian Splitting.
arXiv
preprint arXiv:2503.19443.
Zhang, J.; Li, J.; Yu, X.; Huang, L.; Gu, L.; Zheng, J.; and
Bai, X. 2024. Cor-gs: sparse-view 3d gaussian splatting via
co-regularization. In European Conference on Computer Vi-
sion, 335–352. Springer.
Zhong, L.; Yu, H.-X.; Wu, J.; and Li, Y. 2024. Reconstruc-
tion and simulation of elastic objects with spring-mass 3d
gaussians.
In European Conference on Computer Vision,
407–423. Springer.
Zhu, R.; Qiu, S.; Liu, Z.; Hui, K.-H.; Wu, Q.; Heng, P.-A.;
and Fu, C.-W. 2025a.
Rethinking End-to-End 2D to 3D
Scene Segmentation in Gaussian Splatting. arXiv preprint
arXiv:2503.14029.
Zhu, S.; Mou, L.; Li, D.; Ye, B.; Huang, R.; and Zhao, H.
2025b.
VR-Robo: A Real-to-Sim-to-Real Framework for
Visual Robot Navigation and Locomotion. arXiv preprint
arXiv:2502.01536.
Zhu, S.; Wang, G.; Kong, X.; Kong, D.; and Wang, H. 2024.
3d gaussian splatting in robotics: A survey. arXiv preprint
arXiv:2410.12262.

<!-- page 10 -->
IGFuse: Interactive 3D Gaussian Scene Reconstruction via Multi-Scans Fusion
Supplementary Material
A
Dataset Preparation
To support training and evaluation in both synthetic and real-
world settings, we construct a multi-scan dataset comprising
scenes with multiple static states. In each scene, a subset of
objects undergoes random rigid-body perturbations, such as
translations and rotations. Our dataset consists of 7 simu-
lated scenes and 5 real-world multi-scan scenes. Each scene
includes multiple object configurations, with approximately
100 views captured per scan. In addition to the several train-
ing scan data, we also provide a separate test scan data for
evaluating reconstruction quality using PSNR and SSIM. An
overview of the dataset is illustrated in Figure 7.
A.1
Simulated Scenes
Object Library and Placement
We use Blender (Blender
Online Community 2023) to generate synthetic indoor
scenes populated with objects from the BlenderKit li-
brary (BlenderKit 2023), as shown in Figure 8. The objects
span various semantic categories such as chairs, tables, toys,
and storage items. For each scene, N = 3 ∼15 objects are
randomly selected and placed on physically valid surfaces.
Perturbation Strategy
To generate a number of scans, we
apply rigid perturbations for all the objects, including ran-
dom translations and in-place rotations. The object transfor-
mations can be directly obtained from Blender.
Camera Setup
For each scan, we render m = 100 per-
spective views with a resolution of 640 × 480 using uni-
formly distributed viewpoints on a hemisphere around the
scene. Camera intrinsics (focal length, principal point) are
shared across both scans to simplify alignment.
A.2
Real-World Scenes
Video Capture
We record several short video sequences
(10–20 seconds) of the same scene using a handheld RGB
camera, introducing random changes in object layout be-
tween recordings.
Scene Alignment
To initialize the point cloud and esti-
mate accurate camera poses, we reconstruct the scene us-
ing COLMAP with images from all scans, as shown in Fig-
ure 9. Despite variations in foreground objects, COLMAP
produces a well-aligned background and reliable poses. This
serves as a strong reference for subsequent 3D Gaussian
Splatting reconstruction, ensuring consistent background
alignment in the final model.
Object Alignment and RT Estimation
After scene align-
ment, we extract 3D object masks and compute object-wise
transformations between the two scans. Specifically, we first
perform per-frame segmentation using SAM2 (Ravi et al.
2024). For each matched object pair between two scans, an
initial transformation matrix is obtained by aligning the ob-
ject using the Iterative Closest Point (ICP) algorithm. This
matrix is then refined by optimizing it with supervision from
the object’s RGB mask, leading to a more accurate estima-
tion of the transformation.
Segmentation and ID Consistency
For synthetic data,
ground-truth instance masks are rendered per object. For
real scenes, segmentation masks are initially generated using
SAM2 (Ravi et al. 2024). To ensure cross-scan consistency,
we extract features from images in different scans using DI-
NOv2 (Oquab et al. 2023) and compute the cosine similarity
between them. Regions with similarity above a predefined
threshold are assigned a consistent ID across scans, enabling
consistent association across views and scans.
B
Pseudo State Construction
To facilitate dual-state alignment, we construct a pseudo
state using boundary detection and object overlap avoidance
algorithms. First, we analyze the object positions across all
scans to determine a shared scene boundary. Then, we com-
pute the minimum object spacing threshold by averaging
the three smallest inter-object distances. Objects are then se-
quentially placed into the scene using random rotations and
translations. For each placement, we check whether the ob-
ject is too close to any previously placed object—if the dis-
tance falls below the threshold, a new position is sampled.
This process repeats until all objects are successfully placed.
Finally, we perform a bounding box check based on each ob-
ject’s point cloud to ensure no geometric overlap occurs.
C
Novel State Synthesis
In Figure 10, we show the input scans alongside the resulting
novel states. As illustrated, our method supports the genera-
tion of novel states with arbitrary object arrangements while
maintaining high visual quality. It is also evident that lever-
aging just two input scans already leads to noticeable im-
provements in the reconstruction results.
D
Details of Lr
As referenced in the main text, we elaborate here on the for-
mulation of the overall loss Lr (Ye et al. 2023). We use a
linear layer f followed by a softmax activation to map the
rendered 2D features S into a K-class semantic space. For
the resulting 2D classification, we apply a standard cross-
entropy loss L2d to enforce accurate pixel-level mask pre-

<!-- page 11 -->
Synthetic Dataset
Scan 1
Scan N
Test Scan
Real-world Dataset
Scan 1
Scan N
Test Scan
Scene 2
Scene M
Scene 1
Scene M-1
Scene 2
Scene M
Scene 1
Scene M-1
Figure 7: Overview of our dataset, consisting of 7 synthetic scenes (left) and 5 real-world scenes (right). Each scene includes
several training scans (Scan 1 to Scan N) with random object layouts and one held-out test Scan for evaluation.
Figure 8: A selection of virtual assets used in our simulated
scenes. All objects were sourced from the BlenderKit li-
brary (BlenderKit 2023).
dictions:
L2d = −
X
k∈K
m[k] log (softmax(f(S))[k])
(10)
To ensure consistency in 3D segmentation, we further in-
troduce a 3D segmentation loss L3d. This is a cross-entropy
loss applied to the per-point segmentation features s, using
the fused 3D pointmap labels Ps as pseudo ground truth:
L3d = −
X
k∈K
Ps[k] log (softmax(f(s))[k])
(11)
Scan 1
Scan 2
Scan 3
Figure 9: Illustration of the COLMAP used for scene align-
ment.
In addition, we adopt the standard 3D Gaussian image re-
construction loss as proposed in (Kerbl et al. 2023), which
is a weighted combination of L1 loss and D-SSIM loss:
Limg = (1 −λ)L1 + λLD-SSIM
(12)
The final training objective Lr is a weighted sum of all

<!-- page 12 -->
Input Scans
Novel States
Input Scans
Novel States
Input Scans
Novel States
Input Scans
Novel States
Figure 10: Novel state synthesis results. For each scene, the top row shows the object arrangements in the input scans, while
the bottom row presents the synthesized novel states under randomly arranged object configurations.
three components:
Lr = Limg + λ2dL2d + λ3dL3d
(13)
In our experiments, we set λ2d = 1 and λ3d = 1.
E
Segmentation Improvement
As shown in Figure 11, our multi-scan segmentation method
provides improvements over the feature-based baseline,
Gaussian Grouping, particularly in fine-grained details.

<!-- page 13 -->
GT image
GT segmentation
Grouping
IGFuse(Ours)
Figure 11: Qualitative comparison of segmentation results across different scenes. From top to bottom: ground-truth (GT)
image, ground-truth segmentation, results of Gaussian Grouping, and our method IGFuse. Our method produces more accurate
and consistent segmentations, especially around object boundaries, closely matching the ground truth.
interation 0
interation 1000
interation 2000
interation 3000
interation 4000
interation 5000
Figure 12: Visualizations result over different training iterations.
Gaussian Grouping often produces artifacts around object
boundaries, while our results are much closer to the ground
truth. Quantitatively, the mIoU improves by approximately
4%, as shown in the Table 5.
F
Training Iteration
We provide visualizations in Figure 12 for training iteration.
At iteration 0, when Bidirectional Alignment and Pseudo-
state Guided Alignment are not yet applied, many artifacts
remain in the scene due to inaccurate object transitions. As
optimization proceeds, these artifacts are gradually elimi-
nated. By iteration 5000, the optimization is nearly com-

<!-- page 14 -->
GaussianEditor
Gaussiancut
Gaussian Grouping
IGFuse (ours)
Ground Truth
Scan 2
Scan 1
Figure 13: Multi-scan object observation results. Our method effectively fuses observations from multiple scans to reconstruct
complete object models.
Method
mIoU (%)
Gaussian Grouping
86.8
IGFuse (Ours)
91.0
Table 5: Comparison of mIoU for different methods
plete, and the artifacts have mostly disappeared.
G
Multi-scans Object Observation
In the main text, we demonstrate how multi-scan fusion en-
hances the completeness of the background. This benefit
also extends to object-level reconstruction, as illustrated in
Figure 13. In scan 1 and scan 2, only the top and bottom of
the small red car are visible, respectively, making it difficult
to obtain a complete observation of the object. In contrast,
the second row shows that other methods based on a single
scan (e.g., scan 1) generate novel states where the bottom
of the car is severely missing. By fusing information from
both scan 1 and scan 2, our method reconstructs a complete
object model and supports arbitrary object configurations in
novel scenes.
H
Fair Comparison with Single-Scan
Methods under Multi-Scans Training
Since multi-scan input provides more information than
single-scan settings, direct comparisons may be unfair. To
ensure fair evaluation, we adapt single-scan segmentation
methods to operate under the multi-scan setting as well.
Specifically, we use two scans: the model is first trained
on scan 1 to obtain segmentation results, and then the seg-
mented objects are transferred to scan 2, where training con-
tinues under supervision from the images in scan 2. This ap-
proach produces results that are nearly identical to those ob-
tained by training directly on scan 2. The unoccluded ground
Scan 1
Scan 2
Train in Scan 2
Train in Scan 1 followed by Scan 2
Test Scan
Figure 14: Comparison of training strategies using different
scene scans. The top row shows three static observations:
Scan 1, Scan 2, and a Test Scan. The bottom row presents
the rendering results at the Test Scan after training on Scan
2 only (left) and after sequential training on Scan 1 followed
by Scan 2 (right). The results show that incorporating an ad-
ditional scan does not bring significant improvement over
single-Scan training.
observed in scan 1 is not preserved after the transfer. In-
stead, it becomes missing regions, as shown in Figure 14.
This demonstrates a key limitation of single-scan methods
in fusing information across scans, while our Bidirectional
Alignment strategy effectively addresses this issue.
As shown in the Table 6, incorporating multi-scans brings
only slight improvements in PSNR and SSIM for methods
like Gaussian Grouping. This minor gain is attributed to
small residual Gaussians that may remain on the ground un-
der certain objects, as seen in the bottom left of Figure 14.

<!-- page 15 -->
Training data
Model
PSNR
SSIM
Single-scan
Gaussian Grouping
28.93
0.950
DecoupledGaussian
30.27
0.959
Multi-scans
Gaussian Grouping
28.99
0.952
DecoupledGaussian
30.29
0.959
Multi-scans
IGFuse (Ours)
36.93
0.978
Table 6: Quantitative comparison of 3D Gaussian segmen-
tation methods with single-scan and multi-scans training.
Multi-scans inputs yield slight gains, while inpainting domi-
nates performance. IGFuse achieves the best results without
relying on inpainting.
However, after inpainting is introduced, the performance of
DecoupledGaussian is nearly identical, since the missing re-
gions are mainly filled by the inpainting process and the ad-
ditional scan training provides little further benefit.
I
Depth Consistency Comparison
GT Image
IGFuse (ours)
GaussianCut
GaussianEditor
Gaussian Grouping
DecoupleGaussian
Figure 15: Depth consistency comparison across different
methods.
Although no depth or normal supervision is used and
only the original 3D Gaussians are employed, our method
achieves superior depth reconstruction due to the effective
supervision from multi-scan scenes. Other methods suffer
from visible holes after removal of the object, as shown in
Figure 15. Although the inpainting-based DecoupledGaus-
sian method performs better in filling these gaps, some depth
inconsistencies still remain due to the separate optimization
of foreground and background.
