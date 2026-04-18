<!-- page 1 -->
Dream3DAvatar: Text-Controlled 3D Avatar Reconstruction from a Single Image
Gaofeng Liu1* , Hengsen Li1* , Ruoyu Gao1, Xuetong Li1, Zhiyuan Ma2† , Tao Fang1†
1Department of Automation, Shanghai Jiao Tong University, Shanghai 201100, China.
2Department of Electronic Engineering, Tsinghua University, Beijing 100084, China.
Abstract
With the rapid advancement of 3D representation techniques
and generative models, substantial progress has been made
in reconstructing full-body 3D avatars from a single image.
However, this task remains fundamentally ill-posedness due
to the limited information available from monocular input,
making it difficult to control the geometry and texture of
occluded regions during generation. To address these chal-
lenges, we redesign the reconstruction pipeline and propose
Dream3DAvatar, an efficient and text-controllable two-stage
framework for 3D avatar generation. In the first stage, we de-
velop a lightweight, adapter-enhanced multi-view generation
model. Specifically, we introduce the Pose-Adapter to inject
SMPL-X renderings and skeletal information into SDXL, en-
forcing geometric and pose consistency across views. To pre-
serve facial identity, we incorporate ID-Adapter-G, which in-
jects high-resolution facial features into the generation pro-
cess. Additionally, we leverage BLIP2 to generate high-
quality textual descriptions of the multi-view images, en-
hancing text-driven controllability in occluded regions. In the
second stage, we design a feedforward Transformer model
equipped with a multi-view feature fusion module to re-
construct high-fidelity 3D Gaussian Splat representations
(3DGS) from the generated images. Furthermore, we intro-
duce ID-Adapter-R, which utilizes a gating mechanism to ef-
fectively fuse facial features into the reconstruction process,
improving high-frequency detail recovery. Extensive exper-
iments demonstrate that our method can generate realistic,
animation-ready 3D avatars without any post-processing and
consistently outperforms existing baselines across multiple
evaluation metrics.
1
Introduction
3D digital humans have been widely applied in various fields
such as gaming, animation, and virtual reality. However,
creating realistic 3D avatars typically requires highly spe-
cialized modeling expertise and substantial manual labor.
To alleviate this burden, recent research has increasingly
focused on reconstructing 3D humans from a single im-
age (Li et al. 2025; Zhuang et al. 2025; Qiu et al. 2025a;
Han et al. 2023). Despite these approaches reduce the mod-
eling cost, they struggle to provide text-driven control over
the textures or geometry of occluded regions in the input
*These authors contributed equally to this work.
†Corresponding authors.
Figure 1: We introduces a two stage animatable 3D human
reconstruction method with text control from single image.
image. This lack of controllability introduces a fundamen-
tal ill-posedness in the reconstruction process: the results
heavily rely on prior knowledge learned by the model and
represent only one of many possible interpretations. Given
the inherent information loss in monocular images and the
complexity of human pose, geometry, and texture, gener-
ating multi-view-consistent and photorealistic 3D avatars
from a single image using a text-driven approach remains
a highly challenging task. (Saito et al. 2019; Xiu et al. 2023;
Saito et al. 2020; Zhang, Yang, and Yang 2024; Zheng et al.
2021). Early parametric models (Alldieck et al. 2018; Loper
et al. 2023) leverage human priors to reconstruct 3D bod-
ies but produce only coarse surface textures. Feedforward-
based methods (Zhuang et al. 2025; Qiu et al. 2025a; Zhang
et al. 2024b) allow for fast 3D reconstruction from a single
image but lack controllability and diversity. Although recent
arXiv:2509.13013v1  [cs.CV]  16 Sep 2025

<!-- page 2 -->
advances such as diffusion-based models and iterative opti-
mization (AlBahar et al. 2023; Li et al. 2025; Xiu et al. 2022;
Choi 2025) significantly improve reconstruction quality by
exploiting multi-view information, they still suffer from low
efficiency, limiting their practicality for real-time or high-
throughput applications.
To tackle these challenges, we propose Dream3DAvatar,
a text-driven, lightweight two-stage framework for recon-
structing 3D virtual humans from a single image. Our
method primarily addresses two key issues: 1) alleviat-
ing the ill-posedness caused by monocular input; and
2) achieving lightweight and efficient 3D human recon-
struction based on multi-view images.
We
incorporate
three
lightweight
Adapter
mod-
ules (Alayrac et al. 2022) into the existing generative
pipeline (Podell et al. 2023; He et al. 2022), enabling
efficient fine-tuning while preserving the prior knowledge
of the pretrained model. In the first stage, inspired by MV-
Adapter (Huang et al. 2024), we leverage SDXL (Podell
et al. 2023) to controllably generate multi-view head images
from a single input photo. To impose geometric consistency
on body shape and pose, we introduce a geometry-aware
module, Pose-Adapter, which injects SMPL-X and skeletal
information as additional conditions into SDXL. Simultane-
ously, we employ ID-Adapter-G to process high-resolution
facial features and ensure identity consistency across views.
These conditions are fused within SDXL through multiple
parallel attention mechanisms, facilitating multi-view image
generation with consistent attributes. Additionally, we
utilize BLIP2 (Li et al. 2023) to generate textual descrip-
tions for the multi-view images, enabling text-conditioned
training and enhancing controllability in occluded regions.
In the second stage, we employ a feedforward Trans-
former model equipped with a multi-view feature fusion
module (Zhuang et al. 2025) to reconstruct high-fidelity
3D virtual humans from the generated multi-view images.
To better preserve identity features, we embed ID-Adapter-
R into the Transformer architecture and introduce a gating
mechanism to enhance the integration of facial details into
the full-body reconstruction. With the incorporation of these
lightweight modules, our method requires fine-tuning only
an additional 80M parameters—approximately one-fifteenth
of the total parameters in the Transformer model—greatly
reducing the training cost. Furthermore, benefiting from a
unified 3D human representation, our framework can effort-
lessly generate animated characters in diverse poses.
Dream3DAvatar combines the diversity of diffusion-
based generation with the deterministic efficiency of feed-
forward Transformers, enabling fast and controllable 3D hu-
man reconstruction. We conduct both quantitative and qual-
itative comparisons with other methods in single-image to
multi-view generation and multi-view to 3D reconstruction
tasks. Results demonstrate that Dream3DAvatar consistently
delivers superior performance across various challenging
scenarios. In summary, our main contributions are as fol-
lows:
• We propose Dream3DAvatar, a two-stage framework
for efficient and text-controllable 3D human reconstruc-
tion from a single image, addressing key challenges of
monocular ambiguity and limited controllability.
• We design an SDXL-based module that generates multi-
view images with consistent texture and geometry from
a single image, guided by text.
• We introduce a feed-forward transformer model that
fuses facial features and multi-view features for high-
quality 3D avatars reconstruction.
2
Related Work
2.1
Stable Diffusion for Multi-View Image
Generation
Recent advances have adapted Stable Diffusion for novel-
view synthesis. Early works (Shi et al. 2023a,c) introduced
camera-conditioned diffusion to generate plausible views
from a single image. Later efforts (Liu et al. 2023; Shi
et al. 2023b; Tang et al. 2024; Li et al. 2024; Wu et al.
2024; Long et al. 2024) improved cross-view consistency
and resolution. MV-Adapter (Huang et al. 2024) proposed a
lightweight adapter for efficient fine-tuning, balancing qual-
ity and speed. However, these generic models struggle to
capture articulated human body structures.
To address this, human-specific models have been devel-
oped. Bhunia et al. (Bhunia et al. 2023) used pose condition-
ing to handle large articulations. Others (Shao et al. 2024;
Kant et al. 2025) adopted transformer-based diffusion for
multi-view synthesis. Liu et al. (Liu et al. 2024) extended
diffusion to videos. MagicMan (He et al. 2025a) incorpo-
rated SMPL and normal maps to produce coherent full-body
views, but lacked facial detail. PSHuman (Li et al. 2025)
introduced separate body and face branches for improved
facial fidelity, yet at high computational cost and without
explicit control over occlusions.
2.2
Single-Image 3D Human Reconstruction
Reconstructing 3D humans from a single image is inher-
ently ill-posed due to occlusions and missing views. Implicit
methods (Saito et al. 2019, 2020) learn pixel-aligned occu-
pancy or SDFs for detail recovery but require dense supervi-
sion and struggle with complex poses. Tri-plane representa-
tions (Wang et al. 2023; Zhang et al. 2024a) improve speed
and training efficiency, but suffer from weak priors and over-
smoothing in unseen regions.
SMPL-based approaches (Loper et al. 2023; Pavlakos
et al. 2019; Xiu et al. 2022, 2023) introduce pose and topol-
ogy constraints, enhancing robustness but inheriting fitting
artifacts. Feed-forward methods (Qiu et al. 2025a; Zhuang
et al. 2025; Zhang et al. 2024b) directly predict animatable
3D humans from a single view, offering real-time inference
but lacking view-aware reasoning.
Recent methods (Li et al. 2025; He et al. 2025b; Pan et al.
2024; Ho et al. 2024; Weng et al. 2024) hallucinate pseudo-
views to reduce ambiguity, though often with limited con-
trol and view inconsistency. AniGS (Qiu et al. 2025b) gen-
erates multi-view sequences via video diffusion and fuses
them into 4D Gaussian splats, achieving coherence but with
efficiency trade-offs.

<!-- page 3 -->
Figure 2: The overall framework of Dream3DAvatar. (Left) A diffusion model that generates multi-view images from a
single input image, incorporating SMPL-X geometry, skeleton and high-resolution facial images, and using Pose-Adapter and
ID-Adapter-G modules to achieve pose consistency and identity preservation. (Right) A feed-forward Transformer model that
reconstructs high-fidelity 3D avatars from the generated multi-view images, significantly enhancing detail recovery and identity
consistency through multi-view feature fusion and the ID-Adapter-R module.
Our two-stage framework first synthesizes consistent
multi-view images, then reconstructs a controllable 3D
Gaussian Splatting representation, achieving both high fi-
delity and editability from a single image.
3
Method
3.1
Overview
We propose Dream3DAvatar, a two-stage framework for
reconstructing high-fidelity 3D avatars from a single im-
age (Fig. 2). The pipeline comprises: (1) a multi-view
image generation stage leveraging SDXL (Podell et al.
2023) with Pose-Adapter and ID-Adapter-G modules to
produce consistent multi-view images from single in-
put, where Pose-Adapter enforces geometric consistency
using SMPL-X and skeletal priors while ID-Adapter-G
preserves identity through high-resolution facial features,
guided by text prompts for occluded regions; and (2) a 3D
avatar reconstruction stage employing a feedforward Trans-
former (Zhuang et al. 2025) with multi-view feature fusion
to generate 3D Gaussian splats (3DGS), enhanced by ID-
Adapter-R’s gating mechanism for facial detail recovery.
Section 3.2 details the multi-view generation, Section 3.3
describes the reconstruction model, and Section 3.4 explains
the training methodology.
3.2
Multi-view Image Generation
Geometric and Semantic Conditioning
As shown on the
left side of Figure 2, we augment the pre-trained SDXL
model with several Adapters to generate high-fidelity, con-
sistent multi-view images from a single person image.
Specifically, we train a diffusion model capable of encoding
SMPL-X images, skeletons, high-resolution facial features,
and text, achieving consistency constraints from multiple an-
gles such as geometry, pose, and texture for multi-view im-
ages.
SMPL-X Guider. Training a diffusion model to gener-
ate multi-view images from a single person image while
learning pose information is a significant challenge. To ad-
dress this, we introduce the Pose-Adapter, which encodes
multi-view SMPL-X images and injects them into the SDXL
model, enforcing consistency on the generated multi-view
avatars from the perspectives of geometry and pose. These
multi-view SMPL-X images are estimated from a single in-
put image using a pre-trained pose estimation model, then
rendered according to specified camera parameters.
Skeleton Guider. While SMPL-X provides good geomet-
ric consistency across multi-view images, it fails to maintain
pose consistency for joints such as fingers. Therefore, we
utilize the Pose-Adapter to inject body skeleton information
into the model. The body skeleton is calculated using the
SMPL-X parameters.
Face Guider. The face region occupies only a small por-
tion of the image, yet its reconstruction quality is crucial
for preserving the identity and overall reconstruction accu-
racy. Thus, we extract high-resolution facial features from
the input single-view image and inject them into the diffu-
sion model using the ID-Adapter-G, thereby enhancing the
quality of facial region reconstruction.
Text Guider. During training, we employ an advanced
Vision-Language Model (VLM) to extract descriptive text
from the person image as an additional conditioning input.
While the aforementioned conditions allow SDXL to gen-
erate reasonably coherent multi-view avatars, the texture of
occluded regions in a single-view image might not meet user
expectations due to the partial information it contains. As
shown in Figure 5, by leveraging text conditions, we gain
control over the texture details in the unseen region of gen-
erated multi-view images, ensuring that the textures align
with the desired characteristics.

<!-- page 4 -->
Figure 3: The feedforward transformer block.
Adapter-Enhanced Diffusion model
We design the
Pose-Adapter and ID-Adapter-G to inject human pose in-
formation, and facial information into the pre-trained SDXL
model to generate high-fidelity, consistent multi-view im-
ages while retaining the structure and prior knowledge of
the pre-trained SDXL model.
Pose-Adapter. The Pose-Adapter extracts multi-scale
pose features from multi-view SMPLX images and skeleton
images, and injects them into the encoder of the pre-trained
UNet to add pose constraints. Considering that we use two
pose-guided conditions, which contain different character-
istics of the human body, we are inspired by (Zhu et al.
2024) and introduce self-attention for each pose condition.
This allows the model to capture the semantic information of
different pose conditions. In each layer of the pose adapter,
the two pose features are aggregated through summing and
then input into the corresponding layer of UNet. The Pose-
Adapter can be formulated as

F 1
pose, F 2
pose, F 3
pose, F 4
pose
	
= FP −AD(I1:N
smplx, I1:N
skeleton)
(1)
where F i
pose, i = 1, 2, 3, 4 are the extracted multi-scale pose
features,
FP −AD is Pose-Adapter function,
I1:N
smplx are
multi-view SMPLX images and
I1:N
skeleton are multi-view
skeleton images, N is number of views.
ID-Adapter-G. The face occupies a very small area in
the reference image, which makes it difficult for the model
to capture the information about the face, resulting in distor-
tion of the face in the generated multi-view image. To solve
this problem, We design ID adapter-G to extract facial fea-
tures. Inspired by (Wang et al. 2024a), ID adapter-G uses a
face encoder to extract face ID embedding from reference
face image, which can provide fine-grained facial features.
The face embedding are then injected into the UNet through
a projection layer. We add global cross-attention and local
cross-attention to the self-attention layer of SDXL to fuse
the body features of the reference image extracted from the
referencenet and facial features with multi-view features. To
improve the consistency of multiple views, we added row-
wise self-attention (Li et al. 2024) to the self-attention layer
to enable effective information exchange between views.
Like MV-Adapter (Huang et al. 2024), we adopt a parallel
structure between different attention layers, which can de-
couple the attention layers and preserve the prior knowledge
of SDXL. The attention process can be formulated as
hout = SelfAttn(hin)
+ GlobalCrossAttn(hin, href)
+ LocalCrossAttn(hin, hface)
+ RowWiseAttn(hin) + hin
(2)
where hin refers to the input hidden states in the self atten-
tion layer, href refers to the reference image hidden states,
hface refers to the face hidden states.
3.3
Multi-view Transformer for 3D Recovery
As shown on the right side of Figure 2, we recon-
struct high-fidelity 3D human avatars represented by 3D
Gaussian Splatting from the generated multi-view images.
The reconstruction process is powered by a feedforward
Transformer-based model that integrates multi-view features
and maps them into a 2D UV space defined by the SMPL-X
model. Each 3D Gaussian primitive is associated with at-
tributes—including color, opacity, scale, rotation, and po-
sition—determined within this UV space, which provides
strong geometric priors for body modeling.
To enhance cross-view feature alignment, we design a
dedicated multi-view fusion module that facilitates spatial
correspondence and information aggregation across differ-
ent viewpoints. The use of multi-view inputs significantly
mitigates the ill-posedness introduced by monocular images
by reducing occluded and invisible regions.
Furthermore, to refine the reconstruction of high-
frequency facial details, we incorporate an identity-aware
module, ID-Adapter-R, which injects high-resolution fa-
cial features into the Transformer pipeline. This integration
substantially improves both identity preservation and fine-
grained detail recovery in the final 3D output.
Multi-view Body Feature Fusion
To address the chal-
lenges of occlusions and ambiguities in body geometry
caused by monocular input, we propose a lightweight multi-
view body feature fusion module (MVBF), as shown on the
left side of Figure 3. The module includes parallel spatial at-
tention and view attention mechanisms, where the former fo-
cuses on intra-view relationships to enhance body part con-
sistency, while the latter aligns features across views to re-
solve ambiguities and occlusions. The outputs of both atten-
tion mechanisms are fused and weighted through a learn-
able fusion gate, which adaptively adjusts the contribution
of spatial and view features. Finally, the fused feature map
is passed to a feedforward Transformer for high-quality 3D
reconstruction in the UV space. By efficiently aggregating
multi-view features, our module effectively mitigates the ill-
posed nature of monocular image reconstruction, enabling
consistent and detailed 3D human avatar generation.
Identity-aware 3D Reconstruction
Due to the small size
of the facial region, the generated 3D avatar often exhibits
artifacts on the face. To achieve high-fidelity identity preser-
vation, we introduce ID-ADAPTER-R, shown on the right
side of Figure 3, which injects high-resolution facial features
into the 3D generation pipeline. First, a feature expansion
network maps the facial features to the target UV space fea-
ture dimensions. Then, a gated feature fusion mechanism in-

<!-- page 5 -->
Figure 4: Qualitative comparison of comparison experiments on multi-view human image generation.
Figure 5: The Text Control Results of Multi-view Genera-
tion
.
tegrates the facial features with the UV space, enhancing fa-
cial details while preserving the original body features. Ad-
ditionally, we inject facial features only in the latter half of
the feedforward Transformer, as the first half mainly handles
the low-frequency information related to body geometry re-
construction, while facial details belong to high-frequency
information. The selective feature injection strategy ensures
fine-grained identity preservation while effectively reducing
computational overhead.
Representation for 3D Avatars
Similar to previous
works (Zhuang et al. 2025; Zhang et al. 2024b; Hu et al.
2024), we represent the human body using 3D Gaussian
Splatting (Kerbl et al. 2023) and initialize the Gaussian
primitives with SMPL-X vertices. Formally, each Gaussian
primitive is defined as Gk = {µk, αk, rk, sk, ck}, where µk,
αk, rk, sk, and ck denote the position, opacity, rotation,
scale, and color of the Gaussian sphere, respectively. Each
Gaussian primitive is mapped to a corresponding pixel in the
2D UV space, and its attributes are indirectly regressed by
predicting the properties on the UV plane. This design sig-
nificantly reduces computational complexity and, with the
geometric prior provided by SMPL-X, enables efficient gen-
eration of 3D human animations. Given a target pose repre-
sented by SMPL-X parameters, we update the positions and
rotations of all Gaussian primitives via Linear Blend Skin-
ning (LBS) while keeping other attributes unchanged, en-
abling fast and flexible pose transformations of 3D avatars.
3.4
Training Strategy
Multi-view Image Generation
In the multi-view image
generation stage, we take SMPL-X images, skeleton images,
reference images, reference face images and text as input
samples. During training, we only train Pose-Adapter and
ID-Adapter-G and freeze the parameters of the pre-trained
SDXL. The objective function can be defined as:
Ez0,ϵ∼N (0,I),c,t ∥ϵ −ϵθ(zt, c, t)∥2
(3)
where c = {ct, cr, cf, cp}, ct represents text, cr represents
reference image, cf represents face image, cp represents
SMPL-X images and skeleton images, and θ represents the
parameters of Pose-Adapter and ID-Adapter-G.
Our experiments show that ID-Adapter-G is overly depen-
dent on the features of the reference image, which limits the
learning of face image features. To address this issue, we
randomly mask the face area of the reference image dur-
ing training. Specifically, we mask the face area of the ref-
erence image with a probability of 30% to encourage the
model to make more use of face image features. To enable
class free guidance during inference, we randomly drop text
conditions, reference image conditions, face conditions, and
pose conditions with a probability of 10% during training.
Multi-view Transformer for 3D Recovery
In the sec-
ond stage, we treat the multi-view images IN
body, the high-
resolution facial crop from the frontal view Iface, and the

<!-- page 6 -->
Figure 6: Qualitative comparison of 3D human reconstruction
corresponding SMPL-X and camera parameters as a com-
plete input sample. These inputs are jointly fed into a feed-
forward Transformer model to generate a canonical 3D hu-
man template. The template is then deformed according to
the SMPL-X model and differentiably rendered under the
given camera parameters πt to produce multi-view projec-
tions ˆIN
body. The loss function is computed as follows:
L =λrgb Lrgb(IN
body,ˆIN
body)
+ λlpips Llpips(IN
body,ˆIN
body)
+ λface Llpips(Iface,ˆIface)
(4)
where λrgb, λlpips, and λface are weighting coefficients that
balance the contributions of each loss component.
4
Experiments
4.1
Experimental Setting
Dataset
To fine-tune Dream3DAvatar, we conducted
experiments on several 3D human datasets and video
data,
including
THuman2.1
(Zheng
et
al.
2019),
HuGe100K (Zhuang et al. 2025), and partial video se-
quences with 3D annotations from Human4DiT (Shao et al.
2024). Specifically, for 3D human data, we rendered 72
uniformly distributed views at two resolutions, 768 × 768
and 896 × 640. Video data were cropped and resized to the
same resolutions. In addition, SMPL-X parameters were
estimated using Multi-HMR (Baradel et al. 2024). The
two stages were trained with different data configurations.
In the multi-view image generation stage, one reference
image containing the full face was randomly selected,
and six target views at corresponding intervals were used
for training. In the 3D human reconstruction stage, four
randomly selected images, including at least one frontal
view, were used for self-supervised optimization. For quan-
titative evaluation, the last 50 samples from HuGe100K,
THuman2.1, and Human4DiT were used as the test set. In
addition, real-world examples collected from the internet
were employed for qualitative comparison.
Implementation Details
Dream3DAvatar was fine-tuned
on four NVIDIA A800 GPUs. We adopted the AdamW
optimizer (Kinga, Adam et al. 2015), with initial learning
rates set to 5 × 10−5 and 1 × 10−5 for the two stages, re-
spectively. The multi-view generation and 3D reconstruc-
tion stages were trained for 40k and 10k iterations, respec-
tively, with a mini-batch size of 1. For evaluation, generated
multi-view images were compared with ground truth using

<!-- page 7 -->
Method
MSE ↓
PSNR ↑
SSIM ↑
LPIPS ↓
SV3D
0.0352
15.68
0.8500
0.2008
MagicMan
0.0260
19.22
0.8733
0.1549
PSHuman
0.0149
20.23
0.8958
0.1087
MV-Adapter
0.0298
19.47
0.8688
0.1881
w/o face guider
0.0080
21.11
0.9185
0.0863
w/o skeleton guider
0.0092
20.47
0.9167
0.0891
w/o smplx guider
0.0094
20.36
0.9161
0.0906
Ours
0.0052
22.98
0.9277
0.0711
Table 1: Quantitative comparison of multi-view human im-
age generation and ablation experiments.
Method
MSE ↓
PSNR ↑
LPIPS ↓
DreamGaussian
0.042
14.642
1.654
SIFU
0.054
13.897
1.757
CRM
0.031
16.211
1.592
IDOL
0.011
20.963
1.289
Ours
0.009
21.322
1.097
Table 2: Quantitative comparison of 3D avatar reconstruc-
tion.
Learned Perceptual Image Patch Similarity (LPIPS) (Zhang
et al. 2018), Peak Signal-to-Noise Ratio (PSNR), and mean
squared error (MSE) as quantitative metrics. The total train-
ing time was approximately 4 hours for the multi-view gen-
eration stage and 10 hours for the 3D reconstruction stage.
More details on data pre-processing and implementation are
provided in the supplementary material.
4.2
Multi-view Image Generation
For multi-view generation, we compared our method with
four state-of-the-art baseline methods, including SV3D (Vo-
leti et al. 2024), PSHuman (Li et al. 2025), Magicman (He
et al. 2025a), and MVAdapter (Huang et al. 2024). For the
qualitative results, as shown in left of figure 4, SV3D per-
forms poorly in terms of detail preservation and multi-view
consistency. PSHuman is human-specific multi-view gener-
ation methods. It introduces body-face diffusion, which re-
tains facial information, but still has defects in some detailed
parts such as hands. Due to the lack of human body priors,
MV-Adapter has deformities in human body geometry and
face. Our method can generate high-quality, consistent, and
identity-preserving multi-view images. As for the quantita-
tive results, as shown in table 1, we evaluate our method and
baselines on a subset of THuman2.1, and the results show
that our method outperforms the existing baseline methods
in all four metrics. We attribute this to our designed pose-
adapter that provides body geometry information and ID-
adapter-g that provides face information.
4.3
3D avatar reconstruction
We evaluated our method against four representative base-
lines on the 3D avatar reconstruction task from both quali-
tative and quantitative perspectives. DreamGaussian (Tang
et al. 2023) leverages 2D diffusion priors with score dis-
tillation sampling to iteratively refine 3D representations
and adopts a progressive Gaussian densification strategy
for faster convergence. CRM (Wang et al. 2024b) recon-
structs meshes under multi-view supervision using RGB im-
ages and normal maps, achieving moderate improvements
in multi-view consistency. SIFU (Zhang, Yang, and Yang
2024) rely on iterative geometric optimization and pixel-
level alignment to enhance fine-grained details and high-
fidelity reconstruction. IDOL (Zhuang et al. 2025) em-
ploys a feedforward Transformer to directly predict 3D rep-
resentations from a single image. As shown in Table 2,
Dream3DAvatar consistently outperforms these baselines
across all metrics. Qualitative comparisons in Figure 6 fur-
ther demonstrate its superior reconstruction performance
across various human types and challenging poses. It is
worth noting that, for the sake of fairness, our method also
uses single images for comparison instead of multi-view im-
ages. It can be seen that Dream3DAvatar achieves impres-
sive facial detail preservation, while the baseline method
struggles to recover high-frequency details and maintain tex-
ture consistency.
4.4
Ablation Study and Analysis
Multi-view Image Generation
The right of Figure 4 illus-
trates the contribution of each module to multi-view image
generation. The proposed Pose-Adapter provides strong ge-
ometric constraints, enabling pose-consistent synthesis even
under challenging postures, where most baselines fail. Ad-
ditional skeletal information improves the reconstruction of
key body parts, such as hands. Furthermore, ID-Adapter-G
significantly enhances facial details, ensuring better identity
consistency across views.
3D avatar reconstruction
In the 3D reconstruction stage,
we explore the effects of the number of viewpoints and
ID-Adapter-R. Increasing the number of viewpoints helps
cover more occluded areas, significantly improving the con-
sistency of global geometry and texture. Removing ID-
Adapter-R, on the other hand, results in a loss of facial de-
tails and degradation of identity features, validating its crit-
ical role in the fusion of high-frequency facial information.
More results can be found in the appendix.
5
Conclusion
In this study, we propose Dream3DAvatar, a novel two-
stage framework for efficient, controllable, and high-fidelity
3D human avatar reconstruction from a single image. By
mitigating the ill-posedness inherent in monocular recon-
struction and enhancing controllability, our method effec-
tively bridges the gap between generative diversity and re-
construction efficiency. Specifically, we design a lightweight
multi-view generation module based on SDXL, incorporat-
ing geometric and semantic constraints to achieve view-
consistent image synthesis. This is followed by a feedfor-
ward Transformer network equipped with an ID Adapter to
further improve the accuracy and detail of 3D reconstruc-
tion. Extensive experiments demonstrate that our method

<!-- page 8 -->
achieves state-of-the-art performance on multiple bench-
marks for both multi-view image synthesis and 3D recon-
struction. In future work, we plan to explore driving expres-
sive facial animations, beyond just body motion generation.
References
Alayrac, J.-B.; Donahue, J.; Luc, P.; Miech, A.; Barr, I.; Has-
son, Y.; Lenc, K.; Mensch, A.; Millican, K.; Reynolds, M.;
et al. 2022. Flamingo: a visual language model for few-shot
learning. Advances in neural information processing sys-
tems, 35: 23716–23736.
AlBahar, B.; Saito, S.; Tseng, H.-Y.; Kim, C.; Kopf, J.; and
Huang, J.-B. 2023. Single-image 3d human digitization with
shape-guided diffusion. In SIGGRAPH Asia 2023 Confer-
ence Papers, 1–11.
Alldieck, T.; Magnor, M.; Xu, W.; Theobalt, C.; and Pons-
Moll, G. 2018.
Video based reconstruction of 3d people
models. In Proceedings of the IEEE Conference on Com-
puter Vision and Pattern Recognition, 8387–8397.
Baradel, F.; Armando, M.; Galaaoui, S.; Br´egier, R.; Wein-
zaepfel, P.; Rogez, G.; and Lucas, T. 2024.
Multi-hmr:
Multi-person whole-body human mesh recovery in a single
shot. In European Conference on Computer Vision, 202–
218. Springer.
Bhunia, A. K.; Khan, S.; Cholakkal, H.; Anwer, R. M.;
Laaksonen, J.; Shah, M.; and Khan, F. S. 2023. Person im-
age synthesis via denoising diffusion model. In Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, 5968–5976.
Choi, Y. 2025. SVAD: From Single Image to 3D Avatar via
Synthetic Data Generation with Video Diffusion and Data
Augmentation. In Proceedings of the Computer Vision and
Pattern Recognition Conference, 3137–3147.
Han, S.-H.; Park, M.-G.; Yoon, J. H.; Kang, J.-M.; Park, Y.-
J.; and Jeon, H.-G. 2023. High-fidelity 3d human digitiza-
tion from single 2k resolution images. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 12869–12879.
He, K.; Chen, X.; Xie, S.; Li, Y.; Doll´ar, P.; and Girshick,
R. 2022. Masked autoencoders are scalable vision learners.
In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, 16000–16009.
He, X.; Wu, Z.; Li, X.; Kang, D.; Zhang, C.; Ye, J.; Chen,
L.; Gao, X.; Zhang, H.; and Zhuang, H. 2025a. Magicman:
Generative novel view synthesis of humans with 3d-aware
diffusion and iterative refinement.
In Proceedings of the
AAAI Conference on Artificial Intelligence, 3437–3445.
He, Y.; Zhou, Y.; Zhao, W.; Wu, Z.; Xiao, K.; Yang, W.; Liu,
Y.-J.; and Han, X. 2025b. Stdgen: Semantic-decomposed 3d
character generation from single images. In Proceedings of
the Computer Vision and Pattern Recognition Conference,
26345–26355.
Ho, I.; Song, J.; Hilliges, O.; et al. 2024. Sith: Single-view
textured human reconstruction with image-conditioned dif-
fusion.
In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 538–549.
Hu, L.; Zhang, H.; Zhang, Y.; Zhou, B.; Liu, B.; Zhang, S.;
and Nie, L. 2024. Gaussianavatar: Towards realistic human
avatar modeling from a single video via animatable 3d gaus-
sians. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, 634–644.
Huang, Z.; Guo, Y.-C.; Wang, H.; Yi, R.; Ma, L.; Cao, Y.-P.;
and Sheng, L. 2024. Mv-adapter: Multi-view consistent im-
age generation made easy. arXiv preprint arXiv:2412.03632.
Kant, Y.; Weber, E.; Kim, J. K.; Khirodkar, R.; Zhaoen, S.;
Martinez, J.; Gilitschenski, I.; Saito, S.; and Bagautdinov,
T. 2025. Pippo: High-resolution multi-view humans from a
single image. In Proceedings of the Computer Vision and
Pattern Recognition Conference, 16418–16429.
Kerbl, B.; Kopanas, G.; Leimk¨uhler, T.; and Drettakis, G.
2023. 3D Gaussian splatting for real-time radiance field ren-
dering. ACM Trans. Graph., 42(4): 139–1.
Kinga, D.; Adam, J. B.; et al. 2015. A method for stochastic
optimization. In International conference on learning rep-
resentations (ICLR). California;.
Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023. Blip-2: Boot-
strapping language-image pre-training with frozen image
encoders and large language models. In International con-
ference on machine learning, 19730–19742. PMLR.
Li, P.; Liu, Y.; Long, X.; Zhang, F.; Lin, C.; Li, M.; Qi, X.;
Zhang, S.; Xue, W.; Luo, W.; et al. 2024.
Era3d: High-
resolution multiview diffusion using efficient row-wise at-
tention. Advances in Neural Information Processing Sys-
tems, 37: 55975–56000.
Li, P.; Zheng, W.; Liu, Y.; Yu, T.; Li, Y.; Qi, X.; Chi, X.; Xia,
S.; Cao, Y.-P.; Xue, W.; et al. 2025. Pshuman: Photorealis-
tic single-image 3d human reconstruction using cross-scale
multiview diffusion and explicit remeshing. In Proceedings
of the Computer Vision and Pattern Recognition Conference,
16008–16018.
Liu, Y.; Lin, C.; Zeng, Z.; Long, X.; Liu, L.; Komura, T.;
and Wang, W. 2023. Syncdreamer: Generating multiview-
consistent images from a single-view image. arXiv preprint
arXiv:2309.03453.
Liu, Z.; Dong, H.; Chharia, A.; and Wu, H. 2024.
Human-vdm: Learning single-image 3d human gaussian
splatting from video diffusion models.
arXiv preprint
arXiv:2409.02851.
Long, X.; Guo, Y.-C.; Lin, C.; Liu, Y.; Dou, Z.; Liu, L.; Ma,
Y.; Zhang, S.-H.; Habermann, M.; Theobalt, C.; et al. 2024.
Wonder3d: Single image to 3d using cross-domain diffusion.
In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, 9970–9980.
Loper, M.; Mahmood, N.; Romero, J.; Pons-Moll, G.; and
Black, M. J. 2023. SMPL: A skinned multi-person linear
model. In Seminal Graphics Papers: Pushing the Bound-
aries, Volume 2, 851–866. Association for Computing Ma-
chinery.
Pan, P.; Su, Z.; Lin, C.; Fan, Z.; Zhang, Y.; Li, Z.; Shen,
T.; Mu, Y.; and Liu, Y. 2024. Humansplat: Generalizable
single-image human gaussian splatting with structure pri-
ors. Advances in Neural Information Processing Systems,
37: 74383–74410.

<!-- page 9 -->
Pavlakos, G.; Choutas, V.; Ghorbani, N.; Bolkart, T.; Osman,
A. A.; Tzionas, D.; and Black, M. J. 2019. Expressive body
capture: 3d hands, face, and body from a single image. In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, 10975–10985.
Podell, D.; English, Z.; Lacey, K.; Blattmann, A.; Dockhorn,
T.; M¨uller, J.; Penna, J.; and Rombach, R. 2023. Sdxl: Im-
proving latent diffusion models for high-resolution image
synthesis. arXiv preprint arXiv:2307.01952.
Qiu, L.; Gu, X.; Li, P.; Zuo, Q.; Shen, W.; Zhang, J.; Qiu,
K.; Yuan, W.; Chen, G.; Dong, Z.; et al. 2025a. Lhm: Large
animatable human reconstruction model from a single image
in seconds. arXiv preprint arXiv:2503.10625.
Qiu, L.; Zhu, S.; Zuo, Q.; Gu, X.; Dong, Y.; Zhang, J.; Xu,
C.; Li, Z.; Yuan, W.; Bo, L.; et al. 2025b. Anigs: Animatable
gaussian avatar from a single image with inconsistent gaus-
sian reconstruction. In Proceedings of the Computer Vision
and Pattern Recognition Conference, 21148–21158.
Saito,
S.;
Huang,
Z.;
Natsume,
R.;
Morishima,
S.;
Kanazawa, A.; and Li, H. 2019. Pifu: Pixel-aligned implicit
function for high-resolution clothed human digitization. In
Proceedings of the IEEE/CVF international conference on
computer vision, 2304–2314.
Saito, S.; Simon, T.; Saragih, J.; and Joo, H. 2020.
Pi-
fuhd: Multi-level pixel-aligned implicit function for high-
resolution 3d human digitization.
In Proceedings of the
IEEE/CVF conference on computer vision and pattern
recognition, 84–93.
Shao, R.; Pang, Y.; Zheng, Z.; Sun, J.; and Liu, Y. 2024. Hu-
man4dit: 360-degree human video generation with 4d diffu-
sion transformer. arXiv preprint arXiv:2405.17405.
Shi, R.; Chen, H.; Zhang, Z.; Liu, M.; Xu, C.; Wei, X.; Chen,
L.; Zeng, C.; and Su, H. 2023a. Zero123++: a single im-
age to consistent multi-view diffusion base model. arXiv
preprint arXiv:2310.15110.
Shi, Y.; Wang, J.; Cao, H.; Tang, B.; Qi, X.; Yang, T.; Huang,
Y.; Liu, S.; Zhang, L.; and Shum, H.-Y. 2023b. Toss: High-
quality text-guided novel view synthesis from a single im-
age. arXiv preprint arXiv:2310.10644.
Shi, Y.; Wang, P.; Ye, J.; Long, M.; Li, K.; and Yang, X.
2023c. Mvdream: Multi-view diffusion for 3d generation.
arXiv preprint arXiv:2308.16512.
Tang, J.; Ren, J.; Zhou, H.; Liu, Z.; and Zeng, G. 2023.
Dreamgaussian: Generative gaussian splatting for efficient
3d content creation. arXiv preprint arXiv:2309.16653.
Tang, S.; Chen, J.; Wang, D.; Tang, C.; Zhang, F.; Fan, Y.;
Chandra, V.; Furukawa, Y.; and Ranjan, R. 2024. Mvdiffu-
sion++: A dense high-resolution multi-view diffusion model
for single or sparse-view 3d object reconstruction. In Euro-
pean Conference on Computer Vision, 175–191. Springer.
Voleti, V.; Yao, C.-H.; Boss, M.; Letts, A.; Pankratz, D.;
Tochilkin, D.; Laforte, C.; Rombach, R.; and Jampani, V.
2024. Sv3d: Novel multi-view synthesis and 3d generation
from a single image using latent video diffusion. In Euro-
pean Conference on Computer Vision, 439–457. Springer.
Wang, Q.; Bai, X.; Wang, H.; Qin, Z.; Chen, A.; Li,
H.; Tang, X.; and Hu, Y. 2024a.
Instantid: Zero-shot
identity-preserving generation in seconds.
arXiv preprint
arXiv:2401.07519.
Wang, T.; Zhang, B.; Zhang, T.; Gu, S.; Bao, J.; Baltrusaitis,
T.; Shen, J.; Chen, D.; Wen, F.; Chen, Q.; et al. 2023. Rodin:
A generative model for sculpting 3d digital avatars using
diffusion. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 4563–4573.
Wang, Z.; Wang, Y.; Chen, Y.; Xiang, C.; Chen, S.; Yu, D.;
Li, C.; Su, H.; and Zhu, J. 2024b. Crm: Single image to 3d
textured mesh with convolutional reconstruction model. In
European conference on computer vision, 57–74. Springer.
Weng, Z.; Liu, J.; Tan, H.; Xu, Z.; Zhou, Y.; Yeung-Levy,
S.; and Yang, J. 2024.
Template-free single-view 3d hu-
man digitalization with diffusion-guided lrm. arXiv preprint
arXiv:2401.12175.
Wu, K.; Liu, F.; Cai, Z.; Yan, R.; Wang, H.; Hu, Y.; Duan, Y.;
and Ma, K. 2024. Unique3d: High-quality and efficient 3d
mesh generation from a single image. In The Thirty-eighth
Annual Conference on Neural Information Processing Sys-
tems.
Xiu, Y.; Yang, J.; Cao, X.; Tzionas, D.; and Black, M. J.
2023. Econ: Explicit clothed humans optimized via normal
integration. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, 512–523.
Xiu, Y.; Yang, J.; Tzionas, D.; and Black, M. J. 2022.
Icon: Implicit clothed humans obtained from normals. In
2022 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR), 13286–13296. IEEE.
Zhang, B.; Cheng, Y.; Wang, C.; Zhang, T.; Yang, J.; Tang,
Y.; Zhao, F.; Chen, D.; and Guo, B. 2024a. Rodinhd: High-
fidelity 3d avatar generation with diffusion models. In Eu-
ropean Conference on Computer Vision, 465–483. Springer.
Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang,
O. 2018. The unreasonable effectiveness of deep features as
a perceptual metric. In Proceedings of the IEEE conference
on computer vision and pattern recognition, 586–595.
Zhang, W.; Yan, Y.; Liu, Y.; Sheng, X.; and Yang, X. 2024b.
E 3Gen: Efficient, Expressive and Editable Avatars Genera-
tion. In Proceedings of the 32nd ACM International Confer-
ence on Multimedia, 6860–6869.
Zhang, Z.; Yang, Z.; and Yang, Y. 2024. Sifu: Side-view
conditioned implicit function for real-world usable clothed
human reconstruction.
In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
9936–9947.
Zheng, Z.; Yu, T.; Liu, Y.; and Dai, Q. 2021. Pamir: Para-
metric model-conditioned implicit representation for image-
based human reconstruction. IEEE transactions on pattern
analysis and machine intelligence, 44(6): 3170–3184.
Zheng, Z.; Yu, T.; Wei, Y.; Dai, Q.; and Liu, Y. 2019. Dee-
phuman: 3d human reconstruction from a single image. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, 7739–7749.

<!-- page 10 -->
Zhu, S.; Chen, J. L.; Dai, Z.; Dong, Z.; Xu, Y.; Cao, X.; Yao,
Y.; Zhu, H.; and Zhu, S. 2024. Champ: Controllable and
consistent human image animation with 3d parametric guid-
ance. In European Conference on Computer Vision, 145–
162. Springer.
Zhuang, Y.; Lv, J.; Wen, H.; Shuai, Q.; Zeng, A.; Zhu, H.;
Chen, S.; Yang, Y.; Cao, X.; and Liu, W. 2025. Idol: Instant
photorealistic 3d human creation from a single image. In
Proceedings of the Computer Vision and Pattern Recogni-
tion Conference, 26308–26319.
