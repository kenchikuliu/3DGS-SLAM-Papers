<!-- page 1 -->
Preprint.
UNILAT3D: GEOMETRY-APPEARANCE UNIFIED
LATENTS FOR SINGLE-STAGE 3D GENERATION
Guanjun Wu1,2*
Jiemin Fang1*†
Chen Yang1*
Sikuang Li1,3
Taoran Yi1,2
Jia Lu1,2
Zanwei Zhou1,3
Jiazhong Cen1,3
Lingxi Xie1
Xiaopeng Zhang1
Wei Wei2
Wenyu Liu2
Xinggang Wang2
Qi Tian1†
*Equal Contribution.
†Corresponding Authors.
1Huawei Inc. 2Huazhong University of Science and Technology 3Shanghai Jiaotong University
Figure 1: Gallery of UniLat3D. Our method generates high quality 3D assets in seconds.
ABSTRACT
High-fidelity 3D asset generation is crucial for various industries. While recent
3D pretrained models show strong capability in producing realistic content, most
are built upon diffusion models and follow a two-stage pipeline that first generates
geometry and then synthesizes appearance. Such a decoupled design tends to pro-
duce geometry–texture misalignment and non-negligible cost. In this paper, we
propose UniLat3D, a unified framework that encodes geometry and appearance
in a single latent space, enabling direct single-stage generation. Our key contri-
bution is a geometry–appearance Unified VAE, which compresses high-resolution
sparse features into a compact latent representation – UniLat. UniLat integrates
structural and visual information into a dense low-resolution latent, which can
be efficiently decoded into diverse 3D formats, e.g., 3D Gaussians and meshes.
Based on this unified representation, we train a single flow-matching model to
map Gaussian noise directly into UniLat, eliminating redundant stages. Trained
solely on public datasets, UniLat3D produces high-quality 3D assets in seconds
from a single image, achieving superior appearance fidelity and geometric quality.
More demos and code are available at https://unilat3d.github.io/
1
INTRODUCTION
3D content generation has witnessed rapid growth in recent years, becoming an increasingly es-
sential capability across various applications, including game/film production, virtual/augmented
1
arXiv:2509.25079v1  [cs.CV]  29 Sep 2025

<!-- page 2 -->
Preprint.
reality, industrial design, and embodied AI. Recent advances in 3D generative frameworks (Zhang
et al., 2024c; Hunyuan3D et al., 2025; Lai et al., 2025; Yang et al., 2024c; Zhao et al., 2025; Xiang
et al., 2024; Li et al., 2025a; Hong et al., 2023; Zhang et al., 2024b; Ma et al., 2025; Ren et al.,
2024a; Zou et al., 2024) have demonstrated impressive progress in synthesizing vivid and realistic
3D assets, while some approaches (Li et al., 2024; Wu et al., 2024b; 2025b; Chen et al., 2025d; Li
et al., 2025c; Ye et al., 2025) dive into accurate geometry and fine-grained shape generation.
Despite this rapid progress, the majority of recent high-quality 3D generation frameworks are
diffusion-based, and typically adopt a multi-stage design: geometry is generated first, followed by
texture or appearance synthesis. This paradigm, rooted in the conventional separation of geometry
and appearance, has been adopted by both latent-based pipelines (Xiang et al., 2024) and mesh-based
frameworks (Li et al., 2025a; Hunyuan3D et al., 2025), remaining the prevailing design but entailing
inherent drawbacks. First, the separate generation introduces an inevitable gap between geometry
and appearance, potentially leading to misalignment with the target 3D asset. Second, the two-stage
process introduces additional computation budget, e.g., current mesh-based methods (Hunyuan3D
et al., 2025) first generate the geometry, and then synthesize the corresponding texture based on both
the condition image and geometry generated in the first stage. Notably, the research trajectory in
both vision and graphics (Mildenhall et al., 2020; Kerbl et al., 2023) has long favored unification
over separation – just as object detection evolved from multi-stage Faster R-CNN (Girshick, 2015)
to single-stage YOLO (Redmon et al., 2016). We aim to create a similar unification of geometry and
appearance generation, which is expected to offer more convenience and possibilities for exploring
3D generation under a more extensible and unified framework.
200
300
TripoSR
Step1X-3D
70
80
FDDINOv2
TRELLIS
(Mesh)
Hunyuan3D-2.1
UniLat3D
(Ours-Mesh)
0
20
90
155
Generation Time (s)
45.0
47.5
50.0
52.5
TRELLIS
(GS)
UniLat3D
(Ours-GS)
0.4
0.8
1.5
3
5
Model Size (B)
Figure 2: Evaluation on Toys4K (Stojanov et al., 2021).
Colors stand for model sizes. Lower generation time and
smaller FDDINOv2 indicate better performance, i.e. the left
bottom corner.
To this end, we introduce a unified
3D representation that inherently en-
codes geometry and appearance in a
single latent space, enabling direct
single-stage generation. Our key in-
sight is that such a representation is
naturally aligned—free from geome-
try–texture mismatches—and highly
efficient, as it avoids redundant in-
termediate steps. Inspired by TREL-
LIS (Xiang et al., 2024), we first
transform the 3D asset into sparse
structured features.
A unified vari-
ational autoencoder, UniVAE, is de-
signed to compress high-resolution
sparse features into a compact latent
space, termed UniLat. The UniLat
can then be efficiently upsampled and
sparsified back onto high-resolution
latents that serve as a universal basis
for decoding into various renderable
3D representations, such as 3D Gaus-
sians (Kerbl et al., 2023) and meshes.
Thanks to the simplicity and expres-
sive design of UniLat, we are able to, for the first time, achieve single-stage 3D generation through
one flow-matching model that maps cubic Gaussian noise directly into the geometry-appearance uni-
fied latents. Beyond efficiency, UniLat also offers strong extensibility, which can serve as a versatile
3D prior that can be seamlessly integrated into large multimodal models, facilitating cross-modal
understanding and generation. Our method, UniLat3D, trained only on publicly available datasets,
achieves superior appearance fidelity while maintaining strong geometric accuracy, demonstrating
the effectiveness of unifying geometry and appearance within a single-stage paradigm.
Our contributions are summarized as follows.
• We propose a novel framework, UniLat3D, which bridges the gap between geometry and
appearance by a single diffusion model in high-quality 3D generation.
2

<!-- page 3 -->
Preprint.
• A novel UniLat representation is introduced by encoding geometry and appearance into a
unified latent space, ensuring high-efficiency feature fusion.
• As in Fig. 2, extensive experiments demonstrate UniLat3D’s state-of-the-art performance.
We expect our framework to pave a novel way for exploring 3D generation in a more unified
and scalable paradigm.
2
RELATED WORKS
2.1
3D GENERATION BY LIFTING 2D DIFFUSION MODELS
Lifting 2D diffusion models to 3D has been an effective but challenging approach. DreamFu-
sion (Poole et al., 2022) proposes Score Distillation Sampling (SDS) to distill knowledge from
the 2D diffusion model into a radiance field. Tang et al. (2023); Yi et al. (2023; 2024); Yin et al.
(2023); Ren et al. (2023); Liu et al. (2024); Wang et al. (2023) follow this methodology to generate
high-quality 3D Gaussians (Kerbl et al., 2023) in minutes. Meanwhile, Jain et al. (2022); Liu et al.
(2023b); Shi et al. (2023); Huang et al. (2024); Long et al. (2023); Liu et al. (2023a); Yang et al.
(2024a) fine-tune the image diffusion model to generate multi-view consistent images for synthesiz-
ing 3D assets. Video diffusion models (Yang et al., 2024d; Yu et al., 2024; Xing et al., 2024; Ren
et al., 2025; Zhao et al., 2024; Gao et al., 2024; Wu et al., 2025a; Liang et al., 2024) are also explored
to synthesize high-quality 3D/4D representations (Wu et al., 2024a; Yang et al., 2023; Zhang et al.,
2024d; 2025b). However, most of these methods need iterative optimization from different views
in each generation process, which takes a non-negligible cost, while hallucination may appear, e.g.,
Janus phenomenon, due to the lack of 3D priors.
2.2
3D GENERATION BY PRETRAINING 3D FOUNDATION MODELS
With the emergence of large-scale 3D datasets, e.g., Objaverse (Deitke et al., 2023), 3D foundation
models have been constructed and pretrained to have strong reconstruction and generation abilities.
3D Foundation Reconstruction Models.
Some feed-forward 3D reconstruction methods (Wang
et al., 2024; 2025a; Zhang et al., 2024a; Smart et al., 2024; Li et al., 2025b; Wang et al., 2025b; Yang
et al., 2025a), using vision Transformer (Dosovitskiy et al., 2020) (VIT) to encode and match input
images’ features and recover their relative 3D poses, depths, semantics (Sun et al., 2025; Xu et al.,
2025), and other 3D information (Jiang et al., 2025; Smart et al., 2024). Those methods achieve
nearly real-time reconstruction given an image sequence, while maintaining accurate pose/depth
estimation, and high-quality novel view synthesis.
3D Foundation Generation Models.
A series of 3D foundation models aims to generate high-
quality 3D representations with few or a single image(s) as input in seconds. In the early stage, 3D
Generation mainly focuses on structure&shape generation (Ren et al., 2024b; Vahdat et al., 2022) or
other latent representation (Yang et al., 2024b). Point-E (Nichol et al., 2022) trains a 3D diffusion
model, which is used for generating point clouds from text/image prompts. VecSet (Zhang et al.,
2023) proposes to encode 3D assets into vector representations, which are further applied in the
geometry diffusion models (Chen et al., 2025d; Hunyuan3D et al., 2025; Lai et al., 2025; Zhang
et al., 2024c; Li et al., 2024; Xiong et al., 2025). Then, texture diffusion models (Hunyuan3D et al.,
2025; Li et al., 2025a) are followed to color the high-quality mesh. TRELLIS (Xiang et al., 2024)
and some recent works (Ye et al., 2025; Wu et al., 2025b; Li et al., 2025c; Chen et al., 2025d) encode
multiview images into sparse 3D voxel representations and then decode them into high-quality 3D
assets. Several methods are proposed to generate dynamic objects (Chen et al., 2025a; Zhang et al.,
2025a; Wu et al., 2025c) or extend 3D generation to the part level (Chen et al., 2025b; Dong et al.,
2025; Chen et al., 2025c; Yang et al., 2025b).
We observe that most 3D diffusion models split the generation process into two phases – geometry
and appearance. Our research aims to bridge the gap between geometry and appearance in 3D
generation by introducing a unified latent space while maintaining the strong performance of 3D
diffusion models.
3

<!-- page 4 -->
Preprint.
3
PRELIMINARY
Recently, TRELLIS (Xiang et al., 2024), a powerful 3D generation framework, has enabled gen-
erating high-quality 3D assets in seconds. This is achieved by proposing sparse structured latents
(SLATs) zslat to represent the 3D asset, which can be decoded into different 3D representations.
Sparse Structured Latent Representation.
SLAT is defined as a series of latents located at acti-
vated surface voxels of the 3D asset, which can be formulated as zslat = {zi, pi}L
i=1, where zi ∈Rc
is a c-dimensional latent at the voxel position pi ∈R3, i = {1, 2, ...L}, N denotes the grid resolution
and L << N 3. The coordinates {pi}, representing coarse geometry, are computed by voxelizing
the 3D asset. The latents {zi}, representing appearance and detailed geometry1, are obtained by ag-
gregating and encoding visual features f = {fi, pi}L
i=1, extracted by a vision encoder (Oquab et al.,
2023) from multiple views of the asset. To learn geometry and appearance respectively, TREL-
LIS constructs two separate VAE models, i.e., geometry VAE {Egeo, Dgeo} and appearance VAE
{Eapp, Dapp}.
Specifically, the encoder of the geometry VAE transforms activated voxels p = {pi} to geometry
latents zgeo ∈R
N
s × N
s × N
s ×c with a downsampling factor s:
zgeo = Egeo(p); p = Dgeo(zgeo).
(1)
The sparse appearance VAEs encodes the sparse 3D features f into SLATs zslat, and decodes SLATs
into 3D representations O as:
zslat = Eapp(f); O = Dapp(zslat).
(2)
Note that Eapp only converts f in the feature dimension. The coordinate information is modeled by
Egeo individually.
Sparse Structured Latent Generation.
To generate SLAT zslat, TRELLIS proposes a two-stage
generation pipeline. Given the condition image I, TRELLIS builds a geometry generation flow
Transformer Fgeo to synthesize geometry latents zgeo from the noise ϵ. Then, the activated voxels p
can be decoded by Dgeo:
Fgeo : (ϵ, t, I) →zgeo; p = Dgeo(zgeo),
(3)
where t is the denoising timestep. After that, the appearance noise can be added to the activated
voxels p to get the structured noise ϵapp = {ϵi, pi}. The sparse appearance flow Transformer is
optimized to predict zslat, and the final 3D representation O can be computed by the appearance
decoder Dapp:
Fapp : (ϵapp, t, I) →zslat; O = Dapp(zslat).
(4)
4
METHOD
4.1
OVERALL FRAMEWORK
Geometry-Appearance Unified Latent Representation.
Different from TRELLIS (Xiang et al.,
2024), which obtains sparse structured latents zslat = {zi, pi}L
i=1 in two separate stages, we propose
a dense compressed Latent representation with geometry and appearance Unified (UniLat) zuni ∈
RM×M×M×d which can be obtained in one single stage, where d is the number of unified latent’s
channels, M = N
V , and V denotes the compression ratio. In the reconstruction stage, we construct a
UniLat variational autoencoder (Uni-VAE) {Euni, Duni,{gs,mesh}} to encode the 3D assets efficiently.
The rich geometry and appearance of an assets O can be encoded into the UniLat zuni, which can
be further decoded into 3D representations via decoder Duni as:
zuni ←Euni(O); O = Duni(zuni).
(5)
The unified decoder Duni is composed of a upsampling block Dup and 3D representation decoders
Dgs,mesh. For more details, please refer to Sec. 4.2.2.
1Some detailed geometry properties will be decoded from latents {zi}, e.g. 3D Gaussian positions and mesh
vertices. This will be denoted as ‘appearance’ for short in the following content.
4

<!-- page 5 -->
Preprint.
UniLat3D
Representation
Uni-VAE Encoder 
Sparsification
3D Assets
Noise
UniLat Generation Model
Visual Encoder
K/V
Self-MHA
Cross-MHA
FFN
Linear
Linear
Condition
Image
①
×N
Patchify
Unpatchify
Rendering
Timestep t
②
①
②
③
⊕
PE
Uni-VAE Decoder
3DGS
Projected 
Voxel Feature
Densified
Voxel Feature
Visual Encoder
Projection
Sparse Feature
Predicted 
Occupancy
Upsampled Latents
Sparse Feature 
Densification
Densified Feature 
Compression
UniLat
Mesh
③
④
Sparse Visual Feature 
Extraction & Encoding
④
Encoded 
Voxel Feature
Figure 3: Illustration of the UniLat3D framework. In the reconstruction stage, the encoder of Uni-
VAE Euni converts the 3D asset O to the unified latent – UniLat zuni, which can be directly denoised
from noise ϵ by a single flow model Funi in the generation stage. The obtained UniLat can be
transformed into target 3D representations by the decoder Duni.
Geometry–Appearance Unified Latent Generation.
With geometry and appearance already
fused in our UniLat representation zuni, the generation process becomes naturally streamlined. A
unified generative model Funi is employed to directly denoise compact noises ϵ into UniLat zuni,
which can then be decoded by Duni into the desired 3D representation:
Funi : (ϵ, t, I) →zuni; O = Duni(zuni).
(6)
4.2
UNILAT VARIATIONAL AUTOENCODER
4.2.1
ENCODER
We design an encoder Euni to convert various 3D assets into our UniLats. Euni consists of several
key stages: sparse visual feature extraction, sparse appearance encoding, sparse visual feature den-
sification, and densified feature compression. These stages are supported by two core modules: the
sparse appearance encoding module Msparse and the dense feature compression module Mdense.
The encoding process begins with converting a 3D asset O to sparse visual features f = {fi, pi}, fol-
lowing the multi-view visual feature projection proposed in TRELLIS (Xiang et al., 2024). Then we
employ sparse appearance feature module Msparse to get zsparse by zsparse = Msparse(f). These
two stages are named sparse visual feature extraction and sparse appearance encoding, respectively.
Later, we introduce the sparse feature densification process to fill the empty space in the sparse
latents and get zdense. As computation on zdense is expensive, we perform the densified feature
compression phase, which encodes the processed features into lower-resolution compact latents, i.e.
UniLat zuni. Finally, the UniLat decoder Duni upsamples the compressed unified latents zuni back
onto high-resolution 3D representations, supporting both 3D Gaussian and mesh outputs.
Sparse Feature Densification.
For the sparse feature zsparse with appearance encoded, the geom-
etry is given by indicating which location is empty. To merge both geometry and appearance infor-
mation into unified latents zuni, the structured appearance latents zsparse = {(zsparse,i, pi)}L
i=1 are
converted to dense features zdense. All the empty space is assigned with zero features {0, pj}N 3−L
j̸=i
.
Then, the sparse structured latents can be transformed to dense unified latents:
zdense : {zdense[pi] = zsparse,i; zdense[pj] = 0}.
(7)
Here, zdense is a set of compact dense latents that includes the whole space information.
5

<!-- page 6 -->
Preprint.
Figure 4: Mesh decoder architecture.
Densified Feature Compression.
We use Mdense to encode both the geometry and appearance
features. Similar to 2D/2.5D diffusion models (Chen et al., 2024; Blattmann et al., 2023), Mdense
downsamples zdense ∈N 3 to UniLats zuni ∈M 3 with a downsampling factor s:
zuni = Mdense(zdense).
(8)
The geometry and appearance features are further fused by the downsampling encoding process,
ensuring rich information in the UniLat zuni at the low resolution.
4.2.2
DECODER
Uni-Decoder Duni includes two modules: upsampling block Dup and 3D representation decoders
Dgs,mesh. The high-resolution dense coordinate and features z′
dense ∈RN 3×(C+1) are computed by
Dup, then the pruning process is performed on the dense features z′
dense to obtain sparse features
z′
sparse. Finally, representation decoders Dgs,mesh output the final 3D representations.
Latent Upsampling and Sparsification.
Given a compact but low-resolution UniLat zuni, the
core challenge is to reconstruct high-quality 3D assets in a detailed manner.
To address this,
we introduce an upsampling block that lifts zuni to higher-resolution latents. Leveraging our ge-
ometry–appearance unified representation, we can simultaneously predict voxel occupancy, which
guides a pruning step to remove redundant regions among the upsampled latents. This yields a
sparse set of high-resolution latents that retain both efficiency and fidelity.
Given UniLat zuni ∈RM 3×d, our proposed upsampling blocks Dup compute the appearance and
geometry features at resolution N as :
z′
dense = Dup(zuni).
(9)
Note that both z′
dense = {P ∈RN 3×1, z′ ∈N 3 × c} are high-resolution dense features. Note that
directly performing computation on zdense is expensive, so we propose to prune the low-importance
area to enhance efficiency. The sparse features zsparse are filtered with a signed function:
z′
sparse : {z′
i, pi | P[pi] > 0},
(10)
3D Representation Decoders.
Two 3D representation decoders are designed to transform the
pruned latents into renderable 3D outputs, i.e., 3D Gaussians and meshes. Both decoders share a
backbone of sparse Transformer blocks, similar to TRELLIS, but differ in their task-specific output
heads. For 3D Gaussians, the decoder Dgs maps the latent zuni to attributes of 3D Gaussian primi-
tives Ogs using sparse Transformer blocks and 3D linear projection layers. An additional occupancy
head is employed to predicts voxel occupancy, enabling direct supervision of the reconstructed ge-
ometry.
For meshes, similar to Gaussian decoder, we first upsample zuni and perform sparsification; the
resulting z′
sparse is processed by a stack of sparse Transformer blocks. As illustrated in Fig. 4, hi-
erarchical upsampling is then applied to progressively increase the feature resolution: each stage
performs octree-style subdivision, where each voxel is divided into eight sub-voxels to double the
spatial resolution along each axis, followed by residual sparse 3D convolutions that refine local fea-
tures and preserve gradient flow during training. In practice, three such blocks increase the resolution
from 643 to 5123. After upsampling, a sparse linear output layer predicts SDF values, voxel-corner
6

<!-- page 7 -->
Preprint.
deformations, and interpolation weights (i.e., the SparseFlex (He et al., 2025) parameters), from
which we extract mesh vertices and faces efficiently. To enable multi-scale geometry supervision
and reduce computational overhead, occupancy prediction heads are attached at each resolution and
supervised with corresponding voxel-level occupancy.
To scale Dmesh to higher resolutions, we adopt a pruning strategy that removes voxels entirely
outside or inside object boundaries, thereby reducing computational overhead. We further introduce
a detail augmentation strategy, where depth and normal maps are rendered from zoomed-in camera
views with a differentiable rasterizer, enabling the decoder to learn fine-grained surface details from
localized partial observations. With these techniques, UniLat3D produces meshes at a resolution of
5123, doubling the resolution achieved by TRELLIS.
4.3
UNILAT GENERATION MODEL
With Uni-VAE, we construct a generation model Funi based on rectified flow matching to de-
noise compact noise ϵ into condition-followed UniLats zuni. A single flow Transformer model
Funi with full attention layers is built to predict the velocity at timestamp t under the noise level
as v = Funi(xuni, t, I) and xuni denotes the denoised noise ϵ and timestamp t. The whole flow
Transformer optimization process follows the diffusion guidance given condition I with its condi-
tion encoder. The latent features with both geometry and appearance information are denoised. The
obtained UniLat zuni can be directly fed into the representation decoder Duni to predict the final 3D
representation O.
4.4
OPTIMIZATION
Uni-VAE.
We use both geometry and appearance supervision to train the Duni. Following TREL-
LIS (Xiang et al., 2024), we joint optimize Euni and Dgs with the following loss:
L = λl1Ll1 + λlpipsLlpips + λssimLssim + λklLkl + λdiceLdice + λregLreg.
(11)
Ll1 denotes the L1 color loss, and Llpips and Lssim stand for inception-based losses. Lkl is employed
for optimizing Euni. Ldice and Lreg are used to supervise geometry and decoded representations.
For the mesh decoder, we adopt a hierarchical supervision aligned with the multi-scale upsampling
described in Sec. 4.2.2. Occupancy prediction heads are attached at each resolution, and are trained
with corresponding voxel-level occupancy targets. The overall mesh objective is
Lmesh = λgeoLgeo + λcolorLcolor + λregLreg + λoccLocc,
(12)
where Lgeo, Lcolor, and Lreg follow TRELLIS. To alleviate the computational cost at high res-
olutions, training proceeds in two stages. Stage-1 optimizes Dmesh up to 2563 resolution using
Eq. equation 12. Stage-2 introduces an independent 256→512 upsampling block with its own prun-
ing head; this new block is optimized while the Stage-1 pathway remains frozen. After decoding,
lightweight post-processing removes invisible or degenerate faces and fills small holes.
Rectified Flow Models.
Once the Uni-VAE has been trained, all the UniLat zuni are predicted by
the Uni-Encoder Euni. For optimizing the rectified flow Transformer, we mainly follow the CFM
Loss. Given encoded latents xuni and noise ϵ, we minimize the objective function LCFM (Lipman
et al., 2022) as:
LCFM(θ) = Et,x0,ϵ∥v(xuni, t) −(ϵ −zuni)∥2
2.
(13)
5
EXPERIMENTS
5.1
IMPLEMENTATION DETAILS
Our framework is implemented in PyTorch (Paszke et al., 2019) and built upon the open-source
project TRELLIS (Xiang et al., 2024). FlashAttention-3(Shah et al., 2024) is employed to accelerate
Transformer training, yielding a 1.5× speedup. Both VAE and flow models are trained on 64 GPUs
within two weeks.
7

<!-- page 8 -->
Preprint.
Uni-VAE.
To accelerate and stabilize Uni-VAE training, we initialize Esparse and Dsparse with
the pretrained weights from TRELLIS. During the first 240k iterations, only Edense and Dup are
optimized, after which the entire Uni-VAE is trained end-to-end for an additional 90k iterations
following TRELLIS. For the mesh decoder, we freeze Duni and train our high-resolution mesh
decoder from scratch. Unless otherwise specified, Adam (Kingma & Ba, 2014) is used with a
learning rate of 1 × 10−4.
UniLat Flow Transformer.
For training the rectified flow models, we adopt DINOv3 (Sim´eoni
et al., 2025) as the image encoder and apply classifier-free guidance (Ho & Salimans, 2022) with a
drop rate of 0.1. The model is first trained for 500k iterations with a batch size of 256 and a learning
rate of 1 × 10−4, and then fine-tuned for 160k iterations with a batch size of 1024 and a learning
rate of 1 × 10−5.
5.2
EXPERIMENTAL SETUP
Training Datasets.
UniLat3D is trained exclusively on publicly available datasets. Following the
data preparation pipeline of TRELLIS (Xiang et al., 2024), we curate and process approximately
450k high-quality 3D assets from Objaverse (XL) (Deitke et al., 2023), ABO (Collins et al., 2022),
3D-FUTURE (Fu et al., 2021), and HSSD (Khanna* et al., 2023). To enable occupancy supervision
at multiple scales, we perform voxelization at each resolution. Additional details on data prepro-
cessing can be found in (Xiang et al., 2024).
Evaluation Datasets.
The evaluation is performed on two datasets. One is the whole Toys4K (Sto-
janov et al., 2021) dataset, including 3218 high-quality 3D assets, which is also used in the previous
method (Xiang et al., 2024). However, we observe that many samples of Toys4K tend to have
simple geometry or appearance details. We construct a more complex dataset for comprehensive
evaluation, including 500 high-quality assets collected from the Sketchfab platform and 500 assets
sampled from Toys4K. Condition images for qualitative comparisons and user studies are collected
from Chen et al. (2025d); Wu et al. (2025b) or generated via VLMs.
Evaluation Setups.
For VAE reconstruction evaluation, we use the PSNR, SSIM, and LPIPS met-
rics. For appearance generation quality, we compute the CLIP (Radford et al., 2021) score – sim-
ilarity between rendered images and condition images, and FD (Fr´echet distance) (Heusel et al.,
2017) measured by DINOv2 (Oquab et al., 2023) on 4 views of each generated asset and ground
truth images. We evaluate and compare our method with recent SOTA 3D generation models, i.e.
Hunyuan3D-2.1 (Hunyuan3D et al., 2025), TRELLIS (Xiang et al., 2024), Step1X-3D (Li et al.,
2025a), TripoSR (Tochilkin et al., 2024) for image-conditioned generation, and Stable3DGen (Ye
et al., 2025) and Direct3D-S2 (Wu et al., 2025b) for geometry generation quality comparison. We
report Uni3D (Zhou et al., 2023) and ULIP (Xue et al., 2023) metrics for mesh geometry quality. For
mesh rendering, we mainly use Blender (Blender Foundation, 2025) as a mesh renderer to render
high-quality images. We set FOV=40, render resolution=512, and set normalization to each loaded
object.
5.3
RESULTS
Step1X-3D
9.8%
Ours
38.6%
TRELLIS
22.7%
Hunyuan3D-2.1
28.9%
Figure 9: User study on dif-
ferent models.
We provide qualitative comparisons in Fig. 5 and Fig. 6, where our
method achieves competitive generation quality and demonstrates
stronger alignment with the conditional image, benefiting from
the unified representation. Note that Hunyuan3D-2.1 (Hunyuan3D
et al., 2025), Step1X-3D (Li et al., 2025a), and TripoSR (Tochilkin
et al., 2024) only provide mesh-based results. Importantly, Ours,
TripoSR, TRELLIS, and Direct3D-S2 are trained exclusively on
publicly available datasets, while other methods leverage additional
private data. We also provide qualitative comparisons among some
commercial models in Fig. 7. Results show that even compared
with commercial models, UniLat3D still delivers competitive per-
formance with notably better consistency between the generated 3D
content and the input image. Fig. 8 displays diverse 3D mesh assets
8

<!-- page 9 -->
Preprint.
Condition
TRELLIS (GS)
Step1X-3D
Hunyuan3D-2.1
Ours (GS)
TripoSR
Ours (Mesh)
Figure 5: Qualitative comparisons with other methods. Thanks to our unified representation, Uni-
Lat3D achieves superior performance and better correspondence with input images.
generated by UniLat3D, demonstrating its superior performance in producing high-quality geometry
and realistic appearance.
Quantitative evaluations on Toys4k (Stojanov et al., 2021) are reported in Table 1. Additional results
on our self-collected complex set are provided in the Table 2. Compared with other two-stage
methods, UniLat3D achieves leading appearance performance, reaching 47.68 in FDDINOv2. The
CLIP score of 90.87 further demonstrates the effectiveness of UniLat3D in aligning images and
3D assets. In terms of geometry synthesis, our mesh version also achieves competitive results in
9

<!-- page 10 -->
Preprint.
Condition
TRELLIS (GS)
Step1X-3D
Hunyuan3D-2.1
Ours (GS)
TripoSR
Ours (Mesh)
Figure 6: Additional qualitative comparisons with other methods.
Table 1: Comparisons on the Toys4K dataset. “#Params” denotes the number of model parameters.
The “ULIP” and “Uni3D” metrics are multiplied by 102.
Model
Rep.
#Param.
Time
CLIP↑
FDDINOv2↓
ULIP↑
Uni3D↑
TripoSR
Mesh
0.4B
13s
88.76
279.06
35.30
30.98
TRELLIS
Mesh
1.31B
21s
87.81
79.52
42.51
37.67
TRELLIS
3DGS
1.31B
5s
90.70
52.54
–
–
Stable3DGen†⋆
Mesh
2.63B
4s
–
–
40.33
35.98
Step1X-3D†
Mesh
4.8B
152s
85.85
146.08
41.37
36.51
Direct3D-S2⋆
Mesh
2.1B
185s
–
–
41.51
36.64
Hunyuan3D-2.1†
Mesh
5.3B
90s
88.44
74.16
42.67
37.74
Ours
Mesh
1.58B
36s
87.93
71.81
42.69
37.62
Ours
3DGS
1.55B
8s
90.87
47.68
–
–
† Using proprietary or non-public training data.
⋆Only generating geometry without appearance.
ULIP (Xue et al., 2023), with a score of 42.69. Beyond accuracy, UniLat3D also demonstrates
notable efficiency: 3D Gaussian generation is completed within 8 seconds on a single A100 GPU
and can be further reduced to 3 seconds with FlashAttention-3 (Shah et al., 2024). Mesh generation
requires 36 seconds, primarily due to the higher resolution with more vertices and longer post-
processing compared to TRELLIS, but remains competitive considering the improved output quality.
Besides, we conducted a user study with 19 participants over 3D assets generated from 23 image
prompts. Four models with both geometry and appearance generation are involved. For each prompt,
participants judged generated assets by both image alignment and object quality, and chose the
overall best case. As shown in Fig. 9, UniLat3D received over 35% of the votes, outperforming
Huanyuan3D-2.1 and other models.
5.4
ABLATION STUDY
Table 3: VAE reconstruction results with latents of different
resolutions.
Model
Res.
PSNR↑
SSIM↑
LPIPS↓
TRELLIS (Mesh)
643
31.91
97.44
0.0328
Ours (Mesh)
163
32.35
98.03
0.0305
TRELLIS (GS)
643
34.74
98.52
0.0146
Ours (GS)
83
33.51
98.13
0.0200
Ours (GS)
163
34.80
98.49
0.0158
Ours (GS)
323
34.92
98.53
0.0145
Resolution of Latents.
We explore
the latent space of reconstruction
quality in Uni-VAE. We train Uni-
VAE at different latent resolutions,
including 83, 163, and 323. As shown
in Table 3, higher UniLat resolutions
lead to better reconstruction results.
Note that our Uni-VAE achieves sim-
ilar or even better reconstruction per-
10

<!-- page 11 -->
Preprint.
Condition Image
Commercial 
Model A
Commercial 
Model B
Commercial
Model C
Commercial 
Model D
Ours
Figure 7: Qualitative comparisons with commercial models. Our UniLat3D shows competitive
performance even with only publicly available training data.
formance than TRELLIS with smaller resolutions. In our experiments, when training the flow Trans-
former at a higher resolution of 32, the computational cost increases evidently. We would explore
more efficient approaches on flow Transformers for higher resolutions in future works, e.g., block-
wise computation and lightweight attention.
Visual Encoder of Condition Images
Recently, DINOv3 (Sim´eoni et al., 2025) emerges as a
strong visual encoder model that could extract high-quality details from the image. We compare
the performance between DINOv2 and DINOv3 for encoding condition images. Flow models with
different visual encoders are trained for 500 iterations and tested on Toys4K. In our experiments,
11

<!-- page 12 -->
Preprint.
Figure 8: 3D mesh assets generated by our UniLat3D.
Table 2: Comparisons on the self-collected complex test dataset. “Rep.” denotes the output repre-
sentation type, and “#Params” denotes the number of model parameters. The “ULIP” and “Uni3D”
metrics are multiplied by 102.
Model
Rep.
#Param.
Time
CLIP↑
FDDINOv2↓
ULIP↑
Uni3D↑
TripoSR
Mesh
0.4B
13s
88.00
369.86
33.61
30.44
TRELLIS
Mesh
1.31B
21s
86.40
164.57
41.52
37.30
TRELLIS
3DGS
1.31B
5s
89.67
108.27
-
-
Stable3DGen†⋆
Mesh
2.63B
4s
-
-
39.79
35.93
Step1X-3D†
Mesh
4.8B
152s
84.74
210.49
40.53
36.46
Direct3D-S2⋆
Mesh
2.1B
185s
-
-
40.77
36.47
Hunyuan3D-2.1†
Mesh
5.3B
90s
87.41
150.39
41.70
37.48
Ours
Mesh
1.58B
36s
86.44
149.62
41.71
37.24
Ours
3DGS
1.55B
8s
89.83
97.22
-
-
† Using proprietary or non-public training data.
⋆Only generating geometry without appearance.
Table 4: Ablation study on the visual encoder for condition images.
Model Cond. Encoder CLIP↑FDdinov2↓
Ours
DINOV2
90.83
52.58
Ours
DINOV3
90.60
49.90
the flow Transformer with the DINOv3 encoder shows better quality on complex object generation,
which leads to a better FDdinov2 result as shown in Table 4.
6
DISCUSSION & CONCLUSION
We propose a novel 3D generation framework – UniLat3D to achieve high-quality 3D asset gen-
eration in seconds with a single-stage flow model. Apart from that the proposed method unifies
geometry and appearance in a single, concise framework, it achieves quite competitive performance
compared with popular two-stage methods. We expect our exploration to provide a more convenient
and extensible choice to the 3D generation field, e.g., further unifying object and scene generation
with the compact unified representation, extending UniLat to 4D representations, and integrating
UniLat into large multimodal models etc.
However, the UniLat3D model implemented in this paper is still a preliminary exploration. The
training data we used just follows TRELLIS, totally from public datasets. Injecting more high-
quality data for training will undoubtedly improve the performance and may further scale up the
model. Exploring more efficient designs on the flow model would adapt to higher resolutions of
latents, leading to more detailed generation results.
12

<!-- page 13 -->
Preprint.
ACKNOWLEDGEMENT
We would like to thank Junjie Wang and Zhikuan Bao for their valuable contributions to this project.
We are also grateful to Jinfeng Yao and Lianghui Zhu for their valuable input during the initial stages
of the project.
REFERENCES
Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik
Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling
latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
Blender Foundation. Blender: Open source 3d creation suite. https://www.blender.org,
2025. Accessed: 2025-09-24.
Jianqi Chen, Biao Zhang, Xiangjun Tang, and Peter Wonka. V2m4: 4d mesh animation reconstruc-
tion from a single monocular video. arXiv preprint arXiv:2503.09631, 2025a.
Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, and
Song Han. Deep compression autoencoder for efficient high-resolution diffusion models. arXiv
preprint arXiv:2410.10733, 2024.
Minghao Chen, Roman Shapovalov, Iro Laina, Tom Monnier, Jianyuan Wang, David Novotny, and
Andrea Vedaldi. Partgen: Part-level 3d generation and reconstruction with multi-view diffusion
models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5881–
5892, 2025b.
Minghao Chen, Jianyuan Wang, Roman Shapovalov, Tom Monnier, Hyunyoung Jung, Dilin Wang,
Rakesh Ranjan, Iro Laina, and Andrea Vedaldi. Autopartgen: Autogressive 3d part generation
and discovery. arXiv preprint arXiv:2507.13346, 2025c.
Yiwen Chen, Zhihao Li, Yikai Wang, Hu Zhang, Qin Li, Chi Zhang, and Guosheng Lin. Ultra3d:
Efficient and high-fidelity 3d generation with part attention. arXiv preprint arXiv:2507.17745,
2025d.
Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu,
Xi Zhang, Tomas F Yago Vicente, Thomas Dideriksen, Himanshu Arora, Matthieu Guillaumin,
and Jitendra Malik. Abo: Dataset and benchmarks for real-world 3d object understanding. CVPR,
2022.
Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig
Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of anno-
tated 3d objects. In CVPR, pp. 13142–13153, 2023.
Shaocong Dong, Lihe Ding, Xiao Chen, Yaokun Li, Yuxin Wang, Yucheng Wang, Qi Wang, Jae-
hyeok Kim, Chenjian Gao, Zhanpeng Huang, et al. From one to more: Contextual part latents for
3d generation. arXiv preprint arXiv:2507.08772, 2025.
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An
image is worth 16x16 words: Transformers for image recognition at scale.
arXiv preprint
arXiv:2010.11929, 2020.
Huan Fu, Rongfei Jia, Lin Gao, Mingming Gong, Binqiang Zhao, Steve Maybank, and Dacheng
Tao. 3d-future: 3d furniture shape with texture. International Journal of Computer Vision, 129
(12):3313–3337, 2021.
Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul
Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view
diffusion models. arXiv preprint arXiv:2405.10314, 2024.
Ross Girshick. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision,
pp. 1440–1448, 2015.
13

<!-- page 14 -->
Preprint.
Xianglong He, Zi-Xin Zou, Chia-Hao Chen, Yuan-Chen Guo, Ding Liang, Chun Yuan, Wanli
Ouyang, Yan-Pei Cao, and Yangguang Li. Sparseflex: High-resolution and arbitrary-topology
3d shape modeling. arXiv preprint arXiv:2503.21732, 2025.
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter.
Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in
neural information processing systems, 30, 2017.
Jonathan Ho and Tim Salimans.
Classifier-free diffusion guidance.
arXiv preprint
arXiv:2207.12598, 2022.
Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli,
Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint
arXiv:2311.04400, 2023.
Zehuan Huang, Yuan-Chen Guo, Haoran Wang, Ran Yi, Lizhuang Ma, Yan-Pei Cao, and
Lu Sheng.
Mv-adapter: Multi-view consistent image generation made easy.
arXiv preprint
arXiv:2412.03632, 2024.
Team Hunyuan3D, Shuhui Yang, Mingxin Yang, Yifei Feng, Xin Huang, Sheng Zhang, Zebin He,
Di Luo, Haolin Liu, Yunfei Zhao, et al. Hunyuan3d 2.1: From images to high-fidelity 3d assets
with production-ready pbr material. arXiv preprint arXiv:2506.15442, 2025.
Ajay Jain, Ben Mildenhall, Jonathan T Barron, Pieter Abbeel, and Ben Poole. Zero-shot text-guided
object generation with dream fields. In CVPR, pp. 867–876, 2022.
Haoyi Jiang, Liu Liu, Tianheng Cheng, Xinjie Wang, Tianwei Lin, Zhizhong Su, Wenyu Liu, and
Xinggang Wang. Gausstr: Foundation model-aligned gaussian transformer for self-supervised 3d
spatial understanding. In Proceedings of the Computer Vision and Pattern Recognition Confer-
ence, pp. 11960–11970, 2025.
Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splat-
ting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), July 2023.
URL https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/.
Mukul Khanna*, Yongsen Mao*, Hanxiao Jiang, Sanjay Haresh, Brennan Shacklett, Dhruv Batra,
Alexander Clegg, Eric Undersander, Angel X. Chang, and Manolis Savva. Habitat Synthetic
Scenes Dataset (HSSD-200): An Analysis of 3D Scene Scale and Realism Tradeoffs for Object-
Goal Navigation. arXiv preprint, 2023.
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint
arXiv:1412.6980, 2014.
Zeqiang Lai, Yunfei Zhao, Haolin Liu, Zibo Zhao, Qingxiang Lin, Huiwen Shi, Xianghui Yang,
Mingxin Yang, Shuhui Yang, Yifei Feng, et al. Hunyuan3d 2.5: Towards high-fidelity 3d assets
generation with ultimate details. arXiv preprint arXiv:2506.16504, 2025.
Weiyu Li, Jiarui Liu, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. Crafts-
man: High-fidelity mesh generation with 3d native generation and interactive geometry refiner.
arXiv preprint arXiv:2405.14979, 2024.
Weiyu Li, Xuanyang Zhang, Zheng Sun, Di Qi, Hao Li, Wei Cheng, Weiwei Cai, Shihao Wu, Jiarui
Liu, Zihao Wang, et al. Step1x-3d: Towards high-fidelity and controllable generation of textured
3d assets. arXiv preprint arXiv:2505.07747, 2025a.
Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo
Kanazawa, Aleksander Holynski, and Noah Snavely. Megasam: Accurate, fast and robust struc-
ture and motion from casual dynamic videos. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pp. 10486–10496, 2025b.
Zhihao Li, Yufei Wang, Heliang Zheng, Yihao Luo, and Bihan Wen. Sparc3d: Sparse representa-
tion and construction for high-resolution 3d shapes modeling. arXiv preprint arXiv:2505.14521,
2025c.
14

<!-- page 15 -->
Preprint.
Hanwen Liang, Yuyang Yin, Dejia Xu, Hanxue Liang, Zhangyang Wang, Konstantinos N Platanio-
tis, Yao Zhao, and Yunchao Wei. Diffusion4d: Fast spatial-temporal consistent 4d generation via
video diffusion models. arXiv preprint arXiv:2405.16645, 2024.
Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching
for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
Fangfu Liu, Diankun Wu, Yi Wei, Yongming Rao, and Yueqi Duan. Sherpa3d: Boosting high-
fidelity text-to-3d generation via coarse 3d prior. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pp. 20763–20774, 2024.
Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Zexiang Xu, Hao Su, et al.
One-2-3-45:
Any single image to 3d mesh in 45 seconds without per-shape optimization.
arXiv preprint
arXiv:2306.16928, 2023a.
Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick.
Zero-1-to-3: Zero-shot one image to 3d object. arXiv preprint arXiv:2303.11328, 2023b.
Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma,
Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d
using cross-domain diffusion. arXiv preprint arXiv:2310.15008, 2023.
Ziqiao Ma, Xuweiyi Chen, Shoubin Yu, Sai Bi, Kai Zhang, Chen Ziwen, Sihan Xu, Jianing Yang,
Zexiang Xu, Kalyan Sunkavalli, et al. 4d-lrm: Large space-time reconstruction model from and
to any view at any time. arXiv preprint arXiv:2506.18890, 2025.
Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and
Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, pp.
405–421, 2020.
Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-e: A system
for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022.
Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov,
Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning
robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-
performance deep learning library. Advances in neural information processing systems, 32, 2019.
Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d
diffusion. arXiv preprint arXiv:2209.14988, 2022.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In ICML, pp. 8748–8763. PMLR, 2021.
Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified,
real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern
recognition, pp. 779–788, 2016.
Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. Dream-
gaussian4d: Generative 4d gaussian splatting. arXiv preprint arXiv:2312.17142, 2023.
Jiawei Ren, Cheng Xie, Ashkan Mirzaei, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler,
Seung Wook Kim, Huan Ling, et al. L4gm: Large 4d gaussian reconstruction model. Advances
in Neural Information Processing Systems, 37:56828–56858, 2024a.
Xuanchi Ren, Jiahui Huang, Xiaohui Zeng, Ken Museth, Sanja Fidler, and Francis Williams.
Xcube: Large-scale 3d generative modeling using sparse voxel hierarchies. In Proceedings of
the IEEE/CVF conference on computer vision and pattern recognition, pp. 4209–4219, 2024b.
15

<!-- page 16 -->
Preprint.
Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas
M¨uller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video
generation with precise camera control.
In Proceedings of the Computer Vision and Pattern
Recognition Conference, pp. 6121–6132, 2025.
Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao.
Flashattention-3: Fast and accurate attention with asynchrony and low-precision. Advances in
Neural Information Processing Systems, 37:68658–68685, 2024.
Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view
diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023.
Oriane Sim´eoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose,
Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, et al. Dinov3. arXiv
preprint arXiv:2508.10104, 2025.
Brandon Smart, Chuanxia Zheng, Iro Laina, and Victor Adrian Prisacariu.
Splatt3r: Zero-shot
gaussian splatting from uncalibrated image pairs. arXiv preprint arXiv:2408.13912, 2024.
Stefan Stojanov, Anh Thai, and James M Rehg. Using shape to categorize: Low-shot learning
with an explicit shape bias. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pp. 1798–1808, 2021.
Xiangyu Sun, Liu Liu, Seungtae Nam, Gyeongjin Kang, Wei Sui, Zhizhong Su, Wenyu Liu,
Xinggang Wang, Eunbyung Park, et al. Uni3r: Unified 3d reconstruction and semantic under-
standing via generalizable gaussian splatting from unposed multi-view images. arXiv preprint
arXiv:2508.03643, 2025.
Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative
gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653, 2023.
Dmitry Tochilkin, David Pankratz, Zexiang Liu, Zixuan Huang, Adam Letts, Yangguang Li, Ding
Liang, Christian Laforte, Varun Jampani, and Yan-Pei Cao. Triposr: Fast 3d object reconstruction
from a single image. arXiv preprint arXiv:2403.02151, 2024.
Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, Karsten Kreis, et al. Lion:
Latent point diffusion models for 3d shape generation. NeurIPS, 35:10021–10039, 2022.
Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David
Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision
and Pattern Recognition Conference, pp. 5294–5306, 2025a.
Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Con-
tinuous 3d perception model with persistent state. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pp. 10510–10522, 2025b.
Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Ge-
ometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pp. 20697–20709, 2024.
Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolific-
dreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. arXiv
preprint arXiv:2305.16213, 2023.
Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian,
and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceed-
ings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20310–20320,
2024a.
Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Alek-
sander Holynski. Cat4d: Create anything in 4d with multi-view video diffusion models. In Pro-
ceedings of the Computer Vision and Pattern Recognition Conference, pp. 26057–26068, 2025a.
16

<!-- page 17 -->
Preprint.
Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao.
Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. arXiv preprint
arXiv:2405.14832, 2024b.
Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Yikang Yang, Yajie Bao, Jiachen Qian, Siyu
Zhu, Philip Torr, Xun Cao, and Yao Yao. Direct3d-s2: Gigascale 3d generation made easy with
spatial sparse attention. arXiv preprint arXiv:2505.17412, 2025b.
Zijie Wu, Chaohui Yu, Fan Wang, and Xiang Bai. Animateanymesh: A feed-forward 4d foundation
model for text-driven universal mesh animation. arXiv preprint arXiv:2506.09982, 2025c.
Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin
Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. arXiv
preprint arXiv:2412.01506, 2024.
Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu,
Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images
with video diffusion priors. In European Conference on Computer Vision, pp. 399–417. Springer,
2024.
Bojun Xiong, Si-Tong Wei, Xin-Yang Zheng, Yan-Pei Cao, Zhouhui Lian, and Peng-Shuai Wang.
Octfusion: Octree-based diffusion models for 3d shape generation. In Computer Graphics Forum,
volume 44, pp. e70198. Wiley Online Library, 2025.
Yueming Xu, Jiahui Zhang, Ze Huang, Yurui Chen, Yanpeng Zhou, Zhenyu Chen, Yu-Jie Yuan,
Pengxiang Xia, Guowei Huang, Xinyue Cai, et al. Uniugg: Unified 3d understanding and gener-
ation via geometric-semantic encoding. arXiv preprint arXiv:2508.11952, 2025.
Le Xue, Mingfei Gao, Chen Xing, Roberto Mart´ın-Mart´ın, Jiajun Wu, Caiming Xiong, Ran Xu,
Juan Carlos Niebles, and Silvio Savarese. Ulip: Learning a unified representation of language,
images, and point clouds for 3d understanding. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pp. 1179–1189, 2023.
Chen Yang, Sikuang Li, Jiemin Fang, Ruofan Liang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and
Qi Tian. Gaussianobject: High-quality 3d object reconstruction from four views with gaussian
splatting. ACM Transactions on Graphics (TOG), 43(6):1–13, 2024a.
Haitao Yang, Yuan Dong, Hanwen Jiang, Dejia Xu, Georgios Pavlakos, and Qixing Huang. Atlas
gaussians diffusion for 3d generation. arXiv preprint arXiv:2408.13055, 2024b.
Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai,
Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one
forward pass. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp.
21924–21935, 2025a.
Xianghui Yang, Huiwen Shi, Bowen Zhang, Fan Yang, Jiacheng Wang, Hongxu Zhao, Xinhai Liu,
Xinzhou Wang, Qingxiang Lin, Jiaao Yu, et al. Hunyuan3d 1.0: A unified framework for text-to-
3d and image-to-3d generation. arXiv preprint arXiv:2411.02293, 2024c.
Yunhan Yang, Yufan Zhou, Yuan-Chen Guo, Zi-Xin Zou, Yukun Huang, Ying-Tian Liu, Hao Xu,
Ding Liang, Yan-Pei Cao, and Xihui Liu. Omnipart: Part-aware 3d generation with semantic
decoupling and structural cohesion. arXiv preprint arXiv:2507.06165, 2025b.
Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Real-time photorealistic dynamic scene repre-
sentation and rendering with 4d gaussian splatting. arXiv preprint arXiv:2310.10642, 2023.
Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang,
Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models
with an expert transformer. arXiv preprint arXiv:2408.06072, 2024d.
Chongjie Ye, Yushuang Wu, Ziteng Lu, Jiahao Chang, Xiaoyang Guo, Jiaqing Zhou, Hao Zhao,
and Xiaoguang Han. Hi3dgen: High-fidelity 3d geometry generation from images via normal
bridging. arXiv preprint arXiv:2503.22236, 3:2, 2025.
17

<!-- page 18 -->
Preprint.
Taoran Yi, Jiemin Fang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and
Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussian splatting with point
cloud priors. arXiv preprint arXiv:2310.08529, 2023.
Taoran Yi, Jiemin Fang, Zanwei Zhou, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang,
Wenyu Liu, Xinggang Wang, and Qi Tian. Gaussiandreamerpro: Text to manipulable 3d gaussians
with highly enhanced quality. arXiv preprint arXiv:2406.18462, 2024.
Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, and Yunchao Wei. 4dgen: Grounded 4d content
generation with spatial-temporal consistency. arXiv preprint arXiv:2312.17225, 2023.
Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-
Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for
high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024.
Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka.
3dshape2vecset: A 3d shape
representation for neural fields and generative diffusion models. ACM Transactions On Graphics
(TOG), 42(4):1–16, 2023.
Bowen Zhang, Sicheng Xu, Chuxin Wang, Jiaolong Yang, Feng Zhao, Dong Chen, and Baining
Guo. Gaussian variation field diffusion for high-fidelity video-to-4d synthesis. arXiv preprint
arXiv:2507.23785, 2025a.
Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, De-
qing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the
presence of motion. arXiv preprint arXiv:2410.03825, 2024a.
Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu.
Gs-lrm: Large reconstruction model for 3d gaussian splatting. European Conference on Computer
Vision, 2024b.
Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan
Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d
assets. ACM Transactions on Graphics (TOG), 43(4):1–20, 2024c.
Shuai Zhang, Guanjun Wu, Zhoufeng Xie, Xinggang Wang, Bin Feng, and Wenyu Liu.
Dy-
namic 2d gaussians: Geometrically accurate radiance fields for dynamic objects. arXiv preprint
arXiv:2409.14072, 2024d.
Shuai Zhang, Huangxuan Zhao, Zhenghong Zhou, Guanjun Wu, Chuansheng Zheng, Xinggang
Wang, and Wenyu Liu. Togs: Gaussian splatting with temporal opacity offset for real-time 4d dsa
rendering. IEEE Journal of Biomedical and Health Informatics, 2025b.
Yuyang Zhao, Chung-Ching Lin, Kevin Lin, Zhiwen Yan, Linjie Li, Zhengyuan Yang, Jianfeng
Wang, Gim Hee Lee, and Lijuan Wang. Genxd: Generating any 3d and 4d scenes. arXiv preprint
arXiv:2411.02319, 2024.
Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng,
Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for
high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025.
Junsheng Zhou, Jinsheng Wang, Baorui Ma, Yu-Shen Liu, Tiejun Huang, and Xinlong Wang. Uni3d:
Exploring unified 3d representation at scale. arXiv preprint arXiv:2310.06773, 2023.
Zi-Xin Zou, Zhipeng Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, Yan-Pei Cao, and Song-Hai
Zhang. Triplane meets gaussian splatting: Fast and generalizable single-view 3d reconstruction
with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition, pp. 10324–10335, 2024.
18

<!-- page 19 -->
Preprint.
A
APPENDIX
A.1
MODEL ARCHITECTURE
In this section, we mainly provide the model architecture about our Uni-VAE {Euni, Duni} and
UniLat generation model F.
A.1.1
UNI-VAE
For the sparse encoder Msparse, we mainly follow TRELLIS’s configurations to build a sparse
Transformer. For the dense encoder Mdense, a set of conv3D layers is used as the main architecture.
The settings of Esparse, Dup are shown in Table 5 and details of Euni are provided in Table 6.
Table 5: Model details of Uni-VAE modules Mdense, Dup. “Channels” denotes model channels
after each up/downsampled convolution layer.
Model
ResBlocks
Channels
Esparse
4
[32, 128, 512]
Dup
4
[512, 128, 32]
Table 6: Model details of Uni-VAE modules Msparse, Dgs,mesh.
Model
Latent Res.
Model Channels
Latent. Channels
Blocks
Attn. Heads
Window Size
Msparse, Dsparse
64
768
8
12
12
8
A.1.2
UNILAT FLOW TRANSFORMER
Structure details about our UniLat flow Transformer Funi are provided in the Table 7. The main
architecture of Funi is similar to TRELLIS’s sparse structure flow Transformer. The input noise ϵ
would be flattened to 1D tensors. Positional encoding is applied to a flattened tensor, and it would
be fed to Transformer blocks with self&cross-attention layer and modulated by condition signal &
timestamps. Finally, the flattened tensor would be unpatchified to 3D results, the shape is the same
as ϵ.
Table 7: Model details of UniLat3D flow Transformer.
Model
Params
Latent Res.
Latent Channels
Model Channels
Cond. Channels
Blocks
Attn. Heads
Funi
1.30B
16
32
1280
1280
36
32
19
