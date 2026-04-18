<!-- page 1 -->
MatSpray: Fusing 2D Material World Knowledge on 3D Geometry
Philipp Langsteiner
philipp.langsteiner@uni-tuebingen.de
Jan-Niklas Dihlmann
jan-niklas.dihlmann@uni-tuebingen.de
Hendrik Lensch
hendrik.lensch@uni-tuebingen.de
Figure 1. MatSpray Overview we utilize 2D material world knowlegde from 2D diffusion models to reconstruct 3D relightable objects.
Given multi-view images of a target object, we first generate per-view PBR material predictions (base color, roughness, metallic) using any
2D diffusion-based material model. These 2D estimates are then integrated into a 3D Gaussian Splatting reconstruction via Gaussian ray
tracing. Finally, a neural refinement stage applies a softmax-based restriction to enforce multi-view consistency and enhance the physical
accuracy of the materials. The resulting 3D assets feature high-quality, fully relightable PBR materials under novel illumination. Project
page: https://matspray.jdihlmann.com/
Abstract
Manual modeling of material parameters and 3D geometry
is a time consuming yet essential task in the gaming and
film industries. While recent advances in 3D reconstruction
have enabled accurate approximations of scene geometry
and appearance, these methods often fall short in relighting
scenarios due to the lack of precise, spatially varying ma-
terial parameters. At the same time, diffusion models oper-
ating on 2D images have shown strong performance in pre-
dicting physically based rendering (PBR) properties such
as albedo, roughness, and metallicity. However, transfer-
ring these 2D material maps onto reconstructed 3D geom-
etry remains a significant challenge. We propose a frame-
work for fusing 2D material data into 3D geometry using a
combination of novel learning-based and projection-based
approaches. We begin by reconstructing scene geometry
via Gaussian Splatting. From the input images, a diffusion
model generates 2D maps for albedo, roughness, and metal-
lic parameters. Any existing diffusion model that can con-
vert images or videos to PBR materials can be applied. The
predictions are further integrated into the 3D representa-
tion either by optimizing an image-based loss or by directly
projecting the material parameters onto the Gaussians us-
ing Gaussian ray tracing. To enhance fine-scale accuracy
and multi-view consistency, we further introduce a light-
weight neural refinement step (Neural Merger), which takes
ray-traced material features as input and produces detailed
adjustments.
Our results demonstrate that the proposed
methods outperform existing techniques in both quantitative
metrics and perceived visual realism. This enables more
accurate, relightable, and photorealistic renderings from
1
arXiv:2512.18314v1  [cs.CV]  20 Dec 2025

<!-- page 2 -->
reconstructed scenes, significantly improving the realism
and efficiency of asset creation workflows in content pro-
duction pipelines. Project page: https://matspray.
jdihlmann.com/
1. Introduction
Editing and relighting real scenes captured with casual cam-
eras is central to many vision and graphics applications.
While modern neural 3D reconstruction methods can pro-
duce impressive geometry and appearance from images,
they often entangle illumination with appearance, yielding
textures or coefficients that are not physically meaningful
for relighting. Classical inverse rendering requires strong
assumptions about lighting and exposure and remains frag-
ile when materials vary spatially. In parallel, recent 2D ma-
terial predictors learn rich priors from large-scale data and
can produce plausible material maps from images, yet they
operate in 2D and are not directly consistent across views
or attached to a 3D representation.
We introduce a method to transfer 2D material pre-
dictions onto a 3D Gaussian representation to obtain re-
lightable assets with spatially varying base color, rough-
ness, and metallic parameters. The approach projects 2D
material maps to 3D via efficient ray-traced assignment,
refines materials with a small MLP to reduce multi-view
inconsistencies, and supervises rendered material maps di-
rectly with the 2D predictions to preserve plausible pri-
ors while discouraging baked-in lighting. This combina-
tion yields cleaner albedo, more accurate roughness, and in-
formed metallic estimates, enabling higher-quality relight-
ing compared to pipelines that learn only appearance. Our
contributions are:
• World Material Fusion. A plug-and-play pipeline that,
to our knowledge, is the first to fuse swappable diffusion-
based 2D PBR priors (“world material knowledge”) with
3D Gaussian material optimization via Gaussian ray trac-
ing and PBR consistent supervision to obtain relightable
assets.
• Neural Merger. A softmax neural merger that aggregates
per-Gaussian, multi-view material estimates, suppresses
baked-in lighting, and enforces cross-view consistency
while stabilizing joint environment map optimization.
• Faster Reconstruction.
A simple projection and op-
timization scheme that reconstructs high-quality re-
lightable 3D materials with 3.5× less per-scene optimiza-
tion time than IRGS [9].
2. Related Work
Materials
Spatially varying BRDFs (svBRDFs) have
long been studied, with early high-resolution texel-based
representations enabling point-wise material parameteriza-
tion [25, 44].
In this work, we adopt a Cook–Torrance
variant, which is based on the widely used Disney prin-
cipled BRDF and real-time formulations in major engines
[5, 19, 23]. Building on this foundation, recent research has
explored richer material parameterizations and extended the
expressiveness of svBRDF models [12].
Diffusion
Diffusion models enable high-fidelity image
synthesis and conditioning, with efficient latent-space for-
mulations and extensions to video generation [2, 13, 14, 42].
For material estimation from 2D images, large-scale diffu-
sion priors have been used to infer PBR maps [8, 20, 27,
55, 59]. Of particular relevance is DiffusionRenderer by
Huang et al. [16], whose results on high-quality material
maps motivated this work. Diffusion approaches have been
further explored for related tasks such as HDR prediction,
texture estimation, and relighting [1, 30, 54].
Gaussian Ray Tracing
We exploit Gaussian Ray Trac-
ing as a mechanism for transferring 2D material data to
3D. Although the field remains in its early stages, emerging
works have explored stochastic and explicit Gaussian ray-
tracing methods [34, 39, 47] and related neural optimization
schemes [10]. Our formulation builds upon insights from
Mai et al. [34] and Moenne-Loccoz et al. [39], extending
them toward material-aware 3D reconstruction.
Novel View Synthesis and Scene Reconstruction
Neu-
ral representations for novel view synthesis and scene re-
construction have rapidly advanced 3D modeling, with
NeRF and Gaussian Splatting providing strong foundations
for radiance-based scene representations [21, 37]. Material
modeling atop radiance fields has been explored through in-
verse rendering and relighting [3, 4, 43, 57], where cou-
pling with signed distance fields (SDFs) improves physical
plausibility [36, 50]. Gaussian-based inverse rendering ap-
proaches such as R3DGS [9] reconstruct materials via per-
scene optimization, while IRGS [11] extends this with 2D
Gaussians and deferred shading for improved appearance
modeling. Complementary efforts leverage diffusion mod-
els for 3D reconstruction and sparse-view recovery [31, 38],
and hybrid cues or mesh integration further enhance geo-
metric fidelity [28, 29, 48, 52]. Beyond these, extensions of
2D and 3D Gaussians have enabled spatially varying mate-
rials, advanced relighting, and richer reflectance modeling
[6, 7, 15, 18, 22, 24, 26, 32, 33, 46, 51, 53, 58].
In contrast to R3DGS and IRGS, which optimize mate-
rial and geometry parameters per scene, our approach em-
ploys Gaussian Ray Tracing to lift 2D material estimates
by diffusion models into 3D representations, exploiting the
world-knowledge priors learned by the diffusion models
and the broader understanding of material behavior they
2

<!-- page 3 -->
Figure 2. Pipeline. From multi-view images, a diffusion predictor yields per-view material maps. We reconstruct the object’s geometry
using 3D Gaussian Splatting. Then we project 2D materials to 3D via ray tracing, and refine per Gaussian materials with our Neural Merger
that has a softmax output layer, choosing between the projected values. We then supervise the produced material maps using the predicted
2D material maps. Additionally, using deferred shading we supervise by a PBR-based photometric rendering loss with the multi-view
ground truth images of the object.
provide. This enables faster, more consistent reconstruction
and material reasoning across scenes.
3. Method
Our method recovers consistent 3D PBR materials from
multiple views by combining 2D diffusion predictions with
a 3D Gaussian representation. We first obtain material maps
(base color, roughness, metallic) per view from any diffu-
sion material predictor, making our approach compatible
with a wide range of existing and future diffusion models.
Scene geometry is reconstructed via a relightable Gaussian
Splatting pipeline (R3DGS) [9, 21], which provides both
geometry and normals. The 2D material estimates are then
transferred to 3D and jointly refined for multi-view con-
sistency by a newly introduced Neural Merger step. The
materials maps and normals are further refined based on a
rendering loss, evaluated with deferred rendering. The il-
lumination is modeled by an optimizable environment map
during refinement.
Specifically, we (1) lift 2D materials to 3D via Gaus-
sian ray tracing, (2) refine per-Gaussian material parameters
with the Neural Merger across views, and (3) supervise ren-
dered material maps to preserve plausible 2D priors while
suppressing baked-in lighting.
3.1. Diffusion Material Prediction
We leverage pretrained diffusion priors to predict physically
meaningful per-view material maps, enabling accurate re-
construction of 3D PBR materials. Each material channel,
base color (three channels), roughness (single channel), and
metallic (single channel), is inferred explicitly. In practice,
we evaluate multiple prebuilt diffusion-based predictors and
select the one that provides the best fidelity and consistency
balance for our data. In our experiments, this is Diffusion-
Renderer [27]. We also tested Marigold [20] and RGB-to-
X [55]. We choose DiffusionRenderer because it achieves
about thirty percent higher PSNR than the other methods.
DiffusionRenderer predicts material maps from short
frame batches and uses a limited temporal context to im-
prove consistency and material understanding within each
batch. While this improves local consistency, the predicted
materials in a batch can still differ across views. The model
cannot recover the complete environment illumination from
a small input batch, which may lead to small shifts in color,
roughness, or metallic appearance within a batch. In addi-
tion, results from separate batches may not align, as these
images may introduce new information that was not visible
in previous batches. These variations make a direct pro-
jection of the predicted maps unreliable and often result in
blurry and washed out material maps. Our method resolves
this by refining and merging the estimates into one consis-
tent 3D representation.
3.2. 2D-to-3D Material Lifting
For each view vi, we collect the material attributes (base
color, roughness, metallic) corresponding to every Gaussian
g from the pixels within its projected footprint fpp using
3

<!-- page 4 -->
Gaussian ray tracing, following the formulation of Mai et
al. [35]. Their approach determines each Gaussian or ellip-
soid’s contribution to a ray based on density. Because the
opacity α used in Gaussian Splatting [21] does not directly
correspond to a physical density, we adopt the formulation
by Moenne-Loccoz et al. [39], which allows the direct use
of Gaussian Splatting opacity α in ray tracing.
For a Gaussian with mean µ and covariance Σ, the point
of maximum response xmax along a ray with origin o and
direction d is
τmax = (µ −o)⊤Σ−1d
d⊤Σ−1d
,
xmax = o + τmaxd.
(1)
The corresponding opacity αmax, given a base opacity α
and a falloff parameter λ > 0, is
αmax = α · exp

−1
2λ(xmax −µ)⊤Σ−1(xmax −µ)

.
(2)
Material values per pixel mp are then assigned to the
Gaussians intersected by the ray and aggregated across all
pixels in each Gaussian’s footprint fpg,vi per view. To re-
duce color distortion from outliers and overlapping foot-
prints, we compute a median of all assigned material pa-
rameters mg,vi per Gaussian:
mg,vi = medianp∈fpg,vi (mp).
(3)
After computing the median for each view, the resulting val-
ues are assigned to their corresponding Gaussians. Gaus-
sians not intersected in any view are removed. Grid-based
supersampling per pixel is used to ensure stable Gaussian
hits.
For each Gaussian g, we now obtain arrays of material
estimates across all views:
basecolorg = {bg,1, bg,2, . . . , bg,n},
(4)
metallicg = {mg,1, mg,2, . . . , mg,n},
(5)
roughnessg = {rg,1, rg,2, . . . , rg,n}.
(6)
These arrays contain the inconsistent material values per
view produced by DiffusionRenderer. These inconsisten-
cies motivate the subsequent Neural Merger step, which re-
fines the estimates into a coherent 3D representation.
3.3. Neural Merger
To reduce inconsistencies across views, we introduce the
Neural Merger, which predicts weights per view for the
material parameters collected during the projection step for
each Gaussian. It fuses the predictions into a single, con-
sistent estimate. The key idea is to interpolate between the
predicted values rather than allowing the network to freely
generate new colors or material values. This ensures that the
merged results remain consistent with the world knowledge
captured by the diffusion priors while enforcing coherence
across views.
For each Gaussian g, the Neural Merger takes as in-
put the projected material values mg,v for all views v ∈
{1, . . . , V }, along with the Gaussian position pg, encoded
using a positional encoding. The input is processed by a
lightweight MLP fθ to produce unnormalized weights hg,v
for each view:
[hg,1, hg,2, . . . , hg,V ] = fθ

pg, mg,1, . . . , mg,V

.
(7)
A softmax function then converts these outputs into normal-
ized weights:
wg,v =
exp(hg,v)
PV
v′=1 exp(hg,v′)
,
V
X
v=1
wg,v = 1.
(8)
The merged material mg for the Gaussian is computed as
the weighted sum of the per-view predictions:
mg =
V
X
v=1
wg,v mg,v.
(9)
The Neural Merger is optimized during the refinement ex-
plained in the next section. Using the softmax weighting
is crucial. Without it, the merger can converge faster than
the environment map optimization, producing unrealistic
material values that match the ground truth only superfi-
cially.
By interpolating among the predicted values, the
Neural Merger produces physically plausible material esti-
mates while allowing the environment map to converge re-
liably. In our framework, we use a separate Neural Merger
for each material channel, enabling improved disentangle-
ment of the materials.
3.4. Refinement, Supervision and Loss Functions
The Neural Merger produces material values per Gaussian,
which are then rasterized into material maps. These maps
are iteratively refined using two complementary supervised
losses.
First, we supervise the rendered material maps against
the diffusion model’s 2D material predictions using an L1
loss. This loss is applied exclusively to the material param-
eters, thereby optimizing only the Neural Merger. Given the
rendered material maps Mrender and the diffusion-predicted
material maps M2D, the material supervision loss LImage is
defined as:
LImage = ∥Mrender −M2D ∥1.
(10)
Second, the rasterized materials are used for deferred
shading to generate a physically based rendering (PBR) im-
age. This rendered image is then compared to the ground-
truth input using the loss introduced in Gaussian Splat-
ting [21]. The rendering supervision loss is defined as:
L3DGS = λ L1(IPBR, IGT)+(1−λ) LSSIM(IPBR, IGT), (11)
4

<!-- page 5 -->
Figure 3. Relighting Comparison between our method, an extended version of R3DGS [9] and IRGS [11]. The objects are all relit under
the same environment maps. In IRGS, reconstructed scene geometry might partially occlude the environment map.
where IPBR denotes the rendered image, IGT is the ground-
truth image, and λ ∈[0, 1] (typically set to 0.8).
This
loss supervises both the Neural Merger and the environment
map estimation, ensuring that the final rendering is consis-
tent with the input views.
4. Experiments
We evaluate our method on both synthetic and real-world
datasets from the Navi dataset [17], comparing against
state-of-the-art approaches for 3D material estimation and
relighting, specifically an extended version of R3DGS [9]
(modified to support metallic materials) and IRGS [11].
Our evaluation includes qualitative comparisons of material
maps and relighting quality, quantitative metrics across ma-
terial channels, computational performance, and an ablation
study.
4.1. Implementation Details
All methods use the same experimental setup for fair com-
parison. Synthetic scenes are initialized with a unit cube
point cloud containing 100,000 points sampled uniformly
at random. Real-world scenes use structure-from-motion
reconstruction from COLMAP [45] for both initialization
and camera inference.
We evaluate on 17 synthetic objects with ground-truth
material maps for quantitative analysis.
The synthetic
dataset uses 100 training images and 200 evaluation images
per object. Real-world objects use all available images (av-
erage of 27 per object).
To handle highly specular objects, we train Gaussian
Splatting on DiffusionRenderer normals as RGB, which
helps guide the geometry more effectively, since Gaussian
Splatting alone often struggles with specular surfaces and
can produce holes in the reconstruction.
The Neural Merger consists of separate MLPs for each
material channel (basecolor, roughness, metallic), each with
3 hidden layers of 128 neurons and ReLU activations. The
final layer outputs view-specific weights passed through
softmax to form a probability distribution.
We spend 30.000 iterations for the 3D Gaussian geome-
try optimization and 10.000 for the material refinement. All
experiments run on an NVIDIA RTX 4090 GPU [40].
We
evaluate
material
estimation
using
PSNR,
SSIM [49],
and LPIPS [56] between predicted and
ground-truth material maps.
For relighting, we ren-
der novel views under different environment maps and
compare against ground-truth renderings using the same
metrics.
4.2. Qualitative Results
Relighting Quality
Figure 3 compares relighting quality
across methods. Our method produces results that more
closely resemble ground truth, particularly for specular ob-
jects. The extended R3DGS struggles with specular ma-
terials, often producing brighter images than ground truth
5

<!-- page 6 -->
Figure 4. Material Maps produced by our method compared to extended R3DGS [9], which can also predict metallic material maps, IRGS
[11], and the DiffusionRenderer material output produced on the test images that are not used for training. We show four images each,
where the top left is the base color, top right is the roughness, bottom left is metallic, and bottom right are the normals.
Figure 5. Real-World Comparison of our method, extended R3DGS [9] and IRGS [11]. The ground truth images show the object masked
(top) and unmasked (bottom) to give a better understanding of the object and the surrounding lighting.
due to unconstrained material maps during joint environ-
ment map optimization. In contrast, our method constrains
material maps, enabling more accurate environment map
optimization and improved material estimation.
IRGS exhibits artifacts such as floaters and overly flat
surfaces. While it reconstructs flat surfaces well using 2D
6

<!-- page 7 -->
Table 1. Quantitative Comparison of material estimation and relighting results on 17 synthetic objects. We compare against IRGS [11]
and an extended version of R3DGS [9] that supports metallic materials. * For non-metallic objects, our model correctly optimizes the
parameter to zero, which can result in infinite PSNR when all metallic maps are predicted as zero.
Task
Ours
Ext. R3DGS
IRGS
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
PSNR ↑
SSIM ↑
LPIPS ↓
Relighting
27.282
0.897
0.080
25.483
0.875
0.094
24.409
0.850
0.166
BaseColor
21.341
0.873
0.125
18.360
0.832
0.158
19.204
0.750
0.139
Roughness
15.331
0.820
0.181
14.473
0.763
0.216
16.182
0.744
0.192
Metallic
∞∗/27.202
0.893
0.106
10.073
0.693
0.261
N/A
N/A
N/A
Gaussians, it loses fine detail compared to our method.
Moreover, IRGS cannot predict metallic materials, prevent-
ing accurate reconstruction of highly specular metallic sur-
faces. Our method achieves clear advantages in reconstruct-
ing metallic and highly specular materials while preserving
geometric detail across both diffuse and flat surfaces.
On the real-world relighting results in Figure 5 one can
observe the most consistent results with our method. With
R3DGS, the estimated color appears too bright, the objects
are too shiny and the geometry is less smooth, while IRGS
produces smooth geometry but too dark colors.
Material Maps
Figure 4 compares material maps across
methods. Our method effectively removes baked-in light-
ing effects from basecolor maps, producing nearly diffuse
basecolors with minimal shadowing, while other methods
exhibit visible shadows and residual lighting.
Compared to DiffusionRenderer’s original predictions,
which exhibit significant view-dependent inconsistencies,
our Neural Merger enhances multi-view consistency.
This is particularly evident in roughness and metallic
maps, where DiffusionRenderer’s per-view predictions vary
across views.
Our approach removes these inconsisten-
cies while preserving high-quality spatially varying mate-
rial properties.
Our metallic maps match ground truth well, with pre-
dictions substantially closer than competing methods. For
roughness, our method guides material maps toward con-
sistent values for surfaces sharing the same properties, im-
proving upon the 2D predictions.
While DiffusionRen-
derer struggles with roughness accuracy (reflected in darker
roughness maps), our Neural Merger refines initial predic-
tions by enforcing view consistency, resulting in more phys-
ically plausible parameters. In contrast, R3DGS tends to
overestimate roughness due to specular highlights appear-
ing in only a subset of the images, biasing optimization
toward diffuse surfaces.
IRGS produces overly uniform
roughness where fine details are difficult to discern.
Although IRGS normals are slightly closer to ground
truth in some regions, our method significantly improves
normals compared to extended R3DGS despite starting
from the same 3D Gaussian geometry. Overall, our method
produces qualitatively superior material maps across basec-
olor, normals, metallic, and roughness channels.
4.3. Quantitative Results
Table 1 shows quantitative results on 17 synthetic objects.
Our method consistently outperforms all baselines in re-
lighting quality and basecolor estimation, consistent with
the qualitative comparisons.
For roughness, IRGS achieves slightly higher PSNR
(16.182 vs. 15.331), but our method achieves better SSIM
(0.820 vs. 0.744) and LPIPS (0.181 vs. 0.192), indicating
better preservation of structural information and perceptual
quality. All methods face challenges in roughness estima-
tion, which remains a difficult problem.
For metallic maps, our method shows substantial im-
provement. When correctly predicting fully non-metallic
objects, PSNR becomes infinite, which occurs exclusively
for our method. Those objects are left out in the actual
PSNR calculation. While DiffusionRenderer often predicts
partial metallicity in certain views our method enforces
view-consistent material estimation, producing stable non-
metallic predictions.
Table 2. Runtime Comparison between our method and IRGS
on the Navi dataset. All timings are reported in seconds and mea-
sured on an NVIDIA RTX 4090 GPU. Our method is approxi-
mately 3.5× faster than IRGS.
Stage
Ours (s)
IRGS (s)
Diffusion Predictions
112
-
Gaussian Splatting
131
2490
Normal Generation (R3DGS)
270
-
Material Optimization
975
2857
Total
1488
5347
7

<!-- page 8 -->
4.4. Computation Time
Table 2 shows runtime breakdown on the Navi dataset [17].
DiffusionRenderer requires 112 seconds (∼6 seconds per
image) to process the full image set. Gaussian Splatting
takes 131 seconds on average (ranging from 64 to 274 sec-
onds depending on object complexity). Normal generation
using R3DGS takes 270 seconds on average (247–347 sec-
onds). Material optimization requires 975 seconds on av-
erage, extending up to 3,631 seconds (approximately one
hour) for complex objects.
In total, our method takes 1,488 seconds (∼25 minutes)
on average, approximately 3.5× faster than IRGS (5,347
seconds, ∼89 minutes).
4.5. Ablation Study
Table 3 demonstrates that the Neural Merger is the key com-
ponent responsible for our method’s superior performance.
The full model achieves the highest scores across all met-
rics (PSNR: 29.164, SSIM: 0.9105, LPIPS: 0.0626), signif-
icantly outperforming all ablated variants. This improve-
ment stems from the Neural Merger’s ability to enforce
multi-view consistency while preserving high-quality ma-
terial properties predicted by the diffusion model.
Figure 6. Qualitative Comparison of our ablations. The differ-
ences are most apparent in the base color, which has the largest
impact on the qualitative appearance during relighting and con-
sists of three channels rather than one.
Table 3. Quantitative Ablation study on a subset of the eval-
uated objects. The Supervised variant uses only the predicted 2D
diffusion-based material maps for supervision, without any 3D op-
timization. The Proj. Average variant directly projects all 2D pre-
dictions onto the Gaussians and averages them, without applying
any further training or optimization. The Full model includes the
Neural Merger.
Method
PSNR ↑
SSIM ↑
LPIPS ↓
Full
29.164
0.9105
0.0626
Supervised
24.809
0.889
0.0792
Proj. Average
25.555
0.866
0.122
The Supervised variant performs worse than the full
model, relying solely on diffusion predictions without ge-
ometric or photometric optimization.
Interestingly, it is
even worse than the Proj. Average baseline, which simply
projects diffusion predictions into Gaussians without train-
ing. We attribute this to view-dependent effects captured in
optimized Gaussians that are absent without optimization.
The qualitative results in Figure 6 show that the Neu-
ral Merger produces cleaner and more consistent material
maps. Visualized using base color (where baked-in light-
ing effects are most evident), the Neural Merger yields sub-
stantial improvements in both sharpness and color fidelity
compared to the Proj. Average baseline, which serves as our
initialization. These results validate that the Neural Merger
enhances visual quality and enforces consistency with un-
derlying physical properties, resulting in more accurate and
realistic relighting outcomes.
5. Conclusion
MatSpray enables casual acquisition and high-quality re-
construction of photorealistic relightable 3D assets with
spatially varying materials. It effectively lifts 2D material
predictions to 3D to fuse them with the 3D Gaussian ge-
ometry. It employs pretrained 2D diffusion-based material
estimators without requiring additional expensive training
on large-scale 3D PBR data sets.
Introducing the Neu-
ral Merger, our method significantly improves multi-view
consistencies, which even video-based material prediction
models still struggle with.
The resulting relightable 3D
models feature improved quality both in the estimated ma-
terial maps as well as in the final relit appearance, even
more so for highly specular objects. As demonstrated, Mat-
Spray outperforms current state-of-the-art methods and ex-
cels in reconstructing accurate metallic maps for both syn-
thetic and real-world inputs. The approach provides a pow-
erful tool for easy 3D content generation.
Limitations
While our approach drastically improves
multi-view consistency, the overall material quality re-
mains dependent on the performance of the chosen diffusion
model. However, our PBR-to-image loss partially corrects
small deviations in the diffusion predictions 4 Diffusion-
Renderer vs. MatSpray.
Our method struggles when inconsistent geometry
and normals are produced by the underlying R3DGS
method [9], though the photometric loss may partially miti-
gate these issues. Additionally, very small or flat Gaussians
might sometimes be missed during ray tracing. Future work
could address missing Gaussians through a projection trans-
former combination assignment similar to [41].
8

<!-- page 9 -->
Future Work
The high quality of the resulting 3D
geometry-material association could be exploited for accu-
rate 3D object part segmentation. This segmentation, paired
with matching language features, might enable object-
specific constraints in reconstruction or a natural interface
for manipulating both geometry and reflections.
6. Acknowledgements
Funded by the Deutsche Forschungsgemeinschaft (DFG,
German Research Foundation) under Germany’s Excel-
lence Strategy – EXC number 2064/1 – Project number
390727645.
This work was supported by the German
Research Foundation (DFG): SFB 1233, Robust Vision:
Inference Principles and Neural Mechanisms, TP 02,
project number: 276693517.
This work was supported
by the T¨ubingen AI Center.
The authors thank the In-
ternational Max Planck Research School for Intelligent
Systems (IMPRS-IS) for supporting Jan-Niklas Dihlmann.
References
[1] Hrishav Bakul Barua, Kalin Stefanov, Ganesh Krishnasamy,
KokSheik Wong, and Abhinav Dhall. Physhdr: When light-
ing meets materials and scene geometry in hdr reconstruc-
tion, 2025. 2
[2] Andreas Blattmann, Robin Rombach, Kaan Oktay, and
Bj”orn Ommer. Align your latents: High-resolution video
synthesis with latent diffusion models.
arXiv preprint
arXiv:2304.08818, 2023. 2
[3] Mark Boss, Raphael Braun, Varun Jampani, Jonathan T Bar-
ron, Ce Liu, and Hendrik Lensch. Nerd: Neural reflectance
decomposition from image collections. In Proceedings of
the IEEE/CVF International Conference on Computer Vi-
sion, pages 12684–12694, 2021. 2
[4] Mark Boss, Varun Jampani, Raphael Braun, Ce Liu,
Jonathan Barron, and Hendrik Lensch.
Neural-pil: Neu-
ral pre-integrated lighting for reflectance decomposition.
Advances in Neural Information Processing Systems, 34:
10691–10704, 2021. 2
[5] Brent Burley. Physically based shading at disney. In ACM
SIGGRAPH 2012 Courses: Physically Based Shading in
Theory and Practice, Los Angeles, CA, 2012. Course Notes.
2
[6] Hongze Chen, Zehong Lin, and Jun Zhang. Gi-gs: Global
illumination decomposition on gaussian splatting for inverse
rendering. arXiv preprint arXiv:2410.02619, 2024. 2
[7] Jan-Niklas Dihlmann, Arjun Majumdar, Andreas Engel-
hardt, Raphael Braun, and Hendrik Lensch. Subsurface scat-
tering for gaussian splatting. Advances in Neural Informa-
tion Processing Systems, 37:121765–121789, 2024. 2
[8] Andreas Engelhardt, Mark Boss, Vikram Voleti, Chun-Han
Yao, Hendrik P.A. Lensch, and Varun Jampani. SViM3D:
Stable video material diffusion for single image 3d genera-
tion. In ICCV, 2025. 2
[9] Jian Gao, Chun Gu, Youtian Lin, Zhihao Li, Hao Zhu, Xun
Cao, Li Zhang, and Yao Yao. Relightable 3d gaussians: Re-
alistic point cloud relighting with brdf decomposition and
ray tracing. In European Conference on Computer Vision,
pages 73–89. Springer, 2024. 2, 3, 5, 6, 7, 8, 12, 13
[10] Shrisudhan Govindarajan, Daniel Rebain, Kwang Moo Yi,
and Andrea Tagliasacchi. Radiant foam: Real-time differen-
tiable ray tracing. arXiv preprint arXiv:2502.01157, 2025.
2
[11] Chun Gu, Xiaofei Wei, Zixuan Zeng, Yuxuan Yao, and Li
Zhang. Irgs: Inter-reflective gaussian splatting with 2d gaus-
sian ray tracing. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pages 10943–10952, 2025.
2, 5, 6, 7, 12, 13
[12] Yu Guo, Zhiqiang Lao, Xiyun Song, Yubin Zhou, Zongfang
Lin, and Heather Yu. epbr: Extended pbr materials in im-
age synthesis. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pages 327–336, 2025. 2
[13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising dif-
fusion probabilistic models. In Advances in Neural Informa-
tion Processing Systems, 2020. 2
[14] Jonathan Ho, William Chan, Chitwan Saharia, David J. Fleet,
Mohammad Norouzi, and Tim Salimans.
Video diffusion
models. In Advances in Neural Information Processing Sys-
tems, 2022. 2
[15] Chenxiao Hu, Meng Gai, Guoping Wang, and Sheng Li.
Real-time global illumination for dynamic 3d gaussian
scenes. arXiv preprint arXiv:2503.17897, 2025. 2
[16] Yifeng Huang, Zhang Chen, Yi Xu, Minh Hoai, and Zhong
Li. Dualmat: Pbr material estimation via coherent dual-path
diffusion. arXiv preprint arXiv:2508.05060, 2025. 2
[17] Varun Jampani, Kevis-Kokitsi Maninis, Andreas Engel-
hardt, Arjun Karpur, Karen Truong, Kyle Sargent, Stefan
Popov, Andre Araujo, Ricardo Martin-Brualla, Kaushal Pa-
tel, Daniel Vlasic, Vittorio Ferrari, Ameesh Makadia, Ce
Liu, Yuanzhen Li, and Howard Zhou.
Navi: Category-
agnostic image collections with high-quality 3d shape and
pose annotations. In NeurIPS, 2023. 5, 8
[18] Joanna Kaleta, Kacper Kania, Tomasz Trzci´nski, and Marek
Kowalski. Lumigauss: Relightable gaussian splatting in the
wild. In 2025 IEEE/CVF Winter Conference on Applications
of Computer Vision (WACV), pages 1–10. IEEE, 2025. 2
[19] Brian Karis.
Real shading in unreal engine 4.
Technical
report, Epic Games, 2013. 2
[20] Bingxin Ke, Kevin Qu, Tianfu Wang, Nando Metzger,
Shengyu Huang, Bo Li, Anton Obukhov, and Konrad
Schindler.
Marigold: Affordable adaptation of diffusion-
based image generators for image analysis, 2025. 2, 3
[21] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 2, 3, 4
[22] Georgios Kouros, Minye Wu, and Tinne Tuytelaars. Rgs-dr:
Reflective gaussian surfels with deferred rendering for shiny
objects. arXiv preprint arXiv:2504.18468, 2025. 2
[23] S´ebastien Lagarde and Charles de Rousiers. Moving frost-
bite to physically based rendering 3.0. In ACM SIGGRAPH
2014 Courses: Physically Based Shading in Theory and
Practice, Vancouver, Canada, 2014. Course Slides. 2
9

<!-- page 10 -->
[24] Shuichang Lai, Letian Huang, Jie Guo, Kai Cheng, Bowen
Pan, Xiaoxiao Long, Jiangjing Lyu, Chengfei Lv, and Yan-
wen Guo. Glossygs: Inverse rendering of glossy objects with
3d gaussian splatting. IEEE Transactions on Visualization
and Computer Graphics, 2025. 2
[25] Hendrik P. A. Lensch, Jan Kautz, Michael Goesele, Wolf-
gang Heidrich, and Hans-Peter Seidel. Image-based recon-
struction of spatially varying materials. In Proceedings of the
12th Eurographics Conference on Rendering, page 103–114,
Goslar, DEU, 2001. Eurographics Association. 2
[26] Jingzhi Li, Zongwei Wu, Eduard Zamfir, and Radu Timofte.
Recap: Better gaussian relighting with cross-environment
captures. In Proceedings of the Computer Vision and Pat-
tern Recognition Conference, pages 21307–21316, 2025. 2
[27] Ruofan Liang, Zan Gojcic, Huan Ling, Jacob Munkberg, Jon
Hasselgren, Zhi-Hao Lin, Jun Gao, Alexander Keller, Nan-
dita Vijaykumar, Sanja Fidler, and Zian Wang. Diffusion-
renderer: Neural inverse and forward rendering with video
diffusion models. In The IEEE Conference on Computer Vi-
sion and Pattern Recognition (CVPR), 2025. 2, 3, 12
[28] Zhihao Liang, Qi Zhang, Ying Feng, Ying Shan, and Kui Jia.
Gs-ir: 3d gaussian splatting for inverse rendering. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 21644–21653, 2024. 2
[29] Lianjun Liao, Chunhui Zhang, Tong Wu, Henglei Lv, Bailin
Deng, and Lin Gao. Rosgs: Relightable outdoor scenes with
gaussian splatting, 2025. 2
[30] Yehonathan Litman, Fernando De la Torre, and Shubham
Tulsiani. Lightswitch: Multi-view relighting with material-
guided diffusion. arXiv preprint arXiv:2508.06494, 2025.
2
[31] Bo Liu, Runlong Li, Li Zhou, and Yan Zhou. Dt-nerf: A
diffusion and transformer-based optimization approach for
neural radiance fields in 3d reconstruction, 2025. 2
[32] Shiyong Liu, Xiao Tang, Zhihao Li, Yingfan He, Chongjie
Ye, Jianzhuang Liu, Binxiao Huang, Shunbo Zhou, and Xi-
aofei Wu. Occlugaussian: Occlusion-aware gaussian splat-
ting for large scene reconstruction and rendering.
arXiv
preprint arXiv:2503.16177, 2025. 2
[33] Zhenyuan Liu, Yu Guo, Xinyuan Li, Bernd Bickel, and
Ran Zhang.
Bigs:
Bidirectional gaussian primitives
for relightable 3d gaussian splatting.
arXiv preprint
arXiv:2408.13370, 2024. 2
[34] Alexander Mai, Peter Hedman, George Kopanas, Dor
Verbin, David Futschik, Qiangeng Xu, Falko Kuester,
Jonathan T Barron, and Yinda Zhang. Ever: Exact volumet-
ric ellipsoid rendering for real-time view synthesis. arXiv
preprint arXiv:2410.01804, 2024. 2
[35] Alexander Mai, Peter Hedman, George Kopanas, Dor
Verbin, David Futschik, Qiangeng Xu, Falko Kuester,
Jonathan T Barron, and Yinda Zhang. Ever: Exact volumet-
ric ellipsoid rendering for real-time view synthesis. arXiv
preprint arXiv:2410.01804, 2024. 4
[36] Shi Mao, Chenming Wu, Zhelun Shen, Yifan Wang, Dayan
Wu, and Liangjun Zhang. Neus-pir: Learning relightable
neural surface using pre-integrated rendering.
Computa-
tional Visual Media, 2025. 2
[37] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
2
[38] Niluthpol Chowdhury Mithun, Tuan Pham, Qiao Wang,
Ben Southall, Kshitij Minhas, Bogdan Matei, Stephan
Mandt, Supun Samarasekera, and Rakesh Kumar. Diffusion-
guided gaussian splatting for large-scale unconstrained 3d
reconstruction and novel view synthesis.
arXiv preprint
arXiv:2504.01960, 2025. 2
[39] Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Ric-
cardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja
Fidler, Nicholas Sharp, and Zan Gojcic. 3d gaussian ray trac-
ing: Fast tracing of particle scenes. ACM Transactions on
Graphics and SIGGRAPH Asia, 2024. 2, 4
[40] NVIDIA Corporation.
Nvidia geforce rtx 4090 specifica-
tions, 2022. 24GB GDDR6X VRAM, 16,384 CUDA cores,
450W TDP. 5
[41] Kerui Ren, Jiayang Bai, Linning Xu, Lihan Jiang, Jiangmiao
Pang, Mulin Yu, and Bo Dai. Mv-colight: Efficient object
compositing with consistent lighting and shadow generation.
arXiv preprint arXiv:2505.21483, 2025. 8
[42] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj”orn Ommer. High-resolution image
synthesis with latent diffusion models.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 2022. 2
[43] Viktor Rudnev, Mohamed Elgharib, William Smith, Lingjie
Liu, Vladislav Golyanik, and Christian Theobalt. Nerf for
outdoor scene relighting. In European Conference on Com-
puter Vision, pages 615–631. Springer, 2022. 2
[44] Yoichi Sato, Mark D. Wheeler, and Katsushi Ikeuchi. Ob-
ject shape and reflectance modeling from observation.
In
Proceedings of the 24th Annual Conference on Computer
Graphics and Interactive Techniques, page 379–387, USA,
1997. ACM Press/Addison-Wesley Publishing Co. 2
[45] Johannes L Schonberger and Jan-Michael Frahm. Structure-
from-motion revisited.
In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
4104–4113, 2016. 5
[46] Hanxiao Sun, Yupeng Gao, Jin Xie, Jian Yang, and Beibei
Wang. Svg-ir: Spatially-varying gaussian splatting for in-
verse rendering. In Proceedings of the Computer Vision and
Pattern Recognition Conference, pages 16143–16152, 2025.
2
[47] Xin Sun, Iliyan Georgiev, Yun Fei, and Miloˇs Haˇsan.
Stochastic ray tracing of 3d transparent gaussians.
arXiv
preprint arXiv:2504.06598, 2025. 2
[48] Zipeng Wang and Dan Xu. Hyrf: Hybrid radiance fields for
efficient and high-quality novel view synthesis. The Thirty-
Ninth Annual Conference on Neural Information Processing
Systems (NeurIPS), 2025. 2
[49] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
moncelli. Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing,
13(4):600–612, 2004. 5
10

<!-- page 11 -->
[50] Sean Wu, Shamik Basu, Tim Broedermann, Luc Van Gool,
and Christos Sakaridis.
Pbr-nerf: Inverse rendering with
physics-based neural fields. In Proceedings of the Computer
Vision and Pattern Recognition Conference, pages 10974–
10984, 2025. 2
[51] Zirui Wu, Jianteng Chen, Laijian Li, Shaoteng Wu, Zhikai
Zhu, Kang Xu, Martin R Oswald, and Jie Song. 3d gaus-
sian inverse rendering with approximated global illumina-
tion. arXiv preprint arXiv:2504.01358, 2025. 2
[52] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann
Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys.
Depthsplat: Connecting gaussian splatting and depth.
In
Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 16453–16463, 2025. 2
[53] Xingyuan Yang and Min Wei. Gogs: High-fidelity geometry
and relighting for glossy objects via gaussian surfels, 2025.
2
[54] Zhi Ying, Boxiang Rong, Jingyu Wang, and Maoyuan Xu.
Chord: Chain of rendering decomposition for pbr material
estimation from generated texture images, 2025. 2
[55] Zheng Zeng, Valentin Deschaintre, Iliyan Georgiev, Yannick
Hold-Geoffroy, Yiwei Hu, Fujun Luan, Ling-Qi Yan, and
Miloˇs Haˇsan. Rgb↔x: Image decomposition and synthe-
sis using material- and lighting-aware diffusion models. In
ACM SIGGRAPH 2024 Conference Papers, New York, NY,
USA, 2024. Association for Computing Machinery. 2, 3
[56] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman,
and Oliver Wang. The unreasonable effectiveness of deep
features as a perceptual metric. Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
586–595, 2018. 5
[57] Xiuming Zhang, Pratul P Srinivasan, Boyang Deng, Paul De-
bevec, William T Freeman, and Jonathan T Barron. Ner-
factor: Neural factorization of shape and reflectance under
an unknown illumination. ACM Transactions on Graphics
(ToG), 40(6):1–18, 2021. 2
[58] Xiaoming Zhao, Pratul Srinivasan, Dor Verbin, Keunhong
Park, Ricardo Martin Brualla, and Philipp Henzler. Illumin-
erf: 3d relighting without inverse rendering. Advances in
Neural Information Processing Systems, 37:42593–42617,
2024. 2
[59] Rongjia Zheng, Qing Zhang, Chengjiang Long, and Wei-Shi
Zheng. Dnf-intrinsic: Deterministic noise-free diffusion for
indoor inverse rendering. arXiv preprint arXiv:2507.03924,
2025. 2
11

<!-- page 12 -->
MatSpray: Fusing 2D Material World Knowledge on 3D Geometry
Supplementary Material
Overview
This supplementary material provides extended results,
analyses, and implementation details that complement the
findings in the main paper. For ease of navigation, the main
components are summarized here and referenced through
the corresponding section labels.
• Additional Videos and Real-World Objects (A): A de-
tailed collection of video comparisons and reconstruc-
tions of real objects that highlight the performance and
stability of our method relative to earlier approaches.
• Neural Merger Ablation (B): An extended analysis of
the importance of the final Softmax layer in the Neu-
ral Merger, supported by qualitative and quantitative evi-
dence.
• Tone Mapping Analysis (C): A discussion of the tone
mapping behaviour of DiffusionRenderer, how this af-
fects predicted base color, roughness and metallic maps,
and why this creates a mismatch when compared to linear
ground truth.
• Implementation Details (D): A description of our train-
ing setup, super sampling strategy, Neural Merger inputs
and other practical considerations that are important for
stable optimization.
A. Additional Videos and Real-World Objects
Figure 7 shows the thumbnail that links to all addi-
tional videos included with this supplementary material.
These videos provide an extensive visual comparison of our
method with Extended R3DGS [9], IRGS [11], and the for-
ward renderer of DiffusionRenderer [27]. While the main
paper presents representative examples, the extended videos
Figure 7. Thumbnail showing six videos that can be viewed here.
Three videos show relighting comparison and three show material
prediction comparisons.
give a more complete picture of the consistency and stabil-
ity of our approach, especially compared to the produced
material maps of DiffusionRenderer.
Across the set of videos, our method consistently pro-
duces reconstructions that remain stable across all view-
points, without the flickering or structural collapse that can
be observed in the other methods. This is particularly visi-
ble in objects with complex geometry or pronounced spec-
ular highlights such as the Kettle. Extended R3DGS of-
ten fails to maintain surface smoothness and yields unsta-
ble representations. On the other hand IRGS tends to over-
smooth surfaces and tends to bake in specular reflections
of metallic objects into its base color. In contrast, our ap-
proach maintains coherent structure even under strong light-
ing variations.
To illustrate this, we provide three relighting videos:
White Golden Airplane, Stone Birdhouse, and Kettle. Ad-
ditionally, three videos visualize predicted material proper-
ties: Yellow Airplane, Birdhouse with Yellow Flower, and
Chair. These examples show that DiffusionRenderer, de-
spite being trained on its own dataset, still produces incon-
sistent material maps that vary strongly with camera angle
and lighting. Our method mitigates these issues and aligns
predictions across views more reliably.
Real-World Objects
Figure 8 shows additional real ob-
jects reconstructed by our method and by Extended R3DGS
and IRGS. Here, each method is evaluated under two re-
lighting settings and compared in base color and normals.
The differences are most obvious in the base color: our base
color is locally sharp and coherent across the surface, while
both baselines exhibit noise, distortions, or view-dependent
artifacts. The relighting results further demonstrate that our
predicted materials generalize well across lighting condi-
tions, while the other methods still have lighting effects
baked into their materials (R3DGS) or tend to be washed
out (IRGS).
B. Neural Merger Ablation
The Neural Merger plays a key role in ensuring that
the material parameters assigned to each Gaussian remain
stable and consistent across all viewpoints.
One cen-
tral element of the Neural Merger is the final Softmax
layer, which normalizes its output into weights acting as a
weighted average of the inputs. Although this layer may ap-
12

<!-- page 13 -->
Figure 8. Additional real objects reconstructed with our method, Extended R3DGS [9] and IRGS [11]. The figure includes relighting under
two environments, base color and normal maps.
Figure 9. The impact of the Softmax layer in the Neural Merger. Without it, lighting and shadow patterns leak into the material maps,
leading to inconsistent relighting.
13

<!-- page 14 -->
Figure 10. Tone mapping applied by DiffusionRenderer significantly alters the appearance of material maps. The alpha mask removes
background content and focuses on the region of interest.
pear to be a small architectural detail, it has a sizable impact
on the quality of the final reconstruction.
Without the Softmax normalization, the Neural Merger
becomes unconstrained and starts to absorb illumination
cues directly from the training images.
In other words,
instead of learning clean, view-independent materials, the
MLP blends in signals that correspond to lighting variations
and shadows. Because these patterns differ between view-
points, the network produces material values that vary from
view to view, which leads to inconsistency during render-
ing. Although this might also be additionally influenced by
slight variations in the 2D Diffusion predictions geometry.
This behaviour becomes especially problematic under re-
lighting, because the embedded shadows and highlights in-
terfere with the simulated lighting and produce unrealistic
results.
Figure 9 shows a comparison between the full method,
the version without the Softmax, and the linear ground
truth. The differences become clear when observing fine
geometric structures and shadow placement.
Without
Softmax, shadows from the input images appear in the
base color maps and the renderings become blurry in high
detail areas. These issues are especially visible in the lower
birdhouse example, where the version without Softmax
fails to maintain consistent materials on the swim ring and
the surrounding areas.
We further quantify these findings in Table 4, which re-
ports results across all scenes in the dataset. The full model
outperforms the version without the Softmax across all
metrics, with especially large gains in perceptual similarity
(LPIPS). This confirms that the Softmax-based normaliza-
tion is not merely a numerical improvement but a key com-
ponent that ensures robustness and prevents the network
from encoding view-dependent appearance into the mate-
rials.
C. DiffusionRenderer Tone Mapping Analysis
One recurring observation in our experiments was that
the base color predicted by our method tended to appear
darker than the linear ground truth material map. This ap-
peared to be a miss-prediction of the 2D material maps by
DiffusionRenderer for a few objects. However, this discol-
oration appeared in almost all objects that we tested, hint-
ing towards a systemic problem in DiffusionRenderer. Fig-
ure 10 illustrates this systemic discoloration of the predicted
base color. This indicates that during training Diffusion-
Renderer was supervised using tone-mapped ground truth
images.
Our analysis suggests that DiffusionRenderer employs a
filmic or AgX tone mapping curve. These tone-mapping
algorithms compress high dynamic range values into the
Table 4. Ablation Study on all objects. Removing the Softmax
layer causes the network to encode lighting, which degrades all
metrics.
Method
PSNR ↑
SSIM ↑
LPIPS ↓
Full
27.282
0.897
0.080
Without Softmax
24.600
0.874
0.114
14

<!-- page 15 -->
range expected by standard displays, which improves visual
quality but complicates the recovery of physically meaning-
ful material parameters. In particular, these tone mappings
are not analytically invertible, and even approximate inverse
curves introduce errors, especially near shadows or high-
lights.
Base color is affected in a predictable way, because tone
mapping acts like a softened gamma curve. Applying an
inverse gamma of roughly one point eight partially recov-
ers the linear values but cannot undo the full nonlinearity.
Roughness is affected more severely, because its values oc-
cupy a small part of the zero to one interval, which collapses
under tone mapping. Metallic values, on the other hand, re-
main closer to either zero or one and thus suffer less from
compression. These effects explain why our predicted ma-
terial maps sometimes differ from the linear ground truth as
they closely match DiffusionRenderer’s tone-mapped out-
put.
D. Implementation Details
Our experiments were performed on an NVIDIA RTX
4090 GPU with PyTorch, C++ and Optix. To keep the input
consistent with the internal resolution of DiffusionRenderer,
we render all training views at a resolution of 512×512
pixel. This choice ensures that the reconstruction quality
aligns with the scale at which DiffusionRenderer was origi-
nally trained. In scenes with strong specular highlights, we
disable geometry learning entirely and keep the Gaussian
positions fixed, because additional geometric optimization
tends to destabilize the representation under these condi-
tions.
The Neural Merger is optimized using a learning rate of
zero point zero zero one. Material supervision uses an L1-
loss with a weight of 1.0, as we found that this balance
Figure 11. Super sampling avoids missed Gaussians and ensures
reliable projection of material supervision. Lower sampling rates
cause holes and unstable geometry.
prevents the model from overfitting shadows while still en-
forcing high fidelity in the material maps. During training,
we also apply random view sampling to avoid biasing the
model toward any particular viewpoint.
Super Sampling
A key technical detail is the use of super
sampling during the projection of material values into the
Gaussian representation. We employ a 16×16 grid of rays
per pixel to ensure that even small or distant Gaussians re-
ceive material parameters. With fewer samples, Gaussians
are occasionally missed leading to a patchy geometry and
a low resolution material parameter transferal. Figure 11
shows an example where a lower sampling rate produces
obvious reconstruction defects.
Merger Inputs
Finally, Figure 12 illustrates the input to
the Neural Merger. The features consist of a NeRF-style
positional encoding of the Gaussian location along with the
projected base color, roughness and metallic values. The
combination of positional encoding and projected materi-
als allows the network to balance local detail with global
consistency, which is essential for producing clean results
under relighting.
Figure 12. The input to the Neural Merger includes positional
codes and projected material parameters.
15
