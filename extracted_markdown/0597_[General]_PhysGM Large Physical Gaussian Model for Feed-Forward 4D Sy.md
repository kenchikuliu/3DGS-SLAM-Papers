<!-- page 1 -->
PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis
Chunji Lv1*
Zequn Chen2∗†
Donglin Di2
Weinan Zhang3,5
Hao Li2
Wei Chen2
Yinjie Lei4
Changsheng Li1‡
1Beijing Institute of Technology
2Li Auto
3Harbin Institute of Technology
4Sichuan University
5Suzhou Research Institute, Harbin Institute of Technology
3120250994@bit.edu.cn
lcs@bit.edu.cn
Physical Gaussian reconstruction < 1s
Simulation videos
M
P
M
Physics-based simulation & rendering ～30s
Nu:0.3112
E:2.3e5
Density:2000
Material
Plasticine
Nu:0.2500
E:2.0e8
Density:7800
Material
metal
Input images
Physical Gaussian
PhysGM
Nu:0.0401
Density:1200
E:2.0e4
Material
jelly
Figure 1. Overview of PhysGM. Given a single image, PhysGM performs a single feed-forward pass to directly predict 3D Gaussian
Splatting (3DGS) representation and its associated physical properties (e.g., stiffness, mass). This prediction is optimization-free and
completes in under one second. The generated parameters then initialize a Material Point Method (MPM) simulator, producing the final,
physically plausible 4D animation.
Abstract
Despite advances in physics-based 3D motion synthesis,
current methods face key limitations:
reliance on pre-
reconstructed 3D Gaussian Splatting (3DGS) built from
dense multi-view images with time-consuming per-scene
optimization; physics integration via either inflexible, hand-
specified attributes or unstable, optimization-heavy guid-
ance from video models using Score Distillation Sam-
pling (SDS); and na¨ıve concatenation of prebuilt 3DGS
with physics modules, which ignores physical information
embedded in appearance and yields suboptimal perfor-
mance.
To address these issues, we propose PhysGM,
a feed-forward framework that jointly predicts 3D Gaus-
sian representation and physical properties from a sin-
gle image, enabling immediate simulation and high-fidelity
∗Equal contribution.
†Project Leader.
‡Corresponding author.
4D rendering. Unlike slow appearance-agnostic optimiza-
tion methods, we first pre-train a physics-aware recon-
struction model that directly infers both Gaussian and
physical parameters.
We further refine the model with
Direct Preference Optimization (DPO), aligning simula-
tions with the physically plausible reference videos and
avoiding the high-cost SDS optimization. To address the
absence of a supporting dataset for this task, we pro-
pose PhysAssets, a dataset of 50K+ 3D assets anno-
tated with physical properties and corresponding refer-
ence videos.
Experiments show that PhysGM produces
high-fidelity 4D simulations from a single image in one
minute, achieving a significant speedup over prior work
while delivering realistic renderings. Our project page is
at:https://hihixiaolv.github.io/PhysGM.github.io/
1
arXiv:2508.13911v4  [cs.CV]  17 Mar 2026

<!-- page 2 -->
1. Introduction
Recent advances in 3D representation, particularly 3D
Gaussian Splatting (3DGS) [20], have revolutionized novel-
view synthesis for static scenes. The next frontier is to im-
bue these representations with dynamic, physically plau-
sible behavior, unlocking applications in virtual reality,
robotics [30], and autonomous systems [67].
However,
creating high-fidelity, physics-based 4D content remains a
significant challenge, often demanding computationally ex-
pensive, per-scene optimization pipelines that are ill-suited
for real-time or large-scale deployment.
Current paradigms for physics-based 4D synthesis are
hampered by a fundamental bottleneck: a reliance on slow,
iterative optimization. The typical workflow involves first
reconstructing a 3DGS model from dense multi-view im-
ages, then manually specifying its physical properties (e.g.,
stiffness, mass) via configuration files [55], and finally run-
ning a physics simulation. This process is not only compu-
tationally prohibitive but also lacks scalability and general-
ization. While recent works have explored learning physi-
cal properties via Score Distillation Sampling (SDS) from
video models [26, 64], which rely on gradient backpropa-
gation through a differentiable physics simulator, incurring
heavy per-scene optimization and poor efficiency; hence,
they do not resolve the core efficiency bottleneck. More-
over, the common practice of naively concatenating a pre-
built 3DGS with a physics module overlooks physical cues
encoded in appearance, causing suboptimal performance.
To face these limitations, we ask a fundamental ques-
tion: Can we bypass per-scene optimization entirely and
instead learn a generative model that produces a com-
plete, physically-grounded 4D simulation in a single, feed-
forward pass? This requires reframing the problem from
one of slow, iterative reconstruction to one of amortized,
feed-forward inference. To this end, we propose PhysGM,
a feed-forward framework that enables optimization-free
synthesis of dynamic 4D scenes from image inputs. Our
key insight is a novel two-stage training paradigm designed
to learn a generalizable physical prior and perceptual re-
alism.
In the first stage, we pre-train our model on a
substantial dataset to jointly predict a 3D Gaussian repre-
sentation and its corresponding physical properties—this
joint optimization not only enhances the accuracy of both
geometry and physics predictions but also establishes a
well generative prior [9, 50]. Critically, it eliminates the
reliance on pre-reconstructed 3DGS that demands multi-
view images and time-consuming per-scene optimization,
while addressing the flaw of overlooking visually embed-
ded physical information caused by the naive concatena-
tion of separate Gaussian and physics prediction modules.
In the second stage, we employ Direct Preference Opti-
mization (DPO) [37] to fine-tune the model. By ranking
generated simulations against ground-truth videos, we cre-
ate preference pairs that effectively guide the model to-
wards producing physically plausible and temporally co-
herent dynamic sequences-unlike SDS-based methods, this
preference-driven approach completely eliminates the need
for both time-consuming per-scene optimization processes
and a differentiable physics engine.
Given the lack of datasets pairing 3D assets with physi-
cal annotations and reference simulations for this task, we
construct and release the PhysAssets Dataset, a substan-
tial benchmark of over 50,000 3D assets, each annotated
with its physical material properties and a corresponding
physically-plausible simulation video, providing a critical
resource for training and evaluating generative 4D models.
Our main contributions are summarized as follows:
• We propose PhysGM, the first feed-forward framework
capable of generating a physically-grounded 4D Gaussian
simulation from an image input in one minute.
• We construct a novel two-stage training paradigm that
combines large-scale supervised pre-training with DPO-
based refinement, enabling the model to learn a physical
prior and then align it with perceptual quality.
• We present the PhysAssets Dataset, a new substantial
benchmark with 50,000+ 3D assets annotated with phys-
ical properties and simulation videos, to spur future re-
search in this challenging domain.
• Extensive experiments show that PhysGM produces high-
fidelity 4D simulations from a single image in one minute,
achieving substantial speedups over SDS-based baselines
while delivering superior visual quality across metrics.
2. Related Work
2.1. 4D Content Generation
Recent advances in 4D content generation—dynamic 3D
scenes—primarily follow two paradigms.
The first dis-
tills knowledge from pre-trained 2D/video diffusion mod-
els into a dynamic 3D representation using Score Distil-
lation Sampling (SDS)[27, 34, 40, 59] or trajectory guid-
ance [48].
To ensure temporal coherence, methods em-
ploy explicit deformation models[49] or multi-view video
supervision [15, 60]. However, the iterative nature of SDS
is computationally intensive and prone to instability. Miti-
gation strategies include pre-synthesizing videos for direct
supervision [33, 40] or training generalizable models to by-
pass per-scene optimization [41, 57].
The second paradigm trains generative models directly
on 3D data [17, 32]. These models have evolved to generate
diverse representations, from point clouds [32] to implicit
functions (SDFs)[17] and explicit 3D Gaussians[46]. Its
primary limitation is the scarcity of large-scale 3D datasets.
Despite impressive visual results, both paradigms are
fundamentally physics-agnostic. They produce temporally
coherent but often physically implausible animations, as
2

<!-- page 3 -->
they lack an explicit model of real-world dynamics.
2.2. Feed-forward 3D Gaussian Models
Parallel to 4D generation, another line of research has fo-
cused on accelerating the creation of static 3D scenes. Tra-
ditional 3DGS reconstruction [20] requires slow, per-scene
optimization. To address this, feed-forward models have
emerged to perform 3D reconstruction from sparse or even
single views in a single pass [24, 44, 46, 53, 56, 61]. Mod-
els like Splatter Image [44] and LGM [46] use U-Net-based
architectures to directly regress Gaussian parameters from
input images, while LVSM [16] outputs only rendered im-
ages instead of 3D Gaussians. Other works focus on scene-
level generation by aggregating features from multiple im-
ages [3, 5, 11, 23, 25, 51, 68].
However, these feed-forward models have been exclu-
sively developed for static scene generation. They lack any
mechanism to represent or predict dynamic, physics-based
behavior, limiting their use to non-interactive applications.
Our work is the first to embed physical reasoning directly
into a feed-forward Gaussian generation framework.
2.3. Physics-Grounded 4D Synthesis
The integration of physics into 4D generation is a nascent
but critical field. The pioneering PhysGaussian [55] first
coupled 3DGS with Material Point Method (MPM) simu-
lations based on warp [31], but its reliance on manual, per-
scene parameter tuning highlighted the need for automation.
Subsequent research has focused on automating this
process.
A dominant approach uses Score Distillation
Sampling (SDS) to distill motion priors from video mod-
els, enabling data-driven optimization of material proper-
ties [12, 28, 45, 64]. Others leverage Large Language Mod-
els (LLMs) to infer parameters from text [1, 4, 29, 66],
though accurately predicting continuous values remains a
challenge. More advanced methods learn neural constitu-
tive models [26].Concurrently, efforts have been made to
construct dedicated physical 3D datasets [2].
However, existing approaches are fundamentally lim-
ited by their reliance on pre-reconstructed models and slow,
per-scene optimization. We propose PhysGM to overcome
these bottlenecks. Our framework enables rapid, end-to-
end 4D synthesis from sparse inputs by learning a physical
prior, unifying efficiency with physical realism.
3. Method
We present PhysGM, a transformer-based physical recon-
struction model that, given the posed RGB images I =
{Ii ∈RH×W ×3}n
i=1, the model predicts in one forward
pass: (1) a set of 3D Gaussians for geometry and appear-
ance, and (2) a vector of physical properties for physical
simulation. We first review the fundamentals of 3D Gaus-
sian Splatting (Sec. 3.1). We then detail our transformer-
based architecture that jointly predicts 3DGS and physics
parameters (Sec. 3.2).
Next, we explain how these pa-
rameters drive a Material Point Method (MPM) simulation
(Sec. 3.3) and introduce our DPO-based strategy for refin-
ing the model with ground-truth dynamics (Sec. 3.4). Fi-
nally, we present PhysAssets, a new dataset created to train
and evaluate our model (Sec. 3.5).
3.1. 3D Gaussian Splatting
The core of 3DGS [20] is to model the scene as a set of N
anisotropic 3D Gaussians. Each Gaussian is parameterized
by: position µ ∈R3, defining its center in world space;
covariance Σ ∈R3×3 defining its shape and orientation;
opacity α ∈[0, 1] controlling its transparency; color rep-
resented by Spherical Harmonics (SH) coefficients c that
enable modeling of view-dependent appearance effects.
The covariance matrix can be computed with a rota-
tion matrix Rmat and a diagonal scaling matrix S: Σ =
RmatSST Rmat
T , where S = diag(s). During rendering,
these 3D Gaussians are projected onto the 2D image plane
for a given camera view. The final color C for a pixel is
computed by alpha-blending the N Gaussians that overlap
the pixel with its αi and color ci . The objective of all pa-
rameters ψ = {(µi, qi, si, αi, ci)}N
i=1 is to minimize the
difference between the rendered and ground-truth images.
3.2. Model Architecture
We propose a transformer-based architecture PhysGM,
designed to jointly regress 3DGS parameters and physi-
cal attributes from posed images.
Our framework com-
prises three main components: multi-modality encoding, a
transformer-based backbone, and decoders for 3DGS and
physics properties. An overview is shown in Figure 2.
Multi-modality tokenization
We adopt DINOv3 [42] as
image encoder Eimg. Each input image Ii is patchified into
tokens, which are then projected by a linear layer to pro-
duce high-dimensional image features.
To explicitly en-
code camera geometry, we represent the principal ray of
each pixel for view Ii using Pl¨ucker ray coordinates Ci. We
process Ci by a dense representation encoder [65] Ecam to
produce geometry tokens. The resulting camera tokens are
then concatenated with corresponding image tokens. The
concatenated tokens (ti)N
i=1 are formed by concatenating
the corresponding image and camera tokens. We further
prepend three learnable global tokens g1, g2, g3 to the se-
quence. These tokens aggregate global scene information
and will be used for physics prediction. The concatenated
tokens ti and input tokens Tin are thus formed as:
ti = concat(Eimg(Ii), Ecam(Ci)),
(1)
Tin = (ti)N
i=1 ∪gk, k = 1, 2, 3.
(2)
3

<!-- page 4 -->
DPO
Single Image
MV Adapter
Multi-view Images 
and Cameras
Physical Gaussian Model
Phys head
DPT head
Young's modulus: 𝐄
Poisson's ratio: 𝝂
Material class: C
MPM
Probability Modeling
Rendered Videos
GT Video
Mean: 𝝁
Scale: 𝐬
Opacity: 𝛼
Rotation: 𝐑
SH: 𝐜
Sampling
NLL Loss
MSE Loss
Optional
Forward Pass
Gradient Flow
𝟏×𝐇×𝐖×𝟑
𝟒×𝐇×𝐖×(𝟑+ 𝟔)
(𝟒×𝒉×𝒘)×𝐂
R
R
Rasterization
R
+
Self-Attention
MLP
+
Transformer Blocks*n
𝑬!"#
𝑬$%"
Figure 2. Pipeline of PhysGM. The model conditions on one or four input views and their corresponding camera parameters, which are
processed by a transformer-based model to produce output tokens. These tokens then decoded by two parallel heads: (1) a DPT Head
predicting the initial 3D Gaussian scene parameters ψ, and (2) a Physics Head that predicts a distribution over the object’s physical
properties θ. The sampled parameters (ψ, θ) initialize a Material Point Method (MPM) simulator to generate the final dynamic sequence.
The entire architecture is trained in a two-stage paradigm: first, supervised pre-training on ground-truth data establishes a well generative
prior. Subsequently, a DPO-based fine-tuning stage uses the ranks against a ground-truth video and aligns the model with physically
plausible results.
During inference with a single input image,
we
use MVAdapter [13] to synthesize three fixed auxiliary
views—rear, left, and right—while the original image
serves as the frontal view.
Transformer Backbone
The complete sequence of input
tokens Tin is fed into our 24-layer transformer backbone to
learn the contextualized representations. We collect the out-
put tokens from the intermediate layers, which provide a
multi-scale representation of the scene, capturing both high-
level semantics and low-level details.
Prediction heads
We employ two distinct, specialized
heads to map the backbone’s output tokens to 3DGS pa-
rameters and physical attributes.
To predict the 3DGS representation, we adopt a Dense
Prediction Transformer (DPT) [38, 58] head fgs. This head
takes the multi-scale features from the backbone and pro-
gressively upsamples them through a series of refinement
stages. For each input view, it outputs per-pixel maps corre-
sponding to the 3DGS parameters ψi = fgs(ti). The Gaus-
sians predicted from all views are then aggregated to form
the final, coherent 3D scene representation. During pre-
training, we optimize the network by jointly minimizing
the MSE, alpha and LPIPS [63] losses between rendered
images and the ground-truth views.
We predict three physical attributes that determine ki-
netic behavior of the particles: a material class C, Young’s
modulus E (stiffness) and Poisson’s ratio ν (compressibil-
ity). We categorize materials into Nc classes, where each
class corresponds to a specific constitutive model used in
the subsequent MPM simulation. From the three global to-
kens gk, we predict these properties using two specialized
heads: A classification head fmaterial to determine the mate-
rial class; two regression heads fphys predicting the distri-
bution for the continuous physical properties E, ν respec-
tively.
Specifically, fphys outputs the mean µθ and log-
variance log σ2
θ for these properties:
(µθ, log σ2
θ) = fphys(gk).
(3)
This defines a conditional probability distribution over the
physical properties, allowing us to model the inherent un-
certainty in estimating physics from visual data:
P(θ|I) = N(θ|µθ, diag(σ2
θ)).
(4)
4

<!-- page 5 -->
At inference time, we sample from these learned distribu-
tions to obtain the scene’s parameters: θsampled ∼P(θ|I).
This probabilistic formulation is crucial, as it allows the
model to generate diverse physical parameters, enabling the
subsequent preference-based refinement with GT videos.
GT Video
Rendered Videos
SAM 2 + CoTracker 3
L2 Distance
Affine Transformation
Figure 3. Preference calculation. We use SAM-2 [39] for segmen-
tation and CoTracker-3 for trajectory extraction across the GT and
simulated videos. The extracted point tracks quantify the fidelity
of each candidate to the GT, yielding a ranked preference tuple.
3.3. Physics-based Dynamics via MPM
To simulate physics-based dynamics, we employ the Mate-
rial Point Method (MPM) [14, 43], a hybrid Lagrangian-
Eulerian approach that discretizes an object into a set of
material points. Each point p carries its own physical state,
including mass mp, position xp, velocity vp, and the affine
velocity matrix Cp. Crucially, it also tracks the deforma-
tion gradient Fp, which maps vectors from the material’s
rest configuration to its current deformed state.
The simulation evolves these states through a two-step
process at each time step ∆t. First, in the Particle-to-Grid
(P2G) transfer, particle properties are mapped to a back-
ground Eulerian grid. The mass mi and momentum pi at
each grid node i are computed via weighted summation:
mi =
X
p
mpN(xi −xp),
(5)
pi =
X
p
mp(vp + Cp(xi −xp))N(xi −xp),
(6)
where N(·) is a B-spline interpolation kernel. On the grid,
internal forces (derived from Fp and a constitutive model)
and external forces (e.g., gravity) are computed and used to
update grid velocities.
Second, in the Grid-to-Particle (G2P) transfer, the up-
dated grid velocity field is interpolated back to update the
particle states. The particle’s velocity and deformation gra-
dient are updated as follows:
vn+1
p
=
X
i
pn+1
i
mi
N(xi −xp),
(7)
Fn+1
p
=
 
I + ∆t
X
i
pn+1
i
mi
∇N(xi −xp)T
!
Fn
p.
(8)
The particle’s final position is xn+1
p
= xn
p + ∆t vn+1
p
.
Following PhysGaussian [55], we directly couple this
physical simulation with our 3D Gaussian representation.
The MPM simulation is driven by the physical properties
predicted by our model.
We establish a one-to-one cor-
respondence between each material point and a Gaussian
primitive. The updated particle position xp directly defines
the Gaussian’s mean µ. More importantly, the deformation
gradient Fp dictates the Gaussian’s anisotropic shape and
orientation. We perform a polar decomposition on Fp to
factor it into a rotation matrix Rp and a symmetric positive
semi-definite stretch tensor Sp, Fp = RpSp. The resulting
rotation Rp and the diagonal elements of the stretch tensor
Sp are then used to define the Gaussian’s rotation matrix
Rmat and scaling vector s, respectively. These are used
to construct the covariance matrix Σ, ensuring the rendered
geometry mirrors the physically simulated deformation.
3.4. Preference-based Fine-tuning with DPO
While supervised pre-training provides a robust physical
prior, it may not fully capture the subtle dynamics that lead
to high perceptual quality. To bridge this gap, we intro-
duce a fine-tuning stage using Direct Preference Optimiza-
tion (DPO) [37]. This approach enables refining our gen-
erative model using feedback from physics simulation and
rendering pipeline without requiring differentiability.
We treat the pre-trained model as a fixed reference pol-
icy, πref. The model being optimized, πω, is then refined
using a dataset of preference pairs D = {(z, ϕw, ϕl)}. For
each scene context z, we perform the following steps to
create a preference pair. First, we draw a set of K can-
didate parameter vectors {ϕ1, . . . , ϕK} from our current
policy, where ϕk ∼πω(·|z). Each ϕ consists of the pre-
dicted physical properties θ and fixed appearance parame-
ters ψ. For each candidate ϕk, we run the MPM simulation
and render the resulting 3D Gaussian sequence to produce a
short video clip Vk. We compare each generated clip Vk to
the ground-truth video Vgt using a perceptual distance met-
ric d(Vk, Vgt).The parameter set ϕk that yields the lowest
perceptual distance is designated the “winner” ϕw. Another
sample with the highest distance is chosen as the “loser” ϕl.
5

<!-- page 6 -->
Material : “Ceramics”
E : 2.0e7     nu : 0.2637
Material : “Jelly”
E : 2.0e4     nu : 0.2434
Material : “Stone”
E : 2.0e7     nu : 0.1547
Material : “Snow”
E : 8.0e4
nu : 0.2579
Material : “Plasticine”
E : 2.3e5  
nu : 0.3112
Image
Simulated & Rendered videos
Predicted Physical Property
Figure 4. Qualitative results by PhysGM. For different objects, we show the single input image (left), keyframes from the resulting
physically-plausible simulation (middle), and the physical properties predicted by our model (right). Our method generates these high-
fidelity 4D sequences in under one minute from a single view, without any per-scene optimization.
These dynamically generated preference pairs are used
to fine-tune πω by minimizing the DPO loss. This objective
directly increases the likelihood of the model generating the
“winner” parameters while decreasing the likelihood of the
“loser” parameters, relative to the reference policy:
LDPO(πω, πref) = −E(z,ϕw,ϕl)∼D[log σ(p1 −p2)],
(9)
p1 = β log πω(ϕw|z)
πref(ϕw|z), p2 = β log πω(ϕl|z)
πref(ϕl|z),
(10)
where β is a temperature parameter controlling the opti-
mization strength. By optimizing this objective, we steer the
learned distribution towards regions of the parameter space
that produce perceptually and physically superior simula-
tions, without the need for optimization upon complex dif-
ferentiable MPM and rendering process.
3.5. PhysAssets Dataset
We introduce PhysAssets, a large dataset of over 50,000
3D objects aggregated from established datasets, includ-
ing Objaverse [8], OmniObject3D [54], ABO [6] and
HSSD [21]. To annotate the physical attributes, we devel-
oped a pipeline leveraging a large Multimodal Large Lan-
guage Model (MLLM) Qwen3VL [47]. For each object, we
first obtain its corresponding material category directly us-
ing a well-designed prompt with multi-view images. Con-
currently, we extract the physical property parameters of the
object through predefined textual descriptions of Young’s
modulus and Poisson’s ratio. For each object, we generate
ground-truth (GT) videos using Framepack [62], a physics-
based simulation framework. The simulation is conditioned
on text instructions that specify the detailed physical prop-
erties of the object, ensuring realistic dynamic behaviors.
To enable DPO training, we automate preference label-
ing d(Vk, Vgt) via a comparative analysis pipeline, as shown
in Figure 3. This involves using SAM-2 [22, 39] for in-
stance segmentation and CoTracker-3 [18, 19] for trajec-
tory extraction across the GT and simulated videos. The
extracted point tracks then quantify the fidelity of each can-
didate relative to the GT, yielding a ranked preference tu-
ple. Comprehensive details on the composition of the data
set, video generation pipeline, and the preference calcula-
tion process are provided in the supplementary materials.
4. Experiments
In this section, we conduct a comprehensive comparison
against several baseline methods, evaluating the effective-
ness and efficiency of our model. Furthermore, we perform
ablation studies on our two-stage training strategy.
6

<!-- page 7 -->
4.1. Experimental settings
Dataset
We train our model on our newly created
PhysAssets Dataset. For qualitative visualizations, we show
results on the test set from our dataset as well as on in-the-
wild images to demonstrate generalization.
Baselines
We compare our method against two base-
line models in physics-based dynamic generation: Omni-
PhysGS [26] and DreamerPhysics [12].
These methods
represent the paradigm of per-scene optimization, where
physical properties are learned by distilling knowledge from
video models using Score Distillation Sampling (SDS).
These methods provide strong benchmarks for simulation
quality, albeit at a significant computational cost.
Simulation Process
The physical properties predicted by
our model directly drive the subsequent Material Point
Method (MPM) simulation. The predicted material class
determines which constitutive model is employed for the
object’s dynamics. For instance, a prediction of ‘rubber’
would select a Neo-Hookean constitutive model. The pre-
dicted Young’s modulus (E) and Poisson’s ratio (ν) then
serve as the specific material parameters for this chosen
model. We use a fixed sub-step time of 2 × 10−5s and a
frame time of 4×10−2s, generating 50 frames per sequence.
Additional simulation parameters, such as boundary condi-
tions, are detailed in the Appendix.
Training Details
Our model is trained from scratch and
leverages FlashAttention v2 [7] for efficient computation,
with training conducted on 32 NVIDIA A800 GPUs for
3 days and a batch size of 8 per GPU. The training pro-
cess consists of two stages: in the pre-training phase, we
jointly optimize the model for physics property prediction
and Gaussian parameter prediction under the supervision of
the PhysAssets dataset; In the fine-tuning phase, we employ
Direct Preference Optimization (DPO) to further enhance
the physical realism and temporal coherence of the gener-
ated dynamics, where the model’s backbone remains frozen
and only the predictive heads responsible for physics prop-
erties are fine-tuned.
Evaluation Metrics
We conduct a comprehensive eval-
uation using two complementary metrics:
an objective,
model-based score and a subjective, human-based assess-
ment. (1) CLIPsim [36] quantifies the semantic similarity
between the rendered visual outcomes and their correspond-
ing textual descriptions of the physical phenomena.
(2)
User Preference Rate (UPR). We conduct a four-alternative
forced-choice user study; UPR is the percentage of trials
in which a method is selected as the most realistic or visu-
ally appealing. More details are provided in the appendix.
(3) We evaluate reconstruction quality with three metrics:
PSNR measures pixel-level fidelity via the log ratio of peak
signal to MSE (higher is better); SSIM [52] assesses struc-
Figure 5. Other Results by PhysGM. PhysGM can demonstrate ro-
bust generalization to diverse physical interactions. It accurately
simulates complex deformations like stretching and twisting, han-
dles multi-object dynamics with varied materials, and processes
real-world data, highlighting its extensibility to novel scenarios.
tural similarity (0-1, higher is better); and LPIPS [63] com-
putes learned perceptual distance (lower is better).
4.2. Results and Analysis
In this section, we empirically validate the effectiveness of
PhysGM. We conduct a comprehensive comparison against
baseline methods, and a targeted ablation study to isolate
the contributions of our two-stage training paradigm.
Comparison with baseline methods
We demonstrate the
core capability of optimization-free 4D generation in Fig-
ure 4. From a single input image, our pipeline generates
a 3D Gaussian representation with physical properties via
one feed-forward pass, enabling physics-based simulation
in under one minute. Figure 5 present its performance on a
variety of diverse and challenging scenarios in Figure 5.
We evaluated PhysGM against two optimization-based
baselines, OmniPhysGS and DreamerPhysics, under identi-
cal settings for a fair comparison. Our model used 4-view
images as input, while the baselines were initialized with
Gaussian splats generated by our model. All physical pa-
7

<!-- page 8 -->
Table 1. Quantitative comparisons. We evaluate our method and baseline models on 5 different material types. Evaluation is based on the
CLIPsim score (higher is better ↑) and UPR (higher is better ↑).
Method
metal
jelly
plasticine
snow
sand
average
CLIPsim UPR
CLIPsim UPR
CLIPsim UPR
CLIPsim UPR
CLIPsim UPR
CLIPsim UPR
OmniPhysGS [26]
0.2149
5%
0.2291
12%
0.2135
8%
0.1834
9%
0.2047
16%
0.2091
10%
DreamPhysics [12]
0.2273
16%
0.2459
11%
0.2437
23%
0.2071
18%
0.2217
18%
0.2291
17.2%
PhysGM (w/o DPO)
0.2698
30%
0.2700
33%
0.2547
31%
0.2541
26%
0.2980
30%
0.2693
30%
PhysGM (w/ DPO)
0.2732
49%
0.2774
44%
0.2691
38%
0.2548
47%
0.2997
36%
0.2748
42.8%
rameters were kept consistent, except for those each base-
line is designed to optimize. Both qualitative results in Fig-
ure 6 and quantitative metrics in Table 1 show that PhysGM
outperforms the baselines across a diverse range of materi-
als. This proves our feed-forward approach does not trade
quality for speed; by learning a robust physical prior, it sur-
passes the perceptual realism and physical plausibility of
slower, per-scene optimization techniques.
Ours
Dream
Physics
Omni
PhysGS
Figure 6. Qualitative comparisons. We selected two distinct phys-
ical materials for visual comparison.
Table 2. Quantitative comparisons for multi-view synthesis on
GSO dataset. We matched the baseline settings by comparing with
LGM and GS-LRM, We achieve better results while using only
10% of the data compared to the GS-LRM.
Methods
Res. PSNR ↑SSIM ↑LPIPS ↓
LGM [46]
256
21.44
0.832
0.122
Our
256
25.47
0.916
0.071
GS-LRM [61]
512
30.52
0.952
0.050
Ours
512
28.95
0.953
0.039
Additionally, we validate MVS effectiveness in Table 2
by comparing against Gaussian-based reconstruction meth-
ods. Qualitative results are in the appendix. Note that our
model is trained only at 512 resolution.
Ablation Study
To validate the effectiveness of our two-
stage training strategy, we conduct an ablation focused on
the DPO fine-tuning stage. We compare the full PhysGM
against a baseline variant (“w/o DPO”), which is trained
only with supervised pre-train, also against prior methods.
As shown in Table 3 and Figure 7, the DPO stage is
critical, the full model achieves consistently higher scores
across all categories and both metrics. In effect, DPO con-
verts a statistically sound prior into a perceptually supe-
rior generator by leveraging feedback from the full, non-
differentiable simulation pipeline.
Compared with other methods, PhysGM attains bet-
ter or comparable visual and physical fidelity at substan-
tially lower optimization cost. Inference requires no per-
scene optimization—only a single forward pass followed by
MPM rollout—enabling end-to-end generation in under one
minute, while competing approaches typically rely on iter-
ative, scene-specific optimization. This yields a markedly
better quality–time trade-off.
Table 3. Comparison with state-of-the-art methods. It can be ob-
served that DPO achieves superior performance in generalization,
inference time, and simulation quality.
Methods
Training
Gen.
Infer time
CLIPsim
OmniPhysGS [26]
SDS
×
>12h
0.2091
DreamPhysics [12]
SDS
×
>0.5h
0.2291
Ours
DPO
✓
<1min
0.2748
5. Discussion
The prohibitive computational cost of the MPM remains the
primary obstacle to its use in large-scale, real-time applica-
tions, a challenge exacerbated by the lack of viable alterna-
tives for complex physics like fluid and fracture. Critically,
the persistent sim-to-real gap—stemming from inherent dis-
crepancies between synthetic training data and physical re-
ality, including simplified constitutive models—hinders ro-
bust real-world deployment and limits generalization capa-
bilities. Future work must therefore prioritize two goals:
developing more efficient simulation frameworks and bridg-
ing the sim-to-real gap.
8

<!-- page 9 -->
Material : “Stone”
E : 2.0e4     nu : 0.2212
w/o DPO
Material : “Stone”
E : 2.0e7     nu : 0.1547
w/ DPO
Material : “Metal”
E : 2.0e4     nu : 0.2108
w/o DPO
Material : “Metal”
E : 3.3e6     nu : 0.2563
w/ DPO
Gen-Video
"An object made of stone falls 
down onto the invisible ground."
Gen-Video
"An object made of metal falls 
down onto the invisible ground."
Figure 7. Ablation results of DPO. The results indicate that after the two-stage DPO training, the model predicts physical attributes with
greater accuracy, enabling the generation of 4D videos that exhibit higher physical fidelity.
6. Conclusion
We presented PhysGM, a feed-forward framework for
rapid, physically grounded 4D synthesis from sparse in-
puts.
Our model is first optimized for 3D Gaussian re-
construction and physical properties prediction, then fine-
tuned with Direct Preference Optimization (DPO) to learn
from a non-differentiable simulator, eliminating the need
for per-scene optimization. Empirical analysis reveals our
approach yields physically realistic simulation and render-
ing in under a minute. PhysGM’s efficiency paves the way
for scalable applications in embodied AI, autonomous driv-
ing, and interactive virtual reality.
Appendix
A. More Details on Implementation
A.1. Compare with Other Methods
Table
4
provides
a
qualitative
comparison
between
our method,
PhysGM, and other state-of-the-art ap-
proaches [12, 26, 35, 40, 55, 64, 66]. We evaluate each
method across five critical dimensions: two concerning
input requirements (the need for pre-optimized 3D Gaus-
sians or pre-defined physical parameter) and three concern-
ing core capabilities (generalizability, independence from
Large Language Models, and inference speed). The com-
parison highlights that our approach is the only one to oper-
ate without these stringent prerequisites. PhysGM simulta-
neously achieves strong generalization and maintains a very
short inference time of under 30 seconds.
A.2. Simulation Details
This section elaborates on the key parameters used to con-
figure our Material Point Method (MPM) simulations, as
referenced in the main text. The configuration is detailed
below, categorized by function.
MPM Grid Resolution
The simulation domain is dis-
cretized into a background grid of 200 * 200 * 200 cells.
This grid is fundamental to the MPM algorithm for comput-
ing particle interactions and mapping data between particles
and the grid.
9

<!-- page 10 -->
Table 4. Comparison with state-of-the-art methods, highlighting PhysGM’s unique advantages. Unlike prior work, our method eliminates
the need for both pre-optimized 3D Gaussian and pre-defined physical parameters. This allows it to achieve strong generalization while
maintaining a significantly shorter inference time (< 30s). “only E” represents that only Young’s modulus is automatically predicted, “only
material” represents that only material is automatically predicted.
Method
No Pre-opt.
3D Gaussians
Auto Param
Computation
Generalizable
Without LLM
Inference Time
PhysGaussian [55]
×
×
×
✓
-
DreamPhysics [12]
×
only E
×
✓
>0.5h
PhysDreamer [64]
×
only E
×
✓
>1h
OmniPhysGS [26]
×
only material
×
✓
>12h
DreamGaussian4D [40]
✓
×
✓
×
6.5min
Feature Splatting [35]
×
×
×
✓
>1h
PhysSplat [66]
×
✓
✓
×
<2min
PhysGM (Ours)
✓
✓
✓
✓
<30s
Camera Position
For different objects, the camera is ini-
tialized at an azimuth of -45 or 135 degrees, an elevation of
0 degrees, and a radius of 1.8 or 1.3 units.
Camera Motion
The camera is configured to be static
during the simulation.
Other Parameters
Gravity is applied in the falling scene,
and force in the corresponding direction is applied in the
collision scene.
A.3. Training and Evaluation Details
Network Architecture.
We employ DINOv3 [42] (ViT-
L/16) pre-trained on LVD-1689M as our image encoder,
producing 1024-dimensional features.
The transformer
backbone consists of 24 layers with a hidden dimension
of 1024 and attention head dimension of 64.
We use a
patch size of 16 and incorporate 3 learnable global tokens
for physics prediction. For 3D Gaussian representation, we
set the spherical harmonics degree to 0, with near and far
planes at 0.001 and 2.0, respectively.
Training Configuration.
We train our model on 32
NVIDIA A800 GPUs using a two-stage process for about 3
days in total with a batch size of 8 per GPU. The base learn-
ing rate is set to 2∗10−4 with AdamW optimizer (β1 = 0.9,
β2 = 0.95, weight decay = 0.05). We employ a cosine
learning rate schedule with 5K warmup steps and clip gradi-
ents to a maximum norm of 10.0. Mixed precision training
is enabled using bfloat16 to accelerate computation.
Data Configuration.
During training, input images are
resized to 512 × 512 resolution with square cropping. Each
training sample consists of 4 input views and 8 target views,
where target views include the input views for consistency.
We use 8 workers for data loading with a prefetch factor of
128 to ensure efficient GPU utilization.
Evaluation Protocol.
We evaluate on the complete GSO
dataset [10] containing 1,009 objects. For each object, we
render 32 views with 4 elevation angles (0, 20, 40, 60) and 8
azimuthal angles. During testing, we sample fixed 4 views
as input and evaluate reconstruction quality on 8 randomly
selected novel views. We report PSNR, SSIM [52], and
LPIPS [63] averaged across all test views and objects.
A.4. User Preference Evaluation
To complement quantitative metrics with human perception
assessment, we conduct a user study to evaluate the percep-
tual quality and physical plausibility of generated 4D se-
quences across different methods.
Study Design.
We employ a Four-Alternative Forced
Choice (4AFC) protocol, where participants are presented
with four videos simultaneously showing the same object
simulated by different methods: our PhysGM, and three
baseline methods (PhysGaussian [55], OmniPhysGS [26],
and DreamGaussian4D [40]). The videos are displayed in
randomized positions to eliminate order bias. Participants
are instructed to select the single video that exhibits the
most realistic physical behavior and visual quality, consid-
ering factors such as motion naturalness, material response,
temporal coherence, and rendering fidelity.
Stimuli and Sampling.
We carefully select 5 representa-
tive test scenes spanning diverse object categories and phys-
ical scenarios (dropping and stretching). For each scene,
we generate 4D sequences using all methods with identical
input views and physical interaction setups to ensure fair
comparison.
10

<!-- page 11 -->
Participants and Procedure.
We recruited 103 partici-
pants comprising graduate students and researchers with
backgrounds in computer graphics, computer vision, or re-
lated fields.
Each participant completed a questionnaire
containing 5 comparison trials (one per test scene). Before
the formal study, participants underwent a training phase
with two practice trials to familiarize themselves with the
task and interface. Participants could replay videos multi-
ple times before making their selection and were allowed to
take breaks between trials. The entire study took approxi-
mately 10 minutes per participant.
Data Validation and Filtering.
To ensure data quality,
we implemented several validation mechanisms:
• Attention checks: Two control trials with obvious qual-
ity differences were inserted to identify inattentive partic-
ipants.
• Completion time: Responses completed too quickly (<5
seconds per trial) were flagged.
• Response consistency: Participants showing random se-
lection patterns were identified via entropy analysis.
After applying these criteria, we excluded 3 invalid re-
sponses (2.9% exclusion rate) due to failed attention checks
or suspiciously short completion times, resulting in 100
valid responses for analysis.
User Preference Rate (UPR).
We define the User Prefer-
ence Rate as the percentage of participants who selected a
given method as the most realistic:
UPRm =
1
S · N
S
X
s=1
N
X
i=1
1[choices,i = m] × 100%
(11)
where m denotes the method, S is the number of test scenes,
N = 100 is the number of valid participants, and 1[·] is the
indicator function. A higher UPR indicates stronger human
preference. Under random chance, each method would re-
ceive 25% preference rate in a 4AFC setup.
B. Material Constitutive Models
In continuum mechanics and physics-based simulation, a
constitutive model (or constitutive equation) is a fundamen-
tal mathematical relationship that describes how a mate-
rial responds to external stimuli.
Specifically, it defines
the relationship between the internal forces (stress) and the
material’s deformation (strain). The choice of a constitu-
tive model is critical as it dictates the material’s behav-
ior—whether it behaves as a rigid solid, an elastic solid, a
fluid, or a hyperelastic material like rubber.Our simulation
framework employs different constitutive models based on
the predicted material class. This allows us to capture a di-
verse range of dynamic behaviors.
B.1. The Neo-Hookean Model
For materials predicted to be “jelly” or other soft, rubber-
like substances, we employ the Neo-Hookean model. This
is a classic hyperelastic model, meaning its stress-response
is derived from a strain energy density function. It is ideal
for capturing large, nonlinear deformations while remain-
ing computationally efficient, making it a staple in computer
graphics and simulation. The model’s formulation is based
on the statistical mechanics of polymer chains, which ac-
curately describes the behavior of materials like rubber that
can stretch significantly without permanent deformation.
The core idea is to split the material’s response into two
parts: a part that resists changes in shape (deviatoric) and
a part that resists changes in volume (volumetric). This al-
lows for a robust simulation of compressible, soft-bodied
dynamics.
The model defines the Kirchhoff stress (τ),
which is a measure of internal force suitable for large-
deformation analysis. The Kirchhoff stress τ for a com-
pressible Neo-Hookean material is given by:
τ = µ ∗J−2/3 ∗dev(B) + (λ/2) ∗(J2 −1) ∗I,
(12)
where τ is the Kirchhoff stress tensor. B = FFT is the
left Cauchy-Green deformation tensor, where F is the de-
formation gradient and dev(B) is the deviatoric (volume-
preserving) part of B. J = det(F) is the determinant of the
deformation gradient, representing the volume change. µ
and λ are the Lam´e parameters, which characterize the ma-
terial’s stiffness. They are derived from the Young’s modu-
lus (E) and Poisson’s ratio (ν) predicted by our model.
B.2. The Fixed Corotational Constitutive Model
For materials predicted to be “metal” or other similarly stiff
elastic solids, we employ the Fixed Corotational (FCR) con-
stitutive model. This model is particularly well-suited for
scenarios where a material undergoes large rigid-body mo-
tions (i.e., translation and rotation) but experiences only
small elastic deformations. The core principle of any coro-
tational model is to decouple the object’s overall rotation
from its internal strain. The FCR model begins with the po-
lar decomposition of the deformation gradient F = RS,
where R is a pure rotation matrix, and S is the right
stretch tensor, which is symmetric and positive definite. The
model defines a linear relationship between the First Piola-
Kirchhoff stress (P) and a measure of strain. The First Piola-
Kirchhoff stress is energetically conjugate to the deforma-
tion gradient F and is given by:
P = 2µ(F −R) + λ(J −1)J(F−T ),
(13)
where P is the First Piola-Kirchhoff stress tensor. For force
calculations within our MPM simulation, we use the Kirch-
hoff stress (τ). The relationship between Kirchhoff stress
and the First Piola-Kirchhoff stress is: τ = PFT .
11

<!-- page 12 -->
B.3. The Drucker-Prager Plasticity Model
For materials exhibiting both frictional and cohesive prop-
erties, such as sand, snow, and plasticine, we employ the
Drucker-Prager elastoplasticity model. This model is ideal
for materials whose strength is dependent on the hydrostatic
pressure they are under (e.g., sand becomes stronger when
compressed). It defines a yield criterion, which is a surface
in stress space that separates elastic (temporary) deforma-
tion from plastic (permanent) deformation. The core of the
model is the predictor-corrector algorithm, also known as
return mapping: First, the model assumes the material be-
haves purely elastically during a time step and calculates
a “trial stress”. It then checks if this trial stress lies out-
side the Drucker-Prager yield surface.
If the trial stress
is outside the surface (i.e., the material has yielded), the
stress is mathematically projected back onto the yield sur-
face. This correction step accounts for the plastic flow and
ensures the material’s stress state remains physically plausi-
ble. The Drucker-Prager yield criterion defines the bound-
ary between elastic and plastic states. The yield function is
given by:
f(τ) = ||dev(τ)|| + α ∗tr(τ) −k ≤0,
(14)
where τ is the Kirchhoff stress tensor. dev(τ) is the devia-
toric part of the stress, representing shear. ||dev(τ)|| is the
Frobenius norm of the deviatoric stress, measuring the mag-
nitude of the shear stress. tr(τ) is the trace of the stress, pro-
portional to the hydrostatic pressure (positive for tension,
negative for compression). α is a dimensionless friction pa-
rameter, controlling how much the material’s strength in-
creases with pressure. k is the cohesion of the material,
representing its intrinsic shear strength at zero pressure.
The key insight is that different materials like sand,
snow, and plasticine can be simulated with the same un-
derlying model by simply adjusting the cohesion (k) and
friction (α) parameters.
For instance: Sand (k = 0.0)
has negligible cohesion; its strength comes almost entirely
from inter-particle friction.
Snow (k = 1000.0) repre-
sents an intermediate case with some cohesion. Plasticine
(k = 5000.0) has significant cohesion, allowing it to hold
its shape even without compressive pressure.
C. PhysAssets Dataset Statistics
C.1. Dataset Composition
PhysAssets comprises a comprehensive collection of 3D as-
sets with annotated physical properties. The dataset consists
of two main components: a training set containing 49,206
objects aggregated from multiple public repositories (Obja-
verse [8], OmniObject3D [54], ABO [6], and HSSD [21]),
and a held-out test set of 1,009 objects from the Google
Scanned Objects (GSO) dataset [10], totaling 50,215 anno-
tated 3D objects. The primary objective of this effort was
Table 5. Material distribution in PhysAssets dataset. The 14 pri-
mary materials account for 97% of the dataset, while 32 rare ma-
terials provide additional diversity.
Rank
Material
Count
Percentage
1
Plastic
13,696
27.3%
2
Wood
8,443
16.8%
3
Metal
7,353
14.6%
4
Fabric
7,255
14.5%
5
Ceramic
3,023
6.0%
6
Stone
2,135
4.3%
7
Paper
1,432
2.9%
8
Leather
1,132
2.3%
9
Glass
955
1.9%
10
Rubber
687
1.4%
11
Foam
168
0.3%
12
Snow
147
0.3%
13
Sand
58
0.1%
14
Other (32 materials)
1,731
3.4%
to create a comprehensive, diverse, and standardized collec-
tion of Physical-based assets annotated with 20+ views ren-
dered images, physical properties, and corresponding guid-
ing videos.
C.2. Material and Physical Property Distribution
The dataset exhibits rich material diversity, covering 46 dis-
tinct material categories. Among these, 14 primary mate-
rials constitute the majority of the dataset, while 32 addi-
tional rare materials provide coverage for specialized phys-
ical scenarios. Table 5 presents the distribution of the 14
primary materials.
The most represented material is Plastic, with 13,696
samples (27.3%), reflecting its prevalence in manufactured
objects. Wood constitutes the second largest category with
8,443 samples (16.8%), followed by Metal with 7,353 sam-
ples (14.6%) and Fabric with 7,255 samples (14.5%). Ce-
ramic objects account for 3,023 samples (6.0%). Medium-
frequency materials include Stone (2,135 samples, 4.3%),
Paper (1,432 samples, 2.9%), Leather (1,132 samples,
2.3%), Glass (955 samples, 1.9%), and Rubber (687 sam-
ples, 1.4%). Low-frequency but physically interesting ma-
terials comprise Foam (168 samples, 0.3%), Snow (147
samples, 0.3%), and Sand (58 samples, 0.1%). The remain-
ing 32 rare materials collectively account for approximately
3.0% of the dataset, providing diversity for edge cases and
specialized physical behaviors.
This heterogeneous material distribution enables our
model to learn a comprehensive physical prior spanning
rigid bodies (metal, stone), deformable materials (rubber,
foam), granular substances (sand, snow), and everyday ma-
terials (plastic, wood, fabric). The long-tail distribution also
facilitates studying generalization to rare material types.
12

<!-- page 13 -->
Young’s Modulus (E): Measures material stiffness, rang-
ing from soft materials (103 Pa) to rigid materials (4 ∗1011
Pa). The dataset contains 10 distinct values spanning this
range.
Poisson’s Ratio (ν): Characterizes material compress-
ibility, typically ranging from 0.01 to 0.49. The dataset in-
cludes 10 representative values covering common material
behaviors.
C.3. Source Datasets
Our dataset aggregates models from the following four
sources, each contributing unique characteristics:
OmniObject3D
A high-fidelity dataset featuring approx-
imately 6,000 real-world scanned objects across 190 com-
mon categories (e.g., cups, chairs, animal models). It pro-
vides rich multi-modal data, including textured meshes with
millimeter-level geometric accuracy and multi-view ren-
dered images. For our purposes, we primarily leveraged its
high-resolution rendered views (e.g., the 24-view set with
associated camera parameters) to extract detailed appear-
ance and geometric information.
HSSD Dataset.
The Habitat Synthetic Scenes Dataset
(HSSD) [21], contains over 18,000 high-quality indoor
scenes with photorealistic rendering and detailed seman-
tic annotations. The dataset features diverse residential and
commercial environments with realistic layouts and furnish-
ings.
Amazon Berkeley Objects (ABO)
ABO offers a collec-
tion of approximately 8,000 high-quality, industry-standard
3D models covering 98 everyday object categories. The
data includes textured CAD models (.obj/.glb), which we
utilized to generate consistent multi-view renderings that
align with our standardized format.
Objaverse
A 10M+ dataset containing millions of 3D
objects, Objaverse offers unparalleled diversity in object
shape, category, and style. We selected a substantial sub-
set from this collection to significantly broaden the scope
and variety of our final dataset, as detailed in the following
section.
C.4. Data Processing
Filter and Render
To ensure quality and consistency
across the heterogeneous source datasets, we established
a systematic data curation and processing pipeline.
The
Objaverse dataset, while extensive, is characterized by its
considerable size and variable data quality. Consequently,
to extract a high-quality subset, we employed a systematic
curation strategy analogous to the one applied to gobja-
verse. The screening procedure is outlined as follows: (1)
A geometric similarity clustering algorithm was employed
to identify and remove near-duplicate models. Any model
exhibiting a similarity score of over 85% with another was
considered redundant and removed; (2) To filter out objects
with non-standard or incomplete textures, we performed an
analysis in the HSV (Hue, Saturation, Value) color space.
Models where white pixels constituted more than 75% of
the surface texture were discarded, as this often indicates
missing or placeholder textures. In the end, we filtered ap-
proximately about 20k data points in the Objaverse.For the
other datasets, we used the full data without applying a fil-
tering process. For datasets that do not provide enough view
rendering view, we use the rendering script provided by the
corresponding dataset for enough view rendering. This pro-
cedure ensures that every object in our dataset is represented
by a consistent set of views, capturing its complete geomet-
ric features for subsequent learning tasks.
D. Dataset Construction Pipeline.
As shown in Figure 8, we construct PhysAssets through
an automated pipeline which predicting physical proper-
ties (material class, Young’s modulus, Poisson’s ratio) from
8 selected views using Qwen3-VL [47] and generating
ground-truth reference videos using FramePack [62] condi-
tioned on predicted properties. This pipeline enables scal-
able annotation of 50,215 objects with physical properties
and reference dynamics.
D.1. Physical Property Annotation Pipeline
We develop a semi-automatic annotation pipeline lever-
aging multimodal large language models to predict three
critical physical properties for each object: material class,
Young’s modulus (E), and Poisson’s ratio (ν).
This
approach enables scalable annotation of large-scale 3D
datasets while maintaining consistency with real-world ma-
terial physics.
D.1.1. Visual Feature Extraction
For each 3D object, we choose eight uniformly distributed
views at fixed elevation. These multi-view RGB images
provide comprehensive visual coverage of the object’s ge-
ometry, texture, and appearance. The views are then fed into
Qwen3-VL [47], a state-of-the-art vision-language model
pre-trained on diverse visual and textual data.
Material Classification
Material classification is per-
formed through vision-language alignment.
We define a
closed vocabulary of 45 primary materials commonly found
in everyday objects: Wood, Metal, Plastic, Glass, Fabric,
Leather, Ceramic, Stone, Rubber, Paper, Sand, Snow, Plas-
ticine, Foam, etc.. The model is queried with the following
prompt:
Material Classification Prompt:
13

<!-- page 14 -->
Image Prompt
Image Tokenizer
Text Tokenizer
Text Prompt
Q3:Based on these images, determine the Poisson's Ratio  of the 
object .
Q2:Based on these images, determine the Young's Modulus of the 
object.this list: <Material format details>
Q1:What is the primary material of the object in these images? Answer 
with a single word from this list: <Material format details>
…
…
Vision-Language Model (VLM)
De-tokenizer
A1:
”Wood"
A2: " 10000000.0"
A3: " 0.45"
Physical Parameters
Final Data
Json:
{“E”: 10000000.0,
“nu”: 0.49,
"material": ”Wood"}
Image Prompt
Text Prompt
An object made of {material} falls straight down from 
the air onto the invisible ground, white background, no 
extra background, no shadow.
Video Generation Model 
Text Tokenizer
…
Image Tokenizer
…
material
De-tokenizer
Video
Video
+
Figure 8. Automated dataset construction pipeline. We predict physical properties using Qwen3-VL, and generate reference videos using
FramePack.
“What is the primary material of the object in these im-
ages?
Answer with a single word from this list: Wood,
Metal, Plastic, Glass, Fabric, Leather, Ceramic, Stone,
Rubber, Paper, Sand, Snow, Plasticine, Foam, etc.”
The model processes all eight views and outputs a ma-
terial label based on cross-modal similarity between visual
features and material descriptions. In cases of ambiguity,
a weighted voting mechanism across views determines the
final material class.
Young’s Modulus Prediction
Young’s modulus (E)
characterizes material stiffness—the resistance to elastic
deformation under stress.
We discretize the continuous
range of Young’s modulus values into 10 interpretable cat-
egories spanning from extremely soft materials (e.g., gel,
foam) to ultra-stiff materials (e.g., diamond, tungsten). The
model is prompted with:
“Based on these images, determine the Young’s Modulus
(E) of the object.
What is Young’s Modulus?
Young’s Modulus (E) measures a material’s stiffness or
resistance to elastic deformation. It indicates how much
stress is needed to produce a given amount of strain (defor-
mation).
Select the most appropriate description:
1. extremely soft - Like gel or foam (e.g., jelly, soft foam) ∼
1 KPa
2. very soft - Like rubber or sponge (e.g., rubber bands,
foam mattress) ∼100 KPa
3. soft - Like soft plastics or leather (e.g., leather, soft PVC)
∼1 MPa
4. moderately soft - Like hard rubber (e.g., tire rubber) ∼10
MPa
5. moderate - Like nylon or cork (e.g., nylon, wood cork)
∼100 MPa
6. moderately stiff - Like hard plastics (e.g., ABS plastic,
acrylic) ∼1 GPa
7. stiff - Like glass or ceramics (e.g., glass, porcelain) ∼10
GPa
8. very stiff - Like aluminum (e.g., aluminum, brass) ∼70
GPa
9. extremely stiff - Like steel (e.g., steel, iron) ∼200 GPa
10. ultra stiff - Like tungsten or diamond (e.g., tungsten, di-
amond) ∼400 GPa
Answer with ONLY ONE of these exact keywords.”
The predicted categorical label is then mapped to a nu-
merical value in Pascals (Pa) using the mapping defined in
Table 6.
Poisson’s Ratio Prediction
Poisson’s ratio (ν) quantifies
the ratio of lateral strain to axial strain when a material is
deformed. We similarly discretize Poisson’s ratio into 10
categories representing different material behaviors, from
auxetic materials (negative Poisson’s ratio) to nearly in-
compressible materials (approaching 0.5). The prediction
prompt is:
Poisson’s Ratio Prediction Prompt:
“Based on these images, determine the Poisson’s Ratio
(ν) of the object.
What is Poisson’s Ratio?
Poisson’s Ratio (ν) measures how much a material ex-
pands laterally when compressed axially, or contracts lat-
14

<!-- page 15 -->
erally when stretched. It describes the relationship between
lateral strain and axial strain.
Select the most appropriate description:
1. nearly incompressible - Almost no volume change (e.g.,
rubber) ∼0.50
2. high resistance - High lateral expansion (e.g., soft rub-
ber) ∼0.45
3. moderately high - Moderately high deformation (e.g.,
gold, lead) ∼0.40
4. moderate high - Above average deformation (e.g., plas-
tic, aluminum) ∼0.35
5. moderate - Typical for many metals (e.g., steel, iron)
∼0.30
6. moderate low - Below average deformation (e.g., glass)
∼0.25
7. low - Low lateral expansion (e.g., concrete, ceramics)
∼0.20
8. very low - Very low lateral expansion (e.g., cork) ∼0.15
9. extremely low - Minimal lateral deformation (e.g., foam)
∼0.10
10. auxetic - Negative Poisson’s ratio materials ∼0.01
Answer with ONLY ONE of these exact keywords.”
The categorical output is converted to a dimensionless
numerical value using the mapping in Table 7.
Property Mapping Tables
The categorical predictions
from the vision-language model are mapped to numerical
physical property values suitable for Material Point Method
(MPM) simulation. Tables 6 and 7 present the complete
mappings.
Table 6. Young’s Modulus categorical to numerical mapping. Val-
ues span 8 orders of magnitude, covering materials from soft gels
to ultra-hard ceramics.
Category
Example Materials
Value (Pa)
extremely soft
Gel, foam, jelly
1.0 × 103
very soft
Rubber, sponge, silicone
1.0 × 105
soft
Leather, soft PVC, fabric
1.0 × 106
moderately soft
Hard rubber, tire rubber
1.0 × 107
moderate
Nylon, cork, paper
1.0 × 108
moderately stiff
Hard plastic (ABS, acrylic)
1.0 × 109
stiff
Glass, ceramic, porcelain
1.0 × 1010
very stiff
Aluminum, brass, bronze
7.0 × 1010
extremely stiff
Steel, iron, stainless steel
2.0 × 1011
ultra stiff
Tungsten, diamond, carbide
4.0 × 1011
D.2. Video Generation and Preference Calculation
To facilitate the second stage of our training, which employs
Direct Preference Optimization (DPO), we established a
systematic pipeline for generating a dataset of preference
Table 7. Poisson’s Ratio categorical to numerical mapping. Values
range from 0.01 (auxetic materials) to 0.49 (nearly incompressible
materials).
Category
Example Materials
Value
auxetic
Special engineered materials
0.01
extremely low
Foam materials
0.10
very low
Cork, engineered materials
0.15
low
Concrete, ceramics, brick
0.20
moderate low
Glass, cast iron
0.25
moderate
Steel, iron, brass, titanium
0.30
moderate high
Plastic, aluminum, copper
0.35
moderately high
Gold, lead, clay
0.40
high resistance
Soft rubber, flexible polymers
0.45
nearly incompressible
Rubber, elastomers
0.49
tuples.
This process is crucial for providing the high-
quality, ranked data required to fine-tune our model on the
nuances of physical dynamics.
The pipeline consists of
three main steps:
D.2.1. Ground-Truth Video Generation
We generate reference videos using FramePack [62], guided
by text prompts describing the physical scenario.
After
evaluating multiple prompt formulations (detailed below),
we selected the following template for its optimal balance
of simplicity and physical realism:
“An object made of {material} falls straight
down from the air onto the invisible ground, white
background, no extra background, no shadow.”
D.2.2. Alternative Prompt Variants
For reference and reproducibility, we document the alter-
native prompt variants explored during our experimenta-
tion. These prompts represent different trade-offs between
prompt complexity, physical constraints, and generation
control.
Prompt 2 (Detailed Physics Description):
“A {material} toy centered on a plain pure white back-
ground. The {material} toy falls straight down verti-
cally from the center of the frame to the bottom edge,
obeying the laws of physics (gravity, acceleration).
Show the entire descent: starting stationary at center,
accelerating downwards, hitting the bottom edge with
a subtle impact, and coming to a complete stop. The
{material} toy remains rigid and inanimate through-
out, showing no deformation or independent move-
ment.
Fixed, static camera view.
No anthropomor-
phism, no unexpected motion, only the physics-based
vertical fall and stop.”
15

<!-- page 16 -->
Limitation: Overly detailed constraints sometimes led to
inconsistent generation or failure to satisfy all specified con-
ditions.
Prompt 5 (Identity Preservation Focus):
“Generate a short, high-fidelity video based on the pro-
vided object image, where the absolute highest prior-
ity is to strictly maintain the object’s identity through-
out the entire sequence. The scene features a seam-
less white background and a solid, invisible, horizontal
white floor. The video begins with the object perfectly
still in mid-air, then it is released to fall straight down
vertically under gravity.
Crucially, the object must
maintain its initial orientation during the fall, without
any tumbling, spinning, or rotation. The object is made
of {material}, and its impact and subsequent behav-
ior must realistically simulate the physical properties
of this material.”
Limitation: While improving identity preservation, this
prompt occasionally resulted in unrealistic motion due to
strict orientation constraints.
Prompt 6 (Photorealism Emphasis):
“Generate a short, photorealistic video based on the
provided input image, simulating the object falling and
impacting the ground.
Throughout the entire video,
the object must retain its original visual identity—its
shape, texture, and color. The fall itself must be com-
pletely inanimate and passive; the object must descend
in a pure vertical drop without any rotation, spinning,
or tumbling. Upon impact with the flat white ground,
the object’s physical reaction must precisely mimic the
properties of {material}. The entire event takes place
in a seamless, infinite white studio environment.”
Limitation: We found that excessively long prompts with
detailed constraints often compromise generation quality,
leading to inconsistent or unnatural motion.
The selected prompt consistently produced the most
physically plausible and visually coherent videos across di-
verse materials and object geometries.
D.2.3. Candidate Video Generation
Leveraging the model pre-trained in Stage 1, we generate
a set of plausible, yet varied, simulation outcomes. Specif-
ically, we sample three distinct sets of physical properties
(e.g., Young’s modulus, Poisson’s ratio) from the learned
probability distribution associated with the object. Each of
these property sets is then used to run a new simulation,
producing three unique candidate videos that represent dif-
ferent potential physical behaviors.
D.2.4. Preference Labeling via Trajectory Alignment
To create preference pairs for DPO training, we develop
an automatic labeling pipeline that compares simulated dy-
namics against reference videos through three-stage trajec-
tory alignment. We employ SAM-2 [39] for object segmen-
tation and CoTracker-3 [18] for dense trajectory extraction
across both ground-truth and simulated sequences.
Spatial Alignment.
Due to different camera viewpoints
and object scales between reference and simulated videos,
direct trajectory comparison is infeasible. We address this
through bounding box normalization: for each video, we
compute the object’s bounding box from its segmentation
mask as B = (xmin, ymin, xmax, ymax). Point trajectories
from the ground-truth video are first normalized to [0, 1] co-
ordinates relative to its bounding box:
pnorm =

x −xmin
xmax −xmin
,
y −ymin
ymax −ymin

(15)
These normalized coordinates are then mapped to the sim-
ulated video’s coordinate frame using its bounding box pa-
rameters. This spatial alignment ensures correspondence
between trajectories regardless of viewpoint or scale differ-
ences.
Landing Frame Alignment.
Physical simulations may
exhibit different temporal dynamics (e.g., falling speeds)
even with similar physical properties. To enable fair com-
parison, we align sequences based on a key physical event:
the object’s landing moment. Specifically, we identify the
landing frame as the temporal turning point where the ob-
ject’s vertical motion reverses. For each video, we track
the point with maximum y-coordinate in the first frame
(typically the object’s bottom) and monitor its trajectory.
The landing frame f ∗is detected when the vertical velocity
changes sign:
f ∗= arg min
t {t | yt ≤yt−1, t > 0}
(16)
where yt represents the tracked point’s y-coordinate at
frame t. This frame marks the transition from falling to
resting/bouncing phases.
Temporal Alignment.
Using the detected landing frames
(f ∗
GT, f ∗
sim) as temporal anchors, we align the post-landing
phases of both sequences. We determine the comparable
duration as T = min(TGT −f ∗
GT, Tsim −f ∗
sim), where TGT
and Tsim are the total frame counts. Additionally, we com-
pute a spatial offset (∆x, ∆y) between the landing positions
in both videos and apply this correction to the simulated tra-
jectories:
paligned
sim
= psim + (∆x, ∆y)
(17)
This ensures that both sequences are aligned not only tem-
porally but also spatially at the critical landing event.
16

<!-- page 17 -->
Similarity Metric.
After three-stage alignment, we com-
pute the trajectory dissimilarity as:
d(Vsim, VGT) =
1
NT
N
X
n=1
T
X
t=1
∥pGT
n,t −psim
n,t∥2
(18)
where N is the number of tracked points and T is the
aligned sequence length. Lower dissimilarity indicates bet-
ter physical plausibility. For each scene, we rank K can-
didate simulations by this metric and select the best as the
“winner” and worst as the “loser” for DPO training.
D.2.5. Additional Data Sources
It is also worth noting that the PhysX [2] dataset was re-
leased concurrently with our research, offering 3D objects
annotated with physical properties ,which is also suitable
for our dataset process. Given the timing constraints, its
integration was not feasible for the present study. Never-
theless, we acknowledge its significance and view it as a
promising avenue for extending our work in the future.
Image
Generated three auxiliary views images
Figure 9. Multi-view generation using MVAdapter. Given a
single frontal view image as input (left), MVAdapter [13] gener-
ates three auxiliary views: rear, left, and right (right three panels).
These synthesized views, together with the input frontal view, pro-
vide comprehensive angular coverage for our 3D Gaussian recon-
struction and physics prediction pipeline. The generated views
maintain consistent geometry and appearance while capturing dif-
ferent perspectives of the object.
E. Additional Results
To fully demonstrate the versatility and effectiveness of
our approach, we present an extended suite of supple-
mentary experiments with comprehensive qualitative in-
sights. Specifically, Figure 10 provides detailed visualiza-
tions of the Multi-View Stereo (MVS) module, showcas-
ing its exceptional ability to accurately reconstruct 3D ge-
ometry from multi-view inputs. Given four randomly se-
lected input views, the model generates novel viewpoints,
and we visualize four representative views sampled from
eight randomly selected output views as examples. Comple-
mentarily, Figure 9 offers visualizations of the MVAdapter
component, clearly revealing how it effectively bridges do-
main gaps and enhances feature alignment across diverse
input modalities. Beyond the core component validations,
Figure 11 and Figure 12 exhibit the model’s performance
on fundamental stretching and dropping scenarios, respec-
tively. We further push the envelope to validate its effective-
ness under more challenging configurations: Figure 13 il-
lustrates strong robustness in cluttered/complex background
scenes, while Figure 14 highlights its superior capability in
handling intricate multi-object interactions and other results
in Figure 15 16 17.
F. Limitations and Future Work
While PhysGM demonstrates significant advances in fast,
physically-grounded 3D synthesis, it is important to ac-
knowledge its current limitations, which also highlight
promising directions for future research.
Data Dependency and Generalization.
Our model’s
performance is inherently tied to the scope and diversity
of the PhysAssets dataset. While large, the dataset primar-
ily consists of rigid objects. Consequently, the model may
not generalize well to out-of-distribution categories, such as
highly deformable or articulated objects. Future work could
focus on expanding the dataset and exploring domain adap-
tation techniques to handle a wider variety of object types.
Simplified Physics Representation.
PhysGM currently
predicts a single, “lumped” vector of physical properties
(e.g., one mass, one friction coefficient) for the entire ob-
ject. This assumes uniform material composition, which is
not true for many real-world objects (e.g., a hammer with
a metal head and wooden handle). A significant next step
would be to extend our framework to predict spatially vary-
ing material properties, enabling more complex and realistic
simulations.
17

<!-- page 18 -->
Input Image
Generated 3D Gaussians
Figure 10. Qualitative results for Multi-View Stereo. Our method generates Gaussian splatting with remarkable visual quality on various
challenging images.
18

<!-- page 19 -->
Figure 11. Qualitative results for stretching scenarios. Our method correctly captures the distinct responses of different materials under
tensile forces.
19

<!-- page 20 -->
Figure 12. Qualitative results for dropping scenarios. Our model accurately predicts the physical properties of different materials, leading
to plausible deformation and final states upon impact with the ground.
20

<!-- page 21 -->
Figure 13. Demonstration of our model’s robustness in in-the-wild scenes.
21

<!-- page 22 -->
Figure 14. Qualitative results for multi-object interaction scenarios. Our approach can handle more complex scenes involving simultaneous
collisions and interactions, generating physically consistent outcomes for all objects.
22

<!-- page 23 -->
Figure 15. Other results.
23

<!-- page 24 -->
Figure 16. Other results.
24

<!-- page 25 -->
Figure 17. Other results.
25

<!-- page 26 -->
Acknowledgements.
This work was supported by the
NSFC under Grants U2441242.
This work is also sup-
ported by the National Natural Science Foundation of China
(U23B2013 and 62276176)
References
[1] Junhao Cai, Yuji Yang, Weihao Yuan, Yisheng He, Zilong
Dong, Liefeng Bo, Hui Cheng, and Qifeng Chen.
Gic:
Gaussian-informed continuum for physical property identifi-
cation and simulation. Advances in Neural Information Pro-
cessing Systems, 37:75035–75063, 2024. 3
[2] Ziang Cao, Zhaoxi Chen, Linag Pan, and Ziwei Liu. Physx:
Physical-grounded 3d asset generation.
arXiv preprint
arXiv:2507.12465, 2025. 3, 17
[3] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and
Vincent Sitzmann. pixelsplat: 3d gaussian splats from image
pairs for scalable generalizable 3d reconstruction. In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition, pages 19457–19467, 2024. 3
[4] Boyuan Chen, Hanxiao Jiang, Shaowei Liu, Saurabh Gupta,
Yunzhu Li, Hao Zhao, and Shenlong Wang.
Physgen3d:
Crafting a miniature interactive world from a single image.
In Proceedings of the Computer Vision and Pattern Recogni-
tion Conference, pages 6178–6189, 2025. 3
[5] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang,
Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei
Cai. Mvsplat: Efficient 3d gaussian splatting from sparse
multi-view images. In European Conference on Computer
Vision, pages 370–386. Springer, 2024. 3
[6] Jasmine Collins, Shubham Goel, Kenan Deng, Achlesh-
war Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas
F Yago Vicente, Thomas Dideriksen, Himanshu Arora, et al.
Abo: Dataset and benchmarks for real-world 3d object un-
derstanding.
In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 21126–
21136, 2022. 6, 12
[7] Tri Dao. Flashattention-2: Faster attention with better paral-
lelism and work partitioning, 2023. 7
[8] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs,
Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana
Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse:
A universe of annotated 3d objects.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 13142–13153, 2023. 6, 12
[9] Alexey Dosovitskiy.
An image is worth 16x16 words:
Transformers for image recognition at scale. arXiv preprint
arXiv:2010.11929, 2020. 2
[10] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kin-
man, Ryan Hickman, Krista Reymann, Thomas B McHugh,
and Vincent Vanhoucke. Google scanned objects: A high-
quality dataset of 3d scanned household items. In 2022 In-
ternational Conference on Robotics and Automation (ICRA),
pages 2553–2560. IEEE, 2022. 10, 12
[11] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou,
Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao
Tan. Lrm: Large reconstruction model for single image to
3d. arXiv preprint arXiv:2311.04400, 2023. 3
[12] Tianyu Huang, Haoze Zhang, Yihan Zeng, Zhilu Zhang, Hui
Li, Wangmeng Zuo, and Rynson WH Lau. Dreamphysics:
Learning physics-based 3d dynamics with video diffusion
priors. In Proceedings of the AAAI Conference on Artificial
Intelligence, pages 3733–3741, 2025. 3, 7, 8, 9, 10
[13] Zehuan Huang, Yuan-Chen Guo, Haoran Wang, Ran Yi,
Lizhuang Ma, Yan-Pei Cao, and Lu Sheng.
Mv-adapter:
Multi-view consistent image generation made easy.
In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 16377–16387, 2025. 4, 17
[14] Chenfanfu Jiang, Craig Schroeder, Joseph Teran, Alexey
Stomakhin, and Andrew Selle. The material point method
for simulating continuum materials. In Acm siggraph 2016
courses, pages 1–52. 2016. 5
[15] Yanqin Jiang, Chaohui Yu, Chenjie Cao, Fan Wang, Weim-
ing Hu, and Jin Gao. Animate3d: Animating any 3d model
with multi-view video diffusion. Advances in Neural Infor-
mation Processing Systems, 37:125879–125906, 2024. 2
[16] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi,
Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang
Xu. Lvsm: A large view synthesis model with minimal 3d
inductive bias. arXiv preprint arXiv:2410.17242, 2024. 3
[17] Heewoo Jun and Alex Nichol.
Shap-e:
Generat-
ing conditional 3d implicit functions.
arXiv preprint
arXiv:2305.02463, 2023. 2
[18] Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia
Neverova, Andrea Vedaldi, and Christian Rupprecht. Co-
tracker3:
Simpler and better point tracking by pseudo-
labelling real videos. In Proc. arXiv:2410.11831, 2024. 6,
16
[19] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia
Neverova, Andrea Vedaldi, and Christian Rupprecht. Co-
tracker: It is better to track together. In European conference
on computer vision, pages 18–35. Springer, 2024. 6
[20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis.
3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 2, 3
[21] Mukul Khanna*, Yongsen Mao*, Hanxiao Jiang, Sanjay
Haresh, Brennan Shacklett, Dhruv Batra, Alexander Clegg,
Eric Undersander, Angel X. Chang, and Manolis Savva.
Habitat Synthetic Scenes Dataset (HSSD-200): An Analy-
sis of 3D Scene Scale and Realism Tradeoffs for ObjectGoal
Navigation. arXiv preprint, 2023. 6, 12, 13
[22] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao,
Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer White-
head, Alexander C Berg, Wan-Yen Lo, et al. Segment any-
thing. In Proceedings of the IEEE/CVF international confer-
ence on computer vision, pages 4015–4026, 2023. 6
[23] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun
Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg
Shakhnarovich, and Sai Bi.
Instant3d:
Fast text-to-3d
with sparse-view generation and large reconstruction model.
arXiv preprint arXiv:2311.06214, 2023. 3
[24] Zhengqin Li, Dilin Wang, Ka Chen, Zhaoyang Lv, Thu
Nguyen-Phuoc, Milim Lee, Jia-Bin Huang, Lei Xiao, Yufeng
Zhu, Carl S Marshall, et al.
Lirm: Large inverse render-
ing model for progressive reconstruction of shape, materials
26

<!-- page 27 -->
and view-dependent radiance fields. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
505–517, 2025. 3
[25] Chieh Hubert Lin, Zhaoyang Lv, Songyin Wu, Zhen Xu,
Thu Nguyen-Phuoc, Hung-Yu Tseng, Julian Straub, Numair
Khan, Lei Xiao, Ming-Hsuan Yang, et al. Dgs-lrm: Real-
time deformable 3d gaussian reconstruction from monocular
videos. arXiv preprint arXiv:2506.09997, 2025. 3
[26] Yuchen Lin,
Chenguo Lin,
Jianjin Xu,
and Yadong
Mu.
Omniphysgs:
3d constitutive gaussians for gen-
eral physics-based dynamics generation.
arXiv preprint
arXiv:2501.18982, 2025. 2, 3, 7, 8, 9, 10
[27] Huan Ling, Seung Wook Kim, Antonio Torralba, Sanja Fi-
dler, and Karsten Kreis. Align your gaussians: Text-to-4d
with dynamic 3d gaussians and composed diffusion models.
In Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 8576–8588, 2024. 2
[28] Fangfu Liu, Hanyang Wang, Shunyu Yao, Shengjun Zhang,
Jie Zhou, and Yueqi Duan. Physics3d: Learning physical
properties of 3d gaussians via video diffusion. arXiv preprint
arXiv:2406.04338, 2024. 3
[29] Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shen-
long Wang. Physgen: Rigid-body physics-grounded image-
to-video generation. In European Conference on Computer
Vision, pages 360–378. Springer, 2024. 3
[30] Guanxing Lu, Shiyi Zhang, Ziwei Wang, Changliu Liu, Ji-
wen Lu, and Yansong Tang. Manigaussian: Dynamic gaus-
sian splatting for multi-task robotic manipulation.
arXiv
preprint arXiv:2403.08321, 2024. 2
[31] Miles Macklin. Warp: A high-performance python frame-
work for gpu simulation and graphics. In NVIDIA GPU Tech-
nology Conference (GTC), 2022. 3
[32] Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela
Mishkin, and Mark Chen. Point-e: A system for generat-
ing 3d point clouds from complex prompts. arXiv preprint
arXiv:2212.08751, 2022. 2
[33] Zijie Pan, Zeyu Yang, Xiatian Zhu, and Li Zhang.
Effi-
cient4d: Fast dynamic 3d object generation from a single-
view video. arXiv preprint arXiv:2401.08742, 2024. 2
[34] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Milden-
hall.
Dreamfusion: Text-to-3d using 2d diffusion.
arXiv
preprint arXiv:2209.14988, 2022. 2
[35] Ri-Zhao Qiu, Ge Yang, Weijia Zeng, and Xiaolong Wang.
Feature splatting: Language-driven physics-based scene syn-
thesis and editing. arXiv preprint arXiv:2404.01223, 2024.
9, 10
[36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervi-
sion. In International conference on machine learning, pages
8748–8763. PmLR, 2021. 7
[37] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D Manning, Stefano Ermon, and Chelsea Finn. Direct
preference optimization: Your language model is secretly a
reward model. Advances in neural information processing
systems, 36:53728–53741, 2023. 2, 5
[38] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-
sion transformers for dense prediction. In Proceedings of
the IEEE/CVF international conference on computer vision,
pages 12179–12188, 2021. 4
[39] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang
Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman
R¨adle, Chloe Rolland, Laura Gustafson, et al.
Sam 2:
Segment anything in images and videos.
arXiv preprint
arXiv:2408.00714, 2024. 5, 6, 16
[40] Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao,
Gang Zeng, and Ziwei Liu.
Dreamgaussian4d: Genera-
tive 4d gaussian splatting. arXiv preprint arXiv:2312.17142,
2023. 2, 9, 10
[41] Jiawei Ren, Cheng Xie, Ashkan Mirzaei, Karsten Kreis, Zi-
wei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim,
Huan Ling, et al. L4gm: Large 4d gaussian reconstruction
model. Advances in Neural Information Processing Systems,
37:56828–56858, 2024. 2
[42] Oriane Sim´eoni, Huy V. Vo, Maximilian Seitzer, Federico
Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov,
Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa,
Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan
Wang, Timoth´ee Darcet, Th´eo Moutakanni, Leonel Sentana,
Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt,
Camille Couprie, Julien Mairal, Herv´e J´egou, Patrick La-
batut, and Piotr Bojanowski. Dinov3, 2025. 3, 10
[43] Alexey Stomakhin, Craig Schroeder, Lawrence Chai, Joseph
Teran, and Andrew Selle. A material point method for snow
simulation. ACM Transactions on Graphics (TOG), 32(4):
1–10, 2013. 5
[44] Stanislaw Szymanowicz, Chrisitian Rupprecht, and Andrea
Vedaldi.
Splatter image: Ultra-fast single-view 3d recon-
struction.
In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 10208–
10217, 2024. 3
[45] Xiyang Tan, Ying Jiang, Xuan Li, Zeshun Zong, Tianyi
Xie, Yin Yang, and Chenfanfu Jiang. Physmotion: Physics-
grounded dynamics from a single image.
arXiv preprint
arXiv:2411.17189, 2024. 3
[46] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang,
Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian
model for high-resolution 3d content creation. In European
Conference on Computer Vision, pages 1–18. Springer, 2024.
2, 3, 8
[47] Qwen Team. Qwen3 technical report, 2025. 6, 13
[48] Chen Wang, Chuhao Chen, Yiming Huang, Zhiyang Dou,
Yuan Liu, Jiatao Gu, and Lingjie Liu. Physctrl: Generative
physics for controllable and physics-grounded video genera-
tion. arXiv preprint arXiv:2509.20358, 2025. 2
[49] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang,
Xiang Wang, and Shiwei Zhang. Modelscope text-to-video
technical report. arXiv preprint arXiv:2308.06571, 2023. 2
[50] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea
Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Vi-
sual geometry grounded transformer. In Proceedings of the
Computer Vision and Pattern Recognition Conference, pages
5294–5306, 2025. 2
27

<!-- page 28 -->
[51] Peng Wang, Hao Tan, Sai Bi, Yinghao Xu, Fujun Luan,
Kalyan Sunkavalli, Wenping Wang, Zexiang Xu, and Kai
Zhang.
Pf-lrm:
Pose-free large reconstruction model
for joint pose and shape prediction.
arXiv preprint
arXiv:2311.12024, 2023. 3
[52] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Si-
moncelli. Image quality assessment: from error visibility to
structural similarity. IEEE transactions on image processing,
13(4):600–612, 2004. 7, 10
[53] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi
Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scal-
able image-to-3d generation via 3d latent diffusion trans-
former. Advances in Neural Information Processing Systems,
37:121859–121881, 2024. 3
[54] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren,
Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian,
et al. Omniobject3d: Large-vocabulary 3d object dataset for
realistic perception, reconstruction and generation. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 803–814, 2023. 6, 12
[55] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng,
Yin Yang, and Chenfanfu Jiang.
Physgaussian: Physics-
integrated 3d gaussians for generative dynamics. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 4389–4398, 2024. 2, 3, 5, 9, 10
[56] Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen,
Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon Wet-
zstein. Grm: Large gaussian reconstruction model for ef-
ficient 3d reconstruction and generation. In European Con-
ference on Computer Vision, pages 1–20. Springer, 2024. 3
[57] Zhen Xu, Zhengqin Li, Zhao Dong, Xiaowei Zhou, Richard
Newcombe, and Zhaoyang Lv. 4dgt: Learning a 4d gaus-
sian transformer using real-world monocular videos. arXiv
preprint arXiv:2506.08015, 2025. 2
[58] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi
Feng, and Hengshuang Zhao. Depth anything: Unleashing
the power of large-scale unlabeled data. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 10371–10381, 2024. 4
[59] Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao,
and Yunchao Wei.
4dgen:
Grounded 4d content gen-
eration with spatial-temporal consistency.
arXiv preprint
arXiv:2312.17225, 2023. 2
[60] Haiyu Zhang, Xinyuan Chen, Yaohui Wang, Xihui Liu, Yun-
hong Wang, and Yu Qiao. 4diffusion: Multi-view video dif-
fusion model for 4d generation. Advances in Neural Infor-
mation Processing Systems, 37:15272–15295, 2024. 2
[61] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao,
Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large recon-
struction model for 3d gaussian splatting. In European Con-
ference on Computer Vision, pages 1–19. Springer, 2024. 3,
8
[62] Lvmin Zhang and Maneesh Agrawala. Packing input frame
context in next-frame prediction models for video genera-
tion. arXiv preprint arXiv:2504.12626, 2025. 6, 13, 15
[63] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shecht-
man, and Oliver Wang. The unreasonable effectiveness of
deep features as a perceptual metric. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 586–595, 2018. 4, 7, 10
[64] Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y
Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and
William T Freeman. Physdreamer: Physics-based interac-
tion with 3d objects via video generation. In European Con-
ference on Computer Vision, pages 388–406. Springer, 2024.
2, 3, 9, 10
[65] Yuchen Zhang, Nikhil Keetha, Chenwei Lyu, Bhuvan Jhamb,
Yutian Chen, Yuheng Qiu, Jay Karhade, Shreyas Jha, Yaoyu
Hu, Deva Ramanan, Sebastian Scherer, and Wenshan Wang.
Ufm: A simple path towards unified dense correspondence
with flow, 2025. 3
[66] Haoyu Zhao, Hao Wang, Xingyue Zhao, Hao Fei, Hongqiu
Wang, Chengjiang Long, and Hua Zou.
Efficient physics
simulation for 3d scenes via mllm-guided gaussian splatting.
arXiv preprint arXiv:2411.12789, 2024. 3, 9, 10
[67] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang,
Deqing Sun, and Ming-Hsuan Yang.
Drivinggaussian:
Composite gaussian splatting for surrounding dynamic au-
tonomous driving scenes. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition,
pages 21634–21643, 2024. 2
[68] Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yi-
cong Hong, Li Fuxin, and Zexiang Xu. Long-lrm: Long-
sequence large reconstruction model for wide-coverage
gaussian splats. In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision, pages 4349–4359,
2025. 3
28
