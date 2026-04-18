<!-- page 1 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
1
DreamScene: 3D Gaussian-based End-to-end
Text-to-3D Scene Generation
Haoran Li, Yuli Tian, Kun Lan, Yong Liao* Member, IEEE, Lin Wang Member, IEEE, Pan Hui Fellow, IEEE,
Peng Yuan Zhou Member, IEEE
Abstract—Generating 3D scenes from natural language holds
great promise for applications in gaming, film, and design.
However, existing methods struggle with automation, 3D con-
sistency, and fine-grained control. We present DreamScene, an
end-to-end framework for high-quality and editable 3D scene
generation from text or dialogue. DreamScene begins with a
scene planning module, where a GPT-4 agent infers object
semantics and spatial constraints to construct a hybrid graph.
A graph-based placement algorithm then produces a structured,
collision-free layout. Based on this layout, Formation Pattern
Sampling (FPS) generates object geometry using multi-timestep
sampling and reconstructive optimization, enabling fast and
realistic synthesis. To ensure global consistent, DreamScene
employs a progressive camera sampling strategy tailored to both
indoor and outdoor settings. Finally, the system supports fine-
grained scene editing, including object movement, appearance
changes, and 4D dynamic motion. Experiments demonstrate that
DreamScene surpasses prior methods in quality, consistency,
and flexibility, offering a practical solution for open-domain
3D content creation. Code and demos are available at https:
//jahnsonblack.github.io/DreamScene-Full/.
Index Terms—Text-to-3D, text-to-3D scene, scene generation,
scene editing, 3D Gaussian.
I. INTRODUCTION
T
HE progress made in text-to-3D scene generation signi-
fies a significant step forward in the field of 3D content
creation [1]–[12]. It has extended its reach from generating
simple objects to building intricate, detailed scenes straight
from the textual descriptions. This advancement not only light-
ens the burden on 3D modelers but also stimulates expansion
in industries like gaming, film, and architecture.
Text-to-3D methods [1]–[12] typically use pre-trained 2D
text-to-image models [13]–[15] as prior supervision to create
object-centric 3D differentiable representations [16]–[20] by
rendering image from the camera’s perspective facing towards
the object. Generating text-to-3D scenes require rendering
This work was supported by Anhui Province Science and Technology Inno-
vation Breakthrough Plan (202423l10050033) and the National Key Research
and Development Program of China (2022YFB3105405, 2021YFC3300502).
Corresponding author: Yong Liao.
Haoran
Li,
Yuli
Tian,
Kun
Lan
and
Yong
Liao
are
with
Uni-
versity
of
Science
and
Technology
of
China,
Hefei,
China (e-mail:
lhr123@mail.ustc.edu.cn; yltian@mail.ustc.edu.cn; lankun@mail.ustc.edu.cn;
yliao@ustc.edu.cn).
Lin
Wang
is
with
the
School
of
Electrical
and
Electronic
Engineering,
Nanyang
Technological
University,
Singapore
(email:eee-
addison.wang@ntu.edu.sg).
Pan Hui is with the Computational Media and Arts thrust, Hong Kong
University of Science and Technology (Guangzhou), China, and Department
of Computer Science, University of Helsinki, Finland (email:panhui@ust.hk).
Peng Yuan Zhou is with the Department of Electrical and Computer
Engineering, Aarhus University, Denmark (email: pengyuan.zhou@ece.au.dk).
from preset camera positions outward, capturing the scene
from these specific viewpoints. However, as shown in Fig. 1,
these text-to-3D generation techniques face several significant
obstacles, including: 1) A lack of automation, often relying
on manual layout design or hardcoded placement trajectories,
thereby reducing usability and scalability [21]–[24]; 2) Incon-
sistent 3D visual cues [21]–[23], [25]–[28], with satisfactory
outputs restrained to only training camera poses, similar to
360-degree photography, which limits their applicability in
interactive or exploratory tasks within the generated 3D en-
vironment.; 3) An inefficient generation process often results
in subpar outputs [21], [25], [26], [29] and extended comple-
tion times [22], [27]; 4) The inability to distinguish objects
from their environments, which obstructs flexible editing on
individual components [22], [23], [25], [27].
To address these limitations, we present DreamScene,
an end-to-end framework that enables automated, efficient,
scene-consistent, and flexibly editable 3D scene generation.
Firstly, we perform scene planning by decomposing the scene
into structured object-level and environment-level components.
Given either an open-ended scene prompt or an interactive
dialogue, a GPT-4 agent [30] infers detailed information
for each object, including its category, real-world size, and
descriptive prompt. Based on these results, the agent assigns
coarse placements by predicting region-level anchors (e.g.,
center, side, corner) and inter-object spatial relations (e.g.,
next to, opposite). We organize these spatial constraints into
a hybrid constraint graph, capturing both object-to-object and
object-to-scene relationships. To compute a valid layout, we
propose a graph-based constraint placement (GCP) algorithm
that incrementally assigns position and orientation to each
object while avoiding collisions. This yields a physically plau-
sible, semantically consistent object arrangement and provides
affine parameters—scaling s,translation t and rotation r—for
each object to be used in downstream generation.
Secondly, we generate 3D object representations using For-
mation Pattern Sampling (FPS) guided by descriptive prompts
from the planning stage. Based on the observed patterns in 3D
representation formation, FPS utilizes multi-timestep sampling
(MTS) to balance semantic information and shape consistency,
enabling the rapid generation of high-quality, semantically
rich 3D representations. FPS ensures stable generation perfor-
mance by eliminating redundant internal 3D Gaussians during
optimization. And, by employing DDPM [31] with small
timestep sampling and 3D reconstruction techniques [18],
FPS efficiently generates surfaces with plausible textures from
various viewpoints in just tens of seconds.
arXiv:2507.13985v2  [cs.CV]  29 Jul 2025

<!-- page 2 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
2
low quality
automation/scene-wide consistency /high quality/flexible editing 
low consistency
manual layout
Entangled geometry
limited editability
Other Scene Generation Methods
DreamScene
Fig. 1. DreamScene exhibits significant advantages compared with current state-of-the-art text-to-3D scene generation methods. Text2Room [22] and Set-
the-Scene [21] require complex user-specified object placement. Text2Room, Text2NeRF [25] and many inpainting-based methods suffer from low scene
consistency, producing incoherent geometry across camera poses. GALA3D [32], CG3D [33] generate scenes with low visual quality and do not generate
3D environments. Moreover, most existing methods [22], [23], [25], [27] produce entangled geometry without object-level separation [34], leading to limited
or no editability. In contrast, DreamScene supports automatic layout planning, ensures scene-wide consistency, achieves high visual fidelity, and enables
flexible editing of each individual objects.
Third, we insert the optimized objects into the scene accord-
ing to the layout inferred in the planning stage, applying their
predicted affine transformations to initialize the composition.
We then introduce a progressive three-step camera sampling
strategy to create an environment and guarantee 3D consis-
tency. 1), we generate a coarse environment representation
by positioning the camera at the center of the scene. 2), we
modify ground formation according to the scene type: a) for
indoor scenes, by dividing them into regions and choosing a
random camera position for rendering; b) for outdoor scenes,
by arranging them into concentric circles based on the radius,
and sampling camera poses at different circles along the same
direction. 3), we solidify the scene through reconstructive
generation in FPS, using all camera poses to further refine
the scene. This process results in a semantically aligned and
visually consistent scene, mitigating issues such as the multi-
headed artifact commonly found in prior text-to-3D scene
generation methods [22], [23], [25], [27] .
Finally, DreamScene supports flexible scene editing through
three core operations: object relocation, appearance modifica-
tion, and temporal movement. Object positions can be adjusted
by modifying affine parameters and re-invoking scene planning
module. Appearance edits, including shape or texture changes,
are enabled via an MTS-based 2D optimization pipeline. For
dynamic behaviors, we assign time-dependent transformations
to selected objects, allowing them to follow user-specified
motion trajectories in 4D scene generation.
This
work
is
an
improvement
over
our
ECCV2024
work [24], achieved by substantially extending the method and
experiment in the following ways: (I) We introduce a novel
Scene Planning module to automatically generate structured,
layout-aware 3D scenes. Instead of manually defining object
placements, we leverage GPT-4 [30] as an agent to infer
object categories, physical dimensions, and spatial constraints
from either direct descriptions or multi-turn dialogues. A
hybrid constraint graph is constructed to represent object-
to-object and object-to-scene relations, and a graph-based
constraint placement (GCP) algorithm assigns valid, collision-
free positions and orientations. The inferred layout aligns with
common sense and physical feasibility, serving as a strong
prior for downstream environment generation and helping
prevent artifacts such as multi-headed scenes. (Sec. IV-A). (II)
We develop a flexible editing framework for post-hoc scene
control, supporting: (a) object relocation via affine updates
and planning re-execution; (b) appearance editing by adapting
MTS-based 2D diffusion to our 3D pipeline; and (c) motion
editing through time-varying transformations for dynamic 4D
scene composition (Sec. IV-D). (III) We provide a theoretical
explanation of Multi-Timestep Sampling (MTS), showing its
connection to 2D editing frameworks (Sec.I in Supp.). (IV)
We provide a more comprehensive analysis and evaluation of
current text-to-3D scene generation methods. This includes
an expanded discussion of a technical comparison between
DreamScene and prior approaches and layout generation
strategies (Sec.II-C), along with additional camera sampling
details (Sec.V-A) and extended qualitative and quantitative
experiments on layout generation, scene generation quality,
scene editing and camera sampling (Sec. V-A, Sec. V-B,
Sec. V-C, Sec. V-D).
II. RELATED WORK
A. Differentiable 3D Representation
Utilizing differentiable approaches such as NeRF [16], [35],
SDF [17], [20], and 3D Gaussian Splatting [18], it becomes
possible to represent, manipulate, and render 3D objects and
scenes effectively. These kinds of representations work well
with optimization algorithms like gradient descent, making it
feasible to automatically adjust the parameters of 3D represen-
tations by minimizing loss. A notable recent development [18]
involves the use of differentiable 3D Gaussians to model 3D
scenes, which has resulted in exceptional real-time rendering
performance through the splatting technique. In comparison
to implicit representations [16], [19], [35], 3D Gaussians
present a more explicit framework that eases the integration
of multiple scenes. Consequently, we select 3D Gaussians for
their straightforward, explicit representation and the simplicity
associated with merging scenes.
B. Text-to-3D Generation
Currently, the main approaches to generating 3D represen-
tations in text-to-3D tasks involve either direct methods [11],

<!-- page 3 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
3
[12], [36] or distillation from pre-trained 2D text-to-image
models [1]–[3], [37]. Direct techniques require annotated 3D
datasets for quick generation, but they frequently face issues
such as lower quality and increased GPU demands, often
acting as initial stages for distillation methods [8], [38]. For
instance, Point-E [11] creates an image by employing a diffu-
sion model based on text, which is subsequently transformed
into a point cloud. Conversely, Shap-E [12] links 3D assets to
implicit function parameters using an encoder and trains the
diffusion model based on these parameters with conditions.
The prevailing approach in the field has become the dis-
tillation of 3D representations from pre-trained 2D text-to-
image diffusion models [1]–[3], [5]–[7]. A pioneer, Dream-
Fusion [1], blazed a trail by introducing Score Distillation
Sampling (SDS), ensuring that images rendered from multiple
viewpoints align with the distribution of 2D text-to-image
models [13]–[15]. Subsequent advancements [2]–[12] have
built upon this, refining 3D generation in terms of quality,
speed, and diversity. For instance, LucidDreamer [8] employs
DDIM inversion [39], [40] to ensure 3D consistency during
the object generation process, while DreamTime [6] hastens
the generation convergence via monotonically non-increasing
sampling of timestep t in a 2D text-to-image model. Drawing
inspiration from these pioneering works, our method offers a
more efficient route to generate high-quality and semantically
rich 3D representations.
C. Text-to-3D Scene Generation Methods
Contemporary text-to-3D scene generation techniques, as
depicted in Fig.1, encounter considerable constraints. We can
classify these methods into three categories: Inpainting-based
methods [22], [23], [25], Combination-based methods [21],
[28], and Layout generation methods [29], [32], [33].
Inpainting-based methods [22], [23], [25] utilize text-to-
image inpainting techniques for generating scenes and cur-
rently serve as the main approach for scene generation. These
methods initiate an image, partially mask it to represent a dif-
ferent viewpoint, and then employ pretrained image inpainting
models like Stable Diffusion [14] along with depth estimation
to reconstruct the concealed parts of the image and infer
their depths. The entire scene is iteratively composed through
depth and image alignment. Although these methods can yield
visually appealing results at specific camera positions(e.g.,
the scene’s center) during the generation process, their vis-
ible range faces substantial limitations. Exploring beyond the
predefined camera areas used during generation leads to scene
deterioration, as illustrated in Fig. 8 and Fig. 9, highlighting a
lack of 3D consistency throughout the scene. More critically,
generated scenes exhibit a ”multi-head” issue, similar to the
multiple heads appearing in object generation methods [1]–[3].
In the scene context, this translates to multiple identical objects
appearing in various directions, such as several sofas facing
different directions in a living room. By employing a carefully
devised camera sampling strategy and pre-positioning objects
in the scene to guide the generation of the surrounding
environment, DreamScene attains scene-wide consistency and
reasonable environmental content creation.
Combination-based methods [21], [28] leverage an assembly
approach for scene construction. They grapple with issues such
as subpar generation quality and sluggish training rates. In
addition, [28] makes use of multiple 3D representations (such
as NeRF+DMTet) for integrating objects and scenes, which
heightens the intricacy of scene representation and restricts
the number of objects that can be incorporated within the
scene (2-3 objects), thereby impacting their utility. Conversely,
DreamScene’s FPS method can swiftly generate high-quality
3D content, by using a solitary 3D representation to assemble
the entire scene, which allows for the inclusion of over 20
objects within the scene.
Layout generation methods adopt diverse strategies. Meth-
ods [29], [32], [33], [41], [42], such as CG3D [33], typically
rely on structured scene prompts and optimize layout param-
eters via image-based supervision. They focus primarily on
the logical assembly of a small set of objects while neglecting
broader environmental context, resulting in basic arrangements
rather than comprehensive scenes. These methods also strug-
gle with occlusion and local minima as layout complexity
increases. CC3D [43] generates layout-conditioned 3D scenes
by back-projecting 2D diffusion outputs into NeRF fields, but
requires the layout to be explicitly provided. BerfScene [44]
reconstructs fused volumetric 3D scenes from single images
without object-level structure or layout control. ATISS [45]
autoregressively generates indoor layouts from structured pri-
ors using Transformers, yet remains limited to closed indoor
domains and requires floorplan input. In contrast, Dream-
Scene supports open-ended prompts or dialogues and gener-
ates diverse and reasonable layouts instead of a single fixed
arrangement. Furthermore, unlike Scene-LLM [46] and 3D-
LLM [47], which focus on understanding or interacting with
existing 3D scenes/layouts and rely heavily on limited indoor
datasets for supervision [48], our approach generates complex
3D scenes entirely from scratch. By leveraging GPT-4’s [30]
broad knowledge of the physical world, DreamScene supports
open-domain scene generation beyond the constraints of pre-
collected 3D data.
DreamScene exhibits a significant edge by autonomously
generating 3D scenes with efficiency, consistency, and flexi-
bility, surpassing prior methods.
III. PRELIMINARY
Diffusion Models [31], [49] facilitate the generation of data
x(x ∼p(x)) by approximating the gradients of log probabil-
ity density functions, represented as ∇x log pdata(x). During
training, noise is progressively added to the input x across t
distinct steps:
xt = √¯αtx +
√
1 −¯αtϵ,
(1)
where ¯αt denotes a predetermined coefficient and ϵ, represent-
ing noise, is drawn from a normal distribution N(0, I). The
noise prediction network ϕ then optimized by reducing the
prediction loss Lt:
Lt = Ex,ϵ∼N (0,I)
h
∥ϵϕ(xt, t) −ϵ∥2i
.
(2)
In the sampling phase, the method deduces x using both the
noisy input and its estimated noise ϵϕ(zt, t).

<!-- page 4 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
4
composition
Step1: surroundings
Step2: ground
Step3: all
+
Indoor
Outdoor
environments 
initialize
Point cloud
Camera Sampling (Sec. IV-C)
…
3D Gaussians
Scale
Translation
Rotation
x
“An autumn park “
“A DSLR photo of an 
ornate stone fountain.”
Valid Layout Output
Scene Planning (Sec. IV-A)
GCP Algorithm 
Add Constraint 
“Please keep the color 
tone warm”
LLM 
Agent
Object Category/Count/Size
Appearance Prompt
Spatial Relation
Constraint Graph
Ø Anchor Selection
Ø Constraint Propagation
Ø Collision Check (AABB)
Ø Fallback Placement
“Add another girl”
Scene Editing (Sec. IV-D)
Object Relocation
Temporal Movement
“Let the man opposite the girl”
𝑠!"#, 𝑡!"#, 𝑟!"#
𝑠!"#
$
, 𝑡!"#
$
, 𝑟!"#
$
modify
add
“Let the man move backward.”
𝑠!"#(𝒕), 𝑡!"#(𝒕), 𝑟!"#(𝒕)
𝑠!"#, 𝑡!"#, 𝑟!"#
𝑠%&'(, 𝑡%&'(, 𝑟%&'(
Appearance Modification
𝜖!(𝑥"!; 𝑦#$%", 𝑡%) −𝜖!(𝑥"!; 𝑦, 𝑡% )
“Change the weather to spring”
…
𝑇!"#
𝑡$
𝑡%
𝑡&
Point-E
(optional)
object initialize
Point cloud
3D Gaussians
rendering
3D Gaussian 
Filtering
𝜖'(𝑥(&, 𝑡), 𝑦)
𝜖'(𝑥(&, 𝑡), ∅)
add noise
Multi-timestep Sampling
U-Net
𝐿!"# = 𝐸$,&,'[𝜔(𝑡) ) ||𝜖((𝑥$%; 𝑦, 𝑡)) −𝜖((𝑥$%; ∅, 𝑡))
*
)+,
||-]
Formation Pattern Sampling (Sec. IV-B)
Reconstructive Generation
freeze  parameters
Pretrained 2D 
Diffusion Model 
…
Fig. 2.
Our framework enables automatic 3D scene generation from natural language, supporting both direct descriptions and interactive dialogues. A GPT-4
agent first performs scene decomposition by inferring object semantics, layout constraints, and spatial relations, and constructs a constraint graph to plan
collision-free object placements. Each object is generated using Formation Pattern Sampling (FPS), which integrates multi-timestep sampling, 3D Gaussian
filtering, and reconstructive generation. These objects are placed into the global scene using predicted affine transformations. We then apply a three-stage
camera sampling strategy to optimize the environment and ensure scene-wide consistency. DreamScene also supports structure-aware scene editing, including
object relocation, appearance modification, and 4D editing.
Score Distillation Sampling (SDS) technique, introduced by
DreamFusion [1], aims to distill 3D representations from a
pre-trained 2D text-to-image diffusion model. The approach
involves a differentiable 3D representation, parameterized by
θ and a rendering function, g. For a specified camera pose c,
the image x is rendered as x = g(θ, c). Subsequently, SDS
employs a 2D diffusion model ϕ with fixed parameters to
distill θ by:
∇θLSDS(θ) = Et,ϵ,c

w(t)(ϵϕ(xt; y, t) −ϵ)∂g(θ, c)
∂θ

,
(3)
where w(t) serves as a weighting function that adjusts based
on the timesteps t and y represents the text embedding derived
from the input prompt.
Classifier Score Distillation (CSD) [7] is a variation of Score
Distillation Sampling(SDS) and takes its cue from Classifier-
Free Guidance (CFG) [50]. This technique differentiates the
noise variance in SDS into two components: the generation
prior, noted as ϵϕ(xt; y, t)−ϵ, and the classifier score, noted as
ϵϕ(xt; y, t) −ϵϕ(xt; ∅, t), ∅represents the empty prompt. This
approach suggests that the classifier score is robust enough to
facilitate text-to-3D translation, and it is outlined as follows:
∇θLCSD(θ) = Et,ϵ,c
h
w(t)(ϵϕ(xt; y, t) −ϵϕ(xt; ∅, t)) ∂g(θ,c)
∂θ
i
. (4)
DreamTime [6] is an SDS-based [1] time sampling strategy
that posits that sampling larger timestep t at the beginning
of the iteration and smaller timestep t later can accelerate
convergence of 3D model generation. Therefore, it introduces
a monotonically non-increasing sampling of timestep t. Specif-
ically, it defines a function W(t) for t, where larger values
indicate that the current t is significant and should be sampled
flatly, while smaller values suggest a steep sampling.
W(t) = 1
Z
r
1 −αt
¯αt
e−(t−m)2
2s2
,
(5)

<!-- page 5 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
5
“a modern living room”
Open-ended Input
Interactive Dialogue Input
“I want to create a living room.”
“Which style do you prefer:
modern, classical, or minimalist?”
“Would you like to add any 
decorations like plants or lamps?”
“Modern, with a large sofa and a 
TV on the wall.”
“Yes, place a plant next to the TV.”
…
Scene Object Analysis
🧠“Determine appropriate 
objects for this scene.”
🧠“Estimating the number 
and size of each object.”
🧠“Inferring the style of each object. 
Generate the object/scene prompt.”
2 sofas : 2.0 m×0.8 m×1.0 m
“a modern gray sofa”
TV stand
sofa
TV
plant
table
Constraint Graph
Scene
opposite
next
front
GCP Algorithm
Layout Constraint Inference
🧠“Inferring the placement of 
each object in the scene.”
🧠“Inferring spatial relations 
between objects.”
TV → side
potted plant → corner
coffee table → center
sofa ⟷opposite ⟷TV
TV → over → TV stand
plant ⟷next to ⟷TV
“Placing objects while ensuring a 
collision-free layout.”
Graph +Placement Strategy + AABB
Valid Layout
→
🧠Reasoning Source  
User dialogue  
Real-world priors 
Fig. 3.
Overview of the Scene Planning process. Given either an open-ended prompt or an interactive dialogue, a GPT-4 agent infers object categories,
real-world sizes, textual prompts, spatial placements, and inter-object relations. These constraints are used to plan the layout through a constraint graph and
GCP algorithm. The resulting arrangement provides a physically plausible and semantically coherent layout that supports environment generation.
where Z = PT
t=1
q
1−αt
¯αt e−(t−m)2
2s2
, s and m are hyper-
parameters. In fact, such timestep t sampling can indeed
increase the model’s convergence speed, but it has a little
impact on the improvement of 3D representation quality.
3D Gaussian Splatting [18], [51] represents a novel approach
in 3D reconstruction. It involves a 3D Gaussian defined by a
comprehensive 3D covariance matrix Σ which is established
in the world space and centered at a specific point, known as
the mean µ:
G(x) = e−1
2 xT Σ−1x,
(6)
spherical harmonics(SH) coefficients and opacity α. By im-
plementing interlaced optimization and density control of
these 3D Gaussians, particularly through the tuning of the
anisotropic covariance, we can get highly accurate recon-
struction representations. Additionally, a tile-based rendering
strategy is utilized to facilitate efficient anisotropic splatting,
which not only speeds up the training process but also enables
real-time rendering capabilities.
IV. METHOD
We present an end-to-end framework DreamScene for auto-
matic 3D scene generation from natural language inputs, sup-
porting both direct descriptions and interactive dialogues. The
system jointly infers object/scene semantics, spatial layout,
and stylistic consistency, and produces high-quality scenes
with scene-wide consistency and flexibility for editing.
The generation process begins with a Scene Planning mod-
ule, where a GPT-4 agent infers object categories, real-world
sizes, detailed textual descriptions y, spatial relations, and
region-level placement anchors. It constructs a hybrid con-
straint graph and applies a graph-based constraint placement
(GCP) algorithm to produce a structured, collision-free object
arrangement, from which we derive the affine transformation
parameters for each object, including scaling s, rotation r,
and translation t. Each object is subsequently generated us-
ing Formation Pattern Sampling (FPS), conditioned on the
corresponding description y. FPS incorporates multi-timestep
sampling (MTS), 3D Gaussian filtering, and reconstructive
generation, enabling the rapid synthesis of high-quality 3D
content using a minimal number of Gaussians. For environ-
ment generation, we first initialize cuboid 3D Gaussians to
represent indoor elements such as walls, floors, and ceilings,
and hemispherical Gaussians for outdoor backgrounds like
ground and distant surroundings. We then place each of the
N generated objects into the global scene coordinate system
using the predicted affine transformations:
world(xi) = ri · si · oi(x) + ti, i = 1, ..., N,
(7)
where xi denotes the coordinates of all 3D Gaussians belong-
ing to object i. Finally, we implement a camera sampling
strategy to guide the three-stage optimization of the envi-
ronment, ensuring scene-wide 3D consistency and mitigating
common scene-level issues such as “multi-headed” layouts,
where identical objects (e.g., sofas) appear redundantly across
multiple directions. Our framework further supports structure-
aware 3D scene editing, including object-level relocation
via affine transformation updates and flexible modification
of scene content using our editing optimization algorithm
Additionally, we extend the editing capability to the temporal
dimension, enabling 4D scene editing with controllable object
motion over time.
A. Scene Planning
To support the goal of DreamScene, which aims to gen-
erate diverse and open-domain 3D scenes, we adopt GPT-
4 [30] as the core reasoning agent for scene planning. Unlike
methods [46], [47] constrained by specific indoor datasets, our
approach requires the ability to infer rich world knowledge, re-
solve spatial relationships, and generate layout-aware prompts
across a wide range of scenes.
As illustrated in Fig. 3, user input can take the form
of either an open-ended description (e.g., “a modern living
room”) or an interactive dialogue where the agent proactively
queries preferences, such as style or functional constraints.
These interactions form a contextual history that, together with
commonsense priors, guides the generation of all downstream
prompts. Specifically, we prepend each GPT-4 query with

<!-- page 6 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
6
the phrase “Based on the user history dialogue and real-
world priors” to ensure that the generated descriptions and
layouts satisfy user intent and adhere to real-world spatial and
functional constraints.
1) Scene Object Analysis: Based on the user dialogue and
scene intent, the GPT-4 agent first infers a list of candidate
objects that are likely to appear in the scene as shown in
Fig. 3. For each object, it predicts the category, count, real-
world size, and a fine-grained textual description yi. These
descriptions capture both functional roles (e.g., “a low wooden
coffee table”) and stylistic attributes (e.g., “a modern gray
sofa”) inferred from the dialogue and high-level scene goal.
To guide this process, we design a structured prompt that
instructs GPT-4 to act as a professional scene designer and
return object-level information in JSON format. The output
includes the number of instances, physical dimensions (in
meters), and a descriptive caption starting with “A DSLR photo
of” to encourage photorealistic generation. The full prompt
template and an example are provided in the supplementary
material. To reduce computational cost, we generate one object
instance per category and replicate it according to the predicted
count. To introduce diversity, these replicas can be associated
with slightly varied prompts, allowing the system to produce
stylistic variations of the same object type without regenerating
the geometry from scratch.
2) Layout Constraint Inference: To obtain plausible and
controllable spatial layouts, we prompt the GPT-4 agent to in-
fer layout constraints from the object list O = {o1, o2, ..., oN}
. This includes two levels of constraint generation: (1) object-
to-scene region anchors Ai and (2) object-to-object spatial
relations. These constraints serve as soft guidance for down-
stream layout search, enabling position reasoning without
relying on supervised 3D layout annotations.
For region anchoring A , we divide the scene into coarse
semantic zones. In indoor scenes, these include center, side,
corner, and others, while in outdoor scenes we exclude the
corner zone due to the lack of enclosing structure. The GPT-
4 agent is prompted to assign each object to an appropriate
zone based on its name, function, and contextual relevance to
the scene. For example, coffee tables are typically centered
in a living room, while plants or shelves may be placed at
the periphery or in corners. A visual illustration of the region
definitions for an indoor scene is shown in the top-right corner
of Fig. 3. To enhance the plausibility of object placements, we
further query GPT-4 to infer pairwise spatial relations among
objects using a limited relation set: left, right, front, back,
over, under, next and opposite. These relations are simple yet
expressive, capturing typical scene configurations such as “TV
opposite sofa” or “lamp next to table.” The prompt templates
used to generate both region anchors and object relations are
provided in the supplementary material.
3) Constraint-based Layout Generation: Given the layout
constraints inferred by GPT-4, we construct a constraint graph
G where nodes V represent objects and edges E encode pair-
wise spatial relations. To realize a plausible and collision-free
layout, we propose a graph-based constraint placement (GCP)
algorithm, as shown in Algorithm 1, which incrementally
assigns object positions and rotations within the scene. We
Algorithm 1 Graph-based Constraint Placement
Require: Object set O = {o1, o2, ..., oN}
1: Constraint graph G = (V, E) with spatial relations
2: Anchor region Ai and real-world size for each object oi
Ensure: Position/Translation {ti}, rotation {ri}, and scaling
{si} for all objects
3: Compute scaling factor si as the ratio between real-world
size and model size for each oi
4: Initialize candidate positions Ci from anchor region Ai
5: Select anchor object oa (e.g., most connected or central)
6: Estimate initial rotation ra based on anchor orientation
7: Initialize placement queue Q ←[oa] and mark oa as
placed
8: while Q not empty do
9:
Pop object oi with known position ti and rotation ri
10:
for each unplaced neighbor oj of oi in G do
11:
Retrieve spatial relation rij from G
12:
Use ti and ri to infer oj’s directional constraint
13:
Filter Cj to satisfy rij and avoid AABB collisions
14:
if Cj is not empty then
15:
Select tj ∈Cj, infer rj accordingly
16:
Mark oj as placed, enqueue oj into Q
17:
else
18:
Defer placement of oj
19:
end if
20:
end for
21: end while
22: for each unplaced object ok do
23:
Assign fallback position tk and estimate rk heuristi-
cally
24: end for
25: return {ti, ri, si} for all oi
begin by computing the scaling factor si for each object oi,
defined as the ratio between its real-world dimensions and the
default size of its generated 3D model. This ensures correct
physical scale in the scene and provides a reliable basis for
collision checking. Based on the region anchors Ai, we sample
a set of candidate positions Ci for each object on a discretized
spatial grid. We then select an anchor object oa, typically
the one with the most relational connections, and initialize
its rotation ra according to its anchor direction (e.g., facing
the center if placed at the boundary). Object rotations serve as
the spatial reference frame to resolve directional constraints
such as left, front, or opposite. Starting from oa, we propagate
placements through the graph in a breadth-first manner. For
each neighboring object oj, we use the relation rij and the
current object’s pose to filter valid candidates from Cj, retain-
ing only those that satisfy the directional constraint and avoid
AABB collision. If such candidates exist, we assign one based
on simple heuristics such as proximity to anchor or alignment
with room center, and infer rj accordingly; otherwise, we
defer placement. After traversal, deferred objects are assigned
fallback positions, and their rotation is estimated based on
nearby anchors or previously placed objects. The final output
of this process is a complete layout specification {ti, ri, si}

<!-- page 7 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
7
for each object.
The resulting layout aligns with real-world spatial logic and
provides a strong structural prior for downstream environment
generation, effectively mitigating multi-headed arrangements
in the scene.
B. Formation Pattern Sampling
We have enhanced and expanded the concept of employ-
ing monotonically non-increasing sampling of timestep t in
DreamTime [6]. Our research indicates that developing high-
quality, semantically rich 3D representations greatly benefits
from integrating information across multiple timesteps at
each iteration of a pre-trained text-to-image diffusion model.
This approach stands in contrast to other methods using
Score Distillation Sampling [1](SDS), which typically rely
on information from a single timestep during each iteration.
In the optimization’s early to mid stages, which target the
initial shaping of forms, a decremental time window Tend
is implemented, linearly reducing through iterations. This
window is segmented into m intervals; within each t is
randomly selected for gradient aggregation. Although this
method quickly produces rich semantic 3D representations,
it may also generate unnecessary massive 3D Gaussians. To
counter this, we employ 3D Gaussian filtering, selectively
sampling critical surface Gaussians only. In later optimization
stages, to make the surface textures of representations more
plausible, we sample t from a range between 0 and 200 using
3D reconstruction techniques [18] to expedite this process.
Since this method for generating 3D representations follows
the patterns of 3D model development, sampling different time
steps t in various iterations and targeting 3D Gaussians on
the model’s surface, we aptly named it Formation Pattern
Sampling (FPS).
To capture the varied information offered by the 2D text-
to-image diffusion model across timestep t ranging from
0 to 1000, we utilize pseudo-Ground-Truth(pseudo-GT) im-
ages generated from a single denoising step within Lucid-
Dreamer [8]. By introducing noise across t timestep into the
images x0 to generate xt , we calculate the pseudo-GT ˆxt
0
using the following equation:
ˆxt
0 = xt −√1 −¯αtϵϕ(xt; y, t)
√¯αt
.
(8)
1) Multi-timestep Sampling: As illustrated in Fig. 5 (a),
we observe that at smaller timestep t, the 2D diffusion model
produces detailed and realistic surface textures that align well
with the current 3D shape, but lack comprehensive semantic
information from the prompt y. Conversely, at a larger timestep
t, the model provide richer semantic details, though these may
not conform to the existing 3D shape(discrepancies in the
orientation of the man, the color of the chair, or the direction
of a cooker between timestep 600 and 800).
To address this, we suggest blending information from mul-
tiple timesteps in each iteration of a 2D diffusion model. This
integration aims to maintain shape accuracy while enhancing
semantic information. For example, during the 300-th iteration
for the man in Fig. 5 (a), we utilize timesteps 200 to 400 for
ECCV
TPAMI
Fig. 4. Comparison of the generation quality between the ECCV version and
the TPAMI version of DreamScene.
shape accuracy, while timesteps 400 to 600 and 600 to 800
enrich the semantic context. However, by the 1000-th iteration
for the cooker, we note that the shape already encapsulates suf-
ficient semantic details, and incorporating further information
from a larger timestep might detract from the optimization
process. So the timestep t for i-th sample can be described as
follows:
ti = T iter
end · random(i −1
m , i
m), i = 1, ..., m,
(9)
where Tend represents a linearly decreasing time window, akin
to the approach used in DreamTime [6], iter indicates the
current iteration, and m specifies the number of intervals.
Some studies [8], [52] have found that using ordinary dif-
ferential equation(ODE) processes in sampling can ensure a
certain level of consistency. Naturally, combining our multi-
step consideration, we use DDIM Inversion to calculate xti
between t1 and tm:
xti+1 = √¯αti+1
xti−√
1−¯αtiϵϕ(xti;∅,ti)
√
¯αti
+ p1 −¯αti+1ϵϕ(xti; ∅, ti), (10)
where ∅represents the empty prompt.
Therefore, the combination of MTS and CSD [7] method
can be articulated as follows:
∇θLMTS(θ) = Et,ϵ,c
 m
P
i=1
w(ti)(ϵϕ(xti; y, ti) −ϵϕ(xti; ∅, ti)) ∂g(θ,c)
∂θ

. (11)
Although MTS is initially motivated by empirical ob-
servations across diffusion timesteps, we further provide a
theoretical explanation by linking it to trajectory alignment
in 2D editing methods [39], [53]. In addition, we reduce
the estimation error within MTS, which leads to improved
generation quality as shown in Fig. 4. Details are presented
in the supplementary material. Details are presented in the
supplementary material.
2) 3D Gaussian Filtering: Excessive 3D Gaussians can im-
pede the optimization process. Unlike traditional methods [54],
[55] that use ground truth images to filter reconstructed 3D
Gaussians, our strategy requires filtering to be integrated into
the optimization phase. Regarding rendering, 3D Gaussians
located nearer to the rendering plane have a more pronounced
effect, for which a specialized score function is utilized to
evaluate their impact. For 3D Gaussians along the rendering
ray rj, their contributions are assessed based on the inverse
square of their distance to the rendering plane, factoring in
the 3D Gaussians’ volume. This technique prioritizes 3D
Gaussians that are both closer to the rendering plane and have
a larger volume, as illustrated in Fig. 5 (b). By scoring various
viewpoints, we can effectively discard 3D Gaussians that do
not meet a set threshold.
Score(i) =
H×W ×M
X
j=1
V (i)
D(rj, i)2 × maxV (rj),
(12)

<!-- page 8 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
8
(a)
(b)
(c)
Fig. 5. Formation Pattern Sampling. (a)Multi-timestep Sampling. At varying
timesteps, the 2D text-to-image diffusion model provides different informa-
tion(represented by the pseudo-GT ˆxt
0) obtained from xt in a single-step
by Eq. 8 in LucidDreamer [8]. (b)3D Gaussian Filtering. 3D Gaussians
that are located closer to the rendering plane and possess larger volumes
make a greater contribution to the rendering process. (c)Reconstructive
Generation. During the later stages of optimization, generation can be directly
accomplished using reconstruction based on denoised images, leading to 3D
representations with refined and plausible textures.
where H and W indicate the height and width of the rendered
image, respectively, M represents the number of rendered
images, V (i) is the volume of the i-th 3D Gaussian(calculated
using the covariance matrix), maxV (rj) is the maximum
volume of the 3D Gaussians on rj, and D(rj, i) represents
the distance of the i-th 3D Gaussian from the rendering plane
along the rj. It’s important to note that this procedure is
designed to simulate the rendering process rather than perform
actual rendering.
3) Reconstructive Generation: We can use 3D reconstruc-
tion techniques to accelerate the creation of realistic surface
textures [18]. We observed that when sampling very small
timestep t(ranging from 0 to 200), the image predicted by
Eq. 8 maintains the same 3D shape as the input image x0
but reveals more detailed and plausible textures. Thus, to
maintain shape consistency, we directly generate a new 3D
representation via 3D reconstruction [18]. As depicted in
Fig.5 (c), after achieving a coarse texture but rich semantic
3D structure, we render K images xi, for i = 1, ..., K from
various camera poses ci around the 3D representation. By
adding t timestep of noise to these images to obtain xit using
Eq.1, we estimate the images ˆxt
i0 with plausible textures by
step1
step2
step3
Indoor
Outdoor
Fig. 6. Schematic diagram of camera sampling in environment generation.
Eq.8. We then reconstruct them onto the coarse representation
by minimizing the following reconstruction loss:
Lrec =
X
i
||g(θ, ci) −ˆxt
i0||2.
(13)
This process efficiently generates a representation featuring
detailed and plausible textures within 15 seconds.
C. Camera Sampling
Camera sampling is the primary strategy for environmental
generation within a scene. Before this, it is necessary to place
objects generated in the previous step into the scene based on
coordinates derived from Eq. 7. This approach prevents the
”multi-head” phenomenon in scene generation, where cameras
in other methods [22], [23], [25], [27] cannot truly perceive
orientation, resulting in similar content generated from the
same textual prompts at various angles. Consequently, in living
rooms generated by some methods [22], [23], [25], [27], a
sofa appears in every direction as shown in Fig. 9. Utilizing
the human prior knowledge embedded in GPT-4 [30], we
have prearranged the layout, ensuring that the rendered scene
environment images with information about different objects
from different angles, thereby allowing the camera to perceive
the room’s orientation.
To maintain high quality in scene generation, existing ap-
proaches [21], [23], [25], [27], [56] typically restrict camera
sampling to a narrow range, which does not provide compre-
hensive coverage of the scene-wide observations. Employing
simple random camera sampling throughout the scene can lead
to the breakdown of scene generation during optimization. In
response, we have developed a structured, incremental three-
step camera sampling strategy, illustrated in Fig. 6:
In the initial stage, we create a basic representation of the
surrounding environment, focusing on indoor walls and distant
outdoor elements. We lock the parameters of the 3D Gaussians
for the ground and objects, limiting camera sampling to
coordinates within a certain proximity to the center, to refine
the generation of these surroundings.
During the second stage, our focus shifts to generating
the coarse ground. At this point, the parameters for the 3D
Gaussians representing environments and objects are frozen.
For indoor scenes, the space is segmented into distinct regions
based on object placement. Camera poses are strategically
sampled to target key areas, including objects and the ground,
in each iteration. For outdoor scenes, the area is divided into
concentric circles determined by their radius. A consistent
direction is selected for sampling camera poses around these

<!-- page 9 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
9
circles in each iteration, enhancing ground generation. This
method ensures thorough coverage of the entire ground area,
with a particular focus on zones where the ground meets
objects and the surrounding environment.
In the third stage, we utilize all previously sampled camera
poses to ensure a comprehensive view of the entire scene,
focusing on refining all environmental elements. This includes
meticulous optimization of parameters for both the ground
and the surrounding features. Building on the 3D consistency
achieved in earlier two sreps, we then move to the reconstruc-
tive generation method in Sec. IV-B3 aimed at acquiring more
detailed and plausible textures for the scene.
Camera positions might be obstructed by objects within the
scene, requiring collision detection between the camera and
these objects. If collisions are detected, the affected camera
positions should be discarded to ensure clear visibility.
D. Scene Editing
Thanks to compositional scene generation strategy [21],
[33], [57], DreamScene supports flexible and fine-grained
editing of individual objects or environmental elements (e.g.,
walls, floors, ground), enabling the construction of new scenes
through targeted modifications. We organize editing capabil-
ities into three complementary operations: object relocation,
appearance modification, and temporal movement.
Object Relocation. We enable editing by adjusting the ob-
ject’s affine transformation parameters (s′, t′, r′), which con-
trol its scale, position, and orientation, respectively. These
parameters can be updated without regenerating geometry,
allowing fast and lightweight manipulation. Users may provide
explicit coordinates or high-level spatial commands (e.g.,
“move the man backward,” “rotate the chair to face the
TV”), which are translated into updated affine parameters. For
minor adjustments, such as repositioning a single object, we
directly apply the new parameters and verify collision-free
placement using simple AABB collision detection. In cases
where multiple objects are significantly repositioned or layout
structure is altered, we re-invoke the scene planning module to
re-evaluate spatial constraints and update relationships among
objects. To maintain scene plausibility, we also sample new
camera poses around the relocated objects and re-optimize
the local environment (e.g., floor textures or wall geometry)
accordingly. This ensures that the resulting scene remains
consistent, context-aware, and physically valid after editing.
Similarly, when adding a new object, we assign it a valid
location using the same constraint-based reasoning. For object
removal, we just simply clear its position.
Appearance Modification. To support high-fidelity object
editing, we enable appearance modifications that span both
texture geometry refinements. Instead of regenerating the
object from scratch, we preserve its existing 3D Gaussians
and re-optimize appearance and positional parameters under a
new textual description yedit.
We directly adapt the 2D editing process into our MTS
method for 3D appearance editing. Traditional 2D editing
methods typically consist of two stages: image reconstruction
and image editing. In the reconstruction stage, methods such
Fig. 7. Diversity of layout generation.
as NTI [39] and PTI [53] gradually align the latents in the
diffusion process to obtain accurate noising and denoising tra-
jectory for the input image. Then, during the editing stage, they
inject the target prompt yedit into the denoising trajectory to
guide generation. In our MTS setting, we adopt the same idea
on random rendered images in each optimazation. Specifically,
we approximate the noising trajectory using DDIM inversion
in Eq. 10 and denosing trajectory using DDIM, just replacing
the empty prompt ∅with the current object prompt y to obtain
an approximate reconstruction trajectory (x represnts the latent
in the noising trajectory and ˜x represent the latent in the
denoising trajectory):



xti+1 = √¯αti+1
xti−√
1−¯αtiϵθ(xti;y,ti)
√
¯αti
+ p1 −¯αti+1ϵθ(xti; y, ti)
˜xti = √¯αti
˜xti+1−√1−¯αti+1ϵθ(˜xti+1;y,ti+1)
√¯αti+1
+ √1 −¯αtiϵθ(˜xti+1; y, ti+1).
(14)
Then we directly replace y with yedit in the denoising process
to simulate the 2D editing behavior. This leads to the following
MTS-based editing equation:
∇θLMTS Editing(θ) =
Et,ϵ,c
 m
P
i=1
w(ti)(ϵϕ(xti; yedit, ti) −ϵϕ(xti; y, ti)) ∂g(θ,c)
∂θ

,
(15)
, and this can be viewed as guiding the optimization to move
away from the original semantics encoded in y, and toward
those specified by the target prompt yedit.
Temporal Movement. To support 4D scene generation with
dynamic object motion, we extend the 3D Gaussian repre-
sentation by introducing a temporal dimension. For static
elements such as walls, floors, or backgrounds, the Gaus-
sian parameters remain constant over time. In contrast, for
dynamic objects, we apply time-dependent affine transforma-
tions (si(t), ri(t), ti(t)) to adjust their position, orientation,
and scale at each time step. Given an animation description
from the user (e.g., “the man walks from left to right”), a
GPT-4 agent automatically generates a discrete sequence of
affine transformations that simulates a continuous trajectory,
reflecting the intended motion. This mechanism expands the
capability of DreamScene, enabling its application to tasks
such as animation creation and virtual environment simulation.

<!-- page 10 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
10
Text2Room
(~13.3h)
Outdoor Scene: “A DSLR photo of an autumn park.”
Text2Nerf
(~7.5h)
ProlificDreamer
(~12h)
Set-the-Scene
(~1.5h)
Ours
(~1.5h)
Our Depth
Fig. 8. Visual consistency and generation quality under diverse scene-wide camera poses in the outdoor scenes.
V. EXPERIMENT
Implementation Details. We employ GPT-4 [30] as our Large
Language Model(LLM) for decomposing scene prompts and
Point-E [11] for generating initial sparse point clouds of
objects. For 2D image generation, we use Stable Diffusion
2.1. The maximum number of iterations for objects is set at
1,500, and for the environment, it is 2,000. The value of the
time interval m is 4. In the reconstructive generation phase,
we generate 20 rendering images. To ensure a fair comparison,
we tested DreamScene and all baselines on the same NVIDIA
3090 GPU.
Baselines. For the comparative analysis of text-to-3D scene
generation, we utilize the current open-sourced state-of-the-
art(SOTA)
methods
as
our
baselines:
Text2Room
[22],
Text2NeRF
[25],
ProlificDreamer
[27],
and
Set-the-
Scene [21]. In the domain of text-to-3D generation, our
selected
baselines
are
DreamFusion
[1],
Magic3D
[2],
DreamGaussian [9], and LucidDreamer [8](ProlificDreamer,
DreamFusion and Magic3D have been reimplemented by
Three-studio [58]).
Evaluation Metrics. We assessed the generation time for each
method [1], [2], [8], [9], [21], [22], [25], [27] and compared
the editing capabilities outlined in their respective published
papers. We use R-Precision(same setting in DreamTime [6])
to calculate the similarity between the rendered image of
the generated 3D representation and the text description.
Additionally, we conducted a user study with 100 participants,
where each one rated the quality, consistency, and rationality of
the videos on a scale from 1 to 5. These 30-second videos were
generated by each method across five different scenes—three
indoor and two outdoor.
A. Qualitative Results
Layout Generation. We believe that layout generation should
be diverse, as illustrated in Fig. 7 , which showcases various
layouts for an outdoor park and an indoor bedroom. In

<!-- page 11 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
11
Text2Room
(~13.3h)
Indoor Scene:
Text2Nerf
(~7.5h)
ProlificDreamer
(~12h)
Set-the-Scene
(~1.5h)
Ours
(~1.5h)
Our Depth
“A DSLR photo of a modern living room.”
Fig. 9. Visual consistency and generation quality under diverse scene-wide camera poses in the indoor scenes.
Fig. 10. Comparison with baselines in text-to-3D object generation.
the DreamScene layout generation process, the use of GPT-
4’s question-and-answer capability results in varied responses
each time, although some elements, like the fountain often
being at the center of the park, may be consistent. Additionally,
during the object placement stage, varying the search order
and placement settings (such as centering or edge positioning
within an area) contributes to the creation of diverse layouts.
Scenes and objects generation. To evaluate scene-wide 3D
consistency and generation quality, we conduct comparisons
between DreamScene and representative baselines under di-
verse camera poses. To ensure fairness, we follow each base-
line’s official camera configurations during training. During
testing, we adopt a unified camera trajectory for all methods:
the camera first moves in some straight lines across the scene,
then circles around the scene center, simulating natural human
exploration behavior. It can be observed that Text2Room [22],
Text2NeRF [25], and ProlificDreamer [27] exhibit poor gen-
eralization to novel poses. Even minor viewpoint shifts often
lead to severe distortions or structural collapse, indicating a
lack of true 3D consistency. In contrast, Set-the-Scene [21],
which shares a similar modular scene composition philosophy
with DreamScene, achieves relatively stable structure under
indoor settings. However, due to its reliance on conventional
SDS [1] optimization, the visual quality is significantly lower
and fails to generalize to outdoor scenes. In comparison,
as shown in RGB and depth results, DreamScene generates
complete 3D structure, with the best 3D consistency and visual
quality among all methods. Additional video results and depth

<!-- page 12 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
12
origin
add a man
delete the bed
relocate the objects
cyberpunk style
Ukiyo-e style
minecraft style
(a)
(b)
t=2
t=3
t=4
(c)
t=1
view 1
view 2
origin
Fig. 11.
DreamScene editing results. (a) shows object-level edits, including
relocation, addition, and removal. (b) demonstrates style modifications applied
to both objects and environments. (c) presents the 4D generation results from
multiple viewpoints.
TABLE I
QUANTITATIVE RESULTS OF DREAMSCENE COMPARED WITH BASELINES.
↑MEANS THE MORE THE BETTER AND ↓MEANS THE LOWER THE BETTER.
Q MEANS ”QUALITY”, C MEANS ”CONSISTENCY” AND R MEANS
”RATIONALITY”.
Method
Time
Editing
User Study
(hours) ↓
Q↑
C↑
R↑
Text2Room [22]
13.3
✗
2.93
2.57
2.60
Text2NeRF [25]
7.5
✗
3.05
2.71
2.98
ProlificDreamer [27]
12.0
✗
3.48
3.19
2.95
Set-the-Scene [21]
1.5
✓
2.45
3.52
2.88
Ours
1.5
✓
3.92
4.24
4.05
maps from other methods are provided in the supplementary
material. Fig. 10 reveals that our FPS is capable of producing
high-quality 3D representations in a brief period, adhering
to the text prompts. Although DreamGaussian [9] produces
results more quickly, it sacrifices the generation quality.
B. Quantitative Results
Compare with text-to-3D scene methods. To ensure a fair
comparison, we calculate the generation time of our environ-
ment generation stage, as the baseline methods [22], [25],
[27] cannot generate objects in the environment indepen-
dently. The left side of Tab. I demonstrates that our method
achieves the shortest generation time for environments with
editing capabilities. The right side presents results from a
user study, where DreamScene significantly outperforms the
baseline methods [21], [22], [25], [27] in terms of consistency
and rationality, while maintaining high generation quality.
TABLE II
QUANTITATIVE RESULTS OF DREAMSCENE COMPARED WITH
DREAMTIME. ↑MEANS THE MORE THE BETTER.
R-Precision↑
Ours
Ours(w/o annealing)
3DGS+DreamTime
ViT-L/14
71.9%
71.9%
34%
ViT-BigG/14
70.6%
68.6%
33.3%
Fig. 12. Ablation results of time window strategy in MTS.
Compare with DreamTime. We use the same evaluation
settings as DreamTime [6] to demonstrate that our sampling
strategy not only accelerates convergence but also significantly
enhances the quality of generation. As illustrated in Tab. II ,
our approach yields better results in terms of CLIP R-Precision
after the same 1500 iterations. Additionally, it is observed that
the annealing strategy for the time window T slightly affects
the result of generation.
C. Scene Editing
Fig. 11 showcases the flexible editing capabilities of Dream-
Scene, as discussed in Sec. IV-D. DreamScene supports object-
level relocation, addition, and removal by adjusting the ob-
ject’s affine transformation parameters. During these edits,
we resample camera poses at both the original and updated
locations to maintain visual consistency. As illustrated in
Fig. 11(b), modifying the text prompts enables changes in
object appearance or environmental style via Eq. 15. Further-
more, as shown in Fig. 11(c), by adding temporal control to
the affine transformations, we enable continuous object motion
over time, achieving 4D generation. This process also allows
multi-view observations of dynamic scenes.
D. Ablations
Time window strategies in MTS. As illustrated in Fig. 12,
the first image demonstrates the result of using fixed-step
sampling in MTS rather than random sampling within the
time interval. This strategy resulted in notably low quality of
generation. Other images depict different strategies for setting
time windows in MTS: maintaining a fixed maxstep of 1000,
employing the strategy used in Eq. 5, and using a linearly
decreasing strategy. We found that the linearly decreasing
strategy outperforms the others. As discussed in Sec. IV-B,
large timesteps t provide valuable semantic information. How-
ever, in DreamTime, there are very few sampling points at
large t. In the later stages of optimization, large t may mislead
the optimization direction and result in suboptimal surface
outcomes, as seen in the ”non-decreasing” strategy.
3D Gaussian filtering. The method we propose is specifically
designed for optimization tasks and can be directly applied
to reconstruction tasks [18] as well. Fig. 13 illustrates the
outcomes of both reconstruction and generation tasks before

<!-- page 13 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
13
ground truth
after filtering
371MB
97MB
before filtering
183MB
after filtering
62MB
(a)
(b)
Fig. 13. Ablation results of 3D Gaussian filtering algorithm in reconstruction
and generation tasks. (a) Data in NeRF-360 [16]. (b) Data in generating
process.
(a)
(b)
(c)
Fig. 14.
The ablation results of various camera sampling strategies. (a)
Randomly camera sampling. (b) No distinction between environment and
ground. (c) DreamScene three-step camera sampling strategy
and after using the Gaussian filtering algorithm for compres-
sion. In the reconstruction task, our method reduced 73.9%
memory consumption for storing 3D Gaussians, at the cost of
a slightly blurred image with some loss of detail. Conversely,
in the generation task, the compression resulted in a 66.1%
reduction, with no significant loss of quality.
Camera sampling. Fig. 14 (a) depicts a scene generated by
randomly sampling camera positions within the scene. Due to
the challenges in maintaining consistency of scene-wide views
at the same location, the optimization process often tends to
collapse. Fig. 14 (b) adopts a strategy that progresses from the
center to the periphery, where the environment and ground are
not distinguished. This approach results in improved scene
consistency, but the integration between the ground and the
scene is poorly executed, and the ground is prone to being
populated with coarse 3D Gaussians. Fig. 14 (c) showcases our
three-step strategy, which significantly enhances the quality
of generation while ensuring the consistency of both the
surrounding environment and the ground.
VI. CONCLUSION AND FUTURE WORK
We propose DreamScene, an end-to-end framework for
generating 3D scenes from natural language. The process
starts with a scene planning module, where a GPT-4 agent
predicts object categories, sizes, descriptions, and spatial re-
lations to build a constraint graph. Based on this, we place
objects into the scene with a layout algorithm that ensures
reasonable structure and avoids collisions. Then, we generate
object geometry using Formation Pattern Sampling, and refine
the scene using a three-stage camera sampling strategy for
better consistency. DreamScene also supports scene editing,
including moving, adding, or removing objects, changing style,
and controlling object motion over time. Our experiments
show that DreamScene can generate consistent, realistic, and
editable 3D scenes, making it suitable for a wide range of
applications such as VR/AR, Metaverse and simulation.
In future work, we plan to enhance the scene planning pro-
cess by capturing more realistic spatial relationships, including
fine-grained object placement such as arranging small items
on shelves. We also aim to extend the framework to model
complex 4D dynamics, including both object-level motion and
global scene evolution over time.
REFERENCES
[1] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, “Dreamfusion: Text-
to-3d using 2d diffusion,” arXiv preprint arXiv:2209.14988, 2022.
[2] C.-H. Lin, J. Gao, L. Tang, T. Takikawa, X. Zeng, X. Huang, K. Kreis,
S. Fidler, M.-Y. Liu, and T.-Y. Lin, “Magic3d: High-resolution text-to-
3d content creation,” in Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2023, pp. 300–309.
[3] R. Chen, Y. Chen, N. Jiao, and K. Jia, “Fantasia3d: Disentangling
geometry and appearance for high-quality text-to-3d content creation,”
arXiv preprint arXiv:2303.13873, 2023.
[4] R. Liu, R. Wu, B. Van Hoorick, P. Tokmakov, S. Zakharov, and C. Von-
drick, “Zero-1-to-3: Zero-shot one image to 3d object,” in Proceedings
of the IEEE/CVF International Conference on Computer Vision, 2023,
pp. 9298–9309.
[5] G. Metzer, E. Richardson, O. Patashnik, R. Giryes, and D. Cohen-Or,
“Latent-nerf for shape-guided generation of 3d shapes and textures,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 12 663–12 673.
[6] Y. Huang, J. Wang, Y. Shi, X. Qi, Z.-J. Zha, and L. Zhang, “Dreamtime:
An improved optimization strategy for text-to-3d content creation,” arXiv
preprint arXiv:2306.12422, 2023.
[7] X. Yu, Y.-C. Guo, Y. Li, D. Liang, S.-H. Zhang, and X. Qi, “Text-to-
3d with classifier score distillation,” arXiv preprint arXiv:2310.19415,
2023.
[8] Y. Liang, X. Yang, J. Lin, H. Li, X. Xu, and Y. Chen, “Luciddreamer:
Towards high-fidelity text-to-3d generation via interval score matching,”
arXiv preprint arXiv:2311.11284, 2023.
[9] J. Tang, J. Ren, H. Zhou, Z. Liu, and G. Zeng, “Dreamgaussian:
Generative gaussian splatting for efficient 3d content creation,” arXiv
preprint arXiv:2309.16653, 2023.
[10] W. Li, R. Chen, X. Chen, and P. Tan, “Sweetdreamer: Aligning geo-
metric priors in 2d diffusion for consistent text-to-3d,” arXiv preprint
arXiv:2310.02596, 2023.
[11] A. Nichol, H. Jun, P. Dhariwal, P. Mishkin, and M. Chen, “Point-e: A
system for generating 3d point clouds from complex prompts,” arXiv
preprint arXiv:2212.08751, 2022.
[12] H. Jun and A. Nichol, “Shap-e: Generating conditional 3d implicit
functions,” arXiv preprint arXiv:2305.02463, 2023.
[13] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchical
text-conditional image generation with clip latents,” arXiv preprint
arXiv:2204.06125, vol. 1, no. 2, p. 3, 2022.
[14] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-
resolution image synthesis with latent diffusion models,” in Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition,
2022, pp. 10 684–10 695.
[15] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton,
K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans
et al., “Photorealistic text-to-image diffusion models with deep language
understanding,” Advances in Neural Information Processing Systems,
vol. 35, pp. 36 479–36 494, 2022.
[16] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi,
and R. Ng, “Nerf: Representing scenes as neural radiance fields for view
synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106,
2021.
[17] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove,
“Deepsdf: Learning continuous signed distance functions for shape rep-
resentation,” in Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, 2019, pp. 165–174.
[18] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian
splatting for real-time radiance field rendering,” ACM Transactions on
Graphics, vol. 42, no. 4, 2023.
[19] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics
primitives with a multiresolution hash encoding,” ACM Transactions on
Graphics (ToG), vol. 41, no. 4, pp. 1–15, 2022.
[20] T. Shen, J. Gao, K. Yin, M.-Y. Liu, and S. Fidler, “Deep marching
tetrahedra: a hybrid representation for high-resolution 3d shape synthe-
sis,” Advances in Neural Information Processing Systems, vol. 34, pp.
6087–6101, 2021.
[21] D. Cohen-Bar, E. Richardson, G. Metzer, R. Giryes, and D. Cohen-
Or, “Set-the-scene: Global-local training for generating controllable nerf
scenes,” arXiv preprint arXiv:2303.13450, 2023.

<!-- page 14 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
14
[22] L. H¨ollein, A. Cao, A. Owens, J. Johnson, and M. Nießner, “Text2room:
Extracting textured 3d meshes from 2d text-to-image models,” arXiv
preprint arXiv:2303.11989, 2023.
[23] H. Ouyang, K. Heal, S. Lombardi, and T. Sun, “Text2immersion:
Generative
immersive
scene
with
3d
gaussians,”
arXiv
preprint
arXiv:2312.09242, 2023.
[24] H. Li, H. Shi, W. Zhang, W. Wu, Y. Liao, L. Wang, L.-h. Lee, and
P. Zhou, “Dreamscene: 3d gaussian-based text-to-3d scene generation
via formation pattern sampling,” arXiv preprint arXiv:2404.03575, 2024.
[25] J. Zhang, X. Li, Z. Wan, C. Wang, and J. Liao, “Text2nerf: Text-driven
3d scene generation with neural radiance fields,” IEEE Transactions on
Visualization and Computer Graphics, 2024.
[26] R. Po and G. Wetzstein, “Compositional 3d scene generation using
locally conditioned diffusion,” arXiv preprint arXiv:2303.12218, 2023.
[27] Z. Wang, C. Lu, Y. Wang, F. Bao, C. Li, H. Su, and J. Zhu, “Prolific-
dreamer: High-fidelity and diverse text-to-3d generation with variational
score distillation,” Advances in Neural Information Processing Systems,
vol. 36, 2024.
[28] Q. Zhang, C. Wang, A. Siarohin, P. Zhuang, Y. Xu, C. Yang, D. Lin,
B. Zhou, S. Tulyakov, and H.-Y. Lee, “Scenewiz3d: Towards text-guided
3d scene composition,” arXiv preprint arXiv:2312.08885, 2023.
[29] Y. Lin, H. Bai, S. Li, H. Lu, X. Lin, H. Xiong, and L. Wang,
“Componerf: Text-guided multi-object compositional nerf with editable
3d scene layout,” arXiv preprint arXiv:2303.13843, 2023.
[30] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman,
D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4
technical report,” arXiv preprint arXiv:2303.08774, 2023.
[31] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,”
Advances in neural information processing systems, vol. 33, pp. 6840–
6851, 2020.
[32] X. Zhou, X. Ran, Y. Xiong, J. He, Z. Lin, Y. Wang, D. Sun, and M.-H.
Yang, “Gala3d: Towards text-to-3d complex scene generation via layout-
guided generative gaussian splatting,” arXiv preprint arXiv:2402.07207,
2024.
[33] A. Vilesov, P. Chari, and A. Kadambi, “Cg3d: Compositional generation
for text-to-3d via gaussian splatting,” arXiv preprint arXiv:2311.17907,
2023.
[34] K. Lan, H. Li, H. Shi, W. Wu, L. Wang, and Y. Liao, “2d-guided 3d
gaussian segmentation,” in 2024 Asian Conference on Communication
and Networks (ASIANComNet).
IEEE, 2024, pp. 1–5.
[35] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla,
and P. P. Srinivasan, “Mip-nerf: A multiscale representation for anti-
aliasing neural radiance fields,” in Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision, 2021, pp. 5855–5864.
[36] Y. Shi, P. Wang, J. Ye, M. Long, K. Li, and X. Yang, “Mvdream: Multi-
view diffusion for 3d generation,” arXiv preprint arXiv:2308.16512,
2023.
[37] H. Li, Y. Tian, Y. Wang, Y. Liao, L. Wang, Y. Wang, and P. Y. Zhou,
“Text-to-3d generation by 2d editing,” arXiv preprint arXiv:2412.05929,
2024.
[38] T. Yi, J. Fang, G. Wu, L. Xie, X. Zhang, W. Liu, Q. Tian, and X. Wang,
“Gaussiandreamer: Fast generation from text to 3d gaussian splatting
with point cloud priors,” arXiv preprint arXiv:2310.08529, 2023.
[39] R. Mokady, A. Hertz, K. Aberman, Y. Pritch, and D. Cohen-Or, “Null-
text inversion for editing real images using guided diffusion models,”
in Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2023, pp. 6038–6047.
[40] A. Hertz, R. Mokady, J. Tenenbaum, K. Aberman, Y. Pritch, and
D. Cohen-Or, “Prompt-to-prompt image editing with cross attention
control,” arXiv preprint arXiv:2208.01626, 2022.
[41] J. Zhou, X. Li, L. Qi, and M.-H. Yang, “Layout-your-3d: Control-
lable and precise 3d generation with 2d blueprint,” arXiv preprint
arXiv:2410.15391, 2024.
[42] U. Nath, R. Goel, R. Khurana, K. Min, M. Ollila, P. Turaga, V. Jampani,
and T. Gowda, “Decompdreamer: Advancing structured 3d asset gen-
eration with multi-object decomposition and gaussian splatting,” arXiv
preprint arXiv:2503.11981, 2025.
[43] S. Bahmani, J. J. Park, D. Paschalidou, X. Yan, G. Wetzstein, L. Guibas,
and A. Tagliasacchi, “Cc3d: Layout-conditioned generation of com-
positional 3d scenes,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision, 2023, pp. 7171–7181.
[44] Q. Zhang, Y. Xu, Y. Shen, B. Dai, B. Zhou, and C. Yang, “Berfscene:
Bev-conditioned equivariant radiance fields for infinite 3d scene genera-
tion,” in Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 2024, pp. 6839–6849.
[45] D. Paschalidou, A. Kar, M. Shugrina, K. Kreis, A. Geiger, and S. Fidler,
“Atiss: Autoregressive transformers for indoor scene synthesis,” Ad-
vances in Neural Information Processing Systems, vol. 34, pp. 12 013–
12 026, 2021.
[46] R. Fu, J. Liu, X. Chen, Y. Nie, and W. Xiong, “Scene-llm: Extending
language model for 3d visual understanding and reasoning,” arXiv
preprint arXiv:2403.11401, 2024.
[47] Y. Hong, H. Zhen, P. Chen, S. Zheng, Y. Du, Z. Chen, and C. Gan,
“3d-llm: Injecting the 3d world into large language models,” Advances
in Neural Information Processing Systems, vol. 36, pp. 20 482–20 494,
2023.
[48] Y. Wang, S.-Y. Chen, Z. Zhou, S. Li, H. Li, W. Zhou, and H. Li, “Root:
Vlm based system for indoor scene understanding and beyond,” arXiv
preprint arXiv:2411.15714, 2024.
[49] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,”
arXiv preprint arXiv:2010.02502, 2020.
[50] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv
preprint arXiv:2207.12598, 2022.
[51] G. Chen and W. Wang, “A survey on 3d gaussian splatting,” arXiv
preprint arXiv:2401.03890, 2024.
[52] Z. Wu, P. Zhou, X. Yi, X. Yuan, and H. Zhang, “Consistent3d: Towards
consistent high-fidelity text-to-3d generation with deterministic sampling
prior,” in Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 2024, pp. 9892–9902.
[53] W. Dong, S. Xue, X. Duan, and S. Han, “Prompt tuning inversion for
text-driven image editing using diffusion models,” in Proceedings of
the IEEE/CVF International Conference on Computer Vision, 2023, pp.
7430–7440.
[54] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, and Z. Wang, “Lightgaussian:
Unbounded 3d gaussian compression with 15x reduction and 200+ fps,”
arXiv preprint arXiv:2311.17245, 2023.
[55] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, “Compact 3d gaussian
representation for radiance field,” arXiv preprint arXiv:2311.13681,
2023.
[56] I. Hwang, H. Kim, and Y. M. Kim, “Text2scene: Text-driven in-
door scene stylization with part-aware details,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2023, pp. 1890–1899.
[57] H. Li, L. Ma, H. Shi, Y. Hao, Y. Liao, L. Cheng, and P. Y. Zhou, “3d-
goi: 3d gan omni-inversion for multifaceted and multi-object editing,”
in European Conference on Computer Vision. Springer, 2024, pp. 390–
406.
[58] Y.-C. Guo, Y.-T. Liu, C. Wang, Z.-X. Zou, G. Luo, C.-H. Chen, Y.-P.
Cao, and S.-H. Zhang, “threestudio: A unified framework for 3d content
generation,” 2023.
APPENDIX A
THEORETICAL DERIVATION OF MULTI-TIMESTEP
SAMPLING (MTS)
Our Multi-timestep Sampling (MTS) strategy is grounded
in a key empirical observation in diffusion-based generation:
different timesteps encode information at varying levels of
semantic granularity. This motivates the use of multiple de-
noising steps to improve generation quality and optimization
stability. In this section, we present a theoretical analysis
of MTS and establish its connection to diffusion-based 2D
editing methods. This analysis also confirms that MTS is not
a heuristic mechanism, but a principled strategy supported by
the underlying behavior of diffusion models.
1.Derivation and Approximation
We first obtain a latent noisy trajectory xt0, xt1, ..., xtm
using DDIM Inversion as follows:
xti+1 = p¯αti+1
xti −√1 −¯αtiϵθ(xti, ti, ∅)
√¯αti
+
p
1 −¯αti+1ϵθ(xti, ti, ∅),
(16)

<!-- page 15 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
15
where ∅is an empty prompt used to preserve the original
image content.
We then denoise the latents along the trajectory using
DDIM:
˜xti =
p
¯αti
xti+1 −p1 −¯αti+1˜ϵθ(xti+1, ti+1, y, ∅)
√¯αti+1
+
p
1 −¯αti˜ϵθ(xti+1, ti+1, y, ∅),
(17)
˜ϵθ(xt, t, ∅, y) = ϵθ(xt, t, ∅) + λ(ϵθ(xt, t, y) −ϵθ(xt, t, ∅)),
(18)
where y is the target prompt and λ is the guidance scale.
By simplifying Eq. 16 and Eq. 17, we obtain:











xti+1
pαti+1
−xti
√αti
= (
s
1 −αti+1
αti+1
−
s
1 −αti
αti
)ϵθ(xti, ti, ∅),
xti+1
pαti+1
−
˜xti
√αti
= (
s
1 −αti+1
αti+1
−
s
1 −αti
αti
)˜ϵθ(xti+1, ti+1, y, ∅),
(19)
Subtracting the two equations gives:
xti −˜xti = (
s
1 −αti+1
αti+1
−
s
1 −αti
αti
)
×(˜ϵθ(xti+1, ti+1, y, ∅) −ϵθ(xti, ti, ∅)).
(20)
When ti+1 is close to ti, we can approximate:
˜ϵθ(xti+1, ti+1, y, ∅) ≈˜ϵθ(xti, ti, y, ∅).
(21)
Substituting Eq. 21 into Eq. 20 yields:
xti −˜xti ≈(
s
1 −αti+1
αti+1
−
s
1 −αti
αti
)(˜ϵθ(xti, ti, y, ∅) −ϵθ(xti, ti, ∅))
= (
s
1 −αti+1
αti+1
−
s
1 −αti
αti
)
× (ϵθ(xti, ti, ∅) + λ(ϵθ(xti, ti, y) −ϵθ(xti, ti, ∅)) −ϵθ(xti, ti, ∅))
= λ(
s
1 −αti+1
αti+1
−
s
1 −αti
αti
)
× (ϵθ(xti, ti, y) −ϵθ(xti, ti, ∅))
(22)
Thus, we have:
xti −˜xti ∝ϵθ(xti, ti, y) −ϵθ(xti, ti, ∅),
Therefore, it can be regarded as xti −˜xti ∝(ϵθ(xti, ti, y) −
ϵθ(xti, ti, ∅)), where (ϵθ(xti, ti, y) −ϵθ(xti, ti, ∅)) is the in-
formation at timestep ti in MTS.
In fact, this approximation in Eq. 21 introduces certain
errors, which become more significant as the ∆T = ti+1 −ti
increases, as illustrated in Fig. 19. Therefore, reducing ∆T
leads to higher generation quality. However, this also increases
the number of diffusion steps, resulting in higher computa-
tional cost. Considering computational constraints, we set ∆T
to 50 ∼100 in our implementation.
2. Connection to 2D Editing
Next, we interpret xti −˜xti from the perspective of 2D
image editing using diffusion models.
Text-guided 2D image editing aims to modify an input
image according to a target text prompt. Existing diffusion-
based 2D editing methods generally consist of two main
stages. The first stage is inversion, which focuses on preserving
the content of the input image. This is typically done by
aligning a complete noising and denoising trajectory, enabling
faithful reconstruction of the original image. During the nois-
ing process, DDIM Inversion with an empty text prompt is
often used to preserve the input image’s content. The denoising
path is then aligned through optimization over text embeddings
at each timestep. The second stage is editing, which aims to
inject the semantic content of the target text into the input
image. In this stage, the image is progressively denoised using
the target text prompt, which naturally integrates new content
into the reconstructed image. Multi-step trajectory modeling is
also critical in 2D editing. In the inversion stage, it helps align
content across multiple granularities to enhance reconstruction.
In the editing stage, injecting the target text across timesteps
allows fine-grained control over the strength and scope of the
edits [40].
Under a similar setting to MTS, these ,method denote
the noising trajectory as ˜xt0, ˜xt1, ..., ˜xtm and the denoising
trajectory as ˜xt0, ˜xt1, ..., ˜xtm In the inversion stage, the goal
is to align these two trajectories to reconstruct the original
image. Since the exact prompt that describes the input image is
unknown, recent approaches (e.g., NTI [39], PTI [53]) leverage
differentiable null-text prompts ∅t or conditional target texts
yt to optimize this alignment. This process can be formulated
as:
αti = arg min
αti
||xti −˜xti(ti, αti)||2
2,
(23)
where i = m, ..., 0 and αti = ∅ti or yti. This alignment
process is typically achieved by minimizing the difference
xti −˜xti, effectively guiding ˜xti →xti. In MTS, we observe
a similar mechanism. As shown in Eq.22, the difference
ϵθ(xti, ti, y) −ϵθ(xti, ti, ∅) is proportional to xti −˜xti except
that the direction is reversed: we aim to move xti →˜xti, since
˜xti contains semantic information from the target text prompt
and this information needs to be backpropagated through xti
into the 3D representation.
Editing methods align with multi-step denoising trajectories
in diffusion processes to produce high-quality images. This
alignment mechanism similarly enables MTS to align with
high-quality denoising trajectories, thereby achieving efficient
generation. It also explains why traditional SDS [1] methods
tend to produce oversaturated results: they typically use a
single-step denoising process with a large timestep , which
leads to coarse and imprecise supervision. In contrast, standard
diffusion models perform multi-step denoising with smaller
timestep, allowing for more accurate approximation of the
underlying data distribution.
APPENDIX B
SCENE PLANNING TEMPLATE
We use the prompts shown in Fig. 15, Fig. 16, and Fig. 17
to obtain structured information from GPT-4 [30], which is
then parsed using Python. From the user’s open-ended prompt
or dialogue, we extract the corresponding {User Constraint}.
We prepend each prompt with the instruction: ”You are a
professional scene designer. Based on the user requirements

<!-- page 16 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
16
Object Information Prompt
You are a professional scene designer. Based on the user requirements User Constraint and your domain knowledge, your task is to generate a list
of objects commonly found in the described scene. For each object, please include its frequency of appearance, typical dimensions ([x, y, z] in
meters), and a brief description starting with ”A DSLR photo of”. Ensure that the object descriptions are consistent with the scene’s style and reflect
common human understanding. Output should be formatted as follows in JSON:
Input:
a living room
Output:
{”sofa”: {”number”:2, ”size”:[2.0,1.0,0.8], ”description”:”A DSLR photo of a plush, grey sectional sofa, featuring deep cushions and soft fabric.”},
”coffee table”:{”number”:1, ”size”:[1.5,1.0,0.5], ”description”: ”A DSLR photo of a round, glass-top coffee table with a modern design and a sturdy
metal base.”},
”TV”:{”number”:1, ”size”:[1.4, 0.8, 0.1], ”description”: ”A DSLR photo of a large flat-screen TV, featuring a wide, slim display on the TV stand.”},
”TV stand”: {”number”:1, ”size”:[1.0, 0.4, 0.5], ”description”: ”A DSLR photo of a sleek, modern TV stand featuring open shelving and a minimalist
design.”}
”potted plant”: {”number”:2, ”size”:[0.5, 0.5, 1.0], ”description”: ”A DSLR photo of a vibrant, lush plant with broad green leaves in a decorative
pot.”} }
Now, let’s design the scene: {input}.
Fig. 15. Prompt template for object information with GPT-4.
Layout Information Prompt
You are a scene placement expert. Based on the user requirements User Constraint and your domain knowledge, your task is to determine the spatial
relationship between an object and its environment based on the object’s name and common human understanding. There are four relationships to
choose from: 1. CENTER, the object is in the center of the scene 2. SIDE, the object is at the boundary of the scene 3. CORNER, the object is in
the corner of the scene 4. OTHERS, the object is in other places. When dealing with multiple similar objects, arrange their positions reasonably to
prevent conflicts. Please return in the following example format in JSON format.
Input:
”scene type”:”indoor scene”, ”scene text”:”a living room”, ”objects list”:[”sofa1”, ”sofa2”, ”coffee table1”, ”TV1”,”TV stand1”, ”potted plant1”,
”potted plant2”]
Output:
{”sofa1”: SIDE, ”sofa2”: SIDE, ”coffee table1”: CENTER, ”TV1”: SIDE, ”TV stand1”: SIDE, ”potted plant1”: CORNER, ”potted plant2”: CORNER}
Now, I need select for {input}.
Fig. 16. Prompt template for layout information with GPT-4.
Objects Constraints Prompt
You are an expert in scene arrangement. Based on the user requirements User Constraint, the given environment, and your domain knowledge, your
task is to select objects from the provided list that are relevant to the current object based on common human usage, and describe their spatial
or functional relationships. The possible relationships include: 1.LEFT, indicating the current object is at the left of the selected object. 1.RIGHT,
indicating the current object is at the right of the selected object. 3.FRONT, indicating the current object is at the front of the selected object.
4.BEHIND, indicating the current object is at the behind of the selected object. 5.OVER, indicating the current object is above the selected object.
6.UNDER, indicating the current object is below the selected object. 7.NEXT, indicating the current object is near the selected object. 8.OPPOSITE,
indicating the current object is opposite the selected object. Output the selected object and their relationship in JSON format. For example:
Input:
”scene type”: ”indoor scene”, ”scene text”: ”a living room”,”current object”: ”sofa1”, ”objects list”: [”sofa2”,”coffee table1”,”TV1” ”TV stand1”,
”potted plant1”, ”potted plant2”]
Output:
{”sofa2”: NEXT, ”coffee table1”: FRONT, ”TV1”: OPPOSITE, ”TV stand1”: OPPOSITE}
Now, I need design for {input}.
Fig. 17. Prompt template for objects constraints with GPT-4.
User Constraint, and your domain knowledge...” This approach
allows us to leverage both the user’s specific intent and GPT-
4’s rich scene prior knowledge.
APPENDIX C
SCENE PLANNING TEMPLATE
We provide a detailed algorithmic description of the training
process of DreamScene as shown in 2.
APPENDIX D
ADDITIONAL EXPERIMENTS
A. Camera Configuration in Training and Testing
To ensure fair and meaningful comparison across methods,
we analyze the training-time camera pose strategies of exist-
ing baselines and apply a unified testing-time trajectory for
all. Tab. III provides a detailed comparison of training and
evaluation camera pose strategies.
As shown in Table, for training-time camera poses, each
baseline employs a distinct sampling strategy based on its

<!-- page 17 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
17
TABLE III
COMPARISON OF TRAINING AND EVALUATION CAMERA POSE STRATEGIES ACROSS DIFFERENT METHODS.
Method
Training Camera Poses
Evaluation Camera Poses
Text2Room [22]
Training camera poses are sampled along a predefined con-
tinuous trajectory. Camera orientations are adjusted based on
heuristic tilt and rotation rules defined in the original imple-
mentation, allowing moderate variation in viewpoint along the
path.
the camera starts from the scene center, first moves
along straight paths in multiple directions across the
scene, and then performs a circular motion around the
center, with the radius of the circular path set to two-
thirds of the scene diameter.
Text2NeRF [25]
The cameras are placed within the scene, facing outward, and
sampled within a spherical region with a ±60° pitch angle.
ProlificDreamer [27]
Same as Text2NeRF; uses object-centric sampling without
scene layout awareness.
Set-the-Scene [21]
The entire scene is placed at the center, and camera poses are
sampled from the bounding sphere, with the cameras oriented
toward the interior of the scene.
DreamScene (Ours)
(1) Sample camera poses near the scene center within a
constrained radius; (2) For indoor scenes, divide space into
regions based on object layout and sample camera poses within
each region; for outdoor scenes, sample camera poses along
concentric circles with fixed angular direction; (3) Combine all
sampled poses across stages.
(a) SDS [1]
(b) DreamTime [6]
(c) MTS
(d) FPS
Fig. 18. The ablation results of different sampling strategies.
architectural design:
• Text2Room [22] samples camera poses along a pre-
defined continuous trajectory. Camera orientations are
adjusted using heuristic tilt and rotation rules from the
original implementation, allowing moderate viewpoint
variation along the path.
• Text2NeRF [25] and ProlificDreamer [27] place cam-
eras within the scene, facing outward, and sample them
within a spherical region constrained by a ±60° pitch
angle.
• Set-the-Scene [21] centers the scene within a bounding
sphere and samples camera poses from its surface, ori-
enting the cameras inward toward the scene’s interior.
To enable a more fair and meaningful comparison, we adopt
a unified camera trajectory during evaluation for all methods.
Specifically, we test on the same scenes used for training
but replace each method’s original training-time poses with
a continuous camera trajectory that mimics natural human
exploration behavior. The camera begins at the scene center,
moves along straight paths in multiple directions across the
environment, and then performs a circular sweep around the
center. The radius of this circular path is set to two-thirds of
the scene diameter. This unified trajectory better reflects real-
world usage patterns and offers a more reliable measure of
robustness and practical usability.
B. Multi-head Scene
Fig.8 in the main paper illustrates the ”multi-head”
phenomenon
observed
in
ProlificDreamer
[27]
and
Text2Room [22]. For methods relying on SDS [1], [2],
[27], the camera pose is randomly sampled during the
optimization process, and the model lacks the ability to
perceive the orientation of the scene. Consequently, the
same prompt is optimized in any direction, often leading
to the repetitive generation of objects, such as sofas, from
all angles in scenarios like ”a living room“, resulting in
an overwhelming presence of sofas in the final scene. For
inpainting-based methods [22], [23], [25], the model retains
some orientation awareness as it continuously expands on a
fixed-size rendering image—rotating a certain degree each
time and completing it with the diffusion model. In cases
where sofas have already appeared, these methods usually
do not generate the same content again. However, if the
rotation angle is large enough that the sofa disappears from
the original view, the method will regenerate the sofa content.
Overall, the ”multi-head” issue is more pronounced with
methods based on SDS than with inpainting-based methods.
In our approach, because the scene layout is predefined, our
model can utilize the orientation information and the existing
object layout to enhance the environmental generation. This
significantly mitigates the ”multi-head” problem by ensuring
that the environment generation is coherent and contextually
appropriate.
C. Ablations
Different sampling strategies. We examined the effects of
different sampling strategies on the generation results of a 3D
object. Fig. 18 (c) displays the outcomes after 30 minutes

<!-- page 18 -->
IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE
18
Algorithm 2 DreamScene
1: Input: A simple scene text yS, the maximum number of iteration
iterm, iteration for Gaussian filtering iterf, the number of
intervals m, compression ratio η, x are the coordinates for 3D
Gaussians [18].
2: Initialize Stable Diffusion [5], Point-E [11];
3: Generate
objects
descriptions
y1, y2, ..., yN
and
layouts
l1, l2, ..., lN(l = [s, t, r], s is the scale coefficient, t is the
translation coefficient and r is the roation coefficient) by Scene
Planning Module;
4: for n in [1, 2, ..., N, S] do
5:
if n is not S then
6:
Initialize 3D Gaussian of objn by Point-E
7:
else
8:
Initialize cuboid or hemispherical 3D Gaussian for the
scene
9:
end if
10:
for iter = [0, 1, ..., max iter] do
11:
if n is not S then
12:
Spherical sample camera pose c
13:
else
14:
Sample camera pose c following strategy in Sec.IV-C
15:
end if
16:
x0 = g(θ, c)
17:
Tend = (1 −
iter
iterm ) × 1000
18:
for i = [1, 2, ..., m] do
19:
ti = Tend · random( i−1
m , i
m)
20:
xi = DDIM(xi−1, i)
21:
ϵϕ(xti; yn, ti) =U-Net(xti, yn, ti)
22:
ϵϕ(xti; ∅, ti) =U-Net(xti, ∅, ti)
23:
end for
24:
∇θLMTS(θ) =
25:
Et,ϵ,c
 m
P
i=1
w(ti)(ϵϕ(xti; yn, ti) −ϵϕ(xti; ∅, ti)) ∂g(θ,c)
∂θ

26:
Update θ
27:
if iter%iterf = 0 then
28:
Scorek = PH×W ×M
j=1
V (k)
D(rj,k)2×maxV (rj)
29:
Sort(Scorek)
30:
Delete last η 3D Gaussians
31:
end if
32:
end for
33:
Generate K images ˆxt
0 using ˆxt
0 =
xt−√1−¯αtϵϕ(xt;y,t)
√¯αt
by
sampling timestep t ∈(0, 200) from different camera poses.
34:
Generate detailed and plausible textures by Lrec
=
P
i ||g(θ, ci) −ˆxt
i0||2.
35:
if n is S then
36:
Save 3D Gaussian Representation of the entire scene
37:
break
38:
end if
39:
Save 3D Gaussian Representation objn of text yn
40:
world(x) = rn · sn · objn(x) + tn
41:
Add objn to the Scene by coordinate world(x)
42: end for
of optimization under the prompt ”A DSLR photo of Iron
Man.” As shown, multi-timestep sampling (MTS) establishes
superior geometric structures and textures compared to both
the monotonically non-increasing sampling strategy in [6] and
Score Distillation Sampling (SDS) technique in [1]. Building
upon the strengths of MTS, Formation Pattern Sampling (FPS)
employs a reconstruction method to produce smoother and
more realistic textures.
Different sampling step sizes. We conduct ablation studies
using different sampling step sizes ∆T = ti+1 −ti. As shown
in Eq.21 and discussed in Sec.A, smaller values of δT lead to
t=300
t=150 (ECCV)
t=100
t=50 (TPAMI)
Fig. 19. The ablation results of different timestep size ∆T.
higher-quality and more detailed generation results, as further
evidenced in Fig. 19. However, smaller step sizes require more
sampling steps. Considering hardware limitations, we adopt
δT values in the range of 50 to 100 in our experiments.
