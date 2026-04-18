<!-- page 1 -->
Cog2Gen3D: Sculpturing 3D Semantic-Geometric
Cognition for 3D Generation
Haonan Wang1, Hanyu Zhou2⋆, Haoyue Liu1, Tao Gu3, and Luxin Yan1
1 School of Artificial and Automation, Huazhong University of Science and Technology
2 School of Computing, National University of Singapore
3 School of Computing, Macquarie University
whn_aurora@hust.edu.cn, hy.zhou@nus.edu.sg
Abstract. Generative models have achieved success in producing seman-
tically plausible 2D images, but it remains challenging in 3D generation
due to the absence of spatial geometry constraints. Typically, existing
methods utilize geometric features as conditions to enhance spatial aware-
ness. However, these methods can only model relative relationships and
are prone to scale inconsistency of absolute geometry. Thus, we argue
that semantic information and absolute geometry empower 3D cogni-
tion, thereby enabling controllable 3D generation for the physical world.
In this work, we propose Cog2Gen3D, a 3D cognition-guided diffusion
framework for 3D generation. Our model is guided by three key designs:
1) Cognitive Feature Embeddings. We encode different modalities
into semantic and geometric representations and further extract logical
representations. 2) 3D Latent Cognition Graph. We structure dif-
ferent representations into dual-stream semantic-geometric graphs and
fuse them via common-based cross-attention to obtain a 3D cognition
graph. 3) Cognition-Guided Latent Diffusion. We leverage the fused
3D cognition graph as the condition to guide the latent diffusion process
for 3D Gaussian generation. Under this unified framework, the 3D cog-
nition graph ensures the physical plausibility and structural rationality
of 3D generation. Moreover, we construct a validation subset based on
the Marble World Labs. Extensive experiments demonstrate that our
Cog2Gen3D significantly outperforms existing methods in both semantic
fidelity and geometric plausibility.
Keywords: 3D Generation · 3D Cognition · Latent Diffusion Models
1
Introduction
Generative models have achieved success in synthesizing semantically plausible
2D images, which are widely used in image editing [1] and video generation [2].
When applying these models to the physical world, some researchers rely on
the semantic priors of 2D diffusion models and extend them to 3D generation
through Score Distillation Sampling [3,4], as illustrated in Fig. 1 (a). However, this
⋆Corresponding author.
arXiv:2603.05845v1  [cs.CV]  6 Mar 2026

<!-- page 2 -->
2
H. Wang et al.
(a) Semantics-Guided 3D Generation
(c) 3D Cognition-Guided 3D Generation 
(b) 2D Geometry-Guided 3D Generation
Text & Image
Text & Image
Text & Image
3D Cognition Graph
Semantic
Features
Generative
Models
Generative
Models
Generative Models
3D Rep.
3D Rep.
3D Rep.
Geometry Collapse
Scale Inconsistency
Plausible Physics
Sem. Graph
Geo. Graph
Fuse
Global semantic
response
Local geometry
response
Structured cognition
topology
2D Geo.
Feat.
Sem.
Feat. ⊕
A wooden chair sits
beside a table holding
a mug, a book, and an
apple, with a white vase
resting on the shelf.
A wooden chair sits
beside a table holding
a mug, a book, and an
apple, with a white vase
resting on the shelf.
A wooden chair sits
beside a table holding
a mug, a book, and an
apple, with a white vase
resting on the shelf.
Fig. 1: Paradigm shift of 3D generation. Semantic-Guided Generation relies heavily
on 2D semantic priors, resulting in physical violations such as object intersections.
2D Geometry-Guided 3D Generation introduces relative geometric constraints but
frequently suffers from scale inconsistency due to the lack of absolute metric awareness.
Our 3D Cognition-Guided paradigm leverages a 3D cognition graph as the condition,
ensuring 3D generation with fidelity and plausibility.
paradigm is prone to structural collapse due to the lack of geometric constraints
required by the physical world. In this work, our purpose is to integrate geometric
and semantic representations to jointly achieve high-quality 3D generation.
As shown in Fig. 1 (b), existing methods mainly incorporate geometric priors
such as scene graphs [5–9] and layouts [10–14] to enhance spatial awareness in
the 3D generation process. For example, EchoScene [9] utilizes scene graphs to
define relational dependencies between objects, while Layout2Scene [14] relies
on bounding box layouts to guide the coarse spatial arrangement of the scene.
However, these approaches are limited to modeling 2D relative spatial relation-
ships and fail to capture 3D absolute geometry. This leads to scale inconsistency
and geometric collapse, making it difficult to satisfy the rigid constraints of the
physical world. Therefore, we argue that the key to 3D generation for the physical
world lies in the integration of high-level semantics and absolute geometry.
To address the above issues, we propose that high-level semantics and 3D
absolute geometry jointly empower 3D cognition and enhance the physical spatial
awareness of 3D generation, as illustrated in Fig. 1 (c). Our design is grounded
in two key insights: 1) Geometric features provide essential physical plausibility.
We figure out that geometric encoders inherently capture dense correspondence
and absolute metric information, which motivates us to leverage these cues
to enforce strict physical constraints and precise scale consistency within the
generative process. 2) Latent scene graphs provide structural rationality and
superior reasoning robustness. We discover that latent scene graphs robustly infer
topological relationships in a latent embedding space, which inspires the design
of our graph module to ensure the coherent structure of the generated scenes.
Thus, these observations motivate us to design a common-based fusion scheme
that integrates semantic knowledge and geometric information to represent 3D
cognition, thereby enabling controllable 3D generation for the physical world.

<!-- page 3 -->
Cog2Gen3D
3
Image
Text
S
E
G
E
L
E
A modern blue-themed living 
room featured a deep blue plush 
sofa, a wooden coffee table on a 
cream rug, an abstract painting,
a floor lamp, and a potted plant.
Semantic Tokens
Geometric Tokens
Logical Tokens
sT
G
T
LT
sT
G
T
LT
LT
Semantic
Graph
Encoder
Geometric
Graph
Encoder
LT
Semantic Graph
Geometric Graph
Bridged Graph Fusion
3D Cognition Graph
LG
Sofa
Lamp
Art
Table
Plant
hang above
next to
next to
opposite to
Explicit Scene Graph
Generated 3D Gaussians
GS
D
Latent Diffusion
GS
E
GS
D
Pretrained Gaussian Encoder-Decoder
z
Latent Code
Noise
Condition
Cognitive Feature Embeddings
3D Latent Cognition Graph
Cognitive-Guided Latent Diffusion
Bridge
pull
pull
Fig. 2: Overview of Cog2Gen3D. The model first extracts multiple cognitive tokens
(TS, TG, TL). These tokens are structured into a 3D Cognition Graph, where the logical
tokens act as a bridge for semantic-geometric alignment. Finally, the awakened 3D
cognition steers a latent diffusion process to generate 3D Gaussians.
In this work, we propose Cog2Gen3D, a novel 3D cognition-guided diffusion
framework for 3D generation. Our model consists of three key components: 1)
Cognitive Feature Embeddings. We encode different input modalities into
semantic and geometric representations and further extract logical representation
to serve as high-level guidance. 2) 3D Latent Cognition Graph. We structure
these representations into dual-stream semantic-geometric graphs and utilize the
logical representation as a bridge to fuse them via common-based cross-attention,
obtaining a unified 3D cognition graph. 3) Cognition-Guided Latent Diffusion.
We leverage the 3D cognition graph as condition to guide the diffusion process for
3D Gaussian generation. Through this design, our approach effectively awakens 3D
cognition, ensuring that the generated scenes possess both high-fidelity semantics
and plausible geometry. Our main contributions are summarized as follows:
– We propose Cog2Gen3D, an innovative framework that introduces 3D
cognition to guide 3D generation, effectively bridging semantic priors with
geometric constraints to support versatile controllable 3D object and scene
generation from arbitrary combinations of visual and textual prompts.
– We observe that geometric features provide geometry consistency and latent
scene graphs offer structural rationality. This motivates the design of cognitive
feature embeddings and 3D latent cognition graph for sculpturing a robust 3D
representation that captures appearance attributes and spatial interactions.
– We develop a cognition-guided latent diffusion mechanism that steers 3D
Gaussian generation using the cognition graph, ensuring both semantic fidelity
and geometric plausibility of the generated 3D scenes.
– We integrate a series of 3D datasets and construct a curated validation subset,
collectively named the cognition scene graph 3D dataset (CogSG-3D), along
with explicit scene graph labels for supervision. Extensive experiments have
demonstrated the significant effectiveness of our method.
2
Related Works
2.1
Semantics-Guided 3D Generation
The success of 2D diffusion models has significantly advanced 3D generation.
Current methods predominantly leverage their rich semantic priors to supervise

<!-- page 4 -->
4
H. Wang et al.
3D representations [3,4,15–18]. For instance, DreamFusion [3] lifts 2D image priors
via Score Distillation Sampling (SDS) to iteratively optimize a 3D model. While
capable of synthesizing visually impressive objects, these approaches struggle with
severe geometric collapse. This limitation arises because they treat 3D generation
as multi-view 2D in-painting, lacking an intrinsic perception of physical spatial
structures. Therefore, our goal is to introduce geometric cues into the generative
process to ensure both appearance fidelity and structural rationality.
2.2
2D Geometry-Guided 3D Generation
To mitigate structural inconsistencies, recent approaches explicitly incorporate 2D
geometric priors, such as scene graphs [5–8] and layouts [10–13], to organize global
scene composition. For instance, EchoScene [9] utilizes topological structures to
constrain the semantic interactions among entities, whereas Layout2Scene [14]
relies on bounding boxes to restrict the relative positions of generated objects.
While these structured representations successfully provide a reasonable blueprint
to produce roughly arranged scenes, they persistently struggle to ensure precise
physical plausibility and strict structural fidelity in complex environments. This
limitation fundamentally arises because these methods predominantly model 2D
relative spatial relationships rather than the precise absolute metric geometry of
the 3D physical world. Consequently, our work focuses on capturing 3D absolute
geometry to sculpt structurally rigorous and physically plausible 3D scenes.
2.3
Spatial Geometric Perception
Spatial geometric perception is crucial for 3D generation, as it bridges image-text
prompts and structurally rigorous 3D scenes. The core of this perception lies
in accurately capturing absolute metric geometry rather than merely coarse
topological relations. Recent spatial representation models, such as VGGT [19],
have demonstrated exceptional capabilities in encoding fine-grained 3D spatial
geometry. However, effectively integrating such perception into generative models
remains a challenge. Existing Multimodal Large Language Models often resort
to naive feature concatenation [20,21], which fails to deeply assimilate precise
geometric constraints. In contrast, we propose to construct a structured geometric
graph and fuse it with a corresponding semantic graph. This design effectively
awakens 3D cognition, ensuring both the physical plausibility and structural
rationality of the synthesized 3D scenes.
3
Our Cog2Gen3D
Overview. As shown in Fig. 2, our Cog2Gen3D achieves semantically-coherent
and geometrically-plausible 3D generation through the following three stages:
1) Cognitive Feature Embeddings. This is the cognitive representation
stage that we transform input images I and texts T into disentangled semantic,
geometric, and logical representations TS, TG, TL:
 \s mal l \ setlength  \abovedisplayskip {2pt} \setlength \belowdisplayskip {2pt} \begin {aligned} T_S,T_G,T_L=E_{S,G,L}(I,T), \end {aligned} 
(1)

<!-- page 5 -->
Cog2Gen3D
5
View A
ResNet50
VGGT Encoder
View B
Keypoint-1
Keypoint-2
Keypoint-3
1
2
3
1
1
2
2
3
3
1
2
3
(a) Input Images
(c) t-SNE Visualization
(b) Heatmap Visualization of Attention Maps
Keypoint 1
Keypoint2
Keypoint3
Fig. 3: Cross-view feature correspondence analysis. The attention map and
the t-SNE visualization demonstrate that the VGGT encoder has superior cross-view
geometric consistency. This validates its capability of capturing absolute geometry
information, which motivates us to introduce VGGT encoder as our geometric expert.
which serve as the fundamental building blocks for subsequent cognitive reasoning.
2) 3D Latent Cognition Graph. This is the 3D cognition construction stage
that we encode the extracted features into dual-stream graphs Gsem, Ggeo and
fuse them via a common-based fusion with the logical bridge TL:
 \sm a ll \se tlen gth \ aboved ispl aysk i p {4pt} \setlen gth \ belowdisplayskip {4pt} \begin {aligned} G_{sem}=GE(T_S,T_L), G_{geo}=GE(T_G,T_L), G_{cog}=ComFusion(G_{sem},G_{geo},T_L). \end {aligned} 
(2)
The resulting 3D cognition graph Gcog precisely captures both the extrinsic
semantics and intrinsic geometry of the scene.
3) Cognition-Guided Latent Diffusion. This is the cognition-guided genera-
tion stage that the 3D cognition graph acts as a structural condition to steer the
latent diffusion process, ultimately generating 3D Gaussians ˆG with high-fidelity
appearance and rational structure:
 \s m all \s e tlengt h \ abovedisplayskip {4pt} \setlength \belowdisplayskip {4pt} \begin {aligned} \hat {\textbf {z}}_0 = \text {LDM}(\textbf {z}_T \mid G_{cog}), \hat {\mathcal {G}} = D_{GS}(\hat {\textbf {z}}_0). \end {aligned} 
(3)
Remarks. The output ˆG denotes the generated 3D Gaussians representing the
3D scene with high-fidelity appearance and rational structure. Our framework
utilizes a 3D latent cognition graph as structural conditions to enable semantically-
coherent and geometrically-plausible 3D generation in diffusion models.
3.1
Cognitive Feature Embeddings
While conventional generative models rely predominantly on semantic priors, we
establish a comprehensive foundation for 3D cognition by integrating semantic
and geometric information alongside logical constraints.
Semantic Encoder(ES). To capture rich visual appearance, we utilize a pre-
trained ResNet50 [22] as the semantic encoder ES. It extracts high-level visual
features from input images and projects them into Semantic Tokens TS, ensuring
high-fidelity appearance in the generated 3D Gaussians.
Geometric Encoder (EG). As illustrated in Fig. 3, we evaluate the feature
representations of ResNet50 [22] and the VGGT encoder [19] by computing cross-
view attention between the keypoint-specific features of View A and the spatial
feature maps of View B. Visualizations reveal that ResNet50 struggles with feature
drift while VGGT distinctly separates keypoint features, demonstrating robust

<!-- page 6 -->
6
H. Wang et al.
Visualization of Different Features
Different Input Prompts
V
Different Input Prompts
a. Prompt with correct relation
A wooden chair next to a round table.
b. Prompt without relation
A wooden chair. A round table.
c. Prompt with wrong relation
A wooden chair inside a round table.
Prompt a
Chair Features
Prompt b
Prompt c
Table Features
Explicit Graph
Latent Graph
Fig. 4: Feature distribution of explicit vs. latent scene graphs under prompt
perturbations. Explicit graphs diverge significantly when given missing or incorrect
relations compared to the correct baseline. Conversely, our latent scene graph maintains
a stable distribution, demonstrating superior robustness to unpromising prompts.
cross-view geometric consistency. Motivated by this capability, we adopt VGGT
as our geometric encoder EG, which provides essential geometric grounding,
ensuring rigorous structural plausibility and metric accuracy.
Logical Encoder (EL). To bridge raw features and structural reasoning, we
adopt CLIP ViT and CLIP Text encoders [23] as the logical encoder EL. By
processing image-text pairs, EL extracts Logical Tokens TL that encapsulate
high-level relational contexts and abstract concepts, serving as the essential
guidance for constructing the 3D cognition of the scene.
Through this tri-stream architecture, we obtain a comprehensive set of cog-
nitive tokens {TS, TG, TL}, which collectively define the cognitive feature space
required for the subsequent 3D Latent Cognition Graph construction.
3.2
3D Latent Cognition Graph
3D generation requires robust guidance equipped with both rich semantics and
absolute geometry. Fig. 4 shows that explicit scene graphs are highly sensitive to
noisy inputs, suffering from severe feature divergence under prompt perturbations.
To address this, we propose the 3D Latent Cognition Graph. By processing
cognitive tokens through a dual-stream latent graph encoder and a common-
based fusion mechanism, this module implicitly sculptures a noise-resistant and
structurally rigorous cognitive representation.
Dual-Stream Latent Graph Encoder. To capture coherent appearance and
precise geometry, we construct two parallel graphs: a semantic graph and a
geometric graph. Both share a similar topological encoding paradigm but differ
fundamentally in their input tokens and positional embedding strategies.
Semantic Graph Construction. We formulate the semantic query QS by con-
catenating tokens TS with a 2D positional embedding PE(xp, yp). Through
cross-attention between QS and TS, we extract n initial nodes N S
i (i ∈[0, n]).
To establish relationships, the cross-attention utilizes the shared logical tokens

<!-- page 7 -->
Cog2Gen3D
7
(a) Cognition Graph Construction Modules
(b) Visualization of Cognition Graph Attention
Semantic Graph Construction
sT
s
Q
i
N
S
LT
⊕PE(x , y )
i
i
C-Atten
C-Atten
MLP
Logical Guidance
ij
E
S
Node Update
Geometric Graph Construction
G
T
G
Q
LT
⊕PE(x , y , z )
i
i
i
C-Atten
C-Atten
MLP
Logical Guidance
Node Update
Edges
Edges
A red ceramic vase
sitting on a wooden stool.
Node: Vase
Node: Stool
Edge: Sitting
ti
M d l
ti
Bridged Graph Fusion
j
N
S
i
N
G
ij
E
G
j
N
G
LT
L
Q
SG
K
SG
V
Q
W
K
W
V
W
Projection
Concate
Cross-Attention
N
cog
N
cog
^
Gcog
LN
MLP
3D Cognition
Graph
Gsem
Ggeo
Gsem
Ggeo
Fig. 5: Architectural details and interpretability of the 3D Latent Cognition
Graph. (a) The pipeline for constructing cognition graph from cognitive tokens. (b)
The correspondence visualization showing how graph components precisely align with
3D entities and spatial boundaries.
TL as logical guidance to formulate semantic edges ES
ij between N S
i and N S
j
(i, j ∈[0, n]). An MLP then utilizes the message mS
ij from nodes and corre-
sponding edges to update the nodes, thus obtaining the semantic graph Gsem:
 \
sm a ll \s e
t l e n
g t h 
\abo
vedi s p l
a y
s
k
ip {4pt} \setlength \belowdisplayskip {4pt} \begin {aligned} m_{ij}^S=\text {MLP}(N_i^S,N_j^S,E_{ij}^S),\quad G_{sem}=N_i^S+\sum _{j}m_{ij}. \end {aligned} 
(4)
Geometric Graph Construction. To move beyond 2D constraints and model
absolute 3D metrics, the geometric query QG employs a specialized 3D positional
embedding PE(xq, yq, zq). Here, zq is a learnable embedding explicitly introduced
to capture underlying spatial geometry. Similar to the semantic stream, cross-
attention modules extract initial nodes N G
i
and utilize TL to formulate geometric
edges EG
ij. An MLP then updates them to produce the geometric graph Ggeo.
Driven by the learnable zq dimension, this graph effectively models absolute 3D
metric relationships between entities.
Common-based Cross-Attention Fusion. While the semantic and geometric
graphs encapsulate coherent appearance and precise geometry respectively, they
remain in separate feature spaces. Since both graphs utilize the same logical
tokens TL for relation edge Eij formulation, their topologies inherently share a
common logical foundation.
Exploiting this shared foundation, we introduce a common-based cross-
attention fusion mechanism. To integrate the distinct feature spaces, we treat the
logical tokens TL as a unifying anchor. We project TL to form a shared logical
query QL, while the semantic and geometric nodes are concatenated along the
feature dimension to jointly formulate the keys Ksg and values Vsg:
 \ s mall 
\se t length \abovedi
spl a yskip {4pt} \s etlength \belowdisplayskip {4pt} \begin {aligned} \mathbf {Q}_L = T_L \mathbf {W}_Q,\quad \mathbf {K}_{SG} = [G_{sem} || G_{geo}] \mathbf {W}_K,\quad \mathbf {V}_{SG} = [G_{sem} || G_{geo}] \mathbf {W}_V, \end {aligned} 
(5)
where [·||·] denotes the concatenation operation, and WQ, WK, WV are learnable
linear projection matrices. The unified 3D cognition graph nodes, denoted as
Ncog, are then computed via the scaled dot-product cross-attention:
 \sm a ll \set
lengt
h 
\ab
o
vedisplayskip {4pt} \setlength \belowdisplayskip {4pt} \begin {aligned} \mathbf {N}^{cog} = \text {Softmax}\left ( \frac {\mathbf {Q}_L \mathbf {K}_{SG}^T}{\sqrt {d_k}} \right ) \mathbf {V}_{SG}, \end {aligned} 
(6)

<!-- page 8 -->
8
H. Wang et al.
where dk is the scaling factor representing the channel dimension. In this formu-
lation, the logical query QL acts as an intelligent bridge, adaptively assigning
attention weights to extract and align corresponding semantic textures and struc-
tural constraints.Finally, we apply a residual connection and a MLP to stabilize
the fused representation:
 \sma l l \setlength  \above
disp l aysk ip {4p t } \setlength \belowdisplayskip {4pt} \begin {aligned} \mathbf {\hat {N}}^{cog} = \text {LayerNorm}(T_L + \mathbf {N}^{cog}),\quad \mathbf {G}_{cog} = \text {MLP}(\mathbf {\hat {N}}^{cog}) + \mathbf {\hat {N}}^{cog}. \end {aligned} 
(7)
Ultimately, this explicit mathematical fusion yields a holistic 3D cognition
graph Gcog that perfectly balances semantic coherence with geometric rationality,
ready to structurally guide the subsequent 3D generative process.
3.3
Cognition-Guided Latent Diffusion
With the comprehensive 3D cognition graph formulated, the final stage of our
framework is to effectively generate the 3D scenes with semantic fidelity and
geometric plausibility. Therefore, we propose a cognition-guided latent diffusion
process that operates within a compressed space of 3D Gaussians.
Conditioned Diffusion Process. Standard diffusion models acting directly on
explicit 3D representations are often computationally prohibitive. Therefore, we
perform the generative process in a learned latent space. Let the ground-truth
latent representation of the 3D scene be denoted as z0. During the forward
diffusion process, Gaussian noise ϵ is progressively added to z0 over t timesteps,
producing a noisy latent zt.In the reverse denoising process, our goal is to predict
and remove this noise. Crucially, instead of relying on rudimentary text or layout
conditions, we inject our fused cognition graph, Gcog, as the structural condition.
The denoising network predicts the added noise as:
 \ s mall \ se tlength \abovedisplayskip {4pt} \setlength \belowdisplayskip {4pt} \begin {aligned} \hat {\epsilon } = \epsilon _\theta (\mathbf {z}_t, t, \mathbf {G}_{cog}). \end {aligned} 
(8)
Within the denoising network, the high-fidelity semantics and plausible ge-
ometry embedded in Gcog effectively guide the generation of the latent space.
This cognition-guided paradigm effectively mitigates the geometric ambiguity
and layout distortion commonly observed in standard 2D-prior-based generation.
Latent-Gaussian Encoder-Decoder. To bridge the gap between the com-
pact latent space and the explicit 3D representation, we employ a pre-trained
latent-Gaussian encoder-decoder architecture as shown in Fig. 2. A standard
3D Gaussian splatting scene is explicitly parameterized by a set of Gaussians G.
The Gaussian encoder EGS compresses the high-dimensional Gaussians into the
compact latent space: z0 = EGS(G). Then we apply noise to obtain zt. Conversely,
after the conditioned diffusion process yields the denoised latent ˆz0, the decoder
DGS projects it back into the explicit 3D Gaussian space: ˆG = DGS(ˆz0). This
symmetric design ensures that our model can efficiently learn complex spatial
distributions while ultimately generating high-fidelity 3D Gaussians.

<!-- page 9 -->
Cog2Gen3D
9
3.4
Optimization
Our proposed Cog2Gen3D framework is supervised by training loss consists
of three primary components: a latent diffusion loss Ldiff, an explicit node
grounding loss LG, and a 3D Gaussian reconstruction loss Lrecon.
Latent Diffusion Loss Ldiff. To train the conditioned denoising network, we
employ the standard latent diffusion objective, minimizing the mean squared
error between the added noise ϵ and the predicted noise ˆϵ:
 \sma l l \setl
e
ngt h  \abo
v
e
displayskip {4pt} \setlength \belowdisplayskip {4pt} \begin {aligned} \mathcal {L}_{diff} = \mathbb {E}_{\mathbf {z}_0, \epsilon , t} \left [ || \epsilon - \hat {\epsilon } ||_2^2 \right ]. \end {aligned} 
(9)
Explicit Node Grounding Loss Lg. Explicit scene graphs typically lack precise
spatial grounding and contain sparse relations. Supervising latent edges with
such limited data would bottleneck spatial reasoning. Therefore, we discard edge
supervision and solely anchor the semantic identities of the nodes. Crucially, the
dense nodes in our cognition graph do not strictly align with the sparse entities
in the explicit graph. To address this, we dynamically select the top-K most
critical latent nodes via cross-attention weight ranking that correspond to the
K explicit entities. We pass these selected nodes through a classification head
to output semantic probabilities psem
i
. The loss is computed as a Cross-Entropy
(CE) objective against the explicit semantic labels ysem
i
:
 \ s m
a
l
l
 \s
etlengt
h
 \abo
v
edisplayskip {4pt} \setlength \belowdisplayskip {4pt} \begin {aligned} \mathcal {L}_g = \frac {1}{K} \sum _{i=1}^{K} \text {CE}(p_i^{sem}, y_i^{sem}). \end {aligned} 
(10)
This minimalist top-K supervision ensures semantic fidelity while granting
the latent edges absolute freedom to autonomously infer complex 3D topology.
3D Gaussian Reconstruction Loss Lrecon. To guarantee metric precision
and visual fidelity, we supervise the Gaussians ˆG using the ground-truth G. This
supervision is applied in the image space to ensure multi-view consistency. For
a set of viewpoints v, we render the predicted Gaussians into images ˆIv and
compare them against the GT renders Iv, calculating L1 loss and D-SSIM loss:
 \smal l
 
\
s
etlength  \abov e display s kip {4pt}  \se
t
length \belowdisplayskip {4pt} \begin {aligned} \mathcal {L}_{recon} = \sum _{v} \left ( \lambda _{L1} || \hat {I}_v - I_v ||_1 + \lambda _{ssim} (1 - \text {SSIM}(\hat {I}_v, I_v)) \right ). \end {aligned} 
(11)
Total Loss. The entire framework is optimized end-to-end using a weighted sum
of the above three losses, which is formulated as:
 \smal l  \setle n gth \ abovedisplayskip {4pt} \setlength \belowdisplayskip {4pt} \begin {aligned} \mathcal {L}_{total} = \lambda _1 \mathcal {L}_{diff} + \lambda _2 \mathcal {L}_g + \lambda _3 \mathcal {L}_{recon}, \label {eq1} \end {aligned} 
(12)
where λ1, λ2, λ3 are hyperparameters balancing the contributions of diffusion
fidelity, structural cognition, and visual reconstruction.
4
Training Datasets and Pipeline
4.1
Our CogSG-3D Dataset
As shown in Fig. 6, to effectively train our Cog2Gen3D, we propose CogSG-3D,
a comprehensive dataset aggregating pre-eminent public 3D datasets and self-
built supplementary data from Marble World Labs [25]. It covers two categories:

<!-- page 10 -->
10
H. Wang et al.
Text
Image
Scene Graph
3D Gaussians
A warm, neutral-toned modern bedroom.
A brown bed faces a floating TV stand,
with a beige sofa placed at the foot of
the bed, and abstract art hanging above
the headboard.
A bright, wood-accented attic bedroom. 
A bed rests against a wooden wall panel 
with a nightstand beside it, tall white 
wardrobes lining the right side, and a 
potted plant standing near the window.
A bright, modern living room with geome-
tric patterns. A beige sofa sits on a black 
and white striped rug in front of a coffee 
table, with a tall wooden bookshelf 
standing against the patterned back wall.
Sofa
Bed
TV
Art
TV-stand
Nightstand
hang above
hang above
next to opposite to
behind
Lamp
Plant
next to
Bed
Art
Wardrobe
Nightstand
hang above
next to opposite to
hang above
hang above
Clock
Art
Plant
next to
Table
Bookshelf
Chair
Sofa
behind
next to
next to
next to
Scene Level
19.8k
Object Level
50k
ModelNet
12k
Objarverse-XL
11.6k
ShapeNet
10k
ABO
10k
OmniObject3D
6k
Pix3D
0.4k
3D-Front
18k
ScanNet
1.5k
Self-built
0.3k
(b) Self-built Dataset Visualization
(a) CogSG-3D Dataset Statistics
Fig. 6: Statistics and examples of our proposed CogSG-3D dataset.
object-level (ShapeNet [26], Objaverse-XL [27], OmniObject3D [28], Pix3D [29],
ABO [30], ModelNet [31]) and scene-level (ScanNet [32], 3D-Front [33]), ensuring
diverse semantic and geometric distributions. To unify these heterogeneous source
formats into a standardized format for training, our data processing pipeline
encompasses three steps: 1) Image prompt rendering. We normalize the canonical
poses of all 3D models and render 2D multi-view images to serve as visual
prompts. 2) Text prompt acquisition. For datasets lacking explicit annotations, we
leverage advanced Vision-Language Models (Gemini) to generate dense descriptive
texts, ensuring rigorous vision-language alignment. 3) Unified 3D Gaussians
representation. We convert all 3D ground truths into directly optimizable 3D
Gaussians, overcoming the topological constraints of traditional meshes.
4.2
Training Pipeline
We design a progressive, three-stage training paradigm to ensure stable conver-
gence and optimal semantic-geometric alignment:
Stage 1: Geometric-Latent Alignment. This stage aims to acquire latent
representations of 3D scenes and a robust decoder. We pretrain a Gaussians auto-
decoder where the encoder EGS maps 3D Gaussians into a compact latent variable
z and the decoder DGS is simultaneously optimized to flawlessly reconstruct the
3D Gaussians from z, preserving rich semantic and geometric details.
Stage 2: Cognitive-Generative Alignment. The objective here is to synthe-
size semantically and geometrically coherent 3D latents guided by multimodal
priors. Keeping the autoencoder frozen, we exclusively train the Latent Diffusion
Model (LDM) in the z-space. The denoising network is conditioned on the 3D
Latent Cognition Graph to enforce strict structural constraints.
Stage 3: End-to-End Fine-Tuning. This stage harmonizes the generative
latent space with the explicit 3D rendering space. We unfreeze the decoder DGS
and jointly optimize it with the LDM end-to-end. This mitigates feature mis-
match between synthesized latents and the decoder’s expected input distribution,
significantly enhancing the overall visual fidelity and geometric precision.

<!-- page 11 -->
Cog2Gen3D
11
Table 1: Text-to-3D comparison.
Method
T3Bench [34]
Single Obj.↑Single w/ Surr.↑Multi Obj.↑Average↑
DreamFusion [3]
24.4
19.8
11.7
18.7
Magic3D [17]
37
35.4
25.7
32.7
SJC [4]
24.7
19.8
11.7
18.7
Fantasia3D [15]
26.4
27
18.5
24
ProlificDreamer [16]
49.4
44.8
35.8
43.3
GaussianDreamer [18]
54
48.6
34.5
45.7
Ours
58.3
57.9
53.6
56.6
Table 2: Image-to-3D objects comparison.
Method
ShapeNet [26]
OmniObject3D [28]
FID↓KID(%)↓MMD(%)↓FID↓KID(%)↓MMD(‰)↓
EG3D [35]
28.12
1.25
7.14
41.56
1.13
28.21
GET3D [36] 38.62
1.85
6.45
49.41
1.53
13.57
DiffRF [37]
98.53
6.14
8.59
147.59
8.82
16.07
DiffTF [38]
26.58
1.15
6.15
25.36
0.81
6.64
DiffGS [39]
23.45
1.05
5.31
27.35
0.84
7.05
LN3Diff [40] 21.51
1.08
5.25
21.42
0.73
4.87
Ours
14.54
0.67
3.25
15.94
0.58
3.02
(1) Single Obj.
A castle-shaped 
sand castle.
(2) Single w/ Surr.
A red rose in a 
crystal vase.
(3) Multi-Obj.
A man is holding
an umbrella 
against rain. 
(b) Dreamfusion
(a) Input Text
(c) LatentNeRF
(d) SJC
(e) ProlificDreamer (f) GaussianDreamer
(g) Ours
Fig. 7: Visualization of Text-to-3D on the T3-Bench dataset.
5
Experiments
5.1
Implementation Details
Our 3D latent cognition graph contains 64 hidden nodes by default, which can be
manually adjusted according to scene complexity. For the diffusion process, we
employ T = 1000 training timesteps, while a 50-step DDIM sampler is utilized
for efficient inference. The total loss is optimized using the AdamW optimizer
with a initial learning rate of 1 × 10−5, where the loss weights in Eq. (12) are
set as: λ1 = 0.8, λ2 = 0.2, and λ3 = 0.8. All training and testing procedures are
conducted on 8 NVIDIA A800 GPUs.
5.2
Comparison Experiments
Text-to-3D Generation. We evaluate text-to-3D generation on T3Bench [34]
against SOTA methods [3,4,15–18]. We adopt the T3Bench quality scores across
three complexity levels and their average. As shown in Tab. 1 and Fig. 6, the results
demonstrate the superiority of our approach. Quantitatively, Cog2Gen3D achieves
the highest scores across all metrics, with notable margins in the challenging
multi-object task. Qualitatively, while baseline methods often suffer from blurred
details, geometric distortions, or structural collapse in complex scenes due to
the lack of spatial constraints, our method consistently produces high-fidelity
3D assets. Benefiting from the structural constraints of our 3D cognition graph,
Cog2Gen3D maintains precise geometries and coherent multi-entity relationships,
demonstrating significant advantages in holistic 3D scene generation.

<!-- page 12 -->
12
H. Wang et al.
Chairs
Cars
Planes
(a) Input
(b) DiffTF
(c) DiffGS
(d) LN3Diff
(e) Ours
Fig. 8: Visualization of Image-to-3D objects on the ShapeNet dataset.
Image-to-3D Object Generation. We evaluate image-to-3D object generation
on ShapeNet [26] and OmniObject3D [28] against leading baselines [35–40]. We
adopt standard evaluation metrics such as FID, KID, and MMD. Tab. 2 and
Fig. 8 shows that Cog2Gen3D establishes a clear advantage over existing baselines.
Quantitatively, our model achieves the top scores across all metrics and datasets.
Qualitatively, our framework reliably reconstructs detailed 3D assets. Leveraging
the semantic and geometric perception of our 3D cognition graph, Cog2Gen3D
excels at preserving high-fidelity appearances and plausible geometric structures.
Image-to-3D Scene Generation. We validate complex scene generation on
3D-Front [33] and our CogSG-3D datasets against semantic [41–45] and 2D
geometry-guided [9,14] baselines. We employ standard metrics such as Chamfer
Distance (CD), F-Score, and IoU to evaluate the structural plausibility and
visual fidelity of the generated 3D scenes. The quantitative results in Tab. 3
shows that Cog2Gen3D consistently outperforms all competing baselines. Visual
comparisons in Fig. 9 reveal that existing approaches often struggle with scale
inconsistencies, chaotic spatial layouts, or structural collapse when synthesizing
cluttered spaces due to the lack of spatial awareness. Conversely, our model
generates highly realistic and well-organized 3D environments by embedding
semantic and absolute geometric cognitive features into the generation process.
5.3
Ablation Study
Effectiveness of Cognitive Feature Embeddings. A user study (Tab. 4)
evaluates the distinct roles of the three cognitive tokens across Semantic Fidelity
(SemF), Geometric Plausibility (GeoP), and Relational Coherence (RelC). Ab-
lating the semantic, geometric, or logical tokens severely degrades SemF, GeoP,
or RelC, respectively. This direct correspondence confirms that integrating all

<!-- page 13 -->
Cog2Gen3D
13
Table 3: Image-to-3D scenes comparison.
Method
3D-Front [33]
Chamfer Dist.↓F-Score↑
IoU↑
LucidDreamer [41]
0.083
50.79
0.536
SSR [42]
0.140
39.76
0.311
Gen3DSR [43]
0.123
40.07
0.363
REPARO [44]
0.129
41.68
0.339
MIDI [45]
0.080
50.19
0.518
EchoScene [9]
0.105
45.62
0.458
Layout2Scene [14]
0.094
48.36
0.492
Ours
0.063
58.43
0.682
Table 4: Ablation study on cognitive tokens.
Cognitive Tokens
User Study ↑
Semantic Geometric Logical SemF GeoP RelC
✓
4.12
2.38
2.15
✓
2.23
4.08
2.74
✓
2.41
2.58
3.92
✓
✓
4.38
4.35
2.82
✓
✓
4.42
2.75
4.28
✓
✓
3.17
4.31
4.34
✓
✓
✓
4.65
4.58
4.62
(b) Gen3DSR
(a) Input Image
(c) LucidDreamer
(d) Ours
Fig. 9: Visualization of Image-to-3D Scenes on 3D-Front and our CogSG-3D datasets.
three embeddings is essential for synthesizing scenes with high-fidelity semantics,
plausible geometry, and coherent logic.
Superiority of Latent Graph Construction. We validate our structured
topology by replacing it with a flattened token sequence (Tab. 5). This ablation
degrades performance, indicating that flat sequences fail to capture complex
3D spatial dependencies. Conversely, our structured graph effectively models
semantic-geometric interactions, significantly enhancing the structural rationality
and geometric plausibility of generated scenes.
Impact of Dual-Stream Scene Graphs. We ablate the semantic and geometric
streams to assess their contributions. Fig. 10 shows that removing either stream
degrades performance. Moreover, omitting semantic graphs compromises texture
fidelity and lacking geometric graphs causes severe structural distortions. This
confirms both streams are indispensable for high-quality 3D generation.
5.4
Discussion
Correspondence between the Cognition Graph and 3D Scenes. To
interpret the latent cognition graph, we compute attention maps between the
generated 3D scenes and node/edge embeddings. Fig. 5 shows that high-attention
regions precisely align with target objects and spatial boundaries. This confirms

<!-- page 14 -->
14
H. Wang et al.
Table 5: Graph cons. ablation.
Settings
3D-Front [33]
CD ↓F-Score ↑IoU ↑
w/o Graph
0.089
49.16
0.503
w/. Graph 0.063
58.43
0.683
Table 6: Geometry encoders.
Settings
3D-Front [33]
CD ↓F-Score ↑IoU ↑
ResNet50
0.091
48.75
0.498
CLIP ViT-L 0.076
53.88
0.591
VGGT
0.063
58.43
0.683
Table 7: Fusion strategy.
Settings
3D-Front [33]
CD ↓F-Score ↑IoU ↑
Concat.
0.085
50.41
0.523
Weighted 0.072
55.12
0.635
Common 0.063
58.43
0.683
4) w/o Geometric Graph
1) Input
3) w/o Semantic Graph
2) Full Model
Chamfer
Distance
F-Score
IoU
w/o Semantic Graph
w/o Geometric Graph
Full Model
0.02
0
0.04 0.06 0.08 0.10 0.12 0.14
0.063
0.075
0.103
10
0
20
30
40
50
60
70
58.43
45.76
39.61
0.1
0
0.2
0.3
0.4
0.5
0.6
0.7
0.683
0.587
0.452
Fig. 10: Ablation of dual-stream scene graphs. Quantitative metrics drop no-
ticeably without either stream. Qualitative examples reveal that omitting semantic or
geometric graphs causes severe texture degradation or structural collapse, respectively.
that the graph acts as a structured cognitive map, successfully localizing semantic-
geometric entities to bridge abstract prompts with 3D scenes.
Influence of the Geometry Perception Backbone. We evaluate different ge-
ometry encoders in Tab. 6. The VGGT encoder [19] achieves optimal performance
by excelling at extracting robust geometric features and spatial structures. This
provides precise absolute geometry guidance for the diffusion process, significantly
enhancing the geometric plausibility of the generated 3D scenes.
Effectiveness of Common-based Fusion Strategy. We compare our common-
based fusion against standard concatenation and weighted fusion. Tab. 7 shows
that our approach effectively synergizes semantic and geometric features. This
deep alignment preserves modality-specific priors, yielding the highest generation
quality in both semantic fidelity and geometric plausibility.
Limitations. While Cog2Gen3D demonstrates strong capabilities in generating
high-quality and geometrically plausible static 3D scenes, it currently struggles
with dynamic 4D generation. This limitation primarily arises from the absence of
temporal modeling within our framework. Our 3D cognition graph and Gaussian
representation [24] are restricted to static constraints, failing to capture motion
or topological evolution. Future work will integrate spatio-temporal graphs and
4D Gaussian Splatting [46] to enable the generation of spatially grounded and
temporally consistent dynamic scenes.
6
Conclusion
In this work, we proposed Cog2Gen3D, a 3D cognition-guided diffusion framework
designed to sculpture semantic-geometric cognition for high-quality 3D generation.

<!-- page 15 -->
Cog2Gen3D
15
By integrating absolute geometric priors and semantic constraints, our method
effectively resolves the challenges of scale inconsistency and the absence of
spatial awareness prevalent in existing models. We introduced cognitive feature
embeddings to map multi-modal information into a unified semantic-geometric
space, and proposed the 3D latent cognition graph to capture complex spatial
topological relationships and structural dependencies within 3D environments.
Furthermore, we constructed the CogSG-3D dataset, providing extensive explicit
scene graph and 3D Gaussian annotations to support training. The superiority
of our approach has been demonstrated through extensive experiments across
multiple tasks, including text-to-3D, image-to-3D object, and complex scene
generation, affirming its effectiveness in modeling the physical world.
References
1. Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to
text-to-image diffusion models. In Proceedings of the IEEE/CVF international
conference on computer vision, pages 3836–3847, 2023. 1
2. Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej
Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al.
Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv
preprint arXiv:2311.15127, 2023. 1
3. Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion:
Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 1, 4, 11
4. Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich.
Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
pages 12619–12629, 2023. 1, 4, 11
5. Kai Wang, Yu-An Lin, Ben Weissmann, Manolis Savva, Angel X Chang, and Daniel
Ritchie. Planit: Planning and instantiating indoor scenes with relation graph and
spatial prior networks. ACM Transactions on Graphics (TOG), 38(4):1–15, 2019.
2, 4
6. Helisa Dhamo, Fabian Manhardt, Nassir Navab, and Federico Tombari. Graph-
to-3d: End-to-end generation and manipulation of 3d scenes using scene graphs.
In Proceedings of the IEEE/CVF International Conference on Computer Vision,
pages 16352–16361, 2021. 2, 4
7. Guangyao Zhai, Evin Pınar Örnek, Shun-Cheng Wu, Yan Di, Federico Tombari,
Nassir Navab, and Benjamin Busam. Commonscenes: Generating commonsense
3d indoor scenes with scene graph diffusion.
Advances in Neural Information
Processing Systems, 36:30026–30038, 2023. 2, 4
8. Chenguo Lin and Yadong Mu. Instructscene: Instruction-driven 3d indoor scene
synthesis with semantic graph prior. arXiv preprint arXiv:2402.04717, 2024. 2, 4
9. Guangyao Zhai, Evin Pınar Örnek, Dave Zhenyu Chen, Ruotong Liao, Yan Di,
Nassir Navab, Federico Tombari, and Benjamin Busam. Echoscene: Indoor scene
generation via information echo over scene graph diffusion. In European Conference
on Computer Vision, pages 167–184. Springer, 2024. 2, 4, 12, 13
10. Manyi Li, Akshay Gadi Patil, Kai Xu, Siddhartha Chaudhuri, Owais Khan, Ariel
Shamir, Changhe Tu, Baoquan Chen, Daniel Cohen-Or, and Hao Zhang. Grains:
Generative recursive autoencoders for indoor scenes. ACM Transactions on Graphics
(TOG), 38(2):1–16, 2019. 2, 4

<!-- page 16 -->
16
H. Wang et al.
11. Sherwin Bahmani, Jeong Joon Park, Despoina Paschalidou, Xingguang Yan, Gordon
Wetzstein, Leonidas Guibas, and Andrea Tagliasacchi. Cc3d: Layout-conditioned
generation of compositional 3d scenes. In Proceedings of the IEEE/CVF Interna-
tional Conference on Computer Vision, pages 7171–7181, 2023. 2, 4
12. Ryan Po and Gordon Wetzstein. Compositional 3d scene generation using locally
conditioned diffusion. In 2024 International Conference on 3D Vision (3DV), pages
651–663. IEEE, 2024. 2, 4
13. Xiuyu Yang, Yunze Man, Junkun Chen, and Yu-Xiong Wang. Scenecraft: Layout-
guided 3d scene generation. Advances in Neural Information Processing Systems,
37:82060–82084, 2024. 2, 4
14. Minglin Chen, Longguang Wang, Sheng Ao, Ye Zhang, Kai Xu, and Yulan Guo. Lay-
out2scene: 3d semantic layout guided scene generation via geometry and appearance
diffusion priors. arXiv preprint arXiv:2501.02519, 2025. 2, 4, 12, 13
15. Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling
geometry and appearance for high-quality text-to-3d content creation. In Proceedings
of the IEEE/CVF international conference on computer vision, pages 22246–22256,
2023. 4, 11
16. Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and
Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with
variational score distillation. Advances in neural information processing systems,
36:8406–8441, 2023. 4, 11
17. Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun
Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d:
High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages 300–309, 2023. 4, 11
18. Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang,
Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from
text to 3d gaussians by bridging 2d and 3d diffusion models. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, pages 6796–6807,
2024. 4, 11
19. Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rup-
precht, and David Novotny. Vggt: Visual geometry grounded transformer. In
Proceedings of the Computer Vision and Pattern Recognition Conference, pages
5294–5306, 2025. 4, 5, 14
20. Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang
Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models.
Advances in Neural Information Processing Systems, 36:20482–20494, 2023. 4
21. Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua
Lin. Pointllm: Empowering large language models to understand point clouds. In
European Conference on Computer Vision, pages 131–147. Springer, 2024. 4
22. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning
for image recognition. In Proceedings of the IEEE conference on computer vision
and pattern recognition, pages 770–778, 2016. 5
23. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al. Learning transferable visual models from natural language supervision. In
International conference on machine learning, pages 8748–8763. PmLR, 2021. 6
24. Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis, et al.
3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph.,
42(4):139–1, 2023. 14

<!-- page 17 -->
Cog2Gen3D
17
25. World Labs. Marble: A multimodal world model. https://www.worldlabs.ai/
blog/marble-world-model, November 2025. Accessed: 2026-03-02. 9
26. Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qix-
ing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su,
et al.
Shapenet: An information-rich 3d model repository.
arXiv preprint
arXiv:1512.03012, 2015. 10, 11, 12
27. Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya
Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al.
Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information
Processing Systems, 36:35799–35813, 2023. 10
28. Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu,
Lei Yang, Jiaqi Wang, Chen Qian, et al. Omniobject3d: Large-vocabulary 3d object
dataset for realistic perception, reconstruction and generation. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, pages 803–814,
2023. 10, 11, 12
29. Xingyuan Sun, Jiajun Wu, Xiuming Zhang, Zhoutong Zhang, Chengkai Zhang,
Tianfan Xue, Joshua B Tenenbaum, and William T Freeman. Pix3d: Dataset and
methods for single-image 3d shape modeling. In Proceedings of the IEEE conference
on computer vision and pattern recognition, pages 2974–2983, 2018. 10
30. Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan
Gundogdu, Xi Zhang, Tomas F Yago Vicente, Thomas Dideriksen, Himanshu Arora,
et al. Abo: Dataset and benchmarks for real-world 3d object understanding. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
pages 21126–21136, 2022. 10
31. Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou
Tang, and Jianxiong Xiao. 3d shapenets: A deep representation for volumetric
shapes. In Proceedings of the IEEE conference on computer vision and pattern
recognition, pages 1912–1920, 2015. 10
32. Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser,
and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor
scenes. In Proceedings of the IEEE conference on computer vision and pattern
recognition, pages 5828–5839, 2017. 10
33. Huan Fu, Bowen Cai, Lin Gao, Ling-Xiao Zhang, Jiaming Wang, Cao Li, Qixun
Zeng, Chengyue Sun, Rongfei Jia, Binqiang Zhao, et al. 3d-front: 3d furnished
rooms with layouts and semantics. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 10933–10942, 2021. 10, 12, 13, 14
34. Yuze He, Yushi Bai, Matthieu Lin, Wang Zhao, Yubin Hu, Jenny Sheng, Ran Yi,
Juanzi Li, and Yong-Jin Liu. T3bench: Benchmarking current progress in text-to-3d
generation. arXiv preprint arXiv:2310.02977, 2023. 11
35. Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini
De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis,
et al. Efficient geometry-aware 3d generative adversarial networks. In Proceedings
of the IEEE/CVF conference on computer vision and pattern recognition, pages
16123–16133, 2022. 11, 12
36. Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing
Li, Or Litany, Zan Gojcic, and Sanja Fidler. Get3d: A generative model of high
quality 3d textured shapes learned from images. Advances in neural information
processing systems, 35:31841–31854, 2022. 11, 12
37. Norman Müller, Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulo, Peter
Kontschieder, and Matthias Nießner. Diffrf: Rendering-guided 3d radiance field

<!-- page 18 -->
18
H. Wang et al.
diffusion. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 4328–4338, 2023. 11, 12
38. Ziang Cao, Fangzhou Hong, Tong Wu, Liang Pan, and Ziwei Liu. Large-vocabulary
3d diffusion model with transformer. arXiv preprint arXiv:2309.07920, 2023. 11, 12
39. Junsheng Zhou, Weiqi Zhang, and Yu-Shen Liu. Diffgs: Functional gaussian splatting
diffusion. Advances in Neural Information Processing Systems, 37:37535–37560,
2024. 11, 12
40. Yushi Lan, Fangzhou Hong, Shuai Yang, Shangchen Zhou, Xuyi Meng, Bo Dai,
Xingang Pan, and Chen Change Loy. Ln3diff: Scalable latent neural fields diffusion
for speedy 3d generation. In ECCV, 2024. 11, 12
41. Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu
Lee. Luciddreamer: Domain-free generation of 3d gaussian splatting scenes. arXiv
preprint arXiv:2311.13384, 2023. 12, 13
42. Yixin Chen, Junfeng Ni, Nan Jiang, Yaowei Zhang, Yixin Zhu, and Siyuan Huang.
Single-view 3d scene reconstruction with high-fidelity shape and texture. In 2024
International Conference on 3D Vision (3DV), pages 1456–1467. IEEE, 2024. 12,
13
43. Andreea Ardelean, Mert Özer, and Bernhard Egger. Gen3dsr: Generalizable 3d
scene reconstruction via divide and conquer from a single view. In 2025 International
Conference on 3D Vision (3DV), pages 616–626. IEEE, 2025. 12, 13
44. Haonan Han, Rui Yang, Huan Liao, Jiankai Xing, Zunnan Xu, Xiaoming Yu, Junwei
Zha, Xiu Li, and Wanhua Li. Reparo: Compositional 3d assets generation with
differentiable 3d layout alignment. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 25367–25377, 2025. 12, 13
45. Zehuan Huang, Yuan-Chen Guo, Xingqiao An, Yunhan Yang, Yangguang Li, Zi-Xin
Zou, Ding Liang, Xihui Liu, Yan-Pei Cao, and Lu Sheng. Midi: Multi-instance
diffusion for single image to 3d scene generation. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 23646–23657, 2025.
12, 13
46. Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei,
Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time
dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pages 20310–20320, 2024. 14
