<!-- page 1 -->
Animate Any Character in Any World
Yitong Wang1,2*
Fangyun Wei2*
Hongyang Zhang3
Bo Dai4†
Yan Lu2
1Fudan University
2Microsoft Research
3University of Waterloo
4The University of Hong Kong
https://snowflakewang.github.io/AniX/
Any Character
Any Scene
Long-Horizon, Temporally Coherent Interaction
Controllable Camera Behavior
Consistent Environment and Character Fidelity
Rich Action Repertoire
User
Run Forward
Run to the Right
Play a Guitar
Figure 1. AniX enables users to provide 3DGS scene along with a 3D or multi-view character, enabling interactive control of the character’s
behaviors and active exploration of the environment through natural language commands. The system features: (1) Consistent Environment
and Character Fidelity, ensuring visual and spatial coherence with the user-provided scene and character; (2) a Rich Action Repertoire cov-
ering a wide range of behaviors, including locomotion, gestures, and object-centric interactions; (3) Long-Horizon, Temporally Coherent
Interaction, enabling iterative user interaction while maintaining continuity across generated clips; and (4) Controllable Camera Behavior,
which explicitly incorporates camera control—analogous to navigating 3DGS views—to produce accurate, user-specified viewpoints.
Abstract
Recent advances in world models have greatly enhanced in-
teractive environment simulation. Existing methods mainly
fall into two categories: (1) static world generation models,
which construct 3D environments without active agents, and
(2) controllable-entity models, which allow a single entity
to perform limited actions in an otherwise uncontrollable
environment. In this work, we introduce AniX, leveraging
*Equal contribution.
†Corresponding author.
the realism and structural grounding of static world gen-
eration while extending controllable-entity models to sup-
port user-specified characters capable of performing open-
ended actions. Users can provide a 3DGS scene and a char-
acter, then direct the character through natural language
to perform diverse behaviors—from basic locomotion to
object-centric interactions—while freely exploring the envi-
ronment. AniX synthesizes temporally coherent video clips
that preserve visual fidelity with the provided scene and
character, formulated as a conditional autoregressive video
generation problem. Built upon a pre-trained video gen-
1
arXiv:2512.17796v1  [cs.CV]  18 Dec 2025

<!-- page 2 -->
erator, our training strategy significantly enhances motion
dynamics while maintaining generalization across actions
and characters. Our evaluation covers a broad range of
aspects, including visual quality, character consistency, ac-
tion controllability, and long-horizon coherence.
1. Introduction
Recent advances in world models have led to substan-
tial progress in simulating dynamic and interactive envi-
ronments. Existing methods generally fall into two cate-
gories: (1) static world generation approaches [43, 55, 89],
which construct explorable 3D environments but lack ac-
tive agents; and (2) controllable-entity approaches [45, 90],
which allow a single agent to execute only a limited set
of actions, such as steering a vehicle along a predefined
path [14, 29, 73], while leaving the environment itself
uncontrollable.
In this work, we propose an alternative
framework that combines the strengths of both paradigms:
leveraging the realism and structural grounding of static
world generation while extending controllable-entity mod-
els to support user-specified characters capable of perform-
ing open-ended actions.
Specifically, users can provide a 3D Gaussian Splatting
(3DGS) scene [38]—representing either a synthetic envi-
ronment or a real-world reconstruction—along with a 3D
or multi-view character. Through natural language instruc-
tions, users can control the character’s behavior and enable
active exploration within the environment.
At each iter-
ation, the model generates a video clip that captures the
evolving states of both the character and the environment,
resulting in coherent and temporally consistent generation.
We name our method AniX. As illustrated in Figure 1, AniX
exhibits several key capabilities:
1. Consistent Environment and Character Fidelity. The
visual contents appearing in the generated video clips
exhibit strong consistency in visual identity and spatial
layout with the user-provided scene and character.
2. Rich Action Repertoire. Unlike previous works [7, 29,
45] that limit the controllable entity to basic locomotion,
our model enables the character to perform up to hun-
dreds of distinct actions, encompassing not only naviga-
tion behaviors (e.g., “moving forward”, “turning left”)
but also body-language gestures (e.g., “waving hands”,
“saluting”) and object-centric interactions (e.g., “making
a phone call”, “playing a guitar”).
3. Text Instruction as the Interface. Users can directly
guide the character through natural language commands.
4. Long-Horizon,
Temporally Coherent Interaction.
Users can interact with the model iteratively, generating
new video clips that remain temporally consistent with
previously produced sequences.
5. Controllable Camera Behavior. Our model supports
flexible and intuitive camera control, allowing behav-
iors such as following a character’s trajectory or orbiting
around it to achieve user-specified viewpoints. Unlike
previous methods [27, 28] that encode camera trajec-
tories into Pl¨ucker embeddings [71] and inject them as
conditioning signals into the generation network, our ap-
proach achieves camera control in a more geometrically
grounded manner. Specifically, given a user-provided
3DGS scene and a defined camera path, we directly ren-
der a projection scene video along the specified trajec-
tory. This rendered video serves as an explicit condi-
tioning input, enabling the generation model to produce
videos that accurately follow the desired camera motion.
We formulate the entire process as a conditional autore-
gressive video generation problem. Concretely, the objec-
tive is to synthesize a video clip at each iteration, condi-
tioned on a set of multi-modal inputs including: (a) the
user-provided scene and character, which establish the spa-
tial and visual grounding; (b) a text instruction, which spec-
ifies the intended behavior of the character; and (c) the pre-
viously generated video clips, which serve as temporal ref-
erences to ensure consistency across iterations.
Additionally, we adopt a pre-trained video generator [31]
as the foundation of our framework.
We find that fine-
tuning it on a small dataset containing basic locomotion ac-
tions across diverse characters not only preserves the gener-
alization ability of the pre-trained model but also enhances
overall motion quality compared to the original generator.
This phenomenon is further analyzed in Section 4.1.
In our experiments, we comprehensively evaluate the
proposed model from multiple perspectives, including:
(a) visual quality, assessed using the WorldScore bench-
mark [11]; (b) character consistency, which measures the
alignment between the character appearing in the generated
video and the user-provided reference; (c) action control
success rate, which quantifies how accurately the charac-
ter’s behavior follows the input text instruction across a di-
verse set of up to around 150 actions; and (d) long-horizon
generation quality, which evaluates the model’s ability to
maintain temporal coherence and visual fidelity over ex-
tended interaction sequences. We compare AniX with both
video generation foundation models [39, 78, 88] and dedi-
cated world models [7, 29, 45]. Experimental results show
that our method consistently outperforms both categories
across nearly all evaluation metrics.
2. Related Works
Controllable Video Generation. Recent foundation mod-
els for video generation [4, 21, 25, 39, 40, 60, 78, 88] have
greatly improved modality alignment across text, image,
and video, enabling large-scale pre-training for both text-
to-video and image-to-video synthesis. Building on these
advances, subsequent research has pursued finer-grained
2

<!-- page 3 -->
controllability by introducing mechanisms such as explicit
subject control [6, 9, 13, 34–36, 51, 54] and camera con-
trol [1, 15, 22, 24, 57, 67, 80, 85, 86, 95, 96, 102]. For
subject control, a typical approach [31, 49] is to extract
visual embeddings from reference images and use them
within a Multimodal Diffusion Transformer [12, 63] to
guide the generated subjects to remain consistent with the
reference appearance.
For camera control, one common
practice [27, 28] is to convert the camera path into Pl¨ucker
embeddings [71] and inject them into the main network,
guiding the synthesized video to follow the specified tra-
jectory. In contrast, our model controls the camera by nav-
igating through 3DGS views: given a 3DGS scene and a
camera path, we render a projection video along the path,
which conditions the generator to follow the desired motion
faithfully.
Memory Mechanism in Video Generation. Recent works
incorporate memory mechanisms to improve long-term spa-
tial and temporal consistency in video generation. These
approaches retrieve generation history to localize relevant
content across modalities such as RGB [93] and depth [7],
often using surfel-indexed view selection [46] or camera
FOV overlap [82, 93]. Other methods [32, 33, 81, 92] main-
tain a global point cloud map during generation, enabling
the model to identify and reuse the most relevant spatial re-
gions, thereby maintaining coherence across continuously
generated scenes. In our work, the video generation with
memory mechanism is realized by conditioning on both the
character and the 3DGS scene. The 3DGS scene serves as
a spatial memory that explicitly encodes the geometric and
appearance information of the environment, while the char-
acter provides dynamic cues for motion and interaction.
World Models for Static Scene Creation. Existing world
models that generate static yet explorable environments can
be broadly categorized into two types. The first type [5, 10,
14, 20, 29, 37, 41, 42, 45, 58, 73, 77, 84, 90, 94, 99] stores
the world implicitly within neural networks, using video
generation models [21, 39, 78, 88] to visualize it. Users pro-
vide navigation commands (e.g., “camera forward”) drawn
from a predefined set of camera trajectories, and the model
synthesizes new frames along this path while maintain-
ing spatial consistency with past generations. The second
type [47, 70] explicitly constructs a 3DGS world, where
multi-view images are optimized to form a manipulable 3D
representation, allowing users to render novel views from
arbitrary camera poses. Further developments extend this
paradigm by using panoramic inputs [89, 100] or directly
generating 3DGS representations from text or a single im-
age through feed-forward networks [18, 50, 55, 74, 87],
while others [8, 26, 72, 79, 98] integrate video generation
to streamline and accelerate 3DGS creation. In this work,
users can either create or specify a 3DGS scene before gen-
eration. When users do not provide one, we adopt Mar-
ble [43] to automatically generate a static 3DGS world.
3. Method
Problem Formulation.
Given a pre-generated or real-
world reconstructed 3DGS scene S and a user-specified 3D
character C, AniX enables users to iteratively control the
character C through text instructions T within the scene
S, generating long-horizon, temporally coherent video clips
that remain visually consistent with both S and C.
We formulate this process as a conditional autoregressive
video generation problem. At each iteration, AniX synthe-
sizes the current video clip conditioned on multiple multi-
modal inputs: the previous generated clip, character rep-
resentations, scene representations, and the current text in-
struction. Overviews of the training and inference pipelines
are shown in Figure 2 and Figure 4, respectively.
3.1. Training
3.1.1. Training Data Pre-Processing
Training Set Construction. As shown in Figure 2(a), our
training data is GTA-V [17], a game where players can con-
trol a character to perform basic actions such as “run for-
ward”. We record gameplay sequences and segment them
into short video clips, ensuring that each clip (1) contains
only a single action and (2) has a fixed length of 129 frames.
For each clip V , we apply the following steps:
1. Character Segmentation. We use Grounded-SAM-2 [66]
to segment characters and extract their bounding-box
mask sequence, denoted as M.
2. Scene Inpainting. We remove the segmented characters
and apply DiffuEraser [48] to fill the missing regions,
yielding the inpainted scene video S.
3. Action Labeling. Each clip is then annotated with a con-
cise text label T describing the action performed by the
character, such as “The character is running forward”.
GTA-V also provides access to the 3D character mod-
els used in the game.
To ease the character modeling,
we represent each character using four viewpoint ren-
ders [2, 3, 97]—front, left, right, and back—denoted as
C = {CF , CL, CR, CB}.
Finally, each processed training sample is represented as
a tuple {V , S, M, T , C}, forming a structured dataset.
Token Extraction. As shown in Figure 2(a), given a train-
ing sample {V , S, M, T , C}, we adopt the video VAE en-
coder from HunyuanCustom [31] to extract tokens for the
video V , scene S, and mask M. The resulting token se-
quences are denoted as TV , TS and TM, respectively. The
video VAE encoder operates with a spatial downsampling
rate of 8 and a temporal downsampling rate of 4.
Note that the video VAE encoder can also be applied to
single images. Therefore, for the multi-view character C =
{CF , CL, CR, CB}, we use the same encoder to extract
3

<!-- page 4 -->
Multi-View Character 
Tokens
Video VAE
Encoder
Video VAE
Encoder
LLaVA
Encoder
Text: “The Character 
is running.”
Video VAE
Encoder
Video VAE
Encoder
Rendered Video
Animation with 
Game Engine
Character Mask
3D Character
Scene Video
Multi-View 2D Character
F
L
B
R
F
L
B
R
…
Scene Tokens
…
Target Tokens
…
Mask Tokens
…
…
…
F
L
B
R
…
Noisy 
Target Tokens
…
Projector
Denoised 
Target Tokens
Decoder
: Addition                 : Concatenation
C
C
(a) Training Data Preprocessing
(b) Architecture
Segmentation 
and Inpainting
Conditional
Noisy
Tokens
=
Multi-Modal Diffusion Transformer
LoRA
Figure 2. (a) Each training sample consists of a 3D character and a video depicting the character performing an action described by a short
text. Through segmentation and inpainting, we obtain the corresponding scene video and character mask sequence. The VAE encoder is
then applied to encode these inputs into tokens. (b) AniX predicts target video tokens conditioned on scene, mask, text, and multi-view
character tokens within a Multi-Modal Diffusion Transformer, trained using Flow Matching [52]. Refer to Figure 3 for the training process
of the auto-regressive mode, which enables iterative interaction with AniX, and Figure 4 for the inference.
tokens from each view, resulting in the multi-view character
token set TC = {TCF , TCL, TCR, TCB}.
Finally, to extract text tokens, following HunyuanCus-
tom [31], we employ the multi-modal encoder LLaVA [53],
which takes both the text instruction T and character tokens
TC as input. The resulting encoded text tokens are denoted
as TT . Implementation details are provided in the appendix.
Now, the training sample {V , S, M, T , C} is fully en-
coded into the latent space as {TV , TS, TM, TT , TC}.
3.1.2. Architecture
Training Objective. Figure 2(b) illustrates the architec-
ture of AniX, whose backbone consists of a stack of full-
attention Transformer blocks.
We adopt Flow Match-
ing [52] for model training, conditioned on multiple inputs
(i.e., TS, TM, TT , and TC), to guide the generation process
from pure noise to TV .
Concretely, given target video tokens TV , we first sample
t ∈[0, 1] from a logit-normal distribution and initialize the
noise x0 ∼N(0, I) following Gaussian distribution. The
intermediate sample xt = (1 −t)x0 + tTV is then obtained
via linear interpolation. The model is trained to predict the
velocity ut = dxt/dt by minimizing the mean squared er-
ror between the predicted velocity vt and the ground-truth
velocity ut:
L = Et,x0,TV ∥vt −ut∥2 .
(1)
Condition Modeling. We incorporate multiple condition-
ing signals to guide the learning process, including text to-
kens TT , multi-view character tokens TC, scene tokens TS,
and mask tokens TM. As illustrated in Figure 2(b), to inject
scene and mask priors, we directly fuse TS and TM into the
noisy target tokens xt via: x′
t = xt + Projector([TS; TM]),
where [ ; ] denotes channel-wise concatenation.
The
Projector maps the concatenated tokens [TS; TM] to the
same dimensionality as xt using a lightweight linear layer.
We refer to the resulting x′
t as the conditional noisy tokens.
At last, to integrate the text tokens TT and multi-view
character tokens TC, we concatenate TT , TC, and x′
t along
the sequence dimension. The concatenated sequence is then
fed into the backbone—implemented as a stack of full-
attention Transformer blocks—to denoise xt under the su-
pervision of Eq. 1.
Positional Embeddings. Following HunyuanCustom [31],
no positional embeddings are added to the text tokens
4

<!-- page 5 -->
Multi-Modal Diffusion Transformer
F
L B R
…
…
…
Conditional
Noisy
Tokens
Augmented 
Preceding Video 
Tokens
Multi-View 
Character 
Tokens
Text 
Tokens
…
Denoised Target Tokens
Figure 3. Illustration of the auto-regressive mode. The only dif-
ference from the original architecture in Figure 2 is the addition
of an extra conditioning input—the preceding video tokens. Note
that a misalignment exists between training and inference: during
training, the preceding video tokens are derived from ground-truth
videos, whereas during inference, they are generated by the model
itself. To mitigate this discrepancy, we add a small Gaussian noise
to the preceding video tokens during training and refer to the re-
sulting tokens as augmented preceding video tokens.
TT .
A standard 3D-RoPE (over time, height, and
width dimensions) is applied to the conditional noisy to-
kens x′
t.
For the multi-view character tokens TC
=
{TCF , TCL, TCR, TCB}, each TCF , TCL, TCR, and TCB
represents single-view character tokens. For each view, a
shifted-3D-RoPE is applied. Taking TCF as an example,
PETCF (i, j) = 3D-RoPE(−1, i + w, j + h),
(2)
where (w, h) denotes the spatial size of TCF , and the
shifts along the temporal and spatial dimensions are −1 and
(w, h), respectively. For TCL, TCR and TCB, the spatial
shift remains the same, while the temporal shifts are set to
−2, −3, and −4, respectively.
3.1.3. Auto-Regressive Mode
AniX supports multi-round user interaction while maintain-
ing temporal continuity and semantic coherence between
adjacent video clips. To achieve this, we extend AniX into
an auto-regressive mode. Specifically, we divide the tar-
get video tokens TV along the temporal dimension into two
parts: the first quarter, denoted as TV 1, serves as the preced-
ing video tokens, and the remaining three quarters, denoted
as TV [2:4], serve as the new target video tokens.
The model is trained to generate TV [2:4] conditioned on
both the preceding video tokens TV 1 and the other condi-
tioning signals introduced in Section 3.1.2. As shown in
Figure 3, to incorporate the newly added condition TV 1, we
prepend it to the conditional noisy tokens. The fusion strat-
egy for the other conditioning signals remains unchanged,
as described in Section 3.1.2.
3.2. Inference and Acceleration
Inference.
Figure 4 illustrates the inference pipeline of
AniX, which consists of three main stages:
AniX
Previously Generated Clip
3DGS Scene
Virtual
Camera
Character
User
Character Anchor
Instruction: “The character is running forward.”
Instruction Parsing
Locomotion
Camera Behavior
Gesture
Object-Centric Action
Camera Path
Scene Video
Instruction
Character
Scene Video
Anchor
(a) User Configuration
(b) Scene Video Rendering
(c) AniX Inference
Output
Figure 4. Inference of AniX. (a) Users first specify the inputs,
including the character, 3DGS scene, virtual camera location, and
character anchor. (b) The user-provided text instruction is parsed,
and a corresponding camera path is generated. Applying this path
to the 3DGS scene produces a rendered scene video. (c) AniX then
takes multiple inputs as conditions to generate the final output.
Steps (b) and (c) can be performed iteratively, enabling temporally
consistent, long-horizon interactions.
1. User Configuration. Users first specify a character and
a 3DGS scene. They can position a virtual camera at
any desired location within the scene and define a char-
acter anchor (i.e., a bounding-box mask) that determines
where the character appears in the generated video. The
character anchor remains consistent across all generated
frames. Users may also employ existing models to gen-
erate the 3D character (e.g., Hunyuan3D [44]) or the
3DGS scene (e.g., World Labs Marble [43]).
2. Scene Video Rendering. Next, users provide a text in-
struction to AniX. The instruction is parsed into four cat-
5

<!-- page 6 -->
Table 1. WorldScore metrics for evaluating generation quality, categorized into three groups: (1) controllability, (2) quality, and (3)
dynamics. The static and dynamic scores are computed by aggregating metrics from these three groups. † denotes dedicated world models.
Ctrl: Controllability; Align: Alignment; Const: Consistency; Photo: Photometric; Acc: Accuracy; Mag: Magnitude; Smooth: Smoothness.
Method
WorldScore
Controllability
Quality
Dynamics
Static Dyna-
mic
Camera
Ctrl
Object
Ctrl
Content
Align
3D
Const
Photo
Const
Style
Const
Subjective
Quality
Motion
Acc
Motion
Mag
Motion
Smooth
CogVideoX1.5-I2V (5B) [88]
60.08
56.77
42.13
100.00
31.12
68.65 81.43 77.86
19.39
54.92
24.58
67.62
HunyuanVideo-I2V (13B) [39] 56.43
55.14
27.30
100.00
13.70
58.24 93.22 91.97
10.58
59.36
24.46
72.58
Wan2.1-I2V (14B) [78]
57.91
55.87
37.32
98.33
26.34
81.88 83.77 65.03
12.73
59.59
28.60
65.07
Wan2.2-I2V (14B) [78]
54.52
51.74
24.79
100.00
24.03
57.95 59.44 98.91
16.51
59.60
38.93
37.26
Wan-VACE (14B) [35]
51.54
52.03
21.29
100.00
30.39
27.53 54.18 99.02
17.39
56.73
34.78
67.98
Wan-VACE (1.3B) [35]
50.75
49.38
31.12
100.00
11.54
29.66 61.00 97.67
24.26
55.87
29.48
53.17
HunyuanCustom (13B) [31]
62.64
61.11
47.19
100.00
72.07
48.06 31.40 97.47
25.84
59.04
24.05
89.56
DeepVerse† [7]
52.63
47.63
52.48
75.00
18.80
35.47 83.16 92.39
11.09
33.30
33.61
40.97
Hunyuan-GameCraft† [45]
69.92
57.77
77.45
83.33
51.16
85.91 82.12 93.39
16.11
16.65
31.83
39.77
Matrix-Game-2.0† [29]
52.26
43.98
15.10
99.17
12.38
60.29 68.35 97.60
12.92
3.07
47.36
23.59
AniX (Ours)
84.64
77.22
88.91
100.00
75.73
83.57 87.68 98.91
57.72
61.08
24.12
94.47
egories: (a) locomotion, (b) gesture, (c) object-centric
action, and (d) camera behavior. Each category deter-
mines a different camera path.
For (a), AniX gener-
ates a camera trajectory consistent with the motion de-
scribed in the text—for example, for “The character is
running forward”, the camera follows a forward-moving
path. For (b) and (c), the camera remains stationary. For
(d), AniX generates a trajectory matching the specified
camera motion—for instance, for “The camera circles
around the character”, the camera follows a circular path
around the character. AniX then renders a scene video
clip along the corresponding camera trajectory.
3. AniX Inference. Finally, AniX encodes the text instruc-
tion, character, scene, character anchor, and the previ-
ously generated clip (optional, for auto-regressive mode)
into tokens using the encoders illustrated in Figure 2.
These tokens are then fed into the AniX model to gener-
ate the final output.
Note that Steps 2 and 3 can be performed iteratively, en-
abling temporally consistent, long-horizon generation.
Acceleration.
To
accelerate
inference,
we
adopt
DMD2 [91] to distill the original 30-step denoising model
into a more efficient 4-step version.
4. Experiment
Training Details.
Our network is initialized from Hun-
yuanCustom [31], which contains 13B parameters.
We
freeze the LLaVA encoder and the scene condition projec-
tor, and apply LoRA-style [30] fine-tuning to the backbone
with a rank of 64. Two separate models are trained for 360P
and 720P data using the AdamW optimizer with a learn-
ing rate of 1e-4, each for 5,000 steps under the ZeRO-2
strategy. The 360P model is trained on 8× NVIDIA H100
GPUs, while the 720P model, owing to its higher resolution,
is trained on 8× NVIDIA B200 GPUs. Further details on
training and acceleration are provided in the appendix.
Training Data. Following the data generation procedure
described in Section 3.1.1, we construct a training set
comprising 2,084 video samples featuring five characters.
Each sample is annotated with text labels describing either
locomotion actions—{“run forward”, “run tho the left”,
“run to the right”, “run backward”}—or camera behav-
iors—{“orbit”, “follow” }. A key observation of AniX is
that post-training on such simple locomotion and camera-
behavior data can substantially enhance pre-trained models,
improving both action dynamics and camera control capa-
bility. For each sample, we generate two resolutions—360P
and 720P—to train models of different quality levels. Un-
less otherwise specified, the 360P version is used by default.
Additional details are provided in the appendix.
Evaluation. We evaluate our model across four aspects:
(1) visual quality, (2) camera controllability, (3) action con-
trol capability and generalization to novel actions, and (4)
character consistency on novel characters. For (1) and (2),
we adopt the WorldScore [11] metrics to assess the gen-
erated samples. For (3), we measure the control success
rate via human evaluation and report the CLIP [64] text-
to-image similarity score, covering both the four seen lo-
comotion actions and up to 142 novel actions. For (4), we
assess character similarity between the ground-truth charac-
ter and the generated one using 30 novel characters, evalu-
ated by DINOv2 [61] and CLIP [64] scores. By default, we
use the 360P version of our model. At each iteration, our
model generates 96 frames, using the previously generated
33 frames as conditions when available. Unless otherwise
noted, evaluations are conducted on the first generated clip.
4.1. Main Results
Visual Quality Evaluation. We adopt the metrics proposed
by WorldScore [11] to evaluate the visual quality of gen-
6

<!-- page 7 -->
Table 2. Action control and character consistency evaluation on general foundation models and dedicated world models†. Locomotion
actions include “run forward”, “run to the left”, “run to the right”, and “run backward”, while richer actions encompass 142 gesture and
object-centric actions. Note that the three world models restrict action control to locomotion only.
Method
Action Control
Character Consistency
Success Rate (%)
CLIP Text-Image Score
DINOv2
Score
CLIP
Score
Locomotion
Actions
Richer
Actions
Locomotion
Actions
Richer
Actions
CogVideoX1.5-I2V (5B) [88]
23.3
21.1
0.261
0.273
0.594
0.611
HunyuanVideo-I2V (13B) [39]
26.7
35.2
0.272
0.293
0.645
0.709
Wan2.1-I2V (14B) [78]
26.7
64.8
0.267
0.302
0.627
0.678
Wan2.2-I2V (14B) [78]
53.3
74.6
0.272
0.303
0.650
0.704
Wan-VACE (14B) [35]
43.3
73.2
0.270
0.303
0.398
0.541
Wan-VACE (1.3B) [35]
26.7
13.4
0.261
0.269
0.504
0.548
HunyuanCustom (13B) [31]
56.7
51.4
0.273
0.297
0.558
0.665
DeepVerse† [7]
6.7
-
0.259
-
0.291
0.523
Hunyuan-GameCraft† [45]
16.7
-
0.255
-
0.329
0.529
Matrix-Game-2.0† [29]
3.3
-
0.255
-
0.339
0.524
AniX (Ours)
100.0
80.7
0.279
0.305
0.698
0.721
Figure 5. Screenshot visualizations of videos generated by AniX, showcasing different characters performing various actions across two
scenes. Additional examples are provided in the appendix.
Figure 6.
Using both visual conditions—the 3DGS scene and
multi-view character—significantly improves long-horizon inter-
active video generation across diverse video clips.
erated videos. WorldScore defines a suite of metrics cat-
egorized into three groups: Controllability, Quality, and
Dynamics. The first two primarily assess visual fidelity in
static regions, while the last evaluates motion quality in dy-
namic regions (i.e., the character in our case).
We compare our model with dedicated world mod-
els—including DeepVerse [7], Hunyuan-GameCraft [45],
and Matrix-Game-2.0 [29]—that focus on controlling the
main entity, as well as with general video generation foun-
dation models listed in Table 1.
For each model, we use the first two metric groups to
evaluate 60 generated videos covering 30 different char-
acters and two camera behaviors, {“orbit”,“follow”}. The
third metric group is evaluated on 146 videos, where each
video features the same character performing a distinct ac-
tion—either one of four locomotion actions {“run forward”,
“run to the right”, “run to the left”, “run backward”} or one
of 142 novel actions unseen during training. Note that (1)
both foundation and dedicated models are evaluated using
their original control mechanisms for character and cam-
era, without any modification; (2) the three dedicated world
models only support locomotion actions, thus they are eval-
uated solely on those; and (3) for models requiring an initial
image input, we use Google Gemini [19] to render the cor-
responding character within the scene as the initialization
image. The results are shown in Table 1.
Action Control and Generalization. In Table 2, we eval-
uate the model’s action control capability on both seen ac-
tions (four locomotion actions) and 142 novel actions (re-
ferred to as “richer actions”). The novel actions cover both
7

<!-- page 8 -->
Table 3. Using multi-view character inputs improves character
consistency. By default, we employ all four views.
Character View
DINOv2 Score
CLIP Score
Front
0.556
0.628
+Back
0.613
0.678
+Right and Left
0.698
0.721
gesture-based behaviors (e.g., “wave hands”) and object-
centric interactions (e.g., “play a guitar”). For each model,
we conduct 30 evaluations on locomotion actions and 142
evaluations on distinct novel actions using the same set of
characters. We report the action control success rate via hu-
man evaluation and the CLIP text-image similarity score,
computed as the average frame-wise text-image score.
The results in Table 2 reveal that our model outperforms
the base model, HunyuanCustom [31]. This phenomenon
can be interpreted through the lens of post-training in large
language models [23, 62, 65, 83], where fine-tuning typi-
cally does not disrupt the pre-trained representation space;
rather, it adjusts the response style—for example, to make
the outputs more helpful or harmless—while preserving the
extensive knowledge acquired during pre-training. In our
case, the structurally simple fine-tuning data—composed
primarily of fundamental locomotion behaviors—serve to
refine motion dynamics and align human embodiment rep-
resentations, rather than to redefine the model’s generative
space. Consequently, our fine-tuning strategy enhances mo-
tion fidelity and behavioral coherence while maintaining the
broad semantic and generative generalization inherited from
large-scale pre-training.
Character Consistency Evaluation. A key feature of our
model is its ability to maintain consistent visual identity be-
tween the provided character and the one appearing in the
generated videos. In Table 2, we evaluate character consis-
tency using DINOv2 and CLIP scores. Both metrics mea-
sure the similarity between the generated and ground-truth
characters. During evaluation, we crop the character re-
gion from each generated frame and compute the similar-
ity score; the final result is averaged across all frames. For
our method, since multi-view character inputs are provided,
each generated frame is compared against multiple ground-
truth views, and the maximum similarity score is taken as
the frame’s score. Each model is evaluated 30 times, each
using a different character performing locomotion actions.
Visualization. Figure 5 presents screenshots from the gen-
erated videos. More examples are provided in the appendix.
4.2. Ablation Study
Multi-View Character Condition. We compare our four-
view character-conditioned model with two baselines: a
single-view model and a front–back-view model. To evalu-
ate character consistency, we generate videos by instruct-
Table 4. Using per-frame character anchors helps the model dis-
tinguish dynamic entities from the static scene, leading to higher
DINOv2 and CLIP character consistency scores.
Character Anchor Type
DINOv2 Score
CLIP Score
w/o Anchor
0.477
0.529
w/ First-Frame Anchor
0.597
0.645
w/ Per-Frame Anchor
0.698
0.721
ing characters to run toward the front, left, right, and
back, which naturally reveals their appearance from multi-
ple viewpoints. We then compute DINOv2 and CLIP scores
following the protocol in Section 4.1. We use the same
dataset as that used in Table 2. As shown in Table 3, char-
acter consistency improves as more view inputs are used.
Character Anchor Condition.
As illustrated in Fig-
ure 2, we introduce an additional condition—the character
mask—to help the model distinguish between dynamic en-
tities and the static scene. In our default configuration, a
mask is extracted for each frame during training. During
inference, as shown in Figure 4, users only need to spec-
ify a single character anchor (i.e., a bounding-box mask),
which is shared across all generated frames. Table 4 com-
pares our default model (with per-frame masks) against two
variants: (1) without any anchor mask during both train-
ing and inference, and (2) using only the first-frame anchor
mask throughout training and inference.
Visual Conditions Enhance Long-Horizon Generation.
We introduce two types of visual conditions—multi-view
character and 3DGS scene—to ensure both character and
scene consistency during generation.
Beyond improving
spatial coherence, these visual conditions also alleviate the
issue of visual quality degradation over long-horizon gen-
eration. To validate this, we compare AniX operating in
the auto-regressive mode (see Section 3.1.3) against two
variants: (1) using only the multi-view character condition,
where the 3DGS scene condition is replaced with textual
scene descriptions; and (2) using only the 3DGS scene con-
dition, where the multi-view character condition is replaced
with textual character descriptions.
Figure 6 presents a
comparison with the two variants, using the CLIP-Aesthetic
and DINOv2 scores as metrics to evaluate visual quality and
character consistency, respectively. We conduct 10 evalua-
tions for each model, using a different character in each run.
Acceleration. Our 13B-parameter base model generates a
93-frame 360P video in 121s on a single NVIDIA H100 us-
ing a 30-step denoising schedule. By applying DMD2 [91],
we distill it into a 4-step version, reducing latency to
21s with only slight drops in DINOv2 (0.698→0.669)
and CLIP-Aesthetic (5.665→5.583) scores. Latencies for
higher resolutions are provided in the appendix.
8

<!-- page 9 -->
5. Conclusion
We present AniX, a novel framework that allows users to
provide a character and a 3DGS scene, enabling itera-
tive interaction for both character control and world explo-
ration. Unlike prior controllable-entity models that restrict
the agent to a small set of predefined actions, AniX sup-
ports open-ended control over a wide range of behaviors
through natural commands. AniX delivers substantial im-
provements in motion dynamics and character consistency
over its base model, as validated across a broad set of met-
rics including visual quality, action controllability, character
fidelity, and long-horizon generation capability.
9

<!-- page 10 -->
Animate Any Character in Any World
Supplementary Material
6. More Implementation Details
Text Token Extraction. Following HunyuanCustom [31],
we use the multi-modal encoder LLaVA [53] to extract text
tokens while incorporating the multi-view character images.
Concretely, given the text instruction T and the character
views C = {CF , CL, CR, CB}, we construct the follow-
ing prompt:
“A character is [Action]. <SEP>The character front
view looks like E(CF ).
<SEP>The character left
view looks like E(CL). <SEP>The character back
view looks like E(CB). <SEP>The character right
view looks like E(CR).”
Here, [Action] denotes the action description, E(·) is the
LLaVA image encoder, and <SEP>is the separation token
used to distinguish text and visual modalities.
More Training Details.
The model is initialized from
the pre-trained weights of HunyuanCustom [31], a subject-
conditioned variant of HunyuanVideo [39]. Core compo-
nents—including the LLaVA encoder, the scene-condition
projector, and the MMDiT [12, 63] backbone—are kept
frozen. Trainable parameters are introduced by injecting
LoRA modules [30] (rank 64) into the attention query, key,
value, and projection matrices, as well as into the fully con-
nected layers. We optimize the model using AdamW [56]
with a learning rate of 1e-4 and 500 warm-up steps. The
scene condition is randomly dropped with a probability of
0.3, encouraging the model to rely more heavily on text and
multi-view character references.
Our model supports two generation modes: (1) First-clip
generation. The model generates 93 frames, corresponding
to 24 video latents (because the VAE encoder temporally
downsamples a video of N frames into (N −1)/4+1 video
latents). In this setting, no preceding clip is provided. (2)
Auto-regressive clip generation. When previous clips al-
ready exist, the model conditions on the last 33 frames (9
video latents) of the preceding clip and generates 96 new
frames (24 video latents).
Inference Acceleration. To accelerate inference, we adopt
the DMD2 distillation framework [91], employing teacher,
student, and fake-score models initialized from our trained
model. The teacher model remains fully frozen, while the
student and fake-score models are fine-tuned using LoRA
modules with rank 64. Following the DMD2 protocol, the
fake-score model is updated at every iteration, whereas the
student model is updated once every five iterations. This
setup—instantiating three 13B models while training two
sets of LoRA parameters—incurs a substantial GPU mem-
ory overhead.
As a result, DMD2 distillation is applied
Table 5. Hybrid training with game and real-world data helps the
model disentangle game-engine rendering from real-world visual
characteristics, yielding higher DINOv2 and CLIP character con-
sistency scores.
Training Data Type
DINOv2 Score
CLIP Score
Game Data
0.686
0.718
+Real-World Data
0.755
0.729
only to the 360P model and is trained for 4,000 steps on
8× NVIDIA B200 GPUs, using the ZeRO-3 optimization
strategy to manage memory efficiently.
7. More Experiments
Game-Real Hybrid Data Enhances Real-World Charac-
ter Fidelity. Training solely on GTA-V [17] videos intro-
duces a challenge: the model tends to inherit the game-
engine rendering style, causing synthesized characters to
appear stylized even when conditioned on real-world multi-
view characters at inference time.
To improve the real-
ism of generated real-world characters—while retaining the
diverse and dynamic motion patterns learned from game
data—we curate an additional real-world dataset and jointly
train AniX on both sources. Specifically, we record 400
video clips of real individuals performing the same set of
locomotion actions as in the game dataset. These videos
are processed using the same pipeline described in the main
paper and standardized to 360P resolution.
The model is then trained on a hybrid dataset combining
the aforementioned GTA-V and newly collected real-world
videos. To help the model differentiate between rendered
and real-world visual styles, we apply a simple data-tagging
strategy [101]: GTA-V samples are tagged with the key-
word “rendered” (e.g., “The rendered character is running
forward”), while real-world samples are tagged with “real”
(e.g., “The real character is running forward”). Aside from
this tagging mechanism, the overall training procedure re-
mains identical to that described in the main paper.
To evaluate the effectiveness of the mixed-data strategy,
we collect multi-view captures of two unseen real-world in-
dividuals. Following the evaluation protocol in the main
paper, we compute DINOv2 [61] and CLIP [64] scores to
evaluate the character consistency. As shown in Table 5,
training on hybrid data produces measurable improvements
in real-world character fidelity. The qualitative results in
Figure 7 further illustrate that the hybrid-trained model
achieves noticeably higher realism, capturing fine-grained
details—such as dynamic clothing wrinkles—that are ab-
10

<!-- page 11 -->
Figure 7.
Evaluation of the hybrid data training strategy.
(a)
Training solely on game data causes the model to inherit a game-
engine rendering style in the synthesized characters. (b) Incorpo-
rating real-world data improves photorealism, enabling the model
to capture high-frequency details—such as dynamic clothing wrin-
kles—that are absent from the GTA-V dataset.
sent in models trained solely on GTA-V data.
Inference Acceleration.
To qualitatively assess the ef-
fectiveness of our inference acceleration strategy, Figure 8
compares three variants: (1) the original model with a 30-
step denoising schedule (no acceleration), (2) the acceler-
ated model with a 4-step denoising schedule, and (3) the
original model also restricted to 4 denoising steps (no ac-
celeration but fewer steps).
Latencies for Higher-Resolution Inference. We further
report the inference cost and performance for generating
higher-resolution outputs using 8× NVIDIA H100 GPUs.
When producing a 93-frame video clip at 720P, the 720P
model outperforms the 360P model in both DINOv2 [61]
(0.698 →0.704) and CLIP-Aesthetic [69] (5.665 →5.887)
metrics, with an observed latency of 159 seconds.
8. More Visualizations
Action Control and Generalization. In the main paper,
we quantitatively report the success rate of controlling a
character to perform 142 novel actions. In Figure 9, we
provide qualitative results by visualizing 84 randomly se-
Figure 8. Qualitative comparison of three models: (a) the origi-
nal model with a 30-step denoising schedule (no acceleration), (b)
the accelerated model with a 4-step schedule, and (c) the original
model restricted to 4 steps (no acceleration but fewer steps). The
results show that our 4-step model matches the visual quality of
the original model while achieving a 7.5× inference speedup.
lected actions using the same character. Figure 10 further
illustrates the model’s generalization by showing 25 ran-
domly selected actions—with text annotations—performed
by a different character.
Scene Customization. Our model supports flexible scene
customization. Using state-of-the-art 3DGS scene genera-
tors, users can create diverse environments and control any
character to explore these worlds. In this work, most 3DGS
11

<!-- page 12 -->
scenes are sourced from World Labs Marble [43]. Figure 11
shows examples of characters navigating a variety of gener-
ated worlds.
Character Customization.
Our model demonstrates
strong generalization in controlling previously unseen
characters.
Leveraging mature 3D character-generation
tools—such as Hunyuan3D [76], Tripo [68], Meshy [59],
and Rodin [75]—or sourcing assets from online reposito-
ries like Sketchfab [16], diverse 3D characters can be easily
acquired and used directly for inference. Figures 12 and 13
illustrate examples of these characters performing locomo-
tion actions.
Long-Horizon Generation.
Our model supports auto-
regressive generation, enabling the creation of temporally
coherent video sequences that build upon previously gen-
erated clips.
This capability allows for extended, long-
horizon user–model interactions. Figures 14 and 15 present
two examples of long-horizon generation.
References
[1] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lian-
rui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai,
Pengfei Wan, et al.
Recammaster:
Camera-controlled
generative rendering from a single video. arXiv preprint
arXiv:2503.11647, 2025. 3
[2] Maciej Bala, Yin Cui, Yifan Ding, Yunhao Ge, Zekun
Hao, Jon Hasselgren, Jacob Huffman, Jingyi Jin, JP Lewis,
Zhaoshuo Li, et al. Edify 3d: Scalable high-quality 3d asset
generation. arXiv preprint arXiv:2411.07135, 2024. 3
[3] Raphael Bensadoun, Tom Monnier, Yanir Kleiman, Filip-
pos Kokkinos, Yawar Siddiqui, Mahendra Kariya, Omri
Harosh, Roman Shapovalov, Benjamin Graham, Emi-
lien Garreau, et al.
Meta 3d gen.
arXiv preprint
arXiv:2407.02599, 2024. 3
[4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel
Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi,
Zion English, Vikram Voleti, Adam Letts, et al.
Stable
video diffusion: Scaling latent video diffusion models to
large datasets. arXiv preprint arXiv:2311.15127, 2023. 2
[5] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack
Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai,
Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al.
Genie: Generative interactive environments. In Forty-first
International Conference on Machine Learning, 2024. 3
[6] Jinshu Chen, Xinghui Li, Xu Bai, Tianxiang Ma, Pengze
Zhang, Zhuowei Chen, Gen Li, Lijie Liu, Songtao Zhao,
Bingchuan Li, et al. Omniinsert: Mask-free video insertion
of any reference via diffusion transformer models. arXiv
preprint arXiv:2509.17627, 2025. 3
[7] Junyi Chen, Haoyi Zhu, Xianglong He, Yifan Wang, Jian-
jun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Zhou-
jie Fu, Jiangmiao Pang, et al. Deepverse: 4d autoregres-
sive video generation as a world model.
arXiv preprint
arXiv:2506.01103, 2025. 2, 3, 6, 7
[8] Luxi Chen, Zihan Zhou, Min Zhao, Yikai Wang, Ge Zhang,
Wenhao Huang, Hao Sun, Ji-Rong Wen, and Chongxuan
Li.
Flexworld: Progressively expanding 3d scenes for
flexiable-view synthesis. arXiv preprint arXiv:2503.13265,
2025. 3
[9] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace,
Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir
Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, and Sergey
Tulyakov. Multi-subject open-set personalization in video
generation. In Proceedings of the Computer Vision and Pat-
tern Recognition Conference, pages 6099–6110, 2025. 3
[10] Etched Decart, Quinn McIntyre, Spruce Campbell, Xinlei
Chen, and Robert Wachen. Oasis: A universe in a trans-
former. URL: https://oasis-model. github. io, 2024. 3
[11] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and
Jiajun Wu. Worldscore: A unified evaluation benchmark for
world generation. arXiv preprint arXiv:2504.00983, 2025.
2, 6
[12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim
Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik
Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling recti-
fied flow transformers for high-resolution image synthesis.
In Forty-first international conference on machine learning,
2024. 3, 10
[13] Zhengcong Fei, Debang Li, Di Qiu, Jiahua Wang, Yikun
Dou, Rui Wang, Jingtao Xu, Mingyuan Fan, Guibin Chen,
Yang Li, et al. Skyreels-a2: Compose anything in video
diffusion transformers. arXiv preprint arXiv:2504.02436,
2025. 3
[14] Ruili Feng, Han Zhang, Zhantao Yang, Jie Xiao, Zhilei
Shu, Zhiheng Liu, Andy Zheng, Yukun Huang, Yu Liu,
and Hongyang Zhang. The matrix: Infinite-horizon world
generation with real-time moving control. arXiv preprint
arXiv:2412.03568, 2024. 2, 3
[15] Wanquan Feng, Jiawei Liu, Pengqi Tu, Tianhao Qi,
Mingzhen Sun, Tianxiang Ma, Songtao Zhao, Siyu Zhou,
and Qian He.
I2vcontrol-camera: Precise video camera
control with adjustable motion strength.
arXiv preprint
arXiv:2411.06525, 2024. 3
[16] Epic Games. Sketchfab. https://sketchfab.com/
feed, 2025. 12
[17] Rockstar Games. Grand theft auto v. https://www.
rockstargames.com/gta-v, 2025. 3, 10
[18] Hyojun Go, Byeongjun Park, Jiho Jang, Jin-Young Kim,
Soonwoo Kwon, and Changick Kim. Splatflow: Multi-view
rectified flow model for 3d gaussian splatting synthesis. In
Proceedings of the Computer Vision and Pattern Recogni-
tion Conference, pages 21524–21536, 2025. 3
[19] Google. Gemini. https://gemini.google.com/
app, 2025. 7
[20] Google.
Genie 3:
A new frontier for world models.
https://deepmind.google/blog/genie-3-a-
new-frontier-for-world-models/, 2025. 3
[21] Google.
Veo.
https://deepmind.google/
models/veo/, 2025. 2, 3
[22] Zekai Gu, Rui Yan, Jiahao Lu, Peng Li, Zhiyang Dou,
Chenyang Si, Zhen Dong, Qifeng Liu, Cheng Lin, Ziwei
Liu, et al. Diffusion as shader: 3d-aware video diffusion
for versatile video generation control. In Proceedings of
12

<!-- page 13 -->
the Special Interest Group on Computer Graphics and In-
teractive Techniques Conference Conference Papers, pages
1–12, 2025. 3
[23] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi
Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reason-
ing capability in llms via reinforcement learning.
arXiv
preprint arXiv:2501.12948, 2025. 8
[24] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang,
Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin,
and Bo Dai. Animatediff: Animate your personalized text-
to-image diffusion models without specific tuning. arXiv
preprint arXiv:2307.04725, 2023. 3
[25] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel
Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy
Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weiss-
buch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and
Ofir Bibi. Ltx-video: Realtime video latent diffusion. arXiv
preprint arXiv:2501.00103, 2024. 2
[26] Junlin Hao, Peiheng Wang, Haoyang Wang, Xinggong
Zhang, and Zongming Guo. Gaussvideodreamer: 3d scene
generation with video diffusion and inconsistency-aware
gaussian splatting. arXiv preprint arXiv:2504.10001, 2025.
3
[27] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo
Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: En-
abling camera control for text-to-video generation. arXiv
preprint arXiv:2404.02101, 2024. 2, 3
[28] Hao He, Ceyuan Yang, Shanchuan Lin, Yinghao Xu, Meng
Wei, Liangke Gui, Qi Zhao, Gordon Wetzstein, Lu Jiang,
and Hongsheng Li. Cameractrl ii: Dynamic scene explo-
ration via camera-controlled video diffusion models. arXiv
preprint arXiv:2503.10592, 2025. 2, 3
[29] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang,
Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An,
Yangyang Ren, et al. Matrix-game 2.0: An open-source,
real-time, and streaming interactive world model.
arXiv
preprint arXiv:2508.13009, 2025. 2, 3, 6, 7
[30] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-
Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
et al. Lora: Low-rank adaptation of large language models.
ICLR, 1(2):3, 2022. 6, 10
[31] Teng Hu, Zhentao Yu, Zhengguang Zhou, Sen Liang, Yuan
Zhou, Qin Lin, and Qinglin Lu.
Hunyuancustom:
A
multimodal-driven architecture for customized video gen-
eration. arXiv preprint arXiv:2505.04512, 2025. 2, 3, 4, 6,
7, 8, 10
[32] Junchao Huang, Xinting Hu, Boyao Han, Shaoshuai Shi,
Zhuotao Tian, Tianyu He, and Li Jiang. Memory forcing:
Spatio-temporal memory for consistent scene generation on
minecraft. arXiv preprint arXiv:2510.03198, 2025. 3
[33] Tianyu Huang, Wangguandong Zheng, Tengfei Wang,
Yuhao Liu, Zhenwei Wang, Junta Wu, Jie Jiang, Hui Li,
Rynson WH Lau, Wangmeng Zuo, and Chunchao Guo.
Voyager:
Long-range and world-consistent video diffu-
sion for explorable 3d scene generation.
arXiv preprint
arXiv:2506.04225, 2025. 3
[34] Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si,
Dahua Lin, Yu Qiao, Chen Change Loy, and Ziwei Liu.
Videobooth: Diffusion-based video generation with image
prompts.
In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 6689–
6700, 2024. 3
[35] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang,
Yulin Pan, and Yu Liu.
Vace:
All-in-one video cre-
ation and editing.
In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision, pages 17191–
17202, 2025. 6, 7
[36] Xuan Ju, Weicai Ye, Quande Liu, Qiulin Wang, Xintao
Wang, Pengfei Wan, Di Zhang, Kun Gai, and Qiang Xu.
Fulldit: Multi-task video generative foundation model with
full attention. arXiv preprint arXiv:2503.19907, 2025. 3
[37] Anssi Kanervisto, Dave Bignell, Linda Yilin Wen, Mar-
tin Grayson, Raluca Georgescu, Sergio Valcarcel Macua,
Shan Zheng Tan, Tabish Rashid, Tim Pearce, Yuhan Cao,
et al. World and human action models towards gameplay
ideation. Nature, 638(8051):656–663, 2025. 3
[38] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis. 3d gaussian splatting for real-time
radiance field rendering. ACM Transactions on Graphics,
42(4), 2023. 2
[39] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo
Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jian-
wei Zhang, et al.
Hunyuanvideo: A systematic frame-
work for large video generative models.
arXiv preprint
arXiv:2412.03603, 2024. 2, 3, 6, 7, 10
[40] Kuaishou.
Kling ai.
https://klingai.com/
global/, 2025. 2
[41] Dynamics Lab. Magica 2: The next leap in generative world
engines. https://blog.dynamicslab.ai/, 2025.
3
[42] World Labs. Rtfm: A real-time frame model. https:
//rtfm.worldlabs.ai/, 2025. 3
[43] World Labs. Marble. https://marble.worldlabs.
ai/, 2025. 2, 3, 5, 12
[44] Zeqiang Lai, Yunfei Zhao, Haolin Liu, Zibo Zhao, Qingx-
iang Lin, Huiwen Shi, Xianghui Yang, Mingxin Yang,
Shuhui Yang, Yifei Feng, et al.
Hunyuan3d 2.5: To-
wards high-fidelity 3d assets generation with ultimate de-
tails. arXiv preprint arXiv:2506.16504, 2025. 5
[45] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan
Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin
Lu. Hunyuan-gamecraft: High-dynamic interactive game
video generation with hybrid history condition.
arXiv
preprint arXiv:2506.17201, 2025. 2, 3, 6, 7
[46] Runjia Li,
Philip Torr,
Andrea Vedaldi,
and Tomas
Jakab.
Vmem: Consistent interactive video scene gen-
eration with surfel-indexed view memory. arXiv preprint
arXiv:2506.18903, 2025. 3
[47] Xinyang Li, Zhangyu Lai, Linning Xu, Yansong Qu, Liu-
juan Cao, Shengchuan Zhang, Bo Dai, and Rongrong Ji. Di-
rector3d: Real-world camera trajectory and 3d scene gener-
ation from text. Advances in neural information processing
systems, 37:75125–75151, 2024. 3
13

<!-- page 14 -->
[48] Xiaowen Li, Haolan Xue, Peiran Ren, and Liefeng Bo. Dif-
fueraser: A diffusion model for video inpainting.
arXiv
preprint arXiv:2501.10018, 2025. 3
[49] Zhaoyang Li, Dongjun Qian, Kai Su, Qishuai Diao, Xi-
angyang Xia, Chang Liu, Wenfei Yang, Tianzhu Zhang,
and Zehuan Yuan.
Bindweave: Subject-consistent video
generation via cross-modal integration.
arXiv preprint
arXiv:2510.00438, 2025. 3
[50] Hanwen Liang, Junli Cao, Vidit Goel, Guocheng Qian,
Sergei Korolev, Demetri Terzopoulos, Konstantinos N Pla-
taniotis, Sergey Tulyakov, and Jian Ren. Wonderland: Nav-
igating 3d scenes from a single image. In Proceedings of
the Computer Vision and Pattern Recognition Conference,
pages 798–810, 2025. 3
[51] Sen Liang, Zhentao Yu, Zhengguang Zhou, Teng Hu,
Hongmei Wang, Yi Chen, Qin Lin, Yuan Zhou, Xin Li,
Qinglin Lu, et al. Omniv2v: Versatile video generation and
editing via dynamic content manipulation. arXiv preprint
arXiv:2506.01801, 2025. 3
[52] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maxim-
ilian Nickel, and Matt Le. Flow matching for generative
modeling. arXiv preprint arXiv:2210.02747, 2022. 4
[53] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae
Lee. Visual instruction tuning. Advances in neural infor-
mation processing systems, 36:34892–34916, 2023. 4, 10
[54] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Ji-
awei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu.
Phantom: Subject-consistent video generation via cross-
modal alignment. arXiv preprint arXiv:2502.11079, 2025.
3
[55] Yifan Liu, Zhiyuan Min, Zhenwei Wang, Junta Wu, Tengfei
Wang, Yixuan Yuan, Yawei Luo, and Chunchao Guo.
Worldmirror: Universal 3d world reconstruction with any-
prior prompting. arXiv preprint arXiv:2510.10726, 2025.
2, 3
[56] Ilya Loshchilov and Frank Hutter. Decoupled weight decay
regularization. arXiv preprint arXiv:1711.05101, 2017. 10
[57] Yawen Luo, Jianhong Bai, Xiaoyu Shi, Menghan Xia,
Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and
Tianfan Xue. Camclonemaster: Enabling reference-based
camera control for video generation.
arXiv preprint
arXiv:2506.03140, 2025. 3
[58] Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wen-
shuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu
Qiao, and Kaipeng Zhang.
Yume: An interactive world
generation model. arXiv preprint arXiv:2507.17744, 2025.
3
[59] Meshy. Meshy ai. https://www.meshy.ai/, 2025.
12
[60] OpenAI.
Sora.
https://sora.chatgpt.com/
explore, 2025. 2
[61] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy
Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,
Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al.
Dinov2: Learning robust visual features without supervi-
sion. arXiv preprint arXiv:2304.07193, 2023. 6, 10, 11
[62] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Car-
roll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini
Agarwal, Katarina Slama, Alex Ray, et al. Training lan-
guage models to follow instructions with human feedback.
Advances in neural information processing systems, 35:
27730–27744, 2022. 8
[63] William Peebles and Saining Xie. Scalable diffusion mod-
els with transformers.
In Proceedings of the IEEE/CVF
international conference on computer vision, pages 4195–
4205, 2023. 3, 10
[64] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learn-
ing transferable visual models from natural language super-
vision. In International conference on machine learning,
pages 8748–8763. PmLR, 2021. 6, 10
[65] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D Manning, Stefano Ermon, and Chelsea Finn. Direct
preference optimization: Your language model is secretly a
reward model. Advances in neural information processing
systems, 36:53728–53741, 2023. 8
[66] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang
Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng
Yan, et al. Grounded sam: Assembling open-world models
for diverse visual tasks. arXiv preprint arXiv:2401.14159,
2024. 3
[67] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling,
Yifan Lu, Merlin Nimier-David, Thomas M¨uller, Alexan-
der Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed
world-consistent video generation with precise camera con-
trol. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pages 6121–6132, 2025. 3
[68] VAST AI Research. Tripo. https://www.tripo3d.
ai/, 2025. 12
[69] Christoph Schuhmann. Clip+mlp aesthetic score predictor.
https://github.com/christophschuhmann/
improved-aesthetic-predictor, 2025. 11
[70] Jaidev Shriram, Alex Trevithick, Lingjie Liu, and Ravi Ra-
mamoorthi. Realmdreamer: Text-driven 3d scene gener-
ation with inpainting and depth diffusion. arXiv preprint
arXiv:2404.07199, 2024. 3
[71] Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh
Tenenbaum, and Fredo Durand. Light field networks: Neu-
ral scene representations with single-evaluation rendering.
Advances in Neural Information Processing Systems, 34:
19313–19325, 2021. 2, 3
[72] Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen,
Yueqi Duan, Jun Zhang, and Yikai Wang.
Dimensionx:
Create any 3d and 4d scenes from a single image with con-
trollable video diffusion. arXiv preprint arXiv:2411.04928,
2024. 3
[73] Wenqiang Sun, Fangyun Wei, Jinjing Zhao, Xi Chen, Zi-
long Chen, Hongyang Zhang, Jun Zhang, and Yan Lu.
From virtual games to real-world play.
arXiv preprint
arXiv:2506.18901, 2025. 2, 3
[74] Stanislaw Szymanowicz, Jason Y Zhang, Pratul Srinivasan,
Ruiqi Gao, Arthur Brussee, Aleksander Holynski, Ricardo
Martin-Brualla, Jonathan T Barron, and Philipp Henzler.
Bolt3d: Generating 3d scenes in seconds. In Proceedings of
14

<!-- page 15 -->
the IEEE/CVF International Conference on Computer Vi-
sion, pages 24846–24857, 2025. 3
[75] Deemos Technologies.
Rodin.
https://hyper3d.
ai/, 2025. 12
[76] Tencent. Tencent hunyuan3d. https://3d.hunyuan.
tencent.com/, 2025. 12
[77] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi
Fruchter.
Diffusion models are real-time game engines.
arXiv preprint arXiv:2408.14837, 2024. 3
[78] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao,
Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianx-
iao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang,
Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang
Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi
Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng,
Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang,
Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang,
Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen,
Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu,
Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang,
Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu,
Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong
Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu
Liu. Wan: Open and advanced large-scale video genera-
tive models. arXiv preprint arXiv:2503.20314, 2025. 2, 3,
6, 7
[79] Hanyang Wang, Fangfu Liu, Jiawei Chi, and Yueqi Duan.
Videoscene: Distilling video diffusion model to generate
3d scenes in one step. In 2025 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages
16475–16485. IEEE, 2025. 3
[80] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li,
Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan.
Motionctrl: A unified and flexible motion controller for
video generation. In ACM SIGGRAPH 2024 Conference
Papers, pages 1–11, 2024. 3
[81] Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei
Liu, Dahua Lin, and Gordon Wetzstein.
Video world
models with long-term spatial memory.
arXiv preprint
arXiv:2506.05284, 2025. 3
[82] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai
Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Long-
term consistent world simulation with memory.
arXiv
preprint arXiv:2504.12369, 2025. 3
[83] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen
Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv
preprint arXiv:2505.09388, 2025. 8
[84] Mingyu Yang, Junyou Li, Zhongbin Fang, Sheng Chen,
Yangbin Yu, Qiang Fu, Wei Yang, and Deheng Ye. Playable
game generation. arXiv preprint arXiv:2412.00887, 2024.
3
[85] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma,
Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao.
Direct-a-video: Customized video generation with user-
directed camera movement and object motion.
In ACM
SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 3
[86] Xiaoda Yang, Jiayang Xu, Kaixuan Luan, Xinyu Zhan,
Hongshun Qiu, Shijun Shi, Hao Li, Shuai Yang, Li
Zhang, Checheng Yu, et al.
Omnicam: Unified multi-
modal video generation via camera control. arXiv preprint
arXiv:2504.02312, 2025. 3
[87] Yuanbo Yang, Jiahao Shao, Xinyang Li, Yujun Shen, An-
dreas Geiger, and Yiyi Liao. Prometheus: 3d-aware latent
diffusion models for feed-forward text-to-3d scene gener-
ation. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pages 2857–2869, 2025. 3
[88] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding,
Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong,
Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-
video diffusion models with an expert transformer. arXiv
preprint arXiv:2408.06072, 2024. 2, 3, 6, 7
[89] Zhongqi Yang, Wenhang Ge, Yuqi Li, Jiaqi Chen, Haoyuan
Li, Mengyin An, Fei Kang, Hua Xue, Baixin Xu, Yuyang
Yin, et al. Matrix-3d: Omnidirectional explorable 3d world
generation. arXiv preprint arXiv:2508.08086, 2025. 2, 3
[90] Deheng Ye, Fangyun Zhou, Jiacheng Lv, Jianqi Ma, Jun
Zhang, Junyan Lv, Junyou Li, Minwen Deng, Mingyu
Yang, Qiang Fu, et al. Yan: Foundational interactive video
generation. arXiv preprint arXiv:2508.08601, 2025. 2, 3
[91] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard
Zhang, Eli Shechtman, Fredo Durand, and William T Free-
man. Improved distribution matching distillation for fast
image synthesis. In NeurIPS, 2024. 6, 8, 10
[92] Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent,
Michael Rubinstein, William T Freeman, Forrester Cole,
Deqing Sun, Noah Snavely, Jiajun Wu, et al. Wonderjour-
ney: Going from anywhere to everywhere. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 6658–6667, 2024. 3
[93] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xin-
tao Wang, Pengfei Wan, Di Zhang, and Xihui Liu.
Context as memory:
Scene-consistent interactive long
video generation with memory retrieval.
arXiv preprint
arXiv:2506.03141, 2025. 3
[94] Jiwen Yu, Yiran Qin, Xintao Wang, Pengfei Wan, Di
Zhang, and Xihui Liu.
Gamefactory:
Creating new
games with generative interactive videos. arXiv preprint
arXiv:2501.08325, 2025. 3
[95] Mark YU, Wenbo Hu, Jinbo Xing, and Ying Shan. Tra-
jectorycrafter: Redirecting camera trajectory for monoc-
ular videos via diffusion models.
arXiv preprint
arXiv:2503.05638, 2025. 3
[96] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu
Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying
Shan, and Yonghong Tian. Viewcrafter: Taming video dif-
fusion models for high-fidelity novel view synthesis. arXiv
preprint arXiv:2409.02048, 2024. 3
[97] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu,
Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi
Yu.
Clay: A controllable large-scale generative model
for creating high-quality 3d assets. ACM Transactions on
Graphics (TOG), 43(4):1–20, 2024. 3
[98] Shengjun Zhang, Jinzhao Li, Xin Fei, Hao Liu, and Yueqi
Duan. Scene splatter: Momentum 3d scene generation from
15

<!-- page 16 -->
single image with video diffusion model. In Proceedings of
the Computer Vision and Pattern Recognition Conference,
pages 6089–6098, 2025. 3
[99] Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang,
Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric
Li, Yang Liu, et al. Matrix-game: Interactive world foun-
dation model. arXiv preprint arXiv:2506.18701, 2025. 3
[100] Zhaoyang Zhang, Yannick Hold-Geoffroy, Miloˇs Haˇsan,
Ziwen Chen, Fujun Luan, Julie Dorsey, and Yiwei Hu. Gen-
erating 360◦video is what you need for a 3d scene, 2025.
3
[101] Qi Zhao, Xingyu Ni, Ziyu Wang, Feng Cheng, Ziyan
Yang, Lu Jiang, and Bohan Wang.
Synthetic video en-
hances physical fidelity in video synthesis. arXiv preprint
arXiv:2503.20822, 2025. 10
[102] Jensen Zhou, Hang Gao, Vikram Voleti, Aaryaman Va-
sishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian
Rupprecht, and Varun Jampani. Stable virtual camera: Gen-
erative view synthesis with diffusion models. arXiv preprint
arXiv:2503.14489, 2025. 3
16

<!-- page 17 -->
Figure 9. Visualization of a character performing 84 randomly selected novel actions.
17

<!-- page 18 -->
Figure 10. Visualization of a character performing 25 randomly selected novel actions with text annotations.
Figure 11. Visualization of a character exploring various 3DGS worlds.
18

<!-- page 19 -->
Figure 12. Visualization of diverse characters performing locomotion actions (Part 1).
19

<!-- page 20 -->
Figure 13. Visualization of diverse characters performing locomotion actions (Part 2).
20

<!-- page 21 -->
Figure 14. Visualization of long-horizon generation (Example 1).
21

<!-- page 22 -->
Figure 15. Visualization of long-horizon generation (Example 2).
22
