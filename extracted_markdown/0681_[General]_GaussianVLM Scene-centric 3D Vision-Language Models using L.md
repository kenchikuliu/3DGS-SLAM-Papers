<!-- page 1 -->
arXiv:2507.00886v1  [cs.CV]  1 Jul 2025
GaussianVLM: Scene-centric 3D Vision-Language Models using
Language-aligned Gaussian Splats for Embodied Reasoning and Beyond
Anna-Maria Halacheva1, Jan-Nico Zaech1, Xi Wang1,2,3, Danda Pani Paudel1, Luc Van Gool1
1INSAIT, Sofia University “St. Kliment Ohridski”, 2ETH Zurich, 3TU Munich
Fig. 1: The proposed GaussianVLM
performs comprehensive scene understanding in natural language for 3D scenes
represented as Gaussian Splats. It adopts a fully scene-centric approach, building a global, language-augmented scene
representation. This enables effective handling of both scene- and object-level tasks – requiring multi-object reasoning, spatial
understanding, global context, and fine-grained analysis – suitable for embodied reasoning and beyond.
Abstract— As multimodal language models advance, their
application to 3D scene understanding is a fast-growing frontier,
driving the development of 3D Vision-Language Models (VLMs).
Current methods show strong dependence on object detectors,
introducing processing bottlenecks and limitations in taxonomic
flexibility. To address these limitations, we propose a scene-
centric 3D VLM for 3D Gaussian splat scenes that employs
language- and task-aware scene representations. Our approach
directly embeds rich linguistic features into the 3D scene
representation by associating language with each Gaussian
primitive, achieving early modality alignment. To process the
resulting dense representations, we introduce a dual sparsifier
that distills them into compact, task-relevant tokens via task-
guided and location-guided pathways, producing sparse, task-
aware global and local scene tokens. Notably, we present the
first Gaussian splatting-based VLM, leveraging photorealistic
3D representations derived from standard RGB images, demon-
strating strong generalization: it improves performance of prior
3D VLM (LL3DA [9]) five folds, in out-of-the-domain settings.
We provide open access to all assets.
I. INTRODUCTION
To act intelligently in the physical world, embodied
agents benefit from a rich, structured understanding of 3D
scenes – capturing not only objects but also spatial context,
relationships, and semantics [40], [41], [47], [36]. Such scene
understanding enables agents to move toward advanced tasks
like embodied reasoning and planning, spanning multiple
modalities [29], [30], [31], [20], [46] . While recent 3D VLMs
have advanced towards addressing 3D vision-language tasks
for embodied agents, they are predominantly object-centric,
introducing a critical dependency on object detectors [23],
[9], [22], [55], [20]. This creates a mismatch with the core
objective of generic scene understanding, forcing models into
predefined granularities, limited taxonomies, and neglecting
global context and spatial relationships [33], [17]. In this
work, we propose to shift from object-centric to scene-centric
representations by embedding language features directly into
the spatial structure of the environment. Each element of the
3D scene, represented either as a point or a Gaussian splat,
is enriched with continuous language features, e.g. CLIP
[34], SigLIP [51]. This allows us to construct a language-
aligned scene representation without relying on predefined
object categories. Our scene-centric 3D VLM thus can answer
complex questions related to both objects and scenes, as
shown in Fig. 1.
However, directly embedding language features at the fine-
granularity of the scene elements results in extremely dense

<!-- page 2 -->
representations in the tens of thousands tokens per scene.
We argue that using the existing solutions, meaningfully
understanding such representations via LLMs is a very
challenging task – due to the high density of high-dimensional
language features. To address this, we introduce a dual
sparsifier module that efficiently utlizes dense language
representations while preserving semantic fidelity. The dual
nature of our sparsifier has two pathways: task-guided and
location-guided. The task-guided sparsifier selects scene
tokens based on global task relevance, and the location-
guided sparsifier retrieves fine-grained features conditioned
on spatial cues in the task, as shown in Fig. 2. The location-
based sparsifier selects the language features of the Gaussians
within the Region-of-Interest (ROI) around the location from
the task, reducing them to a few ROI tokens. The task-
guided sparsifier takes as input the dense scene tokens and
the task tokens, using the latter in cross-attention to guide
the sparsification process. As a result, the dense features
are reduced to 128 task-selected scene tokens. The obtained
sparse scene representation, consisting of the ROI tokens and
task-selected tokens, is passed together with the task tokens
to an LLM for response generation.
Finally, we develop the first 3D VLM operating on
Gaussian Splatting (GS) that naturally fuses geometry and
appearance information [27]. Note that unlike point clouds,
Gaussian splats capture detailed 3D textures – in addition
to the geometry – which is necessary for generic 3D scene
understanding of our interest. For the more, with the recent
developments the high-quality 3DGS can be realistically
acquired using only RGB cameras. We demonstrate that our
model, GaussianVLM, maintains strong task performance in
real-world settings. We evaluate GaussianVLM and a state-
of-the-art (SOTA) point-cloud based VLM [9] on an in-house
question-answering task for counting objects in ScanNet++
scenes [49]. On the utilized out-of-domain ScanNet++ scene
representations, derived from RGB images, the GS-based
GaussianVLM outperforms the SOTA point cloud-based 3D
VLM five folds in terms of accuracy (Tab. III).
We evaluate GaussianVLM
on a comprehensive suite
of 3D vision-language tasks spanning both scene-centric
(Tab. I) and object-centric settings (Tab. II). Across the
board, GaussianVLM achieves state-of-the-art performance,
outperforming the SOTA baselines [9], [23] on every bench-
mark. Showing the advantages of scene-centrism, Gaussian-
VLM significantly outperforms previous methods on embod-
ied scene-centric tasks, e.g., embodied reasoning (SQA3D
[30] 49.4% vs. 47.0% top-1 exact match) and substantially
improving dialogue and planning metrics (e.g., +155.3 CIDEr
in Embodied Planning [20]). Importantly, the detector-free
GaussianVLM also excels on object-centric benchmarks, e.g.,
achieving improved object captioning on Nr3D [1] (+15.0
METEOR, +9.3 ROUGE).
Overall, this work makes the following contributions:
• We introduce a fully scene-centric 3D VLM that achieves
SOTA results, without requiring any dependencies on
object detectors, on benchmark datasets for reasoning
tasks required for embodied vision and beyond.
• We propose a dual sparsification mechanism to efficiently
distill dense language-augmented scenes into compact,
task-relevant representations, suitable for LLMs.
• We present the first language-grounded 3D VLM directly
operating on 3D Gaussian Splat representations.
II. RELATED WORK
A. Scene-Level Reasoning of Embodied Agents
Early benchmarks in embodied question answering (EQA)
[16], [46] pioneered tasks requiring agents to reason from
egocentric observations, primarily focusing on situated,
navigation-oriented challenges. Subsequent research expanded
this scope to include multi-hop and commonsense reasoning
[30], as well as embodied planning and dialogue tasks [20].
Early solutions adapted architectures like MCAN [50] and
ClipBERT [24], with ScanQA [3] introducing 3D scene-
grounded QA via explicit reconstructions. This progression
has culminated in generalist 3D VLMs [23], [56], [9], [55]
unifying 3D scene understanding, reasoning, and planning.
B. 3D Scene Tokenization
For effective VLMs, 3D scene tokenization transforms
complex geometry into language-processable, semantically
rich representations. Two prevalent strategies exist:
Object-Level Tokenization. A common paradigm [23],
[22], [26], [11], [53], [55], [45] involves detecting individual
objects, extracting their point clouds, and independently
encoding them with a 3D encoder to generate object-level
tokens. This method, while semantically intuitive, is limited by
object detector performance and neglects vital scene context
(e.g., room layout, walls).
Region-Based Tokenization. Another approach [54], [56]
encodes the entire scene into per-point features, then groups
these points into a fixed number of regions (e.g., via kNN
[54] or graph-based segmentation [56]). Averaging features
within these regions creates region-level tokens, capturing
broader context at reduced granularity. However, this risks
over-smoothing by collapsing diverse information into single
tokens. Additionally, predefining the number of regions is
challenging: too many can introduce irrelevant data and
increase cost, while too few may lose fine-grained details.
In contrast, we introduce language-guided scene tokeniza-
tion, an approach that dynamically re-tokenizes the scene
based on linguistic input and per-point/per-Gaussian language
features. By leveraging language to direct the tokenization,
our method ensures that the resulting tokens focus on the
scene regions most pertinent to the current task.
C. Vision-Language-Aligned 3D Scene Understanding
Integrating language into 3D scene understanding intro-
duces challenges, particularly in (1) achieving effective cross-
modal alignment [23], [9], and (2) ensuring semantically rich
vision features [56].
Text-Vision Alignment. Prior work commonly aligns 3D
visual features with language by projecting each modality
independently into a shared embedding space [23], [22], [18],
[45]. However, this often results in weak alignment due to

<!-- page 3 -->
the largely separate processing of the two modalities. In
line with 2D vision-language models [25], other approaches
employ learnable query tokens that attend to both visual and
textual features, separately, [56], [9], aiming for information
fusion. Nevertheless, these query-based methods frequently
refine visual features before language interaction, limiting the
language’s impact on the initial visual encoding. Critically, a
shared limitation across these strategies is that the 3D encoder
features are generated without incorporating any language or
task-relevant semantic cues, ultimately leading to a shallow
alignment [43]. Our approach, in contrast, ensures strong text-
vision alignment by embedding language features directly
into the fine-grained spatial structure of the 3D scene.
Vision Feature Quality. Recent efforts in 3D scene
representation and sparsification have aimed to improve VLM
performance by increasing the vision feature expressiveness.
Many approaches leverage multi-modal visual data (2D
images, point clouds, meshes) [22], [56], [20] for rich
scene information, yet they are computationally intensive
and architecturally complex, often also with inefficient, task-
agnostic sparsification. Region highlighting techniques [54],
[19], [9] attempt to emphasize key regions alongside a global
scene representation, but the persistent use of dense global
representations limits scalability and and attentional focus. We
avoid these limitations by (a) using easy-to-obtain expressive
language-aligned features [43] as our scene representation,
and (b) generating all scene tokens conditioned on the task.
III. METHOD
We introduce GaussianVLM, a 3D VLM for indoor scene
understanding. Given a 3D scene represented as Gaussian
splats and a natural language prompt, GaussianVLM fuses
language and 3D vision at multiple stages to generate a textual
response. Notably, GaussianVLM
is the first to leverage
Gaussian splats as the 3D scene representation, and function
exclusively in the language space, achieving this object
detector-free. GaussianVLM relies on three key innovations:
(1) a language-aware Gaussian splatting backbone [27] that
predicts language features for each Gaussian, enabling direct
language-based alignment between the scene and the prompt;
(2) a task-guided sparsifier module generating a sparse scene
representation by performing task-aware re-tokenization of
the dense 3D backbone output; and (3) a location-guided
sparsifier module for detector-free extraction of Region-of-
Interest (ROI) information. We detail the GaussianVLM and
the sparsifier components in the subsequent sections.
A. GaussianVLM
Unlike previous approaches that rely on purely visual
representations, our method integrates a 3D transformer
that produces inherently language-grounded vision features.
Specifically, we adopt SceneSplat [27] as our 3D vision
module. SceneSplat processes scenes represented via Gaussian
splats and predicts a SigLIP2 [43] language feature for each
Gaussian end-to-end. To sparsify the resulting dense language
features with a task-awareness, we introduce a dual sparsifier
module. The sparsifier takes as input the dense language
features and outputs sparse task-aware tokens. The sparse
scene tokens are projected from the SigLIP2 space into
the LLM space via a single linear projection. The resulting
vision tokens are then concatenated with the user task tokens,
tokenized via the LLM’s tokenizer, and input into a frozen
LLM augmented with Low-Rank Adaptation (LoRA) [21].
The LLM autoregressively generates responses to the user
query, conditioned jointly on both visual and textual context.
GaussianVLM (OPT-1.3B [52] as LLM) has a size of 1.8B
parameters out of which 19M are learnable.
Training Objective. Similarly to many VLM training
protocols [28], [23], [53], we follow a two-stage training with
alignment and fine-tuning phase. During the alignment phase
we freeze the 3D backbone and LLM tokenizer, training the
sparsifier modules and the transformer for textual alignment
of the vision tokens. The LLM is adapted using LoRA. Both
stages share a unified training objective. Following [23], [6],
[35], we use a prefix language modeling, where the model is
conditioned on an input prefix and trained to autoregressively
generate the target continuation:
L(θ, B) = −
X
{sprefix,sgt}∈B
|sgt|
X
t=1
log pθ

s(t)
gt | s(<t)
gt
, sprefix

,
(1)
with θ as the model parameters, B - a batch of samples
of prefix input sprefix (task prompt and vision tokens), and
ground truth response sgt. s(t)
gt denotes the t-th token in the
ground truth response sequence.
To enhance spatial grounding, we initially pre-train the
task-guided sparsifier on understanding 3D location features.
We leverage an object captioning task in which the model is
provided the 3D coordinates of a labeled object instance and
trained to generate a visual token embedding similar to the
embeddings of the corresponding label token. The location
is encoded through learnable Fourier embeddings (Eq. 3) to
a single feature and passed to the sparsifier. The label text
is embedded using the SigLIP-2 tokenizer. This pre-training
stage uses a one-sided contrastive objective [34], encouraging
the output embedding of the task-guided sparsifier si to match
its corresponding label embedding li, while being distant from
all other labels lj (j ̸= i):
Lcontrast = −log
exp(s⊤
i li/τ)
P j = 1N exp(s⊤
i lj/τ),
(2)
where si is the output scene token of the sparsifier for the i-th
instance, li is the SigLIP-2 embedding of the corresponding
label, N is the number of labels in the batch, and τ is a
temperature hyperparameter which we set to 0.07.
B. Dual Sparsifier
Task-Guided Sparsification. SceneSplat processes 3D
Gaussians into a dense sequence of tokens (one per Gaussian).
Following established practices [9], [54], [19], [10], sampling
40k Gaussians yields a corresponding 40k output tokens,
originating from different SceneSplat decoder layers (specif-
ically, 589, 2.4k, and 40k). To address the computational

<!-- page 4 -->
Fig. 2: The GaussianVLM architecture processes a user task prompt (query and optional location) and a 3D scene (Gaussian
Splat representation). A 3D vision module (SceneSplat Transformer) predicts per-Gaussian language features. These dense
features are then sparsified by a dual sparsifier module. The decoder’s hidden states also inform the task-guided sparsifier. The
dual sparsifier comprises: 1) a location-guided pathway that selects language features from Gaussians within a ROI around the
task location, producing ROI tokens; and 2) a task-guided pathway that attends to dense scene tokens and SceneSplat decoder
hidden states using task tokens (via cross-attention) to produce 128 task-selected scene tokens. The resulting sparse scene
representation (ROI tokens + task-selected tokens), along with the task tokens, is input to an LLM for response generation.
demands of this dense representation and prioritize task-
relevant information, we introduce a novel task-guided
sparsification module. This module re-tokenizes the dense
scene representation into a more compact, sparse one by
selectively attending to the most important visual features
based on the textual query.
Our sparsifier employs the language task to generate
queries that guide the filtering of visual input via depth-
wise cross-attention [5]. This task-guided sparsification is
applied iteratively to the output of each SceneSplat decoder
layer, enabling a dynamic and context-aware reduction of
visual information.
To mitigate the computational overhead of cross-attention
on a large number of tokens, we first apply a simple uniform
downsampling strategy to reduce the representation to 512
tokens per decoder layer. Our ablation study (Sec. IV-G)
demonstrates the sufficiency of this efficient approach, negat-
ing the need for more complex initial downsampling methods
like kNN used in other models [19], [54]. Subsequently, we
further sparsify to 128 tokens by performing cross-attention
between the tokens of the user’s prompt (tokenized using
SigLIP2, consistent with SceneSplat’s language features) and
these weakly-sparsified 3D features. For any spatial locations
locxyz ∈R3 mentioned in the prompt, we encode them using
learnable Fourier embeddings [9]:
pos(locxyz) = [sin(2π locxyz · B) ; cos(2π locxyz · B)] (3)
where B ∈R3×(d/2) is a learnable matrix. If the prompt
includes a bounding box, we extract locxyz as its center.
To generate queries for cross-attention, we apply attention
pooling to the embeddings of the task tokens, resulting in
a fixed set of 128 query vectors. Corresponding to each
SceneSplat decoder block is a cross-attention sparsifier block.
The initial layer of these blocks performs cross-attention
between the SceneSplat visual tokens and the task tokens.
The resulting intermediate visual features are then processed
through subsequent layers, further sparsifying the scene fea-
tures and refining their semantic alignment with the language
in a depth-wise manner. This process yields language-aware
vision tokens integrating global scene understanding from the
earlier decoder layers with instance-level awareness derived
from the per-Gaussian language features.
In the final sparsifier layer, we additionally inject positional
information by encoding the center of the downsampled
512 Gaussian splats using Eq. 3. This step is crucial for
instilling position awareness into the otherwise location-
agnostic Gaussian language features.
Location-Guided Sparsification. For tasks that provide
a location, such as object captioning, we introduce an ROI
magnifier. This module extracts features from a spherical
region around the object location indicated by the prompt.
For click locations, we use the click’s xyz point; for bounding
boxes, we use the center. Given a location point, we select
neighboring points within a 15cm radius, chosen to focus
on small objects. If no points are captured in the ROI, we
iteratively increase the radius by 15cm until the ROI is not
empty. We then apply attention pooling to the language
features of these selected points to generate 4 ROI tokens
summarizing the region.
IV. EXPERIMENTS
A. Dataset
We evaluate our model under the LL3DA, a SOTA 3D
VLM, training protocol [9]. We also evaluate on embodied
reasoning (SQA3D [30]), a popular 3D VLM benchmark
[23], [56], where we follow the LEO [23] training protocol.

<!-- page 5 -->
LL3DA Training Protocol. For the LL3DA protocol, we
follow their one-stage joint training procedure. Training is
performed on ScanRefer [8] (object captioning), ScanQA
[3] (general question-answering (QA)), Nr3D [1] (object
captioning), and the ScanNet subset of 3D-LLM [20] (diverse
scene-centric tasks), focusing on multitask learning.
LEO Training Protocol. In the LEO setting, we adopt a
two-phase training strategy with alignment and instruction
tuning. To maintain compatibility with our scene-centric
design and the LL3DA setup, we restrict training to the
ScanNet subset of the LEO dataset. We align the visual and
language modalities using the ReferIt3D dataset [1] providing
detailed object captions. This phase helps the model ground
linguistic features directly into the 3D scene representation.
During the second stage, the model is further trained to follow
natural language instructions across multiple tasks using the
SQA3D (situated QA) , ScanRefer, and ScanQA datasets.
B. Tasks
We evaluate our model on a diverse set of 3D vision-
language tasks drawn from the LL3DA and LEO benchmarks.
These tasks fall into two broad categories: object-centric
and scene-centric, reflecting differing demands on spatial
grounding and semantic abstraction.
Object-Centric Tasks require reasoning about discrete ob-
jects in the scene, often relying on explicit object annotations
or localized queries. Those tasks include:
Object Captioning. We use ScanRefer [8], and Nr3D [1] for
evaluating object captioning. Each training instance provides
a natural language expression referring to a specific object
in the scene. Using the annotated instance IDs, we extract
the corresponding 3D bounding box and use its center as
the target location. The model is prompted to generate a
caption for the object at this location, conditioned on the full
scene representation. To encourage linguistic diversity, we
use GPT-4o to generate 40 paraphrased variants per prompt
with varied syntax and vocabulary.
Object-Centric Question Answering. We use ScanQA [3],
which includes questions about object attributes, counts, and
presence (e.g., “What is the color of the chair?”). As these
questions typically target individual entities rather than spatial
relationships or global context, the dataset aligns with object-
centric evaluation.
Scene-Centric Tasks, in contrast, require holistic reasoning
about the environment, its layout, and the agent’s situated
context—without reducing the scene to individual object
tokens. The Situated Question Answering (SQA3D) [30]
requires the model answering answer spatial or functional
questions grounded in the scene (e.g., “What is on my left?”),
given a situational context (e.g., “I am washing my hands”).
These questions require understanding the scene’s layout,
affordances, and agent-relative positioning. For the Embodied
Planning [20] task, the model generates high-level plans
to complete tasks, leveraging the full scene structure to
identify relevant objects and transitions. In Scene Captioning
[20], the model produces free-form descriptions summarizing
the entire scene, requiring it to integrate geometry, object
presence, and semantics into coherent language. Embodied
Dialogue [20] introduces an interactive setting, where the
model answers context-aware questions or participates in a
dialogue about the scene, requiring dynamic grounding and
multi-turn understanding.
C. Metrics
For scene-centric tasks, where captions and answers
typically encompass diverse and richly descriptive content,
we report standard metrics including CIDEr [44], BLEU-4
[32], METEOR [4], ROUGE [14], exact-match accuracy, and
Sentence-BERT [37] similarity.
For object-centric tasks, we exclude BLEU-4 and CIDEr.
BLEU, a precision-based metric, and CIDEr are overly
sensitive to superficial n-gram overlap, rendering them
unsuitable for evaluating long-form object captions [2]. These
captions extend beyond simple object naming (e.g., "a bed")
to include context (e.g., "the bed is rectangular and has a
white bedspread. it is located between the end table with the
lamp on it (...)"). Consequently, BLEU and CIDEr can assign
misleadingly high scores to captions that correctly describe
the scene context but identify the wrong object. This occurs
because these metrics reward overlapping phrases and frequent
n-grams, even with an incorrect core referent. In contrast, we
employ METEOR, ROUGE, and Sentence-BERT similarity,
which offer superior handling of semantic alignment and
partial matches [2], [37]. Specifically, METEOR incorporates
synonym matching and alignment at the word and phrase
level. ROUGE captures structural similarity and emphasizes
recall without over-rewarding redundant context. Sentence-
BERT directly evaluates semantic similarity in embedding
space for robustness to paraphrasing.
D. Implementation Details
Following prior work, we represent each 3D scene using
40k randomly sampled Gaussians from the GaussianWorld
[27] Gaussian splats scene. For the language model, we
adopt OPT-1.3B [52] for LL3DA settings and Vicuna-7B
[13] for LEO, as per their respective training protocols. Both
LLMs are loaded in float16 for memory efficiency and fine-
tuned using LoRa. Our training procedure adheres to the
standard protocols: 5 epochs of alignment followed by 10
epochs of instruction tuning for LEO, and 32 epochs for
LL3DA. Training completes in under one day on 8 A100-80
GPUs. Additionally, we pre-train our task-guided sparsifier
on the object captioning task for 5 epochs We employ the
AdamW optimizer with a weight decay of 0.1 and a cosine
annealing learning rate schedule, decaying from (10−4) to
(10−6). Evaluation is performed every 8 epochs for LL3DA
and every epoch for LEO.
E. Results and Analysis
The evaluation results, shown in Tab. I and Tab. II, highlight
GaussianVLM’s effectiveness across both training protocols
and task types. On the scene-centric SQA3D benchmark,
GaussianVLM achieves an exact match accuracy of 49.4%,
surpassing LEO’s 47.0% by 2.4 percentage points. Under

<!-- page 6 -->
Embodied Dialogue
Embodied Planning
Scene Captioning
Sim
C
B-4
M
R
Sim
C
B-4
M
R
Sim
C
B-4
M
R
OPT-1.3B [52]
-
0.31
0.23
5.62
4.83
-
0.16
0.13
0.24
3.56
-
0.0
0.84
8.40
11.7
OPT-2.7B [52]
-
0.38
0.39
7.38
6.28
-
0.10
0.26
3.59
4.35
-
0.11
0.00
6.60
12.32
OPT-6.7B [52]
-
0.25
0.43
6.88
6.16
-
0.00
0.28
3.65
3.94
-
0.06
1.13
8.99
16.96
LLAMA-7B [42]
-
0.27
0.50
7.81
6.68
-
0.04
0.29
3.53
4.71
-
0.2
0.92
7.00
12.31
LL3DA* [9]
48.2
145.9
22.2
40.9
36.7
50.2
65.1
7.1
20.8
32.2
66.4
0.2
3.0
19.4
18.4
GaussianVLM (Ours)
72.3
270.1
31.5
55.7
48.6
59.0
220.4
20.3
44.5
48.0
65.8
0.8
6.4
23.5
21.1
(a) LL3DA Scene-Centric Benchmarks. We compare 3D VLMs and frozen LLMs, following [9]. Our method, GaussianVLM,
outperforms all baselines by a large margin.
SQA3D
EM1
C
B-4
M
R
GPT3 [7]
41.0
-
-
-
-
ClipBERT [24]
43.3
-
-
-
-
SQA3D [30]
46.6
-
-
-
-
3D-VisTA [55]
48.5
-
-
-
-
PQ3D [56]
47.1
-
-
-
-
LEO* [23]
47.0
124.7
9.4
25.5
48.4
GaussianVLM (Ours)
49.4
129.6
17.1
26.4
50.2
(b) LEO Scene-Centric Benchmarks
TABLE I: Evaluation of SOTA 3D VLMs on scene-centric
3D vision-language tasks. (a) Results on the scene-centric
benchmarks from LL3DA. (b) Results on the scene-centric
benchmarks from LEO. We report results from specialist
models (top) and generalist 3D VLMs (bottom). (*): repro-
duced. Evaluation metrics include CIDEr (C), BLEU-4 (B-4),
METEOR (M), ROUGE (R), Sentence Similarity (Sim), and
Top-1 Exact Match (EM1).
ScanRefer
ScanQA
Nr3D
Sim
M
R
EM1
M
R
Sim
M
R
Scan2Cap [12]
-
21.4 43.5
-
-
-
-
-
-
VoteNet+
MCAN [50]
-
-
-
17.3
11.4 29.8
-
-
-
ScanQA [3]
-
-
-
-
13.14 33.3
-
-
-
3D-LLM [20]
-
13.1 33.2 19.3
13.8 34.0
-
-
-
3D-VLP [48]
-
-
-
-
13.5 34.5
-
-
-
Scene-LLM [18]
-
21.8 45.6
-
15.8
-
-
-
-
LL3DA* [9]
55.9 51.6 54.8 14.3
22.8 34.7 48.1 5.8
9.9
GaussianVLM
(Ours)
59.1 52.4 57.4 14.4
22.9 34.8 48.2 20.8 19.2
TABLE II: Evaluation on object-centric LL3DA benchmarks.
We report both specialist models (top), and 3D VLMs
(bottom). (*): reproduced. Models focusing on grounding
(3D-LLM, 3D-VLP, Scene-LLM) and specialist models were
not reproduced due to differing objectives.
the LL3DA protocol, GaussianVLM significantly improves
embodied dialogue and planning tasks, with CIDEr scores
increasing from 145.9 to 270.1 (+124.2) and from 65.1
to 220.4 (+155.3), respectively, demonstrating enhanced
multi-object reasoning and spatial context understanding. In
object-centric evaluations (Tab. II), GaussianVLM achieves
comparable (ScanQA) or superior results (e.g., Nr3D with
a METEOR score of 20.8 versus 5.8) to existing methods,
despite not employing object detectors.
F. Real-World Generalization
To assess generalization to data obtainable in realistic
real-world settings, we also evaluate GaussianVLM
and
LL3DA on scene representations derived from RGB image
data. Unlike traditional point cloud-based VLMs, which
often rely on laser-scanned geometry, our model is trained
on photorealistic Gaussian splats, potentially offering better
robustness to less structured inputs. For this experiment,
we use the ScanNet++ [49] (validation split), which is out-
of-domain (OOD) for our setup consisting exclusively of
ScanNet scenes. Specifically, we utilize GaussianWorld’s
[27] ScanNet++ scenes, generated from RGB data, for
our 3DGS representation, while the point cloud baseline
(LL3DA) employs COLMAP [39], [38] reconstructions from
ScanNet++. To address the lack of suitable benchmarks on
ScanNet++, we introduce a novel object counting question-
answering dataset. This dataset, automatically constructed
using ScanNet++ segmentation annotations, comprises 1000
question-answer pairs focused on object counts. We exclude
non-object categories to ensure focused evaluation. We
evaluate our model and LL3DA on this OOD dataset using
standard question-answering evaluation protocols, specifically
Exact Match, ROUGE, METEOR, CIDEr, as well as Accuracy.
The results reveal a significant performance advantage for our
Gaussian splat-based model, outperforming the point cloud-
based SOTA VLM (LL3DA) by 474% in accuracy on the
GS scenes (Tab. III). Further details on dataset construction
and statistics are provided in the supplementary material.
G. Ablation Study
To understand the contribution of different components
to GaussianVLM’s performance, we conducted an ablation
study. Our analysis reveals that GaussianVLM’s superior
results are primarily due to: (a) the task-guided sparsifier,
which leverages global context to provide task-specific scene-
level awareness, and (b) the location-guided sparsifier, which
offers localized information crucial for object-centric tasks.
As shown in Tab. IV, removing either of these modules results

<!-- page 7 -->
Model
Accuracy (%)
EM
CIDEr
METEOR
ROUGE
LL3DA [9]
4.2
1.5
54.4
25.5
26.8
GaussianVLM (Ours)
24.1
9.3
120.0
35.2
47.3
Improvement %
+474.0%
+520.0%
+120.6%
+38.0%
+76.5%
TABLE III: Evaluation of QA on object counts on the out-of-domain ScanNet++ validation scenes.
Scene-Centric Tasks
Embodied Dialogue
Embodied Planning
Scene Captioning
Sim
C
B-4
M
R
Sim
C
B-4
M
R
Sim
C
B-4
M
R
(1) No Vision Tokens
13.5
0
0
1.1
0
15.7
0
0
0.4
0
0.3
0
0
0.9
0.4
(2) No Scene Tokens
69.3 234.9 28.0 52.0 45.3 54.1 156.1
3.9
36.9 40.1 61.5 0.7 1.3 15.4 17.4
(3) No ROI Tokens
68.9 233.4 28.1 52.0 44.9 56.8 195.0 12.0 41.0 44.8 63.4 2.5 3.1 19.6 20.9
(4) Only Vision Tokens 34.7
67.5
8.8
24.9 19.9 37.0
46.6
4.1
21.1 25.8 37.8
0
0
0.2
0.3
(5) No Depth-Wise CA 71.2 269.1 30.9 55.2 48.3 58.3 209.3 18.6 44.2 47.9 64.4 2.4 4.9 21.8 21.1
(6) No Text-Guidance
71.4 267.0 31.3 55.5 48.5 58.2 218.5 17.7 44.2 47.8 59.6 0.1 1.6 15.1 17.9
(7) kNN Sparsification
71.2 261.6 31.1 54.9 47.8 58.0 218.0 17.1 44.2 47.8 63.3 1.7 5.4 22.0 20.0
GaussianVLM (Ours) 72.3 270.1 31.5 55.7 48.6 59.0 220.4 20.3 44.5 48.0 65.8 0.8 6.4 23.5 21.1
Object-Centric Tasks
ScanQA
Nr3D
EM1
M
R
Sim
M
R
(1) No Vision Tokens
0
1.6
0
32.0 10.2
9.6
(2) No Scene Tokens
15.4 20.6 32.1 44.3 20.3 18.7
(3) No ROI Tokens
14.2 21.5 34.2 44.1 19.0 18.9
(4) Only Vision Tokens 10.1 14.4 23.5 44.8 19.6 17.7
(5) No Depth-Wise CA 13.9 22.4 33.9 47.9 20.8 19.0
(6) No Text-Guidance
13.6 22.2 33.5 48.2 20.8 19.1
(7) kNN Sparsification
14.3 23.9 35.8 48.8 20.8 18.7
GaussianVLM (Ours)
14.4 22.9 34.8 48.2 20.8 19.2
TABLE IV: Ablation Study of GaussianVLM. [A] Component
Ablation on removing different token types: (1) Vision tokens absent
(text-only input), (2) Task-guided scene tokens absent, (3) Location-
guided ROI tokens absent, (4) Prompt tokens absent (vision-only
input). [B] Task-Guided Sparsifier Architecture Ablation: (5)
The three blocks of cross-attention (CA) are applied only to the
final decoder output, not to hidden states, (6) Task prompt-based
queries replaced with task-unaware learnable queries, (7) Uniform
downsampling replaced with a kNN and attention pooling strategy.
All reported metrics are consistent with Tab. I.
in a substantial performance decrease. We further investigated
the architecture of the task-guided sparsifier.
Task-Guided Sparsifier. We first examined the impact
of task guidance. Replacing text-prompt-based queries with
learnable queries caused a substantial performance decrease,
especially for scene-centric tasks (Tab. IV), where the varied
nature of prompts necessitates dynamic and task-aware selec-
tion of diverse visual cues. Next, we evaluated the benefit of
our depth-wise sparsification strategy. Utilizing only the final
SceneSplat output, instead of leveraging intermediate decoder
features, led to a significant performance drop (Tab. IV)
primarily on scene-centric tasks that require the global context
provided by earlier decoder layers. Finally, we compared our
uniform downsampling strategy to a more advanced language-
unaware alternative (attention pooling for early layers, k-NN
for the final layer, where spatial information is available).
This alternative did not yield improved performance (Tab. IV),
confirming the efficiency of our simpler approach without
compromising information.
V. CONCLUSION
We introduced GaussianVLM, a 3D VLM utilzing
language-aligned Gaussian splats. With GaussianVLM, we
proposed a paradigm shift in 3D vision-language under-
standing by moving away from object-centric representations
towards a holistic, scene-centric and language-based approach.
By directly embedding language features into the spatial
structure of 3D scenes, GaussianVLM, overcomes the inherent
limitations of object detector dependencies, enabling a more
natural and comprehensive understanding of complex envi-
ronments. We also proposed a dual sparsification module that
effectively tackles the challenge of dense language-augmented
scenes. The task-guided component distills the representation
into compact, task-relevant features through task-guided
selection on global context. Notably, with GaussianVLM,
we presented a pioneering 3D VLM operating on Gaus-
sian Splats, leveraging their rich geometric and appearance
information for enhanced scene understanding/reasoning
tailored to the embodied vision and beyond. Our extensive
evaluations across a diverse suite of 3D vision-language
tasks demonstrate the clear advantages of our scene-centric
approach. GaussianVLM consistently achieves state-of-the-
art performance, significantly outperforming existing methods
on scene-centric tasks and also exhibiting strong results
on object-centric benchmarks despite being detector-free.
Finally, we empirically validated the practical generalization
of our method, showing its improved performance on 3D
data collected with more readily available equipment.

<!-- page 8 -->
ACKNOWLEDGMENT
This research was partially funded by the Ministry of Edu-
cation and Science of Bulgaria (support for INSAIT, part of
the Bulgarian National Roadmap for Research Infrastructure).
REFERENCES
[1] Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed Elhoseiny,
and Leonidas J. Guibas. ReferIt3D: Neural listeners for fine-grained
3d object identification in real-world scenes. In ECCV, 2020.
[2] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould.
Spice: Semantic propositional image caption evaluation. In ECCV,
2016.
[3] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe.
Scanqa: 3d question answering for spatial scene understanding. In
CVPR, 2022.
[4] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for
mt evaluation with improved correlation with human judgments. In
Proceedings of the ACL workshop on intrinsic and extrinsic evaluation
measures for machine translation and/or summarization, 2005.
[5] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael
Equi, and et al. π0: A vision-language-action flow model for general
robot control. arXiv:2410.24164, 2024.
[6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D
Kaplan, and et al. Language models are few-shot learners. In NeurIPS,
volume 33, 2020.
[7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D
Kaplan, et al. Language models are few-shot learners. NeurIPS, 33,
2020.
[8] Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. Scanrefer:
3d object localization in rgb-d scans using natural language. ECCV,
2020.
[9] Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu, et al.
Ll3da: Visual interactive instruction tuning for omni-3d understanding
reasoning and planning. In CVPR, 2024.
[10] Sijin Chen, Hongyuan Zhu, Xin Chen, Yinjie Lei, et al. End-to-end
3d dense captioning with vote2cap-detr. In CVPR, 2023.
[11] Yilun Chen, Shuai Yang, Haifeng Huang, Tai Wang, Ruiyuan Lyu,
et al. Grounded 3d-llm with referent tokens. arXiv:2405.10370, 2024.
[12] Zhenyu Chen, Ali Gholami, Matthias Nießner, and Angel X Chang.
Scan2cap: Context-aware dense captioning in rgb-d scans. In CVPR,
2021.
[13] Wei-Lin Chiang, Zhuohan Li, Ziqing Lin, Ying Sheng, Zhanghao Wu,
et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%*
chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023),
2(3), 2023.
[14] Lin Chin-Yew.
Rouge: A package for automatic evaluation of
summaries. In Proceedings of the Workshop on Text Summarization
Branches Out, 2004, 2004.
[15] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas
Funkhouser, and Matthias Nießner.
Scannet: Richly-annotated 3d
reconstructions of indoor scenes. In CVPR, 2017.
[16] Abhishek Das, Samyak Datta, Georgia Gkioxari, Stefan Lee, Devi
Parikh, and Dhruv Batra. Embodied question answering. In CVPR,
2018.
[17] Alexandros Delitzas, Ayca Takmaz, Federico Tombari, Robert Sumner,
Marc Pollefeys, et al. SceneFun3D: Fine-Grained Functionality and
Affordance Understanding in 3D Scenes. In CVPR, 2024.
[18] Rao Fu, Jingyu Liu, Xilun Chen, Yixin Nie, and Wenhan Xiong. Scene-
llm: Extending language model for 3d visual reasoning. In IEEE/CVF
Winter Conference on Applications of Computer Vision (WACV), 2025.
[19] Mei Guofeng, Lin Wei, Riz Luigi, Wu Yujiao, Poiesi Fabio, and Wang
Yiming. Perla: Perceptive 3d language assistant. In CVPR, 2025.
[20] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du,
Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into
large language models. NeurIPS, 2023.
[21] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi
Li, et al. Lora: Low-rank adaptation of large language models. ICLR,
1(2), 2022.
[22] Haifeng Huang, Yilun Chen, Zehan Wang, Rongjie Huang, Runsen
Xu, et al. Chat-scene: Bridging 3d scene and large language models
with object identifiers. In NeurIPS, 2024.
[23] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao
Li, et al. An embodied generalist agent in 3d world. In ICML, 2024.
[24] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, et al. Less
is more: Clipbert for video-and-language learning via sparse sampling.
In CVPR, 2021.
[25] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.
Blip-2:
Bootstrapping language-image pre-training with frozen image encoders
and large language models. In ICML, 2023.
[26] Mingsheng Li, Xin Chen, Chi Zhang, Sijin Chen, Hongyuan Zhu, et al.
M3dbench: Towards omni 3d assistant with interleaved multi-modal
instructions. In ECCV, 2025.
[27] Yue Li, Qi Ma, Runyi Yang, Huapeng Li, Mengjiao Ma, Bin Ren,
Nikola Popovic, Nicu Sebe, Ender Konukoglu, Theo Gevers, et al.
Scenesplat: Gaussian splatting-based scene understanding with vision-
language pretraining. arXiv:2503.18052, 2025.
[28] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual
instruction tuning. In A. Oh, T. Naumann, A. Globerson, K. Saenko,
M. Hardt, and S. Levine, editors, NeurIPS, 2023.
[29] Qi Lv, Hao Li, Xiang Deng, Rui Shao, et al. Robomp2: A robotic
multimodal perception-planning framework with mutlimodal large
language models. In ICML, 2024.
[30] Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, et al.
Sqa3d: Situated question answering in 3d scenes. In ICLR, 2023.
[31] Yunze Man, Liang-Yan Gui, and Yu-Xiong Wang. Situational awareness
matters in 3d vision language reasoning. In CVPR, 2024.
[32] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a
method for automatic evaluation of machine translation. In Proceedings
of the 40th annual meeting of the Association for Computational
Linguistics, 2002.
[33] Songyou Peng, Kyle Genova, Chiyu "Max" Jiang, Andrea Tagliasacchi,
Marc Pollefeys, and Thomas Funkhouser.
Openscene: 3d scene
understanding with open vocabularies. In CVPR, 2023.
[34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel
Goh, et al. Learning transferable visual models from natural language
supervision. In ICML, 2021.
[35] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu.
Exploring the limits of transfer learning with a unified text-to-text
transformer. Journal of Machine Learning Research, 21(140), 2020.
[36] Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, et al.
Sayplan: Grounding large language models using 3d scene graphs for
scalable task planning. In CoRL, 2023.
[37] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embed-
dings using Siamese BERT-networks. In Empirical Methods in Natural
Language Processing. ACL, 2019.
[38] Johannes L Schonberger and Jan-Michael Frahm.
Structure-from-
motion revisited. In CVPR, 2016.
[39] Johannes L Schönberger, Enliang Zheng, Jan-Michael Frahm, and Marc
Pollefeys. Pixelwise view selection for unstructured multi-view stereo.
In ECCV, 2016.
[40] Ola Shorinwa, Johnathan Tucker, Aliyah Smith, Aiden Swann, et al.
Splat-mover: Multi-stage, open-vocabulary robotic manipulation via
editable gaussian splatting. In CoRL, 2024.
[41] Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, et al.
Habitat 2.0: Training home assistants to rearrange their habitat. In
NeurIPS, 2021.
[42] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad
Almahairi, et al. Llama 2: Open foundation and fine-tuned chat models.
arXiv:2307.09288, 2023.
[43] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad
Naeem, Ibrahim Alabdulmohsin, et al. Siglip 2: Multilingual vision-
language encoders with improved semantic understanding, localization,
and dense features. arXiv:2502.14786, 2025.
[44] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider:
Consensus-based image description evaluation. In CVPR, 2015.
[45] Zehan Wang, Haifeng Huang, Yang Zhao, Ziang Zhang, and Zhou Zhao.
Chat-3d: Data-efficiently tuning large language model for universal
dialogue of 3d scenes, 2023.
[46] Erik Wijmans, Samyak Datta, Oleksandr Maksymets, Abhishek Das,
Georgia Gkioxari, Stefan Lee, Irfan Essa, Devi Parikh, and Dhruv
Batra. Embodied question answering in photorealistic environments
with point cloud perception. In CVPR, 2019.
[47] Jimmy Wu, Rika Antonova, Adam Kan, Marion Lepert, et al. Tidybot:
Personalized robot assistance with large language models. In IROS,
2023.
[48] Dejie Yang, Zhu Xu, Wentao Mo, Qingchao Chen, Siyuan Huang, and

<!-- page 9 -->
Yang Liu. 3d vision and language pretraining with large-scale synthetic
data. arXiv:2407.06084, 2024.
[49] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela
Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In ICCV,
2023.
[50] Zhou Yu, Jun Yu, Yuhao Cui, Dacheng Tao, and Qi Tian. Deep modular
co-attention networks for visual question answering. In CVPR, 2019.
[51] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer.
Sigmoid loss for language image pre-training. In ICCV, 2023.
[52] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya
Chen, et al.
Opt: Open pre-trained transformer language models.
arXiv:2205.01068, 2022.
[53] Yue Zhang, Zhiyang Xu, Ying Shen, Parisa Kordjamshidi, and Lifu
Huang. Spartun3d: Situated spatial understanding of 3d world in large
language models. arXiv:2410.03878, 2024.
[54] Hongyan Zhi, Peihao Chen, Junyan Li, Shuailei Ma, Xinyu Sun, et al.
Lscenellm: Enhancing large 3d scene understanding using adaptive
visual preferences. arXiv:2412.01292, 2024.
[55] Ziyu Zhu, Xiaojian Ma, Yixin Chen, Zhidong Deng, Siyuan Huang,
and Qing Li. 3d-vista: Pre-trained transformer for 3d vision and text
alignment. In CVPR, 2023.
[56] Ziyu Zhu, Zhuofan Zhang, Xiaojian Ma, Xuesong Niu, Yixin Chen,
et al.
Unifying 3d vision-language understanding via promptable
queries. In ECCV, 2025.
APPENDIX
This supplementary material provides additional results,
implementation details, and data information supporting the
main paper. In Sec. VI, we present qualitative examples,
extended performance comparisons, and ablation studies to
further validate the effectiveness of our model. Sec. VII
outlines the training and inference configurations used in our
experiments. In Sec. VIII we include further dataset informa-
tion, including a description of the object counting dataset
used for the OOD evaluation, along with relevant licensing
details on all datasets utilized in this work. Finally, we discuss
the limitations of our current approach (Sec. IX)and reflect
on the broader impact of our work (Sec. X).
VI. RESULTS AND ANALYSIS
A. Qualitative Results
We present qualitative examples illustrating our model’s
performance on both scene-centric and object-centric tasks.
As shown in Figure 3, our scene-centric model offers a
more comprehensive understanding of the 3D environment.
Compared to the baseline, it produces fewer false positives
in identified objects and avoids repetitive phrasing in its
responses. This indicates not only stronger spatial and
contextual grounding, but also higher linguistic fluency and
semantic relevance in the generated answers.
In the object-centric examples shown in Figure 4, the
advantages of using Gaussian splats emerge as our model
more accurately identifies fine-grained appearance characteris-
tics—such as color, material, or texture. Moreover, in several
prompts, the baseline model fails to produce meaningful
responses, sometimes returning empty strings or outputs
unrelated to the question. In contrast, our model consistently
generates relevant and visually grounded answers.
B. Further Results
We present additional quantitative comparisons to highlight
the strengths and limitations of our model, GaussianVLM,
across diverse question types. Table V shows performance
on the ScanQA dataset, disaggregated by question categories.
On "How many" questions, which require accurate object
counting, LL3DA slightly outperforms GaussianVLM in
Exact Match (EM) and ROUGE, though our model achieves
a higher METEOR. For "What is" questions, GaussianVLM
consistently outperforms LL3DA across all metrics, indicating
improved reasoning for open-ended identification tasks. The
advantage of our model is more pronounced in "Appearance"
questions, where it achieves the highest scores across EM,
METEOR, and ROUGE—highlighting the benefit of texture-
aware Gaussian Splatting in capturing fine visual details.
On the "Where" category, both models struggle in EM, but
LL3DA scores higher in the other metrics, possibly due to
its design employing object detector, this being more tailored
to location-based queries.
In Table VI, we compare GaussianVLM against LEO on the
SQA3D benchmark across the diverse SQA3D question types.
GaussianVLM demonstrates consistent gains in EM and EM-
Refined metrics for the majority of question categories, includ-
ing "What", "Is", "How", and "Others", with improvements
up to +5.38 EM-Refined points on "How" questions. These
gains suggest our model’s robustness across diverse query
types, particularly those requiring spatial-semantic reasoning.
C. Ablation Experiments
Minimal ROI Radius. We conduct an ablation study on
the minimal radius used for capturing the ROI, comparing two
settings: 15 cm and 30 cm. This analysis is carried out across
three benchmarks, two object-centric and one scene-centric.
Given that the ROI-based sparsifier is designed to enhance
performance on object-centric tasks, we focus primarily on
the object-centric tasks, for which we choose ScanRefer and
ScanQA. To evaluate its impact on scene-centric performance,
we also consider the SQA3D benchmark. The results demon-
strate that a smaller ROI of 15 cm significantly benefits
object-centric tasks, while only causing a negligible drop in
performance on the scene-centric benchmark (Tab. VII).
VII. HYPERPARAMETER CHOICE
We evaluate our model under two established training
protocols from prior state-of-the-art 3D VLMs. We follow
the training setup of LL3DA [9], and for the embodied
reasoning task, SQA3D [30], we adopt the LEO protocol
[23], a benchmarked training strategy for instruction tuning
and alignment in 3D vision-language models.
Our architectural hyperparameters are selected based on
findings from prior work. We set the number of ROI tokens to
4, aligning with LL3DA’s ablation on location-aware feature
encoding using click-based inputs. The number of scene
tokens is fixed at 128, based on LSceneLLM [54], which
demonstrates that this token budget provides a good trade-
off between performance and efficiency for global scene
representations.
The detailed hyperparameter configurations used in our
implementation of the LEO training protocol are listed in
Tables VIII,IX,X. Table VIII
shows the settings for the
alignment stage, Table IX for the instruction-tuning stage,

<!-- page 10 -->
Fig. 3: Qualitative results on scene-centric tasks.

<!-- page 11 -->
Fig. 4: Qualitative results on object-centric tasks.
Task
Method
EM
METEOR
ROUGE
How many
LL3DA
0.2723
0.2868
0.5023
GaussianVLM
0.2277
0.3039
0.4892
What is
LL3DA
0.1128
0.1759
0.2647
GaussianVLM
0.1142
0.1763
0.2606
Appearance
LL3DA
0.2862
0.2288
0.4298
GaussianVLM
0.3170
0.2363
0.4565
Where
LL3DA
0.0000
0.1911
0.2310
GaussianVLM
0.0000
0.1142
0.2009
TABLE V: Performance metrics comparison on different ScanQA categories.
and X for inference. Modifications we made relative to the
original LEO setup (e.g., GPU type, output length) are marked
with an asterisk. These configurations ensure a consistent and
fair comparison while allowing us to scale to high-resolution,
language-enriched 3D scenes using Gaussian splats.
VIII. DATASETS INFORMATION
In this section, we describe the datasets used for our experi-
ments, spanning both established 3D vision-language datasets
as well as a new benchmark introduced in this work for
evaluating object counting in real-world 3D reconstructions.
We provide details on the construction, purpose, and licensing
of the evaluation datasets. In particular, we emphasize the
Object Counts Dataset, built on ScanNet++, which allows
us to evaluate the generalization of our model to out-of-
distribution reconstructions created from RGB data Sec. VIII-
A. Table XI and Sec. VIII-B provide a comprehensive
overview of dataset licensing to ensure full transparency
and reproducibility.
A. Object Counts Dataset
To assess the robustness of GaussianVLM to deployment
scenarios involving diverse data sources, we evaluate its
performance on 3D Gaussian splat representations constructed
from RGB images captured in real-world environments. Un-
like point clouds derived from high-precision laser scanners,
RGB-based reconstructions are more accessible and better
reflect casual data collection. Our motivation stems from
the observation that VLMs trained solely on LiDAR- or
scanner-derived point clouds may struggle to generalize to
reconstructions without such professional setups. Since our
setup comprises only the ScanNet dataset, we leverage the
validation split of ScanNet++ – a high-quality 3D indoor
scene dataset collected independently from ScanNet – for

<!-- page 12 -->
Metric
LEO
GaussianVLM
Improvement
EM Refined (What)
38.45
42.46
+4.01
EM Refined (Is)
63.34
67.94
+4.60
EM Refined (How)
41.72
47.10
+5.38
EM Refined (Can)
69.23
66.27
-2.96
EM Refined (Which)
48.72
46.15
-2.57
EM Refined (Others)
49.29
51.41
+2.12
EM (What)
33.74
37.49
+3.75
EM (Is)
61.66
65.80
+4.14
EM (How)
41.51
46.88
+5.37
EM (Can)
69.23
66.27
-2.96
EM (Which)
47.29
45.30
-1.99
EM (Others)
44.70
48.94
+4.24
TABLE VI: Comparison of EM metrics on SQA3D between
LEO and GaussianVLM for different question groups. EM-
Refined represents an EM adaptation by LEO.
Metric
ROI 15
ROI 30
ScanRefer
Sentence Similarity
0.5914
0.5791
ScanQA
EM
0.1443
0.1401
SQA3D
EM Overall
0.4936
0.4942
EM (What)
0.3749
0.3740
EM (Is)
0.6580
0.6488
EM (How)
0.4688
0.4624
EM (Can)
0.6627
0.6538
EM (Which)
0.4530
0.5014
EM (Others)
0.4894
0.4859
TABLE VII: Comparison of GaussianVLM with ROI thresh-
old 15cm and with 30cm.
evaluation. We use the Gaussian splat representations of these
scenes provided by GaussianWorld, generated from RGB
images, and compare against the COLMAP-derived point
clouds available for ScanNet++. To the best of our knowledge,
no object captioning or question answering benchmarks exist
for this dataset. To address this, we construct a new benchmark
focused on object counting. Using ScanNet++ segmentation
annotations, we automatically extract instance counts and
generate 1,000 question-answer pairs of the form "How many
<label> are in the scene?", with 10 synonym question variants
and 5 possible answer rephrasings per instance (e.g., "3", "3
chairs", "I can count 3", etc.). Labels corresponding to non-
countable "stuff" categories (e.g., wall, floor, windowsill) and
artifacts (e.g., "SPLIT", "REMOVE") are excluded.
Figures 7 and
8 visualize the distribution of object
count questions across object class labels for all 1,000
questions, overlaid with those correctly answered by Gaus-
sianVLM and LL3DA, respectively. These show that Gaus-
sianVLM answers correctly across a wider range of object
types. Complementary Figures 5 and 6 break this down for
GaussianVLM (254 correct answers) and LL3DA (44 correct
answers), highlighting the per-class accuracy gap.
Separately, Figures 10 and 11 show the distribution of
questions by object count values (e.g., “1 chair”, “5 doors”),
Hyperparameter
Value
Optimizer
AdamW
Weight decay
0.05
Betas
[0.9, 0.999]
Learning rate
3 × 10−4
Warmup steps
400
Number of workers
4
Parallel strategy
DDP
Type of GPUs*
NVIDIA A100-80
Number of GPUs*
8
Accumulate gradient batches*
4
Batch size per GPU
4
Training precision
bfloat16
Gradient norm
5.0
Epochs
5
TABLE VIII: Hyperparameters choice of LEO protocol [23]
for alignment stage. (*) marks our modifications.
Hyperparameter
Value
Optimizer
AdamW
Weight decay
0.05
Betas
[0.9, 0.999]
Learning rate
3 × 10−5
Warmup steps
400
Number of workers
4
Parallel strategy
DDP
Type of GPUs*
NVIDIA A100-80
Number of GPUs*
8
Accumulate gradient batches*
4
Batch size per GPU
4
Training precision
bfloat16
Gradient norm
5.0
Epochs
10
TABLE IX: Hyperparameters choice of LEO protocol [23]
for instruction-tuning stage. (*) marks our modifications.
again overlaid with correctly answered instances. These plots
demonstrate that GaussianVLM
generalizes well across
a broader range of object counts, including mid-to-high
cardinalities, while LL3DA struggles with higher counts.
Figure 9 shows the global distribution of object counts in the
benchmark, confirming that the dataset includes a wide and
balanced spectrum of count values.
We evaluate on both LL3DA and GaussianVLM; note that
novel data evaluation is limited to LL3DA due to LEO’s lack
of an integrated object detector enabling evaluation on further
datasets. We evaluate using standard QA metrics – exact
match, ROUGE, METEOR, CIDEr, BLEU – and a custom
accuracy metric. Accuracy accounts for rephrasings and
approximate number matching by extracting numeric tokens
from predictions and ground truths (including both digit
and word forms) and comparing them after normalization,
regardless of the sentence context.
All data will be made publicly available.
B. Dataset Licenses
We summarize the licenses and terms of use for all datasets
used in this work in Table XI. All datasets are publicly
released, and we adhere strictly to the respective terms.
Notably, ScanNet [15] and ScanNet++ [49] are governed
by their own custom terms of use, while other datasets adopt

<!-- page 13 -->
Fig. 5: Distribution of the questions on object counts, answered correctly by GaussianVLM. The distribution is according
to object class labels. Overall, 254 questions answered correctly.
Fig. 6: Distribution of the questions on object counts, answered correctly by LL3DA. The distribution is according to object
class labels. Overall, 44 questions answered correctly.
standard open-source licenses. GaussianWorld (SceneSplat-
7K) inherits licensing from the datasets it reprocesses – in
our case, ScanNet and ScanNet++ – and therefore follows
the same terms.
IX. LIMITATIONS
While GaussianVLM demonstrates strong generalization
and performance across a variety of 3D vision-language tasks,
1Available
at
https://kaldir.vc.in.tum.de/scannet/
ScanNet_TOS.pdf (last accessed: 19/05/2025).
2Available
at
https://kaldir.vc.in.tum.de/scannetpp/
static/scannetpp-terms-of-use.pdf
(last
accessed:
19/05/2025).
several limitations remain.
First, although our model maintains computational parity
with other recent 3D VLMs – measured in total training
hours under comparable training protocols – the broader
class of vision-language models for 3D reasoning remains
computationally intensive. As a result, real-time or resource-
constrained inference may still pose practical challenges.
While our multi-phase, multi-branch sparsification strategy is
specifically designed to reduce computational bottlenecks, the
underlying 3D backbone architecture, though SOTA, remains
heavy.
Second, training these VLMs is also resource-intensive,

<!-- page 14 -->
Fig. 7: Distribution of object count questions (correcly answered by GaussianVLM, vs all questions) according to object
class labels.

<!-- page 15 -->
Fig. 8: Distribution of object count questions (correcly answered by LL3DA, vs all questions) according to object class
labels.

<!-- page 16 -->
Fig. 9: Distribution of object count questions according to object count labels.
Fig. 10: Distribution of object count questions (correcly answered by GaussianVLM, vs all questions) according to object
count labels. Overall, 254 questions answered correctly. Logarithmic scaling for the distribution.
Fig. 11: Distribution of object count questions (correcly answered by LL3DA, vs all questions) according to object count
labels. Overall, 44 questions answered correctly. Logarithmic scaling for the distribution.
Fig. 12: Distribution of object count questions based on ground truth object counts labels (log scale). We show the initial
distribution upon generating the questions (red) and the distribution of the questions used in our evaluations (blue).

<!-- page 17 -->
Fig. 13: Distribution of object count questions based on object type label. We show the initial distribution upon generating
the questions (red) and the distribution of the questions used in our evaluations (blue).

<!-- page 18 -->
Hyperparameters
Value
Number of beams
5
Maximum output length*
768
Minimum output length
1
Top p
0.9
Repetition penalty
3.0
Length penalty
1.0
Temperature
1.0
TABLE X: Hyperparameters choice of LEO protocol [23] for
inference. (*) marks our modifications.
Dataset
License / Terms of Use
ScanNet [15]
ScanNet Terms of Use1
ScanNet++ [49]
ScanNet++ Terms of Use2
ScanRefer [8]
Creative Commons BY-NC-SA 3.0
ScanQA [3]
Apache 2.0
SQA3D [30]
Apache 2.0
ReferIt3D [1]
MIT
3D-LLM [20]
MIT
GaussianWorld
(SceneSplat-7K) [27]
Same as ScanNet, ScanNet++
TABLE XI: Licenses and terms of use for datasets employed
in our study.
requiring a dedicated A100-80 node (8 GPUs) for 24 hours.
Third, our evaluation focuses on static 3D scenes. Dynamic
or multi-agent environments – common in robotics and
AR/VR – are not addressed. Extending the model to handle
time-varying inputs and temporal reasoning is a potential
direction for future work.
Fourth, while Gaussian splatting enables realistic recon-
structions from RGB, the quality and completeness of
reconstructions can vary significantly depending on the
capture process. Our experiments on ScanNet++ assume clean
reconstructions; model performance may degrade on lower-
quality or outdoor scenes.
Finally, our object counting benchmark on ScanNet++
covers only one type of task in the out-of-distribution (OOD)
evaluation setting. A broader set of benchmarks across diverse
OOD conditions is necessary to fully assess the generalization
of 3D VLMs to unconstrained environments.
X. BROADER IMPACT
Our work aims to expand the capabilities of vision-
language models (VLMs) for holistic 3D scene understanding,
moving beyond object-centric paradigms that rely heavily
on predefined taxonomies and bounding-box supervision. By
leveraging scene-centric representations and operating directly
on expressive 3D inputs such as Gaussian splats, our approach
offers potential benefits for real-world applications that require
open-ended, spatially grounded reasoning, such as robotics,
assistive technologies, and AR/VR systems.
However, our approach also comes with potential environ-
mental implications. Although our model is designed with
efficiency in mind – via sparsification and modularization –
training large-scale VLMs, including our own, still requires
significant computational resources and GPU hours. This high
energy consumption contributes to environmental concerns,
showing the need for future research on methods that reduce
training footprints.
