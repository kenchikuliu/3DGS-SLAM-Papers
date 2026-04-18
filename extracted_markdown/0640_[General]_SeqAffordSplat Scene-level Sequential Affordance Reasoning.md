# SeqAffordSplat: Scene-level Sequential Affordance Reasoning on 3D Gaussian Splatting

Di Li1, Jie Feng1\*, Jiahao Chen2, Weisheng Dong1,

Guanbin Li2, Yuhui Zheng3,Mingtao Feng1, Guangming Shi1

1Xidian University

3Qinghai Normal University

dili@stu.xidian.edu.cn, jiefeng0109@163.com,chenjh328@mail2.sysu.edu.cn, wsdong@mail.xidian.edu.cn, liguanbin@mail.sysu.edu.cn, zhengyh@vip.126.com, mintfeng@hnu.edu.cn, gmshi@xidian.edu.cn

<!-- image-->  
Figure 1: (Left)We introduce Sequential 3DGS Affordance Reasoning Task for complex, multi-step agent interactions. (Center)To support this, we present SeqAffordSplat, a large-scale dataset with over 1,700 3DGS scenes and 12,000 instruction pairs.(Right) Our model, SeqSplatNet, sets a new state-of-the-art, improving performance by 6.5% on single-step tasks and 14.1% on our sequential benchmark. Please zoom in for better visual effects.

## Abstract

3D affordance reasoning, the task of associating human instructions with the functional regions of 3D objects, is a critical capability for embodied agents. Current methods based on 3D Gaussian Splatting (3DGS) are fundamentally limited to single-object, single-step interactions, a paradigm that falls short of addressing the long-horizon, multi-object tasks required for complex real-world applications. To bridge this gap, we introduce the novel task of Sequential 3D Gaussian Affordance Reasoning and establish SeqAffordSplat, a largescale benchmark featuring 1800+ scenes to support research on long-horizon affordance understanding in complex 3DGS environments. We then propose SeqSplatNet, an end-to-end framework that directly maps an instruction to a sequence of 3D affordance masks. SeqSplatNet employs a large language model that autoregressively generates text interleaved with special segmentation tokens, guiding a conditional decoder to produce the corresponding 3D mask. To handle complex

scene geometry, we introduce a pre-training strategy, Conditional Geometric Reconstruction, where the model learns to reconstruct complete affordance region masks from known geometric observations, thereby building a robust geometric prior. Furthermore, to resolve semantic ambiguities, we design a feature injection mechanism that lifts rich semantic features from 2D Vision Foundation Models (VFM) and fuses them into the 3D decoder at multiple scales. Extensive experiments demonstrate that our method sets a new state-of-theart on our challenging benchmark, effectively advancing affordance reasoning from single-step interactions to complex, sequential tasks at the scene level.

## Introduction

3D affordance reasoning, which identifies interactive regions on objects to enable specific actions in 3D space, is a fundamental perceptual capability for embodied agents (Deng et al. 2021; Yang et al. 2023). By linking perception and action, it underpins essential functionalities in a spectrum of applications, including robotic manipulation (Yamanobe et al. 2017), augmented reality (Steffen et al. 2019; Nagarajan and Grauman 2020), and virtual reality (Dalgarno and Lee 2010; Venkatakrishnan et al. 2023). This has motivated early explorations into affordance reasoning using point cloud representations (Yang et al. 2023; Li et al. 2024b; Deng et al. 2021). While these approaches demonstrate potential in predicting affordances from 3D geometry, they are often constrained by the inherent sparsity and discrete nature of point clouds, which impedes their ability to capture the fine-grained, continuous structures essential for precise interaction.

Table 1: Comparison of Existing 3D Affordance Datasets with Ours.
<table><tr><td>Benchmark</td><td>Vision Type</td><td>Scene-level Support</td><td>Sequence Support</td><td>#Object Cat.</td><td>#Afford. Type</td></tr><tr><td rowspan="3">3D AffordanceNet (CVPR&#x27;21) LASO (CVPR&#x27;24)</td><td>PointCloud</td><td>Ã</td><td>Ã</td><td>23</td><td>17</td></tr><tr><td>PointCloud</td><td>Ã</td><td>Ã</td><td>23</td><td>17</td></tr><tr><td>PointCloud</td><td>Ã</td><td>â</td><td>23</td><td>18</td></tr><tr><td rowspan="2">SeqAfford (CVPR&#x27;25) 3DAffordSplat (ACM MM&#x27;25) Ours</td><td>Gaussians</td><td></td><td>Ã&gt;</td><td>21</td><td>18</td></tr><tr><td>Gaussians</td><td>Ã&gt;</td><td></td><td>21</td><td>18</td></tr></table>

Inspired by the high-fidelity representations of 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023), the transition from sparse point clouds to 3DGS has drawn increasing attention for 3D scene understanding (Mohammadi et al. 2023; Zhu et al. 2024). The pioneering work (Wei et al. 2025) demonstrates promising improvement for precise affordance reasoning in 3DGS scenes, suggesting its superiority in capturing fine-grained affordance detail. However, this method targets at a specific and controlled task prototype, where each scene consists of a single instances and each instruction requires just a single atomic action for execution.

Due to the inherent high-level succinctness of real-world instructions, 3D affordance reasoning requires both (i) the composition of ordered primitive instructions (each corresponding to a primitive action), and (ii) the dynamic shift of actionable regions across object instances in complex scenes. As illustrated in Fig. 1, an instruction such as operating a microwave to warm up food in a bowl necessitates multiple interdependent actions across three distinct actionable regions from different instancesâa composite capability beyond current methods due to their constrained task prototypes. This reveals a critical gap in 3D affordance reasoning: the absence of a task formulation for sequential interaction in cluttered scenes. To bridge this gap, we introduce the Scene-level Sequential 3D Gaussian Affordance Reasoning taskâa novel prototype designed for long-horizon and succinct instructions in complex environments with multiple interactive regions and distractor objects, fundamentally advancing beyond prior object-centric, single-action approaches.

To facilitate research in this new direction, we introduce SeqAffordSplat, the first comprehensive benchmark designed for long-horizon, scene-wide affordance reasoning on 3DGS. The benchmark features a new large-scale dataset containing over 1,800+ complex scenes, 14,000+ affordance masks, and 8,000+ sequential instructions. To establish a complete evaluation framework, we complement the dataset with a suite of novel sequential metricsâsIoU, sAUC, sSIM, and sMAEâtailored to holistically assess performance on multi-step tasks.

To handle the shift in task prototype, we introduce SeqSplatNet, the first framework designed to solve this new scene-level sequential task. Our approach uniquely integrates the hierarchical planning capabilities of Large Language Models (LLMs) with the rich representational power of 3DGS in a unified, end-to-end architecture. Given a complex, long-horizon instruction, our model reasons about the userâs intent and grounds a sequence of actionable affordance masks directly onto the 3DGS scene representation. To handle complex scene geometry, we design a selfsupervised pre-training strategy, Conditional Geometric Reconstruction, where the model learns to reconstruct complete affordance regions from partial geometric observations, thereby building a robust geometric prior. Furthermore, to resolve semantic ambiguities, we design a Semantic Feature Injection mechanism that lifts rich semantic features from a frozen 2D Vision Foundation Model (VFM) (Oquab et al. 2023; Radford et al. 2021) via multi-view rendering and fuses them into the 3D decoder at multiple scales. This unified approach bridges the critical gap between high-level task planning and low-level, fine-grained 3D perception in complex environments. Our main contributions are summarized as follows:

â¢ We introduce the new task of Sequential 3D Gaussian Affordance Reasoning and develop SeqAffordSplat, the first large-scale benchmark with over 1,800+ 3DGS scenes and 14,000+ ground-truth affordance segmentations.

â¢ We propose SeqSplatNet, the first framework to unify high-fidelity 3DGS representation, long-horizon sequential planning, and complex scene-level understanding.

â¢ We demonstrate through extensive experiments that our approach achieves a 14.1% performance improvement over sequential baselines on our challenging new benchmark.

## Related Work

## Affordance Learning

Affordance Learning focuses on identifying interactive regions in a scene, driven by either a closed-set of action types or open-vocabulary language instructions. This originates from 2D image affordance segmentation, where each pixel is assigned to a predefined affordance category (Do, Nguyen, and Reid 2018; Roy and Todorovic 2016). To generalize to unseen affordance types, recent approaches integrate Vision-Language Models (VLMs), aligning language instructions with visual affordances (Chen et al. 2025; Li et al. 2024a; Qian et al. 2024). However, these image-based approaches intrinsically lack the ability to capture explicit 3D spatial informationâa critical requirement for robotic manipulation applications.

To overcome the limitations of image-based perception, researchers turns to 3D representations for acurrate gemoetry awareness in 3D space. Pioneering studies (Deng et al. 2021; Xu et al. 2022; Mo et al. 2022) established benchmarks for affordance segmentation on 3D point clouds for predefined affordance type sets. This foundation later has evolved toward open-vocabulary affordance reasoning, where models identify actionable regions in response to language instructions by leveraging the cross-modality reasoning capability of foundational models (Li et al. 2024b; Xu et al. 2022; Shao et al. 2025; Lu et al. 2025). Despite this evolution, these approaches still struggle with long-horizon reasoning, which stems from their constrained task prototype that each instruction envolves a single action.

To address this limitation, we highlight sequential affordance reasoning as a more practical task prototype, where each instruction maps to a sequence of atomic affordances. Compared with the contemporary SeqAfford (Yu et al. 2025), which is capable of generating sequential affordance masks on 3D cloud points, our SeqSplatNet overcomes its inefficiency in localizing fine-grained actionable regions through high-fidelity 3DGS representation.

## Affordance Learning on 3DGS

Beyond 3D point clouds, 3DGS offers a more expressive representation through its explicit point-based structure and real-time rendering capabilities (Kerbl et al. 2023). These characteristics facilitate the development of various embodied AI systems (Zheng et al. 2024; Shorinwa et al. 2024; Lu et al. 2024), as 3DGS enables both instantaneous environmental perception and direct association of semantic information with spatially precise geometric locations.

In affordance learning, 3DAffordSplat (Wei et al. 2025) established the first large-scale benchmark for affordance reasoning using 3DGS. While notable for its substantial instance volume and diverse affordance coverage, this benchmark adopts a constrained task prototype where each instruction maps exclusively to a single discrete affordance mask, inherently omitting sequential reasoning pathways essential for complex manipulation scenarios.

In summary, existing approaches remain confined to either sequential reasoning on sparse point clouds with inadequate localization fidelity or single-step interactions on 3DGS without action sequencing capability. To the best of our knowledge, we pioneer the first framework for scene-level affordance learning in 3DGS that simultaneously achieves fine-grained affordance localization and sequential reasoning.

## Task Definition and Dataset

## Task Definition

Scene-Level Sequential 3D Gaussian Affordance Reasoning presents a challenging task requiring the identification of step-wise affordance regions in a complex 3DGS scene containing multiple interactive instance parts, in accordance with a succinct input instruction that outlines a composite process of multiple ordered primitive actions.

Specifically, consider a 3D scene represented by a 3DGS model G comprising N Gaussian primitives, $\mathcal { G } = \backslash G _ { i } \} _ { i = 1 } ^ { N }$ Each primitive $G _ { i }$ is parameterized by its position, opacity, scale, rotation, and spherical harmonics coefficients. Given a succinct instruction $Q _ { i n s t } .$ , this task aims to predict an ordered sequence of $T$ binary affordance masks $\mathcal { M } = ( M _ { 1 } , M _ { 2 } , . . . , M _ { T } )$ . Each mask $\mathbf { \bar { \boldsymbol { M } } } _ { t } \in \{ 0 , 1 \} ^ { N }$ identifies the subset of Gaussians that constitute the functional region for the t-th atomic instruction, which is implicitly defined by the instructionâs action plan, as illustrated in Fig. 1. The objective of this task is to find an optimal mapping $F$ that satisfies

$$
\mathcal { M } = F ( Q _ { i n s t } , \mathcal { G } ) .\tag{1}
$$

This formulation extends the traditional affordance segmentation paradigm from identifying what affordances exist to reasoning about in what order they must be actualized to fulfill a userâs complex intent.

## SeqAffordSplat Dataset Collection

To facilitate research into long-horizon, scene-level affordance reasoning, we introduce SeqAffordSplat, the first large-scale benchmark designed to evaluate sequential affordance grounding directly on 3DGS representations. The construction of the SeqAffordSplat benchmark is a twostage process meticulously designed to ensure high fidelity and ecological validity across its two core components: the 3D scene geometry, and the sequential, language-grounded instructions.

Step 1: 3DGS Data Collection. The foundation of our benchmark lies in the quality and complexity of its 3D environments. To properly evaluate long-horizon reasoning, which often involves interactions among multiple objects, single-object models are insufficient.To generate realistic environments, we manually composed scenes by positioning multiple objects from the 3D-AffordanceNet (Wei et al. 2025) dataset using geometric transformations, including translation, rotation, and scaling, to emulate plausible realworld scenarios.

Step 2: Instruction and Affordance Annotation. Rather than annotating from scratch, we transfer affordance labels from established benchmarks, primarily 3D-AffordanceNet (Wei et al. 2025), which provides dense, point-wise affordance labels for thousands of object shapes. The transfer is performed via a semi-automated pipeline: we programmatically match object instances in our scenes with annotated categories in 3DAffordSplat (Wei et al. 2025), project the point-wise labels onto our 3DGS representation, and then use a custom 3D annotation tool for manual verification. For a sequential instruction, the ground truth is stored as an ordered list of affordance masks, explicitly encoding the temporal and causal order required for the task. To generate a large and diverse set of long-horizon instructions, we utilize the multimodal large language model (MLLM) GPT-4o (Achiam et al. 2023) inspired by (Yu et al. 2025). We employ a sophisticated prompt engineering strategy that provides the LLM with rich context for each scene, including Visual Context, Textual Context, Role Prompting and Goal Specification. The generated instruction-sequence pairs undergo a final human-in-the-loop curation process to ensure they are logical, physically possible, and correspond to available affordances. Additional details are provided in the supplementary materials.

<!-- image-->  
Figure 2: An overview of the proposed SeqSplatNet architecture. The architecture comprises four main components: a Large Language Model, a 3DGS Encoder with Conditional Geometric Reconstruction Pre-train, and a Conditional Affordance Decoder with VFM Semantic Feature Injection.

As shown in 1, the final benchmark contains 1800+ unique 3DGS scenes, annotated with over 14,000+ distinct affordance masks across 21 object categories and 18 affordance types. The language component features approximately 8000+ instructions. A key characteristic is its focus on long-horizon tasks.

## Evaluation Configurations and Metrics

Experimental Settings. Inspired by the evaluation settings in prior work (Yu et al. 2025; Li et al. 2024b), we establish three distinct configurations to comprehensively evaluate our method:

â¢ Single: Evaluates the modelâs ability to predict individual, unordered affordance regions.

â¢ Sequential (with gt seq): Assesses affordance grounding accuracy given a ground-truth action sequence.

â¢ Sequential: Tests the full task, where the model must infer and execute the entire action sequence from a single high-level instruction.

Evaluation Metrics. For single-step prediction, we adopt standard metrics mIoU, AUC, SIM, MAE to ensure a fair comparison with prior works like LASO (Li et al. 2024b). For sequential task , we introduce a suite of sequential metrics: sIoU, sAUC, sSIM, and sMAE. The calculation is straightforward: we first align the predicted and groundtruth sequences to the same length by padding the shorter sequence with empty frames. This approach is reasonable because it inherently penalizes discrepancies in sequence length. Additional details are provided in the supplementary materials.

## SeqSplatNet

## Architecture

Our SeqSplatNet features an end-to-end architecture that directly maps language instructions to sequential 3D affordance masks. Through an autoregressive process, the model generates interleaved language tokens and special <SEG> tokens, where each <SEG> emission dynamically triggers the affordance decoder to produce a 3D affordance mask. This design inherently unifies task planning and localization by embedding action sequencing within the generative process, eliminating explicit hierarchical decomposition.

As illustrated in Fig. 2, our SeqSplatNet comprises three core components: a 3DGS Encoder, a Large Language Model (LLM) and a Conditional Affordance Decoder, collectively constituting our base model. We augment this framework with two key enhancements: Conditional Geometric Reconstruction Pre-train for improved 3DGS Encoder initialization, and VFM Semantic Feature Injection to enrich geometric representations with semantic knowledge extracted from 2D VFMs.

3DGS Encoder. We adopt a PointNet-based encoder (Qi et al. 2017) to extract geometric information from a 3DGS scene G. Consistent with 3DAffordSplat (Wei et al. 2025), our encoder processes the geometric attributes (position, rotation and $\mathtt { S C a l e } )$ of Gaussian primitives in ${ \mathcal { G } } ,$ generating point-wise geometric features $\dot { F } _ { \mathrm { g e o } } \in \mathbb { R } ^ { N \times d }$

LLM. Our LLM serves as the central reasoning engine, processing an input instruction $Q _ { \mathrm { i n s t r } }$ to autoregressively generate a primitive instruction sequence. Inspired by recent advancements in Multimodal LLM (Li et al. 2024b; Wei et al. 2025), we augment the token vocabulary with a special token <SEG>. Within the interleaved sequence of language tokens and <SEG> tokens, each <SEG> simultaneously activates affordance mask decoding for its associated primitive instruction and provides a dynamic instruction vector $h _ { \mathrm { s e g } } \in \mathbb { R } ^ { d }$ derived from its hidden state. Benifiting from the masked attention mechanism in LLM, this vector effectively encodes the contextual dependencies from $Q _ { \mathrm { i n s t r } }$ and its preceding primitives.

Conditional Affordance Decoder. This decoder generates the affordance mask conditioned on each obtained dynamic instruction vector $h _ { \mathrm { s e g } }$ . Built upon recent querybased segmentation paradigm (Cheng, Schwing, and Kirillov 2021), it employs each LLM-derived dynamic instruction vector $h _ { \mathrm { s e g } }$ as a latent query to decode its corresponding 3D affordance mask $M _ { t }$ from the encoded geometric feature and injected semantic information (as detailed subsequently).

This tight integration of reasoning and perception within a unified autoregressive framework enables end-to-end sequential reasoning in complex 3DGS scenes with our SeqSplatNet.

End-to-end Training of SeqSplatNet. Our SeqSplatNet aims to generate fine-grained affordance masks associated for accurately reasoned primitive instruction sequences for succinct input instructions. To this end, its overall loss summarizes both the language-modeling misalignments $\mathcal { L } _ { \mathrm { l a n g } }$ and the affordance segmentation errors $\mathcal { L } _ { \mathrm { m a s k } }$

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { l a n g } } + \lambda _ { \mathrm { m a s k } } \sum _ { t = 1 } ^ { T } \mathcal { L } _ { \mathrm { m a s k } } ,\tag{2}
$$

where $\lambda _ { \mathrm { m a s k } }$ is a balancing hyperparameter. In this work, we adopt the standard autoregressive cross-entropy loss over the predicted token sequence for $\mathcal { L } _ { \mathrm { l a n g } }$ . The segmentation loss $\mathcal { L } _ { \mathrm { m a s k } }$ , activated at each ${ < } \mathrm { S E G } >$ token, is a combination of Binary Cross-Entropy (BCE) and Dice losses to ensure both pixel-wise accuracy and structural similarity:

$$
\mathcal { L } _ { \mathrm { m a s k } } = \mathcal { L } _ { \mathrm { B C E } } ( \hat { M } _ { t } , M _ { t } ^ { \mathrm { g t } } ) + \mathcal { L } _ { \mathrm { D i c e } } ( \hat { M } _ { t } , M _ { t } ^ { \mathrm { g t } } ) ,\tag{3}
$$

where $\hat { M } _ { t }$ and $M _ { t } ^ { \mathrm { g t } }$ are the predicted and ground-truth masks for step t, respectively.

## Conditional Geometric Reconstruction Pre-train

Obaining an effective 3DGS encoder from scratch for complex scene-level 3DGS is challenging, due to its reliance on a huge amount of annotated samples. To tackle this issue, we propose a pre-training strategy for providing improved initialization to our perception modules. Our core idea is to instill a geometric prior into the 3DGS encoder by tasking it with reconstructing a spatial affordance region conditioned solely on an abstract semantic embedding.

Specifically, given a 3DGS scene $\mathcal { G }$ and an affordance mask $M ^ { \mathrm { g t } } \in \mathsf { \bar { \{ 0 , 1 \} } } ^ { N }$ , our architecture employs a dualencoder design. The 3DGS encoder $\Phi _ { \mathrm { e n c } }$ projects the scene G into per-point feature embeddings $F _ { \mathrm { g e o } }$ , and the mask encoder $\Phi _ { \mathrm { m a s k } }$ maps the mask $M ^ { \mathrm { g t } }$ into a single conditional embedding $e _ { \mathrm { m a s k } }$ that represents the abstract affordance concept. The reconstruction is then conditioned on this embedding. Specifically, $e _ { \mathrm { m a s k } }$ serves as a query to attend to the per-point geometric features $F _ { \mathrm { g e o } } ,$ producing a fused representation $F _ { \mathrm { f u s e d } } .$ . A decoder, $\Phi _ { \mathrm { d e c } } ,$ then processes these features to reconstruct the final mask $\hat { M } \colon$

$$
F _ { \mathrm { f u s e d } } = \mathrm { A t t e n t i o n } ( Q = e _ { \mathrm { m a s k } } , K = F _ { \mathrm { g e o } } , V = F _ { \mathrm { g e o } } )
$$

$$
\hat { M } = \Phi _ { \mathrm { d e c } } ( F _ { \mathrm { f u s e d } } )\tag{4}
$$

(5)

This pre-training task compels the network to learn a powerful mapping from an abstract semantic concept to its corresponding spatial geometry. By explicitly conditioning the reconstruction on the mask embedding, the model learns a disentangled representation that separates the affordance concept from the geometric structure, providing a superior initialization for the downstream task.

## VFM Semantic Feature Injection

Interpreting nuanced language instructions to identify affordances requires a deep semantic understanding that pure geometric representations cannot provide for complex scenes. To bridge this gap, we leverage the high-fidelity rendering capability of 3DGS to inject potent semantic knowledge from pre-trained 2D Vision Foundation Models (VFM).

For a scene represented by a set of n 3D Gaussians, we first generate m multi-view 2D feature maps $\{ F ^ { ( v ) } \} _ { v = 1 } ^ { m } .$ Each feature map is obtained by processing a rendered RGB image $I ^ { ( v ) }$ with a frozen, pre-trained VFM, Î¨VFM (e.g., DI-NOv2 (Oquab et al. 2023),CLIP (Radford et al. 2021)):

$$
F ^ { ( v ) } = \Psi _ { \mathrm { V F M } } ( I ^ { ( v ) } ) \in \mathbb { R } ^ { H \times W \times d _ { \mathrm { s e m } } }\tag{6}
$$

To lift these 2D features into the 3D space, we employ a learning-free aggregation process akin to an inverse rendering operation following the learning-free lifting paradigm (Marrie et al. 2024) . This approach correctly handles the contribution of multiple Gaussians to each rendered pixel via alpha-blending. The semantic feature vector $f _ { i } ^ { \mathrm { s e m } }$ for each Gaussian i is computed as a weighted average of all the 2D pixel features it influences across all views. The weight for each pixel feature $F _ { p } ^ { ( v ) }$ (from view v at pixel p) is its rendering weight $w _ { i } ( v , p )$ , which represents the influence of Gaussian i on that pixel. The lifted feature is defined as:

$$
f _ { i } ^ { \mathrm { s e m } } = \frac { \sum _ { ( v , p ) \in S _ { i } } w _ { i } ( v , p ) F _ { p } ^ { ( v ) } } { \sum _ { ( v , p ) \in S _ { i } } w _ { i } ( v , p ) }\tag{7}
$$

where $S _ { i }$ is the set of all view-pixel pairs $( v , p )$ that Gaussian i contributes to. This aggregation method inverts the rendering process to produce a semantically-rich feature bank $F _ { \mathrm { s e m } } \in \mathbf { \mathbb { R } } ^ { n \times d _ { \mathrm { s e m } } }$ attached to the Gaussians.

The the lifted semantic features $F _ { \mathrm { s e m } }$ are injected into the Conditional Affordance Decoder at multiple scales using additive fusion, this multi-scale strategy enhances semantic consistency in segmentation by informing the decoding process at all levels of granularity, from coarse to fine-grained.

Table 2: Results on SeqAffordSplat dataset
<table><tr><td>Main results</td><td>Method</td><td>Source</td><td>mloU/sIoUâ</td><td>AUC/sAUCâ</td><td>SIM/sSIMâ</td><td>MAE/sMAEâ</td></tr><tr><td rowspan="3">Single</td><td rowspan="3">3DAffordSplat PointRefer IAGNet OURS</td><td rowspan="3">ACM MM&#x27;25 CVPR&#x27;24 ICCV&#x27;23</td><td>30.5</td><td>92.7</td><td>0.395</td><td>0.065</td></tr><tr><td>31.3</td><td>92.1</td><td>0.411</td><td>0.055</td></tr><tr><td>17.6</td><td>85.2</td><td>0.328</td><td>0.056</td></tr><tr><td rowspan="3">Sequential (with GT seq)</td><td rowspan="3">3DAffordSplat PointRefer IAGNet</td><td rowspan="3">- ACM MM&#x27;25 CVPR&#x27;24 ICCV&#x27;23</td><td>37.0</td><td>94.0</td><td>0.470</td><td>0.049</td></tr><tr><td>26.1</td><td>91.2</td><td>0.343</td><td>0.072</td></tr><tr><td>30.3 13.9</td><td>91.2 88.0</td><td>0.418 0.325</td><td>0.055 0.062</td></tr><tr><td rowspan="2">Sequential</td><td rowspan="2">OURS SeqAfford</td><td rowspan="2">- CVPR&#x27;25</td><td>36.0</td><td>95.6</td><td>0.457</td><td>0.036</td></tr><tr><td>12.1</td><td>73.0</td><td>0.122</td><td>0.230</td></tr><tr><td></td><td>OURS</td><td>-</td><td>26.2</td><td>80.6</td><td>0.312</td><td>0.132</td></tr></table>

## Experiments

## Experimental Settings

Baseline Models. Since our method is the first sequence reasoning approach based on 3DGS, for a fair comparison, we selected the following baselines: 3DAffordSplat (Wei et al. 2025), a single-step reasoning method based on 3DGS; PointRefer (Li et al. 2024b) and IAGNet (Yang et al. 2023), single-step methods based on point clouds; and SeqAfford (Yu et al. 2025), a sequence reasoning method based on point clouds. We evaluate these methods under various settings.

Implementation Details. For our primary results, we chose the Qwen-3-0.6B (Yang et al. 2025) as the LLM, formatting inputs with its official chat template. The model was finetuned for our task using Low-Rank Adaptation (LoRA) (Hu et al. 2022). Both geometric and semantic feature dimensions projected set to 512. The model was first pre-trained for 10 epochs at a learning rate of $1 \times 1 0 ^ { - 4 }$ . It was then trained for 50 epochs, where all non-LLM parameters used a decaying learning rate of $1 \times 1 0 ^ { - 4 }$ . For Qwenâs LoRA fine-tuning, we targeted the q proj, k proj, v proj, and lm head layers with a rank of 8 and an alpha of 16. The Adam optimizer (Kingma and Ba 2014) with weight decay was used for both training stages. Experiments were conducted on 8 GeForce RTX 3090 GPUs.

## Results on SeqAffordSplat Dataset

We performed the Sequential 3D Gaussian Affordance Reasoning task on the SeqAffordSplat Dataset. As outlined in the Dataset section, this task is categorized into three distinct settings based on the nature of the instructions: Single, Sequential (with ground-truth sequence), and end-to-end Sequential. The main results are presented in Table 2.

Single. In the single-step setting, models predict an affordance mask for a single, explicit instruction. Our method achieves a state-of-the-art performance with an mIoU of 37.0 and an AUC of 94.0. This represents a significant improvement of 5.7 points in mIoU over the strongest point-cloud-based baseline, PointRefer (31.3), and a 6.5- point improvement over the 3DGS-based baseline, 3DAffordSplat (30.5). Notably, the performance of 3DAfford-Splat is slightly lower than that of PointRefer. A possible reason is that 3DAffordSplat treats the 3D Gaussians simply as a point cloud enriched with features like opacity, scale, and rotation, without fully leveraging the high-fidelity rendering capabilities inherent to 3DGS. This demonstrates the superior capability of our architecture in understanding and grounding instructions even in non-sequential scenarios.

Table 3: Results on 3DAffordSplat dataset
<table><tr><td>Method</td><td>mIoUâ</td><td>AUCâ</td><td>SIMâ</td><td>MAEâ</td></tr><tr><td>3DAffordSplat</td><td>30.3</td><td>83.9</td><td>0.440</td><td>0.210</td></tr><tr><td>IAGNet</td><td>14.6</td><td>56.7</td><td>0.350</td><td>0.410</td></tr><tr><td>PointRefer</td><td>18.4</td><td>78.5</td><td>0.430</td><td>0.200</td></tr><tr><td>OURS</td><td>40.2</td><td>89.3</td><td>0.530</td><td>0.169</td></tr></table>

Sequential(with GT seq). This setting evaluates the modelâs ability to ground affordances given a ground-truth sequence of sub-instructions. This isolates the performance of the perception module from the language reasoning component. Our method continues to outperform all baselines, achieving an sIoU of 36.0. This is 5.7 points higher than the next-best baseline, PointRefer, indicating that our conditional decoder excels at accurately interpreting specific subtasks and generating precise affordance masks.

Sequential. This is the full end-to-end task, where the model must reason about a complex instruction and generate the entire sequence of affordance masks. Since other baselines do not support end-to-end sequential reasoning, we compare our method against SeqAfford (Yu et al. 2025), the only available baseline for this task, which operates on point clouds. In this challenging setting, our method demonstrates a remarkable improvement. We achieve an sIoU of 26.2, which is more than double the performance of SeqAfford (12.1). This substantial gain of 14.1 points in sIoU underscores the effectiveness of our integrated reasoning and perception framework, which leverages the LLM to decompose tasks and the decoder to ground them in the 3DGS representation.

## Results on 3DaffordSplat Datasets

To validate the generalization capability of our approach, we also evaluated it on the existing 3DAffordSplat dataset (Wei et al. 2025), which focuses on single-step affordance reasoning on 3D Gaussian data. As shown in Table 3, our method achieves an mIoU of 40.2, significantly outperforming all prior methods. This result is 9.9 points higher than the original 3DAffordSplat benchmark, confirming that the architectural designs and training strategies proposed in our work are robust and effective beyond our new sequential task, setting a new state-of-the-art on this established benchmark as well.

<!-- image-->

<!-- image-->  
Figure 3: Visual Results of our proposed methods.

Table 4: Ablation Study of Main Components
<table><tr><td colspan="2">Component</td><td rowspan="2">sloUâ</td><td rowspan="2">sAUCâ</td><td rowspan="2">sSIMâ</td><td rowspan="2">sMAEâ</td></tr><tr><td>Pretrain</td><td>Feature</td></tr><tr><td>Ã</td><td>Ã</td><td>20.3</td><td>76.3</td><td>0.229</td><td>0.169</td></tr><tr><td>â</td><td>Ã</td><td>24.1</td><td>78.5</td><td>0.302</td><td>0.141</td></tr><tr><td>â</td><td>CLIP</td><td>24.2</td><td>79.1</td><td>0.290</td><td>0.141</td></tr><tr><td>â</td><td>DINO v2</td><td>26.2</td><td>80.6</td><td>0.312</td><td>0.132</td></tr></table>

Table 5: Ablation Study of LLM backbones
<table><tr><td>LLM</td><td>sIoUâ</td><td>sAUCâ</td><td>sSIMâ</td><td>sMAEâ</td></tr><tr><td>GPT2-small(0.1B)</td><td>12.1</td><td>43.9</td><td>0.156</td><td>0.488</td></tr><tr><td>Qwen3-0.6B</td><td>26.2</td><td>80.6</td><td>0.312</td><td>0.132</td></tr><tr><td>Qwen3-1.7B</td><td>26.4</td><td>79.5</td><td>0.291</td><td>0.147</td></tr><tr><td>Qwen3-8B</td><td>24.2</td><td>78.6</td><td>0.285</td><td>0.148</td></tr></table>

## Ablation Study

We conducted ablation studies to analyze the contribution of each key component in our framework on Sequential task.

Effects of Different Components. The effectiveness of our main components is validated in Table 4. Our Conditional Geometric Reconstruction pre-training provides a substantial performance boost over a baseline model, improving the sIoU from 20.3 to 24.1. The subsequent injection of rich semantic features from DINOv2 further lifts the performance to 26.2 sIoU. This demonstrates that both our pretraining strategy and semantic feature fusion are critical to the modelâs success.

Effects of Different LLM Encoders. We investigated the impact of the LLM backbone on reasoning performance, with results shown in Table 5. The results demonstrate that the choice of LLM is critical for the sequential reasoning task. The GPT2-small (0.1B)(Radford et al. 2019) model serves as a baseline, achieving a mere 12.1 sIoU, which underscores the necessity of a more capable language model. Among the Qwen3 series, our primary model, Qwen3-0.6B, performs very competitively with an sIoU of 26.2, while also achieving the best sAUC (80.6) and sSIM (0.312) scores. Interestingly, the largest model tested, Qwen3-8B, shows a performance degradation with an sIoU of 24.2. This suggests that simply increasing parameter count does not guarantee better performance on this task. Consequently, we selected Qwen3-0.6B for our main experiments as it offers an excellent trade-off between high performance across multiple metrics and model efficiency.

## Qualitative Results

Figure 3 presents a qualitative comparison of our method against baselines. In single-step scenarios (a), our model demonstrates superior precision. For example, it correctly identifies the specific liftable part of an bag based on a nuanced instruction, while competing methods segment the incorrect region. More critically, for the sequential task (b), our model successfully decomposes a high-level command (âlisten to music using a laptopâ) into a logical, multi-step sequence of affordances. In contrast, the baseline method fails to generate a coherent plan, visually validating our frameworkâs advanced reasoning and grounding capabilities for complex, long-horizon tasks.

## Conclusion

In this paper, we advance 3D affordance reasoning from single-step, object-centric interactions to complex, sequential tasks at the scene level. We introduced SeqSplatNet, the first framework to unify a causal language model with a high-fidelity 3DGS representation for this new paradigm. Bolstered by novel geometric pre-training and semantic feature injection techniques, our method establishes a new state-of-the-art, outperforming the prior sequential baseline by 14.1% on our newly proposed SeqAffordSplat benchmark. This work provides a critical foundation for developing more capable embodied agents that can understand and execute long-horizon instructions in complex environments.

## References

Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Chen, D.; Kong, D.; Li, J.; and Yin, B. 2025. MaskPrompt: Open-Vocabulary Affordance Segmentation with Object Shape Mask Prompts. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 2034â2042.

Cheng, B.; Schwing, A.; and Kirillov, A. 2021. Per-pixel classification is not all you need for semantic segmentation. Advances in neural information processing systems, 34: 17864â17875.

Dalgarno, B.; and Lee, M. J. 2010. What are the learning affordances of 3-D virtual environments? British journal of educational technology, 41(1): 10â32.

Deng, S.; Xu, X.; Wu, C.; Chen, K.; and Jia, K. 2021. 3d affordancenet: A benchmark for visual object affordance understanding. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 1778â1787.

Do, T.-T.; Nguyen, A.; and Reid, I. 2018. Affordancenet: An end-to-end deep learning approach for object affordance detection. In 2018 IEEE international conference on robotics and automation (ICRA), 5882â5889. IEEE.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; Chen, W.; et al. 2022. Lora: Low-rank adaptation of large language models. ICLR, 1(2): 3.

Kerbl, B.; Kopanas, G.; Leimkuhler, T.; and Drettakis, G. Â¨ 2023. 3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4): 139â1.

Kingma, D. P.; and Ba, J. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

Li, G.; Sun, D.; Sevilla-Lara, L.; and Jampani, V. 2024a. One-shot open affordance learning with foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3086â3096.

Li, Y.; Zhao, N.; Xiao, J.; Feng, C.; Wang, X.; and Chua, T.- s. 2024b. Laso: Language-guided affordance segmentation on 3d object. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14251â14260.

Lu, D.; Kong, L.; Huang, T.; and Lee, G. H. 2025. Geal: Generalizable 3d affordance learning with cross-modal consistency. In Proceedings of the Computer Vision and Pattern Recognition Conference, 1680â1690.

Lu, G.; Zhang, S.; Wang, Z.; Liu, C.; Lu, J.; and Tang, Y. 2024. Manigaussian: Dynamic gaussian splatting for multitask robotic manipulation. In European Conference on Computer Vision, 349â366. Springer.

Marrie, J.; MenÂ´ egaux, R.; Arbel, M.; Larlus, D.; and Mairal, Â´ J. 2024. LUDVIG: Learning-free uplifting of 2d visual features to Gaussian splatting scenes. arXiv preprint arXiv:2410.14462.

Mo, K.; Qin, Y.; Xiang, F.; Su, H.; and Guibas, L. 2022. O2o-afford: Annotation-free large-scale object-object affordance learning. In Conference on robot learning, 1666â 1677. PMLR.

Mohammadi, S. S.; Duarte, N. F.; Dimou, D.; Wang, Y.; Taiana, M.; Morerio, P.; Dehban, A.; Moreno, P.; Bernardino, A.; Del Bue, A.; et al. 2023. 3dsgrasp: 3d shape-completion for robotic grasp. arXiv preprint arXiv:2301.00866.

Nagarajan, T.; and Grauman, K. 2020. Learning affordance landscapes for interaction exploration in 3d environments. Advances in Neural Information Processing Systems, 33: 2005â2015.

Oquab, M.; Darcet, T.; Moutakanni, T.; Vo, H.; Szafraniec, M.; Khalidov, V.; Fernandez, P.; Haziza, D.; Massa, F.; El-Nouby, A.; et al. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193.

Qi, C. R.; Yi, L.; Su, H.; and Guibas, L. J. 2017. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems, 30.

Qian, S.; Chen, W.; Bai, M.; Zhou, X.; Tu, Z.; and Li, L. E. 2024. Affordancellm: Grounding affordance from vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7587â 7597.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748â8763. PmLR.

Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.; Sutskever, I.; et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8): 9.

Roy, A.; and Todorovic, S. 2016. A multi-scale cnn for affordance segmentation in rgb images. In European conference on computer vision, 186â201. Springer.

Shao, Y.; Zhai, W.; Yang, Y.; Luo, H.; Cao, Y.; and Zha, Z.-J. 2025. Great: Geometry-intention collaborative inference for open-vocabulary 3d object affordance grounding. In Proceedings of the Computer Vision and Pattern Recognition Conference, 17326â17336.

Shorinwa, O.; Tucker, J.; Smith, A.; Swann, A.; Chen, T.; Firoozi, R.; Kennedy III, M.; and Schwager, M. 2024. Splat-mover: Multi-stage, open-vocabulary robotic manipulation via editable gaussian splatting. arXiv preprint arXiv:2405.04378.

Steffen, J. H.; Gaskin, J. E.; Meservy, T. O.; Jenkins, J. L.; and Wolman, I. 2019. Framework of affordances for virtual reality and augmented reality. Journal of management information systems, 36(3): 683â729.

Venkatakrishnan, R.; Venkatakrishnan, R.; Raveendranath, B.; Pagano, C. C.; Robb, A. C.; Lin, W.-C.; and Babu, S. V. 2023. How virtual hand representations affect the perceptions of dynamic affordances in virtual reality. IEEE Transactions on Visualization and Computer Graphics, 29(5): 2258â2268.

Wei, Z.; Lin, J.; Liu, Y.; Chen, W.; Luo, J.; Li, G.; and Lin, L. 2025. 3DAffordSplat: Efficient Affordance Reasoning with 3D Gaussians. arXiv preprint arXiv:2504.11218.

Xu, C.; Chen, Y.; Wang, H.; Zhu, S.-C.; Zhu, Y.; and Huang, S. 2022. Partafford: Part-level affordance discovery from 3d objects. arXiv preprint arXiv:2202.13519.

Yamanobe, N.; Wan, W.; Ramirez-Alpizar, I. G.; Petit, D.; Tsuji, T.; Akizuki, S.; Hashimoto, M.; Nagata, K.; and Harada, K. 2017. A brief review of affordance in robotic manipulation research. Advanced Robotics, 31(19-20): 1086â 1101.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yang, Y.; Zhai, W.; Luo, H.; Cao, Y.; Luo, J.; and Zha, Z.-J. 2023. Grounding 3d object affordance from 2d interactions in images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 10905â10915.

Yu, C.; Wang, H.; Shi, Y.; Luo, H.; Yang, S.; Yu, J.; and Wang, J. 2025. Seqafford: Sequential 3d affordance reasoning via multimodal large language model. In Proceedings of the Computer Vision and Pattern Recognition Conference, 1691â1701.

Zheng, Y.; Chen, X.; Zheng, Y.; Gu, S.; Yang, R.; Jin, B.; Li, P.; Zhong, C.; Wang, Z.; Liu, L.; et al. 2024. Gaussiangrasper: 3d language gaussian splatting for open-vocabulary robotic grasping. IEEE Robotics and Automation Letters.

Zhu, S.; Wang, G.; Kong, X.; Kong, D.; and Wang, H. 2024. 3d gaussian splatting in robotics: A survey. arXiv preprint arXiv:2410.12262.