# Zero-Shot Robotic Manipulation via 3D Gaussian Splatting-Enhanced Multimodal Retrieval-Augmented Generation

Zilong Xie1\*, Jingyu Gong1, 2, 3\*, Xin Tan1, 2, Zhizhong Zhang1, 3â , Yuan Xie1, 2

1School of Computer Science and Technology, East China Normal University, Shanghai, China

2Chongqing Key Laboratory of Precision Optics, Chongqing Institute of East China Normal University, Chongqing, China

3Shanghai Key Laboratory of Computer Software Evaluating and Testing, Shanghai, China

zilong6037@gmail.com, {jygong, xtan, zzzhang, yxie}@cs.ecnu.edu.cn

## Abstract

Existing end-to-end approaches of robotic manipulation often lack generalization to unseen objects or tasks due to limited data and poor interpretability. While recent Multimodal Large Language Models (MLLMs) demonstrate strong commonsense reasoning, they struggle with geometric and spatial understanding required for pose prediction. In this paper, we propose RobMRAG, a 3D Gaussian Splatting-Enhanced Multimodal Retrieval-Augmented Generation (MRAG) framework for zero-shot robotic manipulation. Specifically, we construct a multi-source manipulation knowledge base containing object contact frames, task completion frames, and pose parameters. During inference, a Hierarchical Multimodal Retrieval module first employs a threepriority hybrid retrieval strategy to find task-relevant object prototypes, then selects the geometrically closest reference example based on pixel-level similarity and Instance Matching Distance (IMD). We further introduce a 3D-Aware Pose Refinement module based on 3D Gaussian Splatting into the MRAG framework, which aligns the pose of the reference object to the target object in 3D space. The aligned results are reprojected onto the image plane and used as input to the MLLM to enhance the generation of the final pose parameters. Extensive experiments show that on a test set containing 30 categories of household objects, our method improves the success rate by 7.76% compared to the best-performing zero-shot baseline under the same setting, and by 6.54% compared to the state-of-the-art supervised baseline. Our results validate that RobMRAG effectively bridges the gap between high-level semantic reasoning and low-level geometric execution, enabling robotic systems that generalize to unseen objects while remaining inherently interpretable.

Code â https://github.com/XieZilongAI/RobMRAG

## Introduction

Robotic manipulation fundamentally requires precise interaction with diverse objects, where accurate prediction of contact points and 3D poses is critical for task success. Without reliable estimation of object poses, robots are often forced into trial-and-error adjustments, resulting in inefficiency and failure (Zhang et al. 2024). Therefore, robots must possess strong reasoning and generalization capabilities to accurately predict low-level action parameters and adapt to dynamic environments.

Although existing works (Geng et al. 2023a,b,c) have explored end-to-end learning for grasp pose prediction, their generalization remains limited due to constrained data scale and the black-box nature of deep models, which lack commonsense reasoning capabilities. Recent advances in Multimodal Large Language Models (MLLMs) (Zhang et al. 2023; Li et al. 2022, 2024a) have demonstrated promising performance in vision-language understanding and cross-modal reasoning. Several efforts (Huang et al. 2023; Zitkovich et al. 2023; Li et al. 2024b) utilize MLLMs for high-level instruction generation. For example, ManipLLM (Li et al. 2024b) employs chain-of-thought prompting and multi-task finetuning to guide category-level tasks. However, current MLLMs still lack sufficient understanding of geometric structure and spatial layout, limiting their effectiveness in precise grasp pose prediction, as shown in Figure 1(a).

Moreover, zero-shot generalization is a key capability for building general-purpose agents, enabling robots to manipulate unseen objects and adapt to varying embodiments and environments without task-specific training. Existing methods often rely on expert demonstrations (Vuong et al. 2023; Khazatsky et al. 2024) or transfer from human interaction data such as HOI (Liu et al. 2022; Grauman et al. 2022; Luo et al. 2022) and internet videos (Chen et al. 2025). RAM (Kuang et al. 2024) performs zero-shot manipulation via cross-domain affordance retrieval, yet its 2D-to-3D lifting pipeline is brittle under viewpoint or geometric shifts, and the MLLMâs reasoning remains outside the geometric core, limiting pose precision in complex scenes.

To address these challenges, we propose RobMRAG, a 3D Gaussian Splatting-Enhanced Multimodal Retrieval-Augmented Generation (MRAG) framework for zero-shot robotic manipulation (see Figure 1(b)). First, a multi-source manipulation knowledge base is built from simulation, robotic dataset, and Internet data, containing rich multimodal information such as object contact frames, taskcompletion frames, and contact poses. During inference, a hierarchical multimodal retrieval module first uses a threepriority hybrid retrieval strategy to find task-relevant operation prototypes. Then, pixel-level cosine similarity and Instance Matching Distance (IMD) identify the closest geometric reference. Within the MRAG framework, a 3D Gaussian Splatting-based pose refinement module is introduced to perform rigid transformations on retrieved reference poses for precise 3D alignment with target objects. The aligned poses are then reprojected onto the 2D image plane and fed into the MLLM to guide accurate manipulation pose generation. Extensive experiments show that RobM-RAG outperforms state-of-the-art methods under both zeroshot and supervised settings, effectively bridging task understanding and concrete operation for open-world robotic manipulation.

<!-- image-->  
Figure 1: (a) Although multimodal large language models (MLLMs) possess the commonsense knowledge that opening a drawer requires grasping the handle and can roughly localize it, the predicted 2D contact points consistently fall outside the table itself, let alone on the handle. (b) We incorporate contact pose prediction using a multimodal retrieval-augmented generation (MRAG) framework. By retrieving relevant manipulation examples, our method enables more accurate inference of the contact pose on the target object.

The key contributions of this work are summarized as follows:

â¢ We propose a Multimodal Retrieval-Augmented Generation (MRAG) framework for zero-shot robotic manipulation, which enables manipulation on unseen objects through a multi-source knowledge base.

â¢ We integrate a 3D-Aware Pose Refinement module into the MRAG framework, enabling precise pose alignment between reference and target objects, thereby enhancing the geometric consistency of the retrieved results.

â¢ Experimental results demonstrate that, on a test set comprising 30 categories of household objects, the proposed method achieves a 7.76% improvement in success rate compared with the SOTA zero-shot baseline, and a 6.54% improvement compared with the SOTA supervised baseline.

## Related Works

## Zero-shot Robotic Manipulation

Zero-shot robotic manipulation enables robots to perform novel tasks without task-specific training. Existing methods fall into two categories. The first attempts to directly learn transferable manipulation policies from human videos (Bharadhwaj et al. 2024; Chang, Prakash, and Gupta 2023; Bharadhwaj et al. 2023; Grauman et al. 2022; Xu et al. 2023; Chen et al. 2025), either through imitation learning by extracting intermediate representations such as hand trajectories and poses, or by constructing reward functions for reinforcement learning. However, these methods often rely on manually collected video datasets (Smith et al. 2020; Xiong et al. 2021) or require online fine-tuning (Bahl, Gupta, and Pathak 2022), which limits their applicability in real-world scenarios. The second category builds databases of diverse manipulation demonstrations (Nguyen et al. 2022; Ju et al. 2024; Kuang et al. 2024), retrieving demonstrations similar to the current task and adapting them to robot execution. Yet, the effectiveness of such methods heavily depends on the coverage of the demonstration database, which limits their generalization to novel tasks or unseen scenes.

In contrast, our method explicitly bridges the inherent gap between task-level intent and instance-level geometric parameters by constructing a multimodal manipulation knowledge base and integrating Hierarchical Multimodal Retrieval with a 3D-Aware Pose Refinement module.

## Multimodal Large Language Models for Robotics

Multimodal Large Language Models (MLLMs) unify visual perception, language understanding, and action planning into a single framework (Zhang et al. 2023; Li et al. 2022). Models like RT-2 (Zitkovich et al. 2023) and Robo-Mamba (Liu et al. 2024) leverage large-scale pretraining to translate natural language into executable control commands. Recent works (Zhen et al. 2024; Kim et al. 2024; Yue et al. 2024) adopt Transformer-based architectures for grounding semantics to actions. To improve adaptability in dynamic environments, while VoxPoser (Huang et al. 2023)

introduces language-driven affordance reasoning, dynamically constructing 3D semantic value maps to guide manipulation. Beyond task-specific designs, foundation models such as PaLM-E (Driess et al. 2023) and Flamingo (Alayrac et al. 2022) further extend cross-modal reasoning capabilities. ManipLLM (Li et al. 2024b) further enhances generalization and stability in grasp pose prediction by incorporating adapters and chain-of-thought reasoning mechanisms.

Building upon this foundation, we combine multimodal retrieval-augmented generation with MLLMs, significantly improving zero-shot generalization in robotic manipulation tasks.

## Multimodal Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) mitigates limitations of Large Language Models (LLMs) like outdated knowledge and hallucination by leveraging external memory (Lewis et al. 2020; Guu et al. 2020). Extending to multimodal domains, Multimodal RAG (MRAG), which integrates both visual and textual inputs, has become a prominent research direction (Chen et al. 2022; Ma et al. 2024; Bonomo and Bianco 2025; Liu et al. 2025b; Yu et al. 2025; Liu et al. 2025c). Representative MRAG approaches, such as MuRAG (Chen et al. 2022), M2RAG (Ma et al. 2024), and MRAMG (Yu et al. 2025), typically enhance the quality of generation in tasks like visual question answering by retrieving image-text pairs from external memory, overcoming conventional RAGâs limitations in visual understanding. Meanwhile, the incorporation of graph neural networks and knowledge graphs (Dong et al. 2024; Edge et al. 2024; Guo et al. 2024; Liu et al. 2025a; Wu et al. 2025), exemplified by GraphRAG (Edge et al. 2024) and LightRAG (Guo et al. 2024), captures complex cross-modal relations for better semantic reasoning.

In this work, we integrate MRAG techniques with MLLMs in the domain of robotic manipulation, enabling high-precision prediction of operational poses.

## Method

Our goal is to develop a framework for zero-shot robotic manipulation that can accurately and efficiently respond to user instructions in unseen environments. To this end, as illustrated in Figure 2, we present RobMRAG, a 3D Gaussian Splatting-Enhanced Multimodal Retrieval-Augmented Generation framework for zero-shot robotic manipulation. In the following sections, we elaborate on the details of each component of RobMRAG.

## Multi-Source Knowledge Base Construction

We construct a multimodal manipulation knowledge base that integrates data from simulation, real-world or synthetic robotic dataset, and the Internet. Each manipulation case contains one or two key action frames: the object contact frame ${ \mathcal { F } } _ { \mathrm { c t c } } .$ , capturing the moment when the robot (or human) correctly makes contact with the object and initiates the manipulation, and the task-completion frame $\mathcal { F } _ { \mathrm { s u c } } .$ , showing the final successful state. It also includes a corresponding instruction text $I _ { e }$ describing the task. Specifically, for simulation data, we directly obtain precise 6D grasp parameters: the 2D contact point $p _ { e } = ( x , y )$ and end-effector directions $D _ { e } = ( d _ { u } , d _ { f } )$ , where the gripper up direction $d _ { u } =$ $\left( x _ { u } , y _ { u } , z _ { u } \right)$ . These data are collected from multiple successful executions across various object categories and manipulation types.

Real-world data are primarily sourced from human-object interaction video datasets (e.g. HOI4D (Liu et al. 2022)), where 2D contact points and motion directions are extracted via hand-keypoint detection and annotated as arrows on the contact-frame images. Internet data offer a more diverse range of manipulation scenarios, including examples extracted from instructional videos and animations. Although these cases often lack accurate 3D parameters, we apply a semi-automatic annotation pipeline to generate 2D grasp sketches. Ultimately, we construct a comprehensive multi-source manipulation knowledge base $\boldsymbol { \mathcal { K } } = \dot { \left\{ ( I _ { e } , \mathcal { F } _ { \mathrm { c t c } } , \mathcal { F } _ { \mathrm { s u c } } , p _ { e } , D _ { e } ) \right\} }$ }, which furnishes rich crossdomain priors for subsequent retrieval augmentation.

## Hierarchical Multimodal Retrieval

Our framework employs a hierarchical retrieval strategy to identify the optimal reference example from the knowledge base. This process cascades from high-level textual semantics down to visual similarity and, finally, to fine-grained geometric matching, ensuring a precise instruction-to-example alignment.

Textual Semantic Retrieval. The first layer operates on the textual modality, utilizing a three-priority hybrid strategy to find semantically relevant candidates. Given a natural language instruction $I = \{ w _ { i } \} _ { i = 1 } ^ { L }$ , this strategy first executes its top priority: using sparse retrieval (BM25) to find manipulation examples with matching object names, primarily targeting our high-quality simulation data source. If no direct match is found (e.g., for a novel object category not in K), the system defaults to its second priority: dense retrieval based on semantic embeddings. At this stage, the instruction text is projected into a dense vector by a pre-trained language encoder, and the cosine similarity $S _ { \mathrm { t e x t } }$ between semantic embeddings is then computed:

$$
S _ { \mathrm { t e x t } } ( I , I _ { e } ) = \frac { \hat { I } \cdot \hat { I _ { e } } } { \Vert \hat { I } \Vert \Vert \hat { I _ { e } } \Vert } .\tag{1}
$$

Here, $\hat { I } = \mathrm { E n c } ( I )$ and $\hat { I } _ { e } = \mathrm { E n c } ( I _ { e } )$ are the semantic embeddings of the instruction and an example text, respectively. As a final fallback, should the top similarity score fall below a threshold $\tau _ { d e n } ,$ , the third priority is triggered: the retrieval scope is expanded to our broader robotic dataset and Internet data sources. This textual retrieval stage yields a candidate set of examples that are conceptually aligned with the given task.

Visual Similarity Filtering. Once the candidate set is obtained, the second layer refines it based on coarse visual similarity. The system computes the cosine similarity SCLIP between the current observation image ${ \mathcal { F } } _ { \mathrm { o b s } }$ and each candidateâs contact frame $\mathcal { F } _ { \mathrm { c t c } }$ using a CLIP image encoder, selecting the top-n visually similar samples:

$$
S _ { \mathrm { C L I P } } ( \mathcal { F } _ { \mathrm { o b s } } , \mathcal { F } _ { \mathrm { c t c } } ) = \frac { f _ { \mathrm { o b s } } \cdot f _ { \mathrm { c t c } } } { \| f _ { \mathrm { o b s } } \| \| f _ { \mathrm { c t c } } \| }\tag{2}
$$

<!-- image-->  
Figure 2: Overview of the RobMRAG: a 3D Gaussian splatting-enhanced multimodal retrieval-augmented generation framework.

where $f _ { \mathrm { o b s } }$ and $f _ { \mathrm { c t c } }$ are the visual feature vectors extracted by CLIP from the ${ \mathcal { F } } _ { \mathrm { o b s } }$ and ${ \mathcal { F } } _ { \mathrm { c t c } } .$ , respectively. This step effectively filters out candidates that, while semantically related, are visually distinct from the current scene.

Geometric Matching. In the final layer, we perform finegrained geometric matching on the remaining candidates to find the best reference prototype. We use Instance Matching Distance (IMD) to measure the geometric discrepancy between the observed object and each candidate by accounting for local feature consistency:

$$
\begin{array} { r l } {  { \mathrm { I M D } ( \mathcal { F } _ { \mathrm { o b s } } , \mathcal { F } _ { \mathrm { c t c } } , \mathcal { M } _ { \mathrm { o b s } } ) } \quad } & { } \\ & { = \underset { p \in \mathcal { M } _ { \mathrm { o b s } } } { \sum } \Vert \mathrm { F } ^ { \mathrm { o b s } } ( p ) - \mathrm { N N } ( \mathrm { F } ^ { \mathrm { o b s } } ( p ) , \mathrm { F } ^ { \mathrm { c t c } } ) \Vert _ { 2 } } \end{array}\tag{3}
$$

where $\mathcal { M } _ { \mathrm { o b s } }$ is the observed instance mask, $\mathbf { F } ( p )$ is the dense feature vector at pixel p, and $\operatorname { N N } ( \cdot )$ finds the nearest neighbor match. The example with the minimum IMD is selected. If this score is below an empirically-determined threshold ÏIMD, it is used directly as the final reference. Otherwise, it serves as a geometric prior for the subsequent pose refinement module.

## 3D-Aware Pose Refinement

For reference examples requiring further alignment, our framework employs a 3D-aware pose refinement module. We first generate a 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023; Gong et al. 2021a,b) representation from the referenceâs RGB-D image using a pre-trained generative model TRELLIS (Xiang et al. 2025). This allows us to re-render the object from the single input viewpoint. The core formulation expresses the object surface through a differentiable collection of Gaussian distributions $\mathcal { G } = \overline { { \{ ( \mu _ { i } , \Sigma _ { i } , c _ { i } ) \} _ { i = 1 } ^ { N } } }$ where $\mu _ { i } \in \mathbb { R } ^ { 3 }$ denotes the Gaussian center position, $\Sigma _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ controls the spatial distribution, and $c _ { i }$ represents the color attributes.

To eliminate the viewpoint discrepancy, our method applies a predefined set of small-angle rotational transformations $\{ \bar { \bf R } _ { \bf k } \} _ { k = 1 } ^ { K }$ to the reference grasp pose. This strategy is sufficient as the preceding IMD filtering has already ensured a high degree of initial geometric alignment. We first project the 2D reference contact point $p _ { e }$ to 3D space $p _ { e } ^ { \prime }$ using depth information, then transform both the 3D reference contact point and orientation:

$$
p _ { k } ^ { \prime } = \mathbf { R _ { k } } p _ { e } ^ { \prime } , D _ { k } ^ { r } = D _ { e } \otimes \operatorname { Q u a t } ( \mathbf { R _ { k } } )\tag{4}
$$

where $\otimes$ denotes quaternion multiplication. The differentiable rendering of 3DGS generates corresponding contact frames $\{ \mathcal { F } _ { \mathrm { c t c } } ^ { r } \}$ for each candidate pose after reprojecting the 3D contact points to the 2D image plane:

$$
p _ { k } ^ { r } = \pi ( \mathbf { M } [ \mathbf { R } _ { \mathbf { k } } | t ] p _ { k } ^ { \prime } )\tag{5}
$$

where $\pi ( \cdot )$ is the camera projection function, M is the intrinsic matrix, and t denotes the translational compensation. Finally, we obtain the re-rendered reference set $\{ \dot { K _ { k } ^ { r } } \} _ { k = } ^ { K }$ =1 and retain the pose whose frames yield the lowest IMDk as the final reference inputs for the MLLM.

## Optional LoRA Fine-Tuning and Inference

The framework employs a joint loss function combining Masked Language Modeling (MLM) with pose regression tasks, achieving precise grasp pose prediction through Low-Rank Adaptation (LoRA) of the multimodal large language model.

MLM randomly masks grasp parameters in input text, forcing the model to infer masked numerical characters from context. The loss function is defined as character-level crossentropy at masked positions:

$$
\mathcal { L } _ { \mathrm { M L M } } = - \sum _ { i \in \mathcal { M } } \sum _ { c \in \mathcal { C } _ { i } } \log P ( c \mid w _ { \backslash \mathcal { M } } , \mathcal { F } _ { \mathrm { o b s } } , \mathcal { F } _ { \mathrm { c t c } } )\tag{6}
$$

where $\mathcal { M }$ denotes the set of masked positions, $\mathcal { C } _ { i }$ represents the character sequence at the i-th masked position, $w _ { \backslash M }$ indicates the unmasked text context, and $\mathcal { F } _ { \mathrm { o b s } } / \mathcal { F } _ { \mathrm { c t c } }$ are the observation and reference images respectively.

The fine-tuning process supervises structured outputs with contact point (x,y) constrained by mean squared error and end-effector directions $\left( d _ { u } , d _ { f } \right)$ regularized through cosine similarity:

$$
\mathcal { L } _ { \mathrm { p o s e } } = \lambda _ { 1 } \Vert ( x , y ) - ( \hat { x } , \hat { y } ) \Vert _ { 2 } ^ { 2 } + \lambda _ { 2 } ( 2 - d _ { u } \cdot \hat { d } _ { u } - d _ { f } \cdot \hat { d } _ { f } )\tag{7}
$$

where $\lambda _ { 1 } , \lambda _ { 2 }$ are balance weights, $\hat { d } _ { u }$ and $\hat { d } _ { f }$ denote predicted direction vectors. The total loss combines both components: $\mathcal { L } = \mathcal { L } _ { \mathrm { M L M } } + \mathcal { L } _ { \mathrm { p o s e } }$

During inference, the framework first retrieves the most task-relevant reference through Hierarchical Multimodal Retrieval and 3D-Aware Pose Refinement module, which are then provided as additional inputs to the MLLM for generating the operational pose. The final output adopts a structured format containing normalized 2D contact point and two 3D end-effector directions (representing gripper up and forward directions). 2D contact point is projected to 3D manipulation space via depth map ${ \mathcal { F } } _ { \mathrm { d e p } } .$ , constructing complete endeffector poses. We apply active impedance adaptation policy (Li et al. 2024b) to adjust movement direction.

## Experiments

## Datasets and Evaluation Metrics

We construct interactive manipulation environments based on the SAPIEN simulator and the PartNetMobility dataset (Xiang et al. 2020), which provides part-level motion annotations for a wide range of object categories. Following the baseline setup (Li et al. 2024b), simulated manipulation is performed using a Franka Panda robotic arm equipped with a suction gripper, rendered through the VulkanRenderer engine. To generate training data, we randomly sample contact points on movable parts and use the inverse surface normal as the initial operation direction, focusing on pull-type actions to ensure consistency between the motion and manipulation directions. In total, we collect approximately 20,000 successful manipulation samples across 20 object categories as our training set. Following the same procedure, we generate the test set, which is further divided into a Test Seen Split, containing objects of seen categories but with different instances, and a Test Unseen Split, containing entirely novel object categories.

The primary evaluation metric is manipulation average success rate (ASR) (Li et al. 2024b), defined as the proportion of successful trials exceeding a displacement threshold $\delta .$ We use two levels: $\delta = 0 . 0 1$ for verifying initial pose prediction, and $\delta = 0 .$ 1 for assessing sustained motion. Active impedance control is applied to adapt motion direction in response to interaction uncertainties.

## Competitor Methods

To validate our Multimodal Retrieval-Augmented Generation (MRAG) framework, we compare it against six robotic manipulation methods under identical train/test splits and end-effector configurations (suction grippers substituted for original parallel grippers in baselines). We compare our method with the following baselines: Where2Act (Mo et al. 2021) predicts pixel-level affordances for generalization; UMPNet (Xu, He, and Song 2022) generates 6DoF actions from monocular input via temporal causality; Flowbot3D (Eisner, Zhang, and Held 2022) infers 3D motion fields from point clouds for planning; Implicit3D (Zhong et al. 2023) extends Transporter to 3D using spatio-temporal keypoints; RAM (Kuang et al. 2024) is a zero-shot framework lifting retrieved 2D affordances to 3D actions; ManipLLM (Li et al. 2024b) uses MLLM chain-of-thought for planning and pose prediction; CrayonRobo (Li et al. 2025) is a prompt-driven VLA model using 2D visual prompts on images to predict SE(3) actions.

For evaluation, we consider the following settings: Zeroshot: Directly applies the method without task-specific training or fine-tuning. All: After fine-tuning, the model is evaluated on the complete test set to assess its generalization capability in real-world scenarios. Local: After fine-tuning, evaluation is performed only on samples where the contact point is correctly predicted, representing the theoretical upper bound given accurate contact point localization.

## Comparative Analysis

As shown in Table 1, our frameworkâs efficacy is first demonstrated in the zero-shot setting, where RobMRAG achieves an average success rate of 43.53% on the unseen split, surpassing the RAM baseline by 7.76 percentage points. This highlights the strength of our framework, which combines the advantages of MLLMs and MRAG. This design enables effective utilization of external knowledge bases, allowing the model to generalize to novel objects without task-specific training, as seen in high-performing tasks like Table (82.75%). In the fine-tuned All setting, our modelâs advantage becomes more pronounced: it achieves a 57.54% average SR on unseen categories, which is 6.54 percentage points higher than the strongest baseline, ManipLLM. On certain tasks, our model reaches exceptionally high success rates, such as Oven (86.00% vs. ManipLLMâs 42.00%) and Laptop (96.08% vs. 43.00%). In the Local evaluation setting, where a perfect contact point is provided, our pose generation process still demonstrates superior robustness. Specifically, we observe average success rate improvements of 14.20% on the seen split and 13.77% on the unseen split. These gains are attributed to our 3D-Aware Pose Refinement module, which aligns retrieved examples in 3D space using Gaussian Splatting and reprojects them to provide the MLLM with more accurate geometric context. This enriched spatial information leads to more precise 6-DoF pose estimation.

<table><tr><td rowspan="2">Method</td><td colspan="10">Test Seen Split</td><td colspan="4"></td><td colspan="3"></td></tr><tr><td>O </td><td></td><td></td><td>B</td><td>m</td><td></td><td>I1 </td><td>â </td><td></td><td></td><td></td><td>:</td><td>i</td><td>*</td><td>B</td><td></td><td>â¢</td></tr><tr><td>RAM (Kuang et al. 2024) (Zero-shot)â </td><td>43.14</td><td>25.67</td><td>14.00</td><td>35.29</td><td>52.94</td><td>35.49</td><td>49.02</td><td>45.39</td><td>52.00</td><td>49.23</td><td></td><td>45.76</td><td>36.02</td><td>24.00</td><td>61.58</td><td>24.00</td><td>72.00</td></tr><tr><td>CrayonRobo(Li et al. 2025) (Zero-shot)</td><td>17.65 62.75</td><td>21.59 47.45</td><td>28.00</td><td>47.06 6.67</td><td>62.18</td><td>33.33</td><td>57.45</td><td>58.54</td><td>48.00</td><td>49.19</td><td></td><td>37.24 49.02</td><td>57.39 38.00</td><td>26.00 22.00</td><td>73.33 67.85</td><td>12.00 18.00</td><td>68.00 78.00</td></tr><tr><td>Ours (Zero-shot) Where2Act (Mo et al. 2021)</td><td>26.00</td><td>36.00</td><td>20.00 19.00</td><td>27.00</td><td>50.98 23.00</td><td>117.65</td><td>559.18 15.00</td><td>72.05 47.00</td><td>64.00 14.00</td><td>43.14 24.00</td><td>13.00</td><td></td><td>12.00</td><td></td><td>68.00</td><td></td><td>40.00</td></tr><tr><td>UMPNet (Xu, He, and Song 2022) Flowbot3D (Eisner, Zhang, and Held 2022)</td><td>46.00</td><td>3.00</td><td>15.00</td><td>28.00</td><td>54.00</td><td>11.00 32.00</td><td>28.00</td><td>6.00</td><td>44.00</td><td>40.00</td><td></td><td>10.00</td><td>23.00</td><td>56.00 18.00</td><td>54.00</td><td>7.00 20.00</td><td>42.00</td></tr><tr><td>Implicit3D (Zhong et al. 2023)</td><td>6700</td><td>55.00</td><td>20.00</td><td>32.00</td><td>27.00</td><td>31.00</td><td>6100</td><td>68.00</td><td>15.00</td><td>28.00</td><td></td><td>36.00</td><td>18.00</td><td>21.00</td><td>70.0</td><td>18.00</td><td>26.00</td></tr><tr><td>ManipLLM (Li et al. 2024b)</td><td>53.0</td><td>558.00 64.00</td><td>35.00 36.00</td><td>55.00 7700</td><td>28.00</td><td>6.00</td><td>58.00</td><td>51.00</td><td>52.00</td><td>57.00</td><td>45.00</td><td></td><td>34.00</td><td>41.00</td><td>54.00</td><td>39.00</td><td>43.00 64.00</td></tr><tr><td>ManipLLM (All)â </td><td>68.00</td><td>51.57</td><td>46.00</td><td>52.94</td><td>43.00</td><td>6200</td><td>6.00</td><td>61.00</td><td>6.00</td><td>52.0</td><td>3.00</td><td></td><td>40.00</td><td>64.00</td><td>71.00</td><td>600.00</td><td>78.00</td></tr><tr><td>Ours (All)</td><td>60.78 6.67</td><td>51.56</td><td>22.00</td><td>82.27</td><td>37.25 96.08</td><td>33.33 18.73</td><td>63.51 82.27</td><td>62.56 87.12</td><td>46.00 66.00</td><td>43.14 65.85</td><td>25.49 75.62</td><td></td><td>54.00</td><td>42.00 32.00</td><td>65.83 77.21</td><td>22.00 22.00</td><td>96.00</td></tr><tr><td>ManipLLM (Local)â </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>56.32</td><td></td><td></td><td></td><td></td></tr><tr><td>Ours (Local)</td><td>72.73 74.07</td><td>67.89 6.01</td><td>47.83 24.29</td><td>63.41 85.05</td><td>44.96 98.00</td><td>64.00</td><td>75.37</td><td>82.43 100.0</td><td>63.16</td><td></td><td>54.29</td><td>26.00</td><td>84.62</td><td>63.64</td><td>72.07 82.12</td><td>29.04 31.25</td><td>80.80 97.96</td></tr><tr><td>Method</td><td></td><td></td><td></td><td></td><td></td><td>21.86</td><td>100.0</td><td></td><td></td><td>87.24</td><td>72.12</td><td>88.24</td><td>100.0</td><td>71.43</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>Test Seen Split</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Test Unseen Split</td><td></td><td>B</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>*</td><td>8</td><td>$\fra2}$</td><td>R</td><td>AVG</td><td>8</td><td>Ã</td><td></td><td>9</td><td>Â¤</td><td></td><td></td><td></td><td>D</td><td>$\fra}$</td><td></td><td>AVG</td></tr><tr><td>RAM (Kuang et al. 2024) (Zero-shot)â </td><td>32.45</td><td>52.64</td><td>21.61</td><td>38.93</td><td>40.56</td><td>42.38</td><td>25.60</td><td>42.83</td><td>56.07</td><td>30.47</td><td>38.00</td><td></td><td>32.00</td><td>43.06</td><td>18.37</td><td>28.96</td><td>35.77</td></tr><tr><td>CrayonRobo(Li et al. 2025) (Zero-shot) Ours (Zero-shot)</td><td>47.06 48.98</td><td>54.90 76.14</td><td>16.33 15.69</td><td>51.02 67.47</td><td>43.31 49.25</td><td>48.67</td><td>29.92</td><td>31.67</td><td>58.36 68.38</td><td>18.68 228.69</td><td></td><td>24.00</td><td>38.00</td><td>41.18</td><td>15.69</td><td>27.45</td><td>33.36 43.53</td></tr><tr><td>Where2Act (Mo et al. 2021)</td><td></td><td></td><td></td><td></td><td></td><td>60.78</td><td>2549</td><td>82.75</td><td></td><td></td><td></td><td>18.00</td><td>40.00</td><td>64.71</td><td>13.73</td><td>32.76</td><td></td></tr><tr><td>UMPNet (Xu, He, and Song 2022)</td><td>13.00 22.00</td><td>18.00 33.00</td><td>13.00 26.00</td><td>40.00 64.00</td><td>26.00</td><td>18.00</td><td>35.00</td><td>38.00</td><td></td><td>28.00</td><td>5.00</td><td>21.00</td><td>17.00</td><td>20.00</td><td>15.00</td><td>15.00</td><td>21.00 28.00</td></tr><tr><td>Flowbot3D (Eisner, Zhang, and Held 2022)</td><td>17.00</td><td>53.00</td><td>29.00</td><td>42.00</td><td>35.00 37.00</td><td>42.00 23.00</td><td>20.00 10.00</td><td>35.00 60.00</td><td></td><td>42.00 29.00 39.00</td><td>27.00</td><td>20.00 42.00</td><td>26.00 28.00</td><td>28.00</td><td>25.00 13.00</td><td>15.00</td><td>32.00</td></tr><tr><td>Implicit3D (Zhong et al. 2023)</td><td>27.00</td><td>665.00</td><td>20.00</td><td>33.00</td><td>46.00</td><td>45.00</td><td>17.00</td><td>8000</td><td></td><td>53.00</td><td>15.00</td><td>9.00</td><td>41.00</td><td>51.00 31.00</td><td>3000</td><td>23.00 31.0</td><td>41.00</td></tr><tr><td>ManipLLM (Li et al. 2024b)</td><td>41.00</td><td>75.00</td><td>44.00</td><td>667.00</td><td>56.00</td><td>38.00</td><td>22.00</td><td>81.00</td><td></td><td>866.00</td><td>38.00</td><td>85.00</td><td>42.00</td><td>83.00</td><td>26.00</td><td>38.00</td><td>51.00</td></tr><tr><td>ManipLLM (All)+</td><td>24.49</td><td>64.78</td><td>31.37</td><td>58.92</td><td>48.20</td><td>29.41</td><td>17.65</td><td>80.27</td><td></td><td>78.51</td><td>29.03</td><td>72.00</td><td>28.00</td><td>66.67</td><td>15.69</td><td></td><td>45.34</td></tr><tr><td>Ours (All)</td><td>72.92</td><td>83.51</td><td>20.84</td><td>68.06</td><td>62.15</td><td>65.21</td><td></td><td>79.45</td><td></td><td>78.21</td><td>556.09</td><td>42.00</td><td>86.00</td><td></td><td></td><td>36.20</td><td>57.54</td></tr><tr><td>ManipLLM (Local)â </td><td>35.90</td><td>73.36</td><td>53.85</td><td>69.74</td><td>61.25</td><td>41.67</td><td>17.84 19.09</td><td>87.50</td><td></td><td>85.28</td><td>37.34</td><td>85.29</td><td>43.33</td><td>67.84 76.67</td><td>25.65 20.77</td><td>57.13 40.35</td><td>53.73</td></tr><tr></table>

Table 1: Comparisons of our method against baseline methods, â indicates reproduced results (ASR, %).

Despite the strong overall performance, our analysis also identifies challenges. The first stems from the inherent properties of the evaluation dataset itself. For instance, some tasks exhibit low success rates due to severely degraded visual input; certain Door images are reduced to a few feature lines with nearly 80% pixel loss, making robust feature extraction nearly impossible. Similarly, tasks like Faucet (25.65% SR) suffer from ambiguous instructions in the dataset, where the prompt does not distinguish between "lever adjustment" and "spout repositioning." These dataset-level issues present a fundamental challenge to any vision-language-based method. The second challenge, however, reflects a genuine limitation of our current framework: manipulation of objects with minuscule interaction regions. The low SR on tasks requiring sub-centimeter precision, such as Scissors (17.84%) and Pliers (32.00%), indicates that our model still struggles with the fine-grained physical reasoning and exacting grasping point prediction necessary for such millimeter-scale operations.

<table><tr><td>Ablation Method</td><td>Test seen</td><td>Test unseen</td></tr><tr><td>Main Component Ablation</td><td></td><td></td></tr><tr><td>w/o Retrieval (only vision input)</td><td>19.46</td><td>19.01</td></tr><tr><td>w/ Textual Retrieval + CosSim only</td><td>37.05</td><td>34.62</td></tr><tr><td>w/ Textual Retrieval + IMD only</td><td>45.55</td><td>41.27</td></tr><tr><td>w/ Textual Retrieval + CosSim + IMD</td><td>47.34</td><td>42.08</td></tr><tr><td>Full MRAG (w/ 3D Pose Align)</td><td>49.25</td><td>43.53</td></tr><tr><td>Different MLLM Backbones</td><td></td><td></td></tr><tr><td>Qwen2-VL-7B-Instruct (base)</td><td>62.15</td><td>57.54</td></tr><tr><td>LLaMA3.2-11B-Vision-Instruct</td><td>62.93</td><td>58.18</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>65.52</td><td>60.63</td></tr></table>

Table 2: Ablation studies on different components of our MRAG framework (ASR, %).

## Ablation Studies

We conduct a series of ablation studies to evaluate the contributions of different components in the MRAG framework. As shown in Table 2, the first block presents the zero-shot performance of the MLLM without utilizing our proposed framework. We observe that using only visual input without incorporating any additional signals leads to poor performance (19.46% seen / 19.01% unseen). Introducing multimodal retrieval based on either cosine similarity or Instance Matching Distance (IMD) significantly improves the results, and combining both further enhances generalization capability. Building on this, we further incorporate the 3D-Aware Pose Refinement module, which yields the best zero-shot results (49.25% seen / 43.53% unseen), highlighting the crucial role of spatial grounding in improving retrieval performance.

<!-- image-->  
Figure 3: Ablation study on the number of retrieval candidates (Top-n) and rotation samples (K).

We then evaluate the frameworkâs performance with different MLLM backbones in the fine-tuned All setting. As shown in the table, leveraging more advanced models consistently enhances results, demonstrating our frameworkâs scalability. Specifically, upgrading the backbone from the base Qwen2-VL-7B-Instruct to Qwen2.5-VL-7B-Instruct achieves the overall peak performance of 65.52% on the seen split and 60.63% on the unseen split. Per-category success rates are detailed in the Appendix. These results clearly indicate that our RobMRAG architecture effectively capitalizes on the enhanced reasoning and visual understanding capabilities of more powerful MLLMs, establishing a robust and scalable foundation for robotic manipulation.

We conducted controlled ablation studies on the retrieval candidate number (top-n) and rotation sampling count (K) separately on the test seen split, as shown in Figure 3. Experiments reveal: peak success rate of 62.15% occurs at top-n=5, with larger values degrading performance by 0.61â 1.48% due to geometric noise; optimal K=6 achieves the best performance, validating small-angle rotation compensation. Both parameters exhibit an initial increase followed by a decline, demonstrating the necessity to balance search and sampling breadth with geometric consistency. Excessive values reduce success rates and increase computational overhead at the same time.

## Qualitative Results

Figure 4 presents qualitative examples demonstrating the robustness of our three-priority hybrid retrieval strategy. The left example showcases the second-priority fallback: for the target object Scissors, which is absent from our simulation dataset, the system uses dense retrieval to successfully match Pliers as a functionally analogous counterpart for affordance reference. The right example illustrates the thirdpriority case: given a Toilet with no direct or semantically close match in simulation, the strategy expands the search to robotic and Internet datasets to retrieve a valid reference.

Figure 5 illustrates the geometric alignment process for reference poses using 3D-Aware Pose Refinement module. Due to viewpoint discrepancies between the retrieved reference pose and the observed target object, we employ 3D Gaussian Splatting to generate candidate views by applying multiple rotational transformations. The IMD is then recalculated for each candidate, and the pose with the minimal IMD is selected to achieve precise alignment.

<!-- image-->  
Figure 4: Visualization of the three-priority hybrid retrieval strategy: â  sparse retrieval (simulation), â¡ dense retrieval (simulation), and â¢ cross-source retrieval (robotic/Internet).

<!-- image-->  
Figure 5: Visualization of 3D-Aware Pose Refinement.

## Conclusion

We propose a novel robot manipulation framework based on Multimodal Retrieval-Augmented Generation (MRAG). By integrating a multi-source manipulation knowledge base, hierarchical multimodal retrieval, and a 3D-aware pose refinement module, our approach improves the average manipulation success rates by 7.76% over state-of-the-art baselines in a zero-shot setting across various objects. Furthermore, with LoRA fine-tuning, the performance improves by up to 6.54% compared to the baseline. While our method demonstrates strong capabilities in semantic understanding and pose adaptation, challenges remain in achieving millimeterlevel precision and complex spatial reasoning. Future work will focus on enhancing fine-grained physical reasoning and spatial understanding.

## Acknowledgements

This work is supported by the National Natural Science Foundation of China (Grant No. 62176092, 62222602, 62302167, U23A20343, 62476090, 62502159), Natural Science Foundation of Shanghai (Grant No. 25ZR1402135), Shanghai Sailing Program (Grant No. 23YF1410500), Young Elite Scientists Sponsorship Program by CAST (Grant No. YESS20240780), the Chenguang Program of Shanghai Education Development Foundation and Shanghai Municipal Education Commission (Grant No. 23CGA34), Natural Science Foundation of Chongqing (Grant No. CSTB2023NSCQ-JQX0007,CSTB2023NSCQ-MSX0137, CSTB2025NSCQ-GPX0445), Open Project Program of the State Key Laboratory of CAD&CG (Grant No. A2501), Zhejiang University, Open Research Fund of Key Laboratory of Advanced Theory and Application in Statistics and Data Science-MOE, ECNU.

## References

Alayrac, J.-B.; Donahue, J.; Luc, P.; Miech, A.; Barr, I.; Hasson, Y.; Lenc, K.; Mensch, A.; Millican, K.; Reynolds, M.; et al. 2022. Flamingo: a visual language model for few-shot learning. NeurIPS, 35: 23716â23736.

Bahl, S.; Gupta, A.; and Pathak, D. 2022. Human-to-robot imitation in the wild. arXiv preprint arXiv:2207.09450.

Bharadhwaj, H.; Gupta, A.; Kumar, V.; and Tulsiani, S. 2024. Towards generalizable zero-shot manipulation via translating human interaction plans. In ICRA, 6904â6911. IEEE.

Bharadhwaj, H.; Gupta, A.; Tulsiani, S.; and Kumar, V. 2023. Zero-shot robot manipulation from passive human videos. arXiv preprint arXiv:2302.02011.

Bonomo, M.; and Bianco, S. 2025. Visual RAG: Expanding MLLM visual knowledge without fine-tuning. arXiv preprint arXiv:2501.10834.

Chang, M.; Prakash, A.; and Gupta, S. 2023. Look ma, no hands! agent-environment factorization of egocentric videos. NeurIPS, 36: 21466â21486.

Chen, H.; Sun, B.; Zhang, A.; Pollefeys, M.; and Leutenegger, S. 2025. VidBot: Learning Generalizable 3D Actions from In-the-Wild 2D Human Videos for Zero-Shot Robotic Manipulation. In CVPR, 27661â27672.

Chen, W.; Hu, H.; Chen, X.; Verga, P.; and Cohen, W. W. 2022. MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text. In EMNLP, 5558â5570.

Dong, Y.; Wang, S.; Zheng, H.; Chen, J.; Zhang, Z.; and Wang, C. 2024. Advanced RAG Models with Graph Structures: Optimizing Complex Knowledge Reasoning and Text Generation. In ISCEIC, 626â630. IEEE.

Driess, D.; Xia, F.; Sajjadi, M. S.; Lynch, C.; Chowdhery, A.; Wahid, A.; Tompson, J.; Vuong, Q.; Yu, T.; Huang, W.; et al. 2023. Palm-e: An embodied multimodal language model. openreview.

Edge, D.; Trinh, H.; Cheng, N.; Bradley, J.; Chao, A.; Mody, A.; Truitt, S.; Metropolitansky, D.; Ness, R. O.; and Larson, J. 2024. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130.

Eisner, B.; Zhang, H.; and Held, D. 2022. FlowBot3D: Learning 3D Articulation Flow to Manipulate Articulated Objects. In RSS.

Geng, H.; Li, Z.; Geng, Y.; Chen, J.; Dong, H.; and Wang, H. 2023a. Partmanip: Learning cross-category generalizable part manipulation policy from point cloud observations. In CVPR, 2978â2988.

Geng, H.; Xu, H.; Zhao, C.; Xu, C.; Yi, L.; Huang, S.; and Wang, H. 2023b. Gapartnet: Cross-category domaingeneralizable object perception and manipulation via generalizable and actionable parts. In CVPR, 7081â7091.

Geng, Y.; An, B.; Geng, H.; Chen, Y.; Yang, Y.; and Dong, H. 2023c. Rlafford: End-to-end affordance learning for robotic manipulation. In ICRA, 5880â5886. IEEE.

Gong, J.; Xu, J.; Tan, X.; Song, H.; Qu, Y.; Xie, Y.; and Ma, L. 2021a. Omni-supervised point cloud segmentation via gradual receptive field component reasoning. In CVPR, 11673â11682.

Gong, J.; Xu, J.; Tan, X.; Zhou, J.; Qu, Y.; Xie, Y.; and Ma, L. 2021b. Boundary-aware geometric encoding for semantic segmentation of point clouds. In AAAI, volume 35, 1424â 1432.

Grauman, K.; Westbury, A.; Byrne, E.; Chavis, Z.; Furnari, A.; Girdhar, R.; Hamburger, J.; Jiang, H.; Liu, M.; Liu, X.; et al. 2022. Ego4d: Around the world in 3,000 hours of egocentric video. In CVPR, 18995â19012.

Guo, Z.; Xia, L.; Yu, Y.; Ao, T.; and Huang, C. 2024. Lightrag: Simple and fast retrieval-augmented generation. openreview.

Guu, K.; Lee, K.; Tung, Z.; Pasupat, P.; and Chang, M. 2020. Retrieval augmented language model pre-training. In ICML, 3929â3938. PMLR.

Huang, W.; Wang, C.; Zhang, R.; Li, Y.; Wu, J.; and Fei-Fei, L. 2023. Voxposer: Composable 3d value maps for robotic manipulation with language models. In CoRL.

Ju, Y.; Hu, K.; Zhang, G.; Zhang, G.; Jiang, M.; and Xu, H. 2024. Robo-abc: Affordance generalization beyond categories via semantic correspondence for robot manipulation. In ECCV, 222â239. Springer.

Kerbl, B.; Kopanas, G.; LeimkÃ¼hler, T.; and Drettakis, G. 2023. 3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4): 139â1.

Khazatsky, A.; Pertsch, K.; Nair, S.; Balakrishna, A.; Dasari, S.; Karamcheti, S.; Nasiriany, S.; Srirama, M. K.; Chen, L. Y.; Ellis, K.; et al. 2024. Droid: A large-scale in-the-wild robot manipulation dataset. In RSS.

Kim, M. J.; Pertsch, K.; Karamcheti, S.; Xiao, T.; Balakrishna, A.; Nair, S.; Rafailov, R.; Foster, E.; Lam, G.; Sanketi, P.; et al. 2024. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246.

Kuang, Y.; Ye, J.; Geng, H.; Mao, J.; Deng, C.; Guibas, L.; Wang, H.; and Wang, Y. 2024. Ram: Retrieval-based affordance transfer for generalizable zero-shot robotic manipulation. arXiv preprint arXiv:2407.04689.

Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.; Goyal, N.; KÃ¼ttler, H.; Lewis, M.; Yih, W.-t.; RocktÃ¤schel, T.; et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. NeurIPS, 33: 9459â9474.

Li, B.; Ge, Y.; Ge, Y.; Wang, G.; Wang, R.; Zhang, R.; and Shan, Y. 2024a. Seed-bench: Benchmarking multimodal large language models. In CVPR, 13299â13308.

Li, J.; Li, D.; Xiong, C.; and Hoi, S. 2022. Blip: Bootstrapping language-image pre-training for unified visionlanguage understanding and generation. In ICML, 12888â 12900. PMLR.

Li, X.; Xu, J.; Zhang, M.; Liu, J.; Shen, Y.; Ponomarenko, I.; Xu, J.; Heng, L.; Huang, S.; Zhang, S.; et al. 2025. Object-Centric Prompt-Driven Vision-Language-Action Model for Robotic Manipulation. In CVPR, 27638â27648.

Li, X.; Zhang, M.; Geng, Y.; Geng, H.; Long, Y.; Shen, Y.; Zhang, R.; Liu, J.; and Dong, H. 2024b. Manipllm: Embodied multimodal large language model for object-centric robotic manipulation. In CVPR, 18061â18070.

Liu, J.; Liu, M.; Wang, Z.; Lee, L.; Zhou, K.; An, P.; Yang, S.; Zhang, R.; Guo, Y.; and Zhang, S. 2024. Robomamba: Multimodal state space model for efficient robot reasoning and manipulation. arXiv preprint arXiv:2406.04339.

Liu, J.; Meng, S.; Gao, Y.; Mao, S.; Cai, P.; Yan, G.; Chen, Y.; Bian, Z.; Shi, B.; and Wang, D. 2025a. Aligning Vision to Language: Text-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning. arXiv preprint arXiv:2503.12972.

Liu, J.; Tao, Y.; Wang, F.; Li, H.; and Qin, X. 2025b. SiQA: A Large Multi-Modal Question Answering Model for Structured Images Based on RAG. In ICASSP, 1â5. IEEE.

Liu, P.; Liu, X.; Yao, R.; Liu, J.; Meng, S.; Wang, D.; and Ma, J. 2025c. Hm-rag: Hierarchical multi-agent multimodal retrieval augmented generation. In ACM MM, 2781â2790.

Liu, Y.; Liu, Y.; Jiang, C.; Lyu, K.; Wan, W.; Shen, H.; Liang, B.; Fu, Z.; Wang, H.; and Yi, L. 2022. Hoi4d: A 4d egocentric dataset for category-level human-object interaction. In CVPR, 21013â21022.

Luo, H.; Zhai, W.; Zhang, J.; Cao, Y.; and Tao, D. 2022. Learning affordance grounding from exocentric images. In CVPR, 2252â2261.

Ma, Z.-A.; Lan, T.; Tu, R.-C.; Hu, Y.; Huang, H.; and Mao, X.-L. 2024. Multi-modal retrieval augmented multi-modal generation: A benchmark, evaluate metrics and strong baselines. arXiv preprint arXiv:2411.16365.

Mo, K.; Guibas, L. J.; Mukadam, M.; Gupta, A.; and Tulsiani, S. 2021. Where2act: From pixels to actions for articulated 3d objects. In ICCV, 6813â6823.

Nguyen, T.; Gopalan, N.; Patel, R.; Corsaro, M.; Pavlick, E.; and Tellex, S. 2022. Affordance-based robot object retrieval. Autonomous Robots, 46(1): 83â98.

Smith, L.; Dhawan, N.; Zhang, M.; Abbeel, P.; and Levine, S. 2020. Avid: Learning multi-stage tasks via pixel-level translation of human videos. In RSS.

Vuong, Q.; Levine, S.; Walke, H. R.; Pertsch, K.; Singh, A.; Doshi, R.; Xu, C.; Luo, J.; Tan, L.; Shah, D.; et al. 2023. Open x-embodiment: Robotic learning datasets and rt-x models. In CoRL.

Wu, J.; Zhu, J.; Qi, Y.; Chen, J.; Xu, M.; Menolascina, F.; and Grau, V. 2025. Medical graph rag: Evidencebased Medical Large Language Model via Graph Retrieval-Augmented Generation. In ACL, 28443â28467.

Xiang, F.; Qin, Y.; Mo, K.; Xia, Y.; Zhu, H.; Liu, F.; Liu, M.; Jiang, H.; Yuan, Y.; Wang, H.; et al. 2020. Sapien: A simulated part-based interactive environment. In CVPR, 11097â 11107.

Xiang, J.; Lv, Z.; Xu, S.; Deng, Y.; Wang, R.; Zhang, B.; Chen, D.; Tong, X.; and Yang, J. 2025. Structured 3d latents for scalable and versatile 3d generation. In CVPR, 21469â 21480.

Xiong, H.; Li, Q.; Chen, Y.-C.; Bharadhwaj, H.; Sinha, S.; and Garg, A. 2021. Learning by watching: Physical imitation of manipulation skills from human videos. In IROS, 7827â7834. IEEE.

Xu, M.; Xu, Z.; Chi, C.; Veloso, M.; and Song, S. 2023. Xskill: Cross embodiment skill discovery. In CoRL, 3536â 3555. PMLR.

Xu, Z.; He, Z.; and Song, S. 2022. Universal manipulation policy network for articulated objects. IEEE RA-L, 7(2): 2447â2454.

Yu, Q.; Xiao, Z.; Li, B.; Wang, Z.; Chen, C.; and Zhang, W. 2025. MRAMG-Bench: A BeyondText Benchmark for Multimodal Retrieval-Augmented Multimodal Generation. arXiv preprint arXiv:2502.04176.

Yue, Y.; Wang, Y.; Kang, B.; Han, Y.; Wang, S.; Song, S.; Feng, J.; and Huang, G. 2024. Deer-vla: Dynamic inference of multimodal large language models for efficient robot execution. NeurIPS, 37: 56619â56643.

Zhang, K.; Ren, P.; Lin, B.; Lin, J.; Ma, S.; Xu, H.; and Liang, X. 2024. Pivot-r: Primitive-driven waypoint-aware world model for robotic manipulation. NeurIPS, 37: 54105â 54136.

Zhang, R.; Han, J.; Liu, C.; Gao, P.; Zhou, A.; Hu, X.; Yan, S.; Lu, P.; Li, H.; and Qiao, Y. 2023. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199.

Zhen, H.; Qiu, X.; Chen, P.; Yang, J.; Yan, X.; Du, Y.; Hong, Y.; and Gan, C. 2024. 3d-vla: A 3d vision-language-action generative world model. arXiv preprint arXiv:2403.09631.

Zhong, C.; Zheng, Y.; Zheng, Y.; Zhao, H.; Yi, L.; Mu, X.; Wang, L.; Li, P.; Zhou, G.; Yang, C.; et al. 2023. 3d implicit transporter for temporally consistent keypoint discovery. In ICCV, 3869â3880.

Zitkovich, B.; Yu, T.; Xu, S.; Xu, P.; Xiao, T.; Xia, F.; Wu, J.; Wohlhart, P.; Welker, S.; Wahid, A.; et al. 2023. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In CoRL, 2165â2183. PMLR.