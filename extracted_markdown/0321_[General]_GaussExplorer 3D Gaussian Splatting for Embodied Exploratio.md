# GaussExplorer: 3D Gaussian Splatting for Embodied Exploration and Reasoning

Kim Yu-Ji1 Dahye Lee2 Kim Jun-Seong1 GeonU Kim1 Nam Hyeon-Woo1   
Yongjin Kwon3 Yu-Chiang Frank Wang4 Jaesung Choe4 Tae-Hyun Oh2

1POSTECH 2KAIST 3ETRI 4NVIDIA

<!-- image-->  
Figure 1. GaussExplorer aims at the embodied exploration and reasoning based on 3D Gaussian Splatting. Given an input question, we first identify initial viewpoints by searching for relevant 3D Gaussians. These viewpoints are then refined into novel-view images with the VLM-as-Judge mechanism, which evaluates rendered views to maximize visual evidence and decide the final viewpoints. Finally, the final views are processed by a VLM to generate the response. We further extend our framework to support fine-grained 3D localization tasks with complex language queries such as 3D referring segmentation.

## Abstract

We present GaussExplorer, a framework for embodied exploration and reasoning built on 3D Gaussian Splatting (3DGS). While prior approaches to language-embedded 3DGS have made meaningful progress in aligning simple text queries with Gaussian embeddings, they are generally optimized for relatively simple queries and struggle to interpret more complex, compositional language queries. Alternative studies based on object-centric RGB-D structured memories provide spatial grounding but are constrained by pre-fixed viewpoints. To address these issues, GaussExplorer introduces Vision-Language Models (VLMs) on top of 3DGS to enable question-driven exploration and reasoning within 3D scenes. We first identify pre-captured images that are most correlated with the query question, and subsequently adjust them into novel viewpoints to more accurately capture visual information for better reasoning by VLMs. Experiments

show that ours outperforms existing methods on several benchmarks, demonstrating the effectiveness of integrating VLM-based reasoning with 3DGS for embodied tasks.

## 1. Introduction

Developing embodied agents that can interpret naturallanguage instructions and operate intelligently within 3D environments has been a long-standing goal in computer vision and robotics. Embodied exploration and reasoning in 3D scenes require the ability to jointly understand both the semantic content of the queries and the geometric structure of their surroundings.

Recent advancements have seen the development of Vision-Language Models (VLMs) [10, 18], which typically integrate a powerful visual encoder with a Large Language Model (LLM). These VLMs have demonstrated remarkable emergent reasoning capabilities and vision-language understanding. As these models have proved their proficiency in interpreting static scenes, the research focus naturally shifted towards applying their reasoning abilities to more dynamic and embodied tasks. A central challenge in this transition has been equipping VLMs with an understanding of 3D space and a memory of their observations over time.

Previous works [1, 4, 8, 19, 34] demonstrate that VLMs can perform complex reasoning across multi-view images when provided with structured memory built from RGB-D observations. While these approaches make significant progress toward memory-based embodied reasoning, they typically operate on pre-captured image sequences and object-level abstractions, which naturally constrain the exploration in 3D environments. As a result, they are not designed to synthesize novel-view observations beyond the recorded viewpoints, limiting their capability to support fine-grained spatial reasoning. To this end, despite their strong reasoning capabilities, 3D-Mem [34] remains inadequate for tasks that demand accurate exploration from the 3D primitives.

Recent advances in 3D Gaussian Splatting (3DGS) [13] have established a promising foundation for language-driven 3D scene understanding [9, 12, 23, 29, 33], enabling openvocabulary 3D scene understanding. While these approaches successfully demonstrate the effectiveness of embeddingbased alignment between language and 3D representations, their similarity computation strategies are primarily optimized for simple, category-level text queries (e.g., âchairâ or âtableâ), and extending them to support the complex reasoning and compositional semantics required for embodied understanding remains challenging.

To address these challenges, we introduce GaussExplorer, a unified framework that performs VLM-guided embodied exploration and reasoning directly on 3DGS, as shown in Fig. 1. Given a natural-language query, GaussExplorer (1) selects view candidates by matching simplified queries to 3D Gaussians, then (2) conducts VLM-driven novel-view adjustment. Additionally, given the final views, our model can finely localize the relevant Gaussians in 3D space to perform 3D referring segmentation. By doing so, our pipeline can enhance the VLM-based reasoning and exploration through the novel-view synthesis by 3D Gaussians. The contributions of our paper are summarized as follows:

â¢ Introduce VLM-guided embodied reasoning on 3D Gaussian Splatting (3DGS).

â¢ Demonstrate the necessity of novel-view adjustment for the better VLM-based reasoning task, overcoming the limitations of fixed-view and memory-based approaches.

â¢ Support precise visual grounding and scene-scale 3D object localization, demonstrating strong performance.

## 2. Related Work

Exploration and Reasoning with VLMs. To perform embodied reasoning tasks [1, 4, 19], recent systems are composed of a pair of Vision Language Models (VLMs) with a memory module built from RGB or RGB-D [8, 17, 18, 34] observations. A text-centric method generates captions [2] from RGB data, which are then fed to Large Language Models (LLMs) [21, 30] that reason without direct visual context. A structured abstraction method builds scene graphs from RGB-D data [8, 15, 27, 36, 38], where objects are represented as nodes and their relationships are represented as edges. A significant limitation of both approaches, however, is their reliance on high-level abstraction. By discarding low-level geometric and visual details, these methods hinder the fine-grained understanding required for complex tasks.

While the development of VLMs has enabled the use of visually rich, multi-view images, this introduces a new challenge: long visual sequences often overwhelm models and hinder effective reasoning [7, 28, 32]. 3D-Mem [34] offers a solution by compressing these observations into âMemory Snapshots.â This method removes visual redundancy and stores object-level annotations. Consequently, VLMs can access only the most relevant snapshots for a given query, allowing them to reason efficiently while maintaining the visual information needed for complex question answering and spatial reasoning. A primary bottleneck of these memorybased systems is their reliance on pre-captured, static views. While they may exhibit strong semantic reasoning, they fall short on tasks that demand 3D localization, novel-view exploration, or precise spatial grounding.

Language-driven 3D Gaussian Splatting. 3D Gaussian Splatting (3DGS) [13] is a modern 3D representation that enables fast, high-fidelity 3D reconstruction and real-time rendering. It has evolved beyond rendering, as several works [9, 11, 12, 23, 29, 33, 35, 39] have augmented it with language features for open-vocabulary reasoning. For example, prior approaches [12, 23, 29, 33] have embedded CLIP features [24] into each Gaussian, enabling tasks such as open-vocabulary object retrieval and segmentation.

The core challenge with existing embedding-augmented 3DGS approaches is that they primarily rely on static embedding similarity. While this works with simple, category-level queries, it struggles to capture richer linguistic reasoning, such as compositional instructions. For example, given the query âWhat color are pillows in the kitchen?â, such methods may retrieve all pillows in the scene, rather than those specifically associated with the kitchen. This behavior arises because their visual reasoning is carried out at the level of individual Gaussians, without explicit mechanisms for iterative grounding or contextual aggregation.

More recently, ReferSplat [9] aligns fine-grained textual descriptions with Gaussian features, leading to a deeper understanding of spatial relationships. However, because its feature updates are rendering-based, it remains less suitable for direct 3D search [12].

By coupling VLM reasoning [21, 22] with a modern 3D representation, specifically 3D Gaussian Splatting [13], our method enables direct search and grounding in 3D space. Our approach also allows agents to actively synthesize novel viewpoints, disambiguate queries, and ground targets with high spatial accuracy through advanced VLM reasoning.

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 2. Overview. (a) We first build a semantic 3DGS scene, where input views I and their semantic information produced by foundation models are lifted into 3D. (b) In initial view selection, (b-1) the query is first rephrased into relevant semantic categories Cevidence by LLM, (b-2) activated 3D Gaussians associated with those categories are grouped into spatial clusters, and (b-3) representative training camera poses covering these clusters are selected. (c) Finally, the selected views are rendered and passed to a Vision-Language Model (VLM) for fine-grained reasoning, which identifies the most informative views and produces the final answer to the query.

## 3. Method

Given a question about a scene and its corresponding multiview images, our goal is to identify the most relevant camera viewpoints from the reconstructed 3D Gaussians, thereby enabling embodied exploration and reasoning. We begin by constructing semantic Gaussians that embed category or feature embedding information (Sec. 3.1). Next, we perform a direct search in 3D space using the user query to determine the optimal viewpoints among the available cameras for answering the question (Sec. 3.2). Based on these viewpoints, we render novel candidate views in their local neighborhoods and interact with modern Vision-Language Models (VLMs) [21, 22] leveraging the differentiable rendering capabilities of 3D Gaussian Splatting (3DGS) [13] (Sec. 3.3). The overall pipeline is illustrated in Fig. 2. Note that N input images are associated with camera parameters $\mathcal { H } = \{ H _ { i } \} _ { i = 1 } ^ { N }$ where $H = [ \mathbf { R } | \mathbf { t } ] \in \mathbb { R } ^ { 3 \times 4 }$ is the extrinsic camera matrix.

## 3.1. Construction of Semantic 3D Gaussians

In this section, we explain a way of building semantic 3D Gaussians following either closed-set categories [34] or open-set semantic embeddings [12]. First, we reconstruct a 3D scene from N input images $\mathcal { T } = \{ I _ { i } \} _ { i = 1 } ^ { N }$ with 3D Gaussians [13]. The 3D scene is represented as a set of M 3D Gaussians, $\mathcal { G } = \{ G _ { j } \} _ { j = 1 } ^ { M }$ . Second, we extract semantic information from images. Depending on the tasks, we follow two different strategies for a fair comparison with competing methods. For the embodied reasoning task, we use a closedset object detector [31]. Each Gaussian $G _ { j }$ along with its semantic information $C _ { j } { \in } C$ is then lifted into 3DGS representation following [34]. For the visual grounding and 3D localization task, we use the open-set segmentation pipelines combined with CLIP [24] and SAM [16]. Each Gaussian $G _ { j }$ along with its semantic CLIP embedding $f _ { j }$ , is then lifted into 3DGS representation following [12]. Lastly, we leverage 3D direct registration approach [12] to infuse semantic object category information into the 3D Gaussians.

## 3.2. Initial View Selection in Semantic Gaussians

We leverage the semantic information encoded in the 3D Gaussians to select views from the pre-recorded images that can be used for answering the given question. Specifically, during this stage, we extract âevidence categoriesâ relevant to the question, identify their corresponding 3D Gaussian clusters (i.e., 3D instances), and select the view per cluster where these instances exhibit the highest visibility.

Gaussian search from query. To query relevant Gaussians, we first use Large-Language Models (LLMs) to extract âevidence categoriesâ from the query question. For example:

<!-- image-->  
Figure 3. Visibility score. We evaluate visibility by checking whether Gaussians with the highest rendering weight belong to the target instance cluster. If they align, the instance is considered visible; if occluded, the selected Gaussians originate from other regions, resulting in no overlap and a low visibility score.

Prompt: You should retrieve helpful objects in order.   
Question: Where can I take a nap?   
Categories (C): radiator, cushion, sink, pillow, picture, ...   
Evidence categories (C selected): pillow, cushion.

These evidence categories (e.g., pillow, cushion) are then used as a query for searching Gaussians. For the embodied reasoning tasks, since each Gaussian $G _ { j }$ has a parameter of categorical information $C _ { j } ,$ , we can directly retrieve all 3D Gaussians whose categories match the evidence categories by:

$$
\mathcal { G } ^ { \mathrm { a c t } } = \{ G _ { j } \ | \ C _ { j } \in \mathcal { C } ^ { \mathrm { s e l e c t e d } } \} ,\tag{1}
$$

where ${ \mathcal { G } } ^ { \mathrm { a c t } }$ is a set of the activated 3D Gaussians conditioned to the selected category Cselected.

For the visual localization tasks, each Gaussian $G _ { j }$ is evaluated by its similarity to the evidence categories:

$$
\mathcal { G } ^ { \mathrm { a c t } } = \left\{ G _ { j } \left. \sum _ { c \in \mathcal { C } ^ { \mathrm { s e l e c t e d } } } \mathrm { s i m } ( f _ { j } , t _ { c } ) \geq \tau \right. \right\} ,\tag{2}
$$

where $f _ { j }$ denotes the feature embedding of $G _ { j } , t _ { c }$ is the text embedding of category c, sim(Â·) is the similarity function, and Ï is the threshold for activating 3D Gaussians.

Clustering 3D Gaussians for instance assignment. A single Gaussian does not represent an entire object. Therefore, we classify these Gaussians to group them into their respective object instances. We assume that close 3D Gaussians with the same category $C _ { j }$ represent the same object instance. We apply the clustering algorithm to activate 3D Gaussians based on HDBSCAN [5, 20] with an additional cluster merge operation:

$$
\{ \mathcal { G } _ { l } ^ { \mathrm { c l u s t e r } } \} _ { l = 1 } ^ { L } = \mathrm { C l u s t e r } ( \mathcal { G } ^ { \mathrm { a c t } } ) ,\tag{3}
$$

<!-- image-->  
Figure 4. EQA pipeline comparison. The competing method [34] attempts to answer queries directly using initial views selected from given images, lacking any viewpoint refinement. In contrast, our method automatically explores nearby novel viewpoints via VLMbased pose adjustment, maximizing visual evidence more clearly. Finally, a VLM-based verification step compares the initial and refined views to select the one providing the most reliable evidence. Note that A represents the predicted answer to the given question.

where ${ \mathcal { G } } ^ { \mathrm { c l u s t e r } }$ is a resulting set of Gaussian clusters. The number of clusters L is dynamically determined by merging physically close clusters.

Initial view selection. We aim to find L initial camera views ${ \mathcal { H } } ^ { \mathrm { i n i t } } { = } \{ H _ { l } \} _ { l = 1 } ^ { L }$ , where each view corresponds to one of the L target instances $\{ \mathcal { G } _ { l } ^ { \mathrm { c l u s t e r } } \} _ { l = 1 } ^ { L }$ . These initial views are selected from the available camera sets $\mathcal { H } { = } \{ H _ { i } \} _ { i { = } 1 } ^ { N }$ satisfying $\mathcal { H } ^ { \mathrm { i n i t } } \subset \mathcal { H }$ . To answer questions related to target instances, the corresponding instances must be visible.

To find visible Gaussians, we first render an activation map from Gcluster with camera view Hi based on the splatting renderer [13, 40]. We then acquire a binary visible mask $M _ { i } ^ { \mathrm { { \bar { v i s i b l e } } } }$ by thresholding the activation map. With the visible mask, we identify the visible Gaussians with:

$$
\mathcal { G } _ { l } ^ { \mathrm { c l u s t e r , v i s } } = \bigcup _ { \mathbf { u } _ { i } \in M _ { i } ^ { \mathrm { v i s i b l e } } } \underset { j \in \mathcal { G } ^ { \mathrm { c l u s t e r } } } { \arg \operatorname* { m a x } } \bigg ( w _ { j } ( I _ { i } , \mathbf { u } _ { i } ; G _ { j } ) \bigg ) ,\tag{4}
$$

by selecting Gaussians with the hightest weight $w _ { j }$ at each mask pixels $\mathbf { u } _ { i }$ and aggregating them. We then use visibility as our scoring metric, defined as follows:

$$
V _ { l , i } = \frac { | \mathcal { G } _ { l } ^ { \mathrm { c l u s t e r , v i s } } | _ { i } } { | \mathcal { G } _ { l } ^ { \mathrm { c l u s t e r } } | } ,\tag{5}
$$

where l is the target instance, i is the index of a training camera view, | Â· | counts the number of Gaussians. This score function is used to compute a score matrix $\mathbf { V } \in \mathbb { R } ^ { L \times N }$ where N is the total number of training views. Finally, we apply an argmax operation along the second axis (i.e., over the N views) to find the initial camera views for each target instance. As shown in Fig. 3, this pipeline is to become robust to occlusion by leveraging the Gaussian weight w.

<!-- image-->  
Figure 5. Novel-view adjustment. From an initial camera pose, we generate multiple novel-view candidates and refine them through a novel view adjustment module. Each candidateâs view is rendered and evaluated by a VLM with a visual-QA prompt to obtain initial answer predictions. These answers, together with the question, are then passed to an LLM to select the most informative final viewpoint.

## 3.3. Novel-view Adjustment using VLMs

Unlike previous study [34] that is constrained by prerecorded images to perform visual reasoning, our model introduces a novel-view synthesis into the visual reasoning pipeline as illustrated in Fig. 4. This pipeline is effective when the initial camera views are suboptimal due to scene ambiguities such as occlusions or complex object relationships. We therefore adjust the initial viewpoints via novel-view rendering. In the adjustment stage, we generate perturbed initial views and employ a VLM-as-Judge module to assess their informativeness, ultimately selecting more suitable viewpoints for the given questions.

For each L initial camera views ${ \mathcal { H } } ^ { \mathrm { i n i t } } = \{ H _ { l } \} _ { l = 1 } ^ { L }$ , we generate perturbed novel views $\mathcal { H } ^ { \mathrm { n o v e l } } = \{ H _ { k } \} _ { k = 1 } ^ { K }$ with $K = L \times 4$ by shifting the camera left or right, or applying zoom-in/-out around the camera center. Each pose is rendered into an RGB image via 3DGS, producing a small batch of novel viewpoints.

VLM-as-Judge for estimating view informativeness. Because it is difficult to determine whether the novel views are informative solely from the visibility score, we incorporate VLMs to guide final view selection in 3D space. As illustrated in Fig. 4, VLMs serve as judges for both novel-view adjustments and verification. For the novel-view adjustment stage, we first generate candidate final views ${ \mathcal { H } } ^ { \mathrm { c a n d i } } = \{ H _ { l } , H _ { k } , . . . , H _ { k + 3 } \}$ as shown in Fig. 5. Each candidate view is then evaluated independently by feeding its rendered image and the question to VLMs, because VLMs tend to struggle with long visual sequences [7, 28, 32]. From these per-view predictions, we prompt the LLM again to select the best answer, which implicitly determines the corresponding best final view. We perform answer reasoning using the selected final camera views ${ \mathcal { H } } ^ { \mathrm { f i n a l } } = \{ H _ { f } \} _ { f = 1 } ^ { L }$ followed by an additional verification stage to determine whether selected views are genuinely necessary. In the end, we obtain a verified answer to the userâs question using the best final camera views, enabling both reliable rendering and question-based 3D referring segmentation for fine-grained 3D localization.

## 4. Experiments

GaussExplorer consists of 3D Gaussian Splatting (3DGS) and Vision-Language Models (VLMs), enabling selection of final novel viewpoints that directly correspond to the userâs query. The final viewpoints highlight the specific regions relevant to the userâs question, enabling embodied behavior such as embodied question-answering in Sec. 4.1 or questionbased 3D referring segmentation Sec. 4.2. The necessity of our pipeline is shown by ablation studies in Sec. 4.3.

Dataset. We evaluate GaussExplorer on two datasets that assess its capabilities in embodied reasoning and 3D scene understanding. To measure embodied reasoning capability, we use the evaluation dataset released by OpenEQA [19]. Specifically, we focus on the Episodic-Memory Embodied Question Answering (EM-EQA) task, which reflects practical scenarios, e.g., smart glasses that rely on the history of past observations to assist users. OpenEQA question-answer pairs are constructed from Habitat-Matterport 3D [25] and ScanNet [3, 26], covering seven types of questions with over 1,600 questions in about 180 environments.

In addition, we introduce question-based 3D referring segmentation for fine-grained 3D object localization, which aims to identify the objects relevant to a user query among multiple instances in the scene. We manually curate a benchmark by selecting five representative scenes from ScanNet and annotating 24 distinct objects with 49 human-written questions that cover various spatial and attribute-based relationships (e.g., âWhich bed is closest to the window?â). Importantly, each scene contains multiple instances from the same object category, and questions are deliberately designed to distinguish among these instances using spatial cues or attributes such as attribute or relative position. This design enables fine-grained evaluation of 3D localization performance, assessing how well Gaussian-based representations capture spatially grounded scene understanding.

Question : Which bed is closest to the window? Evidence category: bed  
<!-- image-->  
Figure 6. Qualitative results of the 3D referring segmentation. Compared to the competing methods, ours more accurately identifies the fine-grained target locations referenced by the question. Dr. Splat is designed to find objects without distinguishing instances, which limits its ability to localize fine-grained target regions (category) and, consequently, makes it struggle to directly interpret complex language queries (question). ReferSplat is designed to respond to handle complex language queries, but is rendering-baesd optimization is not compatible with direct 3D search, leading to limited 3D localization capability.

<table><tr><td>Method</td><td>LLM-Match (â)</td><td>Average Frames</td></tr><tr><td>BlindLLM [21]</td><td>34.8</td><td>0</td></tr><tr><td>Frame Captions</td><td>24.1</td><td>0</td></tr><tr><td>CG Captions* [8]</td><td>36.5</td><td>0</td></tr><tr><td>SVM Captions* [19]</td><td>38.9</td><td>0</td></tr><tr><td>Multi-Frame</td><td>49.1</td><td>3.0</td></tr><tr><td>3D-Mem [34]</td><td>54.6</td><td>2.7</td></tr><tr><td>GaussExplorer (ours)</td><td>57.8</td><td>2.6â </td></tr><tr><td>Human</td><td>86.8</td><td>Full</td></tr></table>

Table 1. Evaluation on EM-EQA. We measure the semantic equivalent using LLM-Match and frame efficiency using the average number of frames. Methods marked with \* and Human score are reported from OpenEQA, while all other results are reproduced.

## 4.1. Embodied Question Answering (EQA)

Comparisons. We compare our method with six types of agents. BlindLLM [21] infers their answer without visual input and only with the user question. Frame Captions uses LLaVA-v1.5 [18] to caption 50 sampled frames, feeding captions and the question to GPT-4 [21]. CG Captions and SVM Captions construct textual scene graphs from episodic memory using ConceptGraphs [8] and Sparse Voxel Map [19], respectively. Multi-Frame directly provides three frames through unified sampling along with the question. 3D-Mem [34] leverages object-centric multi-view Memory Snapshots to preserve spatial and semantic consistency while reducing redundant frames. Ours utilizes semantic 3DGS as an episodic memory integrated with VLM interaction, enabling a compact, informative, and effective representation. We reproduce all results except for CG and SVM Captions, for which the code and prompts are not publicly available. For the 3D-Mem evaluation, we follow the extracted snapshots provided by the 3D-Mem authors, which contain a small omission (about 1%), and therefore evaluate 1,623 questions for all reproductions. As 3D-Mem constructs its memory based on closed-set categories, we also use closedset semantic categories in our method for a fair comparison.

<table><tr><td>Method</td><td>3D mIoU (â)</td><td>Acc@5 (â)</td><td>Acc@8 (â)</td><td>Acc@10 (â)</td></tr><tr><td>Dr. Splat (category) [12] Dr. Splat (question) [12]</td><td>10.03</td><td>55.14 56.86</td><td>41.71</td><td>33.86</td></tr><tr><td>ReferSplat [9]</td><td>10.56 2.34</td><td>14.29</td><td>45.21 2.04</td><td>43.21 0.00</td></tr><tr><td>Ours</td><td>12.46</td><td>52.00</td><td>45.14</td><td></td></tr><tr><td>Ours + novel-view</td><td>12.87</td><td>52.57</td><td>45.71</td><td>45.14 45.71</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 2. 3D referring segmentation results. We evaluate recent studies, such as Dr. Splat, using both category-level and questionbased queries. ReferSplat is tested with question-based queries, as it is specifically designed to interpret detailed textual descriptions. Our method follows the pipeline of GaussExplorer.

Settings. We follow the EM-EQA setting proposed in OpenEQA [19], where the agent answers a user question solely from past visual observations without further exploration. Each agent receives episodic memories or sampled frames as input and generates an open-ended textual answer through its corresponding LLMs. Performance is evaluated using the LLM-Match metric [19], which measures semantic agreement between the predicted and the groundtruth answers by prompting a strong language model [21] to judge whether the two answers are semantically equivalent. This metric better captures conceptual correctness than exact string matching, providing a more reliable assessment of reasoning and grounding performance in embodied settings.

Question: Where can I sit and eat if I donât want to use the dining table? Ground truth: Use the kitchen bar counter  
<!-- image-->  
Pred: âYou coud sit and eat on the chair by the window in the bedroom.â

<!-- image-->  
Pred: âYou can sit and eat at the kitchen counter, which has bar stools.â

Question: What shape are the decorations above the bed? Ground truth: Triangle  
<!-- image-->  
Pred: âThe decoration above the bed are small wall-mounted shelves.â 3D-Mem

<!-- image-->  
Ours

Figure 7. Initial view selection. Compared to 3D-Mem, our visibility-based policy can select more informative views by accounting for actual object visibility, rather than relying solely on category presence.  
<!-- image-->

Question: What is the blue object left of the downstars bed? Ground truth : A humidifier.  
<!-- image-->

<!-- image-->  
Initial View Selection  
Final View Selection  
Figure 8. Final view selection. We apply novel view synthesis to refine initial selected views, leveraging a unique capability of 3DGS. Our results show that even when the initial views lack sufficient information to answer the question, the novel-view adjustment process can recover the necessary viewpoints, enabling more accurate question-aligned answers.

Results. Table 1 reports the LLM-Match performance and average frames across different baselines. BlindLLM and Captions achieve relatively low scores of 34.8 and 24.1, respectively, indicating that relying solely on textual information is insufficient for embodied reasoning. Incorporating visual signals significantly boosts performance: Multi-Frame reaches 49.1 LLM-Match, highlighting the importance of visual context for understanding and reasoning. 3D-Mem further improves performance through its compact yet informative Snapshot Memory architecture. Finally, GaussExplorer achieves the highest performance while maintaining a similar average number of frames, validating the effectiveness and efficiency of our semantic 3DGS representation and novel-view adjustment approach.

## 4.2. Question-based 3D Referring Segmentation

Competing methods. We compare our method against recent language-guided 3D Gaussian Splatting approaches: Dr. Splat [12] and ReferSplat [9]. Dr. Splat lifts multi-view CLIP [24] image embeddings into the 3D Gaussian space for language-based querying. We optimize the model following the original paperâs settings. ReferSplat [9] optimizes 3D Gaussians using fine-grained textual expressions for referring understanding. As ReferSplat relies on paired textual annotations for supervision but lacks an official automatic data generation pipeline, we manually construct the required training data by generating textual descriptions for object masks with foundation models [16, 37].

Settings. We measure the 3D mean Intersection over Union (3D mIoU) between the activated Gaussians and the groundtruth Gaussians. Since the scene is represented by Gaussians, we compute the mIoU based on the volumetric overlap induced by the Gaussian scales and opacities, following [12]. Because no fine-grained object localization ground truth exists, we manually annotate the specific target instances and construct question sets. The ground-truth 3D Gaussian labels are generated by computing the Mahalanobis distance between the ground-truth point clouds from the dateset [3] and 3D Gaussians, following the procedure in [12]. Given a question, each model identifies the corresponding activations over all 3D Gaussians in the scene. A higher mIoU indicates better localization accuracy, meaning that the activated Gaussians more precisely align with the target region referenced by the question. For details of our dataset construction, please refer to the supplementary material.

<table><tr><td>Method</td><td>LLM-Match (â)</td><td>Average Frames</td></tr><tr><td>3D-Mem [34]</td><td>45.4</td><td>3.1</td></tr><tr><td>Volume score</td><td>47.8</td><td>2.7</td></tr><tr><td>Visibility score (ours)</td><td>48.2</td><td>2.7</td></tr></table>

Table 3. Ablation on initial view selection strategy (184 questions). Because 3DGS is volumetric, we also compute volumetric visibility scores in addition to the non-volumetric version. However, visibility scores computed directly from the volume may be influenced by random variations in Gaussian sizes rather than true surface visibility, which could explain their lower performance compared to the non-volumetric variant.

<table><tr><td>Method</td><td>LLM-Match (â)</td><td>Average Frames</td></tr><tr><td>3D-Mem [34]</td><td>45.4</td><td>3.1</td></tr><tr><td>Initial view selection</td><td>48.2</td><td>2.7</td></tr><tr><td>Final view selection (ours)</td><td>50.5</td><td>2.7â </td></tr></table>

Table 4. Ablation study on view selection (184 questions). We demonstrate the necessity of each component in our method. Please note that â  indicates the use of novel views during the view adjustment stage, but the number of input frames fed into VLMs remains the same as reported average frames.

Results. Table 2 reports 3D mIoU and Acc@k scores. Acc@k denotes the percentage of predictions whose IoU with the ground truth exceeds k. We normalize the computed Gaussian volumes by the 90% value and clip them range between 0 and 1, which helps suppress excessive floating Gaussians that appear in vanilla 3DGS [6]. While Refer-Splat is designed to interpret detailed textual descriptions, its rendering-based optimization is not well aligned with direct 3D search, leading to lower mIoU. In contrast, GaussExplorer more reliably identifies the question-related 3D regions thanks to the combination of directly registered semantic 3DGS and VLM-based reasoning.

In Figure 6, we observe that our method aligns more closely with the ground truth, demonstrating stronger finegrained localization capability. This capability is particularly important for embodied exploration and navigation tasks, where the system must precisely reflect user queries.

## 4.3. Ablation Study

We use 184 questions for initial design choices, and evaluate final view selection on the full set of 1,623 questions.

Score function of initial view selection. We compare two scoring functions for selecting initial camera views: a vol-

<table><tr><td>Method</td><td>LLM-Match (â)</td><td>Average Frames</td></tr><tr><td>3D-Mem [34]</td><td>54.6</td><td>2.7</td></tr><tr><td>Visual + textual prompt</td><td>56.2</td><td>2.6â </td></tr><tr><td>Visual prompt only (ours)</td><td>57.8</td><td>2.6â </td></tr></table>

Table 5. Ablation study on final reasoning pipeline (1,623 questions). Comparison of different prompting strategies for the laststage VLM reasoning. Our visual prompt only design achieves the highest accuracy.

<table><tr><td>Method</td><td>LLM-Match (â)</td><td>Average Frames</td></tr><tr><td>Final view</td><td>54.5</td><td>2.6â </td></tr><tr><td>Final view + verification (ours)</td><td>57.8</td><td>2.6â </td></tr></table>

Table 6. Ablation study on verification process (1,623 questions). Using only initial or final views yields limited performance, whereas incorporating the full verification step significantly improves LLM-Match accuracy with no increase in rendering cost.

ume score, computed by replacing the Gaussian count with

$$
\begin{array} { r } { \mathrm { V o l u m e } ( G _ { j } ) = s _ { j } ^ { x } s _ { j } ^ { y } s _ { j } ^ { z } \alpha _ { j } , } \end{array}\tag{6}
$$

where s denotes the scale values of each axis and Î± indicates the opacity of each Gaussian. The volume-based formulation performs worse because large outlier Gaussians can disproportionately dominate the score, introducing strong bias and resulting in unreliable view selection. In contrast, the visibility score effectively acts as a proxy for object surface coverage, independent of Gaussian scale or training dynamics. This yields more stable visibility estimates and produces noticeably better LLM-Match performance, as shown in Table 3.

View selection pipeline. As shown in Table 4, the initial view selection alone already improves performance over 3D-Mem, demonstrating the effectiveness of our visibility score. Figure 7 presents snapshots of 3D-Mem and our initial view selection. We observe that the visibility scores successfully guide the selection of more informative views. With the final view selection, integrated with novel views and the verification stage, performance further increases, indicating that 3DGS helps uncover more informative viewpoints. As shown in Fig. 8, final views better capture evidence relevant to the question, which in turn supports VLMs in generating more accurate answers.

Final VLM reasoning. We evaluate different prompting strategies for the final-stage VLM reasoning. Mixed visualtextual prompt gains over the 3D-Mem, but our visual-only prompting achieves the highest accuracy as shown in Table 5. This suggests that eliminating auxiliary textual cues helps the VLM rely more directly on visual evidence.

Verification process. Using only the final views yields limited performances compared to the views after the verification stage, as shwon in Table 6. When combined with our verification step, accuracy improves without increasing rendering cost, demonstrating that comparing responses across views is crucial for selecting the most reliable evidence.

## 5. Conclusion

We introduced GaussExplorer, a unified framework that integrates 3D Gaussian Splatting with VisionâLanguage Models to enable embodied exploration and visual reasoning. By leveraging semantic 3D Gaussians as a compact episodic memory and employing a visibility-based camera selection strategy, our method identifies informative viewpoints that align with user queries. Furthermore, GaussExplorer exploits the inherent novel-view synthesis capability of 3DGS to refine initial viewpoints, allowing the system to overcome occlusions, resolve ambiguities, and reveal spatial details that fixed-view memory systems have struggled to capture.

# GaussExplorer: 3D Gaussian Splatting for Embodied Exploration and Reasoning

Supplementary Material

A. Implementation Details

B. Experimental Setup

C. Benchmark Dataset Curation

D. Additional Results

E. Discussion and Limitation

## Supplementary Material

In this supplementary material, we provide details and additional results that could not be included in the main paper due to space constraints. We present implementation and experimental details in Sec. A and Sec. B. Our novel curated benchmark for question-based 3D referring segmentation is described in Sec. C. Additional experimental results are provided in Sec. D. Finally, limitations and future directions are discussed in Sec. E.

## A. Implementation Details

## A.1. Semantic Gaussian Construction

Reconstruction with 3DGS. We optimize 3D Gaussain Splatting (3DGS) following the original paper [13], and apply depth regularization [14] to further improve reconstruction quality. Each Gaussian $G _ { j }$ is parameterized by:

$\mu _ { j } = [ x , y , z ] ^ { \top }$ : the center position.

$S _ { j } \in \{ s _ { j } ^ { x } , s _ { j } ^ { y } , s _ { j } ^ { z } \} , R _ { j }$ : scaling and rotation components that define the 3D covariance matrix $\Sigma _ { j }$

$\alpha _ { j } \in \mathbb { R } \colon$ the opacity.

$\mathbf { s p h } _ { j } \in \mathbb { R } ^ { ( d + 1 ) ^ { 2 } \times 3 }$ : the spherical harmonics (SH) coefficients of degree d.

$C _ { j }$ : the category.

$w _ { j } = T _ { j } \cdot \tilde { \alpha } _ { j } \mathrm { : }$ the weight computed by transmittance $T _ { j }$ and effective opacity $\tilde { \alpha } _ { j }$

The weight $w _ { j }$ is used in the visibility score function for initial view selection. These Gaussians $\mathcal { G }$ are rendered to a 2D pixel color c as:

$$
\mathbf { c } ( \mathbf { u } ) = \sum _ { j \in M _ { r } } T _ { j } \tilde { \alpha } _ { j } \mathbf { c } _ { j } , \mathrm { s . t . } \tilde { \alpha } _ { j } = \sigma _ { j } G _ { j } ^ { 2 D } ( \mathbf { u } ) ,\tag{7}
$$

where $T _ { j }$ is a transmittance, ${ \tilde { \alpha } } _ { j }$ is an accumulated opacity, the color $\mathbf { c } _ { j }$ is derived from the spherical harmonics coefficients sphj given the viewing direction of the camera, $G _ { j } ^ { 2 D }$ is the 2D Gaussian function at a pixel location u by splatting the Gaussian parameters [13, 40], the pixel location u is computed from camera parameters $\mathcal { H } = \{ K _ { i } , [ R _ { i } | \mathbf { t } _ { i } ] \} _ { i = } ^ { N }$ 1 where $K \in \mathbb { R } ^ { 3 }$ is an intrinsic matrix and $[ R | \mathbf { t } ] \in \mathbb { R } ^ { 3 \times 4 }$ is an extrinsic matrix. The N input images are associated with camera parameters $\mathcal { H } = \{ \bar { \mathbf { K } } _ { i } , [ \mathbf { R } _ { i } | \bar { \mathbf { t } } _ { i } ] \} _ { i = 1 } ^ { N }$ , where $\mathbf { K } _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ is the intrinsic matrix and $[ \mathbf { R } _ { i } | \mathbf { t } _ { i } ] \in \mathbb { R } ^ { 3 \times 4 }$ is the extrinsic matrix. These rendered 2D images are then subsequently fed to VLMs for embodied reasoning.

Types of semantic information. Based on optimized 3DGS, we lift the semantic information on that. As noted in the main paper, our method is compatible with both types of semantics: closed-set category labels [31] and open-set CLIP [24] embeddings. For a fair comparison, we follow the semantics used in each competing method. As 3D-Mem [34] is built on the closed-set object detector, we also adopt closed-set category semantics for the EM-EQA evaluation. In contrast, Dr. Splat [12] is based on open-set CLIP image embedding, and thus we employ open-set segmentation pipelines and CLIP embeddings for question-based 3D referring segmentation.

Evidence categories from question query. We extract evidence categories from the question queries because closedset semantics can only be retrieved through category labels, and open-set embeddings often struggle to capture the complex language descriptions. Therefore, we design our system to operate at the category level. Using the ScanNet200 [26] categories, we obtain categorical information $C _ { j }$ for each Gaussian $G _ { j }$ in the scene S. Based on the categories $C _ { j }$ , we use Large-Language Models (LLMs) to identify the top-k categories that are most relevant to the question, which serve as our evidence objects. Following 3D-Mem [34], we set $k = 3$ . We use the same prefiltering prompt for obtaining evidence objects as in [34].

## A.2. Initial View Selection

For the initial view selection, we cluster for the activated Gaussians ${ \mathcal { G } } ^ { \mathrm { a c t } }$ to split them into instance-level clusters. We apply a two-step clustering procedure:

$$
\{ \mathcal { G } _ { l } ^ { \mathrm { c l u s t e r } } \} _ { l = 1 } ^ { L } = \mathrm { M e r g e } ( \mathrm { H D B S C A N } ( \mathcal { G } ^ { \mathrm { a c t } } , m _ { c } , m _ { s } ) , \epsilon ) ,\tag{8}
$$

where Gcluster denotes the resulting L clusters, $m _ { c }$ is the minimum cluster size, and $m _ { s }$ is the minimum number of nearby samples. Although HDBSCAN captures hierarchical density structures to handle clusters of varying densities, it struggles to detect clusters of diverse sizes. To mitigate this, we first use HDBSCAN clustering with a small $m _ { c }$ to preserve small-object clusters, and then apply a merge function Merge that combines nearby clusters within a distance threshold Ïµ, producing instance-level groupings. The combination of HDBSCAN and Merge is our clustering function Cluster. The parameter Ïµ controls the average number of frames: a larger Ïµ merges more clusters and thus reduces the average number of frames, while a smaller Ïµ yields finegrained clusters, resulting in a smaller average number of frames.

<!-- image-->  
Figure S1. Benchmark dataset generation pipeline. Because no paired question, 3DGS instance annotations exist, we manually curate instance-specific question descriptions for each 3D Gaussian instance based on the ScanNet [3] instance ground truth.

## A.3. Final View Selection

We integrate Vision-Language Models (VLMs) and Large-Language Models (LLMs) into the novel-view adjustment stage to measure their informativeness. For the initial answer prediction stage using a visual QA prompt, we employ GPT-4o as the VLM. For both the final view selection and the verification stages, we use GPT-5-mini as the LLM. We utilize the same VLMs to evaluate the LLM-Match metrics, following the setting of our competing methods [19, 34]. We describe our prompt for each step in Figs. S4, S5, S6, and S7.

## B. Experimental Setup

## B.1. EM-EQA Evaluation

The original episodic-memory embodied question answering (EM-EQA) benchmark from OpenEQA [21] contains 1,636 questions. In our experiments, we follow the extracted snapshot information provided by the 3D-Mem authors, which already includes a small omission. As a result, we evaluate on 1,623 questions (approximately 1% fewer), matching the setting used in their released data and results. The omitted case corresponds to scene0648 01 from the ScanNet [26] dataset. We reproduce all previous methods reported in Table. 1 in the main paper, except for CG Captions [8], SVM Captions [19], and the Human results [21]. The first two cannot be reproduced because their official code is unavailable, while the Human score is an absolute reference collected over all 1,636 questions and thus cannot be re-evaluated under our experimental setting.

## B.2. Question-based 3D Referring Segmentation

The original 3D object localization task focuses on querying object categories from the semantic 3D Gaussians [12]. Our goal is to extend this task to a fine-grained setting, where the system must handle complex natural-language queries, specifically with questions, and identify the specific object instance among multiple candidates/proposals.

Dr. Splat [12] is the representative work that shows such 3D object localization task, and we follow the official implementation. To address compositional natural language queries, ReferSplat [12] introduces a referring segmentation framework for 3D Gaussian Splatting. However, ReferSplat requires a dedicated referring-segmentation dataset for each 3D scene, and no official automatic data-generation pipeline is available. To address this, we manually design a custom data generation pipeline using foundation models [16, 37]. For simplicity, our pipeline produces a descriptive sentence for each 3D instance, forming a referring based 3D finegrained localization dataset.

To conduct question-based 3D referring segmentation, we select activated Gaussians within the frustum of the chosen camera. This design allows us to evaluate how 3D referring segmentation behaves in novel-view adjustment stages and highlights the effectiveness of our approach. Our 3D referring segmentation stage operates as follows: 1) obtain the activated Gaussians, ${ \mathcal { G } } ^ { \mathrm { a c t } } , 2 )$ cluster the activated Gaussians into instance units, $\{ \mathcal { G } _ { l } ^ { \mathrm { c l u s t e r } } \} _ { l = 1 } ^ { L } , 3 )$ select the most informative camera viewpoint for each cluster, $\mathcal { H } _ { l = 1 } ^ { L }$ (initial views), 4) determine the final view set, ${ \mathcal { H } } ^ { \mathrm { f i n a l } } = { \dot { \{ H _ { f } \} } } _ { f = 1 } ^ { L } ,$ by considering both the initial and generated novel views, 5) select a single answer view Hanswer by feeding final views together with user question into the VLMs and choosing the view most aligned with the question, and 6) retain only the activated 3D Gaussians that lie within the frustum of the answer view, Hanswer.

<!-- image-->  
Figure S2. Final view selection. We apply novel view synthesis to refine the initially selected views, leveraging the unique capability of 3DGS. Our results show that when initial views are insufficient, the novel-view adjustment can recover informative viewpoints, yielding more accurate question-aligned answers. Empirically, VLMs tend to prefer evidence-centered or zoomed-in images compared to the initially selected view.

## B.3. Ablation Study of EM-EQA

We conduct ablation studies on a subset of 184 questions to enable faster and more efficient design exploration. This subset is identical to the active embodied question answering (A-EQA) split used in prior work [19, 34].

## C. Benchmark Dataset Curation

## C.1. Dataset Construction

We construct a new benchmark dataset designed to evaluate question-based 3D referring segmentation from complex question queries within a Gaussian-based scene representation. Starting from the ScanNet [26] dataset, we extract instance-level segmentation information from the official ground truth (GT) annotations. Since no dataset provides paired {question description, 3D instance segmentation} annotations, we manually create instance-specific question descriptions (i.e., human annotations), as illustrated in Fig. S1.

<table><tr><td>Method</td><td># Questions</td><td>GT Type</td><td>Metric</td><td>Instance</td></tr><tr><td>ReferSplat [9]</td><td>59</td><td>2D Image</td><td>2D mIoU</td><td>Single</td></tr><tr><td>Ours</td><td>49</td><td>3D Gaussians</td><td>3D mIoU</td><td>Multi</td></tr></table>

Table S1. Dataset statistics comparison: ReferSplat vs. Ours. While our benchmark has a comparable number of questions to ReferSplat, it offers significant advantages: the ground truth (GT) is distilled directly into 3D Gaussians, and the evaluation metrics can be measured in actual 3D space. ReferSplat focuses on identifying a single target instance, whereas our setting requires distinguishing the correct instance among multiple candidates, making it a more challenging task.

As our target task requires localizing objects directly within the 3D Gaussians, we distill the point-cloud based semantic GT into the Gaussian representation. Specifically, we compute the Mahalanobis distance between each point and all Gaussians, accumulate these point-to-Gaussian assignments, and then apply an argmax operation to determine the instance label for each Gaussian. The detailed groundtruth distillation procedure follows the method described in [12]. Finally, the generated queries cover diverse reasoning types, including spatial relationships (e.g., on top of, close to, near, closest, farthest) and fine-grained object attributes (e.g., floral pattern, leather, ivory-colored). These rich and compositional query types reflect practical scenarios in embodied settings, where an agent must understand nuanced descriptions and locate specific objects within complex 3D environments.

<!-- image-->

<!-- image-->  
Question : Which book is placed on the desk? Evidence category: book

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Question : Which towel is spread out the widest? Evidence category: towel

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Question : "Which towel is next to the shower curtain? Evidence category: towel

<!-- image-->  
RGB

<!-- image-->  
Dr. Splat (category)

<!-- image-->  
Dr. Splat (question)

<!-- image-->  
ReferSplat

<!-- image-->  
Ours

<!-- image-->  
GT

Figure S3. Question-based 3D referring segmentation. Compared to the competing methods, ours more accurately identifies the fine-grained target locations referenced by the question. Dr. Splat is designed to find objects without distinguishing instances, which limits its ability to localize fine-grained target regions (category) and, consequently, makes it struggle to directly interpret complex language queries (question). ReferSplat is designed to respond to handle complex language queries, but is rendering-baesd optimization is not compatible with direct 3D search, leading to limited 3D localization capability.

## C.2. Statistics of Dataset

Our dataset comprises 24 object instances across 10 classes, including towel, desk, plant, bed, and sofa. Each instance is annotated with at least two question queries, yielding a total of 49 {question, 3DGS instance} pairs. Among these queries, 40 focus on spatial relations, 8 target object attributes, and 1 incorporates both aspects. Compared to referring-segmentation datasets for 3D Gaussians [9], our dataset contains a comparable number of questions but is constructed and evaluated directly in 3D space. Both datasets aim to understand spatial relationships; however, our setting requires distinguishing the correct instance among multiple candidates, making it a more challenging task and better aligned with practical embodied reasoning scenarios. In contrast, ReferSplat focuses on identifying a single target instance, as shown in Table S1. In ReferSplat, the number of questions is split into train and test sets, but since we use it only for evaluation, we report the size of the test split.

## D. Additional Results

## D.1. Final View Selection

We provide additional qualitative results of novel view adjustment based on the final selected views, as shown in Fig. S2.

Compared to the initially selected views, our method can capture more informative evidence relevant to the question, further demonstrating the necessity of using VLMs-as-Judge in the novel-view adjustment stage. In practice, VLMs often favor views that bring the key evidence closer to the center or zoom in on the relevant region.

## D.2. Question-based 3D Referring Segmentation

We report additional question-based 3D referring segmentation results in Fig. S3. We evaluate Dr. Splat [12] at both the category level and the question level. At the category level, Dr. Splat can retrieve all relevant categories but fails to localize objects that are specific to the user query. At the question level, it struggles to directly interpret complex language descriptions.

ReferSplat [9] is designed to handle spatial language queries, but because it is optimized using a rendering-based loss, it cannot perform direct search in 3D. In contrast, our method can finely distinguish the specific objects referred to by the user question.

## E. Discussion and Limitation

We present GaussExplorer, a framework for constructing informative and compact episodic memories using 3D Gaussians and for enabling VLM-based 3D scene exploration and visual reasoning. Our initial view selection stages identify informative camera poses through visibility-based strategies, and the final view selection stage introduces novel viewpoints that reveal previously under-observed regions and provide evidence aligned with the user query. Through this process, GaussExplorer can localize fine-grained, questionrelevant 3D regions, supporting embodied question reasoning and 3D referring segmentation, both of which are directly linked to embodied task scenarios.

While our work primarily targets episodic-memory settings such as smart-glasses applications, our framework is not directly applied for fully active embodied exploration or navigation in robotic settings, where action policies and sequential decision-making are required. Extending Gauss-Explorer toward such active settings, where the agent must actively move, plan, and reason over long-horizon trajectories, would be a promising direction for future work.

<!-- image-->

Figure S4. Prompt of visual question-answering at novel-view adjustment stage. The placeholders {question} and [img] are replaced with the user question and novel view images, respectively.  
<!-- image-->  
Figure S5. Prompt of view selection at novel-view adjustment stage. The placeholders {question} and {initial answer prediction} are replaced with the user question and predicted initial answer from the initial answer prediction stage, respectively.

<!-- image-->  
Figure S6. Prompt of verification stage. The placeholder {question} is replaced with the user question. {candidate answer 0} denotes the answer predicted without the novel-view adjustment stage (i.e., only with initial view selection), {candidate answer 1} denotes the answer predicted with the with novel-view adjustment stage (i.e., with the final view selection). The verification stage is crucial for determining whether the selected novel view is truly informative.

<!-- image-->  
Figure S7. Prompt of embodied question-answering. The placeholders {question} and [img] are replaced with the user question and novel view images, respectively. We append an output-format constraint that requires the VLM to specify which image is responsible for its answer. The selected index is then used to determine the corresponding camera, from which we perform fine-grained 3D object localization.

## References

[1] CatË alina Cangea, Eugene Belilovsky, Pietro Li Ë o, and Aaron \` Courville. Videonavqa: Bridging the gap between visual and embodied question answering. In BMVC, 2019. 2

[2] Soravit Changpinyo, Doron Kukliansy, Idan Szpektor, Xi Chen, Nan Ding, and Radu Soricut. All you may need for VQA are image captions. In North American Chapter of the Association for Computational Linguistics, 2022. 2

[3] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias NieÃner. Scannet: Richlyannotated 3d reconstructions of indoor scenes. In CVPR, pages 5828â5839, 2017. 5, 7, 2

[4] Abhishek Das, Samyak Datta, Georgia Gkioxari, Stefan Lee, Devi Parikh, and Dhruv Batra. Embodied question answering. In CVPR, 2018. 2

[5] Martin Ester, Hans-Peter Kriegel, Jorg Sander, Xiaowei Xu, Â¨ et al. A density-based algorithm for discovering clusters in large spatial databases with noise. In International Conference on Knowledge Discovery and Data Mining (KDD), 1996. 4

[6] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang, et al. Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. In NeurIPS, 2024. 8

[7] Junqi Ge, Ziyi Chen, Jintao Lin, Jinguo Zhu, Xihui Liu, Jifeng Dai, and Xizhou Zhu. V2pe: Improving multimodal longcontext capability of vision-language models with variable visual position encoding. In ICCV, 2025. 2, 5

[8] Qiao Gu, Ali Kuwajerwala, Sacha Morin, Krishna Murthy Jatavallabhula, Bipasha Sen, Aditya Agarwal, Corban Rivera, William Paul, Kirsty Ellis, Rama Chellappa, et al. Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning. In IEEE International Conference on Robotics and Automation, 2024. 2, 6

[9] Shuting He, Guangquan Jie, Changshuo Wang, Yun Zhou, Shuming Hu, Guanbin Li, and Henghui Ding. Refersplat: Referring segmentation in 3d gaussian splatting. In ICML, 2025. 2, 6, 7, 3, 4

[10] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card, 2024. 1

[11] Yuzhou Ji, He Zhu, Junshu Tang, Wuyi Liu, Zhizhong Zhang, Xin Tan, and Yuan Xie. Fastlgs: Speeding up language embedded gaussians with feature grid mapping. In AAAI, 2025. 2

[12] Kim Jun-Seong, GeonU Kim, Kim Yu-Ji, Yu-Chiang Frank Wang, Jaesung Choe, and Tae-Hyun Oh. Dr. splat: Directly referring 3d gaussian splatting via direct language embedding registration. In CVPR, 2025. 2, 3, 6, 7, 8, 1, 4

[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and Â¨ George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM TOG, 2023. 2, 3, 4, 1

[14] Bernhard Kerbl, Andreas Meuleman, Georgios Kopanas, Michael Wimmer, Alexandre Lanvin, and George Drettakis. A hierarchical 3d gaussian representation for real-time rendering of very large datasets. ACM Transactions on Graphics (TOG), 43(4):1â15, 2024. 1

[15] Nuri Kim, Obin Kwon, Hwiyeon Yoo, Yunho Choi, Jeongho Park, and Songhwai Oh. Topological semantic graph memory for image-goal navigation. In Annual Conference on Robot Learning, 2022. 2

[16] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023. 3, 7, 2

[17] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 2

[18] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, 2024. 1, 2, 6

[19] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al. Openeqa: Embodied question answering in the era of foundation models. In CVPR, 2024. 2, 5, 6, 3

[20] Claudia Malzer and Marcus Baum. A hybrid approach to hierarchical density-based cluster selection. In 2020 IEEE International Conference on multisensor fusion and integration for intelligent systems (MFI). IEEE, 2020. 4

[21] OpenAI. Gpt-4 technical report. https://arxiv.org/ abs/2303.08774, 2023. 2, 3, 6

[22] OpenAI. Gpt-4v(ision) system card. https://openai. com/research/gpt-4v-system-card, 2023. 2, 3

[23] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20051â20060, 2024. 2

[24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 3, 7, 1

[25] Santhosh Kumar Ramakrishnan, Aaron Gokaslan, Erik Wijmans, Oleksandr Maksymets, Alexander Clegg, John M Turner, Eric Undersander, Wojciech Galuba, Andrew Westbury, Angel X Chang, et al. Habitat-matterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai. In Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2021. 5

[26] David Rozenberszki, Or Litany, and Angela Dai. Languagegrounded indoor 3d semantic segmentation in the wild. In ECCV, 2022. 5, 1, 2, 3

[27] Saumya Saxena, Blake Buchanan, Chris Paxton, Peiqi Liu, Bingqing Chen, Narunas Vaskevicius, Luigi Palmieri, Jonathan Francis, and Oliver Kroemer. Grapheqa: Using 3d semantic scene graphs for real-time embodied question answering. In Annual Conference on Robot Learning, 2025. 2

[28] Aditya Sharma, Michael Saxon, and William Yang Wang. Losing visual needles in image haystacks: Vision language models are easily distracted in short and long contexts. In Findings of the Association for Computational Linguistics: EMNLP 2024, 2024. 2, 5

[29] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua Guan. Language embedded 3d gaussians for open-vocabulary scene understanding. In CVPR, 2024. 2

[30] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. https: //arxiv.org/abs/2307.09288, 2023. 2

[31] Rejin Varghese and M Sambath. Yolov8: A novel object detection algorithm with enhanced performance and robustness. In 2024 International conference on advances in data engineering and intelligent computing systems (ADICS). IEEE, 2024. 3, 1

[32] Zhaowei Wang, Wenhao Yu, Xiyu Ren, Jipeng Zhang, Yu Zhao, Rohit Saxena, Liang Cheng, Ginny Wong, Simon See, Pasquale Minervini, Yangqiu Song, and Mark Steedman. Mmlongbench: Benchmarking long-context vision-language models effectively and thoroughly. In NeurIPS, 2025. 2, 5

[33] Yanmin Wu, Jiarui Meng, Haijie Li, Chenming Wu, Yahao Shi, Xinhua Cheng, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, et al. Opengaussian: Towards point-level 3d gaussian-based open vocabulary understanding. In NeurIPS, 2024. 2

[34] Yuncong Yang, Han Yang, Jiachen Zhou, Peihao Chen, Hongxin Zhang, Yilun Du, and Chuang Gan. 3d-mem: 3d scene memory for embodied exploration and reasoning. In CVPR, 2025. 2, 3, 4, 5, 6, 8, 1

[35] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit anything in 3d scenes. In ECCV, 2024. 2

[36] Hang Yin, Xiuwei Xu, Zhenyu Wu, Jie Zhou, and Jiwen Lu. Sg-nav: Online 3d scene graph prompting for llm-based zero-shot object navigation. In NeurIPS, 2024. 2

[37] Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, and Jianke Zhu. Osprey: Pixel understanding with visual instruction tuning. In CVPR, 2024. 7, 2

[38] Chenyangguang Zhang, Alexandros Delitzas, Fangjinhua Wang, Ruida Zhang, Xiangyang Ji, Marc Pollefeys, and Francis Engelmann. Open-vocabulary functional 3d scene graphs for real-world indoor spaces. In CVPR, 2025. 2

[39] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In CVPR, 2024. 2

[40] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross. Ewa volume splatting. In Proceedings Visualization, 2001. VISâ01. IEEE, 2001. 4, 1