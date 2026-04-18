# TOWARDS PHYSICALLY EXECUTABLE 3D GAUSSIAN FOR EMBODIED NAVIGATION

Bingchen Miao1,2, Rong Wei2,, Zhiqi Ge1, Xiaoquan Sun2,3, Shiqi Gao1, Jingzhe Zhu1, Renhan Wang2, Siliang Tang1, Jun Xiao1, Rui Tang2, Juncheng Li1â

Zhejiang University 1, Manycore Tech Inc 2, Huazhong University of Science and Technology3

## ABSTRACT

3D Gaussian Splatting (3DGS), a 3D representation method with photorealistic real-time rendering capabilities, is regarded as an effective tool for narrowing the sim-to-real gap. However, it lacks fine-grained semantics and physical executability for Visual-Language Navigation (VLN). To address this, we propose SAGE-3D (Semantically and Physically Aligned Gaussian Environments for 3D Navigation), a new paradigm that upgrades 3DGS into an executable, semantically and physically aligned environment. It comprises: (1) Object-Centric Semantic Grounding, which adds object-level fine-grained annotations to 3DGS; and (2) Physics-Aware Execution Jointing, which embeds collision objects into 3DGS and constructs rich physical interfaces. We release InteriorGS, containing 1K object-annotated 3DGS indoor scene data, and introduce SAGE-Bench, the first 3DGS-based VLN benchmark with 2M VLN data. Experiments show that 3DGS scene data is more difficult to converge, while exhibiting strong generalizability, improving baseline performance by 31% on the VLN-CE Unseen task. Our data and code are available at: https://sage-3d.github.io.

<!-- image-->  
Figure 1: Traditional 3DGS vs. Our work. Compared with traditional 3DGS, our InteriorGS provides object-level 3DGS annotations, while SAGE-Bench contains semantically VLN data and detailed physical interfaces, representing an semantically and physically aligned 3DGS paradigm.

## 1 INTRODUCTION

Vision-and-Language Navigation (VLN) is a core capability for Vision-Language Action (VLA) models, enabling them to follow natural language instructions and navigate complex indoor spaces (Wei et al., 2025; Zhang et al., 2024). Direct real-world training is costly and risky, motivating the widely adopted sim-to-real paradigm (Qi et al., 2025; Zun Wang, 2023). Reducing the resulting sim-to-real gap has driven the evolution of scene representations, from early scanned mesh reconstructions such as Matterport3D (Chang et al., 2017) and HM3D (Ramakrishnan et al., 2021), to most recently 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023).

Compared with prior VLN work (Krantz et al., 2020; Song et al., 2025) using scanned mesh reconstructions from RGB-D scans, 3DGS offers two key advantages: 1) Easier and more reliable object-level semantics. Scanned mesh reconstructed from noisy depth scans forms a single continuous surface that merges objects into surrounding structures, making later separation costly (Cheng et al., 2025). In contrast, 3DGS represents scenes with discrete Gaussians that can be directly labeled. 2) View-consistent and photorealistic appearance. Scanned mesh textures, stitched from sparse RGB viewpoints, often break under novel views, where incomplete coverage yields seams, stretching, or blur (Dalal et al., 2024). 3DGS instead optimizes a continuous radiance field, yielding consistent, photorealistic views from any navigable positionâcrucial for free-moving navigation.

Table 1: Comparisons with benchmarks for continuous navigation tasks. Here, âInstruction with Causalityâ: tasks have causal dependencies rather than being mere âA-to-Bâ navigation; âScene Geometryâ: whether the scene mesh is an imperfect estimate or accurate ground truth.
<table><tr><td>Benchmarks</td><td>Num. of Task</td><td>Num. of Scenes</td><td>Scene Source</td><td>Instruction with Casuality</td><td>Scene Geometry</td><td>3D Representation</td></tr><tr><td>VLN-CE (Krantz et al., 2020)</td><td>4.5k</td><td>90</td><td>MP3D</td><td>X</td><td>Estimated</td><td>Scanned Mesh</td></tr><tr><td>OVON (Yokoyama et al., 2024)</td><td>53k</td><td>181</td><td>HM3D</td><td>X</td><td>Estimated</td><td>Scanned Mesh</td></tr><tr><td>GOAT-Bench (Khanna* et al., 2024)</td><td>725k</td><td>181</td><td>HM3D</td><td>X</td><td>Estimated</td><td>Scanned Mesh</td></tr><tr><td>IR2R-CE (Krantz et al., 2022)</td><td>414</td><td>71</td><td>MP3D</td><td>X</td><td>Estimated</td><td>Scanned Mesh</td></tr><tr><td>LHPR-VLN (Song et al., 2025)</td><td>3.3k</td><td>216</td><td>HM3D</td><td>X</td><td>Estimated</td><td>Scanned Mesh</td></tr><tr><td>OctoNav-Bench (Gao et al., 2025)</td><td>45k</td><td>438</td><td>MP3D, HM3D</td><td>X</td><td>Estimated</td><td>Scanned Mesh</td></tr><tr><td>SAGE-Bench</td><td>2M</td><td>1000</td><td>InteriorGS</td><td>â</td><td>Ground Truth</td><td>3DGS-Mesh Hybrid Representation</td></tr></table>

Despite these advantages, the current 3DGS is solely used for high-fidelity rendering (Wang, 2024), as shown in the upper left corner of Fig. 1. It is unsuitable for effective application in VLN tasks due to its two significant limitations: (1) 3DGS is deficient in fine-grained object-level semantics. Existing 3DGS scenes contain only color and density information, with no instance IDs or object attributes (Li et al., 2024). This makes it impossible to uniquely ground VLN instructions such as âgo to the red chair next to the white bookshelfâ, and any attempt to recover object boundaries requires complex and error-prone post-processing. (2) Lack of a physically executable structure. Gaussian Splatting is, by nature, a volumetric rendering technique; although recent efforts (e.g., SuGaR (Guedon & Lepetit, 2024)) attempt to infer surface information from Gaussians, obtaining Â´ smooth surfaces remains challenging. Consequently, deriving reliable collision geometries from 3DGS is difficult, and aligning semantics with appearance is non-trivial.

In this work, we present SAGE-3D (Semantically and Physically Aligned Gaussian Environments for 3D Navigation), a paradigm that upgrades 3DGS from a purely perceptual scene representation to an executable, semantically and physically aligned environment foundation for embodied navigation. This transformation is enabled by two core components: (1) Object-Level Semantic Grounding. We sample 3DGS data from artist-created mesh scenes and create an object-level annotated indoor dataset through careful manual labeling and double verification, thereby endowing 3DGS with fine-grained semantics. Additionally, we design a 2D semantic top-down map derived from 3DGS to support instruction generation and path planning. (2) Physics-Aware Execution Jointing. We introduce a 3DGS-Mesh Hybrid Representation: starting from our mesh scene data, we extract collision bodies for each object as the physics layer, while using 3DGS to provide photorealistic appearance. This decoupled design preserves high-fidelity rendering through 3DGS and enables accurate physical simulation based on mesh-based collision bodies, with connectivity to rich robotics APIs. Together, these two components transform 3DGS into a practical embodied navigation environment substrate and open new avenues for future embodied intelligence research.

Building on this, we release InteriorGSâa dataset of 1,000 manually object-annotated 3DGS scenes. It covers mostly furnished indoor environments plus venues like concert halls and amusement parks, totaling over 554k object instances across 755 categories. We also introduce SAGE-Bench (Tab. 1), the first fully 3DGS-based VLN benchmark with 2M new trajectory-instruction pairs and 554k detailed collision bodies. For data, we provide a hierarchical instruction scheme that combines high-level semantic goals (especially task-causal ones like âIâm thirsty, get water from the tableâ) with low-level actions (e.g., âmove from stool to sofaâ). For evaluation, we design three metrics for navigation natural continuity: Continuous Success Ratio, Integrated Collision Penalty, and Path Smoothness, to assess VLN models from the perspective of continuous motion.

Extensive experiments on SAGE-Bench yield several key insights: (1) 3DGS scene data renders faster but is harder to converge than scanned mesh data. 3DGS has a per-frame rendering time of 6.2ms, outperforming scaned meshâs 16.7ms. Yet reaching 40% Success Rate (SR) needs 160 iterations (6.2h) for 3DGS vs. 120 iterations (4.8h) for scaned meshâthis slower convergence stems from our 3DGS dataâs higher demands, as its richness and photorealism better mirror real-world

## Semantically and Physically Aligned Gaussian Environments for 3D Nav

<!-- image-->  
Figure 2: Overview of SAGE-3D, which consists of two key components: (1) Object-Level Semantic Grounding, 3DGS data is annotated by expect at the object level, then be transformed into 2D semantic maps for path planning and instruction generation; (2) Physics-Aware Execution Jointing, where scene and object collision bodies are generated via convex hull decomposition, integrated into 3DGS to form a 3DGS-Mesh Hybrid Representation, with extensive physics simulation interfaces.

complexity. (2) Our scene-rich, photorealistic 3DGS VLN data exhibits strong generalizability. Models trained entirely on this data achieve a significant performance improvement (31% SR increase) over baselines in unseen VLN-CE environments (Krantz et al., 2020), a result driven by the dataâs alignment with real-world scenarios. (3) Our newly proposed three continuity metrics enable studying navigationâs natural continuity, addressing gaps in conventional metrics. Our newly designed navigation natural continuity metrics reveal that conventional metrics fail to capture model issues like continuous collisions and unsmooth motion, for example, in one experiment case, our ICP (indicating continuous collisions) reaches 0.87, while the traditional collision rate is only 1.

In summary, our contributions are as follows:

â¢ We construct the first large-scale dataset of 1k fully furnished indoor 3DGS reconstructions with dense object-level annotations, released as InteriorGS.

â¢ We propose SAGE-3D, a new paradigm that augments 3DGS with semantic granularity and physical validity, transforming it into an executable environment foundation.

â¢ We build SAGE-Bench, a VLN benchmark based on 3DGS with fine-grained semantics, accurate per-object physical simulation, and rich interfaces for robot embodiments.

â¢ We conduct extensive experiments based on our new paradigm and derive several novel insights in the VLN domain and validate the superiority of our newly introduced data.

## 2 SAGE-3D

In this section, we systematically introduce SAGE-3D, a novel embodied learning paradigm based on 3DGS, as illustrated in Fig. 2. We first provide a formal definition of this paradigm (Section 2.1), followed by an introduction add fine-grained semantic labels to 3DGS through manual annotation and the generation of 2D top-down semantic maps (Section 2.2). We then utilize convex hull decomposition to extract collision bodies and construct a rich physical simulation interface (Section 2.3).

## 2.1 SAGE-3D PARADIGM

We propose SAGE-3D (Semantically and Physically Aligned Gaussian Environments for 3D Navigation), a new paradigm that uses 3DGS as the environment foundation for training and evaluating embodied agent. This paradigm upgrades 3DGS, originally used solely for photorealistic rendering, into an executable, semantically and physically aligned environment foundation that supports continuous Vision-and-Language navigation and related tasks.

Formally, we define SAGE-3D as the process of transforming a Gaussian primitive set G from a 3DGS scene, with added semantics M and physics Î¦, into an executable environment:

$$
G \ : + \ : M \ : + \ : \Phi \ : \longrightarrow \ : { \mathcal E } _ { \mathrm { e x e c } }
$$

where $G = \{ g _ { i } \} _ { i = 1 } ^ { N }$ is the set of Gaussian primitives, M is the semantic layer (e.g., instance/category maps, attributes), and Î¦ is the physics layer (e.g., collision bodies, dynamics). The resulting

environment can be formalized as a semantics- and physics-augmented POMDP (Partially Observable Markov Decision Process):

$$
\mathcal { E } = ( \mathcal { U } , \mathcal { S } , \mathcal { A } , \mathcal { O } , T , Z ; M , \Phi ) ,
$$

where U is the instruction space, S the continuous state space, A the action space, O the multimodal observation space, and T, Z are physics-driven state transition and rendering functions.

The core goal of this paradigm is to preserve the photorealistic rendering quality of 3DGS while introducing object-level semantics and physical executability, making 3DGS a viable environment foundation for training and evaluating embodied agents.

## 2.2 OBJECT-LEVEL SEMANTIC GROUNDING

Conventional 3DGS encodes appearance (e.g., color, density) but lacks instance IDs or object attributes, limiting precise object-level VLN instructions (Chen & Wang, 2025; Li et al., 2024). To overcome this, we release InteriorGS, a manually annotated 3DGS dataset with object-level semantics, and introduce a 2D top-down semantic map generator to support instruction generation.

InteriorGS. We construct InteriorGS: a dataset of 1k high-fidelity indoor 3DGS scenes (752 residential interior scenes and 248 public spaces such as concert halls, amusement parks, and gyms) with double-verified object-level annotations, including object categories, instance IDs, and bounding box information. The dataset contains over 554k object instances across 755 categories, providing a dense, semantically consistent, and broadly diverse foundation for training and evaluation.

InteriorGSâs 3DGS data is sampled from our artist-created mesh scenes. To achieve reliable 3DGS reconstruction in occlusion-rich indoor environments, we render an average of 3,000 camera views per scene and use the open-source GSplat pipeline (Ye et al., 2025) to estimate the 3DGS parameters. The detailed sampling process is provided in Appendix B.

2D Semantic Top-Down Map Generation. Unlike scanned mesh workflows that build NavMesh (e.g., by exhaustive scene traversal in Habitat) (Song et al., 2025; Krantz et al., 2022), 3DGS lacks inherent semantics and discrete entities, making such representations infeasible. We therefore design a 2D semantic top-down map by projecting annotated 3D objects from InteriorGS onto the ground plane, with doors tagged by state (open / closed / half-open) and walls marked as non-traversable. Although annotations are stored as axis-aligned 3D boxes, we refine each footprint into an irregular mask by sampling object surface points, projecting them, and taking a 2D convex hull to optimize:

$$
{ \mathcal { M } } _ { k } = \operatorname { F u s e } \left( \operatorname { H u l l } \left\{ \Pi _ { \mathrm { t o p } } ( p ) \mid p \in \operatorname { S u r f } ( o _ { k } ) \right\} \right)
$$

where $\mathcal { M } _ { k }$ is the 2D mask for object $o _ { k } , \operatorname { S u r f } ( o _ { k } )$ is the set of sampled surface points of object $o _ { k } , \Pi _ { \mathrm { t o p } }$ is the projection onto the ground plane, Hull(Â·) denotes the 2D convex-hull operator, and Fuse(Â·) merges multi-view masks into a consistent footprint.

## 2.3 PHYSICS-AWARE EXECUTION JOINTING

3DGS with semantics still cannot serve directly as a VLN environment, as it allows issues such as mesh penetration that hinder embodied learning (Yue et al., 2024). To overcome this, we extract object-level collision geometry, derive navigable space, and provide a physics simulation interface.

Physics Simulation with 3DGSâMesh Hybrid Representation. Starting with version 5.0, Isaac Sim supports rendering 3DGS assets from USDZ files exported by 3DGUT (Wu et al., 2025a). However, the imported 3DGS are appearance-only and do not carry physics. To enable physically executable scenes, we take the artist-created triangle meshes of each object and apply CoACD (Wei et al., 2022) for convex decomposition, yielding per-object collision bodies. We then assemble a USDA scene where the collision bodies are authored as invisible rigid shapes (driving contact and dynamics), while the 3DGS file remain visible and provide photorealistic appearance. Concretely, each object is instantiated as a USD prim and augmented with $\Phi _ { k }$ (rigid-body and contact parameters), where static-scene objects default to static bodies, and a curated subset is configured as movable or articulated to support extended interactions. This 3DGSâMesh Hybrid Representation authoring removes the need to ray trace the artist meshes at runtime, preserves high-fidelity rendering through 3DGS, and supplies accurate collision geometry for physics.

<!-- image-->  
Figure 3: Overview of SAGE-Bench. SAGE-Bench includes a hierarchical instruction generation scheme, two major task types, two episode complexity categories, and three newly designed natural continuity metrics for navigation.

Agents, Control, and Observations in a Continuous Environment. The simulator exposes robot APIs for legged and wheeled ground platforms (e.g., Unitree G1 / Go2 / H1) and aerial robots (e.g., quadrotor UAVs). Action interfaces support both discrete commands (e.g., turn/forward/stop) and continuous controlâvelocity commands (v, Ï) for ground robots and 6-DoF velocity/attitude commands for UAVsâexecuted in a continuous environment (metric 3D space, no teleportation between panoramic nodes). The environment provides synchronized RGB, depth, semantic segmentation, poses, and contact events, along with built-in collision detection, stuck/interpenetration monitoring, and recovery. Offline-generated collision bodies are cached to accelerate loading and ensure stable, repeatable evaluation.

## 3 SAGE-BENCH

In this section, we introduce SAGE-Bench, the first 3DGS-based VLN benchmark, as shown in Fig. 3. It includes a hierarchical instruction generation scheme (Section 3.1), a three-axis evaluation framework (Section 3.2), and three navigation natural continuity metrics (Section 3.3).

## 3.1 DATA GENERATION

Hierarchical Instruction Generation. To address the limitations of current benchmarks (Zun Wang, 2023), particularly the lack of tasks with causal dependencies such as âIâm thirsty, get water from the tableâ, we introduce a hierarchical scheme that combines high-level semantics with low-level action primitives for more realistic navigation.

We define two levels of instructions: High-level instructions emphasize task semantics and humanoriented intent, and comprise 5 categories: Add Object (introducing causal objects or actions that make a trajectory contextually meaningful); Scenario Driven (embedding specific situational motives that make the destination a reasonable place for execution); Relative Relationship (distinguishing similar nearby targets via spatial relations such as ânext toâ or âoppositeâ); Attribute-based (identifying a unique target using perceivable attributes like color, state, or contents); Area-based (directing the agent toward a general functional area rather than a specific object). Low-level instructions focus on control and kinematic evaluation, including primitive actions such as forward moves. Detailed design and explanation can be found in Appendix C.

Low-level instructions are created by templating the start and end waypoints. High-level instructions are generated by feeding an MLLM with a prompt (detailed in Appendix C) constructed from object categories, attributes, and spatial relations in the 2D semantic map.

Trajectory Generation. Using the collision bodies from Section 2.1, we construct the final navigation map by combining a 1.2 m-height occupancy map with the 2D semantic map. Then we run A\*-based shortest-path search to generate trajectories, more details can be found in Appendix A.

In total, we produce 2M new instructionâtrajectory pairs for VLN. We balance the data distribution and select 1,148 samples to form the SAGE-Bench test split, including 944 high-level and 204 low-level samples across 35 distinct scenes, with the remainder used for training and validation.

## 3.2 THREE-AXIS EVALUATION FRAMEWORK

SAGE-Bench introduces a three-axis evaluation framework that orthogonally combines task types, instruction level, and episode complexity into discrete evaluation slices.

Task Types. This axis specifies the task paradigm and input form, considering two fundamental navigation tasks: VLN and No-goal Navigation (Nogoal-Nav). Nogoal-Nav aims to drive the model to explore the environment as much as possible in order to test policy understanding of the environment and the safety of exploration. We select 100 scenes as the test set for Nogoal-Nav.

Instruction Level. This axis measures how semantic and structural complexity affects the model, and it is aligned with the hierarchical instruction generation scheme described in Section 3.1.

Episode Complexity. This axis quantifies task complexity, covering both scene complexity and path complexity. Scene complexity primarily refers to asset density: we define scenes with more than 376 assets as âmanyâ and those with fewer than 184 assets as âfewâ. Path complexity considers path length: we define paths longer than 29.0 m as âlongâ and those shorter than 8.4 m as âshortâ.

## 3.3 NAVIGATION NATURAL CONTINUITY METRIC

As a new continuous navigation benchmark, to assess VLN model performance from the perspective of continuous motion, SAGE-Bench introduces three natural continuity metrics for navigation.

Continuous Success Ratio (CSR). It indicates the fraction of time the agent stays within a permissible corridor around the reference path. SR makes a 0/1 judgment only at the endpoint, whereas CSR measures the proportion of time the agent stays within a permissible corridor around the reference path while satisfying task conditions, thus reflecting âgoal-consistentâ behavior throughout the $s ( t ) = { \left\{ \begin{array} { l l } { 1 , } & { \operatorname { p o s } ( t ) \in { \mathcal { C } } } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }$ and task conditions satisfied episode. Given a trajectory of length T , let

where C is defined by buffering the reference path with radius $r _ { \mathrm { t o l } }$ , then

$$
\mathrm { C S R } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } s ( t ) .
$$

Integrated Collision Penalty (ICP). It measures the time-averaged collision intensity along the trajectory, capturing both the frequency and duration of contacts. Traditional collision rate (CR) does not distinguish between occasional contact and persistent scraping. ICP integrates the collision intensity sequence $c ( t ) \in [ 0 , 1 ]$ over time as a penalty:

$$
\mathrm { I C P } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } c ( t ) ,
$$

Path Smoothness (PS). It evaluates a normalized smoothness score derived from consecutive heading-change (or curvature) magnitudes, where higher values indicate smoother paths. Smoother paths reduce abrupt turns and acceleration changes, benefiting real robot feasibility and stable planning. PS is computed from the variance of consecutive heading changes:

$$
\mathrm { P S } = 1 - \frac { 1 } { T - 1 } \sum _ { t = 2 } ^ { T } \operatorname* { m i n } \left( \frac { \left| \Delta \theta _ { t } \right| } { \pi } , 1 \right) , \quad \Delta \theta _ { t } = \theta _ { t } - \theta _ { t - 1 } ,
$$

Here $\theta _ { t }$ denotes the agentâs heading angle at trajectory time step t, and $\Delta \theta _ { t }$ is the change in heading between two consecutive time steps.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Baseline. Considering the current generalization capability of MLLM models, we conducted evaluations on a wide range of models. (1) Closed-source MLLMs as Agent: Includes Qwen-VL-MAX (Bai et al., 2023), GPT-4.1, GPT-5. (2) Open-source MLLMs as Agent: Qwen2.5-VL-7B (Bai et al., 2023), InternVL-2.5-8B (Zhu et al., 2025a), InternVL-3-8B (Chen et al., 2024),

Table 2: Comparison of different models on VLN and Nogoal-Nav tasks on SAGE-Bench. Bold values represent the best performance across all methods. Gray values indicate that these metrics lack comparative significance due to the low navigation performance of the models.
<table><tr><td rowspan="2">Methods</td><td colspan="7">VLN (High-level Instruction)</td><td colspan="2">Nogoal-Nav</td></tr><tr><td>SR â</td><td>OSR â</td><td>SPL â</td><td>CR â</td><td>CSR â</td><td>ICP â</td><td>PS â</td><td>Episode Time â</td><td>Explored Areas â</td></tr><tr><td colspan="10">Closed-source MLLMs as Agent</td></tr><tr><td>Qwen-VL-MAX</td><td>0.14</td><td>0.25</td><td>0.12</td><td>0.85</td><td>0.21</td><td>0.41</td><td>0.79</td><td>64.74</td><td>6.40</td></tr><tr><td>GPT-4.1</td><td>0.13</td><td>0.21</td><td>0.12</td><td>0.72</td><td>0.19</td><td>0.35</td><td>0.81</td><td>67.70</td><td>3.00</td></tr><tr><td>GPT-5</td><td>0.12</td><td>0.18</td><td>0.11</td><td>0.63</td><td>0.18</td><td>0.24</td><td>0.86</td><td>64.60</td><td>2.16</td></tr><tr><td colspan="10">Open-source MLLMs as Agent</td></tr><tr><td>Qwen2.5-VL-7B</td><td>0.13</td><td>0.14</td><td>0.13</td><td>0.71</td><td>0.21</td><td>0.27</td><td>0.87</td><td>42.19</td><td>6.88</td></tr><tr><td>InternVL-2.5-8B</td><td>0.10</td><td>0.13</td><td>0.10</td><td>0.52</td><td>0.14</td><td>0.33</td><td>0.88</td><td>28.82</td><td>4.28</td></tr><tr><td>InternVL-3-8B</td><td>0.12</td><td>0.20</td><td>0.11</td><td>0.64</td><td>0.17</td><td>0.32</td><td>0.82</td><td>34.70</td><td>6.34</td></tr><tr><td>Llama-3.2-11B</td><td>0.13</td><td>0.18</td><td>0.14</td><td>0.74</td><td>0.16</td><td>0.29</td><td>0.83</td><td>38.45</td><td>6.68</td></tr><tr><td colspan="10">Vision-Language Model</td></tr><tr><td>NaviLLM</td><td>0.05</td><td>0.06</td><td>0.05</td><td>0.21</td><td>0.09</td><td>0.24</td><td>0.90</td><td>18.73</td><td>5.74</td></tr><tr><td>NavGPT-2</td><td>0.10</td><td>0.12</td><td>0.11</td><td>0.33</td><td>0.14</td><td>0.29</td><td>0.83</td><td>24.51</td><td>3.36</td></tr><tr><td>CMA</td><td>0.13</td><td>0.15</td><td>0.14</td><td>0.54</td><td>0.26</td><td>0.28</td><td>0.86</td><td>44.26</td><td>3.22</td></tr><tr><td>NaVid</td><td>0.15</td><td>0.17</td><td>0.15</td><td>1.24</td><td>0.29</td><td>0.33</td><td>0.89</td><td>56.13</td><td>4.28</td></tr><tr><td>NaVILA</td><td>0.39</td><td>0.47</td><td>0.34</td><td>3.28</td><td>0.48</td><td>0.61</td><td>0.68</td><td>77.82</td><td>8.40</td></tr><tr><td>NaVid-base</td><td>0.10</td><td>0.13</td><td>0.10</td><td>0.33</td><td>0.15</td><td>0.28</td><td>0.84</td><td>20.37</td><td>3.42</td></tr><tr><td>NaVid-SAGE (Ours)</td><td>0.36</td><td>0.46</td><td>0.32</td><td>2.12</td><td>0.48</td><td>0.66</td><td>0.54</td><td>60.35</td><td>5.66</td></tr><tr><td>NaVILA-base</td><td>0.21</td><td>0.26</td><td>0.22</td><td>3.53</td><td>0.33</td><td>0.72</td><td>0.41</td><td>58.26</td><td>6.52</td></tr><tr><td>NaVILA-SAGE (Ours)</td><td>0.46</td><td>0.55</td><td>0.48</td><td>2.67</td><td>0.57</td><td>0.54</td><td>0.74</td><td>82.48</td><td>8.74</td></tr></table>

Llama-3.2-11B. (3) Vision-Language Models: We selected VLN models that have been widely used in recent years, including NaviLLM (Zheng et al., 2024), NavGPT-2 (Zhou et al., 2024), CMA (Krantz et al., 2020), NaVid (Zhang et al., 2024), and NaVILA (Cheng et al., 2025).

Evaluation Metric. (1) For the VLN task. In addition to the three novel metrics we proposed in Section 3.3 for evaluating the natural continuity of model navigation â CSR, ICP, and PS â we also adopt common metrics used in VLN tasks, including success rate (SR), oracle success rate (OSR), and success weighted by path length (SPL) and Collision Rate (CR). (2) For the No-goalNav task. There are two metrics: Episode Time and Explored Areas. An episode is terminated immediately if a collision occurs, and the maximum episode time is set to 120 seconds.

Implementation Details. We selected 500k âtrajectoryâinstructionâ pairs from SAGE-Bench, with no overlap with the test set. We trained two models on this subset: one based on NaV-ILAâs pre-trained model navila-siglip-llama3-8b-v1.5-pretrain (denoted as NaVILA-base), producing NaVILA-SAGE; and the other based on Navidâs pre-trained model navid-7b-full-224 (denoted as NaVid-base), producing NaVid-SAGE. Training details are shown in Appendix A.

## 4.2 RESULTS AND INSIGHTS

Overall Comparison on SAGE-Bench. Tab. 2 presents the experimental results of MLLMs and VLN models on SAGE-Bench. (1) SAGE-Bench poses a novel and challenging VLN task for current VLN models and MLLMs. Except for the recent SOTA VLN model NaVILA, other models achieve SR values no higher than 0.15. For instance, NaVid, which achieves 0.37 SR and 0.49 OSR on VLN-CE R2R Val-Unseen, only obtains 0.15 SR and 0.17 OSR on SAGE-Bench. Similarly, NaVILA, which achieves 0.54 SR and 0.63 OSR on VLN-CE R2R Val-Unseen, records only 0.39 SR and 0.47 OSR on SAGE-Bench. (2) MLLMsâ multimodal understanding inherently gives them some VLN capability. Both the latest open-source and closed-source MLLMs achieve VLN SRs ranging from 0.10 to 0.14 on SAGE-Bench, comparable to dedicated VLN models such as CMA (0.13 SR) and NaVid (0.15 SR), and even surpass VLN models in OSR. For example, the 0.20 OSR achieved by InternVL-3 exceeds that of NaVid (0.17 OSR). Notably, several baseline models with weak VLN performance (SR < 0.20) fail to understand navigation instructions or environmental information in our challenging tasks, behaving like ârandom or single-action predictionâ (e.g., continuous straight movement), rendering their CR, ICP, and PS metrics non-comparable.

Table 3: Rendering speed and training convergence comparison.
<table><tr><td>Environment Type</td><td>Avg. Render Time / Frame (ms) â Avg. Memory (MB) â Iters to SR=40% (k) â Time-to-SR=40% (hrs) â</td><td></td><td></td><td></td></tr><tr><td>Scanned Mesh (MP3D/HM3D)</td><td>16.7</td><td>850</td><td>120</td><td>4.8</td></tr><tr><td>3DGSMesh Hybrid Representation (Ours)</td><td>6.2</td><td>220</td><td>160</td><td>6.2</td></tr></table>

<!-- image-->  
Figure 4: Visualization case study of navigation natural continuity. The red trajectory is the ground truth, and the blue Trajectory is the trajectory of NaVILA.

Table 4: Results on VLN-CE.
<table><tr><td rowspan="2">Methods</td><td colspan="3">R2R Val-Unseen</td></tr><tr><td>SR â</td><td>OSR â</td><td>SPL â</td></tr><tr><td>Seq2Seq</td><td>0.25</td><td>0.37</td><td>0.22</td></tr><tr><td>Navid-base</td><td>0.22</td><td>0.32</td><td>0.17</td></tr><tr><td>Navid-SAGE (Ours)</td><td>0.31</td><td>0.42</td><td>0.29</td></tr><tr><td>CMA</td><td>0.32</td><td>0.40</td><td>0.30</td></tr><tr><td>NaVid</td><td>0.37</td><td>0.49</td><td>0.36</td></tr><tr><td>NaVILA-base</td><td>0.29</td><td>0.38</td><td>0.27</td></tr><tr><td>NaVILA-SAGE (Ours)</td><td>0.38</td><td>0.51</td><td>0.36</td></tr><tr><td>NaVILA</td><td>0.50</td><td>0.58</td><td>0.45</td></tr></table>

Table 5: Results on different instruction levels.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Instruction Level</td><td colspan="6">SAGE-Bench VLN</td></tr><tr><td>SR â</td><td>OSR â</td><td>SPL â</td><td>CSR â</td><td>ICP â</td><td>PS â</td></tr><tr><td rowspan="2">GPT-4.1</td><td>Low-level</td><td>0.22</td><td>0.37</td><td>0.19</td><td>0.27</td><td>0.60</td><td>0.70</td></tr><tr><td>High-level</td><td>0.13</td><td>0.21</td><td>0.12</td><td>0.19</td><td>0.35</td><td>0.81</td></tr><tr><td rowspan="2">InternVL-3-8B</td><td>Low-level</td><td>0.20</td><td>0.35</td><td>0.18</td><td>0.26</td><td>0.61</td><td>0.69</td></tr><tr><td>High-level</td><td>0.12</td><td>0.20</td><td>0.11</td><td>0.17</td><td>0.32</td><td>0.82</td></tr><tr><td rowspan="2">NaVid</td><td>Low-level</td><td>0.24</td><td>0.42</td><td>0.21</td><td>0.34</td><td>0.63</td><td>0.64</td></tr><tr><td>High-level</td><td>0.15</td><td>0.17</td><td>0.15</td><td>0.29</td><td>0.33</td><td>0.89</td></tr><tr><td rowspan="2">NaVILA</td><td>Low-level</td><td>0.56</td><td>0.66</td><td>0.50</td><td>0.58</td><td>0.48</td><td>0.75</td></tr><tr><td>High-level</td><td>0.39</td><td>0.47</td><td>0.34</td><td>0.48</td><td>0.61</td><td>0.68</td></tr></table>

Insight 1: 3DGS scene data renders faster than scanned mesh data but is harder to converge. We randomly selected 10k training samples and 1k validation samples from both traditional scanned mesh data and our 3DGS data, and conducted experiments with the NaVILA-base model on an NVIDIA H20 GPU. Tab. 3 compares the rendering speed and model convergence between scanned mesh VLN data and our 3DGS VLN data. The results show that 3DGS scene data achieves a perframe rendering time of 6.2 ms and an average memory usage of 220 MB, outperforming the 16.7 ms and 850 MB of scanned mesh data. However, in training, to reach the same 40% SR, the 3DGSbased model required about 160 iterations and 6.2 hours, while the scanned mesh-based model needed only about 120 iterations and 4.8 hours. This indicates that although 3DGS scene data offers faster rendering, it presents greater training difficulty and is relatively harder to converge.

Insight 2: 3DGS scene data exhibits strong generalizability. To evaluate the effectiveness of our novel 3DGS-based scene data, we tested the NaVILA-SAGE and NaVid-SAGE models, which were trained solely on our SAGE-Bench dataset, on the VLN-CE benchmark. As shown in Tab. 4, models trained entirely on SAGE-Bench data (without any VLN-CE data) achieved clear performance improvements over their respective baselines. For example, NaVILA-SAGE achieved a 31% relative SR improvement on R2R Val-Unseen (from 0.29 to 0.38) and a 34% relative OSR improvement (from 0.38 to 0.51), with similar gains observed for the NaVid model.

Insight 3: Our newly proposed three continuity metrics enable effective study of navigationâs natural continuity, filling key gaps left by conventional metrics. In Tab. 2, we report results for our three navigation natural continuity metrics. We observe that CSR is generally higher than SR, indicating a more inclusive and robust metric that does not require the model to fit the ground-truth trajectory exactly. For ICP and PS, although NaVILA attains relatively high task completion (0.39 SR, 0.47 OSR), it lacks natural motion continuity: an ICP of 0.61 indicates sustained collisions during navigation, and a PS of 0.68 reflects large, mechanical turning angles rather than smooth, natural motion. Additional visual examples in Fig. 4 corroborate this finding: the NaVILA model (blue trajectory) exhibits unsmooth movement and persistent collisions that conventional metrics fail to reveal. For instance, in Case 1, the model hugs the wall for a long period, yet the collision rate CR is only 1, while our ICP reaches 0.87.

Table 6: Impact of the number of scenes and samples on model performance.
<table><tr><td colspan="2">Data in # Train</td><td colspan="7">SAGE-Bench VLN</td></tr><tr><td>#Scenes</td><td>#Samples</td><td>SR â</td><td>OSR â</td><td>SPL â</td><td>CSR â</td><td></td><td>ICP â</td><td>PS â</td></tr><tr><td>800</td><td>240k</td><td>0.42</td><td>0.47</td><td>0.42</td><td></td><td>0.50</td><td>0.61</td><td>0.63</td></tr><tr><td>800</td><td>120k</td><td>0.40</td><td>0.43</td><td></td><td>0.40</td><td>0.48</td><td>0.62</td><td>0.62</td></tr><tr><td>800</td><td>60k</td><td>0.36</td><td>0.42</td><td></td><td>0.38</td><td>0.46</td><td>0.64</td><td>0.58</td></tr><tr><td>400</td><td>120k</td><td>0.34</td><td>0.39</td><td></td><td>0.35</td><td>0.44</td><td>0.67</td><td>0.54</td></tr><tr><td>400</td><td>60k</td><td>0.31</td><td>0.37</td><td></td><td>0.33</td><td>0.43</td><td>0.67</td><td>0.52</td></tr><tr><td>400</td><td>30k</td><td>0.28</td><td>0.35</td><td></td><td>0.31</td><td>0.43</td><td>0.69</td><td>0.49</td></tr><tr><td>400</td><td>15k</td><td>0.25</td><td>0.31</td><td></td><td>0.27</td><td>0.39</td><td>0.70</td><td>0.46</td></tr><tr><td>200</td><td>60k</td><td>0.27</td><td>0.33</td><td></td><td>0.29</td><td>0.41</td><td>0.70</td><td>0.47</td></tr><tr><td>100</td><td>60k</td><td>0.23</td><td>0.29</td><td></td><td>0.26</td><td>0.38</td><td>0.71</td><td>0.44</td></tr><tr><td colspan="2">NaVILA-base</td><td></td><td>0.21</td><td>0.26</td><td>0.22</td><td>0.36</td><td>0.72</td><td>0.41</td></tr></table>

<!-- image-->  
Figure 5: Model performance change curve (number of scenes vs. sample size).

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 6: Results under Different Evaluation Slice.

## 4.3 MORE FINDINGS

High-level Instructions vs. Low-level Instructions. Tab. 5 compares the performance of different models on high-level and low-level instructions in the VLN task.Compared with low-level instruction data, which are composed of basic step-by-step actions that guide the model gradually through the task, VLN models perform worse when executing high-level instructions. Even the recent SOTA model NaVILA achieves only a 0.39 success rate on high-level instructions, significantly lower than its 0.56 success rate on low-level instructions. Notably, high-level instructions, with their more natural semantics, are closer to real-life scenarios, presenting greater challenges for the future development of VLN models.

Number of training Scenes vs. Training Sample Size. Tab. 6 and Fig. 5 illustrate the influence of varying the number of scenes and the number of samples. We observe that increasing the number of scenes in the training data, while keeping the sample size constant, yields greater performance gains than merely increasing the number of samples. Specifically, with the same number of augmented scenes (800), increasing the sampling density progressively improves the VLN modelâs performance on the val-unseen split. Conversely, generating the same number of samples (700k) from a larger number of environments produces better results. These findings indicate that the number of scenes (Scenes) has a greater impact than the number of samples (Samples), suggesting that diversity of environments is more critical for learning VLN.

Results under Different Evaluation Slice. Based on our three-axis evaluation framework, we further present experimental results in Fig. 6 for different high-level instruction types, trajectory lengths, and scene complexities. The results show that VLN models perform worse on the âRelative Relationshipâ and âAttribute-basedâ instruction types, with SR scores for both NaVILA and NaVid more than 2% lower than those for other types. In addition, as trajectory length increases and scene complexity grows, model performance drops significantly.

## 5 RELATED WORK

Vision-and-Language Navigation (VLN) was first introduced by (Anderson et al., 2018) on early Matterport3D-based discrete panoramic graphs, later extended to multilingual / longer-horizon settings by Ku et al. (2020) and remote object grounding by Qi et al. (2020); research shifted to continuous control with (Krantz et al., 2020) (VLN-CE) on Habitat (Savva et al., 2019), though mainstream benchmarks still rely on scan-mesh reconstructions (with texture/semantic limitations). 3D Gaussian Splatting (3DGS)ârepresenting scenes efficiently via anisotropic Gaussian primitives for photorealistic real-time renderingâhas been integrated into embodied learning, such as coupling with MuJoCo/Isaac Sim (Jia et al., 2025; Zhu et al., 2025b), adopting dual-representation (Gaussians for rendering, meshes for collision) (Lou et al., 2025; Wu et al., 2025b), and enhancing with lighting estimation (Phongthawee et al., 2024); however, native 3DGS lacks object-level semantics, needs cumbersome manual appearance/physics alignment, and struggles with precise VLN language grounding (Krantz et al., 2020; Savva et al., 2019).

## 6 CONCLUSION

We presented SAGE-3D, a paradigm that upgrades 3D Gaussian Splatting from a purely perceptual scene representation to an executable, semantically and physically aligned environment foundation for embodied navigation. We release InteriorGS, the first large-scale dataset of 1K fully furnished indoor 3DGS reconstructions with dense object-level annotations, which enables robust semantic grounding in photorealistic environments. By unifying InteriorGS with a physics-aware execution layer and a hierarchical instruction-evaluation benchmark, SAGE-Bench, our framework provides a coherent pipeline from high-fidelity data generation to physically valid evaluation. We expect SAGE-3D to serve as a foundation for future research in richer multi-step and semantic-aware navigation tasks, interactive manipulation, and broader sim-to-real studies.

## REFERENCES

Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sunderhauf, Ian Reid, Â¨ Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3674â3683, 2018.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.

Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. International Conference on 3D Vision (3DV), 2017.

Guikun Chen and Wenguan Wang. A survey on 3d gaussian splatting, 2025. URL https:// arxiv.org/abs/2401.03890.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Xueyan Zou, Jan Kautz, Erdem Biyik, Hongxu Yin, Sifei Liu, and Xiaolong Wang. Navila: Legged robot vision-language-action model for navigation. In RSS, 2025.

Anurag Dalal, Daniel Hagen, Kjell G. Robbersmyr, and Kristian Muri Knausgard. Gaussian splat- Ë ting: 3d reconstruction and novel view synthesis: A review. IEEE Access, 12:96797â96820, 2024. doi: 10.1109/ACCESS.2024.3408318.

Chen Gao, Liankai Jin, Xingyu Peng, Jiazhao Zhang, Yue Deng, Annan Li, He Wang, and Si Liu. Octonav: Towards generalist embodied navigation, 2025. URL https://arxiv.org/abs/ 2506.09839.

Antoine Guedon and Vincent Lepetit. Sugar: Surface-aligned gaussian splatting for efficient 3d Â´ mesh reconstruction and high-quality mesh rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5354â5363, 2024.

Yufei Jia, Guangyu Wang, Yuhang Dong, Junzhe Wu, Yupei Zeng, Haonan Lin, Zifan Wang, Haizhou Ge, Weibin Gu, Kairui Ding, Zike Yan, Yunjie Cheng, Yue Li, Ziming Wang, Chuxuan Li, Wei Sui, Lu Shi, Guanzhong Tian, Ruqi Huang, and Guyue Zhou. Discoverse: Efficient robot simulation in complex high-fidelity environments, 2025. URL https://arxiv.org/ abs/2507.21981.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis. 3d gaussian splat- Â¨ ting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), July 2023. URL https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/.

Mukul Khanna\*, Ram Ramrakhya\*, Gunjan Chhablani, Sriram Yenamandra, Theophile Gervet, Matthew Chang, Zsolt Kira, Devendra Singh Chaplot, Dhruv Batra, and Roozbeh Mottaghi. Goatbench: A benchmark for multi-modal lifelong navigation. In CVPR, 2024.

Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Vision-and-language navigation in continuous environments. In Computer Vision â ECCV 2020: 16th European Conference, Glasgow, UK, August 23â28, 2020, Proceedings, Part XXVIII, pp. 104â120, Berlin, Heidelberg, 2020. Springer-Verlag. ISBN 978-3-030-58603-4. doi: 10.1007/ 978-3-030-58604-1 7. URL https://doi.org/10.1007/978-3-030-58604-1_7.

Jacob Krantz, Shurjo Banerjee, Wang Zhu, Jason Corso, Peter Anderson, Stefan Lee, and Jesse Thomason. Iterative vision-and-language navigation. arXiv preprint arXiv:2210.03087, 2022.

Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie, and Jason Baldridge. Room-across-room: Multilingual vision-and-language navigation with dense spatiotemporal grounding. arXiv preprint arXiv:2010.07954, 2020.

Haijie Li, Yanmin Wu, Jiarui Meng, Qiankun Gao, Zhiyao Zhang, Ronggang Wang, and Jian Zhang. Instancegaussian: Appearance-semantic joint gaussian representation for 3d instance-level perception, 2024. URL https://arxiv.org/abs/2411.19235.

Haozhe Lou, Yurong Liu, Yike Pan, Yiran Geng, Jianteng Chen, Wenlong Ma, Chenglong Li, Lin Wang, Hengzhen Feng, Lu Shi, et al. Robo-gs: A physics consistent spatial-temporal model for robotic arm with hybrid representation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 15379â15386. IEEE, 2025.

Pakkapon Phongthawee, Worameth Chinchuthakun, Nontaphat Sinsunthithet, Varun Jampani, Amit Raj, Pramook Khungurn, and Supasorn Suwajanakorn. Diffusionlight: Light probes for free by painting a chrome ball. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 98â108, 2024.

Yuankai Qi, Zizheng Pan, Shengping Zhang, Anton van den Hengel, and Qi Wu. Object-and-action aware model for visual language navigation. In European conference on computer vision, pp. 303â317. Springer, 2020.

Zhangyang Qi, Zhixiong Zhang, Yizhou Yu, Jiaqi Wang, and Hengshuang Zhao. Vln-r1: Visionlanguage navigation via reinforcement fine-tuning, 2025. URL https://arxiv.org/abs/ 2506.17221.

Santhosh Kumar Ramakrishnan, Aaron Gokaslan, Erik Wijmans, Oleksandr Maksymets, Alexander Clegg, John M Turner, Eric Undersander, Wojciech Galuba, Andrew Westbury, Angel X Chang, Manolis Savva, Yili Zhao, and Dhruv Batra. Habitat-matterport 3d dataset (HM3d): 1000 large-scale 3d environments for embodied AI. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https: //openreview.net/forum?id=-v4OuqNs5P.

Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, et al. Habitat: A platform for embodied ai research. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9339â9347, 2019.

Xinshuai Song, Weixing Chen, Yang Liu, Weikai Chen, Guanbin Li, and Liang Lin. Towards longhorizon vision-language navigation: Platform, benchmark and method. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Zhengren Wang. 3d representation methods: A survey, 2024. URL https://arxiv.org/abs/ 2410.06475.

Meng Wei, Chenyang Wan, Xiqian Yu, Tai Wang, Yuqiang Yang, Xiaohan Mao, Chenming Zhu, Wenzhe Cai, Hanqing Wang, Yilun Chen, et al. Streamvln: Streaming vision-and-language navigation via slowfast context modeling. arXiv preprint arXiv:2507.05240, 2025.

Xinyue Wei, Minghua Liu, Zhan Ling, and Hao Su. Approximate convex decomposition for 3d meshes with collision-aware concavity and tree search. ACM Transactions on Graphics (TOG), 41(4):1â18, 2022.

Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas Moenne-Loccoz, and Zan Gojcic. 3dgut: Enabling distorted cameras and secondary rays in gaussian splatting. Conference on Computer Vision and Pattern Recognition (CVPR), 2025a.

Yuxuan Wu, Lei Pan, Wenhua Wu, Guangming Wang, Yanzi Miao, Fan Xu, and Hesheng Wang. Rl-gsbridge: 3d gaussian splatting based real2sim2real method for robotic manipulation learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 192â198. IEEE, 2025b.

Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, and Angjoo Kanazawa. gsplat: An open-source library for gaussian splatting. Journal of Machine Learning Research, 26(34):1â17, 2025.

Naoki Yokoyama, Ram Ramrakhya, Abhishek Das, Dhruv Batra, and Sehoon Ha. Hm3d-ovon: A dataset and benchmark for open-vocabulary object goal navigation, 2024. URL https:// arxiv.org/abs/2409.14296.

Lu Yue, Dongliang Zhou, Liang Xie, Feitian Zhang, Ye Yan, and Erwei Yin. Safe-vln: Collision avoidance for vision-and-language navigation of autonomous robots operating in continuous environments. IEEE Robotics and Automation Letters, 9(6):4918â4925, June 2024. ISSN 2377- 3774. doi: 10.1109/lra.2024.3387171. URL http://dx.doi.org/10.1109/LRA.2024. 3387171.

Jiazhao Zhang, Kunyu Wang, Rongtao Xu, Gengze Zhou, Yicong Hong, Xiaomeng Fang, Qi Wu, Zhizheng Zhang, and He Wang. Navid: Video-based vlm plans the next step for vision-andlanguage navigation. Robotics: Science and Systems, 2024.

Duo Zheng, Shijia Huang, Lin Zhao, Yiwu Zhong, and Liwei Wang. Towards learning a generalist model for embodied navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13624â13634, 2024.

Gengze Zhou, Yicong Hong, Zun Wang, Xin Eric Wang, and Qi Wu. Navgpt-2: Unleashing navigational reasoning capability for large vision-language models. arXiv preprint arXiv:2407.12366, 2024.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025a.

Shaoting Zhu, Linzhan Mou, Derun Li, Baijun Ye, Runhan Huang, and Hang Zhao. Vr-robo: A real-to-sim-to-real framework for visual robot navigation and locomotion. IEEE Robotics and Automation Letters, 2025b.

Yicong Hong Yi Wang Qi Wu Mohit Bansal Stephen Gould Hao Tan Yu Qiao Zun Wang, Jialu Li. Scaling data generation in vision-and-language navigation. In ICCV 2023, 2023.

## APPENDIX

## OVERVIEW

This is the Appendix for the paper âTowards Physically Executable 3D Gaussian for Embodied Navigationâ. In this supplementary material we present:

â¢ The implementation details of trajectory generation and training are provided in Section A.

â¢ The detailed InteriorGS data sampling and construction is described in Section B.

â¢ The more specific explanation of the hierarchical instruction system and prompts are presented in Section C.

â¢ The comparison of our 3DGS-Mesh Hybrid Representation data with traditional Matterport 3D data is illustrated in Section D.

â¢ The more visualizations of InteriorGS scenes are shown in Section E.

â¢ The visualization of the data distribution of our InteriorGS is presented in Section F.

## A IMPLEMENTATION DETAILS

Trajectory Generation. We run A\*-based shortest-path search to generate trajectories with a cost function that integrates free-space distance, narrow-passage penalties, and area preferences to ensure both obstacle avoidance and task feasibility. To diversify the dataset, startâend pairs are sampled across different rooms, functional areas, and object instances, and a minimum safety distance is enforced to avoid overly close viewpoints that would reduce the richness of the visual signal.

Training. We selected 500k âtrajectoryâinstructionâ pairs from SAGE-Bench, with no overlap with the test set. We trained two models on this subset: one based on NaVILAâs pre-trained model navila-siglip-llama3-8b-v1.5-pretrain (denoted as NaVILA-base), producing NaVILA-SAGE; and the other based on Navidâs pre-trained model navid-7b-full-224 (denoted as NaVid-base), producing NaVid-SAGE. Training was carried out on 8 NVIDIA Tesla H20 GPUs with a batch size of 256 and a learning rate of $2 \times \mathrm { 1 0 ^ { - 5 } }$ . The training data did not include any VLN-CE R2R or RxR samples.

## B DETAILED SAMPLING METHOD OF INTERIORGS

To obtain reliable 3D Gaussian Splatting (3DGS) reconstructions in occlusion-rich indoor settings, we render on average $\sim 3 \small { , } 0 0 0$ camera views per scene with a ray tracing renderer and estimate 3DGS parameters using the renderer-provided poses via the open-source gsplat pipeline. To mitigate undersampling, we employ two complementary camera placement policies:

(1) Perimeter-aware floorplan sweeps (âsurroundâ). For each room polygon $P ,$ we generate m inwardly offset polygons $\{ P ^ { ( j ) } \} _ { j = 1 } ^ { m }$ according to a prescribed distance schedule, and allocate a global camera budget n across polygons proportionally to their perimeters. Along each $P ^ { ( j ) }$ cameras are uniformly spaced with optical axes aligned to the inward edge normals. At every placement, we instantiate three tangential baselines (left / center / right) and three vertical tiers: outer tiersâlower at 150 mm above the floor pitched +30â¦ (up), middle at mid-height with $0 ^ { \circ }$ pitch, and upper at 500 mm below the ceiling pitched $- 3 0 ^ { \circ }$ (down); interior tiers $( j > \bar { 1 } )$ )âheights are interpolated between the corresponding outer tiers, with upper tiers pitched â15â¦, lower tiers $+ 1 5 ^ { \circ }$ â¦ , and the middle tier matching the outer middle.

(2) Volume-uniform sampling. We distribute the global camera budget across rooms in proportion to room volume to favor coverage in smaller compartments, then draw 3D positions via Poissondisk sampling for space-filling uniformity. At each sampled position, six cameras with canonical yawâpitch templates are instantiated, and a shared small random perturbation is applied to their orientations. Together, these policies emphasize inward-facing, depth-aware viewpoints that broaden coverage and reduce undersampling-induced 3DGS underfitting.

To select viewpoints at appropriate distances from mesh surfaces, Figure 7 presents the camerasampling outcomes: viewpoints shown in green are retained as the final selections, whereas those in red are discarded for being too close to the nearest mesh surface (below a safety threshold).

<!-- image-->  
Figure 7: Camera pose sampling across four distinct floorplans. Green markers denote the final selected camera poses; red markers indicate poses discarded for being too close to the nearest mesh surface. Red outlines highlight ceilingâwall intersection regions, while white outlines indicate floorâwall intersections.

## C HIERARCHICAL INSTRUCTION GENERATION SCHEME

Grounded in 3DGS reconstructions and automatically generated 2D semantic top-down maps, we design a benchmarking-oriented hierarchical instruction generation scheme to close the gap left by prior VLN benchmarks that largely focus on low-semantic-granularity directives (e.g., âgo from A to Bâ or atomic action sequences).

## C.1 HIGH-LEVEL INSTRUCTIONS: TASK-SEMANTIC ORIENTED

The most representative subset of High-Level Instructions is Single-Goal Semantic Instructions, which enriches basic âFrom A to Bâ navigation trajectories with semantic meaning. This subset addresses the limitation of traditional VLN benchmarks by linking navigation goals to human daily scenarios, object properties, or spatial relationships. Detailed categories and examples are provided below:

(1) Add Object This category supplements a logical causal relationship between the start point and destination by introducing contextually relevant objects, making the navigation trajectory conform to human daily behavior. Without such causality, a directive like âfrom the sofa to the bookshelfâ lacks practical meaning; adding a causal object (e.g., âbooksâ) transforms it into a goal-driven task.

â¢ Case1: âPlease move the book from the coffee table to the bookshelf in the study.â

â¢ Case2: âPlease move the teacup from the coffee table to the bookshelf in the study.â

(2) Scenario Driven This category embeds a specific human-centric scenario or motive, framing the destination as a reasonable location to fulfill a practical need. The instruction directly reflects human intentions (e.g., thirst, hunger, rest), enabling the agent to associate navigation with task utility.

â¢ Case1: âIâm thirsty, please bring me a drink from the fridge.â

â¢ Case2: âI want to rest, please take me to the sofa in the living room.â

(3) Relative Relationship This category defines the target using relative spatial terms to distinguish similar or adjacent objectsâan essential capability for navigating cluttered environments (e.g., multiple chairs, tables). Common spatial terms include ânext to,â âbehind,â âthe one on the left,â âacross from,â and âin front of.â

â¢ Case1: âMove to the chair next to that table.â

â¢ Case2: âWalk to the cabinet across from the fridge in the kitchen.â

(4) Attribute-Based This category describes the target using perceivable, unique attributes to guide the agent in identifying a specific object among similar candidates. Attributes include color (e.g., âredâ), state (e.g., âopen,â âonâ), content (e.g., âempty,â âfullâ), size (e.g., âlargeâ), or decoration (e.g., âwith a flower patternâ).

â¢ Case1: âFind an empty table in the dining hall.â

â¢ Case2: âTurn off the lit table lamp in the bedroom.â

(5) Area-Based This category directs the agent to a general functional area rather than a specific object, focusing on spatial zones with practical purposes (e.g., cooking, resting, working). This is particularly useful for scenarios where the exact target object is unspecified but the functional context is clear.

â¢ Case1: âWalk from here to the kitchen area.â

â¢ Case2: âNavigate to the lounge area in the living room.â

## C.2 LOW-LEVEL INSTRUCTIONS: BASIC NAVIGATION & ACTION ORIENTED

Complementing the task-semantic focus of High-Level Instructions, Low-Level Instructions prioritize fundamental kinematic control and goal-directed point-to-point navigation without embedding complex contextual or functional semantics. These instructions serve two core purposes in our VLN framework: (1) evaluating an agentâs basic motion execution capability (e.g., precise rotation, step control) and (2) providing a foundational navigation substrate for higher-level semantic tasksâacting as the âexecution layerâ that translates abstract High-Level goals into concrete movements. Unlike High-Level Instructions that answer âwhy to navigate,â Low-Level Instructions focus solely on âhow to moveâ or âwhere to go (without context).â

Below are the two primary categories of Low-Level Instructions, each tailored to assess distinct aspects of an agentâs low-level navigation competence:

## C.2.1 1. BASE-ACTION: FUNDAMENTAL CONTROL BEHAVIORS

This category consists of goal-free primitive motions that test an agentâs ability to execute basic locomotor or rotational commands with precision. These actions lack any spatial target (e.g., no specific object or area to reach) and instead focus on refining motion accuracyâ a critical prerequisite for smooth, collision-free navigation in continuous environments. Common Base-Actions include step-based forward/backward movement and fixed-angle rotation.

â¢ Case1:âMove forward two steps.â

â¢ Case2:âTurn 90 degrees to the right in place.â

â¢ Case3:âTurn 180 degrees to the left in place.â

â¢ Case4:âMove backward one step.â

## C.2.2 2. SINGLE-GOAL: POINT-TO-POINT NAVIGATION

This category defines targeted point-to-point navigation tasks without additional semantic contextâfocusing solely on guiding the agent from a start location to a predefined end location. The end location can be a room, object, or functional zone, and the instruction is structured as a direct âgo from X to Yâ directive (or simplified to âgo to Yâ when the start location is implicit). This category is further subdivided based on the type of start and end targets, covering common indoor navigation scenarios:

â¢ Room-to-Room: Navigate between two functional rooms. Case1:âWalk to the bedroom.â Case2:âGo from the kitchen to the living room.â

â¢ Room-to-Object: Navigate from a room to a specific object within (or outside) the room. Case1:âWalk to the sofa in the living room.â Case2:âGo from the study to the chair on the balcony.â

â¢ Object-to-Object: Navigate between two distinct objects. Case1:âWalk from the table to the door.â Case2:âGo from the fridge to the dining table.â

â¢ Object-to-Room: Navigate from a specific object to a target room. Case1:âGo from the air conditioner to the kitchen.â Case2:âWalk from the desk to the bedroom.â

â¢ Zone-to-Zone: Navigate between two functional sub-zones within a larger space. Case1:âWalk from the center of the kitchen to the sink area.â Case2:âGo from the TV area in the living room to the window.â

These Single-Goal Low-Level Instructions are critical for benchmarking an agentâs spatial grounding ability (e.g., recognizing âbedroomâ or âsofaâ as navigation targets) without the confounding effects of semantic context, making them ideal for initial model training or control-focused evaluations.

## C.3 PROMPT FOR TRAJORIES TO INSTRUCTIONS

Prompt for Instruction Generation   
You are a specialized data annotator for robotics.   
Your mission is to act as a human providing natural language   
instructions for a home or service robot. You will generate a diverse   
set of human-centric navigation instructions (of INSTRUCTION TYPE   
ââHigh-Level-Deliverââ) based on a symbolic TEXT MAP, STARTING POINT,   
and END POINT.   
You need to generate at least 2--4 instructions for each of the seven   
INSTRUCTION TYPES defined below, ensuring variety and diversity.   
â¨Inputâ©   
1. TEXT MAP: A textual description of an environment, including   
named areas, objects, and their unique IDs (e.g., Bar counter 0,   
chair 5). This map is the single source of truth.   
2. STARTING POINT: The starting point of the trajectory,   
represented by an Object ID (e.g., chair 5).   
Example: ââstarting pointââ: ââchair 5ââ   
3. END POINT: The endpoint of the trajectory, represented by an   
Object ID (e.g., sofa 0).   
Example: ââend pointââ: ââsofa 0ââ   
<Task>   
Generate multiple natural language instructions for a trajectory from   
the STARTING POINT to the END POINT (an optimal short path obtained   
via A\*). Use the TEXT MAP to understand the environment.   
Generate at least 2--4 instructions for each of the INSTRUCTION TYPES   
below, ensuring diversity.   
<Principles>   
1. Donât Embellish or Exaggerate: You do not know the intermediate   
path points or turns. Do not invent waypoints (e.g., \pass   
through desk 2") or directional commands (e.g., \turn left")   
unless explicitly stated in the map.   
2. NEVER Use Internal IDs: Never include object IDs like chair 5.   
Instructions must be understandable to someone without the map.   
3. Stay Grounded in the Map: Do not invent objects, properties, or   
spatial relationships not described or reasonably inferable from   
the TEXT MAP.   
4. Be Natural and Concise: Use everyday language. Keep   
instructions between 5{20 words. Avoid robotic or overly formal   
phrasing.   
5. Be Creative and Diverse: Vary sentence structure, vocabulary,   
and perspective. Small wording changes should yield   
meaningfully different instructions.   
6. Avoid Repetition: Within each type, ensure instructions are   
semantically distinct|not just synonyms or minor rewordings.

7. Ensure Executability: Every instruction must be actionable   
using only the provided map.   
8. Strictly Adhere to Types: Each instruction must clearly match   
its assigned type definition.   
<Instruction Types>   
1. Add Object   
Description: Adds a reasonable causality object (e.g., an   
object to carry) to justify the movement.   
Examples:   
2. ââPlease move the book from the coffee table to the bookshelf in   
the study.ââ   
3. ââPlease move the teacup from the coffee table to the bookshelf   
in the study.ââ   
4. Scenario Driven   
Description: Embeds the instruction in a human-centered   
scenario or goal.   
Example: ââIâm thirsty, please bring me a drink from the   
fridge.ââ   
5. Relative Relationship   
Description: Uses relative spatial terms (e.g., âânext toââ,   
ââbehindââ, ââthe one on the leftââ) to identify the target.   
Example: ââMove to the chair next to that table.ââ   
6. Attribute-based   
Description: Describes the target using perceivable attributes   
(e.g., empty, with a vase, near a window).   
Example: ââFind an empty table in the dining hall.ââ   
7. Area-based   
Description: Directs the robot to a functional area rather than   
a specific object.   
Example: ââWalk from here to the kitchen area.ââ   
<Output Format>   
For each instruction type, generate 2--4 diverse instructions. Output   
as a JSON array:   
[   
{   
ââinstruction_typeââ: ââAdd_Objectââ,   
ââstartââ: ââ[provided_starting_object_id]ââ,   
ââendââ: ââ[provided_end_object_id]ââ,   
ââgenerated_instructionââ: ââ[instruction_text]ââ   
},   
{   
ââinstruction_typeââ: ââArea-basedââ,   
ââtrajectory_idââ: ââ[provided_id]ââ,   
ââstartââ: ââ[provided_starting_object_id]ââ,   
ââendââ: ââ[provided_end_object_id]ââ,   
ââgenerated_instructionââ: ââ[instruction_text]ââ   
},   
...   
]

## D COMPARISON OF OUR DATA WITH MATTERPORT3D

In this section, we compare our 3DGS-Mesh Hybrid Representation with traditional Matterport3D data. As shown in Fig. 8, Matterport3Dâs mesh, derived from scanning, exhibits clear boundary ambiguity and object interpenetration, whereas our data uses collision bodies from the original mesh via convex decomposition, representing the ground truth. Rendered RGB images also show that our data is more photorealistic.

<!-- image-->  
Matterport 3D Estimated Mesh

<!-- image-->  
Our Ground Truth Mesh

<!-- image-->  
Matterport 3D Rendering Scene

<!-- image-->  
Ours High-quality Rendering Scene  
Figure 8: Comparison of Our data with Matterport3D.

## E MORE VISUALIZATION OF INTERIORGS

This section presents additional InteriorGS scenes. As shown in Fig. 9, these scenes are highly detailed and photorealistic, demonstrating the high quality of our indoor data. We anticipate that InteriorGS will become a foundation for future embodied learning research.

<!-- image-->  
Figure 9: More Visualization of InteriorGS.

<!-- image-->  
Figure 10: Distribution of non-home scenes of InteriorGS.  
Figure 11: Distribution of assets of InteriorGS.

## F DISTRIBUTION OF DATA FROM OUR INTERIORGS

In this section, we further detail InteriorGSâs data distribution. Fig. 10 presents the distribution of 244 non-home scenes, categorized by function into Services, Office, Retail, Entertainment, and Fitness; Fitness has the fewest scenes, while the others are similarly distributed. Fig. 11 shows the asset distribution, including Furniture, Lighting, Food & Drinks, Daily Items, Decorations, and Others; books within Others are the most numerous assets.