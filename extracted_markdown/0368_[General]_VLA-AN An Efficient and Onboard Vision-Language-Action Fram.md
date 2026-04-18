# VLA-AN: An Efficient and Onboard Vision-Language-Action Framework for Aerial Navigation in Complex Environments

Yuze Wuâ ,1,2, Mo Zhuâ ,1,2, Xingxing Li2, Yuheng Du2, Yuxin Fan1,2, Wenjun Li1,2, Zhichao Han1,2, Xin Zhouâ,2, Fei Gaoâ,1,2

1 Zhejiang University, 2 Differential Robotics.

â These authors contributed equally to this work.

âCorresponding author. E-mail: iszhouxin@zju.edu.cn (Xin Zhou), fgaoaa@zju.edu.cn (Fei Gao)

## Abstract

This paper proposes VLA-AN (Vision-Language-Action framework for Aerial Navigation), an efficient and onboard Vision-Language-Action (VLA) framework dedicated to autonomous drone navigation in complex environments. VLA-AN addresses four major limitations of existing large aerial navigation models: the data domain gap, insufficient temporal navigation with reasoning, safety issues with generative action policies, and onboard deployment constraints. First, we construct a high-fidelity dataset utilizing 3D Gaussian Splatting (3D-GS) to effectively bridge the domain gap. Second, we introduce a progressive three-stage training framework that sequentially reinforces scene comprehension, core flight skills, and complex navigation capabilities. Third, we design a lightweight, real-time action module coupled with geometric safety correction. This module ensures fast, collision-free, and stable command generation, mitigating the safety risks inherent in stochastic generative policies. Finally, through deep optimization of the onboard deployment pipeline, VLA-AN achieves a robust real-time inference rate of 2â3 Hz on resource-constrained UAVs. Extensive experiments demonstrate that VLA-AN significantly improves spatial grounding, scene reasoning, and long-horizon navigation, achieving a maximum single-task success rate of 98.1%, and providing an efficient, practical solution for realizing full-chain closed-loop autonomy in lightweight aerial robots.

Keywords Aerial Navigation, Vision-Language-Action Model, Multi-stage Training, Multimodal Reasoning, Onboard Deployment

## 1 Introduction

The rapid and concurrent advances of Multimodal Large Language Models (MLLMs) [1â4] have fundamentally reshaped the imagination for the intelligence and autonomy of Unmanned Aerial Vehicles (UAVs). The emerging paradigm envisions aerial systems endowed with human-like cognitiveâaction capabilities, such as interpreting complex natural language instructions, understanding fine-grained scene semantics, inferring nuanced task intentions, and generating context-aware action sequences. However, most existing UAV systems remain largely dependent on manual control or limited autonomy. Their designs typically follow a cascaded perceptionâmappingâplanningâcontrol pipeline, composed of carefully engineered modules tailored to specific tasks such as tracking [5], landing [6], or physical interaction [7]. While effective in constrained settings, this modular paradigm introduces two fundamental limitations. First, errors tend to accumulate across stages, degrading overall system robustness. Second, these systems lack the capacity to reason over open-ended language and high-level intent. Extending them to new tasks often requires substantial manual redesign, making adaptation slow and resource-intensive. Consequently, there is a pressing need for a more general and unified approach that can reliably translate abstract task descriptions into precise control commands, thereby supporting aerial navigation in complex environments.

<!-- image-->  
Figure 1: Overview of dataset, training, deployment for VLA-AN. First, VLA-AN constructs a large-scale high-fidelity dataset to bridge the data domain gap. Second, VLA-AN introduces a progressive three-stage training framework to sequentially strengthen spatial grounding, scene reasoning and long-horizon navigation. Third, VLA-AN produces fast, stable, and collisionfree action commands that can be executed onboard in real time at 2â3 Hz.

To address these limitations, data-driven navigation approaches based on Vision Language Action (VLA) [8, 9] or Vision Language Model (VLM) [10â12], have emerged as a promising direction for enhancing drone autonomy. By leveraging largescale pretraining and cross-modal alignment, these models significantly enhance semantic understanding and enable languageconditioned action generation. With recent advances in model compression techniques such as quantization and knowledge distillation [13,14], the resulting models have become increasingly acceptable for deployment on onboard platforms. However, most existing VLM/VLA systems are primarily developed for ground-based platforms with fixed viewpoints or for relatively open and simple outdoor environments. As a result, they fall short of meeting the distinctive demands of aerial navigation, particularly in scenarios involving indoor flight, aggressive maneuvers, and tightly constrained spaces. We outline the core challenges inherent in deploying large navigation models on agile drones as follows:

1) Data Distribution Mismatch. Most large models are pretrained on static images or data captured from fixed or height-constrained viewpoints. This training distribution differs substantially from the highly dynamic, first-person aerial perspective encountered during flight, leading to degraded semantic perception and unstable spatial localization. In addition, acquiring high-quality navigation data from real UAVs is inherently expensive and entails considerable operational risk, further limiting the availability of representative training data.

2) Insufficient Temporal Reasoning for Navigation. Existing approaches predominantly rely on single-frame inference and have limited ability to encode temporal context. As a result, they struggle to exploit historical observations, reason under complex scenes, or execute long-horizon navigation tasks. These shortcomings hinder stable and continuous operation in real-world environments. Moreover, the reliance on preconstructed or known maps further restricts adaptability when operating in previously unseen spaces.

3) Safety Limitations in Generative Action Models. Many state-of-the-art VLA models often utilize generative models, such as diffusion policy [15] or flow matching [16], to generate continuous control sequences. While flexible, these approaches introduce stochasticity and generative noise that can significantly increase collision risk, particularly in confined environments. Additionally, these models struggle to incorporate explicit geometric constraints in training collision-free action generation in constrained 3D space.

4) Constraints on Onboard Deployment. Current VLA models [8] impose substantial computational demands, typically requiring high-performance GPUs, which limits their deployment on UAVs with constrained onboard compute and strict payload restrictions. Although approaches such as model distillation [17] can partially alleviate these constraints, they often incur performance degradation. Consequently, efficient and lightweight frameworks specifically designed for resource-limited UAV deployment remain largely underdeveloped.

To systematically address these challenges, we propose an efficient and onboard VLA framework for Aerial Navigation (named VLA-AN) in complex environments, establishing a unified pipeline that integrates high-fidelity data generation, multi-stage training, robust action generation, and onboard deployment to achieve full-chain autonomous aerial navigation.

Collecting real-world UAV navigation data is labor-intensive; for instance, UAV-FLOW Colosseo [18] spends several weeks to acquire 30K trajectories. Conventional simulators, such as AerialVLN [19], often lack photorealism, with limited lighting fidelity and texture realism. To bridge the domain gap between real and synthetic data, we employ 3D Gaussian Splatting (3D-GS) [20] to construct high-fidelity navigation data. Compared with standard mesh-based rendering, 3D-GS offers superior continuous geometry representation, consistent lighting, and high rendering efficiency, producing synthetic visual scenes that more closely resemble the real world. Leveraging 3D-GS, we build a large-scale, high-fidelity multimodal navigation dataset that spans diverse indoor and outdoor environments, illumination conditions, occlusion patterns, and dynamic elements. The resulting dataset comprises over 100K navigation trajectories and more than 1M multimodal samples, providing a unified foundation for learning robust semantic navigation across scenes and viewpoints.

Building upon this hybrid dataset, as shown in Fig. 1, we introduce a progressive three-stage training framework for VLA-AN that systematically enhances navigation with temporal reasoning, particularly long-horizon reasoning, spatial grounding, and fine-grained navigation. Stage I conducts supervised fine-tuning (SFT) on large-scale multimodal data to strengthen scene comprehension, logical reasoning, and spatial inference. Stage II incorporates navigation-centric data to impart essential flight skills, including 3D waypoints generation, yaw prediction, and dynamic re-planning, thereby improving robustness in cluttered environments. Inspired by recent advances in leveraging reinforcement learning to enhance large model capabilities [21, 22], Stage III applies reinforcement learning fine-tuning (RFT) to further refine complex decision-making and enable precise navigation under challenging conditions. Collectively, this framework equips the model to reliably follow complex instructions and perform accurate aerial navigation in real-world scenarios.

A stable action generation module is also essential for reliable navigation. VLA-AN incorporates a real-time robust action module that performs continuous command generation coupled with geometric safety correction. In contrast to large action experts (e.g., the 0.3B flow-matching policy used in Ï0 [9]), our design eliminates inference-latency bottlenecks and enables fast, collision-free navigation in dense and previously unseen environments. Remarkably, VLA-AN can run in real time on lightweight onboard inference platforms, enabled by extensive optimizations throughout our deployment pipeline. These improvements allow the system to sustain inference rates of 2â3 Hz under continuously varying inputs on a 100 TOPS computation board. Our ablation studies further show that the optimized model attains performance comparable to the general pipeline, demonstrating a practical and efficient pathway for real-time onboard deployment on lightweight aerial robots.

Contributions. We propose VLA-AN, an efficient and onboard Vision-Language-Action framework for aerial navigation in complex environments. First, we construct a large-scale, high-fidelity 3D-GS dataset for superior realism compared to meshbased methods, effectively bridging the data domain gap. Second, we introduce a progressive three-stage training framework to sequentially strengthen scene grounding, impart aerial navigation, and enhance complex long-horizon reasoning. Third, to overcome the safety limitations of stochastic generative policies, we design a lightweight, real-time action module coupled with geometric safety correction, ensuring fast, collision-free, and stable command generation without the latency bottlenecks of large action experts. Finally, through extensive system-level optimizations, we enable practical real-time onboard deployment on resource-constrained UAVs, achieving an 8.3Ã improvement in inference throughput and demonstrating a full-chain closedloop autonomy pathway for lightweight aerial robots. Extensive simulation and real-world experiments validate its reasoningenhanced navigation and long-horizon task execution, achieving an average success rate exceeding 90%.

## 2 Related Works

## 2.1 Traditional UAV Navigation

Early UAV autonomous navigation predominantly relied on modular, cascaded architectures following the perceptionâdecisionâ planningâcontrol paradigm [23â26]. Localization and environmental perception were typically achieved through LiDAR- or vision-based modules [27, 28], while scene semantics were extracted using object detectors such as YOLO [29], Faster R-CNN [30], and SceneGraph [31]. High-level task logic was then resolved by rule-based or tree-structured decision modules, followed by motion plannersâsuch as A\* [32], RRT [33], Fast-Planner [34], and Ego-Planner [35]âto generate reference trajectories, which were finally executed by trajectory-tracking controllers to accomplish autonomous flight.

These approaches offer clear structure, strong interpretability, and low deployment cost. However, they are usually tailored for specific task scenarios such as target tracking [36], gap traversal [37], or fast landing [38], but lack the capability to interpret natural-language instructions or leverage the rich semantic information contained in visual observations. Moreover, the serial multi-module pipeline makes the system prone to error accumulation, resulting in notable performance degradation in confined spaces, dynamic environments, and complex three-dimensional spaces.

## 2.2 Vision-Language-Action Navigation for Drones

Recent advances in large-scale language models have enabled a new paradigm for language-driven UAV autonomous navigation, allowing drones to perform cross-modal reasoning between visual semantics and linguistic instructions [39, 40]. Representative works include AerialVLN [19], OpenUAV [41], CityNav [42], and more recent frameworks such as FlightGPT [43] and NavAgent [44], which have demonstrated the potential of VLMs in city-scale navigation tasks.

Although existing efforts attempt to construct UAV-specific datasets and achieve instruction alignment (e.g., OpenFly [45], AVDN [46]), high-fidelity data with consistent aerial viewpoints remain extremely scarce, constraining the transferability and generalization of these aerial navigation models. Furthermore, mainstream models are trained primarily on static images or ground-view videos [47, 48], exhibiting a substantial domain gap from the UAVâs high-speed first-person aerial perspective [49, 50], which often leads to semantic drift and spatial errors. Most methods rely on single-frame inference and lack temporal modeling capabilities, making them incapable of handling occlusions, historical states, and task progress [39, 40]âultimately resulting in behavioral deviations during continuous control. Additionally, many models depend on pre-built maps, limiting their applicability to unknown environments [41], and the absence of real-time replanning mechanisms further reduces stability in fast-changing scenes.

## 2.3 Onboard Foundation Models for Robotics

With recent advances in edge computing, deploying foundation modelsâsuch as LLMs [2], VLMs [10], and VLA models [8]â directly onboard robotic platforms [51, 52], particularly UAVs, has become an emerging trend. Typical approaches utilize quantization, pruning, distillation, and operator fusion to reduce inference cost in order to meet requirements on privacy, safety, and real-time performance [53]. However, it is highly challenging to maintain cross-modal alignment and to achieve lowlatency inference on onboard limited computation platforms. For UAV platforms with stringent payload constraints, the model must simultaneously perform visual understanding, language reasoning, and action generation within a unified architecture, imposing extreme demands on computational efficiency and energy consumption.

Existing large-scale robotic models (e.g., the RT models [54, 55], OpenVLA [8], Ï0 series [9, 56]) rely heavily on highperformance GPUs and cannot be directly deployed on UAVs. Although recent studies have explored lightweight ViTs [57], policy distillation [58], mixed-precision inference [59], and hardware âoptimized accelerators for onboard deployment, ensuring closed-loop control stability under aggressive quantization remains an unsolved problem. Overall, despite notable progress in onboard foundation models, meeting the stringent accuracy, robustness, and latency requirements under UAV resource constraints remains a major bottleneck. This calls for UAV-oriented efficient model architectures, multimodal compression strategies, and deployable real-time inference frameworks.

## 3 Methods

As illustrated in Fig. 1, we construct a hybrid dataset enriched with 3D-GS navigation data to support efficient model training and enhance spatial understanding across diverse viewpoints. The proposed VLA-AN framework adopts a hierarchical architecture that couples semantic reasoning with fine-grained spatial navigation, enabling robust performance on complex, long-horizon tasks under open-vocabulary instructions. Given a natural-language command (e.g., âgo to the right side of the white chair with clothes on itâ), the model identifies the referenced target entity (the white chair), extracts its attribute constraints (with clothes), parses the corresponding spatial relation (âright sideâ), and subsequently generates a 3D navigation waypoints along with the desired yaw orientation. By comparing the current visual observation with the initial frame, the model continuously evaluates task progress; if the target has not yet been reached, it initiates a global replanning process to update the navigation goal until successful completion. To ensure safer and more robust action generation, as shown in Fig. 2, the projector module and the action module jointly validate and refine the local action sequences, improving obstacle avoidance and guaranteeing secure, efficient UAV maneuvering in cluttered environments. Finally, with extensive optimization for on-device inference, the full VLA-AN system is deployed on a lightweight onboard computing platform, achieving real-time, high-precision autonomous navigation in real-world scenarios.

<!-- image-->  
Figure 2: Model architecture of VLA-AN. The VLA-AN model architecture is designed for navigation with reasoning, effectively processing single images and multiple images, and consists of four main components: (1) a Vision Transformer for encoding visual inputs; (2) a MLP projector that maps visual encodings to a latent space aligned with a LLM; (3) the LLM itself for textual understanding and reasoning; and (4) the projector and action module check the desired commands and generate action sequences.

## 3.1 Problem Definition

The vision-language navigation task aims to enable an aerial robot to execute autonomous behaviors in unknown environments by interpreting natural language commands and real-time onboard perception. Formally, given a language instruction L and multimodal observations $\mathcal { O } _ { t } = \{ I _ { t } ^ { r g b } , I _ { t } ^ { d e p t h } , p _ { t } \}$ , where $I _ { t } ^ { r g b }$ and $I _ { t } ^ { d \bar { e } p t h }$ denote the RGB and depth images, and $p _ { t } = ( x _ { t } , y _ { t } , z _ { t } , \psi _ { t } )$ represents the UAV pose, the objective is to learn a VLA model ÏVLA-AN that generates an action sequence $a _ { 1 : T }$ maximizing the task success probability P (success $L , { \mathcal { O } } _ { 1 : T } )$

$$
\pi _ { \operatorname { V L A - A N } } = \arg \operatorname* { m a x } _ { \pi } ~ \mathbb { E } _ { a _ { 1 : T } \sim \pi ( \cdot | L , \mathcal { O } _ { 1 : T } ) } \left[ P ( \operatorname { s u c c e s s } \mid L , \mathcal { O } _ { 1 : T } , a _ { 1 : T } ) \right]\tag{1}
$$

Unlike conventional modular pipelines, our framework directly employs the VLA Model to map visual-linguistic inputs to semantic control actions:

$$
a _ { t } = \pi _ { \mathrm { V L A - A N } } ( L , I _ { t } ^ { r g b } , I _ { t } ^ { d e p t h } ) .\tag{2}
$$

To maintain task consistency, a temporal comparison module evaluates task completion by contrasting the initial and current frames:

$$
s _ { t } = f _ { \mathrm { c m p } } ( I _ { 0 } ^ { r g b } , I _ { t } ^ { r g b } , L ) ,\tag{3}
$$

where $s _ { t } \in \{ 0 , 1 \}$ denotes the task completion flag. The overall UAV system is formulated as a state transition process:

$$
\mathcal { S } = \{ \mathrm { I D L E } , \mathrm { N A V I G A T I N G } , \mathrm { R E P L A N N I N G } , \mathrm { T A S K } . \mathrm { C O M P L E T E } \} ,\tag{4}
$$

ensuring closed-loop autonomous coordination.

This work introduces a comprehensive navigation framework for UAVs in unknown environments, consisting of four core components: a high-fidelity vision-language dataset, a three-stage training procedure, a robust action module, and an onboard deployment framework.

## 3.2 High-Fidelity Hybrid Data Collection

To bridge the significant domain gap between existing datasets and real-world scenarios regarding visual realism and UAV motion characteristics, we propose an automated data generation pipeline integrating 3D-GS with the Unity engine. This pipeline is designed to generate not only high-fidelity visual observations but also flight trajectories that strictly conform to UAV dynamics. The specific construction process is as follows:

Stage I: High-fidelity scene reconstruction via 3D-GS. Distinct from traditional mesh-based rendering, we employ 3D-GS technology to capture real-world illumination variations and continuous geometric details. We initially collect video streams covering diverse indoor (e.g., offices, corridors) and outdoor (e.g., parks, streets) environments, converting them into highquality 3D-GS scene representations before importing them into Unity. The 3D-GS approach preserves real-world visual textures while endowing the simulator with exceptional rendering efficiency and realism, thereby significantly narrowing the sim-to-real visual gap.

Stage II: Automated trajectory generation and multimodal data collection. To generate large-scale, semantically aligned, and physically constrained navigation data, we establish a closed-loop automated collection system:

Task Definition and Randomization: First, human experts formulate the metadata for each subtask, including feasible natural language instructions, trajectory start points, and target endpoints. To enhance data diversity, we perform random sampling of start and end coordinates within reasonable ranges defined by experts, ensuring that tasks remain both challenging and solvable.

Kinodynamic Planning: Leveraging the gradient-based trajectory planner, we generate reference trajectories within complex 3D environments. This algorithm fully exploits the spatial maneuverability of UAVs; it not only handles horizontal obstacle avoidance but also flexibly navigates through the Z-axis to evade vertical obstacles, achieving collision-free planning within the full volumetric space.

Multi-view Synchronous Data Collection: Within the Unity simulation, the UAV agent flies autonomously along the planned trajectory. The system synchronously records multimodal data at 10 Hz, encompassing RGB images from four views (front, rear, left, right), depth maps, and the UAVâs 4D pose in the world coordinate system.

Stage III: Hybrid Dataset Construction. To strike a balance between sim-to-real visual consistency and environmental diversity in training data, we proportionally blend high-fidelity 3D-GS data with editable mesh data and real-world data. Specifically, 3D-GS scenes effectively bridge the visual domain gap via their photorealism, whereas mesh scenes leverage their high editability to enable flexible adjustments of lighting conditions, occlusion patterns, and target objects. This complementary strategy ensures the modelâs adaptability to real-world visual features while significantly enhancing robustness against complex and changing environments through domain randomization.

## 3.3 Multi-stage Training for VLA

To address the coexistence of multiple tasks with significantly different scales, we design a progressive three-stage training framework that systematically enhances cross-task generalization for aerial navigation. This framework integrates multi-frame visual question answering, spatial grounding and reasoning data, as well as high-quality aerial navigation demonstrations. By combining supervised fine-tuning (SFT) with reinforcement learning fine-tuning (RFT), the framework gradually strengthens the modelâs spatial understanding, task planning, and high-level reasoning capabilities in complex aerial scenarios, ultimately enabling a unified VLA-based aerial navigation agent.

Stage I: Grounding-reasoning-enhanced SFT. In the first stage, we perform full-parameter SFT on a mixture of large-scale general-purpose datasets, including VQA, spatial grounding, reasoning, and STEM tasks, to enhance the modelâs fundamental visual understanding, logical reasoning, and spatial relation modeling capabilities. The training corpus further incorporates multi-frame and multi-view image reasoning, temporal consistency modeling across action sequences, and complex aerialscene analysis. These designs enable the model to expand from object-level perception to scene-level spatial inference, forming a multi-scale understanding mechanism that supports high-level navigation planning. This stage equips the model with essential capabilities such as interpreting natural-language instructions, understanding physical interactions, and parsing real-world spatial layouts, laying a robust foundation for subsequent navigation-specialized training.

Stage II: Navigation-specific SFT. The second stage injects high-quality UAV navigation data and proportionally mixes curated VQA reasoning data to construct intensive post-training tailored for aerial navigation. This stage explicitly teaches core UAV skills required in real-world environments, including 3D waypoints planning, desired yaw prediction, and dynamic task re-planning. The model undergoes SFT to maintain broad reasoning abilities inherited from Stage I while adapting to the structured knowledge representation and multi-view observation patterns characteristic of navigation tasks. As a result, the model achieves stronger robustness in autonomous navigation under complex environments and establishes a stable semantic and command-level backbone for the subsequent reinforcement learning stage.

Stage III: RFT-enhanced navigation with reasoning. In the third stage, we apply a GRPO-based RFT strategy to further improve accuracy, reliability, and decision consistency. A reward design is constructed from a high-quality thinking dataset to supervise UAV behavior and reasoning processes. For each instruction, multiple responses are sampled, and advantage estimates are computed through within-group normalization. Reward functions are designed to cover visual reasoning, spatial pointing, command generation, etc; visual reasoning is evaluated via strict answer matching, while spatial tasks are assessed using the Intersection over Union (IoU) between predicted and ground-truth bounding boxes or masks. This stage emphasizes correcting failure patterns identified earlier, and under strict format-template validation, the model systematically learns to generate more stable, compliant, and high-confidence outputs. Consequently, the model achieves substantial improvements in executing complex instructions and performing reliable aerial navigation in real-world scenarios.

## 3.4 Robust Action Module

Existing VLA models rely on generative models to produce action chunking, but inference noise and bias may induce collisions in confined environments, posing significant risks to structurally fragile drones. To mitigate this, we introduce a robust real-time action module as a continuous command generator to ensure safe and accurate trajectory tracking. The module extracts local obstacle information from the depth map only when potential intersections between the trajectory and surrounding obstacles are detected, and generates differentiable repulsive gradient forces that instantly adjust the trajectory. This robust mechanism is computationally efficient and highly responsive, enabling UAVs to achieve low-latency, collision-free navigation in dense, previously unseen environments.

Specifically, the module constructs a reference trajectory from the current state to the target state using onboard sensor data. If the initial trajectory intersects with obstacles, local geometric cues are converted into surface anchors and repulsive directions to refine the flight path and control actions, enabling continuous collision-free adjustment. To preserve dynamic feasibility, the system subsequently re-estimates the temporal allocation and reconstructs the trajectory under safety constraints. Unlike conventional large generative modelsâsuch as the 0.3B flow-matching action expert used in the Ï0 model [9]âthe proposed module eliminates the inference-latency bottleneck inherent to such approaches. As a result, the system supports reliable high-speed flight, achieving sustained low-latency, safe, and robust navigation in cluttered and unknown environments.

## 3.5 Onboard Deployment Framework

Given the strict payload constraints of UAV platforms, most consumer-grade and high-performance computing modules remain impractical for onboard deployment. For instance, NVIDIA Jetson Thor introduces substantial mass relative to typical UAV payload limits, and excessive loading directly reduces flight endurance. Moreover, UAVs are frequently deployed in wide-area, communication-limited, or otherwise challenging environments, where long-duration autonomous operation imposes stringent requirements on onboard inference hardware. Under these constraints, we adopt the lightweight NVIDIA Jetson Orin NX (abbreviated as Orin NX) as our onboard computing unit. After onboard integration and optimization, the total added mass is approximately 80 grams, which is acceptable for micro-scale UAV platforms.

Despite its favorable sizeâpower characteristics, the ARM-based Orin NX offers limited support for the inference of general VLA models, particularly for Vision Transformer (ViT) workloads. In practice, some vision network inference still relies on CPU execution due to a lack of GPU acceleration support, which becomes a major computational bottleneck given the modest CPU performance of Orin NX. Moreover, inference acceleration techniques tailored to VLA models remain nascent; optimizations such as efficient Transformer execution, operator fusion, and KV-cache management have not yet been fully leveraged, preventing effective deployment of VLAs in resource-constrained edge platforms.

To address these challenges, we present a systematic investigation of inference performance optimization for VLA deployment on UAV platforms, with the goal of breaking through the real-time inference bottleneck on limited-resource devices. First, we target multiple acceleration mechanisms across the inference pipeline. Specifically, we integrate a Flash-Attention mechanism, fuse feed-forward network (FFN) layers with normalization operators, and employ a KV-cache pre-loading strategy. These optimizations streamline key computational paths such as memory access, redundant operator execution, and attention computation, thereby substantially reducing latency.

Next, we conduct a fine-grained analysis of computational resource usage across model components, revealing that the ViT module dominates the computational workload. The bottlenecks are twofold: the reliance on CPU for vision network inference, and the ARM architectureâs sub-optimal support for vision operators. To mitigate these, we perform tailored optimizations for ViTâs attention architecture, feature-map dimensionality, and operator scheduling. We restructure operator ordering, optimize memory access patterns, and leverage Single Instruction Multiple Data (SIMD) instruction sets to accelerate execution on ARM.

Furthermore, we incorporate CUDA graph to manage process-level parallelism and pipeline scheduling, reducing inter-process scheduling overhead and enabling the entire VLA inference chain to be executed rapidly within a single process. Collectively, these optimizations significantly enhance the energy efficiency and real-time responsiveness of VLA models on Orin NX, providing a viable deployment path for large models on resource-constrained robotic platforms.

Object navigation

<!-- image-->  
Head to the white circular column  
Precise navigation

<!-- image-->  
Go to the right of the chair on the left  
Spatial grounding

<!-- image-->  
Find the third tree from near to far  
Long-horizon navigation

<!-- image-->  
Head to the black chair, then the front circular pillar, and finally land on the carpet  
Scene reasoning

<!-- image-->  
Find a voltmeter with a reading of 48V  
Language control

<!-- image-->  
Turn 45 degrees to the right and then go forward 10 meters  
Target searching

<!-- image-->  
Search for higher electric fans  
Target tracking

<!-- image-->  
Follow the man in the black coat

Figure 3: Results of benchmark evaluations on diverse navigation tasks using OpenVLA [8], Ï0 [9], Groot N1.5 [60], and Ours (the bar chart values represent the success rate (%) ).

## 4 Results

To comprehensively evaluate the performance of VLA-AN, we conduct extensive experiments and ablation studies focusing on four key questions: (1) How does VLA-AN compare with state-of-the-art VLA methods? (2) How well does VLA-AN perform in real-world robotic environments? (3) What level of performance can be achieved under onboard, resource-constrained deployment? (4) Are the proposed design components essential and effective?

To achieve this, we conduct comprehensive benchmark tests in simulation, comparing against mainstream VLA methods, followed by extensive real-world experiments to validate the effectiveness and robustness of our approach. Additionally, we evaluate the inference performance of VLA-AN under onboard deployment. Finally, several ablation studies are performed to assess the contribution of key components in our framework.

## 4.1 Benchmark Results

We evaluate VLA-AN against several representative VLA baselines, including OpenVLA [8], Ï0 [9], and Groot N1.5 [60]. Experiments span eight representative navigation scenarios, covering both diverse seen and unseen tasks. Each navigation episode is conditioned on real-time RGB inputs from multiple onboard cameras along with a natural-language instruction. Success Rate (SR) is used as the primary metric. As shown in Fig. 3, VLA-AN consistently outperforms all baseline models across a wide range of aerial navigation tasks, even achieving an average SR exceeding 98% for object navigation tasks.

In open-vocabulary object navigation and precise navigation, VLA-AN achieves significant improvements over other models. This advantage arises from the fact that our method does not directly predict low-level motor commands of drones; instead, it outputs target 3D waypoints and desired yaw angles. As a result, the model can learn navigation-specific information from heterogeneous robot datasets (e.g., drones of varying configurations, and even humanoids, quadrupeds), greatly improving

1

âPlease locate the door on the right, then proceed to the farthest square table, and finally stop above the square table on the left.â

Onboard view

## 2

âPlease go to the sink, then find the first room at the back, and stop on the windowsill after entering the room.â

Onboard view

## 3

âPlease find the coffee machine, then locate the farthest table, and finally go to the entrance of the meeting room.â

Onboard view

<!-- image-->  
Figure 4: Simulation tests for navigation scenarios utilizing both visual and language inputs.

the training efficiency of the dataset. This abstraction eliminates the need for precise motor-level predictions and enables a robust action module to apply safe obstacle-avoidance constraints. In challenging precise navigation scenarios, such as narrow corridors, cluttered forests, or crowded spaces, models like OpenVLA, Ï0, and Groot N1.5 often produce unsafe motion, leading to numerous failures and a significant decrease in their success rates.

To validate the modelâs generalization and reasoning capabilities, we also include complex navigation tasks, such as spatial grounding, long-horizon navigation, scene reasoning, language control, and target searching. The test results, shown in Fig. 3, indicate that VLA-AN significantly outperforms other VLA models across all tasks. This improvement is largely attributed to our multi-stage training strategy, where we inject and enhance substantial reasoning and decision-making capabilities to improve navigation performance in complex scenarios. In contrast, we observe that OpenVLA and $\pi _ { 0 }$ tend to catastrophically forget their vision-language understanding during task-specific fine-tuning, which degrades multimodal alignment and semantic comprehension. Groot N1.5, due to freezing the VLM weights during training, exhibits stronger reasoning abilities compared to OpenVLA and Ï0. In spatial reasoning tasks, where the model needs to understand proximity and counting capabilities from multimodal inputs to map to the desired reference state, both OpenVLA and Ï0 achieve relatively low SR. For instance, in the task âFind the third tree from near to far,â the model must integrate distance relationships and counting capabilities, which are essential prerequisites for correct action execution.

VLA-AN, surprisingly, also demonstrates strong performance in target tracking, as shown in Fig. 3. Despite tracking-specific samples constituting less than 1% of the entire training dataset, the model maintains a stable target tracking capability, achieving an 82.0% SR on relevant tasks. This robust performance is hypothesized to stem from the fact that our navigation-oriented training implicitly promotes mission re-planning. This mechanism enables the model to autonomously infer whether the target state has changed (e.g., moved or been lost) by aggregating and integrating multi-frame temporal observations. In direct contrast, other existing VLA models exhibit considerably lower SRs in similar dynamic tracking scenarios.

## 4.2 Real-World Results

<!-- image-->  
Figure 5: Hardware and software architectures of two different experimental drones.

Real-world deployment: We deploy VLA-AN on two different UAV platforms, with detailed hardware and software architectures illustrated in Fig. 5. Each platform is equipped with an Intel RealSense camera for visual perception, and the onboard module performs all perception, language processing, and action generation. During flight, the robots asynchronously transmit the latest observations while executing the previously produced action sequences. In Fig. 6, we evaluate tasks including object navigation, spatial grounding, scene reasoning, object searching, and target tracking. The success rates remain comparable to those in simulation, demonstrating strong real-world generalization. We further assess long-horizon language instructions, such as âTurn left 90 degrees, then turn right toward the yellow cabinet, and finally locate the clock icon.â VLA-AN correctly decomposes sequential sub-tasks and reliably completes such multi-step instructions.

Onboard inference performance: As noted earlier, we adopt a lightweight NVIDIA Jetson Orin NX 16 GB module for onboard inference. This lightweight computing board is suitable for small-payload aerial robots and possesses approximately 100 TOPS of computing capability. We deploy VLA-AN models of different scales (7B/3B/2B) under the 30 W performance mode, and the performance results are summarized in Tab. 1. These results benefit from several system-level optimization strategies, including the integration of flash-attention mechanisms, FFNâNormer operator fusion, KV-cache preloading, and CUDA graph scheduling. Overall, inference throughput improves by 7.1Ãâ9.4Ã over the unoptimized baselines. During closedloop navigation, asynchronous image transmission combined with continuous action execution enables high-frequency realtime action inference, which is sufficient for agile autonomous flight in complex environments.

Table 1: Onboard inference performance on different models.
<table><tr><td>Model</td><td></td><td>Floating Point Operations Decode Inference Speed (Second Per Token)</td></tr><tr><td>7B-AWQ</td><td>â¼ 1815.8 GFLOPs</td><td>0.110</td></tr><tr><td>3B-AWQ</td><td>~ 946.7 GFLOPs</td><td>0.051</td></tr><tr><td>2B-AWQ</td><td>~ 563.3 GFLOPs</td><td>0.032</td></tr></table>

Overall, these findings demonstrate that VLA-AN maintains stable and efficient inference performance even under lightweight onboard computing resources, enabling practical real-world deployment of VLA models on drones.

## 4.3 Ablation Studies

Different navigation datasets: To investigate the effectiveness of 3D-GS data in bridging the domain gap between simulated and real-world visual perception, we evaluate three training configurations: only real-world data, only 3D-GS data, and only mesh data. Notably, the seen dataset contains all types of data, whereas the unseen dataset predominantly consists of real-world data, as our objective is to assess the modelâs generalization to real-world navigation. The results are summarized in Tab. 2.

## 1

âGo to the chair where the clothes are placed, and then find the delivery box.â

Onboard view

## 2

âTurn left 90 degrees, then turn right toward the yellow cabinet, and finally locate the clock icon.â

Onboard view

## 3

âGo to the right side of the black rockery.â

Onboard view

## 4

âPlease follow the person in the gray top.â

Onboard view

## 5

âGo around this tree from the left of the nearest oneâ

Onboard view

<!-- image-->  
Figure 6: Real-world experiments for navigation scenarios utilizing both visual and language inputs.

Models trained solely with 3D-GS data achieve performance comparable to those trained on real-world data across both seen and unseen datasets. This can be attributed to 3D-GS providing superior continuous geometry representation, consistent lighting, and efficient rendering, which enhances the realism of synthetic visual features. In contrast, the model trained exclusively on mesh data exhibits noticeably weaker performance. While it achieves competitive results on the seen dataset, its performance deteriorates on the unseen dataset, which contains more real-world samples, indicating limited generalization to real-world scenarios. Importantly, we adopt a hybrid dataset combining three data types, as it provides a larger and more diverse training set. Although mesh data alone lacks realistic visual fidelity, many open-source datasets are mesh-based, which helps cover additional image viewpoints and further improves navigation success rates.

Table 2: Ablation study of different training navigation datasets (Success rates %).
<table><tr><td rowspan="2">Datasets</td><td colspan="2">Seen Dataset</td><td colspan="2">Unseen Dataset</td></tr><tr><td>Object Long-horizon Navigation Navigation</td><td>Spatial Grounding</td><td>Object Long-horizon Navigation Navigation</td><td>Spatial Grounding</td></tr><tr><td>Only real-world data</td><td>98.8 89.1</td><td>95.0</td><td>96.3 82.7</td><td>90.4</td></tr><tr><td>Only 3D-GS data</td><td>98.6 87.4</td><td>95.2</td><td>95.8 82.9</td><td>91.1</td></tr><tr><td>Only mesh data</td><td>92.2 78.0</td><td>88.3</td><td>70.3 61.8</td><td>74.2</td></tr><tr><td>Hybrid data</td><td>99.2 88.2</td><td>96.5</td><td>97.6 83.5</td><td>92.8</td></tr></table>

Multi-stage training strategy: To validate the effectiveness of the proposed progressive three-stage training framework for multi-task aerial navigation, we conduct systematic ablation studies (Tab. 3). The results indicate that different training stages contribute distinctly to various task performances. Stage I (VLA-AN w/ reasoning) significantly enhances scene reasoning and spatial grounding, achieving 87.2% and 94.5% success rates, respectively, demonstrating that early full-parameter SFT effectively strengthens the modelâs visual grounding and logical reasoning capabilities. Stage II alone (VLA-AN w/ navigation) excels in target and long-horizon navigation tasks, with success rates of 95.3% and 82.9%, respectively, indicating that navigation-specific subsequent training markedly improves path planning and action execution. When Stage I and Stage II are combined (VLA-AN w/ full SFT), high performance is maintained across all tasks, confirming that multi-stage training simultaneously enhances visual reasoning and navigation proficiency. The introduction of Stage III reinforcement learning fine-tuning (VLA-AN w/ RFT) further improves long-range navigation success, highlighting RFTâs potential in correcting navigation failure patterns and optimizing decision consistency. Ultimately, the fully integrated three-stage model (Ours) achieves the superior performance across the majority of tasks: scene reasoning 85.5%, spatial localization 94.6%, target navigation 98.1%, and long-range navigation 85.7%, fully demonstrating the effectiveness and necessity of the progressive multi-stage framework for cross-task generalization. These results suggest that by combining our progressive three-stage training, the model can perform robustly and efficiently in complex aerial environments.

Table 3: Ablation study of multi-stage training strategy (Success rates %).
<table><tr><td rowspan="2">Model</td><td rowspan="2"></td><td rowspan="2">Stage I Stage â¡I Stage III</td><td rowspan="2"></td><td rowspan="2">Scene Reasoning</td><td rowspan="2">Spatial Grounding</td><td rowspan="2">Object Navigation</td><td rowspan="2">Long-horizon Navigation</td></tr><tr><td></td></tr><tr><td>Baseline</td><td>Ã</td><td>Ã</td><td>Ã</td><td>77.4</td><td>68.2</td><td>34.2</td><td>20.4</td></tr><tr><td>VLA-AN w/ reasoning</td><td>â</td><td>Ã</td><td>Ã</td><td>87.2</td><td>94.5</td><td>38.1</td><td>22.5</td></tr><tr><td>VLA-AN w/ navigation</td><td>Ã</td><td>â</td><td>Ã</td><td>75.6</td><td>64.3</td><td>95.3</td><td>74.9</td></tr><tr><td>VLA-AN w/ full SFT</td><td>â</td><td>â</td><td>Ã</td><td>82.3</td><td>91.1</td><td>92.9</td><td>72.2</td></tr><tr><td>VLA-AN w/ RFT</td><td>Ã</td><td>Ã</td><td>â</td><td>82.6</td><td>76.1</td><td>45.4</td><td>36.3</td></tr><tr><td>Ours</td><td>â</td><td>â</td><td>â</td><td>85.5</td><td>94.6</td><td>98.1</td><td>85.7</td></tr></table>

Onboard inference optimization: Finally, we benchmark our optimized method on the onboard NVIDIA Jetson Orin NX under ARM architecture against several popular frameworks. Specifically, we employ multiple system-level optimization strategies, including flash-attention mechanisms, FFNâ(RMS)Norm fusion, KV-cache preloading, and CUDA graph scheduling. For detailed inference evaluation, we use 720P relatively high-resolution images as visual input and variable-length, non-empty textual prompts (near 50 tokens on average), reflecting interactive onboard usage. The performance is compared against the unoptimized baseline, and Fig. 7 illustrates the inference time improvements contributed by each optimization. Notably, the total inference time is reduced from about 4100 ms to 494 ms (8.3Ã speedup, 87.9% reduction). The largest absolute reduction occurs in the ViT stage (2350 â 120 ms), while both projector + LLM prefill (550 â 130 ms) and decoding (1000 â 230 ms) are substantially accelerated, together enabling real-time capable onboard deployment under constrained computational resources. In addition, we apply pipelined execution to the CPU-intensive action module, effectively overlapping computation and control logic to further reduce end-to-end latency on the ARM-based onboard platform.

<!-- image-->  
Figure 7: Ablation study of onboard inference optimization. The figure illustrates the original inference time and the inference time achieved after applying various optimization techniques.

## 5 Conclusion

This work presents VLA-AN, an integrated framework that advances aerial robot autonomy by combining high-fidelity data generation, multimodal learning, safety-aware action generation, and efficient onboard deployment. Using 3D-GS method, we build a realistic UAV-centric dataset that narrows the domain gap between synthetic and real-world observations. The threestage training paradigm endows the model with strong scene comprehension, temporal reasoning, and long-horizon navigation, while the lightweight real-time action module ensures safe and collision-free control in constrained, unseen environments. Experiments show that VLA-AN significantly improves semantic grounding, trajectory robustness, and cross-scene generalization, achieving reliable performance even on resource-limited platforms, offering a pathway toward scalable and cognitively capable aerial agents for diverse UAV applications.

Looking forward, we aim to expand the aerial navigation dataset to capture richer UAV observations and actions across varying altitudes, thereby enhancing 3D spatial navigation and reasoning capabilities. We also plan to further optimize inference speed using more advanced techniques and higher-performance hardware (e.g., NVIDIA Jetson Thor) to approach real-time inference rates near 10 Hz, meeting the demands of responsive decision-making. Additionally, we intend to develop navigation-adapted attention mechanisms to train a more intelligent flight foundation navigation model capable of sophisticated perception and action in complex environments.

## References

[1] H. Liu, C. Li, Q. Wu, Y. J. Lee, Visual Instruction Tuning (2023), https://arxiv.org/abs/2304.08485.

[2] OpenAI, GPT-4 Technical Report. arXiv preprint arXiv:2303.08774 (2023).

[3] Google DeepMind, Gemini: A Family of Highly Capable Multimodal Models. arXiv preprint arXiv:2312.11805 (2023).

[4] J. Bai, et al., Qwen-VL: A Versatile Vision-Language Model for Understanding, Reasoning, and Generation. arXiv preprint arXiv:2308.12966 (2023).

[5] J. Ji, N. Pan, C. Xu, F. Gao, Elastic Tracker: A Spatio-temporal Trajectory Planner for Flexible Aerial Tracking, in 2022 International Conference on Robotics and Automation (ICRA) (2022), pp. 47â53, doi:10.1109/ICRA46639.2022. 9811688.

[6] J. Ji, T. Yang, C. Xu, F. Gao, Real-Time Trajectory Planning for Aerial Perching, in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (2022), pp. 10516â10522, doi:10.1109/IROS47612.2022.9981489.

[7] Y. Wu, et al., Ring-Rotor: A Novel Retractable Ring-Shaped Quadrotor With Aerial Grasping and Transportation Capability. IEEE Robotics and Automation Letters 8 (4), 2126â2133 (2023), doi:10.1109/LRA.2023.3245499.

[8] M. J. Kim, et al., OpenVLA: An Open-Source Vision-Language-Action Model (2024), https://arxiv.org/abs/ 2406.09246.

[9] K. Black, et al., Ï0: A Vision-Language-Action Flow Model for General Robot Control (2024), https://arxiv.org/ abs/2410.24164.

[10] S. Bai, et al., Qwen3-VL Technical Report (2025), https://arxiv.org/abs/2511.21631.

[11] W. Wang, et al., InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency (2025), https://arxiv.org/abs/2508.18265.

[12] Y. Xia, H. Huang, J. Zhu, Z. Zhao, Achieving Cross Modal Generalization with Multimodal Unified Representation. Advances in Neural Information Processing Systems 36 (2024).

[13] X. Cui, et al., Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language Models. AAAI Conference on Artificial Intelligence (AAAI) (2025).

[14] S. Yan, et al., Scaling Laws for Speculative Decoding. arXiv preprint arXiv:2505.07858 (2025).

[15] C. Chi, et al., Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, in Proceedings of Robotics: Science and Systems (RSS) (2023).

[16] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le, Flow Matching for Generative Modeling. arXiv preprint arXiv:2210.02747 (2022), https://arxiv.org/abs/2210.02747.

[17] Z. Ravichandran, et al., Distilling On-device Language Models for Robot Planning with Minimal Human Intervention (2025), https://arxiv.org/abs/2506.17486.

[18] X. Wang, et al., UAV-Flow Colosseo: A Real-World Benchmark for Flying-on-a-Word UAV Imitation Learning (2025), https://arxiv.org/abs/2505.15725.

[19] S. Liu, et al., AerialVLN: Vision-and-Language Navigation for UAVs (2023), https://arxiv.org/abs/2308.06735.

[20] B. Kerbl, G. Kopanas, T. Leimkuhler, G. Drettakis, 3D Gaussian Splatting for Real-Time Radiance Field Rendering. Â¨ ACM Transactions on Graphics (TOG) 42 (4), 139:1â139:14 (2023).

[21] Z. Shao, et al., DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (2024), https: //arxiv.org/abs/2402.03300.

[22] H. Li, et al., SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning (2025), https://arxiv.org/abs/ 2509.09674.

[23] F. Gao, et al., Teach-Repeat-Replan: A Complete and Robust System for Aggressive Flight in Complex Environments. IEEE Transactions on Robotics 36 (5), 1526â1545 (2020), doi:10.1109/TRO.2020.2993215.

[24] Y. Zhang, X. Chen, C. Feng, B. Zhou, S. Shen, FALCON: Fast Autonomous Aerial Exploration Using Coverage Path Guidance. IEEE Transactions on Robotics 41, 1365â1385 (2025), doi:10.1109/TRO.2024.3522148.

[25] Y. Wu, et al., Shape-Adaptive Planning and Control for a Deformable Quadrotor (2025), https://arxiv.org/abs/ 2505.15010.

[26] R. Zhang, Y. Wu, L. Zhang, C. Xu, F. Gao, Autonomous and Adaptive Navigation for Terrestrial-Aerial Bimodal Vehicles. IEEE Robotics and Automation Letters 7 (2), 3008â3015 (2022), doi:10.1109/LRA.2022.3145505.

[27] T. Qin, P. Li, S. Shen, VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator. IEEE Transactions on Robotics 34 (4), 1004â1020 (2018), doi:10.1109/TRO.2018.2853729.

[28] W. Xu, F. Zhang, FAST-LIO: A Fast, Robust LiDAR-Inertial Odometry Package by Tightly-Coupled Iterated Kalman Filter. IEEE Robotics and Automation Letters 6 (2), 3317â3324 (2021), doi:10.1109/LRA.2021.3064227.

[29] A. Wang, et al., YOLOv10: Real-Time End-to-End Object Detection. arXiv preprint arXiv:2405.14458 (2024).

[30] S. Ren, K. He, R. Girshick, J. Sun, Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, in Advances in Neural Information Processing Systems (NeurIPS), vol. 28 (2015).

[31] H. Chen, et al., IRS: Instance-Level 3D Scene Graphs via Room Prior Guided LiDAR-Camera Fusion (2025), https: //arxiv.org/abs/2506.06804.

[32] P. E. Hart, N. J. Nilsson, B. Raphael, A Formal Basis for the Heuristic Determination of Minimum Cost Paths, in IEEE Transactions on Systems Science and Cybernetics, vol. 4 (1968), pp. 100â107.

[33] S. Karaman, E. Frazzoli, Sampling-Based Algorithms for Optimal Motion Planning, in The International Journal of Robotics Research (IJRR), vol. 30 (2011), pp. 846â894.

[34] B. Zhou, F. Gao, L. Wang, C. Liu, S. Shen, Robust and Efficient Quadrotor Trajectory Generation for Fast Autonomous Flight. IEEE Robotics and Automation Letters 4 (4), 3529â3536 (2019), doi:10.1109/LRA.2019.2927938.

[35] X. Zhou, Z. Wang, H. Ye, C. Xu, F. Gao, EGO-Planner: An ESDF-Free Gradient-Based Local Planner for Quadrotors. IEEE Robotics and Automation Letters 6 (2), 478â485 (2021), doi:10.1109/LRA.2020.3047728.

[36] Z. Han, R. Zhang, N. Pan, C. Xu, F. Gao, Fast-Tracker: A Robust Aerial System for Tracking Agile Target in Cluttered Environments, in 2021 IEEE International Conference on Robotics and Automation (ICRA) (2021), pp. 328â334, doi: 10.1109/ICRA48506.2021.9561948.

[37] T. Wu, Y. Chen, T. Chen, G. Zhao, F. Gao, Whole-Body Control Through Narrow Gaps from Pixels to Action, in 2025 IEEE International Conference on Robotics and Automation (ICRA) (2025), pp. 11317â11324, doi:10.1109/ICRA55743. 2025.11128088.

[38] Y. Gao, et al., Adaptive Tracking and Perching for Quadrotor in Dynamic Scenarios. IEEE Transactions on Robotics 40, 499â519 (2024), doi:10.1109/TRO.2023.3335670.

[39] C. Y. Hu, et al., See, Point, Fly: A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation (2025), https://arxiv.org/abs/2509.22653.

[40] Q. Chen, et al., GRaD-Nav++: Vision-Language Model Enabled Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics (2025), https://arxiv.org/abs/2506.14009.

[41] X. Wang, et al., Towards Realistic UAV Vision-Language Navigation: Platform, Benchmark, and Methodology (2024), https://arxiv.org/abs/2410.07087.

[42] J. Lee, et al., CityNav: A Large-Scale Dataset for Real-World Aerial Navigation (2025), https://arxiv.org/abs/ 2406.14240.

[43] H. Cai, et al., FlightGPT: Towards Generalizable and Interpretable UAV Vision-and-Language Navigation with Vision-Language Models (2025), https://arxiv.org/abs/2505.12835.

[44] Y. Liu, et al., NavAgent: Multi-scale Urban Street View Fusion For UAV Embodied Vision-and-Language Navigation (2024), https://arxiv.org/abs/2411.08579.

[45] Y. Gao, et al., OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation (2025), https://arxiv. org/abs/2502.18041.

[46] Y. Fan, et al., Aerial Vision-and-Dialog Navigation (2023), https://arxiv.org/abs/2205.12219.

[47] A.-C. Cheng, et al., NaVILA: Legged Robot Vision-Language-Action Model for Navigation (2025), https://arxiv. org/abs/2412.04453.

[48] J. Zhang, et al., Embodied Navigation Foundation Model (2025), https://arxiv.org/abs/2509.12129.

[49] J. Zhang, et al., Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks (2025), https://arxiv.org/abs/2412.06224.

[50] H. Yu, et al., Thinking in 360Â°: Humanoid Visual Search in the Wild (2025), https://arxiv.org/abs/2511.20351.

[51] NVIDIA, NVIDIA Jetson Thor: Advanced AI Platform for Robotics, https://www.nvidia.com/en-us/ autonomous-machines/embedded-systems/jetson-thor (2025), accessed: 2025-12-13.

[52] NVIDIA, NVIDIA Jetson AGX Orin Series, https://www.nvidia.com/en-eu/autonomous-machines/ embedded-systems/jetson-orin/ (2025), accessed: 2025-12-13.

[53] Y. Ma, Y. Zhou, Y. Yang, T. Wang, H. Fan, Running VLAs at Real-time Speed (2025), https://arxiv.org/abs/2510. 26742.

[54] A. Brohan, et al., RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (2023), https: //arxiv.org/abs/2307.15818.

[55] A. Brohan, et al., RT-1: Robotics Transformer for Real-World Control at Scale (2023), https://arxiv.org/abs/ 2212.06817.

[56] P. Intelligence, et al., Ï0.5: a Vision-Language-Action Model with Open-World Generalization (2025), https://arxiv. org/abs/2504.16054.

[57] S. Mehta, M. Rastegari, MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer (2022), https://arxiv.org/abs/2110.02178.

[58] A. A. Rusu, et al., Policy Distillation (2016), https://arxiv.org/abs/1511.06295.

[60] NVIDIA Research, GR00T N1.5: An Improved Open Foundation Model for Generalist Humanoid Robots, https:// research.nvidia.com/labs/gear/gr00t-n1_5/ (2025), accessed: 2025-12-13.

[59] P. Micikevicius, et al., Mixed Precision Training (2018), https://arxiv.org/abs/1710.03740.