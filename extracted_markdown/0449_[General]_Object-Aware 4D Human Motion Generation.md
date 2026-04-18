# Object-Aware 4D Human Motion Generation

Shurui Guiâ 1, 2â  Deep Patelâ 2 Xiner Li1 Martin Renqiang Min2

1Texas A&M University 2NEC Laboratories America

âEqual contribution.

## Abstract

Recent advances in video diffusion models have enabled the generation of high-quality videos. However, these videos still suffer from unrealistic deformations, semantic violations, and physical inconsistencies that are largely rooted in the absence of 3D physical priors. To address these challenges, we propose an object-aware 4D human motion generation framework grounded in 3D Gaussian representations and motion diffusion priors. With pre-generated 3D humans and objects, our method, Motion Score Distilled Interaction (MSDI), employs the spatial and prompt semantic information in large language models (LLMs) and motion priors through the proposed Motion Diffusion Score Distillation Sampling (MSDS). The combination of MSDS and LLMs enables our spatial-aware motion optimization, which distills score gradients from pre-trained motion diffusion models, to refine human motion while respecting object and semantic constraints. Unlike prior methods requiring joint training on limited interaction datasets, our zero-shot approach avoids retraining and generalizes to out-of-distribution object aware human motions. Experiments demonstrate that our framework produces natural and physically plausible human motions that respect 3D spatial context, offering a scalable solution for realistic 4D generation.

## 1. Introduction

Recent advancements in video generation have led to impressive results in generating realistic and semantically rich visual content. Video diffusion models [5, 8, 9, 32, 33, 51, 62] have achieved high visual quality on diverse tasks. Despite the progress, state-of-the-art models, including large-scale systems like Sora [30], still face persistent challenges such as unrealistic deformation, object penetration, and semantic violations. These issues often stem from the lack of explicit physical and spatial constraints, which are difficult to capture in purely 2D representations [1, 53].

To address these limitations, there has been a growing interest in incorporating 3D priors into generative modeling. The success of methods like DreamFusion [38] has demonstrated that distilling 2D priors from pre-trained diffusion models can guide 3D content generation, which has motivated many 3D and 4D generation works [11, 13, 22, 26, 28, 35, 60, 63]. However, 4D generation methods that rely solely on video diffusion models inherit the spatial ambiguity and semantic misalignment problems. For instance, prompts involving spatial relations (e.g., âa dog under the bedâ) often produce incorrect visual arrangements. To mitigate this, compositional 4D generation approaches [3, 4, 54] have been proposed to combine multiple priors and synthesize novel distributions. Yet, these methods still face a fundamental bottleneck: human motions generated from pre-trained video models often suffer from distortions and fail to respect the physical constraints of interactions with static objects.

In this work, we tackle the challenging problem of zeroshot object-aware 4D human motion generation. Specifically, we aim to generate realistic 3D human motion interacting with a 3D static object over time, without requiring additional training on paired human-object data. Unlike prior methods [7, 12, 50, 55] that rely on training dedicated models with limited joint human-object datasets, our framework leverages a compositional approach with strong generalization capability. Our method, Motion Score Distilled Interaction (MSDI), builds on recent advances in 3D Gaussian representations, i.e., motion diffusion models, and spatial reasoning with large language models (LLMs).

Specifically, we first generate high-fidelity human and object 3D Gaussians using HumanGaussian [27] and Dream-Gaussian [44], respectively. To control the temporal motion, we propose to guide human trajectories using LLMgenerated spatial instructions, which provide coarse but plausible global motion plans. Then, instead of directly sampling from pre-trained motion diffusion models, which are often unreliably out of distribution, MSDS distills guidance from the motion diffusion model to form an optimization process that adjusts human poses and trajectories to align with both learned motion priors and interaction constraints. Furthermore, we formulate a constrained optimization framework that combines MSDS loss with smoothness, trajectory alignment, and collision-avoidance terms. This allows us to generate motion sequences that are realistic, smooth, and physically plausible with respect to the static object. Our zero-shot formulation ensures that the system can benefit from future improvements in motion diffusion models without the need for retraining, offering a scalable path toward generalizable and realistic object-aware 4D human motion generation. Experiments on multiple zero-shot prompts show, that our generated 4D scenes produce realistic motions with high physical constraint obedience ability while previous 4D generation methods can only generate unnatural distortions without plausible motion.

## 2. Related Work

Video generation. Video generation models have been widely used to generate realistic videos. Although the video diffusion models [5, 8, 9, 32, 33, 51, 62] have shown promising results in various areas, unrealistic deformation, twisted, penetration, and semantics violations still exist even in large video generation model Sora [30]. These issues are often considered as the lack of physics information learned [2, 48, 56]. Despite many studies in addressing these issues by using trajectory tracking [14], occlusion masks [19], and semantic masks [34], we argue that it is not feasible to solve this problem in 2D space without introducing extra information equivalent to physical information in 3D space, and the natural and intrinsic way to tackle these challenges should lie in the use of 3D space.

3D generation. While 3D generation has been explored in recent years [13, 26, 28, 42, 44, 47, 63], one of the most popular and convincing directions is to extract prior knowledge from 2D diffusion models. Specifically, DreamFusion [38], the first method introducing Score Distillation Sampling (SDS), generates 3D content by leveraging information from 2D image diffusion models. This work inspires a significant amount of following works [10, 52, 58] on improving 3D content quality, optimization efficiency, and human avatar generations [16, 21, 27].

4D Generation. Aligning with the philosophy of extracting information from pre-trained image diffusion models, many 4D generation works adopt pre-trained video diffusion models [25, 57, 61], to tackle challenges in image to 4D [40] and video to 4D [11, 17, 22, 23, 31, 35, 59, 60] tasks. However, the generation ability of these studies cannot go beyond the original distribution of the pre-trained diffusion models, and shares the same limitations as the original 2D pre-trained diffusion models. For example, most of the pre-trained video diffusion models face difficulties in understanding spatial information, $e . g .$ , generating with a prompt âa human walks towards the tableâ can produce unrealistic results, such as deformed bodies, poor framing that shows only the legs, or the human being omitted from the scene entirely. In order to solve this challenge, one convincing direction is to apply compositional 4D generation, which incorporates multiple prior distributions and combine them to generate samples with novel distributions. Recently, 4DFy [4], Comp4D [54], and TC4D [3] have shown promising results on 4D compositional generation. However, although generating contents in 3D space helps with spatial information/trajectory planning, all the motion information from the pre-trained video diffusion model inherits its original distortions, especially on human-related motions. Motivated by this problem, we consider distilling information from dedicated motion models to guide the motion optimization process.

Different from interaction generations between humans and objects/scenes [7, 50, 55], our method is zero-shot, which does not need to train specific dedicated models; thus, it widens the application range. Human-object interaction generations like InterDiff [55] and CG-HOI [12] require the joint distribution of humans and objects for training, while the sizes of these datasets are still limited, which cannot extend to any out-of-distribution scenarios. Since our work focuses on the interaction between humans and static objects, our setting is more similar to the HUMAN-ISE task [7, 50]. Compared with them, our setting eliminates the object-locating phase and focuses on the human trajectory and human motion generation with the static object. While these works require training extra models for human trajectory and human motion generation just for the additional object, our method does not require any additional motion diffusion model training and can achieve realistic interactions between humans and static objects with motion diffusion model score distillation sampling (MSDS). This zero-shot behavior enables this framework to improve as the motion diffusion model iterates in the future without extra distribution and retraining requirements.

## 3. Preliminaries

## 3.1. 3D Gaussian Splatting

3D Gaussian Splatting [20] (3DGS) is a dominating representation in the 3D field, due to its explicit 3D space representation and high efficient optimization. The individual units of 3DGS are 3D Gaussian ellipsoids, where each 3D Gaussian is parameterized by position ??, anisotropic covariance Î£ as its shape, and opacity ?? and spherical harmonic coefficients ??â as its optical characteristics, where ??â is a view-dependent property. The shape of the 3D Gaussian Î£ can be considered as the composition of a scaling and a rotation as follows:

$$
\Sigma = R S S ^ { T } R ^ { T } ,\tag{1}
$$

where the scaling matrix ?? can be denoted as a 3D vector ??, and the rotation matrix ?? as a quaternion $q \in { \bf S 0 } ( 3 )$ .

Therefore, the formal definition of a Gaussian centered at point $\mu$ is:

$$
G ( \mathbf { x } , \mu ) = e ^ { - \frac { 1 } { 2 } ( \mathbf { x } - \mu ) ^ { T } \Sigma ^ { - 1 } ( \mathbf { x } - \mu ) } ,\tag{2}
$$

where x is a random variable in 3D space.

To render 3D Gaussians into a 2D image, 3DGS considers the additional opacity ?? and spherical harmonic coefficients by utilizing a tile-based rasterizer and point-based ??-blend rendering. For each pixel ??, its color ?? (??) is rendered under the following calculation:

$$
\begin{array} { r } { { \cal C } \left( u \right) = \displaystyle \sum _ { i \in N } T _ { i } c _ { i } \alpha _ { i } S \mathcal { H } \left( s h _ { i } , \nu \right) , } \\ { \displaystyle } \\ { { \cal T } _ { i } = G \left( \mathbf { x } , \mu _ { i } \right) \displaystyle \prod _ { j = 1 } ^ { i - 1 } \left( 1 - \alpha _ { j } G \left( \mathbf { x } , \mu _ { j } \right) \right) , } \end{array}\tag{3}
$$

where ???? denotes the transmittance for the ??-th Gaussian, SH denotes the spherical harmonic function, and ?? represents the viewing direction. The 3D Gaussian optimization process includes adjusting all 3D Gaussian properties $\{ \mu , q , s , \sigma , c \}$ and the high-level 3D Gaussian density modifications using densifying and pruning processes.

## 3.2. SMPL-X

SMLP models [37] represent a human by transforming the human mesh of a standard pose, the canonical model, into the observation space, using pose parameter ??, shape parameter $\beta ,$ and expression parameter ??:

$$
\begin{array} { r } { M ( \beta , \theta , \phi ) = \mathrm { L B S } ( T ( \beta , \theta , \phi ) , J ( \beta ) , \theta , \mathcal { W } ) , } \\ { T ( \beta , \theta , \phi ) = \mathbf { T } + B _ { s } ( \beta ) + B _ { e } ( \phi ) + B _ { p } ( \theta ) , } \end{array}\tag{4}
$$

where ?? is the function mapping parameters to a transformed human mesh model; ?? represents the transformed human key points/vertices adjusted by different human shapes, expressions, and poses through corresponding functions $B _ { s } , B _ { e }$ , and $B _ { p } ,$ respectively. Given the transformed vertices, the skins of the human mesh need to be adjusted according to the transformations of several nearby joints, which is done by the linear blend skinning function LBS(Â·) where $\mathcal { W }$ stands for blend weights that determine the effects from different joints. Specifically, the LBS process is defined as follows:

$$
\mathbf { v } _ { o } = { \mathcal { G } } \cdot \mathbf { v } _ { c } , \quad { \mathcal { G } } = \sum _ { k = 1 } ^ { K } w _ { k } { \mathcal { G } } _ { k } \left( { \boldsymbol { \theta } } , j _ { k } \right) ,\tag{5}
$$

where the vertices ${ \bf v } _ { o }$ in the observation space is deformed from the canonical pose vertices ${ \bf v } _ { c }$ by the deformation $\mathcal { G } .$ The deformation is determined by the affine deformation $\mathcal { G } _ { k } \left( \theta , j _ { k } \right)$ that merges the warping effects from K neighboring joints, simulating the smooth position changes of vertices.

## 4. Motion Score Distilled Interaction

Our object-aware human motion generation (OAHM) framework addresses the challenging zero-shot generation problem by leveraging pre-trained motion priors within an explicit optimization paradigm. In this section, we first introduce the overall OAHM generation pipeline, followed by a description of our spatially-aware coarse motion generation strategy. Finally, we detail our Motion Score Distilled Interaction (MSDI) method, where we propose motion score distillation sampling (MSDS) and incorporate spatial and physical constraints to optimize human motion trajectories, enabling the synthesis of physically plausible and natural motion sensitive to object.

## 4.1. OAHM Generation Framework

With expressive 3D representations as the foundation, we employ HumanGaussian [27] to generate high-fidelity 3D human Gaussians $G _ { h }$ from textual prompts, and utilize DreamGaussian [44] to synthesize 3D objects from an initial shape-e geometry [18], as illustrated in Figure 1. Given a motion sequence ??, we establish a correspondence between the motion trajectory and the human Gaussian points $G _ { h }$ enabling dynamic Gaussian-based human motion.

Concretely, we initialize the SMPL-X model in a rest pose, consistent with the canonical configuration of the human Gaussian points. Each Gaussian point is mapped to the nearest barycentric coordinate on the corresponding SMPL-X mesh face. By preserving these fixed barycentric correspondences, any transformation applied to the SMPL-X mesh is faithfully propagated to the associated Gaussian points, ensuring coherent deformation of the 3D human representation.1 Note that this mapping is differentiable, enabling gradients back propagation.

With a controllable Gaussian human and an independently generated Gaussian object co-located in the same coordinate system, we render interactive sequences via Gaussian splatting. While high-quality 3D human and object representations can be readily obtained, the availability of joint 4D human-object distributions remains limited [6], making it infeasible to train generative models directly on such data. To this end, we propose a zero-shot OAHM generation framework. Constructing this framework and achieving realistic object-aware human motion remains highly challenging due to the need for temporally consistent, physically plausible, and semantically appropriate interactions. Therefore, this framework presents two major challenges: (1) extracting meaningful and realistic human motion distributions, and (2) enforcing human-object interaction constraints on generated motions.

## 4.2. Spatial-Aware Coarse Motion Generation

Motion diffusion models. To address the first challenge, we avoid relying on video diffusion models, as prior work [4] has shown that distilling human motion from such models often leads to unrealistic results. Instead, we leverage dedicated human motion diffusion models (MDMs) [41, 46], which are currently state-of-the-art for generating plausible human motions.

Our motion representation consists of an ??-length motion sequence $X = \{ \bar { x } ^ { i } \} _ { i = 1 } ^ { N }$ , where each $\boldsymbol { x } ^ { i } \in \mathbb { R } ^ { 3 + 6 + J \times 6 }$ encodes the pose parameters $\bar { \theta ^ { i } } \in \mathbb { R } ^ { J \times 6 }$ , global translation $r ^ { i } \in \mathbb { R } ^ { 3 }$ , and 6D global orientation $\gamma ^ { i } \in \mathbb { R } ^ { 6 }$ . Other parameters, such as body shape, are omitted for simplicity. During the MDM process, the ??-frame motion sequence ?? is subject to ?? steps of Gaussian noise:

$$
q \left( X _ { t } \mid X _ { t - 1 } \right) = N \left( \sqrt { \alpha _ { t } } X _ { t - 1 } , ( 1 - \alpha _ { t } ) I \right) ,\tag{6}
$$

where $t \in \{ 1 , \ldots , T \}$ denotes the diffusion step and $X _ { T } \sim$ $N ( 0 , I )$ . The MDM is trained to predict the clean motion ??Ë0 given a noisy motion $X _ { t }$ and a text condition ?? encoded by a CLIP-based text encoder.

LLM-based trajectory generation. Despite the advantages of MDMs, directly sampling from these models does not guarantee meaningful object-aware motions, as they lack explicit spatial awareness necessary for modeling relationships between humans and objects. Attempts to use guidance from 2D image and video diffusion models also failed to yield reliable spatial supervision signals.

To overcome this, we harness the spatial reasoning capabilities of LLMs. Given the initial coordinates of the human and object, along with a textual motion instruction, the LLM generates a coarse global trajectory for the human. This LLM-derived trajectory, denoted as $r _ { \mathrm { L L M } } ^ { i } \mathrm { f o r } i = 1 , . . . , N .$ can be further refined using trajectory interpolation and collision detection, enabling the system to produce physically plausible paths, such as automatic detours around obstacles. For example, when instructed to âwalk four meters toward a table two meters away,â the LLM can synthesize a motion that navigates around the object.

The LLM-generated trajectory is used to initialize the global translation in the MDM framework [41]. With estimated time/frames and extracted pure motion prompt as two additional inputs, the MDM can yield a coarse motion sequence that incorporates spatial awareness. While the resulting motions may lack fine-grained realism, they provide a strong starting point for subsequent optimization. Detailed examples of LLM prompts are provided in the Appendix.

## 4.3. Constrained Motion Optimization

The core challenge in generating realistic object-aware human motion lies not only in producing plausible human motion, but also in enforcing physical constraints such as collision avoidance and trajectory fidelity. These challenges are not easily addressed by existing generative models. As mentioned above, two major issues must be overcome: (1) generating meaningful, in-distribution human motion sequences, and (2) ensuring these motions respect spatial and physical constraints posed by objects in the environment.

While our LLM-guided approach and MDM address the extraction of plausible motion trajectories, these solutions alone cannot guarantee realistic interactions. Specifically, directly applying the LLM-generated coarse trajectories often results in infeasible or unnatural motions, as these trajectories may violate object penetration constraints or fall outside the motion distribution captured by the pre-trained MDM. Moreover, existing diffusion models are not inherently designed to encode or enforce collision and spatial constraints.

To overcome these limitations, we introduce a constrained motion optimization framework, namely, motion score distilled interaction (MSDI), that jointly refines human motion by leveraging the prior knowledge encoded in motion diffusion models, while explicitly enforcing trajectory, smoothness, and collision-avoidance constraints.

Motion Diffusion Score Distillation Sampling (MSDS). Instead of generating human motions directly from the diffusion model, we extract the score (gradient) information from the MDM to guide the optimization of both trajectories and poses under physical constraints. Specifically, we propose Motion Diffusion Score Distillation Sampling (MSDS), which optimizes human motion ?? by maximizing the log-likelihood under the MDM prior. The gradient of the MSDS objective is given by:

$$
\nabla _ { X } \mathcal { L } _ { \mathrm { M S D S } } ( \phi ) \triangleq \mathbb { E } _ { t , \epsilon } \left[ w ( t ) \left( X - \mathbf { M D M } _ { \phi } ( X _ { t } , t , c ) \right) \right] .\tag{7}
$$

where $\mathbf { M D M } _ { \phi }$ denotes the pre-trained MDM and $w ( t )$ is a weighting function over diffusion steps. This process aligns the optimized motion with the learned distribution of human poses and trajectories.

Constrained Optimization Objectives. To ensure the resulting motions are physically plausible and interact naturally with the object, we further introduce explicit constraints:

â¢ Trajectory Alignment. We regularize the optimized trajectory to remain close to the LLM-generated coarse trajectory. The trajectory loss is defined as:

$$
\begin{array} { r c l } { { { \mathcal L } _ { \mathrm { t r a j } } } } & { { = } } & { { \displaystyle \lambda _ { \mathrm { m i d d l e } } \cdot \sum _ { i = 2 } ^ { N - 1 } { | | r ^ { i } - r _ { L L M } ^ { i } | | _ { 2 } ^ { 2 } } + } } \\ { { } } & { { } } & { { \displaystyle \lambda _ { \mathrm { e n d } } \cdot \sum _ { i \in \{ 1 , N \} } { | | r ^ { i } - r _ { L L M } ^ { i } | | _ { 2 } ^ { 2 } } , } } \end{array}\tag{8}
$$

where $\lambda _ { \mathrm { m i d d l e } }$ and $\lambda _ { \mathrm { e n d } }$ control the fidelity at middle and endpoint frames, respectively.

<!-- image-->  
Figure 1. Method overview. The framework includes 4 components: human and object 3D generation, coarse trajectory generation, constrained motion optimization, and rendering.

â¢ Motion Smoothness. To prevent unnatural or abrupt changes in motion, we introduce a jerk (third derivative) regularization:

$$
\mathcal { L } _ { \mathrm { s m o o t h } } = \sum _ { i = 1 } ^ { N } \left\| \frac { d ^ { 3 } r ^ { i } } { d t ^ { 3 } } \right\| _ { 2 } ^ { 2 } ,\tag{9}
$$

where, in practice, the derivative is approximated using finite differences over adjacent frames.

â¢ Collision Avoidance. To prevent human-object penetration, we employ a two-stage collision detection and penalty scheme. First, we compute the intersection C of the 3D bounding boxes for the human and object. If C is non-empty, we evaluate pairwise collisions between object points $o _ { i } ~ \in ~ C$ and their nearest human points $h _ { j } \in C .$ . The collision loss is then given by:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c o l l i s i o n } } = \operatorname* { m a x } \left( \mathbf { n } _ { j } \cdot ( h _ { j } - o _ { j } ) , - \epsilon _ { c } \right) , } \end{array}\tag{10}
$$

where $\mathbf { n } _ { j }$ is the normal vector at $h _ { j }$ and $\epsilon _ { c }$ is a collision margin hyperparameter.

MSDI Objective. The final loss function for human motion optimization combines the above terms:

$$
\begin{array} { r c l } { \mathcal { L } } & { = } & { \lambda _ { \mathrm { M S D S } } \cdot \mathcal { L } _ { \mathrm { M S D S } } + \lambda _ { \mathrm { t r a j } } \cdot \mathcal { L } _ { \mathrm { t r a j } } + } \\ & & { \lambda _ { \mathrm { s m o o t h } } \cdot \mathcal { L } _ { \mathrm { s m o o t h } } + \lambda _ { \mathrm { c o l l i s i o n } } \cdot \mathcal { L } _ { \mathrm { c o l l i s i o n } } , } \end{array}\tag{11}
$$

where ??MSDS, $\lambda _ { \mathrm { t r a j } } , \lambda _ { \mathrm { s m o o t h } }$ , and $\lambda _ { \mathrm { { c o l l i s i o n } } }$ are hyperparameters balancing the different objectives.

This MSDI constrained optimization aggregates all gradients to the motion sequence ?? and updates it. Through this optimization, we ensure that the generated motions ?? are not only realistic according to the learned motion diffusion prior but also spatially and physically consistent with the surrounding environment and objects.

## 5. Experiments

In this section, we evaluate the effectiveness of our proposed MSDI framework for object aware 4D human motion generation. Our experiments include both qualitative and quantitative analyses, benchmarking against the state-of-the-art 4Dfy method. We report results across a suite of objective metrics designed to assess motion realism, diversity, and physical plausibility. Ablation studies further demonstrate the importance of main components within our pipeline.

## 5.1. Metrics

To quantitatively assess the quality of generated objectaware human motion, we adapt several metrics that collectively measure pose realism, motion diversity, and temporal dynamics. We evaluate using established metrics like Optical Flow Score [29] and we introduce three metrics designed to asses motion dynamics: Pose Plausibility, Pose Variation and Trajectory Length. We propose this suite of metrics because there is no single universally accepted metric to quantify the perceptual quality of human motion. It is important to consider these metrics in combination, as any individual metric can be trivially satisfied by a degenerate solution (e.g., a high trajectory score with a static, implausible pose). Since the metrics operate on different scales and cannot be combined arithmetically, their value lies in the holistic, comparative assessment of different methods. A model can only be judged to produce high-quality motion if it demonstrates strong and balanced performance across this entire suite.

For each generated video, we extract per-frame 3D human meshes using HMR2.0 [15], which estimates SMPL parameters [37] for every detected human instance using a

<!-- image-->  
Figure 2. Qualitative Results. Generated videos from 4Dfy and MSDI across various text prompts. Each row corresponds to a different prompt. Within each row, columns display frames sampled at incremental timesteps from the generated video, illustrating temporal progression and motion characteristics. The frames are center cropped for better visibility.

ViTDet detector [24]. From these per-frame SMPL models and multi-view RGB frames, we compute the following metrics:2

Pose Plausibility. We evaluate the realism of each human pose using VPoser [36], a variational autoencoder trained on large-scale pose data. For each frame ??, we convert the predicted SMPL body pose parameters $\pmb { \theta } _ { t } ^ { \mathrm { p o s e } }$ into VPosercompatible axis-angle representation $\pmb { \phi } _ { t } \in \mathbb { R } ^ { N _ { V } \times 3 }$ , then encode these to obtain a posterior $q ( \boldsymbol { z } _ { t } | \phi _ { t } )$ over the latent pose space. The plausibility for each frame is quantified by the KL divergence to a standard normal prior ??(??).

Pose Variation. To quantify diversity and motion magnitude, we measure the temporal standard deviation of the pose vector $\phi _ { t }$ (flattened dimension $K = N _ { V } \times 3 )$ across all frames. High variation reflects diverse and dynamic motions.

Trajectory Length. To assess the extent of global character movement within the 3D space, we calculate the trajectory length of the root joint. For each frame ??, HMR2.0 provides the 3D keypoint coordinates. The total trajectory length is the sum of Euclidean distances between the root joint positions in consecutive frames. A longer trajectory length suggests more substantial displacement of the character over time.

Optical Flow Score. To quantify the amount of motion and temporal dynamics, we compute an Optical Flow Score [29]. For each of the $N _ { c \nu } = 4$ views, we estimate dense optical flow between consecutive frames using RAFT [45].

The score for each view is the average magnitude of these flow vectors across all pixels and frames. The final Optical Flow Score is the average of these per-view scores. A higher score signifies a more pronounced motion.

User Study. To complement our quantitative metrics, we also conducted a user study to qualitatively assess the performance of our method against 4D-fy. The study was designed to measure human perception of motion quality, physical plausibility, and overall realism. We followed the human evaluation setup established by 4D-fy and MAV3D [43].

## 5.2. Results

We conduct a comprehensive set of experiments to evaluate our method. The results demonstrate that MSDI consistently outperforms baseline 4D-fy method overall across all metrics.

Quantitative and Qualitative Analysis. As shown in our quantitative analysis (Figure 3), MSDI shows a clear advantage. Specifically, it achieves substantially higher scores in both Pose Variation and Optical Flow, indicating diverse and larger motion. Furthermore, MSDI produces better Pose Plausibility and longer Trajectory Lengths for most prompts, while remaining comparable on others.

This numerical advantage translates directly to visually perceptible improvements, as shown in our qualitative comparisons in Figure 2. In contrast, 4Dfy often produces videos where frames appear largely similar, with only minor arm or leg movements, and the human subject frequently remains static in the same position. Itâs limited human motion could be attributed to the inherent constraints of the underlying video diffusion model, such as VideoCrafters [8] used for Score Distillation Sampling. MSDI using Motion SDS generates coherent and physically-grounded interactions with human subject showing larger change in their positions over time.

<!-- image-->

<!-- image-->

<!-- image-->

<!-- image-->  
Figure 3. Quantitative Results. Quantitative comparison of MSDI and 4Dfy. The bar chart displays scores for 4 key metrics across 10 text prompts. An âXâ marker indicates that the metric failed to detect any humans in all four generated views for that particular prompt.

User Study. To validate that our quantitative and qualitative findings align with human judgment, we performed a formal user study comparing MSDI against 4D-fy. The results, summarized in Table 1, show a decisive preference for MSDI across all categories. Crucially, the preference in Motion Quality (MQ) is an overwhelming 87%, directly validating our core technical contribution. This result also confirms that the higher Pose Plausibility and Variation captured by our metrics correspond to motions that humans perceive as significantly more natural and realistic. The high preference for Appearance (AQ) (79%), 3D Structure (SQ) (71%) and Text Alignment (TA) (75%) further suggests that physically plausible motion enhances overall visual fidelity and text alignment. The 80% overall preference underscores that generating believable 4D videos hinges not just on appearance, but critically on the quality of the motion itself.

## 5.3. Ablation study

We conduct qualitative ablation studies to demonstrate the importance of key components in our proposed method. We focus on the prompt âthe human jumps onto the tableâ

<table><tr><td>Preferred Method</td><td>AQ</td><td>SQ</td><td>MQ</td><td>TA</td><td>Overall</td></tr><tr><td>MSDI (%)</td><td>79%</td><td>71%</td><td>87%</td><td>75%</td><td>80%</td></tr><tr><td>4D-fy (%)</td><td>21%</td><td>29%</td><td>13%</td><td>25%</td><td>20%</td></tr></table>

Table 1. User study results comparing MSDI with 4D-fy. We report the percentage of times users preferred our method. MSDI significantly outperforms 4D-fy across all metrics, with all results being statistically significant (?? < 0.001).

<!-- image-->  
Figure 4. Ablation study on key components of MSDI. We visualize the impact of removing our main loss terms for the prompt âthe human jumps onto the tableâ.

to highlight specific failure modes when certain losses are excluded. For all variants, the high-level LLM planned trajectory remains consistent. Visualizations are provided in Figure 4.

Effect of Collision Loss $( \mathcal { L } _ { \mathrm { c o l l i s i o n } } )$ . Without collision loss $\mathcal { L } _ { \mathrm { c o l l i s i o n } }$ , the optimization fails to enforce physical nonpenetration constraints. The generated human visibly penetrates or pierces into the table surface during the landing phase of the jump.

Effect of Motion Diffusion Score Distillation Sampling (LMSDS). Excluding the MSDS $\mathcal { L } _ { \mathrm { M S D S } }$ , significantly degrades the quality of the human motion and object interaction, particularly contact points. Without ${ \mathcal { L } } _ { \mathrm { m s d s } }$ , the human appears to float during the jump and makes unnatural contact, with improperly planted feet.

## 6. Conclusion

In this work, we introduced Motion Score Distilled Interaction (MSDI), a novel zero-shot framework for object-aware human motion generation. Our approach uniquely combines the strengths of 3D Gaussian representations for highfidelity visuals, motion diffusion models for realistic human movement priors, and large language models for spatial reasoning and initial trajectory planning. A key component of our framework is Motion Diffusion Score Distillation Sampling (MSDS), which allows us to refine human motion by leveraging gradients from pre-trained motion diffusion models. This, coupled with our constrained optimization strategy that considers trajectory alignment, motion smoothness, and collision avoidance, enables the generation of interactions that are not only natural but also physically plausible and respectful of object presence.

Unlike previous methods that often require extensive training on specific datasets, MSDI operates in a zero-shot manner. This means it can generalize to novel interactions without retraining, making it a scalable and adaptable solution. Our experiments have shown that MSDI can produce realistic human motions interacting with static 3D objects, overcoming common issues like unnatural distortions and physical violations seen in outputs from methods relying solely on video diffusion models. We believe MSDI offers a promising direction for creating more dynamic and believable 4D content by effectively integrating 3D physical and semantic priors into the generation process.

## References

[1] Sherwin Bahmani, Jeong Joon Park, Despoina Paschalidou, Hao Tang, Gordon Wetzstein, Leonidas Guibas, Luc Van Gool, and Radu Timofte. 3d-aware video generation. arXiv preprint arXiv:2206.14797, 2022. 1

[2] Sherwin Bahmani, Jeong Joon Park, Despoina Paschalidou, Hao Tang, Gordon Wetzstein, Leonidas Guibas, Luc Van Gool, and Radu Timofte. 3d-aware video generation. arXiv preprint arXiv:2206.14797, 2022. 2

[3] Sherwin Bahmani, Xian Liu, Wang Yifan, Ivan Skorokhodov, Victor Rong, Ziwei Liu, Xihui Liu, Jeong Joon Park, Sergey Tulyakov, Gordon Wetzstein, et al. Tc4d: Trajectoryconditioned text-to-4d generation. In European Conference on Computer Vision, pages 53â72. Springer, 2024. 1, 2

[4] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7996â8006, 2024. 1, 2, 4, 11

[5] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, et al. Lumiere: A space-time diffusion model for video generation. In SIGGRAPH Asia 2024 Conference Papers, pages 1â11, 2024. 1, 2

[6] Bharat Lal Bhatnagar, Xianghui Xie, Ilya A Petrov, Cristian Sminchisescu, Christian Theobalt, and Gerard Pons-Moll. Behave: Dataset and method for tracking human object interactions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15935â 15946, 2022. 3

[7] Zhi Cen, Huaijin Pi, Sida Peng, Zehong Shen, Minghui Yang, Shuai Zhu, Hujun Bao, and Xiaowei Zhou. Generating human motion in 3d scenes from text descriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1855â1866, 2024. 1, 2

[8] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 1, 2, 7

[9] Shoufa Chen, Mengmeng Xu, Jiawei Ren, Yuren Cong, Sen He, Yanping Xie, Animesh Sinha, Ping Luo, Tao Xiang, and Juan-Manuel Perez-Rua. Gentron: Diffusion transformers for image and video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6441â6451, 2024. 1, 2

[10] Zilong Chen, Yikai Wang, Feng Wang, Zhengyi Wang, and Huaping Liu. V3d: Video diffusion models are effective 3d generators. arXiv preprint arXiv:2403.06738, 2024. 2

[11] Wen-Hsuan Chu, Lei Ke, and Katerina Fragkiadaki. Dreamscene4d: Dynamic multi-object scene generation from monocular videos. arXiv preprint arXiv:2405.02280, 2024. 1, 2

[12] Christian Diller and Angela Dai. Cg-hoi: Contact-guided 3d human-object interaction generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19888â19901, 2024. 1, 2

[13] Matt Deitke; Dustin Schwenk; Jordi Salvador; Luca Weihs; Oscar Michel; Eli VanderBilt; Ludwig Schmidt; Kiana Ehsani; Aniruddha Kembhavi; Ali Farhadi. Objaverse: A universe of annotated 3d objects. IEEE, 1314. 1, 2

[14] Daniel Geng, Charles Herrmann, Junhwa Hur, Forrester Cole, Serena Zhang, Tobias Pfaff, Tatiana Lopez-Guevara, Carl Doersch, Yusuf Aytar, Michael Rubinstein, et al. Motion prompt-

ing: Controlling video generation with motion trajectories. arXiv preprint arXiv:2412.02700, 2024. 2

[15] Shubham Goel, Georgios Pavlakos, Jathushan Rajasegaran, Angjoo Kanazawa, and Jitendra Malik. Humans in 4D: Reconstructing and tracking humans with transformers. In ICCV, 2023. 5

[16] Shoukang Hu, Tao Hu, and Ziwei Liu. Gauhuman: Articulated gaussian splatting from monocular human videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20418â20431, 2024. 2

[17] Yanqin Jiang, Li Zhang, Jin Gao, Weimin Hu, and Yao Yao. Consistent4d: Consistent 360 {\deg} dynamic object generation from monocular video. arXiv preprint arXiv:2311.02848, 2023. 2

[18] Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions, 2023. 3

[19] Lei Ke, Yu-Wing Tai, and Chi-Keung Tang. Occlusion-aware video object inpainting. In Proceedings of the IEEE/CVF international conference on computer vision, pages 14468â 14478, 2021. 2

[20] Bernhard Kerbl, Georgios Kopanas, Thomas LeimkÂ¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):1â14, 2023. 2

[21] Muhammed Kocabas, Jen-Hao Rick Chang, James Gabriel, Oncel Tuzel, and Anurag Ranjan. Hugs: Human gaussian splats. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 505â515, 2024. 2

[22] Yao-Chih Lee, Yi-Ting Chen, Andrew Wang, Ting-Hsuan Liao, Brandon Y Feng, and Jia-Bin Huang. Vividdream: Generating 3d scene with ambient dynamics. arXiv preprint arXiv:2405.20334, 2024. 1, 2

[23] Jiahui Lei, Yijia Weng, Adam Harley, Leonidas Guibas, and Kostas Daniilidis. Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds. arXiv preprint arXiv:2405.17421, 2024. 2

[24] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Exploring plain vision transformer backbones for object detection. In European conference on computer vision, pages 280â296. Springer, 2022. 6

[25] Hanwen Liang, Yuyang Yin, Dejia Xu, Hanxue Liang, Zhangyang Wang, Konstantinos N Plataniotis, Yao Zhao, and Yunchao Wei. Diffusion4d: Fast spatial-temporal consistent 4d generation via video diffusion models. arXiv preprint arXiv:2405.16645, 2024. 2

[26] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9298â 9309, 2023. 1, 2

[27] Xian Liu, Xiaohang Zhan, Jiaxiang Tang, Ying Shan, Gang Zeng, Dahua Lin, Xihui Liu, and Ziwei Liu. Humangaussian: Text-driven 3d human generation with gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6646â6657, 2024. 1, 2, 3

[28] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023. 1, 2

[29] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22139â22149, 2024. 5, 6

[30] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177, 2024. 1, 2

[31] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. arXiv preprint arXiv:2308.09713, 2023. 2

[32] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video generation. arXiv preprint arXiv:2303.08320, 2023. 1, 2

[33] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 1, 2

[34] Junting Pan, Chengyu Wang, Xu Jia, Jing Shao, Lu Sheng, Junjie Yan, and Xiaogang Wang. Video generation from single semantic label map. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3733â3742, 2019. 2

[35] Zijie Pan, Zeyu Yang, Xiatian Zhu, and Li Zhang. Fast dynamic 3d object generation from a single-view video. arXiv preprint arXiv:2401.08742, 2024. 1, 2

[36] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. Expressive body capture: 3d hands, face, and body from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10975â10985, 2019. 6

[37] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3D hands, face, and body from a single image. In Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), pages 10975â10985, 2019. 3, 5

[38] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 1, 2

[39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748â8763. PMLR, 2021. 11

[40] Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. Dreamgaussian4d: Generative 4d gaussian splatting. arXiv preprint arXiv:2312.17142, 2023. 2

[41] Yonatan Shafir, Guy Tevet, Roy Kapon, and Amit H Bermano. Human motion diffusion as a generative prior. arXiv preprint arXiv:2303.01418, 2023. 4

[42] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023. 2

[43] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, et al. Text-to-4d dynamic scene generation. arXiv preprint arXiv:2301.11280, 2023. 6, 11

[44] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653, 2023. 1, 2, 3

[45] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer VisionâECCV 2020: 16th European Conference, Glasgow, UK, August 23â28, 2020, Proceedings, Part II 16, pages 402â419. Springer, 2020. 6

[46] Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit H Bermano. Human motion diffusion model. arXiv preprint arXiv:2209.14916, 2022. 4

[47] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. arXiv preprint arXiv:2403.12008, 2024. 2

[48] Kaihong Wang, Kumar Akash, and Teruhisa Misu. Learning temporally and semantically consistent unpaired video-tovideo translation through pseudo-supervision from synthetic optical flow. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2477â2486, 2022. 2

[49] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In European Conference on Computer Vision, pages 396â416. Springer, 2024. 11

[50] Zan Wang, Yixin Chen, Tengyu Liu, Yixin Zhu, Wei Liang, and Siyuan Huang. Humanise: Language-conditioned human motion generation in 3d scenes. Advances in Neural Information Processing Systems, 35:14959â14971, 2022. 1, 2

[51] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623â7633, 2023. 1, 2

[52] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. arXiv preprint arXiv:2405.14832, 2024. 2

[53] Tianyi Xie, Zeshun Zong, Yuxin Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. Physgaussian: Physicsintegrated 3d gaussians for generative dynamics. arXiv preprint arXiv:2311.12198, 2023. 1

[54] Dejia Xu, Hanwen Liang, Neel P Bhatt, Hezhen Hu, Hanxue Liang, Konstantinos N Plataniotis, and Zhangyang Wang. Comp4d: Llm-guided compositional 4d scene generation. arXiv preprint arXiv:2403.16993, 2024. 1, 2

[55] Sirui Xu, Zhengyuan Li, Yu-Xiong Wang, and Liang-Yan Gui. Interdiff: Generating 3d human-object interactions with physics-informed diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14928â 14940, 2023. 1, 2

[56] Xindi Yang, Baolu Li, Yiming Zhang, Zhenfei Yin, Lei Bai, Liqian Ma, Zhiyong Wang, Jianfei Cai, Tien-Tsin Wong, Huchuan Lu, et al. Towards physically plausible video generation via vlm planning. arXiv preprint arXiv:2503.23368, 2025. 2

[57] Zeyu Yang, Zijie Pan, Chun Gu, and Li Zhang. Diffusion 2: Dynamic 3d content generation via score composition of orthogonal diffusion models. arXiv preprint arXiv:2404.02148, 2024. 2

[58] Taoran Yi, Jiemin Fang, Zanwei Zhou, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Xinggang Wang, and Qi Tian. Gaussiandreamerpro: Text to manipulable 3d gaussians with highly enhanced quality. arXiv preprint arXiv:2406.18462, 2024. 2

[59] Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, and Yunchao Wei. 4dgen: Grounded 4d content generation with spatial-temporal consistency. arXiv preprint arXiv:2312.17225, 2023. 2

[60] Yifei Zeng, Yanqin Jiang, Siyu Zhu, Yuanxun Lu, Youtian Lin, Hao Zhu, Weiming Hu, Xun Cao, and Yao Yao. Stag4d: Spatial-temporal anchored generative 4d gaussians. arXiv preprint arXiv:2403.14939, 2024. 1, 2

[61] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023. 2

[62] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022. 1, 2

[63] Junzhe Zhu, Peiye Zhuang, and Sanmi Koyejo. Hifa: Highfidelity text-to-3d generation with advanced diffusion guidance. arXiv preprint arXiv:2305.18766, 2023. 1, 2

<!-- image-->  
Figure 5. Video Language Score Comparison of MSDI and 4Dfy.

## 7. Technical Appendices and Supplementary Material

## 7.1. Evaluation Metrics

Pose Plausibility. A lower KL divergence indicates that the pose is more similar to those seen during VPoserâs training, and thus more plausible. The final Pose Plausibility score for a video is the average $\mathcal { L } _ { \mathrm { p l a u s } , t }$ over all ?? frames:

$$
M _ { \mathrm { P l a u s i b i l i t y } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathcal { L } _ { \mathrm { p l a u s } , t } .\tag{12}
$$

It is worth noting that pose plausibility utilizes a pretrained variational autoencoder, whose performance can be constrained by its original training data, potentially limiting generalization to out-of-distribution poses.

Pose Variation. we first compute the standard deviation ???? for each of the ?? pose parameters across time:

$$
\sigma _ { j } = \operatorname { s t d } \bigl ( \{ \phi _ { 1 , j } , \phi _ { 2 , j } , \ldots , \phi _ { T , j } \} \bigr ) , \quad j = 1 , \ldots , K .\tag{13}
$$

A higher value indicates more significant changes in pose throughout the video, suggesting more dynamic motion. The Pose Variation metric is then the mean of these standard deviations:

$$
M _ { \mathrm { V a r i a t i o n } } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \sigma _ { j } .\tag{14}
$$

Trajectory Length. To assess the extent of global character movement within the 3D space, we calculate the trajectory length of the root joint. For each frame ??, HMR2.0 provides the 3D keypoint coordinates. We extract the root jointâs 3D position $\mathbf { k } _ { t } ~ = ~ ( x _ { t } , y _ { t } , z _ { t } )$ . The total trajectory length is the sum of Euclidean distances between the root joint positions in consecutive frames:

$$
M _ { \mathrm { T r a j e c t o r y } } = \sum _ { t = 1 } ^ { T - 1 } | | \mathbf { k } _ { t + 1 } - \mathbf { k } _ { t } | | _ { 2 } .\tag{15}
$$

A longer trajectory length suggests more substantial displacement of the character over time.

Video-Language Score. To measure the semantic alignment between the input text prompt and the generated video, we use InternVideo2 [49], a video-text foundation model. For each of the $N _ { c \nu } = 4$ generated views, we compute the cosine similarity between the text embedding of the prompt and the video embedding. The final Video-Language Score is the average of these similarity scores across all views. A higher score indicates better prompt-video alignment.

4Dfy often achieves a higher Video Language Score (See Figure 5), this may stem from a bias in the metric towards static or motion-limited scenes. Consequently, the metric might prioritize overall scene-text alignment over nuanced motion quality, potentially favoring 4Dfy despite its weaker human motion dynamics. This observation is pertinent, as previous works have often relied on image-based metrics (e.g., CLIP [39] scores) for video-text alignment, which are arguably even less sensitive to temporal dynamics. Moreover, the human evaluation study shows that video generated by our method has high preference (75%) over 4Dfy for Text Alignment (TA). This shows that the video text alignment scores using video language models does not truly capture the human perception of motion quality.

## 7.2. User Study Methodology.

We followed human evaluation methodology established by 4D-fy [4] and MAV3D [43]. We collected responses from 11 human evaluators. For a diverse set of 10 text prompts, each evaluator was shown a pair of videos generated by MSDI and 4D-fy. Participants were asked to choose the superior video based on five criteria:

â¢ Appearance Quality (AQ): The visual clarity and appeal of the generated human and object.

â¢ 3D Structure Quality (SQ): The realism and consistency of the 3D shapes across multiple viewpoints.

â¢ Motion Quality (MQ): The naturalness, dynamism, and physical plausibility of the humanâs movements.

â¢ Text Alignment (TA): How accurately the videoâs content reflects the input text prompt.

â¢ Overall Preference (OP): The evaluatorâs subjective choice for the better video, considering all the above aspects.

## 7.3. Evaluation Prompts

Table 2 lists the text prompts used for the quantitative and qualitative evaluation.

## 7.4. Limitations

Despite its advancements, MSDI has several limitations offering avenues for future work.

First, the final output quality is tied to the pre-generated 3D assets and their initial placement and orientation. Suboptimal inputs or challenging initial setups (e.g., incorrect facing, distant objects) can hinder the generation of plausible interactions, as our framework doesnât currently optimize this initial scene layout.

Second, our reliance on LLMs for initial âcoarseâ trajectory generation can be a bottleneck. LLMs may produce suboptimal, physically impractical, or semantically incorrect paths for complex prompts or environments, providing a poor starting point for optimization.

<!-- image-->  
Figure 6. Generated motion for the prompt: âthe human is playing a drumâ. Top: 4Dfy. Bottom: MSDI

<table><tr><td>Prompt ID</td><td>Prompt Text</td></tr><tr><td>0</td><td>the human walks around the table in a circle and stops close to the start position</td></tr><tr><td>1</td><td>the human prepares to jump for 1 second</td></tr><tr><td>2</td><td>then jumps over the fence the human jumps from the stepstool onto</td></tr><tr><td>3</td><td>the ground the human walks on the clouds</td></tr><tr><td>4</td><td>the human walks towards the lamp</td></tr><tr><td>5</td><td>the human falls down from the stepstool</td></tr><tr><td>6</td><td>the human crawls under the table</td></tr><tr><td>7</td><td>the human prepares to jump for 1 second then jumps onto the table and stops on</td></tr><tr><td>8</td><td>the surface of the table for 1 second the human falls down on the ground</td></tr><tr><td>9</td><td>the human sits down on ground with legs</td></tr><tr><td></td><td>cross</td></tr></table>

Table 2. List of text prompts used for evaluation.

Third, the framework struggles with fine-grained interactions, especially detailed hand and finger movements (e.g., realistically playing a drum, Figure 6). Current motion models and representations lack the specificity for such dexterous tasks, leading to generalized rather than precise contact.

Fourth, while our collision avoidance works for general movements, it may be less robust or efficient for highly complex object geometries or very intricate, close-quarters interactions.

Fifth, MSDI is currently designed for human interactions with static objects. Handling dynamic objects or multi-agent scenarios remains a future challenge.

Finally, the systemâs performance is dependent on the capabilities of the underlying pre-trained motion diffusion models, and the optimization process requires careful hyperparameter tuning to balance different objectives.

## 7.5. Compute Resources

All experiments were conducted on a system equipped with 1 NVIDIA A100 GPUs, 128 CPU cores, and 1TB of CPU memory. Generating a single 4D video clip with 4Dfy (all three of its stages) required approximately 24 hours. MSDI completed the generation of human and object artifacts followed by the optimization process in approximately 5 hours per prompt using the same computational resources.

## 7.6. Multi View Qualitative Results

Figures 7, 8, 9, 10, shows comparison of generated motion with 4Dfy and MSDI from different camera angles.

<table><tr><td> $\because a = - 1$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>111</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>AkAkAAiAd</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>HAAAAAAAA</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>AA4</td><td></td><td></td><td></td><td></td><td></td></tr></table>

<table><tr><td colspan="1" rowspan="1"> $\frac { 3 } { 1 }$ </td><td colspan="1" rowspan="1"> $1 7$ </td><td colspan="1" rowspan="1"> $2 + \cdots$ </td><td colspan="1" rowspan="1"> $y = \frac { 1 } { 2 }$ </td><td colspan="1" rowspan="1"> $1 4$ </td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"> $5 7 ^ { \frac { 3 } { 2 } }$ </td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"> $\frac { 3 } { 7 }$ </td></tr><tr><td colspan="1" rowspan="1">17</td><td colspan="1" rowspan="1"> $m ^ { 2 }$ </td><td colspan="1" rowspan="1"> $\dot { \frac { 3 } { 4 } }$ </td><td colspan="1" rowspan="1"> $a _ { 4 4 }$ </td><td colspan="1" rowspan="1"> $A = 1 0$ </td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">m17</td><td colspan="1" rowspan="1">19</td><td colspan="1" rowspan="1">19  </td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"> $\cdots$ </td><td colspan="1" rowspan="1"> $\frac { A } { H }$ </td><td colspan="1" rowspan="1"> $a _ { T }$ </td><td colspan="1" rowspan="1"> $1 9 - 1$ </td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">LUT</td></tr><tr><td colspan="1" rowspan="1">de 79</td><td colspan="1" rowspan="1">25 </td><td colspan="1" rowspan="1">$</td><td colspan="1" rowspan="1"> $H _ { 2 } = \frac { A } { 2 }$ </td><td colspan="1" rowspan="1"> $\boxed  \begin{array} { r c l } { \boxed { \begin{array} { r l } { \hat { \mathbf { u } } } & { \hat { \mathbf { u } } } & { } \\ { \hat { \mathbf { \mathbf { \phi } } } } & { \ddots } \end{array} } } \end{array}$ </td><td colspan="1" rowspan="1">H</td><td colspan="1" rowspan="1"> $\frac { 3 } { 5 }$ </td><td colspan="1" rowspan="1"> $\gamma _ { H }$ </td><td colspan="1" rowspan="1">Qan  7</td></tr><tr><td>5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>h</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Figure 7. Generated motion for the prompt: âa human walks around a table in a circle and stops close to the start positionâ. Top: 4Dfy. Bottom: MSDI. The four rows illustrate the motion from different camera viewpoints

<table><tr><td rowspan=1 colspan=1> $\mathbf { \underline { { \widehat { \phi } } } } _ { i = 1 } ^ { \theta }$ </td><td rowspan=1 colspan=1> $\begin{array} { c } { { \frac { \hat { \mathbf { a } } } { 2 } } } \\ { { \mathrm { i n i t } 1 } } \end{array}$ </td><td rowspan=1 colspan=1> $\underset { \mathrm { i i f f } } { \overset { \underset { \mathrm { A } } { } } { } }$ </td><td rowspan=1 colspan=1> $\underset { \mathrm { i n i } } { \hat { \boldsymbol { \hat { u } } } } _ { }$ </td><td rowspan=1 colspan=1> $\mathbf { \Sigma } _ { \mathbb { F } ^ { \pm } + 1 } ^ { \mathbb { A } }$ </td><td rowspan=1 colspan=1> $\underset { i = 1 } { \overset { \underset { x } { \iint } } }$ </td><td rowspan=1 colspan=1> $\{ 5$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>*</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $i ^ { \frac { 3 } { 4 } }$ </td><td rowspan=1 colspan=1>y</td><td rowspan=1 colspan=1> $\mathbf { \chi } _ { \mathrm { ~ i ~ } } ^ { \cdot }$ </td><td rowspan=1 colspan=1> $\mathbf { \boldsymbol { \mathsf { x } } } _ { \mathrm { ~ i ~ } }$ </td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>y</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $\frac { 8 } { 1 0 0 }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $\frac { 4 } { 2 }$ </td><td rowspan=1 colspan=1> $\underbrace { \hat { \mathbf { u } } } _ { \mathrm { ~ i ~ f ~ f ~ } }$ </td><td rowspan=1 colspan=1> $\mathbf { \frac { \partial \psi } { \partial t } } _ { \mathbf { \ell } } ^ { \star }$ </td><td rowspan=1 colspan=1> $\underset { \mathbb { T } ^ { 1 } \times 1 } { \mathbb { Z } }$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>o4?</td><td rowspan=1 colspan=1> $\boldsymbol { \underline { { \underline { { \delta } } } } } _ { \mathrm { ~ i ~ } }$ </td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>e</td><td rowspan=1 colspan=1> $\mathrm { ~ i ~ } ^ { \sharp }$ </td><td rowspan=1 colspan=1>Aait</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

Figure 8. Generated motion for the prompt: âthe human prepares to jump for 1 second then jumps over the fenceâ. Top: 4Dfy. Bottom: MSDI. The four rows illustrate the motion from different camera viewpoints

<!-- image-->

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>,8</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>ES</td><td rowspan=1 colspan=1>dEn</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>Qa</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>44  0245)</td><td rowspan=1 colspan=1>4424)</td><td rowspan=1 colspan=1>44(a0)</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>&quot;5</td><td rowspan=1 colspan=1>.Sr</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>it</td></tr><tr><td rowspan=1 colspan=1>N*s</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>*E4</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>dht </td><td rowspan=1 colspan=1>&quot;D&quot; </td><td rowspan=1 colspan=1>&quot;(</td><td rowspan=1 colspan=1>OR:&quot;(</td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>(2)</td><td rowspan=1 colspan=1>i</td></tr></table>

Figure 9. Generated motion for the prompt: âthe human walks towards the lampâ. Top: 4Dfy. Bottom: MSDI. The four rows illustrate the motion from different camera viewpoints

<table><tr><td rowspan=1 colspan=1>t</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr></table>

<table><tr><td rowspan=1 colspan=1> $\frac { 8 } { 1 7 }$ </td><td rowspan=1 colspan=1> $\frac { 8 } { 1 7 }$ </td><td rowspan=1 colspan=1> $\therefore$ </td><td rowspan=1 colspan=1> $5 4$ </td><td rowspan=1 colspan=1> $\boldsymbol { \tau } +$ </td><td rowspan=1 colspan=1> $\oint \limits _ { T } f ( x ) d x = \frac { f ( x ) } { 2 }$ </td><td rowspan=1 colspan=1> $r ^ { \frac { 8 } { 7 } }$ </td><td rowspan=1 colspan=1> $\frac { - \frac { 8 } { 2 } } { 1 - \frac { 8 } { 3 } }$ </td><td rowspan=1 colspan=1> $\frac { 8 } { 5 }$ </td></tr><tr><td rowspan=1 colspan=1>28v17</td><td rowspan=1 colspan=1>tez1Y</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1> $1 4 5$ </td><td rowspan=1 colspan=1> $H _ { 1 } = \frac { \sqrt { 3 } } { 3 }$ </td><td rowspan=1 colspan=1> $H _ { 1 } = \frac { 3 } { 2 }$ </td><td rowspan=1 colspan=1> $m ^ { 2 }$ </td><td rowspan=1 colspan=1> $\underset { H } { \overset { \mathcal { X } } { \mathrm { \Sigma } } }$ </td><td rowspan=1 colspan=1>e</td></tr><tr><td rowspan=1 colspan=1> $\frac { 9 } { 7 9 } = \frac { 1 } { 7 9 }$ </td><td rowspan=1 colspan=1> $\frac { 8 } { 7 } = \frac { 1 } { 7 }$ </td><td rowspan=1 colspan=1> $4 7$ </td><td rowspan=1 colspan=1> $\oint \limits _ { t } ^ { \infty }$ </td><td rowspan=1 colspan=1> $\frac { 4 } { 2 }$ </td><td rowspan=1 colspan=1> $\frac { 9 } { 9 }$ </td><td rowspan=1 colspan=1> $\oint \limits _ { \cdot } ^ { \cdot }$ </td><td rowspan=1 colspan=1> $1$ </td><td rowspan=1 colspan=1> $\sharp$ </td></tr><tr><td rowspan=1 colspan=1>i 77</td><td rowspan=1 colspan=1>iass7</td><td rowspan=1 colspan=1> $\beta _ { n } = \frac { \sqrt { 3 } } { 2 }$ </td><td rowspan=1 colspan=1> $\beta _ { n }$ </td><td rowspan=1 colspan=1> $\beta _ { \texttt { \^ H } }$ </td><td rowspan=1 colspan=1> $\ddot { \ddot { \rho } } _ { \perp }$ </td><td rowspan=1 colspan=1> $\therefore F _ { H }$ </td><td rowspan=1 colspan=1> $\bf \Pi _ { \Pi \Pi \hat { H } } ^ { ' \dagger }$ </td><td rowspan=1 colspan=1> $\sharp$ </td></tr></table>

Figure 10. Generated motion for the prompt: âthe human prepares to jump for 1 second then jumps onto the table and stops on the surface of the table for 1 secondâ. Top: 4Dfy. Bottom: MSDI. The four rows illustrate the motion from different camera viewpoints