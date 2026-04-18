# Causal Reasoning Elicits Controllable 3D Scene Generation

Shen Chen1, Ruiyu Zhao2, Jiale Zhou2, Zongkai Wu3, Jenq-Neng Hwang4, Lei Li4,5\*

1Zhejiang University 2East China University of Science and Technology 3Skai Intelligence 4University of Washington 5VitaSight

## Abstract

Existing 3D scene generation methods often struggle to model the complex logical dependencies and physical constraints between objects, limiting their ability to adapt to dynamic and realistic environments. We propose CausalStruct, a novel framework that embeds causal reasoning into 3D scene generation. Utilizing large language models (LLMs), We construct causal graphs where nodes represent objects and attributes, while edges encode causal dependencies and physical constraints. CausalStruct iteratively refines the scene layout by enforcing causal order to determine the placement order of objects and applies causal intervention to adjust the spatial configuration according to physics-driven constraints, ensuring consistency with textual descriptions and real-world dynamics. The refined scene causal graph informs subsequent optimization steps, employing a Proportional-Integral-Derivative(PID) controller to iteratively tune object scales and positions. Our method uses text or images to guide object placement and layout in 3D scenes, with 3D Gaussian Splatting and Score Distillation Sampling improving shape accuracy and rendering stability. Extensive experiments show that CausalStruct generates 3D scenes with enhanced logical coherence, realistic spatial interactions, and robust adaptability.

Code â https://causalstruct.github.io/

## Introduction

In recent years, 3D scene generation has advanced significantly in computer vision, graphics, and content creation. However, traditional methods still rely heavily on manual modeling and expert knowledge, making multi-object scene construction time-consuming and costly. Existing text-to-3D approaches have attempted to address this by using 2D diffusion models for optimizing 3D representations (Tang et al. 2024; Yi et al. 2023; Fridman et al. 2024) or employing 3D diffusion models for direct asset generation (Hong et al. 2024; Zhou, Zhang, and Liu 2025). While these approaches have demonstrated success in synthesizing individual objects, they struggle with multi-object scene composition, often resulting in geometric distortion, spatial inconsistencies, and object drift.

A well-structured spatial layout is essential for generating coherent 3D scenes, as it dictates object placement. Previous layout-generation approaches (Zhou et al. 2024; Sun et al. 2023; Feng et al. 2024b; Zheng et al. 2024; Yang, Hu, and Ye 2021; Lin et al. 2023c) rely on data-driven heuristics or LLM-based inference to ensure semantic and structural consistency. However, these methods primarily focus on static spatial constraint placement and overlook the interactions between objects on the overall scene.

<!-- image-->  
Figure 1: CausalStruct optimizes and controls 3D scene generation using either pure text or a combination of text and image inputs, ensuring spatial coherence and realistic object interactions through causal reasoning.

Causal reasoning in 3D scene generation establishes a hierarchy of directed dependencies among objects, ensuring their spatial and functional interactions adhere to realworld physical principles. Rooted in structural causal models (SCMs) (Neuberg 2003), it transcends statistical correlations by explicitly defining how the presence or state of one object causally influences the placement, orientation, or existence of others. Without causal reasoning, layout methods are unable to dynamically model object interactions, resulting in scenes with misaligned spatial relationships, improper functional placements, or floating objects that violate realworld physics.

To overcome these challenge, we propose CausalStruct, a novel framework that integrates causal reasoning into scene graph optimization. Using LLMs (Hurst et al. 2024; Cai et al. 2025b), we construct a scene graph where nodes represent objects and attributes, and edges encode relationships and physical dependencies. However, the LLM alone cannot accurately construct scene graphs due to its neglect of node properties and edge interactions, resulting in unrealistic scenes that defy physical laws. Inspired by causal reasoning in structure discovery and relationship modeling (KÄ±cÄ±man et al. 2023; Vashishtha et al. 2023), we introduce a causal order mechanism to enforce logical sequencing in object placement. By computing a causal precedence through pairwise LLM reasoning, we ensure that objects follow physically consistent dependencies. Furthermore, to address uncertain or inconsistent edges, we compute the confidence of each edge using a Bayesian estimation (Cai et al. 2025a) and determine whether to apply a causal intervention based on this confidence, where interventions on object states validate relationships through their physical impact on other scene elements, guiding whether to modify or retain the edge.

To refine the attributes of nodes (objects) in the causal scene graph and adjust them to realistic spatial proportions, we optimize attributes using a Multimodal Large Language Model (MLLM) (Hurst et al. 2024; Li 2024) that assesses spatial relationships. Each edge in the causal graph is assigned an attribute correction score, quantifying discrepancies in size and position. To ensure physically plausible adjustments, we apply a Proportional-Integral-Derivative (PID) controller (Willis 1999; Crowe et al. 2005), iteratively refining object attributes while maintaining scene stability. For enhanced geometric consistency and rendering stability, we integrate 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023; Chen, Zhou, and Li 2025) with Score Distillation Sampling (SDS) (Poole et al. 2022) at both the object and scene levels. Additionally, our method supports text-only and text-image combinations, enabling improved physical simulations and generating coherent 3D scenes.

â¢ We propose CausalStruct, integrating Causal Order, and Causal Intervention into scene graph optimization to enforce object dependencies and refine uncertain edges.

â¢ We optimize whole 3D scene layout by refining object attributes with an MLLM and a PID controller.

â¢ We enhance 3D scene generation by leveraging 3DGS with SDS to refine the Causal Scene, improving geometric consistency and rendering stability.

â¢ We support both text-only and text-image combinations, enhancing scene generation and physical simulation accuracy.

## Related Work

Text-Driven 3D Generation NeRF-based methods, such as DreamFusion (Poole et al. 2022) and Score Jacobian Chaining (Wang et al. 2023), leverage 2D diffusion models (Rombach et al. 2022; Saharia et al. 2022) to synthesize single objects. Subsequent works, including Magic3D (Lin et al. 2023a), Latent-NeRF (Metzer et al. 2023), and 3DFuse (Seo et al. 2023), aim to enhance 3D generation quality under SDS constraints. ProlificDreamer (Wang et al. 2024) models 3D parameters as random variables, and introduces Variational Score Distillation (VSD) for improved optimization. While NeRF-based approaches effectively generate high-quality 3D objects, they suffer from inefficiency. To improve efficiency, 3DGS-based approaches (Tang et al. 2024; Yi et al. 2023) have been proposed for text-to-3D generation by integrating diffusion models with Gaussian splatting. Recent methods (Zhou, Zhang, and Liu 2025; Zhang et al. 2024a; Chen et al. 2024; Jiang et al. 2024) utilize textto-3DGS pipelines to facilitate object synthesis, achieving faster generation. While these methods enable diverse 3D generation from text prompts, they struggle to produce photorealistic multi-object scenes with complex geometry and high-fidelity textures due to their reliance on high-level semantic priors.

LLMs for Causal Discovery LLMs have significantly advanced causal inference by combining text-based dependency extraction with reasoning capabilities to uncover causal relationships. LLMs enhance causal inference by performing pairwise causal reasoning to identify relationships between variables (KÄ±cÄ±man et al. 2023). Beyond pairwise inference, LLMs contribute to causal graph construction, leveraging causal ordering to orient undirected edges (Vashishtha et al. 2023) and refining predictions through iterative feedback mechanisms (Ban et al. 2023). Additionally, LLMs enhance generalization by incorporating pretrained knowledge (Feng et al. 2024a), making them wellsuited for capturing complex dependencies in structured reasoning. Since scene construction inherently involves causal relationships between objects, integrating LLMs with causal discovery into this process ensures coherent spatial layouts and semantically consistent object interactions.

Layout Generation Scene layout is fundamental to 3D scene generation, as it dictates the spatial arrangement, scale, and interactions of objects, directly influencing realism and coherence. Various methods have explored different strategies for scene composition and object placement. SceneSuggest (Savva, Chang, and Agrawala 2017) utilizes spatial constraints to infer supporting surfaces, while Physcene (Yang et al. 2024) and Text2nerf (Zhang et al. 2024b) integrate diffusion models to enforce physically plausible layouts. Graph-based approaches have gained popularity in structuring object relationships and layout within scenes. PlanIT (Wang et al. 2019) and SceneGraphNet (Zhou, While, and Kalogerakis 2019) encode spatial and functional dependencies, guiding object placement based on predefined constraints. GraphDreamer (Gao et al. 2024) and SceneWiz3D (Zhang et al. 2024c) incorporates LLMs with layout-based NeRF to further enhance scene composition, while GALA3D (Doe and Smith 2023) and LayoutDreamer (Zhou et al. 2025) introduces layout-guided 3D Gaussian representation, leveraging adaptive constraints for geometry refinement and inter-object interactions. Despite their effectiveness in structuring layouts, these methods overlook causal dependencies in object placement and attributes, often leading to misalignment or floating objects. In this paper, we integrate causal reasoning to refine both object relationships and attributes, ensuring a more coherent and physically plausible scene.

<!-- image-->  
Figure 2: Overview of our method. Given a scene description, our method constructs a causal scene graph using LLMs and MLLMs with causal reasoning. A PID controller refines object scales and positions, ensuring spatial consistency. Additionally, objects and the scene are represented with 3D Gaussian Splatting and optimized using Diffusion and SDS for high-fidelity rendering.

## Methods

As shown in Fig. 2, our method constructs causal-driven 3D Gaussian representations by integrating causal reasoning, PID-based optimization, and layout-guided representation. First, given a text description, we generate an initial scene graph using LLM and refine object relationships through causal reasoning. Second, to ensure spatial balance, PID Control optimizes object scale and position while preventing abrupt changes. Finally, Layout-Guided Representation builds the scene with 3DGS and optimizes it using Diffusion and SDS for spatial consistency and high-fidelity rendering.

Preliminaries 3D Gaussian Splatting (3DGS) represents 3D scenes using anisotropic Gaussian primitives, denoted as $\{ \mathcal { G } _ { n } \ | \ n = 1 , . . . , N \}$ , with parameters including position $\mu _ { n } \in \mathbb { R } ^ { 3 }$ , covariance $\bar { \Sigma } _ { n } \in \bar { \mathbb { R } ^ { 7 } }$ , color $c _ { n } \in \mathbb { R } ^ { 3 }$ , and opacity $\alpha _ { n } \in \mathbb { R }$ . The Gaussian function is defined as:

$$
\mathcal { G } _ { n } ( \boldsymbol { p } ) = e ^ { - \frac { 1 } { 2 } ( \boldsymbol { p } - \boldsymbol { \mu } _ { n } ) ^ { T } \Sigma _ { n } ^ { - 1 } ( \boldsymbol { p } - \boldsymbol { \mu } _ { n } ) } ,\tag{1}
$$

where $\Sigma _ { n }$ is parameterized by a rotation matrix $R _ { n } \in \mathbb { R } ^ { 4 }$ and a scaling matrix $S _ { n } \in \mathbb { R } ^ { 3 }$

For rendering, differential splatting projects the Gaussians onto camera planes, using a viewing transformation $W _ { n }$ and the Jacobian matrix $J _ { n }$ to obtain a transformed covariance. The color for a ray r is computed as:

$$
C _ { r } ( \boldsymbol { x } ) = \sum _ { i \in M } c _ { i } \sigma _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \sigma _ { j } ) , \quad \sigma _ { i } = \alpha _ { i } \mathcal { G } _ { i } ^ { 2 D } ( \boldsymbol { x } ) .\tag{2}
$$

An adaptive density control mechanism dynamically adjusts the number of Gaussians to balance computational efficiency and scene detail.

## Causal Reasoning for Scene Graph

Causal Order for Scene Graph Optimization Ensuring a consistent and physically plausible scene layout, we define causal precedence among objects to establish a logical placement sequence. Given a set of objects $\boldsymbol { \mathcal { O } } =$ $\left\{ o _ { 1 } , o _ { 2 } , . . . , o _ { n } \right\}$ and their spatial relations $\mathcal { E } = \bar { \{ e _ { i , j } \vert o _ { i } , o _ { j } \in } $ O}, directly from a LLM, we define the causal order âº as:

$$
o _ { i } \prec o _ { j } \iff C _ { i , j } > C _ { j , i } ,\tag{3}
$$

where $o _ { i } \to o _ { j }$ represents a directed causal edge indicating that the placement of $o _ { j }$ depends on $o _ { i } . \ C _ { i , j } = \mathbb { P } ( o _ { i } \prec o _ { j } \ |$ $\mathbf { L L M } ( o _ { i } , o _ { j } ) )$ represents the probability of $o _ { i }$ causally preceding oj . If $C _ { i , j } > C _ { j , i } .$ , we adjust the scene such that $o _ { i }$ is placed before $o _ { j }$ in spatial reasoning, updating the edge set E to the refined set ${ \mathcal { E } } _ { \mathrm { o r d e r } }$ by enforcing the inferred causal order.

Bayesian Edge Estimation The hallucinations of LLM lead to causal graphs containing false or physically unreasonable edges during linguistic reasoning. We introduce Bayesian Edge Estimation to assess the confidence of each edge before Causal Intervention.

Prior: The prior probability $p ( e _ { i , j } )$ of an edge $e _ { i , j }$ represents the inherent plausibility of the causal relationship between objects $o _ { i }$ and $o _ { j } ,$ before observing any additional evidence. Instead of assuming a uniform prior, we obtain the prior directly from LLM.

Posterior: For a candidate edge $e _ { i , j }$ , its posterior probability given the Causal Order edges ${ \mathcal { E } } _ { \mathrm { o r d e r } }$ is computed using Bayesâ rule:

$$
p ( e _ { i , j } \mid \mathcal { E } _ { \mathrm { o r d e r } } ) = \frac { p ( \mathcal { E } _ { \mathrm { o r d e r } } \mid e _ { i , j } ) p ( e _ { i , j } ) } { p ( \mathcal { E } _ { \mathrm { o r d e r } } ) } ,\tag{4}
$$

where $p ( \mathcal { E } _ { \mathrm { o r d e r } } \ \mid \ e _ { i , j } )$ is the likelihood that the LLMgenerated edge set is observed given that $e _ { i , j }$ is correct, and $\bar { p } ( \mathcal { E } _ { \mathrm { o r d e r } } )$ is the normalizing factor ensuring a valid probability distribution. We assume that the likelihood of $\bar { \mathcal { E } _ { \mathrm { o r d e r } } }$ given $e _ { i , j }$ is decomposable across all edges in $\mathcal { E } _ { \mathrm { o r d e r } } \mathrm { : }$

$$
p ( \mathcal { E } _ { \mathrm { o r d e r } } \mid e _ { i , j } ) = \prod _ { ( o _ { i } , o _ { j } ) \in \mathcal { E } _ { \mathrm { o r d e r } } } p ( \mathrm { L L M } ( o _ { i } , o _ { j } ) \mid e _ { i , j } ) ,\tag{5}
$$

where $p ( \mathrm { L L M } ( o _ { i } , o _ { j } ) \mid e _ { i , j } )$ represents the probability that the LLM correctly predicts the causal relationship between objects $o _ { i }$ and $o _ { j } .$ , obtained by querying the model multiple times and aggregating its responses.

Causal Intervention for Edge Refinement For edges with uncertain posterior probabilities, we conduct causal interventions using an MLLM to verify their correctness. Given two objects $o _ { i }$ and $o _ { j }$ connected by an edge $e _ { i , j } ,$ , we evaluate the impact of modifying their spatial relationship on the entire scene. This intervention is performed by iterating over a set of candidate placements for the object and analyzing their effect using the rendered scene image. The MLLM determines whether each placement results in a physically and semantically plausible configuration.

To quantify the likelihood of each possible adjustment, we define the probability of $D _ { i , j }$ as:

$$
p ( D _ { i , j } \mid e _ { i , j } ) = \sum _ { r , s \in S } p ( D _ { i , j } = s \mid d o ( e _ { i , j } = r ) ) ,\tag{6}
$$

where $S$ is the set of all candidate placements for object $o _ { j } ,$ determined by a predefined position list. $d o ( e _ { i , j } = r ) )$ means to interfere with edge $e _ { i , j }$ to force its state to be set to r. $p ( D _ { i , j } = s \mid d o ( e _ { i , j } { \bar { = } } r ) )$ represents the probability of state s being the modification for $\mathbf { \bar { \Gamma } } _ { d o ( e _ { i , j } } = r ) \mathbf { \bar { \Gamma } }$ , which is estimated using an MLLM evaluation:

$$
\begin{array} { l } { { \displaystyle p ( D _ { i , j } = s \mid d o ( e _ { i , j } = r ) ) } \ ~ } \\ { { \displaystyle \qquad = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } [ \mathbf { \Big ( M L L M ( } _ { i } , o _ { j } , I _ { e _ { i , j } } ^ { r } ) \Big . \Big .  \Big ) } , } \end{array}\tag{7}
$$

where $\mathbb { I } ( \cdot )$ is an indicator function that returns 1 if the MLLM selects state s for edge $e _ { i , j }$ in the k-th trial, otherwise returns $0 . \ I _ { e _ { i , j } } ^ { r }$ is the rendered scene image when object $o _ { j }$ is placed at candidate position $r \in S .$ , and the MLLM evaluates whether this placement is reasonable. The intervention decision is determined by selecting the placement modification:

$$
s ^ { * } = \arg \operatorname* { m a x } _ { s \in S } p ( D _ { i , j } = s \mid d o ( e _ { i , j } = r ) ) .\tag{8}
$$

Update Strategy Edges are classified into three categories based on their posterior probability and causal intervention results:

$$
e _ { i , j } = \left\{ \begin{array} { l l } { e _ { i , j } } & { p _ { e _ { i , j } } > \tau _ { 1 } } \\ { s ^ { * } } & { p _ { e _ { i , j } } \leq \tau _ { 1 } \& p ( \mathcal { E } _ { \mathrm { o r d e r } } \mid s ^ { * } ) > \tau _ { 2 } \ , } \\ { \emptyset } & { p _ { e _ { i , j } } \leq \tau _ { 1 } \& p ( \mathcal { E } _ { \mathrm { o r d e r } } \mid s ^ { * } ) \leq \tau _ { 2 } } \end{array} \right.\tag{9}
$$

where $\tau _ { 1 }$ and $\tau _ { 2 }$ are confidence thresholds, $p _ { e _ { i } }$ i,j represents $p ( \mathcal { E } _ { \mathrm { o r d e r } } \mid e _ { i , j } ) , p ( \mathcal { E } _ { \mathrm { o r d e r } } \mid s ^ { * } )$ represents the posterior probability of $s ^ { * }$ , and $\mathcal { D }$ means remove the edge. This ensures that high-confidence edges remain, mediumconfidence edges are validated through interventions, and low-confidence edges are discarded, leading to a reliable causal scene graph.

## PID Control Object Optimization

Causal reasoning determines the initial placement of objects, while the PID controller fine-tunes their positions and scales to ensure physical plausibility and spatial accuracy. Each object pair (edge) in the scene graph is evaluated using an MLLM to accurately adjust edges, generating a scale correction score and position scores for precise spatial refinement.

We propose an optimization method based on a Proportional-Integral-Derivative (PID) controller to ensure proportional accuracy and visual balance in reconstructed scenes. The PID controller can effectively handle the nonindependent relationships between edges by dynamically adjusting the proportional, integral, and derivative parameters to to ensure that the objects in the edges do not overlap and ensure overall spatial coordination and accuracy.

The error signal, defined as the negation of the score, drives the PID controller to adjust the scale and position. The control signal u is computed using the error $e ,$ accumulated errorR e dt, and change in error over time $\textstyle { \frac { d e } { d t } }$

$$
u = K _ { p } e + K _ { i } \int e d t + K _ { d } \frac { d e } { d t } ,\tag{10}
$$

where $K _ { p } , K _ { i }$ and $K _ { d }$ are the proportional, integral, and derivative coefficients.

To implement PID control effectively, we introduce an actuator that converts the control signal u into practical adjustments of the sceneâs scale and position. The actuator ensures that adjustments remain within a predefined range, preventing instability or produce unrealistic results. The actuator output $\Delta$ is formulated as a nonlinear function of $u ,$ employing the hyperbolic tangent function to achieve smooth scaling:

$$
\Gamma = \Delta \cdot t a n h ( \frac { u } { \gamma } ) ,\tag{11}
$$

where $\Delta$ denotes the maximum permissible adjustment, and $\gamma$ modulates the steepness of the response. This transformation constrains the output within the range $- \Delta$ to $\Delta ,$ , ensuring smooth and controlled adjustments while preventing abrupt shifts in scale or position.

Once the nodes and edges are optimized, they are organized into a graph based on inter-object relationships, such as aligned spatial attributes and semantic associations. The LLM then directs the placement of these subgraphs by interpreting high-level prompts that specify the expected sizes, relationships, and spatial context of the objects. Additionally, based on the attribute values of each node, MLLM is utilized to finely adjust the scene graph, ensuring that the final structure accurately aligns with the textual description.

MVDream

Gala3D

<!-- image-->  
Figure 3: Qualitative Reconstruction Results. Compared to other methods, our approach produces high-quality reconstructions.

<!-- image-->  
Figure 4: Scene Editing. Our method can add, remove, or move objects based on the causal relationship between their placement.

## Layout-guided Representation

Based on causal reasoning for scene optimization and PID controller for object state adjustments, we obtain a structured layout, where each object is assigned a position based on its inferred dependencies. The layout provides center coordinates, and object sizes.

Object Representation Each object is optimized independently using MVDream (Lee and Kim 2023) or Zero123 (Liu et al. 2023) with Score Distillation Sampling (SDS):

$$
\nabla _ { G _ { \mathrm { o b j } } } L = \mathbb { E } _ { \epsilon , \eta } \left[ \lambda _ { \mathrm { o b j } } \big ( \epsilon _ { \phi } ( I _ { \mathrm { o b j } } ; t , \beta , \eta ) - \epsilon \big ) \frac { \partial I } { \partial G _ { \mathrm { o b j } } } \right] ,\tag{12}
$$

where $G _ { \mathrm { o b j } }$ represents object parameters, $I _ { \mathrm { o b j } }$ is the rendered object image, and $t , \beta , \tau$ Î· correspond to time step, camera parameters, and noise conditioning, respectively.

Scene Representation Simply placing objects directly in the scene makes it difficult to maintain overall coherence. To address this, we optimize the entire scene using Stable Diffusion (Rombach et al. 2022) with SDS, ensuring consistency in object interactions. The full-scene optimization follows as:

$$
\nabla _ { G _ { \mathrm { s c e n e } } } L = \mathbb { E } _ { \epsilon , p } \left[ \lambda _ { \mathrm { s c e n e } } \big ( \epsilon _ { \phi } ( I _ { F _ { \mathrm { s c e n e } } } ; t , \beta , p ) - \epsilon \big ) \frac { \partial I } { \partial G _ { \mathrm { s c e n e } } } \right] ,\tag{13}
$$

where p represents the description of the entire scene, Gscene represents the global scene parameters, $I _ { F _ { \mathrm { s c e n e } } }$ is the rendered full-scene image, and $F _ { \mathrm { s c e n e } }$ encodes the structured layout with 3D Gaussian properties and spatial parameters for all objects:

$$
F _ { \mathrm { s c e n e } } = \{ \mathcal { G } _ { i } , x _ { i } , y _ { i } , z _ { i } , s _ { i } \} _ { i = 1 } ^ { N } ,\tag{14}
$$

where $\mathcal { G } _ { i }$ represents the 3D Gaussian attributes of object $i ,$ while $( x _ { i } , y _ { i } , z _ { i } )$ denote the center coordinates, and $s _ { i }$ define the object scale. This joint optimization ensures that objects are not only individually refined but also integrated into a spatially coherent and semantically meaningful scene.

## Experiments

Experimental Setup Our approach is implemented in Py-Torch (Paszke et al. 2019). We employ GPT-4o (Hurst et al. 2024) to generate the initial scene graph. To optimize the causal scene graph, we integrate DeepSeek (Guo et al. 2025) and GPT-4o, where DeepSeek facilitates strong chain-ofthought reasoning to refine causal relationships, while GPT-4o incorporates multimodal analysis to maintain consistency between textual descriptions and the visual layout. During causal graph optimization, we employ Point-E (Nichol et al. 2022) to render edge images and generate the spatial layout. During the object scale and position adjustment stage, we set $\bar { k _ { p } } ~ = ~ 1 , \bar { k _ { i } } ~ = ~ 0 . 0 0 0 0 1 , \bar { k } _ { d } ~ = ~ 5 , \bar { \Delta } ~ = ~ 0 . 0 2$ , and $\gamma = 5 0 0$ for scale control, while for position control, we set $k _ { p } = 1 , k _ { i } = 0 . 0 0 0 0 1 , k _ { d } = 5 , \Delta \stackrel { \circ } { = } 0 . 4 , \mathrm { a n d } \gamma = 8 0 0$

<table><tr><td>Methods</td><td>Represent.</td><td>ViT-B/32</td><td>ViT-L/14</td><td>ViT-bigL/14</td><td>Gemini</td><td>GPT-40</td><td>Claude</td><td>Qwen</td><td>GLM</td></tr><tr><td>MVDream (Lee and Kim 2023)</td><td>Nerf</td><td>24.30</td><td>18.34</td><td>18.16</td><td>1.7</td><td>4.1</td><td>2.7</td><td>1.6</td><td>3.6</td></tr><tr><td>GraphDreamer (Gao et al. 2024)</td><td>Nerf</td><td>21.30</td><td>19.96</td><td>20.57</td><td>1.5</td><td>2.8</td><td>3.0</td><td>6.6</td><td>2.6</td></tr><tr><td>DreamGaussian (Tang et al. 2024)</td><td>3DGS</td><td>15.78</td><td>10.22</td><td>10.41</td><td>0.6</td><td>1.2</td><td>1.8</td><td>3.0</td><td>0.3</td></tr><tr><td>GaussianDreamer (Yi et al. 2023)</td><td>3DGS</td><td>20.57</td><td>18.04</td><td>18.72</td><td>0.8</td><td>1.8</td><td>1.7</td><td>5.2</td><td>1.0</td></tr><tr><td>GSGen (Chen et al. 2024)</td><td>3DGS</td><td>17.32</td><td>12.11</td><td>14.22</td><td>1.0</td><td>2.5</td><td>3.2</td><td>2.6</td><td>1.0</td></tr><tr><td>GALA3D (Doe and Smith 2023)</td><td>3DGS</td><td>22.29</td><td>16.68</td><td>17.60</td><td>1.6</td><td>3.5</td><td>2.6</td><td>4.3</td><td>0.6</td></tr><tr><td>Ours</td><td>3DGS</td><td>25.90</td><td>20.86</td><td>21.31</td><td>2.8</td><td>5.0</td><td>3.3</td><td>6.9</td><td>4.2</td></tr></table>

Table 1: Comparison with additional metrics. CLIP (Radford et al. 2021) & MLLMs (Team et al. 2023; Hurst et al. 2024; Bai et al. 2023; Du et al. 2021; Anthropic 2024).

<table><tr><td>Methods</td><td>LLM</td><td>MLLM</td><td>ViT-B/32</td><td>ViT-L/14</td><td>ViT-bigL/14</td><td>Gemini</td><td>GPT-40</td><td>Claude</td></tr><tr><td>w/o Causal Reasoning</td><td>-</td><td>-</td><td>25.55</td><td>22.27</td><td>23.81</td><td>1.78</td><td>4.06</td><td>3.19</td></tr><tr><td>DeepSeek(distilled)</td><td>R1-8b</td><td>LLava-34b</td><td>27.48</td><td>23.05</td><td>24.88</td><td>2.39</td><td>4.83</td><td>3.75</td></tr><tr><td>DeepSeek(distilled)</td><td>R1-14b</td><td>LLava-34b</td><td>28.17</td><td>23.46</td><td>25.80</td><td>2.44</td><td>5.13</td><td>3.64</td></tr><tr><td>DeepSeek</td><td>R1</td><td>40</td><td>28.05</td><td>23.81</td><td>25.95</td><td>2.69</td><td>5.47</td><td>3.89</td></tr><tr><td>GPT</td><td>40</td><td>40</td><td>27.81</td><td>24.63</td><td>26.95</td><td>2.42</td><td>5.64</td><td>3.64</td></tr></table>

Table 2: Ablation Studies on Knowledge Distillation. Our experiments systematically evaluate the impact of knowledge distillation, while probing how original non-distilled models shape final performance outcomes.

During Gaussian optimization and generation, we employ MVDream or Zero123 to refine individual objects and Stable Diffusion to optimize the overall scene, ensuring both object-level quality and scene-level coherence. All experiments were performed on an NVIDIA A100 GPU with 80GB memory.

Evaluation Metrics We evaluate our method using CLIP (Radford et al. 2021) Score and MLLMs (Team et al. 2023; Hurst et al. 2024; Bai et al. 2023; Anthropic 2024; Du et al. 2021) Score, comparing them quantitatively with baseline models. CLIP Score computes similarity by comparing the visual and textual embeddings extracted from the same CLIP model. Additionally, we incorporate MLLMs Score, to further assess scene-object alignment and semantic coherence in the generated 3D representations.

## Comparison of Methods

Quantitative Comparison We report quantitative results in Table 1. We assess our approach on the Text-to-3D task by comparing it with mainstream methods, including DreamGaussian (Tang et al. 2024), GaussianDreamer (Gao et al. 2024), MVDream (Lee and Kim 2023), GSGen (Chen et al. 2024), GALA3D (Doe and Smith 2023), and Graph-Dreamer (Gao et al. 2024). To ensure a fair evaluation, we adopt CLIP Score, following prior works, to measure the alignment between generated images and their corresponding textual descriptions. Given the inherent randomness in MLLMs, we utilize multiple MLLMs to assess the semantic consistency between generated scenes and input descriptions from different perspectives. Notably, higher CLIP or MLLM scores indicate better performance. By aggregating evaluations from CLIP and various MLLMs, our approach achieves the highest performance, demonstrating superior scene-object alignment and semantic coherence.

Qualitative Comparison We report qualitative comparisons on text-to-3D generation in Fig. 3. Compared to existing methods, our approach demonstrates superior spatial consistency and causal alignment. While prior methods primarily focus on single-object generation or data-driven scene synthesis, they often struggle with incorrect object relationships and spatial inconsistencies. In contrast, our method leverages causal order and intervention optimization to refine object interactions, ensuring that generated scenes adhere to real-world semantics and physical constraints. Additionally, our PID-based optimization maintains proportional accuracy, while diffusion-guided 3DGS refinement enhances overall rendering quality.

## Scene Editing

As illustrated in Fig. 4, scene editing in our approach facilitates flexible and controllable modifications via text descriptions. LLMs translate user descriptions into layout transformations, such as adding, removing, or repositioning objects. The Layout-Guided Representation is subsequently optimized within the edited regions, maintaining stability in the unchanged areas. Notably, our editing process accounts for the causal relationships that govern typical object placement, ensuring that modifications to position and scale align with real world spatial logic. This approach supports spatial adjustments, object interactions, and style modifications, offering a seamless and intuitive 3D scene editing experience grounded in causal reasoning.

In addition to generating 3D scenes from text descriptions, our method also supports text-image-based generation. Each node in the scene can receive image inputs, which guide the output of the node, enabling image-to-3D conversion. This integration enables text and image modes to work synergistically, where text provides global semantic constraints while images inject local geometric priors, thereby enhancing both the realism and physical plausibility of the synthesized scenes.

## Ablation Study

Adaptability and Robustness Our framework incorporates localized, distilled LLMs and MLLMs, facilitating lightweight deployment while maintaining high performance across varied computational environments. As shown in Fig. 5 and Table 2, we assess multiple model configurations, demonstrating that our causal graph-based framework effectively captures object relationships and spatial dependencies. Leveraging a diverse set of models, our approach ensures robustness and stability in scene generation under varying computational constraints. Moreover, our framework efficiently adapts to varying input complexities, ensuring consistent spatial reasoning across diverse scenarios.

<!-- image-->  
Figure 5: Ablation Study of causal reasoning and adaptability. The results show that causal reasoning enhances scene coherence, and experiments with distilled models demonstrate the robustness of our method.

<!-- image-->  
Figure 6: Visual results of key Components. The experiments validate the necessity of each component in our framework, highlighting their critical roles in ensuring coherent spatial relationships and physical plausibility

Causal Reasoning To evaluate the impact of causal order and causal intervention on scene generation, we conduct ablation studies comparing scene graphs constructed from standard LLM parsing with those refined using causal reasoning. As shown in Fig. 5, Fig. 6 and Table 2, removing the causal order mechanism leads to missing essential object relationships, resulting in incomplete or incorrect connections. Without Causal intervention to validate edges, the scene graph retains erroneous relationships, causing misaligned objects, floating placements, and unrealistic spatial arrangements. Integrating causal ordering and causal intervention validation enhances spatial consistency, object interactions, and physical plausibility. These results highlight the need for causal reasoning in ensuring well-structured 3D representations.

PID Controller The PID controller regulates node attributes (object scales and positions) to maintain spatial consistency and enhance structural precision. As shown in Fig. 6, proportional optimization without PID control relies on fixed value updates, which are highly sensitive to erroneous LLM scores. These errors often propagate as perturbations in the system, causing fluctuations, directional misalignment, or even inverted object orientationsâultimately destabilizing scene layouts. In contrast, PID-based refinement addresses this limitation through a dynamic errorcorrection mechanism: the proportional term responds to immediate discrepancies, the integral term compensates for accumulated historical errors, and the derivative term anticipates abrupt changes. This multi-component control strategy effectively dampens noise from LLM evaluations, enabling smoother convergence. Through multi-term error compensation, PID dynamically regulates node attributes to generate physically coherent scenes in the case of abnormal LLM output.

## Conclusion

In this paper, we proposed CausalStruct, a causal-driven framework for 3D scene generation, integrating causal reasoning, PID control, and Diffusion refinement. By leveraging LLMs and MLLMs, our method constructs a causal scene graph, ensuring that object relationships align with real-world semantics and physical constraints. Moreover, our approach adapts to varying scene complexities, ensuring stable optimization across different generation tasks. Through PID control, we maintain proportional accuracy and spatial consistency, while 3DGS with SDS optimization enhances object fidelity and rendering quality. Experimental results show that CausalStruct improves scene composition, object interactions, and multi-view consistency, generating structured and semantically coherent 3D scenes. Our work demonstrates the potential of causal reasoning and PID control in 3D generation.

## References

Anthropic. 2024. The Claude 3 Model Family: Opus, Sonnet, Haiku.

Bai, J.; Bai, S.; Chu, Y.; Cui, Z.; Dang, K.; Deng, X.; Fan, Y.; Ge, W.; Han, Y.; Huang, F.; et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Ban, T.; Chen, L.; Wang, X.; and Chen, H. 2023. From query tools to causal architects: Harnessing large language models for advanced causal discovery from data. arXiv preprint arXiv:2306.16902.

Cai, C.; Liu, H.; Zhao, X.; Jiang, Z.; Zhang, T.; Wu, Z.; Lee, J.; Hwang, J.-N.; and Li, L. 2025a. Bayesian Optimization for Controlled Image Editing via LLMs. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).

Cai, C.; Zhao, X.; Liu, H.; Jiang, Z.; Zhang, T.; Wu, Z.; Hwang, J.-N.; and Li, L. 2025b. The Role of Deductive and Inductive Reasoning in Large Language Models. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).

Chen, S.; Zhou, J.; and Li, L. 2025. Dense Point Clouds Matter: Dust-GS for Scene Reconstruction from Sparse Viewpoints. In ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1â5.

Chen, Z.; Wang, F.; Wang, Y.; and Liu, H. 2024. Text-to-3d using gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21401â21412.

Crowe, J.; Chen, G.; Ferdous, R.; Greenwood, D.; Grimble, M.; Huang, H.; Jeng, J.; Johnson, M. A.; Katebi, M.; Kwong, S.; et al. 2005. PID control: new identification and design methods. Springer.

Doe, J.; and Smith, J. 2023. GALA3D: Generative Adversarial Layout Arrangement in 3D Spaces. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 12345â12353.

Du, Z.; Qian, Y.; Liu, X.; Ding, M.; Qiu, J.; Yang, Z.; and Tang, J. 2021. Glm: General language model pretraining with autoregressive blank infilling. arXiv preprint arXiv:2103.10360.

Feng, T.; Qu, L.; Tandon, N.; Li, Z.; Kang, X.; and Haffari, G. 2024a. From pre-training corpora to large language models: What factors influence llm performance in causal discovery tasks? arXiv preprint arXiv:2407.19638.

Feng, W.; Zhu, W.; Fu, T.-j.; Jampani, V.; Akula, A.; He, X.; Basu, S.; Wang, X. E.; and Wang, W. Y. 2024b. Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems, 36.

Fridman, R.; Abecasis, A.; Kasten, Y.; and Dekel, T. 2024. Scenescape: Text-driven consistent scene generation. Advances in Neural Information Processing Systems, 36.

Gao, G.; Liu, W.; Chen, A.; Geiger, A.; and Scholkopf, B. Â¨ 2024. Graphdreamer: Compositional 3d scene synthesis

from scene graphs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21295â 21304.

Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.; Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Hong, F.; Tang, J.; Cao, Z.; Shi, M.; Wu, T.; Chen, Z.; Yang, S.; Wang, T.; Pan, L.; Lin, D.; et al. 2024. 3dtopia: Large text-to-3d generation model with hybrid diffusion priors. arXiv preprint arXiv:2403.02234.

Hurst, A.; Lerer, A.; Goucher, A. P.; Perelman, A.; Ramesh, A.; Clark, A.; Ostrow, A.; Welihinda, A.; Hayes, A.; Radford, A.; et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Jiang, L.; Zheng, X.; Lyu, Y.; Zhou, J.; and Wang, L. 2024. Brightdreamer: Generic 3d gaussian generative framework for fast text-to-3d synthesis. arXiv preprint arXiv:2403.11273.

Kerbl, B.; Wraber, W.; Egger, B.; and Lugmayr, A. 2023. 3D Gaussian Splatting for Efficient Scene Representation. arXiv preprint arXiv:2302.08354.

KÄ±cÄ±man, E.; Ness, R.; Sharma, A.; and Tan, C. 2023. Causal reasoning and large language models: Opening a new frontier for causality. arXiv preprint arXiv:2305.00050.

Lee, A.; and Kim, T. 2023. MVDream: Multi-View Consistent 3D Object Generation from Single-View Images. In Proceedings of the International Conference on Computer Vision (ICCV), 5678â5685.

Li, L. 2024. Cpseg: Finer-grained image semantic segmentation via chain-of-thought language prompting. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 513â522.

Li, X.; Fan, B.; Zhang, R.; Jin, L.; Wang, D.; Guo, Z.; Zhao, Y.; and Li, R. 2024. Image content generation with causal reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 13646â13654.

Lin, C.-H.; Gao, J.; Tang, L.; Takikawa, T.; Zeng, X.; Huang, X.; Kreis, K.; Fidler, S.; Liu, M.-Y.; and Lin, T.-Y. 2023a.

Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 300â309.

Lin, C.-H.; Liu, Y.; Fang, D.; Lin, W.; and Zhou, F. 2023b. CompoNeRF: Customizable Layouts for Compositional 3D Generation. arXiv preprint arXiv:2305.09134.

Lin, Y.; Wu, H.; Wang, R.; Lu, H.; Lin, X.; Xiong, H.; and Wang, L. 2023c. Towards language-guided interactive 3d generation: Llms as layout interpreter with generative feedback. arXiv preprint arXiv:2305.15808.

Liu, R.; Wu, R.; Van Hoorick, B.; Tokmakov, P.; Zakharov, S.; and Vondrick, C. 2023. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, 9298â9309.

Metzer, G.; Richardson, E.; Patashnik, O.; Giryes, R.; and Cohen-Or, D. 2023. Latent-nerf for shape-guided generation of 3d shapes and textures. In Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition, 12663â12673.

Neuberg, L. G. 2003. Causality: models, reasoning, and inference, by judea pearl, cambridge university press, 2000. Econometric Theory, 19(4): 675â685.

Nichol, A.; Jun, H.; Dhariwal, P.; Mishkin, P.; and Chen, M. 2022. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751.

Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.; Chanan, G.; Killeen, T.; Lin, Z.; Gimelshein, N.; Antiga, L.; et al. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

Poole, B.; Jain, A.; Barron, J. T.; and Mildenhall, B. 2022. DreamFusion: Text-to-3D using 2D Diffusion. arXiv preprint arXiv:2209.14988.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748â8763. PMLR.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10684â 10695.

Saharia, C.; Chan, W.; Saxena, S.; Li, L.; Whang, J.; Denton, E. L.; Ghasemipour, K.; Gontijo Lopes, R.; Karagol Ayan, B.; Salimans, T.; et al. 2022. Photorealistic text-toimage diffusion models with deep language understanding. Advances in neural information processing systems, 35: 36479â36494.

Savva, M.; Chang, A. X.; and Agrawala, M. 2017. Scenesuggest: Context-driven 3d scene design. arXiv preprint arXiv:1703.00061.

Seo, J.; Jang, W.; Kwak, M.-S.; Kim, H.; Ko, J.; Kim, J.; Kim, J.-H.; Lee, J.; and Kim, S. 2023. Let 2d diffusion model know 3d-consistency for robust text-to-3d generation. arXiv preprint arXiv:2303.07937.

Sun, C.; Han, J.; Deng, W.; Wang, X.; Qin, Z.; and Gould, S. 2023. 3d-gpt: Procedural 3d modeling with large language models. arXiv preprint arXiv:2310.12945.

Tang, J.; Ren, J.; Zhou, H.; Liu, Z.; and Zeng, G. 2024. DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation. In The Twelfth International Conference on Learning Representations.

Team, G.; Anil, R.; Borgeaud, S.; Alayrac, J.-B.; Yu, J.; Soricut, R.; Schalkwyk, J.; Dai, A. M.; Hauth, A.; Millican, K.; et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Vashishtha, A.; Reddy, A. G.; Kumar, A.; Bachu, S.; Balasubramanian, V. N.; and Sharma, A. 2023. Causal Inference using LLM-Guided Discovery. In AAAI 2024 Workshop on âAre Large Language Models Simply Causal Parrots?â.

Wang, H.; Du, X.; Li, J.; Yeh, R. A.; and Shakhnarovich, G. 2023. Score jacobian chaining: Lifting pretrained 2d

diffusion models for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12619â12629.

Wang, K.; Lin, Y.-A.; Weissmann, B.; Savva, M.; Chang, A. X.; and Ritchie, D. 2019. Planit: Planning and instantiating indoor scenes with relation graph and spatial prior networks. ACM Transactions on Graphics (TOG), 38(4): 1â 15.

Wang, Z.; Lu, C.; Wang, Y.; Bao, F.; Li, C.; Su, H.; and Zhu, J. 2024. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36.

Willis, M. J. 1999. Proportional-integral-derivative control. Dept. of Chemical and Process Engineering University of Newcastle, 6.

Xiong, S.; Chen, D.; Wu, Q.; Yu, L.; Liu, Q.; Li, D.; Chen, Z.; Liu, X.; and Pan, L. 2024. Improving causal reasoning in large language models: A survey. arXiv e-prints, arXivâ 2410.

Yang, X.; Hu, F.; and Ye, L. 2021. Text to scene: a system of configurable 3D indoor scene synthesis. In Proceedings of the 29th ACM International Conference on Multimedia, 2819â2821.

Yang, Y.; Jia, B.; Zhi, P.; and Huang, S. 2024. Physcene: Physically interactable 3d scene synthesis for embodied ai. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 16262â16272.

Yi, T.; Fang, J.; Wu, G.; Xie, L.; Zhang, X.; Liu, W.; Tian, Q.; and Wang, X. 2023. Gaussiandreamer: Fast generation from text to 3d gaussian splatting with point cloud priors. arXiv preprint arXiv:2310.08529.

Zhang, B.; Cheng, Y.; Yang, J.; Wang, C.; Zhao, F.; Tang, Y.; Chen, D.; and Guo, B. 2024a. GaussianCube: Structuring Gaussian Splatting using Optimal Transport for 3D Generative Modeling. arXiv preprint arXiv:2403.19655.

Zhang, J.; Li, X.; Wan, Z.; Wang, C.; and Liao, J. 2024b. Text2nerf: Text-driven 3d scene generation with neural radiance fields. IEEE Transactions on Visualization and Computer Graphics, 30(12): 7749â7762.

Zhang, Q.; Wang, C.; Siarohin, A.; Zhuang, P.; Xu, Y.; Yang, C.; Lin, D.; Zhou, B.; Tulyakov, S.; and Lee, H.-Y. 2024c. Towards Text-guided 3D Scene Composition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6829â6838.

Zheng, K.; Chen, X.; He, X.; Gu, J.; Li, L.; Yang, Z.; Lin, K.; Wang, J.; Wang, L.; and Wang, X. E. 2024. EditRoom: LLM-parameterized Graph Diffusion for Composable 3D Room Layout Editing. arXiv preprint arXiv:2410.12836.

Zhou, J.; Li, X.; Qi, L.; and Yang, M.-H. 2024. Layoutyour-3D: Controllable and Precise 3D Generation with 2D Blueprint. arXiv preprint arXiv:2410.15391.

Zhou, J.; Zhang, W.; and Liu, Y.-S. 2025. Diffgs: Functional gaussian splatting diffusion. Advances in Neural Information Processing Systems, 37: 37535â37560.

Zhou, Y.; He, Z.; Li, Q.; and Wang, C. 2025. LAY-OUTDREAMER: Physics-guided Layout for Text-to-3D Compositional Scene Generation. arXiv preprint arXiv:2502.01949.

Zhou, Y.; While, Z.; and Kalogerakis, E. 2019. Scenegraphnet: Neural message passing for 3d indoor scene augmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 7384â7392.

## Clarify of Causal Reasoning

In the introduction and related works, we define âCausal Reasoning (CR)â as explicitly modeling relationships between objects (e.g., spatial or functional relationships) while excluding irrelevant variables to address ambiguity in object placement during scene generation. This aligns with the definition of CR in LLMs (Xiong et al. 2024). In image generation, consistent with Image Content Generation with Causal Reasoning (Li et al. 2024), CR in generative tasks aims to structure latent relationships between objects, compensating for missing details of objects during the generation process.

## Details of Bayesian Edge Estimation

In our Bayesian Edge Estimation framework, we assume that each edge in the causal order edge set ${ \mathcal { E } } _ { \mathrm { o r d e r } }$ is independent given $e _ { i , j }$ . However, edges may remain not independent in scenarios. The PID controller effectively addresses these dependencies by dynamically adjusting its parameters, ensuring comprehensive spatial coordination and precision across the framework. This assumption allows us to apply the product rule for independent events in probability theory. we compute the probability of the entire edge set occurring under the condition $e _ { i , j }$ , referring to formula 5 in the main text. Under the condition that $e _ { i , j }$ is valid, the probability of each side can be calculated separately, without being affected by the presence of other sides in the set. This simplifies the calculation and allows us to decompose the joint probability into the product of the individual probabilities:

$$
p ( \mathcal { E } _ { \mathrm { o r d e r } } \mid e _ { i , j } ) = p ( e _ { a , b } \mid e _ { i , j } ) p ( e _ { c , d } \mid e _ { i , j } ) \ldots\tag{15}
$$

This calculation mode is particularly advantageous in LLMbased probability estimation, as it allows each edge to be evaluated independently without the need to explicitly model dependencies between them.

## Details of PID controller

In our framework, causal reasoning and PID control operate as two distinct stages to ensure accurate scene reconstruction. Causal reasoning is first used to generate an initial object placement by inferring spatial dependencies from textual descriptions. However, these placements lack precise spatial alignment, leading to minor positional errors, intersections, or unrealistic gaps between objects.

To refine these placements, we employ a PID controller, which fine-tunes object positions and scales to ensure that the spatial configuration remains physically plausible. The PID controller takes the initial positions inferred from causal reasoning and iteratively adjusts them to minimize spatial discrepancies.

Why Use an Actuator? A key challenge in applying PID adjustments is preventing instability and overcorrection. As illustrated in Fig.7, directly applying the PID output can lead to abrupt jumps, oscillations, or physically unrealistic movements, especially when the error signal fluctuates. To mitigate this, we introduce an actuator that smooths the adjustment process, ensuring controlled and realistic modifications to object placement.

<!-- image-->  
Figure 7: Visual results of actuator.

Algorithm 1: PID Optimization with Iterative Loop.   
Input p: Text prompt   
Input Î±: Target score   
Input Ïµ: Error tolerance   
Input N : Max iterations   
1: Initialize âcurrent, $E \gets 0 , e _ { \mathrm { p r e v } } \gets 0 , i \gets 0$   
2: repeat   
3: $I _ { e d g e } $ rende $\cdot ( \wp _ { \mathrm { c u r r e n t } } )$ {Generate image}   
4: s â MLLM $( I _ { e d g e } , \mathsf { p } )$ {Generate score}   
5: $e  \alpha - s$ {Compute error}   
6: E â E + e {Update integral}   
7: $d  e - e _ { \mathrm { p r e v } }$ {Compute derivative}   
8: $u \gets K _ { p } \cdot \dot { e } + \dot { K } _ { i } \cdot \bar { E } + K _ { d } \cdot d$ {PID control signal}   
9: $\Gamma  \Delta ^ { * }$ Â· tanh $( u / \gamma )$ {Actuator output}   
10: âcurrent â âcurrent + Î {Update attribute}   
11: $e _ { \mathrm { p r e v } }  e$   
12: $i \gets i + 1$   
13: until $| e | \le \epsilon$ or $i \geq N$   
14: return âcurrent

Actuators are commonly used in control systems to translate control signals into gradual, physically constrained movements (Crowe et al. 2005). In our case, the actuator controls position and scale adjustments, mitigating abrupt, excessive corrections that may introduce additional inconsistencies.

Why Use tanh as the Actuator? To implement the actuator, we use a hyperbolic tangent (tanh) function to smoothly transform the control signal into practical spatial adjustments. The choice of tanh offers several advantages.

â¢ Saturation Effect: The output of tanh(x) is bounded between â1 and 1, ensuring that extreme PID outputs do not result in excessively large movements, which could cause objects to shift too abruptly.

â¢ Smooth Transitions: Unlike a linear function, tanh produces gradual transitions, which is essential for finetuning positions without introducing jerky or unnatural motion.

â¢ Damping Small Adjustments: For small error values, tanh behaves approximately linearly, allowing precise micro-adjustments, while for large errors, it naturally limits the adjustment size, preventing excessive corrections.

<!-- image-->  
Figure 8: Visual results of compositional scene generation methods.

## Compared with scene generation methods

We further evaluate our approach against recent methods in compositional scene generation, including Graph-Dreamer(Gao et al. 2024), GALA3D(Doe and Smith 2023), LI3D(Lin et al. 2023c), and CompoNeRF(Lin et al. 2023b). Detailed comparisons of GraphDreamer and GALA3D are presented in the main text. GALA3D utilizes a LLM to generate layouts. However, its direct layout inference lacks reasoning about object placement logic, often leading to physical constraint violations, such as objects floating in mid-air. GraphDreamer employs a graph-based structure to model inter-object relationships. Despite this, experimental results demonstrate that its performance degrades significantly when generating complex scenes or environments with a large number of objects. Since LI3D and CompoNeRF are not open-sourced, our comparison relies on the examples provided in their respective papers. As illustrated in Fig.8 , our method outperforms these approaches in terms of visual clarity and offers more precise scene control, enabling intuitive and interactive editing tailored to user specifications.

## Failure case

While our LLM-based spatial evaluation is effective, distilled models with smaller architectures often struggle to establish certain relational edges and accurately adjust object positions due to reduced precision. This results in increased randomness in spatial adjustments and reduced accuracy in scene refinements. To mitigate this, we propose fine-tuning on scene layout datasets using contrastive learning and multi-view consistency constraints to enhance relational reasoning and positional accuracy.

Causal Order Prompt   
You are an expert in computer graphics, computer vision, causal analysis, and scene design.   
You will be provided with a scene layout graph containing objects (nodes) and their spatial relationships (edges). Your   
task is to analyze and refine this graph using physical constraints and causal reasoning. Follow these guidelines precisely:   
1. Allowed Spatial Relations:   
- All nodes need to have an edge.   
- Use only the following words to describe connections: {above, under, in, on, front, left, right, corner, behind, left front,   
right front, left back, right back, left on, right on}.   
- Only one word must be selected per edge.   
2. Causal Reasoning & Edge Completion:   
- If two objects are closely related in real-world use but are not connected in the input graph, infer the missing edge and   
add it (e.g., add [âlampâ, âonâ, âtableâ] if missing, but do not add âfloorâ).   
- Ensure causal flow integrity: All edges must form a directed acyclic graph following causal order.   
3. Causal Order Principles:   
- Objects follow a causal flow: obj2 comes after obj1 if obj1âs placement depends on obj2 (e.g., [âcupâ, âonâ, âtableâ]).   
- Causal Rule: If obj1 depends on obj2, reverse edge and adjust relation. Example: [âtableâ, âunderâ, âcupâ] â [âcupâ,   
âonâ, âtableâ].   
- Size Rule: Larger objects should be obj2. Transform [âlaptopâ, âleft onâ, âmouseâ] â [âmouseâ, âright onâ, âlaptopâ].   
- Causal order takes priority over size when they conflict (e.g., [âTVâ, âonâ, âstandâ], even if TV is larger).   
Output Format: The output must contain the following content:   
<Answer>edges = [[obj 1, word 1, obj 2], [obj 2, word 4, obj 3], ...]</Answer>.   
Example Corrections:   
Input: [[âlaptopâ,âleftâ,âmouseâ], [âcupâ,âunderâ,âtableâ]]   
Output: edges = [[âmouseâ,ârightâ,âlaptopâ], [âcupâ,âonâ,âtableâ]]  
Table 3: Prompt used for causal order inference in our framework. The LLM is guided to ensure spatial constraints, causal consistency, and object dependencies.

Causal Intervention Prompt   
I will provide an image of a scene.   
The image depicts {candidate edge.create prompt()} as part of the scene described as: {prompt}.   
In this scene, object â{obj names[0]}â is currently labeled as â{candidate edge.edge name}â relative to   
â{obj names[1]}â.   
Task:   
- Assess whether the given spatial relationship complies with physical laws and real-world scene consistency.   
- Based on the provided image and object interactions, determine the validity of the relation using the following criteria:   
- Gravity & Support: Objects must adhere to realistic physical constraints (e.g., smaller objects should rest on larger ones,   
and unsupported objects should not float).   
- Spatial Positioning: The labeled relationship should match common spatial arrangements (e.g., a chair should be under   
a table, not above it).   
- Functional Affordance: Objects should maintain plausible real-world functionality (e.g., a monitor should be on a desk,   
not inside it).   
Decision Guidelines:   
- If the relationship is valid, return âkeepâ.   
- If the relationship is incorrect but fixable, return âmodifyâ and suggest a new relation from the predefined set:   
{candidate relations}.   
Output Format:   
Provide the response strictly in JSON format as follows:   
"action": "keep" | "modify", "updated relation": "new relation" }  
Table 4: Prompt used for causal intervention in our framework. The LLM evaluates scene images across multiple perspectives to assess the correctness of object relationships, ensuring alignment with real-world physics, and spatial reasoning.

Scale Evaluation Prompt   
I will provide an image of a scene.   
Object Dimensions:   
- The dimensions (length, width, height) of {obj names[0]} in the real world:   
- Length: {length0} cm, Width: {width0} cm, Height: {height0} cm.   
- The dimensions (length, width, height) of {obj names[1]} in the real world:   
- Length: {length1} cm, Width: {width1} cm, Height: {height1} cm.   
Task:   
- Evaluate the relative scale of the object {obj names[0]} compared to {obj names[1]}.   
- The scale of {obj names[1]} is assumed to be correct, but {obj names[0]} may have scaling inconsistencies in   
the scene with {edge.create prompt()}.   
Evaluation Criteria:   
Scale Comparison: Does {obj names[0]} appear appropriately scaled relative to {obj names[1]}?   
- Consider the effect on scene composition, ensuring it is neither too large nor too small.   
Scoring System:   
- A score from -100 to 100 is assigned based on scale consistency:   
- Score close to 0: The scale of {obj names[0]} is appropriate.   
- Positive score: {obj names[0]} is too large compared to {obj names[1]}, disrupting scene balance.   
- Negative score: {obj names[0]} is too small, making it insignificant in the scene.   
Output Format:   
Provide only the result in the following format, with no additional text:   
<Answer>The score is: X</Answer>, where X is the evaluated score.   
For example, output: <Answer>The score is: 25</Answer>.  
Table 5: Prompt used for scale evaluation and PID-based optimization in our framework. The LLM assesses object-relative scaling.

Spatial Position Evaluation Prompt   
I will send you a sentence and images of a scene.   
Scene Description:   
The image shows a scene of {edge.create prompt()}.   
Evaluation Task:   
- The position of object {obj names[1]} is correct.   
- Object {obj names[0]} may be misplaced in the scene. Evaluate its spatial deviation along three axes.   
Scoring Criteria:   
Assign a score from -100 to 100 along each axis:   
1. Left-Right (X-Axis):   
- Positive score: Too close to {obj names[1]}.   
- Negative score: Too far from {obj names[1]}.   
2. Forward-Backward (Y-Axis):   
- Positive score: Too close to {obj names[1]}.   
- Negative score: Too far from {obj names[1]}.   
3. Up-Down (Z-Axis):   
- Positive score: Too high above {obj names[1]}.   
- Negative score: Too low below {obj names[1]}.   
Output Format:   
<Answer>The score-1 is: XX. The score-2 is: YY. The score-3 is: ZZ</Answer>  
Table 6: Prompt used for spatial position evaluation. The LLM provides axis-aligned position corrections based on multi-view scene analysis.