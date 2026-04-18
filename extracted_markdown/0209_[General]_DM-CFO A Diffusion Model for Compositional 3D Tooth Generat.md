# DM-CFO: A Diffusion Model for Compositional 3D Tooth Generation with Collision-Free Optimization

Yan Tian, Pengcheng Xue, Weiping Ding, Senior Member, IEEE, Mahmoud Hassaballah, Member, IEEE, Karen Egiazarian, Fellow, IEEE, Aura Conci, Abdulkadir Sengur, Leszek Rutkowski, Life Fellow, IEEE

AbstractâThe automatic design of a 3D tooth model plays a crucial role in dental digitization. However, current approaches face challenges in compositional 3D tooth generation because both the layouts and shapes of missing teeth need to be optimized. In addition, collision conflicts are often omitted in 3D Gaussianâbased compositional 3D generation, where objects may intersect with each other due to the absence of explicit geometric information on the object surfaces. Motivated by graph generation through diffusion models and collision detection using 3D Gaussians, we propose an approach named DM-CFO for compositional tooth generation, where the layout of missing teeth is progressively restored during the denoising phase under both text and graph constraints. Then, the Gaussian parameters of each layout-guided tooth and the entire jaw are alternately updated using score distillation sampling (SDS). Furthermore, a regularization term based on the distances between the 3D Gaussians of neighboring teeth and the anchor tooth is introduced to penalize tooth intersections. Experimental results on three tooth-design datasets demonstrate that our approach significantly improves the multiview consistency and realism of the generated teeth compared with existing methods. Project page: https://amateurc.github.io/CF-3DTeeth/

Index TermsâDiffusion Model, 3D Gaussian Splatting, 3D Editing, Dental Model Design, Dental Digitization.

## I. INTRODUCTION

Manuscript received September 9, 2025; revised December 22, 2025. This work was supported by the Zhejiang Province Natural Science Foundation (No. LZ24F020001), AGH University of Krakow âExcellence initiative - research universityâ, Polish Ministry of Science and Higher Education funds (No. UMO-2021/01/2/ST6/00004 and No. ARTIQ/0004/2021), the Opening Foundation of the Tongxiang Institute of General Artificial Intelligence (No. TAGI2-B-2024-0009), and the State Key Laboratory of Advanced Medical Materials and Devices (No. SQ2022SKL01089-2025-14).

Karen Egiazarian is with the Department of Computing Sciences, Tampere University, Tampere 33720, Finland.

Aura Conci is with the Department of Computer Science, Universidade Federal Fluminense, Niteroi 24210-346, Brazil.

Abdulkadir Sengur is with the Department of Electrical and Electronic Engineering, Faculty of Technology, Firat University, 23000 Elazig, Turkey.

Leszek Rutkowski is with the Systems Research Institute of the Polish Academy of Sciences, 01-447 Warsaw, Poland, with AGH University of Krakow, 30-059 Krakow, and with the SAN University, 90-113 Lodz, Poland.

Corresponding author: Yan Tian and Weiping Ding, email: tianyan@zjgsu.edu.cn, dwp9988@163.com.

T OOTH model design, a specialized area within themedical applications of computer graphics, involves the medical applications of computer graphics, involves the creation of accurate and detailed representations of teeth for various dental applications, including prosthetics, orthodontics, and restorative dentistry [1], [2]. This design process is crucial to ensure that dental restorations are properly fitted and function effectively. The field has been notably advanced through the integration of computer-aided design (CAD) technology, as well as recent developments in deep learning techniques.

Although contemporary text-to-3D models [3] and imageto-3D models [4] demonstrate the capability to generate individual 3D teeth, they face significant challenges in the compositional generation of multiple 3D teeth [5], [6], where multiple missing 3D teeth in a jaw are simulated using contextual information, such as geometric knowledge of neighboring teeth. This difficulty arises from the need to optimize both layouts and shapes of missing teeth. Large language models (LLMs), such as GPT-3.5, have been utilized to explicitly generate scene graphs [7], [8], which facilitate the supervision of shape generation for each individual instance. An example is GALA3D [7], illustrated in Fig. 1(a). However, this exploration is restricted to pairwise relationships, which inform the 3D synthesis but result in geometric inconsistencies among objects due to the lack of higher-order information. Additionally, collision conflicts are frequently overlooked in 3D Gaussianâbased compositional generation, as objects may intersect due to the absence of explicit geometric information regarding their surfaces. DreamScape [9] introduces a collision-loss mechanism that calculates the aggregate distance between closely located points of two objects, using a predetermined threshold. Nevertheless, this approach is insufficient for instances characterized by varying scales.

In recent years, denoising diffusion models [10] have exhibited exceptional generative capabilities in the domain of graph generation [11]. These models offer several advantages, including stable training processes and the ability to generalize across various graph structures. Consequently, graph diffusionâbased models [11] can be utilized effectively in the design of compositional tooth models by optimizing the arrangement of missing teeth within the jaw. Moreover, since a tooth can be approximated as a cylindrical shape, the spatial relationships between 3D Gaussians corresponding to different teeth can be leveraged to impose penalties for collision conflicts.

In this study, we propose an approach named DM-CFO, where the jaw configuration, including absent teeth, is generated through a graph diffusion model, in which a target graph is incrementally restored during the denoising process by integrating both textual and graphical constraints, as illustrated in Fig. 1(b). Following this, we alternately update the Gaussian parameters of each layout-guided tooth, as well as those of the entire jaw, using score distillation sampling (SDS) [3]. A regularization term, based on the distances between the 3D Gaussian representations of neighboring teeth and the anchor tooth, is introduced to mitigate tooth intersections.

<!-- image-->  
Fig. 1. Illustration of approaches using the pairwise relations and the proposed approach. (a) GALA3D uses only the text description to optimize the layout (teeth positions), receiving limited results. (b) Our DM-CFO constructs a graph to represent the jaw with multiple missing teeth, then the target graph is incrementally restored during the denoising process through a graph diffusion model.

The principal contributions of this paper are summarized as follows.

â¢ A novel framework utilizing 3D Gaussian splatting is proposed for the automatic generation of multiple missing teeth. This framework operates by alternately updating the Gaussian parameters associated with both the overall scene and the individual instances.

â¢ The configuration of teeth within a jaw, which may include the occurrence of several missing teeth, is produced utilizing a graph diffusion model. In this model, a target graph is systematically reconstructed during the denoising phase employing both textual and graphical constraints.

â¢ A collision loss based on 3D Gaussians is proposed to penalize intersections between teeth, thereby improving the geometric quality of the designed tooth model.

The experimental results obtained from Shining3D [12], Aoralscan3 [13], and DeepBlue [14] tooth design datasets demonstrate that the methodology presented in this article is competitive with current state-of-the-art (SOTA) techniques in the design of compositional tooth models.

The remainder of this paper is organized as follows: Section II reviews studies on 3D generation and editing, Section III introduces the proposed approach, Section IV presents the experimental results, and Section V concludes the article.

## II. RELATED WORK

In this section, we provide a succinct review of the literature on 3D generation and editing, with particular emphasis on diffusion models and 3D Gaussian splatting representations.

## A. 3D Tooth Generation

The field of 3D tooth generation is driven by the integration of deep learning techniques that automate and personalize restorative design. Traditional methods employ the encoderdecoder architecture to reconstruct an integrated point cloud when a partial point cloud is provided. For example, VF-Net [15] is a fully probabilistic point cloud model closely resembling variational autoencoders to replace the Chamfer distance and enable working with probability densities. Recently, research efforts are increasingly focused on transformer-based architectures and multimodal data fusion. For example, TranS-DFNet [16] introduces a voxel-based truncated signed distance field (TSDF) to improve smooth reconstruction. The point-tomesh completion network [17] generates watertight meshes directly from partial scans, leveraging implicit neural representations for improved marginal fit. SSEN [18] learns directly from unlabeled dental data by identifying multimodal features and topological relationships. VBCD [19] employs a coarseto-fine architecture and loss of curvature and margin line to reconstruct dental crowns.

Generative adversarial networks (GANs) [20] are another important architecture for 3D tooth generation, adding a discriminator to evaluate the effectiveness of distribution simulation. A two-stage GAN framework [21] divides the generation task into segmentation and depth estimation stages to improve morphological consistency, while a three-stage architecture [22] adds an additional image inpainting stage to handle challenging cases. MVDC [23] synthesizes occlusal, buccal, and lingual depth maps to reconstruct crown geometries with high fidelity, emphasizing holistic shape inference.

Diffusion models have recently been used for the synthesis of photorealistic textures, which improve aesthetic outcomes by generating realistic enamel surfaces [24], [25]. However, multiple-tooth generation is still underexplored, since both tooth layout and tooth surface information are required in the inference stage, which increases the complexity of the generation task.

## B. 3D Gaussian Splatting-based Generation and Editing

3D Gaussian Splatting (3DGS) utilizes millions of Gaussian ellipsoidal point clouds to accurately represent objects or scenes and facilitate view rendering through rasterization. To address computational initialization, LGM [26] proposes a regression network that directly initializes Gaussian parameters from multiview images end-to-end. Moreover, 3DGS is limited in surface modeling due to the intrinsic properties of its representations, particularly in scenarios involving multiple instances within the target scene. To mitigate this issue, a mesh is extracted from 3DGS using a local density query [27].

In the context of image editing, the rendered image and textual embeddings are generally aligned to accurately identify the target within the rendered image [28]. Certain methodologies [29] attribute inaccuracies in controlling the specified appearance and location to the inherent limitations of text descriptions. These approaches accommodate both text and image prompts simultaneously to delineate the editing region. Furthermore, Gaussian semantic features [30], [31] are utilized to identify or track the edited region from multiple viewpoints.

Collision conflicts are frequently overlooked in 3D Gaussianâbased compositional tooth generation, as teeth may intersect due to the absence of explicit geometric information regarding their surfaces. DreamScape [9] introduces a collisionloss mechanism that calculates the aggregate distance between closely situated points of two objects using a predetermined threshold. Nevertheless, this approach is insufficient for teeth characterized by varying scales.

## C. Diffusion Model-based Generation and Editing

Significant advances have been made in 3D generation and editing, largely attributable to the rapid development of diffusion models [32]. Direct 3D editing utilizes semantic 3D representations, such as neural shape representations, to manipulate the appearance, shape, or existence of target objects. The optimization of both the entire scene and its zoomed-in sections is performed jointly to address the multi-face problem and improve detail [33].

In scenarios where limited 3D scene data are available, various strategies [3] have been developed to improve 3D representations by leveraging prior information extracted from 2D diffusion models. To address the issue of multiview inconsistency, multiview diffusion models [34]â[36] are employed to generate coherent images from novel target viewpoints.

To generate objects characterized by intricate textures and complex geometries, it is essential to integrate global features with local features [37]. This integration provides precise guidance for the SDS, which aligns effectively with the input data. However, extending this approach to the generation of multiple missing teeth presents challenges, as the local patterns of distinct teeth are independent of each other, and neighboring teeth may encounter collision conflicts due to the lack of contextual information from their surroundings.

## D. Compositional 3D Generation

Compositional 3D generation, which entails the synthesis of scenes comprising multiple instances, poses considerable challenges within 3D contexts. Certain 3D reconstruction methodologies, such as Comboverse [5] and REPARO [38], independently segment, complete, and generate multiple instances, subsequently optimizing their spatial relationships by aligning them with reference images. An alternative approach [4] emphasizes compositional text-to-3D generation, in which the reference image is generated by a text-to-image model. Although Deep Prior Assembly (DPA) [39] and the Divide-and-Conquer (DAC) strategy [40] utilize depth priors to aid in scene assembly, they face challenges in compositional 3D generation [5], [6], as both the layout of the scene and the shapes of the instances require optimization.

Layout-your-3D [41] adopts 2D layouts to enhance spaceaware SDS, ensuring precise control over the generation process. Nevertheless, reliance on 2D guidance often presents challenges in accurately composing multiple objects with diverse attributes and interrelationships into a cohesive scene. To address the differentiation of various attributes within implicit 2D diffusion priors, GALA3D [7] generates a coarse 3D layout prior and subsequently learns a layout-guided Gaussian representation. Within this framework, MVDream [42] functions as the instance-level diffusion prior, while ControlNet [43] operates as the scene-level diffusion prior. SceneWiz3D [44] employs particle swarm optimization (PSO) to refine the layout of the scene. In addition, large language models, such as GPT-3.5, are employed to explicitly generate scene graphs [7], [8], [45] that supervise the shape generation of each instance. However, the current approach relies solely on adjacency relations to guide 3D synthesis, which results in geometric inconsistencies among instances due to the lack of higherorder information.

To enhance the understanding of relationships among instances, MIDI [46] introduces a multi-instance attention mechanism that effectively captures complex inter-object interactions. Simultaneously, ComboVerse [5] modifies the attention map of position tokens, which represent spatial relationships for score distillation. However, the issue of collision conflicts remains unaddressed in 3D Gaussianâbased compositional generation, where objects may intersect due to the absence of explicit geometric information regarding their surfaces. PhyCAGE [47] introduces an innovative refinement of SDS by integrating physics-based simulation. Rather than relying solely on traditional gradient updates, the method repurposes the SDS loss gradient as the initial velocity in a dynamic physical simulation. DreamScape [9] proposes a collisionloss mechanism that calculates the sum of distances between points that are in close proximity across two objects, based on a predetermined threshold. However, this approach proves inadequate for instances with varying scales, particularly in scenarios that require stringent collision rejection.

## III. THE PROPOSED APPROACH

3D Gaussian splatting, diffusion models, and score distillation sampling collectively establish a foundational framework for 3D editing. In the context of a jaw model that exhibits multiple missing teeth, we construct and optimize the arrangement of all teeth using a graph diffusion model. We propose a duallevel optimization approach aimed at achieving instance-level realism and global consistency, thereby enhancing the fidelity of the synthesized teeth while mitigating collision conflicts. The details of our DM-CFO are illustrated in Fig. 2.

## A. Preliminaries

3D Gaussian Splatting. 3DGS [48] represents an explicit radiance field that employs anisotropic 3D Gaussians for scene representation, thereby enabling high-quality, real-time, and high-resolution rendering. To enhance rendering efficiency, 3DGS incorporates a tile-based rasterizer that segments the image into tiles, which are subsequently filtered and sorted according to the projected 3D Gaussians. The color C of each pixel within the patch is defined as follows:

<!-- image-->  
Fig. 2. An illustration of the proposed DM-CFO. Given 3D Gaussian representing jaw with missing teeth, the graph diffusion generates layout of missing teeth by progressively denoising graph with both text and graph constraints. Then, the Gaussian parameters of each layout-guided tooth and the whole jaw are alternately updated using SDS. A regularization term based on 3D Gaussians of neighboring teeth is explored to penalize the tooth intersection.

$$
C = \sum _ { i \in \mathcal { N } } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{1}
$$

where $c _ { i }$ is derived from spherical harmonics, which characterize the color of a point among the N ordered points that intersect with the pixel, and Î± is the opacity of the point, which is scaled by a 2D Gaussian.

Diffusion Model. Diffusion models are a category of generative models that systematically denoise samples I initially drawn from a Gaussian distribution, with the ultimate goal of aligning these samples with the real data distribution p(I). These models are based on two primary processes. The forward process utilizes Gaussian noise scheduling to perturb the data, which is represented as $\mathbf { I } \sim p ( \mathbf { I } )$ . A reverse process incrementally removes this noise and reintroduces structure into an intermediate latent variable, expressed as ${ \mathbf I } _ { \eta } = \alpha _ { \eta } { \mathbf I } + \sigma _ { \eta } \epsilon$ , where $\alpha _ { \eta }$ and $\sigma _ { \eta }$ are noise schedules at timestep Î·, and Ïµ is Gaussian noise. The reverse process is typically parameterized by a conditional neural network $\epsilon _ { \phi }$ that is trained to predict the noise Ïµ using a simplified objective function.

Score Distillation Sampling. SDS utilizes a pre-trained DM to improve the optimization of a differentiable and parametric image rendering function $g ( \theta , \pi )$ , where Î¸ is a 3D representation, and Ï denotes the camera pose from which the image I is produced. Specifically, the parameters Î¸ are updated by employing the gradient:

$$
\nabla _ { \phi } L _ { S D S } ( \phi , \theta ) = \mathbb { E } _ { \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) , \eta \sim T } [ \omega ( \eta ) ( \hat { \epsilon } _ { \phi } ( \mathbf { z } _ { \eta } , \eta , \mathbf { c } ) - \epsilon ) \frac { \partial \mathbf { z } _ { \eta } } { \partial \theta } ] ,\tag{2}
$$

where $\mathbf { z } _ { \eta } = E ( g ( \theta , \pi ) )$ represents the latent representation obtained by the encoder E.

## B. Layout Editing Based on Graph Diffusion

Graph models are employed to represent spatial relationships among instances with the assistance of LLMs, they encounter difficulties in addressing complex intrinsic dependencies. Unlike image data, the nodes within these models do not adhere to the assumption of being independent and identically distributed (i.i.d.). The complexity inherent in the graph structure presents significant obstacles in the generation of the desired graphs.

Drawing upon recent advances in language-guided 3D scene synthesis, we transform scenes into semantic graphs and utilize a conditional diffusion model to learn the conditional distribution of the target scene graph, which is illustrated in Fig. 3.

<!-- image-->  
Fig. 3. An illustration of the layout editing module. Given the original jaw graph, noise is progressively added in the forward diffusion process, and then graph is progressively updated in the reverse denoising process to obtain the augmented jaw graph. The yellow arrow represents the iteration step.

In this study, a 3D jaw model characterized by multiple missing teeth is used to facilitate the segmentation of each individual tooth using a commercially available approach [49]. An original jaw graph $\mathcal { G } _ { s } ~ = ~ ( \mathbf { V } _ { s } , \mathbf { E } _ { s } )$ is constructed to represent the jaw layout $\mathbf { L } _ { s } ,$ , where each node $\mathbf { v } _ { i } ~ \in ~ \mathbf { V } _ { s }$ represents a specific tooth. Each node encompasses a discrete category $c _ { i } ,$ a tooth layout $\mathbf { L } _ { i } .$ and semantic features $\mathbf { f } _ { i }$ derived from the segmentation process. The tooth layout $\mathbf { L } _ { i } = \{ x _ { i } , y _ { i } , z _ { i } , h _ { i } , w _ { i } , l _ { i } , k _ { i } , r _ { i } \}$ , where $( x _ { i } , y _ { i } , z _ { i } )$ represent the spatial coordinates of the i-th tooth; $( h _ { i } , w _ { i } , l _ { i } )$ are the length, width, and height of the tooth boundary, $k _ { i }$ and $r _ { i }$ are the rotation angles of the dental and buccal sides. The discrete category $c _ { i }$ of missing teeth is based on established dental knowledge. In contrast, the corresponding tooth layout $\mathbf { L } _ { i }$ and semantic features $\mathbf { f } _ { i }$ are initially set to $\cdot _ { \mathcal { O } } ,$ and are progressively refined throughout the optimization process. Each edge ${ \bf e } _ { i j } \in { \bf E } _ { s }$ corresponds to a unique type of relationship, which can be specifically classified into the following categories: [âNeighborâ, âSymmetryâ, âArchâ]. Therefore, $\mathcal { G } _ { s } = ( \mathbf { C } _ { s } , \mathbf { L } _ { s } , \mathbf { F } _ { s } , \mathbf { E } _ { s } )$ , where ${ \bf C } _ { s } , { \bf L } _ { s } , { \bf F } _ { s }$ , and $\mathbf { E } _ { s }$ encapsulate the aggregated classifications, layouts, and features of all teeth, as well as the interrelationships among teeth within a given jaw.

Subsequently, the original jaw graph denoted $\mathcal { G } _ { s }$ is optimized to produce the augmented jaw graph $\begin{array} { r l } { \mathcal { G } _ { t } } & { { } = } \end{array}$ $\left( \mathbf { C } _ { t } , \mathbf { L } _ { t } , \mathbf { F } _ { t } , \mathbf { E } _ { t } \right)$ using a jaw text prompt $\mathbf { y } _ { s }$ via a discrete graph diffusion model $\epsilon _ { g } ,$ , where $\mathbf { C } _ { t } , \mathbf { L } _ { t } , \mathbf { F } _ { t } .$ , and $\mathbf { E } _ { t }$ represent the assembled classes, layouts, features of all teeth, and the interrelationships between the teeth within the jaw, respectively. The objective of this module is to examine the conditional distribution $q ( \mathcal { G } _ { t } | \mathcal { G } _ { s } , \mathbf { y } _ { s } )$ . In the diffusion phase, Gaussian noise is incrementally introduced to the augmented jaw graph $\mathcal { G } _ { t }$ to obtain $\mathcal { G } _ { t } ^ { \eta }$ at the timestep $\eta .$ In the denoising phase, the graph diffusion model $\epsilon _ { g }$ systematically reconstructs the image $\mathcal { G } _ { t } ^ { 0 }$ using the control signal $\mathcal { G } _ { s }$ and the vector $\mathbf { y } _ { s }$ . Each element of the source scene graphs is concatenated with the noisy target scene graphs to provide contextual information, and then a graph Transformer with a frozen text encoder iteratively removes noise and updates graph configurations. The graph Transformer consists of a stack of M blocks, each comprising graph attention, cross-attention, and multilayer perceptron (MLP), where cross-attention layers are utilized to integrate linguistic features. Assume that

$$
L _ { \eta - 1 } = D _ { K L } [ q ( \mathcal { G } _ { t } ^ { \eta - 1 } | \mathcal { G } _ { t } ^ { \eta } , \mathcal { G } _ { t } ^ { 0 } ) | | p _ { \epsilon _ { g } } ( \mathcal { G } _ { t } ^ { \eta - 1 } | \mathcal { G } _ { t } ^ { \eta } , \mathcal { G } _ { s } , \mathbf { y } _ { s } ) ] ,\tag{3}
$$

where $D _ { K L }$ indicates the KL divergence. Then, the variational lower bound (VLB) of the likelihood is derived as follows:

$$
L _ { g } = \mathbb { E } _ { q ( \mathcal { G } _ { t } ^ { 0 } ) } [ \sum _ { \eta = 2 } ^ { T } L _ { \eta - 1 } - \mathbb { E } _ { q ( \mathcal { G } _ { t } ^ { 1 } \vert \mathcal { G } _ { t } ^ { 0 } ) } [ \log p _ { \epsilon _ { g } } ( \mathcal { G } _ { t } ^ { 0 } \vert \mathcal { G } _ { t } ^ { 1 } , \mathcal { G } _ { s } , \mathbf { y } _ { s } ) ] ] .\tag{4}
$$

During the inference phase, the layout $\mathbf { L } _ { s }$ and the corresponding original scene graph $\mathcal { G } _ { s }$ are generated based on the results of 3D segmentation. Subsequently, conditioned on the original scene graph $\mathcal { G } _ { s }$ and the textual prompt $\mathbf { y } _ { s } ,$ the graph diffusion model $\epsilon _ { g }$ predicts the augmented scene graph $\mathcal { G } _ { t }$

The optimization details are presented in Algorithm 1. Our approach provides two notable advantages. First, the structural dependencies of complex graphs are effectively modeled and optimized through the noising and denoising phases of the diffusion model. Furthermore, despite the inherent discreteness of the graph structure, which complicates the computation of model gradients, continuous noise is integrated during the backpropagation training process for graph generation.

## C. Compositional Optimization

The generation of multiple teeth presents a significant challenge, as it necessitates the optimization of both layout and geometry. This process employs implicit diffusion priors to maintain consistency among the various teeth while simultaneously preventing collision conflicts with adjacent teeth.

Consequently, we propose a novel approach for synthesizing multiple teeth through the iterative optimization of the SDS loss, taking into account both scene and instance perspectives. This method employs multiview diffusion prior and incorporates heterogeneous constraints to enhance control, which is illustrated in Fig. 4.

Assume that the jaw geometry parameters $\mathbf { G } _ { s } = \{ \mathbf { O } _ { i } \} , i \in$ $\{ 1 , 2 , . . . , N \}$ are composed of tooth geometry parameters $\mathbf { O } _ { i }$ with the tooth index i guided by the layout, and the tooth geometry parameters $\mathbf { O } _ { i } = \{ \mathbf { L } _ { i } , \mathbf { G } _ { i } \}$ include the tooth layout $\mathbf { L } _ { i }$ and the Gaussians of the tooth $\mathbf { G } _ { i } .$ . The tooth layout $\mathbf { L } _ { i }$ was initially established based on the results of the graph diffusion.

Algorithm 1: Layout Optimization Based on Graph   
Diffusion   
Input: Original jaw graph $\mathcal { G } _ { s } = ( \mathbf { V } _ { s } , \mathbf { E } _ { s } )$ with node   
attributes $\left( \mathbf { C } _ { s } , \mathbf { L } _ { s } , \mathbf { F } _ { s } \right)$ , text prompt ${ \bf y } _ { s } ,$   
pre-trained graph diffusion model $\epsilon _ { g } ;$   
Output: Augmented jaw graph $\mathcal { G } _ { t } = ( \mathbf { C } _ { t } , \bar { \mathbf { L } } _ { t } , \mathbf { F } _ { t } , \mathbf { E } _ { t } ) ;$   
1 Initialization: Target graph $\mathcal { G } _ { t } ^ { 0 }  \mathcal { G } _ { s } ;$   
2 Forward Diffusion Process:   
3 for $\eta = 1$ to $T$ do   
4 Add Gaussian noise: $\mathcal { G } _ { t } ^ { \eta }  \alpha _ { \eta } \mathcal { G } _ { t } ^ { \eta - 1 } + \sigma _ { \eta } \epsilon ;$   
5 end   
6 Reverse Denoising Process:   
7 for $\eta = T$ to 1 do   
8 Concatenate context: $\mathcal { G } _ { t } ^ { \eta } \gets [ \mathcal { G } _ { s } ; \mathcal { G } _ { t } ^ { \eta } ] ;$   
9 Apply graph transformer: $\hat { \epsilon } _ { g } \dot { \mathbf { \eta } }  \epsilon _ { g } ( \dot { \mathcal { G } } _ { t } ^ { \eta } , \eta , \mathcal { G } _ { s } , \mathbf { y } _ { s } ) ;$   
10 Update graph: $\begin{array} { r } { \mathcal G _ { t } ^ { \eta - 1 } \gets \frac { 1 } { \alpha _ { \eta } } ( \mathcal G _ { t } ^ { \eta } - \bar { \sigma } _ { \eta } \hat { \epsilon } _ { g } ) + \sigma _ { \eta } \mathbf z ; } \end{array}$   
11 KL-divergence: $L _ { \eta } \gets \dot { D _ { K L } } ( q ( \mathcal { G } _ { t } ^ { \eta - 1 } | \mathcal { G } _ { t } ^ { \eta } , \mathcal { G } _ { t } ^ { 0 } ) | | p _ { \epsilon _ { g } } ) ;$   
12 end   
13 Post-Processing: Extract final layout $\mathbf { L } _ { t }$ from $\mathcal { G } _ { t } ^ { 0 } .$

<!-- image-->  
Fig. 4. An illustration of the compositional optimization module. Instancelevel diffusion and scene-level diffusion are jointly optimized.

The Gaussians of the tooth $\mathbf { G } _ { i }$ are made up of anisotropic Gaussian functions. These functions are characterized by several parameters, including center $\mathbf { p } _ { i } ,$ color $\mathbf { c } _ { i } ,$ opacity $\alpha _ { i } ,$ , and covariance matrix $\Sigma _ { i } .$ . The Gaussian splatting rendering [48] described in Eq. 1 is denoted by $g ( \ u )$ . In each training iteration, the tooth Gaussians $\mathbf { G } _ { i }$ are rendered to produce the tooth image $\mathbf { I } _ { i } ^ { r } = g ( \mathbf { G } _ { i } )$ , while the jaw geometry parameters $\mathbf { G } _ { s }$ are rendered to generate the jaw image $\mathbf { I } _ { s } ^ { r } = g ( \mathbf { G } _ { s } )$ . The gradient for the geometry parameters of the i-th tooth according to Eq. 2 can be expressed as follows:

$$
\nabla _ { \mathbf { O } _ { i } } L _ { S D S } ^ { i } = \mathbb { E } _ { \epsilon , \eta } [ \omega ( \eta ) ( \epsilon _ { \phi } ( \mathbf { I } _ { i } ^ { r } ; \mathbf { y } _ { i } , \pi _ { i } , \eta ) - \epsilon ) \frac { \partial \mathbf { I } _ { i } ^ { r } } { \partial \mathbf { O } _ { i } } ] ,\tag{5}
$$

where Ïµ represents the noise introduced, while Î· denotes the time step. The function $\omega ( \eta )$ serves as a weighting function. Additionally, $\epsilon _ { \phi }$ refers to the denoising function employed in the image diffusion process. The variable $\mathbf { y } _ { i }$ signifies the text prompt associated with the i-th tooth, and $\pi _ { i }$ indicates the extrinsic matrix of the camera.

Conditioned diffusion is used to improve the global scene by generating restored teeth while maintaining the original layout. In particular, ControlNet is fine-tuned to facilitate the rendering of layouts from multiple viewpoints as input, thereby producing 2D diffusion supervision that ensures consistency between layout and text. The gradient for the jaw geometry parameters can be articulated according to Eq. 2 as follows:

$$
\nabla _ { \mathbf { G } _ { s } } L _ { S D S } ^ { s } = \mathbb { E } _ { \epsilon , \eta } [ \omega ( \eta ) ( \epsilon _ { \phi } ( \mathbf { I } _ { s } ^ { r } ; \mathbf { y } _ { s } , \delta _ { s } , \eta ) - \epsilon ) \frac { \partial \mathbf { I } _ { s } ^ { r } } { \partial \mathbf { G } _ { s } } ] ,\tag{6}
$$

where Ïµ represents the noise introduced; Î· denotes the time step; $\omega ( \eta )$ serves as a weighting function; $\epsilon _ { \phi }$ refers to the denoising function utilized in the diffusion process of 3DGS; $\mathbf { y } _ { s }$ signifies the text prompt associated with the scene; and $\delta _ { s }$ represents the conditional input for the ControlNet, which is derived from rendering images based on the layouts.

However, the compositional loss is inadequate to address the significant collisions resulting from occlusion. Most existing methods rely on SDF to handle collision conflict $[ 5 0 ] ,$ and the conversion process between 3D Gaussians and SDF is intricate and not conducive to real-time applications. DreamScape [9] presents a collision loss mechanism that utilizes 3D Gaussian representations. This methodology calculates the cumulative distances between points that are in close proximity across two objects, employing a predetermined threshold. Nevertheless, this approach proves inadequate for instances that exhibit variability in scale.

To address this challenge, we propose a collision loss mechanism grounded in a 3D Gaussian representation, which serves to distinguish between improperly overlapping instances. This methodology iteratively evaluates the distances from the 3D Gaussian representations of adjacent teeth to the anchor tooth, thereby imposing penalties for tooth intersections. An illustration of the proposed collision loss is presented in Fig. 5.

<!-- image-->  
Fig. 5. An illustration of collision loss. Our approach employs an intravariance $R _ { i }$ rather than a fixed threshold to avoid collision conflict.

Assume that the Gaussians $\mathbf { P } ^ { i } \ = \ \{ \mathbf { p } _ { 1 } ^ { i } , \mathbf { p } _ { 2 } ^ { i } , . . . , \mathbf { p } _ { K _ { i } } ^ { i } \}$ for the i-th tooth are represented as a combination of Gaussian coordinates $\mathbf { p } _ { k } ^ { i }$ , where k is the Gaussian index and $K _ { i }$ is the number of Gaussians that comprise the i-th tooth. Initially, the mean coordinate of the i-th tooth is determined as $ { \mathbf { p } } _ { m } ^ { i }$ , and the intravariance $\begin{array} { r } { R _ { i } = \frac { 1 } { K _ { i } } \sum _ { k = 1 } ^ { K _ { i } } | | { \bf p } _ { k } ^ { i } - { \bf p } _ { m } ^ { i } | | _ { 2 } } \end{array}$ is calculated to evaluate the sparsity of the Gaussian distributions associated with the i-th tooth. Subsequently, the mean distances between the distributions of neighboring teeth are computed. Specifically, the points associated with the i-1-th and i+1-th teeth are utilized to compute the distances to $ { \mathbf { p } } _ { m } ^ { i }$ and are then compared to the intra-variance $R _ { i }$ to impose penalties for collisions occurring between adjacent teeth.

$$
\begin{array} { r } { L _ { c o l } ^ { i } = \displaystyle \sum _ { k = 1 } ^ { K _ { i - 1 } } \operatorname* { m a x } ( 0 , R _ { i } - | | \mathbf { p } _ { k } ^ { i - 1 } - \mathbf { p } _ { m } ^ { i } | | _ { 2 } ) + } \\ { \displaystyle \sum _ { k = 1 } ^ { K _ { i + 1 } } \operatorname* { m a x } ( 0 , R _ { i } - | | \mathbf { p } _ { k } ^ { i + 1 } - \mathbf { p } _ { m } ^ { i } | | _ { 2 } ) . } \end{array}\tag{7}
$$

When two teeth are in conflict, the distance from the points within the intersection region to the center of the affected tooth is less than the intravariance of that tooth.

Given the tooth Gaussian loss $L _ { S D S } ^ { i }$ , the jaw Gaussian loss $L _ { S D S } ^ { s }$ , the collision regulation term $L _ { c o l }$ , the total loss is summarized as

$$
L ^ { t o t a l } = \lambda _ { 1 } \sum _ { i = 1 } ^ { N } L _ { S D S } ^ { i } + \lambda _ { 2 } L _ { S D S } ^ { s } + \sum _ { i = 1 } ^ { N } L _ { c o l } ^ { i } ,\tag{8}
$$

The weights $\lambda _ { 1 }$ and $\lambda _ { 2 }$ balance the influence of various terms and are determined through a grid search methodology.

The intravariance $R _ { i }$ is not a fixed hyperparameter but a property learned and optimized along with the Gaussian parameters (position, color, opacity, covariance) for each tooth. During SDS optimization, if teeth begin to intersect, the points in the overlapping region will cause the collision loss $L _ { c o l } ^ { i }$ to increase. The gradient from this loss will push the Gaussians of both teeth apart. Consequently, the positions $\mathbf { p } _ { k } ^ { i }$ are updated to minimize $L _ { c o l } ^ { i } ,$ which inherently influences the calculated $R _ { i }$ for the next iteration. This creates a feedback loop that continuously refines the tooth shape and spacing to avoid collisions.

The optimization details are encapsulated in Algorithm 2. Our methodology presents two main advantages. Firstly, the dual-level optimization facilitates the attainment of instancelevel realism alongside global consistency, which enhances the fidelity of the synthesized dental structures. Secondly, it mitigates the issues of intersection and misalignment among the teeth, thereby addressing the potential spatial biases that may arise from the 3D Gaussian representations produced by LLMs and ensuring physical accuracy.

## IV. RESULTS

The effectiveness of the proposed methodology is evaluated and compared with its counterparts using the Shining3D [12], Aoralscan3 [13], and DeepBlue [14] tooth design datasets.

## A. Datasets and Evaluation Criteria

The Shining3D tooth design dataset [12] consists of 1,416 meshes generated from 3D scans of dental plaster models, each obtained from a randomly selected patient in a dental hospital. The dataset is divided into training, validation, and testing subsets, which contain 1,150, 133, and 133 samples, respectively. A point cloud is extracted from each mesh, and an instance segmentation method [49] is used to classify and delineate the 3D region of each tooth. Subsequently, the 3D regions of interest are cropped, including incisors, canines, and molars, while the remaining portions serve as conditions for simulation. Each scene within the dataset comprises 60 images without visible teeth and 40 images that prominently feature them. Masks representing the two-dimensional regions of the teeth of interest are generated using the approach described in [51] and are made available for training and testing purposes.

Algorithm 2: Compositional Optimization for Scene   
and Instance   
Input: Original jaw Gaussians $\mathbf { G } _ { s } = \{ \mathbf { O } _ { i } \}$ , target   
layout $\mathbf { L } _ { t } ,$ text prompts $\{ \mathbf { y } _ { s } , \mathbf { y } _ { i } \} ;$   
Output: Optimized scene Gaussians $\mathbf { G } _ { s } ^ { j i n a l }$ and   
instance Gaussians $\{ \mathbf { G } _ { i } ^ { f i n a l } \}$   
1 Initialize teeth Gaussians $\left\{ \mathbf { G } _ { i } \right\}$ using layout $\mathbf { L } _ { t } ;$   
2 while not converged do   
3 Scene-level Optimization:   
4 for each camera view $\pi \in \Pi$ do   
5 Render scene image $\mathbf { I } _ { s } ^ { r } = g ( \mathbf { G } _ { s } \cup \{ \mathbf { G } _ { i } \} , \pi ) \mathrm { , }$   
6 Compute SDS loss $L _ { S D S } ^ { s }$ via Eq.(6);   
7 Update $\mathbf { G } _ { s }$ using $\nabla _ { \mathbf { G } _ { s } } L _ { S D S } ^ { s } ;$   
8 end   
9 Instance-level Optimization:   
10 for each missing tooth $i \in L _ { t }$ do   
11 for each camera view $\pi \in \Pi$ do   
12 Render instance image $\mathbf { I } _ { i } ^ { r } = g ( \mathbf { G } _ { i } , \pi ) ;$   
13 Compute SDS loss $L _ { S D S . } ^ { i }$ via Eq.(5);   
14 Compute collision loss $L _ { c o l } ^ { i }$ via Eq.(7);   
15 Update $\mathbf { G } _ { i }$ using $\nabla _ { \mathbf { G } _ { i } } ( \lambda _ { 1 } \overset { \sim } { L } _ { S D S } ^ { i } + L _ { c o l } ^ { i } ) ;$   
16 end   
17 end   
18 Update layout $\mathbf { L } _ { t }$ based on updated Gaussians.   
19 end

The Aoralscan3 dataset [13] and the DeepBlue dataset [14] comprise 1,999 and 2,061 samples, respectively, with each sample derived from an anonymous patient. While these datasets are primarily intended for the pose estimation of teeth, the 3D regions both with and without the teeth of interest can be utilized as paired data for the design of tooth models. The construction of these datasets parallels that of the Shining3D tooth design dataset, in which each tooth is segmented using a distinct approach [52]. The Aoralscan3 dataset includes training, validation, and test sets containing 1,667, 156, and 176 samples, respectively, while the DeepBlue dataset comprises training, validation, and test sets containing 1,573, 244, and 244 samples, respectively. Data distributions across multiple datasets, including age, pathology, and types of missing teeth, are illustrated in Fig. 6.

To assess the quality of the generated teeth, we compare the generated multiview images with the corresponding rendered views from other approaches by calculating the Peak Signalto-Noise Ratio (PSNR), the Frechet Inception Distance (FID), Â´ and the average Learned Perceptual Image Patch Similarity (LPIPS). To quantitatively evaluate the results, we use the Chamfer Distance (CD) in millimeters (mm) and the F-Score to measure the similarity between the predicted teeth and the ground truth. Moreover, to evaluate the effect of collision avoidance, we adopt the penetration distance (PD) of nearby teeth in millimeters (mm) as an additional evaluation metric.

<!-- image-->  
(a) Age  
(b) Pathology  
(c) Type  
Fig. 6. Data distributions in multiple datasets of (a) age, (b) pathology, and (c) types of missing teeth.

## B. Implementation Details

A workstation featuring an Intel i9-9980X 3.0 GHz CPU, 128 GB of RAM, and four NVIDIA RTX 4090D GPUs is used for performance evaluation.

In the layout editing phase, we use a 5-layer, 8-head graph Transformer with 512 attention dimensions and a dropout rate of 0.1. We optimize the layout through 400 iterations.

In the compositional optimization phase, the instance text prompt specifies the particular category of the generated tooth, while the scene text prompt consists of a combination of instance text prompts. MVDream [42] operates as a multiview diffusion model, utilizing a guidance scale of 50. The Control-Net guidance scale is set to 100 to improve scene optimization and reduce the timestep during 3DGS. The learning rates for opacity and position are $5 \times 1 0 ^ { - 2 }$ and $1 . 6 \times 1 0 ^ { - 4 }$ , respectively. The color representation of the 3D Gaussians is achieved through spherical harmonic coefficients, with the degree fixed at 0. The initial learning rate is set to $5 \times 1 0 ^ { - 3 }$ and then attenuated to $5 \times 1 0 ^ { - 4 }$ after epoch 380. The covariance of the 3D Gaussians is decomposed into scaling and rotation for optimization, employing learning rates of $5 \times 1 0 ^ { - 3 }$ and $1 0 ^ { - 3 }$ , respectively. The coefficients $\lambda _ { 1 } = 1 0 . 0$ and $\lambda _ { 2 } = 2 . 5$ are determined using a grid-search methodology. The stopping criterion is defined such that the training loss does not vary by more than 500 over 10 consecutive epochs.

## C. Ablation Study

Extensive ablation studies were conducted to elucidate the impact of several critical components on the performance of the proposed methodology. All experiments described in this subsection were performed using the Shining3D tooth design dataset.

Tooth Number. The impact of the varying number of teeth produced by regression and by GALA3D, compared to our methodology, is illustrated in Fig. 7(a). The X-axis represents the class number, while the Y-axis represents the FID. Layout errors and collisions increase significantly when using regression-based methods for multi-tooth generation. The maximum number of simulated teeth is four, and the curvature of any additional teeth must take into account the dental arch; otherwise, the placement of the generated teeth may not satisfy the occlusal requirements.

TABLE I  
EFFECTIVENESS COMPARISON OF THE SHINING3D, AORALSCAN3, AND DEEPBLUE DATASET. âââ MEANS LOWER IS BETTER WHILE âââ MEANS UPPER IS BETTER.
<table><tr><td rowspan="2">Approach</td><td colspan="3">Shining3D</td><td colspan="3">Aoralscan3</td><td colspan="3">DeepBlue</td></tr><tr><td>FIDâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>FIDâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>FIDâ</td><td>LPIPSâ</td><td>PSNRâ</td></tr><tr><td>DGE [53]</td><td>223.15</td><td>0.70</td><td>12.38</td><td>234.61</td><td>0.75</td><td>10.42</td><td>228.73</td><td>0.72</td><td>11.11</td></tr><tr><td>VcEdit [54]</td><td>221.44</td><td>0.69</td><td>12.84</td><td>233.67</td><td>0.74</td><td>11.02</td><td>227.05</td><td>0.71</td><td>11.85</td></tr><tr><td>Gaussctrl [55]</td><td>220.90</td><td>0.69</td><td>13.09</td><td>233.22</td><td>0.74</td><td>11.15</td><td>226.10</td><td>0.70</td><td>12.20</td></tr><tr><td>CAT3D [34]</td><td>218.50</td><td>0.67</td><td>14.45</td><td>231.12</td><td>0.73</td><td>11.57</td><td>224.24</td><td>0.69</td><td>13.08</td></tr><tr><td>CompGS [4]</td><td>208.82</td><td>0.65</td><td>16.23</td><td>216.75</td><td>0.69</td><td>13.10</td><td>212.45</td><td>0.67</td><td>14.83</td></tr><tr><td>Frankenstein [6]</td><td>205.43</td><td>0.64</td><td>17.06</td><td>214.88</td><td>0.68</td><td>13.67</td><td>208.57</td><td>0.66</td><td>15.39</td></tr><tr><td>ComboVerse [5]</td><td>202.43</td><td>0.63</td><td>17.57</td><td>210.45</td><td>0.67</td><td>14.58</td><td>206.52</td><td>0.65</td><td>15.91</td></tr><tr><td>DIScene [8]</td><td>200.61</td><td>0.62</td><td>18.04</td><td>209.72</td><td>0.66</td><td>15.61</td><td>204.69</td><td>0.64</td><td>16.77</td></tr><tr><td>DreamScape [9]</td><td>198.83</td><td>0.61</td><td>18.87</td><td>208.49</td><td>0.65</td><td>16.30</td><td>203.34</td><td>0.63</td><td>17.60</td></tr><tr><td>SceneWiz3D [44]</td><td>198.59</td><td>0.61</td><td>18.90</td><td>208.40</td><td>0.65</td><td>16.35</td><td>203.40</td><td>0.63</td><td>17.68</td></tr><tr><td>GALA3D [7]</td><td>196.62</td><td>0.60</td><td>19.24</td><td>206.44</td><td>0.64</td><td>17.14</td><td>201.55</td><td>0.62</td><td>18.25</td></tr><tr><td>MIDI [46]</td><td>195.71</td><td>0.59</td><td>20.39</td><td>205.83</td><td>0.63</td><td>17.66</td><td>200.77</td><td>0.61</td><td>18.95</td></tr><tr><td>Ours</td><td>193.29</td><td>0.57</td><td>22.55</td><td>203.56</td><td>0.61</td><td>19.02</td><td>198.41</td><td>0.59</td><td>20.74</td></tr></table>

<!-- image-->  
(a) Tooth Number

<!-- image-->  
(b) Training Loss  
Fig. 7. Illustration of variance on (a) tooth number and (b) training loss.

Tooth Group. The impact of different categories of teeth on the formation of three distinct tooth types is depicted in Fig. 8, for example, a central incisor, a lateral incisor, and a canine. Each column in the figure corresponds to a particular group of dental categories. The findings indicate consistent robustness across the various categories of teeth.

<!-- image-->

<!-- image-->

<!-- image-->  
premolar, a 1 st molar, and a 2nd molarâ  
incisor, and a canineâ  
Fig. 8. Illustration of the impact of different group of tooth categories. Each column represents a specific group of tooth categories.

Training Loss. A comparison of the training loss between GALA3D and our proposed method is presented in Fig. 7(b). The X-axis denotes the epoch number, whereas the Y-axis represents the training loss. Our methodology requires a greater number of epochs to converge, attributable to fluctuations in occlusion resulting from diverse viewpoints. Nevertheless, the final loss achieved is lower than that of the alternative approach.

Effectiveness. The efficacy of each component is detailed in Table II and illustrated in Fig. 9. The baseline utilizes the GALA3D architecture, which employs the layout interpreted by the LLM with subsequent refinement. Note that no additional processing is implemented to address collision conflicts between instances. As indicated in Table II, the graph diffusion model plays a crucial role in optimizing layouts by continuously adjusting them throughout the denoising process. This methodology facilitates more intricately aligned interactions among instances while maintaining adherence to real-world constraints. Furthermore, as shown in Fig. 9, improvements in global scene optimization and geometric conflict resolution have resulted in the generation of 3D scenes that exhibit enhanced textures and scene coherence, effectively mitigating the occurrence of âover-constrainedâ boundaries.

TABLE II  
EFFECTIVENESS COMPARISON OF IMPROVED MODULES ON THE SHINING3D DATASET. GDM REPRESENTS THE GRAPH DIFFUSION MODEL, AND GCL REPRESENTS GAUSSIAN COLLISION LOSS IN COMPOSITIONAL OPTIMIZATION.
<table><tr><td>GDM</td><td>GCL</td><td>FIDâ</td><td>LPIPSâ</td><td>PSNRâ</td></tr><tr><td rowspan="3">â</td><td></td><td>196.62</td><td>0.60</td><td>19.24</td></tr><tr><td></td><td>194.25</td><td>0.58</td><td>21.85</td></tr><tr><td></td><td>194.49</td><td>0.59</td><td>20.71</td></tr><tr><td>â</td><td>&gt;&gt;</td><td>193.29</td><td>0.57</td><td>22.55</td></tr></table>

## D. Evaluation of the Tooth Design Datasets

Quantitative Comparison. A comparative analysis of various compositional 3D generation methodologies is conducted alongside our proposed method, utilizing identical parameters to ensure a fair evaluation, and the results are reported in Table I. The methodologies referenced in the literature [34], [53]â[55] generate multiple teeth by simulating a single tooth in each iteration. This approach results in limited 3D consistency and incurs substantial computational time and resource costs. Methods [4]â[6] segment, complete, and generate multiple instances while neglecting layout information, which may lead to a deterioration in the quality of the generated output.

<!-- image-->  
Fig. 9. Evaluation of the proposed modules on the Shining3D tooth dataset. Two samples are separated by the black dotted line. The baseline (GALA3D) refines layout without handling collisions. Integrating Graph Diffusion Model (GDM) improves tooth layout in a jaw, while adding Geometry Collision Loss (GCL) effectively resolves collisions, improving geometry quality.

TABLE III  
3D METRIC COMPARISON OF THE SHINING3D, AORALSCAN3, AND DEEPBLUE DATASET. âââ AND âââ MEANS LOWER AND UPPER IS BETTER, RESPECTIVELY. â\*â MEANS THE APPROACH IS IMPLEMENTED BY OURSELVES.
<table><tr><td rowspan="2">Approach</td><td colspan="3">Shining3D</td><td colspan="3">Aoralscan3</td><td colspan="3">DeepBlue</td></tr><tr><td>CDâ</td><td>F-Scoreâ</td><td>PDâ</td><td>CDâ</td><td>F-Scoreâ</td><td>PDâ</td><td>CDâ</td><td>F-Scoreâ</td><td>PDâ</td></tr><tr><td>TranSDFNet [16]</td><td>0.33</td><td>0.80</td><td>0.16</td><td>0.37</td><td>0.77</td><td>0.19</td><td>0.36</td><td>0.78</td><td>0.18</td></tr><tr><td>Point-to-mesh [17]*</td><td>0.28</td><td>0.82</td><td>0.14</td><td>0.34</td><td>0.79</td><td>0.18</td><td>0.30</td><td>0.81</td><td>0.16</td></tr><tr><td>SSEN [18]*</td><td>0.25</td><td>0.83</td><td>0.14</td><td>0.30</td><td>0.81</td><td>0.17</td><td>0.27</td><td>0.82</td><td>0.16</td></tr><tr><td>VBCD [19]</td><td>0.24</td><td>0.83</td><td>0.12</td><td>0.27</td><td>0.81</td><td>0.15</td><td>0.26</td><td>0.82</td><td>0.14</td></tr><tr><td>DPD [20]</td><td>0.34</td><td>0.79</td><td>0.17</td><td>0.38</td><td>0.77</td><td>0.21</td><td>0.36</td><td>0.78</td><td>0.19</td></tr><tr><td>2Stage [21]*</td><td>0.32</td><td>0.80</td><td>0.16</td><td>0.37</td><td>0.78</td><td>0.20</td><td>0.35</td><td>0.79</td><td>0.19</td></tr><tr><td>3Stage [22]*</td><td>0.31</td><td>0.81</td><td>0.15</td><td>0.33</td><td>0.79</td><td>0.18</td><td>0.32</td><td>0.80</td><td>0.17</td></tr><tr><td>MVDC [23]</td><td>0.28</td><td>0.82</td><td>0.14</td><td>0.32</td><td>0.80</td><td>0.18</td><td>0.30</td><td>0.81</td><td>0.16</td></tr><tr><td>DM [24]</td><td>0.30</td><td>0.81</td><td>0.15</td><td>0.32</td><td>0.79</td><td>0.17</td><td>0.31</td><td>0.80</td><td>0.16</td></tr><tr><td>Ours</td><td>0.22</td><td>0.86</td><td>0.07</td><td>0.26</td><td>0.84</td><td>0.10</td><td>0.24</td><td>0.85</td><td>0.09</td></tr></table>

Methods [7]â[9], [44], [46] explore layouts using large language models or pairwise relations. However, these methods may produce geometric inconsistencies among instances due to their inability to incorporate higher-order information. In contrast, our approach demonstrates a significant improvement over all other methods in terms of FID and LPIPS. For example, our method improves FID by 2.42% compared to MIDI [46] on the Shining3D tooth design dataset, thereby underscoring its efficacy in compositional 3D generation.

The 3DGS are converted to mesh using Dreamgaussian [27]. A comparative analysis of the quality of the generated mesh is conducted between different tooth generation methods, and the results are reported in Table III. Both encoder-decoder architecture [16]â[19] and the GAN-based approaches [20]â [23] receive similar effectiveness. Our approach improves CD by a margin of 6.0%-8.0% and PD by 7.0%-8.0% due to the layout prior, dual optimization of instance and scene, and collision avoidance regularization.

Qualitative Comparison. Qualitative comparisons of rendered images among MVDC, VBCD, ComboVerse, GALA3D, and our approach using the Shining3D tooth design dataset are illustrated in Fig. 10. Our approach, which employs a graph diffusion model and a collision-loss mechanism based on 3D Gaussians, demonstrates the ability to generate multiple teeth that seamlessly integrate with the surrounding dentition, producing visually coherent and high-quality editing results. Qualitative comparisons of the corresponding meshes among these approaches are illustrated in Fig. 11. The results are consistent with those observed in the rendered-image comparisons: both the locations and surfaces of missing teeth are accurately predicted, thanks to our modelâs awareness of global structure and local spatial relationships.

User Study. To assess the subjective aspects of scene editing, we conducted a user study comparing our approach with state-of-the-art (SOTA) alternatives. The study collected a total of 241 votes based on three primary criteria: 3D consistency, collision avoidance, and fidelity to textual descriptions. As shown in Fig. 12, our method was predominantly favored across these metrics.

Resource Consumption. The proposed approach demonstrates a consistent reduction in instance conflicts across various settings, requiring an additional 0.4 GB of memory compared to GALA3D, with a training duration of one hour. Furthermore, the efficiency comparison of different methodologies applied to the Shining3D dataset is presented in Table IV. Although the proposed method exhibits slower performance relative to ComboVerse and GALA3D, this additional computational overhead is justified by a 3.3-point improvement in FID over GALA3D and by the generation of collision-free geometry, making it suitable for precision-critical applications in dentistry.

## E. Discussion

We propose a compositional 3D generation approach for oral scenarios that predicts the layout using a graph diffusion model. Subsequently, optimizations are performed iteratively at both the scene and instance levels. Among these optimizations, we introduce a collision-loss function based on 3D Gaussians to penalize tooth intersections, thereby enabling the simulation of multiple missing teeth when only text and image prompts are provided.

<!-- image-->  
Fig. 10. Qualitative comparisons of rendered images between MVDC, VBCD, ComboVerse, GALA3D, and our approach on the Shining3D tooth design dataset are illustrated. The results of the two samples are divided by the black dotted line. Colorful bounding boxes are used for comparison of multiview consistency.

<!-- image-->  
Fig. 11. Qualitative comparisons of meshes between MVDC, VBCD, ComboVerse, GALA3D, and our approach on the Shining3D tooth design dataset are illustrated. Each row represents a specific sample. GT means the ground truth.

<!-- image-->  
Fig. 12. User Study. In a comprehensive user study encompassing three evaluation criteria that are 3D consistency, collision avoidance, and text fidelity, our approach achieves the highest scores of all the criteria.

TABLE IV  
EFFICIENCY COMPARISON OF DIFFERENT APPROACHES IN THESHINING3D DATASET.
<table><tr><td>Approach</td><td>MVDream</td><td>GALA3D</td><td>ComboVerse</td><td>Ours</td></tr><tr><td>Time (minute)</td><td>2.5</td><td>4.2</td><td>4.4</td><td>4.7</td></tr></table>

The experimental results derived from two public datasets illustrate the efficacy of the proposed approach. Specifically, the layout prior improves significantly in compositional instance editing, as the spatial distribution of multiple neighboring elements is considered simultaneously during inference. In addition, collision conflicts are mitigated by incorporating a shape prior based on intravariance within the 3D Gaussian splatting representation.

In our experiments, the average intravariance R for typical teeth ranges from 3.0 mm to 6.0 mm, depending on the size of the tooth (e.g., molars exhibit larger R values). Assume that h is the distance between the 3D Gaussian points of neighboring teeth and the center of the anchor tooth, the collision loss function effectively resolves overlaps when h < R, with a tolerance of approximately 0.1â0.3 mmâwell within the clinical requirements for dental models. For example, if R = 3.0 mm and h = 2.8 mm, the loss penalizes overlaps of 0.2 mm or greater. In cases where h > R, minor overlaps may persist, but our dual-level optimization (Algorithm 2) ensures that such residuals are minimized through global scene consistency.

The efficacy of layout optimization via graph diffusion lies in its ability to model intricate structural dependencies and the anatomical constraints inherent in dental arrangements. Unlike discrete graph generation, the diffusion process systematically denoises the target layout while integrating textâgraph joint conditioning through cross-attention layers. This methodology ensures semantic alignment with clinical requirements (e.g., tooth types) and geometric coherence (e.g., symmetry and occlusion). Regularization based on KL divergence enforces consistency between the denoised and original jaw graphs, thereby maintaining biomechanical plausibility. By conceptualizing teeth as interconnected nodes with hierarchical relationships, the model adeptly resolves multiscale spatial conflicts, thereby avoiding the local minima that are common in pairwise relationâbased approaches. This iterative refinement achieves a harmonious balance between global scene consistency and instance-level realism, as demonstrated by the improved FID and LPIPS metrics.

<!-- image-->  
Fig. 13. Illustration of the drawbacks in our approach. Multiple generated teeth may adhere to neighboring teeth.

Although our approach enhances the quality and robustness of compositional 3D tooth models, some limitations in generation remain to be addressed. In compositional optimization for scenes and instances, the generated teeth may adhere to neighboring teeth due to their high similarity in appearance. Examples are illustrated in Fig. 13. The layout prior partially mitigates this issue by integrating biological knowledge and contextual information related to the jaw model; however, the problem of tooth adherence between adjacent teeth still persists. Moreover, the efficiency of inference requires improvement. Our approach requires approximately five minutes to generate multiple teeth, incorporating layout optimization through graph diffusion and compositional optimization for both scenes and instances.

For teeth with severe malformations, atypical implants, or pathological geometries, the symmetric intravariance $R _ { i }$ (derived from Gaussian sparsity) becomes an unreliable collision threshold. Instead, adaptive collision modeling should be introduced: 1) Surface-Aware Metrics: Replace centroidbased $R _ { i }$ with curvature-aware or boundary-focused distances. Extract mesh surfaces from Gaussians via lightweight Poisson reconstruction, then compute pointwise SDFs or nearest-edge distances for collision zones. 2) Hierarchical intravariance: Decompose irregular teeth into subregions (e.g., crown/root) and compute localized values of $R _ { i }$ . Apply the collision loss per subregion to handle asymmetric shapes. To decrease inference time in compositional 3D tooth generation, a coarse-tofine optimization strategy may be implemented. Additionally, progressive Gaussian pruning could remove low-opacity splats during early optimization stages, reducing computational overhead in later refinements.

## V. CONCLUSION

In this paper, we present an approach for the compositional generation of 3D teeth. This approach infers the tooth layout by progressively denoising the source graph using text and graph constraints. Additionally, collision conflicts are mitigated by integrating a tooth shape prior based on 3D Gaussian splatting. Comprehensive experiments from multiple datasets demonstrated that our approach consistently outperforms stateof-the-art approaches, including MVDC, VBCD, ComboVerse, and GALA3D, in terms of 3D consistency, collision prevention, and text fidelity. The integration of a dual-level optimization scheme further ensures global scene stability while minimizing local overlaps within clinically acceptable precision tolerances.

Despite these advancements, certain challenges remain. In particular, compositional optimization may occasionally cause adjacent teeth to adhere due to high inter-tooth similarity. Future research will explore refined biological priors, instancelevel disentanglement strategies, and hybrid diffusion-neural field representations to further enhance structural differentiation and realism in complex dental reconstructions.

## REFERENCES

[1] H. Li, H. Zhai, X. Yang, Z. Wu, Y. Zheng, H. Wang, J. Wu, H. Bao, and G. Zhang, âImtooth: Neural implicit tooth for dental augmented reality,â IEEE Transactions on Visualization and Computer Graphics, vol. 29, no. 5, pp. 2837â2846, 2023.

[2] M. Dastan, M. Fiorentino, E. D. Walter, C. Diegritz, A. E. Uva, U. Eck, and N. Navab, âCo-designing dynamic mixed reality drill positioning widgets: A collaborative approach with dentists in a realistic setup,â IEEE Transactions on Visualization and Computer Graphics, vol. 30, no. 11, pp. 7053â7063, 2024.

[3] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, âDreamFusion: Textto-3D using 2D diffusion,â in International Conference on Learning Representations, 2023, pp. 2201â2218.

[4] C. Ge, C. Xu, Y. Ji, C. Peng, M. Tomizuka, P. Luo, M. Ding, V. Jampani, and W. Zhan, âCompGS: Unleashing 2D compositionality for compositional text-to-3D via dynamically optimizing 3D Gaussians,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 18 509â18 520.

[5] Y. Chen, T. Wang, T. Wu, X. Pan, K. Jia, and Z. Liu, âComboVerse: Compositional 3D assets creation using spatially-aware diffusion guidance,â in European Conference on Computer Vision, 2024, pp. 128â146.

[6] H. Yan, Y. Li, Z. Wu, S. Chen, W. Sun, T. Shang, W. Liu, T. Chen, X. Dai, C. Ma et al., âFrankenstein: Generating semantic-compositional 3D scenes in one tri-plane,â in SIGGRAPH Asia Conference, 2024, pp. 881â891.

[7] X. Zhou, X. Ran, Y. Xiong, J. He, Z. Lin, Y. Wang, D. Sun, and M.- H. Yang, âGALA3D: Towards text-to-3D complex scene generation via layout-guided generative Gaussian splatting,â in International Conference on Machine Learning, 2024, pp. 862â872.

[8] X.-L. Li, H. Li, H.-X. Chen, T.-J. Mu, and S.-M. Hu, âDIScene: Object decoupling and interaction modeling for complex scene generation,â in SIGGRAPH Asia Conference, 2024, pp. 991â1012.

[9] X. Yuan, H. Yang, Y. Zhao, and D. Huang, âDreamScape: 3D scene creation via gaussian splatting joint correlation modeling,â Computational Visual Media, vol. 10, no. 6, pp. 161â172, 2024.

[10] J. Ho, A. Jain, and P. Abbeel, âDenoising diffusion probabilistic models,â in Advances in Neural Information Processing Systems, 2020, pp. 6840â 6851.

[11] L. Kong, J. Cui, H. Sun, Y. Zhuang, B. A. Prakash, and C. Zhang, âAutoregressive diffusion model for graph generation,â in International Conference on Machine Learning, 2023, pp. 17 391â17 408.

[12] P. Wang, Y. Tian, N. Liu, J. Wang, S. Chai, X. Wang, and R. Wang, âA tooth surface design method combining semantic guidance, confidence, and structural coherence,â IET Computer Vision, vol. 16, no. 8, pp. 727â 735, 2022.

[13] Y. Tian, G. Jian, J. Wang, H. Chen, L. Pan, Z. Xu, J. Li, and R. Wang, âA revised approach to orthodontic treatment monitoring from oralscan video,â IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 12, pp. 5827â5836, 2023.

[14] Y. Tian, H. Fu, H. Wang, Y. Liu, Z. Xu, H. Chen, J. Li, and R. Wang, âRgb oralscan video-based orthodontic treatment monitoring,â Science China Information Sciences, vol. 67, no. 1, p. 112107, 2024.

[15] J. Z. Ye, T. Ãrkild, P. L. SÃ¸ndergard, and S. Hauberg, âVariational point encoding deformation for dental modeling,â in International Conference on Machine Learning Workshop on Structured Probabilistic Inference {\&} Generative Modeling, 2023.

[16] X. Shen, C. Zhang, X. Jia, D. Li, T. Liu, S. Tian, W. Wei, Y. Sun, and W. Liao, âTranSDFNet: Transformer-based truncated signed distance fields for the shape design of removable partial denture clasps,â IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 10, pp. 4950â 4960, 2023.

[17] G. Hosseinimanesh, A. Alsheghri, J. Keren, F. Cheriet, and F. Guibault, âPersonalized dental crown design: A point-to-mesh completion network,â Medical Image Analysis, vol. 101, p. 103439, 2025.

[18] Y. Shi, Y. Wang, H. Li, and L. Chen, âSelf-structure enhance network for digital molar wax-up design,â in IEEE International Conference on Acoustics, Speech and Signal Processing, 2025, pp. 1â5.

[19] L. Wei, C. Liu, W. Zhang, Z. Zhang, S. Zhang, and H. Li, âVBCD: A voxel-based framework for personalized dental crown design,â in International Conference on Medical Image Computing and Computer-Assisted Intervention, 2025, pp. 627â636.

[20] I. Chafi, Y. Zhang, Y. Ladini, F. Cheriet, J. Keren, and F. Guibault, âExploring the use of generative adversarial networks for automated dental preparation design,â in International Symposium on Biomedical Imaging, 2025, pp. 1â5.

[21] J. Roh, J. Kim, and J. Lee, âTwo-stage deep learning framework for occlusal crown depth image generation,â Computers in Biology and Medicine, vol. 183, p. 109220, 2024.

[22] J. Wu, Y. Huang, J. He, K. Chen, W. Wang, and X. Li, âAutomatic restoration and reconstruction of defective tooth based on deep learning technology,â BMC Oral Health, vol. 25, no. 1, pp. 1â29, 2025.

[23] X. Yang, Q. Deng, M. Huang, L. Jiang, and D. Zhang, âMVDC: A multiview dental completion model based on contrastive learning,â in IEEE International Conference on Acoustics, Speech and Signal Processing, 2025, pp. 1â5.

[24] O. Saleh, B. Spies, L. Brandenburg, M. Metzger, J. Luchtenborg, M. Blatz, and F. Burkhardt, âFeasibility of using two generative AI models for teeth reconstruction,â Journal of Dentistry, vol. 151, p. 105410, 2024.

[25] C. Wang, G. Wei, J. K. H. Tsoi, Z. Cui, S. Lu, Z. Liu, and Y. Zhou, âDiff-OSGN: Diffusion-based occlusal surface generation network with geometric constraints,â Computational Visual Media, vol. 11, no. 4, pp. 817â832, 2025.

[26] J. Tang, Z. Chen, X. Chen, T. Wang, G. Zeng, and Z. Liu, âLGM: Large multi-view Gaussian model for high-resolution 3D content creation,â in European Conference on Computer Vision, 2024, pp. 1111â1118.

[27] J. Tang, J. Ren, H. Zhou, Z. Liu, and G. Zeng, âDreamGaussian: Generative gaussian splatting for efficient 3D content creation,â in International Conference on Learning Representations, 2024, pp. 2301â 2318.

[28] J. Wang, J. Fang, X. Zhang, L. Xie, and Q. Tian, âGaussianEditor: Editing 3D Gaussians delicately with text instructions,â in IEEE Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 902â20 911.

[29] J. Zhuang, D. Kang, Y.-P. Cao, G. Li, L. Lin, and Y. Shan, âTIP-Editor: An accurate 3D editor following both text-prompts and image-prompts,â ACM Transactions on Graphics, vol. 43, no. 4, pp. 1â12, 2024.

[30] Y. Chen, Z. Chen, C. Zhang, F. Wang, X. Yang, Y. Wang, Z. Cai, L. Yang, H. Liu, and G. Lin, âGaussianEditor: Swift and controllable 3D editing with Gaussian splatting,â in IEEE Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 476â21 485.

[31] M. Ye, M. Danelljan, F. Yu, and L. Ke, âGaussian grouping: Segment and edit anything in 3D scenes,â in European Conference on Computer Vision, 2024, pp. 162â179.

[32] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, âHigh-resolution image synthesis with latent diffusion models,â in IEEE Conference on Computer Vision and Pattern Recognition, 2022, pp. 10 684â10 695.

[33] Y. Cao, Y.-P. Cao, K. Han, Y. Shan, and K.-Y. K. Wong, âDreamAvatar: Text-and-shape guided 3D human avatar generation via diffusion models,â in IEEE Conference on Computer Vision and Pattern Recognition, 2024, pp. 958â968.

[34] R. Gao, A. Holynski, P. Henzler, A. Brussee, R. Martin-Brualla, P. Srinivasan, J. T. Barron, and B. Poole, âCAT3D: Create anything in 3D with multi-view diffusion models,â in Advances in Neural Information Processing Systems, 2024, pp. 4476â4485.

[35] Z. Chen, Y. Wang, F. Wang, Z. Wang, and H. Liu, âV3D: Video diffusion models are effective 3D generators,â IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 1, pp. 1â18, 2025.

[36] J. Han, F. Kokkinos, and P. Torr, âVFusion3D: Learning scalable 3D generative models from video diffusion models,â in European Conference on Computer Vision, 2024, pp. 333â350.

[37] F. Yang, J. Zhang, Y. Shi, B. Chen, C. Zhang, H. Zhang, X. Yang, J. Feng, and G. Lin, âMagic-Boost: Boost 3D generation with mutliview conditioned diffusion,â arXiv preprint arXiv:2404.06429, 2024.

[38] H. Han, R. Yang, H. Liao, J. Xing, Z. Xu, X. Yu, J. Zha, X. Li, and W. Li, âREPARO: Compositional 3D assets generation with differentiable 3D layout alignment,â in International Conference on Computer Vision, 2025, pp. 2840â2850.

[39] J. Zhou, Y.-S. Liu, and Z. Han, âZero-shot scene reconstruction from single images with deep prior assembly,â in Advances in Neural Information Processing Systems, 2024, pp. 6411â6420.

[40] A. Dogaru, M. Ozer, and B. Egger, âGeneralizable 3D scene recon- Â¨ struction via divide and conquer from a single view,â in International Conference on 3D Vision, 2024, pp. 1476â1485.

[41] J. Zhou, X. Li, L. Qi, and M.-H. Yang, âLayout-your-3D: Controllable and precise 3D generation with 2D blueprint,â in International Conference on Learning Representations, 2025, pp. 462â472.

[42] Y. Shi, P. Wang, J. Ye, M. Long, K. Li, and X. Yang, âMVDream: Multi-view diffusion for 3D generation,â in International Conference on Learning Representations, 2024, pp. 4401â4418.

[43] L. Zhang, A. Rao, and M. Agrawala, âAdding conditional control to text-to-image diffusion models,â in IEEE International Conference on Computer Vision, 2023, pp. 3836â3847.

[44] Q. Zhang, C. Wang, A. Siarohin, P. Zhuang, Y. Xu, C. Yang, D. Lin, B. Zhou, S. Tulyakov, and H.-Y. Lee, âTowards text-guided 3D scene composition,â in IEEE Conference on Computer Vision and Pattern Recognition, 2024, pp. 6829â6838.

[45] G. Gao, W. Liu, A. Chen, A. Geiger, and B. Scholkopf, âGraphDreamer: Compositional 3D scene synthesis from scene graphs,â in IEEE Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 295â 21 304.

[46] Z. Huang, Y.-C. Guo, X. An, Y. Yang, Y. Li, Z.-X. Zou, D. Liang, X. Liu, Y.-P. Cao, and L. Sheng, âMIDI: Multi-instance diffusion for single image to 3D scene generation,â in IEEE Conference on Computer Vision and Pattern Recognition, 2025, pp. 2568â2577.

[47] H. Yan, M. Zhang, Y. Li, C. Ma, and P. Ji, âPhyCAGE: Physically plausible compositional 3D asset generation from a single image,â arXiv preprint arXiv:2411.18548, 2024.

[48] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3D Gaussian splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, pp. 139â150, 2023.

[49] Y. Tian, Y. Zhang, W.-G. Chen, D. Liu, H. Wang, H. Xu, J. Han, and Y. Ge, â3D tooth instance segmentation learning objectness and affinity in point cloud,â ACM Transactions on Multimedia Computing, Communications, and Applications, vol. 18, no. 4, pp. 1â16, 2022.

[50] Y. Li, Y. Dou, Y. Shi, Y. Lei, X. Chen, Y. Zhang, P. Zhou, and B. Ni, âFocalDreamer: Text-driven 3D editing via focal-fusion assembly,â in AAAI Conference on Artificial Intelligence, 2024, pp. 3279â3287.

[51] Y. Tian, G. Cheng, J. Gelernter, S. Yu, C. Song, and B. Yang, âJoint temporal context exploitation and active learning for video segmentation,â Pattern Recognition, vol. 100, p. 107158, 2020.

[52] X. Zhao, J. Jiang, Y. Tian, L. Wu, Z. Xu, W.-f. Yang, Y. Zou, and X. Wang, âInnovative tooth segmentation using hierarchical features and bidirectional sequence modeling,â Pattern Recognition, vol. 175, p. 113045, 2026.

[53] M. Chen, I. Laina, and A. Vedaldi, âDGE: Direct gaussian 3D editing by consistent multi-view editing,â in European Conference on Computer Vision, 2024, pp. 2311â2318.

[54] Y. Wang, X. Yi, Z. Wu, N. Zhao, L. Chen, and H. Zhang, âViewconsistent 3D editing with Gaussian splatting,â in European Conference on Computer Vision, 2024, pp. 404â420.

[55] J. Wu, J.-W. Bian, X. Li, G. Wang, I. Reid, P. Torr, and V. A. Prisacariu, âGaussCtrl: Multi-view consistent text-driven 3D Gaussian splatting editing,â in European Conference on Computer Vision, 2024, pp. 2204â 2220.

<!-- image-->

Yan Tian received his Ph.D. degree from Beijing University of Posts and Telecommunications, Beijing, China, in 2011. Then he held a postdoctoral research fellow position (2012-2015) in the Department of Information and Electronic Engineering, Zhejiang University, Hangzhou, China. He is currently a Professor at the School of Computer Science and Technology of Zhejiang Gongshang University, China. His current interests are machine learning and video analysis.

<!-- image-->

Pengcheng Xue received his bachelorâs degree from the School of Computer Science and Technology, Changchun University of Finance and Economics, China, in 2022. He is currently pursuing his Masterâs degree at the School of Computer Science and Technology, Zhejiang Gongshang University, China. His research interests include machine learning and computer vision.

<!-- image-->

Weiping Ding received the Ph.D. degree in Computer Science, Nanjing University of Aeronautics and Astronautics, Nanjing, China, in 2013. From 2014 to 2015, he was a Postdoctoral Researcher at the Brain Research Center, National Chiao Tung University, Hsinchu, Taiwan, China. In 2016, he was a Visiting Scholar at National University of Singapore, Singapore. From 2017 to 2018, he was a Visiting Professor at University of Technology Sydney, Australia. Now he is the Full Professor of Nantong University. His research directions involve

granular data mining and multimodal machine learning. He serves as an Associate Editor/Area Editor/Editorial Board member of more than 10 international prestigious journals, such as IEEE Transactions on Neural Networks and Learning Systems, IEEE Transactions on Fuzzy Systems, IEEE/CAA Journal of Automatica Sinica, IEEE Transactions on Emerging Topics in Computational Intelligence, IEEE Transactions on Intelligent Transportation Systems, Information Fusion, Neurocomputing, Applied Soft Computing, et al. He was the Leading Guest Editor of Special Issues in several prestigious journals, including IEEE Transactions on Evolutionary Computation, IEEE Transactions on Fuzzy Systems, Information Fusion, et al.

<!-- image-->

Mahmoud Hassaballah received the Doctor of Engineering in Computer Science from Ehime University, Japan in 2011. He is currently a Professor of Computer Science at the Department of Computer Science, Prince Sattam Bin Abdulaziz University, Saudi Arabia. Also, he is a full Professor at the Department of Computer Science, Qena University, Egypt. He serves as a reviewer for several Journals such as IEEE Transactions on Image Processing, IEEE Transactions on Circuits and Systems for Video Technology, IEEE Transactions on Industrial

Informatics, IEEE Transactions on Fuzzy Systems. Also, he is a TPC member of many conferences. He is an Editorial Board member of Pattern Analysis and Applications, Real-Time Image Processing, IET Image Processing, and Imaging Science Journal. His research interests include human-centered artificial intelligence, machine learning, computer vision, biometrics, image processing, feature extraction, object detection/recognition, and data security.

<!-- image-->

Karen Egiazarian received Ph.D. degree in physics and mathematics from Moscow State University, Russia, in 1986, and Doctor of Technology in signal processing from Tampere University of Technology, Finland, in 1994. He is a Professor of Signal Processing at the Department of Computing Sciences, Tampere University, Tampere, Finland. He is an IEEE Fellow. He was Editor-in-Chief of the Journal of Electronic Imaging, served as associate editor of the IEEE Transactions on Image Processing. His main research interests are in the field of computational imaging, compressed sensing, efficient signal processing algorithms, image/video restoration and compression.

<!-- image-->

Aura Conci is an engineer with M.Sc. and Ph.D. in structures, professor at Universidade Federal Fluminense (UFF). She works now in the areas of computer modeling, computer vision, image analysis and bioinformatics. She oriented hundred students and is a member of the: ACM, ISGG, ABCM, ISPRS, SBrT and SBC. She acts in the editorial office of a number of international journals and has cooperated on research with many scholars in various countries. She has a number of high quality publications (around 6k citations, h index=41 and i10=120). She has funding from Brazilian and EU governments for coordinating over 30 research projects.

<!-- image-->

Abdulkadir Sengur received the B.Sc. degree in electronics and computers education, the M.Sc. degree in electronics education, and the Ph.D. degree in electrical and electronics engineering from Firat University, Turkey, in 1999, 2003, and 2006, respectively. He became a Research Assistant with the Technical Education Faculty, Firat University, in February 2001. He is currently a Professor with the Technology Faculty, Firat University. His research interests include signal processing, image segmentation, pattern recognition, medical image processing,

and computer vision.  
<!-- image-->

Leszek Rutkowski received the M.Sc., Ph.D., and D.Sc. degrees from the WrocÅaw University of Technology, WrocÅaw, Poland, in 1977, 1980, and 1986, respectively, and the Honoris Causa degree from the AGH University of Science and Technology, KrakÂ´ow, Poland, in 2014. He is with the Systems Research Institute of the Polish Academy of Sciences, Warsaw, Poland, and with the Institute of Computer Science, AGH University of Science and Technology, Krakow, Poland, in both places serving as a professor. He is an Honorary Professor of the

Czestochowa University of Technology, Poland, and he also cooperates with the University of Social Sciences in ÅÂ´odÂ´z, Poland. His research interests include machine learning, data stream mining, big data analysis, neural networks, stochastic optimization and control, agent systems, fuzzy systems, image processing, pattern classification, and expert systems. He has published seven monographs and more than 300 technical papers, including more than 40 in various series of IEEE Transactions. He is the president and founder of the Polish Neural Networks Society. He is on the editorial board of several most prestigious international journals. He is a recipient of the IEEE Transactions on Neural Networks Outstanding Paper Award. He is a Full Member (Academician) of the Polish Academy of Sciences, elected in 2016, and a Member of the Academia Europaea, elected in 2022. He is also a Life Fellow of IEEE.