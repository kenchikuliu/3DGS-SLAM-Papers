# Large Material Gaussian Model for Relightable 3D Generation

Jingrui Yeâ, Lingting Zhuâ, Runze Zhang, Zeyu Hu, Yingda Yin, Lanjiong Li, Lequan Yuâ , Qingmin Liaoâ 

AbstractâThe increasing demand for 3D assets across various industries necessitates efficient and automated methods for 3D content creation. Leveraging 3D Gaussian Splatting, recent large reconstruction models (LRMs) have demonstrated the ability to efficiently achieve high-quality 3D rendering by integrating multiview diffusion for generation and scalable transformers for reconstruction. However, existing models fail to produce the material properties of assets, which is crucial for realistic rendering in diverse lighting environments. In this paper, we introduce the Large Material Gaussian Model (MGM), a novel framework designed to generate high-quality 3D content with Physically Based Rendering (PBR) materials, i.e., albedo, roughness, and metallic properties, rather than merely producing RGB textures with uncontrolled light baking. Specifically, we first fine-tune a new multiview material diffusion model conditioned on input depth and normal maps. Utilizing the generated multiview PBR images, we explore a Gaussian material representation that not only aligns with 2D Gaussian Splatting but also models each channel of the PBR materials. The reconstructed point clouds can then be rendered to acquire PBR attributes, enabling dynamic relighting by applying various ambient light maps. Extensive experiments demonstrate that the materials produced by our method not only exhibit greater visual appeal compared to baseline methods but also enhance material modeling, thereby enabling practical downstream rendering applications.

Index Termsâ3D Generation, Material Modeling, Large Reconstruction Model, Gaussian Splatting.

## I. INTRODUCTION

Automatic 3D content creation holds immense potential across various domains, including digital gaming, virtual reality, and filmmaking. Recently, 3D generative models achieve significant advancements, and they have remarkably reduced huge manual labor of professional 3D artists and democratized creativity in 3D asset generation for non-experts. Previous optimization-based methods [1, 2, 3, 4, 5, 6] in 3D generation have focused predominantly on score distillation sampling (SDS) with 2D diffusion model priors, which is often hindered by expensive optimization, the Janus problem, and inconsistency between different views. Recent advancements have significantly shortened inference time by leveraging large reconstruction model (LRM), which usually take multiview images as input. Some approaches [7, 8, 9, 10, 11, 12, 13, 14] utilize transformers [15] to learn the triplane-based [16] neural radiance fields (NeRF) [17], but fail to reconstruct detailed geometry and texture due to the low-resolution nature and inferior post-conversion. More recently, several works [10, 11, 14, 18, 19, 20] employ 3D Gaussian Splatting [21] or its variant [22] as 3D representation and facilitate high-resolution training with expressiveness, combined with attention mechanism to generate Gaussians from multiview images.

Despite these advancements, such approaches still struggle to control light shading and do not account for the material properties of the 3D objects, e.g., Physically Based Rendering (PBR) materials. Consequently, the reconstructed 3D objects often exhibit uncontrolled light baking in textures, resulting in a lack of realistic rendering effects and an inability to adapt appearances under varying ambient lighting conditions. This significantly limits their applicability in downstream applications. To address these limitations, as far as we know, we are the first to propose a novel method for generating 3D Gaussians with PBR materials, which are adaptable for relighting during Gaussians rendering. Our approach consists of two main components: a generative model and a reconstruction model, starting from a text prompt and finally producing 3D rendering incorporating PBR. Initially, we train a textto-PBR multiview diffusion model on 3D material datasets, built upon multiview diffusion models, e.g., MVDream [23]. Subsequently, a large reconstruction model based on 2D Gaussian Splatting [22] is employed to reconstruct the sparse views of the generated PBR images following the designs and architectures of recent Gaussian-based reconstruction models [10, 14]. To ensure global consistency across both the generation and reconstruction stages, we incorporate geometry data, i.e., depth and normal maps, to serve as proxy information. The integration offers benefits in three folds: 1) We can seamlessly adopt geometry generation models, e.g., [12, 24, 25, 26, 27] for additional geometry guidance and mitigate multiview inconsistency; 2) The assistance of geometry guidance models, e.g., text-to-geometry models and text-to-depth/normal models [6], can be optionally removed when untextured 3D models are available to produce rendering geometry maps, aligning more closely with practical usages; 3) Since the PBR images inherently lack lighting cues and differ significantly from the distribution of shaded images, relying solely on them for training radiance fields produces Gaussian points with inaccurate spatial details, indicating the geometry guidance is necessary in reconstruction. In practice, we design a geometry-conditioned multiview generation model with ControlNet [28] in the generation stage, and then introduce geometry features injection and utilize geometric guidance to produce additional supervision in reconstruction. Incorporating efficient rasterization, our method achieves high-resolution renderings along with albedo, roughness, and metallic maps, enabling dynamic relighting effects under various ambient lighting conditions.

Extensive qualitative and quantitative experiments validate the effectiveness of our method, demonstrating that it matches or exceeds state-of-the-art Gaussian Splatting based generation quality while achieving realistic rendering results under diverse lighting conditions. Our contributions are summarized as follows:

1) We pioneer to propose a novel framework to generate high-resolution Gaussians with material properties from text prompts, capable of being dynamically relighted through physically based rendering.

2) We introduce principled models tailored to relightable Gaussians, including a controllable multiview PBR model and a unified material Gaussian reconstruction model.

3) Extensive experiments demonstrate that our method excels in both appearance and material quality with efficiency.

## II. RELATED WORK

## A. Large Reconstruction Model

Recent advancements in NeRF [17] and 3DGS [21] enable high-quality novel view synthesis capabilities. Pioneered by the large reconstruction model (LRM) [7], recent works [8, 9, 12, 13] demonstrate that image tokens can be directly mapped to 3D representations, typically triplane-NeRF, in a feed-forward manner via a scalable transformer-based architecture [15] with large-scale 3D training data [29, 30, 31]. Among them, Instant3D [8] integrates LRM with multiview diffusion models [23, 32, 33, 34, 35], using four views of generated images for better quality. To avoid inefficient volume rendering and limited triplane resolution, some concurrent works [10, 11, 36] follow Instant3D and introduce 3DGS or its variants [22] into sparse-view LRM variants for more efficient rendering. Specifically, LGM [10] combines 3D Gaussians from different views using a convolution-based asymmetric U-Net [37], along with other Gaussian reconstruction models like [18, 19, 38, 39]. GRM [11] and GS-LRM [36] use pixelaligned Gaussian [40, 41] with a pure transformer-based reconstruction model. LaRa [14] adopts 2DGS [22] representations and models scenes as Gaussian volumes and group attention layers for better quality and faster convergence.

## B. Optimization-based 3D Generation

In 3D generation, due to the scarcity of 3D data, leveraging 2D priors has become an explorable method. DreamFusion [1] pioneers the optimization of 3D assets by distilling from pretrained image diffusion models [42], followed by a large group of successors [2, 3, 43, 44] with more advanced distillation techniques. Some methods explore more efficient 3D representations, e.g., hashgrid [45] and 3DGS [21], to accelerate the optimization [46, 47, 48, 49]. DreamGaussian proposes a two-stage coarse-to-fine Gaussian generation method, and performs texture refinement in UV space. GaussianDreamer [48] combines Gaussian distribution with 3D and 2D diffusion models [42, 50], and introduces noise point growth and color perturbation to enrich the details. However, these optimization processes still incur additional computational costs at test time. Moreover, since 2D diffusion models inherently lack multiview awareness, these methods frequently struggle with Janus problem. And approaches based on SDS loss tend to produce over-saturated colors, leading to suboptimal generation results.

## C. PBR Material and Relighting

3D objects include both geometric and material properties. Predicting only the radiance has a significant limitation: it fails to produce convincing results when relighting, as material properties such as reflectance and roughness are not captured. Therefore, a stream of researches [6, 51, 52, 53, 54, 55] focus on material generation and downstream relighting, Fantasia3D [52] introduces the bidirectional reflectance distribution function (BRDF) [56] to the task of text-to-3D, and optimizes PBR materials with SDS loss. By fine-tuning multiview diffusion, attempts like UniDream [51] and RichDreamer [6] adopt an albedo-normal aligned multiview diffusion and progressive generation for geometry and albedo-textures based on SDS, which achieves multiview consistency, but suffer from high computational costs.

Some works aim at producing PBR maps for untextured meshes. Some of them [53, 57, 58, 59] generate single-view or multiview PBR image by fine-tuning diffusion model, and continuously update the UV map with RePaint [60] and UV projection. Another type of methods directly trains a diffusion model [61, 62] or uses 2D diffusion priors [63] in the UV space to model PBR, improving global consistency but may suffer from poor generalization.

Other works [55, 64, 65, 66, 67, 68, 69] train models on multiview data with varying lighting conditions and viewpoints. ReLitLRM [64] proposes a geometry-based transformer and adopts a diffusion-based relightable appearance generator, where the geometry is separated from the appearance to better handle the uncertainty caused by lighting. ARM [65] decouples geometry from appearance, processes appearance within the UV space, and introduces a material prior that encodes semantic appearance information to address the material and illumination ambiguities present in sparseview input images. However, these methods do not model material properties of objects and often require high training costs.

## III. PRELIMINARY

## A. 2D Gaussian Splatting

3DGS [21] achieves great success in 3D rendering with radiance field. Yet, It models angular radiance as a single blob, limiting high-quality surface reconstruction. 2DGS [22] further takes advantage of standard surfel modeling [70, 71], adopting 2D oriented disks as surface elements and better align with thin surfaces. Specifically, 2DGS evaluates Gaussian values at 2D disks and utilizes explicit ray-splat intersection, resulting in a perspective-correct splatting:

$$
\mathcal { G } ( \pmb { u } ) = \exp \left( - \frac { \boldsymbol { u } ( \pmb { r } ) ^ { 2 } + \boldsymbol { v } ( \pmb { r } ) ^ { 2 } } { 2 } \right) ,\tag{1}
$$

where $\textbf { \em u } = ~ ( u ( \boldsymbol { r } ) , v ( \boldsymbol { r } ) )$ is the intersection point between ray r and the primitive in UV space. And each Gaussian primitive has its own view-dependent color $^ { c , }$ defined by SH coefficients $f$ with degree $k .$ In summary, each 2D Gaussian can be characterized as:

$$
\Theta = \{ x , s , \pmb q , \alpha , \pmb f \} ,\tag{2}
$$

Albedo

Roughness

Metallic

Normal

<!-- image-->  
Fig. 1: The text prompt of the above case is ${ } ^ { \ast } a$ green single-door toy $c a r ^ { \prime \prime }$ . Taking the textual description as input, our method can generate high-quality Gaussians consisting of albedo, roughness, and metallic that can be relighted for photo-realistic rendering under any new illumination environments. We invite readers to read our appendix for more relighting results.

with position $\pmb { x } \in \mathbb { R } ^ { 3 }$ , scaling vector $\boldsymbol { s } \in \mathbb { R } ^ { 2 }$ , rotation vector $q \in \mathbb { R } ^ { 4 }$ , opacity $\alpha \in \mathbb { R } ^ { 1 }$ and SH coefficients $\pmb { f } \in \mathbb { R } ^ { 3 \times ( k + 1 ) ^ { 2 } }$ For rendering, Gaussians are sorted according to their centers and composed into pixels using front-to-back alpha blending:

$$
\mathbf { c } ( { \pmb r } ) = \sum _ { i = 1 } { \pmb { c } } _ { i } \alpha _ { i } \hat { \mathcal { G } } _ { i } ( { \pmb u } ) T _ { i } ,\tag{3}
$$

where $T _ { i }$ is the approximated accumulated transmittances defined by $\begin{array} { r } { \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } \hat { \mathcal { G } } _ { j } ( { \pmb u } ) ) } \end{array}$ where $\hat { \mathcal G }$ denotes the results after a low-pass filter, and the integration process is terminated when the accumulated opacity reaches saturation.

## B. Large Gaussian Models

Leveraging the efficient rasterization of 3DGS [21], recent methods such as LGM [10] and LaRa [14] introduce Gaussian-based reconstruction models. They follow the LRM approach [7], demonstrating significant advancements in 3D content reconstruction and creation. Our material Gaussian reconstruction model is built upon LaRa [14], which takes M images $\pmb { I } = ( I _ { 1 } , \ldots , \pmb { I } _ { M } )$ with camera parameters $( \pmb { \pi } _ { 1 } , \dots , \pmb { \pi } _ { M } )$ to reconstruct radiance fields as a collection of 2D Gaussians. The model operates on a 3D voxel grid and consists of three volumes: an image feature volume $V _ { \mathrm { f } }$ coping with image conditions, an embedding volume $V _ { \mathrm { e } }$ describing prior learned from data, and a Gaussian volume $V _ { \mathrm { g } }$ representing the radiance field.

Given multiview images, feature maps of M views are extracted by DINO [72] encoder with Plucker ray directions, Â¨ then lifted to 3D to form the image feature volume $V _ { \mathrm { f } } .$ A volume transformer containing a set of group attention layers is used to predict the Gaussian volumes and each group attention layer contains three sublayers: group cross-attention, a MLP, and 3D convolution. The image feature volume $V _ { \mathrm { f } }$ and embedding volume $V _ { \mathrm { e } }$ are unfolded into G groups along each axis, and cross-attention is applied between the corresponding groups of embedding tokens $V _ { \mathrm { e } } ^ { g , j }$ and feature tokens $V _ { \mathrm { f } } ^ { g }$ where j denotes the index of the layer starting from 1 and $\{ V _ { \mathrm { e } } ^ { g , 1 } \} _ { g } = V _ { \mathrm { e } }$ . The embedding groups are then updated using an MLP to produce $\{ \ddot { V } _ { \mathrm { e } } ^ { g , j } \} _ { g = 1 } ^ { G }$ 1 which are reassembled into $\ddot { V } _ { \mathrm { e } } ^ { j }$ and fed into a 3D convolutional layer. The process is described as:

$$
\dot { V } _ { \mathrm { e } } ^ { g , j } = \mathrm { G r o u p C r o s s A t t n } \left( \mathrm { L N } \left( V _ { \mathrm { e } } ^ { g , j } \right) , V _ { \mathrm { f } } ^ { g } \right) + V _ { \mathrm { e } } ^ { g , j } ,\tag{4}
$$

$$
\ddot { V } _ { \mathrm { e } } ^ { g , j } = \mathrm { M L P } \left( \mathrm { L N } \left( \dot { V } _ { \mathrm { e } } ^ { g , j } \right) \right) + \dot { V } _ { \mathrm { e } } ^ { g , j } ,\tag{5}
$$

$$
\begin{array} { r } { V _ { \mathrm { e } } ^ { j + 1 } = 3 \mathrm { D C N N } \left( \mathrm { L N } \left( \ddot { V } _ { \mathrm { e } } ^ { j } \right) \right) + \ddot { V } _ { \mathrm { e } } ^ { j } . } \end{array}\tag{6}
$$

After passing through all group attention layers, LaRa employs a 3D transposed CNN to upscale the updated embedding volume $\dot { V _ { \mathrm { e } } }$ and obtain the Gaussian volume: $V _ { \mathrm { g } } ~ =$ Transpose-3DCNN $\left( \dot { V } _ { \mathrm { e } } \right)$ . Finally, 2D Gaussian primitives are decoded from the Gaussian volume using a coarse-to-fine strategy and achieve efficient high-resolution image rendering through the rasterization in 2DGS [22].

## IV. METHODOLOGY

In this section, we discuss the full pipeline of our Large Material Gaussian Model (MGM). To begin with, we introduce the designs of multiview PBR diffusion in Section IV-A. Then, we discuss our material Gaussian reconstruction model in Section IV-B. Finally, we present that with our material Gaussian representation, downstream relighting can be seamlessly integrated in Section IV-C. An overview of MGM is shown in Figure 2.

## A. Multiview PBR Diffusion

We aim to synthesize PBR materials, including albedo of 3 channels, roughness and metallic of 1 channel each. To achieve this, we modify base model that is originally designed for shaded image space generation and fails to decompose lighting effects to suit the need of material attributes with additional channels and modalities. To re-target the base Stable Diffusion [42] or multiview models like MVDream [23] for downstream applications, the implementations have been relatively standard in recent years, i.e., preparation of downstream data and re-purposing the base models for new targets with optionally new modules, demonstrated by a wide range of applications [6, 59, 73, 74, 75]. Specifically, we train two submodels where one for albedo and one for roughness/metallic. For the latter, we inflate the outermost blocks in the input and output blocks, adopting parameter copy for initialization of additional branches. We leverage the 3D datasets [30] to render a consistent multiview PBR image dataset for fine-tuning, denoted as $\mathscr { X } _ { m v - P B R } .$ Following the same training regimen of multiview diffusion models, our model can synthesize texture images from four viewpoints, and align precisely with the input geometry with multiple ControlNet branches [28] that take each target viewâs rendered depth map ${ \cal I } _ { d }$ and normal map $I _ { n }$ as input control images $I _ { c } = \{ I _ { d } , I _ { n } \}$

<!-- image-->  
Fig. 2: Overview of our method. Starting with a text input, our MGM generates multiview depth and normal maps, serving as geometry priors. These maps, along with the text prompt, guide the generation of multiview PBR images through our specialized multiview PBR diffusion process. Subsequently, these images inform our material Gaussian reconstruction model, which reconstructs the material Gaussians complete with PBR components. The integration of PBR materials ensures that these Gaussians can be seamlessly utilized to achieve realistic lighting effects in various rendering scenarios.

Formally, given a set of noisy image $\pmb { x } _ { t } \in \mathbb { R } ^ { F \times H \times W \times C }$ the feature of text prompt y, and a set of extrinsic camera parameters $\pmb { c } \in \mathbb { R } ^ { F \times 1 6 }$ , where F is the view number dimension, our multiview PBR diffusion is trained to generate a set of images $\pmb { x } _ { 0 } ~ \in ~ \mathbb { R } ^ { F \times H \times W \times C }$ of the same scene from F different view angles with $C = 5$ channels. For training samples $\{ \pmb { x } , \pmb { y } , \pmb { I _ { c } } , \pmb { c } \} \in \mathcal { X } _ { m v - P B R }$ , the loss is defined as:

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } , \boldsymbol { \chi } _ { m v - P B R } ) = \mathbb { E } _ { \boldsymbol { x } , \boldsymbol { y } , \boldsymbol { c } , t , \boldsymbol { \epsilon } } \Big [ \| \boldsymbol { \epsilon } - \boldsymbol { \epsilon } _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { t } ; \boldsymbol { y } , I _ { c } , \boldsymbol { c } , t ) \| _ { 2 } ^ { 2 } \Big ] , } \end{array}\tag{7}
$$

where $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ is the noisy image diffused with random noise $\epsilon ,$ and the $\epsilon _ { \theta }$ is the output of multiview material diffusion model Î¸.

During inference, we use text prompts along with the geometry information to generate multiview PBR diffusion model. To obtain the multiview depth and normal conditions, we can resort to pre-trained 3D geometry generation models for supplying the proxy information, denoted as multiview depth and normal generation block shown in Figure 2. In practice, we use CraftsMan [24] to get a 3D mesh and render the multiview images at desired viewpoints in inference. Though existing geometry generation models fail to produce high-quality untextured meshes that are really close to the distribution of the high-quality 3D assets in the datasets, thanks to the generalization capabilities of our trained multiview PBR diffusion, our method can still cope with generated multiview geometry conditions. Furthermore, as sometimes untextured 3D meshes are already available in real-world pipelines, we can directly render it to acquire geometry guidance for PBR diffusion models, indicating the wide use-cases of our model.

## B. Material Gaussian Reconstruction Model

A naive approach to acquire Gaussians set with multiple attributes is to train three reconstruction models for PBR material channels respectively, but this would reconstruct three point clouds with different geometry, leading to obvious artifacts in the PBR rendering and a lack of practical methods for their integration. To address this, we designed a material Gaussian representation that contains multi-channel material properties. We keep all geometric and position-related Gaussian parameters the same as Gaussian Splatting, and re-target the SH coefficients to degraded view-independent colors comprising of 5 channels, where the first 3 channels correspond to albedo, and the last 2 channels represent roughness and metallic. Here we consider that albedo does not contain any lighting components, so orders higher than 1 in SH coefficients should not be used. In summary, our material Gaussian can be represented as:

$$
\Theta = \{ x , s , r , \alpha , f _ { a } , f _ { r } , f _ { m } \} ,\tag{8}
$$

where $f _ { a } \in \mathbb { R } ^ { 3 }$ and $\pmb { f } _ { r / m } \in \mathbb { R } ^ { 1 }$ . This unified representation aligns with the design of 2DGS but also models materials and enables training a unified model operating on PBR-aware Gaussians.

Since PBR images lack lighting clues compared to shaded images, we find that the geometry reconstruction performance when directly using PBR data for model training is inferior to that achieved with shaded data, this trend occurs for both the unified model and the three separate models. In addition, it is essential that the reconstructed geometry aligns with the proxy geometry provided to the multiview PBR diffusion model. To address the problem above, we take depth and normal maps for the second use in the full pipeline, incorporating them into the reconstruction model to enhance the modelâs perception of the spatial location, redesigning the reconstruction model and adopting additionally geometry supervision. As a result, the predicted positions and opacities of the Gaussians can be more accurate under the guidance. To be specific, given $5 \times N$ images containing albedo $\pmb { I _ { a } } = ( I _ { a } ^ { 1 } , \ldots , \pmb { I _ { a } ^ { N } } )$ roughness $\pmb { I _ { r } } = ( \bar { I } _ { r } ^ { 1 } , \ldots , \pmb { I _ { r } ^ { N } } )$ , metallic $\pmb { I _ { m } } = ( I _ { m } ^ { 1 } , \ldots , I _ { m } ^ { N } )$ depth $\pmb { I _ { d } } = ( \pmb { I } _ { d } ^ { 1 } , \dots , \pmb { I } _ { d } ^ { N } )$ , and normal $\pmb { I _ { n } } = ( I _ { n } ^ { 1 } , \ldots , I _ { n } ^ { N } )$ ; along with N camera embedding $\pmb { c } = ( c ^ { 1 } , \dots , c ^ { N } )$ and $\pmb { c } ^ { i } = ( \pmb { d } ^ { i } , ( \pmb { o } ^ { i } \times \pmb { d } ^ { i } ) ) \in \mathbb { R } ^ { 6 }$ , where o is the camera location and d is the view direction vector. We first apply the image encoder [76] to extract features and concatenate with camera embedding for each type of image:

<!-- image-->  
Fig. 3: Detailed designs of generative modules and reconstruction modules. a) Multiview PBR Diffusion. With the input text prompt and camera embeddings, and the rendered geometry views the controllable diffusion with adapted blocks aim at synthesizing multiview materials. b) Material Gaussian Reconstruction Model. Multiview images are passed through the Image Encoder to extract features with view direction injected and the Volume Decoder predicts the Gaussian volume.

$$
\begin{array} { r } { h _ { a } = \mathrm { C o n c a t } ( \mathcal { F } _ { a } ( I _ { a } ) , \mathcal { F } _ { g } ( I _ { d } ) , \mathcal { F } _ { g } ( I _ { n } ) , c ) , } \end{array}\tag{9}
$$

$$
h _ { r / m } = \mathrm { C o n c a t } ( \mathcal { F } _ { r / m } ( I _ { r / m } ) , \mathcal { F } _ { g } ( I _ { d } ) , \mathcal { F } _ { g } ( I _ { n } ) , c ) ,\tag{10}
$$

where $\mathcal { F } _ { a } , \mathcal { F } _ { g } , \mathcal { F } _ { r / m }$ are the image encoders for albedo, depth/normal, and roughness/metallic, respectively. With these image feature $\mathcal { H } _ { f } = ( h _ { a } , h _ { r } , h _ { m } )$ and an embedding volume $\mathcal { H } _ { e }$ which is a learnable query embedding that aims to capture data prior dynamically, we employ a volume decoder $\Phi _ { g }$ based on transformer from LaRa [14] to predict the 3D Gaussian volume, and the input is unfolded:

$$
\mathcal { H } _ { g } = \Phi _ { g } ( \mathrm { F l a t t e n } ( \mathcal { H } _ { f } ) , \mathrm { F l a t t e n } ( \mathcal { H } _ { e } ) ) .\tag{11}
$$

Finally, our material Gaussian primitives Î can be derived from $\mathcal { H } _ { g }$ through a lightweight parameter decoding network. Please refer to the Appendix for more details.

Reconstruction Loss. Since our material Gaussian representation comprises of multiple channels, it is crucial to optimize the PBR components jointly to achieve a satisfactory appearance and geometry. We adopt the standard reconstruction loss in differentiable rendering to guide the appearance, utilizing 2D Gaussian Splatting [22]. At each training iteration, we render the albedo ${ \hat { I } } _ { a } ,$ roughness $\hat { I } _ { r }$ and metallic $\hat { I } _ { m }$ from 8 views, including 4 reconstructed input views and 4 novel views. Following other works in LRM series [7, 10, 11, 14], we apply a straightforward image reconstruction loss between the PBR renderings from Gaussians, i.e., $\hat { \pmb { I } } = ( \hat { I } _ { a } , \hat { I } _ { r } , \hat { I } _ { m } )$ and ground-truth images, i.e., I:

$$
\mathcal { L } _ { \mathrm { I m a g e } } = \mathcal { L } _ { \mathrm { M S E } } ( \hat { \boldsymbol { I } } , \boldsymbol { I } ) + \mathcal { L } _ { \mathrm { S S I M } } ( \hat { \boldsymbol { I } } , \boldsymbol { I } ) ,\tag{12}
$$

where LMSE is the pixel-wise L2 loss, and $\mathcal { L } _ { \mathrm { S S I M } }$ is the structural similarity loss.

Geometry Regularization. The geometric priors, i.e., $I _ { g } = $ $( I _ { d } , I _ { n } )$ , are used to extract image feature. This not only ensures that the 2D Gaussians can be optimized to produce better geometry, but also helps to maintain the consistency with the PBR generation stage, as initially controlled by the proxy geometry generation. We render the depth and normal maps of 4 input views from 2D Gaussians, i.e., $\hat { \pmb { I } } _ { g } = ( \hat { \pmb { I } } _ { d } , \hat { \pmb { I } } _ { n } )$

and calculate regularization loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { G e o m e t r y } } = \mathcal { L } _ { \mathrm { M S E } } ( \hat { \pmb { I } } _ { d } , \pmb { I } _ { d } ) + \mathcal { L } _ { \mathrm { M S E } } ( \hat { \pmb { I } } _ { n } , \pmb { I } _ { n } ) . } \end{array}\tag{13}
$$

2DGS Regularization. Moreover, we add two regularization terms following 2DGS [22]: depth distortion and normal consistency. Specifically, for a ray $\mathbf { u } ( \mathbf { x } )$ originating from pixel x, the weight distribution is concentrated by minimizing the distance between ray-primitive intersections, leading to the distortion loss: $\begin{array} { r } { \mathcal { L } _ { \mathrm { d } } = \sum _ { i , j } \omega _ { i } \omega _ { j } \big | z _ { i } - z _ { j } \big | } \end{array}$ , where $\omega _ { i } =$ $\begin{array} { r } { \alpha _ { i } \mathcal { G } _ { i } ( \mathbf { u } ( \mathbf { x } ) ) \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } \mathcal { G } _ { j } ( \mathbf { u } ( \mathbf { x } ) ) ) } \end{array}$ is the blending weight of iâth intersection and $z _ { i }$ is the intersection depth. Since 2DGS explicitly models the normals, we can align the rendered normals $\mathbf { n } _ { i }$ with the normals N computed from depth via consistency loss: $\begin{array} { r } { \mathcal { L } _ { \mathrm { n } } = \sum _ { i } \omega _ { i } ( 1 - \mathbf { n } _ { i } ^ { \top } \mathbf { N } ) } \end{array}$ . The regularization term for the ray u(x) is: $\mathcal { L } _ { \mathrm { R e g } } = \gamma _ { \mathrm { d } } \mathcal { L } _ { \mathrm { d } } + \gamma _ { \mathrm { n } } \mathcal { L } _ { \mathrm { n } } .$ , and we set $\gamma _ { \mathrm { d } } = 1 0 0 0$ and $\gamma _ { \mathrm { n } } = 0 . 2$ following LaRa [14]. Our total loss function is defined as:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { I m a g e } } + \mathcal { L } _ { \mathrm { R e g } } + \mathcal { L } _ { \mathrm { G e o m e t r y } } .\tag{14}
$$

In practice, training a multi-channel model directly can be challenging, especially due to the limited availability of highquality training data for roughness and metallic components compared to albedo. To address this, we implement a two-stage training strategy. In the first stage, we freeze the roughness and metallic components of the material Gaussian, focusing exclusively on training the albedo component. In the second stage, we include all material components in the training process. This approach improves the training quality and leads to faster convergence.

## C. Relighting

In our framework, we leverage the classic rendering equation to formulate the outgoing radiance of a surface point x with normal n under varying lighting conditions:

$$
L _ { o } ( \pmb { x } , \pmb { v } ) = \int _ { \Omega } L _ { i } ( \pmb { x } , \pmb { l } ) f _ { r } ( \pmb { l } , \pmb { v } ) ( \pmb { l } \cdot \pmb { n } ) d \pmb { l } ,\tag{15}
$$

where â¦ represents the upper hemisphere centered at x, l and v denote incident and view directions respectively. $\boldsymbol { L } _ { i } ( \boldsymbol { x } , l )$ denotes the radiance received at x from l. We follow Cook-Torrance microfacet model [56, 77] to formulate the bidirectional reflectance distribution function (BRDF) $f _ { r }$ as a function of albedo $a \ \in \ [ 0 , 1 ] ^ { 3 }$ , metallic $m \in \ [ 0 , 1 ]$ , and roughness $\rho \in [ 0 , 1 ]$ :

$$
f _ { r } ( l , { \pmb v } ) = \underbrace { ( 1 - m ) \frac { \pmb a } { \pi } } _ { \mathrm { d i f f u s e } } + \underbrace { \frac { D F G } { 4 ( \pmb n \cdot l ) ( \pmb n \cdot { \pmb v } ) } } _ { \mathrm { s p e c u l a r } } ,\tag{16}
$$

where microfacet distribution function D, Fresnel reflection $F _ { ; }$ and geometric shadowing factor G are related to the surface roughness $\rho .$ With our material Gaussians that store all the PBR properties and given the view direction l, we can render the albedo a, roughness $\rho ,$ metallic m and normal map n with 2DGS rasterization, then we can apply different environment light maps to the BRDF function $f _ { r } ( \cdot )$ to acquire the relighting results.

## V. EXPERIMENTS

## A. Implementation Details

In our multiview PBR model, we train two sub-models for albedo and roughness/metallic, initialized from MV-Dream [23]. For the albedo model, we train the first two blocks in the input blocks and the last one block in the output blocks. For roughness/metallic model, we inflate the trainable parts mentioned in albedo model and train the full parameters due to the larger gap between roughness/metallic and shaded images. We use depth and normal ControlNets [28] for both models. The multiview PBR dataset is rendered from objects in the Objaverse [30] dataset with Blender, which contains about 80,000 entries, each with a descriptive caption sourced from Cap3D [78].

In our material Gaussian reconstruction model, the input images are all at a resolution of $5 1 2 \times 5 1 2$ . After DINO encoding, the shape of the image feature is $7 6 8 \times 3 2 \times 3 2$ and the resolution of the embedding volume is 32 with 256 channels. Following LaRa, the volume decoder consists of 12 group attention layers with $G = 1 6$ groups, producing a Gaussian volume of size $6 4 \times 6 4 \times 6 4 \times 8 0$ . We choose $K = 2$ primitives for each voxel and constrain the offset radius to $r = 1 / 3 2$ of the length of the bounding box. The model is trained for a total of 80 epochs, with the first and second stages each accounting for half of the total training duration. We train the model on G-buffer Objaverse (Gobjaverse) [6] dataset and filter out the âlow-qualityâ objects as defined in the dataset.

## B. Training Datasets

For our multiview PBR diffusion, we use images rendered from objects in the Objaverse [30] dataset with Blender. We filter out objects without material maps. For roughness and metallic maps, we select four pairs of roughness and metallic rendered images with elevations of 0 and azimuths of 0, 90, 180, and 270 degrees to calculate the CLIP score [80]. Objects with scores exceeding 0.95 are filtered out, as they have almost identical roughness and metallic maps, suggesting low-quality material maps. Following the camera settings in MVDream [23], we render each object from 32 views at a fixed elevation angle and camera distance in each setting, and sample three times on elevation and camera distance. The rendered image sets include albedo, roughness, metallic, normal, and depth map for each view. For normal rendering, we disable the normal UV map with fine details that comes with the model, considering that normals are generally calculated from the modelâs geometry and do not exhibit such fine details during inference. The filtered subset contains about 80,000 entries, each with a text prompt captioned by Cap3D [78].

For our material Gaussian reconstruction model, we use G-buffer Objaverse (Gobjaverse) [6] dataset, which is based on Objaverse [30]. This multiview rendering dataset includes 280,000 scenes with PBR data. After removing blank albedo maps and âlow-qualityâ objects as defined in the dataset, we obtain a subset of 100,000 albedo maps. For roughness and metallic maps, we notice that many images are single-colored, likely due to default settings in Gobjaverse for missing channels. To address this, we implement a simple filtering strategy to exclude images where the roughness or metallic foreground pixel value is unique, resulting in a subset of 60,000 images.

<!-- image-->  
Fig. 4: Qualitative comparison. Given various text prompts, our method generates 3D assets with material properties, and achieves better geometry and appearance quality comparing to other Gaussian-based generative methods.

TABLE I: Quantitative comparison. All scores are mean value across all samples. Here GA is GaussianAnything [79].
<table><tr><td>Metrics / Methods</td><td>GA</td><td>LaRa</td><td>LGM</td><td>Paint-it</td><td>DreamMat</td><td>Ours</td></tr><tr><td>Geometry CLIP â</td><td>26.66</td><td>27.02</td><td>27.84</td><td>-</td><td>-</td><td>29.87</td></tr><tr><td>Appearance CLIP â</td><td>28.28</td><td>28.77</td><td>29.31</td><td>27.65</td><td>28.49</td><td>30.48</td></tr><tr><td>FID â</td><td>121.49</td><td>97.95</td><td>101.60</td><td>113.87</td><td>105.32</td><td>89.55</td></tr><tr><td>Time â</td><td>-</td><td>-</td><td>-</td><td>40min</td><td>1h15min</td><td>30sec</td></tr></table>

## C. Details on Proxy Geometry Generation

Since our multiview PBR diffusion is conditioned on the multiview depth and normal maps as geometry prior, we need to generate them from text input. Initially, we plan to use the multiview depth and normal diffusion provided by Richdreamer [6] to obtain four multiview depth and normal maps directly from the input text. However, upon testing, we find that this model directly resize the image to 32 Ã 32 as the latent feature and dose not train the VAE with depth and normal additionally, resulting in generated images that are very blurry and noisy. As a result, we shift our approach to generate a mesh from the text prompt and then rendering the mesh. We first input the text into Stable Diffusion 2.1 [42] to obtain the corresponding image, and then remove the background of the image and import it into CraftsMan [24] to create a 3D model. The resulting mesh goes through a round of redundant faces deletion, and then we normalize it to fit within a bounding box of [â0.5, 0.5] and translate it to the origin. Finally we render the mesh with Pytorch3D [81] to obtain multi-view depth and normal maps, the camera distance is set to 1.5.

## D. Coarse-to-Fine Decoding of Reconstruction Model

Following LaRa [14], we extract 2D Gaussian primitive shape and appearance parameters from the Gaussian volume using a coarse-to-fine decoding process. In the coarse decoding stage, Gaussian volume features are fed into a lightweight MLP, which outputs a set of K Gaussian parameters for each voxel. In order to better preserve the appearance, a fine decoding stage is added to guide the fine texture prediction. Specifically, the primitive centers $p _ { i } ^ { k }$ are projected onto the coarse renderings, i.e., RGB image ËI, depth image DË , and alpha map AË), to incorporate the coarse renderings for each primitive using the camera poses Ï,

$$
\mathcal { X } _ { p _ { i } ^ { k } } = \left( I _ { p _ { i } ^ { k } } , \hat { I } _ { p _ { i } ^ { k } } , \hat { D } _ { p _ { i } ^ { k } } , \hat { \mathrm { A } } _ { p _ { i } ^ { k } } \right) = \Phi \left( \mathcal { P } \left( p _ { i } ^ { k } , p i \right) , \oplus \left[ I , \hat { I } , \hat { D } , \hat { \mathrm { A } } \right] \right)\tag{17}
$$

where P denotes the point projection, â is a concatenation operation along the channel dimension, and Î¦ represents bilinear interpolation. We then implement a point-based crossattention layer between the features of point $\mathcal { X } _ { p _ { i } ^ { k } }$ and the primitive voxel. The results are subsequently fed into an MLP, which is designed to predict the residual spherical harmonics:

<!-- image-->  
Fig. 5: Comparison with post-processing for material estimation. For Ours, the first, second, and third columns are albedo, roughness, and metallic, respectively. For LGM [10] and LaRa [14], the first column is the shaded RGB images, and the last three columns are decomposed albedo/roughness/metallic by IDArb [54], denote as LGM+IDArb and LaRa+IDArb. The text prompts are âAn orange and red carpâ, âA cartoon-style purple robotâ, and âA wooden barrelâ, respectively.

TABLE II: User study. The rating is of scale 1-5, where the higher the better.
<table><tr><td>Methods</td><td>Multiview Consistency â</td><td>Textual Alignment â</td><td>Geometric Integrity</td><td>Overall Quality â</td></tr><tr><td>GA [79]</td><td>2.85</td><td>3.30</td><td>3.15</td><td>2.96</td></tr><tr><td>LGM [10]</td><td>3.38</td><td>3.74</td><td>3.51</td><td>3.63</td></tr><tr><td>LaRa [14]</td><td>3.47</td><td>3.62</td><td>3.65</td><td>3.45</td></tr><tr><td>Paint-it [63]</td><td>3.06</td><td>3.30</td><td>-</td><td>3.27</td></tr><tr><td>DreamMat [82]</td><td>3.15</td><td>3.55</td><td>-</td><td>3.40</td></tr><tr><td>MGM (Ours)</td><td>3.66</td><td>3.78</td><td>3.92</td><td>3.80</td></tr></table>

$$
\mathrm { S H } _ { i , k } ^ { r e s i d u a l s } = \mathrm { M L P } \left( \mathrm { C r o s s A t t n } \left( \mathcal { X } _ { p _ { i } ^ { k } } , V _ { \mathcal { G } } ^ { i } \right) \right) ,\tag{18}
$$

$$
\mathrm { S H } _ { i , k } ^ { f i n e } = \mathrm { S H } _ { i , k } ^ { c o a r s e } + \mathrm { S H } _ { i , k } ^ { r e s i d u a l s }\tag{19}
$$

Finally, our work takes advantage of 2DGS [22] to enable efficient high-resolution rendering, and both coarse and fine stages are differentiable and updated simultaneously.

## E. Baselines and Evaluation Metrics

We conduct comparisons primarily with Gaussian Splatting based 3D generation methods and other PBR texture generation methods, including LaRa [14], LGM [10], GaussianAnything [79], DreamMat [82] and Paint-it [63]. LaRa and LGM are two methods that follow LRM [7] fashion with Gaussian Splatting representation. GaussianAnything is a native 3D diffusion model that employs a 3D VAE to decode and generate 2D Gaussians. Itâs important to note that these methods are limited to generate shaded textures rather than PBR materials. All the 3D Gaussians based methods achieve fast generation, requiring less than a minute to produce one scene. DreamMat and Paint-it are two PBR texture generation methods based on 2D-lifting optimization, we use the geometry generated by CraftsMan [24] as input for these two methods.

Evaluating the quality of text-to-3D models is challenging due to the lack of standard metrics. To objectively evaluate the quality, we adopt the evaluation method from RichDreamer [6] and use the CLIP model (ViT-B-32) to calculate the Geometry CLIP Score and Appearance CLIP Score. The CLIP score ranges from 0 to 1 and is multiply by 100. We also calculated FID [83] to evaluate the generated appearance quality, as well as the time required for the generation to measure the efficiency of different methods. We select 148 text prompts from various 3D generation works [10, 79, 84, 85, 86], render 24 predefined views for each object, and compute the average Geometry/Appearance CLIP score and FID. For our approach, we randomly select one of six ambient maps as the light source to synthesize the final rendering for evaluation, the light maps are shown in the Appendix.

Since DreamMat [82] and Paint-it [63] do not generate geometry, we do not calculate the geometry-related metrics of these two methods, i.e., Geometry CLIP score and Geometric Integrity. We also do not calculate the inference time of Gaussian-based generation methods, i.e., GaussianAnything [79], LGM [10] and LaRa [14], which are roughly comparable to our method.

## F. Main Comparison

Qualitative Results. As shown in Figure 4, we render images from the generated 3D assets for visualization. The results produced by our method align appropriately with the input text and demonstrate strong generalization capabilities. For the geometry quality, other methods sometimes exhibit artifacts such as fragmentation or floating elements, whereas our method preserves a complete and detailed geometric topology. For the generated textures, other methods may produce blurry or unrealistic textures and the lighting factors cannot be removed. In contrast, our approach yields high-quality PBR materials with rich texture details, and the materials are closely aligned with the text inputs. Notably, the albedo maps generated by our approach do not contain any lighting or shadow effects, and the roughness and metallic maps accurately reflect the necessary details for different regions of the model. In addition to the single-object results, we also find that the multiobject generated by our method exhibits superior geometry and texture appearance compared to the baseline methods. This improvement can be attributed to our incorporation of geometric prior guidance from multiview depth and normal maps in the inference stage of the multiview PBR diffusion. We include the results of multi-object generation in Appendix.

Quantitative Results. Table I demonstrates that our method achieves state-of-the-art performances on geometry CLIP score, appearance CLIP score and FID compared to other methods. In particular, the geometry CLIP score is significantly better than other methods, indicating that our geometry prior is of great help in improving the geometric quality. Our method outperforms other methods in quantitative assessments and the results consistently align with human subjective judgments in qualitative evaluations. And our method can generate within one minute, while methods like DreamMat [82] and Paint-it [63] usually require more than dozens of minutes for optimization.

User Study. To further assess the visual quality of the generated 3D models, we conduct a comprehensive user study with 12 volunteers. Each participant receives 8 examples, along with the corresponding text prompts. There are 4 evaluation indicators: overall quality, text consistency, multiview consistency, and geometric integrity, each scored from 1 to 5, with 1 being the worst and 5 being the best. Table II presents the results of our user study. In terms of multiview consistency, text consistency, geometric integrity, and overall quality, our method receives the highest scores. We also conducted a user study on the quality of PBR generation in Table III, comparing with DreamMat [82], Paint-it [63], and LGM [10]/LaRa [14] with IDArb [54] post-processing. Our method achieved the highest scores in albedo, roughness, and metallic quality.

In the appendix, we further present several generation examples based on corresponding text prompts with all the baseline methods, and we add GVGEN [18] for comparison which is also a 3D diffusion model using a 3D U-Net to output 3D Gaussians. And we show more of the generated material Gaussian with appropriate PBR properties, with the realistic rendering results under different physics world lighting. In addition to single-object generation, 3D generation also includes multi-object generation, we show the comparison results of multi-object generation in the appendix as well. It can be seen that under the guidance of geometric priors, the multiple objects we generate have a more complete and reasonable geometry, as well as rich texture colors and semantically aligned PBR materials. And we present the quantitative evaluation and the user study of multi-object generation, respectively. The superiority of our method over the single-object generation is more prominent obviously.

We also compared our multiview PBR diffusion with other multiview generation models, ie, MV-Adapter [87] and MV-Dream [23], both of which can produce multiview images based on the text input. Some generation results are shown in the appendix. We can find that MV-Adapter and MVDream can only generate images with undecoupled light and shadow, while our multiview PBR diffusion generates PBR images without any light and shadow, and the roughness and metallic are highly consistent with the text prompts and have rich details. The results of the quantitative comparison of multiview generation are all given in the appendix.

TABLE III: User Study on PBR. The rating is of scale 1-5, where the higher the better.
<table><tr><td>Methods</td><td>Albedo Quality â</td><td>Roughness Quality</td><td>Metallic Quality</td></tr><tr><td>LGM+IDArb [10, 54]</td><td>2.43</td><td>2.94</td><td>2.75</td></tr><tr><td>LaRa+IDArb [14, 54]</td><td>2.56</td><td>2.90</td><td>2.78</td></tr><tr><td>Paint-it [63]</td><td>3.17</td><td>3.10</td><td>3.13</td></tr><tr><td>DreamMat [82]</td><td>3.29</td><td>3.45</td><td>3.36</td></tr><tr><td>MGM (Ours)</td><td>3.97</td><td>3.55</td><td>3.60</td></tr></table>

## G. Comparison with Post-Processing on Material Estimation

To demonstrate that our model can generate semantically accurate materials with high-quality details, we consider adopting post-processing modules like those in [54, 88, 89] based on other large reconstruction models that producing shaded images to extract materials. Here, we choose IDArb [54] as it decomposes PBR materials from multiview shaded images with high quality. Figure 5 shows our materials and the results of LGM and LaRa after PBR extraction by IDArb. It is observed that the albedo generated through the postprocessing method can partially eliminate light and shadow effects. However, the extracted roughness and metallic maps are monochromatic, lacking detail and exhibiting poor alignment with the text. In contrast, the PBR materials we generate not only produce an albedo free from lighting interference but also include highly detailed roughness and metallic maps that align closely with the textual semantics.

## H. Ablation Study

To validate the effectiveness of each component of our model, we conduct an ablation study on the text prompt âA green potted succulent plantâ to generate materials. The qualitative results are shown in Figure 6, and the quantitative results is presented in Table IV, it can be seen that the indicators deteriorate when our proposed components are disabled one by one. Additionally, we train the models using the firstorder and second-order spherical harmonic coefficients, which the metrics are also slightly lower, proving that PBR images without lighting information are more suitable for the simplest color modeling, i.e., directly using 3 color channels.

Disable Depth and Normal. Depth and normal maps provide crucial spatial information for the Gaussian points. When these images are not included, the quality of the reconstructed geometry significantly deteriorates, becoming smoother and losing a lot of details with slightly blurry textures.

<!-- image-->  
Albedo  
Roughness  
Metallic

Fig. 6: Ablation study. The first row is the results generated by our multiview PBR diffusion. The next four lines are the reconstruction results corresponding to the ablation method. For the unified geometry model (full model, w/o depth and normal), we visualize the normal on albedo. For the reconstruction model with three independent channels, we visualize the normals on the PBR separately.  
TABLE IV: Quantitative comparison on ablation study. All scores were mean value across all samples, values the higher the better. Here SHs-degree=1 and SHs-degree=2 represent using first- and second-order spherical coefficients to model color in 2DGS, respectively.
<table><tr><td>Metrics / Methods</td><td>Full Model</td><td>w/o Depth and Normal</td><td>Seperated Models</td><td>PBR-LaRa</td><td>SHs-degree=1</td><td>SHs-degree=2</td></tr><tr><td>Geometry CLIP â</td><td>29.87</td><td>28.92</td><td>29.39</td><td>28.70</td><td>29.68</td><td>29.51</td></tr><tr><td>Appearance CLIP â</td><td>30.48</td><td>29.85</td><td>29.64</td><td>29.22</td><td>30.14</td><td>30.05</td></tr><tr><td>FID </td><td>89.55</td><td>95.47</td><td>93.92</td><td>97.20</td><td>90.56</td><td>90.83</td></tr></table>

<!-- image-->  
Fig. 7: Training PSNR curves of one-stage and two-stage strategies respectively.

Three Independent Gaussian Models. Training PBR component separately does not yield a substantial difference in reconstruction quality compared to combined training. However, this approach results in three different geometries during inference, as highlighted by the red boxes in the fourth row of Figure 6. This geometric inconsistency will lead to a notable decline in relighting quality.

Reconstruction with the Model Trained on Shaded Data. Since PBR images do not contain light and shadow clues, the reconstruction model pre-trained on shaded data fails to produce satisfactory results. If we integrate our multiview PBR diffusion directly into LaRa, the quality declines significantly, as LaRa struggles to accurately predict the geometric details of the object due to the domain gap between these data distributions.

In Figure 7, we also plot the training PSNR curves for both one-stage and two-stage training strategies proposed at the end of section IV-B, demonstrating that the two-stage strategy yields superior reconstruction quality and faster convergence.

## VI. LIMITATIONS

Although our method successfully generates diverse and high-quality 3D assets with PBR material according to the text prompts, it still has several limitations. First, since material estimation is an inherently ill-posed problem, our multiview PBR diffusion exhibits inaccurate roughness and metallic generation under some circumstances. Second, our method struggles with materials that exhibit properties like transparency, high reflection, or subsurface scattering. This limitation arises primarily from our choice of the Bidirectional Reflectance Distribution Function (BRDF) model, which is not equipped to handle more advanced and complex materials. Furthermore, for the reconstructed model, our approach faces challenges in recovering high-frequency geometry and texture details, mainly due to the limitation of Gaussian volume representation based reconstruction model.

## VII. CONCLUSION

In this works, we pioneeringly propose a large multiview Gaussian model for generating high-quality 3D assets with PBR material, and support relighting with different environment maps. Distinct from previous methods can only generate shaded RGB texture, we purpose a novel material Gaussian representation together with the well designed training strategy to learn the material efficiently. Additionally, we inject the geometry prior into both our generation and reconstruction stages to better supervise the material reconstruction, improving the completeness and detail of generated geometry and appearance. In summary, we presents the first generative methods for producing Gaussians with PBR and achieves stateof-the-art text-to-3D generation quality based on Gaussian Splatting representation, the versatility and applicability have been proved across various contexts.

## REFERENCES

[1] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, âDreamfusion: Textto-3d using 2d diffusion,â arXiv preprint arXiv:2209.14988, 2022.

[2] Z. Wang, C. Lu, Y. Wang, F. Bao, C. Li, H. Su, and J. Zhu, âProlificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation,â Advances in Neural Information Processing Systems, vol. 36, 2024.

[3] C.-H. Lin, J. Gao, L. Tang, T. Takikawa, X. Zeng, X. Huang, K. Kreis, S. Fidler, M.-Y. Liu, and T.-Y. Lin, âMagic3d: High-resolution text-to-3d content creation,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 300â309.

[4] Z. Chen, R. Su, J. Zhu, L. Yang, J.-H. Lai, and X. Xie, âVividdreamer: Towards high-fidelity and efficient text-to-3d generation,â arXiv preprint arXiv:2406.14964, 2024.

[5] Y. Liang, X. Yang, J. Lin, H. Li, X. Xu, and Y. Chen, âLuciddreamer: Towards high-fidelity text-to-3d generation via interval score matching,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6517â6526.

[6] L. Qiu, G. Chen, X. Gu, Q. Zuo, M. Xu, Y. Wu, W. Yuan, Z. Dong, L. Bo, and X. Han, âRichdreamer: A generalizable normal-depth diffusion model for detail richness in text-to-3d,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 9914â9925.

[7] Y. Hong, K. Zhang, J. Gu, S. Bi, Y. Zhou, D. Liu, F. Liu, K. Sunkavalli, T. Bui, and H. Tan, âLrm: Large reconstruction model for single image to 3d,â arXiv preprint arXiv:2311.04400, 2023.

[8] J. Li, H. Tan, K. Zhang, Z. Xu, F. Luan, Y. Xu, Y. Hong, K. Sunkavalli, G. Shakhnarovich, and S. Bi, âInstant3d: Fast text-to-3d with sparseview generation and large reconstruction model,â arXiv preprint arXiv:2311.06214, 2023.

[9] X. Wei, K. Zhang, S. Bi, H. Tan, F. Luan, V. Deschaintre, K. Sunkavalli, H. Su, and Z. Xu, âMeshlrm: Large reconstruction model for highquality mesh,â arXiv preprint arXiv:2404.12385, 2024.

[10] J. Tang, Z. Chen, X. Chen, T. Wang, G. Zeng, and Z. Liu, âLgm: Large multi-view gaussian model for high-resolution 3d content creation,â in European Conference on Computer Vision. Springer, 2025, pp. 1â18.

[11] Y. Xu, Z. Shi, W. Yifan, H. Chen, C. Yang, S. Peng, Y. Shen, and G. Wetzstein, âGrm: Large gaussian reconstruction model for efficient 3d reconstruction and generation,â arXiv preprint arXiv:2403.14621, 2024.

[12] J. Xu, W. Cheng, Y. Gao, X. Wang, S. Gao, and Y. Shan, âInstantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models,â arXiv preprint arXiv:2404.07191, 2024.

[13] Y. Xu, H. Tan, F. Luan, S. Bi, P. Wang, J. Li, Z. Shi, K. Sunkavalli, G. Wetzstein, Z. Xu et al., âDmv3d: Denoising multi-view diffusion using 3d large reconstruction model,â arXiv preprint arXiv:2311.09217, 2023.

[14] A. Chen, H. Xu, S. Esposito, S. Tang, and A. Geiger, âLara: Efficient large-baseline radiance fields,â in European Conference on Computer Vision. Springer, 2025, pp. 338â355.

[15] A. Vaswani, âAttention is all you need,â Advances in Neural Information Processing Systems, 2017.

[16] E. R. Chan, C. Z. Lin, M. A. Chan, K. Nagano, B. Pan, S. De Mello, O. Gallo, L. J. Guibas, J. Tremblay, S. Khamis et al., âEfficient geometry-aware 3d generative adversarial networks,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 16 123â16 133.

[17] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[18] X. He, J. Chen, S. Peng, D. Huang, Y. Li, X. Huang, C. Yuan, W. Ouyang, and T. He, âGvgen: Text-to-3d generation with volumetric representation,â in European Conference on Computer Vision. Springer, 2025, pp. 463â479.

[19] Z. Wang, Y. Wang, Y. Chen, C. Xiang, S. Chen, D. Yu, C. Li, H. Su, and J. Zhu, âCrm: Single image to 3d textured mesh with convolutional reconstruction model,â in European Conference on Computer Vision. Springer, 2025, pp. 57â74.

[20] J. Chen, C. Li, J. Zhang, L. Zhu, B. Huang, H. Chen, and G. H. Lee, âGeneralizable human gaussians from single-view image,â arXiv preprint arXiv:2406.06050, 2024.

[21] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[22] B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, â2d gaussian splatting for geometrically accurate radiance fields,â in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1â11.

[23] Y. Shi, P. Wang, J. Ye, M. Long, K. Li, and X. Yang, âMvdream: Multiview diffusion for 3d generation,â arXiv preprint arXiv:2308.16512, 2023.

[24] W. Li, J. Liu, R. Chen, Y. Liang, X. Chen, P. Tan, and X. Long, âCraftsman: High-fidelity mesh generation with 3d native generation and interactive geometry refiner,â arXiv preprint arXiv:2405.14979, 2024.

[25] D. Tochilkin, D. Pankratz, Z. Liu, Z. Huang, A. Letts, Y. Li, D. Liang, C. Laforte, V. Jampani, and Y.-P. Cao, âTriposr: Fast 3d object reconstruction from a single image,â arXiv preprint arXiv:2403.02151, 2024.

[26] K. Wu, F. Liu, Z. Cai, R. Yan, H. Wang, Y. Hu, Y. Duan, and K. Ma, âUnique3d: High-quality and efficient 3d mesh generation from a single image,â arXiv preprint arXiv:2405.20343, 2024.

[27] J. Chen, L. Zhu, Z. Hu, S. Qian, Y. Chen, X. Wang, and G. H. Lee, âMar-3d: Progressive masked auto-regressor for high-resolution 3d generation,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 11 083â11 092.

[28] L. Zhang, A. Rao, and M. Agrawala, âAdding conditional control to text-to-image diffusion models,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3836â3847.

[29] T. Wu, J. Zhang, X. Fu, Y. Wang, J. Ren, L. Pan, W. Wu, L. Yang, J. Wang, C. Qian et al., âOmniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 803â814.

[30] M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. VanderBilt, L. Schmidt, K. Ehsani, A. Kembhavi, and A. Farhadi, âObjaverse: A

universe of annotated 3d objects,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13 142â13 153.

[31] X. Yu, M. Xu, Y. Zhang, H. Liu, C. Ye, Y. Wu, Z. Yan, C. Zhu, Z. Xiong, T. Liang et al., âMvimgnet: A large-scale dataset of multi-view images,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 9150â9161.

[32] P. Wang and Y. Shi, âImagedream: Image-prompt multi-view diffusion for 3d generation,â arXiv preprint arXiv:2312.02201, 2023.

[33] R. Shi, H. Chen, Z. Zhang, M. Liu, C. Xu, X. Wei, L. Chen, C. Zeng, and H. Su, âZero123++: a single image to consistent multi-view diffusion base model,â arXiv preprint arXiv:2310.15110, 2023.

[34] R. Liu, R. Wu, B. Van Hoorick, P. Tokmakov, S. Zakharov, and C. Vondrick, âZero-1-to-3: Zero-shot one image to 3d object,â in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 9298â9309.

[35] Y. Lu, J. Zhang, S. Li, T. Fang, D. McKinnon, Y. Tsin, L. Quan, X. Cao, and Y. Yao, âDirect2. 5: Diverse text-to-3d generation via multi-view 2.5 d diffusion,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8744â8753.

[36] K. Zhang, S. Bi, H. Tan, Y. Xiangli, N. Zhao, K. Sunkavalli, and Z. Xu, âGs-lrm: Large reconstruction model for 3d gaussian splatting,â in European Conference on Computer Vision. Springer, 2025, pp. 1â19.

[37] O. Ronneberger, P. Fischer, and T. Brox, âU-net: Convolutional networks for biomedical image segmentation,â in Medical image computing and computer-assisted interventionâMICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer, 2015, pp. 234â241.

[38] Z.-X. Zou, Z. Yu, Y.-C. Guo, Y. Li, D. Liang, Y.-P. Cao, and S.-H. Zhang, âTriplane meets gaussian splatting: Fast and generalizable single-view 3d reconstruction with transformers,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 10 324â10 335.

[39] C. Zhang, H. Song, Y. Wei, Y. Chen, J. Lu, and Y. Tang, âGeolrm: Geometry-aware large reconstruction model for high-quality 3d gaussian generation,â arXiv preprint arXiv:2406.15333, 2024.

[40] S. Szymanowicz, C. Rupprecht, and A. Vedaldi, âSplatter image: Ultrafast single-view 3d reconstruction,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 10 208â10 217.

[41] D. Charatan, S. L. Li, A. Tagliasacchi, and V. Sitzmann, âpixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 457â19 467.

[42] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, âHighresolution image synthesis with latent diffusion models,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684â10 695.

[43] H. Wang, X. Du, J. Li, R. A. Yeh, and G. Shakhnarovich, âScore jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 12 619â12 629.

[44] J. Sun, B. Zhang, R. Shao, L. Wang, W. Liu, Z. Xie, and Y. Liu, âDreamcraft3d: Hierarchical 3d generation with bootstrapped diffusion prior,â arXiv preprint arXiv:2310.16818, 2023.

[45] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics Â¨ primitives with a multiresolution hash encoding,â ACM transactions on graphics (TOG), vol. 41, no. 4, pp. 1â15, 2022.

[46] J. Lorraine, K. Xie, X. Zeng, C.-H. Lin, T. Takikawa, N. Sharp, T.-Y. Lin, M.-Y. Liu, S. Fidler, and J. Lucas, âAtt3d: Amortized text-to-3d object synthesis,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 17 946â17 956.

[47] J. Tang, J. Ren, H. Zhou, Z. Liu, and G. Zeng, âDreamgaussian: Generative gaussian splatting for efficient 3d content creation,â arXiv preprint arXiv:2309.16653, 2023.

[48] T. Yi, J. Fang, G. Wu, L. Xie, X. Zhang, W. Liu, Q. Tian, and X. Wang, âGaussiandreamer: Fast generation from text to 3d gaussian splatting with point cloud priors,â arXiv preprint arXiv:2310.08529, 2023.

[49] Z. Chen, F. Wang, Y. Wang, and H. Liu, âText-to-3d using gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 401â21 412.

[50] H. Jun and A. Nichol, âShap-e: Generating conditional 3d implicit functions,â arXiv preprint arXiv:2305.02463, 2023.

[51] Z. Liu, Y. Li, Y. Lin, X. Yu, S. Peng, Y.-P. Cao, X. Qi, X. Huang, D. Liang, and W. Ouyang, âUnidream: Unifying diffusion priors for relightable text-to-3d generation,â in European Conference on Computer Vision. Springer, 2025, pp. 74â91.

[52] R. Chen, Y. Chen, N. Jiao, and K. Jia, âFantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation,â in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 22 246â22 256.

[53] Y. Wang, X. Xu, L. Ma, H. Wang, and B. Dai, âBoosting 3d object generation through pbr materials,â in SIGGRAPH Asia 2024 Conference Papers, 2024, pp. 1â11.

[54] Z. Li, T. Wu, J. Tan, M. Zhang, J. Wang, and D. Lin, âIdarb: Intrinsic decomposition for arbitrary number of input views and illuminations,â arXiv preprint arXiv:2412.12083, 2024.

[55] D. Shim, Y. Shi, K. Li, H. J. Kim, and P. Wang, âMvlight: Relightable text-to-3d generation via light-conditioned multi-view diffusion,â arXiv preprint arXiv:2411.11475, 2024.

[56] R. L. Cook and K. E. Torrance, âA reflectance model for computer graphics,â ACM Transactions on Graphics (ToG), vol. 1, no. 1, pp. 7â 24, 1982.

[57] B. Xiong, J. Liu, J. Hu, C. Wu, J. Wu, X. Liu, C. Zhao, E. Ding, and Z. Lian, âTexgaussian: Generating high-quality pbr material via octreebased 3d gaussian splatting,â arXiv preprint arXiv:2411.19654, 2024.

[58] X. Huang, T. Wang, Z. Liu, and Q. Wang, âMaterial anything: Generating materials for any 3d object via diffusion,â arXiv preprint arXiv:2411.15138, 2024.

[59] L. Zhang, Z. Wang, Q. Zhang, Q. Qiu, A. Pang, H. Jiang, W. Yang, L. Xu, and J. Yu, âClay: A controllable large-scale generative model for creating high-quality 3d assets,â ACM Transactions on Graphics (TOG), vol. 43, no. 4, pp. 1â20, 2024.

[60] A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, and L. Van Gool, âRepaint: Inpainting using denoising diffusion probabilistic models,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 11 461â11 471.

[61] X. Yu, P. Dai, W. Li, L. Ma, Z. Liu, and X. Qi, âTexture generation on 3d meshes with point-uv diffusion,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4206â4216.

[62] X. Yu, Z. Yuan, Y.-C. Guo, Y.-T. Liu, J. Liu, Y. Li, Y.-P. Cao, D. Liang, and X. Qi, âTexgen: a generative diffusion model for mesh textures,â ACM Transactions on Graphics (TOG), vol. 43, no. 6, pp. 1â14, 2024.

[63] K. Youwang, T.-H. Oh, and G. Pons-Moll, âPaint-it: Text-to-texture synthesis via deep convolutional texture map optimization and physicallybased rendering,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 4347â4356.

[64] T. Zhang, Z. Kuang, H. Jin, Z. Xu, S. Bi, H. Tan, H. Zhang, Y. Hu, M. Hasan, W. T. Freeman et al., âRelitlrm: Generative relightable radiance for large reconstruction models,â arXiv preprint arXiv:2410.06231, 2024.

[65] X. Feng, C. Yu, Z. Bi, Y. Shang, F. Gao, H. Wu, K. Zhou, C. Jiang, and Y. Yang, âArm: Appearance reconstruction model for relightable 3d generation,â arXiv preprint arXiv:2411.10825, 2024.

[66] M. He, P. Clausen, A. L. TasÂ¸el, L. Ma, O. Pilarski, W. Xian, L. Rikker, X. Yu, R. Burgert, N. Yu et al., âDiffrelight: Diffusion-based facial performance relighting,â in SIGGRAPH Asia 2024 Conference Papers, 2024, pp. 1â12.

[67] X. Chen, J. Zheng, H. Huang, H. Xu, W. Gu, K. Chen, H.-a. Gao, H. Zhao, G. Zhou, Y. Zhang et al., âRgm: Reconstructing high-fidelity 3d car assets with relightable 3d-gs generative model from a single image,â arXiv preprint arXiv:2410.08181, 2024.

[68] Z. He, T. Wang, X. Huang, X. Pan, and Z. Liu, âNeural lightrig: Unlocking accurate object normal and material estimation with multilight diffusion,â arXiv preprint arXiv:2412.09593, 2024.

[69] L. Zhu, J. Ye, R. Zhang, Z. Hu, Y. Yin, L. Li, J. Chen, S. Qian, X. Wang, Q. Liao et al., âMuma: 3d pbr texturing via multi-channel multi-view generation and agentic post-processing,â arXiv preprint arXiv:2503.18461, 2025.

[70] H. Pfister, M. Zwicker, J. Van Baar, and M. Gross, âSurfels: Surface elements as rendering primitives,â in Proceedings of the 27th annual conference on Computer graphics and interactive techniques, 2000, pp. 335â342.

[71] W. Yifan, F. Serena, S. Wu, C. Oztireli, and O. Sorkine-Hornung, Â¨ âDifferentiable surface splatting for point-based geometry processing,â ACM Transactions on Graphics (TOG), vol. 38, no. 6, pp. 1â14, 2019.

[72] M. Caron, H. Touvron, I. Misra, H. Jegou, J. Mairal, P. Bojanowski, and Â´ A. Joulin, âEmerging properties in self-supervised vision transformers,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 9650â9660.

[73] B. Ke, A. Obukhov, S. Huang, N. Metzger, R. C. Daudt, and K. Schindler, âRepurposing diffusion-based image generators for monocular depth estimation,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 9492â9502.

[74] X. Fu, W. Yin, M. Hu, K. Wang, Y. Ma, P. Tan, S. Shen, D. Lin, and X. Long, âGeowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image,â in European Conference on Computer Vision. Springer, 2025, pp. 241â258.

[75] X. Liu, J. Ren, A. Siarohin, I. Skorokhodov, Y. Li, D. Lin, X. Liu, Z. Liu, and S. Tulyakov, âHyperhuman: Hyper-realistic human generation with latent structural diffusion,â arXiv preprint arXiv:2310.08579, 2023.

[76] H. Zhang, F. Li, S. Liu, L. Zhang, H. Su, J. Zhu, L. M. Ni, and H.-Y. Shum, âDino: Detr with improved denoising anchor boxes for end-toend object detection,â arXiv preprint arXiv:2203.03605, 2022.

[77] B. Walter, S. R. Marschner, H. Li, and K. E. Torrance, âMicrofacet models for refraction through rough surfaces.â Rendering techniques, vol. 2007, p. 18th, 2007.

[78] T. Luo, C. Rockwell, H. Lee, and J. Johnson, âScalable 3d captioning with pretrained models,â Advances in Neural Information Processing Systems, vol. 36, 2024.

[79] Y. Lan, S. Zhou, Z. Lyu, F. Hong, S. Yang, B. Dai, X. Pan, and C. C. Loy, âGaussiananything: Interactive point cloud latent diffusion for 3d generation,â arXiv preprint arXiv:2411.08033, 2024.

[80] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., âLearning transferable visual models from natural language supervision,â in International conference on machine learning. PMLR, 2021, pp. 8748â8763.

[81] N. Ravi, J. Reizenstein, D. Novotny, T. Gordon, W.-Y. Lo, J. Johnson, and G. Gkioxari, âAccelerating 3d deep learning with pytorch3d,â arXiv:2007.08501, 2020.

[82] Y. Zhang, Y. Liu, Z. Xie, L. Yang, Z. Liu, M. Yang, R. Zhang, Q. Kou, C. Lin, W. Wang et al., âDreammat: High-quality pbr material generation with geometry-and light-aware diffusion models,â ACM Transactions on Graphics (TOG), vol. 43, no. 4, pp. 1â18, 2024.

[83] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, âGans trained by a two time-scale update rule converge to a local nash equilibrium,â Advances in neural information processing systems, vol. 30, 2017.

[84] S. Su, X. Cai, L. Gao, P. Zeng, Q. Du, M. Li, H. T. Shen, and J. Song, âGt23d-bench: A comprehensive general text-to-3d generation benchmark,â arXiv preprint arXiv:2412.09997, 2024.

[85] J. Xiang, Z. Lv, S. Xu, Y. Deng, R. Wang, B. Zhang, D. Chen, X. Tong, and J. Yang, âStructured 3d latents for scalable and versatile 3d generation,â arXiv preprint arXiv:2412.01506, 2024.

[86] Y. Siddiqui, T. Monnier, F. Kokkinos, M. Kariya, Y. Kleiman, E. Garreau, O. Gafni, N. Neverova, A. Vedaldi, R. Shapovalov et al., âMeta 3d assetgen: Text-to-mesh generation with high-quality geometry, texture, and pbr materials,â arXiv preprint arXiv:2407.02445, 2024.

[87] Z. Huang, Y.-C. Guo, H. Wang, R. Yi, L. Ma, Y.-P. Cao, and L. Sheng, âMv-adapter: Multi-view consistent image generation made easy,â arXiv preprint arXiv:2412.03632, 2024.

[88] Y. Hong, Y.-C. Guo, R. Yi, Y. Chen, Y.-P. Cao, and L. Ma, âSupermat: Physically consistent pbr material estimation at interactive rates,â arXiv preprint arXiv:2411.17515, 2024.

[89] Z. Zeng, V. Deschaintre, I. Georgiev, Y. Hold-Geoffroy, Y. Hu, F. Luan, L.-Q. Yan, and M. Hasan, âRgb Ë âx: Image decomposition and synthesis using material-and lighting-aware diffusion models,â in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1â11.