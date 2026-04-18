# AvatarBack: Back-Head Generation for Complete 3D Avatars from Front-View Images

Shiqi Xin , Xiaolin Zhang , Yanbin Liu , Peng Zhang , Caifeng Shan

AbstractâRecent advances in Gaussian Splatting have significantly boosted the reconstruction of head avatars, enabling high-quality facial modeling by representing an 3D avatar as a collection of 3D Gaussians. However, existing methods predominantly rely on frontal-view images, leaving the backhead poorly constructed. This leads to geometric inconsistencies, structural blurring, and reduced realism in the rear regions, ultimately limiting the fidelity of reconstructed avatars. To address this challenge, we propose AvatarBack, a novel plug-andplay framework specifically designed to reconstruct complete and consistent 3D Gaussian avatars by explicitly modeling the missing back-head regions. AvatarBack integrates two core technical innovations, i.e., the Subject-specific Generator (SSG) and the Adaptive Spatial Alignment Strategy (ASA). The former leverages a generative prior to synthesize identity-consistent, plausible backview pseudo-images from sparse frontal inputs, providing robust multi-view supervision. To achieve precise geometric alignment between these synthetic views and the 3D Gaussian representation, the later employs learnable transformation matrices optimized during training, effectively resolving inherent pose and coordinate discrepancies. Extensive experiments on NeRSemble and K-hairstyle datasets, evaluated using geometric, photometric, and GPT-4o-based perceptual metrics, demonstrate that Avatar-Back significantly enhances back-head reconstruction quality while preserving frontal fidelity. Moreover, the reconstructed avatars maintain consistent visual realism under diverse motions and remain fully animatable.

Index Termsâ3D Head Reconstruction, Gaussian Splatting, Generative Adversarial Networks.

## I. INTRODUCTION

3 D avatar head reconstruction is an important topic in computer vision and computer graphics, with applications to virtual humans, digital entertainment, virtual reality, and human-computer interaction. Typical approaches use parametric mesh models to compactly and interpretably represent facial geometry, e.g., 3DMM [1], FLAME [2] and DECA [3]. These methods suffer from limitations in depicting fine-grained details of target heads. Recently, 3D Gaussian

<!-- image-->  
Fig. 1. Motivation for our AvatarBack. (a) Current datasets are limited to frontal views, a result of expensive capture processes. (b) Existing methods, which rely on facial landmarks for alignment, cannot be applied to the back of the head. (c) Head reconstruction techniques with Gaussian splatting e.g., GaussianAvatars [5], fail to reconstruct the backhead regions. (d) Our proposed method reconstructs the back of the head while maintaining competitive frontal face quality.

Splatting (3DGS) [4] has emerged as an efficient alternative to achieve both real-time rendering and photo-realistic reconstructions by modeling 3D points as differentiable Gaussian distributions. Building on 3DGS, methods, e.g., GaussianAvatars [5], SplattingAvatar [6], FlashAvatar [7], Surf-Head [8], and PSAvatar [9], combine mesh priors, rigging, and Gaussian primitives to generate photorealistic and animatable head avatars. These avatars capture dynamic expressions and detailed appearance, leveraging the flexibility and efficiency of Gaussian-based representations. However, despite these impressive advances, a persistent challenge in 3D head reconstruction, particularly for 3DGS-based methods [5], [6], [7], [8], [9], [10], arises from their reliance on frontal or limited-view datasets. This limitation stems mainly from two factors: 1) acquiring full 360â¦ multi-view data is expensive and time-consuming (Fig. 1(a)), and 2) most frontal reconstruction models are designed around facial detection and landmark alignment, focusing almost exclusively on the visible front (Fig. 1(b)). As a result, these methods struggle to reconstruct the crucial but under-constrained back-head, often producing incomplete 3D portraits where rear regions are poorly represented or entirely missing. As shown in Fig. 1(c), this leads to unrealistic and visually jarring back-head geometry, which significantly reduces the realism and usability of reconstructed avatars.

Although some recent methods attempt to address missing views, they remain fundamentally ineffective in practice. For example, PanoHead [11] generates full-head appearances from a single frontal image using latent feature synthesis. However, it relies on implicit geometry representations and does not produce explicit representations, making it incompatible with pipelines that require controllable and animatable 3D geometry. Other 3DGS-based methods focus on enhancing facial detail and expression control but leave the back-head underconstrained due to the absence of direct supervision [6], [7], [9]. As a result, no existing approach effectively combines plausible back-head completion with explicit, high-fidelity, and fully animatable 3D reconstruction.

Motivated by this gap, we tackle the challenging task of reconstructing complete 3D Gaussian head avatars when backview regions are missing from the input data. Our key idea is to leverage generative models to synthesize the unobserved back-head regions in a way that remains consistent with the subjectâs frontal appearance and hairstyle (Fig. 1(d)). While such generative supervision is plausible in image space, integrating it into a point-based 3DGS framework introduces two main challenges: 1) maintaining visual coherence with the frontal face to avoid identity or hairstyle artifacts, and 2) accurately aligning the pseudo-image content with the 3DGS spatial domain to ensure effective and stable reconstruction. Addressing these challenges requires a novel solution that bridges generative supervision and explicit, animatable 3D Gaussian modeling.

To this end, we propose AvatarBack, a novel framework designed to complete the missing back-head regions in 3D Gaussian head avatars. AvatarBack integrates two core modules: the Subject-specific Generator (SSG) and the Adaptive Spatial Alignment Strategy (ASA). SSG leverages a generative prior to synthesize plausible back-view pseudo-images that are consistent with the subjectâs frontal appearance and hairstyle, providing crucial supervision signals for the 3DGS model. To ensure these synthetic views are effectively utilized, an ASA employs learnable transformation matrices, optimized during training, to align the pseudo-image content accurately with the 3DGS model space. Together, these modules form a unified reconstruction framework. AvatarBack achieves significantly more complete and geometrically consistent 3D head avatars, with notable improvements in the rear regions. Meanwhile, it preserves the explicit, animatable, and high-fidelity properties of Gaussian-based methods.

To the best of our knowledge, this work presents the first approach specifically designed to complete the back-head regions in Gaussian-based 3D head avatars. The proposed AvatarBack framework is a plug-and-play solution that integrates flexibly into state-of-the-art systems such as GaussianAvatars [5] and SurfHead [8]. We establish a comprehensive evaluation protocol combining geometric, photometric, and perceptual metrics, including a GPT-4o-based scoring method for assessing fine-grained, human-aligned visual quality of the reconstructed back-view. Experiments on the NeRSemble [12] and K-hairstyle [13] datasets show that AvatarBack significantly improves back-head reconstruction while preserving frontal fidelity. Moreover, the enhanced avatars remain fully animatable and reliably respond to diverse driving signals, consistently delivering high visual realism across a wide range of motions and expressions.

In summary, our main contributions are as follows:

â¢ We propose AvatarBack, the first framework to complete the back-head regions in Gaussian-based 3D head avatars, combining generative supervision with explicit 3D Gaussian modeling.

â¢ We design a plug-and-play solution with two novel components: SSG for synthesizing consistent back-view, and ASA for accurate integration into the 3DGS space.

â¢ We establish a comprehensive evaluation protocol, including geometric, photometric, and GPT-4o-based perceptual metrics, demonstrating superior reconstruction quality and full animatability on challenging benchmarks.

## II. RELATED WORK

## A. 3D Gaussian Splatting for Head Avatars

The recent 3D Gaussian Splatting (3DGS) [4] offers a new paradigm with state-of-the-art, real-time rendering. However, it represents geometry as an unstructured Gaussian cloud, which lacks the explicit topology for animation and editing. A dominant strategy is to impose structure by anchoring the Gaussians to an explicit mesh, enabling coherent control. For example, SplattingAvatar [6] embeds Gaussians within a mesh using barycentric coordinates, enabling mesh-driven animation. Similarly, GaussianAvatars [5], HERA [14], SVG-Head [15], and MeGA [16] all leverage an underlying surface (explicit or implicit) to provide topological consistency and control, allowing for high-fidelity rendering and coherent deformation. SurfHead [8] further refines this by using 2D Gaussian surfels constrained on a deforming mesh, adeptly handling extreme poses. Another significant research thrust focuses on creating compact and parametrically controllable models. This is often achieved by integrating Gaussians with established parametric head models. For example, GPHM [17] and NPGA [18] leverage expression and pose parameters to directly drive the attributes of the Gaussians, enabling semantic control. Others achieve compactness through learned representations, such as reduced Gaussian blendshapes [19], graph neural networks [20], or efficient latent embeddings in texture space [21] or high-dimensional spaces [22], all aiming for efficient animation and reduced model size. Furthermore, recent efforts have pushed the boundaries of realism and interactivity. A key focus is relightability, where methods such as BecomingLit [23], HRAvatar [24], and LightHeadEd [25] disentangle intrinsic surface properties from illumination, allowing avatars to be realistically rendered under novel lighting conditions. Specialized components, such as dynamic and plausible hair, have also been addressed by dedicated hybrid models [26], [27]. Moreover, the field is moving towards greater user control, enabling direct textural editing [28] or creating generative, interactive avatars [29]. While these methods perform well in supervised areas, they often lack sufficient guidance in sparse regions, such as the back of the 3D avatar head. As a result, recovering complete head geometry remains a challenging task.

## B. 3D-aware GANs

Generative Adversarial Networks (GANs) [30] have achieved major advances in image generation tasks since their introduction. The style-based GAN architecture (Style-GAN) [31], in particular, generated high-quality images by disentangling style and content. This powerful disentanglement paved the way for advanced latent space editing techniques for various applications [32], [33], [34]. Recently, GANs have also been applied to the field of 3D modeling. 3D-GAN [35] pioneered the application of adversarial training to voxel grids. It synthesized 3D shapes from random noise using convolutional voxel generators and discriminators. However, limitations in voxel resolution hindered the representation of fine details. Subsequently, researchers developed GAN frameworks for various 3D representations, such as point clouds [36], [37], explicit meshes [38], [39], and implicit fields [40], [41]. These advancements significantly enhanced generation quality and diversity.

<!-- image-->  
Fig. 2. Overview of the AvatarBack framework. The framework leverages a Subject-specific Generator to produce pseudo back-head views via generator optimization (w, Î). These views are adaptively aligned during reconstruction through an Adaptive Spatial Alignment Strategy module and jointly supervised with real frontal images. This hybrid training strategy enables identity-consistent and complete full-head reconstructions, effectively compensating for missing supervision in invisible regions.

GAN-based methods have also started showing promise in 3D human head reconstruction. EG3D [40] introduced a hybrid architecture combining tri-planes and neural rendering. This approach decouples feature generation from the rendering process. It enables the real-time generation of multi-view consistent, high-resolution images and high-quality geometry. Building on this, PanoHead [11] incorporated facial priors and geometric regularization. This allows the generation of complete 360Â° head models from a single frontal image. It effectively completes the structure and texture of unseen areas, such as the back regions of the 3D avatar head. Furthermore, Pivotal Tuning Inversion (PTI) [42] introduced an innovative GAN inversion technique. It fine-tunes a pre-trained StyleGAN generator to better match an input image. This enables highquality reconstruction and editing. Applying PTI to PanoHead further enhances its reconstruction accuracy.

However, these 3D GAN methods can still produce artifacts, such as view inconsistencies or structural misalignments. This often occurs when handling extreme viewpoints or sparse real image inputs. Moreover, these models typically store 3D information solely as latent variables. This makes it difficult to directly generate drivable 3D human head structures with rich details.

## III. METHOD

## A. Overall Framework

Given a training set of multi-view frontal images $\{ I _ { i } \} _ { i = 1 } ^ { N _ { o r i } }$ for a specific target subject, we follow state-of-the-art methods, namely GaussianAvatars [5] and SurfHead [8], for 3D head reconstruction. The reconstructed 3D head avatar integrates 3D Gaussian Splatting (3DGS) with a parametric FLAME mesh representation. The FLAME model provides the underlying geometry and animatable structure, while the 3DGS captures detailed textures to render realistic appearance. These two representations are coupled by binding the relative positions of Gaussian kernels near the mesh surface.

Formally, we denote a parametric FLAME mesh as M(Ï), where Ï encodes identity, expression, and pose parameters. For each triangle center $p _ { k }$ on the mesh, a Gaussian kernel is attached and parameterized as

$$
\mathcal { G } _ { k } = \{ \mu _ { k } , \Sigma _ { k } , \mathbf { R } _ { k } , \alpha _ { k } , \mathbf { c } _ { k } \} ,\tag{1}
$$

where $\pmb { \mu _ { k } } ~ \in ~ \mathbb { R } ^ { 3 }$ denotes the mean position, $\Sigma _ { k } ~ \in ~ \mathbb { R } ^ { 3 \times 3 }$ represents the anisotropic scale, $\mathbf { R } _ { k } \ \in \ S O ( 3 )$ specifies the rotation, $\alpha _ { k }$ is the opacity, and $\mathbf { c } _ { k }$ is the view-dependent color. During animation, the deformation of triangles driven by pose and expression changes is propagated to each Gaussian kernel $\mathcal { G } _ { k } .$ , ensuring consistent motion and appearance across the 3D avatar.

Following such representation and captured frontal face images, it can produce a photorealistic and temporally stable reconstruction of the visible facial regions in the training views. However, the back-head remains unobserved and unsupervised due to view sparsity, resulting in incomplete geometry and texture. As shown in Fig. 2, we propose AvatarBack, a framework that incorporates synthesized images as pseudosupervisions to complement the reconstruction of unobserved regions, e.g., back-head areas. The framework introduces a 3DGS-aware closed feedback loop between reconstruction and generative synthesis. Instead of treating 3DGS reconstruction and image synthesis as independent stages, AvatarBack couples them so that: 1) an initial 3DGS head model provides geometry- and pose-consistent cues to guide plausible backview synthesis, and 2) the synthesized views are fed back into the 3DGS pipeline as pseudo-supervision to progressively refine the missing regions.

To fulfill both purposes, we propose a Subject-specific Generator (SSG) module for providing pseudo-supervision of the unobserved back-head regions. Additionally, an Adaptive Spatial Alignment (ASA) module aligns the synthesized and captured supervisions with the underlying 3D coordinates. Specifically, SSG (Sec. III-B) uses a hybrid supervision set that combines captured real views and 3DGS-rendered novel views to drive a geometry-conditioned GAN inversion process, producing identity-consistent back-head views even under extreme view sparsity. ASA (Sec. III-C) ensures pixel-accurate integration of the generated images into the 3DGS coordinate space via a learnable geometric transformation.

By enabling feedback between generated pseudo images of unobserved regions and rendered realistic images of the 3DGS head avatars, both components can be mutually reinforced, yielding complete, animatable, and photorealistic 3D head avatars. Particularly, the 3DGS model gains dense supervision from realistic, identity-preserving novel views. The generated pseudo-images can be further improved by integrating explicit 3DGS geometry for appearance-consistent synthesis.

## B. Subject-specific Generator for Unseen Back-head Regions

Given a 3DGS reconstruction of a head avatar from a state-of-the-art approach, $e . g .$ , GaussianAvatars and SurfHead, the 3DGS head accurately reconstructs the frontal region but leaves the back-head largely unconstrained due to the absence of such views in the training set. We utilize a pretrained 3DGAN model, $i . e .$ , PanoHead [11], to synthesize the unseen structure of the missing regions, and the synthetic images of the unobserved regions are employed as pseudo-supervision for the optimization of back-head regions. To meet the fundamental consistency requirements of the synthetic images with the target subject, an inversion of the 3DGAN model is a natural choice. However, directly performing 3DGAN inversion on PanoHead produces artifacts in the back-head regions and does not provide effective supervision for the back-head completion task.

Therefore, instead of directly applying a generic inversion pipeline, which is prone to identity drift under sparse and biased viewpoints, we adopt a feedback loop and introduce a subject-specific generator strategy. This strategy explicitly exploits the initialized 3DGS model to strengthen inversion and, in turn, uses the inversion outputs to enhance the 3DGS reconstruction.

Concretely, the Subject-specific Generator consists of two phases: the Hybrid Subject-specific Generator and Novel

Back-view Synthesis. The Hybrid Subject-specific Generator aims to adapt the hidden weights of PanoHead to a specific subject. Instead of using only limited frontal-view images, we adopt a hybrid multi-view inversion strategy that incorporates both the captured real images and the rendered images from the 3DGS avatars. As the reconstruction process relies on the supervision of both types of images, the inversion process can therefore benefit from the updated reconstructions.

Specifically, the inversion process can be formulated as an optimization task as in Eq. (2):

$$
\begin{array} { r l r } {  { ( \Theta ^ { * } , \mathbf { w } ^ { * } ) = \arg \operatorname* { m i n } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathcal { L } _ { \mathrm { i n v } } \bigl ( G _ { \Theta } ( \mathbf { w } , c _ { i } ) , I _ { i } \bigr ) , } } \\ & { } & { I _ { i } \in \mathcal { T } ^ { h y b r i d } = \{ I _ { i } ^ { \mathrm { o r i } } \} _ { i = 1 } ^ { N _ { \mathrm { o r i } } } \cup \{ I _ { i } ^ { \mathrm { r e n d e r } } \} _ { i = 1 } ^ { N _ { \mathrm { r e n d e r } } } , } \end{array}\tag{2}
$$

where $N = N _ { \mathrm { o r i } } + N _ { \mathrm { r e n d e r } } ,$ , and w and Î are the latent code and the weights adapted to the target subject, respectively. $\left\{ c _ { i } \right\}$ denotes the calibrated camera pose with respect to the given image $I _ { i } , \ { \mathcal { L } } _ { \mathrm { i n v } }$ combines pixel-wise, perceptual, and latent regularization losses, enforcing appearance coherence across both captured and 3DGS-rendered views. The hybrid dataset $\boldsymbol { \mathcal { T } ^ { h y b r i d } }$ mitigates the underconstrained nature of sparse frontal captures, providing diverse geometry-aware cues to the inversion stage.

Different from the vanilla inversion method, which uses a single frontal-view image to obtain the latent code of a generator, we utilize a hybrid multi-view inversion strategy to obtain an appearance-consistent model across different views. For a specific subject, the original captured images $\{ I _ { i } ^ { \mathrm { o r i } } \} _ { i = 1 } ^ { N _ { \mathrm { o r i } } }$ normally include only sparse views of the frontal face region. By incorporating the rendered images from the updated 3DGS head, the inversion process is constrained to maintain identity and multi-view consistency, improving the details of the generated pseudo-images for back-head regions.

After adapting the optimal latent code $\mathbf { w } ^ { * }$ and generator weights $\Theta ^ { * }$ to the target head, we utilize them in the synthesis process of the novel back-head views. Given a specific novel view $c _ { j } ^ { \mathrm { b a c k } }$ of the back-head, we render the pseudo-image $I ^ { \mathrm { b a c k } }$ by forwarding the adapted generator following Eq. (3):

$$
I ^ { \mathrm { b a c k } } = G _ { \Theta ^ { * } } \left( \mathbf { w } ^ { * } , c _ { j } ^ { \mathrm { b a c k } } \right) .\tag{3}
$$

To obtain images of the unobserved back-head regions, $\{ c _ { j } ^ { \mathrm { b a c k } } \}$ are sampled from azimuths $[ 9 0 ^ { \circ } , 2 7 0 ^ { \circ } ]$ . This novel back-view synthesis process can produce a set of synthetic back-head images, and these images maintain strong identity coherence thanks to the participation of the rendered 3DGS images.

Furthermore, the synthesized back-view are refined using the face-oriented super-resolution network [43], which enhances high-frequency details such as hair strands, edges, and contour sharpness. These enhanced images $\{ \tilde { I } _ { j } ^ { \mathrm { b a c k } } \}$ form the high-quality pseudo-supervision set used in subsequent 3DGSbased reconstruction, ensuring that fine details are preserved in the final completed 3D head model.

Through this closed-loop process, the inversion stage is no longer an isolated pre-processing step. Instead, it is explicitly coupled with the 3DGS model, both benefiting from and contributing to it. This design enables reliable, identity-consistent back-view generation even under severe viewpoint sparsity.

## C. Adaptive Spatial Alignment

Precise pixel-level alignment between rendered 3DGS head views and generated pseudo-images is essential for supervising the back-head regions. For instance, if a generated view depicts a back-head hair part at a specific image location, the corresponding rendering from the 3DGS head model must reproduce that detail at the same position and scale. Otherwise, the supervision may introduce inconsistencies or errors. However, due to differences in the coordinate systems of the Gaussian avatar and the generative model, their rendered images are misaligned.

To resolve this, we introduce an Adaptive Spatial Alignment (ASA) module, which learns a transformation matrix $\mathrm { ~ \bf ~ T ~ } \in$ R4Ã4 that aligns the coordinate systems of the two models. The transformation is applied to the FLAME mesh vertices V(Ï), thereby indirectly guiding the 3DGS head to align with the generative model coordinate. This matrix is optimized jointly during training via the following objective:

$$
\begin{array} { r l } & { \mathbf T ^ { * } = \arg \underset { \mathbf T } { \operatorname* { m i n } } \sum _ { j = 1 } ^ { M } \mathcal L \left( \mathcal R ( \mathcal G ( \mathbf T \cdot \mathcal V ( \phi ) , \ c _ { j } ^ { \mathrm { b a c k } } ) , \ I _ { j } ^ { \mathrm { b a c k } } \right) } \\ & { \quad \quad + \mathcal L _ { \mathrm { F L A M E } } ( \phi ) , } \end{array}\tag{4}
$$

where $\mathcal { G }$ is the set of 3D Gaussians, $c _ { j } ^ { \mathrm { b a c k } }$ denotes a back-view camera pose, and $I _ { j } ^ { \mathrm { b a c k } }$ is the pseudo back-head image. The function R denotes the differentiable Gaussian rendering.

The loss $\mathcal { L }$ enforces photometric consistency between rendered and pseudo images, while LFLAME regularizes the alignment to remain faithful to the original FLAME geometry:

$$
\mathcal { L } _ { \mathrm { F L A M E } } ( \phi ) = \lambda _ { \mathrm { f l a m e } } \cdot \left. \phi - \phi _ { \mathrm { o r i g } } \right. _ { 2 } ^ { 2 } ,\tag{5}
$$

where $\phi$ and $\phi _ { \mathrm { o r i g } }$ denote the current and initial FLAME parameters, and $\lambda _ { \mathrm { f l a m e } }$ controls the regularization strength.

Directly optimizing the entire transformation matrix T can easily introduce undesired effects beyond intended scaling, rotation, and translation. It may also lead to coupling between different geometric components, making independent control difficult. Therefore, we decompose T into a scale matrix S, a rotation matrix R, and a translation vector t, which are optimized separately. Formally, the transformation matrix T can be expressed as:

$$
\mathbf { T } = \left[ \begin{array} { l l } { \mathbf { R } \mathbf { S } } & { \mathbf { t } } \\ { \mathbf { 0 } ^ { \top } } & { 1 } \end{array} \right] ,\tag{6}
$$

where $\mathbf { R } \in \mathbb { R } ^ { 3 \times 3 } , \mathbf { S } \in \mathbb { R } ^ { 3 \times 3 } , \mathbf { t } \in \mathbb { R } ^ { 3 }$ , and $\mathbf { 0 } \in \mathbb { R } ^ { 3 }$

To facilitate optimization, we parameterize the spatial transformation using low-dimensional vectors. Specifically, the transformation consists of a learnable scale vector $\textbf { s } \in \ \mathbb { R } ^ { 3 }$ a rotation vector $\textbf { r } \in \ \mathbb { R } ^ { 3 }$ , and a translation vector $\textbf { t } \in \mathbb { R } ^ { 3 }$ The scale and rotation vectors are converted into matrix form during the forward pass, while the translation vector is directly applied without further transformation.

The scale matrix S is defined as:

$$
\mathbf { S } = \mathrm { d i a g } ( \mathbf { s } ) = \left[ \begin{array} { c c c } { s _ { 1 } } & { 0 } & { 0 } \\ { 0 } & { s _ { 2 } } & { 0 } \\ { 0 } & { 0 } & { s _ { 3 } } \end{array} \right] ,\tag{7}
$$

and the rotation matrix R is computed using Rodriguesâ rotation formula:

$$
\mathbf { R } = \mathbf { I } + { \frac { \sin \theta } { \theta } } [ \mathbf { r } ] _ { \times } + { \frac { 1 - \cos \theta } { \theta ^ { 2 } } } [ \mathbf { r } ] _ { \times } ^ { 2 } ,\tag{8}
$$

where $\theta = \| \mathbf { r } \|$ is the magnitude of the rotation vector, and $[ \mathbf { r } ] _ { \times } \in \mathbb { R } ^ { 3 \times 3 }$ is the corresponding skew-symmetric matrix:

$$
[ \mathbf { r } ] _ { \times } = \left[ { \begin{array} { c c c } { 0 } & { - r _ { 3 } } & { r _ { 2 } } \\ { r _ { 3 } } & { 0 } & { - r _ { 1 } } \\ { - r _ { 2 } } & { r _ { 1 } } & { 0 } \end{array} } \right] .\tag{9}
$$

Here, I denotes the $3 \times 3$ identity matrix. The rotation matrix R is thus obtained by exponentiating $[ \mathbf { r } ] _ { \times } ,$ , ensuring a smooth and stable representation during optimization.

## IV. EXPERIMENTS

## A. Implementation Details

As a plug-and-play algorithm, we verify the proposed method by integrating it with state-of-the-art approaches, $e . g .$ GaussianAvatars [5] and SurfHead [8]. We augment the training process with supervision from pseudo-images of the backhead. These pseudo-images are integrated into the pipeline alongside an ASA module, which is co-optimized with the 3DGS head avatar. This unified framework allows for joint optimization using a combined objective over real and spatiallyaligned pseudo-images. For training, real multi-view images of the frontal face are utilized at a resolution of $8 0 2 \times 5 5 0$ pixels, while pseudo-images generated by our proxy module are 512Ã512 pixels. The entire training process employs the Adam optimizer, with the loss weight for pseudo-real images set to $\lambda ~ = ~ 0 . 0 1$ . The weight of the FLAME parameters constraint is $\lambda _ { \mathrm { f l a m e } } ~ = ~ 0 . 5$ to enforce geometric consistency. The initial learning rate for scale factor s, rotation matrix $\mathbf { r } ,$ translation vector t is 0.005, following a cosine annealing schedule. All experiments are conducted on a workstation equipped with four NVIDIA RTX 3090 Ti GPUs, each with 24 GB of memory.

## B. Experimental Setup

Datasets. We employ two publicly available datasets, NeRSemble [12] and K-hairstyle [13], to comprehensively evaluate the proposed method on both frontal and back-head reconstruction tasks.

â¢ NeRSemble [12] is a widely used benchmark for frontal face reconstruction. It provides high-quality multi-view video sequences specifically designed for 3D avatars. For each subject, a total of 11 video sequences are recorded, including four emotion (EMO) sequences, six expression (EXP) sequences, and one free performance (FREE) sequence. Each video frame captures 16 synchronized camera views distributed across approximately 120 degrees in the front. The evaluation follows two protocols: 1) novelview synthesis, where head poses and expressions from training sequences are used to render the subject from unseen camera viewpoints. 2) self-reenactment, where a held-out sequence with unseen poses and expressions is

<!-- image-->  
SurfHead

Fig. 3. Qualitative comparison of back-head reconstructions. The first column displays ground-truth images, followed by five rendered views under azimuth angles from 60â¦ to 180â¦. The proposed AvatarBack method can be integrated into existing models, e.g., GaussianAvatars and SurfHead, to accurately reconstruct the unseen back and side head regions.

SurfHead + AvatarBack

used to drive the avatar, and rendering is performed across all 16 camera views.

â¢ K-hairstyle [13] provides multi-view image sequences of multiple subjects, captured at 6-degree intervals across a full 360Â° azimuth range. We utilize the back-view images (within 90â¦â270â¦) as pseudo-ground-truth references to evaluate our synthesized results, addressing the lack of real supervisory signals in these regions.

Criteria. SurfHead+AvatarBack follows the evaluation setup of SurfHead [8], where nine reconstructed avatars are selected for evaluation. Among these sequences, EMO-1 is used for self-reenactment evaluation, while the remaining eight sequences, excluding the FREE sequence, are used for training. For training, 15 out of the 16 camera views are used, excluding the 8th view. The 8th view, which corresponds to the central frontal camera, is reserved as the novel-view synthesis evaluation. GaussianAvatars+AvatarBack follows the evaluation protocol of GaussianAvatars [5], using the same number of subjects and the same partitioning strategy. The specific subject IDs used in both protocols are provided in the

TABLE I  
BACK-HEAD REGION EVALUATION. QUANTITATIVE COMPARISON OF BASELINES AND AVATARBACK ENHANCEMENTS. $\mathbf { D _ { G A } }$ AND $\mathbf { D _ { S F } }$ DENOTE THE SUBJECT SPLITS FOLLOWING THE GAUSSIANAVATARS AND SURFHEAD PROTOCOLS, RESPECTIVELY.
<table><tr><td>Metric</td><td colspan="3"> $\mathbf { D _ { G A } }$ </td><td colspan="2"> $\mathbf { D _ { S F } }$ </td></tr><tr><td></td><td>PanoHead [11]</td><td>GaussianAvatars [5]</td><td>GaussianAvatars+AvatarBack</td><td>SurfHead [8]</td><td>SurfHead+AvatarBack</td></tr><tr><td>Clarityâ</td><td>6.583</td><td>6.778</td><td>8.444</td><td>7.014</td><td>8.556</td></tr><tr><td>Structural Integrityâ</td><td>7.833</td><td>6.556</td><td>8.306</td><td>6.653</td><td>8.361</td></tr><tr><td>Texture Qualityâ</td><td>6.319</td><td>6.000</td><td>7.806</td><td>6.181</td><td>8.125</td></tr><tr><td>Color &amp; Lighting Consistencyâ</td><td>7.056</td><td>6.306</td><td>8.181</td><td>7.125</td><td>8.639</td></tr><tr><td>Overall Perceptionâ</td><td>6.917</td><td>6.375</td><td>8.278</td><td>6.653</td><td>8.444</td></tr><tr><td>Overall Scoreâ</td><td>6.94</td><td>6.40</td><td>8.20</td><td>6.73</td><td>8.43</td></tr></table>

TABLE II

BACK-HEAD REGION EVALUATION. QUANTITATIVE COMPARISON OF BACK-HEAD RECONSTRUCTION QUALITY ON THE K-HAIRSTYLE DATASET.
<table><tr><td>Model</td><td>FIDâ</td><td>KIDâ</td></tr><tr><td>GaussianAvatars [5]</td><td>218.34</td><td>0.202</td></tr><tr><td>GaussianAvatars+AvatarBack</td><td>146.73</td><td>0.120</td></tr><tr><td>SurfHead [8]</td><td>232.46</td><td>0.227</td></tr><tr><td>SurfHead+AvatarBack</td><td>165.06</td><td>0.146</td></tr></table>

Supplementary Material.

To comprehensively assess the quality of 3D avatar reconstruction, we employ different evaluation protocols for the frontal and back-head regions.

â¢ Frontal Region Evaluation: we adopt Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS) to respectively assess pixel-level accuracy, structural consistency, and perceptual similarity. These metrics are consistent with the evaluation protocols of GaussianAvatars [5] and SurfHead [8], facilitating fair comparison. Among them, higher PSNR and SSIM values indicate better reconstruction quality, while lower LPIPS values correspond to higher perceptual similarity.

â¢ Distribution-Level Evaluation Metrics: We evaluate the similarity between the distributions of reconstructed and reference images using Frechet Inception Distance (FID) and Kernel Inception Distance (KID), computed from Inception network [44] features. Lower values indicate better alignment. The evaluation compares synthesized images at azimuth angles from $9 0 Â°$ to 270â¦ across all time steps of nine subjects with corresponding reference images from the K-hairstyle dataset.

â¢ GPT-4o-based Perceptual Scoring: To capture subtle perceptual differences aligned with human visual judgment, we adopt a GPT-4o-based [45] perceptual scoring system. The evaluation focuses on three representative back-head viewpoints: the exact rear view (180â¦), the left-back view (135â¦), and the right-back view (225â¦). Each view is assigned a weight reflecting its contribution to overall reconstruction quality: 50% for the rear view and 25% for each side-back view. The final perceptual score S is computed as a weighted sum:

$$
S = 0 . 5 \times S _ { 1 8 0 ^ { \circ } } + 0 . 2 5 \times S _ { 1 3 5 ^ { \circ } } + 0 . 2 5 \times S _ { 2 2 5 ^ { \circ } } ,\tag{10}
$$

where $S _ { \theta }$ denotes the GPT-4o perceptual score at azimuth angle Î¸. Each per-view score $S _ { \theta }$ is computed based on five equally weighted criteria: clarity, structural integrity, texture quality, color and lighting consistency, and overall perception. The detailed definitions of these criteria are provided in the Supplementary Material.

TABLE III  
GAUSSIANAVATARS. BOTTOM: RESULTS FOLLOWING THE PROTOCOL OF SURFHEAD.
<table><tr><td rowspan="2">Method</td><td colspan="2">Novel-View Synthesis</td><td colspan="2">Self-Reenactment</td></tr><tr><td>|PSNRâ SSIMâLPIPSâ|PSNRâ SSIMâ LPIPSâ</td><td></td><td></td><td></td></tr><tr><td>AvatarMAV [46]</td><td>29.5 0.913</td><td>0.152</td><td>24.3</td><td>0.887 0.168</td></tr><tr><td>PointAvatar [47]</td><td>25.8 0.893</td><td>0.097</td><td>23.4</td><td>0.884 0.104</td></tr><tr><td>INSTA [48]</td><td>26.7 0.899</td><td>0.122</td><td>26.3</td><td>0.906 0.110</td></tr><tr><td>GaussianAvatars [5]</td><td>31.6 0.938</td><td>0.065</td><td>26.0</td><td>0.910 0.076</td></tr><tr><td>GaussianAvatars+AvatarBack</td><td>31.8 0.939</td><td>0.064</td><td>26.1</td><td>0.912 0.075</td></tr><tr><td>PointAvatar [47]</td><td>20.56 0.844</td><td>0.206</td><td>20.59</td><td>0.854 0.190</td></tr><tr><td>Flare [49]</td><td>21.91 0.814</td><td>0.228</td><td>21.11</td><td>0.802 0.227</td></tr><tr><td>SplattingAvatars [6]</td><td>23.68 0.858</td><td>0.232</td><td>20.25</td><td>0.828 0.265</td></tr><tr><td>GaussianAvatars [5]</td><td>30.29 0.934</td><td>0.067</td><td>23.43</td><td>0.891 0.093</td></tr><tr><td>SurfHead [8]</td><td>30.07 0.934</td><td>0.079</td><td>23.53</td><td>0.892 0.103</td></tr><tr><td>SurfHead+AvatarBack</td><td>32.75 0.940</td><td>0.069</td><td>26.53</td><td>0.907 0.089</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

<!-- image-->  
Fig. 4. Qualitative comparison of reconstructed frontal-views. The proposed AvatarBack method consistently enhances fine-grained details, leading to more faithful frontal reconstructions.

Baselines. We validate the proposed AvatarBack framework by integrating it with two state-of-the-art 3D head reconstruction methods, i.e., GaussianAvatars [5] and Surf-Head [8], yielding GaussianAvatars+AvatarBack and Surf-Head+AvatarBack. For frontal-view evaluations, we quantitatively compare the proposed method with the state-of-theart facial reconstruction methods. For back-head views, we propose to evaluate across multiple criteria.

<!-- image-->  
Fig. 5. SSG improves back-head geometry, while Lflame preserves frontal facial structure and symmetry. The full model integrates both components to produce anatomically plausible and view-consistent 3D head reconstructions.

## C. Quantitative Comparison

Back-head Region. We assess the quality of the backhead regions with the cutting-edge GPT-4o model as shown in Tab. I. It shows that AvatarBack obtains a profound enhancement in perceived visual quality. The overall score for GaussianAvatars increases from 6.40 to 8.20, and for SurfHead, it rises from 6.73 to 8.43. The improvements are attributed to the enhancement of clarity, structural integrity, and texture quality. PanoHead achieves a score of 6.94, higher than GaussianAvatars (6.40) but lower than the AvatarBack-enhanced methods (8.20). This shows that despite its multi-view generation capability, it still struggles to produce consistent and realistic back-head textures, underscoring the necessity of our framework. Tab. II further shows the evaluation results of the distribution statistics regarding the reconstructed back-head regions, i.e., FID and KID. Our framework yields remarkable improvements for both integrated methods. Specifically, GaussianAvatars+AvatarBack demonstrates a notable FID reduction of 71.61, and SurfHead+AvatarBack achieves a substantial decrease of 67.40. These consistent and considerable improvements in FID and KID metrics underscore that the synthesized textures are much closer to the distribution of real-world images, thereby enhancing visual fidelity and realism. This qualitative assessment confirms that our framework effectively elevates the visual quality of back-head reconstructions.

Frontal Region. Tab. III presents a comprehensive quantitative comparison of our proposed AvatarBack framework against state-of-the-art 3D head reconstruction methods, rigorously assessing the quality of reconstructed frontal face regions. The results show that the proposed AvatarBack approach achieves competitive accuracy on both novel-view synthesis and self-reenactment tasks under multiple evaluation metrics. Notably, SurfHead+AvatarBack delivers a substantial PSNR gain of 2.68 in novel-view synthesis and an even more significant increase of 2.99 in self-reenactment. In the lower half of Tab. III, although GaussianAvatars records a marginally better LPIPS in the split following SurfHead, our approach exhibits the most balanced and comprehensively superior performance across all benchmarks. These results confirm that our AvatarBack method not only effectively completes the unseen back-head regions but also maintains competitive reconstruction quality for the frontal face.

<!-- image-->  
Fig. 6. Ablation comparison of mesh quality and rendered images under four canonical viewpoints. SSG enhances mesh completeness and fidelity, while AvatarBack enforces geometric regularity and recovers plausible back-head structures under reliable supervision.

## D. Qualitative Comparison

Back-head Region. Fig. 3 provides a qualitative comparison of back-head reconstructions, contrasting the baseline GaussianAvatars and SurfHead models against their AvatarBack-enhanced counterparts. Each row visualizes outputs at challenging, unseen viewpoints ranging from $6 0 ^ { \circ }$ to a full $1 8 0 ^ { \circ }$ . The baseline methods, trained solely on frontalfacing data, exhibit a progressive decline in reconstruction quality as the viewpoint shifts toward the back. Around $9 0 Â°$ noticeable hollow regions begin to appear in the reconstructed geometry. At 180 , corresponding to the direct back view, the reconstructions collapse entirely, manifesting as disorganized floating artifacts and incoherent structures, rather than resembling a plausible head or hair configuration. In stark contrast, the models enhanced with AvatarBack seamlessly address these deficiencies. They successfully fill the geometric void with dense, plausible hair volume that maintains the subjectâs overall head shape and hairstyle. Even at a full $1 8 0 ^ { \circ }$ posterior view, the models render a solid and continuous surface. The synthesized texture is not only internally coherent but also integrates naturally with the visible hair from the original views, creating a complete and visually convincing 360-degree appearance. These results suggest that AvatarBack significantly enhances the visual realism and structural integrity of reconstructions under unseen back-view conditions.

Frontal Region. Fig. 4 presents frontal-view comparisons based on GaussianAvatars [5] and SurfHead [8]. In the first row, AvatarBack contributes to more defined tooth structures, particularly in (b) and (d), compared to the baseline outputs (a) and (c). The second row highlights improved reconstruction of earrings, with clearer and sharper results in the enhanced versions. The third row shows better definition around the mouth corners, especially when comparing (b) to (a). In the fourth row, hair details are more faithfully reproduced with AvatarBack. These results indicate that our AvatarBack not only significantly enhances the quality of local facial features but also ensures consistent improvements, thereby elevating overall structural and perceptual fidelity.

## E. Ablation Studies

We evaluate the two core components of AvatarBack: the subject-specific generator (SSG) and FLAME-based regularization ${ \mathcal { L } } _ { \mathrm { f l a m e } }$ . SSG generates reliable back-head geometry, mitigating semantic leakage from single-view training, while ${ \mathcal { L } } _ { \mathrm { f l a m e } }$ preserves frontal facial structure and symmetry. Removing either module leads to distorted or incomplete reconstructions, whereas the full model produces anatomically plausible and view-consistent 3D heads.

Analysis of Mesh Quality. We qualitatively compare reconstructed meshes and rendered images across four canonical viewpoints (0 , 90 , 180 , 270 ) as shown in Fig. 6. PanoHead exhibits severe surface irregularities, profile distortions, and back-view artifacts, including misreconstructed facial features. SSG improves mesh completeness and fidelity across views but still lacks structured regularity. AvatarBack w/o SSG introduces some geometric regularity, yet inaccurate back-head supervision results in implausible geometry. Our full Avatar-Back pipeline, guided by SSG, produces smooth surfaces, realistic facial structures, and consistent head geometry across all viewpoints.

Additional ablation results, including the effects of the FLAME-based regularization term, are provided in the Supplementary Material.

## V. CONCLUSION

In this paper, we proposed AvatarBack, a plug-and-play framework that leverages generative supervision and adaptive spatial alignment to complete the missing back-head regions in Gaussian-based 3D head avatars. Our method achieves geometrically consistent, photorealistic, and fully animatable reconstructions, while surpassing existing approaches in both fidelity and completeness. This work paves the way toward more complete, realistic, and controllable 3D human head modeling.

## REFERENCES

[1] V. Blanz and T. Vetter, âA morphable model for the synthesis of 3d faces,â in Seminal Graphics Papers: Pushing the Boundaries, 2023, vol. 2, pp. 157â164.

[2] T. Li, T. Bolkart, M. J. Black, H. Li, and J. Romero, âLearning a model of facial shape and expression from 4d scans,â ACM Transactions on Graphics, vol. 36, no. 6, 2017.

[3] Y. Feng, H. Feng, M. J. Black, and T. Bolkart, âLearning an animatable detailed 3d face model from in-the-wild images,â ACM Transactions on Graphics, vol. 40, no. 4, 2021.

[4] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, 2023.

[5] S. Qian, T. Kirschstein, L. Schoneveld, D. Davoli, S. Giebenhain, and M. Niessner, âGaussianavatars: Photorealistic head avatars with rigged 3d gaussians,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 299â20 309.

[6] Z. Shao, Z. Wang, Z. Li, D. Wang, X. Lin, Y. Zhang, M. Fan, and Z. Wang, âSplattingavatar: Realistic real-time human avatars with mesh-embedded gaussian splatting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1606â1616.

[7] J. Xiang, X. Gao, Y. Guo, and J. Zhang, âFlashavatar: High-fidelity head avatar with efficient gaussian embedding,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1802â1812.

[8] J. Lee, T. Kang, M. Buehler, M.-J. Kim, S. Hwang, J. Hyung, H. Jang, and J. Choo, âSurfhead: Affine rig blending for geometrically accurate 2d gaussian surfel head avatars,â in The Thirteenth International Conference on Learning Representations, 2025.

[9] Z. Zhao, Z. Bao, Q. Li, G. Qiu, and K. Liu, âPsavatar: A point-based shape model for real-time head avatar animation with 3d gaussian splatting,â 2024. [Online]. Available: https://arxiv.org/abs/2401.12900

[10] Y. Zhong, X. Zhang, L. Liu, Y. Zhao, and Y. Wei, âAvatarmakeup: Realistic makeup transfer for 3d animatable head avatars,â 2025. [Online]. Available: https://arxiv.org/abs/2507.02419

[11] S. An, H. Xu, Y. Shi, G. Song, U. Y. Ogras, and L. Luo, âPanohead: Geometry-aware 3d full-head synthesis in 360Â°,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 20 950â20 959.

[12] T. Kirschstein, S. Qian, S. Giebenhain, T. Walter, and M. NieÃner, âNersemble: Multi-view radiance field reconstruction of human heads,â ACM Transactions on Graphics, vol. 42, no. 4, 2023.

[13] T. Kim, C. Chung, S. Park, G. Gu, K. Nam, W. Choe, J. Lee, and J. Choo, âK-hairstyle: A large-scale korean hairstyle dataset for virtual hair editing and hairstyle classification,â in IEEE International Conference on Image Processing, 2021, pp. 1299â1303.

[14] H. Cai, Y. Xiao, X. Wang, J. Li, Y. Guo, Y. Fan, S. Gao, and J. Zhang, âHera: Hybrid explicit representation for ultra-realistic head avatars,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2025, pp. 260â270.

[15] H. Sun, C. Wang, T.-X. Xu, J. Huang, D. Kang, C. Guo, and S.-H. Zhang, âSvg-head: Hybrid surface-volumetric gaussians for high-fidelity head reconstruction and real-time editing,â 2025. [Online]. Available: https://arxiv.org/abs/2508.09597

[16] C. Wang, D. Kang, H. Sun, S. Qian, Z. Wang, L. Bao, and S.-H. Zhang, âMega: Hybrid mesh-gaussian head avatar for high-fidelity rendering and head editing,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 26 274â26 284.

[17] Y. Xu, Z. Su, Q. Wu, and Y. Liu, âGphm: Gaussian parametric head model for monocular head avatar reconstruction,â 2024. [Online]. Available: https://arxiv.org/abs/2407.15070

[18] S. Giebenhain, T. Kirschstein, M. Runz, L. Agapito, and M. NieÃner, Â¨ âNpga: Neural parametric gaussian avatars,â in SIGGRAPH Asia 2024 Conference, 2024.

[19] L. Li, Y. Li, Y. Weng, Y. Zheng, and K. Zhou, âRgbavatar: Reduced gaussian blendshapes for online modeling of head avatars,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 10 747â10 757.

[20] X. Wei, P. Chen, M. Lu, H. Chen, and F. Tian, âGraphavatar: compact head avatars with gnn-generated 3d gaussians,â in Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence, 2025.

[21] G. Li, P. Gotardo, T. Bolkart, S. Garbin, K. Sarkar, A. Meka, A. Lattas, and T. Beeler, âTega: Texture space gaussian avatars for high-resolution dynamic head modeling,â in Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference, 2025.

[22] G. Serifi and M. C. Buhler, âHypergaussians: High-dimensional Â¨ gaussian splatting for high-fidelity animatable face avatars,â 2025. [Online]. Available: https://arxiv.org/abs/2507.02803

[23] J. Schmidt, S. Giebenhain, and M. Niessner, âBecominglit: Relightable gaussian avatars with hybrid neural shading,â 2025. [Online]. Available: https://arxiv.org/abs/2506.06271

[24] D. Zhang, Y. Liu, L. Lin, Y. Zhu, K. Chen, M. Qin, Y. Li, and H. Wang, âHravatar: High-quality and relightable gaussian head avatar,â in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 26 285â26 296.

[25] P. Manu, A. Srivastava, A. Raj, V. Jampani, A. Sharma, and P. J. Narayanan, âLightheaded: Relightable & editable head avatars from a smartphone,â 2025. [Online]. Available: https://arxiv.org/abs/2504.09671

[26] Z. Liao, Y. Xu, Z. Li, Q. Li, B. Zhou, R. Bai, D. Xu, H. Zhang, and Y. Liu, âHhavatar: Gaussian head avatar with dynamic hairs,â 2024. [Online]. Available: https://arxiv.org/abs/2312.03029

[27] Y. Zheng, M. Chai, D. Vicini, Y. Zhou, Y. Xu, L. Guibas, G. Wetzstein, and T. Beeler, âGroomlight: Hybrid inverse rendering for relightable human hair appearance modeling,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 16 040â16 050.

[28] J. Zhang, Z. Wu, Z. Liang, Y. Gong, D. Hu, Y. Yao, X. Cao, and H. Zhu, âFate: Full-head gaussian avatar with textural editing from monocular video,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 5535â5545.

[29] Z. Yu, T. Li, J. Sun, O. Shapira, S. Park, M. Stengel, M. Chan, X. Li, W. Wang, K. Nagano, and S. De Mello, âGaia: Generative animatable interactive avatars with expression-conditioned gaussians,â in Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference, 2025.

[30] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, âGenerative adversarial networks,â 2014. [Online]. Available: https://arxiv.org/abs/1406.2661

[31] T. Karras, S. Laine, and T. Aila, âA style-based generator architecture for generative adversarial networks,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 4401â4410.

[32] O. Patsouras, A. Tefas, N. Nikolaidis, and I. Pitas, âStyleclip: Text-driven manipulation of stylegan imagery,â in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 2085â2094.

[33] M. W. Khan, M. Jia, X. Zhang, E. Yu, C. Shan, and K. Musial-Gabrys, âInstaface: Identity-preserving facial editing with single image inference,â 2025. [Online]. Available: https://arxiv.org/abs/2502.20577

[34] Y. Zhong, X. Zhang, Y. Zhao, and Y. Wei, âDreamlcm: Towards high quality text-to-3d generation via latent consistency model,â in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 1731â1740.

[35] J. Wu, C. Zhang, T. Xue, W. T. Freeman, and J. B. Tenenbaum, âLearning a probabilistic latent space of object shapes via 3d generativeadversarial modeling,â in Proceedings of the International Conference on Neural Information Processing Systems, 2016, p. 82â90.

[36] R. Li, X. Li, K.-H. Hui, and C.-W. Fu, âSp-gan: sphere-guided 3d shape generation and manipulation,â ACM Transactions on Graphics, vol. 40, no. 4, 2021.

[37] Z. Yang, Y. Chen, X. Zheng, Y. Chang, and X. Li, âConditional gan for point cloud generation,â in 16th Asian Conference on Computer Vision, 2022, p. 117â133.

[38] A. Pemasiri, K. Nguyen, S. Sridharan, and C. Fookes, âAccurate 3d hand mesh recovery from a single rgb image,â Scientific Reports, vol. 12, 2022.

[39] J. Gao, T. Shen, Z. Wang, W. Chen, K. Yin, D. Li, O. Litany, Z. Gojcic, and S. Fidler, âGet3d: A generative model of high quality 3d textured shapes learned from images,â in Proceedings of the International Conference on Neural Information Processing Systems, 2022, pp. 31 841 â 31 854.

[40] E. R. Chan, C. Z. Lin, M. A. Chan, K. Nagano, B. Pan, S. de Mello, O. Gallo, L. Guibas, J. Tremblay, S. Khamis, T. Karras, and G. Wetzstein, âEfficient geometry-aware 3d generative adversarial networks,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16 102â16 112.

[41] C. Sun, Y. Liu, J. Han, and S. Gould, âNerfeditor: Differentiable style decomposition for 3d scene editing,â in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 7306â 7315.

[42] D. Roich, R. Mokady, A. H. Bermano, and D. Cohen-Or, âPivotal tuning for latent-based editing of real images,â ACM Transactions on Graphics, vol. 42, no. 1, 2022.

[43] J. He, W. Shi, K. Chen, L. Fu, and C. Dong, âGcfsr: a generative and controllable face super resolution method without facial and gan priors,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 1879â1888.

[44] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, âGoing deeper with convolutions,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 1â9.

[45] OpenAI, âGpt-4o system card,â 2024. [Online]. Available: https: //arxiv.org/abs/2410.21276

[46] Y. Xu, L. Wang, X. Zhao, H. Zhang, and Y. Liu, âAvatarmav: Fast 3d head avatar reconstruction using motion-aware neural voxels,â in ACM SIGGRAPH 2023 Conference Proceedings, 2023.

[47] Y. Zheng, W. Yifan, G. Wetzstein, M. J. Black, and O. Hilliges, âPointavatar: Deformable point-based head avatars from videos,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 21 057â21 067.

[48] W. Zielonka, T. Bolkart, and J. Thies, âInstant volumetric head avatars,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 4574â4584.

[49] S. Bharadwaj, Y. Zheng, O. Hilliges, M. J. Black, and V. F. Abrevaya, âFlare: Fast learning of animatable and relightable mesh avatars,â ACM Transactions on Graphics, vol. 42, no. 6, 2023.

# Supplementary Material for AvatarBack: Back-Head Generation for Complete 3D Avatars from Front-View Images

Shiqi Xin , Xiaolin Zhang , Yanbin Liu , Peng Zhang , Caifeng Shan

## I. ADDITIONAL EXPERIMENTAL DETAILS

Evaluation Protocol Details. Table S1 lists the specific subject IDs used in both protocols on the NeRSemble dataset. As described in the main paper, $\mathbf { D _ { G A } }$ and $\mathbf { D _ { S F } }$ denote the subject selections following the protocols of GaussianAvatars [1] and SurfHead [2], respectively.

TABLE S1  
SUBJECT IDS USED FOR EVALUATION ON THE NERSEMBLE DATASET.
<table><tr><td> $\mathbf { D _ { G A } }$ </td><td>074</td><td>104</td><td>218</td><td>253</td><td>264</td><td>302</td><td>304</td><td>306</td><td>460</td></tr><tr><td> $\mathbf { D _ { S F } }$ </td><td>074</td><td>140</td><td>175</td><td>210</td><td>253</td><td>264</td><td>302</td><td>304</td><td>306</td></tr></table>

Details of GPT-4o-based Perceptual Scoring. Table S2 summarizes the five criteria used by GPT-4o to assess the quality of back-of-head reconstruction. Each criterion contributes equally (20%) to the per-view score $S _ { \theta } .$ , which is then aggregated into the overall perceptual score S as described in the main paper.

TABLE S2  
EVALUATION CRITERIA USED BY GPT-4O TO ASSESS THE QUALITY OF BACK-HEAD RECONSTRUCTION.
<table><tr><td>Criterion</td><td>Evaluation Description</td></tr><tr><td>Clarity</td><td>Are the reconstructed details  $\textit { ( e . g . }$  ,hair, skin, contours) sharp and clear? Is the visual clarity comparable to the frontal view?</td></tr><tr><td>Structural Integrity</td><td>Does the back-of-head structure align with the subject&#x27;s frontal structure? Are proportions and symmetry preserved?</td></tr><tr><td>Texture Quality</td><td>Are hair and skin textures realistic and detailed? Do they match the visual style of the input image?</td></tr><tr><td>Color &amp; Lighting Consistency</td><td>Are skin and hair color consistent with the front image? Is the lighting coherent with the frontal view?</td></tr><tr><td>Overall Perception</td><td>Does the image appear realistic and nat- ural overall? Is it visually consistent with the front-view reference?</td></tr></table>

## II. ADDITIONAL ABLATION STUDIES

Subject-specific Generator Module. Notably, as illustrated in Fig. S1(a), when the generator module is trained using only a single frontal image, the reconstructed back-head often contains implausible artifacts, including front-facing features such as eyes, eyebrows, and nose. This indicates an over-reliance on visible cues, resulting in semantic leakage and poor generalization to occluded areas. When such flawed pseudo images are used to guide GaussianAvatars+AvatarBack, the resulting 3DGS head avatar exhibit incomplete geometry and unrealistic back-head appearance. In contrast, our SSG generates more reliable and view-consistent geometry shown in Fig. S1(b), which effectively supports AvatarBack in producing accurate and coherent head reconstructions.

<!-- image-->  
Fig. S1. Qualitative comparison of back-head reconstruction using PanoHead with single-view input versus our multi-view training strategy. (a) Results using only one frontal image. (b) Results using multiple frontal views (Ours). Our approach produces more accurate geometry and detailed textures in the back-head region.

<!-- image-->

<!-- image-->  
(a)

<!-- image-->  
Fig. S2. Effect of FLAME-based regularization. From left to right: Target, (a) mesh without ${ \mathcal { L } } _ { \mathrm { f l a m e } } .$ , and (b) mesh with ${ \mathcal { L } } _ { \mathrm { f l a m e } }$ . The regularized mesh (b) exhibits improved frontal structure and better alignment with true facial geometry.

FLAME-based Regularization. We evaluate the effect of the FLAME-based regularization term ${ \mathcal { L } } _ { \mathrm { f l a m e } }$ , which constrains mesh deformation to ensure plausible and stable facial geometry. As shown in Fig. S2, (a) the mesh without this regularization exhibits obvious distortion in the frontal region, such as unnatural facial contours and collapsed structures. In contrast, (b) the mesh with ${ \mathcal { L } } _ { \mathrm { f l a m e } }$ more closely resembles the ground-truth results, preserving facial symmetry and anatomical plausibility. This effect is also reflected in the full reconstruction pipeline, where omitting ${ \mathcal { L } } _ { \mathrm { f l a m e } }$ leads to visibly distorted features in the frontal view. Compared to the full model, these results highlight the importance of the FLAMEbased prior in enforcing geometric realism and preventing frontal-view collapse.

## REFERENCES

[1] S. Qian, T. Kirschstein, L. Schoneveld, D. Davoli, S. Giebenhain, and M. Niessner, âGaussianavatars: Photorealistic head avatars with rigged 3d gaussians,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20 299â20 309.

[2] J. Lee, T. Kang, M. Buehler, M.-J. Kim, S. Hwang, J. Hyung, H. Jang, and J. Choo, âSurfhead: Affine rig blending for geometrically accurate 2d gaussian surfel head avatars,â in The Thirteenth International Conference on Learning Representations, 2025.