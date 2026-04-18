# FMGS-Avatar: Mesh-Guided 2D Gaussian Splatting with Foundation Model Priors for 3D Monocular Avatar Reconstruction

Jinlong Fana, Bingyu Hua, Xingguang Lib, Yuxiang Yanga, Jing Zhangc

aHangZhou Dianzi University, , HangZhou, , Zhejiang, China bShenzhen Polytechnic University, , ShenZhen, , Guangdong, China cWuHan University, , WuHan, , Hubei, China

## Abstract

Reconstructing high-fidelity animatable human avatars from monocular videos remains challenging due to insufficient geometric information in single-view observations. While recent 3D Gaussian Splatting methods have shown promise, they struggle with surface detail preservation due to the volumetric nature of 3D Gaussian primitives. To address both the representation limitations and information scarcity, we propose a novel method, FMGS-Avatar, that integrates two key innovations. First, we introduce Mesh-Guided 2D Gaussian Splatting, where 2D Gaussian primitives are attached directly to template mesh faces with constrained position, rotation, and movement, enabling superior surface alignment and geometric detail preservation. Second, we leverage foundation models trained on large-scale datasets, such as Sapiens, to complement the limited visual cues from monocular videos. However, when distilling multi-modal prior knowledge from foundation models, conflicting optimization objectives can emerge as different modalities exhibit distinct parameter sensitivities. We address this through a coordinated training strategy with selective gradient isolation, enabling each loss component to optimize its relevant parameters without interference. Through this combination of enhanced representation and coordinated information distillation, our approach significantly advances 3D monocular human avatar reconstruction. Experimental evaluation demonstrates superior reconstruction quality compared to existing methods, with notable gains in geometric accuracy and appearance fidelity while providing rich semantic information. Additionally, the distilled prior knowledge within a shared canonical space naturally enables spatially and temporally consistent rendering under novel views and poses.

Keywords: Human Avatar, 2D Gaussian Splatting, Foundation Model

## 1. Introduction

High-fidelity, animatable digital avatar creation has become increasingly important for applications ranging from entertainment and healthcare to AR/VR and interactive simulations. Traditional Motion Capture (MoCap) approaches, while capable of producing highquality results, require expensive equipment [1, 2] or controlled studio environments [3], limiting their accessibility. The development of methods that can create digital avatars from readily available monocular RGB videos would significantly democratize this technology.

Recent advances in neural rendering have opened new possibilities for digital human reconstruction from monocular videos. Neural Radiance Field (NeRF) [4] based approaches have demonstrated photorealistic rendering capabilities, though their computational requirements often limit real-time applications [5, 6, 7, 8, 9]. 3D Gaussian Splatting (3DGS) [10] has emerged as an attractive alternative, offering efficient rendering while maintaining high visual quality. Methods such as Animatable 3D Gaussians [11], GaussianAvatar [12], and GART [13] have shown promising progress in combining efficient rendering with realistic avatar modeling.

However, monocular avatar reconstruction faces two fundamental, interconnected challenges: geometric ambiguity from single-view data and the limitations of existing representations. While recent works have explored attaching 3D Gaussian primitives to a mesh template (e.g., GoMAvatar [14]), the volumetric nature of 3D Gaussians is suboptimal for representing surfaces, often leading to noisy geometry or depth ambiguity. Furthermore, while foundation models, such as DINOv2 [15], SAM [16], and Sapiens [17], can offer rich 2D priors (depth, normals, semantics) to alleviate geometric ambiguity, systematically distilling the multi-modal knowledge introduces a critical, unaddressed problem: optimization conflicts, where supervisory signals from different modalities compete and interfere with each other during training.

To address these challenges, we propose FMGS-Avatar, a novel method that leverages Foundation Model priors and Mesh-Guided 2D Gaussian Splatting to assist monocular human avatar reconstruction through systematic knowledge distillation. Rather than focusing solely on geometric or appearance enhancement, our approach distills comprehensive 2D knowledge, including semantic understanding, depth information, and surface normals, into 3D human avatars, aiming to improve both geometric and appearance quality while providing semantic annotations.

First, we propose Mesh-Guided 2D Gaussian Splatting, a representation inherently suited for surfaces. Unlike volumetric 3D Gaussians-based methods, our approach employs 2D Gaussian Splatting (2DGS) [18] as the core representation and takes the 2D primitives as surfels, naturally aligning with the surface manifold and providing a more geometrically faithful representation. This Mesh-Guided 2DGS design choice aims to improve surface alignment while maintaining the computational efficiency of 2DGS.

Second, we develop a method to systematically distill priors from multiple modalities, but more importantly, we introduce a Coordinated Training Strategy to resolve the inherent optimization conflicts. This strategy, featuring selective gradient stopping, is a core architectural innovation that enables the stable fusion of competing losses (e.g., depth loss affecting position, normal loss affecting orientation). It transforms the use of foundation models from simple "external supervision" into a deeply integrated and coherent learning process.

The resulting avatar representation, enhanced with distilled 2D knowledge, can be rendered under novel views and poses, naturally maintaining spatial and temporal consistency through the shared canonical space. Our experimental evaluation suggests that this approach achieves improved reconstruction quality compared to existing methods. Our main contributions include:

â¢ A Synergistic Framework for Knowledge Distillation: We present a unified approach where a surface-centric representation (Mesh-Guided 2DGS) and a conflict-aware training strategy (Coordinated Training) work in concert to enable the systematic and stable distillation of comprehensive knowledge (geometry, semantics) from 2D foundation models into a 3D avatar. Our method is designed to be extensible for incorporating additional 2D priors as foundation models continue to advance.

â¢ Mesh-Guided 2D Gaussian Splatting: We demonstrate the superiority of constraining 2D Gaussian primitives to a template mesh through explicit position, rotation, and movement constraints for surface modeling, achieving better geometric fidelity and alignment compared to methods that use volumetric 3D Gaussians.

â¢ Coordinated Training Strategy: We introduce a coordinated training strategy that addresses multimodal optimization conflicts through selective gradient isolation, enabling each loss component to focus on its most relevant parameters while preventing mutual interference. This approach ensures coherent learning across different Gaussian parameters and all representation components.

## 2. Related Work

## 2.1. Monocular Human Avatar Reconstruction

Early approaches for human avatar reconstruction relied on template-based methods that fit parametric models like SMPL to input observations [19, 20] but struggled with capturing clothing details. While NeRF-based methods [21, 22, 8, 23, 24, 25], achieved photorealistic results, their slow rendering speeds have driven the community towards 3D Gaussian Splatting [10] for creating real-time animatable avatars.

Recent 3DGS-based methods have explored various strategies [12, 26, 12, 27, 28]. Some attach Gaussians to a template mesh to enforce structural consistency, such as GoMAvatar [14] and GauHuman [29]. Others, like 3DGS-Avatar [30], focus on learning deformable Gaussian fields. The latest advancements continue to push the boundaries of expressiveness and efficiency. For instance, ExAvatar [31] extends the representation to the full body, including face and hands, by leveraging the SMPL-X model, enabling more expressive animations. Other works tackle the more challenging task of single-image reconstruction; GUAVA [32] achieves rapid upper-body avatar creation, and AniGS [33] focuses on generating animatable avatars from a single, potentially inconsistent image. While these methods demonstrate remarkable progress, they primarily focus on appearance modeling and still inherit the limitations of using volumetric primitives to model thin surfaces. In contrast, our work proposes Mesh-Guided 2D Gaussians representation, providing a more natural and efficient representation for surface geometry.

<!-- image-->  
Figure 1: Overview. Our method distills foundation model priors to enhance monocular human avatar reconstruction through Mesh-Guided 2D Gaussian Splatting and multi-field knowledge distillation. (a) Canonical Space Representation: We constrain 2D Gaussian primitives to template mesh faces for superior surface alignment, while employing separate feature volumes $\mathcal { V } _ { g } , \mathcal { V } _ { c } ,$ , and $\gamma _ { s }$ to store geometry, appearance, and semantic properties, respectively. (b) Multi-Field Distilling: Based on sampled property features, we utilize corresponding property fields, including a pose-dependent geometry residual field, a view-dependent appearance field, and a semantic field, to capture distilled knowledge from foundation models. (c) Skinning Field: The canonical human representation with distilled knowledge is transformed to observation space through forward Linear Blend Skinning (LBS) using learnable pose parameters $\theta _ { p }$ and predicted skinning weights W. (d) Training Strategy: In addition to supervision losses on each rendered modality, we propose a coordinated training strategy to balance the multi-field optimization and resolve potential conflicts. This novel method enables high-quality monocular avatar reconstruction with enhanced geometric details and rich semantic properties through systematic 2D-to-3D knowledge transfer.

## 2.2. Foundation Model Priors for 3D Reconstruction

Foundation models have achieved remarkable success across diverse vision tasks, with general-purpose models like CLIP [34], DINOv2 [15], and SAM [16] demonstrating exceptional zero-shot capabilities and robust feature representations. These models have been increasingly applied to 3D reconstruction, primarily for static scene reconstruction [35, 36, 37]. Concurrently, human-centric foundation models have rapidly emerged as a specialized domain [38]. Notably, Sapiens [17] represents a significant breakthrough, providing stateof-the-art performance across human pose estimation, depth prediction, surface normal estimation, and semantic parsing within a unified framework. Recent works have begun exploring the application of these humancentric foundation model priors to human avatar reconstruction. However, existing approaches predominantly leverage single-modal supervision in isolation. For example, StruGauAvatar [39] utilizes surface normals as pseudo ground truth. The systematic distillation of comprehensive multi-modal foundation model knowledgeâ encompassing depth, normals, and semanticsâinto a dynamic 3D human avatar remains largely unexplored.

Furthermore, the inherent optimization conflicts that arise from simultaneously applying these diverse supervisory signals have not been adequately addressed.

## 3. Preliminaries

2D Gaussian Splatting. Unlike 3DGS, which uses 3D ellipsoids, 2DGS employs flat 2D Gaussian disks embedded in 3D space for scene representation. These primitives distribute densities within planar surfaces (surfels), enabling better surface alignment and improved geometry reconstruction compared to volumetric representations. Each 2D Gaussian primitive is characterized by its center point $\mu \in \mathbb { R } ^ { 3 }$ , opacity $\alpha \in \mathbb { R }$ view-dependent color $\mathbf { c } \in \mathbb { R } ^ { 3 }$ Î±computed via spherical harmonics, scaling vector $\textbf { s } = \ ( s _ { u } , s _ { \nu } ) \ \in \ \mathbb { R } ^ { 2 }$ control-,ling the 2D variance, and rotation matrix $\textbf { R } \in \mathbb { R } ^ { 3 \times 3 }$ . The rotation matrix $\mathbf R = [ \mathbf t _ { u } , \mathbf t _ { \nu } , \mathbf t _ { w } ]$ consists of two orthogonal tangent vectors $\mathbf { t } _ { u } , \mathbf { t } _ { \nu }$ and the normal vector $\mathbf { t } _ { w } = \mathbf { t } _ { u } \times \mathbf { t } $ ,v obtained through cross product. The 2D Gaussian is defined in a local tangent uv plane. For any point $\mathbf { u } = ( u , \nu )$ in uv space, the Gaussian value is computed as $\begin{array} { r } { G ( \mathbf { u } ) = \exp ( - \frac { \mathbf { \bar { \rho } } u ^ { 2 } + \nu ^ { 2 } } { 2 } ) } \end{array}$ . During rendering, 2DGS maps uv space to screen pixels through differentiable Gaussian rasterization:

$$
\mathbf { c } ( \mathbf { x } ) = \sum _ { i } \mathbf { c } _ { i } \alpha _ { i } ^ { 2 D } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ^ { 2 D } ) .\tag{1}
$$

## 4. Method

Fig. 1 illustrates FMGS-Avatar for creating animatable 3D human avatars from monocular videos through systematic knowledge distillation from foundation models. Given a monocular video sequence $\{ I ^ { k } \} _ { k = 1 } ^ { K }$ with fitted SMPL parameters for each frame, including pose $\theta ,$ shape $\beta ,$ and template mesh $M _ { c } ,$ we first extract Î¸ Î²rich 2D priors including foreground masks MÂ¯ , pseudo depth DÂ¯ , surface normals $\bar { N } ,$ and human parsing semantics SÂ¯ using foundation model.

## 4.1. Canonical Space Representation

## 4.1.1. Mesh-Guided 2D Gaussian Splatting

The choice of 2D Gaussians over 3D is deliberate and critical for surface modeling. 3D Gaussians are volumetric ellipsoids. When representing a thin surface like cloth or skin, they must be made extremely flat, leading to training instabilities, or they retain volume, creating depth ambiguity and a "blurry" or "thickened" surface effect. 2D Gaussians, as planar surfels, are inherently surface-based primitives. This makes them a more efficient and geometrically faithful representation for avatar surfaces.

Given the SMPL template mesh $M _ { c }$ with 6,890 vertices, we first upsample it to 30,000 vertices to obtain a denser mesh $\begin{array} { r c l } { \mathbf { \dot { \mathcal { M } } } _ { c } ^ { u p } } & { = } & { \{ \{ \nu _ { i } \} _ { i = 1 } ^ { V } , \{ f _ { j } \} _ { j = 1 } ^ { F } \} } \end{array}$ for en-,hanced surface detail representation. For each face $f _ { j }$ on the upsampled mesh $\mathcal { M } _ { c } ^ { u p }$ , we attach a corresponding 2D Gaussian primitive to establish explicit surface correspondence. This one-to-one mapping ensures that our representation can faithfully capture the underlying mesh topology while benefiting from the efficient rendering properties of Gaussian Splatting. For each 2D Gaussian primitive $k ,$ its position $\mu _ { k }$ is determined by Âµthe barycentric center of its corresponding face:

$$
\mu _ { k } = \frac { 1 } { 3 } \sum _ { i = 1 } ^ { 3 } \nu _ { i } ,\tag{2}
$$

where $\nu _ { i }$ are the three vertices of face $f _ { k }$ . This constraint anchors each Gaussian to a specific mesh location, providing geometric stability and surface coherence.

The rotation matrix $\mathbf R _ { k } \ = \ [ \mathbf t _ { u } , \mathbf t _ { \nu } , \mathbf t _ { w } ]$ for each primi-, ,tive is constructed such that the tangent vectors $( \mathbf { t } _ { u } , \mathbf { t } _ { \nu } )$ ,lie within the tangent plane of the mesh surface, while $\mathbf { t } _ { w }$ aligns with the face normal ${ \bf n } _ { k }$ . This orientation constraint ensures that 2D Gaussian primitives maintain proper surface alignment and follow the natural curvature of the human body.

Unlike conventional 3DGS methods that employ adaptive density control during optimization, our approach fixes the number of 2D Gaussians after mesh upsampling. This design choice prevents uncontrolled primitive proliferation while ensuring sufficient representation density through the systematic upsampling strategy. The fixed correspondence between mesh faces and Gaussians also facilitates consistent property feature learning across the avatar surface.

## 4.1.2. Property Feature Sampling

Instead of using per-point features for encoding different avatar properties in canonical space, we employ separate HashGrid volumes [40] for memory efficiency and multi-level feature fusion. We use independent volumes to store different property features: a geometry feature volume $\mathcal { V } _ { g } ,$ an appearance feature volume $\mathcal { V } _ { c } .$ and a semantic feature volume $\mathcal { \boldsymbol { V } } _ { s }$ . For each 2D Gaussian in canonical space, property features $( \mathbf { f } _ { g } , \mathbf { f } _ { c } , \mathbf { f } _ { s } ) \ \in$ $\mathbb { R } ^ { 3 2 }$ , ,are sampled from their corresponding feature volumes through trilinear interpolation. This design is extensible for incorporating additional foundation model properties by simply adding more feature volumes with minimal computational overhead, making our method adaptable to future foundation model advances.

## 4.2. Multi-Field Distilling

## 4.2.1. Geometry Field

In canonical space, 2D Gaussian primitives constrained by the template mesh have limited representation ability for pose-dependent surface details. To address this, we introduce a pose-dependent geometry residual field to correct pose-related deformations. We formulate this field as a lightweight MLP, which takes geometry feature $\mathbf { f } _ { g }$ and pose latent code $\mathcal { Z } _ { p }$ as input:

$$
( \delta d , \delta \mathbf { s } , \delta \mathbf { r } , \mathbf { z } _ { g } ) = \mathcal { F } _ { \theta _ { g } } ( \mathbf { f } _ { g } , \boldsymbol { \mathcal { Z } } _ { p } ) .\tag{3}
$$

The pose latent code ${ \mathcal { Z } } _ { p }$ encodes SMPL pose and shape parameters $( \theta , \beta )$ using a hierarchical pose encoder [41], providing pose context for the observation space. The canonical Gaussians are corrected as:

$$
\mu ^ { \prime } = \mu + \mathbf { n } \cdot \delta d ,\tag{4}
$$

$$
\mathbf { s } ^ { \prime } = \mathbf { s } \cdot \mathrm { e x p } ( \delta \mathbf { s } ) ,\tag{5}
$$

$$
\mathbf { R } ^ { \prime } = \mathbf { R } \cdot \exp ( [ \delta \mathbf { r } ] _ { \times } ) ,\tag{6}
$$

where n is the surface normal. nÂ· d ensures the position Î´offset only moves along the normal direction, reducing the 3D movement freedom to 1D displacement. And [ r]Ã denotes the skew-symmetric matrix for rotation Î´updates.

## 4.2.2. Appearance Field

Conventional 3DGS methods use spherical harmonics for view-dependent color [42, 43], but in monocular settings, camera directions are limited and may not align with human pose variations. Similar to [9, 30], We canonicalize ray directions d from observation space to canonical space as $\hat { \mathbf { d } } = \mathbf { T } _ { 1 : 3 , 1 : 3 } ^ { - 1 } \mathbf { d }$ using inverse rotation ,matrices from forward skinning (ref. to Sec. 4.3).

Furthermore, local deformations such as clothing wrinkles depend on human pose, motivating pose conditioning for color prediction. Our appearance field takes sampled color feature $\mathbf { f } _ { c } ,$ geometry-encoded feature $\mathbf { z } _ { g } ~ \in ~ \mathbb { R } ^ { 1 6 }$ from geometry field, pose latent code $\boldsymbol { Z } _ { p } \in \mathbb { R } ^ { 1 6 }$ , and canonicalized viewing direction $\gamma ( \hat { \mathbf { d } } )$ as input:

$$
\begin{array} { r } { \mathbf { c } = \mathscr { F } _ { \theta _ { c } } ( \mathbf { f } _ { c } , \mathbf { z } _ { g } , \mathscr { Z } _ { p } , \gamma ( \hat { \mathbf { d } } ) ) . } \end{array}\tag{7}
$$

Following [30], we use a compact MLP with one 64- dimensional hidden layer to prevent overfitting while maintaining sufficient representational capacity.

## 4.2.3. Semantics Field

We leverage the Sapiens foundation model [17] to estimate the human parsing map with 28 semantic classes. To represent these semantics in our canonical space, we sample a feature vector $\mathbf { f } _ { s }$ from the semantic feature volume $\mathcal { V } _ { s }$ . Directly interpreting these features as semantic logits (e.g., via a softmax) is suboptimal, as the feature volume stores a compressed, abstract representation rather than clean, class-specific logits. Therefore, we employ a lightweight MLP, $\mathcal { F } _ { \theta _ { s } }$ , which acts as a seÎ¸mantic decoder. This decoder learns a non-linear mapping from the sampled feature vector $\mathbf { f } _ { s }$ to the final 28- dimensional semantic logits ls:

$$
\mathbf { l } _ { s } = \mathcal { F } _ { \theta _ { s } } ( \mathbf { f } _ { s } ) .\tag{8}
$$

Using a shared MLP decoder provides two key advantages: 1) It significantly increases the modelâs representational power, allowing it to learn complex boundaries between semantic regions. 2) It enhances spatial consistency by applying a single, coherent mapping function across the entire feature space, resulting in smoother and more reliable semantic maps. The rendered logits are then processed with softmax and argmax to obtain the final semantic map Is = arg max(softmax(ls)).

## 4.3. Skinning Field

Since mesh-guided 2D Gaussians corrected by the geometry residual field are no longer strictly on the template mesh, we have to diffuse the skinning weights defined on the mesh into 3D space. To that end, we learn a neural network $\mathcal { F } _ { \theta _ { w } }$ to predict the skinning weights $\mathcal { W } = \{ w _ { b } \} _ { b = 1 } ^ { 2 4 }$ Î¸for any point in the canonical space.

The input to this network are the corrected Gaussian positions $\mu ^ { \prime }$ , which are first encoded into features using Âµa multi-resolution hash encoding, which we denote as $H ( \cdot )$ . The network $\mathcal { F } _ { \theta _ { w } }$ is implemented as a 4-layer MLP Î¸with a hidden dimension of 128. This architecture takes the hash-encoded features as input and outputs a 24- dimensional vector corresponding to the influences of the SMPL joints. The final output is processed through a softmax layer to ensure the skinning weights sum to one $\begin{array} { r } { ( \sum _ { b = 1 } ^ { 2 4 } w _ { b } = 1 ) } \end{array}$ . This process is formulated as:

$$
\mathcal { W } = \operatorname { s o f t m a x } ( \mathcal { F } _ { \theta _ { w } } ( H ( \mu ^ { \prime } ) ) ) .\tag{9}
$$

We then transform the 2D Gaussian positions and rotations from canonical space to the observation space via forward Linear Blend Skinning (LBS):

$$
\mathbf { T } = \sum _ { b = 1 } ^ { 2 4 } w _ { b } \mathbf { B } _ { b } ,\tag{10}
$$

$$
\mu _ { o } = \mathbf { T } \boldsymbol { \mu } ^ { \prime } ,\tag{11}
$$

$$
\mathbf { R } _ { o } = \mathbf { T } _ { 1 : 3 , 1 : 3 } \mathbf { R } ^ { \prime } ,\tag{12}
$$

where $\mathbf { B } _ { b }$ represents the bone transformation matrices of human pose $\theta _ { p } ,$ , and T is the blended transformation Î¸matrix. Finally, images in different modalities are rendered via Eq.1.

## 4.4. Training Objectives and Regularization

Our training objective combines multiple loss terms to ensure high-quality reconstruction while effectively leveraging foundation model priors.

Photometric Loss. We employ a combination of L1 and SSIM losses to ensure photometric consistency between rendered images I and input frames Â¯I:

$$
\mathcal { L } _ { \mathrm { c } } = ( 1 - \lambda _ { \mathrm { s s i m } } ) \mathcal { L } _ { 1 } ( I , \bar { I } ) + \lambda _ { \mathrm { s s i m } } \mathcal { L } _ { \mathrm { S S I M } } ( I , \bar { I } ) ,\tag{13}
$$

where $\lambda _ { \mathrm { s s i m } } ~ = ~ 0 . 2$ balances the contribution of both Î» .terms. This combination captures both fine-grained pixel-level differences and perceptual image quality.

Mask Loss. To ensure accurate foreground segmentation, we apply binary cross-entropy loss on the rendered opacity mask M and ground truth mask MÂ¯ :

$$
{ \mathcal { L } } _ { \mathrm { m } } = { \mathcal { L } } _ { \mathrm { B C E } } ( M , { \bar { M } } ) .\tag{14}
$$

Depth Supervision Loss. Since estimated single-view depth from foundation model exhibits scale ambiguity and may contain absolute depth errors, we employ an ordinal depth ranking loss that focuses on relative depth relationships rather than absolute values. Following [44], We define the ordinal indicator function as:

$$
\begin{array} { r } { \tilde { J } _ { \mathrm { o r d } } ( \mathcal { D } ( x _ { 1 } ) , \mathcal { D } ( x _ { 2 } ) ) = \left\{ \begin{array} { l l } { + 1 , } & { \mathrm { i f } \ \mathcal { D } ( x _ { 1 } ) > \mathcal { D } ( x _ { 2 } ) } \\ { - 1 , } & { \mathrm { i f } \ \mathcal { D } ( x _ { 1 } ) < \mathcal { D } ( x _ { 2 } ) . } \end{array} \right. } \end{array}\tag{15}
$$

The depth ranking loss is then formulated as:

$$
\mathcal { L } _ { \mathrm { d } } = \left. \operatorname { t a n h } \left( \alpha ( \mathcal { D } ( x _ { 1 } ) - \mathcal { D } ( x _ { 2 } ) ) \right) - \bar { Z } _ { \mathrm { o r d } } ( \bar { \mathcal { D } } ( x _ { 1 } ) , \bar { \mathcal { D } } ( x _ { 2 } ) ) \right. _ { 1 } ,\tag{16}
$$

where $\alpha = 1 0$ is a scaling factor, and we randomly sample pixel pairs $( x _ { 1 } , x _ { 2 } )$ during training to compute the ,ranking loss efficiently.

Surface Normal Loss. We supervise surface normal estimation using multiple consistency constraints. We use a self-consistency loss $\mathcal { L } _ { s n } = | | \mathcal { N } - \hat { N } | | _ { 1 }$ between the rendered normals N and the normals derived from the rendered depth map, NË . We also leverage a prior alignment loss $\mathcal { L } _ { n } = 1 - N \cdot \bar { N }$ to encourage consistency with the foundation modelâs predictions NÂ¯ . Additionally, to promote spatial smoothness on the rendered normal map, we apply a total variation (TV) regularization loss, ${ \mathcal { L } } _ { \mathrm { t v } }$ The TV loss is defined as the sum of the absolute differences of neighboring pixel values in the normal map along the horizontal (x) and vertical (y) axes:

$$
\mathcal { L } _ { \mathrm { t v } } ( \boldsymbol { N } ) = \sum _ { i , j } \left( | N _ { i , j + 1 } - N _ { i , j } | + | N _ { i + 1 , j } - N _ { i , j } | \right) .\tag{17}
$$

The complete surface normal loss is the sum of these components:

$$
\mathcal { L } _ { \mathrm { n o r m } } = \mathcal { L } _ { s n } + \mathcal { L } _ { n } + \mathcal { L } _ { \mathrm { t v } } .\tag{18}
$$

Semantic Loss. For semantic supervision, we use standard cross-entropy loss between predicted semantic logits $s$ and foundation model semantic maps $\bar { s } { : }$

$$
\mathcal { L } _ { \mathrm { s } } = \mathcal { L } _ { \mathrm { C E } } ( S , \bar { S } ) .\tag{19}
$$

We also propose semantic-guided regularization that encourages feature consistency within semantic regions while allowing cross-region variations:

$$
\mathcal { L } _ { \mathrm { r e g } } = \frac { 1 } { | C | } \sum _ { c \in C } \frac { 1 } { | \mathscr { G } _ { c } | } \sum _ { i , j \in \mathscr { G } _ { c } } \| \mathbf { f } _ { i } - \mathbf { f } _ { j } \| _ { 2 } ^ { 2 } ,\tag{20}
$$

where C represents the set of semantic classes, $\mathcal { G } _ { c }$ contains all Gaussians belonging to semantic class $^ { c , }$ and $\mathbf { f } _ { i }$ denotes sampled semantic features.

Other Regularization Terms. To prevent overfitting in the monocular setting and maintain geometric consistency, we also include $\mathcal { L } _ { \mathrm { s k i n } }$ regularizes skinning weights, and $\mathcal { L } _ { \mathrm { i s o } }$ encourages as-rigid-as-possible deformation by preserving local distances between neighboring Gaussians.

<!-- image-->  
Figure 2: Qualitative results on ZJUMoCap dataset.

Total Loss Function. Our complete objective function combines all terms with carefully tuned weights:

$$
\begin{array} { r l } { \mathcal { L } = \mathcal { L } _ { \mathrm { c } } + \lambda _ { m } \mathcal { L } _ { \mathrm { m } } + \lambda _ { d } \mathcal { L } _ { \mathrm { d } } } & { { } } \\ { + \lambda _ { n } \mathcal { L } _ { \mathrm { n o r m } } + \lambda _ { s } \mathcal { L } _ { \mathrm { s } } + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { r e g } } } & { { } } \\ { + \lambda _ { \mathrm { s k i n } } \mathcal { L } _ { \mathrm { s k i n } } + \lambda _ { \mathrm { i s o } } \mathcal { L } _ { \mathrm { i s o } } , } \end{array}\tag{21}
$$

where $\{ \lambda _ { m } , \lambda _ { \mathrm { s k i n } } , \lambda _ { \mathrm { i s o } } \}$ follow established practices Î» , Î» , Î»from [30], while $\{ \lambda _ { d } , \lambda _ { n } , \lambda _ { s } , \lambda _ { \mathrm { r e g } } \}$ are determined Î» , Î» ,through empirical validation.

## 4.5. Coordinated Training Strategy

Multi-field distillation faces conflicting optimization requirements where Depth estimation is primarily sensitive to Gaussian positions, surface normals depend critically on Gaussian orientations, and semantic information requires cluster consistency. To resolve these competing objectives targeting the same Gaussian parameters, we implement selective gradient stopping: (1)

depth losses block gradients to Gaussian rotation parameters, focusing optimization on positional updates; (2) normal losses block gradients to Gaussian position parameters, concentrating on orientation refinement; and (3) semantic losses block gradients to both position and rotation parameters, directing optimization toward the semantic field while minimizing interference with geometric Gaussian parameters. This coordinated approach prevents parameter competition, enabling effective multi-modal knowledge distillation while maintaining geometric and semantic consistency for highquality avatar reconstruction.

## 5. Experiments

In this section, we conduct comprehensive evaluations to demonstrate the effectiveness of our approach. We first compare our method with recent state-ofthe-art methods for neural human reconstruction from monocular videos, including both NeRF-based (Neural-Body [8], Anim-NeRF [45], ARAH [46], NVR [47], HumanNeRF [6]) and 3DGS-based (GoMAvatar [14], GauHuman [29], 3DGS-Avatar [30]) approaches. Subsequently, we perform systematic ablation studies to validate the effectiveness of each designed component.

<table><tr><td rowspan=1 colspan=1>Method|</td><td rowspan=1 colspan=5>PSNRâSSIMâ LPIPSâTrainâFPSâ</td></tr><tr><td rowspan=5 colspan=1>NeuralBody [8]Anim-NeRF [45]MonoHuman [5]InstantAvatar [48]HumanNeRF [6]</td><td rowspan=1 colspan=1>29.03</td><td rowspan=1 colspan=1>0.964</td><td rowspan=1 colspan=1>52.29</td><td rowspan=1 colspan=1>10h</td><td rowspan=1 colspan=1>1.5</td></tr><tr><td rowspan=1 colspan=1>29.17</td><td rowspan=1 colspan=1>0.961</td><td rowspan=1 colspan=1>51.98</td><td rowspan=1 colspan=1>13h</td><td rowspan=1 colspan=1>1.1</td></tr><tr><td rowspan=1 colspan=1>30.26</td><td rowspan=1 colspan=1>0.969</td><td rowspan=1 colspan=1>30.92</td><td rowspan=1 colspan=1>6h</td><td rowspan=1 colspan=1>0.1</td></tr><tr><td rowspan=1 colspan=1>29.73</td><td rowspan=1 colspan=1>0.938</td><td rowspan=1 colspan=1>64.41</td><td rowspan=1 colspan=1>5m</td><td rowspan=1 colspan=1>4.2</td></tr><tr><td rowspan=1 colspan=1>30.24</td><td rowspan=1 colspan=1>0.969</td><td rowspan=1 colspan=1>33.38</td><td rowspan=1 colspan=1>10h</td><td rowspan=1 colspan=1>0.3</td></tr><tr><td rowspan=1 colspan=1>GoMAvatar [14]</td><td rowspan=1 colspan=1>30.56</td><td rowspan=1 colspan=1>0.967</td><td rowspan=1 colspan=1>32.55</td><td rowspan=1 colspan=1>15h</td><td rowspan=1 colspan=1>43</td></tr><tr><td rowspan=1 colspan=1>GauHuman [29]</td><td rowspan=1 colspan=1>30.79</td><td rowspan=1 colspan=1>0.960</td><td rowspan=1 colspan=1>32.73</td><td rowspan=1 colspan=1>1m</td><td rowspan=1 colspan=1>180</td></tr><tr><td rowspan=1 colspan=1>3DGS-Avatar [30]</td><td rowspan=1 colspan=1>30.61</td><td rowspan=1 colspan=1>0.965</td><td rowspan=1 colspan=1>30.28</td><td rowspan=1 colspan=1>17m</td><td rowspan=1 colspan=1>50</td></tr><tr><td rowspan=1 colspan=1>FMGS-Avatar</td><td rowspan=1 colspan=2>30.89 0.972</td><td rowspan=1 colspan=1>28.59</td><td rowspan=1 colspan=1>10m</td><td rowspan=1 colspan=1>55</td></tr></table>

Table 1: Quantitative Results on ZJU-MoCap. Cell color indicated Best and Second Best .

## 5.1. Evaluation Datasets

ZJU-MoCap Dataset. ZJU-MoCap dataset [8] serves as our primary testbed for quantitative evaluation. We select six representative sequences (377, 386, 387, 392, 393, 394) from the ZJU-MoCap dataset and follow the standard training/test split established by Human-NeRF [6]. The motion patterns in these sequences are repetitive and do not contain sufficient pose diversity for meaningful novel pose synthesis benchmarks. Therefore, we focus on evaluating novel view synthesis performance using standard metrics (PSNR/SSIM/LPIPS)

<!-- image-->  
Figure 3: Comprehensive comparison of geometric reconstruction quality. From left to right: (a) reconstructed appearance, (b) depth map, (c) predicted normal, (d) surface normal derived from depth gradient. Our method demonstrates superior geometric fidelity.

and provide qualitative results for animation under outof-distribution poses. Note that LPIPS values in all tables are scaled by 1000 for clarity.

PeopleSnapshot Dataset. We also conduct experiments on 4 sequences from PeopleSnapshot dataset [49], which contains monocular videos of people rotating in front of a camera under controlled lighting conditions. We follow the data split protocol established by InstantAvatar [48] and compare directly with their results for fair evaluation. For consistency, we use the SMPL poses optimized by AnimNeRF [45] without further refinement during our training process.

## 5.2. Comparison with Baselines

Quantitative Results. Tab.1 and 2 present quantitative results for novel view synthesis on ZJU-MoCap and PeopleSnapshot datasets, respectively. Our method demonstrates superior rendering quality compared to both NeRF-based baselines and recent 3DGS-based state-of-the-art approaches across all evaluation metrics.

On ZJU-MoCap dataset, our method achieves stateof-the-art performance. Compared to the closest competitor, 3DGS-Avatar, we achieve consistent improvements across all metrics, while maintaining competitive rendering efficiency at 55 FPS. Our method shows particularly notable improvements over GoMAvatar, which shares a similar Gaussian-on-Mesh design philosophy but employs 3D Gaussians. We achieve substantial gains across all metrics. More importantly, our approach provides dramatic efficiency improvements with 90Ã faster training (10 minutes vs. 15 hours) while maintaining comparable inference speed (55 vs. 43 FPS).

<table><tr><td rowspan="2"></td><td colspan="3">male-3-casual</td><td colspan="3">male-4-casual</td><td colspan="3">female-3-casual</td><td colspan="3">female-4-casual</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Anim-NeRF [45]</td><td>23.17</td><td>0.9266</td><td>78.4</td><td>22.30</td><td>0.9235</td><td>91.1</td><td>22.37</td><td>0.9311</td><td>78.4</td><td>23.18</td><td>0.9292</td><td>68.7</td></tr><tr><td>NeuralBody [8]</td><td>24.94</td><td>0.9428</td><td>32.6</td><td>24.71</td><td>0.9469</td><td>42.3</td><td>23.87</td><td>0.9504</td><td>34.6</td><td>24.37</td><td>0.9451</td><td>38.2</td></tr><tr><td>Anim-3DGS [11]</td><td>29.06</td><td>0.9704</td><td>26.4</td><td>26.16</td><td>0.9554</td><td>49.1</td><td>24.59</td><td>0.9535</td><td>39.9</td><td>27.26</td><td>0.9634</td><td>28.1</td></tr><tr><td>InstantAvatar [48]</td><td>29.65</td><td>0.9731</td><td>19.2</td><td>27.97</td><td>0.9649</td><td>34.6</td><td>27.9</td><td>0.9722</td><td>24.9</td><td>28.92</td><td>0.9692</td><td>18.0</td></tr><tr><td>3DGS-Avatar [30]</td><td>30.57</td><td>0.9581</td><td>20.9</td><td>33.16</td><td>0.9678</td><td>15.7</td><td>34.28</td><td>0.9724</td><td>14.9</td><td>30.22</td><td>0.9653</td><td>23.1</td></tr><tr><td>FMGS-Avatar</td><td>30.92</td><td>0.9816</td><td>15.2</td><td>33.78</td><td>0.9753</td><td>23.7</td><td>33.89</td><td>0.9854</td><td>14.2</td><td>30.95</td><td>0.9873</td><td>15.8</td></tr></table>

Table 2: Comparison on PeopleSnapshot Dataset.

On PeopleSnapshot dataset, our method consistently outperforms existing approaches across most sequences. Notably, we achieve the best performance on 3 out of 4 sequences, with particularly strong results on male-3-casual and female-4-casual, demonstrating the effectiveness of our foundation model distillation and mesh-guided representation across diverse subjects.

Our approach achieves significant training acceleration compared to traditional NeRF-based methods, requiring only 10 minutes versus hours for conventional approaches. While InstantAvatar achieves faster training (5 minutes), our method delivers substantially superior inference performance (55 FPS vs. 4.2 FPS), making it more suitable for real-time applications. Although some recent methods like GauHuman achieve faster training (1 minute) and higher inference speeds (180 FPS), our approach provides a better qualityefficiency trade-off, delivering superior reconstruction fidelity while maintaining practical rendering speeds for interactive applications.

Qualitative Analysis. Fig.2 presents qualitative comparisons of novel view rendering results on ZJU-MoCap dataset. Our method produces significantly more detailed and geometrically consistent results compared to NeRF-based baselines while achieving comparable or superior quality to 3DGS-based state-of-the-art methods. NeRF-based methods exhibit characteristic limitations: ARAH [46] shows notable artifacts on human body regions, particularly in areas with complex geometry, while NVR produces overly smooth surfaces that lack fine-grained details such as clothing wrinkles. In contrast, our approach effectively leverages distilled foundation model priors to resolve geometric ambiguities inherent in monocular reconstruction, resulting in enhanced surface details and more realistic appearance.

Fig.3 provides a comprehensive analysis of our methodâs geometric reconstruction capabilities through depth and surface normal visualizations. To systematically validate the effectiveness of our approach, we conduct comparative analysis using 3DGS-Avatar as the 3DGS baseline and an adapted 2DGS baseline where 3D Gaussians are replaced with conventional 2D Gaussians without mesh guidance.

The results demonstrate clear advantages of our approach across multiple geometric aspects. First, 2DGS primitives provide more consistent depth scaling and reasonable surface normal estimation compared to 3DGS, as the planar nature of 2D Gaussians could align with underlying surface geometry better. Second, our mesh-guided design further enhances this alignment by constraining primitives to template mesh faces, ensuring geometrically plausible surface reconstruction. Third, the integration of foundation model priors provides additional geometric cues that resolve ambiguities in monocular settings, leading to more accurate depth estimation and surface normal prediction.

Fig.4 demonstrates the rendered multi-modal results of Subject 392 driven by poses from AIST++ [50] and AMASS [2] sequences, showcasing our methodâs capability to generalize to out-of-distribution poses. This represents a significant advancement in 2D-to-3D knowledge transfer, where 2D semantic and geometric priors from foundation models become effectively drivable in 3D space. This drivable semantic information is particularly valuable for downstream applications such as virtual try-on, motion analysis, or avatar editing, expanding the practical utility of reconstructed avatars beyond basic animation and rendering.

## 5.3. Ablation

We conduct systematic ablation studies on Subject 377 from ZJU-MoCap dataset to validate the effectiveness of each proposed component. Tab. 3 presents quantitative results demonstrating the progressive improvement achieved by each component.

<!-- image-->  
Figure 4: Multi-modal novel pose synthesis results showing: (a) RGB rendering, (b) depth map, (c) predicted normal, (d) surface normal derived from depth, (e) semantic segmentation. The first row shows the canonical rest pose, while subsequent rows demonstrate poses from AIST++ [50] and AMASS [2] sequences.

Foundation Model Supervision. The baseline without foundation model supervision achieves the lowest performance (29.98 dB PSNR). Adding depth supervision $( \mathcal { L } _ { d } )$ provides a +0.33 dB improvement, demonstrating the value of geometric priors. Incorporating self-consistent normal loss $( \mathcal { L } _ { s n } )$ further enhances results by +0.42 dB, while normal supervision $( { \mathcal { L } } _ { n } )$ from foundation models contributes an additional +0.36 dB improvement. Finally, adding semantic supervision $( \mathcal { L } _ { s } )$ achieves the best performance (31.22 dB), validating the effectiveness of comprehensive multi-modal knowledge distillation.

Coordinated Training Strategy. Fig. 5 demonstrates the critical importance of our proposed Coordinated Training Strategy. Without selective gradient blocking, multi-modal losses compete for the same Gaussian parameters, leading to notable artifacts in both semantic and normal fields, evident as black holes in semantic maps and incorrect normals at the head and legs. Our coordinated approach effectively resolves these optimization conflicts, ensuring stable multi-field learning.

## 6. Conclusion

We propose FMGS-Avatar, which leverages meshguided 2D Gaussian Splatting with foundation model priors to enhance monocular human avatar reconstruction through systematic knowledge distillation. Our approach addresses three fundamental challenges: (1) information scarcity inherent in monocular observations through multi-modal foundation model distillation, (2) surface representation limitations of conventional 3D Gaussians through mesh-guided 2D Gaussians, and (3) optimization conflicts in multi-field learning through coordinated training strategies.

<table><tr><td></td><td>Ld Lsn Ln Ls</td><td></td><td></td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td></td><td></td><td></td><td></td><td>29.98</td><td>0.938</td><td>29.53</td></tr><tr><td>â</td><td></td><td></td><td></td><td>30.31</td><td>0.958</td><td>28.23</td></tr><tr><td>â</td><td>â</td><td></td><td></td><td>30.73</td><td>0.963</td><td>28.01</td></tr><tr><td>â</td><td>â</td><td>â</td><td></td><td>31.09</td><td>0.975</td><td>27.42</td></tr><tr><td>â</td><td>â</td><td>â</td><td>â</td><td>31.22</td><td>0.978</td><td>26.53</td></tr></table>

Table 3: Quantitative results of ablation study.

<!-- image-->  
Figure 5: Effectiveness of Coordinated Training Strategy.

Experimental results demonstrate that our method achieves state-of-the-art performance in both geometric accuracy and appearance fidelity while maintaining efficient training and rendering capabilities. The distilled 2D foundation model priors in canonical 3D space could be rendered under novel views and poses through spatially and temporally consistent avatar animation, significantly advancing the practical applicability of monocular avatar reconstruction.

Our method establishes a promising pathway for incorporating rapidly advancing foundation model capabilities into 3D human reconstruction, suggesting significant potential for future research in cross-modal knowledge transfer and neural avatar modeling.

## References

[1] M. Loper, N. Mahmood, M. J. Black, Mosh: motion and shape capture from sparse markers., ACM Trans. Graph. 33 (6) (2014) 220â1.

[2] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, M. J. Black, Amass: Archive of motion capture as surface shapes, in: Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 5442â5451.

[3] H. Joo, H. Liu, L. Tan, L. Gui, B. Nabbe, I. Matthews, T. Kanade, S. Nobuhara, Y. Sheikh, Panoptic studio: A massively multiview system for social motion capture, in: Proceedings of the IEEE international conference on computer vision, 2015, pp. 3334â3342.

[4] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, R. Ng, Nerf: Representing scenes as neural radiance fields for view synthesis, Communications of the ACM 65 (1) (2021) 99â106.

[5] Z. Yu, W. Cheng, X. Liu, W. Wu, K.-Y. Lin, Monohuman: Animatable human neural field from monocular video, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 16943â16953.

[6] C.-Y. Weng, B. Curless, P. P. Srinivasan, J. T. Barron, I. Kemelmacher-Shlizerman, Humannerf: Free-viewpoint rendering of moving people from monocular video, in: Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition, 2022, pp. 16210â16220.

[7] B. Jiang, Y. Hong, H. Bao, J. Zhang, Selfrecon: Self reconstruction your digital avatar from monocular video, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5605â5615.

[8] S. Peng, Y. Zhang, Y. Xu, Q. Wang, Q. Shuai, H. Bao, X. Zhou, Neural body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 9054â9063.

[9] X. Zhou, S. Peng, Z. Xu, J. Dong, Q. Wang, S. Zhang, Q. Shuai, H. Bao, Animatable implicit neural representations for creating realistic avatars from videos, IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (6) (2024) 4147â 4159.

[10] B. Kerbl, G. Kopanas, T. LeimkÃ¼hler, G. Drettakis, 3d gaussian splatting for real-time radiance field rendering., ACM Trans. Graph. 42 (4) (2023) 139â 1.

[11] Y. Liu, X. Huang, M. Qin, Q. Lin, H. Wang, Animatable 3d gaussian: Fast and high-quality reconstruction of multiple human avatars, in: Proceed-

ings of the 32nd ACM International Conference on Multimedia, 2024, pp. 1120â1129.

[12] L. Hu, H. Zhang, Y. Zhang, B. Zhou, B. Liu, S. Zhang, L. Nie, Gaussianavatar: Towards realistic human avatar modeling from a single video via animatable 3d gaussians, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 634â644.

[13] J. Lei, Y. Wang, G. Pavlakos, L. Liu, K. Daniilidis, Gart: Gaussian articulated template models, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 19876â19887.

[14] J. Wen, X. Zhao, Z. Ren, A. G. Schwing, S. Wang, Gomavatar: Efficient animatable human modeling from monocular video using gaussians-on-mesh, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 2059â2069.

[15] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., Dinov2: Learning robust visual features without supervision, arXiv preprint arXiv:2304.07193 (2023).

[16] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al., Segment anything, in: Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015â4026.

[17] R. Khirodkar, T. Bagautdinov, J. Martinez, S. Zhaoen, A. James, P. Selednik, S. Anderson, S. Saito, Sapiens: Foundation for human vision models, in: European Conference on Computer Vision, Springer, 2024, pp. 206â228.

[18] B. Huang, Z. Yu, A. Chen, A. Geiger, S. Gao, 2d gaussian splatting for geometrically accurate radiance fields, in: ACM SIGGRAPH 2024 conference papers, 2024, pp. 1â11.

[19] G. Pavlakos, V. Choutas, N. Ghorbani, T. Bolkart, A. A. Osman, D. Tzionas, M. J. Black, Expressive body capture: 3d hands, face, and body from a single image, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 10975â10985.

[20] M. Kocabas, N. Athanasiou, M. J. Black, Vibe: Video inference for human body pose and shape

estimation, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 5253â5263.

[21] S.-Y. Su, F. Yu, M. ZollhÃ¶fer, H. Rhodin, A-nerf: Articulated neural radiance fields for learning human shape, appearance, and pose, Advances in neural information processing systems 34 (2021) 12278â12291.

[22] W. Jiang, K. M. Yi, G. Samei, O. Tuzel, A. Ranjan, Neuman: Neural human radiance field from a single video, in: European Conference on Computer Vision, Springer, 2022, pp. 402â418.

[23] Y. Xiu, J. Yang, X. Cao, D. Tzionas, M. J. Black, Econ: Explicit clothed humans optimized via normal integration, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 512â523.

[24] J. Pan, X. Li, J. Bai, J. Dai, Litenerfavatar: A lightweight nerf with local feature learning for dynamic human avatar, Pattern Recognition (2025) 112008.

[25] Z. Huang, S. M. Erfani, S. Lu, M. Gong, Efficient neural implicit representation for 3d human reconstruction, Pattern Recognition 156 (2024) 110758.

[26] Z. Li, Z. Zheng, L. Wang, Y. Liu, Animatable gaussians: Learning pose-dependent gaussian maps for high-fidelity human avatar modeling, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 19711â19722.

[27] Z. Shao, Z. Wang, Z. Li, D. Wang, X. Lin, Y. Zhang, M. Fan, Z. Wang, Splattingavatar: Realistic real-time human avatars with meshembedded gaussian splatting, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1606â1616.

[28] A. Moreau, J. Song, H. Dhamo, R. Shaw, Y. Zhou, E. PÃ©rez-Pellitero, Human gaussian splatting: Real-time rendering of animatable avatars, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 788â798.

[29] S. Hu, T. Hu, Z. Liu, Gauhuman: Articulated gaussian splatting from monocular human videos, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 20418â20431.

[30] Z. Qian, S. Wang, M. Mihajlovic, A. Geiger, S. Tang, 3dgs-avatar: Animatable avatars via deformable 3d gaussian splatting, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 5020â5030.

[31] G. Moon, T. Shiratori, S. Saito, Expressive wholebody 3d gaussian avatar, in: European Conference on Computer Vision, Springer, 2024, pp. 19â35.

[32] D. Zhang, Y. Liu, L. Lin, Y. Zhu, Y. Li, M. Qin, Y. Li, H. Wang, Guava: Generalizable upper body 3d gaussian avatar, arXiv preprint arXiv:2505.03351 (2025).

[33] L. Qiu, S. Zhu, Q. Zuo, X. Gu, Y. Dong, J. Zhang, C. Xu, Z. Li, W. Yuan, L. Bo, et al., Anigs: Animatable gaussian avatar from a single image with inconsistent gaussian reconstruction, in: Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 21148â21158.

[34] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., Learning transferable visual models from natural language supervision, in: International conference on machine learning, PmLR, 2021, pp. 8748â8763.

[35] J. Kerr, C. M. Kim, K. Goldberg, A. Kanazawa, M. Tancik, Lerf: Language embedded radiance fields, in: Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 19729â19739.

[36] K. Abou Zeid, K. Yilmaz, D. de Geus, A. Hermans, D. Adrian, T. Linder, B. Leibe, Dino in the room: Leveraging 2d foundation models for 3d segmentation, arXiv e-prints (2025) arXivâ2503.

[37] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari, S. You, Z. Wang, A. Kadambi, Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21676â21685.

[38] S. Tang, Y. Wang, L. Chen, Y. Wang, S. Peng, D. Xu, W. Ouyang, Human-centric foundation models: Perception, generation and agentic modeling, arXiv preprint arXiv:2502.08556 (2025).

[39] Y. Zhi, W. Sun, J. Chang, C. Ye, W. Feng, X. Han, Strugauavatar: Learning structured 3d gaussians for animatable avatars from monocular videos,

IEEE Transactions on Visualization and Computer Graphics (2025).

[40] T. MÃ¼ller, A. Evans, C. Schied, A. Keller, Instant neural graphics primitives with a multiresolution hash encoding, ACM transactions on graphics (TOG) 41 (4) (2022) 1â15.

[41] M. Mihajlovic, Y. Zhang, M. J. Black, S. Tang, Leap: Learning articulated occupancy of people, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 10461â10471.

[42] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, X. Wang, 4d gaussian splatting for real-time dynamic scene rendering, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 20310â20320.

[43] Z. Yang, H. Yang, Z. Pan, L. Zhang, Realtime photorealistic dynamic scene representation and rendering with 4d gaussian splatting, arXiv preprint arXiv:2310.10642 (2023).

[44] L. Qingming, Y. Liu, J. Wang, X. Lyu, P. Wang, W. Wang, J. Hou, Modgs: Dynamic gaussian splatting from casually-captured monocular videos with depth priors, in: The Thirteenth International Conference on Learning Representations, 2025.

[45] S. Peng, J. Dong, Q. Wang, S. Zhang, Q. Shuai, X. Zhou, H. Bao, Animatable neural radiance fields for modeling dynamic human bodies, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 14314â 14323.

[46] S. Wang, K. Schwarz, A. Geiger, S. Tang, Arah: Animatable volume rendering of articulated human sdfs, in: European conference on computer vision, Springer, 2022, pp. 1â19.

[47] C. Geng, S. Peng, Z. Xu, H. Bao, X. Zhou, Learning neural volumetric representations of dynamic humans in minutes, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 8759â8770.

[48] T. Jiang, X. Chen, J. Song, O. Hilliges, Instantavatar: Learning avatars from monocular video in 60 seconds, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 16922â16932.

[49] T. Alldieck, M. Magnor, W. Xu, C. Theobalt, G. Pons-Moll, Video based reconstruction of 3d people models, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 8387â8397.

[50] R. Li, S. Yang, D. A. Ross, A. Kanazawa, Ai choreographer: Music conditioned 3d dance generation with aist++, in: Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 13401â13412.