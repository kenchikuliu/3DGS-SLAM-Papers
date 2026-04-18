# Realistic and Controllable 3D Gaussian-Guided Object Editing for Driving Video Generation

Jiusi Li1, Jackson Jiang2, Jinyu Miao1, Miao Long1, Tuopu Wen1, Peijin Jia1, Shengxiang Liu2 Chunlei Yu2, Maolin Liu2, Yuzhan Cai3, Kun Jiang1â, Mengmeng Yang1, Diange Yang1â

AbstractâCorner cases are crucial for training and validating autonomous driving systems, yet collecting them from the real world is often costly and hazardous. Editing objects within captured sensor data offers an effective alternative for generating diverse scenarios, commonly achieved through 3D Gaussian Splatting or image generative models. However, these approaches often suffer from limited visual fidelity or imprecise pose control. To address these issues, we propose G2Editor, a framework designed for photorealistic and precise object editing in driving videos. Our method leverages a 3D Gaussian representation of the edited object as a dense prior, injected into the denoising process to ensure accurate pose control and spatial consistency. A scenelevel 3D bounding box layout is employed to reconstruct occluded areas of non-target objects. Furthermore, to guide the appearance details of the edited object, we incorporate hierarchical finegrained features as additional conditions during generation. Experiments on the Waymo Open Dataset demonstrate that G2Editor effectively supports object repositioning, insertion, and deletion within a unified framework, outperforming existing methods in both pose controllability and visual quality, while also benefiting downstream data-driven tasks.

Index Termsâautonomous driving, data generation.

## I. INTRODUCTION

Extensive data serves as a foundation for advanced Autonomous Driving (AD) systems. In recent years, large-scale public datasets have played a crucial role in AD model training and validation. However, real-world data often exhibits a longtail distribution [1], [2], making it difficult for models to handle rare yet critical corner cases, which are costly and risky to collect. This challenge has spurred increasing interest in synthetic data generation to supplement real-world data and enhance the robustness and safety of AD systems.

Although existing data generation methods can generate various driving scenarios, they struggle to produce highfidelity synthetic data with precise pose control. Some works leverage diffusion models to generate full-scene data conditioned on scene layouts (e.g., 3D bounding boxes, maps) and text, producing semantically aligned visual data from noise [3], [4]. Yet, these approaches struggle to enable fine-grained control over individual object poses and appearances, failing to support directional generation grounded in real-world data.

To address this, some methods focus on editing objects within captured data [5], [6]. They often frame the task as diffusionbased image inpainting, using 3D bounding boxes to control object poses. While diffusion priors can guide appearance via reference images, 3D box-based pose signals provide insufficient constraints for drastic pose changes, leading to inaccurate object geometry.

To ensure accurate geometric control for edited objects in driving scenes editing, some recent methods instead utilize neural rendering techniques, such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). Some methods like [7], [8] decompose driving scenes into background and foreground objects, modeling objects as individual 3DGS instances to enable accurate geometric control. However, such solution are prone to degradation due to limited and sparse vehicle-mounted views, causing rendering distortion under large pose variations. Additionally, the lack of explicit lighting modeling introduces fidelity issues, such as inconsistent illumination between edited objects and backgrounds.

To bridge precise control and high-fidelity synthesis, we introduce G2Editor, a framework for realistic and geometrically accurate object editing in driving scenes. By integrating the photorealistic appearance synthesis of diffusion models with the explicit geometric control of 3DGS, G2Editor achieves dual objectives: 1) Precise Pose Control: A geometric-aware control signal is proposed that combines projected 3D bounding boxes and rendered 3D Gaussian models onto the image plane, providing dense and accurate spatial positional cues to guide the diffusion denoising process. 2) Appearance Fidelity and Temporal Consistency: Appearance details are captured from the reference image, which are hierarchically injected into the denoising UNet via spatial attention. Temporal layers are trained separately to ensure content consistency across frames. Experiments on Waymo Open Dataset [9] demonstrate that G2Editor enables photorealistic object repositioning, insertion, and deletion, as shown in Fig. 1, outperforming state-ofthe-art methods in pose control accuracy and visual fidelity. Our contributions can be summarized as follows:

â¢ We propose G2Editor, an framework for driving video editing that supports realistic object manipulation.

â¢ We introduce a hybrid pose control strategy combining 3D boxes and 3D Gaussian, ensuring precise control and spatial coherence during editing. A scene-level 3D bounding box layout is employed to effectively infer and recover occluded parts of non-target objects.

â¢ Experiments on the public dataset demonstrate the stateof-the-art performance of G2Editor in terms of pose accuracy and visual fidelity. The generated data can boost downstream AD tasks by providing high-quality and diverse driving scenes.

<!-- image-->  
Fig. 1. G2Editor enables realistic and controllable object editing, including repositioning, insertion and deletion.

## II. RELATED WORKS

Full-scene Driving Video Generation: To tackle the diversity limitations of real world driving data, researchers have employed generative techniques to expand the variety of driving videos. Many efforts focus on controllable full-scene generation that synthesizes street-view images [10], [11] or driving videos [3], [4] aligned with semantic driving scenarios represented via 3D layouts and text prompts. However, these semantic-level controls lack fine-grained object-level constraints, thereby limiting the ability to precisely manipulate individual objects. To enhance object-level manipulation, SubjectDrive [12] introduces a subject bank mechanism. Despite the progress in scalable AD data generation, these methods still struggle to achieve detailed control over specific objects, failing to support directional editing grounded in real-world data for high-fidelity synthesis of long-tail scenarios.

Diffusion Models for Image and Video Editing: Some recent works have shifted the focus from full-scene visual data generation to object-level editing in driving videos. With the rapid development of Stable Diffusion (SD) [13], many diffusion-based models support image editing driven by text [14], [15] or reference images [16], [17]. Although these methods achieve seamless image editing at the 2D level, they lack precise control over object pose and spatial consistency in 3D space, limiting their applicability in AD scenarios. Recently, inspired by these works, GenMM [5] adopts the inpainting framework to tackle driving video editing, enabling object insertion based on a single reference image. DriveEditor [6] further introduces a unified framework supporting multiple editing tasks. However, existing methods still face challenges in accurate pose control and appearance maintenance. To this end, we explicitly incorporate the 3D Gaussian to enhance pose accuracy and improve spatial coherence.

NeRF and 3D Gaussian Editing: With the development of NeRF [18] and 3DGS [19], numerous NeRF-based [20], [21] and 3DGS-based [7], [8] methods have been developed to separately represent dynamic foregrounds and static backgrounds, enabling object editing and simulation in driving scenarios. OmniRe [8], for instance, decomposes static backgrounds, vehicles, and dynamic actors into separate nodes to facilitate object editing. While these methods enable precise object pose control, they face challenges in flexible object editing due to inherent overfitting to training views, along with fidelity issues such as inconsistent lighting and poor shadow rendering. Other approaches like Lift3D [22] and GINA-3D [23] use image synthesis networks or NeRF to construct full 3D assets and integrate them into driving scenes. These approaches also suffer from visual artifacts, most notably in inconsistent illumination and misaligned shadows at object boundaries. Our method incorporates 3D Gaussian information into the diffusion-based framework to enhance physical and visual realism.

## III. METHOD

## A. Problem Formulation: Object Editing as Video Editing

Preliminaries: Stable Diffusion [13]. Our framework is based on SD, which encodes images in a latent space. A denoising UNet learns to predict the added noise Ïµ to the image latent z. This optimization objective can be formulated as:

$$
\mathcal { L } = \mathbb { E } _ { \mathbf { z } _ { t } , \mathbf { c } _ { c } , \epsilon , t } [ \| \epsilon - \epsilon _ { \theta } ( \mathbf { z } _ { t } , \mathbf { c } _ { c } , t ) \| _ { 2 } ^ { 2 } ] ,\tag{1}
$$

where $\mathbf { z } _ { t }$ is the noisy latent at step t and $\mathbf { c } _ { c }$ represents conditional embeddings. During inference, a latent $\mathbf { z } _ { T }$ is sampled from Gaussian noise and progressively denoised to $\mathbf { z } _ { 0 }$ , which is then decoded to obtain the generated image.

Generally, image inpainting aims to fill masked regions of an image with reasonable content, enabling object insertion and deletion. Recently, some inpainting methods are built upon SD to enable photorealistic editing conditioned on text or reference images. Object editing in driving videos, which similarly involves repositioning, insertion and deletion, can be approximated as video inpainting. These inpainting UNets take as input the concatenation of the latent noise $\mathbf { z } _ { t }$ , the Variational Auto-Encoders (VAE) features of the background video with grayed-out foreground pixels $E ( \mathbf { V } _ { b g } )$ , and masks M. E is the VAE encoder. However, in driving videos, object editing emphasizes the control and coherence of the object pose, as well as the maintenance of appearance details. Therefore, we formulate object editing task based on inpainting task:

<!-- image-->  
Fig. 2. The overview of G2Editor. A diffusion-based inpainting framework that includes object pose control and object appearance maintenance. This framework takes as input the assets of the edited object (the reference image and 3D Gaussian model), scene-level 3D boxes and the masked video, and outputs the edited video.

$$
\begin{array} { r } { \mathcal { L } = \mathbb { E } _ { \mathbf { z } _ { t } , \mathbf { c } _ { p } , \mathbf { c } _ { a } , \epsilon , t } ( \left| \left| \epsilon - \epsilon _ { \theta } ( \mathbf { z } _ { t } , \mathbf { c } _ { p } , \mathbf { c } _ { a } , t ) \right| \right| _ { 2 } ^ { 2 } ) , } \end{array}\tag{2}
$$

where $\mathbf { c } _ { p }$ denotes pose-related conditions and $\mathbf { c } _ { a }$ denotes appearance-related ones. Existing methods leverage 3D bounding boxes of the edited object for pose control and object image assets for appearance maintenance, achieving some success but still facing limitations. To enable more flexible and precise editing, we propose a pose control strategy that incorporates information from a 3D Gaussian model and scenelevel 3D boxes (Sec. III-B), and maintain appearance using reference image features with random flipping (Sec. III-C). Our inpainting framework is shown in Fig. 2.

## B. Object Pose Control

To enable precise control over object position and orientation, we introduce scene-level 3D boxes and the 3D Gaussian model as pose-related conditions $\mathbf { c } _ { p }$

We employ depth-aware boxes, similar to [6]. To maintain the layout of non-target objects while precisely controlling the edited objectâs poses, we use scene-level 3D boxes instead of boxes of the single edited object in [6]. Specifically, each face of boxes is processed separately by projecting corner depths onto the image plane and interpolating within the face to form depth-aware boxes $\mathbf { D } _ { b }$ . These are encoded via a ResNet-style depth embedder to extract multi-scale features $f _ { l } ,$ , which are injected into ResBlocks of self-attention layers through a zeroinit fusion layer:

$$
\nu _ { l } = \nu _ { l } + \mathrm { F u s i o n l a y e r } ( f _ { l } ) ,\tag{3}
$$

where $\nu _ { l }$ is the feature map of l-th block, and the fusion layer consists of layer normalization, SiLU activation, and convolution operation. We also project the scene-level 3D boxes onto the image plane as edge masks $\mathbf { M } _ { b }$

To provide dense and accurate positional cues, we introduce a 3D Gaussian model beyond 3D bounding boxes. Specifically, we start with the 3D Gaussian asset of the edited object, which can be obtained from multiview reconstruction or image-to-3D generation. Each Gaussian is defined by its mean $\mu ,$ rotation R, scale S, opacity Î± and color o in a local coordinate. We use the object pose {W, T} to transform 3D Gaussians from local to world coordinate:

$$
{ \pmb { \mu } } _ { w } = { \bf W } { \pmb { \mu } } + { \bf T } ,
$$

$$
\mathbf { R } _ { w } = \mathbf { W } ^ { T } \mathbf { R } .\tag{4}
$$

(5)

The 3D Gaussians are then projected onto the 2D image plane using camera extrinsic and intrinsic. The rendered image of the edited object is computed by blending N ordered Gaussians through the Î±-rendering technique [19]:

$$
\mathbf { I } ( \mathbf { p } ) = \sum _ { i = 1 } ^ { N } \mathbf { o } _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } \left( 1 - \alpha _ { j } \right) ,\tag{6}
$$

where p is the pixel of the rendering image. All rendered images make up the objectâs Gaussian video $\mathbf { V } _ { g } = \{ \mathbf { I } _ { 0 } , . . . , \mathbf { I } _ { t } \}$

Since the Gaussian video is spatially aligned with the edited video, we concatenate their latents. Along with latent noise $\mathbf { z } _ { t } ,$ the latent of the masked video $E ( \mathbf { V } _ { b g } )$ and masks M used in the inpainting task, we concatenate the latent of the Gaussian video $E ( \mathbf { V _ { g } } )$ and edge masks $\mathbf { M } _ { b } ,$ , yielding input for the inpainting UNet.

Some methods [5], [6] directly feed reference images and target object poses to the diffusion process, requiring the model to correctly understand the 3D scene geometry and the viewpoint transformation, an inherently challenging task. In contrast, taking the renderings of the edited object as input, G2Editor edits on slightly blurred Gaussian images, which is more akin to Gaussian image restoration [24]. This significantly simplifies the diffusion modelâs training objective and enhances control over object poses. Moreover, the Gaussian model explicitly ensures the 3D spatial coherence.

In summary, our pose-related conditions $\mathbf { c } _ { p }$ in (2) comprise Gaussian video $\mathbf { V } _ { g } ,$ edge masks of scene-level boxes $\mathbf { M } _ { b }$ and depth-aware boxes $\mathbf { D } _ { b }$

## C. Object Appearance Maintenance

Object editing requires the maintenance and consistency of appearance details. Many image-driven generation methods use the CLIP image encoder to extract features, which are injected into the diffusion process via cross-attention. However, as noted in [25], CLIP effectively captures highlevel information but lacks the ability to preserve fine-grained details. Following [5], we complement CLIP features $\mathbf { c } _ { c }$ with a ReferenceNet that shares the architecture of the original SD, leveraging its pre-trained capability for extracting image features. Since the reference image and latent noise are not spatially aligned and only parts of latent noise require information from the reference, naive injection strategies such as concatenation or addition are suboptimal. Therefore, we replace the self-attention layers in the inpainting UNet with spatial-attention layers to selectively attend to relevant features. Specifically, spatial attention is performed over the concatenation of feature maps (Î½ and $\mathbf { c } _ { r } )$ from the inpainting UNet and ReferenceNet:

$$
\nu = \nu + \mathrm { T S } ( \mathrm { S e l f A t t n } ( [ \nu , \mathbf { c } _ { r } ] ) ) ,\tag{7}
$$

where TS(Â·) is a token selection operation that only retains the features from the inpainting UNet. The reference feature $\mathbf { c } _ { r }$ is replicated t times along the temporal dimension and concatenated with Î½ along the spatial dimension. Overall, the appearance-related condition $\mathbf { c } _ { a }$ in (2) comprises CLIP features $\mathbf { c } _ { c }$ and ReferenceNet features $\mathbf { c } _ { r }$ from the reference image.

Moreover, we observe that selecting a single image from the video clip as the reference image done in [5] may lead the diffusion model to learn pose from the ReferenceNet. As a result, during inference, the model tends to overfit to the pose in the reference image, limiting pose control despite appearance maintenance. Simple random horizontal flipping of the reference image can prevent the diffusion model from learning object pose information from ReferenceNet.

## D. Training Strategy

1) Data Preparation: Object repositioning and insertion can be defined as reconstructing the masked object in the video given the object asset and 3D bounding boxes. To train our inpainting framework, we prepare the dataset. Specifically, we select N frames as a video $\dot { \mathbf { V } } { \in } \mathbb { R } ^ { N \times 3 \times H \times W }$ , each satisfying the following requirements: The edited object appears in all frames with sufficient size, and fewer than two other objects are within 3m around (less occlusion). Each video has a set of 3D bounding boxes. To support object translation and rotation, the 3D bounding boxes are projected onto the image plane and enlarged appropriately to form masks $\mathbf { M } { \in } \mathbb { R } ^ { N \times H \times W }$ , resulting in the masked video $\mathbf { V } _ { b g }$ . Reference images of a clip are obtained by cropping square regions from frames in which the edited object is fully visible. To enable object deletion and realistic inpainting of the background within the masked regions, random masks are applied to regions without objects in images and incorporated into the training data. And we use white images as Gaussian renderings and the reference image for object deletion.

2) Two Stage Training: Our framework adopts a twostage training strategy. In the first stage, the model learns to paint the specified object within the masked region of a single image according to the pose and appearance conditions. The optimization objective is defined in (2), where poserelated conditions include Gaussian video, edge masks and depth-aware boxes, and appearance-related conditions consist of ReferenceNet features and CLIP features. To prevent the model from directly learning pose or background cues from the reference image, we select the reference and edited frames from the same clip with the largest temporal distance. At inference, the reference image is from external sources, and may differ significantly in lighting from the edited video. Therefore, we apply brightness, contrast and saturation augmentations to the reference image, enabling the model to infer lighting conditions from the unmasked background. In the second stage, we aim to enhance the temporal consistency of the edited region and restore the occluded background. Following [5], [25], we incorporate temporal-attention layers after the cross-attention layers in the inpainting UNet, which are selfattention along the temporal dimension. At this stage, we only train the temporal layers.

## IV. EXPERIMENTS

## A. Experimental Setup

1) Dataset: We construct a training dataset based on the Waymo Open Dataset [9]. This autonomous driving dataset comprises 798 training scenes, each providing surround-view images along with 3D bounding box annotations at 10Hz. We use images from the three front-view cameras and resize them to 640Ã960. We set N = 10. After the processing described in Sec. III-D, 19477 video clips are obtained for training. We also use images whose object-free regions are randomly masked at the first stage training with a probability of 0.2.

2) Baselines: Our work focuses on object editing in driving videos, emphasizing pose control and visual fidelity. There are limited comparable methods, and evaluation approaches also require further exploration. To evaluate fine-grained pose control and visual fidelity in edited regions, we design benchmarks for repositioning, insertion and deletion.

For repositioning, we apply the same data preparation to the Waymo validation set. For each target object, we perform three manipulations to assess fine-grained pose control:

<!-- image-->  
Fig. 3. Visualization of rotation and translation. In the first row, black and orange boxes denote the object before and after manipulation. For rotation, G2Editor achieves precise yaw control while maintaining appearance, outperforming 3DGS-based method and other video editing methods prone to artifacts or poor pose control. For translation, G2Editor enables precise pose control and inpaints reasonable background, whereas other methods often leave residual artifacts.

reinsertion (masking the object and inserting it at the same position), clockwise rotation by Î² degrees and one-meter leftward translation. We compare our method with the 3DGSbased method OmniRe [8], as well as two driving video editing methods, GenMM [5] and DriveEditor [6]. For OmniRe, to enable editing of static vehicles, we model static and dynamic vehicles as rigid nodes. Note that the Gaussian videos used in our method are also derived from OmniRe. We use our own implementation of GenMM, aligned with our setup, and apply the official model for DriveEditor inference.

For insertion, we evaluate the visual fidelity, focusing on lighting and shadows. We use the SOTA image-to-3D method, TRELLIS [26], to generate the 3D Gaussian model of the object from the reference image, scaling it to align with aggregated LiDAR points. These models are inserted into reconstructed scenes based on existing object trajectories, and the rendered video clips serve as a baseline. For our method, the Gaussian videos of the edited object provide guidance.

For deletion, we conduct a qualitative comparison with the

3DGS-based method.

3) Metrics: For repositioning, we evaluate both the pose control accuracy and appearance quality. We select 8 challenging scenes from Waymo validation set, which contain editable objects and complex environmental conditions. 180 video clips are obtained for evaluation. We set Î² = 5â¦. For object pose control accuracy, we apply the PGD model [27], a pre-trained vision-centric 3D object detector, to the edited videos. We use the LET metrics [28] (LET-mAP, LET-mAPH, LET-mAPL) provided by Waymo and set 5% longitudinal error tolerance. As scene-level (full-image) metrics are not sensitive to edited object detection, we only consider the ground-truth/detection pairs of the edited instances. For appearance quality, we evaluate only the square region around the edited object to focus on the modifications. We resize the square images to 512 Ã 512 and utilize frame-wise FID [29] and LPIPS [30] between the edited and the original images.

For insertion, we evaluate visual fidelity. We use 3 Gaussian assets generated from TRELLIS and insert them into 9 scenes, resulting in 102 video clips. Since there is no ground-truth for the objectâs appearance in the target video, we consider the reference image and the cropped square images. We also use FID and LPIPS.

TABLE I  
QUANTITATIVE RESULTS FOR POSE CONTROL ON OBJECT REPOSITIONING
<table><tr><td></td><td colspan="3">reinsertion</td><td colspan="3">rotation 5</td><td colspan="3">translation 1m</td></tr><tr><td></td><td>LET-mAPâ</td><td>LET-mAPHâ</td><td>LET-mAPLâ</td><td>LET-mAPâ</td><td>LET-mAPHâ</td><td>LET-mAPLâ</td><td>LET-mAPâ</td><td>LET-mAPHâ</td><td>LET-mAPLâ</td></tr><tr><td>OmniRe [8]</td><td>0.736</td><td>0.727</td><td>0.474</td><td>0.724</td><td>0.714</td><td>0.478</td><td>0.678</td><td>0.668</td><td>0.419</td></tr><tr><td>DriveEditor [6]</td><td>0.562</td><td>0.546</td><td>0.318</td><td>0.532</td><td>0.511</td><td>0.306</td><td>0.280</td><td>0.266</td><td>00.152</td></tr><tr><td>enM [5]</td><td>0.715</td><td>0.704</td><td>0.444</td><td>0.726</td><td>0.709</td><td>0.440</td><td>0.416</td><td>0.408</td><td>0.217</td></tr><tr><td>Ours</td><td>0.781</td><td>0.772</td><td>0.504</td><td>0.806</td><td>0.794</td><td>0.517</td><td>0.725</td><td>0.715</td><td>0.463</td></tr></table>

TABLE II

QUANTITATIVE RESULTS FOR APPEARANCE MAINTENANCE ON OBJECT REPOSITIONING
<table><tr><td></td><td colspan="2">reinsertion</td><td colspan="2">rotation 5</td><td colspan="2">translation 1m</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>LPRefâFID-RefâLPIPSRefâFID-RefâLPIPSRefâFID-Refâ</td><td></td></tr><tr><td>OmniRe [8]</td><td>0.162</td><td>27.297</td><td>0.231</td><td>30.610</td><td>0.460</td><td>37.712</td></tr><tr><td>DriveEditor [6]</td><td>0.274</td><td>25.994</td><td>0.276</td><td>25.659</td><td>0.516</td><td>33.435</td></tr><tr><td>GenMM [5]</td><td>0.150</td><td>12.603</td><td>0.158</td><td>12.968</td><td>0.456</td><td>24.658</td></tr><tr><td>Ours</td><td>0.151</td><td>13.240</td><td>0.207</td><td>14.222</td><td>0.440</td><td>19.745</td></tr></table>

4) Training Setup: We train the first stage for 30,000 iterations using a batch size of 8 per GPU on 4 A800 GPUs, and the second stage for 10,000 iterations using a batch size of 1 clip per GPU on 8 A800 GPUs. We initialize our inpainting UNet and ReferenceNet with SD v1.4-image-variations, and the extra 10 channels in the first layer of inpainting UNet are initialized to zero. We initialize the temporal attention layers with the pretrained motion module in [31].

## B. Performance

1) Repositioning Objects: Fig. 3 shows qualitative editing results of 5â¦ rotation (left 2 columns) and 1m translation (right 2 columns) of the target object. For rotation, the 3DGS-based method can accurately control the rotation angle, but often produces blurring and artifacts due to overfitting to training views during Gaussian optimization. As shown in Fig. 6, this issue becomes increasingly evident as the rotation angle increases. Other video editing methods struggle to respond to pose control inputs. For frames with large pose variations or those temporally distant from the reference image, DriveEditor fails to maintain object appearance. Our method enables precise control over the yaw of the edited object while maintaining high-fidelity appearance. For translation, DriveEditor fails to perform accurate movement, while GenMM and OmniRe can not remove objects from the original location. Our method enables precise control of object pose and inpaints reasonable background at the original location.

Table I and Table II report the quantitative performance of our method and baselines on object pose control and appearance maintenance, respectively. In this experiment, our method significantly outperforms baselines on pose control over the edited objects. For appearance maintenance, our method greatly surpasses OmniRe and DriveEditor, and is on par with GenMM. We attribute this to the fact that GenMM generates parts of the images, which have higher resolution (512 Ã 512) for evaluated regions during inference.

<!-- image-->  
Fig. 4. Based on image-to-3D generation method, G2Editor is capable of synthesizing realistic shadows during object insertion.

<!-- image-->  
Fig. 5. Visualization of object deletion. Compared to the 3DGS-based method, G2Editor enables more realistic object deletion and reasonable background completion, particularly for static objects.

2) Inserting Objects: Fig. 4 shows qualitative results of object insertion using Gaussian models from TRELLIS directly and our Gaussian-guided method. Directly inserting 3D Gaussian models does not model the light source of the scene and cannot render realistic shadows. Our method can synthesize shadows and generate more realistic videos. For quantitative comparison, we measure the LPIPS-Ref metric for both TRELLIS and our method. Our method achieves 0.612 while TRELLIS is 0.639, showing better visual fidelity.

3) Deleting Objects: Fig. 5 shows qualitative results of object deletion. For OmniRe, object deletion is achieved by removing the targetâs Gaussian model when rendering, but this introduces undesirable background distortions (row 1). And the separation of static foreground and background makes static object deletion challenging (row 2). G2Editor enables more realistic object deletion and reasonable background completion, particularly for static objects.

TABLE III  
ABLATION STUDY ON THE REPOSITIONING TASK
<table><tr><td rowspan="3">rreer</td><td rowspan="3">i wop</td><td rowspan="3">Sere-et</td><td rowspan="3">sexos</td><td rowspan="3">Coe</td><td colspan="6">Object Pose Control</td><td colspan="4">Object Appearance Maintenance</td></tr><tr><td colspan="3">rotation 5</td><td colspan="3">translation 1m</td><td colspan="2">rotation 50</td><td colspan="2">translation 1m</td></tr><tr><td>LET-mAPâ</td><td>LET-mAPHâ</td><td>LET-mAPLâ</td><td>LET-mAPâ</td><td>LET-mAPHâ</td><td>LET-mAPLâ</td><td>LPIPS-Refâ</td><td>FID-Refâ</td><td>LPIPS-Refâ</td><td>FID-Refâ</td></tr><tr><td>SDÎµ</td><td></td><td>â</td><td>â</td><td>â</td><td>0.806</td><td>0.794</td><td>0.517</td><td>0.725</td><td>0.715</td><td>0.463</td><td>0.207</td><td>14.222</td><td>0.440</td><td>19.745</td></tr><tr><td>&gt;&gt;</td><td>â â</td><td>â</td><td>â</td><td>Ã</td><td>0.798</td><td>0.786</td><td>0.510</td><td>0.709</td><td>0.701</td><td>0.451</td><td>0.213</td><td>15.397</td><td>0.440</td><td>20.240</td></tr><tr><td>â</td><td>â</td><td>â</td><td>Ã</td><td>Ã</td><td>0.780</td><td>0.767</td><td>0.493</td><td>0.706</td><td>0.697</td><td>0.437</td><td>0.213</td><td>15.530</td><td>0.441</td><td>20.211</td></tr><tr><td>Ã</td><td>â</td><td>â</td><td>Ã</td><td>Ã</td><td>0.668</td><td>0.649</td><td>0.396</td><td>0.416</td><td>0.407</td><td>0.212</td><td>0.201</td><td>14.289</td><td>0.471</td><td>19.804</td></tr><tr><td>â</td><td>â</td><td>Ã</td><td>Ã</td><td>Ã</td><td>0.776</td><td>0.762</td><td>0.495</td><td>0.692</td><td>0.683</td><td>0.439</td><td>0.214</td><td>16.369</td><td>0.441</td><td>21.144</td></tr><tr><td>â</td><td>Ã</td><td>Ã</td><td>Ã</td><td>Ã</td><td>0.782</td><td>0.770</td><td>0.482</td><td>0.706</td><td>0697</td><td>0.441</td><td>0.258</td><td>118.713</td><td>0.457</td><td>22.537</td></tr></table>

TABLE IV

EFFECT OF EDITED DATA ON DETECTOR PERFORMANCE UNDER THREE-VIEW CAMERA SETTING
<table><tr><td rowspan="2">Pretraining (25e) Real data</td><td colspan="2">Finetuning (+5e)</td><td colspan="3">Front</td><td colspan="3">Three Front Camera</td></tr><tr><td>Real data</td><td>Edited data</td><td>LET-mAPâ</td><td>LET-mAPHâ</td><td>LET-mAPLâ</td><td>LET-mAPâ</td><td>LET-mAPHâ</td><td>LET-mAPLâ</td></tr><tr><td></td><td></td><td></td><td>0.612</td><td>0.606</td><td>0.453</td><td>0.536</td><td>0.529</td><td>0.395</td></tr><tr><td>&gt;&gt;</td><td>â</td><td></td><td>0.624</td><td>0.618</td><td>0.463</td><td>0.554</td><td>0.547</td><td>0.409</td></tr><tr><td>â</td><td></td><td>â</td><td>0.628</td><td>0.622</td><td>0.466</td><td>0.556</td><td>0.548</td><td>0.411</td></tr></table>

<!-- image-->  
Fig. 6. Effectiveness of 3DGS. By leveraging the dense and accurate spatial positional cues provided by 3DGS, our method achieves precise control over the target objectâs rotation, despite slightly blurred Gaussian renderings.

## C. Ablation Study

To verify the effectiveness of each component, we conduct ablation studies on the repositioning task, evaluating both pose control and appearance maintenance. We first ablate the temporal layers to obtain an image editing framework. Based on this, we further ablate scene-level boxes, 3DGS renderings, the horizontal flipping of the reference image, and ReferenceNet. As shown in Table III, the 3DGS rendering has a significant effect on the object pose control, whereas ReferenceNet is notably effective on the object appearance maintenance. Note that the model without the 3DGS rendering has better LPIPS when the edited objects are rotated by 5â¦. We attribute this to poor pose control. We find that the edited objects undergo minimal rotation in this setting, resulting in outputs more similar to original images. Additionally, random flipping the reference image horizontally indeed improves pose control accuracy by discouraging the framework from relying on ReferenceNet for object pose.

<!-- image-->  
Fig. 7. Effectiveness of scene-level boxes and temporal layers. Temporal layers and scene-level boxes contribute to the completion of occluded regions of non-target objects in each frame on object deletion.

Fig. 6 illustrates the effectiveness of 3DGS renderings in pose control. As shown in Fig.7, scene-level boxes and temporal layers facilitate the completion of occluded regions of non-target objects in each frame on object deletion.

## D. Training Support for 3D Object Detection

We generate edited data by repositioning (translating and rotating) objects in selected clips of Waymo training data, which is suitable for editing, aiming to enhance the training of the monocular object detector PGD [27]. PGD is trained for 25 epochs following the official setting, and then fine-tuned for 5 additional epochs with real data combined with edited data on three front-view cameras. We copy the corresponding real data to ensure the same total data volume for fair comparison. We evaluate performance using the official Waymo scenelevel LET metrics [28]. As shown in Table IV, repositioning effectively expands the viewpoint distribution of objects and increases data diversity, thereby improving detection performance compared to simple duplication of real data.

## V. CONCLUSION

We propose G2Editor, a diffusion-based framework that supports object repositioning, insertion and deletion in driving videos. To enable precise control over object pose, we introduce a hybrid control strategy combining 3DGS renderings and 3D boxes as pose-related conditions. And we hierarchically inject visual features from a random-flipped reference image into the diffusion process to maintain the edited objectâs appearance. We design fine-grained experiments to evaluate the accuracy of pose control and visual fidelity within the edited region, demonstrating that G2Editor outperforms existing methods and benefits downstream AD tasks.

## REFERENCES

[1] L. Le Mero, D. Yi, M. Dianati, and A. Mouzakitis, âA survey on imitation learning techniques for end-to-end autonomous vehicles,â IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 9, pp. 14128â14147, 2022.

[2] M. GarcÂ´Ä±a, A. Iglesias, M. Sanchez, R. Naranjo, J. A. I. De Gordoa, Â´ M. Nieto, and N. Aginako, âSynthetic dataset generation using logical scenario files for automotive perception testing,â in 2025 IEEE Intelligent Vehicles Symposium (IV), pp. 392â397, IEEE, 2025.

[3] W. Wu, X. Guo, W. Tang, T. Huang, C. Wang, and C. Ding, âDrivescape: High-resolution driving video generation by multi-view feature fusion,â in Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 17187â17196, 2025.

[4] Y. Wen, Y. Zhao, Y. Liu, F. Jia, Y. Wang, C. Luo, C. Zhang, T. Wang, X. Sun, and X. Zhang, âPanacea: Panoramic and controllable video generation for autonomous driving,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6902â 6912, 2024.

[5] B. Singh, V. Kulharia, L. Yang, A. Ravichandran, A. Tyagi, and A. Shrivastava, âGenmm: Geometrically and temporally consistent multimodal data generation for video and lidar,â 2024.

[6] Y. Liang, Z. Yan, L. Chen, J. Zhou, L. Yan, S. Zhong, and X. Zou, âDriveeditor: A unified 3d information-guided framework for controllable object editing in driving scenes,â in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, pp. 5164â5172, 2025.

[7] X. Zhou, Z. Lin, X. Shan, Y. Wang, D. Sun, and M.-H. Yang, âDrivinggaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 21634â21643, 2024.

[8] Z. Chen, J. Yang, J. Huang, R. de Lutio, J. M. Esturo, B. Ivanovic, O. Litany, Z. Gojcic, S. Fidler, M. Pavone, et al., âOmnire: Omni urban scene reconstruction,â arXiv preprint arXiv:2408.16760, 2024.

[9] P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, et al., âScalability in perception for autonomous driving: Waymo open dataset,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2446â2454, 2020.

[10] A. Swerdlow, R. Xu, and B. Zhou, âStreet-view image generation from a birdâs-eye view layout,â IEEE Robotics and Automation Letters, 2024.

[11] Z. Jiang, J. Liu, M. Sang, H. Li, and Y. Pan, âCritical test cases generalization for autonomous driving object detection algorithms,â in 2024 IEEE Intelligent Vehicles Symposium (IV), pp. 1149â1156, IEEE, 2024.

[12] B. Huang, Y. Wen, Y. Zhao, Y. Hu, Y. Liu, F. Jia, W. Mao, T. Wang, C. Zhang, C. W. Chen, et al., âSubjectdrive: Scaling generative data in autonomous driving via subject control,â in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, pp. 3617â3625, 2025.

[13] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, âHighresolution image synthesis with latent diffusion models,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684â10695, 2022.

[14] S. Sheynin, A. Polyak, U. Singer, Y. Kirstain, A. Zohar, O. Ashual, D. Parikh, and Y. Taigman, âEmu edit: Precise image editing via recognition and generation tasks,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8871â 8879, 2024.

[15] J. Zhuang, Y. Zeng, W. Liu, C. Yuan, and K. Chen, âA task is worth one word: Learning with task prompts for high-quality versatile image inpainting,â in European Conference on Computer Vision, pp. 195â211, Springer, 2024.

[16] Y. Song, Z. Zhang, Z. Lin, S. Cohen, B. Price, J. Zhang, S. Y. Kim, and D. Aliaga, âObjectstitch: Object compositing with diffusion model,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18310â18319, 2023.

[17] B. Yang, S. Gu, B. Zhang, T. Zhang, X. Chen, X. Sun, D. Chen, and F. Wen, âPaint by example: Exemplar-based image editing with diffusion models,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 18381â18391, 2023.

[18] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[19] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering.,â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[20] Z. Wu, T. Liu, L. Luo, Z. Zhong, J. Chen, H. Xiao, C. Hou, H. Lou, Y. Chen, R. Yang, et al., âMars: An instance-aware, modular and realistic simulator for autonomous driving,â in CAAI International Conference on Artificial Intelligence, pp. 3â15, Springer, 2023.

[21] Z. Yang, Y. Chen, J. Wang, S. Manivasagam, W.-C. Ma, A. J. Yang, and R. Urtasun, âUnisim: A neural closed-loop sensor simulator,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1389â1399, 2023.

[22] L. Li, Q. Lian, L. Wang, N. Ma, and Y.-C. Chen, âLift3d: Synthesize 3d training data by lifting 2d gan to 3d generative radiance field,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 332â341, 2023.

[23] B. Shen, X. Yan, C. R. Qi, M. Najibi, B. Deng, L. Guibas, Y. Zhou, and D. Anguelov, âGina-3d: Learning to generate implicit neural assets in the wild,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4913â4926, 2023.

[24] C. Ni, G. Zhao, X. Wang, Z. Zhu, W. Qin, G. Huang, C. Liu, Y. Chen, Y. Wang, X. Zhang, et al., âRecondreamer: Crafting world models for driving scene reconstruction via online restoration,â in Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1559â 1569, 2025.

[25] L. Hu, âAnimate anyone: Consistent and controllable image-to-video synthesis for character animation,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8153â 8163, 2024.

[26] J. Xiang, Z. Lv, S. Xu, Y. Deng, R. Wang, B. Zhang, D. Chen, X. Tong, and J. Yang, âStructured 3d latents for scalable and versatile 3d generation,â arXiv preprint arXiv:2412.01506, 2024.

[27] T. Wang, Z. Xinge, J. Pang, and D. Lin, âProbabilistic and geometric depth: Detecting objects in perspective,â in Conference on Robot Learning, pp. 1475â1485, PMLR, 2022.

[28] W.-C. Hung, V. Casser, H. Kretzschmar, J.-J. Hwang, and D. Anguelov, âLet-3d-ap: Longitudinal error tolerant 3d average precision for cameraonly 3d detection,â in 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 8272â8279, IEEE, 2024.

[29] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, âGans trained by a two time-scale update rule converge to a local nash equilibrium,â Advances in neural information processing systems, vol. 30, 2017.

[30] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, âThe unreasonable effectiveness of deep features as a perceptual metric,â in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586â595, 2018.

[31] Y. Guo, C. Yang, A. Rao, Z. Liang, Y. Wang, Y. Qiao, M. Agrawala, D. Lin, and B. Dai, âAnimatediff: Animate your personalized text-toimage diffusion models without specific tuning,â in ICLR, 2024.