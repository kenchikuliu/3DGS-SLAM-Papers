# Monocular Endoscopic Tissue 3D Reconstruction with Multi-Level Geometry Regularization

Yangsen Chen

HKUST(GZ)

ychen950@connect.hkust-gz.edu.cn

Hao Wang\* \*

HKUST(GZ)

haowang@hkust-gz.edu.cn

AbstractâReconstructing deformable endoscopic tissues is crucial for achieving robot-assisted surgery. However, 3D Gaussian Splatting-based approaches encounter challenges in achieving consistent tissue surface reconstruction, while existing NeRFbased methods lack real-time rendering capabilities. In pursuit of both smooth deformable surfaces and real-time rendering, we introduce a novel approach based on 3D Gaussian Splatting. Specifically, we introduce surface-aware reconstruction, initially employing a Sign Distance Field-based method to construct a mesh, subsequently utilizing this mesh to constrain the Gaussian Splatting reconstruction process. Furthermore, to ensure the generation of physically plausible deformations, we incorporate local rigidity and global non-rigidity restrictions to guide Gaussian deformation, tailored for the highly deformable nature of soft endoscopic tissue. Based on 3D Gaussian Splatting, our proposed method delivers a fast rendering process and smooth surface appearances. Quantitative and qualitative analysis against alternative methodologies shows that our approach achieves solid reconstruction quality in both textures and geometries.

Index Termsâ3D Reconstruction, Gaussian Splatting, Robotic Surgery

## I. INTRODUCTION

The 3D reconstruction of surgical scenes from endoscope videos is a critical yet challenging task. It serves as a foundational element in the field of Robot-Assisted Surgery [1], enabling various essential clinical applications [2]â[4]. Despite significant progress in 3D reconstruction techniques for natural scenes [5], [6], endoscopic surgical field reconstruction from videos presents several unresolved challenges.

Firstly, surgical scenarios are typified by deformable structures undergoing significant topological transformations, necessitating the employment of dynamic reconstruction methodologies to accurately encapsulate non-rigidity. While recent advancements in 3D reconstruction techniques [7]â[10] have exhibited substantial strides in the domain of deformable endoscopic tissues, they exhibit a deficiency in surface-aware constraint, resulting in artifacts. To address this issue, we propose two methodologies: surface-aware reconstruction and semi-rigidity deformation. Our surface-aware reconstruction technique leverages mesh surface constraints to confine Gaussian distributions, thereby yielding seamless outcomes. Furthermore, our semi-rigidity deformation approach is developed based on the pretrained key point detection models. Based on the detected key points from the given images, we integrate local rigidity cues and global non-rigidity cues to steer Gaussian Splatting, enhancing reconstruction fidelity.

Secondly, the limited perspectives captured in endoscopic videos, constrained by camera movements, restrict the availability of 3D cues for reconstructing soft tissue. Furthermore, surgical instruments invariably obstruct portions of the soft tissue, impeding the comprehensive reconstruction of the surgical scene. To address this challenge, we leverage the existing 2D optical flow detection model. Technically, we produce optical flows from the endoscopic images, which are used to augment and guide the learning process for 3D reconstruction and deformation. To predict occluded regions based on spatialtemporal cues, we introduce a video inpainting module into our framework. It is notable that our pipeline initially estimates the optical flow of the videos and subsequently incorporates a video inpainting model. We conducted the video inpainting model fine-tuning with a surgical dataset [11]. To our best knowledge, we propose the first work in addressing the spatialtemporal masking issue within this endoscopic tissue 3D reconstruction context.

Moreover, despite the advancements in existing NeRF [5] based methods, which have achieved high-quality reconstruction, their limitations include long training time and slow rendering speeds. In contrast, our proposed novel Gaussian Splatting-based approach enables real-time rendering and significantly reduces training time. Compared to the concurrent 3D Gaussian-based methodologies, our technique yields solid reconstruction quality in both textures and geometries.

Our contributions in this work are threefold:

â¢ We propose the surface-aware endoscopic reconstruction, which integrates RGB, depth, and optical flow data, to achieve consistent and smooth geometry reconstructed results.

â¢ We propose semi-rigidity deformation guidance to model realistic Gaussian deformations through global and local motion learning, which avoids the 3D floaters during the 3D reconstruction process.

â¢ We propose a novel approach with multi-level regularization for 3D dynamic endoscopic tissue reconstruction, which demonstrates superior performance in both textures and geometries.

## II. RELATED WORKS

## A. Endoscopic 3D Reconstruction

Depth estimation based methods such as [12], [13] explored the effectiveness of surgical scene reconstruction via depth estimation. Since most of the endoscopes are equipped with stereo cameras, depth can be estimated from binocular vision. However these methods can not provide good deformable results.

<!-- image-->  
Fig. 1. Our methodology begins with the key point detection, followed by neighborhood identification and mesh based Gaussian Splatting reconstruction. Subsequently, global and local restriction on the deformation is executed to obtain the final dynamic output.

SLAM-based methods [14]â[18] fuse depth maps in 3D space to reconstruct surgical scenes under more complex settings. Nevertheless, these methods either hypothesize scenes as static or surgical tools not present, limiting their practical use in real scenarios.

Sparse warp field-based methods such as SuPer [19] and E-DSSR [20] present frameworks consisting of tool masking, stereo depth estimation to perform singleview 3D reconstruction of deformable tissues. All these methods track deformation based on a sparse warp field [21], which is not robust when deformations are significantly beyond the scope of nontopological changes.

With the development of Neural Radiance Field (NeRF) [5], learning based 3D reconstruction has been much more popular, recent works [8], [22]â[25] utilize NeRF for the reconstruction of endoscopic videos. However, due to the implicit representation nature of NeRF, the rendering speed is far from real time, hindering their real world applications.

Some simultaneous works [7], [9], [10], [26] also used Gaussian Splatting [6], while the surface reconstruction accuracy in endoscopic scenes remains challenging for real-world applications.

## III. METHODS

Given a stereo video of deforming tissues, we aim to reconstruct the surface shape S and texture C. Similar to EndoNeRF [8], we take a sequence of frame data $\left\{ ( \mathbf { I } _ { i } , \mathbf { D } _ { i } , \mathbf { M } _ { i } , \mathbf { P } _ { i } ) \right\} _ { i = 1 } ^ { T }$ as input. Here $T$ stands for the total number of frames. $\mathbf { I } _ { i } \in \bar { \mathbb { R } } ^ { H \times W \times 3 }$ and $\mathbf { D } _ { i } \in \mathbb { R } ^ { H \times W }$ refer to the i-th left RGB image and depth map with height H and width W . Foreground mask $\mathbf { M } _ { i } \in \mathbf { \bar { \mathbb { R } } } ^ { H \times W }$ is utilized to exclude unwanted pixels, such as surgical tools, blood, and smoke. Projection matrix $\mathbf { P } _ { i } \in \mathbb { R } ^ { 4 \times 4 }$ maps 3D coordinates to 2D pixels. In this work we prioritize 3D reconstruction and deformation.

## A. Preparatory Procedures

In this phase, we initiate preparatory procedures for the training of Gaussian Splatting. We conduct sparse key point matching, intended for subsequent utilization in the semi-rigidity deformation stage, and perform video inpainting to mitigate occlusions caused by surgical instruments, aimed for subsequent application in the surface-aware reconstruction phase.

Sparse Feature Point Matching. We initially identify specific feature points for sparse point tracking. These points are predominantly situated at crucial vascular intersections and regions characterized by distinctive features, making them challenging to reconstruct. We employ the Scale-Invariant Feature Transform (SIFT) [27] technique to extract sparse feature points from each frame. Subsequently, we conduct feature matching to ascertain the correspondence of points across frames, thereby establishing the trajectories of sparse key points. The acquisition of sparse tracks furnishes valuable information to facilitate the modeling of tissue deformation. By leveraging the sparse tracks, we can effectively guide the learning process for the deformation dynamics.

Video Inpainting. In this stage, video inpainting is conducted to eliminate occlusions caused by surgical tools. Given the original video sequence of masked surgical tools $X ~ : = ~ \{ X _ { 1 } , \ldots , X _ { T } \}$ , with corresponding annotations of corrupted regions represented by the mask sequence $M : =$ $\{ M _ { 1 } , \ldots , M _ { T } \}$ (where T denotes the length of the video), our objective is to generate the inpainted video sequence ${ \hat { Y } } : = \left\{ { \hat { Y } } _ { 1 } , \ldots , { \hat { Y } } _ { T } \right\}$ while preserving spatio-temporal coherence with the ground truth video sequence $Y : = \{ Y _ { 1 } , \ldots , Y _ { T } \}$

Prior approaches have often overlooked the significance of inpainting, resulting in unnatural visual artifacts within the inpainted regions. To address this, we adopt a Transformer [28]-based inpainting network [29] for the video inpainting process. Specifically, we fine-tune a flow-guided video inpainting model to accommodate tool masks, leveraging data from StereoMIS [11]. In this dataset, continuous areas are randomly masked to simulate the occlusion effects of surgical tools. As illustrated in Figure 3, our inpainting outcomes exhibit improved visual fidelity.

## B. Surface-Aware Reconstruction

In this phase, our aim is to reconstruct the initial frame of the scene with high quality while being surface-aware. Despite the ability of 3DGS to produce realistic real-time rendered images, it faces challenges in accurately representing the surface of the scene. This challenge arises from the use of discrete Gaussian kernels. However, ensuring precision in depicting the underlying surface of the surgical scene is crucial. To tackle this issue, we incorporate mesh with Gaussian Splatting during the reconstruction of the first frame. Our approach focuses on bounding 3D Gaussian kernels onto the mesh surface, thus facilitating subsequent Gaussian deformation processes. The conceptual foundation for our surface-aware reconstruction is inspired by EndoSuRF (Endoscopic Surgical Reconstruction Framework) [22] and Mesh-based Gaussian Splatting [30]. Our primary objective in this phase is to achieve a high-quality reconstruction of the first frame.

Mesh Reconstruction. The initial step involves generating the mesh for the first frame, employing static NeuS2 [31] for mesh reconstruction. Each 3D position x is mapped to its multiresolution hash encodings $h _ { \Omega } ( \mathbf { x } )$ , utilizing learnable hash table entries â¦. Since $h _ { \Omega } ( \mathbf { x } )$ serves as an informative encoding of spatial position, the MLPs responsible for mapping x to its Signed Distance Function (SDF) d and color c can be kept shallow, ensuring efficient training without compromising quality. The SDF network, denoted as $( d , \mathbf { g } ) = f _ { \boldsymbol { \Theta } } ( \mathbf { e } )$ , consists of a shallow MLP with weights Î, where $\mathbf { e } = ( \mathbf { x } , h _ { \Omega } ( \mathbf { x } ) )$ . Here, e encapsulates the 3D position x along with its corresponding hash encoding $h _ { \Omega } ( \mathbf { x } )$ , yielding the SDF value d and a geometry feature vector $\mathbf { g } \in \mathbb { R } ^ { 1 5 }$ . The normal vector n at x is computed as $\begin{array} { r } { \mathbf { n } = \nabla _ { \mathbf { x } } d , } \end{array}$ , where $\nabla _ { \mathbf x } d$ represents the gradient of the SDF with respect to x. This normal, combined with the geometry feature g, the SDF d, the point x, and the ray direction v, serves as input to the color network, expressed as $\mathbf { c } = c _ { \Upsilon } ( \mathbf { x } , \mathbf { n } , \mathbf { v } , d , \mathbf { g } )$ , which predicts the color c of x.

To supervise the learning of NeuS2, we minimize the color difference between the rendered pixels $\hat { C } _ { i }$ with $i \in \{ 1 , \ldots , m \}$ and the corresponding ground truth pixels $C _ { i }$ and also minimize the depth difference between ground truth depth and predicted depth:

$$
\mathcal { L } _ { \mathrm { c o l o r } } = \frac { 1 } { m } \sum _ { i } \mathcal { R } \left( \hat { C } _ { i } , C _ { i } \right) , \mathcal { L } _ { \mathrm { d e p t h } } = \frac { 1 } { m } \sum _ { i } \mathcal { R } \left( \hat { D } _ { i } , D _ { i } \right)\tag{1}
$$

where $\mathcal { R }$ is the Huber loss. We also employ an Eikonal term

$$
\mathcal { L } _ { \mathrm { e i k o n a l } } = \frac { 1 } { m n } \sum _ { k , i } \left( \| \mathbf { n } _ { k , i } \| - 1 \right) ^ { 2 }\tag{2}
$$

to regularize the learned signed distance field, where k indexes the k-th sample along the ray with $k \in \{ 1 , \ldots , n \}$ , n is the number of sampled points, and $\mathbf { n } _ { k , \ast }$ ,i is the normal of a sampled point. Our final loss for the mesh reconstruction of the first frame:

$$
{ \mathcal { L } } _ { \mathrm { m e s h } } = { \mathcal { L } } _ { \mathrm { c o l o r } } + \alpha _ { 1 } { \mathcal { L } } _ { \mathrm { d e p t h } } + \alpha _ { 2 } { \mathcal { L } } _ { \mathrm { e i k o n a l } }\tag{3}
$$

Mesh restricted Gaussian Splatting. Upon acquiring the mesh, Gaussian kernels are positioned at the centroid of each mesh triangle, establishing a direct correspondence between Gaussian kernels and mesh triangles. The initial radius aligns with the size of the inscribed circle within the binding triangle. The initial Gaussian training proceeds for the first frame without heuristics such as deletion or splitting of Gaussians.

To enhance the visual fidelity of 3D Gaussians, a regularization process is implemented to maintain spatial coherence and local consistency. This regularization mitigates potential visual distortions arising from overly expansive Gaussians that cover multiple mesh triangles. To ensure the fidelity of deformation outcomes, a regularization term $L _ { \mathrm { s c a l e } }$ is introduced, the formulation of this regularization term is expressed as:

$$
L _ { \mathrm { s c a l e } } = \frac { 1 } { | \mathcal { G } | } \sum _ { g _ { i } \in \mathcal { G } } \operatorname* { m a x } \left( \operatorname* { m a x } \left( s _ { i } \right) - \gamma _ { 1 } R _ { i } , 0 \right)\tag{4}
$$

where $s _ { i }$ represents the 3D scaling vector of each Gaussian, $R _ { i }$ denotes the radius of the circumcircle of the binding triangle wherein the Gaussian is positioned, and $\gamma _ { 1 }$ denotes the hyperparameter. This loss dynamically adjusts the Gaussian size relative to the binding triangleâs radius during training. This method ensures the acquisition of suitable Gaussian representations and maintains local continuity during deformation.

Additionally, we impose constraints on the displacement of Gaussians, preventing them from shifting away from the binding triangle:

$$
L _ { \mathrm { s h i f t } } = \frac { 1 } { | \mathcal { G } | } \sum _ { g _ { i } \in \mathcal { G } } \operatorname* { m a x } \left( \operatorname* { m a x } \left( \Delta \mu \right) - \gamma _ { 2 } R _ { i } , 0 \right)\tag{5}
$$

where $\Delta \mu$ denotes the shifted distance, and $\gamma _ { 2 }$ represents the hyper-parameters. This loss penalizes large shifts for Gaussians. Therefore, the comprehensive loss function for the first frame Gaussian Splatting is formulated as

$$
{ \mathcal { L } } _ { \mathrm { r e c o n s t r u c t i o n } } = { \mathcal { L } } _ { \mathrm { c o l o r } } + \beta _ { 1 } { \mathcal { L } } _ { \mathrm { d e p t h } } + \beta _ { 2 } { \mathcal { L } } _ { \mathrm { s c a l e } } + \beta _ { 3 } { \mathcal { L } } _ { \mathrm { s h i f t } }\tag{6}
$$

## C. Semi-Rigidity Deformation

After achieving high-quality reconstruction of the initial frame, we advocate for semi-rigidity deformation in the training of subsequent frames using Gaussian Splatting to preserve multi-level geometry regularization. This module is designed to facilitate the acquisition of physically plausible deformations. To address this objective, we introduce two guiding methodologies: local rigidity restriction and global non-rigidity restriction. Our local rigidity restriction aims to guide the learning on the area where there exists key point features, and our global non-rigidity restriction aims to unify the global deformation.

Local Rigidity Restriction. We employ a methodology inspired by the as-rigid-as-possible (ARAP) approach [32] for mesh deformation. Since the strict adherence to rigidity may lead to inaccuracies in the resulting depth, we introduce an ARAP Loss $\mathcal { L } _ { \mathrm { a r a p } }$ to gently guide points within local regions to conform to the principle of maximal rigidity. Given that endoscopic tissues exhibit high deformability, rigidity is primarily localized to small areas. As depicted in Figure 1, we define the region as within a specified distance from the key points; only within these designated areas is the ARAP loss implemented.

<!-- image-->  
Fig. 2. Comparison with other Gaussian Splatting-based methodologies. When we change the viewpoints, we clearly observe the 3D geometries of existing works are distorted, while our proposed framework reconstructs more consistent and much smoother endoscopic tissue surfaces. This demonstrates the usefulness of our proposed multi-level geometry regularization.

To compute $\mathcal { L } _ { \mathrm { a r a p } }$ , we utilize the points at the current time step t and the corresponding points at the previous time step t â 1. For each point i within the region influenced by the ARAP Loss, its nearest sparse key point is represented as k. Initially, we compute the rotation matrix:

$$
{ \hat { R } } _ { i } = \underset { R \in { \bf S } \mathbf { O } ( 3 ) } { \arg \operatorname* { m i n } } \sum _ { k \in { \cal K } } w _ { i k } \left\| \left( \mu _ { i } ^ { t } - \mu _ { k } ^ { t } \right) - R \left( \mu _ { i } ^ { t - 1 } - \mu _ { k } ^ { t - 1 } \right) \right\| ^ { 2 }\tag{7}
$$

This rotation matrix can be calculated easily by using SVD decomposition. After getting the rotation matrix, we can define our ARAP Loss $( { \mathcal { L } } _ { \mathrm { a r a p } } )$ as:

$$
\frac { 1 } { n | S | } \sum _ { i \in S } \sum _ { k \in K } w _ { i k } \left\| ( \mu _ { i , t } - \mu _ { k , t } ) - \hat { R } _ { i } \left( \mu _ { i , t - 1 } - \mu _ { k , t - 1 } \right) \right\| ^ { 2 } .\tag{8}
$$

where n denotes the number of key points, S denotes all the points restricted by the loss, K is the group of key points, $w _ { i k }$ is the contagent as described in [32]. This loss evaluates the degree to which the learned motion deviates from the assumption of local rigidity principle described by the ARAP. By penalizing Larap, the learned motions are encouraged to be locally rigid.

Global Non-Rigidity Restriction. Since the key points are limited and the radius of our ARAP regulation cannot cover all points, we use neighborhood similarity loss to handle changes globally. We explicitly encourage the neighbouring Gaussians to have the same rotation over time:

$$
\mathcal { L } _ { \mathrm { r o t } } = \frac { 1 } { r | S | } \sum _ { i \in S } \sum _ { j \in \mathrm { k n n } _ { i ; r } } \left. \hat { q } _ { j , t } \hat { q } _ { j , t - 1 } ^ { - 1 } - \hat { q } _ { i , t } \hat { q } _ { i , t - 1 } ^ { - 1 } \right. _ { 2 }\tag{9}
$$

where qË denotes the normalized quaternion representation of each Gaussianâs rotation, with j belonging to the neighborhood of i as determined by the k-nearest-neighbor criterion. Unlike other methodologies such as that proposed by [33], which necessitates additional computational steps for neighborhood determination, our approach obviates the need for further classifications of nearby neighbors. This is attributed to the clarity in neighborhood identification facilitated by our surface-aware reconstruction..

We apply ${ \mathcal L } _ { \mathrm { r o t } }$ only between the current timestep and the directly preceding timestep, thus only enforcing these losses over short-time horizons, this sometimes causes elements of the scene to drift apart, thus we apply another loss, the isometry loss, over the long-term:

$$
\mathcal { L } _ { \mathrm { i s o } } = \frac { 1 } { r | S | } \sum _ { i \in S } \sum _ { j \in \mathrm { k n n } _ { i ; r } } w _ { i , j } \left| \left\| \mu _ { j , 0 } - \mu _ { i , 0 } \right\| _ { 2 } - \left\| \mu _ { j , t } - \mu _ { i , t } \right\| _ { 2 } \right|\tag{10}
$$

This is a weaker constraint in that instead of enforcing the positions between two Gaussians to be the same it only enforces the distances between them to be the same.

Therefore, combining global and local constraint, our overall loss for deformable Gaussian Splatting is defined as:

$$
{ \mathcal { L } } _ { \mathrm { d e f o r m } } = { \mathcal { L } } _ { \mathrm { c o l o r } } + \lambda _ { 1 } { \mathcal { L } } _ { \mathrm { d e p t h } } + \lambda _ { 2 } { \mathcal { L } } _ { \mathrm { a r a p } } + \lambda _ { 3 } { \mathcal { L } } _ { \mathrm { r o t } } + \lambda _ { 4 } { \mathcal { L } } _ { \mathrm { i s o } }\tag{11}
$$

<!-- image-->  
Fig. 3. Comparison with other methods on image quality and depth quality, where red boxes denote the incorrect reconstructed areas. Our method presents better reconstructed textures in the occluded regions.

TABLE I  
COMPARISON ON THE ENDONERF DATASET [8]. OUR APPROACH DEMONSTRATES SUPERIOR PERFORMANCE ACROSS MOST METRICS AND SCENARIOS.
<table><tr><td rowspan="2">Models</td><td rowspan="2">PSNR â</td><td rowspan="2">Cutting SSIM â</td><td rowspan="2"></td><td colspan="3">Pulling</td><td rowspan="2">Training Time â</td><td rowspan="2">FPS â</td><td rowspan="2">GPU Usage â</td></tr><tr><td>LPIPS â PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>EndoNeRF</td><td>35.64</td><td>0.930</td><td>0.132</td><td>34.71</td><td>0.920</td><td>0.095</td><td>â¼ 6 hrs</td><td>~ 0.2</td><td>â¼ 20GB</td></tr><tr><td>EndoSurf</td><td>35.89</td><td>0.952</td><td>0.107</td><td>34.91</td><td>0.955</td><td>0.120</td><td>~ 7 hrs</td><td>â¼ 0.04</td><td>â¼ 20GB</td></tr><tr><td>LerPlane</td><td>34.69</td><td>0.901</td><td>0.112</td><td>36.38</td><td>0.937</td><td>0.083</td><td>â¼ 8 min</td><td>â¼ 0.9</td><td>â¼ 20GB</td></tr><tr><td>EndoGS</td><td>36.20</td><td>0.958</td><td>0.044</td><td>38.21</td><td>0.67</td><td>0.066</td><td>â¼ 2 min</td><td>â¼ 60</td><td>â¼ 10GB</td></tr><tr><td>EndoGaussian</td><td>37.21</td><td>0.961</td><td>0.065</td><td>36.10</td><td>0.946</td><td>0.091</td><td>â¼ 2 min</td><td>â¼ 170</td><td>â¼ 3GB</td></tr><tr><td>Ours</td><td>38.05</td><td>0.965</td><td>0.047</td><td>38.27</td><td>0.951</td><td>0.046</td><td>â¼ 2 min</td><td>â¼ 170</td><td>â¼ 3GB</td></tr></table>

TABLE II

COMPARISON ON THE SCARED DATASET [11].
<table><tr><td>Models</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td></tr><tr><td>EndoNeRF [8]</td><td>24.34</td><td>0.759</td><td>0.320</td></tr><tr><td>EndoSurf [22]</td><td>25.02</td><td>0.802</td><td>0.356</td></tr><tr><td>EndoGS [7]</td><td>26.47</td><td>0.798</td><td>0.291</td></tr><tr><td>EndoGaussian [26]</td><td>27.04</td><td>0.825</td><td>0.275</td></tr><tr><td>Ours</td><td>28.31</td><td>0.810</td><td>0.282</td></tr></table>

## IV. EXPERIMENTS

## A. Datasets

We conduct experiments utilizing two publicly available datasets: ENDONERF [8] and SCARED [11]. The EndoNeRF dataset [8] provides two instances of in-vivo prostatectomy data. This dataset includes depth maps estimated using E-DSSR [20], accompanied by manually labeled tool masks. ENDONERF [8] encompasses two cases of in-vivo prostatectomy data captured from stereo cameras at a single viewpoint. It presents challenging scenes featuring non-rigid deformation and tool occlusion. The SCARED dataset [11] consists of ground truth RGBD images of five porcine cadaver abdominal anatomies. SCARED gathers RGBD images of the same anatomies using a DaVinci endoscope and a projector.

## B. Main Results

In our experiments, the loss weight coefficients were determined via a coarse grid search on the validation set. For the mesh reconstruction stage (NeuS2), we set the depth loss weight to $\alpha _ { 1 } = 0 . 1$ and the Eikonal term weight to $\alpha _ { 2 } = 0 . 0 1$ In the mesh-restricted Gaussian Splatting, we used $\beta _ { 1 } = 0 . 5$ for the depth loss, $\beta _ { 2 } ~ = ~ 0 . 1$ for the scale regularization loss, and $\beta _ { 3 } = 0 . 0 5$ for the shift regularization loss. For the deformation stage, the loss terms were weighted as follows: $\lambda _ { 1 } = 0 . 5$ for the depth loss, $\lambda _ { 2 } = 0 . 1$ for the local ARAP loss, $\lambda _ { 3 } = 0 . 0 5$ for the rotation consistency loss, and $\lambda _ { 4 } = 0 . 0 2$ for the isometry loss. Variations of Â±20% in these values resulted in only marginal performance changes, indicating the robustness of our settings.

<!-- image-->  
Fig. 4. Detailed comparisons of the output RGB image are presented in this section. We emphasize regions rich in blood vessels and intricate texture. Our approach demonstrates a notable capability in preserving intricate details when contrasted with alternative methodologies. Furthermore, the inpainted regions in our method exhibit a higher degree of realism and coherence.

We performed a comparative analysis of our proposed approach with other techniques: EndoNeRF [8], EndoSurf [22], LerPlane [23], EndoGS, and EndoGaussian [26]. Among these, EndoNeRF, EndoSurf, and LerPlane are methods based on NeRF, while EndoGS and EndoGaussian utilize Gaussian Splatting.

As depicted in Table I and Table II, while methodologies rooted in NeRF, such as EndoNeRF and EndoSurf, demonstrate the ability to generate reconstructions of deformed tissues at a high-quality level, they come with a significant training cost, requiring hours of optimization and substantial memory resources. LerPlane [23] notably reduces the training duration to approximately 3 minutes per frame but compromises on the fidelity of reconstructions. Although successive iterations of LerPlane have the potential to augment rendering quality, they still encounter impediments in terms of inference speed. Their overall image quality falls short in comparison to Gaussian Splatting-based methodologies.

Comparing with other Gaussian Splatting-based methodologies, as depicted quantitatively in Tables I and II, our approach demonstrates superior performance across three key metrics: PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index Measure), and LPIPS (Learned Perceptual Image Patch Similarity). Furthermore, upon qualitative examination, our model exhibits enhanced 3D visual fidelity, as illustrated in Figure 2. In contrast, other Gaussian-based techniques exhibit deficiencies such as the presence of artifacts and suboptimal surface reconstruction. Our method achieves smoother surface representations, as evidenced in the visual depiction.

Our approach achieves state-of-the-art reconstruction outcomes (PSNR of 38.162) within a 2 minutes per frame of training, and our method achieves real-time rendering rates exceeding 60 frames per second (FPS), signifying a substantial acceleration compared to NeRF based techniques. Additionally, we note that our method utilizes only 2GB of GPU memory for optimization, approximately one-tenth the consumption of prior techniques, thereby mitigating hardware prerequisites for implementation in surgical settings. To provide a more intuitive comparison, we present several qualitative results in Figure 3 and 4. It is evident from these results that our method preserves more details and offers superior visualization of deformable tissues compared to other methods.

TABLE III  
COMPARISON ON FIRST FRAME 3D RECONSTRUCTION QUALITY.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>3DGS [6]</td><td>25.15</td><td>0.803</td><td>0.536</td></tr><tr><td>Dynamic 3D Gaussians [34]</td><td>33.60</td><td>0.914</td><td>0.096</td></tr><tr><td>Depth-Regularized GS [35]</td><td>28.38</td><td>0.913</td><td>0.472</td></tr><tr><td>FS-GS [36]</td><td>30.24</td><td>0.927</td><td>0.124</td></tr><tr><td>Ours</td><td>38.16</td><td>0.965</td><td>0.046</td></tr></table>

To compare with other methods for the Gaussian Splatting reconstruction of the first frame, we first experimented on the efficacy of other Gaussian Splatting based reconstruction methods : the original 3DGS, Dynamic 3D Gaussians [34], Depth-Regularized GS [35] and FS-GS [36], and our proposed surface-aware reconstruction technique. The results are outlined in Table III.

When using the original 3DGS method, the result is undesirable, with the PSNR only at 25.15, SSIM at 0.83 and LPIPS at 0.536. This deficiency is mainly due to its reliance on single-view data. We further train the original 3DGS with additional depth guidance, but simple adding depth guidance do not lead to better reconstruction quality.

For FS-GS, Depth-Regularized GS and Dynamic 3D Gaussians, although they demonstrate an improvement in reconstruction quality. However, despite the enhancements, the outcome remained not competitive with our surface-aware reconstruction. This is due to none of these methods is designed for single view situation, while in endoscopic scenario, there only exists single view. In comparison, our surface-aware reconstruction technique provides higher quantitative results in terms of all three metrics.

## C. Ablation Study

We also carried out the ablation study on our proposed components, the Surface-Aware Reconstruction and the Semi-Rigidity Deformation.

For the Surface-Aware Reconstruction, this component plays a crucial role in ensuring accurate and smooth surface representation by constraining the 3D Gaussian kernels onto the mesh. When we remove this module, our framework exhibits a significant degradation in overall image quality. Specifically, as presented in Table IV, the PSNR drops to 33.60, SSIM decreases to 0.914, and LPIPS increases to 0.096. These results highlight that without the surface-aware reconstruction, our model struggles to effectively represent the underlying geometry, leading to less accurate surface reconstructions with visible artifacts. This is particularly evident in highly deformable regions, where the lack of mesh constraints causes the Gaussians to drift and fail to form a coherent surface.

TABLE IV  
ABLATION STUDY OF OUR PROPOSED MULTI-LEVEL GEOMETRY REGULARIZATION ON THE ENDONERF DATASET [8].
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Ours</td><td>38.16</td><td>0.965</td><td>0.046</td></tr><tr><td>w/o surface-aware reconstruction</td><td>33.60</td><td>0.914</td><td>0.096</td></tr><tr><td>w/o global non-rigidity restriction</td><td>36.29</td><td>0.944</td><td>0.063</td></tr><tr><td>w/o local rigidity restriction</td><td>37.18</td><td>0.950</td><td>0.065</td></tr></table>

For the Semi-Rigidity Deformation, we performed a detailed ablation study on its two key components: the local rigidity restriction and the global non-rigidity restriction. Both components are crucial for preserving the structural consistency of the reconstructed scene during deformation. When the local rigidity restriction is omitted, the overall image quality degrades, with PSNR dropping to 37.18, SSIM to 0.950, and LPIPS increasing to 0.065. This degradation is primarily due to the lack of guidance in regions with key points, which results in less accurate motion learning and leads to inconsistencies in deformable regions, especially around areas with vascular intersections or fine features.

Similarly, when the global non-rigidity restriction is removed, the model suffers from irregular motion of the Gaussians, causing further degradation in the reconstruction quality. In this case, the PSNR drops to 36.29, SSIM to 0.944, and LPIPS increases to 0.063. The absence of this global constraint leads to a lack of cohesion in the deformation process, causing neighboring Gaussians to move inconsistently and resulting in artifacts or distortions in the reconstructed surfaces.

## V. CONCLUSION

In this paper, we propose a novel 3D Gaussian Splatting [6] based framework with multi-level geometry regularization for real-time and high-quality reconstruction of dynamic endoscopic scenes. By employing surface-aware reconstruction and semi-rigidity deformation, we address the challenge of reconstructing deformable tissue. Experimental results have demonstrated that our method achieves state-of-the-art reconstruction quality, with smooth surfaces and realistic deformation. Additionally, we achieve real-time rendering speeds over 100 times faster than previous NeRF [5] based methods, with training times reduced by a factor of 10. We believe that our approach utilizing Gaussian Splatting-based reconstruction techniques can inspire advancements in robotic surgery scene reconstruction.

## VI. ACKNOWLEDGEMENT

This research is supported by the National Natural Science Foundation of China (No. 62406267), Guangzhou-HKUST(GZ) Joint Funding Program (Grant No.2025A03J3956), Education Bureau of Guangzhou Municipality and the Guangzhou Municipal Education Project (No. 2024312122).

## REFERENCES

[1] L. Qian, J. Y. Wu, S. P. DiMaio, N. Navab, and P. Kazanzides, âA review of augmented reality in robotic-assisted surgery,â IEEE Transactions on Medical Robotics and Bionics, vol. 2, no. 1, pp. 1â16, 2020.

[2] J. Han, J. Davids, H. Ashrafian, A. Darzi, D. S. Elson, and M. H. Sodergren, âA systematic review of robotic surgery: From supervised paradigms to fully autonomous robotic approaches,â The International Journal of Medical Robotics and Computer Assisted Surgery, vol. 18, 2021.

[3] M. Zhu, C. Li, S. Zhao, L. Chen, and X. Zhao, âThe role of threedimensional reconstruction of medical images and virtual reality in nursing experimental teaching,â Journal of Healthcare Engineering, vol. 2022, 2022.

[4] C. Li, H. Liu, Y. Liu, B. Y. Feng, W. Li, X. Liu, Z. Chen, J. Shao, and Y. Yuan, âEndora: Video generation models as endoscopy simulators,â in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024.

[5] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, âNerf: Representing scenes as neural radiance fields for view synthesis,â Commun. ACM, vol. 65, pp. 99â106, 2020.

[6] B. Kerbl, G. Kopanas, T. Leimkuehler, and G. Drettakis, â3d gaussian splatting for real-time radiance field rendering,â ACM Transactions on Graphics (TOG), vol. 42, pp. 1 â 14, 2023.

[7] L. Zhu, Z. Wang, Z. Jin, G. Lin, and L. Yu, âDeformable endoscopic tissues reconstruction with gaussian splatting,â 2024.

[8] Y. Wang, Y. Long, S. H. Fan, and Q. Dou, âNeural rendering for stereo 3d reconstruction of deformable tissues in robotic surgery,â ArXiv, vol. abs/2206.15255, 2022.

[9] C. Li, B. Y. Feng, Y. Liu, H. Liu, C. Wang, W. Yu, and Y. Yuan, âEndosparse: Real-time sparse view synthesis of endoscopic scenes using gaussian splatting,â in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 252â262.

[10] H. Liu, Y. Liu, C. Li, W. Li, and Y. Yuan, âLgs: A light-weight 4d gaussian splatting for efficient surgical scene reconstruction,â in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024.

[11] M. Allan, J. Mcleod, C. Wang, J. C. Rosenthal, Z. Hu, N. Gard, P. Eisert, K. X. Fu, T. Zeffiro, W. Xia et al., âStereo correspondence and reconstruction of endoscopic data challenge,â arXiv preprint arXiv:2101.01133, 2021.

[12] P. Brandao, D. Psychogyios, E. B. Mazomenos, D. Stoyanov, and M. Janatka, âHapnet: hierarchically aggregated pyramid network for real-time stereo matching,â Computer Methods in Biomechanics and Biomedical Engineering: Imaging & Visualization, vol. 9, pp. 219 â 224, 2020.

[13] H. Luo, C. Wang, X. Duan, H. Liu, P. Wang, Q. Hu, and F. Jia, âUnsupervised learning of depth estimation from imperfect rectified stereo laparoscopic images,â Computers in biology and medicine, vol. 140, p. 105109, 2021.

[14] K. Wang, C. Yang, Y. Wang, S. Li, Y. Wang, Q. Dou, X. Yang, and W. Shen, âEndogslam: Real-time dense reconstruction and tracking in endoscopic surgeries using gaussian splatting,â 2024.

[15] J. Song, J. Wang, L. Zhao, S. Huang, and G. Dissanayake, âDynamic reconstruction of deformable soft-tissue with stereo scope in minimal invasive surgery,â IEEE Robotics and Automation Letters, vol. 3, no. 1, pp. 155â162, 2017.

[16] H. Zhou and J. Jagadeesan, âReal-time dense reconstruction of tissue surface from stereo optical video,â IEEE transactions on medical imaging, vol. 39, no. 2, pp. 400â412, 2019.

[17] H. Zhou and J. Jayender, âEmdq-slam: Real-time high-resolution reconstruction of soft tissue surface from stereo laparoscopy videos,â in MICCAI. Springer, 2021, pp. 331â340.

[18] F. P. Stilz, M. A. Karaoglu, F. Tristram, N. Navab, B. Busam, and A. Ladikos, âFlex: Joint pose and dynamic radiance fields optimization for stereo endoscopic videos,â 2024.

[19] Y. Li, F. Richter, J. Lu, E. K. Funk, R. K. Orosco, J. Zhu, and M. C. Yip, âSuper: A surgical perception framework for endoscopic tissue manipulation with surgical robotics,â IEEE Robotics and Automation Letters, vol. 5, pp. 2294â2301, 2019.

[20] Y. Long, Z. Li, C.-H. Yee, C.-F. Ng, R. H. Taylor, M. Unberath, and Q. Dou, âE-dssr: Efficient dynamic surgical scene reconstruction with transformer-based stereoscopic depth perception,â in International

Conference on Medical Image Computing and Computer-Assisted Intervention, 2021.

[21] R. A. Newcombe, D. Fox, and S. M. Seitz, âDynamicfusion: Reconstruction and tracking of non-rigid scenes in real-time,â 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 343â352, 2015.

[22] R. Zha, X. Cheng, H. Li, M. Harandi, and Z. Ge, âEndosurf: Neural surface reconstruction of deformable tissues with stereo endoscope videos,â in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2023, pp. 13â23.

[23] C. Yang, K. Wang, Y. Wang, X. Yang, and W.-M. Shen, âNeural lerplane representations for fast 4d reconstruction of deformable tissues,â in International Conference on Medical Image Computing and Computer-Assisted Intervention, 2023.

[24] C. Yang, K. Wang, Y. Wang, X. Yang, and W. Shen, âNeural lerplane representations for fast 4d reconstruction of deformable tissues,â MIC-CAI, 2023.

[25] S. Saha, S. Liu, S. Lin, J. Lu, and M. Yip, âBased: Bundle-adjusting surgical endoscopic dynamic video reconstruction using neural radiance fields,â arXiv preprint arXiv:2309.15329, 2023.

[26] Y. Liu, C. Li, C. Yang, and Y. Yuan, âEndogaussian: Real-time gaussian splatting for dynamic endoscopic scene reconstruction,â 2024.

[27] D. G. Lowe, âObject recognition from local scale-invariant features,â in Proceedings of the seventh IEEE international conference on computer vision, vol. 2. Ieee, 1999, pp. 1150â1157.

[28] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Å. Kaiser, and I. Polosukhin, âAttention is all you need,â Advances in neural information processing systems, vol. 30, 2017.

[29] K. Zhang, J. Fu, and D. Liu, âInertia-guided flow completion and style fusion for video inpainting,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 5982â5991.

[30] L. Gao, J. Yang, B.-T. Zhang, J.-M. Sun, Y.-J. Yuan, H. Fu, and Y.-K. Lai, âMesh-based gaussian splatting for real-time large-scale deformation,â 2024.

[31] Y. Wang, Q. Han, M. Habermann, K. Daniilidis, C. Theobalt, and L. Liu, âNeus2: Fast learning of neural implicit surfaces for multi-view reconstruction,â 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3272â3283, 2022.

[32] O. Sorkine and M. Alexa, âAs-rigid-as-possible surface modeling,â in Proceedings of the Fifth Eurographics Symposium on Geometry Processing, ser. SGP â07. Goslar, DEU: Eurographics Association, 2007, p. 109â116.

[33] Y.-H. Huang, Y.-T. Sun, Z. Yang, X. Lyu, Y.-P. Cao, and X. Qi, âSc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes,â arXiv preprint arXiv:2312.14937, 2023.

[34] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan, âDynamic 3d gaussians: Tracking by persistent dynamic view synthesis,â in 3DV, 2024.

[35] J. Chung, J. Oh, and K. M. Lee, âDepth-regularized optimization for 3d gaussian splatting in few-shot images,â 2024.

[36] Z. Zhu, Z. Fan, Y. Jiang, and Z. Wang, âFsgs: Real-time few-shot view synthesis using gaussian splatting,â 2023.