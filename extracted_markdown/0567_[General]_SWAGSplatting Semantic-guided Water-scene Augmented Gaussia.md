# SWAGSplatting: Semantic-guided Water-scene Augmented Gaussian Splatting

Zhuodong Jiang\*1, Haoran Wang\*1, Guoxi Huang1, Brett Seymour2, Nantheera Anantrasirichai1 1University of Bristol, Bristol, UK, {ci21041, yp22378, guoxi.huang, n.anantrasirichai}@bristol.ac.uk 2Submerged Resources Centre, Denver, CO, USA, Brett Seymour@nps.gov

GT  
<!-- image-->  
3DGS  
UW-GS  
SWAGSplatting  
Fig. 1: Reconstruction performance comparison among 3DGS [1], UW-GS [2], and the proposed SWAGSplatting. Relative to the 3DGS and UW-GS methods, SWAGSplatting markedly suppresses artefact generation and yields more faithful rendering results.

AbstractâAccurate 3D reconstruction in underwater environments remains a challenging task due to light attenuation, scattering, and limited visibility. While recent AI-based approaches have advanced underwater imaging, they often overlook highlevel semantic understanding, which is crucial for reconstructing complex scenes. In this paper, we propose SWAGSplatting, Semantic-guided Water-scene Augmented Gaussian Splatting, a novel multimodal framework that integrates language and vision knowledge into 3D Gaussian Splatting for robust and highfidelity underwater reconstruction. Each Gaussian primitive is augmented with a learnable semantic feature, supervised using CLIP-based embeddings extracted from region-level semantic cues. A dedicated semantic consistency loss enforces alignment between geometric reconstruction and scene semantics. In addition, a stage-wise optimisation strategy combining coarse-tofine learning with late-stage parameter refinement improves training stability and visual quality. Furthermore, we propose a 3D Gaussian Primitives Reallocation strategy to address the imbalanced distribution of primitives introduced by naive point cloud densification. Extensive experiments on the SeaThru-NeRF and Submerged3D datasets demonstrate that SWAGSplatting consistently outperforms state-of-the-art methods across PSNR, SSIM, and LPIPS metrics, achieving up to a 3.48 dB improvement in PSNR, enabling more accurate and semantically coherent underwater scene reconstruction for applications in marine perception and exploration.

Index Termsâ3D gaussian splatting, Semantic-aware 3D reconstruction, Underwater scene reconstruction

## I. INTRODUCTION

Underwater exploration supports a wide range of applications, including marine ecology, archaeology, and robotics. These tasks rely heavily on accurate 3D reconstruction for interpretation and navigation. High-quality 3D visuals are particularly important when viewed through VR headsets, as in educational and creative industry applications. However, underwater environments present unique challenges, such as limited visibility, colour distortion, sparse viewpoints, and noise caused by light attenuation and scattering. Capturing videos in deep underwater settings is especially difficult, as low light conditions require longer exposure times or higher ISO settings, both of which introduce significant motion blur and sensor noise. Still image capture is often limited and unevenly distributed, further compounding the difficulty of reliable 3D reconstruction in such conditions.

Recent advances in Neural Rendering have opened new possibilities for 3D reconstruction through Novel View Synthesis (NVS) methods. Neural Radiance Field (NeRF) [3] has achieved impressive results but requires dense views and long training times, lacking flexibility. 3D Gaussian Splatting (3DGS) [4] offers real-time rendering and higher refresh rates using explicit point-based scene representations. However, most NeRF- and 3DGS-based methods are built based on the assumption of clear media, and their performance degrades severely in turbid underwater conditions.

A critical limitation of existing underwater reconstruction methods [2], [5]â[7] is that they treat all regions uniformly. In underwater environments, salient objects, which typically appear in the foreground, deserve greater attention for both perceptual quality and practical applications such as VR-based peripheral vision. Without explicit guidance, gradients from these semantically important regions become diluted by the background during training, leading to reconstructions where salient objects appear blurred and lack fidelity.

To overcome these limitations, we present SWAGSplatting (Semantic-guided Water-scene Augmented Gaussian Splatting), a novel multimodal framework that integrates semantic and physical understanding for underwater 3D reconstruction. By incorporating semantic cues from visionâlanguage models, our method enables object-aware, highfidelity scene reconstruction under challenging imaging conditions. In addition, the low fidelity of underwater scenes increases the redundancy of 3D Gaussians, which consequently limits rendering quality. To address this, we propose a new representation that adaptively relocates 3D Gaussians to reduce redundancy while improving reconstruction quality.

The main contributions of this work are as follows:

â¢ We present the first semantic-guided 3D Gaussian Splatting framework for underwater scene reconstruction. Each Gaussian is augmented with a learnable semantic feature supervised by CLIP [8] embeddings derived from region-level descriptions, enabling object-aware and semantically consistent reconstruction.

â¢ We propose a novel semantic consistency loss that enforces the alignment between semantic and geometric features, improving both structural coherence and perceptual fidelity.

â¢ We introduce a stage-wise optimisation strategy, a coarseto-fine training scheme that enhances stability and visual quality via late-stage parameter freezing and $\ell _ { 2 }$ fine-tuning.

â¢ Gaussian primitive reallocation is proposed to balance the distribution of the Gaussian point cloud by reallocating low-importance primitives to high-error regions, thereby enhancing NVS quality.

We conduct a comprehensive evaluation on the SeaThru-NeRF [5] and Submerged3D [7] datasets, demonstrating an improvement of up to 3.48 dB in PSNR and consistent gains in SSIM and LPIPS over state-of-the-art baselines. Fig. 1 shows a qualitative comparison against the state-of-the-art method.

## II. RELATED WORK

## A. NeRF-based Underwater Scene Reconstruction

ScatterNeRF [9] extends NeRF [10] to rendering in scattering media by distinguishing volumetric attenuation from object geometry, a principle relevant to underwater imaging. SeaThru-NeRF [5], built upon the revised underwater image formation model proposed in [11], separates transmittance and medium colour through dedicated MLPs to disentangle objects from water effects. SP-SeaNeRF [12] enhances sharpness using learnable illumination embeddings. Water-NeRF [13] employs a physics-based light transport model with optimal transport for colour correction, while WaterHE-NeRF [14] applies a Retinex-based water-ray matching field for colour compensation. UWNeRF [15] distinguishes static and dynamic regions but relies heavily on accurate masking, whereas AquaNeRF [16] mitigates moving-object artefacts using a single-surface-per-ray strategy and Gaussian-weighted transmittance.

## B. 3DGS-based Underwater Scene Reconstruction

Underwater 3DGS remains an emerging research area. Early progress was made by UW-GS [2], which introduced a physics-based density control strategy and motion masks to reduce scattering and dynamic distractors. WaterSplatting [6] improved realism by separating object and medium transmittance, enabling real-time, high-quality rendering. Aquatic-GS [17] advanced this further by coupling implicit water fields with explicit 3DGS, achieving clearer and physically consistent reconstructions. SeaSplat [18] adopted an underwater image formation model to enhance rendering quality. However, it remains limited to static scenes. More recent methods, such as RecGS [19], which improve perceptual quality via recurrent training, and RUSplatting [7], which integrates uncertainty estimation to mitigate the impact of degraded frames, illustrate a growing trend toward robust and adaptive underwater Gaussian splatting. R-Splatting [20] integrates several pretrained underwater-enhancement models into the 3DGS pipeline to handle illumination variations in the input images, which works well for shallow-water scenarios. AtlantisGS [21] decomposes the scene into foreground objects and the background medium, and increases the number of Gaussians representing the foreground.

## III. PRELIMINARIES

## A. 3D Gaussian Splatting (3DGS)

3DGS [4] represents a scene using a set of 3D anisotropic Gaussians, each defined by its position $\pmb { \mu _ { i } } \in \mathbb { R } ^ { 3 }$ , covariance $\Sigma _ { i } ,$ view-dependent colour $c _ { i } ,$ , and opacity $\alpha _ { i }$ . To render an image, each Gaussian $G _ { i }$ is projected onto the 2D image plane, where their contribution to pixel x is given by:

$$
G _ { i } ( { \pmb x } ) = \mathrm { e x p } \left( - \frac { 1 } { 2 } ( { \pmb x } - { \pmb \mu } _ { i } ) ^ { \top } \Sigma _ { i } ^ { - 1 } ( { \pmb x } - { \pmb \mu } _ { i } ) \right) .\tag{1}
$$

The final colour is calculated via alpha blending:

$$
C = \sum _ { i = 1 } ^ { N } \alpha _ { i } c _ { i } ( \mathbf { v } ) \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{2}
$$

in which $c _ { i } ( \mathbf { v } )$ denotes the SH-based view-dependent appearance.

## B. Underwater Image Formation

Underwater scenes are strongly affected by light scattering and absorption, which may significantly degrade image quality and, thus, the reconstruction performance [15]. The observed colour $I _ { c }$ at a camera pixel can be modelled as:

$$
I _ { c } = J \cdot T ^ { D } + B ^ { \infty } \cdot ( 1 - T ^ { B } ) .\tag{3}
$$

Here $J$ is the true scene radiance, $B ^ { \infty }$ stands for the background light, and $T ^ { D } , T ^ { B }$ represent the transmission of direct and backscattered light:

$$
T ^ { D } = \exp ( - \beta ^ { d } \cdot z ) , ~ T ^ { B } = \exp ( - \beta ^ { b } \cdot z ) ,\tag{4}
$$

with $\beta ^ { d }$ and $\beta ^ { b }$ denoting the attenuation coefficients, and $z \ \mathrm { a s }$ the depth. These are important parameters for the model when adapting 3DGS to underwater environments.

## IV. METHODOLOGY

As illustrated in Fig. 2, the proposed SWAGSplatting augments each Gaussian with an additional semantic feature. By aligning this feature with CLIP semantic embeddings, the model captures high-level object structure. Moreover, semantic-guided segmentation provides supervision via $L _ { s } ,$ encouraging the model to suppress redundant background regions. We further introduce a 3D Gaussian primitives reallocation mechanism that redistributes primitives from lowimportance regions to high-error regions, improving reconstruction quality without increasing the overall point budget. Finally, a stage-wise optimisation strategy is applied to maintain geometric stability while enhancing appearance.

<!-- image-->  
Fig. 2: Pipeline of the SWAGSplatting. Yellow highlights indicate the proposed contributions: (1) semantic-guided loss $L _ { s }$ to obtain high-level structure consistency and high fidelity and quality reconstruction; (2) stage-wise optimisation strategy to enhance both training stability and construction quality; (3) 3D Gaussian primitives reallocation balances the point-cloud distribution and improves reconstruction with the same number of primitives.

## A. Semantic-guided Gaussians

Traditional 3DGS methods optimise all Gaussians uniformly based solely on photometric loss, which is not the optimal solution for underwater scenes where (a) salient objects require higher reconstruction priority for perceptual quality, and (b) sparse views and medium distortions make geometric consistency difficult to maintain. We address this by embedding semantic awareness directly into the Gaussian representation.

In SWAGSplatting, each Gaussian is augmented with a learnable semantic feature vector $f _ { s } \in \mathbb { R } ^ { d }$ , where d denotes the dimensionality of the projected semantic embedding space. Unlike spatial or photometric attributes, $f _ { s }$ is optimised under external supervision to encode high-level semantic information that promotes object-aware reconstruction.

To obtain reference embeddings, we first generate textual descriptions for each scene using BLIP3-o [22], which guides Grounded-SAM [23] to capture regions of interest in the input image $I _ { i m g }$

$$
\begin{array} { r l } & { \mathcal { R } = \mathrm { G r o u n d e d - S A M } ( I _ { i m g } , \mathrm { c a p t i o n } ) , } \\ & { I _ { \mathcal { R } } = I _ { i m g } [ \mathcal { R } ] , } \\ & { f _ { \mathrm { r e f } } = \mathrm { C L I P } ( I _ { \mathcal { R } } ) , } \end{array}\tag{5}
$$

where R is the bounding box of the detected object and fref denotes the CLIP embedding of the detected region.

During training, all Gaussians whose projections fall inside the region R are encouraged to align their semantic features with $f _ { \mathrm { r e f } }$ via the following loss:

$$
L _ { s } = \sum _ { i \in \mathcal { R } } \Vert f _ { s } ^ { ( i ) } - f _ { \mathrm { r e f } } \Vert _ { 2 } ^ { 2 } ,\tag{6}
$$

where $f _ { s } ^ { ( i ) }$ denotes the semantic feature $f _ { s }$ of the i-th Gaussian $G _ { i } .$ . This additional supervision enforces semantic-geometric consistency by encouraging Gaussians within the same object region to share similar semantics, thereby preserving object-level coherence under noisy, low-visibility, or sparseview underwater conditions. As a result, the reconstructed scenes exhibit improved structural integrity and perceptual interpretability.

## B. Stage-wise Optimization Strategy

Our training process follows a two-stage optimisation schedule guided by a composite objective function. The baseline losses: reconstruction $L _ { \mathrm { R e c } }$ depth-supervised LDepth, grayworld prior $L _ { g } ,$ and edge-aware smoothness $L _ { \mathrm { S m o o t h } } .$ , are inherited from RUSplatting [7]. To further strengthen stability and fine-grained appearance modelling, we introduce (1) the semantic loss $L _ { s }$ from (6); (2) an additional pixel-wise mean squared error (MSE) term $L _ { 2 }$ and (3) a hinge loss $L _ { h }$ that constrains the predicted attenuation coefficients $\beta ^ { d }$ and $\beta ^ { b }$ . For any desired inequality $a > b ,$ , we use $L _ { \mathrm { h i n g e } } ( a , b ) = \operatorname* { m a x } ( 0 , b -$ $a + m )$ , where $m \geq 0$ is a small margin (i.e., 1e-3). The loss is zero when the constraint holds and increases linearly when violated. Following underwater optics (red attenuates most, blue least), we enforce $\beta _ { r } ^ { * } > \beta _ { g } ^ { * } > \beta _ { b } ^ { * } ~ ( * \in \{ d , b \} )$ with

$$
L _ { h } ^ { * } = \operatorname* { m a x } ( 0 , \beta _ { g } ^ { * } - \beta _ { r } ^ { * } + m ) + \operatorname* { m a x } ( 0 , \beta _ { b } ^ { * } - \beta _ { g } ^ { * } + m ) ,\tag{7}
$$

where subscripts denote the RGB channels. We set the final hinge loss as $L _ { h } ~ = ~ { \textstyle { \frac { 1 } { 2 } } } ( L _ { h } ^ { d } + L _ { h } ^ { b } )$ . The total objective is formulated as:

$$
L _ { \mathrm { f i n a l } } = L _ { \mathrm { R e c } } + L _ { \mathrm { D e p t h } } + L _ { g } + L _ { \mathrm { S m o o t h } } + \lambda _ { s } L _ { s } + \lambda _ { 2 } L _ { 2 } + \lambda _ { h } L _ { h } ,\tag{8}
$$

where $\lambda _ { s } , \lambda _ { 2 }$ and $\lambda _ { h }$ control the weighting of the semantic, MSE and hinge loss terms, respectively. The interpolated frame loss adopts the uncertainty-based weighting:

$$
L _ { \mathrm { f i n a l } } ^ { \prime } = \frac { 1 } { 2 } \cdot \gamma \cdot L _ { \mathrm { f i n a l } } - \frac { 1 } { 2 } \cdot \alpha \cdot \log ( \gamma ) ,\tag{9}
$$

TABLE I: Performance comparison between SWAGSplatting and six baseline models on two datasets. â indicates that higher values are better, while $\downarrow$ indicates that lower values are better. Red, orange, and yellow denote the best, second-best, and third-best results, respectively.  
SeaThru-NeRF
<table><tr><td>Scene</td><td colspan="2">Curacao</td><td colspan="2"></td><td colspan="2">Panama</td><td colspan="2">IUI-Redsea</td><td colspan="3">Japanese-Redsea</td><td colspan="3">Average</td></tr><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Instant-NGP [24]</td><td>27.9215</td><td>0.7013</td><td>0.3961</td><td>23.4341</td><td>0.6294</td><td>0.4361</td><td>20.6162</td><td>0.5594 0.6121</td><td>23.2061</td><td>0.6371</td><td>0.3294</td><td>23.7945</td><td>0.6318</td><td>0.4434</td></tr><tr><td>SeaThru-NeRF [5]</td><td>2.9162</td><td>0.8463</td><td>0.2984</td><td>26.9634</td><td>0.7261</td><td>0.3684</td><td>25.4989</td><td>0.7934 0.3684</td><td>21.8631</td><td>0.7569</td><td>0.3243</td><td>26.604</td><td>0.7807</td><td>0.3399</td></tr><tr><td>3DGS [4]</td><td>29.234</td><td>0.9127</td><td>0.190</td><td>30.7038</td><td>0.606</td><td>0.1560</td><td>25.2240</td><td>0.9005 0.1970</td><td>22.3902</td><td>0.8638</td><td>01960</td><td>2.8353</td><td>0.8594</td><td>0.1820</td></tr><tr><td>WaterSplatting [6]</td><td>31.4616</td><td>0.9256</td><td>0.1560</td><td>31.5367</td><td>09213</td><td>0.1145</td><td>22.1392</td><td>0.6598 0.2984</td><td>24.4323</td><td>0.8834</td><td>0.1494</td><td>27.3925</td><td>0.8475</td><td>0.871</td></tr><tr><td>UW-GS [2] RUSplatting [7]</td><td>29.9721 30.9557</td><td>0.9220</td><td>0.170</td><td>31.0023</td><td>0.9335</td><td>0.1432</td><td>28.5473</td><td>0.9243 0.1885</td><td>23.5013</td><td>0.8625</td><td>0.19166</td><td>28.2558</td><td>0..9106</td><td>0.1751</td></tr><tr><td></td><td></td><td>0.9318</td><td>01611</td><td>31.8683</td><td>0.9339</td><td>01396</td><td>29.8036</td><td>09292 0.1847</td><td>24.5436</td><td>0.8706</td><td>0.1809</td><td>2.2928</td><td>0.9164</td><td>0.166</td></tr><tr><td>SWAGSplatting (ours)</td><td>33.0521 0.9481</td><td>0.1428</td><td></td><td>32.0630</td><td>0.9389</td><td>0.1264 Submerged3D</td><td>30.3602</td><td>0.9306 0.1822</td><td>24.3617</td><td>0.8861</td><td>0.1772</td><td>29.9593</td><td>0.9260</td><td>0.1571</td></tr><tr><td colspan="9"></td><td colspan="6"></td></tr><tr><td>Scene</td><td></td><td>Cormoran</td><td></td><td></td><td> $\mathrm { I s r o }$ </td><td></td><td></td><td> $\mathrm { K w a j }$ </td><td></td><td>Tokai</td><td></td><td></td><td>Average</td><td></td></tr><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Instant-NGP [24]</td><td>19.5271</td><td></td><td>0.5505</td><td>20.1861</td><td></td><td>0.4146</td><td>18.1444</td><td>0.4682</td><td>18.2488</td><td>0.4082</td><td>0.5723</td><td></td><td>0.5102</td><td>0.4977</td></tr><tr><td>SeaThru-NeRF [5]</td><td>16.8816</td><td>0.5055 0.453</td><td>06118</td><td>21.5887</td><td>0.6588 06501</td><td>0.5025</td><td>20.7094</td><td>0.4535 0.5116 0.5240</td><td>17.6072</td><td>0.3720</td><td>0.6360</td><td>19.0266 19.1967</td><td>0.4968</td><td>0.5686</td></tr><tr><td>3DGS [4]</td><td>20.498</td><td>0.6756</td><td>0.3852</td><td>22.8198</td><td>0.8421</td><td>0.3008</td><td>24.0964</td><td>0.8485 02276</td><td>23.2345</td><td>0.6517</td><td>0.4017</td><td>22.6551</td><td>0.7545</td><td>03288</td></tr><tr><td>WaterSplatting [6]</td><td>21.0749</td><td>0.682</td><td>0.3846</td><td>26.2920</td><td>0.8517</td><td>0.2716</td><td>2.8613</td><td>08709 0.2162</td><td>23.1047</td><td>06362</td><td>0.965</td><td>24.5832</td><td>07518</td><td>0.3172</td></tr><tr><td>UW-GS [2]</td><td>20.6942</td><td>0.6800</td><td>0.3412</td><td>24.6752</td><td>0.8649</td><td>0.2683</td><td>28.4065</td><td>0.8868 0.2105</td><td>23.580</td><td>0.6578</td><td>0.3833</td><td>24.3410</td><td>0.7724</td><td>0.3008</td></tr><tr><td>RUSplatting [7]</td><td>21.9625</td><td>0.6846</td><td>0.362</td><td>27.8693</td><td>0.8680</td><td>0.2670</td><td>2.9058</td><td>0.8774 02253</td><td>24.4585</td><td>0.6599</td><td>0.3548</td><td>25.7990</td><td>0.7725</td><td>0.3033</td></tr><tr><td>SWAGSplatting (ours)</td><td>21.9794</td><td>0.7023</td><td>0.3482</td><td>28.1544</td><td>0.8782</td><td>0.2643</td><td>28.8320</td><td>0.8847</td><td>0.2161 25.6028</td><td>0.7391</td><td>0.3612</td><td>26.1422</td><td>0.8011</td><td>0.2974</td></tr></table>

where $\gamma$ is the learned uncertainty and Î± acts as the regularisation term. To ensure robust convergence, we adopt a coarseto-fine stage-wise training schedule.

â¢ Stage 1 (0â60% iterations): Emphasises global structure and robustness using $\ell _ { 1 }$ -based reconstruction loss and semantic alignment.

â¢ Stage 2 (60â100% iterations): Freezes geometric and semantic parameters of the Gaussians (position $f _ { x y z }$ , rotation $f _ { \mathrm { r o t a t i o n } } .$ , scale $f _ { \mathrm { s c a l e } }$ , and semantic feature $f _ { s } )$ and focuses on fine-grained appearance refinement. The $\ell _ { 1 }$ term is downweighted while the $\ell _ { 2 }$ component is strengthened to promote sharper details and accurate colour restoration.

This stage-wise scheme stabilises optimisation in noisy and sparse-view scenarios, mitigates overfitting to medium effects, and achieves visually consistent, high-fidelity reconstructions across diverse underwater conditions.

## C. Gaussian Primitives Reallocation

The 3DGS-based approaches adopt a strategy that uses position gradients as cues to perform point cloud densification. However, such densification can introduce redundancy in the point cloud [25] and limit rendering quality. Inspired by [26], we propose a 3D Gaussian Primitives Reallocation method that adjusts the distribution of the point cloud model by reallocating low-contribution 3D Gaussian primitives to regions exhibiting relatively large errors.

According to Speedy-splat [27], we define the importance score $\tilde { S } _ { i }$ for each Gaussian i as

$$
\tilde { S } _ { i } = \log \left| \nabla _ { I _ { G } } g _ { i } \nabla _ { I _ { G } } g _ { i } ^ { T } \right| ,\tag{10}
$$

which, as $g _ { i }$ is scalar and log is increasing monotonically, can be simplified to

$$
\tilde { S } _ { i } = \left( \nabla _ { I _ { G } } g _ { i } \right) ^ { 2 } ,\tag{11}
$$

where $I _ { G }$ is the rendered image of all Gaussians, and $\nabla _ { I _ { G } } g _ { i }$ is the gradient of $g _ { i }$ with respect to $I _ { G }$ â¢

To improve the reconstruction quality, we introduce an error score ${ \tilde { E } } _ { i }$ that employs the same workflow to quantify the contribution of Gaussian primitives to reconstruction loss $L _ { \mathrm { R e c } }$ as follows:

$$
\tilde { E } _ { i } = \left( \nabla _ { L _ { \mathrm { R e c } } } g _ { i } \right) ^ { 2 }\tag{12}
$$

Every certain number of iterations after densification (3000 in this paper), we compute the importance score $\tilde { S } _ { i }$ and the error score ${ \tilde { E } } _ { i }$ for each primitive. The primitives in the bottom 10% according to ${ \tilde { S } } _ { i } .$ , which are considered redundant, are then removed. The freed point budget is subsequently used to densify the primitives in the top 10% ranked by $\tilde { E } _ { i }$ via a cloning operation, in which a new Gaussian is created at the same location with attributes identical to those of the original primitive, and later optimisation naturally separates the duplicated Gaussians to better represent the scene.

The 3D Gaussian Primitives Reallocation strategy recycles unimportant points to aid the reconstruction of regions that suffer from high errors. This not only alleviates the imbalance introduced by densification but also mitigates overfitting, while keeping the total number of 3D Gaussians unchanged.

## V. EXPERIMENT AND RESULTS

## A. Experiment Setting

All experiments are conducted on a single NVIDIA RTX 4090 GPU. Each training session runs for 20,000 iterations, with the second optimisation stage commencing at the $1 2 { , } 0 0 0 ^ { \mathrm { t h } }$ iteration. To improve computational efficiency, the reference semantic embedding $f _ { \mathrm { r e f } }$ is precomputed at the start of training and cached for subsequent use. The dimensionality of the projected CLIP embedding space d is set to 32. For frame interpolation, we employ a fixed weighting factor of 0.1 rather than adaptive weights, providing a stable balance between reconstruction and interpolation quality across datasets.

<!-- image-->  
Fig. 3: Novel view rendering comparison. The first row shows results from the IUI-Redsea scene from the SeaThru-NeRF dataset, and the second row shows the reconstructed scenes of the Isro from the Submerged3D dataset. The left side of the third row displays the reconstructed scene of the Tokai from the Submerged3D, while the right side shows the Japanese-Redsea.

## B. Datasets and evaluation metrics

We evaluate our method on two public underwater datasets: SeaThru-NeRF [5] and Submerged3D [7], each containing four representative underwater scenes. All images are resized to a resolution of 720p for consistency across experiments. Our approach is compared against six state-of-the-art baselines: Instant-NGP [24], SeaThru-NeRF [5], 3DGS [4], WaterSplatting [6], UW-GS [2], and RUSplatting [7]. For fairness, we use the official implementations of all baseline methods and train each model on identical image sequences following our dataset split protocol. Quantitative performance is assessed using three standard metrics: PSNR, SSIM and LPIPS, which jointly capture pixel accuracy, structural integrity, and perceptual quality.

## C. Quantitative Comparisons

Tab. I presents a quantitative comparison of SWAGSplatting against six state-of-the-art baselines, reporting the rendering performance across all scenes for both datasets. Our method consistently outperforms existing approaches across all three metrics. Compared to RUSplatting, SWAGSplatting achieves, on average, a 0.67 dB improvement in PSNR, a 1.05% increase in SSIM, and a 5.70% reduction in LPIPS on the SeaThru-NeRF dataset. Relative to UW-GS, it yields average gains of 1.80 dB in PSNR and 3.72% in SSIM, along with an average 1.13% decrease in LPIPS across all scenes in the Submerged3D dataset. These results demonstrate the effectiveness of our semantic-guided, stage-wise optimisation and Gaussian reallocation strategies in enhancing both structural consistency and perceptual quality under challenging underwater conditions.

## D. Qualitative Comparisons

The qualitative results in Fig. 3 further demonstrate the superior visual performance of SWAGSplatting compared to existing methods. Competing approaches struggle to reconstruct scene geometry and background clarity, as shown in the yellow and red highlighted regions. For instance, most baselines fail to recover the fine, spiny structure of coral or produce consistent background textures under scattering conditions. In contrast, SWAGSplatting preserves these intricate details, yielding sharper and more faithful reconstructions. The second row in Fig. 3 illustrates the methodâs ability to restore both geometric accuracy and underwater colour balance, where alternatives often exhibit visible artefacts (e.g., the orange region in RUSplatting) or blurred reconstructions (e.g., pink area in UW-GS). The third row further confirms our modelâs advantage in maintaining fine structural features and high perceptual quality. Overall, SWAGSplatting delivers clearer, more stable, and semantically coherent reconstructions across diverse underwater scenes.

## E. Ablation Study

To verify the contribution of each component within SWAGSplatting, we conduct a series of ablation experiments.

TABLE II: Ablation results averaged over all scenes from the SeaThru-NeRF and Submerged3D datasets. Red, orange, and yellow denote the best, second-best, and third-best results, respectively.
<table><tr><td>Variant</td><td>SG</td><td>SO</td><td>PR</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>M1</td><td>X</td><td>&gt;Ã</td><td>&gt;&gt;</td><td>27.4453</td><td>0.8373</td><td>0.2426</td></tr><tr><td>M2</td><td>â</td><td></td><td></td><td>27.8131</td><td>0.8511</td><td>0.2355</td></tr><tr><td>M3</td><td>â</td><td>:</td><td>Ã&gt;</td><td>27.8604</td><td>0.8632</td><td>0.2297</td></tr><tr><td>Ours</td><td>â</td><td></td><td></td><td>28.0508</td><td>0.8636</td><td>0.2272</td></tr></table>

SG: Semantic-guided Gaussians; SO: Stage-wise Optimisation; PR: Primitives Reallocation

The corresponding configurations and quantitative results are summarised in Tab. II. Starting from the full model, removing any individual component consistently degrades reconstruction quality across all metrics, confirming that each module contributes positively. In particular, the variants M1, M2 and M3 all exhibit lower PSNR/SSIM and higher LPIPS than the full SWAGSplatting model. It is worth highlighting that removing primitive reallocation (M3) leads to a relatively smaller drop compared to removing other components. This is expected because primitive reallocation mainly improves the efficiency of Gaussian allocation by redistributing Gaussians from lowimportance areas to high-error regions. Such improvements mainly affect challenging local regions, whose impact is less pronounced when averaged over entire scenes with a fixed number of Gaussians. Overall, the full SWAGSplatting model achieves the best performance, indicating that the three components provide complementary gains when combined.

## VI. CONCLUSIONS

This paper introduces SWAGSplatting, a semantic-guided 3D Gaussian Splatting framework designed for robust and high-fidelity underwater scene reconstruction. Each Gaussian primitive is augmented with a learnable semantic feature, supervised by CLIP-based embeddings to enforce semanticâgeometric consistency. A dedicated semantic loss guides the network toward preserving high-level structural relationships, resulting in perceptually faithful reconstructions. Furthermore, we propose a stage-wise optimisation strategy to enhance training stability and a 3D Gaussian primitive reallocation strategy to improve visual detail. Primitive reallocation can enhance reconstruction performance by redistributing the point cloud within the same point budget. Together, these innovations enable SWAGSplatting to achieve accurate, consistent, and semantically coherent reconstructions across challenging underwater environments, setting a new benchmark for underwater neural rendering. The code will be released upon acceptance.

## REFERENCES

[1] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Â¨ Drettakis, â3D gaussian splatting for real-time radiance field rendering,â ACM Transactions on Graphics, vol. 42, no. 4, July 2023.

[2] Haoran Wang, Nantheera Anantrasirichai, Fan Zhang, and David Bull, âUW-GS: Distractor-aware 3d gaussian splatting for enhanced underwater scene reconstruction,â in WACV, 2025.

[3] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng, âNeRF: Representing scenes as neural radiance fields for view synthesis,â Communications of the ACM, vol. 65, no. 1, pp. 99â106, 2021.

[4] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d gaussian Â¨ splatting for real-time radiance field rendering,â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[5] D. Levy, A. Peleg, N. Pearl, D. Rosenbaum, D. Akkaynak, S. Korman, and T. Treibitz, âSeathru-nerf: Neural radiance fields in scattering media,â in CVPR, 2023, pp. 56â65.

[6] H. Li, W. Song, T. Xu, A. Elsig, and J. Kulhanek, âWaterSplatting: Fast underwater 3D scene reconstruction using gaussian splatting,â 3DV, 2025.

[7] Z. Jiang, H. Wang, G. Huang, B. Seymour, and N. Anantrasirichai, âRUSplatting: Robust 3d gaussian splatting for sparse-view underwater scene reconstruction,â in BMVC, 2025.

[8] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, et al., âLearning transferable visual models from natural language supervision,â in ICML, 2021, pp. 8748â8763.

[9] A. Ramazzina, M. Bijelic, S. Walz, A. Sanvito, D. Scheuble, and F. Heide, âScatternerf: Seeing through fog with physically-based inverse neural rendering,â in ICCV, 2023, pp. 17957â17968.

[10] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng, âNeRF: Representing scenes as neural radiance fields for view synthesis,â in Computer Vision â ECCV 2020, Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm, Eds., 2020, pp. 405â421.

[11] D. Akkaynak and T. Treibitz, âA revised underwater image formation model,â in CVPR, 2018, pp. 6723â6732.

[12] L. Chen, Y. Xiong, Y. Zhang, R. Yu, L. Fang, and D. Liu, âSp-seanerf: Underwater neural radiance fields with strong scattering perception,â Computers & Graphics, vol. 123, pp. 104025, 2024.

[13] A. V. Sethuraman, M. S. Ramanagopal, and K. A. Skinner, âWaterNeRF: Neural radiance fields for underwater scenes,â in OCEANS, 2023.

[14] J. Zhou, T. Liang, D. Zhang, S. Liu, J. Wang, and E. Q. Wu, âWaterHE-NeRF: Water-ray matching neural radiance fields for underwater scene reconstruction,â Information Fusion, vol. 115, pp. 102770, 2025.

[15] Y. Tang, C. Zhu, R. Wan, C. Xu, and B. Shi, âNeural underwater scene representation,â in CVPR, 2024, pp. 11780â11789.

[16] L. Gough, A. Azzarelli, F. Zhang, and N. Anantrasirichai, âAquaNeRF: Neural radiance fields in underwater media with distractor removal,â in ISCAS, 2025.

[17] S. Liu, J. Lu, Z. Gu, J. Li, and Y. Deng, âAquatic-GS: A hybrid 3d representation for underwater scenes,â arXiv:2411.00239, 2024.

[18] D. Yang, J. J. Leonard, and Y. Girdhar, âSeasplat: Representing underwater scenes with 3D gaussian splatting and a physically grounded image formation model,â in ICRA, 2025.

[19] T. Zhang, W. Zhi, B. Meyers, N. Durrant, et al., âRecGS: Removing water caustic with recurrent gaussian splatting,â IEEE Robotics and Automation Letters, vol. 10, no. 1, pp. 668â675, 2025.

[20] G. Huang, H. Wang, Z. Qi, W. Lu, D. Bull, and N. Anantrasirichai, âFrom restoration to reconstruction: Rethinking 3D gaussian splatting for underwater scenes,â arXiv:2509.17789, 2025.

[21] J. Yi, Q. Bi, H. Zheng, H. Huang, H. Zhan, et al., âAtlantisGS: Underwater sparse-view scene reconstruction via gaussian splatting,â in ACM MM, 2025, pp. 7805â7814.

[22] J. Chen, Z. Xu, X. Pan, Y. Hu, C. Qin, et al., âBlip3-o: A family of fully open unified multimodal models-architecture, training and dataset,â arXiv:2505.09568, 2025.

[23] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan, et al., âGrounded sam: Assembling open-world models for diverse visual tasks,â arXiv preprint arXiv:2401.14159, 2024.

[24] T. Muller, A. Evans, C. Schied, and A. Keller, âInstant neural graphics Â¨ primitives with a multiresolution hash encoding,â TOG, vol. 41, no. 4, pp. 1â15, 2022.

[25] G. Fang and B. Wang, âMini-splatting: Representing scenes with a constrained number of gaussians,â in ECCV, 2024, pp. 165â181.

[26] J. Zhu, J. Yue, F. He, and H. Wang, â3D student splatting and scooping,â in CVPR, 2025, pp. 21045â21054.

[27] A. Hanson, A. Tu, G. Lin, V. Singla, M. Zwicker, and T. Goldstein, âSpeedy-splat: Fast 3d gaussian splatting with sparse pixels and sparse primitives,â in CVPR, 2025, pp. 21537â21546.