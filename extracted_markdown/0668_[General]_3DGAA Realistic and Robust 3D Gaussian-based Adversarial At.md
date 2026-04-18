# 3DGAA: Realistic and Robust 3D Gaussian-based Adversarial Attack for Autonomous Driving

Yixun Zhang, Lizhi Wang, Junjun Zhao, Wending Zhao, Feng Zhou, Yonghao Dang, and Jianqin Yin\* School of Intelligent Engineering and Automation, Beijing University of Posts and Telecommunications, China

{zhangyixun, wanglizhi, zhaojunjun, windyz, zhoufeng, dyh2018, jqyin}@bupt.edu.cn

## Abstract

Camera-based object detection systems play a vital role in autonomous driving, yet they remain vulnerable to adversarial threats in real-world environments. Existing 2D and 3D physical attacks, due to their focus on texture optimization, often struggle to balance physical realism and attack robustness. In this work, we propose 3D Gaussianbased Adversarial Attack (3DGAA), a novel adversarial object generation framework that leverages the full 14- dimensional parameterization of 3D Gaussian Splatting (3DGS) to jointly optimize geometry and appearance in physically realizable ways. Unlike prior works that rely on patches or texture optimization, 3DGAA jointly perturbs both geometric attributes (shape, scale, rotation) and appearance attributes (color, opacity) to produce physically realistic and transferable adversarial objects. We further introduce a physical filtering module that filters outliers to preserve geometric fidelity, and a physical augmentation module that simulates complex physical scenarios to enhance attack generalization under real-world conditions. We evaluate 3DGAA on both virtual benchmarks and physical-world setups using miniature vehicle models. Experimental results show that 3DGAA achieves to reduce the detection mAP from 87.21% to 7.38%, significantly outperforming existing 3D physical attacks. Moreover, our method maintains high transferability across different physical conditions, demonstrating a new state-of-the-art in physically realizable adversarial attacks.

## 1. Introduction

Camera-based object detection systems serve as a cornerstone of autonomous driving [1, 2, 38], enabling essential tasks such as obstacle avoidance and lane navigation. However, increasing researches show that these models are highly vulnerable to adversarial examples. In physical world, subtle, structured perturbations often induce misclassifications [4, 10, 11, 28, 31, 48]. This raises serious concerns about the reliability of perception modules in safety-critical scenarios. Thus, systematically exploring such vulnerabilities is essential to understand and improve the real world safety of autonomous driving systems [3, 7, 15, 16, 26, 29, 33, 34, 39â41, 43, 46, 47].

Existing physical adversarial attacks typically fall into two categories: 2D patch-based methods and 3D texturelevel attacks. 2D methods typically apply adversarial patches onto objects [3, 7, 15, 39], achieving high physical realism through minimal appearance changes, but often suffer from poor robustness across viewpoints or physical conditions (Challenge 1). In contrast, recent 3D attacks [26, 33, 34, 40, 41, 43, 46] achieve stronger and more consistent multi-view adversarial performance by directly manipulating surface textures. However, they often rely solely on appearance perturbations, which may result in visual artifacts or unrealistic surface distortions. This limits their applicability in safety-critical settings, where high physical realism is essential for real-world deployment (Challenge 2). This reveals an inherent trade-off between physical realism and adversarial robustness, posing a key challenge for deploying such attacks in real-world scenarios where both are critical (Challenge 3).

To address these challenges, we introduce 3DGAA, an adversarial attack framework that performs joint optimization of geometric and appearance perturbations to generate physically realistic adversarial objects. This is achieved by leveraging 3D Gaussian Splatting (3DGS) [19], a differentiable and compact 3D representation originally designed for photorealistic rendering. Unlike mesh-based or pointbased representations, 3DGS encodes both shape and texture in a unified 14-dimensional parameter space, enabling fine-grained and physically consistent adversarial manipulation [14, 35, 36].

3DGAA comprises three key modules that jointly address the trade-off between physical realism and attack robustness: First, a Physically-Constrained Adversarial Optimization stage jointly perturbs geometry and appearance attributes to degrade detector confidence across diverse viewpoints and camera distances. This overcomes the limited perturbation scope of 2D methods and addresses Challenge 1. Second, we propose a Physical Filtering Module that enhances geometric fidelity by removing topological outliers and denoising structural artifacts. This module enforces surface-level consistency and mitigates unrealistic deformations commonly seen in texture-only adversarial models, thereby preserving the visual plausibility of the 3D object from multiple perspectives. It further enhances physical reality, tackling Challenge 2. Furthermore, our framework leverages the expressive and differentiable nature of 3DGS to strike a unique balance between these two conflicting objectives. We introduce a Physical Augmentation Module that injects environmental variationsâsuch as imaging noise, photometric distortions, shadows, and occlusionsâinto the optimization loop. This improves generalization under realworld conditions, addressing Challenge 3.

Extensive experiments on both virtual benchmarks and real-world miniature vehicle setups validate our approach. Specifically, 3DGAA reduces detection mAP from 87.21% to 7.38%, and consistently succeeds across different lighting and viewpoint scenarios. These results demonstrate its strong physical realism and attack robustness. As shown in Fig. 1, 3DGAA achieves superior physical deployment effectiveness and occupies the optimal region in the realismrobustness space, outperforming existing 2D and 3D baselines. Our contributions are summarized as follows:

â¢ We propose 3DGAA, the novel adversarial framework to adapt 3D Gaussian Splatting for physical adversarial object generation, enabling joint optimization of geometry and appearance in its native parameter space.

â¢ We design two novel modules: a physical filtering module leveraging topological pruning and structural denoising to enforce geometric fidelity, and a physical augmentation module to simulate environmental conditions and improve robustness in camera-based object detection systems.

â¢ We conduct extensive experiments of the proposed framework, including real-world deployment using miniature vehicle models. The results show a stong attack performance, with a detection degradation of 79.83%, verifying the physical effectiveness and cross-domain transferability of 3DGAA.

## 2. Related Work

Physical adversarial attacks. In autonomous driving scenarios, physical adversarial attacks aim to deceive realworld perception systems by perturbing object appearance in a deployable manner. Early methods [3, 7, 20, 24, 39] mainly adopt 2D patch-based perturbations rendered on printable surfaces. UPC [15] introduces universal physical patches that generalize across scenarios, DAS [41] exploits differentiable mesh rendering [18] to inject perturbation via attention manipulation. Although these approaches offer strong attack performance, they often exhibit limited robustness under varying viewpoints and lighting.

<!-- image-->  
(a) Real-world deployment: object goes undetected.

<!-- image-->  
(b) Realism vs. Robustness comparison across methods.  
Figure 1. (a) Physical deployment of 3DGAA on a miniature vehicle, resulting in detection failure. (b) Comparison of adversarial methods in terms of physical realism (PSNR) and attack effectiveness (mAP). 3DGAA achieves state-of-the-art performance on both axes.

To improve generalization, recent works leverage 3D object representations and render adversarial textures onto full object meshes [29, 33, 40, 41, 43, 46, 47]. In particular, FCA [40] generates full-surface camouflage with physical constraints. Environment-aware enhancements [33, 34] further boost robustness, and recent works such as TT3D [16] and PGA [26] explore high-dimensional texture search spaces.

Despite their improved transferability, most existing physical attacks rely solely on texture perturbations, which limit their expressiveness and fail under geometric distortions or sensor variations [17, 22, 27]. These methods lack the ability to jointly manipulate geometric cues such as shape, scale, or position, factors critical for consistent realworld perception. To overcome this limitation, we propose a new class of physical adversarial attacks that extend beyond appearance-only changes.

Our approach leverages 3D Gaussian Splatting [19], a differentiable and expressive 3D representation, to jointly optimize both geometry and appearance in a unified 14- dimensional space [5, 9, 14, 21, 35]. This formulation enables physically realistic perturbations that remain effective under variable real-world conditions [37, 44]. Unlike prior works constrained to textures, our method introduces perturbation through joint optimization of geometry and appearance, enabling physically realistic and adversarially robust object perturbations for evaluating robust perception systems.

## 3. Method

## 3.1. Preliminaries

Formally, we assume access to n multi-view RGB images $\{ I _ { i } \} _ { i = 1 } ^ { n }$ of a target object, captured under calibrated camera poses $\{ P _ { i } \} _ { i = 1 } ^ { n }$ These inputs are fed into a pretrained 3D Gaussian Splatting (3DGS) generation network to generate a base object $\mathcal { G } = \{ \mathbf { g } _ { j } \} _ { j = 1 } ^ { N }$ , where each Gaussian primitive $\mathbf { g } _ { j } \in \mathbb { R } ^ { 1 4 }$ is parameterized by: Position $\mathbf { x } \in \mathbb { R } ^ { 3 }$ , Rotation $\mathbf { q } \in \mathbb { R } ^ { 4 }$ (quaternion), Scale $\mathbf { s } \in \mathbb { R } ^ { 3 }$ , Color $\mathbf { c } \in \mathbb { R } ^ { 3 }$ , Opacity $\alpha \in \mathbb { R }$ . This 14-dimensional representation allows for finegrained control over both geometry and appearance, making it highly suitable for physical adversarial object generation. The 3DGS generation process can be formulated as:

<!-- image-->  
Figure 2. Overview of the proposed 3DGAA pipeline. (i) Given multi-view images and camera calibration, a pretrained backbone generates a vanilla 3D Gaussian Splatting (3DGS) object. (ii) A physical filtering module applies topological pruning and structural denoising to improve physical plausibility. (iii) The adversarial optimization stage perturbs the filtered 3DGS using an adversarial loss $L _ { \mathrm { a d v } }$ to deceive object detectors and a shape loss $L _ { \mathrm { s h a p e } }$ to maintain physical consistency. (iv) The final adversarial 3DGS is trained under physical augmentations including different physical variations.

$$
\mathcal { G } = \mathcal { F } _ { 3 \mathrm { D G S } } \big ( \{ I _ { i } , P _ { i } \} _ { i = 1 } ^ { n } \big ) ,\tag{1}
$$

where $\mathcal { F } _ { \mathrm { 3 D G S } }$ denotes the pretrained Gaussian generation network that maps calibrated RGB images to a set of $N$ Gaussians $\mathcal { G } = \{ \mathbf { g } _ { j } \} _ { j = 1 } ^ { N }$ , with each $\mathbf { g } _ { j } ~ \in \mathbb { R } ^ { 1 4 }$ encoding geometry and appearance attributes.

Optimizing target. Our goal is to perturb the Gaussian parameters such that the object, when rendered into a scene and processed by a detector D, minimizes the predicted confidence for its true class. Let $R ( { \mathcal { G } } , P )$ denote the rendered image from pose P , the core adversarial goal can be formulated as:

$$
\operatorname* { m i n } _ { \mathcal { G } } L _ { \mathrm { a d v } } = \mathbb { E } _ { P \sim \mathcal { V } } \left[ f _ { D } ( R ( \mathcal { G } , P ) ) \right]\tag{2}
$$

where $f _ { D } ( \cdot )$ denotes the detection confidence of the target class and V denotes a distribution over camera viewpoints.

## 3.2. Overview

As shown in Figure 2, our proposed 3DGAA framework generates adversarial 3D objects through a four-stage pipeline: i. 3D Gaussian Generation: Multi-view images and camera parameters are fed into a pretrained backbone to generate an initial 3DGS object. ii. Physical Filtering: Structural artifacts and topological noise are removed to enhance physical realism. iii. Adversarial Optimization: The filtered object is optimized using an adversarial loss $L _ { \mathrm { a d v } }$ and a shape consistency loss $L _ { \mathrm { s h a p e } }$ . iv. Physical Augmentation: Differentiable augmentations simulate environmental conditions to improve real-world robustness. Together, these modules produce adversarial objects that are both physically realistic and robust across diverse viewpoints and environments.

## 3.3. Physically-Constrained Adversarial Optimization

To address the limited perturbation capability of traditional methods, we propose a physically-constrained adversarial optimization framework that perturbs both the geometry and texture of the 3DGS representation. This high-degreeof-freedom space enables more expressive and robust adversarial behavior under varying physical conditions.

Adversarial Loss. To suppress the detectorâs confidence on the target class, we minimize the expected detection score across multiple viewpoints:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { a d v } } = \mathbb { E } _ { P \sim \mathcal { V } } \left[ f _ { D } ( R ( \mathcal { G } , P ) ) \right] , } \end{array}\tag{3}
$$

where $R ( { \mathcal { G } } , P )$ is the rendered image of G from camera pose $P ,$ and $f _ { D } ( \cdot )$ denotes the predicted confidence score of the target object output by the pretrained detector D. Shape Consistency Loss. To preserve the physical reality of the objectâs geometry, we introduce a shape constraint that penalizes deviations from the original structure:

$$
\mathcal { L } _ { \mathrm { s h a p e } } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \left\| \mathbf { p } _ { j } - \mathbf { p } _ { j , 0 } \right\| _ { 2 } ^ { 2 } + \left\| \mathbf { s } _ { j } - \mathbf { s } _ { j , 0 } \right\| _ { 2 } ^ { 2 } + \left\| \mathbf { q } _ { j } \otimes \mathbf { q } _ { j , 0 } ^ { - 1 } \right\| _ { 2 } ^ { 2 }\tag{4}
$$

where $\mathbf { p } _ { j , 0 } , \mathbf { s } _ { j , 0 }$ and $\mathbf { q } _ { j , 0 }$ are the initial position, scale and rotation before optimization. â denotes quaternion multiplication. By using such a shape consistency loss, we can constrain the deformation of 3DGS during the adversarial attack process, making the adversarial perturbation not mainly concentrated in the geometry dimensions, which has a negative impact on physical reality.

Total Loss and Optimization. The final objective combines the adversarial and shape consistency terms:

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { a d v } } \cdot \mathcal { L } _ { \mathrm { a d v } } + \lambda _ { \mathrm { s h a p e } } \cdot \mathcal { L } _ { \mathrm { s h a p e } } . } \end{array}\tag{5}
$$

To dynamically balance attack strength and geometric fidelity, we set $\lambda _ { \mathrm { a d v } }$ to be inversely proportional to the adversarial loss and $\lambda _ { \mathrm { s h a p e } }$ to follow the unnormalized shape deviation. This allows the optimization to prioritize realism when the attack is already strong, and vice versa.

Algorithm 1 summarizes the optimization loop. The 3DGS parameters are iteratively updated via gradient descent, where each iteration involves rendering, applying physical augmentations, computing adversarial and shape losses, and selectively applying gradients to relevant parameter subsets. The dynamic weights $\lambda _ { \mathrm { a d v } }$ and $\lambda _ { \mathrm { s h a p e } }$ are adjusted according to the magnitude of the current loss to balance the strength of the attack and the preservation of the realism. Further details are provided in Appendix A.1.

## 3.4. Physical Filtering Module

Despite the flexibility of 3DGS representations, direct optimization over the entire parameter space often results in artifacts such as isolated floating Gaussians and spatial jitter. These issues compromise the physical reality of the resulting objects, especially under multi-view rendering. We propose a Physical Filtering Module to refine the optimized 3D object and enforce geometric consistency.

Topological Pruning (TP). We first remove outlier Gaussians based on projected spatial density. For each Gaussian, we compute its neighborhood density in screen space across sampled viewpoints. If a Gaussian is consistently located in sparse regions or lacks sufficient overlap with other primitives, it is discarded. This step eliminates fragmented components and prevents unnatural floaters from appearing in rendered views.

Structural Denoising (SD). We further smooth the object by adjusting the scale of neighboring Gaussians to reduce abrupt local variations. We apply a Gaussian-weighted smoothing filter over scale dimension, guided by projected image-space distance and opacity. This encourages gradual transitions in apparent size and reduces aliasing artifacts caused by irregular primitive distribution.

Formally, the filtering process involves (1) pruning lowdensity Gaussians based on adaptive neighborhood analysis, and (2) applying a structural smoothing function in scales and opacities. Full algorithmic details are provided in Appendix A.2.

## 3.5. Physical Augmentation Module

To improve the real-world transferability of adversarial 3D objects, we introduce a differentiable Physical Augmentation Module that simulates deployment-time environmental variations. This module injects physically realistic perturbations into rendered views during optimization and is composed of four sub-transforms:

Imaging Degradation. We simulate sensor noise and lens blur through depth-aware Gaussian noise ${ \mathcal { T } } _ { n o i s e } ,$ where the noise level increases with object-camera distance. This models degradation from low-quality hardware or atmospheric scattering.

Photometric Variation. We apply color distortions via channel-wise affine transforms $\mathcal { T } _ { p h o t o }$ , introducing random brightness and contrast changes to emulate lighting inconsistencies, exposure drift, and white balance variation.

Shadow Projection. To account for natural illumination effects, we simulate soft shadow overlays using depth-based sigmoid masks $\mathcal { T } _ { s h a d o w }$ These synthetic shadows mimic occlusion by other objects or self-casting.

Adaptive Occlusion. We introduce random rectangular occluders with variable transparency $\mathcal { T } _ { o c c l }$ to simulate realworld clutter and partial obstructions, enhancing robustness to unexpected foreground elements.

Each transformation is applied in sequence to the rendered image: $\mathcal { T } = { \mathcal { T } } _ { o c c l } \circ { \mathcal { T } } _ { s h a d o w } \circ { \mathcal { T } } _ { p h o t o } \circ { \mathcal { T } } _ { n o i s e }$ . All operations are differentiable and applied online at every training epoch, allowing gradients to flow through the augmentation pipeline. This enables adversarial optimization to produce samples that remain effective under realistic sensing and environmental distortions. Additional formulation details are provided in Appendix A.3.

## 3.6. Dimension Selection for Physical Realization

Real-world deployment imposes a non-deformability constraint on vehicles. To explicitly accommodate this, we expose a dimension-selection switch in 3DGAA that partitions the 3DGS parameters into $\theta _ { \mathrm { a p p } } = \{ \mathrm { r g b , o p a c i t y } \}$ (appearance) and $\theta _ { \mathrm { g e o } } = \{ \mathrm { p o s } , \mathrm { s c a l e } , \mathrm  \tilde { r o t } \}$ (geometry), and instantiate three modes:

Appearance-only 3DGAAa. We freeze $\theta _ { \mathrm { g e o } }$ and optimize only $\theta _ { \mathrm { a p p } }$ . In this setting, the resulting adversarial object can be fabricated by printable wraps/films without any geometric alteration (see Physical Realization). Technically, this mode achieves the best perceptual fidelity among attack methods (Table 1: lowest LPIPS and highest PSNR.

Algorithm 1 Adversarial 3DGS Optimization   
Require: $T \colon$ Total iterations   
$\theta _ { 0 } \colon$ Initial 3DGS parameters   
$R ( \cdot ) \colon$ 3DGS differentiable renderer   
$\tau ( \cdot ) { : }$ Physical augmentation function   
$\mathcal { L } _ { \mathrm { a d v } } \mathrm { : }$ Adversarial loss   
$\mathcal { L } _ { \mathrm { { s h a p e } } } \mathrm { . }$ Shape consistency loss   
$\kappa \colon$ Selected dimensions for update   
Ensure: Optimized 3DGS parameters Î¸   
1: $\theta  \theta _ { 0 }$ â· Initialization   
2: for t = 1 to T do   
3: $I  R ( \theta )$ â· Render current 3DGS   
4: $\tilde { I }  \mathcal { T } ( I )$ â· Apply physical augmentation   
5: Compute weights $\lambda _ { \mathrm { a d v } } , \lambda _ { \mathrm { s h a p e } }$ based on current loss values   
6: $\delta \gets \nabla _ { \theta } \left[ \lambda _ { \mathrm { a d v } } \cdot \mathcal { L } _ { \mathrm { a d v } } ( \tilde { I } ) + \lambda _ { \mathrm { s h a p e } } \cdot \mathcal { L } _ { \mathrm { s h a p e } } ( \theta ) \right]$   
7: $\theta  \theta - \mathbf { \bar { M a s k } } _ { \mathcal { K } } ( \delta )$ â· Update selected dimensions only   
8: end for   
9: return Î¸

Geometry-only 3DGAAg. We fix $\theta _ { \mathrm { a p p } }$ and optimize $\theta _ { \mathrm { g e o } }$ under tight displacement bounds implicitly encoded by $L _ { \mathrm { s h a p e } } .$ This ablates the contribution of geometric cues and keeps competitive realism (Table 1) while exposing how pose/scale perturbations contribute to detector failures.

Full (3DGAA). We jointly optimize $\theta _ { \mathrm { a p p } } \cup \theta _ { \mathrm { g e c } }$ with $L _ { \mathrm { s h a p e } } .$ This configuration provides the strongest attack effect (highest LCR, lowest mAP in Table 1) at a small realism cost relative to 3DGAAa).

The switchable design aligns optimization with physical constraints: 3DGAAa directly supports geometry-free, printable deployment on vehicles; 3DGAAg isolates geometric sensitivity; and 3DGAA realizes the upper bound of attack efficacy. In practice, 3DGAAg converges faster and with lower memory due to frozen appearance, while 3DGAA achieves the best robustness. Moreover, 3DGAAa serves as the methodological precursor to our fabrication pipeline (see Physical Realization) and the comprehensive comparisons in Exp. 4.2 and 4.7.

## 4. Experiments

## 4.1. Experimental Setup

Baseline Methods. We compare 3DGAA against representative open-source 2D and 3D physical attack baselines, including CAMOU [46], UPC [15], DAS [41], FCA [40], DTA [33], ACTIVE [34], TT3D [16], RAUCA [47] and PACG [29]. For a fair comparison, all methods are adapted to the same synthetic and real-world setups with standard calibration and consistent rendering settings.

Object & Invariance. All physical variants use the same rigid miniature shell; geometry is unchanged and only textures differ (Sec. 4.7). Textures used for fabrication are produced by the appearance-only pipeline 3DGAAa (Sec. 3.6).

Synthetic Scene Selection. We adopt CARLA [8] sampling with fixed seeds to collect the images, covering 20 vehicle models with varying geometries, 5 weather conditions, multi-scale observations and full spherical viewpoints (24 azimuth/elevation angles).

Target Models. We evaluate detection attacks on six widely used object detectors: Faster R-CNN [32], Mask R-CNN [13], SSD [23], YOLOv3 [30], YOLOv5 and YOLOv8. For transferable task segmentation, we use DeepLabv3 [6] and FCN [25] with ResNet-50 and ResNet-101 [12] backbones.

Evaluation Metrics. We report attack effectiveness using the Log Confidence Reduction (LCR), which measures the proportion of images where the detector fails to assign the correct label to the adversarial object:

$$
\mathrm { L C R } = \log \left( { \frac { \mathrm { I n i t i a l ~ C o n f i d e n c e } } { \mathrm { F i n a l ~ C o n f i d e n c e } } } \right) .\tag{6}
$$

Unlike absolute confidence drop, LCR normalizes adversarial effectiveness across different baselines, ensuring a fairer comparison. We also report mean Average Precision with 50% threshold (mAP@0.5) to reflect detection performance decline. To quantify physical realism, we adopt image similarity metrics including LPIPS [45], SSIM [42], and PSNR, computed between rendered adversarial objects and their clean counterparts. All metrics are computed under the same view and lighting protocol as in Sec. 4.2 to ensure fairness.

Implementation Details. All experiments run on a single NVIDIA RTX 4090 with fixed driver/CUDA/PyTorch versions and fixed seeds. We use early stopping under a shared rule; minutes-level wall-clock runtime profiling (including 3DGS generation, adversarial training, and I/O/misc) and peak memory under this protocol are summarized in App. C.3. The two loss terms in our objectiveâadversarial loss ${ \mathcal { L } } _ { \mathrm { a d v } }$ and shape-consistency loss $\mathcal { L } _ { \mathrm { s h a p e } ^ { - \mathrm { a r e } } }$ dynamically weighted by normalizing relative magnitudes to balance gradients: the weight for ${ \mathcal { L } } _ { \mathrm { a d v } }$ is inversely proportional to its scaled value, while the weight for $\mathcal { L } _ { \mathrm { s h a p e } }$ is proportional to its raw value. This dynamic scheme eliminates manual tuning of fixed loss weights.

Generation Pipeline and Camera Setup. Digital rendering: we use a differentiable 3D Gaussian renderer following [35]. All adversarial objects are rendered under multi-view settings with 12 evenly spaced azimuth angles and 3 distances (3 m, 5 m, 10 m), aligned with the protocol in Sec. 4.2. Physical fabrication: textures are UVunwrapped and printed on matte wraps to reduce glare, then applied to the same rigid shell (no body modification); materials, printing ppi, and alignment cues are summarized in App. B.1. Physical images are captured with a Realme GT Neo5 SE; camera/exposure specifics and layout sketches are provided in App. B.1.

## 4.2. Comprehensive Comparison with Benchmark Physical Attacks

We compare 3DGAA with a range of representative physical adversarial attack methods, including both 2D patchbased and 3D texture-based approaches. Our evaluation considers two key dimensions: physical realism, how visually consistent the adversarial object is with its original counterpart, and attack effectiveness, how successfully it deceives object detection models.

Physical Realism. As shown in Table 1, 3DGAAa attains the lowest LPIPS (0.5218), with 3DGAA (0.5373) and 3DGAAg (0.5423) following as second and third best, respectively. While our SSIM is marginally lower than UPC, this largely reflects the inherent approximation of 3DGS geometry rather than adversarial manipulation. Importantly, 3DGAAa achieves the highest PSNR (0.5480) among attack methods, with 3DGAA second (0.4951) and 3DGAAg third (0.4741), indicating superior pixel-level fidelity. These results highlight the benefit of our shape-consistency loss and physical filtering in preserving realism, and show that appearance-only optimization can further tighten perceptual fidelity.

Attack Strength. 3DGAA significantly outperforms all baselines in adversarial robustness (Table 1). It achieves an LCR of 3.5628, exceeding the strongest baseline RAUCA (1.4705), and reduces mAP from 87.21% to 7.38%. Both variants also surpass all baselines: 3DGAAg reaches 3.4410 LCR / 8.03% mAP and 3DGAAa 2.6766 LCR / 13.64% mAP. This confirms that optimizing both geometry and appearance yields the strongest effect, while our selective dimension optimization (Method 3.6) still delivers SOTA degradation.

Viewpoint Robustness. We measure LCR across 12 azimuth angles and 3 distances. Figure 3 shows higher success rates at side views and close range, with performance slightly declining under front views or long distances, validating the multi-view training strategy and 3DGAAâs robustness to viewpoint variations.

Together, these results confirm that 3DGAA achieves strong attack success while preserving physical realism, and remains effective under diverse viewpoints. Moreover, the strong results of 3DGAAa verify a purely appearance-based and geometry-free deployment path, which is critical for physically deployable adversarial objects.

<!-- image-->

Figure 3. Polar plot of LCR across 12 viewpoints and 3 distances in simulation. Attacks are most effective at side views and short range.  
<!-- image-->  
(a) TP

<!-- image-->  
(b) SD

<!-- image-->  
(c) TP+SD  
Figure 4. Geometric refinement analysis through physical filtering module: (a) Topology pruning (b) Structural denoising (c) Topology pruning + Structural denoising. Color scale indicates local deformation energy. Note the artifacts above and below of the car, which are most obvious in (c), indicating the best removal effect.

## 4.3. Effectiveness of the Physical Filtering Module

To evaluate the impact of the physical filtering module introduced in Section 3.4, we compare the results of applying topology pruning (TP), structural denoising (SD), and their combination. As shown in Figure 4, the visual artifacts, such as floating Gaussians and irregular surface patches, are significantly reduced after filtering. The TP step removes geometric outliers, while SD smooths local scale inconsistencies. The combined TP+SD strategy yields the cleanest geometry and most plausible object structure.

Quantitatively, Table 2 shows that TP+SD improves perceptual realism (LPIPS: 0.5373), achieves the highest SSIM (0.0128) and removes 83.1% of artifacts, although with a slight decrease in PSNR due to suppression of highfrequency noise. This trade-off is acceptable, as it prioritizes geometric plausibility over strict pixel-level fidelity, aligning with our goal of producing physically realistic adversarial objects.

## 4.4. Effect of Shape Loss Design

We conduct an ablation study to assess the contribution of the shape loss $\mathcal { L } _ { s h a p e }$ introduced in Section 3.3. This loss aims to preserve geometric consistency while optimizing for adversarial effectiveness.

As shown in Table 3, adding the shape loss slightly improves LPIPS (0.5516 to 0.5503) and SSIM (0.0141 to 0.0146), indicating better perceptual quality and structural preservation. The PSNR value remains similar, with only a minor drop, reflecting a shift from pixel-level perturbation to more physically realistic adjustments. These results show that incorporating $\mathcal { L } _ { s h a p e }$ helps balance attack robustness and physical realism, aligning with our dual-objective design.

Table 1. Benchmark Comparison of 3DGAA against existing physical adversarial attacks. Besides the full model (3DGAA), we report two dimension-selection variants: 3DGAAa (appearance-only optimization) and 3DGAAg (geometry-only optimization). Across metrics, our approach consistently delivers superior physical realism and adversarial robustness.
<table><tr><td>Method</td><td>Vanilla</td><td>CAM.[46]</td><td>UPC[15]</td><td>DAS[41]</td><td>FCA[40]</td><td>DTA[33]</td><td>ACT.[34]</td><td>TT3D[16]</td><td>RAU.[47]</td><td>PACG[29]</td><td>3DGAAa</td><td>3DGAA9</td><td>3DGAA</td></tr><tr><td>LPIPS â</td><td>0.0000</td><td>0.6210</td><td>0.6161</td><td>0.5979</td><td>0.5629</td><td>0.6142</td><td>0.6105</td><td>0.5433</td><td>0.6056</td><td>0.6152</td><td>0.5218</td><td>0.5423</td><td>0.5373</td></tr><tr><td>SSIM â</td><td>1.0000</td><td>0.0212</td><td>0.0224</td><td>0.0132</td><td>0.0123</td><td>0.0198</td><td>0.0193</td><td>0.0122</td><td>0.0178</td><td>0.0204</td><td>0.0123</td><td>0.0193</td><td>.0128</td></tr><tr><td>PSNR â</td><td>â</td><td>0.1829</td><td>0.1749</td><td>0.4590</td><td>0.2987</td><td>0.1804</td><td>0.1814</td><td>0.3876</td><td>0.2572</td><td>0.1956</td><td>0.5480</td><td>0.4741</td><td>0.4951</td></tr><tr><td>LCR â</td><td>0.0000</td><td>0.3128</td><td>0.2095</td><td>0.5366</td><td>1.2810</td><td>0.8784</td><td>1.3778</td><td>1.1202</td><td>1.4705</td><td>1.0299</td><td>2.6766</td><td>3.4410</td><td>35628</td></tr><tr><td>mAP â</td><td>87.21%</td><td>70.21%</td><td>75.42%</td><td>60.12%</td><td>35.89%</td><td>47.44%</td><td>33.56%</td><td>40.12%</td><td>31.47%</td><td>42.71%</td><td>13.64%</td><td>8.03%</td><td>7.38%</td></tr></table>

Table 2. Filtering strategy comparison across different metrics. TP+SD achieves the best performance on physical realism and artifacts removal (AR).
<table><tr><td>Method</td><td>LPIPS â</td><td>SSIM â</td><td>PSNR â</td><td>AR (%) â</td></tr><tr><td>Vanilla</td><td>0.5443</td><td>0.0105</td><td>0.5282</td><td>0.0</td></tr><tr><td>TP</td><td>0.5435</td><td>0.0086</td><td>0.5251</td><td>19.8</td></tr><tr><td>SD</td><td>0.5386</td><td>0.0121</td><td>0.5015</td><td>72.2</td></tr><tr><td>TP + SD</td><td>0.5373</td><td>0.0128</td><td>0.4951</td><td>83.1</td></tr></table>

Table 3. Ablation study of the shape loss component. $\mathcal { L } _ { a d v } +$ $\mathcal { L } _ { s h a p e }$ achieves better physical realism.
<table><tr><td>Method</td><td>LPIPS â</td><td>SSIM â</td><td>PSNR â</td></tr><tr><td> $\mathcal { L } _ { a d v } \mathrm { ~ o n l y }$ </td><td>0.5516</td><td>0.0141</td><td>0.5884</td></tr><tr><td> $\mathcal { L } _ { a d v } + \mathcal { L } _ { s h a p e }$ </td><td>0.5503</td><td>0.0146</td><td>0.5881</td></tr></table>

<!-- image-->  
(a) Grouped optimization.

<!-- image-->  
(b) Single-dimension optimization.  
Figure 5. LCR sensitivity analysis. (a) Optimizing geometry dimensions leads to faster convergence, while texture yields higher final LCR. (b) Position and scale are most sensitive; rotation has minimal effect.

## 4.5. Selective Dimension Optimization

To analyze the sensitivity and effectiveness of individual 3DGS parameters during adversarial optimization, we perform a set of experiments that isolate geometric and appearance dimensions.

<!-- image-->  
Figure 6. Physical-world adversarial attack under varying lighting. Column: ID, IN, OD, ON. Row 1: vanilla, Row 2: w/o Aug., Row 3: w/ Aug. Only the fully augmented adversarial object consistently evades detection.

As shown in Figure 5a, optimizing geometric parameters (position x, scale s) leads to rapid increases in Log Confidence Reduction (LCR), achieving strong attack effectiveness within 30 epochs. In contrast, optimizing texture parameters (opacity Î±, color c) requires more iterations but ultimately reaches comparable or higher LCR. This suggests a time-performance trade-off: geometry enables fast convergence, while texture offers stronger long-term perturbation due to its direct effect on visual semantics.

A detailed dimension-wise analysis in Figure 5b further reveals that position x and scale s are the most sensitive parameters, contributing significantly to early-stage attack progress. Opacity Î± and color c exhibit moderate sensitivity, as they influence visibility and texture realism, respectively. Rotation q, despite affecting global geometry, shows minimal impact, due to its non-local and redundant nature in Gaussian-based representations. These findings validate our design of selective dimension optimization (Section 3.6), which allow practitioners to balance computational efficiency and physical feasibility in real-world deployments. Additional explanations and visual examples are provided in Appendix B.2.

Table 4. LCR (â) for adversarial objects optimized on one detector and tested across others. 3DGAA achieves strong cross-detector generalization.
<table><tr><td>Method</td><td>F-RCNN</td><td>M-RCNN</td><td>SSD</td></tr><tr><td>F-RCNN</td><td>2.819</td><td>2.111</td><td>2.193</td></tr><tr><td>M-RCNN</td><td>2.841</td><td>2.417</td><td>2.232</td></tr><tr><td>SSD</td><td>1.054</td><td>0.765</td><td>2.349</td></tr><tr><td>Method</td><td>YOLOv3</td><td>YOLOv5</td><td>YOLOv8</td></tr><tr><td>F-RCNN</td><td>6.098</td><td>4.684</td><td>4.429</td></tr><tr><td>M-RCNN</td><td>6.180</td><td>4.852</td><td>4.418</td></tr><tr><td>SSD</td><td>5.199</td><td>3.619</td><td>3.210</td></tr></table>

Table 5. LCR (â) of segmentation models under 3DGAA attack. 3DGAA achieves strong performance on segmentation models.
<table><tr><td>Method</td><td>DLv3-R50</td><td>DLv3-R101</td><td>FCN-R50</td><td>FCN-R101</td></tr><tr><td>LCR</td><td>0.628</td><td>0.874</td><td>0.795</td><td>0.893</td></tr></table>

DLv3: DeepLabV3; R50/101: ResNet-50/101 backbones.

## 4.6. Transferability Across Models and Tasks

We evaluate the transferability of 3DGAA adversarial objects across different perception models, including detectors with diverse architectures and segmentation models.

Cross-Detector Generalization. Table 4 reports the LCR performance of adversarial objects optimized on one detector and tested on others. Perturbations generated from two-stage detectors (Faster R-CNN and Mask R-CNN) demonstrate strong transferability, reaching LCR = 6.180 on YOLOv3 and maintaining performance on SSD (e.g., F-RCNN â SSD yields LCR = 2.193). These results suggest that 3DGAA captures transferable vulnerabilities across architectures through multi-dimensional optimization, rather than overfitting to a specific detector.

Segmentation Attack. We further test 3DGAAâs generalization to segmentation models by evaluating LCR on DeepLabV3 and FCN with ResNet backbones. As shown in Table 5, the attacks transfer reasonably well to segmentation settings, despite being optimized for detection. Interestingly, deeper backbones (ResNet-101) appear more vulnerable than ResNet-50 variants, suggesting that network depth may increase susceptibility to 3D physical perturbations.

## 4.7. Physical Realization

To verify real-world effectiveness under the nondeformability constraint, we deploy adversarial textures on the same miniature vehicle shell so that geometry remains unchanged across all variants. Textures are obtained from our pipeline 3DGAAa (Sec. 3.6) and transferred to printable wraps. Practically, this constitutes a geometrypreserving, print-and-apply workflow: textures from 3DGAAa are UV-unwrapped and printed on matte wraps applied to the same rigid shell (no body modification); the procedure scales to full-size vehicles by proportionally adjusting wrap size/ppi and view-distance while keeping the capture protocol fixed.

Table 6. Physical-world AP@0.5 (â) comparison of attack methods. 3DGAA achieves superior physical adversarial effectiveness.
<table><tr><td>Method</td><td>YOLOv5</td><td>SSD</td><td>F-RCNN</td><td>M-RCNN</td></tr><tr><td>Vanilla</td><td>93.75</td><td>84.17</td><td>96.25</td><td>94.58</td></tr><tr><td>Random</td><td>80.00</td><td>76.25</td><td>87.50</td><td>82.50</td></tr><tr><td>CAMOU [46]</td><td>69.58</td><td>64.58</td><td>72.08</td><td>74.17</td></tr><tr><td>DAS [41]</td><td>75.83</td><td>74.17</td><td>79.58</td><td>77.50</td></tr><tr><td>FCA [40]</td><td>60.83</td><td>48.33</td><td>62.92</td><td>67.92</td></tr><tr><td>3DGAAa</td><td>32.08</td><td>17.92</td><td>35.83</td><td>34.58</td></tr></table>

We evaluate four lighting conditions: indoor daylight (ID), indoor night (IN), outdoor daylight (OD), and outdoor night (ON), and compare three texture variants (Fig. 6): (1) vanilla (non-adversarial), (2) 3DGAA w/o Aug. (no physical augmentation), and (3) 3DGAA w/ Aug. The vanilla object is reliably detected in all scenes. Without augmentation, adversarial textures mildly reduce confidence yet show sensitivity to illumination. In contrast, 3DGAA w/ Aug. consistently induces missed detections or misclassifications across ID/IN/OD/ON, indicating that our augmentationâmodeling exposure/lighting/partial occlusion during optimizationâeffectively bridges the sim-toreal gap.

Table 6 quantifies detector performance (AP@0.5) on four architectures. With identical geometry and capture protocol, 3DGAA reduces AP from 84.17-96.25% down to 17.92-35.83%, surpassing all physical baselines. Together with Sec. 3.6, these results establish a geometry-preserving, appearance-only fabrication path for vehicles while retaining strong attack efficacy under diverse real-world lighting.

## 5. Conclusion

We present 3DGAA, a novel adversarial object generation framework that jointly optimizes geometry and appearance in the 3D Gaussian space. Unlike prior methods, 3DGAA enables expressive, multi-view-consistent adversarial robustness. Furthermore, we propose a Physical Filtering Module for physical realism and a Physical Augmentation Module for simulating environmental to develop real-world deployment. Extensive experiments show that 3DGAA significantly degrades detection performance (mAP â to 7.38%) while maintaining strong physical realism across detectors and environments. It also generalizes across viewpoints, model architectures, and segmentation tasks, validating 3DGAA as a robust and transferable pipeline for evaluating safety-critical perception systems, especially in autonomous driving scenarios.

A discussion of security concerns and potential societal risks and mitigation strategies is included in App. C.5.

## References

[1] Baidu Apollo, 2020. http://apollo.auto/. 1

[2] Autoware.ai, 2020. https://www.autoware.ai/. 1

[3] Tom B. Brown, Dandelion Mane, Aurko Roy, Mart Â´ Â´Ä±n Abadi, and Justin Gilmer. Adversarial patch, 2018. 1, 2

[4] Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017 IEEE Symposium on Security and Privacy (SP), 2017. 1

[5] Guikun Chen and Wenguan Wang. A survey on 3d gaussian splatting, 2025. 2

[6] Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam. Rethinking atrous convolution for semantic image segmentation, 2017. 5

[7] Shang-Tse Chen, Cory Cornelius, Jason Martin, and Duen Horng (Polo) Chau. Shapeshifter: Robust physical adversarial attack on faster r-cnn object detector. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2018, Dublin, Ireland, September 10â14, 2018, Proceedings, Part I, page 52â68, Berlin, Heidelberg, 2018. Springer-Verlag. 1, 2

[8] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. CARLA: An open urban driving simulator. In Proceedings of the 1st Annual Conference on Robot Learning (CoRL), pages 1â16, 2017. 5

[9] Ben Fei, Jingyi Xu, Rui Zhang, Qingyuan Zhou, Weidong Yang, and Ying He. 3d gaussian splatting as new era: A survey. IEEE Transactions on Visualization and Computer Graphics, pages 1â20, 2024. 2

[10] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3354â3361, 2012. 1

[11] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples, 2015. 1

[12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015. 5

[13] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Gir- Â´ shick. Mask r-cnn. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 2980â2988, 2017. 5

[14] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d, 2024. 1, 2

[15] Lifeng Huang, Chengying Gao, Yuyin Zhou, Cihang Xie, Alan L. Yuille, Changqing Zou, and Ning Liu. Universal physical camouflage attacks on object detectors. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 1, 2, 5, 7

[16] Yao Huang, Yinpeng Dong, Shouwei Ruan, Xiao Yang, Hang Su, and Xingxing Wei. Towards transferable targeted 3d adversarial attack in the physical world. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24512â24522, 2024. 1, 2, 5, 7, 6

[17] Matthew Hull, Chao Zhang, Zsolt Kira, and Duen Horng Chau. Adversarial attacks using differentiable rendering: A survey, 2024. 2

[18] Hiroharu Kato, Yoshitaka Ushiku, and Tatsuya Harada. Neural 3d mesh renderer. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 2

[19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 1, 2

[20] Mark Lee and Zico Kolter. On physical adversarial patches for object detection, 2019. 2

[21] Longwei Li, Huajian Huang, Sai-Kit Yeung, and Hui Cheng. Omnigs: Fast radiance field reconstruction using omnidirectional gaussian splatting. In Proceedings of the Winter Conference on Applications of Computer Vision (WACV), pages 2260â2268, 2025. 2

[22] Yanjie Li, Bin Xie, Songtao Guo, Yuanyuan Yang, and Bin Xiao. A survey of robustness and safety of 2d and 3d deep learning models against adversarial attacks. ACM Comput. Surv., 56(6), 2024. 2

[23] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C. Berg. SSD: Single shot multibox detector. In 2016 European Conference on Computer Vision (ECCV), 2016. 5

[24] Xin Liu, Huanrui Yang, Linghao Song, Hai Li, and Yiran Chen. Dpatch: Attacking object detectors with adversarial patches. CoRR, abs/1806.02299, 2, 2018. 2

[25] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 5

[26] Tianrui Lou, Xiaojun Jia, Siyuan Liang, Jiawei Liang, Ming Zhang, Yanjun Xiao, and Xiaochun Cao. 3d gaussian splatting driven multi-view robust physical adversarial camouflage generation, 2025. 1, 2

[27] Qiming Lu, Shikui Wei, Haoyu Chu, and Yao Zhao. Towards transferable 3d adversarial attack. In Proceedings of the 3rd ACM International Conference on Multimedia in Asia (MMAsia), New York, NY, USA, 2022. Association for Computing Machinery. 2

[28] Kien Nguyen, Tharindu Fernando, Clinton Fookes, and Sridha Sridharan. Physical adversarial attacks for surveillance: A survey. IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 35(12):17036â17056, 2024. 1

[29] Zhenbang Peng, Jianqi Chen, Zhenwei Shi, and Zhengxia Zou. Physical Adversarial Camouflage Generation in Optical Remote Sensing Images. IEEE Transactions on Information Forensics and Security (TIFS), 20:6308â6323, 2025. 1, 2, 5, 7

[30] Joseph Redmon and Ali Farhadi. Yolov3: An incremental improvement, 2018. 5

[31] Huali Ren and Teng Huang. Adversarial example attacks in the physical world. In Machine Learning for Cyber Security, pages 572â582, Cham, 2020. Springer International Publishing. 1

[32] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 39(6):1137â1149, 2017. 5

[33] Naufal Suryanto, Yongsu Kim, Hyoeun Kang, Harashta Tatimma Larasati, Youngyeo Yun, Thi-Thu-Huong Le, Hunmin Yang, Se-Yoon Oh, and Howon Kim. Dta: Physical camouflage attacks using differentiable transformation network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15305â15314, 2022. 1, 2, 5, 7

[34] Naufal Suryanto, Yongsu Kim, Harashta Tatimma Larasati, Hyoeun Kang, Thi-Thu-Huong Le, Yoonyoung Hong, Hunmin Yang, Se-Yoon Oh, and Howon Kim. Active: Towards highly transferable 3d physical camouflage for universal and robust vehicle evasion. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4282â4291, 2023. 1, 2, 5, 7

[35] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation, 2024. 1, 2, 5

[36] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. In The Twelfth International Conference on Learning Representations (ICLR), 2024. 1

[37] Ziyu Tang, Weicai Ye, Yifan Wang, Di Huang, Hujun Bao, Tong He, and Guofeng Zhang. Nd-sdf: Learning normal deflection fields for high-fidelity indoor reconstruction. In 2025 International Conference on Learning Representations (ICLR), pages 3460â3489, 2025. 2

[38] Tesla, 2025. https://www.tesla.com/fsd/. 1

[39] Simen Thys, Wiebe Van Ranst, and Toon Goedeme. Fooling automated surveillance cameras: adversarial patches to attack person detection. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019. 1, 2

[40] Donghua Wang, Tingsong Jiang, Jialiang Sun, Weien Zhou, Zhiqiang Gong, Xiaoya Zhang, Wen Yao, and Xiaoqian Chen. Fca: Learning a 3d full-coverage vehicle camouflage for multi-view physical adversarial attack. Proceedings of the AAAI Conference on Artificial Intelligence, page 2414â2422, 2022. 1, 2, 5, 7, 8, 6

[41] Jiakai Wang, Aishan Liu, Zixin Yin, Shunchang Liu, Shiyu Tang, and Xianglong Liu. Dual attention suppression attack: Generate adversarial camouflage in physical world. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 1, 2, 5, 7, 8

[42] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4): 600â612, 2004. 5

[43] Tong Wu, Xuefei Ning, Wenshuo Li, Ranran Huang, Huazhong Yang, and Yu Wang. Physical adversarial attack on vehicle detector in the carla simulator, 2020. 1, 2

[44] Jason Y. Zhang, Amy Lin, Moneish Kumar, Tzu-Hsuan Yang, Deva Ramanan, and Shubham Tulsiani. Cameras as rays: Pose estimation via ray diffusion. In The Twelfth International Conference on Learning Representations (ICLR), 2024. 2

[45] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep

features as a perceptual metric. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 5

[46] Yang Zhang, Hassan Foroosh, Philip David, and Boqing Gong. Camou: Learning physical vehicle camouflages to adversarially attack detectors in the wild. International Conference on Learning Representations (ICLR), 2018. 1, 2, 5, 7, 8

[47] Jiawei Zhou, Linye Lyu, Daojing He, and YU LI. RAUCA: A Novel Physical Adversarial Attack on Vehicle Detectors via Robust and Accurate Camouflage Generation. In Fortyfirst International Conference on Machine Learning (ICML), 2024. 1, 2, 5, 7

[48] Heran Zhu and Dazhong Rong. Multiview consistent physical adversarial camouflage generation through semantic guidance. In 2024 International Joint Conference on Neural Networks (IJCNN), pages 1â8, 2024. 1

# 3DGAA: Realistic and Robust 3D Gaussian-based Adversarial Attack for Autonomous Driving Supplementary Material

More details on the method, experimental, and a discussion of some extended questions are included in this supplementary material.

## A. Supplementary of Methods

Note that the selected intermediate physical derivations have been condensed for conciseness.

## A.1. Dynamic Loss Weighting Strategy

To ensure stable optimization between the adversarial objective ${ \mathcal { L } } _ { \mathrm { a d v } }$ and the shape consistency loss $\mathcal { L } _ { \mathrm { s h a p e } }$ , we employ a dynamic weighting mechanism that adaptively balances the two terms based on their magnitudes.

Let $\bar { \mathcal { L } } _ { \mathrm { { a d v } } }$ and $\bar { \mathcal { L } } _ { \mathrm { s h a p e } }$ denote the average adversarial loss and shape loss at a given iteration. We first scale the adversarial loss for numerical stability:

$$
\hat { \mathcal { L } } _ { \mathrm { a d v } } = \bar { \mathcal { L } } _ { \mathrm { a d v } } / \gamma ,\tag{A.1}
$$

where $\gamma$ is a scale factor (set to 10.0 in our experiments).

We then compute unnormalized weights:

$$
w _ { \mathrm { a d v } } = \frac { 1 } { \hat { \mathcal { L } } _ { \mathrm { a d v } } + \varepsilon } , \quad w _ { \mathrm { s h a p e } } = \bar { \mathcal { L } } _ { \mathrm { s h a p e } } ,\tag{A.2}
$$

where Îµ is a small constant to avoid division by zero.

The raw weights are normalized:

$$
\lambda _ { \mathrm { a d v } } = \frac { w _ { \mathrm { a d v } } } { w _ { \mathrm { a d v } } + w _ { \mathrm { s h a p e } } + \varepsilon } , \quad \lambda _ { \mathrm { s h a p e } } = 1 - \lambda _ { \mathrm { a d v } } .\tag{A.3}
$$

To prevent collapse into one objective, we clamp $\lambda _ { \mathrm { s h a p e } }$ to a minimum value (e.g., 0.4), ensuring sufficient geometric consistency:

$$
\lambda _ { \mathrm { s h a p e } } = \mathrm { m a x } ( \lambda _ { \mathrm { s h a p e } } , \lambda _ { \mathrm { m i n } } ) , \quad \lambda _ { \mathrm { a d v } } = 1 - \lambda _ { \mathrm { s h a p e } } .\tag{A.4}
$$

The final loss is a weighted combination:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = - \lambda _ { \mathrm { a d v } } \cdot \bar { \mathcal { L } } _ { \mathrm { a d v } } + \lambda _ { \mathrm { s h a p e } } \cdot \bar { \mathcal { L } } _ { \mathrm { s h a p e } } .\tag{A.5}
$$

This strategy adaptively reduces adversarial emphasis when the attack saturates $\left( \mathrm { i . e . , \ } \mathcal { L } _ { \mathrm { a d v } } \downarrow \right)$ , and shifts focus to geometry preservation. Empirically, we find this dynamic scheme stabilizes training and improves both realism and attack effectiveness.

## A.2. Physical Filtering Module

We introduce a two-stage Physical Filtering Module composed of: (1) Topological Pruning (TP), and (2) Structural Denoising (SD), as described in Section 3.4.

Topological Pruning: To eliminate geometrically inconsistent Gaussians while preserving structural continuity, we develop an adaptive density-aware pruning strategy. The local density $\rho ( \mathbf { g } _ { j } )$ for each Gaussian $\mathbf { g } _ { j }$ is estimated through its k-nearest neighbors in 3D space. We dynamically determine the removal threshold $\tau _ { d }$ as the p-th percentile of density distribution:

$$
\tau _ { d } = \mathrm { Q u a n t i l e } ( \{ \rho ( \mathbf { g } _ { j } ) \} _ { j = 1 } ^ { N } , p )\tag{A.6}
$$

Gaussians satisfying $\rho ( g _ { j } ) < \tau _ { d }$ are considered outliers and removed. This percentile-based thresholding automatically adapts to varying object densities without manual parameter tuning.

Structural Denoising: We propose a camera-aware anisotropic filter that jointly optimizes 3DGS scales $s _ { j }$ and opacities $\alpha _ { j }$ . The filtering intensity $\sigma _ { j }$ for each Gaussian is modulated by its minimum projected distance to camera planes:

$$
\sigma _ { j } \propto \operatorname* { m i n } _ { i } \frac { | | R _ { i } x _ { j } + T _ { i } | | _ { 2 } } { f _ { i } }\tag{A.7}
$$

where $R _ { i } , T _ { i } , f _ { i }$ denote the rotation matrix, translation vector, and focal length of the i-th camera. This spatial adaptation ensures stronger smoothing for distant regions while preserving detail in close-range areas. The filtered parameters are obtained through:

$$
s _ { j } ^ { \prime } = \sqrt { s _ { j } ^ { 2 } + \sigma _ { j } ^ { 2 } I } , \quad \alpha _ { j } ^ { \prime } = \alpha _ { j } \cdot \frac { \operatorname* { d e t } ( s _ { j } ) } { \operatorname* { d e t } ( s _ { j } ^ { \prime } ) }\tag{A.8}
$$

This dual-phase physical filtering module achieves critical improvements: Noise suppression eliminates floating artifacts, and physical consistency ensures multi-view coherence through camera-aware smoothing.

The final filtered 3DGS representation $\mathcal { G } _ { b a s e }$ maintains geometric fidelity while achieving photorealistic rendering quality, providing a robust foundation for subsequent adversarial optimization.

## A.3. Physical Augmentation Module

This section details the exact formulation of the differentiable augmentation components defined in Section 3.5. The

physical augmentation module applies a sequence of four transformations:

$$
\mathcal { T } = \mathcal { T } _ { \mathrm { o c c l } } \circ \mathcal { T } _ { \mathrm { s h a d o w } } \circ \mathcal { T } _ { \mathrm { p h o t o } } \circ \mathcal { T } _ { \mathrm { n o i s e } } ,\tag{A.9}
$$

where each sub-transformation models a common source of physical variation. The operations are applied in right-toleft order during optimization.

Imaging Degradation simulates camera sensor noise, which increases with object distance due to atmospheric interference. This degradation is modeled as an additive Gaussian noise function:

$$
{ \mathcal { T } } _ { n o i s e } = I + { \mathcal { N } } ( 0 , \sigma ^ { 2 } ( d ) ) ,\tag{A.10}
$$

where I is the original rendered image, ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ represents zero-mean Gaussian noise, and $\sigma ( d )$ is the depth-dependent standard deviation:

$$
\sigma ( d ) = \sigma _ { 0 } + \gamma \cdot d .\tag{A.11}
$$

Here, $\sigma _ { 0 }$ is the base noise level, d represents the pixel-wise depth value, and $\gamma$ is a scaling factor controlling noise amplification with distance.

Photometric Variation simulates color distortions caused by varying lighting conditions and sensor imperfections. We model this effect using channel-wise affine transformations:

$$
\mathcal { T } _ { p h o t o } = I _ { c } \cdot \alpha _ { c } + \beta _ { c } ,\tag{A.12}
$$

where $I _ { c }$ is the original intensity of color channel c, $\alpha _ { c } ~ \sim ~ U ( 0 . 9 , 1 . 1 )$ represents contrast variation sampled from a uniform distribution, and $\beta _ { c } \sim U ( - 0 . 0 5 , 0 . 0 5 )$ is an additive shift factor.

These transformations ensure that adversarial perturbations remain effective under different lighting conditions.

Shadow Projection used sampling random light source positions and computing penumbra-umbra transitions via sigmoid intensity mapping, we replicated the soft shadow boundaries observed in natural illumination environments.

$$
\mathcal { T } _ { s h a d o w } = \frac { 1 } { 1 + e ^ { - \alpha ( d ( x , y ) - d _ { \mathrm { t h } } ) } } ,\tag{A.13}
$$

where Î± controls shadow smoothness, $d ( x , y )$ is the depth at pixel $( x , y )$ , and $d _ { \mathrm { t h } }$ is the threshold depth for shadow casting.

Adaptive Occlusion enhances robustness to partial obstructions through random rectangular masks that simulate large-scale object occlusions covering part of the image area.

$$
\mathcal { T } _ { o c c l } = \left\{ \begin{array} { l l } { \{ 0 , r a n d o m ( 0 , 1 ) \} , } & { ( x , y ) \in \mathrm { o c c l u s i o n r e g i o n } , } \\ { 1 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{A.14}
$$

The composite transformation $\mathcal { T } ~ = ~ \mathcal { T } _ { o c c l } \circ \mathcal { T } _ { s h a d o w } ~ \circ$ â¦ $\mathcal { T } _ { p h o t o } \circ \mathcal { T } _ { n o i s e }$ establishes a physical augmentation module

during optimization. This phased approach enables the adversarial samples to develop physical robustness, as quantified in Section 4.

## A.4. Selective Dimension Optimization

We define three fundamental optimization modes based on parameter space decomposition in the 3D Gaussian Splatting representation:

Geometry-only Mode $( { \cal { K } } _ { g } ) { :  }$ : This mode activates all 10 geometry-related dimensions, including position $( x , y , z )$ scale $( s _ { x } , s _ { y } , s _ { z } )$ , and quaternion rotation $( q _ { w } , q _ { x } , q _ { y } , q _ { z } ) \mathrm { . }$

$$
\mathcal { K } _ { g } = \{ x , y , z , s _ { x } , s _ { y } , s _ { z } , q _ { w } , q _ { x } , q _ { y } , q _ { z } \}\tag{A.15}
$$

This mode enables structural manipulation of the object without altering appearance, suitable for settings like geometric camouflage or shape-aware perturbations.

Appearance-only Mode $( K _ { a } )$ : This mode only optimizes the 4 appearance-related parametersâcolor $( c _ { r } , c _ { g }$ $c _ { b } )$ and opacity (Î±):

$$
\mathcal { K } _ { a } = \{ c _ { r } , c _ { g } , c _ { b } , \alpha \}\tag{A.16}
$$

It is ideal for use cases such as 3D-printed objects or painted surfaces, where modifying geometry is impractical or costly.

Full-Dimensional Mode $( { \cal { K } } _ { f u l l } )  : $ For completeness, we denote the full parameter set as:

$$
{ \ K } _ { f u l l } = { \ K } _ { g } \cup { \ K } _ { a }\tag{A.17}
$$

which enables unconstrained optimization across the entire 14D space.

Optimization Behavior. As demonstrated in Section 3.6, different modes exhibit different convergence behavior. Geometry-mode reaches high LCR within 30 epochs, while texture-mode requires longer training (up to 200 epochs) but eventually achieves comparable performance. This difference stems from the sensitivity of geometry dimensions $( \mathrm { e } . \mathrm { g } . , x , s )$ to detector outputs, as discussed in our LCR sensitivity analysis (Fig. 5).

Practical Deployment Implications. In scenarios where 3D shape alteration is restricted (e.g., rigid vehicle body), $\kappa _ { a }$ can be adopted to generate adversarial decals or coatings. For applications involving laser-cut, foam, or parametric shell objects, $\kappa _ { g }$ can produce printable adversarial shapes while maintaining uniform color.

Further visual examples under different dimension selection strategies are presented in Appendix B.2.

## B. Supplementary of Experiments

## B.1. Implementation Details

The 3DGS generation backbone adopts the pretrained LGM architecture [35] with frozen parameters, processing four calibrated views (512 Ã 512 resolution) as input. The physical filtering module applies density-based pruning at $\tau _ { d } =$ 0.105 percentile threshold, followed by structural Gaussian smoothing, as illustrated by Eq. A.8.

<!-- image-->  
All

<!-- image-->

<!-- image-->  
Geometry  
Texture

Figure B.1. Selective Dimensions. Mode-selected dimension optimization visualization.  
<!-- image-->  
All

<!-- image-->  
Position

<!-- image-->  
Scale

<!-- image-->  
Rotation

<!-- image-->  
Opacity

<!-- image-->  
Color  
Figure B.2. Selective Dimensions. Single dimension optimization visualization.

For adversarial optimization, we initialize the learning rate Î· = 0.03 with gradient descent over 50 epochs, optimizing all 14 parameters of the 3D Gaussians (positions $x \in \mathbb { R } ^ { 3 }$ , rotations $q \in \mathbb { R } ^ { 4 }$ , scales $s \in \mathbb { R } ^ { 3 }$ , colors $c \in \mathbb { R } ^ { 3 }$ , opacity $\alpha ~ \in ~ \mathbb { R } ^ { 1 } )$ under the combined loss from Eq. 5. The physical augmentation module probabilistically applies transformations with $\gamma ~ = ~ 0 . 0 0 5$ for scaling factor and $p _ { s i z e } = 0 . 1$ for occlusion size.

All experiments are conducted on an NVIDIA RTX 4090 GPU with PyTorch 2.1 + CUDA 11.8, where each adversarial optimization completes within 1 minute.

## B.2. Selective Dimension Optimization

Our experiments investigate the sensitivity of different dimensions in 3DGS under adversarial optimization. The part-based optimization comparison (Fig. 5a) reveals that modifying geometric dimensions achieves a high LCR within a few training iterations. This indicates that 3DGS is particularly sensitive to geometric perturbations, making it highly susceptible to adversarial attacks in this domain. In contrast, texture-based optimization requires more training steps to reach an optimal adversarial effect but ultimately surpasses geometric perturbations in overall LCR. This suggests that while geometry is more straightforward to manipulate for quick adversarial impacts, texture perturbations offer more effective long-term deception, as shown in Fig. B.1.

Further analysis in all-dimension optimization (Fig. 5b) demonstrates that different dimensions exhibit varying degrees of adversarial sensitivity. The position x and scale s dimensions show high sensitivity, meaning even minor perturbations in these attributes significantly impact the rendered adversarial object. Conversely, opacity Î± and RGB c color dimensions exhibit lower sensitivity, requiring more extensive modifications to influence the detection model. Notably, the quaternion rotation q dimension is the least sensitive, requiring the most training iterations yet achieving the lowest LCR in 400 epochs. The visualized results are shown in Fig. B.2.This implies that 3DGS representations are more resistant to adversarial attacks targeting 3DGS rotational attributes, likely due to their global influence on object orientation rather than direct appearance modifications.

These findings highlight the importance of selecting optimal adversarial dimensions for attack optimization. While geometric perturbations offer immediate adversarial benefits, optimizing texture features leads to superior attack efficacy. Moreover, identifying sensitive dimensions x, s can help refine attack strategies, ensuring effective adversarial perturbations while minimizing unnecessary computational overhead.

<!-- image-->  
Figure B.3. Adversarial Effectiveness of 3DGAA Across Different Vehicle Types.

## B.3. Object Generalization

A critical aspect of physical attacks is their ability to generalize across objects. We evaluate 3DGAA on multiple vehicle types (sedan, SUV, truck, bus). As shown in Fig. B.3, our framework consistently reduces detector confidence across these categories, indicating robustness under diverse 3D geometries.

Cross-vehicle adaptation routes. Because 3DGS couples appearance to a scene-specific Gaussian set (and implicitly to geometry/UV layout), cross-vehicle use is achieved via two lightweight routes: (i) Initialization transfer: use an already optimized source model as initialization on the target model and run a few optimization steps to align views (Sec. 3.6); (ii) UV reparameterization: map the source texture onto the target via UV alignment (piecewise or learned correspondence) and optionally refine with brief optimization. Both routes are compatible with the appearance-only setting 3DGAAa (no body modification) and benefit from our minutes-level runtime (App. C.3), making per-model adaptation practical.

These findings suggest that 3DGAAâs adversarial perturbations adapt effectively across rigid vehicles by choosing appearance-only deployment; for cones/barriers, either appearance-only or simple geometry shells are feasible. Overall, 3DGAA trades a universal single-texture assumption for a controllable, geometry-aware procedure that remains broadly applicable while preserving physical deployability.

## B.4. Multi-Angle Visualization of 3DGAA Results

Figures B.4 and B.5 provide extended multi-perspective visualizations of our methodâs capability to maintain structural authenticity while achieving adversarial effectiveness. Figure B.4 demonstrates the original 3DGS reconstructions across 12 representative viewpoints, highlighting the baseline geometric accuracy and texture fidelity. Correspondingly, Figure B.5 showcases the adversarial counterparts under identical viewing conditions, where optimized texture patterns consistently mislead detectors without introducing noticeable visual disparity.

Table B.1. Realism scores of different adversarial and vanilla textures from participants.
<table><tr><td>Methods</td><td>Scores</td><td>Methods</td><td>Scores</td></tr><tr><td>Vanilla</td><td>9.6</td><td>Random</td><td>1.1</td></tr><tr><td>CAMOU</td><td>2.9</td><td>UPC</td><td>6.4</td></tr><tr><td>DAS</td><td>3.9</td><td>FCA</td><td>2.3</td></tr><tr><td>DTA</td><td>3.2</td><td>ACTIVE</td><td>6.0</td></tr><tr><td>RAUCA</td><td>3.7</td><td>PACG</td><td>3.6</td></tr><tr><td>3DGAAa</td><td>7.1</td><td>3DGAA</td><td>7.9</td></tr></table>

3DGAAa: Appearance-only mode.

The side-by-side comparison reveals two key observations: (1) The adversarial textures preserve high-frequency surface details comparable to original patterns, ensuring physical plausibility, (2) Adversarial effectiveness remains stable across extreme viewing angles, confirming the 3DGAA optimizationâs robustness to perspective variations. These visual results complement our quantitative analyses by demonstrating the spatial consistency of geometric reconstruction and adversarial pattern generation.

## B.5. Human Perception

To further evaluate the realism of adversarial textures generated by 3DGAA, we conducted a human perception study with 50 volunteers. Each participant was shown a series of adversarially perturbed vehicle images and was asked to rate the perceived realism on a 1 to 10 scale, where 10 indicates that the texture appears completely natural, resembling standard vehicle paint or patterns. 1 indicates that the texture is highly unnatural and adversarially modified. Each volunteer was presented with a randomized set of images, including both adversarially perturbed and vanilla textures. This study aimed to assess whether adversarial textures can remain imperceptible to human observers, as a key component of maintaining physical realism in real-world adversarial attacks.

<!-- image-->  
Figure B.4. Multi-View Visualization of Original 3DGS Reconstructions: Twelve representative viewpoints demonstrating baseline geometric accuracy and texture fidelity.

<!-- image-->  
Figure B.5. Consistent Adversarial Effects Across Viewing Angles: Corresponding adversarial 3DGS visualizations under identical viewpoints, showing maintained structural integrity while evading detection.

We analyze the collected ratings by computing the average realism score per condition and compare different adversarial and vanilla textures. The results presented in Table B.1 demonstrate the superior realism of our proposed 3DGAA method compared to existing adversarial attack approaches. Notably, 3DGAA achieves a significantly higher realism score of 7.9, outperforming prior methods such as ACTIVE (6.0), UPC (6.4), and RAUCA (3.7), indicating that adversarial perturbations generated by 3DGAA are less perceptible to human observers. The appearance-selected variant, 3DGAAa (7.1), further validates the effectiveness of our approach, demonstrating that the texture modifications in 3DGAA contribute to improved realism while maintaining adversarial effectiveness. Compared to real vehicle textures (Vanilla: 9.6), our method achieves the closest resemblance, confirming that our Physical Filtering Module and Physical Augmentation Module successfully reduce unrealistic artifacts. Furthermore, conventional adversarial texture-based methods, such as CAMOU (2.9), DAS (3.9), and FCA (2.3), exhibit significantly lower realism scores, primarily due to their reliance on large-scale texture distortions that lack physical consistency. The extreme case of Random (1.1) highlights that naive perturbations are easily identifiable as artificial, further reinforcing the necessity of structured optimization in adversarial texture generation. These results demonstrate that 3DGAA achieves a superior balance between adversarial robustness and realism, making it more applicable to real-world scenarios where physical reality is crucial.

## C. Discussion

## C.1. Adversarial Effectiveness is Derived from 3DGAA

The adversarial effectiveness of 3DGAA originates from its explicit adversarial optimization, which differentiates it from conventional 3D Gaussian Splatting (3DGS) representations. While 3DGS naturally introduces slight distortions in shape and texture due to the limitations of generation and sampling, these variations do not inherently contribute to adversarial behavior. The core objective of 3DGS is to reconstruct high-fidelity 3D objects from multi-view images, ensuring that the generated representation maintains visual consistency rather than misleading object detection models.

In contrast, 3DGAA explicitly optimizes the adversarial properties of the object by introducing targeted perturbations in texture and geometry. This process is guided by an adversarial loss function, which systematically reduces detection confidence while preserving the objectâs structural integrity. To achieve this balance, a shape preservation constraint is integrated into the optimization process, preventing excessive geometric distortions that could otherwise compromise the physical reality of the adversarial object. As a result, 3DGAA generates objects that exhibit strong adversarial effectiveness maintain realistic physical attributes, making them highly transferable across different detection models and environmental conditions.

## C.2. Generalization to Non-Vehicle Objects

The framework of 3DGAA is inherently designed to be object-agnostic, allowing its application beyond vehiclerelated adversarial scenarios. While our experimental evaluations primarily focus on vehicles, the underlying optimization process operates on fundamental 3D attributes, including position, scale, rotation, opacity, and color. These attributes are not limited to any specific category of objects, suggesting that the method can be generalized to a wide range of targets.

The physical filtering and augmentation modules incorporated in 3DGAA are formulated independently of object semantics. These modules refine the adversarial object while ensuring adaptability to varying physical and environmental conditions, reinforcing the methodâs applicability to non-vehicle objects such as pedestrians, animals, or urban infrastructure. Extending 3DGAA to diverse object categories opens promising directions for future research, particularly in adversarial robustness across different realworld scenarios.

## C.3. Optimization Efficiency

Protocol and measurements. We profile three modes (Full, 3DGAAa, 3DGAAg) under a shared setup (same data, views, early-stopping rule) on a single RTX 4090 with fixed driver/CUDA/PyTorch. Wall-clock time is reported from process start to end and decomposed into 3DGS Gen (initial Gaussian generation), Adv. Time (adversarial optimization), and Data/Misc (I/O, logging, warm-up). Adv. Iters denotes the final iteration count under the identical stopping criterion; Time/Iter is the mean per-iteration duration measured after device synchronization. Peak Mem is the maximum GPU memory in MB. We further report G (the number of Gaussians after TP+SD; determined by the backbone and inputs) and $\Delta G$ (the percentage of Gaussians updated during adversarial optimization). For context, we also include two non-3DGS baselines: FCA [40] (fixedmesh/texture pipeline) and TT3D [16] (NeRF-based). For these methods, 3DGS Gen is not applicable (fixed mesh or radiance-field training), so we report their adversarial optimization and I/O times under the same capture/view protocol. See Tab. C.2 for consolidated numbers.

3DGAAa attains the lowest Peak Mem (19,862 MB) and the fastest Time/Iter (0.26 s), but requires more iterations and thus a longer Adv. Time (74.82 s) and Total (116.32 s), which is verified in Fig. 5a. In contrast, jointly optimizing appearance and geometry (Full) reaches the target earlier (27 iters; 10.02 s adversarial time) despite a higher periteration cost (0.37 s) and memory (23,256 MB), yielding a shorter end-to-end time (51.52 s). 3DGAAg sits in between on memory (21,196 MB) and per-iteration time (0.31 s), converging in 24 iterations for a 48.92 s total. When contrasted with FCA and TT3D, the 3DGAA family operates at a markedly different efficiency scale: FCA and TT3D exhibit per-iteration costs in the order of tens of seconds (12.68 s / 21.80 s) and hours-level totals (2,629 s / 4,529 s), even without a 3DGS generation stage, whereas all 3DGAA modes finish within seconds-to-minutes under the same protocol. Overall, these findings substantiate the practicality of 3DGAA; detailed runtime numbers are provided here to illustrate the efficiency of 3DGAA, while the main paper focuses on realism and attack efficacy.

## C.4. Direct Optimization on Segmentation

Under the same 12Ã3 view protocol, we directly optimize a segmentation model (DeepLabv3-R50) by suppressing the car class over the object mask â¦. Concretely, we minimize $\begin{array} { r } { \mathcal { L } _ { \mathrm { s e g } } ~ = ~ \frac { 1 } { | \Omega | } \sum _ { x \in \Omega } p _ { c } ( x ) } \end{array}$ and use $\mathcal { L } ~ = ~ \lambda _ { \mathrm { s e g } } \mathcal { L } _ { \mathrm { s e g } } +$ $\lambda _ { \mathrm { s h a p e } } \mathcal { L } _ { \mathrm { s h a p e } } + \dot { \lambda } _ { \mathrm { p h y s } } \mathcal { L } _ { \mathrm { p h y s } }$ as the total objective.

Table C.3 shows that direct segmentation optimization achieves a larger LCR than DetâSeg transfer, while Vanilla remains at zero by definition. This confirms that, under the same view protocol, optimizing the segmentation loss $( \mathcal { L } _ { \mathrm { s e g } } )$ provides additional leverage beyond detector-driven transfer. In practice, the improvement is consistent across views and does not require changing the body geometry, indicating that 3DGAA can natively attack segmentation models with the same render-and-backprop pipeline.

Table C.2. Efficiency profiling under a shared protocol (same data, views, and early-stopping rule). Times are wall-clock from process start to end; Total includes 3DGS generation, adversarial training, and data I/O/misc.
<table><tr><td>Mode</td><td>GPU (Model)</td><td>Peak Mem (MB)</td><td>3DGS Gen (s)</td><td>Adv. Iters (epochs/iters)</td><td>Time/Iter (s)</td><td>Adv. Time (s)</td><td>Data/Misc (s)</td><td>Total (s)</td><td>G</td><td>âG (%)</td></tr><tr><td>3DGAA (Full)</td><td>RTX 4090</td><td>23256</td><td> $9 . 2 4 \pm 0 . 2 6$ </td><td>27</td><td>0.37</td><td> $1 0 . 0 2 \pm 0 . 2 2$ </td><td> $3 2 . 2 6 \pm 0 . 0 1$ </td><td>51.52</td><td>24378</td><td>98.22</td></tr><tr><td>3DGAAa</td><td>RTX 4090</td><td>19862</td><td> $9 . 2 4 \pm 0 . 2 6$ </td><td>283</td><td>0.26</td><td> $7 4 . 8 2 \pm 3 . 2 0$ </td><td> $3 2 . 2 6 \pm 0 . 0 1$ </td><td>116.32</td><td>24378</td><td>80.25</td></tr><tr><td>3DGAA9</td><td>RTX 4090</td><td>21196</td><td> $9 . 2 4 \pm 0 . 2 6$ </td><td>24</td><td>0.31</td><td> $7 . 4 2 \pm 0 . 1 5$ </td><td> $3 2 . 2 6 \pm 0 . 0 1$ </td><td>48.92</td><td>24378</td><td>75.05</td></tr><tr><td>FCA [40]</td><td>RTX 4090</td><td>23256</td><td></td><td>200</td><td>12.68</td><td> $2 5 3 6 \pm 9 6$ </td><td> $9 2 . 5 2 \pm 1 2 . 1 8$ </td><td>2629</td><td></td><td></td></tr><tr><td>TT3D [16]</td><td>RTX 4090</td><td>23256</td><td></td><td>200</td><td>21.80</td><td> $4 3 6 0 \pm 7 4$ </td><td> $1 6 6 . 2 9 \pm 4 8 . 4 0$ </td><td>4529</td><td></td><td>â</td></tr></table>

Table C.3. Segmentation model (DeepLabv3-R50) optimizing.
<table><tr><td>Method</td><td>Vanilla</td><td>Det â Seg</td><td>Seg</td></tr><tr><td>LCR â</td><td>0.000</td><td>0.628</td><td>1.144</td></tr></table>

## C.5. Ethical Considerations

Potential Societal Risks. Our work introduces 3DGAA, a 3D Gaussian-based framework for generating physically realistic adversarial objects. While this research aims to advance understanding of model vulnerabilities and robustness in safety-critical applications such as autonomous driving, it inherently poses potential risks if misused. These include: (1) intentional deployment of adversarial objects in real-world environments to cause sensor-level misperception, (2) disruption of autonomous navigation systems through physically disguised threats, and (3) security concerns arising from the transferability of attacks across models and viewpoints.

Mitigation and Safeguards. To mitigate misuse, we emphasize the following safeguards: (1) our attacks are conducted strictly in controlled simulation environments or on miniature-scale physical setups, without field deployment; (2) we do not release attack code or physical models publicly at this stage; and (3) our method is also intended to guide future development of more robust perception systems and defense techniques. Moreover, we incorporate physical realism constraints and shape fidelity to reduce the feasibility of applying large-scale, arbitrary perturbations.

Responsible Disclosure and Reproducibility. We advocate for responsible disclosure and reproducibility: the proposed attack can serve as a benchmark tool for stresstesting detection systems under worst-case conditions, and the 3D Gaussian representation supports explainable diagnostics. We believe that transparency around such vulnerabilities can foster long-term security through proactive system hardening, not exploitation.

Use of Human-derived Data. Our work includes a perceptual user study (Appendix B.5) in which human raters evaluate the realism of rendered 3D objects under different conditions. This evaluation was conducted anonymously and non-invasively, without collecting any personally identifiable information, demographic attributes, or biometric data. Participants were not exposed to offensive or sensitive content, and their involvement was limited to providing visual realism ratings on a 10-point scale. Given the benign and low-risk nature of this evaluation, formal IRB approval was not sought, in line with common practice for perceptionfocused visual studies in computer vision.