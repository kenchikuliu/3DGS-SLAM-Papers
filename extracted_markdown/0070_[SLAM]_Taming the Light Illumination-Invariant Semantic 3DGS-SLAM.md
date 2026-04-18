# TAMING THE LIGHT: ILLUMINATION-INVARIANT SEMANTIC 3DGS-SLAM

Shouhe Zhang1,2, Dayong Ren3,\*, Sensen Song1,2,\*, Yurong Qian1,2 and Zhenhong Jia1,4

1School of Computer Science and Technology, Xinjiang University

2Joint International Research Laboratory of Silk Road Multilingual Cognitive Computing Xinjiang University, Urumqi Xinjiang, 830046, China

3National Key Laboratory for Novel Software Technology, Nanjing University, Nanjing 210023, China.   
4Key Laboratory of Signal Detection and Processing, Xinjiang University, Urumqi, 830046, China

## ABSTRACT

Extreme exposure degrades both the 3D map reconstruction and semantic segmentation accuracy, which is particularly detrimental to tightly-coupled systems. To achieve illumination invariance, we propose a novel semantic SLAM framework with two designs. First, the Intrinsic Appearance Normalization (IAN) module proactively disentangles the sceneâs intrinsic properties, such as albedo, from transient lighting. By learning a standardized, illumination-invariant appearance model, it assigns a stable and consistent color representation to each Gaussian primitive. Second, the Dynamic Radiance Balancing Loss (DRB-Loss) reactively handles frames with extreme exposure. It activates only when an imageâs exposure is poor, operating directly on the radiance field to guide targeted optimization. This prevents error accumulation from extreme lighting without compromising performance under normal conditions. The synergy between IANâs proactive invariance and DRB-Lossâs reactive correction endows our system with unprecedented robustness. Evaluations on public datasets demonstrate state-of-the-art performance in camera tracking, map quality, and semantic and geometric accuracy.

Index Termsâ 3DGS, SLAM, semantic segmentation

## 1. INTRODUCTION

Recent advancements in neural radiance fields have revolutionized 3D scene representation, with 3D Gaussian Splatting (3DGS) emerging as a particularly potent technique[1]. Its ability to achieve highfidelity, real-time rendering from sparse image sets has unlocked new possibilities in robotics and augmented reality, most notably in the domain of Simultaneous Localization and Mapping (SLAM). By representing scenes as a collection of explicit 3D Gaussians, 3DGSbased SLAM systems[2, 3] can construct photorealistic maps while simultaneously tracking camera pose[4, 5, 6], paving the way for creating true-to-life digital twins of our environment[1, 7].

However, the transition from controlled datasets to the unpredictable conditions of the real world exposes a critical vulnerability in these systems: the lack of robustness to illumination variations. The foundational assumption of photometric consistencyâthat a scene point maintains a constant appearance across different viewsâis frequently violated in practice[8, 9, 10]. Drastic changes in lighting, such as moving from shadow to direct sunlight, or the automatic exposure adjustments of a moving camera, can cause identical scene elements to appear dramatically different.This variance corrupts the optimization process, leading to geometric artifacts, color inconsistencies in the map, and even catastrophic tracking failure[11, 12, 13]. This challenge is further magnified in the context of semantic SLAM[14, 15, 16, 17, 18, 19], where the goal is not only to map the world but also to understand it. In coupled systems where geometry and semantics are mutually beneficial, illumination variance wreaks havoc on both fronts. The same surface under different exposures can yield vastly different feature representations, confusing semantic segmentation[20, 21, 22, 23, 19, 24] networks and leading to erroneous labeling[13, 25, 26]. This, in turn, feeds incorrect semantic priors back into the geometric reconstruction, creating a vicious cycle of degradation that undermines the entire system[13, 27].

To address these fundamental limitations, we propose a novel illumination-invariant semantic SLAM framework designed to âtameâ the effects of real-world lighting. Our approach is built on a two-pronged strategy: a proactive module that standardizes scene appearance at its core, and a reactive mechanism that dynamically compensates for extreme exposure variations during optimization. Our main contributions are centered around two key designs:

First, we introduce the Intrinsic Appearance Normalization (IAN) module, which proactively disentangles the sceneâs intrinsic properties from transient lighting. Instead of learning continuous colors that are easily influenced by lighting, our module constrains the appearance of each Gaussian to a discretized, canonical color palette. This quantization acts as a powerful regularizer, forcing the model to learn a stable, underlying albedo for scene surfaces, thereby achieving a standardized and illumination-invariant appearance representation across the entire map. Second, to reactively handle severe, per-frame brightness shifts, we propose the Dynamic Radiance Balancing Loss (DRB-Loss). This component learns to model per-image exposure variations through a set of latent parameters. Crucially, our loss function is adaptive and structure-aware; it applies this photometric correction only when it detects a significant structural inconsistency between the rendered and the real image. This dynamic behavior ensures that the system robustly compensates for challenging exposure without using the correction mechanism to mask underlying geometric errors, leading to more stable and accurate optimization. By integrating these proactive and reactive mechanisms, our framework achieves an unprecedented level of robustness. Comprehensive evaluations on challenging public datasets demonstrate that our method delivers state-of-the-art performance in camera tracking, map reconstruction quality, and semantic and geometric accuracy, successfully navigating the complex and varied illumination conditions of real-world environments.

<!-- image-->  
Fig. 1. The framework processes input frames by first disentangling albedo and illumination with the IAN module. It then performs camera tracking, using the DRB-Loss to robustly handle exposure variations. Finally, the system jointly optimizes a dense, high-fidelity map containing both geometric and semantic information.

## 2. METHOD

In this section, we detail our illumination-invariant semantic 3DGS-SLAM framework. The core principle of our method is to disentangle the sceneâs appearance into two components: a stable intrinsic albedo and a varying transient illumination. Based on this decomposition, we introduce two key modules: the Intrinsic Appearance Normalization (IAN) module, which proactively learns a canonical color representation for the scene, and the Dynamic Radiance Balancing Loss (DRB-Loss), which reactively compensates for extreme illumination shifts. By integrating these modules, our system jointly optimizes geometry, appearance, and semantics during tracking and mapping, ultimately reconstructing a high-fidelity, illumination-robust 3D semantic map. The overall pipeline is illustrated in Fig. 1.

## 2.1. Differentiable Gaussian Representation and Rendering

To enable end-to-end optimization, we first define our scene representation based on 3D Gaussians. Each Gaussian primitive $G _ { i }$ is parameterized by a set of attributes:

$$
G _ { i } = \{ \mu _ { i } , r _ { i } , \sigma _ { i } , a _ { i } , l _ { i } , s _ { i } \} .\tag{1}
$$

where ${ \pmb \mu } _ { i } \in \mathbb { R } ^ { 3 }$ is the center, $r _ { i } \in \mathbb { S O } ( 3 )$ is the rotation, and $\sigma _ { i } \in \mathbb { R }$ is the opacity.

Critically, deviating from standard 3DGS, we explicitly model the final appearance color $\mathbf { c } _ { i }$ as the element-wise product of an intrinsic albedo $\mathbf { a } _ { i } \in [ 0 , 1 ] ^ { 3 }$ and a transient illumination factor $\mathbf { \Phi } _ { l _ { i } } \in$ $\mathbb { R } ^ { + }$ . This relationship, $\mathbf { c } _ { i } = \mathbf { a } _ { i } \odot l _ { i } ,$ is central to our framework. Additionally, si represents the semantic label of the Gaussian. These parameters are optimized via a differentiable rendering pipeline.

## 2.1.1. Color, Depth, and Semantic Rendering

Following the standard 3DGS pipeline [1], we project the 3D Gaussians onto the 2D image plane and synthesize the final pixel attributes using volumetric Î±-blending. For a given pixel $p ,$ the rendered attributes are computed as follows:

Color Rendering. The final color $C ( \boldsymbol p )$ is a blend of each Gaussianâs appearance $^ { c _ { i } , }$ which is the product of its intrinsic albedo and

illumination factor.

$$
C ( p ) = \sum _ { i = 1 } ^ { N } \pmb { c } _ { i } \cdot \pmb { f } _ { i } ( p ) \cdot \prod _ { j < i } ( 1 - f _ { j } ( p ) ) .\tag{2}
$$

where $f _ { i } ( \boldsymbol { p } )$ is the influence weight of Gaussian i on pixel $p ,$ determined by its projected 2D covariance and opacity $\sigma _ { i }$

Depth Rendering. The pixel depth $D ( p )$ is similarly computed by blending the depth $d _ { i }$ of each Gaussian center in the camera coordinate system, providing geometric supervision.

$$
D ( p ) = \sum _ { i = 1 } ^ { N } d _ { i } \cdot f _ { i } ( p ) \cdot \prod _ { j < i } ( 1 - f _ { j } ( p ) ) .\tag{3}
$$

Semantic Rendering. To facilitate joint semantic-geometric optimization, the semantic map $S ( p )$ is rendered by blending the semantic labels $\mathbf { \boldsymbol { s } } _ { i }$ of the Gaussians in a manner consistent with color rendering [15].

$$
S ( p ) = \sum _ { i = 1 } ^ { N } \pmb { s } _ { i } \cdot \pmb { f } _ { i } ( p ) \cdot \prod _ { j < i } ( 1 - f _ { j } ( p ) ) .\tag{4}
$$

## 2.2. Intrinsic Appearance Normalization (IAN)

Directly optimizing continuous RGB colors allows the model to bake transient lighting effects into the sceneâs color representation, leading to inconsistencies and artifacts in the reconstructed map. To address this, we introduce the IAN module, which employs color quantization as a strong regularizer to compel the network to learn a stable and canonical intrinsic albedo $\mathbf { \alpha } _  \mathbf { \alpha } \mathbf { \alpha } _ { \mathbf { \alpha } } \mathbf { \alpha } _  \mathbf { \alpha } \mathbf { \alpha } _ { \mathbf { \beta } \mathbf { \alpha } _ { \mathbf { \alpha } \mathbf { \beta } \mathbf { \alpha } _ { \mathbf { \beta } \mathbf { \alpha } _ { \lambda } \mathbf { \alpha } _ { \lambda } \mathbf { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } \mathrm { \alpha } _ { \lambda } } } }$ for scene surfaces, independent of lighting conditions. We discretize the continuous albedo space into a fixed, canonical palette. Specifically, during optimization, we enforce that each RGB channel of the intrinsic albedo ai (normalized to [0, 1]) maps to one of four discrete values. This is achieved by applying the following rule:

$$
a _ { v } ^ { \prime } = \left[ 4 \cdot a _ { v } \right] \cdot 0 . 2 5 + 0 . 1 2 5 .\tag{5}
$$

where $a _ { v }$ is the original continuous channel value and $a _ { v } ^ { \prime }$ is its quantized counterpart. This rule constrains each albedo channel to the set {0.125, 0.375, 0.625, 0.875}. By forcing the albedo into this standardized representation, the model must attribute illumination variations to the dedicated illumination factor $\mathbf { \xi } _ { l _ { i } , \mathrm { ~ ~ } }$ thereby achieving a normalized and robust appearance representation across the map.

<!-- image-->  
Fig. 2. IAN moduleâs effect on intrinsic appearance.

## 2.3. Dynamic Radiance Balancing Loss (DRB-Loss)

While the IAN module provides a stable base appearance for the map, it may not be sufficient to handle drastic, frame-wide radiance changes caused by factors like camera auto-exposure or moving between vastly different lighting environments. We, therefore, design the DRB-Loss as a reactive mechanism that activates only when necessary to model and compensate for these severe per-frame photometric variations. We dynamically detect the need for exposure correction using the Structural Similarity Index (SSIM) between the rendered image $I _ { \mathrm { r e n d e r } }$ and the ground-truth image $I _ { \mathrm { g t } }$ . When drastic illumination shifts occur (e.g., over or underexposure), the imageâs local structure (edges, textures) deviates significantly from that of the map rendered under a stable lighting assumption, leading to a drop in the SSIM score.

$$
S = { \mathrm { S S I M } } ( I _ { \mathrm { r e n d e r } } , I _ { \mathrm { g t } } ) .\tag{6}
$$

If S falls below a threshold TDRB (set to 0.50 in our experiments), the frame is flagged as an âexposure frame,â and the DRB-Loss is activated. Otherwise, the loss remains zero, ensuring no interference under normal lighting conditions.

DRB-Loss Calculation. For flagged frames, we introduce a set of learnable exposure parameters $\theta = \{ g , o \}$ to apply an affine transformation to the rendered image: $I _ { \mathrm { r e n d e r } } ^ { \theta } = g \cdot I _ { \mathrm { r e n d e r } } + o .$ The DRB-Loss then jointly optimizes Î¸ and the scene parameters:

$$
\mathcal { L } _ { \mathrm { { D R B } } } = ( 1 - S ) \cdot \left( \lambda _ { 1 } \cdot \Vert I _ { \mathrm { { r e n d e r } } } ^ { \theta } - I _ { \mathrm { { g t } } } \Vert _ { 1 } + \lambda _ { 2 } \cdot \Vert \nabla I _ { \mathrm { { r e n d e r } } } ^ { \theta } - \nabla I _ { \mathrm { { g t } } } \Vert _ { 1 } \right)\tag{7}
$$

Crucially, the loss is weighted by (1 â S), making the correction adaptive: the greater the structural discrepancy (lower S), the stronger the applied photometric supervision. The loss combines an L1 photometric term and a gradient difference term to preserve both color and structural fidelity. The gain g and offset o are constrained to [0.1, 10] and [â0.2, 0.2] respectively, mimicking the operational range of real camera adjustments.

## 2.4. Camera Tracking

In the tracking stage, we estimate the camera pose for each new frame by leveraging the stable intrinsic map from IAN and the dynamic correction from DRB-Loss.

Camera Pose Estimation. We initialize the pose of the current frame t + 1 using a constant velocity motion model: $E _ { t + 1 } = E _ { t }$ $\left( E _ { t } { \cdot } E _ { t - 1 } ^ { - 1 } \right)$ . This initial pose is then refined by minimizing a tracking loss.

Tracking Loss Optimization. The camera pose is optimized by minimizing the discrepancy between rendered and observed information. The loss function combines geometric constraints (depth), appearance consistency (albedo color), and semantic alignment. The DRB-Loss is conditionally activated for frames with significant exposure issues.

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { t r a c k i n g } } = \lambda _ { D } \cdot \| D ( p ) - D _ { \mathrm { g t } } \| _ { 1 } + \lambda _ { C } \cdot \| C _ { a } ( p ) - C _ { \mathrm { g t } } \| _ { 1 } } \\ & { \phantom { \mathcal { L } _ { \mathrm { t r a c k i n g } } = } + \lambda _ { S } \cdot \| S ( p ) - S _ { \mathrm { g t } } \| _ { 1 } } \\ & { \phantom { \mathcal { L } _ { \mathrm { t r a c k i n g } } = } + \lambda _ { \mathrm { D R B } } \cdot \mathbb { I } ( S < T _ { \mathrm { D R B } } ) \cdot \mathcal { L } _ { \mathrm { D R B } } . } \end{array}\tag{8}
$$

where $C _ { a } ( \boldsymbol { p } )$ is the rendered intrinsic albedo, and I(Â·) is the indicator function. This formulation ensures robust tracking even under challenging lighting.

Keyframe Selection. We employ a two-stage filtering process for keyframe selection. First, a Geometry Filter ensures sufficient visual overlap. We calculate the reprojection ratio Î· of existing map points into the candidate frameâs view:

$$
\eta = \frac { \sum _ { G _ { i } \in G _ { \operatorname* { m a p } } } \mathbb { I } ( \mathrm { i s } _ { - \mathrm { i n } _ { - } \mathrm { v i e w } } ( G _ { i } , E _ { \operatorname { c a n d } } ) ) } { | G _ { \operatorname* { m a p } } | } .\tag{9}
$$

A candidate frame is kept if Î· is within a predefined range. Second, a Semantic Filter promotes informational diversity. We discard candidate keyframes whose rendered semantic map is identical to that of the last selected keyframe. This strategy ensures that new keyframes contribute novel semantic information, preventing redundancy and improving mapping efficiency.

## 2.5. Map Construction

In the mapping stage, we fix the camera poses of selected keyframes and jointly optimize the parameters of all Gaussian primitives $( \mu , r , \sigma , a , l , s )$

Joint Optimization of Map Parameters. The mapping process is driven by a comprehensive loss function that integrates geometric, appearance, and semantic supervision from multiple keyframes. It also incorporates the IAN quantization and the conditional DRB-Loss to ensure the mapâs robustness and consistency.

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { m a p p i n g } } = \displaystyle \sum _ { p \in \mathcal { P } } \left( \lambda _ { D } \cdot \| D ( p ) - D _ { \mathrm { g t } } \| _ { 1 } + \lambda _ { C } \cdot \| C ( p ) - C _ { \mathrm { g t } } \| _ { 1 } \right. } \\ & { \left. ~ + \lambda _ { S } \cdot \mathcal { L } _ { \mathrm { C E } } ( S ( p ) , S _ { \mathrm { g t } } ) + \lambda _ { \mathrm { D R B } } \cdot \mathbb { I } ( S < T _ { \mathrm { D R B } } ) \cdot \mathcal { L } _ { \mathrm { D R B } } \right) . } \end{array}\tag{10}
$$

Here, P denotes the set of sampled pixels across keyframes. Note that for semantic supervision, we use a standard Cross-Entropy loss $( \mathcal { L } _ { \mathrm { C E } } )$ which is more suitable for classification tasks than SSIM. The map is dynamically densified and pruned during this process to reconstruct detailed scene geometry and appearance.

## 3. EXPERIMENTS

We evaluate our method on the synthetic Replica[28] and realworld ScanNet[29] datasets. We assess performance using established metrics for three key aspects: rendering quality (PSNRâ, SSIMâ, LPIPSâ), tracking accuracy (ATE RMSEâ), and semantic understanding (mIoUâ).

## 3.1. Result

To validate the robustness of our method against challenging illumination variations, we evaluate camera tracking accuracy on both Replica and ScanNet. As shown in Table 1, our method demonstrates superior tracking accuracy compared to the baseline and other state-of-the-art methods. This significant improvement is attributed to the stable intrinsic appearance representation provided by our IAN module and the dynamic compensation for exposure shifts by the DRB-Loss. (1) Specifically, our method reduces the average ATE RMSE on Replica from 0.45 cm (SGS-SLAM[15]) to 0.34 cm and on ScanNet from 12.23 cm to 11.30 cm. Notably, our method achieves performance on par with the state-of-the-art Hier-SLAM[16] on Replica[28] and sets a new SOTA on several challenging real-world ScanNet[29] scenes (e.g., âscene0181â, âscene0207â), proving its effectiveness and reliability in real-world conditions. (2) High-fidelity rendering is crucial for creating photorealistic digital twins. As detailed in Table 2, our method achieves state-of-theart performance across all three standard metrics (PSNR, SSIM, and LPIPS) on average. This demonstrates that our framework not only reconstructs accurate geometry but also learns a consistent and true-to-life scene appearance, effectively mitigating artifacts such as color inconsistencies and detail loss that are commonly caused by lighting changes. The superior rendering quality is also visually confirmed in Fig. 3. (3) Our method also excels in the downstream task of semantic segmentation. As shown in Table 3, our system achieves a top-tier mIoU of 92.69%, which is on par with the state-of-the-art SGS-SLAM (92.72%) while surpassing it in several scenes (e.g., R0, Of0). This result underscores a key tenet of our work: a robust geometric and appearance foundation, invariant to lighting, provides a clean and stable input for semantic feature extraction. This prevents misclassifications that can arise from photometric inconsistencies, thereby ensuring high performance in scene understanding tasks.

Table 1. Tracking performance on Replica[28] and ScanNet[29], measured by ATE RMSE (cm, lower is better â). Best results are in bold.
<table><tr><td>Dataset</td><td colspan="8">Replica</td><td colspan="8">ScanNet</td></tr><tr><td>Methods</td><td>Avg.</td><td>R0</td><td>R1</td><td>R2</td><td>Of0</td><td>Of1</td><td>Of2</td><td>Of3</td><td>Of4</td><td>Avg.</td><td>0000</td><td>0059</td><td>0106</td><td>0169</td><td>0181</td><td>0207</td></tr><tr><td>Vox-Fusion[30]</td><td>3.09</td><td>1.37</td><td>4.70</td><td>1.47</td><td>8.48</td><td>2.04</td><td>2.58</td><td>1.11</td><td>2.94</td><td>26.90</td><td>68.84</td><td>24.18</td><td>8.41</td><td>27.28</td><td>23.30</td><td>9.41</td></tr><tr><td>NICE-SLAM[31]</td><td>1.07</td><td>0.97</td><td>1.31</td><td>1.07</td><td>0.88</td><td>1.00</td><td>1.06</td><td>1.10</td><td>1.13</td><td>10.70</td><td>12.00</td><td>14.00</td><td>7.90</td><td>10.90</td><td>13.40</td><td>6.20</td></tr><tr><td>ESLAM[32]</td><td>0.63</td><td>0.71</td><td>0.70</td><td>0.52</td><td>0.57</td><td>0.55</td><td>0.58</td><td>0.72</td><td>0.63</td><td></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Point-SLAM[33]</td><td>0.52</td><td>0.61</td><td>0.41</td><td>0.37</td><td>0.38</td><td>0.48</td><td>0.54</td><td>0.69</td><td>0.72</td><td>12.19</td><td>10.24</td><td>7.81</td><td>8.65</td><td>22.16</td><td>14.77</td><td>9.54</td></tr><tr><td>MonoGS[3]</td><td>0.79</td><td>0.47</td><td>0.43</td><td>0.31</td><td>0.70</td><td>0.57</td><td>0.31</td><td>0.31</td><td>3.20</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SplaTAM[2]</td><td>0.36</td><td>0.31</td><td>0.40</td><td>0.29</td><td>0.47</td><td>0.27</td><td>0.29</td><td>0.32</td><td>0.55</td><td>11.88</td><td>12.83</td><td>10.14</td><td>17.72</td><td>12.08</td><td>11.10</td><td>7.46</td></tr><tr><td>SNI-SLAM[14]</td><td>0.46</td><td>0.50</td><td>0.55</td><td>0.45</td><td>0.35</td><td>0.41</td><td>0.33</td><td>0.62</td><td>0.50</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SemGauss-SLAM[34]</td><td>0.33</td><td>0.26</td><td>0.42</td><td>0.27</td><td>0.34</td><td>0.17</td><td>0.32</td><td>0.36</td><td>0.49</td><td></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Hier-SLAM[16]</td><td>0.33</td><td>0.21</td><td>0.49</td><td>0.24</td><td>0.29</td><td>0.16</td><td>0.31</td><td>0.37</td><td>0.53</td><td>11.80</td><td>12.83</td><td>9.57</td><td>17.54</td><td>11.54</td><td>11.78</td><td>7.55</td></tr><tr><td>SGS-SLAM[15]</td><td>0.45</td><td>0.46</td><td>0.41</td><td>0.35</td><td>0.46</td><td>0.27</td><td>0.56</td><td>0.90</td><td>0.60</td><td>12.23</td><td>14.90</td><td>11.10</td><td>17.83</td><td>12.22</td><td>9.67</td><td>7.63</td></tr><tr><td>Ours</td><td>0.34</td><td>0.25</td><td>0.36</td><td>0.33</td><td>0.50</td><td>0.21</td><td>0.32</td><td>0.30</td><td>0.45</td><td>11.30</td><td>14.39</td><td>9.07</td><td>17.13</td><td>11.91</td><td>8.26</td><td>7.06</td></tr></table>

Table 2. Rendering quality on the Replica[28]. Metrics are PSNR (dB, higher is better â), SSIM (higher is better â), and LPIPS (lower is better â).
<table><tr><td>Methods</td><td>Metrics</td><td>Avg.</td><td>R0</td><td>R1</td><td>R2</td><td>Of0</td><td>Of1</td><td>Of2</td><td>Of3</td><td>Of4</td></tr><tr><td rowspan="3">Vox-Fusion[30]</td><td>PSNRâ</td><td>24.41</td><td>22.39</td><td>22.36</td><td>23.92</td><td>27.79</td><td>29.83</td><td>20.33</td><td>23.47</td><td>25.21</td></tr><tr><td>SSIMâ</td><td>0.80</td><td>0.68</td><td>0.75</td><td>0.80</td><td>0.86</td><td>0.88</td><td>0.79</td><td>0.80</td><td>0.85</td></tr><tr><td>LPIPSâ</td><td>0.24</td><td>0.30</td><td>0.27</td><td>0.23</td><td>0.24</td><td>0.18</td><td>0.24</td><td>0.21</td><td>0.20</td></tr><tr><td rowspan="3">NICE-SLAM[31]</td><td>PSNRâ</td><td>24.42</td><td>22.12</td><td>22.47</td><td>24.52</td><td>29.07</td><td>30.34</td><td>19.66</td><td>22.23</td><td>24.96</td></tr><tr><td>SSIMâ</td><td>0.81</td><td>0.69</td><td>0.76</td><td>0.81</td><td>0.87</td><td>0.89</td><td>0.80</td><td>0.80</td><td>0.86</td></tr><tr><td>LPIPSâ</td><td>0.23</td><td>0.33</td><td>0.27</td><td>0.21</td><td>0.23</td><td>0.18</td><td>0.24</td><td>0.21</td><td>0.20</td></tr><tr><td rowspan="3">ESLAM[32]</td><td>PSNRâ</td><td>28.06</td><td>25.25</td><td>27.39</td><td>28.09</td><td>30.33</td><td>27.04</td><td>27.99</td><td>29.27</td><td>29.15</td></tr><tr><td>SSIMâ</td><td>0.92</td><td>0.87</td><td>0.89</td><td>0.96</td><td>0.93</td><td>0.91</td><td>0.94</td><td>0.95</td><td>0.95</td></tr><tr><td>LPIPS</td><td>0.26</td><td>0.32</td><td>0.30</td><td>0.25</td><td>0.21</td><td>0.25</td><td>0.24</td><td>0.19</td><td>0.21</td></tr><tr><td rowspan="3">SplaTAM[2]</td><td>PSNRâ</td><td>34.11</td><td>32.86</td><td>33.89</td><td>35.25</td><td>38.26</td><td>39.17</td><td>31.97</td><td>29.70</td><td>31.81</td></tr><tr><td>SSIMâ</td><td>0.97</td><td>0.98</td><td>0.97</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.97</td><td>0.95</td><td>0.95</td></tr><tr><td>LPIPSâ</td><td>0.10</td><td>0.07</td><td>0.10</td><td>0.08</td><td>0.09</td><td>0.09</td><td>0.10</td><td>0.12</td><td>0.15</td></tr><tr><td rowspan="3">SNI-SLAM[14]</td><td>PSNRâ</td><td>29.43</td><td>25.91</td><td>28.17</td><td>29.15</td><td>31.85</td><td>30.34</td><td>29.13</td><td>28.75</td><td>30.97</td></tr><tr><td>SSIMâ</td><td>0.92</td><td>0.88</td><td>0.90</td><td>0.92</td><td>0.94</td><td>0.93</td><td>0.93</td><td>0.93</td><td>0.94</td></tr><tr><td>LPIPSâ</td><td>0.23</td><td>0.31</td><td>0.29</td><td>0.26</td><td>0.19</td><td>0.21</td><td>0.23</td><td>0.21</td><td>0.20</td></tr><tr><td rowspan="3">SGS-SLAM[15]</td><td>PSNRâ</td><td>34.66</td><td>32.50</td><td>34.25</td><td>35.10</td><td>38.54</td><td>39.20</td><td>32.90</td><td>32.90</td><td>32.75</td></tr><tr><td>SSIMâ</td><td>0.97</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.97</td><td>0.97</td><td>0.95</td></tr><tr><td>LPIPSâ</td><td>0.10</td><td>0.07</td><td>0.09</td><td>0.07</td><td>0.09</td><td>0.09</td><td>0.10</td><td>0.12</td><td>0.15</td></tr><tr><td rowspan="3">Ours</td><td>PSNRâ</td><td>34.75</td><td>33.05</td><td>35.37</td><td>35.16</td><td>37.80</td><td>38.95</td><td>32.86</td><td>33.16</td><td>31.62</td></tr><tr><td>SSIMâ</td><td>0.97</td><td>0.98</td><td>0.97</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.97</td><td>0.95</td><td>0.95</td></tr><tr><td>LPIPSâ</td><td>0.10</td><td>0.07</td><td>0.10</td><td>0.07</td><td>0.08</td><td>0.09</td><td>0.10</td><td>0.11</td><td>0.15</td></tr></table>

Table 3. Semantic segmentation performance on Replica[28], measured by mIoU (%, higher is better â).
<table><tr><td>Methods</td><td>Avg. mIoU â</td><td>RO [%] â</td><td>R1 [%] â</td><td>R2 [%] â</td><td>Of0 [%] â</td></tr><tr><td>NIDS-SLAM[35]</td><td>82.37</td><td>82.45</td><td>84.08</td><td>76.99</td><td>85.94</td></tr><tr><td>DNS-SLAM[36]</td><td>87.77</td><td>88.32</td><td>84.90</td><td>81.20</td><td>84.66</td></tr><tr><td>SNI-SLAM[14]</td><td>87.41</td><td>88.42</td><td>87.43</td><td>86.16</td><td>87.63</td></tr><tr><td>SGS-SLAM[15]</td><td>92.72</td><td>92.72</td><td>92.91</td><td>92.10</td><td>92.90</td></tr><tr><td>Ours</td><td>92.69</td><td>92.78</td><td>92.85</td><td>92.02</td><td>93.10</td></tr></table>

Table 4. Ablation study on the Replica[28] (Room0) using SGS-SLAM[15] as the baseline.
<table><tr><td>Settings</td><td>Depth L1 [cm] â</td><td>ATE RMSE [cm] â</td><td>PSNR [dB] â</td><td>mIoU [%] â</td></tr><tr><td>Baseline(SGS-SLAM[15])</td><td>0.50</td><td>0.46</td><td>32.41</td><td>92.69</td></tr><tr><td>Baseline + IAN</td><td>0.50</td><td>0.36</td><td>32.22</td><td>92.40</td></tr><tr><td>Baseline + DRB-Loss</td><td>0.53</td><td>0.30</td><td>32.85</td><td>92.56</td></tr><tr><td>Ours (Full Model)</td><td>0.49</td><td>0.25</td><td>33.03</td><td>92.73</td></tr></table>

## 3.2. Ablation Studies

We conduct a thorough ablation study on the Replica[28] to dissect the contribution of our two core components: the proactive Intrinsic Appearance Normalization (IAN) module and the reactive Dynamic Radiance Balancing Loss (DRB-Loss). The results, presented in Table 4, clearly validate our design choices. (1) Effect of IAN: Adding only the IAN module to the baseline dramatically reduces the ATE RMSE from 0.46 cm to 0.36 cm. This provides strong evidence that regularizing the appearance via our quantization scheme yields a stable intrinsic representation that is crucial for robust tracking under varying illumination.(2)Effect of DRB-Loss: Individually, the DRB-Loss module also significantly improves tracking accuracy (ATE to 0.30 cm) while simultaneously boosting the rendering quality (PSNR from 32.41 to 32.85 dB). This highlights its dual benefit in handling extreme exposure frames for both tracking and mapping. (3) Full Model: Finally, our full model, integrating both modules, achieves the best performance across the board, reaching the lowest tracking error (0.25 cm) and the highest rendering quality and semantic accuracy. This demonstrates a clear synergistic effect between our proactive normalization (IAN) and reactive compensation (DRB-Loss) strategies, which work together to create a highly robust SLAM system. The qualitative effect of the IAN module is visualized in Fig. 2.

<!-- image-->  
Fig. 3. Visualization of Our Rendering Performance on the Replica Dataset.

## 4. CONCLUSION

In this work, we introduced an illumination-invariant semantic 3DGS-SLAM framework that effectively âtames the lightâ in realworld scenes. Our core strategy combines a proactive Intrinsic Appearance Normalization (IAN) module, which learns a canonical albedo, with a reactive Dynamic Radiance Balancing (DRB) Loss to compensate for exposure shifts. Extensive experiments and ablation studies validate our approach, demonstrating state-of-the-art performance in tracking, rendering, and semantic segmentation. By successfully disentangling intrinsic scene properties from transient lighting, our work marks a significant step towards deploying robust 3DGS-SLAM systems for real-world applications like robotics and augmented reality.

## 5. REFERENCES

[1] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, â3d Â¨ gaussian splatting for real-time radiance field rendering.â ACM Trans. Graph., vol. 42, no. 4, pp. 139â1, 2023.

[2] N. Keetha, J. Karhade, K. M. Jatavallabhula, G. Yang, S. Scherer, D. Ramanan, and J. Luiten, âSplatam: Splat track & map 3d gaussians for dense rgb-d slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 357â21 366.

[3] H. Matsuki, R. Murai, P. H. Kelly, and A. J. Davison, âGaussian splatting slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 039â18 048.

[4] Y. Guo, Y. Li, D. Ren, X. Zhang, J. Li, L. Pu, C. Ma, X. Zhan, J. Guo, M. Wei et al., âLidar-net: A real-scanned 3d point cloud dataset for indoor scenes,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 989â21 999.

[5] W. Li, J. Liu, Y. Wang, W. Hao, D. Ren, and L. Chen, âDlposenet: A differential lightweight network for pose regression over se (3),â in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 16 834â 16 840.

[6] W. Li, J. Liu, W. Hao, H. Liu, D. Ren, Y. Wang, and L. Chen, âOnline deep bingham network for probabilistic orientation estimation,â IET Computer Vision, vol. 17, no. 6, pp. 663â675, 2023.

[7] A. Kundu, K. Genova, X. Yin, A. Fathi, C. Pantofaru, L. J. Guibas, A. Tagliasacchi, F. Dellaert, and T. Funkhouser, âPanoptic neural fields: A semantic object-aware neural scene representation,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 12 871â12 881.

[8] F. Schmidt, M. Enzweiler, and A. Valada, âNerf and gaussian splatting slam in the wild,â arXiv preprint arXiv:2412.03263, 2024.

[9] P. Bergmann, R. Wang, and D. Cremers, âOnline photometric calibration of auto exposure video for realtime visual odometry and slam,â IEEE Robotics and Automation Letters, vol. 3, no. 2, pp. 627â634, 2017.

[10] P. Liu, X. Yuan, C. Zhang, Y. Song, C. Liu, and Z. Li, âRealtime photometric calibrated monocular direct visual slam,â Sensors, vol. 19, no. 16, p. 3604, 2019.

[11] A. Rosinol, J. J. Leonard, and L. Carlone, âNerf-slam: Realtime dense monocular slam with neural radiance fields,â in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 3437â3444.

[12] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, âMip-nerf 360: Unbounded anti-aliased neural radiance fields,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5470â 5479.

[13] J. McCormac, A. Handa, A. Davison, and S. Leutenegger, âSemanticfusion: Dense 3d semantic mapping with convolutional neural networks,â in 2017 IEEE International Conference on Robotics and automation (ICRA). IEEE, 2017, pp. 4628â 4635.

[14] S. Zhu, G. Wang, H. Blum, J. Liu, L. Song, M. Pollefeys, and H. Wang, âSni-slam: Semantic neural implicit slam,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21 167â21 177.

[15] M. Li, S. Liu, H. Zhou, G. Zhu, N. Cheng, T. Deng, and H. Wang, âSgs-slam: Semantic gaussian splatting for neural dense slam,â in European Conference on Computer Vision. Springer, 2024, pp. 163â179.

[16] B. Li, Z. Cai, Y.-F. Li, I. Reid, and H. Rezatofighi, âHier-slam: Scaling-up semantics in slam with a hierarchically categorical gaussian splatting,â in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 9748â 9754.

[17] S. Song, D. Ren, Z. Jia, and F. Shi, âAdaptive gaussian regularization constrained sparse subspace clustering for image segmentation,â in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 4400â4404.

[18] F. Zhang, F. Shi, D. Ren, and Y. Li, âA fuzzy c-means clustering algorithm for real medical image segmentation,â in ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1â5.

[19] D. Ren, Z. Jia, J. Yang, and N. K. Kasabov, âA practical grabcut color image segmentation based on bayes classification and simple linear iterative clustering,â IEEE Access, vol. 5, pp. 18 480â18 487, 2017.

[20] C. Chen, Y. Wang, H. Chen, X. Yan, D. Ren, Y. Guo, H. Xie, F. L. Wang, and M. Wei, âGeosegnet: point cloud semantic segmentation via geometric encoderâdecoder modeling,â The Visual Computer, vol. 40, no. 8, pp. 5107â5121, 2024.

[21] D. Ren, Z. Wu, J. Li, P. Yu, J. Guo, M. Wei, and Y. Guo, âPoint attention network for point cloud semantic segmentation,â Science China Information Sciences, vol. 65, no. 9, p. 192104, 2022.

[22] D. Ren, Z. Ma, Y. Chen, W. Peng, X. Liu, Y. Zhang, and Y. Guo, âSpiking pointnet: Spiking neural networks for point clouds,â Advances in Neural Information Processing Systems, vol. 36, 2024.

[23] L. Diao, D. Ren, S. Song, and Y. Qian, âZigzagpointmamba: Spatial-semantic mamba for point cloud understanding,â arXiv preprint arXiv:2505.21381, 2025.

[24] D. Ren, S. Yang, W. Li, J. Guo, and Y. Guo, âSae: Estimation for transition matrix in annotation algorithms,â arXiv preprint, 2022.

[25] M. Runz, M. Buffier, and L. Agapito, âMaskfusion: Real-time recognition, tracking and reconstruction of multiple moving objects,â in 2018 IEEE international symposium on mixed and augmented reality (ISMAR). IEEE, 2018, pp. 10â20.

[26] A. Rosinol, M. Abate, Y. Chang, and L. Carlone, âKimera: an open-source library for real-time metric-semantic localization and mapping,â in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 1689â 1696.

[27] X. Zhang, A. Kundu, T. Funkhouser, L. Guibas, H. Su, and K. Genova, âNerflets: Local radiance fields for efficient structure-aware 3d scene representation from 2d supervision,â in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 8274â8284.

[28] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma et al., âThe replica dataset: A digital replica of indoor spaces,â arXiv preprint arXiv:1906.05797, 2019.

[29] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. NieÃner, âScannet: Richly-annotated 3d reconstructions of indoor scenes,â in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5828â5839.

[30] X. Yang, H. Li, H. Zhai, Y. Ming, Y. Liu, and G. Zhang, âVoxfusion: Dense tracking and mapping with voxel-based neural implicit representation,â in 2022 IEEE International Symposium on Mixed and Augmented Reality (ISMAR). IEEE, 2022, pp. 499â507.

[31] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, âNice-slam: Neural implicit scalable encoding for slam,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 12 786â12 796.

[32] M. M. Johari, C. Carta, and F. Fleuret, âEslam: Efficient dense slam system based on hybrid representation of signed distance fields,â in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 17 408â17 419.

[33] E. Sandstrom, Y. Li, L. Van Gool, and M. R. Oswald, âPoint- Â¨ slam: Dense neural point cloud-based slam,â in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18 433â18 444.

[34] S. Zhu, R. Qin, G. Wang, J. Liu, and H. Wang, âSemgaussslam: Dense semantic gaussian splatting slam,â arXiv preprint arXiv:2403.07494, 2024.

[35] Y. Haghighi, S. Kumar, J.-P. Thiran, and L. Van Gool, âNeural implicit dense semantic slam,â arXiv preprint arXiv:2304.14560, 2023.

[36] K. Li, M. Niemeyer, N. Navab, and F. Tombari, âDns-slam: Dense neural semantic-informed slam,â in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 7839â7846.