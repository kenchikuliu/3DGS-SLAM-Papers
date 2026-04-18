# Scalable Adaptation of 3D Geometric Foundation Models via Weak Supervision from Internet Video

Zihui Gao 1 Ke Liu 1 Donny Y. Chen 2 Duochao Shi 1 Guosheng Lin 3 Hao Chen 1 Chunhua Shen 1 4

Geometric foundation models show promise in 3D reconstruction, yet their progress is severely constrained by the scarcity of diverse, large-scale 3D annotations. While Internet videos offer virtually unlimited raw data, utilizing them as a scaling source for geometric learning is challenging due to the absence of ground-truth geometry and the presence of observational noise. To address this, we propose SAGE, a framework for Scalable Adaptation of GEometric foundation models from raw video streams. SAGE leverages a hierarchical mining pipeline to transform videos into training trajectories and hybrid supervision: (1) Informative training trajectory selection; (2) Sparse Geometric Anchoring via SfM point clouds for global structural guidance; and (3) Dense Differentiable Consistency via 3D Gaussian rendering for multiview constraints. To prevent catastrophic forgetting, we introduce a regularization strategy using anchor data. Extensive experiments show that SAGE significantly enhances zero-shot generalization, reducing Chamfer Distance by 20-42% on unseen benchmarks (7Scenes, TUM-RGBD, Matterport3D) compared to state-of-the-art baselines. To our knowledge, SAGE pioneers the adaptation of geometric foundation models via Internet video, establishing a scalable paradigm for general-purpose 3D learning.

## 1. Introduction

Recent advances in 3D geometric foundation models, such as DUSt3R (Wang et al., 2024b) and MASt3R (Leroy et al., 2024), have transformed the pipeline of reconstructing 3D geometry from unposed 2D images. These methods move beyond traditional Structure-from-Motion (SfM) (Snavely et al., 2006) pipelines with end-to-end data-driven networks that directly estimate pixel-aligned pointmaps in a unified coordinate system. Further developments like MV-DUSt3R (Tang et al., 2024), Fast3R (Yang et al., 2025), and VGGT (Wang et al., 2025a) extend the framework to multiview inputs, enabling dense 3D reconstruction in a single forward pass. These 3D foundation models achieve impressive results even with extremely sparse inputs or in zeroshot settings, demonstrating strong generalization across unseen scenes. However, their further scaling is severely bottlenecked by the scarcity of high-quality 3D annotations. Unlike 2D vision models trained on billions of images, 3D models heavily rely on sensor-captured datasets like Scan-Net, which are costly to scale and restricted to static, indoor environments. This reliance on limited supervised data results in a significant distribution shift when applying these models to diverse, in-the-wild scenes.

<!-- image-->  
Figure 1. Overview of the SAGE framework for scaling a General-Purpose 3D Foundation Model. To overcome the Data Scarcity Bottleneck inherent in limited labeled 3D datasets, we propose a pipeline that leverages unlimited Internet videos to achieve robust zero-shot generalization on complex unseen scenes. Bottom right: Scalability analysis demonstrates zero-shot reconstruction performance as training data scales from 100 to 10K video scenes, showing consistent improvement with increased data volume across various benchmarks (MP3D, 7-Scenes, and TUM).

To overcome this data scalability bottleneck, Internet videos emerge as a promising alternative, offering virtually unlimited coverage of diverse real-world environments. For example, the RealEstate10K (Zhou et al., 2018) dataset, sourced from YouTube videos, contains over 10,000 unique scenes, far more than 3D datasets like ScanNet (Dai et al., 2017), which has only around 1,000 depth sensorâannotated scenes. However, unlocking this potential is non-trivial due to the lack of ground-truth geometry and camera poses. A straightforward approach, we called VDS, which fine-tunes models using pseudo-depth labels from video depth estimators, often fails to improve or even degrades performance as shown in our empirical analysis. The main reason is that the predicted video depths contain substantial noise and often lack multi-view consistency, making accurate alignment difficult and ultimately degrading reconstruction quality. Therefore, the core challenge lies in extracting robust supervision signals from noisy, unlabelled video streams without drifting into degenerate solutions.

To tackle this challenge, we propose SAGE, a framework for the Scalable Adaptation of GEometric foundation models. Specifically, we develop a spatio-temporal mining pipeline to curate informative training trajectories as well as complementary supervisions from raw videos. We first perform stochastic trajectory sampling to curate frame sequences balancing viewpoint diversity and geometric overlap, providing the data foundation for multi-view learning. Upon these curated trajectories, we utilize sparse point clouds derived from SfM as geometric anchors to provide reliable global structural guidance. To complement these sparse anchors, we leverage differentiable 3D Gaussian Splatting to enforce dense photometric consistency across video frames. Furthermore, to stabilize the adaptation process on diverse video distributions, we introduce a regularization strategy using a small fraction of anchor data, effectively mitigating catastrophic forgetting of the original geometric priors.

Extensive experiments provide empirical evidence of scalability for our SAGE as shown in Figure 1. SAGE demonstrates a consistent log-linear improvement in zero-shot accuracy as the volume of video data scales from 100 to 10K scenes. This confirms that our weak supervision strategy successfully converts virtually unbounded raw video into effective and scalable training signals, without saturation. We validate our approach by adapting a pretrained multi-view reconstruction model (MV-DUSt3R (Tang et al., 2024)) using over 10,000 video clips from RealEstate10K and DL3DV (Ling et al., 2024), which is 10 times larger than the standard 3D datasets. Both in-distribution performance and zero-shot generalization capabilities are improved by our SAGE. Especially on unseen benchmarks such as 7Scenes, TUM-RGBD, and Matterport3D, our method reduces reconstruction error (Chamfer Distance) by 20-42% compared to the baseline.

Our main contributions are summarized as follows:

â¢ We propose SAGE, the first framework to enable the scalable adaptation of feed-forward geometric foundation models using unlabelled Internet video.

â¢ We design a robust hybrid supervision strategy that integrates sparse geometric anchors, dense spatiotemporal consistency, and regularization to effectively learn from noisy video data.

â¢ We demonstrate significant improvements in zero-shot generalization across 4 benchmarks and the scalability of our approach, establishing Internet video as a viable resource for evolving general-purpose 3D models.

## 2. Related work

## 2.1. Geometric Foundation Models & Data Scarcity

Traditional approaches to pose-free multi-view 3D reconstruction include Structure-from-Motion (SfM) (Snavely et al., 2006; Crandall et al., 2012; Schonberger & Frahm, 2016; Cui et al., 2017) and Multi-View Stereo (MVS) (Seitz et al., 2006; Yao et al., 2018; Goesele et al., 2006). While effective, they rely on complex multi-stage optimization. Recently, the field has shifted towards Geometric Foundation Models such as DUSt3R (Wang et al., 2024b) and VG-GSfM (Wang et al., 2024a), which jointly estimate camera parameters and 3D structure in a data-driven manner. Extensions including MV-DUSt3R, Fast3R (Yang et al., 2025), VGGT (Wang et al., 2025a) and $\pi ^ { 3 }$ (Wang et al., 2025c) further scale this to multi-view inputs.

However, these models are severely bottlenecked by the scarcity of 3D supervision. Unlike 2D models trained on web-scale data, they rely on costly annotated datasets like ScanNet, limiting their generalization to in-the-wild distributions. While methods like CUT3R (Wang et al., 2025b) use video sequences, they directly supervises on noisy SfM poses. In contrast, our approach addresses this scalability bottleneck by leveraging unlabelled Internet video. We propose a weakly-supervised adaptation framework that combines sparse geometric anchors with multi-view consistency, enabling foundation models to learn from diverse, real-world data without ground-truth poses.

## 2.2. Differentiable Rendering as Weak Supervision

Recent works PixelSplat (Charatan et al., 2024), MVSplat (Chen et al., 2024a), and DepthSplat (Xu et al., 2024) predict 3D Gaussian representations from sparse views for novel view synthesis. Compared with optimization-based methods like 3DGS (Kerbl et al., 2023), these feedforward approaches offer better generalization. Extensions such as MVSplat360 (Chen et al., 2024b) and LatentSplat (Wewer et al., 2024) further improve rendering flexibility.

While these methods target view synthesis, we repurpose the differentiable nature of 3D Gaussians as a dense supervision mechanism. In our framework, the Gaussian renderer serves as a differentiable loss function that enforces photometric consistency across video frames. This allows us to propagate dense gradients to the underlying geometry in regions undefined by sparse points. Unlike MV-DUSt3R, which trains a Gaussian head for downstream rendering, we utilize this differentiable consistency to robustly adapt the

geometric backbone itself.

## 2.3. Scalable Representation Learning from Video

Self-supervised learning on video sequences leverages spatiotemporal consistency as an intrinsic supervision signal. Early works (Zhou et al., 2017) use image-warping losses, while subsequent methods (Bian et al., 2019; Godard et al., 2019) introduce scale-consistent predictions for depth estimation. More recently, CroCo (Weinzaepfel et al., 2022b; 2023) extends masked image modeling to cross-view settings for representation learning.

While these methods typically train task-specific networks from scratch, they share the principle of deriving supervision from video consistency. Our work extends this principle to the adaptation of pre-trained geometric foundation models. By integrating sparse geometric constraints (bias-reduction) with dense video consistency (variance-reduction), we propose a hybrid learning framework that effectively fine-tunes general-purpose models on noisy, large-scale video data, significantly enhancing their zero-shot generalization.

## 3. Method

We address the problem of scaling up a pre-trained geometric foundation model, fÎ¸, by adapting it to the domain of unlabelled Internet videos. We formulate this as a weaklysupervised optimization problem, where the goal is to leverage intrinsic spatiotemporal consistency to refine the model without requiring ground-truth 3D annotations.

## 3.1. Preliminary & Problem Formulation

We formulate the training of geometric foundation models as a scalable optimization problem, transitioning from a data-scarce supervised regime to a data-abundant weaklysupervised regime.

## 3.1.1. TASK INSTANTIATION.

We focus on unconstrained sparse-view 3D reconstruction.

Definition 3.1 (pose-free sparse-view 3D reconstruction). Let $f _ { \theta } : \mathcal { T } ^ { N } \to \mathcal { X }$ denote a geometric foundation model parameterized by Î¸. Given a set of unposed RGB images $\bar { I } \in \mathcal { I } ^ { N }$ , the model directly predicts a dense 3D representation X (specifically, pixel-aligned 3D pointmaps) without requiring ground-truth camera poses.

Our objective is to leverage the unbounded scale of Internet video to transcend the generalization limits of $f _ { \theta }$

## 3.1.2. BOTTLENECK & SCALING SOURCE.

Supervised Regime (Bottleneck) Existing models are typically initialized on a supervised dataset $\begin{array} { r l } { \mathcal { D } _ { s u p } } & { { } = } \end{array}$ $\{ ( I , Y ) \}$ , where Y represents expensive ground-truth 3D annotations (e.g., depth, poses). While $\mathcal { D } _ { s u p }$ offers highfidelity supervision, it is fundamentally non-scalable due to the prohibitive cost of 3D data acquisition. Consequently, $| \mathcal { D } _ { s u p } |$ remains small, restricting the modelâs exposure to a narrow distribution of scenes and limiting its zero-shot generalization capabilities to in-the-wild scenarios.

Weakly-Supervised Regime (Scaling Source) To address this bottleneck, we introduce an unbounded stream of Internet videos, denoted as $\mathcal { D } _ { v i d e o } = \{ V _ { j } \}$ . In contrast to $\mathcal { D } _ { s u p }$ , this dataset is scalable $( | \mathcal { D } _ { v i d e o } | \to \infty )$ but unlabelled: it lacks explicit 3D geometry $( Y = \varnothing )$ and camera poses. However, it contains rich intrinsic supervision encoded in physical constraints, specifically spatiotemporal consistency.

## 3.1.3. OPTIMIZATION OBJECTIVE.

We aim to learn an optimal parameter set $\theta ^ { * }$ that absorbs geometric knowledge from massive video data while preserving the structural priors learned from the supervised regime. We formulate it as a constrained optimization problem:

$$
\begin{array} { r l } & { \underset { \theta } { \operatorname* { m i n } } \underbrace { \mathbb { E } _ { V \sim \mathcal { D } _ { v i d e o } } [ \mathcal { L } _ { c o n s i s t e n c y } ( f _ { \theta } ( V ) ) ] } _ { \mathrm { S c a l i n g ~ v i a ~ W e a k ~ S u p e r v i s i o n } } } \\ & { ~ + \gamma \cdot \underbrace { \mathbb { E } _ { ( I , Y ) \sim \mathcal { D } _ { s u p } } [ \mathcal { L } _ { s u p } ( f _ { \theta } ( I ) , Y ) ] } _ { \mathrm { F o u n d a t i o n ~ R e g u l a r i z a t i o n } } , } \end{array}\tag{1}
$$

where $\mathcal { L } _ { { c o n s i s t e n c y } }$ represents a hybrid loss function designed to distill robust geometric signals from noisy video consistency, and the second term serves as a regularization constraint to prevent catastrophic forgetting.

## 3.2. Scaling with Spatio-Temporal Weak Supervision

A fundamental requirement for scaling 3D foundation models is the ability to harvest informative training trajectories and latent supervision from massive, unconstrained video streams. Unlike structured 3D datasets, raw videos are inherently redundant and exhibit inconsistent viewpoint transitions. To address this, we propose a spatio-temporal mining pipeline in Figure 2. First, we curate training trajectories by balancing viewpoint diversity and geometric overlap to ensure representative scene coverage. For these trajectories, we derive two levels of weak supervision: (1) Sparse Geometric Anchors from COLMAP point clouds to provide global structural bias; (2) Dense Differentiable Consistency via 3D Gaussian rendering to propagate gradients through video spatiotemporal consistency. This synergy allows our model to effectively optimize Eq. 1, providing essential supervision for learning robust 3D geometry from diverse real-world scenes in a scalable and stable manner.

<!-- image-->  
Figure 2. Illustration of SAGE. For each video sequence, we sample context frames as model inputs and designate target frames for novel-view supervision, providing photometric constraints to refine the reconstructed 3D point cloud. Furthermore, we incorporate sparse 3D point clouds that provide consistent geometric constraints, complementing the photometric supervision from the target views.

## 3.2.1. SPATIOTEMPORAL SAMPLING FROM VIDEOS

Unlike prior works (Wang et al., 2024b; Tang et al., 2024) that rely on ground-truth depth to determine image overlap, our approach tackles the scaling challenge where only raw video streams are available. We exploit the inherent spatiotemporal continuity of videos as a geometric overlap proxy to mine multi-view training samples.

Given a video sequence, we construct training trajectories V by balancing viewpoint overlap (ensuring stable supervision) and baseline diversity (promoting geometric learning). While uniform temporal sampling with interval ât:

$$
\mathcal { V } _ { \mathrm { u n i f o r m } } = \{ I _ { t _ { 0 } + i \Delta t } \ | \ i = 0 , 1 , \dots , n \}\tag{2}
$$

maintains consistent visual transitions, it constrains viewpoint diversity and may introduce motion pattern bias.

We instead employ stochastic temporal perturbations by sampling offsets $\delta _ { i } \sim \mathcal { U } ( - \epsilon , \epsilon )$

$$
\mathcal { V } _ { \mathrm { p e r t u r b e d } } = \{ I _ { t _ { 0 } + i \Delta t + \delta _ { i } } \ | \ i = 0 , 1 , \ldots , n \} ,\tag{3}
$$

which preserves temporal structure for geometric consistency while maximizing camera pose diversity. This automated curation establishes the data foundation for scalable learning, from which we subsequently derive hierarchical supervision signals (Sec. 3.2.2, 3.2.3).

## 3.2.2. SPARSE GEOMETRIC ANCHORS

While dense pseudo-depth maps can be easily generated by off-the-shelf models, they often suffer from multi-view inconsistency and scale drift, which can destabilize the scaling process. In contrast, sparse point clouds derived from SfM provide globally consistent geometric anchors, offering a reliable structural bias. Specifically, the sparse point cloud is projected onto each image to generate corresponding sparse depth maps, using the camera intrinsics

K and extrinsics [R | t] from COLMAP. Each 3D point $\mathbf { P } = ( x , y , z )$ is projected to image coordinates $( u , v , d )$ via $\left[ u \quad v \quad 1 \right] ^ { \top } \sim \mathbf { K } [ \mathbf { R } | \mathbf { t } ] \left[ x \quad y \quad z \quad 1 \right] ^ { \top }$ , where d denotes the projected depth. Then following DUSt3R and MV-DUSt3R, the confidence-aware 3D loss is defined as:

$$
\mathcal { L } _ { \mathrm { a n c h o r } } = \sum _ { v \in \{ 1 , \ldots , N \} } \sum _ { i \in \mathcal { D } ^ { v } } C _ { i } ^ { v , 1 } \ell _ { \mathrm { r e g r } } ( v , i ) - \alpha \log C _ { i } ^ { v , 1 } ,\tag{4}
$$

$$
\ell _ { \mathrm { r e g r } } ( v , i ) = \left\| \frac { 1 } { z } X _ { i } ^ { v , 1 } - \frac { 1 } { \bar { z } } \bar { X } _ { i } ^ { v , 1 } \right\| ,\tag{5}
$$

where z, zÂ¯ are scale normalization factors and $C _ { i } ^ { v , 1 }$ is the predicted confidence score for pixel i. The term $X _ { i } ^ { v , 1 }$ represents the predicted 3D coordinate in the reference coordinate. This term anchors the predicted geometry to the global SfM structure, mitigating potential geometric drift.

## 3.2.3. DENSE DIFFERENTIABLE CONSISTENCY

Sparse anchors alone leave the majority of the surface undefined. To propagate geometric supervision to dense regions, we derive the Spatiotemporal Consistency from video. We employ 3D Gaussian Splatting as a differentiable rendering module. Unlike prior works using Gaussians for view synthesis tasks, we use it strictly as a supervision mechanism.

Given a sampled video clip of M temporally ordered frames $\mathcal { V } _ { \mathrm { c l i p } } = \{ I _ { t _ { 1 } } , I _ { t _ { 2 } } , . . . , I _ { t _ { M } } \}$ , we divide them into context frames as inputs and target frames as supervision for novel view synthesis. Target views are unseen frames from the same video, used to enforce multi-view consistency during training. To ensure effective supervision, we random sample intermediate frames $t _ { j } \in \left\{ t _ { 2 } , t _ { 3 } , \dots , t _ { M - 1 } \right\}$ as target views, avoiding extrapolated viewpoints that may introduce artifacts in Gaussian splatting reconstructions. $\hat { I } _ { k }$ is rendered from differentiable Gaussian Splatting from the predicted point cloud. The loss function is:

$$
\mathcal { L } _ { \mathrm { r e n } } = \sum _ { I _ { k } \in \{ I _ { c o n } , I _ { t g t } \} } \gamma \cdot \mathrm { L P I P S } ( I _ { k } , \hat { I } _ { k } ) + \Big | \Big | I _ { k } - \hat { I } _ { k } \Big | \Big | _ { 2 } ,\tag{6}
$$

where $I _ { c o n }$ and $I _ { t g t }$ are context and target frames. This loss enforces image consistency through novel view synthesis of 3D Gaussians, thereby constraining the Gaussian centers and improving the accuracy of the predicted 3D point cloud.

## 3.3. Regularization against Catastrophic Forgetting

A critical challenge in adapting foundation models to noisy video data is distribution drift, where the model overfits to the video domain and loses its original general-purpose capabilities. To explicitly mitigate this, we implement the regularization term in Eq. 1 by mixing a small set of anchor data $\mathcal { D } _ { s u p }$ from the original pre-training distribution.

We define the sampling ratio between video data and anchor data as Î·. Empirically, we find that a small fraction $( \eta \approx 3 \% )$ is sufficient to stabilize the optimization landscape. This regularization constrains the model parameters to remain within the valid manifold of plausible 3D geometries, preventing catastrophic forgetting while allowing the model to absorb new diversity from Internet videos.

## 4. Experiments

## 4.1. Experimental Setup

We use the pretrained MV-DUSt3R (Tang et al., 2024) model as our multi-view reconstruction baseline, originally trained on 3D ground-truth annotated datasets, and we finetune it with Internet video data to enhance its 3D reconstruction performance.

Datasets We use videos from RealEstate10K (Zhou et al., 2018) and DL3DV (Ling et al., 2024) as the primary training sources. These datasets contain a large number of diverse scenes of common real-life environments. We preprocess the dataset by filtering out short videos. In total, over 10,000 video clips are utilized for training, resulting in a 10-fold increase in scene diversity compared to standard 3D datasets. For more details, please refer to App. B.1.

To assess the reconstruction performance, we evaluate all models on 4 benchmark. ScanNet (Dai et al., 2017) is divided into training and test sets; all methods are trained on the training split and tested on the official test split. In addition, we include 7Scenes (Shotton et al., 2013), TUM RGB-D (Sturm et al., 2012), and Matterport3D (Chang et al., 2017) as zero-shot datasets, which are not seen during training by any of the methods. These datasets provide a more rigorous assessment of generalization ability to unseen and diverse real-world scenes.

Implementation details Following MV-DUSt3R (Tang et al., 2024), we train on input images of resolution 224 Ã 224. Each training sample consists of 8 input views, with either 2 or 8 novel views rendered for supervision. During training, we mix the video data from RealEstate10K and DL3DV. Additionally, we include approximately 3% (about 400) pretrained samples, resulting in a 30:1 ratio between video data and pretrained supervision data. For more details, please refer to App. D.3.

Evaluation metrics We evaluate reconstruction quality using two main metrics: Chamfer Distance (CD) (AanÃ¦s et al., 2016) and Distance Accuracy @0.2 (DAc), which measures the proportion of pixels whose normalized distance to the ground-truth pointmap is less than or equal to 0.2. CD serves as a global geometric error metric but is sensitive to point cloud density and outliers, and can be affected by prediction sparsity or noise. In contrast, DAc directly quantifies the proportion of well-aligned points and more reliably reflects overall structural consistency, making it particularly suitable for evaluating dense and accurate reconstructions. In more challenging settings, such as the MP3D (Chang et al., 2017) dataset characterized by low viewpoint overlap and textureless surfaces, we use Distance Accuracy @0.5 (DAc@0.5) as a more appropriate metric. It offers a more reliable assessment of reconstruction performance under such conditions.

## 4.2. Sparse 3D Reconstruction from Multi Inputs

We evaluate our SAGE on four widely used datasets Scan-Net, 7Scenes, TUM RGBD, and MP3D. The comparison includes several recent state-of-the-art methods, including Spann3R (Wang & Agapito, 2024), DUSt3R (Wang et al., 2024b), VGGT (Wang et al., 2025a), and MV-DUSt3R (Tang et al., 2024). For a fair comparison, we follow the standard 8-view input setting across all methods. Additional experiments with varying numbers of input views (ranging from 4 to 24) are provided in App C.2. As shown in Table 1, SAGE consistently outperforms all models trained at the standard 224 resolution across four benchmarks, achieving lower Chamfer Distance and higher Distance Accuracy. Some metrics remain below VGGT, which was trained at a higher 518 resolution, reflecting the impact of training resolution on performance. Compared to MV-DUST3R, SAGE reduces CD by approximately 6%, 20%, 29%, and 42% on ScanNet, 7Scenes, TUM-RGBD, and MP3D, respectively. Notably, ScanNet is included during supervised training for all the methods; the remaining three datasets represent zero-shot scenarios. This highlights the benefit of leveraging diverse video data, leading to improved reconstruction quality and stronger generalization to unseen environments, underscoring its value as a scalable and effective resource for 3D reconstruction models.

Table 1. 8-view reconstruction results on four datasets (ScanNet, 7Scenes, TUM-RGBD, and Matterport3D). GA denotes global alignment optimization. â  indicates models trained at a resolution of 518, the default for VGGT; all other models are trained at 224 Ã 224. The higher training resolution gives VGGT an advantage, leading to better performance on some datasets. VDS (Video Depth Supervision) denotes fine-tuning on video data using pseudo depth from Depth Anything Video.
<table><tr><td colspan="2"></td><td colspan="2">In-distribution</td><td colspan="6">Zero-shot Generalization</td></tr><tr><td rowspan="2">Method</td><td rowspan="2">GA</td><td colspan="2">ScanNet DAc@0.2 â CD â</td><td colspan="2">7Scenes DAc@0.2 â CD â</td><td colspan="2">TUM-RGBD DAc@0.2 â CD â</td><td colspan="2">Matterport3D DAc@0.5â</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>CD â</td></tr><tr><td>Spann3R (Wang &amp; Agapito, 2024) </td><td>Ã</td><td>59.0</td><td>13.48 |</td><td>90.0</td><td>1.48 |</td><td>35.7</td><td>7.06</td><td>15.8</td><td>69.71</td></tr><tr><td>DUSt3R (Wang et al., 2024b)</td><td>â</td><td>88.0</td><td>4.50</td><td>89.5</td><td>1.50</td><td>71.4</td><td>4.65</td><td>45.4</td><td>44.10</td></tr><tr><td>VGGTâ  (Wang et al., 2025a)</td><td>Ã</td><td>90.0</td><td>3.12</td><td>99.0</td><td>0.65</td><td>80.0</td><td>3.92</td><td>57.0</td><td>20.36</td></tr><tr><td>MV-DUSt3R (Tang et al., 2024)</td><td>Ã</td><td>88.2</td><td>1.79</td><td>89.6</td><td>0.95</td><td>87.5</td><td>2.60</td><td>30.3</td><td>43.23</td></tr><tr><td>Ours (VDS)</td><td>ÃÃ</td><td>90.0</td><td>1.78</td><td>92.5</td><td>0.89</td><td>84.7</td><td>2.68</td><td>14.3</td><td>82.08</td></tr><tr><td>Ours (SAGE-1K)</td><td></td><td>9000</td><td>1.61</td><td>96.0</td><td>0.79</td><td>91.7</td><td>2.05</td><td>47.4</td><td>31.66</td></tr><tr><td>Ours (SAGE-10K)</td><td>X</td><td>91.2</td><td>1.69</td><td>98.0</td><td>0.76</td><td>93.1</td><td>1.85</td><td>48.0</td><td>24.90</td></tr></table>

Table 2. Ablation study on hybrid supervision and training settings. We evaluate the contribution of each supervision component and the impact of different training strategies.
<table><tr><td></td><td colspan="2">ScanNet</td><td colspan="2">7Scenes</td><td colspan="2">TUM</td><td colspan="2">MP3D</td></tr><tr><td>Method</td><td>DAcâ</td><td>CDâ</td><td>DAcâ</td><td>CDâ</td><td>DAcâ</td><td>CDâ</td><td>DAcâ</td><td>CDâ</td></tr><tr><td>Baseline (Foundation)</td><td>88.2</td><td>1.79</td><td>89.6</td><td>0.95</td><td>87.5</td><td>2.60</td><td>30.3</td><td>43.2</td></tr><tr><td>Ours (Full)</td><td>91.2</td><td>1.69</td><td>98.0</td><td>0.76</td><td>93.1</td><td>1.85</td><td>48.0</td><td>24.9</td></tr><tr><td>Component Ablation:</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o Sparse Anchors (LAnchor)</td><td>77.0</td><td>4.33</td><td>88.5</td><td>1.43</td><td>69.4</td><td>5.20</td><td>18.9</td><td>33.6</td></tr><tr><td>w/o Dense Consistency (Lren)</td><td>89.2</td><td>1.89</td><td>93.5</td><td>0.79</td><td>86.1</td><td>2.26</td><td>49.2</td><td>26.0</td></tr><tr><td>w/o Anchor Regularization</td><td>76.0</td><td>3.32</td><td>88.5</td><td>1.25</td><td>80.6</td><td>2.14</td><td>45.4</td><td>32.5</td></tr><tr><td>w/o Frozen GS Params</td><td>91.2</td><td>1.70</td><td>93.5</td><td>0.72</td><td>91.7</td><td>2.09</td><td>46.7</td><td>23.7</td></tr><tr><td>Strategy Variants:</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/Linear Sampling (vs. Jitter)</td><td>91.0</td><td>1.88</td><td>95.0</td><td>0.84</td><td>90.2</td><td>2.06</td><td>49.0</td><td>25.1</td></tr><tr><td>w/ 8 Target Views (vs. 2)</td><td>90.2</td><td>1.73</td><td>96.5</td><td>0.78</td><td>90.3</td><td>1.84</td><td>49.7</td><td>26.0</td></tr></table>

Qualitative reconstruction comparisons in Figure 4 show that SAGE produces the most accurate and geometrically consistent reconstructions compared to other approaches. More NVS results can be found in App. E.2.

In Table 1, we also evaluate a model trained with pseudo labels generated from video depth estimation model Depth Any Video (Yang et al., 2024), which results in noticeably weaker performance across all datasets. The video depth model remains limited, as multi-frame predictions are often inconsistent and noisy, making it difficult to generate reliable 3D supervision from video data for training 3D reconstruction models.

## 4.3. Ablations

Effect of Hybrid Supervision We evaluate the contribution of each component in our hybrid training strategy by removing one supervision at a time. Specifically, we independently ablate the Sparse Anchors, the Dense Consistency, and the small amount of Regularization data used to stabilize fine-tuning. The results in Table 2 show that the full combination achieves the best reconstruction quality, while excluding any single component leads to a noticeable performance drop. We further examine whether to freeze the pretrained Gaussian representation module during finetuning. Results indicate that freezing the GS module leads to more stable optimization and better adaptation of the reconstruction network, whereas updating GS jointly with the network tends to degrade performance.

Number of Rendered Views We analyze the effect of varying the number of rendered novel views per training sample, as shown in Table 2. By default, we supervise the model with two target views to balance training efficiency and supervision quality. Increasing the number of rendered views (e.g., w/ 8 Target Views) provides stronger geometric constraints but also exposes the model to more unseen regions across views, which may introduce noise into the supervision signal.

Frame Sampling Strategy We ablate the effect of our perturbed frame sampling strategy against linear sampling. As shown in Table 2, compared with video frame sampling without perturbation (w/ Linear Sampling), our perturbed sampling consistently yields higher accuracy and more stable performance across benchmarks. These improvements validate its effectiveness, and we adopt it as the default strategy for constructing multi-view training sequences.

## 5. Discussion

## 5.1. Regularization Strength & Prevention of Forgetting

As formulated in Eq. 1, the supervised anchor data acts as a regularization term. A key question is: what is the minimal regularization strength required to prevent catastrophic forgetting? To answer this, we vary the ratio between weakly-supervised video data $\mathcal { D } _ { v i d e o }$ and supervised anchor data $\mathcal { D } _ { s u p } .$ . As shown in Table 3, a 30 : 1 ratio achieves the optimal balance. Without sufficient regularization (e.g., ratio â â), the model drifts from the metric scale derived from ScanNet. Conversely, even a very sparse signal (100:1) significantly stabilizes the optimization, confirming that the foundation model requires only lightweight anchoring to maintain its priors while scaling up.

Table 3. Effect of Video-to-Supervised Data Ratio. Evaluation on four datasets with different video-to-supervised sample ratios. A 30:1 ratio is adopted as the default, balancing reconstruction quality with reliance on pretrained supervised data.
<table><tr><td colspan="3">ScanNet</td><td colspan="2">7Scenes</td><td>TUM</td><td colspan="2">MP3D</td></tr><tr><td>Ratio</td><td>DAcâ</td><td>CDâ</td><td>DAcâ</td><td>CDâ</td><td>DAcâ CDâ</td><td>DAcâ</td><td>CDâ</td></tr><tr><td>10:1</td><td>90.0</td><td>1.77</td><td>94.5</td><td>0.75</td><td>91.6 1.84</td><td>44.9</td><td>24.55</td></tr><tr><td>20:1</td><td>91.2</td><td>1.62</td><td>98.0</td><td>0.76</td><td>89.0 2.08</td><td>47.2</td><td>24.90</td></tr><tr><td>30:1</td><td>91.2</td><td>1.69</td><td>98.0</td><td>0.76</td><td>93.1 1.85</td><td>48.0</td><td>24.90</td></tr><tr><td>100:1</td><td>90.0</td><td>1.84</td><td>94.5</td><td>0.87</td><td>93.0 1.89</td><td>43.4</td><td>23.73</td></tr></table>

## 5.2. Beyond Scale: How Does the Data Difficulty Spectrum Shape Generalization?

We analyze the training data distribution in terms of sample difficulty, measured by the per-sample training loss of the pretrained model. As shown in the left panel of Figure 3, the Re10K dataset predominantly contains medium-difficulty samples, while DL3DV offers a higher proportion of challenging samples with elevated training losses. This observation aligns with our intuition, as Re10K features more structured indoor scenes, whereas DL3DV encompasses complex real-world environments with diverse lighting conditions and cluttered backgrounds. As shown in the right panel of Figure 3, we conduct a sensitivity analysis by varying the mixing ratio between medium-difficulty samples (Re10K) and high-difficulty samples (DL3DV) to investigate how the difficulty composition of training data influences crossdataset generalization. The pre-trained baseline (red dashed line) consistently exhibits the highest error across all benchmarks, underscoring the necessity of fine-tuning on diverse data. For relatively structured and simple scenes, such as 7Scenes, the model achieves its peak performance with a higher proportion of medium-difficulty data (e.g., the 1:0 ratio). In contrast, for large-scale and complex environments like MP3D and TUM, the inclusion of high-difficulty samples is crucial for pushing the performance boundary. As shown in the right panels of Figure 3, the Chamfer Distance (CD) significantly drops as the DL3DV ratio increases, reaching its optimum at the balanced 1:1 ratio. Beyond this point, however, we observe diminishing returns or slight performance fluctuations. These results suggest that a balanced curriculum of medium and high difficulty is essential for robust generalization.

## 5.3. Capability Gains from Video-based Adaptation

Pose Estimation We observe that the proposed training scheme with video data not only improves the quality of the reconstruction, but also improves the accuracy of the model pose estimation. Following MV-DUSt3R (Tang et al., 2024), we estimate camera poses using a standard RANSAC (FISCHLER AND, 1981) and PnP (Lepetit et al., 2009) pipeline, where 2Dâ3D correspondences are formed between image pixels and the predicted pointmaps generated by the network. In Table 4, our SAGE yields consistently lower rotation and translation errors across multiple datasets, demonstrating that video-based fine-tuning leads to more geometrically reliable outputs even without access to ground-truth annotations of video.

<!-- image-->  
Figure 3. Empirical analysis of training data difficulty and its impact on model generalization. (Left) Training distribution across three datasets. (Right) Generalization performance (CD â) on three test sets under various mixing ratios of Re10K to DL3DV samples. Subplots indicate that while pure medium-hardness data (1:0) benefits simple scenes, a balanced mixture (1:1) yields the most robust generalization on large-scale environments like MP3D. The dashed red lines denote the pre-trained baseline performance.

Table 4. Impact of Fine-tuning with Videos on Pose Estimation. Comparison between the baseline and our method on relative rotation error (RRE), relative translation error (RTE), and mean average error (mAE). Lower is better.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>7Scenes</td><td rowspan=1 colspan=1>TUM RGBD</td><td rowspan=1 colspan=1>MP3D</td></tr><tr><td rowspan=1 colspan=1>MV-DUSt3R</td><td rowspan=1 colspan=1>0.3 3.2 13.8</td><td rowspan=1 colspan=1>2.7 16.0 27.0</td><td rowspan=1 colspan=1>56.9 55.9 67.3</td></tr><tr><td rowspan=1 colspan=1>Ours (SAGE)</td><td rowspan=1 colspan=1>0.2 2.9 13.3</td><td rowspan=1 colspan=1>2.0 15.7 26.4</td><td rowspan=1 colspan=1>41.2 42.5 55.9</td></tr></table>

Evaluation on OOD Outdoor Scenes Although the training videos primarily consist of indoor scenes, they also include a portion of outdoor scenes. As a result shown in Table 5, ours SAGE generalizes surprisingly well to the out-ofdistribution outdoor UASOL dataset, achieving a DAc@0.5 of 27.8 and outperforming the baseline MV-DUSt3R (Tang et al., 2024). This demonstrates the advantage of leveraging diverse video data, where incidental exposure to outdoor environments contributes to improved robustness beyond the training distribution.

<!-- image-->  
Figure 4. Qualitative comparison of reconstructed point clouds across different methods, alongside ground-truth geometry. Zoomed-in views of representative regions are shown on the right. (Note: The 7Scenes ground-truth may appear slightly misaligned due to minor pose inaccuracies in the dataset.)

Table 5. Reconstruction performance on UASOL (out-ofdistribution, outdoor dataset). SAGE significantly outperforms MV-DUSt3R and the pseudo-supervised variant. Despite utilizing more video data, the pseudo-supervised model suffers from inconsistent and noisy multi-frame depth predictions, resulting in inferior 3D reconstruction accuracy.
<table><tr><td>Method</td><td>DAc-0.5 â</td><td>CD â</td></tr><tr><td>MV-DUSt3R</td><td>3.3</td><td>27.8</td></tr><tr><td>Ours (Video Depth Supervision)</td><td>26.0</td><td>31.8</td></tr><tr><td>Ours (SAGE)</td><td>27.8</td><td>18.8</td></tr></table>

## 6. Conclusion

We introduce SAGE, a framework that unlocks the scalability of geometric foundation models by exploiting the intrinsic consistency of Internet video. By formulating adaptation as a regularized weakly-supervised optimization, SAGE balances structural bias (from sparse anchors) and variance reduction (from dense differentiable consistency) to robustly learn from noisy data. Our experiments show log-linear scaling laws in geometric performance, achieving 20-42% improvements in zero-shot generalization. This work establishes a scalable paradigm for evolving general-purpose 3D models using unbounded video streams, turning the data scarcity bottleneck into a data abundance opportunity.

Limitation and Further Discussion Our method builds on a pretrained sparse 3D reconstruction model for static scenes and we primarily leverage Internet video sequences that capture largely static environments. Although we filter out clips with excessive motion to ensure reliable sparse 3D supervision, this strategy cannot fully exploit the vast scale and diversity of Internet video data, where dynamic content is prevalent. Extending our framework to pretrained models capable of handling both static and dynamic scenes represents a highly promising direction, with the potential to unlock the full value of large-scale Internet videos, enhance model generalization across diverse environments, and alleviate the longstanding scarcity and limited diversity of existing 3D training datasets.

## References

AanÃ¦s, H., Jensen, R. R., Vogiatzis, G., Tola, E., and Dahl, A. B. Large-scale data for multiple-view stereopsis. International Journal of Computer Vision, 120:153â168, 2016.

Bian, J., Li, Z., Wang, N., Zhan, H., Shen, C., Cheng, M.- M., and Reid, I. Unsupervised scale-consistent depth and ego-motion learning from monocular video. Advances in neural information processing systems, 32, 2019.

Chang, A., Dai, A., Funkhouser, T., Halber, M., Niessner, M., Savva, M., Song, S., Zeng, A., and Zhang, Y. Matterport3d: Learning from rgb-d data in indoor environments. arXiv preprint arXiv:1709.06158, 2017.

Charatan, D., Li, S. L., Tagliasacchi, A., and Sitzmann, V. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 19457â19467, 2024.

Chen, Y., Xu, H., Zheng, C., Zhuang, B., Pollefeys, M., Geiger, A., Cham, T.-J., and Cai, J. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European Conference on Computer Vision, pp. 370â386. Springer, 2024a.

Chen, Y., Zheng, C., Xu, H., Zhuang, B., Vedaldi, A., Cham, T.-J., and Cai, J. Mvsplat360: Feed-forward 360 scene synthesis from sparse views. arXiv preprint arXiv:2411.04924, 2024b.

Crandall, D. J., Owens, A., Snavely, N., and Huttenlocher, D. P. Sfm with mrfs: Discrete-continuous optimization for large-scale structure from motion. IEEE transactions on pattern analysis and machine intelligence, 35(12): 2841â2853, 2012.

Cui, H., Gao, X., Shen, S., and Hu, Z. Hsfm: Hybrid structure-from-motion. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1212â1221, 2017.

Dai, A., Chang, A. X., Savva, M., Halber, M., Funkhouser, T., and NieÃner, M. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5828â5839, 2017.

FISCHLER AND, M. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Commun. ACM, 24 (6):381â395, 1981.

Godard, C., Mac Aodha, O., Firman, M., and Brostow, G. J. Digging into self-supervised monocular depth estimation.

In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3828â3838, 2019.

Goesele, M., Curless, B., and Seitz, S. M. Multi-view stereo revisited. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPRâ06), volume 2, pp. 2402â2409. IEEE, 2006.

Kerbl, B., Kopanas, G., Leimkuhler, T., and Drettakis, G. 3d Â¨ gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023.

Lepetit, V., Moreno-Noguer, F., and Fua, P. Ep n p: An accurate o (n) solution to the p n p problem. International journal of computer vision, 81:155â166, 2009.

Leroy, V., Cabon, Y., and Revaud, J. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision, pp. 71â91. Springer, 2024.

Ling, L., Sheng, Y., Tu, Z., Zhao, W., Xin, C., Wan, K., Yu, L., Guo, Q., Yu, Z., Lu, Y., et al. Dl3dv-10k: A largescale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22160â22169, 2024.

Schonberger, J. L. and Frahm, J.-M. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4104â4113, 2016.

Seitz, S. M., Curless, B., Diebel, J., Scharstein, D., and Szeliski, R. A comparison and evaluation of multi-view stereo reconstruction algorithms. In 2006 IEEE computer society conference on computer vision and pattern recognition (CVPRâ06), volume 1, pp. 519â528. IEEE, 2006.

Shotton, J., Glocker, B., Zach, C., Izadi, S., Criminisi, A., and Fitzgibbon, A. Scene coordinate regression forests for camera relocalization in rgb-d images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2930â2937, 2013.

Smart, B., Zheng, C., Laina, I., and Prisacariu, V. A. Splatt3r: Zero-shot gaussian splatting from uncalibrated image pairs. arXiv preprint arXiv:2408.13912, 2024.

Snavely, N., Seitz, S. M., and Szeliski, R. Photo tourism: exploring photo collections in 3d. In ACM Siggraph, pp. 835â846, 2006.

Sturm, J., Engelhard, N., Endres, F., Burgard, W., and Cremers, D. A benchmark for the evaluation of rgb-d slam systems. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pp. 573â580. IEEE, 2012.

Szot, A., Clegg, A., Undersander, E., Wijmans, E., Zhao, Y., Turner, J., Maestre, N., Mukadam, M., Chaplot, D., Maksymets, O., Gokaslan, A., Vondrus, V., Dharur, S., Meier, F., Galuba, W., Chang, A., Kira, Z., Koltun, V., Malik, J., Savva, M., and Batra, D. Habitat 2.0: Training home assistants to rearrange their habitat. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Tang, Z., Fan, Y., Wang, D., Xu, H., Ranjan, R., Schwing, A., and Yan, Z. Mv-dust3r+: Single-stage scene reconstruction from sparse views in 2 seconds. arXiv preprint arXiv:2412.06974, 2024.

Wang, H. and Agapito, L. 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061, 2024.

Wang, J., Karaev, N., Rupprecht, C., and Novotny, D. Vggsfm: Visual geometry grounded deep structure from motion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 21686â 21697, 2024a.

Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., and Novotny, D. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5294â5306, 2025a.

Wang, Q., Zhang, Y., Holynski, A., Efros, A. A., and Kanazawa, A. Continuous 3d perception model with persistent state. arXiv preprint arXiv:2501.12387, 2025b.

Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., and Revaud, J. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20697â20709, 2024b.

Wang, Y., Zhou, J., Zhu, H., Chang, W., Zhou, Y., Li, Z., Chen, J., Pang, J., Shen, C., and He, T. Ï3: Scalable permutation-equivariant visual geometry learning, 2025c. URL https://arxiv.org/abs/2507.13347.

Weinzaepfel, P., Leroy, V., Lucas, T., Bregier, R., Cabon, Â´ Y., Arora, V., Antsfeld, L., Chidlovskii, B., Csurka, G., and Revaud, J. Croco: Self-supervised pre-training for 3d vision tasks by cross-view completion. Advances in Neural Information Processing Systems, 35:3502â3516, 2022a.

Weinzaepfel, P., Leroy, V., Lucas, T., Bregier, R., Cabon, Â´ Y., Arora, V., Antsfeld, L., Chidlovskii, B., Csurka, G., and Revaud, J. Croco: Self-supervised pre-training for 3d vision tasks by cross-view completion. Advances in Neural Information Processing Systems, 35:3502â3516, 2022b.

Weinzaepfel, P., Lucas, T., Leroy, V., Cabon, Y., Arora, V., Bregier, R., Csurka, G., Antsfeld, L., Chidlovskii, B., andÂ´ Revaud, J. Croco v2: Improved cross-view completion

pre-training for stereo matching and optical flow. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 17969â17980, 2023.

Wewer, C., Raj, K., Ilg, E., Schiele, B., and Lenssen, J. E. latentsplat: Autoencoding variational gaussians for fast generalizable 3d reconstruction. In European Conference on Computer Vision, pp. 456â473. Springer, 2024.

Xu, H., Peng, S., Wang, F., Blum, H., Barath, D., Geiger, A., and Pollefeys, M. Depthsplat: Connecting gaussian splatting and depth. arXiv preprint arXiv:2410.13862, 2024.

Yang, H., Huang, D., Yin, W., Shen, C., Liu, H., He, X., Lin, B., Ouyang, W., and He, T. Depth any video with scalable synthetic data. arXiv preprint arXiv:2410.10815, 2024.

Yang, J., Sax, A., Liang, K. J., Henaff, M., Tang, H., Cao, A., Chai, J., Meier, F., and Feiszli, M. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. arXiv preprint arXiv:2501.13928, 2025.

Yao, Y., Luo, Z., Li, S., Fang, T., and Quan, L. Mvsnet: Depth inference for unstructured multi-view stereo. In Proceedings of the European conference on computer vision (ECCV), pp. 767â783, 2018.

Zhou, T., Brown, M., Snavely, N., and Lowe, D. G. Unsupervised learning of depth and ego-motion from video. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1851â1858, 2017.

Zhou, T., Tucker, R., Flynn, J., Fyffe, G., and Snavely, N. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018.

## Appendix

## A. Preliminary

## A.1. Common Paradigm of 3D Geometric Foundation Models

Recent 3D geometric foundation models (3D GFMs), represented by DUSt3R (Wang et al., 2024b) and VGGT (Wang et al., 2025a), follow a largely unified pipeline that performs end-to-end geometric reasoning from RGB images, shown in Figure 5. These models first encode one or multiple input views into visual tokens using a shared image encoder, followed by cross-view feature interactionâtypically implemented via correspondence-aware attention or transformer-based aggregationâto capture multi-view geometric consistency. Geometry is then directly regressed in the form of pixel-aligned 3D point maps, depth, and/or camera parameters. DUSt3R formulates reconstruction as a pairwise image matching problem, which offers robustness under minimal view settings but limits scalability to long sequences, whereas VGGT extends this paradigm to explicit multi-view modeling with higher training resolution and stronger geometric supervision, achieving improved performance at increased computational cost. Despite architectural differences, most existing 3D GFMs rely heavily on large-scale, strongly supervised 3D datasets for pretraining, which constrains their scalability and motivates alternative adaptation strategies using weaker but more scalable supervision sources.

<!-- image-->  
Figure 5. Standard 3D GFM Training and Inference Framework. A generic 3D GFM consists of an image encoder, a decoder for cross-view feature interaction, and optional output heads that regress various geometric representations. These models are commonly trained with explicit 3D supervision.

## B. Experiment Details

## B.1. Video Data Processing Pipeline

This section details the video data processing pipeline used to construct the training set for weakly-supervised adaptation of 3D geometric foundation models. An overview of the workflow is illustrated in Fig. 6.

1) Video Acquisition. We collect raw videos from large-scale Internet video datasets, RealEstate10K and DL3DV contain minimal dynamic content, making them well-suited for fine-tuning pretrained 3D reconstruction models. These videos provide rich multi-view observations while avoiding reliance on curated 3D ground-truth geometry. 2) Video Filtering. To ensure sufficient geometric content and temporal coverage for reliable multi-view reconstruction, we further filter out low-quality video sequences, those are too short, exhibit large viewpoint discontinuities, or contain dynamic objects, as they may cause COLMAP reconstruction failures or produce excessively noisy point clouds: (1) minimum duration of 10 seconds, and (2) minimum spatial resolution of 480p. Videos that do not meet these requirements are discarded. This filtering step removes low-quality or short clips that are unlikely to yield stable camera poses or consistent multi-view correspondences. 3) Structure-from-Motion via COLMAP. For each retained video, we run a standard Structure-from-Motion (SfM) pipeline using COLMAP to recover sparse 3D geometry and camera parameters. Specifically, we first extract up to 16K local features per frame using GPU acceleration, which improves robustness under challenging conditions such as motion blur, illumination changes, and textureless regions. We then perform exhaustive feature matching across frames to maximize reliable correspondences and support wide-baseline reconstruction. Finally, COLMAP performs triangulation and bundle adjustment to estimate sparse 3D point clouds, camera extrinsic parameters, and camera intrinsic parameters. Only reconstructions that successfully converge are retained for subsequent training. 4) Output Representation. The final output of the pipeline consists of sparse 3D points with associated camera intrinsics and extrinsics, which serve as weak geometric supervision for model adaptation. Notably, no dense depth maps or ground-truth meshes are used at any stage of training. This design allows the proposed method to scale to large and diverse Internet video collections while maintaining compatibility with standard SfM tools.

<!-- image-->  
Figure 6. Video Data Processing Workflow for Weak Geometric Supervision. Raw Internet videos are filtered based on duration and resolution and processed using a COLMAP-based structure-from-motion pipeline. Feature extraction, exhaustive matching, and sparse reconstruction are performed to obtain sparse 3D points and camera parameters, which serve as weak geometric supervision for scalable model adaptation.

## C. Additional Experimental Results

## C.1. Generalization to Other 3D Foundation Model

To further verify the generality of our training framework, we apply SAGE to another multi-view feedforward 3D reconstruction model, VGGT (Wang et al., 2025a). Specifically, we fine-tune VGGT on a subset of RealEstate10K for approximately five hours on 4 GPUs, following the original training resolution of 518. As shown in Table 6, fine-tuning with SAGE reduces Chamfer Distance across all datasets, with particularly large improvements of up to 26% on 7Scenes and TUM-RGBD.

Table 6. Performance of VGGT before and after fine-tuning with SAGE on four 3D reconstruction benchmarks. Chamfer Distance (CD) decreases consistently, with particularly large improvements on 7Scenes and TUM-RGBD.
<table><tr><td rowspan="2">Method</td><td>ScanNet</td><td>7Scenes</td><td>TUM</td><td></td><td rowspan="2">MP3D</td></tr><tr><td>DAcâ CDâ</td><td>DAcâ CDâ</td><td>DAcâ</td><td>CDâ DAcâ</td></tr><tr><td>VGGT</td><td></td><td>99.0 0.65</td><td>80.0</td><td>3.92</td><td>CDâ 57.0 20.36</td></tr><tr><td>VGGT+SAGE</td><td>90.0 3.12 88.0</td><td>2.79 100.0 0.48</td><td>82.9</td><td>2.89</td><td>55.9 18.21</td></tr></table>

## C.2. Multi-view Reconstruction on ScanNet

In the main paper, we report 3D point cloud reconstruction results using 8 input views. Here, we provide additional results under varying numbers of input views. Table 7 shows the performance on the ScanNet test set across different view counts. As shown in Table 7, our method achieves the best overall reconstruction performance across most settings. However, as the number of input views increases, single forward inference becomes more challenging. We also report more results on MP3D (Chang et al., 2017) with varying number of inputs in Table 8. For multi-view reconstruction, Spann3R (Wang & Agapito, 2024) and DUSt3R (Wang et al., 2024b) are excluded due to high memory usage and optimization overhead, respectively. Only single forward inference methods are compared.

Table 7. Multi-view 3D reconstruction performance on ScanNet. Comparison across five ScanNet scenes with varying number of images (4 to 24). Each column reports Distance Accuracy (DAcâ) and Chamfer Distance (CDâ).
<table><tr><td rowspan="2">Method</td><td>4</td><td>12 16</td><td>20</td><td>24</td></tr><tr><td>DAc CD DAc CD</td><td>DAc CD DAc CD</td><td></td><td>DAc CD</td></tr><tr><td>VGGT</td><td></td><td>72.0 5.19 58.0 4.83 50.0 4.91 45.0 5.56 36.0 5.62</td><td></td><td></td></tr><tr><td>MVD Ours</td><td>88.0 1.67 86.0 2.63 82.0 2.43 80.0 2.95 73.0 2.49</td><td>90.0 1.71 84.0 2.31 82.0 1.98 82.0 2.41 72.0 2.34</td><td></td><td></td></tr></table>

Table 8. Multi-view 3D performance on MP3D. Evaluation across different number of images. Each column reports Distance Accuracy (DAc@0.5â) and Chamfer Distance (CDâ). All methods are evaluated without global alignment.
<table><tr><td rowspan="2">Method</td><td>4</td><td>12</td><td>16</td><td>20</td><td></td><td rowspan="2">24</td></tr><tr><td>DAc CD</td><td></td><td>DAc CD DAc CD</td><td></td><td>DAc CD DAc CD</td></tr><tr><td>VGGT</td><td></td><td></td><td></td><td></td><td>44.1 40.65 24.6 46.28 19.5 57.45 13.3 57.56 13.3 78.27</td><td></td></tr><tr><td>MVD</td><td></td><td></td><td></td><td></td><td>42.9 42.08 23.0 49.28 24.0 62.29 11.7 77.89 7.1 92.19</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>SAGE (Ours) 62.2 19.28 35.2 30.41 30.1 34.73 18.9 39.69 14.3 52.61</td><td></td></tr></table>

## C.3. Effect of Viewpoint Interpolation for Novel View Supervision

As discussed in the main paper, when constructing novel view supervision, we observe that interpolated target views (views that lie spatially between the input images) consistently yield better results than extrapolated ones. This is because extrapolated views tend to fall outside the effective coverage of the input images, making view consistency supervision more susceptible to ambiguity and error, which is also analyzed in Splatt3r (Smart et al., 2024). In contrast, interpolated views benefit from overlapping content on both sides, leading to more stable and effective training signals. In our experiments, we provide a comparative analysis of both strategies.

We ablate the effect of target view selection for novel-view supervision. As shown in Table 9, interpolated views consistently yield better reconstruction performance compared to extrapolated ones under the same training setup. Note that this experiment is conducted under a slightly different configuration from the main paper and serves primarily for comparative analysis.

Table 9. Comparison of novel-view supervision strategies. Performance across four datasets using extrapolated vs. interpolated target views. Interpolated targets consistently yield better reconstruction results.
<table><tr><td rowspan="2">Method</td><td>ScanNet</td><td></td><td>7Scenes</td><td>TUM</td><td>MP3D</td></tr><tr><td>DAc CD</td><td>DAc</td><td>CD</td><td>DAc CD</td><td>DAc CD</td></tr><tr><td>Ours (extra) 88.0 2.31</td><td></td><td></td><td>100.0 0.74</td><td>87.5 2.53</td><td>30.6 39.91</td></tr><tr><td>Ours (inter) 90.0 1.88</td><td></td><td></td><td>100.0 0.48</td><td>91.7 1.78</td><td>46.4 21.80</td></tr></table>

## C.4. Novel View Synthesis Quality Comparison

We evaluate the novel view synthesis performance of our method by rendering images from the predicted 3D Gaussians. Among the test datasets, we choose ScanNet (Dai et al., 2017) for this evaluation due to its high-quality imagery. In contrast, 7Scenes(Shotton et al., 2013) and TUM-RGBD (Sturm et al., 2012) often suffer from motion blur, and MP3D (Chang et al., 2017) lacks sufficient novel views with appropriate overlap, making them less suitable for consistent evaluation. To ensure fairness, we allow training of the Gaussian head in our method, similar to the baseline. As shown in Table 10, our method achieves slightly better perceptual quality than MV-DUSt3R in terms of PSNR, SSIM, and LPIPS. Although novel view synthesis is not our primary target, the improved rendering results further validate the quality of the reconstructed geometry.

Table 10. Novel view synthesis quality on the ScanNet dataset. Comparison between MV-DUSt3R and our method (UnVi3D) using three standard metrics: PSNR (â), SSIM (â), and LPIPS (â). Higher PSNR/SSIM and lower LPIPS indicate better perceptual quality.
<table><tr><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>MV-DUSt3R (Tang et al., 2024)</td><td>20.72</td><td>0.66</td><td>0.188</td></tr><tr><td>SAGE (Ours)</td><td>20.98</td><td>0.67</td><td>0.184</td></tr></table>

## C.5. The Impact of Scene Diversity vs. Sample Density

To investigate whether the generalization capability stems from the total volume of training samples or the diversity of the scenes, we conduct a controlled scaling experiment. We fix the total number of training iterations and the total sample budget to ensure a fair comparison. We then vary the number of unique training video scenes, ranging from 100 to 2,000. In this setup, as the number of scenes increases, the number of samples drawn from each individual video decreases proportionally. We evaluate the zero-shot generalization performance on the 7-Scenes and TUM datasets using the Chamfer Distance (CD) as the primary metric.

The results, illustrated in Figure 7, reveal a clear power-law relationship between training scene diversity and generalization error. On both the 7-Scenes and TUM datasets, we observe a consistent and significant reduction in Chamfer Distance as the number of training scenes increases. These findings underscore the necessity of leveraging large-scale, diverse internet video data. The diminishing returns of over-sampling from the same scene suggest that the model benefits more from exposure to novel geometric contexts and environmental variations, which are abundant in internet-scale datasets.

Crucially, since the total number of training iterations is kept constant, this performance gain is achieved by training on more diverse scenes with fewer samples per scene, rather than seeing more frames from a limited set of environments. This provides empirical evidence that scene-level diversity (the âbreadthâ of data) is a more critical factor for robust generalization than the temporal sampling density within a single video (the âdepthâ of data).

<!-- image-->

<!-- image-->  
Figure 7. Generalization performance under a fixed number of training iterations, as the diversity of training scenes increases from 100 to 2K video scenes, demonstrating consistent performance gains from broader scene coverage.

## D. Implementation Details

## D.1. Video Frame Sampling

When constructing training inputs, we follow the uniform sampling strategy described in the main text. For RealEstate10K (Zhou et al., 2018), we set the temporal interval $\Delta t = 2 0$ based on its typical recording frame rate, while for DL3DV (Ling et al., 2024), we use $\Delta t = 1 5$ due to its faster motion and denser frame sampling. For the perturbed sampling variant, we apply a random offset of up to Â±5 frames around each sampled timestamp to increase viewpoint diversity.

## D.2. Test Dataset Construction

Our test dataset construction follows the protocol proposed in CroCo (Weinzaepfel et al., 2022a) and MV-DUSt3R(Tang et al., 2024), but with key modifications. Unlike MV-DUSt3R, we do not use synthetic images rendered via Habitat-Sim (Szot et al., 2021), as such rendered data often suffers from limited photorealism and domain discrepancies compared to real-world images. Instead, for all test datasets, we begin by randomly selecting the first image in each test sequence. Subsequent views are sampled based on their geometric overlap with the previously selected views, which is computed using ground-truth depth. A predefined threshold is applied to determine whether a candidate frame provides sufficient overlap to be included in the test sequence.

## D.3. Training Process

The resolution of input images is 224 Ã 224. Each training sample consists of 8 input views, with either 2 or 8 novel views rendered for supervision. During training, we mix the video data from RealEstate10K and DL3DV. Additionally, we include approximately 3% (about 400) pretrained samples, resulting in a 30:1 ratio between video data and pretrained supervision data. To preserve the pretrained 3D Gaussian representation, we freeze the gaussian head and only optimize the reconstruction network. The model is trained for 5 epochs. In each epoch, we sample 15,000 training trajectories from each dataset. The entire training process is highly efficient, requiring less than two days on 4 RTX 4090 GPUs using only low-cost video data.

## E. Additional Qualitative Results

## E.1. Qualitative 3D Reconstrution Results

We provide additional qualitative reconstruction results in Figure 8, and further examples on more challenging scenes, low overlap or limited texture, in Figure 9.

## E.2. Qualitative Rendering Results

We visualize the rendering quality of the inferred 3D representation in Figure 10 and Figure 11. Given 8 input images as model input, the network performs a forward pass to infer a 3D Gaussian representation, which is then used to render images from both seen viewpoints and novel viewpoints. Specifically, the reconstructed representation is rendered to reproduce the original 8 input views, as well as 2 additional novel views for evaluating generalization to unseen viewpoints. Due to occlusions and limited field-of-view coverage in the input images, the novel-view renderings are naturally restricted to regions observed by the input views. Despite this limitation, the rendered images exhibit high visual fidelity, sharp geometric structures, and strong consistency with the ground-truth images, demonstrating that the learned 3D Gaussian representation effectively captures the underlying scene geometry and appearance. These qualitative results indicate that the proposed model can reconstruct accurate 3D scene representations from sparse multi-view inputs and generate high-quality renderings across viewpoints.

<!-- image-->  
Figure 8. Qualitative results of reconstructed point clouds with 8 input views across diverse scenes, shown alongside ground-truth geometry. All examples are drawn from unseen scenes and demonstrate the generalization ability of the model.

<!-- image-->  
Figure 9. Qualitative comparisons results of reconstructed point clouds views on challenging scenes, shown alongside ground-truth geometry. These examples feature low image overlap, texture-less surfaces (e.g., walls or floors), or inherently difficult matching conditions, and are drawn from previously unseen test scenes.

<!-- image-->  
Figure 10. Qualitative visualization of rendered images and ground-truth views. 18

<!-- image-->  
Figure 11. Qualitative visualization of rendered images and ground-truth views. 8 images are provided as model input, from which a 3D19 Gaussian representation is inferred and rendered to reconstruct the original 8 views as well as 2 novel viewpoints. Due to occlusions, novel-view renderings are limited to regions covered by the input views. The rendered images exhibit high visual fidelity and consistency with ground truth.