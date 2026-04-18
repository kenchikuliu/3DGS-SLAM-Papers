# Distilled-3DGS: Distilled 3D Gaussian Splatting

Lintao Xiang1,2â Xinkai Chen2â Jianhuang Lai3 Guangcong Wang2â 

1The University of Manchester,

2Vision, Graphics, and X Group, Great Bay University, 3Sun Yat-Sen University

<!-- image-->

<!-- image-->  
Figure 1. Compared with state-of-the-art 3DGS-based methods on Mip360 dataset, our Distilled-3DGS introduces a novel lightweight framework for high-quality view synthesis, achieving better detail preservation with lower storage.

## Abstract

3D Gaussian Splatting (3DGS) has exhibited remarkable efficacy in novel view synthesis (NVS). However, it suffers from a significant drawback: achieving high-fidelity rendering typically necessitates a large number of 3D Gaussians, resulting in substantial memory consumption and storage requirements. To address this challenge, we propose the first knowledge distillation framework for 3DGS, featuring various teacher models, including vanilla 3DGS, noiseaugmented variants, and dropout-regularized versions. The outputs of these teachers are aggregated to guide the optimization of a lightweight student model. To distill the hidden geometric structure, we propose a structural similarity loss to boost the consistency of spatial geometric distributions between the student and teacher model. Through comprehensive quantitative and qualitative evaluations across diverse datasets, the proposed Distilled-3DGSâa simple yet effective framework without bells and whistlesâachieves promising rendering results in both rendering quality and storage efficiency compared to state-of-theart methods. Project page: https://distilled3dgs.github.io. Code: https://github.com/lt-xiang/Distilled-3DGS.

## 1. Introduction

Novel view synthesis (NVS) is a fundamental task [40] in computer vision and computer graphics, serving as a cornerstone in many applications, e.g., VR/AR and autonomous driving. The goal of NVS is to generate photorealistic images from novel, previously unseen viewpoints. This process typically begins by constructing a 3D representation from a set of existing 2D observations. 3D Gaussian Splatting (3DGS) [13] has recently demonstrated remarkable effectiveness in novel view synthesis. This approach employs a point-based representation augmented with 3D Gaussian attributes and utilizes a rasterization-based rendering pipeline to synthesize images. However, 3DGS necessitates a large number of 3D Gaussians to ensure high-fidelity image rendering, particularly in the presence of complex scenes. This limits their applicability on platforms and devices with constrained computational resources and limited memory capacity.

On the other hand, knowledge distillation [3] has proven effective in compressing neural networks across various vision tasks. However, applying it to 3D Gaussian Splatting (3DGS) introduces unique challenges. First, 3DGS is an explicit and unstructured representation composed of variable 3D Gaussians, lacking the consistent latent feature spaces typically leveraged in conventional KD. Second, the Gaussian primitives are scene-dependent and unordered, preventing straightforward correspondence between teacher and student components. Third, since rendering outputs are view-dependent and non-differentiable w.r.t. individual Gaussians, designing stable and informative distillation losses becomes nontrivial. These challenges necessitate careful design of both teacher model ensembles and geometry-aware distillation strategies, as introduced in our Distilled-3DGS framework.

To address this challenge, we propose a lightweight 3D Gaussian representation framework based on knowledge distillation, termed Distilled-3DGS. This approach enhances the performance of a compact student model by distilling knowledge from multiple complex teacher models. The overall pipeline of Distilled-3DGS comprises two main stages: multi-teacher training and student training via distillation. In the multi-teacher training stage, we begin by training a standard 3DGS model. Subsequently, we introduce random perturbations and dropout strategies separately to obtain two additional diverse teacher models. During the distillation-based student training stage, we first aggregate predictions from the teacher ensemble to synthesize a pseudo image. The student model is then supervised by enforcing similarity between its rendered output and this pseudo image. This strategy effectively transfers rich knowledge priors from the teacher models, providing a more comprehensive and robust supervisory signal for optimizing the student model.

In the context of Distilled-3DGS, the teacher model typically contains dense and high-quality point clouds, while the student model is compressed to obtain much sparser points due to efficiency or deployment considerations. Despite its sparsity, the student model is trained to reconstruct the same underlying 3D scene as the teacher. So it is expected to preserve the essential spatial layout and local geometric patterns present in the teacher model. To facilitate this, we propose a spatial distribution distillation strategy that guides the student to align with the teacherâs point distribution in space. Rather than enforcing exact point-wise correspondence, this structure-aware supervision encourages the student to learn how the teacher organizes points, focusing on global and local geometric consistency. In this way, structural knowledge from the teacher can be effectively and comprehensively distilled into the student model.

In summary, our main contributions are as follows: 1) We propose a novel distillation-based 3DGS framework, termed as Distilled-3DGS, which is the first method to leverage multi-teacher knowledge priors to optimize 3DGS and boost rendering quality and storage efficiency. 2) We propose a spatial distribution consistency distillation to enable the student model to learn similar geometric structure distributions from the teacher model. 3) Extensive experiments on several real-world datasetsâincluding Mip-NeRF 360, Tanks&Temples, and Deep Blendingâdemonstrate that the proposed Distilled-3DGS achieves promising performance in both rendering quality and efficiency compared to existing methods.

## 2. Related Work

3D Representation. Radiance fields have been extensively employed for 3D scene reconstruction, particularly in the context of novel view synthesis. Neural Radiance Fields (NeRFs) have achieved remarkable progress by learning neural volumetric representations of 3D scenes, enabling high-fidelity image synthesis via volumetric rendering techniques. After that, many works have focused on improving the rendering quality [1, 2] and accelerating the efficiency [6, 11, 31, 39] of NeRFs. Nevertheless, NeRF-based approaches continue to rely on numerous MLP queries during rendering, thereby limiting their applicability in scenarios with real-time constraints. To enhance the training and rendering efficiency, Plenoxels [6] improve NeRF efficiency by optimizing a sparse voxel grid and removing the need for MLPs, while Instant NGP [39] uses hash-grid encodings to boost expressivity. However, despite these improvements, grid-based methods still struggle to achieve real-time rendering. Recently, 3D Gaussian Splatting (3DGS) [13] has gained significant attention as an efficient and effective approach for 3D scene representation. 3DGS represents 3D scenes explicitly with millions of anisotropic Gaussians and utilizes differentiable rasterization, enabling real-time, photorealistic view synthesis. When 3DGS overfits the scene by optimizing Gaussian properties, it typically produces many redundant Gaussians, thereby reducing rendering efficiency and substantially increasing memory usage.

To tackle these issues, several subsequent approaches have aimed to prune redundant Gaussians based on handcrafted importance criteria. Mini-Splatting [5] addresses overlapping and reconstruction artifacts by employing blur splitting, depth reinitialization, and stochastic sampling. Radsplatting [24] enhances robustness by applying a max operator to derive importance scores from ray contributions. Taming-3DGS [21] leverages pixel saliency and gradient information for selective densification, while LP-3DGS [38] utilizes a learned binary mask for efficient Gaussian pruning. Additionally, Scaffold-GS [20] proposes a structured dual-layer hierarchical scene representation to better regulate the distribution of 3D Gaussian primitives.

Overall, the aforementioned methods that prioritize efficiency generally achieve faster performance, but this often comes at the expense of rendering quality compared to the standard 3DGS. Conversely, approaches that focus on enhancing rendering quality tend to demand substantially higher computational resources. To address this trade-off, we propose a knowledge distillation-based 3DGS framework that simultaneously improves storage efficiency and rendering fidelity.

Knowledge Distillation.Knowledge distillation (KD)

transfers knowledge from a large teacher model to a compact student model. Initially proposed for model compression [3, 10], KD began with matching teacher outputs and was later extended to mimic intermediate representations [27, 32]. KD has since been applied to various tasks, including detection [18, 35],segmentation [12, 19], and generation [17, 36]. To overcome the limitations of singleteacher KD, multi-teacher distillation (MKD) is proposed to aggregate diverse knowledge from multiple teachers. While early approaches assign equal weights [7, 30], recent methods adopt adaptive strategies, such as entropy-based weighting (EB-KD [15]) and confidence-aware distillation (CA-MKD [33]). MMKD [34] further introduces metalearning to jointly distill features and logits. These methods often rely on CNNs for structured feature spaces, facilitating effective alignment via soft labels or intermediate supervision.

However, extending KD to 3D Gaussian Splatting (3DGS) poses new challenges. 3DGS uses an explicit and unstructured representation composed of a variable set of discrete Gaussian primitives. These primitives are unordered, scene-dependent, and lack a shared latent space, making it infeasible to directly align elements between teacher and student. As a result, existing KD strategies must be fundamentally rethought to accommodate the unique properties of 3DGS. Based on the above analysis, we propose to utilize multiple pre-trained 3DGS teacher models to render high-quality images as supervision targets and optimize the student model. Besides, we propose a spatial distribution distillation strategy that guides the student to align with the teacherâs point distribution in space.

## 3. Method

In this section, we present Distilled-3DGS, an efficient 3D Gaussian Splatting framework that distills knowledge from powerful teacher models to a small student model.

## 3.1. Preliminaries

3DGS [13] is a cutting-edge method for novel view synthesis, which fundamentally depends on an explicit pointbased representation to achieve high-fidelity rendering from arbitrary viewpoints. Specifically, 3DGS models a scene as a set of Gaussian distributions. The $i _ { t h }$ Gaussian primitive is denoted as $G _ { i } = ( \mu _ { i } , \Sigma _ { i } , o _ { i } , f _ { i } )$ , where $\mu _ { i }$ is the 3D position, $\Sigma _ { i }$ is the covariance matrix, $f _ { i }$ represents spherical harmonics (SH) coefficients associated with the Gaussian, and $o _ { i }$ indicates opacity. The effect of the $i _ { t h }$ Gaussian primitive at position x is represented by $G _ { i } ( x ) \ =$ $e ^ { - \frac { 1 } { 2 } \left( x - \mu _ { i } \right) ^ { T } \Sigma _ { i } ^ { - 1 } \left( x - \mu _ { i } \right) }$ , where $\Sigma _ { i }$ can be factorized as $\Sigma _ { i } =$ $R S S ^ { T } R ^ { T }$ , with R as a rotation matrix and S as a scaling matrix, both being learnable. Subsequently, the Gaussians are mapped to the 2D image plane via the projection matrix W , resulting in the projected 2D covariance matrix $\Sigma _ { i } ^ { ' } = J W \Sigma _ { i } W ^ { T } J ^ { T }$ , where J represents the Jacobian of the affine projection. The pixel color is computed through alpha blending as follows:

$$
\mathbf { c } = \sum _ { i = 1 } ^ { N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{1}
$$

where N is the number of Gaussians covering the pixel, the color $c _ { i }$ is derived from the spherical harmonics (SH) coefficients of each Gaussian, while $\alpha _ { i }$ is determined by the projected 2D covariance matrices $\Sigma ^ { ' }$ and the associated opacity $o _ { i } .$ . The Gaussian parameters are optimized using a photometric loss [13] function, with the posed training images providing the ground-truth supervision.

## 3.2. Distilled 3D Gaussian Splatting

3DGS has enabled highly detailed and accurate 3D scene reconstruction, yet such state-of-the-art models are often extremely large and computationally expensive, limiting their practicality in real-time and resource-constrained scenarios. Knowledge distillation (KD) has emerged as a highly effective and popular approach for model compression in image classification, semantic segmentation [19], and object detection [4]. Inspired by these observations, one could ask if knowledge distillation works for 3DGS. Different from conventional KD in neural networks, it is an explicit 3D representation with variable unstructured 3D Gaussians, which remains unexplored. To address this problem, we first provide an overview of the proposed Distilled-3DGS, and then detail the design of diverse teacher models and the distillation method, as discussed in the following.

## 3.3. Overview of Distill-3DGS

As shown in Fig.2, we firstly train three independent 3DGS models with diverse strategies to obtain cumbersome yet high-quality teacher models with millions of Gaussian primitives, optimized by the standard photometric loss. Then we leverage the optimized teacher 3DGS representation to generate pseudo labels by fusing multiple teachersâ outputs. In the training process of the student model, pseudo labels are leveraged to transfer prior knowledge from multiple teachers to a single student. To obtain a lightweight student model, we prune the number of Gaussians based on the importance score proposed in Mini-Splatting [5]. To distill knowledge hidden in geometric structure of 3D Gaussians, we propose structural knowledge distillation for unstructured 3D Gaussians to encourage the similar spatial distribution between teacher and student models.

## 3.4. Training Diverse Teacher models

To provide the student model with richer supervision signals and facilitate a better understanding of 3D scene structures and details, we train the base 3DGS model multiple

<!-- image-->  
Figure 2. The architecture of multi-teacher knowledge distillation framework for 3DGS. It consists of two stages. First, a standard teacher model $G _ { s t d }$ is trained, along with two variants: $G _ { p e r b }$ with random perturbation and $G _ { d r o p }$ with random dropout. Then, a pruned student model $G _ { s t d }$ is supervised by the outputs of these teachers. Additionally, a spatial distribution distillation strategy is introduced to help the student learn structural patterns from the teachers.

times using diverse strategies to enhance the robustness and generalization ability of the teacher models.

Regular training. First, we train a vanilla 3DGS model $G _ { s t d }$ with the same settings in [13]. The training loss is defined as:

$$
\mathcal { L } _ { \mathrm { c o l o r } } = ( 1 - \lambda ) \mathcal { L } _ { 1 } ( \hat { I } , I _ { g t } ) + \lambda \mathcal { L } _ { \mathrm { D - S S I M } } ( \hat { I } , I _ { g t } )\tag{2}
$$

Feature Perturbation. Then, we train a 3DGS model $G _ { p e r b }$ with random perturbations on Gaussian parameters, following the same optimization and density control strategies introduced in 3DGS. At each training iteration t, each Gaussian is perturbed as:

$$
G _ { p e r b } ^ { t } = G _ { s t d } ^ { t } + \delta _ { t } ,\tag{3}
$$

where random noises are added to corresponding Gaussian parameters including 3D positions $\mu _ { p } .$ , 3D rotations $R _ { p } ,$ scales $S _ { p } ,$ , and opacities $o _ { p }$ . Since the representation of rotation as a 3D matrix is discontinuous, we instead perturb its continuous 6D representation as :

$$
\hat { \mathbf { R } } _ { p } ^ { t } = f ^ { - 1 } \left( f ( \mathbf { R } _ { p } ^ { t } ) + \delta _ { t } ^ { R } \right) , \quad \delta _ { p } ^ { t } \in \mathbb { R } ^ { 6 }\tag{4}
$$

where $f$ and $f ^ { - 1 }$ are the forward and inverse mappings between the rotation matrix and its 6D representation. By introducing parameter perturbations during training, the model is compelled to learn scene structures that are less dependent on the precise positions and shapes of Gaussian primitives, thereby enhancing its generalization capability.

Random Dropout. Dropout [8, 26] is recognized as one of the most effective techniques for improving model robustness by randomly deactivating a subset of neurons during training. Inspired by the remarkable success of dropout, we introduce a Random Dropout Strategy to further enhance both the robustness and redundancy of the representation of model $G _ { d r o p } .$ Specifically, during training, each Gaussian primitive is randomly deactivated with probability p, while the remaining primitives are optimized to fit the observed views. During inference, all Gaussian primitives are activated to facilitate novel view synthesis. By randomly dropping a subset of Gaussian primitives during training, our approach encourages the model to learn a collaborative and distributed scene representation, rather than relying on a limited set of critical Gaussian primitives. Inspired by [25], the dropping rate $r _ { t }$ is updated based on the current iteration index t as follows:

$$
r _ { t } = r _ { i n i t } \cdot ( t - t _ { 0 } ) / ( t _ { 1 } - t _ { 0 } ) ,\tag{5}
$$

where $t _ { 0 }$ and $t _ { 1 }$ are the starting and end iterations of in-

troducing random dropout strategy. $r _ { i n i t }$ is the initial drop rate.

## 3.5. Training Efficient Student Model

Knowledge distillation is a technique that transfers knowledge from a larger teacher model to a smaller, faster student model. This approach is particularly useful when deploying deep neural networks in resource-constrained environments. The student model, trained under the guidance of the teacher, can achieve comparable performance with significantly fewer parameters. This process mainly consists of pseudo label generation and student training.

Pseudo labeling with teacher model. By evaluating the optimized diverse teachers $G _ { s t d } , G _ { p e r b }$ and $G _ { d r o p } ,$ we can render per-view image denoted as Istd, $I _ { p e r b }$ and $I _ { d r o p } ,$ these rendered images as prior knowledge are further aggregated by average strategy to generate pseudo label $I _ { t e a }$ and guide the learning of the student model.

Conventional Knowledge Distillation. In the training process of the student network, we utilize the ground-truth labels and the pseudo label of multiple teachers as additional knowledge to jointly guide the optimization of student model $G _ { s t u }$ . Following the conventional knowledge distillation loss, we formulate our objective by incorporating fused knowledge from multiple teachers, as follows:

$$
\mathcal { L } _ { \mathrm { k d } } = \mathcal { L } _ { c o l o r } ( I _ { s t u } , I _ { g t } ) + \lambda _ { k d } \mathcal { L } _ { c o l o r } ( I _ { s t u } , I _ { t e a } )\tag{6}
$$

Spatial Distribution Distillation. In the context of 3DGS, these optimized teacher models provide a structure-rich and dense 3D point distribution. In contrast, the student model operates under sparse or limited sampling conditions and aims to reconstruct the similar scene representation. Therefore, we hope to design a structural similarity loss to encourage the student model to capture spatial geometric distributions similar to those of the teacher. However, challenges arise due to varying point densities, sampling noise, and non-uniform point distributions between student and teacher models. Direct coordinate-based distance measures are often insufficiently robust to these variations.

To address this problem, we leverage the voxel histogram representation shown in Fig. 3, which discretizes the 3D space into regular voxels and counts the number of points within each voxel. This approach encodes the spatial distribution of points as a high-dimensional structural feature, inherently robust to point count and density variations. Comparing voxel histograms thus enables efficient and structureaware similarity evaluation between different point clouds. To this end, we propose a voxel histogram-based structural loss to enhance the structural learning capability of the student model.

During the training phase of the student model, we firstly obtain point cloud $P _ { t e a }$ and $P _ { s t u }$ from the optimized standard teacher model $G _ { s t d }$ and student model $G _ { s t u }$ . Then, we determine a common 3D bounding box that encompasses both sets of points.The bounding box is partitioned into a regular voxel grid with a resolution of 128 . Each point from both clouds is assigned to a corresponding voxel cell based on its spatial coordinates. For point cloud $P _ { t e a }$ and $P _ { s t u }$ , we count the number of points falling into every voxel separately, resulting in two high-dimensional voxel occupancy histograms $\mathbf { h } _ { t e a }$ and $\mathbf { h } _ { s t u }$ . These histograms are then normalized to form probability distributions that capture the spatial structure of each cloud, independent of point count or density. Finally, we compute the cosine similarity between their normalized voxel occupancy histograms. The cosine similarity loss is given by:

<!-- image-->  
Figure 3. Overview of Spatial Distribution Distillation.

$$
\mathcal { L } _ { \mathrm { h i s t } } = 1 - \frac { \mathbf { h } _ { t e a } \cdot \mathbf { h } _ { s t u } } { \Vert \mathbf { h } _ { t e a } \Vert _ { 2 } \Vert \mathbf { h } _ { s t u } \Vert _ { 2 } }\tag{7}
$$

This loss quantitatively reflects how closely the Student point cloud matches the Teacherâs structural distribution. The final loss function during the student training phase is defined as:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { k d } } + \mathcal { L } _ { \mathrm { h i s t } }\tag{8}
$$

## 4. Experiments

Datasets. We conducted experiments on three widely used datasets: LLFF [22], Mip360 [2] and two scenes from the Tanks and Temples(T&T) [14]. LLFF contains eight scenes with forward-facing camera. Mip-NeRF360 comprises nine distinct scenes that encompass both expansive outdoor scenes and intricate indoor settings. These scenes exhibit a wide range of capture styles and encompass both bounded indoor environments as well as large, unbounded outdoor settings. To partition the dataset into training and test sets, we follow the protocol of 3DGS by allocating every eight image to the test set. The resolution of all images is kept consistent with that used in 3DGS.

GT  
Ours  
Taming 3DGS  
<!-- image-->  
Mini-splatting  
3DGS

Figure 4. Visualized comparison on the Bicycle, Garden, and Kitchen scenes. As shown in the rendered images and corresponding local regions, the proposed method can better capture fine details.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="4">Mip-NeRF 360</td><td colspan="4">Tanks &amp; Temples</td><td colspan="4">Deep Blending</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>#G(10Â°)â</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>#G(10Â°)</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>#G(10Â°)â</td></tr><tr><td rowspan="7">Quality</td><td>Plenoxels (CVPR&#x27;22)</td><td>23.08</td><td>0.626</td><td>0.463</td><td></td><td>21.08</td><td>0.719</td><td>0.379</td><td>-</td><td>23.06</td><td>0.795</td><td>0.510</td><td>-</td></tr><tr><td>INGP-Big (SIGGRAPH&#x27;22)</td><td>25.59</td><td>0.699</td><td>0.331</td><td>-</td><td>21.92</td><td>0.745</td><td>0.305</td><td>-</td><td>24.96</td><td>0.817</td><td>0.390</td><td>-</td></tr><tr><td>Mip-NeRF360 (CVPR&#x27;22)</td><td>27.69</td><td>0.792</td><td>0.237</td><td>-</td><td>22.22</td><td>0.759</td><td>0.257</td><td>-</td><td>29.40</td><td>0.901</td><td>0.245</td><td>-</td></tr><tr><td>3D-GS (TOG&#x27;23)</td><td>27.26</td><td>0.815</td><td>0.214</td><td>3.5</td><td>23.14</td><td>0.841</td><td>0.183</td><td>2.0</td><td>29.41</td><td>0.903</td><td>0.243</td><td>3.2</td></tr><tr><td>3D-GS*</td><td>27.39</td><td>0.819</td><td>0.219</td><td>3.43</td><td>23.61</td><td>0.849</td><td>0.180</td><td>1.84</td><td>29.55</td><td>0.912</td><td>0.241</td><td>3.24</td></tr><tr><td>ScaffoldGS (CVPR&#x27;24)</td><td>27.60</td><td>0.812</td><td>0.222</td><td>0.6</td><td>24.08</td><td>0.854</td><td>0.165</td><td>0.6</td><td>30.25</td><td>0.907</td><td>0.245</td><td>0.40</td></tr><tr><td>CompactGaussian (CVPR&#x27;24)</td><td>27.08</td><td>0.798</td><td>0.247</td><td>1.388</td><td>23.32</td><td>0.831</td><td>0.201</td><td>0.836</td><td>29.79</td><td>0.901</td><td>0.258</td><td>1.06</td></tr><tr><td rowspan="7">Efficiency</td><td>LP-3DGS (NIPS&#x27;24)</td><td>27.47</td><td>0.812</td><td>0.227</td><td>1.959</td><td>23.60</td><td>0.842</td><td>0.188</td><td>1.244</td><td></td><td></td><td></td><td>-</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.836</td><td></td><td></td><td>-</td><td>-</td><td>0.253</td><td>0.40</td></tr><tr><td>MiniSplatting (CVPR&#x27;24) EAGLES(ECCV&#x27;24)</td><td>27.25 27.20</td><td>0.820</td><td>0.217</td><td>0.5</td><td>23.21 23.26</td><td>0.837</td><td>0.203</td><td>0.32 0.7</td><td>29.98 9.86</td><td>0.908 0.910</td><td>0.246</td><td>1.20</td></tr><tr><td></td><td>27.71</td><td>0.809 0.820</td><td>0.232 0.207</td><td>1.3 0.63</td><td>23.95</td><td>0.837</td><td>0.201 0.201</td><td></td><td>29.82</td><td>0.904</td><td>0.237</td><td>0.27</td></tr><tr><td>Taming 3DGS(SIGGRAPH Asia&#x27;24)</td><td>27.12</td><td>0.806</td><td>0.240</td><td>0.84</td><td>23.44</td><td>0.838</td><td>0.198</td><td>0.29 0.52</td><td>29.90</td><td>0.907</td><td>0.251</td><td></td></tr><tr><td>CompGS (ECCV&#x27;24) Ours</td><td>27.81</td><td>0.827</td><td>0.202</td><td>0.49</td><td>23.76</td><td>0.845</td><td>0.179</td><td>0.25</td><td>29.87</td><td>00.916</td><td>0.251</td><td>0.55</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.33</td></tr></table>

Table 1. Quantitative evaluations across the Mip-NeRF 360, Tanks&Temples, and Deep Blending datasets. Best and second-best results are highlighted for each. \* denotes our re-runs of the existing codebase to ensure a fair evaluation.

Evaluation Metrics. For the evaluation of comparative view synthesis quality, we adopt several widely used quantitative metrics, including Peak Signal-to-Noise Ratio (PSNR), Learned Perceptual Image Patch Similarity (LPIPS)[37], and Structural Similarity Index Measure (SSIM)[28]. PSNR and SSIM primarily assess pixellevel fidelity and structural consistency, respectively, while LPIPS reflects a more human-aligned assessment of visual quality. In addition, we present the average memory usage associated with storing the optimized parameters.

Implementations. Our implementations are based on the official 3DGS codebase. All models were trained on a single NVIDIA RTX 3090 GPU. Training process of Distilled-3DGS contains two stages: In the training phase of teacher models, these teacher models are trained for 30k iterations by following 3DGS. Gaussian densification is stopped after the 15000th iteration as in 3DGS. For the teacher training with random perturbation, random noise $\delta _ { t }$ is applied to those Gaussian primitives exhibiting large view-space positional gradients, as these typically correspond to regions that have not yet been well reconstructed, starting from 500th to 15000th iteration with interval 500. Introducing appropriate perturbations in this manner can enhance the robustness of the model. For the teacher training with random dropout, $r _ { i n i t } , \ t _ { 0 }$ and $t _ { 1 }$ are respectively set 0.2, 500 and 15000. Each Gaussian primitive is randomly deactivated with probability $p ,$ and the remaining primitives are optimized to fit the observed views. For inference, all Gaussian primitives are activated to enable novel view synthesis.

In the training phase of student model, the total number of optimization steps is set to 30K. Densification is applied up to the 15000th iteration, after which simplification is carried out at both the 15000th iterations. Subsequently, the structural similarity loss is computed and applied every 500 iterations throughout the optimization process.

## 4.1. Comparisons with State-of-the-Arts

We evaluate model performance across several real-world datasets, including Mip-NeRF 360, Tanks&Temples, and Deep Blending. For NeRF-based methods, we compare with the state-of-the-art Mip-NeRF 360 [2] and two efficient NeRF variants, INGP [23] and Plenoxels [6]. For 3DGSbased methods, our comparisons include the vanilla 3DGS as well as leading Gaussian simplification techniques such as CompactGaussian [16], LP-3DGS [38], EAGLES [9], MiniSplatting [5], and Taming 3DGS [21]. For vanilla 3D-GS, we include both the metrics reported in [13] and those obtained through our own experimental runs. The quantitative results for all datasets are presented in Table 1. Our method surpasses both the voxel grid-based approach, Plenoxels, and the fast NeRF-based method, INGP, across all datasets and evaluation metrics. Compared to the Mip-NeRF360 baseline, Distilled-3DGS yields PSNR improvements of 0.12 dB, 1.54 dB, and 0.47 dB on the Mip-NeRF360, Tanks&Temples, and Deep Blending datasets, respectively, verifying its effectiveness across diverse datasets.

Compared to other 3DGS-based methods, our proposed Distilled-3DGS consistently outperforms the baseline 3DGS across all three metrics while using significantly fewer Gaussians on Mip360 dataset. We attribute this improvement to the comprehensive knowledge supervision provided by the diverse teacher models. Specifically, compared to the vanilla 3DGS, our Distilled-3DGS achieves PSNR improvements of 0.55 dB on Mip-NeRF360, 0.62 dB on Tanks&Temples, and 0.46 dB on Deep Blending. The number of Gaussians is also reduced by 86.0%, 87.5%, and 89.6% on these three datasets, respectively. Compared with these 3DGS simplification methods, such as Taming 3DGS, our method also can improve rendering quality while maintaining a comparable number of Gaussians. Besides, visual comparison is illustrated in Fig. 4, compared with existing simplification approaches such as Taming-3DGS, Mini-Splatting, and the vanilla 3DGS, our Distilled-3DGS achieves rendering results that preserve fine details most faithfully to the ground truth while utilizing a significantly reduced number of Gaussian primitives.

## 4.2. Ablation Studies and Further Analyses

To study the contribution of each component in the proposed framework, we conducted a series of ablation experiments on the Deep Blending and Tanks&Temples datasets. Effect of the number of teachers. To further understand the contribution of each teacher model in our distillation framework, we conducted ablation studies by gradually removing the Perturbation-based, Dropout-based 3DGS teachers, respectively. The quantitative results are shown in Table 2. Specifically, the student model distilled from all three teachers consistently achieves the best performance, indicating that each teacher provides complementary knowledge. Teacher $G _ { p e r b }$ enhances the studentâs robustness to input variations, while teacher $G _ { d r o p }$ prevents overfitting and encourages generalization. The regular 3DGS teacher $G _ { s t d }$ serves as a strong baseline, providing high-fidelity supervision. The progressive decrease in performance with the removal of these specialized teachers underscores their critical roles in enriching the distilled knowledge, validating the effectiveness of leveraging diverse teacher models for optimal student performance.

<table><tr><td rowspan="2">Method</td><td colspan="3">Deep Blending</td><td colspan="3">Tanks&amp;Temples</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>Ours</td><td>29.87</td><td>0.916</td><td>0.251</td><td>23.76</td><td>0.845</td><td>0.179</td></tr><tr><td>Without  $G _ { d r o p }$ </td><td>29.71</td><td>0.899</td><td>0.257</td><td>23.58</td><td>0.840</td><td>0.186</td></tr><tr><td>Without  $G _ { p e r b }$ </td><td>29.63</td><td>0.878</td><td>0.262</td><td>23.43</td><td>0.838</td><td>0.195</td></tr><tr><td>Without  ${ \mathcal { L } } _ { \mathrm { h i s t } }$ </td><td>29.47</td><td>0.871</td><td>0.263</td><td>23.32</td><td>0.836</td><td>0.197</td></tr></table>

Table 2. Ablation study on two datasets.
<table><tr><td>Grid_size</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Train Mem.(MB)</td></tr><tr><td>32</td><td>27.51</td><td>0.819</td><td>0.198</td><td>9564</td></tr><tr><td>64</td><td>27.62</td><td>0.821</td><td>0.199</td><td>10232</td></tr><tr><td>128</td><td>27.81</td><td>0.827</td><td>0.202</td><td>12235</td></tr><tr><td>256</td><td>27.92</td><td>0.829</td><td>0.203</td><td>15456</td></tr></table>

Table 3. The impact of different grid size in spatial distribution distillation.
<table><tr><td rowspan="2">Method</td><td colspan="4">Room (Mip360)</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td> $\# \mathrm { G } ( 1 0 ^ { 6 } ) \downarrow$ </td></tr><tr><td>Teachers  $( G _ { s t d } \mathbf { + } G _ { d r o p } \mathbf { + } G _ { p e r b } )$ </td><td>32.15</td><td>0.935</td><td>0.185</td><td>1.56</td></tr><tr><td>3DGS</td><td>31.59</td><td>0.920</td><td>0.200</td><td>1.50</td></tr><tr><td>Student(Base)</td><td>31.54</td><td>0.927</td><td>0.193</td><td>0.46</td></tr><tr><td>Student(Big)</td><td>31.89</td><td>0.934</td><td>0.189</td><td>1.13</td></tr><tr><td>Student(Small)</td><td>31.39</td><td>0.923</td><td>0.194</td><td>0.21</td></tr></table>

Table 4. The impact of the number of Gaussians.
<table><tr><td rowspan="2">Teacher</td><td rowspan="2">Student</td><td colspan="4">Room (Mip360)</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPS</td><td>#G(106)â</td></tr><tr><td> $\overline { { G _ { s t d } / G _ { p e r b } / G _ { d r o p } } }$ </td><td rowspan="5"> $G _ { s t u }$ </td><td>31.54</td><td>0.927</td><td>0.193</td><td>0.460</td></tr><tr><td> $G _ { s t d } / G _ { s t d } / G _ { s t d }$ </td><td>31.36</td><td>0.923</td><td>0.191</td><td>0.469</td></tr><tr><td> $G _ { s t d }$ </td><td>31.19</td><td>0.918</td><td>0.187</td><td>0.465</td></tr><tr><td> $G _ { p e r b }$ </td><td>31.31</td><td>0.921</td><td>0.186</td><td>0.453</td></tr><tr><td> $\underbrace { G _ { d r o p } } _ { \phantom { \left( G _ { d r o p } \right. } }$ </td><td>31.23</td><td>0.919</td><td>0.189</td><td>0.459</td></tr></table>

Table 5. The impact of different teachers

Effect of Spatial Distribution Distillation. The results presented in Table 2 verify that spatial distribution distillation plays a crucial role in enhancing rendering quality. Without this, performance in PSNR is decreased by 0.16 dB. In addition, we investigate the impact of varying grid sizes on the student model training using the Mip360 dataset shown in Table 3. Generally, a larger grid size produces smaller voxel dimensions and a greater number of voxels, leading to a more detailed scene representation. While increasing the grid size can improve PSNR performance, it also incurs a substantial increase in GPU memory.

Impact of the number of Gaussians. We conduct experiments on the Room scene from Mip360 to evaluate the impact of Gaussian count. Table 4 reports the reconstruction quality and the number of Gaussians for different model variants. The ensemble of three diverse teacher models achieves the highest PSNR of 32.15 dB. Compared to vanilla 3DGS, the student (Base) modelâtrained via multiteacher distillationâpreserves comparable reconstruction quality while significantly reducing the number of Gaussians. Although the student (Big) model achieves higher PSNR, it uses nearly as many Gaussians as the teacher models. In contrast, the student (Small) model applies further pruning, resulting in only a slight PSNR drop of 0.15 dB.

Impact of different teachers. We analyze the effects of various teacher models on the performance of the student model. As shown in Table 5, employing multiple diverse teachers $( G _ { s t d } , G _ { p e r b } , G _ { d r o p } )$ to distill the student yields the best overall performance. In contrast, using three standard teachers $( G _ { s t d } )$ results in a lower PSNR (31.36), and single-teacher configurations perform even worse compared to these teacher ensembles. These results highlight that diversity among teachers provides richer and more complementary supervisory signals, thereby enhancing student model performance.

## 5. Conclusion

In this paper, we proposed a multi-teacher distillation framework for 3DGS, aiming to preserve reconstruction quality under significantly reduced Gaussian counts. By leveraging knowledge from multiple teacher models, our approach effectively transfers both scene geometry and appearance priors to a more compact student representation. Besides, we leverage a spatial distribution distillation strategy to encourage the student to learn spatial geometric distributions consistent with those of a standard teacher model. Extensive experiments across different scenes demonstrate that our distilled student model-Distilled-3DGS achieves promising performance with substantially fewer Gaussians, highlighting the potential of our method for deployment in memory-constrained or real-time scenes.

Limitation. Distilled-3DGS also has some drawbacks: first, it requires pre-training multiple high-performance teacher models, increasing training time and computational resources by at least N-fold compared to a single model; second, generating distillation soft labels via multi-model inference significantly increases GPU memory usage. Future work could explore end-to-end distillation pipelines or adaptive pruning strategies for Gaussian parameters to further improve efficiency and generalization.

## Acknowledgement

The computational resources are supported by SongShan Lake HPC Center (SSL-HPC) in Great Bay University. This work was also supported by Guangdong Research Team for Communication and Sensing Integrated with Intelligent Computing (Project No. 2024KCXTD047).

## References

[1] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, and Ricardo Martin-Brualla. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5855â5864, 2021. 2

[2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5470â5479, 2022. 2, 5, 7

[3] Cristian Bucilua, Rich Caruana, and Alexandru Niculescu- Ë Mizil. Model compression. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 535â541, 2006. 1, 3

[4] Guobin Chen, Wongun Choi, Xiang Yu, Tony Han, and Manmohan Chandraker. Learning efficient object detection models with knowledge distillation. Advances in neural information processing systems, 30, 2017. 3

[5] Guangchi Fang and Bing Wang. Mini-splatting: Representing scenes with a constrained number of gaussians. In European Conference on Computer Vision, pages 165â181. Springer, 2024. 2, 3, 7

[6] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5501â5510, 2022. 2, 7

[7] Takashi Fukuda, Masayuki Suzuki, Gakuto Kurata, Samuel Thomas, Jia Cui, and Bhuvana Ramabhadran. Efficient knowledge distillation from an ensemble of teachers. In Interspeech, pages 3697â3701, 2017. 3

[8] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pages 1050â1059. PMLR, 2016. 4

[9] Sharath Girish, Kamal Gupta, and Abhinav Shrivastava. Eagles: Efficient accelerated 3d gaussians with lightweight encodings. In European Conference on Computer Vision, pages 54â71. Springer, 2024. 7

[10] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 3

[11] Wenbo Hu, Yuling Wang, Lin Ma, Bangbang Yang, Lin Gao, Xiao Liu, and Yuewen Ma. Tri-miprf: Tri-mip representation for efficient anti-aliasing neural radiance fields. In

Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19774â19783, 2023. 2

[12] Deyi Ji, Haoran Wang, Mingyuan Tao, Jianqiang Huang, Xian-Sheng Hua, and Hongtao Lu. Structural and statistical texture knowledge distillation for semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16876â16885, 2022. 3

[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 1, 2, 3, 4, 7

[14] Arno Knapitsch, Jaesik Park, and Qian-Yi Zhou. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG), 36(4):1â13, 2017. 5

[15] Kisoo Kwon, Hwidong Na, Hoshik Lee, and Nam Soo Kim. Adaptive knowledge distillation based on entropy. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 7409â7413. IEEE, 2020. 3

[16] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3d gaussian representation for radiance field. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21719â 21728, 2024. 7

[17] Muyang Li, Ji Lin, Yaoyao Ding, Zhijian Liu, Jun-Yan Zhu, and Song Han. Gan compression: Efficient architectures for interactive conditional gans. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5284â5294, 2020. 3

[18] Quanquan Li, Shengying Jin, and Junjie Yan. Mimicking very efficient network for object detection. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 6356â6364, 2017. 3

[19] Yifan Liu, Ke Chen, Chris Liu, Zengchang Qin, Zhenbo Luo, and Jingdong Wang. Structured knowledge distillation for semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2604â2613, 2019. 3

[20] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20654â20664, 2024. 2

[21] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Markus Steinberger, Francisco Vicente Carrasco, and Fernando De La Torre. Taming 3dgs: High-quality radiance fields with limited resources. In SIGGRAPH Asia 2024 Conference Papers, pages 1â11, 2024. 2, 7

[22] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (ToG), 38(4):1â14, 2019. 5

[23] Thomas Muller, Alex Evans, Christoph Schied, and Alexan- Â¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1â15, 2022. 7

[24] Michael Niemeyer, Fabian Manhardt, Marie-Julie Rakotosaona, Michael Oechsle, Daniel Duckworth, Rama Gosula, Keisuke Tateno, John Bates, Dominik Kaeser, and Federico Tombari. Radsplat: Radiance field-informed gaussian splatting for robust real-time rendering with 900+ fps. arXiv preprint arXiv:2403.13806, 2024. 2

[25] Hyunwoo Park, Gun Ryu, and Wonjun Kim. Dropgaussian: Structural regularization for sparse-view gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21600â21609, 2025. 4

[26] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1):1929â1958, 2014. 4

[27] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive representation distillation. arXiv preprint arXiv:1910.10699, 2019. 3

[28] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600â612, 2004. 6

[29] Xiaoyang Wu, Daniel DeTone, Duncan Frost, Tianwei Shen, Chris Xie, Nan Yang, Jakob Engel, Richard Newcombe, Hengshuang Zhao, and Julian Straub. Sonata: Selfsupervised learning of reliable point representations. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22193â22204, 2025. 11

[30] Shan You, Chang Xu, Chao Xu, and Dacheng Tao. Learning from multiple teacher networks. In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining, pages 1285â1294, 2017. 3

[31] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. Plenoctrees for real-time rendering of neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5752â 5761, 2021. 2

[32] Sergey Zagoruyko and Nikos Komodakis. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. arXiv preprint arXiv:1612.03928, 2016. 3

[33] Hailin Zhang, Defang Chen, and Can Wang. Confidenceaware multi-teacher knowledge distillation. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4498â4502. IEEE, 2022. 3

[34] Hailin Zhang, Defang Chen, and Can Wang. Adaptive multiteacher knowledge distillation with meta-learning. In 2023 IEEE International Conference on Multimedia and Expo (ICME), pages 1943â1948. IEEE, 2023. 3

[35] Linfeng Zhang and Kaisheng Ma. Improve object detection with feature-based knowledge distillation: Towards accurate and efficient detectors. In International conference on learning representations, 2020. 3

[36] Linfeng Zhang, Xin Chen, Xiaobing Tu, Pengfei Wan, Ning Xu, and Kaisheng Ma. Wavelet knowledge distillation: Towards efficient image-to-image translation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12464â12474, 2022. 3

[37] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586â595, 2018. 6

[38] Zhaoliang Zhang, Tianchen Song, Yongjae Lee, Li Yang, Cheng Peng, and Rama Chellappa. Lp-3dgs: Learning to prune 3d gaussian splatting. Advances in Neural Information Processing Systems, 37:122434â122457, 2024. 2, 7

[39] Yang Katie Zhao, Shang Wu, Jingqun Zhang, Sixu Li, Chaojian Li, and Yingyan Celine Lin. Instant-nerf: Instant on-device neural radiance field training via algorithmaccelerator co-designed near-memory processing. In 2023 60th ACM/IEEE Design Automation Conference (DAC), pages 1â6. IEEE, 2023. 2

[40] Tinghui Zhou, Shubham Tulsiani, Weilun Sun, Jitendra Malik, and Alexei A Efros. View synthesis by appearance flow. In Computer VisionâECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11â14, 2016, Proceedings, Part IV 14, pages 286â301. Springer, 2016. 1

## A. Additional ablation experiments

## A.1. Point cloud structural similarity

Methods for evaluating the similarity of spatial geometric distribution between two point clouds can be broadly categorized into three types: distance-based, distribution matching-based, and learning-based approaches. Our proposed spatial distribution distillation strategy, which utilizes voxel histograms, falls under the distribution matching category. For the distance-based approach, we employ Chamfer Distance (CD) to directly measure the spatial distance between the two point sets. For the learning-based approach, we adopt Sonata [29], a state-of-the-art point cloud representation learning method, to extract point features and compute the similarity loss between the two point clouds.

<table><tr><td rowspan="2">Method</td><td colspan="5">Tanks&amp;Temples</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>Mem.(GB)</td><td>Student Time(min)</td></tr><tr><td rowspan="2">Distance-based Feature-based</td><td>23.69</td><td>0.832</td><td>0.187</td><td>23.82</td><td>50</td></tr><tr><td>23.78</td><td>0.847</td><td>0.178</td><td>40.00</td><td>60</td></tr><tr><td>Ours</td><td>23.76</td><td>0.845</td><td>0.179</td><td>13.83</td><td>30</td></tr></table>

Table 6. Impact of different structural similarity strategy.

As shown in Table 6, both the distance-based and feature-based methods consume considerable memory and time(student model training) yet fail to deliver substantial performance gains. In contrast, our proposed voxel histogram-based method outperforms these two approaches while requiring significantly less memory and computation time.

## B. Per-scene breakdown results

To provide a more detailed evaluation of our model, we present the per-scene breakdown results of Mip-NeRF360, Tanks&Temples and Deep Blending datasets.

<table><tr><td>Scene</td><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>#G(106)â</td></tr><tr><td rowspan="4">Playroom</td><td>EAGLES</td><td>30.38</td><td>0.910</td><td>0.250</td><td>0.80</td></tr><tr><td>Mini-Splatting</td><td>30.62</td><td>0.915</td><td>0.249</td><td>0.41</td></tr><tr><td>3D-GS*</td><td>28.79</td><td>0.911</td><td>0.241</td><td>3.39</td></tr><tr><td>Ours</td><td>30.45</td><td>0.926</td><td>0.243</td><td>0.26</td></tr><tr><td rowspan="4">Johnson</td><td>EAGLES</td><td>29.35</td><td>0.900</td><td>0.240</td><td>1.57</td></tr><tr><td>Mini-Splatting</td><td>29.36</td><td>0.903</td><td>0.260</td><td>0.38</td></tr><tr><td>3D-GS*</td><td>30.31</td><td>0.913</td><td>0.241</td><td>3.08</td></tr><tr><td>Ours</td><td>29.29</td><td>0.906</td><td>0.259</td><td>0.39</td></tr><tr><td rowspan="4">Average</td><td>EAGLES</td><td>29.86</td><td>0.910</td><td>0.250</td><td>1.19</td></tr><tr><td>Mini-Splatting</td><td>29.99</td><td>0.909</td><td>0.255</td><td>0.40</td></tr><tr><td>3D-GS*</td><td>29.55</td><td>0.912</td><td>0.241</td><td>3.24</td></tr><tr><td>Ours</td><td>29.87</td><td>0.916</td><td>0.251</td><td>0.33</td></tr></table>

Figure 6. Quantitative per-scene breakdown results on Deep Blending dataset.

<table><tr><td>Scene</td><td>Method</td><td>PSNR</td><td>SSIM</td><td>LPIPS</td><td>Num.(M)</td></tr><tr><td rowspan="4">Bicycle</td><td>EAGLES</td><td>25.04</td><td>0.750</td><td>0.240</td><td>2.26</td></tr><tr><td>Mini-Splatting</td><td>25.21</td><td>0.760</td><td>0.246</td><td>0.60</td></tr><tr><td>3D-GS*</td><td>25.03</td><td>0.740</td><td>0.241</td><td>5.67</td></tr><tr><td>Ours</td><td>24.97</td><td>0.777</td><td>0.233</td><td>0.59</td></tr><tr><td rowspan="4">Bonsai</td><td>EAGLES</td><td>31.32</td><td>0.940</td><td>0.190</td><td>0.64</td></tr><tr><td>Mini-Splatting</td><td>31.41</td><td>0.940</td><td>0.182</td><td>0.33</td></tr><tr><td>3D-GS*</td><td>31.99</td><td>0.960</td><td>0.170</td><td>1.64</td></tr><tr><td>Ours</td><td>32.79</td><td>0.946</td><td>0.179</td><td>0.31</td></tr><tr><td rowspan="4">Counter</td><td>EAGLES</td><td>28.40</td><td>0.900</td><td>0.200</td><td>0.56</td></tr><tr><td>Mini-Splatting</td><td>28.32</td><td>0.913</td><td>0.181</td><td>0.36</td></tr><tr><td>3D-GS*</td><td>28.89</td><td>0.920</td><td>0.190</td><td>1.58</td></tr><tr><td>Ours</td><td>29.55</td><td>0.914</td><td>0.181</td><td>0.35</td></tr><tr><td rowspan="4">Flowers</td><td>EAGLES</td><td>21.29</td><td>0.58</td><td>0.370</td><td>1.33</td></tr><tr><td>Mini-Splatting</td><td>21.31</td><td>0.614</td><td>0.334</td><td>0.62</td></tr><tr><td>3D-GS*</td><td>21.30</td><td>0.600</td><td>0.359</td><td>3.67</td></tr><tr><td>Ours</td><td>21.45</td><td>0.617</td><td>0.313</td><td>0.62</td></tr><tr><td rowspan="4">Garden</td><td>EAGLES</td><td>26.91</td><td>0.840</td><td>0.150</td><td>1.65</td></tr><tr><td>Mini-Splatting</td><td>26.67</td><td>0.844</td><td>0.153</td><td>0.69</td></tr><tr><td>3D-GS*</td><td>27.32</td><td>0.870</td><td>0.125</td><td>5.92</td></tr><tr><td>Ours</td><td>27.58</td><td>0.871</td><td>0.108</td><td>0.68</td></tr><tr><td rowspan="4">Kitchen</td><td>EAGLES</td><td>30.77</td><td>0.930</td><td>0.130</td><td>1.00</td></tr><tr><td>Mini-Splatting</td><td>31.24</td><td>0.924</td><td>0.123</td><td>0.38</td></tr><tr><td>3D-GS*</td><td>31.43</td><td>0.930</td><td>0.120</td><td>2.01</td></tr><tr><td>Ours</td><td>31.65</td><td>0.932</td><td>0.117</td><td>0.37</td></tr><tr><td rowspan="4">Room</td><td>EAGLES</td><td>31.47</td><td>0.920</td><td>0.200</td><td>0.67</td></tr><tr><td>Mini-Splatting</td><td>31.21</td><td>0.920</td><td>0.191</td><td>0.32</td></tr><tr><td>3D-GS*</td><td>31.59</td><td>0.920</td><td>0.200</td><td>1.99</td></tr><tr><td>Ours</td><td>31.54</td><td>0.927</td><td>0.193</td><td>0.31</td></tr><tr><td rowspan="4">Stump</td><td>EAGLES</td><td>26.78</td><td>0.770</td><td>0.240</td><td>2.22</td></tr><tr><td>Mini-Splatting</td><td>27.32</td><td>0.804</td><td>0.215</td><td>0.61</td></tr><tr><td>3D-GS*</td><td>26.53</td><td>0.770</td><td>0.240</td><td>4.68</td></tr><tr><td>Ours</td><td>27.73</td><td>0.811</td><td>0.193</td><td>0.60</td></tr><tr><td rowspan="4">Treehill</td><td>EAGLES</td><td>22.69</td><td>0.640</td><td>0.340</td><td>1.60</td></tr><tr><td>Mini-Splatting</td><td>22.58</td><td>0.656</td><td>0.331</td><td>0.63</td></tr><tr><td>3D-GS*</td><td>22.43</td><td>0.660</td><td>0.325</td><td>3.67</td></tr><tr><td>Ours</td><td>22.98</td><td>0.645</td><td>0.314</td><td>0.62</td></tr><tr><td rowspan="4">Average</td><td>EAGLES</td><td>27.23</td><td>0.810</td><td>0.240</td><td>1.33</td></tr><tr><td>Mini-Splatting</td><td>27.25</td><td>0.820</td><td>0.217</td><td>0.50</td></tr><tr><td>3D-GS*</td><td>27.39</td><td>0.819</td><td>0.219</td><td>3.43</td></tr><tr><td>Ours</td><td>27.81</td><td>0.827</td><td>0.202</td><td>0.49</td></tr></table>

Table 7. Quantitative per-scene breakdown results on MiP-NeRF360 dataset.

<table><tr><td>Scene</td><td>Method</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>#G(106)â</td></tr><tr><td rowspan="4">Train</td><td>EAGLES</td><td>21.65</td><td>0.800</td><td>0.240</td><td>0.46</td></tr><tr><td>Mini-Splatting</td><td>21.28</td><td>0.801</td><td>0.238</td><td>0.29</td></tr><tr><td>3D-GS*</td><td>21.94</td><td>0.810</td><td>0.200</td><td>1.11</td></tr><tr><td>Ours</td><td>22.14</td><td>0.812</td><td>0.207</td><td>0.23</td></tr><tr><td rowspan="4">Truck</td><td>EAGLES</td><td>25.09</td><td>0.870</td><td>0.160</td><td>0.83</td></tr><tr><td>Mini-Splatting</td><td>25.13</td><td>0.871</td><td>0.166</td><td>0.35</td></tr><tr><td>3D-GS*</td><td>25.31</td><td>0.880</td><td>0.150</td><td>2.54</td></tr><tr><td>Ours</td><td>25.37</td><td>0.878</td><td>0.150</td><td>0.29</td></tr><tr><td rowspan="4">Average</td><td>EAGLES</td><td>23.37</td><td>0.835</td><td>0.200</td><td>0.64</td></tr><tr><td>Mini-Splatting</td><td>23.21</td><td>0.836</td><td>0.203</td><td>0.32</td></tr><tr><td>3D-GS*</td><td>23.62</td><td>0.845</td><td>0.175</td><td>1.83</td></tr><tr><td>Ours</td><td>23.76</td><td>0.845</td><td>0.179</td><td>0.25</td></tr></table>

Table 8. Quantitative per-scene breakdown results on Tanks&Temples dataset.

<!-- image-->  
Figure 5. Visual comparison with different teacher models. Without the guidance of diverse teacher models, the rendering quality of the student 3DGS model gradually deteriorates.