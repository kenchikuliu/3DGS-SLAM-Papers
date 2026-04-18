# SV-GS: Sparse View 4D Reconstruction with Skeleton-Driven Gaussian Splatting

Jun-Jee Chao University of Minnesota

Volkan Isler The University of Texas at Austin

<!-- image-->  
Figure 1. We study the problem of 4D reconstruction from sparse observations. Our method takes the following as input: (a) A set of posed RGB images of an articulated target, captured at sparse time steps (up to 20x fewer than existing methods) from arbitrary viewpoints; (b) An annotated skeleton graph only at the first frame; (c) An initial static 3D reconstruction, derived either from multi-view images or a pre-trained image-to-3D diffusion model. Our goal is to produce a continuous 4D reconstruction of the dynamic target.

## Abstract

Reconstructing a dynamic target moving over a large area is challenging. Standard approaches for dynamic object reconstruction require dense coverage in both the viewing space and the temporal dimension, typically relying on multi-view videos captured at each time step. However, such setups are only possible in constrained environments. In real-world scenarios, observations are often sparse over time and captured sparsely from diverse viewpoints (e.g., from security cameras), making dynamic reconstruction highly ill-posed. We present SV-GS, a framework that simultaneously estimates a deformation model and the objectâs motion over time under sparse observations. To initialize SV-GS, we leverage a rough skeleton graph and an initial static reconstruction as inputs to guide motion estimation. (Later, we show that this input requirement can be relaxed.) Our method optimizes a skeleton-driven deformation field composed of a coarse skeleton joint pose estimator and a module for fine-grained deformations. By

making only the joint pose estimator time-dependent, our model enables smooth motion interpolation while preserving learned geometric details. Experiments on synthetic datasets show that our method outperforms existing approaches under sparse observations by up to 34% in PSNR, and achieves comparable performance to dense monocular video methods on real-world datasets despite using significantly fewer frames. Moreover, we demonstrate that the input initial static reconstruction can be replaced by a diffusion-based generative prior, making our method more practical for real-world scenarios.

## 1. Introduction

Reconstructing dynamic targets from images is a longstanding computer vision problem, with applications in motion analysis [6, 25], AR/VR [60], and dynamic scene understanding [52]. While recent progress in neural [5, 40, 49] and Gaussian-based representations [12, 51, 54, 64] have shown impressive results, most methods rely on monocular or multi-view videos with dense temporal coverage, where rich motion cues and correspondences are available.

In real-world scenarios, however, such dense observations are not always accessible. For example, surveillance cameras often capture moving objects sparsely over time, especially in cluttered environments. Moreover, when multiple cameras are available, their viewpoints can differ drastically, and the observed targets may exhibit significant motion and self-occlusion between observations. Under this setting, temporal correspondences are difficult to establish, as appearance can change dramatically across sparse observations, making dynamic reconstruction highly ill-posed.

In this paper, we address this challenging setting of articulated dynamic reconstruction from sparse temporal observations, where only a few posed images from arbitrary viewpoints are available as illustrated in Fig. 1. To solve this highly ill-posed problem, we consider a setting where we have access to additional structural information. Initially, we assume that a rough skeleton graph and a static reconstruction at the first frame are available. The initial reconstruction can be can be obtained from a standard multiview setup [13, 42, 43]. Later on in Section 4.3, we will show how this assumption can be relaxed with a pre-trained generative model [24, 45, 47] using only a single image. Despite this additional information, the task remains difficult as the inputs do not yield a complete rigged modelâthe skeleton annotation can be noisy and contains only node positions and connectivity, while the joint poses, skinning weights, and point-to-part associations remain unknown.

We present SV-GS which, given the input skeleton graph and initial static reconstruction, learns a skeleton-driven deformation field that models coherent motion under sparse supervision. Our deformation field consists of a coarse skeleton joint pose estimator and a module that models fine-grained motion deformations. By allowing only the joint pose estimator to be time-dependent, our model enables smooth test-time motion interpolation while preserving learned local deformation details. Experiments demonstrate that state-of-the-art (SOTA) dynamic reconstruction methods degrade significantly in this sparse setting, while SV-GS achieves better reconstruction quality. Furthermore, we show that the need for multi-view initialization can be relaxed using a diffusion-based generative prior, enabling dynamic reconstruction in real-world scenarios. Our contributions can be summarized as follows.

â¢ We perform articulated dynamic reconstruction from sparse temporal observations, where only a few frames from arbitrary viewpoints are available.

â¢ We present a skeleton-driven deformation field that enables smooth motion interpolation under sparse supervision, and demonstrate that a pre-trained diffusion prior can be incorporated to fill in missing information.

â¢ Experiments show that our method outperforms SOTA methods by up to 34% in PSNR on synthetic datasets with sparse observations, and achieves comparable performance to dense monocular video methods on real-world datasets with significantly fewer frames.

## 2. Related Work

We review related works on dynamic scene reconstruction and articulated object modeling. As most existing methods rely on video inputs (see Fig. 2), we also discuss recent generative approaches that are related to our sparse-view setting. We further quantify the difficulty of our setup using the metric from [10] in the supplementary material.

Dynamic scene modeling. Some earlier methods apply explicit mesh representation [4, 8] or implicit neural volumes [26] to model dynamic scenes from multi-view videos, leveraging the dense spatial and temporal information. After NeRF [32] was introduced, the field of novel view synthesis became even more popular. D-NeRF [40] and many concurrent works extend the static NeRF representation to dynamic scene by optimizing an additional time-dependent deformation field [11, 17, 35, 36, 48, 57], or by directly modeling the 4D space [5, 9, 44].

3D Gaussian Splatting (3DGS) [13] is another scene representation that has gained popularity due to its fast rendering speed. Many recent works adapt 3DGS for dynamic scene reconstruction [12â14, 23, 28, 50, 54, 62]. 4DGS [54] decouples the scene into a static 3DGS and a deformation field represented with multi-resolution hex-planes [5]. Recently, a line of work attempts to model the dynamic scene with a more controllable representation by using a sparse set of parameters to represent the dense deformation [12, 50]. However, most existing methods rely on monocular videos with dense temporal information, which is unavailable in our sparse observation setup (Fig. 2). Moreover, without structural constraints, these approaches can produce noisy deformations that fail to preserve the objectâs structure under sparse supervision.

<!-- image-->  
Figure 2. Comparison of input configurations across dynamic reconstruction methods. Multi-view and monocular video methods assume small viewpoint changes and dense temporal observations, whereas our method handles sparse temporal observations with large viewpoint variations. Generative methods attempt to synthesize the full motion from a static state.

Articulated object reconstruction. To model dynamic articulated objects, some methods leverage category-specific priors. For example, SMPL [27] focuses on human body modeling, and MANO [41] focuses on human hands. Another line of work tackles the animal category where a kinematic structure is shared among different instances [15, 19, 55, 56, 63, 70]. However, many of these works focus on part discovery from a single image instead of reconstructing the continuous motion for novel view synthesis [19, 55, 63].

More general category-agnostic methods have been explored [33, 51, 61, 64, 67]. Many of these methods focus on simultaneously modeling the dynamic target and extracting the underlying kinematic structure from video input. SK-GS [51] extends SP-GS [50] by first grouping the 3DGS with similar motion into superpoints. Then, they extract a skeleton model from the superpoints based on relative motion and proximity. Similarly, built upon SC-GS [12], RigGS [64] first estimates a set of sparse control points to model the dynamic scene, then the kinematic skeleton is estimated from the motion of these control points. While these methods learn skeleton-driven deformation for 3DGS which is similar to our setup, they take continuous monocular videos as input, and do not perform well when only sparse images are available. Moreover, we show in Section 4.2 that with the same initialization and skeleton input, these methods designed for monocular video fail when only sparse images are available.

Scene reconstruction with generative priors. A pretrained generative model can potentially be applied to fill in the missing information from sparse observations. Many recent works apply pre-trained diffusion models for static scene reconstruction from one or more images [22, 24, 45, 47]. To extend from static to dynamic scene, a popular approach is to apply the SDS loss [39] to guide the motion with a pre-trained video diffusion model [3, 16, 18, 58, 65, 68, 69]. However, these methods focus on the generative setup where the generated motion is expected to be smooth and reasonable but does not need to match any ground truth. On the contrary, our problem setup requires us to estimate the ground truth motion from sparse observations.

## 3. Method

Our goal is to reconstruct an articulated dynamic target from sparse temporal observations $\mathcal { T } = \{ I _ { t } \} _ { t \in [ 0 , 1 ] }$ , where each time step consists of only a single posed image captured from an arbitrary viewpoint. We present SV-GS, which assumes access to a skeleton structure $\mathcal { F }$ as input. The skeleton specifies the 3D locations of J nodes and their parentâchild connectivity, which can be obtained through human annotation or estimated using an off-theshelf method [21, 59]. As illustrated in Fig. 3, SV-GS starts from building an initial static 3D reconstruction of the target. Then we learn a skeleton-driven deformation field that models continuous articulation and motion over time, under sparse temporal supervision.

## 3.1. Scene representation

Initial Static 3D Gaussians. We adopt 3D Gaussian Splatting (3DGS) [13] as our scene representation for its fast optimization speed and explicit, physically interpretable parameterization. 3DGS represents a scene with a collection of Gaussian primitives $\mathcal { G } = \{ g _ { i } \} _ { i \in 1 , . . . , N } .$ , where each Gaussian $g _ { i }$ is defined by a center $\mu _ { i } ,$ a rotation matrix represented with quaternion $q _ { i }$ , a scaling vector $s _ { i } ,$ an opacity value $\sigma _ { i } ,$ , and a set of spherical harmonics coefficients $s h _ { i }$ determining the view-dependent color. Given a camera pose, we can render an image from ${ \mathcal { G } } ,$ where the pixel color is determined by Î±-blending along the ray direction:

$$
c o l o r = \sum _ { k } c _ { k } \alpha _ { k } \prod _ { j = 1 } ^ { k - 1 } ( 1 - \alpha _ { j } )\tag{1}
$$

where k is the index of the Gaussians sorted by depth along the viewing direction, and $c _ { k }$ is the view-dependent color evaluated from the spherical harmonics coefficients. The Î± value is the opacity Ï weighted by the projected 2D Gaussian distribution from the 3D space onto the 2D plane.

In this paper, we assume the initial static 3DGS can be obtained either from multi-view images or potentially from a pre-trained image-to-3D diffusion model. In the multiview setup, we follow the standard pipeline [13] to optimize the Gaussian parameters by minimizing the perceptual loss between the rendered images and the ground truth images. We further showcase in Section 4.3 that the multiview initialization can potentially be replaced with a pretrained generative model. More details can be found in Section 4.3 and the supplementary material.

Skeleton-Driven Deformation. Given the initial static 3DGS $\mathcal { G }$ and an annotated skeleton graph ${ \mathcal F } ,$ our goal is to learn a deformation field that transforms the initial $\mathcal { G }$ to match the observed images at the corresponding sparse time steps. Furthermore, the learned deformation enables continuous motion synthesis for intermediate time steps without direct observations. Note that the input skeleton graph can be noisy and contains only the 3D positions of the nodes and their connectivity, without point-to-part associations or joint parameters.

To derive a deformation that is constrained by the input skeleton while also remaining flexible to match the sparse observations, we draw inspiration from learnable Linear

<!-- image-->  
Figure 3. Given canonical 3D Gaussians and an input skeleton, SV-GS first predicts time-dependent joint poses, regularized with $\mathcal { L } _ { m o t i o n }$ for temporal smoothness. With the predicted skeleton poses, the canonical Gaussians are then transformed via Linear Blend Skinning using learnable per-bone radii and a skinning correction field. Finally, a detail deformation field refines the transformed Gaussians. All parameters are optimized by minimizing the perceptual loss between the rendered and observed images.

Blend Skinning (LBS) techniques [16, 30, 61, 64]. Specifically, we adopt an MLP to model the time-dependent local rotation $q _ { j } ^ { t }$ (represented using quaternions) for each joint $j$ in the skeleton, along with a local translation $p ^ { t } \in \mathbb { R } ^ { 3 }$ only for the root joint.

$$
q ^ { t } , p ^ { t } = M L P _ { \Theta } ( \gamma ( t ) )\tag{2}
$$

where $\gamma ( \cdot )$ denotes the positional encoding [32]. The local rotations are defined for each joint in the local frame, therefore, given the parentâchild hierarchy in the skeleton graph ${ \mathcal F } ,$ we compute the global transformation of each joint using forward kinematics [7]

$$
\hat { \mathbf { R } ^ { t } } , \hat { T ^ { t } } = f k ( \mathcal { F } , q ^ { t } , p ^ { t } )\tag{3}
$$

where $\hat { \mathbf { R } _ { j } ^ { t } }$ and $\hat { T } _ { j } ^ { t }$ denote the global rotation (represented as $3 \times 3$ matrix) and translation of joint $j$ at time t respectively. $f k ( \cdot )$ is the forward kinematics operation that propagates the local transformation of each joint to all child joints.

Next, to guide the Gaussian primitives with the estimated joint poses, we derive a fine-grained motion field based on a learnable LBS deformation. We first construct B bones, where each bone $b _ { j }$ corresponds to the edge connecting joint $j$ and itâs parent [49, 64]. Each Gaussian center $\mu _ { i }$ in the canonical static state is transformed to time t as:

$$
\mu _ { i } ^ { t } = \sum _ { j = 1 } ^ { B } w _ { i , j } ( \hat { \mathbf { R } _ { j } ^ { t } } \mu _ { i } + \hat { T _ { j } ^ { t } } )\tag{4}
$$

where $w _ { i , j }$ is the learnable skinning weight satisfying $\Sigma _ { j } w _ { i , j } ~ = ~ 1$ The rotation part of the Gaussian primitive is similarly approximated by the weighted sum: $\textstyle \sum _ { j = 1 } ^ { B } w _ { i , j } \hat { \mathbf { R } _ { j } ^ { t } } \mathbf { R } _ { i }$

Learnable Skinning Weights. Since the input skeleton can be noisy and lacks skinning and deformation information, we model the skinning effect of each bone as a Radial Basis Function (RBF) kernel in the canonical (static) state. Moreover, to account for the noise in the input skeleton, we learn a position-dependent correction field $M L P _ { \Phi }$ also in the canonical state. Formally, we compute normalized weights as:

$$
w _ { i , j } = \frac { w _ { i , j } ^ { \star } } { \sum _ { j = 1 } ^ { B } w _ { i , j } ^ { \star } } ,\tag{5}
$$

where

$$
\hat { w _ { i , j } } = \Delta w _ { i , j } \ e x p \left( - \frac { d _ { i , j } ^ { 2 } } { 2 r _ { j } ^ { 2 } } \right)\tag{6}
$$

Here $d _ { i , j }$ denotes the the distance between the Gaussian center $\mu _ { i }$ and bone $b _ { j }$ in the canonical frame, and $r _ { j }$ is the learnable influence radius for each bone $j .$ Moreover, the correction field $\Delta w _ { i , j }$ is parameterized with a MLP:

$$
\Delta w _ { i , j } = M L P _ { \Phi } ( \gamma ( \mu _ { i } ) )\tag{7}
$$

where $\gamma ( \cdot )$ again denotes the positional encoding [32] for the Gaussian center $\mu _ { i }$

Detail Deformation. The above skeleton-driven deformation captures coarse articulated motion by propagating the joint transformations to the Gaussian primitives. However, the skeleton is sparse by nature and cannot account for finegrained non-rigid deformations. Inspired by [64], we include an additional pose-dependent detail deformation field M $L P _ { \Psi }$ to refine the local details. For each Gaussian, we predict a small offset by considering the Gaussian center in the canonical frame and the predicted joint poses at that time step. Therefore, the final Gaussian center at time t is:

$$
\hat { \mu _ { i } ^ { t } } = \mu _ { i } ^ { t } + M L P _ { \Psi } ( \gamma ( \mu _ { i } ) , \mathbf { R } ^ { t } )\tag{8}
$$

<!-- image-->  
Figure 4. Qualitative results on the D-NeRF dataset [40] downsampled at 0.1 intervals, yielding 11 frames per motion sequence (up to 20Ã fewer than the original). We compare our method with SOTA methods including 4DGS [54], SK-GS [51], and RigGS [64]. Additionally, we modify RigGS [64] to take in the same skeleton input as ours. Despite all methods being initialized with the same multi-view images at t = 0, existing methods produce noisy deformations and fail to preserve object structure given only sparse temporal observations.

## 3.2. Optimization

The trainable parameters of our deformation field include the joint local pose predictor $M L P _ { \Theta }$ , the bone influence radii $r _ { j } .$ , the skinning correction field $M L P _ { \Phi }$ , and the detail deformation field $M L P _ { \Psi }$ . During training the deformation parameters, we keep the parameters of the static canonical Gaussians G fixed. All deformation parameters are jointly optimized by minimizing the following loss:

$$
\mathcal { L } = \lambda _ { 1 } \mathcal { L } _ { p e r c e p t u a l } + \lambda _ { 2 } \mathcal { L } _ { m o t i o n } + \lambda _ { 3 } \mathcal { L } _ { d e t a i l }\tag{9}
$$

The main objective is to enforce the deformed Gaussians to match the observed images when rendered from the corresponding viewpoints. We follow the perceptual loss used in 3DGS [13], where $\mathcal { L } _ { p e r c e p t u a l }$ is a combination of $\mathcal { L } _ { 1 }$ loss and D-SSIM loss.

However, since only one image observation is available at each sparse time step, regions without direct supervision may undergo unstable or noisy deformation. To address this, we introduce two regularization terms that constrain the skeleton motion and the detail deformation field.

Motion Regularization. Since the joint poses are defined in their respective local frames, we can directly enforce temporal smoothness by minimizing the Laplacian of the predicted values with respect to time:

$$
\mathcal { L } _ { m o t i o n } = \frac { 1 } { T J } \sum _ { t } ^ { T } \sum _ { j } ^ { J } \left| q _ { j } ^ { t - 1 } - 2 q _ { j } ^ { t } + q _ { j } ^ { t + 1 } \right|\tag{10}
$$

where $T$ is uniformly sampled between [0, 1]. This regularization helps mitigate the ambiguity caused by selfocclusions under single-view supervision at each time step, preventing $M L P _ { \Theta }$ from producing abrupt pose changes and encouraging temporally coherent motion.

Detail Deformation Regularization. The detail deformation field $M L P _ { \Psi }$ is defined in the canonical frame to model small offsets for each Gaussian primitive such that the rendered images reflect finer motion details. Since this field is not intended to cause large displacements, we apply an $\mathcal { L } _ { 2 }$ regularization term on the predicted offsets:

$$
\mathcal { L } _ { d e t a i l } = \frac { 1 } { N } \sum _ { i } ^ { N } \left\| \boldsymbol { M } \boldsymbol { L } \boldsymbol { P } _ { \Psi } ( \gamma ( \mu _ { i } ) , \mathbf { R } ^ { t } ) \right\| _ { 2 } ^ { 2 }\tag{11}
$$

## 3.3. Inference

Our ultimate goal is to reconstruct a continuous motion sequence from sparse observations. Since the model is only supervised at a few discrete time steps, the learned M $L P$ may produce temporally inconsistent or jittery motions when queried at unseen time steps. To mitigate this issue, we design the deformation field such that only the local pose prediction $M L P _ { \Theta }$ depends explicitly on time. This allows us to effectively perform interpolation for the joint poses at unseen intermediate time steps while preserving the effect of the skinning correction field and detail deformation field. We show in the supplementary video that our method generates smooth and coherent motion even under sparse temporal supervision.

<!-- image-->  
All observed images from each motion sequenceFigure 5. We show all input views from the downsampled dataset (up to 20Ã fewer frames than the original), illustrating the challenges of establishing correspondences under sparse observations, large viewpoint changes, and self-occlusions.

## 4. Experiments

We first compare our method against existing approaches on novel view synthesis under sparse temporal observations, given a multi-view reconstruction at the initial state (Section 4.2). We then demonstrate that the multi-view initialization can be replaced by a pre-trained diffusion-based generative model (Section 4.3), highlighting the potential of our approach in more challenging scenarios.

## 4.1. Experimental Setup

Datasets. Our experiments are mainly conducted on three datasets: D-NeRF [40], DG-Mesh [20], and ZJU-MoCap [37]. D-NeRF [40] contains 6 synthetic scenes after excluding those with multiple objects or inconsistent motion between training and testing [12]. DG-Mesh [20] includes 5 synthetic sequences of articulated animal models. We normalize the time steps to the [0, 1] range and uniformly subsample frames at 0.1 intervals, resulting in 11 image observations per sequence, where each observation is captured from an arbitrary camera viewpoint at that time step as illustrated in Fig. 5. This corresponds to up to 20Ã fewer time frames compared to the original datasets. Following [64], we evaluate our approach on 6 real-world sequences from the ZJU-MoCap dataset [37]. Since ZJU-MoCap contains longer motion sequences with more complex movements, we downsample the frame rate to 1/10 from the original, where each time step is again arbitrarily selected from the training views. To further demonstrate the generalization ability of our method on in-the-wild data, we additionally test on the camel scene from the DAVIS dataset [38], where no camera pose information is provided.

Metrics. We evaluate the quality of novel view synthesis using three standard metrics: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM) [53], and Learned Perceptual Image Patch Similarity (LPIPS) [66].

Table 1. Quantitative results on the D-NeRF dataset [40] downsampled at 0.1 intervals, yielding 11 frames per motion sequence. We report the average metrics across all test cases / the mean over the worst-performing test case of each scene. â  indicates method initialized with the same skeleton input as ours.
<table><tr><td>Method</td><td>SSIM â</td><td>PSNR â</td><td>LPIPS (Ã100)â</td></tr><tr><td>4DGS [54]</td><td>0.925 / 0.829</td><td>21.70 / 17.01</td><td>7.85 / 12.02</td></tr><tr><td>SK-GS [51]</td><td>0.921 / 0.790</td><td>19.43 / 15.45</td><td>8.8 / 16.38</td></tr><tr><td>RigGS [64]</td><td>0.897 / 0.771</td><td>24.23 / 19.33</td><td>8.28 / 13.32</td></tr><tr><td>RigGS [64]â </td><td>0.839 / 0.739</td><td>22.63 / 19.29</td><td>13.82 / 18.59</td></tr><tr><td>Ours</td><td>0.950 / 0.893</td><td>27.75 / 23.48</td><td>5.79 / 9.43</td></tr></table>

Implementation Details. The experiments are conducted on a single NVIDIA RTX 4080 GPU. Optimizations are done with PyTorch [2] and the ADAM optimizer [34]. We set $\lambda _ { 1 } = 2 , \lambda _ { 2 } = 1 , \lambda _ { 3 } = 1$ . We run the deformation field optimization for 40,000 steps for each scene, and the skeleton graph is initialized with the estimates from [64]. More details can be found in the supplementary material.

## 4.2. Comparison with Existing Methods

Synthetic Datasets. Most existing dynamic scene reconstruction methods rely on either monocular video or multiview video inputs. Therefore, to ensure a fair comparison under our sparse temporal observation setting, we provide all methods with the same multi-view posed images only at the initial time step. We compare our method with 4DGS [54], SK-GS [51], and RigGS [64] for the task of novel view synthesis. 4DGS [54] learns a deformation field for the canonical 3DGS without any explicit structural constraint. In contrast, both SK-GS [51] and RigGS [64] jointly reconstruct the dynamic target and its underlying kinematic structure. Since our method takes a skeleton graph as input, we also modify RigGS [64] to initialize from the same skeleton for direct comparisons.

We present qualitative results on the D-NeRF [40] and DG-Mesh [20] datasets in Fig. 4 and Fig. 6, respectively. As shown, all baselines struggle when only sparse temporal observations are available. Without structural constraints, 4DGS [54] produces diverging deformations that do not preserve object structure. While SK-GS [51] and RigGS [64] consider skeleton constraints, they can generate inaccurate motion or skinning weights which result in blurry renderings. Additionally, we initialize RigGS with the same skeleton as ours, however, without careful design, the noisy skeleton and the absence of ground-truth skinning weights can lead to unstable deformations and degraded reconstruction quality. Quantitative results in Table 1 and Table 2 confirm that our method outperforms all baselines across all evaluation metrics. For the DG-Mesh dataset, we evaluate two temporal downsampling configurations with intervals of 0.05 and 0.1, corresponding to 21 and 11 observable time steps respectively. As shown in Table 2, when more temporal observations are available, the baselines can achieve a closer SSIM score to ours, whereas our method remains robust even under severely sparse temporal inputs.

<!-- image-->  
Figure 6. Qualitative result on the DG-Mesh dataset [20] downsampled at 0.05 intervals, yielding 21 frames per motion sequence. While all methods perform similarly for parts with small motion, our approach better preserves object structure and captures finegrained motion more faithfully.

Table 2. Results on the DG-Mesh dataset [20] downsampled at 0.05 and 0.1 intervals. We present the average across all test cases / the mean over the worst-performing test case of each scene.
<table><tr><td colspan="4">DG-Mesh 0.05</td></tr><tr><td>Method</td><td>SSIM â</td><td>PSNR â</td><td>LPIPS (Ã100)â</td></tr><tr><td>4DGS [54]</td><td>0.918 / 0.822</td><td>23.40 / 17.68</td><td>7.26 / 13.23</td></tr><tr><td>SK-GS [51]</td><td>0.920 / 0.833</td><td>23.32 / 17.68</td><td>7.69 / 13.26</td></tr><tr><td>RigGS [64]</td><td>0.879 / 0.712</td><td>22.81 / 16.87</td><td>8.36 / 15.88</td></tr><tr><td>Ours</td><td>0.929 / 0.824</td><td>25.81 / 19.11</td><td>6.38 / 12.43</td></tr><tr><td colspan="4">DG-Mesh 0.1</td></tr><tr><td>Method</td><td>SSIM â</td><td>PSNR â</td><td>LPIPS (Ã 100) â</td></tr><tr><td>4DGS [54]</td><td>0.887 / 0.774</td><td>21.28 / 16.07</td><td>8.72 / 15.41</td></tr><tr><td>SK-GS [51]</td><td>0.875 / 0.776</td><td>20.56 / 15.92</td><td>10.37 / 17.22</td></tr><tr><td>RigGS [ [64]</td><td>0.855 / 0.694</td><td>21.80 / 16.51</td><td>9.27 / 16.29</td></tr><tr><td>Ours</td><td>0.900 / 0.786</td><td>23.76 / 17.86</td><td>7.59 / 13.78</td></tr></table>

Real-World Dataset. We compare our method against RigGS [64] and AP-NeRF [49] on the real-world ZJU-MoCap [37] dataset. In Table 3, the reported results of RigGSS [64] and AP-NeRF [37] are obtained using all available time steps in the standard monocular video setup, whereas our method runs with only 1/10 and $1 / 5$ of the time steps. Despite having access to significantly fewer temporal observations, our approach achieves comparable performance to these SOTA methods. We show in Fig. 7 that our method is able to reconstruct the motion accurately.

## 4.3. Relaxing the Need for Multi-View Initialization with a Pretrained Generative Model

We demonstrate that the multi-view initialization at the canonical (static) state can potentially be replaced with a pretrained diffusion-based generative model, using only a single observation $I ^ { r }$ at the first time step. Given $I ^ { r } ,$ w e optimize the initial $\mathcal { G }$ with $\mathcal { L } _ { p e r c e p t u a l }$ only at the corresponding viewpoint, and employ the $\mathcal { L } _ { S D S } \left[ 3 9 \right]$ to optimize all other unseen viewpoints. $\mathcal { L } _ { S D S }$ is defined as:

<!-- image-->  
Figure 7. Qualitative result on the real-world ZJU-MoCap dataset. We use only 1/10 of the original video frames, where each frame is sampled from an arbitrary training viewpoint at that time step.

Table 3. Results on the real-world ZJU-MoCap dataset. Note that existing methods are trained with full monocular video sequences, whereas our method uses only 10Ã and 5Ã fewer frames.
<table><tr><td>Method</td><td>SSIM â</td><td>PSNR â</td><td>LPIPS (Ã100) â</td></tr><tr><td>AP-NeRF [49]</td><td>0.919</td><td>25.62</td><td>9.34</td></tr><tr><td>RigGS [64]</td><td>0.975</td><td>33.54</td><td>3.27</td></tr><tr><td>Ours (10Ã)</td><td>0.934</td><td>28.13</td><td>6.53</td></tr><tr><td>Ours (5Ã)</td><td>0.944</td><td>28.83</td><td>5.89</td></tr></table>

$$
\begin{array} { r } { \nabla _ { \mathcal { G } } \mathcal { L } _ { S D S } = \mathbb { E } _ { t , p , \epsilon } \left[ w ( t ) ( \epsilon _ { \phi } ( I ^ { p } ; t , I ^ { r } , \Delta p ) - \epsilon ) \frac { \partial I ^ { p } } { \partial \mathcal { G } } \right] } \end{array}\tag{12}
$$

where $w ( t )$ is the weighting function from DDIM [46] and $\epsilon _ { \phi } ( \cdot )$ is the predicted noise from a pre-trained 2D diffusion model. We use Zero-1-to-3 [24] as the diffusion prior, conditioned on $I ^ { r }$ and the relative camera pose $\Delta p$ from the reference viewpoint r to the rendering viewpoint $p .$ After the canonical $\mathcal { G }$ is initialized, we follow the same process described in Section 3.2 to optimize our deformation field. Since the initial $\mathcal { G }$ can be noisy with only one observed image, we keep $\mathcal { L } _ { S D S }$ in the loss function (Equation (9)) during the optimization to regularize the reconstruction. More details can be found in the supplementary material.

We first present results on the Jumpingjacks scene from D-NeRF [40] in Fig. 8. All methods are trained without access to the multi-view images. Despite using only 11 input images across the entire motion sequence, our method produces more coherent and structurally consistent motion compared to the baselines. We observe that while the baselines fit the input frames well, the sparse observations and self-occlusions lead to inconsistent geometry and unrealistic deformations when viewed from unseen viewpoints.

Additionally, we evaluate our method on the in-the-wild camel scene from the DAVIS dataset [38]. Note that the other side of the target is never seen in this monocular video. Assuming the camera is fixed across the whole sequence, we have only sparse temporal observations from a fixed viewpoint. Despite the challenging setup, we show in Fig. 9 that our method, paired with $\mathcal { L } _ { S D S }$ , successfully reconstructs plausible motion and texture for the visible regions. For the completely unseen part, the overall motion and structure are preserved, while there is oversaturated texture near the edge, which is a known issue of $\mathcal { L } _ { S D S }$ [1, 29, 31].

<!-- image-->  
Figure 8. Comparison of all methods without access to multi-view images at the initial time step. Despite using only 11 sparse input, our method reconstructs motion and preserves object structure more faithfully, whereas baselines are prone to artifacts under self-occlusion and sparse supervisions.

<!-- image-->  
Figure 9. Results on the camel scene from the in-the-wild DAVIS dataset [38] without camera pose information. Note that this is a monocular video with fixed camera and the other side of the target is never seen in the video.

## 4.4. Ablation studies

We conduct ablation studies to evaluate the effect of key components in our framework: the motion regularization term $\mathcal { L } _ { m o t i o n }$ , the skinning weight correction field $M L P _ { \Phi }$ and the detail deformation field $M L P _ { \Psi }$ . As shown in Table 4, both the skinning correction field and detail deformation field contribute to improving rendering quality on the D-NeRF dataset. The skinning correction field refines the learned skinning weights when the RBF-based bone representation is insufficient, while the detail deformation field adjusts the Gaussian primitives for parts that cannot be fully explained by the learned LBS deformation. Although $\mathcal { L } _ { m o t i o n }$ has small impact on quantitative metrics, Fig. 10 shows that it reduces noise in the joint poses predicted by M $L P _ { \Theta }$ , resulting in smoother and more stable motion.

<!-- image-->  
Figure 10. Qualitative comparison of results with and without $\mathcal { L } _ { m o t i o n }$ . This motion regularization term helps reduces noise in joint pose predictions.

Table 4. Ablation study on the D-NeRF dataset. We evaluate the effect of the motion regularization term $\mathcal { L } _ { m o t i o n }$ , skinning correction field $M L P _ { \Phi }$ , and the detail deformation field $M L P _ { \Psi }$
<table><tr><td>Method</td><td>SSIM â</td><td>PSNR â</td><td>LPIPS (Ã100) â</td></tr><tr><td>w/o  $\overline { { \mathcal { L } _ { m o t i o n } } }$ </td><td>0.942</td><td>27.26</td><td>6.08</td></tr><tr><td>w/o  $M L P _ { \Phi }$ </td><td>0.945</td><td>27.28</td><td>5.97</td></tr><tr><td>w/o  $M L P _ { \Psi }$ </td><td>0.931</td><td>26.34</td><td>6.51</td></tr><tr><td>Ours</td><td>0.950</td><td>27.75</td><td>5.79</td></tr></table>

## 5. Conclusion and Future Work

We presented SV-GS, a method for articulated dynamic object reconstruction from sparse temporal observations. SV-GS leverages a rough input skeleton and an initial static reconstruction to learn a skeleton-driven deformation field that models coherent motion across time. Furthermore, we showed that the need for multi-view initialization can be relaxed using a pre-trained diffusion-based generative prior, enabling dynamic reconstruction in real-world scenarios. Experiments on synthetic datasets show that SV-GS outperforms existing methods by up to 34% in PSNR under sparse observations and performs comparably to dense monocular methods on real-world datasets, even though SV-GS uses 10Ã fewer frames. While promising, our approach has limitations. The diffusion-based initialization can fail under severe self-occlusion or uncommon viewpoints, as it relies on a general pre-trained model. Moreover, test-time interpolation may struggle with highly complex motion. A potential future direction is to investigate using category-specific priors or a pre-trained prior conditioned on the noisy skeleton input to guide motion estimation and reconstruction.

## References

[1] Thiemo Alldieck, Nikos Kolotouros, and Cristian Sminchisescu. Score distillation sampling with learned manifold corrective. In European Conference on Computer Vision, pages 1â18, 2024. 8

[2] Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, Geeta Chauhan, Anjali Chourdia, Will Constable, Alban Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael Gschwind, Brian Hirsh, Sherlock Huang, Kshiteej Kalambarkar, Laurent Kirsch, Michael Lazos, Mario Lezcano, Yanbo Liang, Jason Liang, Yinghai Lu, CK Luk, Bert Maher, Yunjie Pan, Christian Puhrsch, Matthias Reso, Mark Saroufim, Marcos Yukio Siraichi, Helen Suk, Michael Suo, Phil Tillet, Eikan Wang, Xiaodong Wang, William Wen, Shunting Zhang, Xu Zhao, Keren Zhou, Richard Zou, Ajit Mathews, Gregory Chanan, Peng Wu, and Soumith Chintala. PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation. In 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS â24). ACM, 2024. 6

[3] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7996â8006, 2024. 3

[4] Michael Broxton, John Flynn, Ryan Overbeck, Daniel Erickson, Peter Hedman, Matthew Duvall, Jason Dourgarian, Jay Busch, Matt Whalen, and Paul Debevec. Immersive light field video with a layered mesh representation. ACM Transactions on Graphics (TOG), 39(4):86â1, 2020. 2

[5] Ang Cao and Justin Johnson. Hexplane: A fast representation for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 130â141, 2023. 1, 2

[6] Jun-Jee Chao, Qingyuan Jiang, and Volkan Isler. Part segmentation and motion estimation for articulated objects with dynamic 3d gaussians. arXiv preprint arXiv:2506.22718, 2025. 1

[7] Jacques Denavit and Richard S Hartenberg. A kinematic notation for lower-pair mechanisms based on matrices. 1955. 4

[8] Mingsong Dou, Sameh Khamis, Yury Degtyarev, Philip Davidson, Sean Ryan Fanello, Adarsh Kowdle, Sergio Orts Escolano, Christoph Rhemann, David Kim, Jonathan Taylor, et al. Fusion4d: Real-time performance capture of challenging scenes. ACM Transactions on Graphics (ToG), 35(4): 1â13, 2016. 2

[9] Sara Fridovich-Keil, Giacomo Meanti, Frederik RahbÃ¦k Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12479â12488, 2023. 2

[10] Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. Monocular dynamic view synthesis: A reality check. Advances in Neural Information Processing Systems, 35:33768â33780, 2022. 2

[11] Xiang Guo, Jiadai Sun, Yuchao Dai, Guanying Chen, Xiaoqing Ye, Xiao Tan, Errui Ding, Yumeng Zhang, and Jingdong Wang. Forward flow for novel view synthesis of dynamic scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16022â16033, 2023. 2

[12] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4220â4230, 2024. 1, 2, 3, 6

[13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 2, 3, 5

[14] Agelos Kratimenos, Jiahui Lei, and Kostas Daniilidis. Dynmf: Neural motion factorization for real-time dynamic view synthesis with 3d gaussian splatting. In European Conference on Computer Vision, pages 252â269. Springer, 2024. 2

[15] Jiahui Lei, Yufu Wang, Georgios Pavlakos, Lingjie Liu, and Kostas Daniilidis. Gart: Gaussian articulated template models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19876â19887, 2024. 3

[16] Xuan Li, Qianli Ma, Tsung-Yi Lin, Yongxin Chen, Chenfanfu Jiang, Ming-Yu Liu, and Donglai Xiang. Articulated kinematics distillation from video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17571â17581, 2025. 3, 4

[17] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural scene flow fields for space-time view synthesis of dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6498â 6508, 2021. 2

[18] Huan Ling, Seung Wook Kim, Antonio Torralba, Sanja Fidler, and Karsten Kreis. Align your gaussians: Text-to-4d with dynamic 3d gaussians and composed diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8576â8588, 2024. 3

[19] Di Liu, Anastasis Stathopoulos, Qilong Zhangli, Yunhe Gao, and Dimitris Metaxas. Lepard: Learning explicit part discovery for 3d articulated shape reconstruction. In Advances in Neural Information Processing Systems, pages 54187â 54198. Curran Associates, Inc., 2023. 3

[20] Isabella Liu, Hao Su, and Xiaolong Wang. Dynamic gaussians mesh: Consistent mesh reconstruction from dynamic scenes. In The Thirteenth International Conference on Learning Representations, 2025. 6, 7

[21] Isabella Liu, Zhan Xu, Yifan Wang, Hao Tan, Zexiang Xu, Xiaolong Wang, Hao Su, and Zifan Shi. Riganything: Template-free autoregressive rigging for diverse 3d assets. ACM Transactions on Graphics (TOG), 44(4):1â12, 2025. 3

[22] Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Ji-

ayuan Gu, and Hao Su. One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10072â10083, 2024. 3

[23] Qingming LIU, Yuan Liu, Jiepeng Wang, Xianqiang Lyu, Peng Wang, Wenping Wang, and Junhui Hou. MoDGS: Dynamic gaussian splatting from casually-captured monocular videos with depth priors. In The Thirteenth International Conference on Learning Representations, 2025. 2

[24] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298â9309, 2023. 2, 3, 7

[25] Shaowei Liu, Saurabh Gupta, and Shenlong Wang. Building rearticulable models for arbitrary 3d objects from 4d point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21138â 21147, 2023. 1

[26] Stephen Lombardi, Tomas Simon, Jason Saragih, Gabriel Schwartz, Andreas Lehrmann, and Yaser Sheikh. Neural volumes: learning dynamic renderable volumes from images. ACM Transactions on Graphics (TOG), 38(4):1â14, 2019. 2

[27] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multi-person linear model. ACM Trans. Graphics (Proc. SIGGRAPH Asia), 34(6):248:1â248:16, 2015. 3

[28] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In 2024 International Conference on 3D Vision (3DV), pages 800â809. IEEE, 2024. 2

[29] Artem Lukoianov, Haitz Sâaez de Ocâariz Borde, Kristjan Greenewald, Vitor Guizilini, Timur Bagautdinov, Vincent Sitzmann, and Justin M Solomon. Score distillation via reparametrized ddim. Advances in Neural Information Processing Systems, 37:26011â26044, 2024. 8

[30] Nadia Magnenat-Thalmann, Richard Laperriere, and Daniel\` Thalmann. Joint-dependent local deformations for hand animation and object grasping. In Proceedings on Graphics interfaceâ88, pages 26â33, 1989. 4

[31] David McAllister, Songwei Ge, Jia-Bin Huang, David W. Jacobs, Alexei A. Efros, Aleksander Holynski, and Angjoo Kanazawa. Rethinking score distillation as a bridge between image distributions. In Advances in Neural Information Processing Systems, 2024. 8

[32] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99â106, 2021. 2, 4

[33] Atsuhiro Noguchi, Umar Iqbal, Jonathan Tremblay, Tatsuya Harada, and Orazio Gallo. Watch it move: Unsupervised discovery of 3d joints for re-posing of articulated objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3677â3687, 2022. 3

[34] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the 3rd In-

ternational Conference on Learning Representations (ICLR 2015), 2015. 6

[35] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5865â5874, 2021. 2

[36] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-Brualla, and Steven M Seitz. Hypernerf: a higherdimensional representation for topologically varying neural radiance fields. ACM Transactions on Graphics (TOG), 40 (6):1â12, 2021. 2

[37] Sida Peng, Yuanqing Zhang, Yinghao Xu, Qianqian Wang, Qing Shuai, Hujun Bao, and Xiaowei Zhou. Neural body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9054â9063, 2021. 6, 7

[38] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 6, 7, 8

[39] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In The Eleventh International Conference on Learning Representations, 2022. 3, 7

[40] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10318â10327, 2021. 1, 2, 5, 6, 7

[41] Javier Romero, Dimitris Tzionas, and Michael J Black. Embodied hands: Modeling and capturing hands and bodies together. ACM Transactions on Graphics, 36(6), 2017. 3

[42] Johannes Lutz Schonberger and Jan-Michael Frahm. Â¨ Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 2

[43] Johannes Lutz Schonberger, Enliang Zheng, Marc Pollefeys, Â¨ and Jan-Michael Frahm. Pixelwise view selection for unstructured multi-view stereo. In European Conference on Computer Vision (ECCV), 2016. 2

[44] Ruizhi Shao, Zerong Zheng, Hanzhang Tu, Boning Liu, Hongwen Zhang, and Yebin Liu. Tensor4d: Efficient neural 4d decomposition for high-fidelity dynamic reconstruction and rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16632â 16642, 2023. 2

[45] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023. 2, 3

[46] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 7

[47] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653, 2023. 2, 3

[48] Edgar Tretschk, Ayush Tewari, Vladislav Golyanik, Michael Zollhofer, Christoph Lassner, and Christian Theobalt. Non- Â¨ rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12959â12970, 2021. 2

[49] Lukas Uzolas, Elmar Eisemann, and Petr Kellnhofer. Template-free articulated neural point clouds for reposable view synthesis. Advances in Neural Information Processing Systems, 36:31621â31637, 2023. 1, 4, 7

[50] Diwen Wan, Ruijie Lu, and Gang Zeng. Superpoint gaussian splatting for real-time high-fidelity dynamic scene reconstruction. In International Conference on Machine Learning, pages 49957â49972. PMLR, 2024. 2, 3

[51] Diwen Wan, Yuxiang Wang, Ruijie Lu, and Gang Zeng. Template-free articulated gaussian splatting for real-time reposable dynamic view synthesis. Advances in Neural Information Processing Systems, 37:62000â62023, 2024. 1, 3, 5, 6, 7

[52] Yikai Wang, Yinpeng Dong, Fuchun Sun, and Xiao Yang. Root pose decomposition towards generic non-rigid 3d reconstruction with monocular videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13890â13900, 2023. 1

[53] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600â612, 2004. 6

[54] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20310â 20320, 2024. 1, 2, 5, 6, 7

[55] Shangzhe Wu, Ruining Li, Tomas Jakab, Christian Rupprecht, and Andrea Vedaldi. Magicpony: Learning articulated 3d animals in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8792â8802, 2023. 3

[56] Yuefan Wu\*, Zeyuan Chen\*, Shaowei Liu, Zhongzheng Ren, and Shenlong Wang. CASA: Category-agnostic skeletal animal reconstruction. In Neural Information Processing Systems (NeurIPS), 2022. 3

[57] Wenqi Xian, Jia-Bin Huang, Johannes Kopf, and Changil Kim. Space-time neural irradiance fields for free-viewpoint video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9421â9431, 2021. 2

[58] Dejia Xu, Hanwen Liang, Neel P Bhatt, Hezhen Hu, Hanxue Liang, Konstantinos N Plataniotis, and Zhangyang Wang. Comp4d: Llm-guided compositional 4d scene generation. arXiv preprint arXiv:2403.16993, 2024. 3

[59] Zhan Xu, Yang Zhou, Evangelos Kalogerakis, Chris Landreth, and Karan Singh. Rignet: Neural rigging for articulated characters. ACM Trans. on Graphics, 39, 2020. 3

[60] Jinbo Yan, Rui Peng, Zhiyan Wang, Luyang Tang, Jiayu Yang, Jie Liang, Jiahao Wu, and Ronggang Wang. Instant gaussian stream: Fast and generalizable streaming of dynamic scene reconstruction via gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16520â16531, 2025. 1

[61] Gengshan Yang, Minh Vo, Natalia Neverova, Deva Ramanan, Andrea Vedaldi, and Hanbyul Joo. Banmo: Building animatable 3d neural models from many casual videos. In CVPR, 2022. 3, 4

[62] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20331â20341. IEEE, 2024. 2

[63] Chun-Han Yao, Wei-Chih Hung, Yuanzhen Li, Michael Rubinstein, Ming-Hsuan Yang, and Varun Jampani. Lassie: Learning articulated shape from sparse image ensemble via 3d part discovery. In NeurIPS, 2022. 3

[64] Yuxin Yao, Zhi Deng, and Junhui Hou. Riggs: Rigging of 3d gaussians for modeling articulated objects in videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5592â5601, 2025. 1, 3, 4, 5, 6, 7

[65] Yifei Zeng, Yanqin Jiang, Siyu Zhu, Yuanxun Lu, Youtian Lin, Hao Zhu, Weiming Hu, Xun Cao, and Yao Yao. Stag4d: Spatial-temporal anchored generative 4d gaussians. 2024. 3

[66] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586â595, 2018. 6

[67] Tingyang Zhang, Qingzhe Gao, Weiyu Li, Libin Liu, and Baoquan Chen. Bags: Building animatable gaussian splatting from a monocular video with diffusion priors, 2024. 3

[68] Yuyang Zhao, Zhiwen Yan, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. Animate124: Animating one image to 4d dynamic scene. arXiv preprint arXiv:2311.14603, 2023. 3

[69] Yufeng Zheng, Xueting Li, Koki Nagano, Sifei Liu, Otmar Hilliges, and Shalini De Mello. A unified approach for textand image-guided 4d scene generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7300â7309, 2024. 3

[70] Silvia Zuffi, Angjoo Kanazawa, David W Jacobs, and Michael J Black. 3d menagerie: Modeling the 3d shape and pose of animals. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6365â6373, 2017. 3