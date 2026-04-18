# LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates

Minkwan Kim1 Seungmin Lee1 Junho Kim1 Young Min Kim1,2

1 Dept. of Electrical and Computer Engineering, Seoul National University

2 Interdisciplinary Program in Artificial Intelligence and INMC, Seoul National University

<!-- image-->  
Initial Gaussian Environments

<!-- image-->  
Temporal Reconstruction

Figure 1. We introduce LTGS to efficiently update the Gaussian reconstruction of the initial environments. Given the spatiotemporally sparse post-change images, our framework tracks object-level changes in 3D and models long-term scene evolution.

## Abstract

Recent advances in novel-view synthesis can create the photo-realistic visualization of real-world environments from conventional camera captures. However, the everyday environment experiences frequent scene changes, which require dense observations, both spatially and temporally, that an ordinary setup cannot cover. We propose longterm Gaussian scene chronology from sparse-view updates, coined LTGS, an efficient scene representation that can embrace everyday changes from highly under-constrained casual captures. Given an incomplete and unstructured 3D Gaussian Splatting (3DGS) representation obtained from an initial set of input images, we robustly model the longterm chronology of the scene despite abrupt movements and subtle environmental variations. We construct objects as template Gaussians, which serve as structural, reusable priors for shared object tracks. Then, the object templates undergo a further refinement pipeline that modulates the priors to adapt to temporally varying environments given fewshot observations. Once trained, our framework is generalizable across multiple time steps through simple transformations, significantly enhancing the scalability for a temporal evolution of 3D environments. As existing datasets do not explicitly represent the long-term real-world changes

with a sparse capture setup, we collect real-world datasets to evaluate the practicality of our pipeline. Experiments demonstrate that our framework achieves superior reconstruction quality compared to other baselines while enabling fast and light-weight updates.

## 1. Introduction

With recent advances in novel-view synthesis, such as Neural Radiance Fields (NeRFs) [25] or 3D Gaussians Splatting (3DGS) [11], a casual user can reconstruct a 3D environment using a conventional camera input, and enjoy a photorealistic experience of exploring the environment. The representation stores the complex distribution of light and geometry of the static scene in an unstructured format. If we model the everyday environments where people live, daily activities often induce changes within the scene, quickly making the reconstruction obsolete. One may need to rerun these algorithms from scratch, or incorporate 4D representations that encapsulate the dynamic movements of the 3D representation [9, 30, 40]. The former discards the previously acquired information, while the latter can only process with continuous observation of smooth motion. Both approaches suffer from significant redundancy and are not desirable for modeling everyday environments in practical applications, such as location-based services, digital twins, or robotic setups.

We argue that a practical strategy for modeling evolving real-world environments is to efficiently detect and update changes. Instead of requiring continuous captures of the entire scene, we suggest a light-weight update of the changed region from a sparse set of images, as demonstrated in Fig. 1. While our setup suggests comparably flexible and realistic input requirements, it is highly underconstrained and therefore requires a strong scene prior. The challenge is to efficiently update the scene without notable artifacts while maintaining the necessary information for photorealistic rendering. Previous works that adapt novelview synthesis into sparse input still suffer from severe artifacts when the viewpoint changes significantly [6, 10, 27]. The continual learning method can sustain the pre-captured information [1, 41, 43], but relies on multiple captures for updates, as it lacks structural priors.

We propose an integrated pipeline that detects and updates Gaussian splatting scene representations in environments with diverse object states from sparse observations. We refer to the framework as long-term Gaussian scene chronology from sparse view updates, or LTGS. Realworld scenes can undergo multiple types of changes, including variations in geometry, appearance, or lighting. Recent works demonstrate that subtle appearance and lighting changes can be partially addressed by adding learnable embeddings or auxiliary neural networks [15, 18, 19, 23]. In this work, we focus on object-level changes, which involve abrupt geometric alterations such as insertions, removals, replacements, or relocations, providing a structural mechanism to efficiently account for the consequences of daily interactions.

Given an initially reconstructed 3DGS without any segmentation, we need to robustly extract the object-level structure, which confines the granularity of change estimation under ambiguous observations. Our scene update involves object tracking, relocalization, and reconstruction for individual objects. By combining multiple image-space cues of segmentation and feature extraction, we detect and distill the change to a 3D representation. We then aggregate the observations to build an object-level Gaussian template that models an object shared across time. The template serves as a reusable 3D prior to relocalize objects at different times, resolving ambiguities in sparse views. Then we reiterate the aggregation step such that the images of multiple time spans refine the shared template to best explain the overall observations via simple transformations. To evaluate our framework in practical scenarios, we captured realworld datasets containing multiple shared objects in various layouts, with few-shot observations spanning multiple time steps. Our pipeline demonstrates robust performance in challenging scenarios where previous approaches struggle. Our key contributions are summarized as follows:

<table><tr><td>Method</td><td>Discont. motion</td><td>Temporal recon.</td><td>Few-shot</td><td>Speed</td></tr><tr><td>3DGS [11]</td><td></td><td></td><td></td><td>Fast</td></tr><tr><td>InstantSplat [6]</td><td></td><td></td><td>Ã &gt;</td><td>Fast</td></tr><tr><td>4DGS [40]</td><td>ÃÃÃ</td><td></td><td></td><td>Moderate</td></tr><tr><td>NSC [18]</td><td>X</td><td>xÃ&gt;</td><td></td><td>Slow</td></tr><tr><td>3DGS-CD [21]</td><td></td><td>X</td><td></td><td>Fast</td></tr><tr><td>CL-NeRF [41]</td><td></td><td>X</td><td></td><td>Slow</td></tr><tr><td>CL-Splats [1]</td><td></td><td>X</td><td>xxÃx</td><td>Fast</td></tr><tr><td>LTGS (Ours)</td><td></td><td>â</td><td></td><td>Fast</td></tr></table>

Table 1. Related methods comparison. Our method captures abrupt geometric changes without requiring continuous motion and maintains reconstructions of multiple timesteps using a decomposable geometric scene prior that is reusable, thus allowing fast updates from a sparse set of images.

â¢ We address the problem of updating an initial 3DGS reconstruction in a highly efficient manner by using a set of spatio-temporally sparse images capturing long-term changes.

â¢ We present LTGS, an integrated strategy to track, associate, and relocalize the objects, and reconstruct the evolving scenes.

â¢ We propose a new real-world dataset, casually capturing environments with dynamic object-level changes across multiple timesteps to evaluate our framework.

## 2. Related Works

We build on Gaussian splatting representation and propose a novel yet challenging practical setup that allows lightweight updates of temporal changes from sparse view inputs. The setting partially shares important properties with recent variations of novel-view synthesis that enable temporal extensions or few-shot reconstruction, as summarized in Tab. 1.

## 2.1. Non-static scene reconstruction

Several works successfully extend NeRFs [17, 30] or 3DGS [22, 39, 40] to model dynamic scenes. These works require dense input observations both temporally and spatially [9], which can be challenging to obtain in practice. Furthermore, the reconstructed dynamics recover slow, continuous motions [17, 22, 30, 39, 40] or color variations on quasi-static geometry [18]. Another line of research integrates a continual learning framework, transforming the initial reconstruction to match gradual changes over time. The extension of 3DGS [1, 43] is inherently faster in rendering compared to NeRFs [41, 45], and therefore enjoys faster training times. However, these works still require more than ten input images to adapt to scene changes and eventually lose information from previous time steps.

## 2.2. Few-shot NeRF and Gaussian splatting

Several works pioneered relieving the dense-view requirement of NeRFs or 3DGS by incorporating geometric priors or regularization techniques [10, 27, 38, 46]. Recent geometric vision foundation models such as MASt3R [16] provide strong geometric priors, enabling effective scene representation when combined with Gaussian splatting [6, 36], particularly in cases where conventional structure-frommotion (SfM) [34] fails. However, in the context of updating scene changes, these methods cannot preserve the initial reconstruction, which leads to significant performance degradation, such as severe floating artifacts in novel views. Our approach instead builds a geometric structure of reusable priors by aggregating actual observations, and sustains consistent long-term temporal reconstruction despite a sparse set of temporal image captures.

## 2.3. Change detection and segmentations in 3DGS

While change detection has remained a long-standing problem in computer vision [31, 32], vision foundation models facilitate more generalizable ways to detect changes. Recent approaches [2, 12, 21] estimate change regions by leveraging the encoded feature embeddings of the segment anything model (SAM) [14]. Similarly, DINO features [28] are leveraged to detect change regions [1, 8]. Recently, several approaches have demonstrated that one can separate 3D objects from 3DGS with 2D masks [35, 44]. Our framework incorporates a similar framework to build Gaussian templates of the shared objects in evolving 3D environments.

## 3. Method

## 3.1. Overview

Given the initial 3DGS reconstruction of the scene $\mathcal { G } _ { 0 } .$ , our goal is to update the scene from a set of images ${ \mathcal { T } } = \{ I ^ { i } \} _ { t } ,$ where $t \in \{ t _ { 1 } , t _ { 2 } , \dots , t _ { M } \}$ represents a sparsely sampled time stamps, and the input images for each time stamp are captured from a small number of viewpoints $i = 1 , \ldots , N _ { t }$ that are not fixed beforehand. Our goal is to acquire the temporal evolution of the scene $\boldsymbol { \mathcal { S } } = \{ \mathcal { G } _ { 0 } , \mathcal { G } _ { 1 } , . . . \mathcal { G } _ { M } \}$ that recovers its states at sampled time steps in the input. The representations are a set of Gaussian splats, and each splat element is parameterized as $\{ \mu , q , s , \alpha , c \}$ , where $\boldsymbol { \mu } \in \mathbb { R } ^ { N \times 3 }$ denotes 3D center position, $q \in \mathbb { R } ^ { N \times 4 }$ represents quaternion representation of orientation (we denote its conversion into a rotation matrix as $R ) , \ s \ \in \ \mathbb { R } ^ { N \times 3 }$ specifies scale, $\boldsymbol { \alpha } \in \mathbb { R } ^ { N \times 1 }$ denotes opacity, and $c \in \mathbb { R } ^ { N \times 4 8 }$ encodes viewdependent color using third-order spherical harmonics.

Assuming the scene is an everyday environment with frequent interactions, it may experience geometric changes due to object displacements. We focus on modeling objectlevel movements and develop a decomposed representation that can serve as a strong geometric prior despite sparse observations, as shown in Fig. 2. We develop a lightweight pipeline that i) performs foundation model inference [14, 16] for dense local matches and instance identification, and ii) reuses these results for subsequent steps in instance association, coarse geometry estimation, and scene updating. Specifically, we robustly detect object-level changes (Sec. 3.2) and collect information for individual objects from the input image set (Sec. 3.3). We separately train Gaussian templates for the objects and static background (Sec. 3.4), such that blending them with the correct transform can best match the input images at the corresponding time step. While we incorporate strong cues of foundation models, they do not directly translate into stable 3D updates. Our aggregation strategy is critical for converting their perview predictions into coherent object-level templates. Overall, our framework can achieve high-quality reconstruction in temporally evolving scenes while suppressing prominent artifacts with minimal input and computation.

## 3.2. Change detection

We first find the exact camera positions of $\{ I ^ { i } \}$ t despite local changes, such that we can compare the observations from different time stamps against the initial Gaussian reconstruction $\mathcal { G } _ { 0 }$ . We use a robust hierarchical localization pipeline [33] and render the initial reconstruction in the same viewpoints $\{ \hat { I } ^ { i } \} _ { t }$

To decompose our scene representation into movable objects and static background, we detect temporal changes. We incorporate both semantic and photometric criteria to identify the differences between the rendered $\{ \hat { I } ^ { i } \}$ and the captured $\{ I ^ { i } \}$ t images. Semantic differences detect object-level changes despite lighting variations and other adversaries, and are measured by the cosine similarity of SAM features [14], similar to recent change detection methods [2, 12, 21]. Photometric differences are evaluated using the structural similarity index measure (SSIM), which can detect subtle object deviations not observable by SAM. The pixel-wise differences of the combined criteria are binarized by a scene-specific threshold chosen by the statistics [29], resulting in an initial pseudo mask for changed regions.

Note that we can robustly extract object-level changes from sparse input with the semantic masks. Specifically, we select the object region to be the set of SAM masks that sufficiently overlap with the pseudo masks while containing semantically dissimilar features compared to the initial image, as the result of change [12]. The aid of SAM masks effectively ignores differences due to floating artifacts in the rendering images and reliably extracts object regions. The resulting detected masks are dilated by 3 pixels to maintain sufficient information for 3D aggregation across multi-view observations, mitigating the impact of pixel-wise errors or slight misalignments. See the supplementary material for detailed implementation details on change detection. The overall pipeline produces stable outputs, even in the presence of potential inaccuracies as discussed in Sec. 4.

<!-- image-->  
Figure 2. Method overview. We propose an integrated pipeline to update an initial reconstruction given the collection of post-change captures. Our pipeline first estimates the camera poses of the input capture and compares them against renderings of the initial reconstruction in the same view to detect object-level changes. We aggregate detected objects from multiple viewpoints and timestamps to create 3D Gaussian templates, and finally update the temporal scenes by compositing the templates at their respective states with the background.

## 3.3. Object tracking and template reconstruction

After obtaining change masks, we match the changed objects within the individual images and extract initial 3D Gaussian templates for them. We then refine the relative transforms between the 3D templates and the image observations, which can serve as input to integrated optimization for 3D scene S. The entire process takes only 30 seconds to track and reconstruct templates for five discrete timesteps.

2D instance matching While one can associate object masks using low-level image features such as SIFT [20], our sparse setting makes it challenging to match small objects with few discriminative features. Also, objects are sometimes dynamic in appearance or geometry, which further complicates the process. To address these issues, we combine the strength of both dense geometric features from MASt3R [16] and semantic features obtained from SAM [14] in Sec. 3.2. We first match multi-view images incorporating MASt3R features [16] for all pairs of images within the same time stamp $\{ I ^ { i } \}$ t. Using the output, we build a graph where the nodes are object masks and the edges are matches with pairwise correspondences. By examining the graph structure, we can assign instance IDs to the matched components and filter out unmatched objects, such that we can naturally overcome artifacts. The intratimestep matching then provides reliable starting points for establishing matchings across different timesteps. We aggregate the SAM features within the object region, which have been computed to detect changes in Sec. 3.2, and build a matrix that records extensive pairwise cosine similarity of the aggregated features. Based on the matrix, we leverage Hungarian matching [26] to match object instances.

Gaussian object template extraction After associating 2D change masks, we are ready to build object-level Gaussian templates, which can be refined and function as a reusable geometric prior. We first decompose the initial Gaussian reconstruction $\mathcal { G } _ { 0 }$ into a set of template objects ${ \mathcal T } _ { 0 } = \{ o _ { 0 , k } \}$ and the background $B _ { 0 }$ . We formulate it as a segmentation problem for individual splats, and solve for the optimal label assignment as proposed in [35]. Objects in a later sequence do not exist in $\mathcal { G } _ { 0 } .$ , and we initialize the templates $\{ \mathcal { T } _ { t } | t > 0 \}$ using 3D point clouds estimated with MASt3R. Note that we reuse the extracted descriptors of MASt3R in previous stages and optimize per-view depth maps with fixed camera parameters from Sec. 3.2 to reconstruct 3D point clouds. The point clouds directly provide the positions and colors of the Gaussian splats, and we simply initialize uniform opacity, identity rotation, and uniform scaling as [6]. For the background, we maintain a single global set of Gaussians $B _ { 0 }$ . When occluded regions in the initial reconstruction become visible after changes, we augment this background representation by initializing the newly observed areas using point maps from MASt3R.

Gaussian object template tracking After initializing 3D templates, we deduce temporal states of the objects by tracking their movements and verifying consistency in 3D. For the object instances matched from different time steps, we compare their 3D overlap using robust point cloud registration. However, the MASt3R points or Gaussian reconstructions are incomplete and noisy with irregular density, and often cannot be registered using conventional point cloud pipelines such as ICP [3] or RANSAC-based approaches [7]. Instead, we establish correspondences by augmenting DINO features [28] to each point and apply a robust point cloud registration pipeline [42] to register templates. The registration yields 6DoF poses between pairs of template points $P _ { t  \tilde { t } , k } = \{ R _ { t  \tilde { t } , k } , T _ { t  \tilde { t } , k } \}$ such that we can assess the geometric consistency across the temporal track by thresholding with the Chamfer distance [5]. If the points are close enough, we avoid redundancy by selecting a single 3D template per matched instance, along with its relative transforms over time. Since we conservatively select templates, we can naturally represent an object under significant geometric variations as different instances without modifying the framework, as shown in Fig. 6. We further refine the shared template in the next stage. See the supplementary material for details on object tracking and object-level template reconstruction.

## 3.4. Long-term Gaussian splats optimization

While the selected templates can explain the other time steps with sufficient geometric overlap, they are derived from a single time step and can be noisy and incomplete, especially in different time steps with significant viewpoint changes. We aggregate the collection of observations and optimize the parameters of Gaussian splats (Sec. 3.1). The templates from initial time step can be transformed into a time step t using the registration parameters $P _ { 0  t , k } ~ =$ $\{ R _ { 0  t , k } , T _ { 0  t , k } \}$ from Sec. 3.3 as following:

$$
\begin{array} { r } { ( \mu _ { t , k } , R _ { t , k } , c _ { t , k } ) = ( \mu _ { 0 , k } R _ { 0  t , k } ^ { \top } + T _ { 0  t , k } ^ { \top } , } \\ { R _ { 0  t , k } R _ { 0 , k } , c _ { 0 , k } \mathcal { R } _ { \mathrm { S H } } ( R _ { 0  t , k } ) ^ { \top } ) . } \end{array}\tag{1}
$$

Here, $R _ { 0 , k }$ and $R _ { t , k }$ are rotation matrices corresponding to $q _ { 0 , k }$ and $q _ { t , k }$ respectively and SH coefficients are rotated via rotation operator $\mathcal { R } _ { \mathrm { S H } }$ [4]. In addition, we apply a temporal opacity filter per object $\mathcal { M } _ { t , o }$ such that transient objects become invisible (zero opacity). Since the 6DoF poses obtained from Sec. 3.3 are not precise at the pixel level, we additionally set the 6DoF poses of the object templates as an optimization parameter.

These strategies enable efficient modeling of long-term environments, but they also risk overfitting the initial assets to post-change views. To address this issue, we additionally leverage training camera poses used at the initial stages to render the scene and enforce consistency in the rendered images. Formally, we can render and update the image from the ith viewpoint at time t by defining the optimization problem as follows:

$$
\hat { I } _ { t } ^ { i } = \mathrm { R a s t e r i z e } ( \mu _ { t , k } , q _ { t , k } , s _ { t , k } , \mathcal { M } _ { t , o } \cdot \alpha _ { t , k } , c _ { t , k } ) ,\tag{2}
$$

$$
\begin{array} { r } { \{ \mu ^ { * } , q ^ { * } , s ^ { * } , \alpha ^ { * } , c ^ { * } , P ^ { * } \} = \arg \operatorname* { m i n } \mathcal { L } _ { \mathrm { p h o t o } } \Big ( \hat { I } _ { t } ^ { i } , I _ { t } ^ { i } \Big ) , } \end{array}
$$

where $I _ { t } ^ { i }$ contains both captured post-change images and renderings from the initial camera poses. For L, we use the standard L1 loss with D-SSIM loss that was used in the original implementation of 3DGS [11]. The background scene is initialized with $B _ { 0 }$ and similarly refined using all the information from different times. As the initial templates provide a reasonable approximation, 5000 iterations are sufficient to refine the parameters without densifying or cloning Gaussians, and we also skip the opacity resetting stages. Once optimized, our framework easily scales to multiple timesteps by simple transformations of template Gaussians.

## 4. Experiments

## 4.1. Datasets & baselines

Datasets We use a synthetic dataset from CL-NeRF [41], which contains three scenes captured at different timesteps: [WHITEROOM, KITCHEN, ROME]. Each timestep includes object-level sequential operations, such as addition, deletion, replacement, and movement. However, the motions are simple and do not exhibit diverse variations between objects in different steps. We additionally captured challenging real-world scenes, where objects may abruptly reappear in different configurations. The dataset consists of image collections captured in five scenes at 5 different timesteps: [CAFE, DININGROOM, HALL, LAB, LIVINGROOM].

Baselines We compare our work to recent NeRFs and 3DGS variants. First, we evaluate against the original (1) 3DGS [11] using all images for different timesteps as a reference. We further compare with (2) InstantSplat [6], a fewshot reconstruction method applied independently at each timestep. To account for frameworks explicitly modeling dynamic scenes, we include (3) 4DGS [40] and (4) Neural Scene Chronology (NSC) [18]. In addition, we compare (5) 3DGS-CD, which explicitly detects and updates object-level changes. Finally, we evaluate against continual learning frameworks, including (6) CL-NeRF [41] and (7) CL-Splats [1]. The works and their capabilities are also summarized in Tab. 1. Refer to the supplement for details.

## 4.2. Comparative studies

We evaluate our method and baselines on novel-view synthesis tasks across scenes with multiple timesteps. For every setting, we use three images at each timestep from various angles to capture scene changes. Our framework largely outperforms baselines both qualitatively and quantitatively, as demonstrated in Tab. 2 and Fig. 3. It successfully reconstructs diverse object-level changes while remaining robust to limitations imposed by sparse views. In particular, InstantSplat [6] is designed for fast and lightweight reconstruction specifically in a few-shot setting, and thus cannot maintain its performance on a free-viewpoint setting covering the full scene. 4DGS [40] and NSC [18] struggle to precisely model the discrete changes, such as added or removed objects. The snapshots in Fig. 3 also include overly smooth results in the change regions.

<!-- image-->  
Figure 3. Qualitative comparisons of our method. We illustrate the results of our method using the CL-NeRF dataset and our dataset.

While 3DGS-CD [21] also quickly handles object-level placement changes, our approach consistently outperforms, as our approach better accounts for added and removed objects. Recent continual-learning frameworks also struggle to address spatio-temporally sparse settings. CL-NeRF [41] performs well on synthetic datasets but cannot track complex real-world changes. Also, the implicit representation of CL-NeRF occasionally results in degraded sharpness, as shown in the appendix. The optimization of CL-Splats [1] fails to maintain its stability in the sparse-view inputs, as its mask estimation becomes unreliable. The effects can also be observed in the LAB scene (4th row) in Fig. 3.

We also report the total time spent reconstructing different timesteps in Tab. 2. In our framework, the total processing time is approximately 6.5 minutes (2.5 min for change detection, 0.5 min for instance matching, 3.5 min for 5000 iteration updates), measured on an NVIDIA RTX 4090. In conclusion, while prior methods either oversmooth dynamic changes or fail under sparse inputs, our framework achieves accurate and efficient reconstruction of evolving scenes with strong object-level consistency.

## 4.3. Ablation studies

We conducted ablation studies on various components of the method, as demonstrated in Tab. 3. Since our components are primarily designed to improve the modeling of changing objects rather than the overall quality of the initially reconstructed scenes, the image quality metric itself did not fully reveal significant improvement. Accordingly, we additionally demonstrate the qualitative comparisons focusing on the change regions Fig. 4.

<table><tr><td rowspan="2">Method</td><td colspan="4">CL-NeRF dataset (synthetic)</td><td colspan="4">Our dataset (real)</td></tr><tr><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>Time â</td><td>PSNR â</td><td>SSIM â</td><td>LPIPS â</td><td>Time â</td></tr><tr><td>3DGS [11]</td><td>24.53</td><td>0.789</td><td>0.392</td><td>6 min</td><td>19.56</td><td>0.857</td><td>0.272</td><td>8 min</td></tr><tr><td>InstantSplat [6]</td><td>18.98</td><td>0.601</td><td>0.466</td><td>3 min</td><td>19.36</td><td>0.785</td><td>0.343</td><td>3 min</td></tr><tr><td>4DGS [40]</td><td>26.13</td><td>0.786</td><td>0.411</td><td>24 min</td><td>21.49</td><td>0.850</td><td>0.322</td><td>29 min</td></tr><tr><td>NSC [18]</td><td>20.63</td><td>0.698</td><td>0.465</td><td>&gt;10 hours</td><td>17.52</td><td>0.755</td><td>0.439</td><td>&gt;10 hours</td></tr><tr><td>3DGS-CD [21]</td><td>23.61</td><td>0.727</td><td>0.437</td><td>2 min</td><td>20.94</td><td>0.774</td><td>0.348</td><td>2 min</td></tr><tr><td>CL-NeRF [41]</td><td>25.53</td><td>0.730</td><td>0.465</td><td>2 hours</td><td>20.95</td><td>0.815</td><td>0.379</td><td>2 hours</td></tr><tr><td>CL-Splats [1]</td><td>25.84</td><td>0.772</td><td>0.416</td><td>3 min</td><td>21.12</td><td>0.829</td><td>0.312</td><td>3 min</td></tr><tr><td>LTGS (ours)</td><td>27.17</td><td>0.795</td><td>0.376</td><td>6 min</td><td>23.46</td><td>0.889</td><td>0.230</td><td>7 min</td></tr></table>

Table 2. Quantitative comparisons on CL-NeRF dataset and our dataset. We compared our method against NeRF-based and Gaussian splatting variants. The best results are highlighted in bold.

<table><tr><td>Configuration</td><td>PSNR â SSIM â</td><td>LPIPS â</td></tr><tr><td>w/o Obj. Tracking</td><td>23.26 0.885</td><td>0.234</td></tr><tr><td>w/o Pose Opt.</td><td>23.33 0.886</td><td>0.232</td></tr><tr><td>w/o BG Init.</td><td>23.29 0.885</td><td>0.233</td></tr><tr><td>w/o Training View</td><td>23.11 0.885</td><td>0.240</td></tr><tr><td>Full (ours)</td><td>23.46</td><td>0.889 0.230</td></tr></table>

Table 3. Ablation study. We demonstrate the effect of different optimization configurations.  
<!-- image-->

<!-- image-->  
Figure 4. Visual comparison of ablation study. We visualize the effect of each components listed in Tab. 3.

We first tested the effect of instance matching, where we removed the template association step and built new objectlevel Gaussians at every time step. Without using object Gaussian templates as a reusable prior, we could not handle sparse view limitations, leaving traces of removed objects as shown in Fig. 4. 6DoF pose updates also increased the reconstruction quality, as they account for pixel-level errors from subtle pose errors after registration.

While our primary ablations focus on reconstructing dynamic objects, we also examined the impact of background initialization. Specifically, when objects disappear and leave previously occluded regions visible, initializing these empty areas with MASt3R [16] point clouds alleviates background artifacts. As shown in Fig. 4, the joint use of global background Gaussians with this initialization strategy further reduces artifacts after object removal. Including training views of initial timesteps also enhanced the quality, as reported in Tab. 3. As we only leverage a few-shot images, some unseen regions, such as under the table or back of the chair in Fig. 4, include sharp artifacts, which degrade the rendering of several viewpoints. We verified that each component clearly affected the enhancement of both object and background reconstructions without compromising the initial reconstruction.

<!-- image-->  
Lab  
Figure 5. Object template visualization. We sampled several captures from the initial state and post-change captures and corresponding object-level Gaussian templates.

## 4.4. Performance Analysis

Object-level reconstruction. After optimization, our framework produces object-level reconstruction, where learned Gaussian templates can be directly leveraged for scene composition and temporal reasoning. To illustrate this, we visualize the optimized object-level Gaussian templates in Fig. 5. We selected some viewpoints from initial and post-change captures, and rasterized the object-level Gaussians to the corresponding viewpoints. Our framework effectively disentangles individual objects from the scene while preserving consistent geometry and appearance across time steps. Notably, even with few-shot observations, the optimized object templates exhibit well-defined shapes without severe artifacts along object boundaries.

<!-- image-->  
(b) Object Gaussian Templates  
Figure 6. Challenging real-world scenarios. We demonstrate (a) reconstruction results with articulations and (b) a visualization of object-level reconstructions.

This suggests that the optimization effectively integrates multi-timestep cues into coherent object-level representations. Such clean object templates also highlight the potential of our method for modeling object-level changes in scenes with longer temporal variances.

Non-rigid transformations or articulations. We further tested our framework on more challenging setups as shown in Fig. 6. In real-world scenarios, object-level changes often involve non-rigid transformations or articulations. Our method handles such cases by defining separate object Gaussian templates for objects in different articulation states. Although these templates are not rigidly tracked, they are all identified as a single object instance due to our robust 2D matching pipeline. We visualize the reconstructed scene with temporal variations and object-level reconstructions across different states for the MAC scene from the world-across-time (WAT) dataset introduced in CLNeRF [45]. Thus, our method provides a principled way to represent objects in different states through independent templates. Moreover, the tracked object templates can be combined with recent works [13], modeling object articulation for more detailed tracking, which is left as future work.

Robustness to errors of foundation models We conducted experiments to verify the robustness of our framework against potential errors occurred from foundation models. Specifically, we injected random noise with varying standard deviations into the MASt3R [16] descriptors and SAM [14] embeddings as demonstrated in Fig. 7. Even with the perturbed MASt3R descriptors and SAM embeddings, our framework achieved reliable reconstruction without severe degradation, with stable instance matching and template reconstruction. The contour-based aggregation helps smooth out local perturbations, and the graph-based instance matching across multiple timesteps further enhances robustness to errors from pretrained models. Moreover, the errors in reconstructed Gaussian templates caused by noisy MASt3R descriptor can be compensated by the proposed long-term optimization process in Sec. 3.4.

<!-- image-->

<!-- image-->  
(b) PSNR comparison  
Figure 7. Robustness evaluation. We demonstrate (a) the effects of varying noise levels on components of our pipeline, and (b) the reconstruction results measured in PSNR.

## 5. Conclusion

We present LTGS, an integrated framework for modeling scenes with long-term changes given spatiotemporally sparse images. Our strategy stably builds and exploits object-centric templates under the challenging setup. Several comparative studies and ablation studies have verified that the combination of our components significantly outperforms the baselines. We further verified our frameworks to be applicable to several extensions, such as object-level reconstruction and reconstructing more challenging setups with non-rigid transformations or articulations. In conclusion, we believe this framework offers a promising foundation for building a coherent structural representation that is reusable for a long temporal horizon. As our framework primarily targets scenes with geometric variations, it poses challenges for scenes with significant lighting changes or severe appearance changes with fixed geometries, such as monitors, if these are not captured as changes. We leave the problem of modeling the heavy lighting changes with shadows for future work.

Acknowledgements This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2023-00216821, Development of Beyond X-verse Core Technology for Hyper-realistic interactions by Synchronizing the Real World and Virtual Space) and Creative-Pioneering Researchers Program through Seoul National University.

## References

[1] Jan Ackermann, Jonas Kulhanek, Shengqu Cai, Xu Haofei, Marc Pollefeys, Gordon Wetzstein, Leonidas Guibas, and Songyou Peng. Cl-splats: Continual learning of gaussian splatting with local optimization. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025. 2, 3, 5, 6, 7, 14, 16

[2] Aikaterini Adam, Konstantinos Karantzalos, Lazaros Grammatikopoulos, and Torsten Sattler. Has anything changed? 3d change detection by 2d segmentation masks. arXiv preprint arXiv:2312.01148, 2023. 3

[3] Paul J Besl and Neil D McKay. Method for registration of 3-d shapes. In Sensor fusion IV: control paradigms and data structures, pages 586â606. Spie, 1992. 5

[4] Jiahao Chang, Yinglin Xu, Yihao Li, Yuantao Chen, Wensen Feng, and Xiaoguang Han. Gaussreg: Fast 3d registration with gaussian splatting. In European Conference on Computer Vision, pages 407â423. Springer, 2024. 5

[5] Haoqiang Fan, Hao Su, and Leonidas J. Guibas. A point set generation network for 3d object reconstruction from a single image. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 5

[6] Zhiwen Fan, Kairun Wen, Wenyan Cong, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic, Marco Pavone, Georgios Pavlakos, Zhangyang Wang, and Yue Wang. Instantsplat: Sparse-view gaussian splatting in seconds, 2024. 2, 3, 4, 5, 7, 16

[7] Martin A Fischler and Robert C Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6):381â395, 1981. 5

[8] Chamuditha Jayanga Galappaththige, Jason Lai, Lloyd Windrim, Donald Dansereau, Niko Sunderhauf, and Dimity Miller. Multi-view pose-agnostic change localization with zero labels. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 11600â11610, 2025. 3

[9] Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. Monocular dynamic view synthesis: A reality check. In NeurIPS, 2022. 1, 2

[10] Guangcong, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu. Sparsenerf: Distilling depth ranking for few-shot novel view synthesis. IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 2, 3

[11] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 1, 2, 5, 7, 13, 14, 16

[12] Jae-Woo Kim and Ue-Hwan Kim. Towards generalizable scene change detection. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 24463â24473, 2025. 3, 11, 12

[13] Seungyeon Kim, Junsu Ha, Young Hun Kim, Yonghyeon Lee, and Frank C Park. Screwsplat: An end-to-end method for articulated object recognition. arXiv preprint arXiv:2508.02146, 2025. 8

[14] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Â´ Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 3, 4, 8, 11, 12

[15] Jonas Kulhanek, Songyou Peng, Zuzana Kukelova, Marc Pollefeys, and Torsten Sattler. WildGaussians: 3D gaussian splatting in the wild. NeurIPS, 2024. 2

[16] Vincent Leroy, Yohann Cabon, and Jerome Revaud. Grounding image matching in 3d with mast3r, 2024. 3, 4, 7, 8, 11, 12, 14

[17] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video synthesis from multi-view video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5521â5531, 2022. 2

[18] Haotong Lin, Qianqian Wang, Ruojin Cai, Sida Peng, Hadar Averbuch-Elor, Xiaowei Zhou, and Noah Snavely. Neural scene chronology. In CVPR, 2023. 2, 5, 7, 16

[19] Jiaqi Lin, Zhihao Li, Xiao Tang, Jianzhuang Liu, Shiyong Liu, Jiayue Liu, Yangdi Lu, Xiaofei Wu, Songcen Xu, Youliang Yan, and Wenming Yang. Vastgaussian: Vast 3d gaussians for large scene reconstruction. In CVPR, 2024. 2

[20] David G Lowe. Distinctive image features from scaleinvariant keypoints. International journal of computer vision, 60(2):91â110, 2004. 4

[21] Ziqi Lu, Jianbo Ye, and John Leonard. 3dgs-cd: 3d gaussian splatting-based change detection for physical object rearrangement. IEEE Robotics and Automation Letters, 2025. 2, 3, 6, 7, 16

[22] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In 3DV, 2024. 2

[23] Ricardo Martin-Brualla, Noha Radwan, Mehdi S. M. Sajjadi, Jonathan T. Barron, Alexey Dosovitskiy, and Daniel Duckworth. NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections. In CVPR, 2021. 2

[24] Ben Mildenhall, Pratul P. Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (TOG), 2019. 13

[25] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 1, 13

[26] James Munkres. Algorithms for the assignment and transportation problems. Journal of the society for industrial and applied mathematics, 5(1):32â38, 1957. 4, 12

[27] Michael Niemeyer, Jonathan T. Barron, Ben Mildenhall, Mehdi S. M. Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2022. 2, 3

[28] Maxime Oquab, Timothee Darcet, Theo Moutakanni, Huy V. Â´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,

Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023. 3, 5

[29] Nobuyuki Otsu et al. A threshold selection method from gray-level histograms. Automatica, 11(285-296):23â27, 1975. 3, 11

[30] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-NeRF: Neural Radiance Fields for Dynamic Scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. 1, 2

[31] Ragav Sachdeva and Andrew Zisserman. The change you want to see, 2022. 3

[32] Ken Sakurada, Mikiya Shibuya, and Weimin Wang. Weakly supervised silhouette-based semantic scene change detection. In 2020 IEEE International conference on robotics and automation (ICRA), pages 6861â6867. IEEE, 2020. 3

[33] Paul-Edouard Sarlin, Cesar Cadena, Roland Siegwart, and Marcin Dymczyk. From coarse to fine: Robust hierarchical localization at large scale. In CVPR, 2019. 3, 12

[34] Johannes Lutz Schonberger and Jan-Michael Frahm. Â¨ Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 3, 13

[35] Qiuhong Shen, Xingyi Yang, and Xinchao Wang. Flashsplat: 2d to 3d gaussian splatting segmentation solved optimally. European Conference of Computer Vision, 2024. 3, 4, 12

[36] Brandon Smart, Chuanxia Zheng, Iro Laina, and Victor Adrian Prisacariu. Splatt3r: Zero-shot gaussian splatting from uncalibrated image pairs. arXiv preprint arXiv:2408.13912, 2024. 3

[37] Robert Tarjan. Depth-first search and linear graph algorithms. SIAM journal on computing, 1(2):146â160, 1972. 11

[38] Mikaela Angelina Uy, Ricardo Martin-Brualla, Leonidas Guibas, and Ke Li. Scade: Nerfs from space carving with ambiguity-aware depth estimates. In Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 3

[39] Yifan Wang, Peishan Yang, Zhen Xu, Jiaming Sun, Zhanhua Zhang, Yong Chen, Hujun Bao, Sida Peng, and Xiaowei Zhou. Freetimegs: Free gaussian primitives at anytime anywhere for dynamic scene reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21750â21760, 2025. 2

[40] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20310â 20320, 2024. 1, 2, 5, 7, 14, 16

[41] Xiuzhe Wu, Peng Dai, Weipeng Deng, Handi Chen, Yang Wu, Yan-Pei Cao, Ying Shan, and Xiaojuan Qi. Cl-nerf: continual learning of neural radiance fields for evolving scene representation. Advances in Neural Information Processing Systems, 36:34426â34438, 2023. 2, 5, 6, 7, 14, 16

[42] H. Yang, J. Shi, and L. Carlone. TEASER: Fast and Certifiable Point Cloud Registration. IEEE Trans. Robotics, 2020. 5

[43] Lin Zeng, Boming Zhao, Jiarui Hu, Xujie Shen, Ziqiang Dang, Hujun Bao, and Zhaopeng Cui. Gaussianupdate: Continual 3d gaussian splatting update for changing environments. arXiv preprint arXiv:2508.08867, 2025. 2

[44] Jiaxin Zhang, Junjun Jiang, Youyu Chen, Kui Jiang, and Xianming Liu. Cob-gs: Clear object boundaries in 3dgs segmentation based on boundary-adaptive gaussian splitting. In CVPR, 2025. 3

[45] Matthias Muller Zhipeng Cai. Clnerf: Continual learning Â¨ meets nerf. In ICCV, 2023. 2, 8

[46] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian splatting. In European conference on computer vision, pages 145â163. Springer, 2024. 3

# LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates

Supplementary Material

## A. Implementation details

## A.1. Change detection

We detect fine object-level changes in 2D by using the semantic prior of the SAM model [14]. We prepare a pair of images $( I _ { t } ^ { i } , \hat { I } _ { t } ^ { i } )$ from captured and rendered images from the corresponding viewpoints. To find abstract differences between the rendered and captured images, we extract features from the segmentation model for both images. We used features obtained from the pretrained SAM encoder and interpolated them to match the original image resolution, after which we computed pairwise cosine similarities for comparison. We additionally use SSIM to capture structural differences, where using only semantic cosine similarity struggles to find slightly deviated objects.

To obtain coarse change masks, we need a binarization by thresholding the obtained differences. Since the binary coarse mask is highly sensitive to manually defined thresholds, which vary significantly across scenes, we adopt Otsuâs method [29] to automatically determine the threshold $\tau _ { c o s }$ . We obtain the coarse binary masks $\mathcal { M } _ { t , \mathrm { c o a r s e } } ^ { i }$ as follows:

$$
\begin{array} { r l } & { M _ { t , \mathrm { c o a r s e } } ^ { i } = \gamma \cdot \mathrm { c o s } ( \mathcal { E } ( I _ { t } ^ { i } ) , \mathcal { E } ( \hat { I } _ { t } ^ { i } ) ) } \\ & { \qquad + \left( 1 - \gamma \right) \cdot \mathrm { S S I M } ( I _ { t } ^ { i } , \hat { I } _ { t } ^ { i } ) \leq \tau _ { \mathrm { c o s } } , } \end{array}\tag{3}
$$

where $\mathcal { E }$ denotes the feature extractor of $\mathrm { S A M } . \gamma = 0 . 7$ was used throughout our experiments.

After obtaining coarse change regions, we extract finegrained object-level change masks to model instance-wise changes. Since precise object-level changes are difficult to capture using the coarse stages described above, we leverage the automatic mask generation from SAM [14] within the coarse binary mask $M _ { t , \mathrm { c o a r s e } } ^ { i } .$ . For each generated mask, we first calculate the intersection-over-union (IoU) between its region and the coarse binary mask. We further compare the cosine similarity between the accumulated features within those regions in the extracted features of the rendered and captured images. The fine object-level masks can be obtained as follows:

$$
\begin{array} { r l } & { \mathcal { O } _ { t } ^ { i } = \bigcup _ { k } \Big \{ o _ { t , k } ^ { i } \Big | \mathrm { I o U } \big ( o _ { t , k } ^ { i } , M _ { t , \mathrm { c o a r s e } } ^ { i } \big ) \geq \tau _ { \mathrm { I o U } } \wedge } \\ & { \qquad \quad \cos \big ( \Phi \big ( o _ { t , k } ^ { i } , \mathcal { E } \big ( I _ { t } ^ { i } \big ) \big ) , \Phi \big ( o _ { t , k } ^ { i } , \mathcal { E } \big ( \hat { I } _ { t } ^ { i } \big ) \big ) \big ) \leq \tau _ { \mathrm { c o s } } \Big \} , } \end{array}\tag{4}
$$

where $o _ { t , k } ^ { i }$ denotes the kth object mask generated by automatic mask generation, and $\begin{array} { r l } { \Phi ( m , X ) } & { { } = } \end{array}$ $\begin{array} { r } { \frac { \overline { { 1 } } } { | m | } \sum _ { p \in \Omega } m ( p ) X ( p ) } \end{array}$ denotes the average pooling operator of feature X within the interior â¦ of mask m. We select and keep only the highly overlapped and semantically different masks following [12]. We further removed masks occupying only small regions to prevent noise, and we dilated the change masks to address pixel-wise errors.

## A.2. Object tracking

We provide additional details of our pipeline to associate with changed instances. Given 2D object-level change masks from Sec. 3.2 of the main paper, we leverage both visual feature and SAM feature matching to associate 2D change masks. As mentioned in Sec. 3.3 of the main paper, we separate the instance matching into intra-timestep matching and cross-timestep matching. Let $\mathcal { O } _ { t } ^ { i } = \{ o _ { t , 1 } ^ { i } , o _ { t , 2 } ^ { i } , . . . , o _ { t , N _ { o } } ^ { i } \}$ be the set of object-level masks of ith viewpoint at timestep t. We first compute pairwise matches across images $I _ { t } ^ { i }$ and $I _ { t } ^ { j }$ using MASt3R within the same timestep. Given the extracted MASt3R descriptors $\mathbf { d } _ { t , k } ^ { i }$ and ${ \bf d } _ { t , l } ^ { j }$ for object $o _ { t , k } ^ { i }$ and $o _ { t , l } ^ { j }$ respectively, we find descriptor matches $\mathcal { M } _ { t } ^ { ( i , k )  ( j , l ) }$ within object change masks as follows:

$$
\begin{array} { r l } & { \mathcal { M } _ { t } ^ { ( i , k )  ( j , l ) } = \operatorname { m a t c h } ( \mathbf { d } _ { t , k } ^ { i } , \mathbf { d } _ { t , l } ^ { j } ) , } \\ & { \quad o _ { t , k } ^ { i } \in \mathcal { O } _ { t } ^ { i } , o _ { t , l } ^ { j } \in \mathcal { O } _ { t } ^ { j } , i \neq j . } \end{array}\tag{5}
$$

Note for matching, we follow the fast reciprocal matching procedure from MASt3R [16]. Based on these matches, we construct a graph $G _ { t } = ( N _ { t } , E _ { t } )$ as:

$$
\begin{array} { c } { N _ { t } = \bigcup _ { i } \mathcal { O } _ { t } ^ { i } , } \\ { E _ { t } = \big \{ ( o _ { t , k } ^ { i } , \ o _ { t , l } ^ { j } , | { \mathcal { M } _ { t } ^ { ( i , k )  ( j , l ) } } | ) \big | i \neq j \big \} . } \end{array}\tag{6}
$$

where the nodes $N _ { t }$ are objects and the edge weight $E _ { t }$ encodes the total number of matches between every pair of objects $o _ { t , k } ^ { i }$ and $o _ { t , l } ^ { j } .$ We then cluster the graph using the depth-first search (DFS) algorithm [37] to identify connected components, where each component corresponds to a unique global object identity. Here, we retain an edge only if the number of matches exceeds a threshold $\tau _ { m a t c h }$ Based on these clustering results, we assign instance IDs to the matched components and filter out unmatched objects for consistency. This filtering strategy gives robustness to detected instances that are inaccurate due to the artifacts in rendered images.

After obtaining object-level matches for every sequence within an identical timestep, it is essential to associate object masks across different timesteps. For each object $o _ { t , i } ^ { i }$ k and $o _ { \tilde { t } , l } ^ { j }$ where t and $\tilde { t }$ are the set of target times after intratimestep matching, we accumulate the SAM [14] features in the object region, and build a matrix $\mathbf { S } _ { k  l }$ that contains cosine similarity among every pair as follows:

$$
\mathbf { S } _ { k  l } = \cos \big ( \Phi ( o _ { t , k } ^ { i } , \mathcal { E } ( I _ { t } ^ { i } ) ) , \Phi ( o _ { \tilde { t } , l } ^ { j } , \mathcal { E } ( I _ { \tilde { t } } ^ { j } ) ) \big ) .\tag{7}
$$

Based on the matrix, we leverage Hungarian matching [26] to solve an optimal assignment problem between the instances as $\begin{array} { r } { \pi ^ { * } = \arg \operatorname* { m a x } _ { \pi } \sum _ { k } \mathbf { S } _ { k  \pi ( k ) } } \end{array}$ , where $\pi ( k )$ denotes the matched object in timestep tË. After semantic matching, we filter out false pairs for those with the cosine similarities lower than $\tau _ { c o s }$ defined in Eq. (3) and Eq. (4). Note that we conduct this process identically for every possible timestep pair for $t \in [ 0 , T ]$

## A.3. Object Gaussian Template reconstruction

Given the tracked object masks $\begin{array} { r l r } { M } & { { } = } & { \{ M _ { t } ^ { i } | i \quad = } \end{array}$ $1 , . . . , N _ { v } ; t ~ = ~ 0 , . . . , T \}$ , we provide additional details of the construction and initialization of object-level Gaussian Splats. Here, we define the total number of instances at the initial timestep as E. For the objects that emerge in initial reconstruction, we separate those using the optimal label assignment problem introduced in FlashSplat [35]. For our task, the problem is defined as follows:

$$
\operatorname* { m i n } _ { \{ P _ { k } \} } \mathcal { F } = \sum _ { i } \left| \sum _ { k } P _ { k } \alpha _ { k } T _ { k } - M _ { 0 } ^ { i } \right| ,\tag{8}
$$

where $\alpha _ { k } , T _ { k }$ each denotes the alpha value and transmittance during volume rendering, and $P _ { k }$ denotes the per-Gaussian 3D label. Among the $P _ { k }$ , index 0 corresponds to the background, while the remaining indices correspond to the foreground. The above equation solves the problem of assigning the 3D label $P _ { k }$ by volume rendering them to the image domain to match the given multiview masks at t = 0. Specifically, we use the majority voting algorithm as follows:

$$
\begin{array} { l } { { \displaystyle P _ { k } = \arg \operatorname* { m a x } _ { n \in \{ 0 , m \} } A _ { n } , } } \\ { { \displaystyle A _ { m } = \sum _ { i } \alpha _ { k } T _ { k } \mathbb { 1 } ( M _ { 0 } ^ { i } , m ) , } } \\ { { \displaystyle A _ { 0 } = \sum _ { i } \sum _ { e \neq m } \alpha _ { k } T _ { k } \mathbb { 1 } ( M _ { 0 } ^ { i } , e ) , } } \end{array}\tag{9}
$$

where $\mathbb { 1 } ( M _ { 0 } ^ { i } , m )$ denotes the indicator function which is equal to 1 if the pixel in mask $M _ { 0 } ^ { i }$ belongs to object m, and 0 otherwise. Eq. (9) solves the assignment problem by allocating the label that maximizes the weighted contribution of Gaussians within the object mask regions. Please refer to the original FlashSplat [35] paper regarding the details and the derivation. We additionally filter Gaussians that are out of the object mask region after directly projecting the centers to remove floating artifacts.

After registration and geometric verification as presented in Sec. 3.3 of the main paper, we initialize Gaussians for objects that do not exist in the initial reconstruction. We first extract point clouds for new objects from the global scene reconstruction of MASt3R [16] with estimated poses from the hierarchical localization pipeline [33]. In the original implementation of MASt3R, camera parameters were optimized jointly with per-view depth maps and global scales. We modify the optimization loop to operate only on depth maps with scale and offset parameters, while the camera poses remain fixed. To reduce noise, we retain only the point clouds with per-pixel confidence values greater than 1.5, and we randomly downsample the point cloud by a factor of 4, as per-pixel point clouds are overly dense, which is inefficient for optimization.

## A.4. Hyperparameters

In this section, we discuss how key hyperparameters are set and evaluate their influence on performance. We analyze the effect of the $\tau _ { c o s }$ in Eq. (3) and Eq. (4), and the connectivity threshold $\tau _ { m a t c h }$ in Sec. A.2. Additionally, we evaluate the geometric verification threshold in Sec. 3.3 of the main paper, where we compare the chamfer distance between the two Gaussian object templates to $\tau _ { o v e r l a p }$ to examine the geometric consistency across the temporal track. We demonstrate the results of variations on above hyperparameters in Tab. 4.

For the change detection, we set $\tau _ { c o s } = 0 . 9$ in Eq. (3) and Eq. (4), which were used throughout our experiments. We sweep the values between [0.8, 0.95], referring to the cosine similarity threshold in [12], originally set to 0.88. We set $\tau _ { m a t c h } = 5 0$ , considering that higher threshold values tend to neglect small objects while tracking. In geometric verification stage, $\tau _ { o v e r l a p } = 0 . 1 5$ worked well to distinguish the geometrically matched objects. Severely lower $\tau _ { o v e r l a p }$ interrupts the object from being rigidly tracked.

To initialize Gaussian primitives, we use the dense point cloudâs position and its corresponding color as the initial position and color. We convert the RGB color into spherical harmonics (SH) coefficients and initialize higher-order SH components with zeros. We use low initial opacity values as Î± = 0.1 for all Gaussians. Rotations are initialized as identity quaternions, i.e., $q = ( 1 , 0 , 0 , 0 )$ for all Gaussians, while the scales are determined by the pairwise squared distance as done in the original 3DGS [11]. For refinement, we update the Gaussian Splats for 5000 iterations using the same learning rate as done in the official 3DGS [11] implementation. We skipped several techniques that were used in the original 3DGS such as opacity resetting, cloning and pruning operations to preserve the original reconstruction.

<table><tr><td> $\tau _ { c o s }$ </td><td>PSNR</td><td> $\tau _ { m a t c h }$ </td><td>PSNR</td><td> $\tau _ { o v e r l a p }$ </td><td>PSNR</td></tr><tr><td>0.80</td><td>22.79</td><td>10</td><td>23.40</td><td>0.05</td><td>23.29</td></tr><tr><td>0.85</td><td>22.76</td><td>50</td><td>23.43</td><td>0.1</td><td>23.39</td></tr><tr><td>0.90</td><td>23.43</td><td>100</td><td>23.33</td><td>0.15</td><td>23.43</td></tr><tr><td>0.95</td><td>23.33</td><td>200</td><td>23.22</td><td>0.20</td><td>23.41</td></tr></table>

Table 4. Effects of the hyperparameter.

<!-- image-->

<!-- image-->

(a) Examples of Capture  
<!-- image-->  
(b) Initial Reconstruction

<!-- image-->  
Figure 8. Examples of our datasets. (a) Illustration of long-term captures of our dataset for each scene. (b) Initial reconstruction of Gaussian splats at t = 0 and tracked instances at the initial timestep.

## B. Datasets

In this section, we provide some additional details for the datasets that we have introduced in our main manuscript. We casually captured video sequences using Galaxy S24 with fixed focal lengths and manually changed several objects between the captures for 5 sequences. After converting video frames to images, we regularly sampled 300-400 images for initial reconstruction to cover the scene of interest, and obtain camera poses using COLMAP SfM [34]. Fig. 8 illustrates the example of our datasets for every scene, captured from different timesteps. In total, our dataset contains five scenes that cover a diverse set of indoor spaces (Cafe, Diningroom, Livingroom, Hall, Lab). We additionally visualized the initial reconstruction of sampled scenes, with tracked instances using our pipeline, which serve as the initial object Gaussian templates before refinement. For evaluation, we selected every 8th frame following the conventional evaluation protocol of neural rendering [24, 25].

## C. Additional performance analysis

## C.1. Lighting changes

To ensure robust change estimation, we use semantic differences and structural differences rather than directly comparing the RGB values. We validate our frameworkâs robustness against varying illumination by applying different exposure, tone, and contrast curves to our dataset, as demonstrated in Fig. 9. To account for the exposure changes in the different input images, we used the exposure compensation provided in the official 3DGS [11] implementation. As a result, our framework successfully isolates object-level modifications without being degraded by global illumination shifts, maintaining high rendering fidelity.

## C.2. Geometry estimation errors

The robustness experiments in the main paper primarily analyze descriptor noise, related to the matching error. To further simulate the actual estimation failures, we conduct an additional analysis by assuming errors in depth maps and camera poses during reconstruction, which closely mimics real-world failure scenarios, as depicted in Fig. 10. While moderate noise is refined during the final refinement step, MASt3R [16] point clouds with severe errors and unstable registration are naturally treated as separate instances and refined independently. Consequently, this isolation strategy prevents initial pose or depth inaccuracies from corrupting the global scene geometry, effectively absorbing the errors to yield coherent structural reconstructions.

## C.3. Lightweight representation capability

In our main experiments, we utilize 3 images per timestep across 4â5 timesteps per scene. To further evaluate the impact of input sparsity, we conduct an additional performance analysis by varying the number of images per timestep on the LIVINGROOM scene, as detailed in Tab. 5. As a reference baseline, we measured the performance of standard 3DGS [11] using 100 densely sampled images per timestep. Notably, our approach achieves comparable reconstruction quality using only sparse updates, eliminating the need for dense image captures.

<!-- image-->  
Figure 9. Different lighting results with estimated change.

<!-- image-->  
Figure 10. Geometry estimation errors and rendering results.

Although we incorporate pretrained modules, our pipeline remains lightweight by leveraging reusable priors from the existing scene. As demonstrated in Tab. 6, our framework effectively models dynamic scenes while maintaining real-time rendering performance after optimization. Furthermore, it significantly reduces the memory footprint required for long-term environment maintenance, keeping memory usage comparable to the initial static 3DGS [11]. Ultimately, these results demonstrate that our updating mechanism not only renders photorealistic images but also provides a highly scalable and memory-efficient solution for long-term scene representations.

## D. Additional comparative studies

## D.1. Additional qualitative comparisons

We present additional qualitative results for both CL-NeRF [41] and our datasets in Fig. 11. We illustrate scenes that are not covered in Fig. 3 of the main manuscript. By incorporating reusable priors into scene representations, we achieve a notable reduction of artifacts, particularly in under-constrained regions where other baselines often struggle. Compared to the baselines, our method produces cleaner geometry and more photorealistic synthesis under novel viewpoints. This improvement highlights the effectiveness of our method in reducing ambiguity and enabling stable reconstructions across diverse scenes.

## D.2. Per-scene quantitative comparisons

We provide the evaluation results of all scenes in terms of PSNR, SSIM, and LPIPS. As demonstrated in Tab. 7, Tab. 8 and Tab. 9, our framework achieves the best results for most scenes.

<table><tr><td># image</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>3DGS [11] (~100)</td></tr><tr><td>PSNR</td><td>23.06</td><td>23.46</td><td>23.92</td><td>24.21</td><td>24.41</td><td>25.66</td></tr><tr><td>Time</td><td>6m 2s</td><td>7m 11s</td><td>8m 42s</td><td>10m 4s</td><td>11m 20s</td><td>31m 32s</td></tr><tr><td>Peak VRAM</td><td>7.8 GB</td><td>8.7 GB</td><td>9.2 GB</td><td>10.3 GB</td><td>11.7 GB</td><td>8.3 GB</td></tr></table>

Table 5. Performance analysis for different number of images.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3DGS [11] (Single time)</td><td rowspan=1 colspan=1>4DGS [40]</td><td rowspan=1 colspan=2>CL-Splats [1]LTGS (Ours)</td></tr><tr><td rowspan=1 colspan=1>MemoryFPS</td><td rowspan=1 colspan=1>125.8 MB190.3</td><td rowspan=1 colspan=1>172.8 MB82.6</td><td rowspan=1 colspan=1>192.9 MB34.1</td><td rowspan=1 colspan=1>128.5 MB102.8</td></tr></table>

Table 6. Memory and dynamic rendering FPS comparison.

<!-- image-->  
Figure 11. Additional qualitative comparisons. We illustrate the results of our method and baselines using CL-NeRF dataset and our dataset.

<table><tr><td></td><td colspan="3">CL-NeRF dataset</td><td colspan="5">Our dataset</td></tr><tr><td>Method</td><td>Whiteroom</td><td>Kitchen</td><td>Rome</td><td>Cafe</td><td>Diningroom</td><td>Livingroom</td><td>Hall</td><td>Lab</td></tr><tr><td>3DGS [11]</td><td>20.66</td><td>24.55</td><td>28.38</td><td>15.97</td><td>18.44</td><td>20.44</td><td>22.35</td><td>20.60</td></tr><tr><td>InstantSplat [6]</td><td>19.35</td><td>17.66</td><td>19.92</td><td>17.71</td><td>22.14</td><td>19.30</td><td>18.66</td><td>18.98</td></tr><tr><td>4DGS [40]</td><td>24.84</td><td>25.41</td><td>28.15</td><td>18.01</td><td>22.84</td><td>22.51</td><td>23.33</td><td>20.76</td></tr><tr><td>NSC [18]</td><td>18.13</td><td>17.40</td><td>26.36</td><td>15.10</td><td>18.21</td><td>19.81</td><td>18.63</td><td>15.86</td></tr><tr><td>3DGS-CD [21]</td><td>23.86</td><td>22.20</td><td>24.77</td><td>17.94</td><td>22.15</td><td>21.00</td><td>22.22</td><td>21.41</td></tr><tr><td>CL-NeRF [41]</td><td>26.22</td><td>25.81</td><td>24.55</td><td>16.53</td><td>22.66</td><td>22.32</td><td>22.40</td><td>20.87</td></tr><tr><td>CL-Splats [1]</td><td>26.84</td><td>24.90</td><td>25.79</td><td>18.71</td><td>19.29</td><td>23.94</td><td>22.21</td><td>21.47</td></tr><tr><td>LTGS (ours)</td><td>26.67</td><td>26.21</td><td>28.65</td><td>20.77</td><td>25.22</td><td>23.50</td><td>25.22</td><td>22.64</td></tr></table>

Table 7. PSNR comparisons on CL-NeRF dataset and our dataset. The first and second best results are highlighted in bold and underlined, respectively.

<table><tr><td rowspan="2">Method</td><td colspan="3">CL-NeRF dataset</td><td colspan="5">Our dataset</td></tr><tr><td>Whiteroom</td><td>Kitchen</td><td>Rome</td><td>Cafe</td><td>Diningroom</td><td>Livingroom</td><td>Hall</td><td>Lab</td></tr><tr><td>3DGS [11]</td><td>0.821</td><td>0.645</td><td>0.899</td><td>0.776</td><td>0.866</td><td>0.897</td><td>0.911</td><td>0.835</td></tr><tr><td>InstantSplat [6]</td><td>0.699</td><td>0.443</td><td>0.660</td><td>0.723</td><td>0.857</td><td>0.795</td><td>0.809</td><td>0.739</td></tr><tr><td>4DGS [40]</td><td>0.827</td><td>0.644</td><td>0.885</td><td>0.772</td><td>0.893</td><td>0.882</td><td>0.892</td><td>0.812</td></tr><tr><td>NSC [18]</td><td>0.710</td><td>0.536</td><td>0.849</td><td>0.705</td><td>0.795</td><td>0.813</td><td>0.804</td><td>0.658</td></tr><tr><td>3DGS-CD [21]</td><td>0.815</td><td>0.563</td><td>0.803</td><td>0.719</td><td>0.792</td><td>0.797</td><td>0.751</td><td>0.812</td></tr><tr><td>CL-NeRF [41]</td><td>0.829</td><td>0.636</td><td>0.725</td><td>0.696</td><td>0.863</td><td>0.881</td><td>0.859</td><td>0.775</td></tr><tr><td>CL-Splats [1]</td><td>0.848</td><td>0.627</td><td>0.840</td><td>0.749</td><td>0.873</td><td>0.836</td><td>0.866</td><td>0.819</td></tr><tr><td>LTGS (ours)</td><td>0.848</td><td>0.645</td><td>0.892</td><td>0.845</td><td>0.911</td><td>0.922</td><td>0.924</td><td>0.840</td></tr></table>

Table 8. SSIM comparisons on CL-NeRF dataset and our dataset. The first and second best results are highlighted in bold and underlined, respectively.

<table><tr><td></td><td colspan="3">CL-NeRF dataset</td><td colspan="5">Our dataset</td></tr><tr><td>Method</td><td>Whiteroom</td><td>Kitchen</td><td>Rome</td><td>Cafe</td><td>Diningroom</td><td>Livingroom</td><td>Hall</td><td>Lab</td></tr><tr><td>3DGS [11]</td><td>0.522</td><td>0.537</td><td>0.116</td><td>0.323</td><td>0.313</td><td>0.216</td><td>0.234</td><td>0.273</td></tr><tr><td>InstantSplat [6]</td><td>0.561</td><td>0.579</td><td>0.257</td><td>0.328</td><td>0.295</td><td>0.367</td><td>0.377</td><td>0.346</td></tr><tr><td>4DGS [40]</td><td>0.539</td><td>0.547</td><td>0.148</td><td>0.363</td><td>0.306</td><td>0.307</td><td>0.299</td><td>0.334</td></tr><tr><td>NSC [18]</td><td>0.585</td><td>0.619</td><td>0.192</td><td>0.441</td><td>0.385</td><td>0.426</td><td>0.431</td><td>0.510</td></tr><tr><td>3DGS-CD [21]</td><td>0.527</td><td>0.571</td><td>0.213</td><td>0.337</td><td>0.366</td><td>0.345</td><td>0.375</td><td>0.318</td></tr><tr><td>CL-NeRF [41]</td><td>0.521</td><td>0.536</td><td>0.339</td><td>0.463</td><td>0.324</td><td>0.328</td><td>0.350</td><td>0.428</td></tr><tr><td>CL-Splats [1]</td><td>0.492</td><td>0.542</td><td>0.214</td><td>0.345</td><td>0.308</td><td>0.323</td><td>0.303</td><td>0.280</td></tr><tr><td>LTGS (ours)</td><td>0.478</td><td>0.522</td><td>0.128</td><td>0.246</td><td>0.253</td><td>0.185</td><td>0.204</td><td>0.260</td></tr></table>

Table 9. LPIPS comparisons on CL-NeRF dataset and our dataset. The first and second best results are highlighted in bold and underlined, respectively.