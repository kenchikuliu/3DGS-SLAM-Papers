# DynamicVGGT: Learning Dynamic Point Maps for 4D Scene Reconstruction in Autonomous Driving

Zhuolin He1,3 Jing Li2 Guanghao Li1,4 Xiaolei Chen1 Jiacheng Tang1 Siyang Zhang3,5 Zhounan Jin2 Feipeng Cai3 Bin Li1 Jian Pu1B Jia Cai3 Xiangyang Xue1B 1Fudan University 2Huawei 3Yinwang Intelligent Technology 4Shanghai Innovation Institute 5CUHK

{zlhe22,ghli22,chenxl23,jiachengtang21}@m.fudan.edu.cn {libin,jianpu,xyxue}@fudan.edu.cn {lijing470,jinzhounan}@huawei.com {caifeipeng,caijia}@yinwang.com siyangzhang@link.cuhk.edu.cn

## Abstract

Dynamic scene reconstruction in autonomous driving remains a fundamental challenge due to significant temporal variations, moving objects, and complex scene dynamics. Existing feed-forward 3D models have demonstrated strong performance in static reconstruction but still struggle to capture dynamic motion. To address these limitations, we propose DynamicVGGT, a unified feed-forward framework that extends VGGT from static 3D perception to dynamic 4D reconstruction. Our goal is to model point motion within feed-forward 3D models in a dynamic and temporally coherent manner. To this end, we jointly predict the current and future point maps within a shared reference coordinate system, allowing the model to implicitly learn dynamic point representations through temporal correspondence. To efficiently capture temporal dependencies, we introduce a Motion-aware Temporal Attention (MTA) module that learns motion continuity. Furthermore, we design a Dynamic 3D Gaussian Splatting Head that explicitly models point motion by predicting Gaussian velocities using learnable motion tokens under scene flow supervision. It refines dynamic geometry through continuous 3D Gaussian optimization. Extensive experiments on autonomous driving datasets demonstrate that DynamicVGGT significantly outperforms existing methods in reconstruction accuracy, achieving robust feed-forward 4D dynamic scene reconstruction under complex driving scenarios.

## 1. Introduction

Visual geometry learning is a fundamental problem in computer vision and serves as a core foundation for various applications in robotics [32] and autonomous driving [29, 30]. In recent years, feed-forward 3D models [4, 21, 24] have achieved remarkable progress in static scene understanding by directly predicting geometric representations such as point clouds and 3D Gaussian from image inputs. However, visual geometry learning in autonomous driving scenarios faces much greater complexity than in static scenes. Real-world driving environments are inherently dynamic, featuring diverse moving objects, changing long-range temporal dependencies. Although feed-forward architectures [26, 27] demonstrate strong performance on static datasets, they struggle to maintain both geometric accuracy and temporal consistency when extended to such dynamic conditions. This motivates the need for a unified feed-forward framework that can jointly model geometry and motion, enabling temporally consistent dynamic scene reconstruction.

<!-- image-->  
Figure 1. DynamicVGGT extends static multi-view 3D perception to dynamic 4D reconstruction by enabling 3D Gaussian rendering and adaptively modeling motion across multiple temporal scales without explicit camera extrinsic alignment.

Current 3D foundation models [3, 9] are typically trained on large-scale, well-labeled datasets and can achieve consistent and accurate 3D reconstruction across most scenes. However, applying them to real-world autonomous driving scenarios remains highly challenging. First, autonomous driving data often exhibit large-scale, high-noise, and sparse-depth characteristics, which can lead to a degradation of the modelâs original dense prediction capability when trained directly on such data. Moreover, beyond static geometric perception, capturing dynamic geometric information is crucial in autonomous driving. Although several recent 3D foundation models [16, 38] have begun to explore dynamic scene modeling, their output representations are still primarily based on static point maps, lacking a unified dynamic representation that can directly support downstream autonomous driving tasks.

To address these issues, we propose DynamicVGGT, a unified framework for high-fidelity dynamic scene reconstruction in a feed-forward manner. As Fig. 1 shows, DynamicVGGT introduces a novel Dynamic Point Map (DPM) [22] mechanism designed by two different dynamic tasks. Specifically, we introduce a Future Point Head that predicts the point map of the next frame and enforces consistency with the current frame, thereby enabling the model to implicitly learn point-wise motion. On the other hand, we introduce a Dynamic 3D Gaussian Splatting Head (DGSHead), which refines the predicted geometry using Gaussian primitives initialized from the geometric priors. It further incorporates a lightweight motion-aware encoder that encodes motion flow through learnable motion tokens, supervised by scene flow. Extensive experiments demonstrate that DynamicVGGT achieves state-of-the-art performance across diverse driving datasets. We summarize our main contributions as follows:

â¢ We introduce a motion-aware temporal attention module that captures temporal dependencies without disrupting VGGTâs spatial attention, preserving stable training and geometric priors.

â¢ We extend point-based representations towards a unified DPM by introducing a future point prediction task and a Dynamic 3D Gaussian Splatting Head. On top of this framework, the model learns point-wise motion through implicit consistency of inter-frame point motion and explicit supervision of Gaussian motion using scene flow.

â¢ A stage-wise training scheme is adopted to mitigate the performance degradation observed on real-world driving data. On the Waymo dataset, our model achieves notable gains over VGGT and StreamVGGT, improving Accuracy by 0.5 and Completeness by 0.2.

## 2. Related work

## 2.1. Feed-Forward Visual Geometry Learning.

Feed-forward 3D reconstruction models [9, 26, 28] aim to recover the geometry of static scenes directly from images under a temporal invariance assumption, providing robust 3D priors for downstream tasks [1, 6, 7, 12â15]. Unlike traditional multi-view geometry pipelines [17, 18] that relied on optimization-based correspondence matching, these learning-based methods predicted depth, camera pose, or dense 3D point maps in an end-to-end manner. Representative frameworks such as DUSt3R [28] had demonstrated that transformer-based architectures can effectively learn direct mappings from image pixels to 3D coordinate fields. Subsequent researches had expanded this paradigm to multi-view and sequential settings, integrating camera pose estimation, depth prediction, and correspondence learning into unified architectures. Among these, VGGT [26] further enhanced the feed-forward formulation by introducing alternating attention mechanisms across spatial and temporal dimensions, achieving joint prediction of multiple geometric quantities through a single, shared model. Anysplat [9] incorporated 3DGS with Feed-Forward Model to achieve high-fidelity reconstruction. Recent extensions of feed-forward visual geometry models have begun to incorporate temporal modeling for dynamic reconstruction. Approaches such as MoVieS[16] and StreamVGGT[38] extend static frameworks like VGGT to handle sequential inputs, but they are primarily designed for indoor environments. Despite these advances, existing 3D feed-forward methods struggled with reconstruct the large scale, dynamic autonomous driving scenes, motivating the need for generalizable 4D feed-forward frameworks capable of capturing scene dynamics over time.

## 2.2. 3DGS Reconstruction for Driving Scenes.

Building photorealistic reconstructions of dynamic urban scenes is crucial for autonomous driving, as it supports closed-loop training and evaluation under realistic motion patterns. Consequently, recent research [31] has shifted focus from static to dynamic scene reconstruction. However, most existing methods [2, 37] rely heavily on dense annotations, which are expensive to obtain and limit scalability. Furthermore, these approaches typically depend on per-scene optimization, making it difficult to exploit largescale data priors and leading to slow reconstruction speeds. Recent advances in feed-forward reconstruction, such as STORM [33] and DrivingForward [24], demonstrate the potential for fast and generalizable 4D scene recovery without per-scene optimization. STORM introduces a feedforward pipeline for dynamic scene reconstruction and editing, achieving high-quality results but still relying on calibrated multi-view inputs. DrivingForward [24] presents a feed-forward 3D Gaussian Splatting framework for driving scenes with flexible surround-view inputs, jointly training pose, depth, and Gaussian heads to infer camera poses and dense geometry without using depth ground truth or provided extrinsics. Our DynamicVGGT generalizes feedforward reconstruction to real-world autonomous driving scenes, jointly modeling geometry and motion without camera parameters or dense annotations.

<!-- image-->  
Figure 2. Proposed DynamicVGGT training framework. Given a sequence of multi-view images $\{ V _ { 1 } , V _ { 2 } , V _ { 3 } \}$ , the model first encodes them using a pretrained DINOv2 backbone to extract patch tokens and camera tokens for each view, while motion tokens are initialized as learnable parameters that encode temporal priors. The patch and camera tokens are processed by the Alternating-Attention (AA) blocks to model intra-frame spatial geometry, whereas the Motion-aware Temporal Attention (MTA) blocks operate in parallel to model inter-frame temporal dependencies using the motion tokens. The resulting temporal features T A are then fed into a Dynamic 3D Gaussian Head (DGSHead) for dynamic 3DGS reconstruction and a Future Point Head for future point prediction.

## 3. Method

Our framework builds upon VGGT and extends it from static 3D perception to dynamic 4D reconstruction. The key idea is to establish a unified geometric representation, namely DPMs, as the core of temporal modeling. Based on this formulation, we introduce temporal reasoning via a MTA module, predict future geometry via a Future Point Head (FPH), and further refine dynamic geometry through a DGSHead. An overview of the proposed architecture is shown in Fig. 2.

## 3.1. Dynamic Point Map and Task Formulation

We denote the camera index by $v \in \{ 1 , \ldots , N _ { v } \}$ and the frame index by $t \in \{ 1 , \ldots , \tau \}$ . Temporal modeling is defined on frame pairs $( v , t )$ and $( v , t + \delta )$ from the same camera stream. Following prior work [22], a static point map is defined as

$$
P _ { v , t } = \pi ^ { - 1 } ( I _ { v , t } ; K _ { v , t } , E _ { v , t } ) \in \mathbb { R } ^ { 3 \times H \times W } .\tag{1}
$$

To model dynamics, previous DPM formulations align all frames into a shared reference frame:

$$
P _ { v , t } ^ { ( \mathrm { r e f } ) } = \mathcal { T } _ { ( v , t )  \mathrm { r e f } } \big ( \pi ^ { - 1 } ( I _ { v , t } ; K _ { v , t } , E _ { v , t } ) \big ) ,\tag{2}
$$

so that temporal motion is expressed as

$$
\begin{array} { r } { \Delta P _ { v , t } ^ { \mathrm { ( r e f ) } } = P _ { v , t + \delta } ^ { \mathrm { ( r e f ) } } - P _ { v , t } ^ { \mathrm { ( r e f ) } } . } \end{array}\tag{3}
$$

In contrast, VGGT directly predicts point maps in a learned canonical frame. Given a multi-view clip $\{ I _ { v , t } \}$ our model predicts

$$
\begin{array} { r } { \hat { P } _ { v , t } , \hat { P } _ { v , t + \delta } = f _ { \theta } ( \{ I _ { v , t } \} ) | _ { ( v , t ) , ( v , t + \delta ) } , } \end{array}\tag{4}
$$

which enables implicit motion learning through $\Delta \hat { P } _ { v , t } =$ $\hat { P } _ { v , t + \delta } - \hat { P } _ { v , t }$ . This formulation avoids explicitly relying on externally specified frame-to-reference transformations in the dynamic formulation, while still preserving the geometric prior of the original VGGT backbone.

It therefore provides a unified basis for dynamic task design, as illustrated in Fig. 3. Specifically, we introduce two complementary tasks built upon this formulation: (i) the FPH, which learns implicit motion by enforcing inter-frame point consistency, and (ii) the DGSHead, which explicitly refines dynamic geometry through scene-flow-supervised Gaussian optimization.

## 3.2. Motion-aware Temporal Attention

Although the DPM formulation captures temporal variations at the geometric level, relying solely on point-wise displacement is still insufficient for accurate dynamic reconstruction. Previous work, such as StreamVGGT [38], also introduces temporal attention to enhance temporal modeling. However, its sequential stacking of AA blocks tends to cause unstable training and degraded performance in the early stages. To address this limitation, we propose a Motion-aware Temporal Attention (MTA) module that explicitly models motion cues at the feature level and integrates temporal reasoning into the VGGT backbone. The key idea is to introduce learnable motion tokens that dynamically encode inter-frame motion information, guiding temporal attention to focus on motion-consistent regions.

<!-- image-->  
Figure 3. Dynamic task formulation. We formulate dynamic point maps by designing two complementary tasks that model point-wise motion over time. The Future Point Head learns implicit motion through inter-frame point consistency, while the Dynamic 3D Gaussian Splatting Head provides explicit motion supervision via scene flow to refine dynamic geometry.

Given the aggregated token features $\tilde { F } _ { v , t } = [ F _ { v , t } ^ { c } ; F _ { v , t } ^ { p } ]$ produced by the AA blocks, we remove the camera token $F _ { v , t } ^ { c }$ and concatenate the patch tokens $F _ { v , t } ^ { p }$ with learnable motion tokens to form the MTA input. For each MTA layer l, temporal correlations are computed in parallel across all frames along the temporal dimension Ï . The input to the l-th MTA layer is defined as

$$
F _ { m , v , t } ^ { ( l ) } = \left\{ \begin{array} { l l } { \mathrm { C o n c a t } \Big ( M _ { v , t } ^ { ( l ) } , F _ { v , t } ^ { p ( l ) } \Big ) , } & { l = 1 , } \\ { \mathrm { C o n c a t } \Big ( M _ { v , t } ^ { ( l ) } , F _ { v , t } ^ { p ( l ) } + F _ { v , t } ^ { p ( l - 1 ) } \Big ) , } & { l > 1 , } \end{array} \right.\tag{5}
$$

where $\boldsymbol { M } _ { v , t } ^ { ( l ) }$ denotes the motion tokens and $F _ { v , t } ^ { p ( l ) }$ denotes the spatial patch tokens from the AA branch at layer l. Temporal attention is computed independently for each patch position and each view:

$$
A _ { t , t ^ { \prime } } ^ { ( l ) } = \mathrm { S o f t m a x } \left( \frac { Q _ { t } ^ { \mathrm { a t t n } , ( l ) } \left( K _ { t ^ { \prime } } ^ { \mathrm { a t t n } , ( l ) } \right) ^ { \top } } { \sqrt { d } } + B _ { t , t ^ { \prime } } ^ { \mathrm { t i m e } } \right) ,\tag{6}
$$

$$
\tilde { F } _ { m , v , t } ^ { ( l ) } = \sum _ { t ^ { \prime } = 1 } ^ { \tau } A _ { t , t ^ { \prime } } ^ { ( l ) } V _ { t ^ { \prime } } ^ { \mathrm { a t t n } , ( l ) } ,\tag{7}
$$

where $t , t ^ { \prime } \in \mathsf { \Omega } \{ 1 , \dots , \tau \}$ denote discrete frame indices within the sampled clip, and $B _ { t , t ^ { \prime } } ^ { \mathrm { t i m e } }$ denotes the temporal positional bias implemented using rotary position embeddings. The updated temporal features are then processed by layer normalization and an MLP with residual connections:

$$
F _ { m , v , t } ^ { ( l + 1 ) } = \mathrm { M L P } ^ { ( l ) } \left( \mathrm { L a y e r N o r m } \Big ( \tilde { F } _ { m , v , t } ^ { ( l ) } \Big ) \right) + F _ { m , v , t } ^ { ( l ) } .\tag{8}
$$

After the final MTA layer, we denote the temporally enhanced feature for view v at time t by

$$
T A _ { v , t } = F _ { m , v , t } ^ { ( L ) } ,\tag{9}
$$

where L is the number of MTA layers. The resulting feature $T A _ { v , t }$ is used by both the Future Point Head and the Dynamic 3DGS Head. This formulation enables simultaneous message passing across temporal spans and enhances the modelâs ability to capture motion continuity and temporally coherent geometry.

## 3.3. Future Point Prediction

Building upon the unified DPM representation and the feature-level temporal modeling provided by MTA, we further introduce a Future Point Head (FPH) to explicitly learn point-wise motion dynamics. Specifically, the FPH predicts the 3D point map of a future frame from the temporally enhanced feature at the current timestep, enabling the network to learn short-term motion continuity in a self-supervised manner. Given the temporally enhanced feature $T A _ { v , t }$ produced by the MTA module, the FPH predicts the point map of the same camera stream at a future timestep:

$$
\hat { P } _ { v , t + \delta } ^ { \mathrm { f u t } } = \mathrm { D P T } _ { p } ( T A _ { v , t } ) ,\tag{10}
$$

where $\mathrm { D P T } _ { p } ( \cdot )$ denotes a DPT head [20] for future point regression. To further encourage physically plausible point motion trajectories, we introduce a temporal consistency regularization:

$$
\mathcal { L } _ { \mathrm { t e m p } } = \frac { 1 } { | \mathcal { N } | } \sum _ { i \in \mathcal { N } } \left. \left( \mathbf { p } _ { v , t + \delta } ^ { ( i ) } - \mathbf { p } _ { v , t } ^ { ( i ) } \right) - \left( \hat { \mathbf { p } } _ { v , t + \delta } ^ { ( i ) } - \hat { \mathbf { p } } _ { v , t } ^ { ( i ) } \right) \right. _ { 1 } ,\tag{11}
$$

where $\mathbf { p } _ { v , t } ^ { ( i ) }$ and $\hat { \mathbf { p } } _ { v , t } ^ { ( i ) }$ denote the ground-truth and predicted 3D coordinates of the i-th valid point at time t, respectively, and $\mathcal { N }$ denotes the set of valid points. The displacement field $\begin{array} { r } { \Delta \mathbf { p } _ { v , t } ^ { ( i ) } = \mathbf { p } _ { v , t + \delta } ^ { ( i ) } - \mathbf { p } _ { v , t } ^ { ( i ) } } \end{array}$ acts as a coarse motion representation that encourages the network to learn interframe point displacement within the shared DPM coordinate space. We emphasize that $\mathcal { L } _ { \mathrm { t e m p } }$ supervises motion implicitly at the point-map level, which complements the explicit motion supervision introduced later in the Dynamic 3DGS Head.

## 3.4. Dynamic 3D Gaussian Splatting Head

To further model dynamic scenes at the primitive level, we introduce a Dynamic 3D Gaussian Splatting (3DGS) Head as a downstream dynamic reconstruction task.

This module takes both the temporally enhanced features $T A _ { v , t }$ from MTA and the RGB appearance cues from the input images as inputs, and converts them into time-varying 3D Gaussian primitives that jointly model geometry, appearance, and motion. We observe that freezing the AA blocks causes the pretrained VGGT backbone to overemphasize geometric reasoning while weakening appearance cues, which degrades Gaussian rendering quality. To compensate for this issue, we fuse image features extracted by Conv(Â·) with geometric features obtained from $T A _ { v , t } \mathrm { i }$

$$
F _ { v , t } ^ { \mathrm { a p p } } = \mathrm { C o n v } ( I _ { v , t } ) ,\tag{12}
$$

$$
F _ { g , v , t } , D _ { g , v , t } = \mathrm { D P T } _ { g } ( T A _ { v , t } ) ,\tag{13}
$$

$$
G _ { v , t } = F _ { v , t } ^ { \mathrm { a p p } } + F _ { g , v , t } ,\tag{14}
$$

where $F _ { g , v , t }$ denotes the Gaussian features used to initialize Gaussian primitives, and $D _ { g , v , t }$ denotes the predicted Gaussian depth. At each timestep, the predicted Gaussian depth $D _ { g , v , t }$ together with the camera predictions from the retained VGGT camera branch are used to reconstruct a point map $P _ { v , t } ^ { g }$ , which initializes the Gaussian centers $\mu _ { i }$ . Each Gaussian primitive is parameterized as $\{ \mu _ { i } , \sigma _ { i } , r _ { i } , c _ { i } , \nu _ { i } \}$ . where $\mu _ { i }$ denotes the center, $\sigma _ { i }$ the scale, $r _ { i }$ the rotation, $c _ { i }$ the color, and $\nu _ { i }$ the velocity vector.

We further leverage the M learnable motion tokens introduced in MTA to decode a set of velocity bases $\nu _ { b } \in \mathbb { R } ^ { 3 }$ ; forming a shared dynamic representation for Gaussian motion. To describe temporal evolution, we assume constant velocity within a short clip:

$$
\begin{array} { r } { \mu _ { i , t + \delta } = \mu _ { i , t } + \delta \cdot \nu _ { i , t } . } \end{array}\tag{15}
$$

During training, scene-flow supervision is applied to encourage each Gaussian primitive to carry a physically meaningful velocity attribute. Unlike $\mathcal { L } _ { \mathrm { t e m p } }$ , which constrains coarse inter-frame displacement at the point-map level, the Gaussian motion supervision explicitly regularizes motion in the dynamic Gaussian space and therefore provides complementary supervision for dynamic reconstruction.

## 3.5. Training Objective

To enable the model to learn dynamic geometry progressively, we adopt a two-stage training strategy that follows a curriculum-style paradigm from synthetic to real-world data. In the first stage, the model is trained on high-quality synthetic autonomous driving datasets, where dense geometry and reliable motion cues are available. The stage-1 training objective combines static and temporal supervision:

<!-- image-->  
Figure 4. Depth and Point Maps Comparison. The sparsity of LiDAR point clouds degrades the results, leading to less smooth depth maps and rougher point clouds.

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s t a g e 1 } } = \mathcal { L } _ { \mathrm { c a m } } + \mathcal { L } _ { \mathrm { d e p t h } } + \mathcal { L } _ { \mathrm { p o i n t } } ^ { ( t ) } + \mathcal { L } _ { \mathrm { p o i n t } } ^ { ( t + \delta ) } + \lambda _ { \mathrm { t e m p } } \mathcal { L } _ { \mathrm { t e m p } } . } \end{array}\tag{16}
$$

The camera loss supervises the predicted camera parameters gË: $\begin{array} { r } { \mathcal { L } _ { \mathrm { c a m } } ~ = ~ \sum _ { i = 1 } ^ { N } \left. \hat { g } _ { i } - g _ { i } \right. _ { \epsilon } } \end{array}$ , where $\hat { g } _ { i }$ and $g _ { i }$ denote the predicted and ground-truth camera parameters, respectively, and $\| \cdot \| _ { \epsilon }$ denotes the Huber loss. The depth loss ${ \mathcal { L } } _ { \mathrm { d e p t h } }$ and point-map losses $\mathcal { L } _ { \mathrm { p o i n t } } ^ { ( t ) }$ and $\mathcal { L } _ { \mathrm { p o i n t } } ^ { ( t + \delta ) }$ follow VGGT [26]. This stage focuses on learning temporal consistency by predicting future point maps, allowing the network to capture short-term motion while preserving the geometric priors of the pretrained backbone.

In the second stage, we fine-tune the model on real driving datasets using the Dynamic 3DGS objective. The overall stage-2 objective is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s t a g e 2 } } = \mathcal { L } _ { \mathrm { s t a g e 1 } } + \mathcal { L } _ { \mathrm { 3 D G S } } , } \end{array}\tag{17}
$$

where

$$
{ \mathcal { L } } _ { \mathrm { 3 D G S } } = { \mathcal { L } } _ { \mathrm { r g b } } + \lambda _ { \mathrm { g s } } { \mathcal { L } } _ { \mathrm { g s d e p t h } } + \lambda _ { \mathrm { d i s t } } { \mathcal { L } } _ { \mathrm { d i s t i l l } } + \lambda _ { \mathrm { f l o w } } { \mathcal { L } } _ { \mathrm { f l o w } } .\tag{18}
$$

For image reconstruction, we use $\begin{array} { r l } { \mathcal { L } _ { \mathrm { r g b } } } & { { } = } \end{array}$ $\mathrm { M S E } ( I _ { v , t } , \hat { I } _ { v , t } )$ , where $I _ { v , t }$ is the ground-truth RGB image and $\hat { I } _ { v , t }$ is the image rendered from the predicted dynamic Gaussian primitives.

We observe that directly using sparse LiDAR point clouds as supervision on real autonomous driving datasets leads to severe performance degradation due to their limited density and uneven spatial distribution. To alleviate this issue, we introduce a depth distillation strategy. Specifically, the depth predicted by the stage-1 point-map branch serves as the teacher signal, while the Gaussian depth branch predicts the student depth: $\mathcal { L } _ { \mathrm { d i s t i l l } } = \left. D _ { g , v , t } - \mathrm { s g } \big ( D _ { v , t } ^ { \mathrm { p m } } \big ) \right. _ { 1 }$ where $D _ { v , t } ^ { \mathrm { p m } }$ denotes the depth predicted by the stage-1 geometry branch and $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operator. In addition, $\mathcal { L } _ { \mathrm { g s d e p t h } }$ is a standard $L _ { 1 }$ loss, whose supervision is provided by the pretrained model [27]. This strategy mitigates the noise caused by point-cloud sparsity and stabilizes Gaussian optimization, as shown in Fig. 4.

Finally, we use scene-flow supervision for explicit Gaussian motion learning: $\mathcal { L } _ { \mathrm { f l o w } } ~ = ~ \mathrm { M S E } ( \mathbf { s } _ { v , t } , \hat { \mathbf { s } } _ { v , t } )$ , where ${ \bf s } _ { v , t }$ and $\hat { \mathbf { s } } _ { v , t }$ denote the ground-truth and predicted scene flow, respectively. Compared with $\mathcal { L } _ { \mathrm { t e m p } } ,$ which supervises coarse point displacement in the DPM space, ${ \mathcal { L } } _ { \mathrm { { f l o w } } }$ explicitly constrains motion in the Gaussian representation. The two losses therefore operate at different levels and are complementary rather than redundant.

## 4. Experiments

This section compares our method to state-of-the-art approaches across multiple tasks to show its effectiveness.

## 4.1. Implementation Details

We use L = 12 MTA layers, resulting in approximately 1.4B parameters. DynamicVGGT is initialized from pretrained VGGT weights, with about 800M parameters (excluding frozen modules) fine-tuned in two stages.

In Stage 1, we train for 10 epochs using AdamW with a hybrid learning rate schedule: a linear warm-up for the first 0.5 epoch followed by cosine decay, with a peak learning rate of $1 \times 1 0 ^ { - 6 }$ . In Stage 2, we further fine-tune the model for 50 epochs with the Gaussian head enabled, using the same schedule but a higher peak learning rate of $5 \times 1 0 ^ { - 5 }$

All input frames, depth maps, and point maps are resized so that the longer image side does not exceed 518 pixels. The temporal offset Î´ is randomly sampled from 1 to 3. We adopt a dynamic batch sizing strategy similar to VGGT, processing 18 images per batch. For the training objective, we set $\lambda _ { \mathrm { t e m p } } = 0 . 0 1 , \lambda _ { \mathrm { g s } } = \lambda _ { \mathrm { d i s } } = 0 . 1$ , and $\lambda _ { \mathrm { f l o w } } = 0 . 0 1$ Further details of the training settings and dataset configurations are provided in the Appendix.

## 4.2. Training Data

The model is trained on a collection of dynamic autonomous driving datasets, including Waymo Open Dataset [23], Virtual KITTI [5], and MVS-Synth [8]. In the first stage, DynamicVGGT is trained on Virtual KITTI and MVS-Synth to learn robust geometric priors and temporal consistency under controlled synthetic settings. In the second stage, the model is fine-tuned with the Dynamic 3D Gaussian Splatting module on Waymo and Virtual KITTI to enhance dynamic geometry refinement and appearance consistency in real driving environments. Evaluation is conducted on the Waymo validation set and the KITTI [25] dataset to assess reconstruction quality.

## 4.3. Point Map Reconstruction

We evaluate point map reconstruction on the KITTI and Waymo datasets, as shown in Table 1. We report Accuracy (Acc.), Completion (Comp.), and Normal Consistency (NC). On the KITTI dataset, which uses monocular input with three consecutive frames per sequence, DynamicVGGT achieves the best results across most metrics, obtaining an accuracy of 0.901 and a normal consistency of 0.939. It consistently outperforms both VGGT [26] and StreamVGGT [38], demonstrating its effectiveness in capturing dynamic geometry and maintaining temporal consistency in monocular sequences.

On the Waymo dataset, which provides synchronized multi-view images from three cameras with a frame stride of four, our model generalizes well to large-scale dynamic driving scenes. It achieves an accuracy of 4.021 and a normal consistency of 0.603. These results confirm that the proposed dynamic formulation effectively enhances cross-view consistency and scene completeness, even under challenging real-world motion and illumination variations, highlighting the scalability of our feed-forward framework for dynamic 4D perception.

## 4.4. 4D scene reconstruction

We further evaluate DynamicVGGT on 4D scene reconstruction using the Waymo validation set, as summarized in Table 2. We compare two categories of methods: per-scene optimization and feed-forward models. DynamicVGGT achieves a PSNR of 18.07 and an SSIM of 0.376 on dynamic regions, and reaches 24.07 and 0.676 on the fullframe evaluation. Although methods such as STORM obtain higher scores with multi-camera inputs and geometric priors, achieving 21.26 in PSNR and 0.535 in SSIM, DynamicVGGT delivers competitive results using only monocular images without relying on camera parameters or scene-specific optimization. These results demonstrate that DynamicVGGT effectively reconstructs dynamic 4D scenes with strong temporal consistency and high visual fidelity through purely image-based self-supervision.

## 4.5. Monocular and MVS depth estimation

We evaluate the monocular and multi-view stereo depth estimation performance of DynamicVGGT on three benchmarks: KITTI and NYU-v2 [19], as shown in Table 3. We report two standard metrics: Absolute Relative Error (Abs Rel) and accuracy under the Î´ < 1.25 threshold.

On monocular KITTI, DynamicVGGT achieves an Abs Rel of 0.070, outperforming all baselines. On NYU-v2, it obtains an Abs Rel of 0.064 and 94.3% accuracy under $\delta \ < \ 1 . 2 5$ , demonstrating strong generalization from outdoor to indoor scenes. Under the multi-view stereo setting, DynamicVGGT achieves the best overall results with an Abs Rel of 0.051 and 97.6% accuracy, surpassing VGGT and StreamVGGT by a clear margin.

Table 1. Point Map Reconstruction on KITTI and Waymo(val). KITTI uses monocular input with every 3 consecutive frames per camera. Waymo uses 3 frames (stride 4) from FRONT, SIDE LEFT, and SIDE RIGHT cameras, totaling 9 images per group.
<table><tr><td rowspan="3">Methods</td><td colspan="6">KITTI (monocular)</td><td colspan="6">Waymo (3 cameras)</td></tr><tr><td colspan="3">Mean</td><td colspan="3">Med.</td><td colspan="3">Mean</td><td colspan="3">Med.</td></tr><tr><td>Acc. â</td><td>Comp. â</td><td>NCâ</td><td>Acc. â</td><td>Comp. â</td><td>NCâ</td><td>Acc. â</td><td>Comp. â</td><td>NC â</td><td>Acc. â</td><td>Comp. â</td><td>NC â</td></tr><tr><td>VGGT[26]</td><td>1.489</td><td>0.690</td><td>0.918</td><td>1.329</td><td>0.535</td><td>0.971</td><td>4.635</td><td>2.667</td><td>0.561</td><td>2.634</td><td>1.734</td><td>0.590</td></tr><tr><td>StreamVGGT[38]</td><td>1.078</td><td>0.495</td><td>0.899</td><td>0.867</td><td>0.390</td><td>0.949</td><td>4.598</td><td>2.626</td><td>0.564</td><td>2.567</td><td>1.789</td><td>0.592</td></tr><tr><td>DynamicVGGT</td><td>0.901</td><td>0.584</td><td>0.939</td><td>0.733</td><td>0.464</td><td>0.963</td><td>4.021</td><td>2.390</td><td>0.562</td><td>1.971</td><td>1.564</td><td>0.603</td></tr></table>

Table 2. Comparison to state-of-the-art methods on Waymo (val). PSNR and SSIM are reported. Full: requires dense scene annotations. Camera: requires camera intrinsics and extrinsics.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Supervision</td><td colspan="2">Dynamic-only</td><td colspan="2">Full image</td></tr><tr><td>PSNR â SSIMâPSNR â SSIM â</td><td></td><td></td><td></td></tr><tr><td>Per-scene optimization</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3DGS [10]</td><td>Full</td><td>17.13</td><td>0.267</td><td>25.13</td><td>0.741</td></tr><tr><td>DeformableGS [34]</td><td>Full</td><td>17.10</td><td>0.266</td><td>25.29</td><td>0.761</td></tr><tr><td>Feed-forward model</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GS-LRM[36]</td><td>Camera</td><td>20.02</td><td>0.520</td><td>25.18</td><td>0.753</td></tr><tr><td>STORM[33]</td><td>Camera</td><td>21.26</td><td>0.535</td><td>25.03</td><td>0.750</td></tr><tr><td>DynamicVGGT</td><td>Image-only</td><td>18.07</td><td>0.376</td><td>24.07</td><td>0.676</td></tr></table>

Table 3. Monocular and MVS depth estimation.
<table><tr><td rowspan="2">Methods</td><td colspan="2">KITTI(Mono)</td><td colspan="2">NYU-v2(Mono)</td><td colspan="2">KITTI(MVS)</td></tr><tr><td></td><td>Abs Rel â Î´ &lt; 1.25 â Abs Rel â Î´ &lt; 1.25 â Abs Rel â Î´ &lt; 1.25 â</td><td></td><td></td><td></td><td></td></tr><tr><td>DUSt3R[28]</td><td>0.109</td><td>0.873</td><td>0.081</td><td>0.909</td><td>0.143</td><td>0.814</td></tr><tr><td>MASt3R[11]</td><td>0.077</td><td>0.948</td><td>0.110</td><td>0.865</td><td>0.115</td><td>0.848</td></tr><tr><td>MonST3R [35]</td><td>0.098</td><td>0.895</td><td>0.094</td><td>0.887</td><td>0.107</td><td>0.884</td></tr><tr><td>VGGT[26]</td><td>0.082</td><td>0.938</td><td>0.059</td><td>0.951</td><td>0.062</td><td>0.969</td></tr><tr><td>StreamVGGT [38]</td><td>0.082</td><td>0.947</td><td>0.057</td><td>0.959</td><td>0.173</td><td>0.721</td></tr><tr><td>DynamicVGGT</td><td>0.070</td><td>0.940</td><td>0.064</td><td>0.943</td><td>0.051</td><td>0.976</td></tr></table>

## 4.6. Ablution study

We conduct an ablation study on the KITTI monocular dataset and the Waymo three-camera dataset to evaluate the contribution of each proposed component, as summarized in Table 4. Starting from the vanilla VGGT baseline, adding temporal attention and the future point prediction head improves accuracy from 1.489 to 0.927 and completeness from 0.690 to 0.600 on KITTI, demonstrating the benefit of temporal modeling in capturing dynamic geometry.

Introducing the Dynamic 3D Gaussian Splatting Head further enhances accuracy to 0.901 and normal consistency to 0.939, producing smoother and more complete reconstructions. On the Waymo dataset, our full model achieves the best overall results with an accuracy error of 4.021 and a normal consistency of 0.603, confirming that the combination of motion-aware temporal attention and dynamic geometry refinement substantially improves temporal coherence and reconstruction quality.

<!-- image-->  
Figure 5. Point map reconstruction. DynamicVGGT reconstructs denser, smoother, and more geometrically consistent point maps than VGGT, maintaining temporal coherence even under large viewpoint or scene changes. Zoom in for better view.

## 4.7. Visualization

Point Map Reconstruction. We provide qualitative comparisons of point map reconstruction results in Fig. 5, covering three configurations: single-frame reconstruction, shortterm multi-frame reconstruction, and long-range temporal reconstruction. Across all settings, DynamicVGGT consistently outperforms VGGT in both geometric completeness and temporal consistency. Even in the single-frame case, our model produces denser and smoother geometry with more accurate structural details, whereas VGGT tends to generate distorted geometry when the viewpoint changes.

When extended to multi-frame and long-sequence inputs, DynamicVGGT demonstrates superior robustness in capturing point-level motion trajectories and maintaining consistent global geometry, particularly in challenging scenarios such as downhill roads and open intersections in autonomous driving. This highlights the modelâs ability to recover fine-grained 3D structures and maintain stable temporal coherence under large scene variations.

Table 4. Ablation study. We evaluate ablated variants of DynamicVGGT on point map estimation over KITTI and Waymo (val). KITTI uses monocular input with three consecutive frames, while Waymo uses 3 frames (stride 4) from the FRONT, SIDE LEFT, and SIDE RIGHT cameras, yielding 9 images per sample. Metrics include Accuracy (Acc.), Completeness (Comp.), Normal Consistency (NC).
<table><tr><td rowspan="3">Methods</td><td colspan="6">KITTI(Mono)</td><td colspan="6">Waymo(3cam)</td></tr><tr><td colspan="3">Mean</td><td colspan="3">Med.</td><td colspan="3">Mean</td><td colspan="3">Med.</td></tr><tr><td>Acc. â</td><td>Comp. â</td><td></td><td>NC â Acc. â</td><td>Comp. â</td><td>NCâ</td><td>Acc. â</td><td>Comp. â</td><td></td><td>NC â Acc. â</td><td>Comp. â</td><td>NC â</td></tr><tr><td>Baseline</td><td>1.489</td><td>0.690</td><td>0.918</td><td>1.329</td><td>0.535</td><td>0.971</td><td>4.635</td><td>2.667</td><td>0.561</td><td>2.634</td><td>1.734</td><td>0.590</td></tr><tr><td>+ TA &amp; FPH(stage1)</td><td>0.927</td><td>0.600</td><td>0.915</td><td>0.857</td><td>0.474</td><td>0.932</td><td>4.330</td><td>2.939</td><td>0.561</td><td>2.224</td><td>1.649</td><td>0.593</td></tr><tr><td>+ DGSHead(stage2)</td><td>0.901</td><td>0.584</td><td>0.939</td><td>0.733</td><td>0.464</td><td>0.963</td><td>4.021</td><td>2.390</td><td>0.562</td><td>1.971</td><td>1.564</td><td>0.603</td></tr></table>

<!-- image-->  
Figure 6. Scene Reconstruction and Novel View Synthesis. Given input frames 0, 2, and 4, DynamicVGGT reconstructs the corresponding scenes and synthesizes novel views for the next frame. For both KITTI and Waymo, multi-view inputs are used; for Waymo, we visualize the front-camera results due to limited view overlap. The model achieves high-quality reconstruction and realistic novel view generation across dynamic driving scenes.

Scene Reconstruction and Novel View Synthesis. Figure 6 presents qualitative results of scene reconstruction and novel view synthesis on the real-world KITTI and Waymo datasets. Given input frames 0, 2, 4, DynamicVGGT reconstructs the corresponding scenes and synthesizes novel views for the subsequent frame, showcasing its ability to model temporal dynamics from purely image-based inputs. For both datasets, multi-view images are used as input; for Waymo, we visualize the front-camera results due to the limited overlap between different camera views.

Our method faithfully reconstructs dynamic scenes with moving vehicles and illumination changes, while maintaining consistent global geometry and realistic appearance across time. Notably, DynamicVGGT achieves temporally coherent novel view synthesis, highlighting its effectiveness as a unified feed-forward framework for dynamic 4D perception in autonomous driving.

## 5. Conclusion

We presented DynamicVGGT, a unified feed-forward framework for dynamic 4D scene reconstruction. By extending VGGT from static geometry perception to temporal dynamics, our model jointly learns geometric and motion representations through Dynamic Point Maps, Motionaware Temporal Attention, Future Point Head and a Dynamic 3D Gaussian Head. This design enables the model to capture temporal dependencies, refine geometry through continuous Gaussian optimization, and maintain feed-forward efficiency. Experimental results demonstrate that DynamicVGGT delivers strong temporal consistency on real-world autonomous driving datasets and simultaneously provides reliable by-products, including camera pose estimation, depth prediction, and novel view synthesis. We believe this direction will push feed-forward 4D reconstruction closer to a unified paradigm for autonomous driving.

Acknowledge This work is supported by NSFC General Program (Grant No. 62576110).

## References

[1] Qi Chen, Yu Cao, Jiawei Hou, Guanghao Li, Shoumeng Qiu, Bo Chen, Xiangyang Xue, Hong Lu, and Jian Pu. Vpl-slam: a vertical line supported point line monocular slam system. IEEE Transactions on Intelligent Transportation Systems, 25(8):9749â9761, 2024. 2

[2] Ziyu Chen, Jiawei Yang, Jiahui Huang, Riccardo de Lutio, Janick Martinez Esturo, Boris Ivanovic, Or Litany, Zan Gojcic, Sanja Fidler, Marco Pavone, et al. Omnire: Omni urban scene reconstruction. arXiv preprint arXiv:2408.16760, 2024. 2

[3] Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic, Marco Pavone, Georgios Pavlakos, et al. Instantsplat: Sparse-view gaussian splatting in seconds. arXiv preprint arXiv:2403.20309, 2024. 1

[4] Xin Fei, Wenzhao Zheng, Yueqi Duan, Wei Zhan, Masayoshi Tomizuka, Kurt Keutzer, and Jiwen Lu. Driv3r: Learning dense 4d reconstruction for autonomous driving. arXiv preprint arXiv:2412.06777, 2024. 1

[5] Adrien Gaidon, Qiao Wang, Yohann Cabon, and Eleonora Vig. Virtual worlds as proxy for multi-object tracking analysis. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4340â4349, 2016. 6

[6] Xin Gao and Jian Pu. Deep incomplete multi-view learning via cyclic permutation of vaes. In The Thirteenth International Conference on Learning Representations, 2025. 2

[7] Xin Gao, Jiyao Liu, Guanghao Li, Yueming Lyu, Jianxiong Gao, Weichen Yu, Ningsheng Xu, Liang Wang, Caifeng Shan, Ziwei Liu, et al. Good: Training-free guided diffusion sampling for out-ofdistribution detection. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 2

[8] Po-Han Huang, Kevin Matzen, Johannes Kopf, Narendra Ahuja, and Jia-Bin Huang. Deepmvs: Learning multi-view stereopsis. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2821â2830, 2018. 6

[9] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng Zhao, et al. Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. arXiv preprint arXiv:2505.23716, 2025. 1, 2

[10] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis. 3d gaussian Â¨ splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139â1, 2023. 7

[11] Vincent Leroy, Yohann Cabon, and JerÂ´ ome Revaud. Ë Grounding image matching in 3d with mast3r. In European Conference on Computer Vision, pages 71â91. Springer, 2024. 7

[12] Guanghao Li, Kerui Ren, Linning Xu, Zhewen Zheng, Changjian Jiang, Xin Gao, Bo Dai, Jian Pu, Mulin Yu, and Jiangmiao Pang. Artdeco: Toward high-fidelity on-the-fly reconstruction with hierarchical gaussian structure and feed-forward guidance. In The Fourteenth International Conference on Learning Representations. 2

[13] Guanghao Li, Yu Cao, Qi Chen, Xin Gao, Yifan Yang, and Jian Pu. Papl-slam: Principal axis-anchored monocular point-line slam. IEEE Robotics and Automation Letters, 2025.

[14] Guanghao Li, Qi Chen, Sijia Hu, Yuxiang Yan, and Jian Pu. Constrained gaussian splatting via implicit tsdf hash grid for dense rgb-d slam. IEEE Transactions on Artificial Intelligence, 2025.

[15] Guanghao Li, Qi Chen, Yuxiang Yan, and Jian Pu. Ecslam: Effectively constrained neural rgb-d slam with tsdf hash encoding and joint optimization. Pattern Recognition, 170:112034, 2026. 2

[16] Chenguo Lin, Yuchen Lin, Panwang Pan, Yifan Yu, Honglei Yan, Katerina Fragkiadaki, and Yadong Mu. Movies: Motion-aware 4d dynamic view synthesis in one second. arXiv preprint arXiv:2507.10065, 2025. 2

[17] Raul Mur-Artal and Juan D Tardos. Orb-slam2: An Â´ open-source slam system for monocular, stereo, and rgb-d cameras. IEEE transactions on robotics, 33(5): 1255â1262, 2017. 2

[18] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos. Orb-slam: A versatile and accurate monocular slam system. IEEE transactions on robotics, 31(5):1147â1163, 2015. 2

[19] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012. 6

[20] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Â´ Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179â12188, 2021. 4

[21] Brandon Smart, Chuanxia Zheng, Iro Laina, and Victor Adrian Prisacariu. Splatt3r: Zero-shot gaussian splatting from uncalibrated image pairs. 2024. 1

[22] Edgar Sucar, Zihang Lai, Eldar Insafutdinov, and Andrea Vedaldi. Dynamic point maps: A versatile representation for dynamic 3d reconstruction. arXiv preprint arXiv:2503.16318, 2025. 2, 3

[23] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James

Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2446â2454, 2020. 6

[24] Qijian Tian, Xin Tan, Yuan Xie, and Lizhuang Ma. Drivingforward: Feed-forward 3d gaussian splatting for driving scene reconstruction from flexible surround-view input. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7374â7382, 2025. 1, 2

[25] Jonas Uhrig, Nick Schneider, Lukas Schneider, Uwe Franke, Thomas Brox, and Andreas Geiger. Sparsity invariant cnns. In 2017 international conference on 3D Vision (3DV), pages 11â20. IEEE, 2017. 6

[26] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294â5306, 2025. 1, 2, 5, 6, 7

[27] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5261â5271, 2025. 1, 5

[28] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697â20709, 2024. 2, 7

[29] Mingwang Xu, Jiahao Cui, Feipeng Cai, Hanlin Shang, Zhihao Zhu, Shan Luan, Yifang Xu, Neng Zhang, Yaoyi Li, Jia Cai, et al. Wam-diff: A masked diffusion vla framework with moe and online reinforcement learning for autonomous driving. arXiv preprint arXiv:2512.11872, 2025. 1

[30] Yifang Xu, Jiahao Cui, Feipeng Cai, Zhihao Zhu, Hanlin Shang, Shan Luan, Mingwang Xu, Neng Zhang, Yaoyi Li, Jia Cai, et al. Wam-flow: Parallel coarse-to-fine motion planning via discrete flow matching for autonomous driving. arXiv preprint arXiv:2512.06112, 2025. 1

[31] Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, and Sida Peng. Street gaussians: Modeling dynamic urban scenes with gaussian splatting. In European Conference on Computer Vision, pages 156â173. Springer, 2024. 2

[32] Yuxiang Yan, Zhiyuan Zhou, Xin Gao, Guanghao Li, Shenglin Li, Jiaqi Chen, Qunyan Pu, and Jian

Pu. Learning spatial-aware manipulation ordering. In Advances in Neural Information Processing Systems (NeurIPS), 2025. 1

[33] Jiawei Yang, Jiahui Huang, Yuxiao Chen, Yan Wang, Boyi Li, Yurong You, Apoorva Sharma, Maximilian Igl, Peter Karkus, Danfei Xu, et al. Storm: Spatiotemporal reconstruction model for large-scale outdoor scenes. arXiv preprint arXiv:2501.00602, 2024. 2, 7

[34] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20331â20341, 2024. 7

[35] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arXiv:2410.03825, 2024. 7

[36] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gslrm: Large reconstruction model for 3d gaussian splatting. In European Conference on Computer Vision, pages 1â19. Springer, 2024. 7

[37] Hongyu Zhou, Jiahao Shao, Lu Xu, Dongfeng Bai, Weichao Qiu, Bingbing Liu, Yue Wang, Andreas Geiger, and Yiyi Liao. Hugs: Holistic urban 3d scene understanding via gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21336â21345, 2024. 2

[38] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. Streaming 4d visual geometry transformer. arXiv preprint arXiv:2507.11539, 2025. 2, 3, 6, 7