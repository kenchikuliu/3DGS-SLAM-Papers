# Learning Efficient and Generalizable Human Representation with Human Gaussian Model

Yifan Liu1,â, Shengjun Zhang1,â, Chensheng Dai1, Yang Chen3, Hao Liu2, Chen Li2, Yueqi Duan1â  1Tsinghua University, 2WeChat Vision, Tecent Inc., 3Nanyang Technological University {liuyifan22, zhangsj23}@mails.tsinghua.edu.cn, duanyueqi@tsinghua.edu.cn

https://github.com/Simon-Dcs/Human_Gaussian_Graph/

<!-- image-->  
(a) Examples of novel views and novel poses from our model

<!-- image-->  
(b) Performance and runtime comparison

Figure 1. Our method achieves state-of-the-art rendering quality while maintaining remarkable fast run-time performance. (a) Qualitative results: Our approach delivers high-fidelity results for both novel view synthesis and novel pose animation. (b) Performance comparison: Our method achieves the highest PSNR in both single view (yellow) and multi-view (blue) settings with superior computational efficiency.

## Abstract

Modeling animatable human avatars from videos is a long-standing and challenging problem. While conventional methods require per-instance optimization, recent feed-forward methods have been proposed to generate 3D Gaussians with a learnable network. However, these methods predict Gaussians for each frame independently, without fully capturing the relations of Gaussians from different timestamps. To address this, we propose Human Gaussian Graph to model the connection between predicted Gaussians and human SMPL mesh, so that we can leverage information from all frames to recover an animatable human representation. Specifically, the Human Gaussian Graph contains dual layers where Gaussians are the first layer nodes and mesh vertices serve as the second layer nodes. Based on this structure, we further propose the intra-node operation to aggregate various Gaussians connected to one

mesh vertex, and inter-node operation to support message passing among mesh node neighbors. Experimental results on novel view synthesis and novel pose animation demonstrate the efficiency and generalization of our method.

## 1. Introduction

Realistic human reconstruction is a fundamental task in computer vision with widespread application in virtual reality, gaming, healthcare and social media. Remarkable progress has been made using neural implicit representations [26, 27, 30] to model flexible topology, but these methods [4, 13, 14, 36, 38] suffer from expensive time consumption in training and rendering. Recently, 3D Gaussian Splatting [18] (3DGS) has drawn increasing attention for explicit Gaussian representations and real-time rendering performance. Benefiting from rasterization-based rendering, 3DGS avoids dense points querying in scene space so that it can maintain high efficiency and quality.

Since conventional methods [12, 20, 22, 33, 34, 39, 42] based on 3DGS require per-instance optimization, recent human reconstruction studies [29, 43, 55] focus on directly regress Gaussian parameters with feed-forward networks. Typically, these methods generate Gaussians with U-Net architectures [55] or latent reconstruction transformers [29] for each frame independently, without leveraging complementary information from other frames. Besides, the lack of alignment between Gaussian representations and the human SMPL [24, 32] mesh further limits the application of novel pose animation in downstream tasks.

To tackle these challenges, we propose Human Gaussian Graph to construct the relations of Gaussian groups from multiple frames and the human structure priors from SMPL mesh. Specifically, our Human Gaussian Graph contains dual layers, where Gaussians from all frames are the firstlayer nodes, and the SMPL vertices, which are equivalent throughout the temporal axis, are the second-layer nodes. Then, we define the edges between two-layer nodes as the alignment of Gaussians and SMPL vertices based on the spatial relations. We present intra-node operations between two-layer nodes to aggregate temporal information from multiple frames. Since the second-layer nodes are naturally connected according to human structures, we further introduce inter-node operation to support message passing between connected SMPL vertices in local regions. As shown in Figure 1, our method can generate high-quality generalizable and animatable human gaussian representations from videos. We conduct experiments on novel view synthesis and novel pose animation. While the optimization-based methods [20, 28, 48] require expansive training time, our method is more efficient in a feed-forward manner. While other generalizable methods fail to benefit from the multiple frames of videos, our graph structure models the relations of temporal information to generate mesh-aligned animatable human Gaussians. As shown in Figure 1, we surpass previous methods with less time consumption. Our main contributions are summarized as follows:

â¢ We introduce Human Gaussian Graph to effectively model the relations of cross-frame Gaussian primitives and human SMPL mesh in videos.

â¢ We present intra-node operations and inter-node operations to process the Human Gaussian Graph, enabling the interaction and aggregation across different nodes.

â¢ Experimental results illustrate that our method can generate high-quality generalizable and animatable human Gaussian representations from videos.

## 2. Related Work

3D Human Reconstruction. Reconstructing 3D humans is a research focus. The parametric template models [21, 25, 31, 35] regulate a strong geometric prior of the human body, fueling a surge of research on human body poses [8, 19, 40, 41, 44]. However, the explicit and predefined topology of parametric meshes cannot capture personalized appearances such as hair, and clothing [2, 3, 9]. Implicit methods [1, 4, 10, 11, 16, 37, 46, 54], utilizing SDF and NeRF, enables accurate depiction of 3D clothed humans. Methods like GTA [54] use transformers to map input images into 3D tri-plane features, which are then used for reconstruction. Recently, 3D Gaussian Splatting [18] has become the new trend in human reconstruction [12, 15, 20, 22, 33, 34, 39, 42]. Optimization-based methods such as GART [20] bind Gaussians onto the SMPL model and utilizes LBS skinning to map the Gaussians to poses in respective frames for supervision. Though high in quality, these methods are neither instant nor generalizable due to the time-consuming optimization process, limiting their downstream applications. We propose a method that effectively bypasses the optimization process and constructs high-quality generalizable avatars within inference time.

Generalizable Gaussian Model. pixelSplat [5] and MVSplat [6] are representative works showing that 3D Gaussians can be directly predicted from image pairs via feed-forward models, avoiding the time-consuming optimizing process. Methods like LGM [43, 50, 51, 53] predict Gaussian attributes for each input pixel in each view using large deterministic models with scaled training, and combine them as the final output. GPS-Gaussian [55] focuses on novel view synthesis by splatting each pixel into space based on the estimated depth, showcasing in experiments that the generalizable Gaussian models have potential in human reconstruction. The most relevant work to ours is HumanSplat [29], which introduces human geometric priors into feed-forward Gaussian networks. HumanSplat tokenizes the SMPL [25, 31] mesh and lets the image features attend to the SMPL tokens, injecting geometric priors into the pipeline. Though effective, their method is unable to deal with video inputs and cannot produce animatable avatars, In contrast, our method directly builds a graph based on the geometric priors to integrate cross-frame information, and our model outputs SMPL-aligned pose-driven Gaussians, opening up new possibilities for downstream applications. To our knowledge, we are the first work to achieve animatable Gaussian avatars within inference time.

## 3. Method

The overall framework is illustrated in Figure 2. Given a video $\{ I ^ { t } \} _ { t = 1 } ^ { T }$ , our goal is to reconstruct an animatable avatar within inference time. In our framework, we first build Gaussian representations $G ^ { t } ~ = ~ \{ \mu ^ { t } , f ^ { t } \}$ for each frame It through a feed-forward 3DGS network [43] with generative priors [23]. Then, we construct Human Gaussian Graph to model the relations between predicted Gaussians from multiple frames and the SMPL mesh [24], where Gaussians are the first layer nodes and SMPL vertices serve as the second layer nodes. To enable cross-frame Gaussian aggregation, we introduce an intra-node operation to extract features from different timesteps on each SMPL vertices. Furthermore, we design an intra-node operation to support message passing between SMPL vertices and their neighbors. In this way, our model reconstructs SMPL-aligned Gaussians that can be driven by pose, fueling downstream application like virtual reality and video games.

<!-- image-->  
Figure 2. Overview. Given an input human video, our goal is to build high-fidelity animatable Gaussian representations within inference time. We first establish frame-wise Gaussian representations through a feed-forward 3DGS network. Then we construct Human Gaussian Graph (HGG) to model the relations between predicted Gaussians from multiple frames and the SMPL mesh (Section 3.2). We introduce two complementary types of operations on the HGG: the intra-node operation that extracts temporal features across multiple timesteps, and the inter-node operation that facilitates robust local message passing between topologically adjacent nodes (Section 3.3). Finally, the Gaussians are updated into SMPL-aligned Gaussians through the HGG framework, enabling novel pose animation. (Section 3.4)

## 3.1. Preliminaries

SMPL model. SMPL [25] is a parametric human mesh model, which is created by skinning and blend shapes. The shape parameters $\beta \in \mathbb { R } ^ { \bar { 1 0 } }$ adjust the body shape and the pose parameters $\theta \in \mathbb { R } ^ { 2 4 \times 3 }$ re-poses the body to various gestures with Linear Blending Skinning (LBS). A 3D point x in the canonical space is transformed to a posed space defined by Î¸ as:

$$
\tilde { x } = \sum _ { k = 1 } ^ { K } w _ { k } \left( G _ { k } ( J , \theta ) x + b _ { k } ( J , \theta , \beta ) \right) ,\tag{1}
$$

where J includes K joint locations, $G _ { k } ( J , \theta )$ and $b _ { k } ( J , \theta , \beta )$ are the transformation matrix and translation vector of joint k, and $w _ { k }$ is the linear blend weight.

3D Gaussian Splatting. 3D Gaussian Splatting [17] represents 3D objects or scenes using a set of Gaussians, including a center position $\mu \in \mathbb { R } ^ { 3 }$ , an opacity $\alpha \in \mathbb { R }$ , a covariance matrix $\bar { \Sigma } \in \mathbb { R } ^ { 3 \times 3 }$ and spherical harmonics denoting the color $\mathbf { c } \in \mathbb { R } ^ { k }$ . The Gaussian function can be formulated as:

$$
G ( x ) = e ^ { - \frac { 1 } { 2 } ( x - \mu ) ^ { \top } \Sigma ^ { - 1 } ( x - \mu ) } ,\tag{2}
$$

where $\Sigma = R S S ^ { \top } R ^ { \top }$ , S is the scaling matrix and R is the rotation matrix. For every pixel, the color is rendered by a set of Gaussians sorted in depth order:

$$
C = \sum _ { i \in N } c _ { i } \alpha _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { i } ) .\tag{3}
$$

## 3.2. Human Gaussian Graph

Dual-layer Nodes Given the input video $\{ I ^ { t } \} _ { t = 1 } ^ { T } \in$ RT ÃHÃW Ã3, we generate multi-view images with a diffusion model [23], and predict the corresponding set of Gaus-

```latex
Algorithm 1 Constructing the Graph
Input: Input video $\{ I ^ { t } \} _ { t = 1 } ^ { T } .$ , SMPL poses $\{ p ^ { t } \} _ { i = 1 } ^ { N }$
Output: Human Gaussian Graph $\{ \mathcal { V } , \mathcal { G } , \mathcal { E } _ { v v } , \mathcal { E } _ { v g } \}$
for t â 1 to T do
$G ^ { t } = \{ g _ { m } ^ { t } \} _ { m = 1 } ^ { M } = \Phi _ { \mathrm { { G } } } ( I ^ { t } )$ â· per-frame Gaussians
end for
for t â 1 to T do
$\mathcal { V } ^ { t } = \{ v _ { i } ^ { t } \} _ { i = 1 } ^ { N } = \psi _ { \mathrm { L B S } } ( \mathcal { V } , p ^ { t } )$
for $m \gets 1$ to M do
$k = \arg$ min $\psi _ { \mathrm { d } } ( \mu _ { m } ^ { t } , v _ { i } ^ { t } )$
i
$\mathcal { E } _ { v g } ( g _ { m } ^ { t } , \stackrel { \cdot } { v } _ { k } ) = 1$ â· Gaussian-Mesh edges
end for
end for
for $s \gets 1$ to $N$ do
for $s \gets 1$ to N do
if $d ( v _ { s } , v _ { n } ) \leq d _ { 0 }$ then
$\mathcal { E } _ { v v } ( v _ { s } , v _ { t } ) = 1$ â· Mesh-level edges
end if
end for
end for
```

sian for each frame $I ^ { t } { : }$

$$
G ^ { t } = \{ g _ { m } ^ { t } \} _ { m = 1 } ^ { M } = \Phi _ { \mathrm { G } } ( I ^ { t } ) , \quad g _ { m } ^ { t } = \{ \mu _ { m } ^ { t } , f _ { m } ^ { t } \} ,\tag{4}
$$

where $\Phi _ { \mathrm { { G } } }$ denotes the Gaussian prediction model, $\mu _ { m } ^ { t } \in \mathbb { R } ^ { 3 }$ is the Gaussian center, and $f _ { m } ^ { t } \in \mathbb { R } ^ { C }$ is the Gaussian features. We define ${ \mathcal G } = \{ G ^ { t } \} _ { t = 1 } ^ { T }$ as the first-layer nodes, serving as the source of the information needed for reconstruction. The SMPL vertices $\mathcal { V } = \{ v _ { i } \} _ { i = 1 } ^ { N }$ serve as the secondlayer nodes, which will be leveraged as a set of equivalent points throughout the temporal axis.

Gaussian-Mesh Edges. To model the relations between Gaussians in different frames, we build the edges $\mathcal { E } _ { v g } \ \in$ $\{ 0 , 1 \} ^ { | \mathcal { G } | \times | \nu | }$ between Gaussian nodes G and mesh vertices $\nu .$ Given a frame $I ^ { t } ,$ we fisrt estimate the SMPL poses $p ^ { t } ~ = ~ \{ \beta ^ { t } , \theta ^ { t } \}$ for the current motion of the human. We transform the second-layer nodes V into the pose of current frame:

$$
\begin{array} { r } { \gamma ^ { t } = \{ v _ { i } ^ { t } \} _ { i = 1 } ^ { N } = \psi _ { \mathrm { L B S } } ( \mathcal { V } , p ^ { t } ) , } \end{array}\tag{5}
$$

where $\psi _ { \mathrm { L B S } }$ denotes the LBS algorithm in Eq. 1. For each Gaussian $g _ { m } ^ { t } \in \mathcal { G } ^ { t }$ , we connect it to its closet node $v _ { k } ^ { t }$ in the second layer:

$$
\mathcal { E } _ { v g } ( g _ { m } ^ { t } , v _ { k } ) = 1 , \quad \mathrm { w h e r e } \ k = \arg \operatorname* { m i n } _ { i } \ \psi _ { \mathrm { d } } ( \mu _ { m } ^ { t } , v _ { i } ^ { t } ) ,\tag{6}
$$

where $\psi _ { \mathrm { d } }$ stands for the Euclidean distance. Notably, Gaussians from different frames are connected to the same set of second-level nodes V, thus reorganizing the Gaussians from temporal grouping to spatial grouping.

Mesh-level Edges. The SMPL mesh provides a natural connectivity for SMPL vertices. We connect vertices with its neigbours in the SMPL mesh. Specifically, let $V _ { n }$ be a second-layer vertex. We define a distance $d ( n , n ^ { \prime } )$ on the mesh, denoting the number of mesh faces needed to connect the two vertices. We consider vertices with a distance no greater than a threshold $d _ { 0 }$ as neighbours:

Algorithm 2 Graph Operations   
Input: Human Gaussian Graph $\{ \gamma , \mathcal { G } , \mathcal { E } _ { v v } , \mathcal { E } _ { v g } \}$   
Output: SMPL-aligned Gaussians $\mathcal { G } ^ { \mathrm { s m p l } }$   
for $l \gets 1$ to L do â· stack L blocks   
for $n \gets 1$ to N do â· intra-node operation   
$B _ { n } ( { \mathcal G } ) = \{ g _ { m } ^ { t } , \mathrm { i f } \ : { \mathcal E } _ { v g } ( g _ { m } ^ { t } , v _ { n } ) = 1 \}$   
$q _ { n } \gets \mathrm { A t t e n t i o n } ( q = q _ { n } , k = v = B _ { n } ( \mathcal { G } ) )$   
$q _ { n } \gets \Phi _ { \mathrm { F F N } } ( q _ { n } )$   
end for   
for $V _ { n } \in \{ V _ { n } \} _ { n = 1 } ^ { N }$ do â· inter-node operation   
$B _ { n } ( \mathcal { V } ) = \{ v _ { s } , \mathrm { i f } \mathcal { E } _ { v v } ( v _ { s } , v _ { n } ) = 1 \}$   
$q _ { n } \gets \mathrm { A t t e n t i o n } ( q = q _ { n } , k = v = B _ { n } ( \mathcal { V } ) )$   
$q _ { n } \gets \Phi _ { \mathrm { F F N } } ( q _ { n } )$   
end for   
end for   
for m â 1 to M do   
$V _ { n }  V _ { n }$ ,where $E _ { v g } ^ { t _ { 0 } } ( m , n ) = 1 \}$   
$g _ { m } ^ { \mathrm { s m p l } } = \Phi _ { \mathrm { A t t } } \left( q = g _ { m } ^ { t _ { 0 } } , k = v = q _ { n } \right) + g _ { m } ^ { t _ { 0 } }$   
end for

$$
\mathcal { E } _ { v v } ( v _ { s } , v _ { t } ) = 1 , \mathrm { ~ i f ~ } d ( v _ { s } , v _ { t } ) \leq d _ { 0 } .\tag{7}
$$

The construction of our Human Gaussian Graph is illustrated in Algorithm 1.

## 3.3. Graph Operations

Based on our Human Gaussian Graph, we introduce two types of operations for information aggregation and interaction. The two operations are illustrated in Figure 2 and Algorithm 2.

Intra-node Operation. To extract rich information of Gaussian from different timesteps, we propose an intranode operation on $\mathcal { E } _ { v g }$ . For each mesh vertex $v _ { n } ,$ we define a learnable query $q _ { n }$ for capturing individual-agnostic features. We define the neighbor Gaussian group of a mesh vertex $v _ { n }$ as:

$$
B _ { n } ( { \mathcal { G } } ) = \{ g _ { m } ^ { t } , { \mathrm { i f } } \ : \mathcal { E } _ { v g } ( g _ { m } ^ { t } , v _ { n } ) = 1 \} .\tag{8}
$$

Thus, the learnable query $q _ { n }$ can be updated by the attention mechanism:

$$
\tilde { q } _ { n } = \Phi _ { \mathrm { A t t } } \left( q = q _ { n } , k = v = B _ { n } ( \mathcal { G } ) \right) ,\tag{9}
$$

where the key, query and value of $\Phi _ { \mathrm { A t t } }$ are formulated as:

$$
Q = h _ { Q } ( q _ { n } ) ,\tag{10}
$$

$$
K = h _ { K } \left( B _ { n } ( { \mathcal { G } } ) \right) ,\tag{11}
$$

$$
V = h _ { K } \left( B _ { n } ( { \mathcal { G } } ) \right) ,\tag{12}
$$

Table 1. Quantitative comparison on novel view synthesis and novel pose animation with other methods. âGen.â indicates generalizable methods and âAni.â denotes capability for novel pose animation. â : LGM is fine-tuned on our dataset. As LGM lacks inherent animation capability, we directly bind its output to SMPL meshes for comprehensive comparative evaluation.
<table><tr><td rowspan="2"></td><td rowspan="2">Setting</td><td rowspan="2">Category Gen. Ani.</td><td colspan="3">Novel View</td><td colspan="3">Novel Pose</td><td rowspan="2">Time</td></tr><tr><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td></tr><tr><td>GART [20]</td><td>Single</td><td>X â</td><td>0.906</td><td>20.109</td><td>0.098</td><td>0.866</td><td>18.964</td><td>0.124</td><td>1.9m</td></tr><tr><td>ExAvatar [28]</td><td>Single</td><td>X â</td><td>0.910</td><td>21.450</td><td>0.077</td><td>0.894</td><td>19.327</td><td>0.087</td><td>2.9h</td></tr><tr><td>LGMâ [ [43]</td><td>Single</td><td>â X</td><td>0.912</td><td>21.274</td><td>0.093</td><td>0.889</td><td>19.560</td><td>0.126</td><td>9.7s</td></tr><tr><td>Ours</td><td>Single</td><td>â â</td><td>0.920</td><td>23.112</td><td>0.080</td><td>0.896</td><td>21.857</td><td>0.111</td><td>9.7s</td></tr><tr><td>4DGS [48]</td><td>Multi</td><td>X Ã</td><td>0.814</td><td>19.072</td><td>0.139</td><td>-</td><td>-</td><td>-</td><td>8.5m</td></tr><tr><td>GPS-Gaussian [55]</td><td>Multi</td><td>â X</td><td>0.827</td><td>19.129</td><td>0.134</td><td>-</td><td>-</td><td>-</td><td>0.6s</td></tr><tr><td>Ours</td><td>Multi</td><td>â â</td><td>0.955</td><td>26.536</td><td>0.048</td><td>0.934</td><td>24.013</td><td>0.066</td><td>0.9s</td></tr></table>

where $h _ { Q }$ $h _ { K }$ , $h _ { V }$ stand for projections. After this, the output is passed through a standard feed-forward network in the transformer:

$$
q _ { n } = \Phi _ { \mathrm { F F N } } ( \tilde { q } _ { n } ) .\tag{13}
$$

Inter-node Operation To support message passing across mesh vertices, we further propose an inter-node operation on ${ \mathcal { E } } _ { v v }$ . Similarly, we define the neighbor Gaussian group of a mesh vertex $v _ { n }$ as:

$$
B _ { n } ( \mathcal { V } ) = \{ v _ { s } , \mathrm { i f } \mathcal { E } _ { v v } ( v _ { s } , v _ { n } ) = 1 \} .\tag{14}
$$

Thus, the learnable query $q _ { n }$ can be updated by:

$$
\tilde { q } _ { n } = \Phi _ { \mathrm { A t t } } \left( q = q _ { n } , k = v = B _ { n } ( \mathcal { V } ) \right) ,\tag{15}
$$

where $\Phi _ { \mathrm { A t t } }$ stand for attention mechanism. After this, the output is passed through a standard feed-forward network in the transformer:

$$
q _ { n } = \Phi _ { \mathrm { F F N } } ( \tilde { q } _ { n } ) .\tag{16}
$$

The learnable query $\{ q _ { n } \} _ { n = 1 } ^ { N }$ is aligned with the SMPL mesh, which has integrated features of Gaussians from different frames. Since the inter-node operation enables the model to achieve integration within a relatively local region, multiple operations can be stacked to broaden the receptive field of each node.

## 3.4. Training Objectives

Given one chosen frame of initial Gaussian $\begin{array} { r l } { G ^ { t _ { 0 } } } & { { } = } \end{array}$ $\{ g _ { m } ^ { t _ { 0 } } \} _ { m = 1 } ^ { M }$ , we introduce the aforementioned $\left\{ q _ { n } \right\}$ to refine the Gaussian features:

$$
g _ { m } ^ { \mathrm { s m p l } } = \Phi _ { \mathrm { A t t } } \left( q = g _ { m } ^ { t _ { 0 } } , k = v = q _ { n } \right) + g _ { m } ^ { t _ { 0 } } ,
$$

$$
\mathrm { w h e r e } n = \operatorname * { a r g m i n } _ { i } \ \psi _ { \mathrm { d } } ( \mu _ { m } ^ { t _ { 0 } } , v _ { i } ^ { t } ) .\tag{17}
$$

(18)

The final SMPL Gaussians $G ^ { s m p l } = \{ g _ { m } ^ { \mathrm { s m p l } } \} _ { m = 1 } ^ { M }$ are temporal-invariant. Specifically, for each frame with multiview images $I _ { m v } ^ { t } ,$ the Gaussian avatar is first re-posed into the estimated SMPL pose of the frame $p _ { t } .$ Then, the reposed Gaussians are rendered using the standard 3D Gaussian rasterization. Following the instruction of LGM [43], we use mean square error loss and LPIPS loss to the RGB image and mean square error loss on the alpha image:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { M S E } } ( \tilde { I } _ { \mathrm { r g b } } , I _ { \mathrm { r g b } } ) { + } \alpha _ { 1 } \mathcal { L } _ { \mathrm { L P I P S } } ( \tilde { I } _ { \mathrm { r g b } } , I _ { \mathrm { r g b } } ) { + } \alpha _ { 2 } \mathcal { L } _ { \mathrm { M S E } } ( \tilde { I } _ { \alpha } , I _ { \alpha } ) ,
$$

where $I _ { \mathrm { r g b } } , I _ { \alpha }$ are the ground truth images and $\tilde { I } _ { \mathrm { r g b } } , \tilde { I } _ { \alpha }$ are the corresponding rendered output images from our model.

## 4. Experiment

## 4.1. Implementation Details

Dataset. We utilize MvHumanNet [49] as our training dataset. MvHumanNet is a large-scale multi-view video human dataset that includes estimated SMPL parameters. For our experiments, we curated a training split comprising 2,000 distinct individuals and sampled 60 frames from each subjectâs sequence. This sampling strategy yielded a diverse corpus of 120,000 poses, featuring varied individuals in a wide range of postures and movements.

Training. Following LGM [43], we leverage the four ground-truth views (front, left, back, and right) from the MvHumanNet dataset as input to the LGM U-Net. We render our posed SMPL-aligned Gaussians onto eight views, comprising the four input views plus four additional randomly sampled viewpoints. The loss function described in Section 3.4 is computed between these rendered images and their corresponding ground-truth counterparts. We keep the LGM U-Net frozen throughout the training process, focusing optimization exclusively on the HGG modules. Training convergence is achieved within two days using eight NVIDIA A800 (40GB) GPUs.

GART

ExAvatar

LGM (ft)

Ours (mono)

GPS-Gaussian

Ours (mv)

GT

Figure 3. Qualitative comparison of ours against GART [20], ExAvatar [28], LGM [43] and GPS-Gaussian [55] on MvHumanNet dataset [49]. Our approach achieves the highest visual fidelity and reconstruction quality in both single-view and multi-view setting.

Inference time. Our framework supports both multiview and single-view video inputs. For multi-view processing, we directly feed four images per frame into the network, generating SMPL-aligned Gaussians in 0.9 seconds. For single-view input, we employ a fine-tuned Wonder3D [23] multi-view diffusion model to synthesize multiview videos, requiring 8.8 seconds for diffusion, resulting in a total inference time of 9.7 seconds per frame. Once constructed, the SMPL-aligned Gaussian Avatar renders novel views and poses at over 120 FPS on an A800 GPU.

## 4.2. Comparison

Setting. Given a video, the model should be able to output arbitrary views of the human in the video. Some of the methods could model an animatable avatar and generate views in novel poses. We evaluated the quality of the novel view and pose using PSNR, SSIM [45] and LPIPS [52] metrics. Table 1 presents a comprehensive comparison between our approach and contemporary Gaussian methods across both monocular and multi-view video settings. Gaussian methods fall into two categories: optimization-based approaches such as 4DGS [47], GART [20], and ExAvatar [28]; and generalizable methods including LGM [43] and GPS-Gaussian [55].

We conduct a quantitative comparison on 10 MvHuman-Net [49] scans, with detailed information about the evaluation split available in the supplementary material. To ensure

Baseline

<!-- image-->  
w.o. intra  
w.o. inter  
Ours  
GT

Figure 4. Qualitative results for novel pose animation. We evaluated each module in our proposed Human Gaussian Graph and analyzed the growth. The results demonstrate the improvement with each component and the overall quality of our novel poses. We kindly refer readers to the supplementary for more visualization results.

fairness, we fine-tuned LGM [43] on our training dataset and fully optimized each optimization-based method by selecting their longest training steps.

Quantitative Comparison. As shown in Table 1, our method outperforms the counterparts on both single-view setting and multi-view setting, as well as both novel view synthesis task and novel pose animation task, demonstrating the effectiveness of our design. Specifically, on novel view synthesis for single view, we achieve a boost of 1.6 dB of PSNR against ExAvatar [28]. Notably, ExAvatar excels in front view reconstruction and especially face modeling, but our overall reconstruction results are better than theirs, indicating the strong 3D reconstruction capabilities of our model thanks to the Human Gaussian Graph. Our advantage over LGM [43], which takes one frame as input, demonstrate the value of gathering information across the frames. For multi-view reconstruction, ours method is significantly ahead. Current methods like 4DGS [48] and GPS-Gaussian [55] are designed only for rather dense-view inputs (i.e. 16 views for GPS-Gaussian), and perform suboptimally when the input multi-view is sparse. In contrast, our method showcase efficient usage of the multi-view information via HGG, enabling realistic reconstruction results. For novel pose animation, our method also demonstrates a significant advantage of 2.3 dB of PSNR. This indicates the high-quality of our reconstructed animatable avatar represented by SMPL-aligned Gaussians, fueling a variety of downstream applications and tasks.

Qualitative Comparison. As illustrated in Figure 3, by integrating cross-frame information via HGG, our model achieves higher fidelity in novel view synthesis against counterparts. For single-view setting, optimization-based methods like GART [20] and ExAvatar [28] reconstructs high-quality training views, but neglects 3D consistency and result in collapsed novel views. LGM [43] can only leverage one frame, and suffers from artifacts generated by the multi-view diffusion in its pipeline. Though our method also faces the multi-view diffusion problem, HGG can effectively suppress the artifacts with extra knowledge from other frames and neighboring second-layer nodes. For multi-view setting, GPS-Gaussian [55] suffers severe hallucination when fusing the sparse multi-views. Our method achieves consistently photorealistic quality with clear contour and satisfactory details, demonstrating the effectiveness of HGG. Notably, our method achieves high-quality face and hand reconstruction without bells and whistles, which is regarded as a severe obstacle [28, 29].

Table 2. Ablation studies. (a) âBaselineâ stands for the model without Human Gaussian Graph and learnable mesh-aligned queries, âintraâ represents the intra-node transformer and âinterâ is the inter-node transformer. Each component can bring improvement to the architecture. (b) While 1 stack of âintra interâ pair already boosts the performance, adding more layers would be beneficial. Metrics are evaluated on the novel pose task in the multi-view setting.  
(a) Ablation study for proposed modules  
(b) Ablation study for scaling up
<table><tr><td>Method</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td></tr><tr><td>Baseline</td><td> $0 . 9 1 5 _ { ( - 0 . 0 1 9 ) }$ </td><td> $2 1 . 3 2 7 _ { ( - 2 . 6 8 6 ) }$ </td><td> $0 . 0 9 2 _ { ( + 0 . 0 2 6 ) }$ </td></tr><tr><td>W.o. intra</td><td> $0 . 9 3 0 _ { ( - 0 . 0 0 4 ) }$ </td><td> $2 1 . 9 0 2 _ { ( - 2 . 1 1 1 ) }$ </td><td> $0 . 0 9 0 _ { ( + 0 . 0 2 4 ) }$ </td></tr><tr><td>W.o. inter</td><td> $0 . 9 3 1 _ { ( - 0 . 0 0 3 ) }$ </td><td> $2 3 . 1 0 5 _ { ( - 0 . 9 0 8 ) }$ </td><td> $0 . 0 8 7 _ { ( + 0 . 0 2 1 ) }$ </td></tr><tr><td>Ours</td><td>0.934</td><td> $\mathbf { 2 4 . 0 1 3 }$ </td><td>0.066</td></tr></table>

## 4.3. Ablation Study

We conduct an ablation study on the challenging novel pose animation for each of the proposed modules in our HGG. The baseline model is to directly bind predicted Gaussians to SMPL and drive them with pose parameters. We provide quantitative results in 2a and qualitative results in Fig. 3.

Learnable mesh queries. The learnable mesh queries, binding to the SMPL vertices, serve as information collectors and human-agnostic bias. Even without the graph architecture and the information-gathering process, the learnable queries can still provide a slight enhancement (i.e. 0.57 in PSNR). We attribute this to the basic human obtained via massive training, which will help regulate the Gaussianaligning process, providing a boost in visual qualities.

Intra-node operation. The intra-node operation is aimed for efficiently extracting information from the Gaussian nodes across the temporal axis. Comparing Line 2 and 4 in Table 2a, the intra-node operation brings about a boost for 2.1 dB in PSNR. As shown in Figure 4, the intra-node operation significantly promotes the details such as in the face, the hands, and the printed patterns on the clothes. It demonstrates that information from other frames can indeed enhance the quality of Gaussians, underlying the importance of cross-frame interaction.

Inter-node operation. The inter-node operation, enabling neigbour communication across second-layer nodes, is effective when intra-node operation is active. Comparing Line 3 and 5 in Table 2a, adding inter-node operation can bring about a further 0.9 dB of PSNR boost. This result proves the necessity of feature aggregation and purification in localities in the second layer. Effectiveness of this module is also illustrated in Fig. 4, where the noise and aliases are significantly suppressed.

<table><tr><td>Num Layers</td><td>SSIMâ</td><td>PSNRâ</td><td>LPIPSâ</td></tr><tr><td>L = 0</td><td> $0 . 9 3 0 _ { ( - 0 . 0 0 4 ) }$ </td><td> $2 1 . 9 0 2 _ { ( - 2 . 1 1 1 ) }$ </td><td> $0 . 0 9 0 _ { ( + 0 . 0 2 4 ) }$ </td></tr><tr><td> $\mathrm { { L } } = 1$ </td><td> $0 . 9 2 6 _ { ( - 0 . 0 0 8 ) }$ </td><td> $2 3 . 4 5 5 _ { ( - 0 . 5 5 8 ) }$ </td><td> $0 . 0 7 1 _ { ( + 0 . 0 0 5 ) }$ </td></tr><tr><td> $\mathrm { L } = 3$ </td><td> $0 . 9 2 9 _ { ( - 0 . 0 0 5 ) }$ </td><td> $2 3 . 9 2 5 _ { ( - 0 . 0 8 8 ) }$ </td><td> $0 . 0 7 3 _ { ( + 0 . 0 0 7 ) }$ </td></tr><tr><td>Ours (L = 6)</td><td>0.934</td><td>24.013</td><td>0.066</td></tr></table>

Scaling up. Theoretically, stacking the âinter-interâ pairs in Section 3.3 can enlarge the perception field for each secondlayer node, analogous to convolutions. We conducted an experiment with the number of layers stacked in Table 2b, which shows the increasing number of layers can demonstrably boost performance.

## 5. Conclusion

We present a pioneering generalizable and animatable Gaussian human network that derives SMPL-aligned Gaussians from monocular or multi-view videos within inference time. This model proposes a novel dual-level Human Gaussian Graph, enabling effective and efficient feature communication both across the temporal axis and within the topological neighbourhood. Extensive quantitative experiments demonstrate that our method surpasses existing methods in both single-view and multi-view settings, including novel view synthesis and novel pose animation. This capability opens up a broad spectrum of downstream applications.

Limitations and Future works. We observed some gaps in the experiment results that could potentially be addressed by future researches. (1) Quality gaps between monocular setting and multi-view setting. The monocular setting lags behind multi-view results in a large scale (i.e. 3.4 dB in PSNR). We attribute it to the absence of open source realistic human diffusion model. Researches on human diffusion models would fuel this field. (2) Detail degradation for novel poses. Though we achieve SOTA in novel pose animation, there is a gap to the results in novel views (i.e. 1.5- 2.5 dB in PSNR). We attribute it to the inaccurate SMPL parameters from the dataset. Future works could address this by introducing SMPL parameter refinement techniques.

Acknowledgments. This work was supported in part by the National Natural Science Foundation of China under Grant 62206147, and in part by 2024 WeChat Vision, Tecent Inc. Rhino-Bird Focused Research Program.

## References

[1] Badour AlBahar, Shunsuke Saito, Hung-Yu Tseng, Changil Kim, Johannes Kopf, and Jia-Bin Huang. Single-image 3d human digitization with shape-guided diffusion. In SIG-GRAPH Asia 2023 Conference Papers, 2023. 2

[2] Thiemo Alldieck, Marcus Magnor, Weipeng Xu, Christian Theobalt, and Gerard Pons-Moll. Video based reconstruction of 3d people models. In CVPR, pages 8387â8397, 2018. 2

[3] Thiemo Alldieck, Marcus Magnor, Weipeng Xu, Christian Theobalt, and Gerard Pons-Moll. Detailed human avatars from monocular video. In 2018 International Conference on 3D Vision (3DV), pages 98â109. IEEE, 2018. 2

[4] Thiemo Alldieck, Mihai Zanfir, and Cristian Sminchisescu. Photorealistic monocular 3d reconstruction of humans wearing clothing. In CVPR, 2022. 1, 2

[5] David Charatan, Sizhe Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. arXiv preprint arXiv:2312.12337, 2023. 2

[6] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. arXiv preprint arXiv:2403.14627, 2024. 2

[7] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, 2023. 1

[8] Shubham Goel, Georgios Pavlakos, Jathushan Rajasegaran, Angjoo Kanazawa, and Jitendra Malik. Humans in 4d: Reconstructing and tracking humans with transformers. arXiv preprint arXiv:2305.20091, 2023. 2

[9] Marc Habermann, Weipeng Xu, Michael Zollhoefer, Gerard Pons-Moll, and Christian Theobalt. Livecap: Real-time human performance capture from monocular video. ACM Transactions On Graphics (TOG), 38(2):1â17, 2019. 2

[10] Sang-Hun Han, Min-Gyu Park, Ju Hong Yoon, Ju-Mi Kang, Young-Jae Park, and Hae-Gon Jeon. High-fidelity 3d human digitization from single 2k resolution images. In CVPR, 2023. 2

[11] Tong He, John P. Collomosse, Hailin Jin, and Stefano Soatto. Geo-PIFu: Geometry and pixel aligned implicit functions for single-view human reconstruction. In NeurIPS, 2020. 2

[12] Liangxiao Hu, Hongwen Zhang, Yuxiang Zhang, Boyao Zhou, Boning Liu, Shengping Zhang, and Liqiang Nie. Gaussianavatar: Towards realistic human avatar modeling from a single video via animatable 3d gaussians. In CVPR, 2024. 2

[13] Shoukang Hu, Fangzhou Hong, Liang Pan, Haiyi Mei, Lei Yang, and Ziwei Liu. Sherf: Generalizable human nerf from a single image. In ICCV, 2023. 1

[14] Yangyi Huang, Hongwei Yi, Weiyang Liu, Haofan Wang, Boxi Wu, Wenxiao Wang, Binbin Lin, Debing Zhang, and Deng Cai. One-shot implicit animatable avatars with modelbased priors. In ICCV, 2023. 1

[15] Rohit Jena, Ganesh Subramanian Iyer, Siddharth Choudhary, Brandon Smith, Pratik Chaudhari, and James Gee. Splatarmor: Articulated gaussian splatting for animatable humans from monocular rgb videos. arXiv preprint arXiv:2311.10812, 2023. 2

[16] Yuheng Jiang, Kaixin Yao, Zhuo Su, Zhehao Shen, Haimin Luo, and Lan Xu. Instant-nvr: Instant neural volumetric rendering for human-object interactions from monocular rgbd stream. In CVPR, 2023. 2

[17] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG), 2023. 3

[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, Â¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 1, 2

[19] Muhammed Kocabas, Nikos Athanasiou, and Michael J Black. Vibe: Video inference for human body pose and shape estimation. In CVPR, pages 5253â5263, 2020. 2

[20] Jiahui Lei, Yufu Wang, Georgios Pavlakos, Lingjie Liu, and Kostas Daniilidis. Gart: Gaussian articulated template models. In CVPR, 2024. 2, 5, 6, 7

[21] Tianye Li, Timo Bolkart, Michael. J. Black, Hao Li, and Javier Romero. Learning a model of facial shape and expression from 4D scans. ACM Transactions on Graphics, (Proc. SIGGRAPH Asia), 36(6):194:1â194:17, 2017. 2

[22] Xinqi Liu, Chenming Wu, Jialun Liu, Xing Liu, Chen Zhao, Haocheng Feng, Errui Ding, and Jingdong Wang. Gva: Reconstructing vivid 3d gaussian avatars from monocular videos. arXiv preprint arXiv:2402.16607, 2024. 2

[23] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, and Wenping Wang. Wonder3d: Single image to 3d using cross-domain diffusion, 2023. 2, 3, 6, 1

[24] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multiperson linear model. ACM transactions on graphics (TOG), 2015. 2, 3

[25] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multiperson linear model. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pages 851â866. 2023. 2, 3

[26] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3d reconstruction in function space. In CVPR, 2019. 1

[27] Ben Mildenhall, Pratul Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 1

[28] Gyeongsik Moon, Takaaki Shiratori, and Shunsuke Saito. Expressive whole-body 3d gaussian avatar. In ECCV, 2024. 2, 5, 6, 7, 8

[29] Panwang Pan, Zhuo Su, Chenguo Lin, Zhen Fan, Yongjie Zhang, Zeming Li, Tingting Shen, Yadong Mu, and Yebin Liu. Humansplat: Generalizable single-image human gaussian splatting with structure priors. arXiv preprint arXiv:2406.12459, 2024. 2, 8

[30] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In CVPR, 2019. 1

[31] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. Expressive body capture: 3d hands, face, and body from a single image. In CVPR, pages 10975â 10985, 2019. 2

[32] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3D hands, face, and body from a single image. In CVPR, 2019. 2

[33] Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide Davoli, Simon Giebenhain, and Matthias NieÃner. Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians. In CVPR, pages 20299â20309, 2024. 2

[34] Zhiyin Qian, Shaofei Wang, Marko Mihajlovic, Andreas Geiger, and Siyu Tang. 3dgs-avatar: Animatable avatars via deformable 3d gaussian splatting. 2024. 2

[35] Javier Romero, Dimitrios Tzionas, and Michael J. Black. Embodied hands: Modeling and capturing hands and bodies together. ACM Transactions on Graphics, (Proc. SIG-GRAPH Asia), 36(6), 2017. 2

[36] Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo Morishima, Angjoo Kanazawa, and Hao Li. Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization. In ICCV, 2019. 1

[37] Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo Morishima, Hao Li, and Angjoo Kanazawa. PIFu: Pixel-aligned implicit function for high-resolution clothed human digitization. In ICCV, 2019. 2

[38] Shunsuke Saito, Tomas Simon, Jason Saragih, and Hanbyul Joo. Pifuhd: Multi-level pixel-aligned implicit function for high-resolution 3d human digitization. In CVPR, 2020. 1

[39] Zhijing Shao, Zhaolong Wang, Zhuang Li, Duotun Wang, Xiangru Lin, Yu Zhang, Mingming Fan, and Zeyu Wang. SplattingAvatar: Realistic Real-Time Human Avatars with Mesh-Embedded Gaussian Splatting. In CVPR, 2024. 2

[40] Yu Sun, Qian Bao, Wu Liu, Yili Fu, Michael J Black, and Tao Mei. Monocular, one-stage, regression of multiple 3d people. In ICCV, pages 11179â11188, 2021. 2

[41] Yu Sun, Qian Bao, Wu Liu, Tao Mei, and Michael J Black. Trace: 5d temporal regression of avatars with dynamic cameras in 3d environments. In CVPR, pages 8856â8866, 2023. 2

[42] David Svitov, Pietro Morerio, Lourdes Agapito, and Alessio Del Bue. Haha: Highly articulated gaussian human avatars with textured mesh prior. arXiv preprint arXiv:2404.01053, 2024. 2

[43] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. ECCV, 2024. 2, 5, 6, 7, 8

[44] Yufu Wang and Kostas Daniilidis. Refit: Recurrent fitting network for 3d human recovery. In ICCV, pages 14644â 14654, 2023. 2

[45] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing (TIP), 2004. 6

[46] Zhenzhen Weng, Jingyuan Liu, Hao Tan, Zhan Xu, Yang Zhou, Serena Yeung-Levy, and Jimei Yang. Template-free single-view 3d human digitalization with diffusion-guided lrm. arXiv preprint arXiv:2401.12175, 2024. 2

[47] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. arXiv preprint arXiv:2310.08528, 2023. 6

[48] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In CVPR, pages 20310â20320, 2024. 2, 5, 7

[49] Zhangyang Xiong, Chenghong Li, Kenkun Liu, Hongjie Liao, Jianqiao Hu, Junyi Zhu, Shuliang Ning, Lingteng Qiu, Chongjie Wang, Shijie Wang, et al. Mvhumannet: A largescale dataset of multi-view daily dressing human captures. In CVPR, pages 19801â19811, 2024. 5, 6

[50] Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen, Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon Wetzstein. Grm: Large gaussian reconstruction model for efficient 3d reconstruction and generation. arXiv preprint arXiv:2403.14621, 2024. 2

[51] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. In ECCV, 2025. 2

[52] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6

[53] Shengjun Zhang, Xin Fei, Fangfu Liu, Haixu Song, and Yueqi Duan. Gaussian graph network: Learning efficient and generatlizable gaussian representations from multi-view images. NeurIPS, 2024. 2

[54] Zechuan Zhang, Li Sun, Zongxin Yang, Ling Chen, and Yi Yang. Global-correlated 3d-decoupling transformer for clothed avatar reconstruction. In NeurIPS, 2023. 2

[55] Shunyuan Zheng, Boyao Zhou, Ruizhi Shao, Boning Liu, Shengping Zhang, Liqiang Nie, and Yebin Liu. Gpsgaussian: Generalizable pixel-wise 3d gaussian splatting for real-time human novel view synthesis. In CVPR, 2024. 2, 5, 6, 7, 8

# Learning Efficient and Generalizable Human Representation with Human Gaussian Model

Supplementary Material

## 6. Implementation Details

Fine-tuning multi-view diffusion. We utilize the Wonder3D [23] as the multi-view diffusion model. As the Wonder3D model is designed for objects and trained on Objaverse [7], it has limited knowledge about humans and directly applying it into our architecture would lead to dissatisfactory results. To mitigate this problem, we fine-tuned Wonder3D with the MvHumanNet dataset. To ensure training efficiency, we only selected 600 humans and used 30 poses for each scan, summing to 18K training data. We input the front view and supervise the output with corresponding front, back, left and right views. The resolution is set to 256, and we use a learning rate of 5e-5 at the mixed training stage and 2.5e-5 at joint training stage. The training converges on 8 Ã Nvidia A800 GPUs in 9 days, with a batch size of 4 per GPU. Though improved, the overall result is still non-optimal, leading to the gap between the monocular setting and the multiview setting in our experiments.

Fine-tuning Gaussian reconstruction model. To obtain an initial set of Gaussians mentioned in Section 3.2:

$$
G ^ { t } = \{ g _ { m } ^ { t } \} _ { m = 1 } ^ { M } , \quad g _ { m } ^ { t } = \{ \mu _ { m } ^ { t } , f _ { m } ^ { t } \} ,\tag{19}
$$

we leverage a fine-tuned LGM model. We use the same dataset mentioned in Section 4.1, and fine-tuned the âlargeââ LGM with an input resolution of 256 and an output Gaussian resolution of 128 per view. We follow the original LGM configuration for the learning rate and batch size. The training converges on 8 Ã Nvidia A800 GPUs in 5 days.

Training details. We train our HGG modules with the finetuned LGM model frozen. We uniformly sample 8 frames from each video and use the 8 frames as input for our module. The learning rate is set to 4e-4, gradient clip is 1.0, batch size to 1 per GPU and gradient accumulation steps to 8. We trained our model on 8 Ã Nvidia A800 GPUs, and it converges in 18 hours.

Evaluation split. Our evaluation split is separated from the training split. We randomly selected 10 scans in the MvHumanNet as the evaluation split. Their IDs are listed as follows: 200102, 200114, 200125, 200134, 200137, 200151, 200535, 202148, 202209, 204157.

## 7. More Analysis

## 7.1. Efficiency

Though processing large amount of information across the frames, HGG is highly-efficient thanks to the design of learnable queries, negligible compared with the LGM Unet. In this paragraph, the efficiency of each module will be theoretically analyzed.

<!-- image-->  
Figure 5. Fail cases. (a) Wonder3D fails to generate reasonable back and side views, resulting in the failure for LGM Gaussian reconstruction. (b) Wonder3D generates multi-views of good quality, but LGM reconstruction fails due to inconsistent camera constraints provided by Wonder3D.

The intra-node transformer enables efficient communication with the Gaussians. If attention is directly applied to the union of all Gaussian sets, the complexity would be $O ( M ^ { 2 } T ^ { 2 } D ^ { 2 } )$ , M is the number of feed-forward Gaussians per frame, T is the number of the frames, and D denotes the dimension of the features. Given that N is often at the 10,000 level and a video typically have hundreds of frames, it would become an unacceptable computation bottleneck both for time and memory. With our HGG, Gaussians are collected by the mesh vertices, and the complexity becomes linear when each mesh node applies cross attention with the affiliated Gaussians with learnable queries:

$$
O ( ( M T / N ) * N * D ^ { 2 } ) = O ( M T D ^ { 2 } )\tag{20}
$$

This would be over $1 0 ^ { 6 }$ times more efficient than vanilla cross attention between Gaussians.

On the other hand, the computation complexity of the inter-node attention is rather small, thanks to the connectivity given by mesh. Empirically, a node has approximately 10 neigbours, so the complexity is only $O ( 1 0 N D ^ { 2 } )$ , which is negligible compared with other modules.

<!-- image-->  
Figure 6. More qualitative results on novel pose animation. The human avatar in various poses indicates the high-quality of our reconstructed 3D avatar.

Table 3. Quantitative results of 50 testing examples.
<table><tr><td rowspan="2"></td><td colspan="3">Deepfashion</td><td colspan="3">MvHumanNet</td></tr><tr><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td><td>PSNRâ</td><td>SSIMâ</td><td>LPIPSâ</td></tr><tr><td>LGM</td><td>18.01</td><td>0.846</td><td>0.196</td><td>19.54</td><td>0.887</td><td>0.129</td></tr><tr><td>Ours (LGM)</td><td>20.31</td><td>0.901</td><td>0.172</td><td>21.82</td><td>0.892</td><td>0.112</td></tr><tr><td>IDOL</td><td>20.24</td><td>0.904</td><td>0.174</td><td>21.03</td><td>0.894</td><td>0.116</td></tr><tr><td>Ours (IDOL)</td><td>22.38</td><td>0.912</td><td>0.154</td><td>23.59</td><td>0.930</td><td>0.092</td></tr></table>

## 7.2. Monocular Setting

As introduced in the limitations, we observe a gap of 3.4 dB in PSNR between monocular settings and multi-view settings. Though we achieved SOTA in monocular setting, the quality is still far from downstream applications. As illustrated in Figure 5, we found that the fail cases largely derives from (1) the total failure of Wonder3D to generate novel views. (2) generated images do not follow camera constraint strictly, causing misalignments across views. These two reasons account for the failure in Gaussian initialization with LGM, and thus lead to corrupted results.

We attribute this issue to a lack of open-sourced realworld human diffusion models. Such work will largely fuel the field of single-view human reconstruction.

## 7.3. Comparison with new methods

IDOL can replace LGM to serve as the single-frame reconstruction module in our pipeline. Therefore, we further evaluate our method with this module. As shown in Table 3, our method surpasses the SOTA method and achieves better results with the stronger backbone IDOL. Yet, AniGS has not open-sourced their codes or test splits.

We conduct experiments on Deepfashion, an in-the-wild fashion clothing dataset. As shown in Table 3 and Figure 7, our model achieves consistent performance improvements with different reconstruction modules.

## 8. More Visualization Results

We present more visualization results for both novel view synthesis and novel pose animation in Figure 6 and 8.

## 9. Broader Impacts

Our modelâs capacity to generate high-quality 3D animatable avatars raises substantial privacy risks. To address

<!-- image-->  
(a) Deepfashion  
(b) MvHumanNet

Figure 7. Visualization on Deepfashion and MVHumanNet.

these, the creation of ethical guidelines and legal frameworks is imperative. This necessitates close collaboration among researchers, developers, and policymakers. Researchers should embed ethical considerations in development, while developers must implement privacy-centric practices. Policymakers need to craft regulations that define proper use, penalize misuse, and safeguard user privacy. Such collaboration is crucial for promoting the responsible application of this technology.

# 1 â 1 -

Figure 8. More qualitative results on novel view synthesis. The novel views in multiple directions indicate the high-quality and potential downstream applications of our reconstructed 3D avatar.